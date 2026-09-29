---
title: "优化专项 04: 吞吐上不去"
type: concept
tags: [ai-infra, inference-optimization, throughput, roofline, batching, kv-cache, parallelism]
created: 2026-09-29
updated: 2026-09-29
---
# 优化专项 04: 吞吐上不去

> 一句话：吞吐被三样东西之一卡住：KV 容量（并发上不去）、GPU 算力或带宽（batch 不够大或 workload 本质 compute-bound）、CPU（引擎跟不上 GPU）；先判断是哪一样，再谈优化。关联日课：[[day03-benchmark]]、[[day05-profiling]]、[[day06-param-tuning]]、[[day09-scheduler]]、[[day10-kv-cache]]、[[day11-quantization]]、[[day13-flash-attention]]、[[day16-tensor-parallel]]、[[day18-data-parallel]]、[[day24-disaggregated]]、[[day26-roofline]]。

吞吐有三种口径，先说清楚在优化哪一个：

- **output tokens/s**：decode 产出速度，chat 和长输出场景的主指标。
- **total tokens/s**（prompt + output）：RAG、摘要这类长输入短输出场景的主指标，prefill 占大头。
- **requests/s** 或 **goodput**：满足 SLO（TTFT、ITL 上限）的请求速率。生产上真正要的是这个，单纯堆 tokens/s 而 P99 爆掉没有意义。

## 1. 症状与判定

| 观察 | 指向 |
|---|---|
| `num_requests_running` 远小于 `max_num_seqs`，`kv_cache_usage_perc` 接近 1 | KV 容量卡住并发 |
| `num_requests_running` 等于 `max_num_seqs`，KV 使用率不高 | `max_num_seqs` 卡住 |
| running 已很高，GPU SM 利用率仍低于 60% | CPU-bound，引擎跟不上 |
| SM 利用率高，output tokens/s 随并发不再增长 | GPU 饱和，到 roofline 上限 |
| prompt tokens 占总 tokens 的 80% 以上 | prefill-bound，需要算力而不是 batch |
| TP 越大单卡吞吐越低 | 通信占比过高 |
| 开结构化输出后吞吐掉一半 | bitmask 生成与 FSM 推进在 CPU |

判定顺序：并发有没有被卡（KV 或 max_num_seqs）→ GPU 是否真的忙（nvidia-smi 的 SM 利用率、nsys 的 kernel 密度）→ workload 是 prefill-bound 还是 decode-bound → 通信占比。

## 2. roofline 视角：为什么 batch 决定 decode 吞吐

decode 每步要把全部权重从 HBM 读一遍，无论 batch 里有几个请求。以 7B BF16 为例，权重约 14 GB，H100 HBM 约 3.35 TB/s，读一遍约 4.2 ms，这是一步的下限。batch=1 时每步只产 1 个 token，吞吐约 240 tokens/s；batch=64 时每步产 64 个 token，权重读取摊薄，只要计算没成为瓶颈，吞吐接近 64 倍。

算术强度 = FLOPs / bytes。batch 增大，每读一字节权重做的乘加增多，算术强度线性上升，直到碰到 GPU 的算力上限（H100 BF16 约 990 TFLOPS，对应算术强度约 300 FLOPs/byte）。粗算这个交叉点在 batch 几百的量级，但 attention 部分要读 KV cache，序列越长 KV 读取越多，交叉点会提前。

结论：decode 吞吐在 batch 不够大时是 memory-bound，加 batch 几乎免费；到交叉点后是 compute-bound，只能靠量化减少 FLOPs、更强的卡或更多卡。数字是示意，用 [[day26-roofline]] 的方法算你的模型和卡。

这也是为什么"并发被 KV 卡住"是最常见的吞吐问题：batch 上不去，GPU 在等内存。

## 3. 根因逐个排查

### 3.1 KV 容量卡住并发

启动日志里 `# GPU blocks` 乘 block_size 就是能同时缓存的 token 总数。除以每个请求的平均上下文长度（prompt + 已生成），就是实际能承载的并发，通常远小于 `max_num_seqs`。

确认：`kv_cache_usage_perc` 接近 1，`num_requests_waiting` 大于 0，`num_requests_running` 远小于 `max_num_seqs`。

修（按性价比排序）：
- `--max-model-len` 缩到业务实际最大值。它影响 profiling run 的 activation 预留和每请求的 block 上限，缩小后 KV 可用量往往明显上升。
- `--kv-cache-dtype fp8`：KV 减半，并发近似翻倍。需要 backend 支持，看启动日志。质量影响见 [[10-quantization-choice]]。
- 权重量化（FP8 或 AWQ/GPTQ）腾出显存给 KV。
- `--gpu-memory-utilization` 从 0.9 提到 0.92 到 0.95，前提是没有其他进程用这张卡，并留意 [[05-oom-preemption]]。
- prefix caching 命中率提升本身就是省 KV：共享前缀只存一份。
- 更多卡：TP 拆权重腾显存，或 DP 加实例。

### 3.2 `max_num_seqs` 卡住

确认：running 长期等于 `max_num_seqs`，KV 使用率不高（示例：低于 0.7）。

修：调大。V1 默认值随版本和场景变化，以启动日志为准。代价：batch 变大后 ITL 上升，见 [[03-tpot-itl-jitter]]。调到 KV 使用率稳态在 0.8 到 0.9 为止。

### 3.3 CPU-bound：引擎跟不上 GPU

确认：running 很高但 SM 利用率低于 60%；EngineCore 进程 CPU 接近 100%；nsys 里 kernel 之间的空隙明显；py-spy 看到 `schedule`、`_prepare_inputs`、sampler 的 CPU 部分占比高。

修：
- `--async-scheduling`：调度与 GPU 执行重叠，decode 主导场景收益最直接。
- 确认 CUDA graph 在用（不是 `--enforce-eager`），capture sizes 覆盖实际 batch。
- 统一采样参数，去掉不必要的 `logprobs`。
- `--api-server-count` 分担前端 CPU（tokenize、detokenize）。
- 小模型高并发时 CPU 是主要瓶颈，DP 多实例比单实例大 batch 更能利用多核。
- 细节见 [[06-gpu-util-low-cpu-bound]]。

### 3.4 TP 过大，通信占比高

TP 每层两次 all-reduce。模型小、batch 小时通信时间和计算时间同量级，TP=8 的单卡吞吐可能只有 TP=1 的一半。

确认：nsys 里 all-reduce kernel 总时长占比；对比 TP=1 单实例吞吐乘卡数与 TP=N 的吞吐。

修：单卡装得下的模型用 DP；装不下的用能装下的最小 TP，剩余卡做 DP；MoE 模型 attention 用 DP、expert 用 EP，见 [[12-moe-serving]] 和 [[13-parallelism-choice]]。

### 3.5 attention backend 不对

确认：启动日志的 backend 名。Hopper 上应是 FA3，Blackwell 上倾向 FlashInfer；Triton 是兜底，长序列下明显慢。FP8 KV、特殊 head size、sliding window 可能让选择器退到慢路径。

修：`VLLM_ATTENTION_BACKEND=` 显式指定并对比；MLA 模型确认用了 FlashMLA 或 CUTLASS MLA 而不是 Triton MLA。见 [[day13-flash-attention]]。

### 3.6 workload 本质是 prefill-bound

长输入短输出（RAG、分类、摘要）下 prompt token 占 80% 以上。prefill 是 compute-bound，batch 再大也不会像 decode 那样摊薄，吞吐上限就是 GPU 算力。

确认：`vllm:prompt_tokens_total` 与 `vllm:generation_tokens_total` 的比例；SM 利用率已高。

修：
- 提升 prefix cache 命中率，直接减少 prefill 计算量。
- 调大 `max_num_batched_tokens`，让 prefill 一步吃完，减少调度和 kernel launch 开销。
- FP8 权重量化：Hopper 上 FP8 GEMM 算力翻倍，prefill 直接受益。
- 更多算力：加卡做 DP，或 P/D 解耦让 prefill 实例专门堆算力。
- 这类 workload 不要开 spec decode，输出短没收益。

### 3.7 结构化输出

每步为每个结构化输出请求生成 vocab 大小的 bitmask，FSM 推进在 CPU。大量结构化请求时 EngineCore 变成 CPU-bound。

确认：关掉 `response_format` 对比吞吐；py-spy 看 `grammar_bitmask` 占比。

修：简化 schema；对比 backend（xgrammar、guidance）；结构化请求单独实例池。见 [[09-structured-output-tool-calls]]。

### 3.8 量化 kernel 不匹配

AWQ/GPTQ 4-bit 在大 batch 下需要反量化再做 GEMM，batch 大时可能比 BF16 慢。Marlin 类 kernel 缓解但有形状限制。FP8 需要 Hopper 及以上才有硬件加速。

确认：同 workload 下量化版与 BF16 版的吞吐曲线，看交叉点；启动日志的量化 kernel 名。

修：高并发用 FP8 而不是 4-bit；4-bit 留给显存受限的低并发场景。见 [[10-quantization-choice]]。

### 3.9 多模态

vision encoder 每张图一次完整 forward，不受 batch 摊薄；图片预处理占前端 CPU。

确认：nsys 里 encoder kernel 段占比；前端进程 CPU。

修：encoder cache 依赖图片 hash 命中；限制分辨率和每请求图片数；前端进程数；部分模型支持 encoder 独立部署。

## 4. 决策表

| 症状组合 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| KV 使用率接近 1，running 远小于 max_num_seqs | 缩 max_model_len、KV FP8 | 权重量化、加卡 | 调大 max_num_seqs |
| running = max_num_seqs，KV 使用率低 | 调大 max_num_seqs | — | 调 budget |
| running 高、SM 利用率低 | async scheduling、确认 CUDA graph | api-server-count、DP 多实例 | 加 GPU |
| SM 利用率高、tokens/s 不再随并发增长 | 已到上限：FP8 权重、加卡 DP | P/D 解耦 | 继续加并发（只会增加 ITL） |
| prompt tokens 占比大于 80% | prefix cache 命中率、调大 budget、FP8 | 加算力 | spec decode |
| TP=N 吞吐低于 TP=1 乘 N 的一半 | 改 DP 或最小 TP | 检查 NVLink 拓扑 | 加大 TP |
| backend 是 Triton | 显式指定 FA3/FlashInfer | 检查 FP8 KV 与 backend 兼容 | 接受现状 |
| 结构化请求多 | 简化 schema、换 backend | 独立实例池 | 调引擎参数 |
| 4-bit 量化在高并发下变慢 | 换 FP8 | 回 BF16 加卡 | 调 batch |
| 低并发（QPS 低）吞吐低 | spec decode（EAGLE-3 / MTP） | 合并实例提高单实例并发 | 加卡 |

低并发是个特例：GPU 本来就没喂饱，此时 spec decode 用闲置算力换 decode 步数，见 [[11-spec-decode-no-gain]]。高并发下它反而抢 batch 的算力。

## 5. 验证实验

固定：模型、dtype、`max_model_len`、`temperature=0`、种子。报告 output tokens/s、total tokens/s、TTFT P99、ITL P99，四个一起看才能判断 goodput。

实验 A，并发扫描找上限：

```bash
for C in 1 4 16 64 128 256; do
  vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --dataset-name random \
    --random-input-len 512 --random-output-len 512 --num-prompts $((C*8)) \
    --max-concurrency $C --request-rate inf
done
```

预期：output tokens/s 先近似线性上升，然后趋平。趋平点对应 roofline 交叉或 KV 耗尽，看 `kv_cache_usage_perc` 区分。同时 ITL P99 会在趋平点前开始上升，goodput 的最优并发通常在趋平点之前。

实验 B，KV 容量三招对比：baseline、`--max-model-len` 从 32768 缩到 8192、`--kv-cache-dtype fp8`、两者叠加。记录启动日志 `# GPU blocks` 和实验 A 的趋平点变化。预期 blocks 数与趋平并发近似成正比。

实验 C，TP 与 DP：8 卡机器上 7B 模型，TP=8 单实例、TP=2 四实例、TP=1 八实例（网关轮询），同一总流量下比较总 tokens/s 和 ITL P99。预期 TP=1 八实例吞吐最高，TP=8 最低，ITL 差距在小 batch 时最明显。

实验 D，prefill-bound 识别：input 4096 / output 32 与 input 32 / output 4096 两种 workload，各自扫并发。预期前者 total tokens/s 很快趋平且 SM 利用率高，后者 output tokens/s 随并发持续上升。前者用 FP8 权重再跑一次，看趋平值是否提升。

实验 E，async scheduling：decode 主导 workload，开关对比。预期吞吐提升几个百分点，CPU-bound 越严重收益越大。

## 6. 容量估算速查

给定一张卡，估算能跑多少并发：

```
可用 KV 显存 ≈ 总显存 × gpu_memory_utilization − 权重 − activation 预留（profiling run 决定）
每 token KV ≈ 2 × layers × kv_heads × head_dim × bytes（GQA 模型远小于 MHA）
可缓存 token 数 = 可用 KV 显存 / 每 token KV
可承载并发 ≈ 可缓存 token 数 / 平均每请求上下文长度
```

以 Qwen2.5-7B（28 层、4 个 KV head、head_dim 128、BF16）为例，每 token 约 57 KB，20 GB KV 能存约 35 万 token，平均上下文 4k 时约 85 个并发。这是示意数字，以启动日志和 [[day01-inference-pipeline]] 的公式为准。算出来的并发低于目标就先做 3.1，高于目标但吞吐仍低就看 3.3 到 3.6。

## 7. 关联

- 总入口：[[01-diagnosis-playbook]]
- 相邻：[[02-ttft-high]]、[[03-tpot-itl-jitter]]（并发与延迟的取舍）、[[05-oom-preemption]]（KV 容量）、[[06-gpu-util-low-cpu-bound]]、[[10-quantization-choice]]、[[11-spec-decode-no-gain]]、[[12-moe-serving]]、[[13-parallelism-choice]]、[[15-cost-per-token]]
- 日课：[[day03-benchmark]]、[[day05-profiling]]、[[day06-param-tuning]]、[[day09-scheduler]]、[[day10-kv-cache]]、[[day11-quantization]]、[[day13-flash-attention]]、[[day16-tensor-parallel]]、[[day18-data-parallel]]、[[day24-disaggregated]]、[[day26-roofline]]
