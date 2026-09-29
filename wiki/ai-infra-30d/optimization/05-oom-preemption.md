---
title: "优化专项 05: OOM 与抢占"
type: concept
tags: [ai-infra, inference-optimization, oom, preemption, kv-cache, gpu-memory, capacity-planning]
created: 2026-09-29
updated: 2026-09-29
---
# 优化专项 05: OOM 与抢占

> 一句话：显存问题分三类，启动 OOM、运行时 OOM、抢占，判定方法和修法各不相同，本页先讲显存由什么构成，再按三类给排查路径和容量规划公式。关联日课：[[day01-inference-pipeline]]、[[day02-vllm-setup]]、[[day06-param-tuning]]、[[day09-scheduler]]、[[day10-kv-cache]]、[[day11-quantization]]、[[day13-flash-attention]]、[[day16-tensor-parallel]]、[[day20-production-deploy]]。

## 1. 显存构成

一张卡上 vLLM 进程的显存，从大到小：

| 项 | 大小量级 | 何时确定 | 受什么参数影响 |
|---|---|---|---|
| 模型权重 | 参数量 × 每参数字节 / TP | 加载时 | 量化、TP |
| KV cache | 剩余显存的绝大部分 | 启动 profiling run 之后 | `gpu_memory_utilization`、`max_model_len`、`kv_cache_dtype`、上面所有项 |
| activation 峰值 | 与一步最多处理的 token 数成正比 | profiling run 实测 | `max_num_batched_tokens`、`max_num_seqs`、`max_model_len`、attention backend |
| CUDA graph 池 | 每个捕获桶一份中间 buffer，几百 MB 到 GB | 捕获时 | `cudagraph_capture_sizes`、`cudagraph_mode` |
| torch.compile 产物 | 几十到几百 MB | 编译时 | 编译级别 |
| 多模态 encoder cache | 可配置 | 启动时 | `--limit-mm-per-prompt`、encoder cache 大小参数 |
| NCCL buffer | TP>1 时几百 MB | 初始化 | TP、自定义 all-reduce |
| 采样 buffer | vocab × max_num_seqs × 4 字节量级 | 启动时 | `max_num_seqs`、`logprobs` |
| CUDA context 与碎片 | 几百 MB | 进程启动 | 不可控 |

启动流程：加载权重 → 跑一次 profiling run（用 `max_num_batched_tokens` 个 token 和 `max_num_seqs` 个序列的最坏 batch 做一次 forward，测 activation 峰值）→ `总显存 × gpu_memory_utilization − 权重 − activation 峰值 − 其他` 就是 KV 预算 → 按每 block 大小算出 `# GPU blocks` → 捕获 CUDA graph。

所以 KV 数量是"剩下的"，任何一项变大都在挤 KV。日志里 `# GPU blocks` 这一行是所有显存问题的第一个检查点。

## 2. 三类问题的判定

| 类型 | 表现 | 时刻 | 第一步 |
|---|---|---|---|
| 启动 OOM | 进程起不来，日志 `CUDA out of memory` 或 "No available memory for the cache blocks" 或 blocks 数为 0 | 加载权重、profiling run、graph 捕获 | 看是哪一步 OOM |
| 运行时 OOM | 服务跑了一段时间后 worker 崩溃，`CUDA out of memory` 出现在 forward 或采样 | 流量高峰、特殊请求 | 看崩溃前一步的 batch 组成 |
| 抢占 | 服务不崩，`vllm:num_preemptions_total` 增长，TTFT 和 ITL 长尾 | KV 接近满 | 看 `kv_cache_usage_perc` |

抢占不是 bug，是 KV 不够时的正常行为；但频繁抢占说明容量规划错了。V1 抢占只有 recompute：被抢占请求回到 waiting 队头，`num_computed_tokens` 清零，靠保留 hash 的 free block 让重算代价降低，但如果那些 block 已被驱逐，就是完整重做 prefill。没有 swap，`--swap-space` 在 V1 里对 KV 无意义。

## 3. 启动 OOM

### 3.1 权重加载就 OOM

确认：日志停在 "Loading model weights"，没有到 profiling。

修：权重放不下，只能量化、TP 拆分或换卡。检查 `--dtype` 没有被设成 float32；确认加载的是目标模型而不是带 vision tower 的变体。

### 3.2 profiling run OOM

确认：权重加载完成，日志在 "Starting profile run" 或 memory profiling 相关行之后 OOM。

原因：最坏 batch 的 activation 峰值超过了权重之外的剩余显存。峰值主要来自 `max_num_batched_tokens` 个 token 的中间张量（尤其是 logits：token 数 × vocab × 4 字节，Qwen 15 万 vocab 下 8192 token 的 logits 就是约 5 GB）和 attention 的临时 buffer。

修（按副作用从小到大）：
- 缩 `--max-num-batched-tokens`，直接降峰值。
- 缩 `--max-num-seqs`，减少采样 buffer 和 InputBatch。
- 缩 `--max-model-len`，减少 attention 相关 buffer 和 block table。
- 量化权重腾空间。
- 加 TP。

### 3.3 KV blocks 为 0 或过少

确认：日志有 blocks 数但极小，或报 "No available memory for the cache blocks"；有时是 `max_model_len` 一个请求就需要的 block 数超过总 blocks，日志会明确说 "The model's max seq len is larger than the maximum number of tokens that can be stored in KV cache"。

修：同 3.2，外加 `--gpu-memory-utilization` 上调（前提是卡上没有别的进程）。较新版本有 `--kv-cache-memory-bytes` 直接指定 KV 预算，绕过比例换算，参数是否存在以 `--help` 为准。

### 3.4 CUDA graph 捕获 OOM

确认：blocks 已分配，日志在 "Capturing CUDA graphs" 阶段 OOM。

修：减少 `cudagraph_capture_sizes` 的桶数或最大桶；`cudagraph_mode` 从 FULL 退到 PIECEWISE；极端情况 `--enforce-eager` 排除问题后再逐步加回。

### 3.5 卡上有别的进程

确认：`nvidia-smi` 在 vLLM 启动前就有占用。`gpu_memory_utilization` 是按总显存算比例，别的进程占的部分不会被扣除，直接导致 OOM。

修：清理进程，或把比例按"实际可用"重算。

## 4. 运行时 OOM

启动成功说明最坏 batch 的 profiling 过了，运行时还 OOM 通常是 profiling 没覆盖到的路径。

### 4.1 profiling run 没覆盖的形状

- 多模态：profiling 用的是每请求最大图片数的假设，实际请求超过就 OOM。修：`--limit-mm-per-prompt` 卡死上限，前端拒绝超限请求。
- `logprobs` 或 `prompt_logprobs`：请求级参数，profiling 时按默认值算，大量请求同时要 top-k logprobs 时采样 buffer 超预期。修：限制 `--max-logprobs`。
- spec decode 的 draft 部分、LoRA adapter 数量超过 `--max-loras`。
- 长 prompt 走了 profiling 没走的 attention 路径（backend 在不同长度下切 kernel）。

确认：崩溃前最后一步的 SchedulerOutput（开 DEBUG 日志）里 token 数、请求数、多模态项、采样参数。

### 4.2 显存碎片与预留不足

`gpu_memory_utilization` 设到 0.95 以上时，PyTorch allocator 的碎片和 NCCL 的临时 buffer 可能挤不进剩余的 5%。

修：回到 0.9；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少碎片（PyTorch 通用选项，vLLM 是否默认设置以版本为准）。

### 4.3 外部进程后来占了显存

监控脚本、另一个服务、notebook 在服务运行后启动。确认：`nvidia-smi` 的进程列表。修：独占卡，或 K8s 里用 GPU 资源请求隔离。

### 4.4 TP 下某一 rank 先 OOM

各 rank 权重均分但 rank 0 多承担采样和输出，多模态 encoder 也可能只在部分 rank 上跑。确认：崩溃的是哪个 rank。修：同 3.2 的参数收缩，通常收 `max_num_seqs` 最有效。

## 5. 抢占

### 5.1 判定

`vllm:num_preemptions_total` 是累计计数，看它的增长速率（每分钟增量）而不是绝对值。配合 `kv_cache_usage_perc`（接近 1）和 `num_requests_running`。

偶发抢占（示例：每小时几次）可以接受；每分钟都在增长就是容量问题，表现为 TTFT 和 ITL 双双长尾，见 [[02-ttft-high]] 和 [[03-tpot-itl-jitter]]。

### 5.2 为什么会抢占

调度器每步先给 running 请求分新 block，分不出时踢掉最晚进入 running 的请求（priority 模式踢优先级最低的）。抢占后这一步不再接收新请求。所以抢占的直接原因是"并发 × 每请求上下文增长"超过了 KV 总量，触发点通常是一批长输出请求同时进入 decode 后期。

### 5.3 修法

| 手段 | 效果 | 副作用 |
|---|---|---|
| 降 `max_num_seqs` | 立刻止血，并发上限与 KV 匹配 | 吞吐略降，排队上升（但排队比抢占便宜） |
| 缩 `max_model_len` | 单请求 block 上限降低，KV 预算变大 | 超长请求被拒 |
| `--kv-cache-dtype fp8` | KV 翻倍 | 质量轻微影响，需 backend 支持 |
| 权重量化 | 腾显存给 KV | 质量影响，kernel 匹配 |
| 网关限制 `max_tokens` | 长输出不会无限增长 | 产品侧约束 |
| 提高 prefix cache 命中 | 共享前缀只存一份 | 无 |
| 加卡 TP 或 DP | 容量线性增加 | 成本 |
| priority scheduling | 抢占先踢低优先级 | 不减少抢占总量 |

不要做的：调大 `gpu_memory_utilization` 到 0.95 以上换来的 KV 很少，却把运行时 OOM 的风险带回来；调大 budget 与抢占无关。

## 6. 容量规划公式

抢占频繁本质是规划错误，用下面的公式反推：

```
每 token KV 字节 = 2 × local_layers × local_kv_heads × head_dim × bytes_per_elem
每请求平均 KV = 每 token KV × (平均 prompt 长度 + 平均输出长度 × 0.5 到 1.0)
    （decode 过程中上下文逐步增长，稳态取均值；保守用 1.0）
可承载并发 = KV 总字节 / 每请求平均 KV
安全并发 = 可承载并发 × 0.8 到 0.9   （留出长尾请求和 prefix cache 的空间）
max_num_seqs 设为安全并发；超出部分让网关排队或扩容
```

示例（示意数字）：Qwen2.5-7B BF16 每 token 约 57 KB，KV 预算 20 GB，平均 prompt 2k、平均输出 1k、保守系数 1.0，每请求约 171 MB，可承载约 120 并发，安全并发约 100。若目标并发 200，要么 KV FP8（约 240），要么加一张卡 DP。

反过来，已知目标并发和 workload，可以算出需要的 KV 预算，再决定卡数和量化方案。这是 [[15-cost-per-token]] 的输入。

## 7. 决策表

| 症状 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| 权重加载 OOM | 量化或 TP | 换卡 | 调 gpu_memory_utilization |
| profiling run OOM | 缩 max_num_batched_tokens | 缩 max_num_seqs、max_model_len | enforce-eager（不解决 activation） |
| blocks 为 0 或极少 | 缩 max_model_len | 量化、kv-cache-memory-bytes | 只调比例 |
| graph 捕获 OOM | 减 capture sizes | PIECEWISE 模式 | 长期 enforce-eager |
| 运行时 OOM，多模态 | limit-mm-per-prompt | 前端拒绝超限 | 提高比例 |
| 运行时 OOM，比例 0.95 以上 | 回 0.9 | expandable_segments | 继续提高比例 |
| 抢占每分钟增长 | 降 max_num_seqs 止血 | KV FP8、缩 max_model_len、限 max_tokens | 提高比例、调 budget |
| 抢占偶发 | 监控即可 | priority 保护关键流量 | 过度收缩并发 |

## 8. 验证实验

固定：模型、dtype、种子、`temperature=0`。

实验 A，显存账本：`--max-model-len` 取 4096 / 16384 / 32768，`--max-num-batched-tokens` 取 2048 / 8192，记录每组启动日志的权重大小、profiling 峰值、`# GPU blocks`。画出 blocks 随两参数的变化，验证"activation 峰值与 budget 成正比、blocks 是剩余量"。

实验 B，触发并消除抢占：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.5 --max-model-len 4096 --max-num-seqs 128
vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --num-prompts 300 --request-rate inf
watch -n1 'curl -s localhost:8000/metrics | grep -E "num_preemptions_total|kv_cache_usage_perc|num_requests_(running|waiting)"'
```

记录抢占开始时的 KV 使用率和 running 数。用第 6 节公式算安全并发，把 `max_num_seqs` 改成该值重跑，预期抢占归零、ITL P99 大幅收窄、吞吐下降不超过一成。

实验 C，KV FP8 的容量收益：同一配置加 `--kv-cache-dtype fp8`，对比 `# GPU blocks` 和实验 B 的抢占阈值。预期 blocks 接近翻倍。

实验 D，运行时 OOM 复现：`--max-logprobs 20` 下发 200 个并发请求全部带 `logprobs=20`，观察是否 OOM；改 `--max-logprobs 5` 重跑。

## 9. 关联

- 总入口：[[01-diagnosis-playbook]]
- 相邻：[[02-ttft-high]]、[[03-tpot-itl-jitter]]（抢占的延迟表现）、[[04-throughput-low]]（KV 容量与并发）、[[07-long-context]]（长上下文下的 KV 压力）、[[10-quantization-choice]]、[[13-parallelism-choice]]、[[15-cost-per-token]]
- 日课：[[day01-inference-pipeline]]、[[day02-vllm-setup]]、[[day06-param-tuning]]、[[day09-scheduler]]、[[day10-kv-cache]]、[[day11-quantization]]、[[day13-flash-attention]]、[[day16-tensor-parallel]]、[[day20-production-deploy]]
