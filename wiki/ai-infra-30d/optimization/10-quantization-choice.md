---
title: "优化专项 10: 量化选型与「量化后反而没变快、变差」"
type: concept
tags: [ai-infra, inference-optimization, quantization, fp8, awq, gptq, kv-cache-fp8, nvfp4]
created: 2026-09-29
updated: 2026-09-29
---

# 优化专项 10: 量化选型与「量化后反而没变快、变差」

> 一句话：按"目标 × 硬件 × workload"选量化格式，并解释为什么量化经常不带来预期收益甚至变慢。关联日课：[[day11-quantization]]、[[day26-roofline]]、[[day10-kv-cache]]、[[day13-flash-attention]]、[[day19-moe-inference]]。

## 0. 先建立正确的预期

量化只在三条路径上产生收益，三条路径的前提互相不同：

| 路径 | 机制 | 什么时候有效 | 什么时候无效 |
|---|---|---|---|
| A. 降低 decode 的权重读带宽 | 每步 decode 要把全部权重从 HBM 读一遍，权重字节数减半，步时间近似减半 | 小 batch、decode 主导、memory-bound（roofline 左侧） | 大 batch 已经 compute-bound；KV 读取量远大于权重读取量（长上下文 + 大 batch） |
| B. 提高 prefill / 大 batch 的算力 | FP8 / INT8 Tensor Core 峰值是 BF16 的 2 倍，NVFP4 再翻倍 | 激活也量化（W8A8、W4A4），且 kernel 真的走了低精度 Tensor Core | 只量化权重不量化激活（W4A16）：矩阵乘仍是 BF16，还多了 dequant |
| C. 省显存换并发 | 权重占用下降，KV 空间变大，可承载并发上升 | 并发被 KV 卡住（`kv_cache_usage_perc` 长期接近 100%、有抢占） | 并发被 `max_num_seqs` 或 CPU 卡住；权重本来就不是显存大头（小模型大上下文） |

"量化后没变快"几乎总是因为选了一条不匹配当前瓶颈的路径。先用 [[01-diagnosis-playbook]] 确认瓶颈，再选格式。

## 1. 症状与判定

| 症状 | 伴随指标 | 最可能的原因 | 跳到 |
|---|---|---|---|
| W4A16（AWQ/GPTQ）上线后吞吐没升，甚至 TPOT 变高 | 并发 ≥ 32，SM 利用率高，nsys 里 GEMM 占比上升 | 大 batch 下 dequant + BF16 矩阵乘变成 compute-bound，W4 比 FP8 慢 | 2.1 |
| FP8 上线后 decode 单请求延迟几乎没变 | 并发 = 1 到 4，TPOT 只降了不到 10% | 权重只占每步读带宽的一部分，KV 读取、非 GEMM kernel、launch 开销没变 | 2.2 |
| 量化后吞吐没升，`kv_cache_usage_perc` 仍接近 100% | `num_preemptions_total` 在增长 | 省下的显存没有转成 KV：`gpu_memory_utilization` 没调、`max_model_len` 没变、并发被 `max_num_seqs` 卡住 | 2.3 |
| 启动日志显示走了非预期 kernel | 日志有 "Using xxx kernel for quantization" 或退回 `torch` 实现 | 硬件或 shape 不满足高性能 kernel 前提（head_dim、group size、SM 版本） | 2.4 |
| 输出质量下降，长回答末尾开始乱 | 长序列、低 temperature 场景更明显 | KV FP8 无校准 scale、或 W4 的 group size 太大 / 校准集不匹配 | 2.5 |
| 量化模型 `--max-model-len` 被拒或 OOM | 启动阶段 | 量化 checkpoint 的 config 与原模型不同，或 KV 量化没开导致 KV 反而是瓶颈 | 2.6 |
| 量化 + LoRA / 量化 + spec decode 报错或变慢 | 启动或运行时 | kernel 组合未支持，回退到慢路径 | 2.7 |

判定用的指标和命令：

```bash
# 启动日志：量化方法、kernel 后端、KV dtype、attention backend
grep -E "quantization|kernel|kv_cache_dtype|Using .* backend" server.log
# 运行时：GEMM 占比（是否 compute-bound）
nsys profile -t cuda --capture-range=cudaProfilerApi -o q python -c "..."   # 配合 --profiler-config 或 /start_profile
# 显存是否转成 KV
grep -E "GPU KV cache size|Maximum concurrency" server.log
```

## 2. 根因逐个排查

### 2.1 W4A16 在大 batch 下比 FP8 慢

**机制**：AWQ/GPTQ 只量化权重，激活仍是 BF16。矩阵乘之前要把 INT4 权重 dequant 成 BF16（或在 kernel 内做），矩阵乘本身仍在 BF16 Tensor Core 上跑。小 batch 时瓶颈是读权重，INT4 只读 1/4 字节，赢；batch 大到 compute-bound 之后，矩阵乘时间由 BF16 算力决定，dequant 是纯增项，输给 FP8（FP8 Tensor Core 算力是 BF16 的 2 倍且无 dequant）。

**确认**：固定 input/output 长度，concurrency 梯度 1 / 8 / 32 / 128 分别跑 BF16、FP8、W4A16。W4A16 曲线通常在低并发领先、高并发落后于 FP8；交叉点随模型和 GPU 变化，示例量级：7B 模型在 H100 上大约在并发 16 到 64 之间。

**修**：高并发、追求吞吐用 FP8（H100 及以上）；显存受限、低并发、追求单请求延迟用 W4A16。两者都要则 W4A8（部分 kernel 支持，版本相关）或 NVFP4（Blackwell）。

**副作用**：FP8 需要 Hopper 及以上；Ampere 上只能 INT8（SmoothQuant W8A8）或 W4A16。

### 2.2 FP8 对小 batch decode 收益有限

**机制**：decode 每步读的字节 = 权重 + 当前 batch 的 KV + 中间激活。权重减半只作用于第一项。上下文长、batch 小时 KV 项可能和权重项同量级；另外 norm、rope、sampling、launch 开销不随量化变化。

**确认**：nsys 看单步 decode 里 GEMM kernel 时间占比。占比 50% 时权重减半最多让步时间降 25%。

**修**：同时开 KV FP8（`--kv-cache-dtype fp8`）把第二项也减半；开 CUDA graph 消 launch 开销（[[06-gpu-util-low-cpu-bound]]）；或者接受这是 memory-bound 的物理上限，转向 spec decode（[[11-spec-decode-no-gain]]）。

### 2.3 省下的显存没变成 KV

**机制**：vLLM 按 `gpu_memory_utilization × 总显存 - 权重 - 激活 profiling` 分配 KV。权重变小，KV 自动变大，但并发还受 `max_num_seqs`、`max_model_len`、调度 budget 限制。

**确认**：对比量化前后启动日志的 "GPU KV cache size" 和 "Maximum concurrency"，再看压测时 `vllm:num_requests_running` 是否真的上去了。

**修**：把 `--max-num-seqs` 调到新的 KV 上限允许的水平；如果目标是并发，KV FP8 通常比权重 W4 更直接（[[05-oom-preemption]]）。

### 2.4 kernel 后端没走高性能路径

vLLM 会按 checkpoint 的 `quantization_config` 自动识别方法（`--quantization` 通常不用手动指定），再按 GPU 架构、权重 shape、group size 选 kernel。确定存在的后端：

| 方法 | 主要 kernel（版本相关，以启动日志为准） | 硬件前提 |
|---|---|---|
| FP8 W8A8 | CUTLASS FP8 GEMM；Blackwell 上有 CUTLASS/FlashInfer 的 FP8 blockwise 路径 | Hopper 及以上；Ada（L40S）支持 FP8 |
| INT8 W8A8（SmoothQuant，llm-compressor 产出） | CUTLASS INT8 GEMM | Ampere 及以上 |
| AWQ / GPTQ W4A16 | Marlin（Ampere+，最常用）、Machete（Hopper，针对大 batch 优化）、旧的 exllama 路径作为回退 | Marlin 要求 group size 和 shape 满足条件，否则回退慢 kernel |
| NVFP4 / MXFP4 | CUTLASS / FlashInfer 的 FP4 GEMM | Blackwell（B200/GB200）；MXFP4 是 gpt-oss 等模型的原生格式 |
| KV FP8 | 由 attention backend 实现：FA3、FlashInfer、Triton backend 支持，FA2 不支持 | 见 2.5 |

**确认**：启动日志 grep "kernel"，或 `VLLM_LOGGING_LEVEL=DEBUG`。看到 "falling back" 或 "not supported, using" 就是退回慢路径。

**修**：换 group size（128 是最常见且被 Marlin 支持的）、换 GPU、或换量化方法。不要自己改 kernel 选择逻辑除非能验证正确性。

### 2.5 质量下降

**KV FP8**：`--kv-cache-dtype fp8` 默认用 e4m3 且 scale 为 1，动态范围窄，激活异常值大的模型（部分 Qwen、GLM）在长序列末端更容易劣化。修：用带 `kv_scale` 的校准 checkpoint（llm-compressor 可产出），或用 `fp8_e5m2`（范围更大精度更低）。前提：attention backend 支持，FA2 不行，日志会说明选用了哪个 backend。

**W4 权重**：group size 越大误差越大；校准数据集和目标任务分布不一致（用英文通用语料校准，跑中文代码任务）。修：group size 128、用与业务同分布的校准集重新量化、或换 AWQ（对激活异常值更鲁棒）。

**验证方法**（每次上线量化都要做）：

1. 任务集：业务样本 200 到 500 条，人评或强模型评分，对比 BF16。
2. logprob 差异：同 prompt 用 `logprobs=5` 取 BF16 和量化版本的 top-1 token 一致率和 KL，示例阈值：top-1 一致率低于 95% 就要看任务集。
3. 长序列末端：max_tokens 2k 以上的样本单独看后 20% 的质量，KV 量化问题在这里最先暴露。
4. 结构化输出：JSON 有效率，量化后 sampler 分布变化会让 grammar 约束更频繁生效。

### 2.6 量化 checkpoint 的配置差异

预量化 checkpoint（llm-compressor、AutoAWQ、GPTQModel 产出）自带 `config.json` 和 tokenizer。常见坑：tokenizer 或 chat template 是量化者从旧版本复制的，与原模型不一致，导致对比实验里"质量下降"其实是 template 错了。上线前 diff 两个 repo 的 `tokenizer_config.json` 和 `generation_config.json`。`max_model_len` 也可能被量化者改小。

在线 FP8（`--quantization fp8` 加载 BF16 权重启动时转换）不存在这个问题，但启动时间更长，且是 per-tensor 动态 scale，精度略低于离线校准的 FP8。

### 2.7 与其他特性的组合

- **量化 + LoRA**：LoRA 作用于 BF16 的 base 输出，量化 base 支持，但 Punica kernel 和某些量化 kernel 的组合有版本限制，看启动是否报错。
- **量化 + spec decode**：target 量化没问题；EAGLE head 通常是 BF16，draft_model 也可以是量化的，接受率要重新测（量化改变了 target 分布，[[11-spec-decode-no-gain]]）。
- **MoE 量化**：专家权重是 MoE 的显存大头，量化收益最大；DeepSeek-V3/R1 原生 FP8（blockwise），不要再二次量化；Qwen3 MoE 的 FP8 版本可用；fused MoE kernel 对量化格式的支持比 dense GEMM 慢一拍，看日志（[[12-moe-serving]]）。
- **量化 + TP**：per-channel/per-group scale 随权重切分，没有问题；per-tensor scale 在 TP 下各 rank 一致。

## 3. 决策表

按目标和硬件：

| 目标 | A100 | H100 / H200 | B200 | L40S / Ada |
|---|---|---|---|---|
| 提高吞吐（高并发） | INT8 W8A8（SmoothQuant） | **FP8 W8A8** | NVFP4 或 FP8 | FP8（算力低，收益有限） |
| 降单请求 TPOT（低并发） | W4A16 Marlin | W4A16 Machete/Marlin，或 FP8 + KV FP8 | NVFP4 | W4A16 |
| 省显存放下模型 | W4A16 | W4A16 | NVFP4 | W4A16 |
| 提并发（KV 受限） | KV 不支持 FP8 于 FA2，用 Triton backend 或降 max_model_len | **KV FP8** + FP8 权重 | KV FP8 | KV FP8（Triton/FlashInfer backend） |
| 质量优先 | BF16 | FP8（几乎无损） | FP8 | BF16 或 FP8 |

按症状：

| 症状 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| 高并发吞吐低，H100 | FP8 | INT8 W8A8 | 换成 W4A16 |
| 低并发 TPOT 高，显存够 | FP8 + KV FP8 + CUDA graph | W4A16 | 先动量化，应先看 spec decode |
| 抢占多、并发上不去 | KV FP8 | W4A16 释放显存 | 只量化权重不调 `max_num_seqs` |
| 质量掉 | 换校准集 / group size 128 / 带 scale 的 KV FP8 | 回 BF16 | 用 e5m2 硬顶 |
| Ampere 上想要 FP8 | 不可行，用 INT8 | W4A16 | 强制 `--quantization fp8`（会走模拟路径或报错） |

## 4. 验证实验

固定：模型、`max_model_len`、input 512 / output 256、temperature 0、warmup 50 条、每组 300 条、同一 GPU、`--no-enable-prefix-caching` 消除缓存干扰。

```bash
for Q in bf16 fp8 awq; do
  case $Q in
    bf16) M=Qwen/Qwen2.5-7B-Instruct; EXTRA="";;
    fp8)  M=Qwen/Qwen2.5-7B-Instruct; EXTRA="--quantization fp8";;
    awq)  M=Qwen/Qwen2.5-7B-Instruct-AWQ; EXTRA="";;
  esac
  vllm serve $M --port 8000 --max-model-len 4096 --no-enable-prefix-caching $EXTRA &
  sleep 120
  for C in 1 8 32 128; do
    vllm bench serve --model $M --dataset-name random --random-input-len 512 --random-output-len 256 \
      --num-prompts 300 --max-concurrency $C --save-result --result-filename q_${Q}_c${C}.json
  done
  pkill -f "vllm serve"; sleep 10
done
```

看四条曲线：output tokens/s 和 TPOT P50 随并发变化。预期方向：

- 并发 1：AWQ 的 TPOT 最低，FP8 次之，BF16 最高。
- 并发 128：FP8 吞吐最高，AWQ 可能低于 BF16。
- 交叉点就是这个模型在这张卡上的选型边界，把它记进 tuning playbook。

再加 KV FP8 一组（`--kv-cache-dtype fp8`），看 "Maximum concurrency" 和并发 128 时的抢占次数。

质量实验：同一 200 条业务 prompt，三个版本各取 `logprobs=1`，算 top-1 一致率，超过 5% 不一致就跑人评。

## 5. 关联

- 诊断入口：[[01-diagnosis-playbook]]
- 上游瓶颈判断：[[04-throughput-low]]、[[05-oom-preemption]]、[[06-gpu-util-low-cpu-bound]]
- 组合方案：[[11-spec-decode-no-gain]]、[[12-moe-serving]]、[[13-parallelism-choice]]、[[15-cost-per-token]]
- 日课：[[day11-quantization]]、[[day26-roofline]]、[[day10-kv-cache]]、[[day13-flash-attention]]

## 6. 速答

- **"量化后显存没降多少"**：看的是 `nvidia-smi` 总占用，而 vLLM 按 `gpu_memory_utilization` 把省下的显存全给了 KV，总占用不变是正常的，看启动日志的 KV 大小。
- **"FP8 比 BF16 慢"**：几乎只发生在 Ampere（走模拟路径）或 shape 不满足 CUTLASS 前提退回 torch 实现时，看日志。
- **"AWQ 和 GPTQ 选哪个"**：同 group size 下质量接近，AWQ 对激活异常值更稳，GPTQ 有更多现成 checkpoint；两者都走 Marlin 时速度相同。
- **"要不要量化 lm_head 和 embedding"**：默认不量化，占比小且对质量敏感，保持默认。
- **"NVFP4 是不是可以无脑用"**：只在 Blackwell，且需要对应格式的 checkpoint；质量上介于 FP8 和 W4A16 之间，仍要跑任务集验证。
