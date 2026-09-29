---
title: "Day 6: vLLM 参数调优第一轮"
type: concept
tags: [day6, vllm, tuning, parameters, optimization]
created: 2026-06-01
updated: 2026-06-01
---

# Day 6: vLLM 参数调优第一轮

## Part 1: 调优的基本框架

推理参数调优的核心矛盾：

```
显存总量 = 模型权重 + KV Cache + 计算缓冲
         固定         ↕可调        小量固定

KV Cache 越大 → 能同时处理更多请求 → 吞吐更高
但：并发越高 → 单个请求的延迟可能增加
```

调优目标通常是在满足 **延迟 SLA**（如 P99 TTFT < 500ms）的前提下**最大化吞吐**（tokens/s）。

### 调优前必须明确的信息

1. **硬件**：GPU 型号、显存大小、GPU 数量
2. **模型**：参数量、KV heads 数、支持的最大上下文长度
3. **Workload**：典型的 input/output 长度分布、QPS 需求
4. **SLA**：TTFT 和 TPOT 的 P99 要求

---

## Part 2: 核心参数逐一分析

### `--gpu-memory-utilization`

**作用**：控制 vLLM 总共使用多少比例的 GPU 显存。

**调优逻辑**：
```
可用于 KV Cache 的显存 = GPU总显存 × gpu_memory_utilization - 模型权重 - 计算缓冲

例：A100 80GB, Qwen2.5-7B (FP16 ≈ 14GB), 计算缓冲 ≈ 2GB
  0.90: 80×0.9 - 14 - 2 = 56 GB KV Cache
  0.85: 80×0.85 - 14 - 2 = 52 GB
  0.95: 80×0.95 - 14 - 2 = 60 GB
```

**实验设计**：
```bash
for util in 0.80 0.85 0.90 0.95; do
  echo "=== gpu_memory_utilization=$util ==="
  vllm serve Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization $util \
    --max-model-len 8192 &
  sleep 30  # 等待启动

  python benchmarks/benchmark_serving.py \
    --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
    --endpoint /v1/completions --dataset-name random \
    --random-input-len 512 --random-output-len 128 \
    --num-prompts 200 --request-rate 20

  kill %1
  sleep 5
done
```

**预期结果**：
- 0.80 → 0.90：throughput 显著提升（更多 KV Cache → 更多并发）
- 0.90 → 0.95：提升较小，但 OOM 风险增加
- 重点看启动日志中 "# GPU blocks" 的变化

### `--max-model-len`

**作用**：限制单个请求的最大上下文长度（input + output）。

**调优逻辑**：

`max_model_len` 会限制单请求最大上下文长度，也会参与调度、内存 profiling 和 CUDA graph 等配置。设得越大，系统需要为更长请求留出可能性，通常会降低可承载并发或增加内存压力，但实际影响要看启动日志里的 KV block 数、`max_num_seqs`、workload 长度分布和是否启用 chunked prefill。

```
如果 max_model_len = 131072 (128K):
  单请求允许的最大 block 上界很高
  在长上下文 workload 下，可同时运行的请求数会明显受限

如果 max_model_len = 8192:
  单请求上界更小
  在业务确实不需要 128K 时，通常能让并发和调度更稳定
```

**实践建议**：设为业务实际需要的最大值，而不是模型支持的最大值。

### `--max-num-seqs`

**作用**：限制 scheduler 单次 iteration 最多同时处理多少个请求。

**调优逻辑**：
- 设太小（如 8）：decode 阶段 batch 太小，GPU 利用率低，throughput 差
- 设太大（如 512）：每次 iteration 的计算量大，TPOT 升高
- 实际上会被 KV Cache 空间限制：即使设 512，如果 block 只够 50 个请求，实际并发就是 ~50

**实验设计**：
```bash
for seqs in 8 32 64 128 256; do
  echo "=== max_num_seqs=$seqs ==="
  # 启动 vLLM 并压测（同上模式）
done
```

**预期结果**：
- 8 → 32：throughput 提升显著
- 32 → 64：throughput 继续提升但幅度减小
- 64+：在某个点后 throughput 不再增加（被 compute 或 KV Cache 限制），但 TPOT 可能开始升高

### `--max-num-batched-tokens`

**作用**：限制单次 iteration 处理的总 token 数。

**影响机制**：

这个参数主要影响 prefill：

```
例：一个 input_len=4096 的请求需要 prefill 4096 个 token

如果 max_num_batched_tokens=8192：
  这个 prefill 可以一次完成

如果 max_num_batched_tokens=2048：
  需要分 2 次完成（需要 chunked prefill 支持）

如果不限制：
  一个超长 prompt 的 prefill 会独占整个 iteration
  → 所有 decode 请求都被阻塞 → TPOT 出现尖峰
```

**调优策略**：
- 关注 ITL/TPOT 稳定性时：设小一些（如 2048-8192），让长 prefill 更容易被切分，减少 decode 被拖慢
- 关注 TTFT 和总体吞吐时：设大一些，减少 prefill 的 iteration 次数
- 默认值随 vLLM 版本、硬件和使用场景变化（V1 按 `UsageContext` 和显存大小分级给默认值）；以当前版本启动日志和官方配置文档为准
- 机制层面：V1 调度器把它当作每步的 token budget，decode 请求每个占 1，剩余全部给 prefill 分片。Day 9 逐行读这段代码
- 相关联的长 prefill 保护参数：`--max-num-partial-prefills`、`--max-long-partial-prefills`、`--long-prefill-token-threshold`

---

## Part 3: 参数之间的联动关系

这些参数不是独立的，它们之间存在复杂的互相影响：

```
gpu_memory_utilization ──→ 可用 KV Cache 显存 ──→ GPU blocks 数
                                                       ↓
max_model_len ──→ 每个请求最大 block 数 ─────────→ 最大并发请求数
                                                       ↓
max_num_seqs ──→ 实际并发上限 = min(max_num_seqs, blocks/per_request_blocks)
                       ↓
max_num_batched_tokens ──→ 每次 iteration 的计算负载 ──→ TPOT
```

### 典型配置示例

**场景 1：短对话（chatbot，input~200, output~200）**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096
# 关注：高并发下的 throughput 和 TPOT
```

**场景 2：RAG（长 context，input~4000, output~500）**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096
# 关注：TTFT（prefix caching 的效果）和 TPOT 稳定性
```

**场景 3：代码生成（长输出，input~500, output~2000）**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --max-num-seqs 64
# 关注：长生成过程中的 TPOT 是否稳定
```

---

## Part 4: 不同模型的调优差异

### Qwen2.5 系列

| 模型 | 权重显存 | KV/token | 24GB GPU 建议 | 80GB GPU 建议 |
|---|---:|---:|---|---|
| Qwen2.5-3B | ~6 GB | 36 KB | max_model_len=16384, seqs=128 | max_model_len=32768, seqs=256 |
| Qwen2.5-7B | ~14 GB | 56 KB | max_model_len=4096, seqs=32 | max_model_len=16384, seqs=128 |
| Qwen2.5-14B | ~28 GB | 192 KB | 需要量化 | max_model_len=8192, seqs=64 |
| Qwen2.5-72B | ~144 GB | 320 KB | 不可行 | TP=2+, max_model_len=4096 |

### DeepSeek 系列

- DeepSeek-V2-Lite (16B MoE, 2.4B active)：
  - 权重 ~32 GB，但推理 FLOPS 只需 ~2.4B 参数的计算量
  - 适合 batch 较大的高吞吐场景
  
- DeepSeek-V3 (671B MoE)：
  - 需要多节点分布式推理
  - MLA 使得 KV Cache 非常紧凑，可以设较大 max_model_len
  - Expert parallel 是核心并行策略（Week 3 详解）

### GLM-4-9B

- 权重 ~18 GB (FP16)
- kv_heads=2 (GQA)，KV Cache 非常紧凑
- 在 24 GB GPU 上可以设 max_model_len=8192, max_num_seqs=64

---

## Part 5: 系统化实验方法

### 固定变量法

每次只变一个参数，其他保持不变：

```
Baseline:
  model=Qwen2.5-7B, gpu_util=0.9, max_model_len=8192, max_num_seqs=64
  workload: input=512, output=128, request_rate=20

实验 1: 变 gpu_memory_utilization
  0.80, 0.85, 0.90, 0.95 → 记录 blocks, throughput, TTFT, TPOT

实验 2: 变 max_model_len
  2048, 4096, 8192, 16384 → 同上

实验 3: 变 max_num_seqs
  8, 32, 64, 128, 256 → 同上

实验 4: 变 max_num_batched_tokens
  2048, 4096, 8192, 16384 → 同上
```

### 记录模板

| 参数 | 值 | GPU Blocks | TTFT P50 | TTFT P99 | TPOT P50 | TPOT P99 | Throughput | Preemptions | OOM? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| gpu_util | 0.80 | ? | ? | ? | ? | ? | ? | ? | ? |
| gpu_util | 0.85 | ? | ? | ? | ? | ? | ? | ? | ? |
| ... | | | | | | | | | |

---

## Part 6: 常见问题和诊断

### OOM (Out of Memory)

```
RuntimeError: CUDA out of memory
```
- 降低 `--gpu-memory-utilization`
- 降低 `--max-model-len`
- 降低 `--max-num-seqs`
- 使用量化模型

### TTFT 太高

诊断路径：
1. 检查 `vllm:num_requests_waiting` → 是否排队
2. 检查 input_len → prefill 是否太重
3. 检查是否开启了 prefix caching（相同前缀的请求应该更快）
4. 检查是否开启了 chunked prefill（长 prompt 是否阻塞了其他请求）

### TPOT 不稳定（P99 远高于 P50）

诊断路径：
1. 检查是否有长 prompt 的 prefill 穿插在 decode 中 → 开启 chunked prefill
2. 检查 KV Cache 使用率是否接近 100% → 降低并发或增大显存
3. 检查是否发生了 preemption → `vllm:num_preemptions_total`

### Throughput 上不去

诊断路径：
1. 检查 `max_num_seqs` 是否太小 → batch 太小无法充分利用 GPU
2. 检查 GPU utilization → 是否有 CPU 端瓶颈
3. 检查 KV Cache 是否是瓶颈 → 增大 gpu_memory_utilization 或用量化减少权重显存

---

## 交付物

| 文件 | 描述 |
|---|---|
| `vLLM 参数调优矩阵 v1.md` | 4 组实验（每组 4-5 个参数值）的完整数据表和结论 |

## 自检问题

1. gpu_memory_utilization 从 0.85 调到 0.95 会发生什么？风险是什么？
2. 为什么 max_model_len 不应该设成模型最大支持值？
3. max_num_seqs=8 和 max_num_seqs=256 在 decode 阶段的 GPU 利用率有什么区别？
4. 一个 RAG 场景（长 input，短 output，大量重复 system prompt）应该怎么配参数？
5. 如果 throughput 不随 max_num_seqs 增加而增加，瓶颈可能在哪里？
