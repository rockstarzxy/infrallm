---
title: "Day 16: Tensor Parallelism"
type: concept
tags: [day16, tensor-parallel, megatron, column-parallel, row-parallel]
created: 2026-06-01
updated: 2026-06-01
---

# Day 16: Tensor Parallelism

## Part 1: TP 的核心思想

Tensor Parallelism 将单个矩阵运算切分到多个 GPU 上并行计算。

```
不用 TP:
  GPU 0: 完整的 W [4096, 4096] × X → Y

TP=2:
  GPU 0: W 的前半列 [4096, 2048] × X → Y_part0
  GPU 1: W 的后半列 [4096, 2048] × X → Y_part1
  → AllReduce or Concat 得到完整 Y
```

每个 GPU 只需要存储 1/TP 的权重和计算 1/TP 的 FLOPS。

---

## Part 2: Megatron-LM 风格的 TP

vLLM 采用了 Megatron-LM 提出的 TP 策略。核心是两种切分方式：

### Column Parallel Linear

```
矩阵 W 按列切分：

W = [W_1 | W_2]     (列切分为 2 份)

GPU 0: Y_1 = X @ W_1    → Y_1 是输出的前半部分
GPU 1: Y_2 = X @ W_2    → Y_2 是输出的后半部分

Y = [Y_1, Y_2]          → 拼接（或直接用局部结果继续计算）
```

用途：QKV projection、FFN 的 gate/up projection

### Row Parallel Linear

```
矩阵 W 按行切分：

W = [W_1]               (行切分为 2 份)
    [W_2]

X = [X_1, X_2]          (输入也要切分)

GPU 0: Y_0 = X_1 @ W_1
GPU 1: Y_1 = X_2 @ W_2

Y = Y_0 + Y_1           → AllReduce(SUM)
```

用途：Attention 的 output projection、FFN 的 down projection

### 一个 Transformer Layer 的 TP 切分

```
Input X (每个 GPU 都有完整副本)
    │
    ↓
┌─── Column Parallel ───┐
│ QKV Projection        │  W_qkv 按列切分
│ GPU0: Q0,K0,V0        │  每个 GPU 得到部分 Q/K/V heads
│ GPU1: Q1,K1,V1        │
└───────────────────────┘
    │
    ↓
┌─── 独立计算 ──────────┐
│ Self-Attention        │  每个 GPU 独立计算自己的 heads
│ GPU0: Attn(Q0,K0,V0)  │  不需要通信！（GQA 下每个 GPU 有自己的 KV heads）
│ GPU1: Attn(Q1,K1,V1)  │
└───────────────────────┘
    │
    ↓
┌─── Row Parallel ──────┐
│ Output Projection     │  W_o 按行切分
│ GPU0: partial_output  │
│ GPU1: partial_output  │
│ ── AllReduce(SUM) ──  │  ← 第一次通信
└───────────────────────┘
    │
    ↓
    RMSNorm + Residual
    │
    ↓
┌─── Column Parallel ───┐
│ FFN gate + up proj    │  W_gate 和 W_up 按列切分
│ GPU0: gate0, up0      │
│ GPU1: gate1, up1      │
│ SiLU(gate) * up       │  每个 GPU 独立计算
└───────────────────────┘
    │
    ↓
┌─── Row Parallel ──────┐
│ FFN down proj         │  W_down 按行切分
│ ── AllReduce(SUM) ──  │  ← 第二次通信
└───────────────────────┘
    │
    ↓
    RMSNorm + Residual
    │
    ↓
Output X' (每个 GPU 都有完整副本)
```

关键点：**每个 Transformer layer 只需要 2 次 AllReduce 通信**。

---

## Part 3: GQA 在 TP 下的处理

GQA 使得 KV heads 数量少于 Q heads，TP 切分时需要注意：

```
Llama-3-8B: Q heads = 32, KV heads = 8

TP=4:
  每个 GPU: 32/4 = 8 个 Q heads, 8/4 = 2 个 KV heads
  → 每 4 个 Q head 共享 1 个 KV head
  → 没问题

TP=8:
  每个 GPU: 32/8 = 4 个 Q heads, 8/8 = 1 个 KV head
  → 还是可以的

TP=16:
  每个 GPU: 32/16 = 2 个 Q heads, 8/16 = 0.5 个 KV head
  → 不整除！需要广播 KV heads
  → 可能需要特殊处理
```

经验法则：**TP size 不应超过 KV heads 的数量**。

中国模型的 KV heads 数量：

| 模型 | KV Heads | 最大推荐 TP |
|---|---:|---:|
| Qwen2.5-7B | 4 | 4 |
| Qwen2.5-72B | 8 | 8 |
| DeepSeek-V2 (MLA) | 特殊 | 取决于 latent dim |
| GLM-4-9B | 2 | 2 |
| Llama-3-8B | 8 | 8 |

---

## Part 4: TP 对性能的影响

### 显存

```
TP=N 下每个 GPU 的显存:
  模型权重: total_params / N
  KV Cache: 也可以分（每个 GPU 只存自己负责的 heads）
  计算缓冲: ~不变（受 batch size 影响）
```

### 吞吐和延迟

```
理想 TP=N:
  prefill FLOPS/GPU: 1/N → prefill 快 N 倍
  decode: 权重量 1/N → 带宽读取 1/N → decode 快 N 倍

实际:
  有 AllReduce 通信开销
  decode 阶段通信比例更大（计算量小，通信量不变）

典型加速比:
  TP=2: ~1.7-1.9x (NVLink)
  TP=4: ~3.2-3.6x (NVLink)
  TP=8: ~5.5-6.5x (NVLink)
  TP=8 (PCIe): ~2-3x （通信瓶颈严重）
```

### 什么时候增加 TP 不值得

```
如果目标是吞吐（不是延迟）:
  TP=4 → 吞吐 T
  TP=8 → 吞吐 1.5T（因为 AllReduce 开销增加）

而 2 个 TP=4 实例 → 吞吐 2T

结论: 超过某个 TP 后，开多实例（DP）比增大 TP 更划算
```

---

## Part 5: 在 vLLM 中使用 TP

```bash
# TP=2
vllm serve Qwen/Qwen2.5-14B-Instruct --tensor-parallel-size 2

# TP=4
vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4

# TP=8
vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 8

# 指定 GPU（注意选择有 NVLink 直连的 GPU）
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4
```

### 启动日志中的 TP 信息

```
INFO: Using tensor parallel size 4
INFO: Model weights loaded in 4 GPU(s)
INFO: GPU 0: Qwen2ForCausalLM layer 0-19 (partial)
INFO: GPU 1: Qwen2ForCausalLM layer 0-19 (partial)
...
```

### 实验：TP size 对比

```bash
for tp in 1 2 4 8; do
  echo "=== TP=$tp ==="
  vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size $tp \
    --max-model-len 4096 &
  sleep 60

  python benchmarks/benchmark_serving.py \
    --backend vllm --model Qwen/Qwen2.5-72B-Instruct \
    --endpoint /v1/completions --dataset-name random \
    --random-input-len 512 --random-output-len 128 \
    --num-prompts 200 --request-rate 10

  kill %1 && sleep 10
done
```

---

## Part 6: DeepSeek-V3 的 TP 特殊性

DeepSeek-V3 是 MoE + MLA 架构，TP 策略有特殊考量：

```
Dense attention layers:
  MLA 的 latent 维度决定了 TP 的切分方式
  → 需要确保 latent projection 的矩阵能被 TP size 整除

MoE FFN layers:
  每个 expert 可以独立放在不同 GPU 上（Expert Parallelism）
  → 不一定用 TP 来切分 MoE 层
  → 更常见的做法: attention 用 TP, FFN 用 EP

通信模式:
  Attention: AllReduce（TP 通信）
  MoE FFN: All-to-All（EP 通信）
  → 两种通信交替出现
```

---

## 交付物

| 文件 | 描述 |
|---|---|
| `tensor-parallel-report.md` | TP=1/2/4/8 的性能对比（或理论分析） |

## 自检问题

1. Column Parallel 和 Row Parallel 各用在哪些层？为什么？
2. 每个 Transformer layer 有几次 AllReduce？通信量是多少？
3. TP size 超过 KV heads 数量会怎样？
4. 为什么 TP=8 的加速比通常达不到 8x？
5. 什么时候应该用 2 个 TP=4 实例，而不是 1 个 TP=8 实例？
