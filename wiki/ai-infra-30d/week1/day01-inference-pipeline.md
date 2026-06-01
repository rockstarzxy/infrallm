---
title: "Day 1: 推理系统全景和 Transformer 推理路径"
type: concept
tags: [day1, inference-pipeline, kv-cache, prefill, decode]
created: 2026-06-01
updated: 2026-06-01
---

# Day 1: 推理系统全景和 Transformer 推理路径

## Part 1: LLM 推理的完整生命周期

一个 LLM 推理请求从进入系统到返回结果，经过以下阶段：

```
HTTP Request
    ↓
1. API Server 接收请求，解析参数（model, messages, max_tokens, temperature...）
    ↓
2. Tokenizer 编码：将 text 转为 token IDs
    ↓
3. Scheduler 调度：决定这个请求何时上 GPU 执行
    ↓
4. Prefill 阶段：一次性处理所有 input tokens，生成 KV Cache
    ↓
5. Decode 循环：每步生成 1 个 token，更新 KV Cache
    ↓  （重复直到遇到 EOS 或达到 max_tokens）
6. Sampling：从 logits 中采样下一个 token（top-k, top-p, temperature）
    ↓
7. Detokenize：token ID → text
    ↓
8. Stream / 返回完整响应
```

### 关键区分：Prefill vs Decode

这是推理优化中最基础也最重要的概念划分。

**Prefill 阶段（首次前向传播）：**
- 输入：完整的 prompt（所有 input tokens）
- 计算：对所有 input tokens 并行做 attention 计算
- 输出：所有层的 KV Cache + 第一个 output token 的 logits
- 计算特征：大矩阵乘法，**通常是 compute-bound**
- 类比：一次性"理解"整个问题

**Decode 阶段（自回归生成）：**
- 输入：上一步生成的 1 个 token
- 计算：这 1 个 token 和之前所有 token 的 KV Cache 做 attention
- 输出：下一个 token 的 logits
- 计算特征：batch size 很小（通常是 1 个 token），矩阵运算退化为矩阵-向量乘法，**通常是 memory-bandwidth-bound**
- 类比：一个字一个字地"写答案"

**为什么 decode 是 memory-bound？**

在 decode 阶段，每一步需要：
1. 从 HBM 加载模型权重（几 GB 到几十 GB）
2. 从 HBM 加载 KV Cache（随序列长度增长）
3. 但实际计算量很小：只处理 1 个 token 的矩阵-向量乘法

GPU 的算力远超这点计算量的需求，瓶颈在于"数据从显存搬到计算单元"的速度——即 HBM 带宽。

量化理解：
- H100 的 HBM 带宽：~3.35 TB/s
- Llama-3-8B FP16 权重：~16 GB
- 单次 decode 至少读一遍权重：16 GB / 3.35 TB/s ≈ 4.8 ms/token
- 如果不做 batching，单卡理论上限约 ~208 tokens/s（受带宽限制，和算力无关）

这就是为什么 **batching 对 decode 阶段至关重要**：多个请求共享一次权重加载，均摊带宽开销。

---

## Part 2: Self-Attention 计算流程

### 标准 Attention

给定输入 X（shape: [seq_len, d_model]），计算过程如下：

```
Q = X @ W_Q    # [seq_len, d_model] @ [d_model, d_head * n_heads] → [seq_len, n_heads * d_head]
K = X @ W_K    # 同上
V = X @ W_V    # 同上

# 重塑为多头：[n_heads, seq_len, d_head]

# 对每个 head：
scores = Q @ K^T / sqrt(d_head)   # [seq_len, seq_len]  ← O(n²) 在这里
attn_weights = softmax(scores)     # [seq_len, seq_len]
output = attn_weights @ V          # [seq_len, d_head]

# 合并多头 → 输出投影
output = concat(all heads) @ W_O   # [seq_len, d_model]
```

计算复杂度：O(n² · d) 其中 n 是序列长度，d 是 head 维度。

### KV Cache 机制

在 decode 阶段，每步只新增 1 个 token。如果每次都重新计算所有 token 的 attention，计算量会随序列增长而重复浪费。

KV Cache 的核心思路：**缓存之前所有 token 的 K 和 V，新 token 只需计算自己的 Q、K、V，然后和缓存的 K/V 做 attention。**

```
# Decode step t：
# 新 token x_t 的投影：
q_t = x_t @ W_Q    # [1, d_head * n_heads]
k_t = x_t @ W_K    # [1, d_head * n_heads]
v_t = x_t @ W_V

# 追加到 cache：
K_cache = concat(K_cache, k_t)   # [t, d_head * n_heads]
V_cache = concat(V_cache, v_t)

# Attention：
scores = q_t @ K_cache^T / sqrt(d_head)  # [1, t]
output = softmax(scores) @ V_cache        # [1, d_head]
```

好处：避免对前 t-1 个 token 重复计算 Q、K、V 投影。
代价：需要在 GPU 显存中保存 K 和 V，且显存占用随序列长度线性增长。

---

## Part 3: MQA 和 GQA

标准 Multi-Head Attention (MHA) 中，每个 head 有独立的 Q、K、V。
KV Cache 大小 = 2（K和V） × n_heads × d_head × seq_len × batch × bytes_per_element。

**Multi-Query Attention (MQA)：**
- 所有 Q head 共享 **1 组** K 和 V
- KV Cache 缩小为原来的 1/n_heads
- 代价：质量可能略有下降

**Grouped-Query Attention (GQA)：**
- 将 Q heads 分成 g 组，每组共享 1 组 K/V
- KV Cache 缩小为原来的 g/n_heads
- 是 MHA 和 MQA 之间的折中
- Llama-3 系列使用 GQA

```
MHA:  每个 Q head 有自己的 K/V head    → n_kv_heads = n_q_heads
GQA:  多个 Q head 共享一组 K/V          → n_kv_heads < n_q_heads（如 8）
MQA:  所有 Q head 共享同一组 K/V        → n_kv_heads = 1
```

**Llama-3-8B 的具体参数：**
- n_q_heads = 32, n_kv_heads = 8 (GQA, 4 个 Q head 共享 1 个 KV head)
- d_head = 128
- n_layers = 32

---

## Part 4: KV Cache 显存计算

### 公式

```
KV Cache 显存 = 2 × n_layers × n_kv_heads × d_head × seq_len × batch_size × bytes_per_element
```

- 2：K 和 V 各一份
- n_kv_heads：GQA 下是 KV head 数（不是 Q head 数）
- bytes_per_element：FP16 = 2 bytes, FP8 = 1 byte, INT8 = 1 byte

### 计算示例

**Llama-3-8B (FP16)：**
- 参数：layers=32, kv_heads=8, head_dim=128, dtype=FP16(2 bytes)

| seq_len | batch=1 | batch=8 | batch=32 | batch=128 |
|---:|---:|---:|---:|---:|
| 1,024 | 128 MB | 1.0 GB | 4.0 GB | 16.0 GB |
| 4,096 | 512 MB | 4.0 GB | 16.0 GB | 64.0 GB |
| 8,192 | 1.0 GB | 8.0 GB | 32.0 GB | 128.0 GB |
| 32,768 | 4.0 GB | 32.0 GB | 128.0 GB | 512.0 GB |
| 131,072 | 16.0 GB | 128.0 GB | 512.0 GB | 2.0 TB |

单位换算：2 × 32 × 8 × 128 × 2 = 131,072 bytes per token per batch element = **128 KB/token**

**Llama-3-70B (FP16)：**
- 参数：layers=80, kv_heads=8, head_dim=128, dtype=FP16(2 bytes)
- Per token: 2 × 80 × 8 × 128 × 2 = 327,680 bytes = **320 KB/token**

| seq_len | batch=1 | batch=8 | batch=32 |
|---:|---:|---:|---:|
| 4,096 | 1.25 GB | 10.0 GB | 40.0 GB |
| 8,192 | 2.5 GB | 20.0 GB | 80.0 GB |

**关键结论：**
- KV Cache 显存可以轻松超过模型权重显存
- 长上下文 + 大 batch 时 KV Cache 是显存瓶颈的主要来源
- 这就是 PagedAttention（Day 4）和 KV Cache 量化（Day 10）要解决的问题

### 模型权重 vs KV Cache 显存对比

| 模型 | 权重显存 (FP16) | 1 个 seq_len=4096 的请求 KV Cache |
|---|---:|---:|
| Llama-3-8B | ~16 GB | 0.5 GB |
| Llama-3-70B | ~140 GB | 1.25 GB |

单个请求看起来 KV Cache 不大，但当并发 64 个请求时：
- 8B: 64 × 0.5 GB = 32 GB —— 已经是权重的 2 倍
- 70B: 64 × 1.25 GB = 80 GB —— 超过单卡显存

---

## Part 5: 模型前向传播的完整计算链

以一个标准的 Decoder-only Transformer（如 Llama）为例，一个 layer 的计算：

```
Input x  →  RMSNorm
         →  Self-Attention (QKV projection → attention → output projection)
         →  Residual Add
         →  RMSNorm
         →  FFN (gate_proj + up_proj → SiLU → down_proj)
         →  Residual Add
         →  Output x'
```

**FFN（SwiGLU variant）的计算：**
```
gate = x @ W_gate          # [seq_len, d_model] @ [d_model, d_ffn]
up   = x @ W_up            # 同上
hidden = SiLU(gate) * up   # element-wise
output = hidden @ W_down   # [seq_len, d_ffn] @ [d_ffn, d_model]
```

完整模型：N 个这样的层 + 最终的 RMSNorm + LM Head（vocabulary projection）。

---

## 交付物模板

### 推理链路图.md

画出从 HTTP request 到 token stream 的完整路径，标注每一步的：
- 计算类型（CPU / GPU）
- 数据格式（text / token IDs / tensors）
- 显存操作（allocate / read / write）
- 是否有批处理（single request vs batched）

### KV Cache 显存估算表.md

用上面的公式计算你计划使用的模型在不同配置下的 KV Cache 显存。重点关注：
- 你的 GPU 有多少显存？
- 扣除模型权重后，能容纳多少 KV Cache？
- 这决定了能同时处理多少并发请求？

---

## 自检问题

1. 一个 request 从到达服务到返回第一个 token，经历了哪些步骤？
2. prefill 阶段为什么可以并行处理所有 input token？
3. decode 阶段为什么必须逐 token 生成？
4. Llama-3-8B 在 seq_len=8192、batch=32 时 KV Cache 占多少显存？（答：32 GB）
5. GQA 把 kv_heads 从 32 降到 8，KV Cache 节省了多少？（答：75%）
6. 为什么说 decode 阶段是 memory-bandwidth-bound 而不是 compute-bound？
7. 如果不做 batching，单卡 H100 上 Llama-3-8B 的 decode 吞吐大约是多少 tokens/s？
