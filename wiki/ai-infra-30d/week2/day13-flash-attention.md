---
title: "Day 13: Attention 优化和 FlashAttention"
type: concept
tags: [day13, flash-attention, io-aware, attention, tiling]
created: 2026-06-01
updated: 2026-06-01
---

# Day 13: Attention 优化和 FlashAttention

## Part 1: 标准 Attention 的 IO 瓶颈

回顾标准 attention 的计算：

```python
# 标准实现（PyTorch 风格）
S = Q @ K.T / sqrt(d)    # [N, N] — 注意力分数矩阵
P = softmax(S, dim=-1)    # [N, N] — 注意力权重
O = P @ V                 # [N, d] — 输出
```

其中 N = 序列长度，d = head dimension。

### 显存问题

S 和 P 矩阵的大小是 [N, N]：
- N = 2048, fp16: S + P = 2 × 2048² × 2 bytes = **16 MB** (per head)
- N = 8192, fp16: S + P = 2 × 8192² × 2 bytes = **256 MB** (per head)
- N = 131072, fp16: S + P = 2 × 131072² × 2 bytes = **64 GB** (per head) — 不可能存下！

即使存得下，这些中间矩阵需要在 HBM 和 SRAM 之间反复搬运。

### GPU 内存层次

```
┌───────────────────────────┐
│     SRAM (on-chip)        │  ← 快，小：~20 MB (H100)
│     - registers           │     带宽：~19 TB/s
│     - shared memory       │
├───────────────────────────┤
│     L2 Cache              │  ← 中等：~50 MB (H100)
├───────────────────────────┤
│     HBM (off-chip)        │  ← 慢，大：80 GB (H100)
│     - global memory       │     带宽：~3.35 TB/s
└───────────────────────────┘
```

SRAM 和 HBM 的带宽差距：~6x。

标准 attention 的 IO 流程：

```
1. 从 HBM 读 Q, K → 计算 S = Q @ K^T → 写 S 到 HBM
2. 从 HBM 读 S → 计算 P = softmax(S) → 写 P 到 HBM
3. 从 HBM 读 P, V → 计算 O = P @ V → 写 O 到 HBM

总 HBM 读写: O(N² + Nd) 
→ 序列越长，IO 开销越大
→ 算术密度 (FLOPS/byte) 很低 → memory-bound
```

---

## Part 2: FlashAttention 的核心思想

FlashAttention 的关键洞察：**不需要把完整的 N×N 矩阵写到 HBM，可以分块在 SRAM 中完成所有计算。**

### Tiling（分块）

将 Q, K, V 按序列维度分成小块（block），每个 block 能放进 SRAM：

```
Q 分成 T_q 个 block: Q_1, Q_2, ..., Q_{T_q}     每个大小 [B_q, d]
K 分成 T_k 个 block: K_1, K_2, ..., K_{T_k}     每个大小 [B_k, d]
V 分成 T_k 个 block: V_1, V_2, ..., V_{T_k}     每个大小 [B_k, d]

B_q, B_k 选择使得 block 能放进 SRAM
```

### Online Softmax

Tiling 的最大挑战是 softmax：标准 softmax 需要看到一整行的所有值才能计算。

```
标准 softmax:
  softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
  → 需要全局的 sum，不能分块计算？

Online softmax 的技巧:
  维护两个累积量: m (当前最大值) 和 l (当前 exp 之和)
  
  初始: m = -inf, l = 0
  
  处理 block_k:
    S_block = Q_block @ K_block^T
    m_new = max(m, max(S_block))
    l_new = l * exp(m - m_new) + sum(exp(S_block - m_new))
    O_new = O * (l * exp(m - m_new) / l_new) + exp(S_block - m_new) @ V_block / l_new
    m = m_new
    l = l_new

  每个 block 只需要 O(B_q × B_k) 的中间存储
  → 全部在 SRAM 中完成！
```

### 完整 FlashAttention 流程

```
For each Q_block (outer loop):
    初始化 O_block = 0, m = -inf, l = 0
    
    For each K_block, V_block (inner loop):
        1. 从 HBM 加载 Q_block, K_block, V_block 到 SRAM
        2. 在 SRAM 中计算 S = Q_block @ K_block^T
        3. 用 online softmax 更新 m, l, O_block
        4. 不需要写 S 或 P 回 HBM！
    
    将 O_block 写回 HBM

总 HBM 读写: O(N²d² / M)    其中 M 是 SRAM 大小
→ 比标准 attention 的 O(N² + Nd) 少很多
→ 节省的 IO ≈ 数倍到数十倍（取决于序列长度）
```

---

## Part 3: FlashAttention-2 的改进

FlashAttention-2 在 FlashAttention 的基础上进一步优化了并行性：

### 改进 1：减少非矩阵乘法的操作

```
FlashAttention-1: 在 SRAM 中做了大量 rescaling 操作（非矩阵乘法）
→ Tensor Core 空闲，只用了普通 CUDA core

FlashAttention-2: 重排计算顺序，减少 rescaling 步骤
→ 更多时间花在 Tensor Core 上的矩阵乘法
→ 更接近 GPU 峰值算力
```

### 改进 2：更好的并行化

```
FlashAttention-1: 外层循环在 K/V block 上（inner loop 在 Q block 上）
→ 反向传播需要额外的同步

FlashAttention-2: 外层循环在 Q block 上（inner loop 在 K/V block 上）
→ 每个 Q block 可以独立并行处理
→ warp 之间不需要共享中间结果
→ 更好的 GPU 利用率
```

### 改进 3：支持更多 head 维度

```
FlashAttention-1: 只支持 head_dim ≤ 128
FlashAttention-2: 支持 head_dim 可以是 64, 128, 256 等
→ 更好地适配不同模型架构
```

### 性能提升

FlashAttention-2 相比 FlashAttention-1 通常快 ~2x，相比标准 attention 快 ~5-10x（取决于序列长度）。

---

## Part 4: FlashDecoding

FlashAttention 在 prefill 阶段效果好（处理大量 token），但 decode 阶段呢？

### Decode 阶段的问题

```
Decode 时:
  Q: [1, d]              ← 只有 1 个 token 的 query
  K: [seq_len, d]         ← 整个 KV Cache
  V: [seq_len, d]

标准 FlashAttention 的并行是在 Q block 上做的
→ Q 只有 1 个 token → 没有 Q 维度可并行
→ GPU 很多 SM 闲着
```

### FlashDecoding 的解决方案

```
将 KV Cache 在序列维度上分成多个 split:

Split 1: K[0:1024],    V[0:1024]     → 并行
Split 2: K[1024:2048], V[1024:2048]  → 并行
Split 3: K[2048:3072], V[2048:3072]  → 并行
Split 4: K[3072:4096], V[3072:4096]  → 并行

每个 split 独立计算 partial attention (partial O, m, l)
最后用 online softmax 的性质合并所有 split 的结果
```

效果：decode 阶段也能充分利用 GPU 的所有 SM。

---

## Part 5: PagedAttention Kernel

PagedAttention 不只是一个内存管理策略，它也需要特殊的 attention kernel 来支持分页的 KV Cache。

### 标准 Attention Kernel 的假设

标准 FlashAttention kernel 假设 K 和 V 是**连续存储**的张量。

### PagedAttention Kernel 的区别

```
标准: K = contiguous tensor [seq_len, n_heads, d]
Paged: K 分散在不同的物理 block 中，通过 block table 索引

kernel 需要:
  1. 读取 block table → 知道第 i 个逻辑 block 在哪个物理地址
  2. 从不同物理地址收集 K/V 数据
  3. 计算 attention
```

vLLM 使用 FlashInfer 或定制的 PagedAttention kernel 来高效处理这种非连续的 KV 访问。

---

## Part 6: 不同模型架构对 Attention 优化的影响

### 标准 GQA 模型（Qwen2.5, Llama-3）

```
Q heads: 32, KV heads: 8 (GQA ratio = 4)
FlashAttention 可以直接使用
每个 KV head 被 4 个 Q head 共享 → K/V 只需读一次
→ GQA 天然减少了 attention 的 IO
```

### DeepSeek-V2/V3 (MLA)

```
MLA 的 attention 计算:
1. 从 cache 读取 compressed latent (小)
2. 解压为 K, V (expand)
3. 做标准 attention

FlashAttention 的适配:
- 解压后可以用标准 FlashAttention
- 或者实现 fused kernel: decompress + attention 一步完成
- vLLM/SGLang 都做了 MLA-specific 的 attention kernel
```

### MiniMax Lightning Attention

```
Lightning Attention 是线性注意力:
  O = Q @ (K^T @ V)    ← 先算 K^T @ V，复杂度 O(d²) 而非 O(N²)

不需要 FlashAttention 的 tiling 技巧
但有自己的精度和表达能力 trade-off
```

### Mistral Sliding Window Attention

```
每个 token 只 attend to 最近 W 个 token
Attention matrix 是 banded（带状）的:

  ████░░░░    ← 只有对角线附近有值
  ░████░░░
  ░░████░░
  ░░░████░

FlashAttention 可以利用这个结构跳过零值 block
→ 长序列性能更好（有效 attention 长度固定为 W）
```

---

## Part 7: Attention 优化对指标的影响

| 优化 | 对 Prefill 的影响 | 对 Decode 的影响 |
|---|---|---|
| FlashAttention | TTFT 大幅降低（尤其长序列） | 间接帮助（省显存给 KV Cache） |
| FlashDecoding | 无（只优化 decode） | TPOT 降低（更好的 GPU 利用率） |
| GQA | 加速（KV IO 减少） | 加速（KV Cache 更小） |
| MLA | 加速（KV 更紧凑） | 加速 + 更多并发 |
| PagedAttention kernel | 微小开销（block table 查找） | 同左，通常可忽略 |

---

## 交付物

| 文件 | 描述 |
|---|---|
| `attention-optimization-notes.md` | FlashAttention 原理笔记、不同 attention 变体对比 |

## 自检问题

1. 标准 attention 的 IO 瓶颈在哪里？为什么 N×N 的中间矩阵是问题？
2. FlashAttention 如何通过 tiling 减少 HBM 读写？
3. Online softmax 的核心技巧是什么？为什么它让分块成为可能？
4. FlashAttention-2 相比 v1 的主要改进是什么？
5. FlashDecoding 解决了 decode 阶段的什么并行性问题？
6. DeepSeek 的 MLA 需要什么样的 attention kernel 适配？
