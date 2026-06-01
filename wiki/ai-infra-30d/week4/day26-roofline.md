---
title: "Day 26: Roofline 和瓶颈判断"
type: concept
tags: [day26, roofline, compute-bound, memory-bound, arithmetic-intensity]
created: 2026-06-01
updated: 2026-06-01
---

# Day 26: Roofline 和瓶颈判断

## Part 1: Arithmetic Intensity

**Arithmetic Intensity (AI)** = FLOPS / Bytes (计算量 / 数据搬运量)

```
AI 高 → 每搬一个 byte 做很多计算 → compute-bound
AI 低 → 每搬一个 byte 做很少计算 → memory-bound

GPU 的"平衡点":
  peak_FLOPS / peak_bandwidth = 临界 AI

H100:
  FP16: 989 TFLOPS / 3.35 TB/s = 295 FLOPS/byte
  FP8:  1979 TFLOPS / 3.35 TB/s = 590 FLOPS/byte

含义: 如果 AI < 295 → memory-bound（H100 FP16）
      如果 AI > 295 → compute-bound
```

---

## Part 2: Roofline 模型

```
Performance (FLOPS)
    ↑
    |                    ╱ ← compute ceiling (989 TFLOPS)
    |                   ╱
    |                  ╱
    |                 ╱
    |                ╱──────────────── ← 实际 roofline
    |               ╱
    |              ╱  ← memory bandwidth ceiling
    |             ╱     (3.35 TB/s × AI)
    |            ╱
    |           ╱
    |          ╱
    |         ╱
    └────────┼──────────────────────→ Arithmetic Intensity (FLOPS/byte)
             ↑
           平衡点 (295 for H100 FP16)
```

读法：
- 左侧斜线区域：memory-bound（性能受限于带宽）
- 右侧水平线区域：compute-bound（性能受限于算力）
- 一个操作在图上的位置由它的 AI 决定

---

## Part 3: 推理中各操作的 AI 分析

### GEMM (矩阵乘法)

```
C = A × B
A: [M, K],  B: [K, N],  C: [M, N]

FLOPS = 2 × M × N × K
Bytes = (M×K + K×N + M×N) × bytes_per_element

AI = 2MNK / ((MK + KN + MN) × bytes)

当 M, N, K 都很大时:
  AI ≈ 2MNK / (2MK × bytes) = N / bytes
  → AI 随矩阵规模增加而增加 → 大 GEMM 是 compute-bound
  
当 M=1 时 (GEMV):
  AI = 2NK / ((K + NK + N) × bytes) ≈ 2 / bytes
  → AI 非常低 → GEMV 是 memory-bound
```

### Prefill vs Decode 的 AI

```
Prefill (seq_len = 2048):
  Linear layer: [2048, 4096] × [4096, 4096]
  FLOPS = 2 × 2048 × 4096 × 4096 = 68.7 GFLOPS
  Bytes = (2048×4096 + 4096×4096) × 2 = 50.3 MB
  AI = 68.7 GFLOPS / 50.3 MB = 1365 FLOPS/byte
  → 1365 >> 295 → compute-bound ✓

Decode (batch=1):
  Linear layer: [1, 4096] × [4096, 4096]
  FLOPS = 2 × 1 × 4096 × 4096 = 33.6 MFLOPS
  Bytes = (4096 + 4096×4096) × 2 = 33.6 MB
  AI = 33.6 MFLOPS / 33.6 MB = 1 FLOPS/byte
  → 1 << 295 → 极度 memory-bound ✓
```

### Batching 如何改变 AI

```
Decode (batch=B):
  Linear: [B, 4096] × [4096, 4096]
  FLOPS = 2 × B × 4096 × 4096
  Bytes ≈ 4096 × 4096 × 2 (权重，batch 共享) + B × 4096 × 2 (input) + B × 4096 × 2 (output)
        ≈ 33.6 MB + 很小

  AI ≈ 2B × 4096 × 4096 / 33.6 MB ≈ B FLOPS/byte

  B=1:   AI ≈ 1   → memory-bound
  B=32:  AI ≈ 32  → memory-bound
  B=128: AI ≈ 128 → memory-bound（接近转折点）
  B=295: AI ≈ 295 → 平衡点
  B=512: AI ≈ 512 → compute-bound

结论: 在 H100 FP16 上，batch 需要 ~300 才能让 decode 的 linear 层变为 compute-bound
```

### Attention 的 AI

```
Standard Attention:
  S = Q × K^T: [B×H, N, D] × [B×H, D, N] → [B×H, N, N]
  P × V:       [B×H, N, N] × [B×H, N, D] → [B×H, N, D]
  
  问题: N×N 的中间矩阵导致大量 HBM 读写 → AI 低 → memory-bound

FlashAttention:
  通过 tiling 减少 HBM 读写 → 有效 AI 大幅提升
  → 从 memory-bound 变为接近 compute-bound
```

---

## Part 4: 量化对 Roofline 的影响

```
FP16 → FP8:
  FLOPS 增加: 989 → 1979 TFLOPS
  Bandwidth 不变: 3.35 TB/s
  但数据量减半: 每个 byte 包含的信息量翻倍

新的平衡点: 1979 / 3.35 = 590 FLOPS/byte

实际效果:
  Decode batch=1:
    FP16: AI = 1 → 极度 memory-bound, 性能 = 1 × 3.35 TB/s = 3.35 GFLOPS
    FP8:  AI = 1 → 仍然 memory-bound, 但数据量减半
          → 性能 ≈ 2 × 3.35 GFLOPS = 6.7 GFLOPS → ~2x 加速 ✓

  Decode batch=256:
    FP16: AI = 256 → 接近 compute-bound
    FP8:  AI = 256 → 仍 memory-bound (平衡点 590)
          → 比 FP16 快但不到 2x

结论: 量化在 low-batch decode 场景的收益最大
```

---

## Part 5: GQA/MQA 对 Roofline 的影响

```
Attention 的 KV 读取量:
  MHA: 每步读取所有 KV heads → 大量 HBM 读取
  GQA: 读取 kv_heads / q_heads 的 KV → 减少读取量
  MQA: 只读 1 组 KV → 最少读取量

效果:
  GQA 通过减少 KV 的 Bytes → 提高了 attention 操作的 AI
  → attention 更接近 compute-bound
  → decode 阶段的 attention 效率更高

这就是为什么 GQA/MQA 对推理性能如此重要:
  不只是节省 KV Cache 显存
  还直接提升了 decode 阶段的计算效率
```

---

## Part 6: 用 Roofline 指导优化

```
判断流程:

1. 确定操作的 AI
2. 和 GPU 的平衡点比较
3. 选择优化方向:

如果 memory-bound (AI < 平衡点):
  → 减少数据搬运: 量化、kernel fusion、缓存
  → 增加 batch size: 提高 AI
  → 减少 KV 读取: GQA/MQA

如果 compute-bound (AI > 平衡点):
  → 优化计算: 更高效的 GEMM kernel
  → 使用 Tensor Core
  → 减少不必要的计算
  → 使用稀疏计算
```

---

## 交付物

| 文件 | 描述 |
|---|---|
| `roofline-inference-notes.md` | Roofline 模型应用 + prefill/decode 的 AI 计算 + 量化/GQA 对 roofline 的影响 |

## 自检问题

1. Arithmetic Intensity 的定义是什么？单位是什么？
2. H100 FP16 的 roofline 平衡点是多少？这个数字告诉你什么？
3. 为什么 decode batch=1 极度 memory-bound？需要多大 batch 才能转为 compute-bound？
4. 量化为什么在低 batch decode 场景收益最大？
5. GQA 除了节省 KV Cache 显存，还有什么性能好处？
