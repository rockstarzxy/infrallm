---
title: "Day 27: 长上下文推理"
type: concept
tags: [day27, long-context, kv-offload, ring-attention, sparse-attention]
created: 2026-06-01
updated: 2026-06-01
---

# Day 27: 长上下文推理

## Part 1: 长上下文的挑战

模型支持的上下文越来越长（Qwen2.5 支持 128K，DeepSeek-V3 也支持 128K+），但推理面临三重挑战：

```
挑战 1: KV Cache 显存线性增长
  128K context, Qwen2.5-72B (FP16):
  KV Cache = 128000 × 320 KB = 40 GB → 单个请求就占满半张卡

挑战 2: Attention 计算量二次增长
  Prefill: attention 的 FLOPS ∝ O(N²·d)
  128K 的 prefill 比 4K 的 prefill 慢 ~1024x (128K/4K)² = 1024

挑战 3: Prefill TTFT 爆炸
  128K prompt 的 prefill 可能需要数秒到数十秒
  → 用户等待时间不可接受
```

---

## Part 2: 解决方案总览

```
                   显存问题 ────┐
                                ↓
            ┌──────────────────────────────────┐
            │     KV Cache 压缩/卸载           │
            │  FP8/INT8 KV ← 压缩精度         │
            │  CPU Offload ← 放到 CPU          │
            │  NVMe Offload ← 放到 SSD         │
            │  MLA (DeepSeek) ← 训练时就压缩    │
            └──────────────────────────────────┘

                   计算问题 ────┐
                                ↓
            ┌──────────────────────────────────┐
            │     注意力计算优化                 │
            │  FlashAttention ← 减少 IO         │
            │  Ring Attention ← 分布式 context  │
            │  Sparse Attention ← 只算一部分     │
            │  Chunked Prefill ← 分段处理       │
            └──────────────────────────────────┘

                   TTFT 问题 ────┐
                                ↓
            ┌──────────────────────────────────┐
            │     预填充优化                     │
            │  Prefix Caching ← 复用已计算部分  │
            │  Chunked Prefill ← 分段不阻塞    │
            │  Disaggregated ← 专用 prefill GPU │
            └──────────────────────────────────┘
```

---

## Part 3: KV Cache 分层存储

### GPU → CPU Offload

```
策略: 将不活跃 token 的 KV Cache 移到 CPU 内存

GPU 显存: 放最近的 token (active window)
CPU 内存: 放较早的 token

当 attention 需要早期 token 的 KV:
  从 CPU 传回 GPU → 计算 → 计算完再放回 CPU

开销:
  PCIe Gen5: 64 GB/s
  传 1 GB KV Cache: ~16 ms
  → decode 每步都需要所有 KV → 每步都要传 → 不可行？

优化: pipeline
  1. 当前 layer 计算的同时，预取下一层的 KV
  2. 计算和传输 overlap
  → 如果传输时间 < 计算时间，overhead 接近 0
```

### GPU → NVMe Offload

```
NVMe SSD: ~7 GB/s (比 PCIe 慢很多)
适合: 极端长上下文 + 低 QPS + batch=1

不适合在线 serving，但适合:
  - 研究用的超长文档分析
  - 离线批量处理
```

### DeepSeek MLA 的天然优势

```
标准 GQA (Qwen2.5-72B):
  KV/token = 320 KB → 128K context = 40 GB

DeepSeek-V3 MLA:
  KV latent/token ≈ 160 KB → 128K context = 20 GB
  → 减半！

MLA 是目前处理长上下文 KV Cache 最优雅的方案之一
因为压缩是在训练时学到的，不会损失精度
```

---

## Part 4: Ring Attention

### 问题

```
128K context 的 attention 计算:
  Q: [128K, d]  ×  K^T: [d, 128K]  →  [128K, 128K]
  
这个 128K × 128K 的矩阵:
  - FP16: 128K × 128K × 2 = 32 GB → 无法放入 SRAM
  - FlashAttention 通过 tiling 解决了 SRAM 问题
  - 但 KV Cache 仍然需要 40 GB → 可能超单卡显存
```

### Ring Attention 原理

```
将 context 分布到多个 GPU 上，每个 GPU 持有一部分 KV：

4 GPU, 128K context:
  GPU 0: KV[0:32K]
  GPU 1: KV[32K:64K]  
  GPU 2: KV[64K:96K]
  GPU 3: KV[96K:128K]

计算过程（ring 结构）:
  Step 1: 每个 GPU 计算本地 Q 和本地 KV 的 attention
  Step 2: 每个 GPU 将 KV 传给下一个 GPU（ring）
  Step 3: 每个 GPU 用新收到的 KV 更新 attention（online softmax）
  Step 4: 重复直到所有 KV 轮转一圈

GPU 0: local → recv from GPU 3 → recv from GPU 2 → recv from GPU 1
GPU 1: local → recv from GPU 0 → recv from GPU 3 → recv from GPU 2
...

通信和计算 overlap:
  当 GPU 0 在用 GPU 3 的 KV 做 attention 时
  同时从 GPU 2 接收下一批 KV
  → 如果 compute time > transfer time → 通信完全隐藏
```

---

## Part 5: Sparse Attention 方案

### Sliding Window Attention

```
Mistral 风格:
  每个 token 只 attend to 最近 W 个 token
  KV Cache 大小固定为 W × KV_per_token

适合: 模型训练时就使用了 sliding window
不适合: 需要引用远距离信息的任务（如"文档开头说了什么？"）
```

### StreamingLLM (Attention Sink)

```
保留:
  [前几个 token (attention sink)] + [最近 N 个 token]

效果:
  KV Cache 从 O(seq_len) 降到 O(N + sink_size) = O(1)
  可以处理无限长上下文（理论上）

限制:
  丢弃的中间 token 信息不可恢复
  → 如果问"第 50 段说了什么？"答不出来
  → 适合连续生成/对话，不适合文档 QA
```

### H2O (Heavy-Hitter Oracle)

```
动态保留最重要的 token:
  每层独立维护 top-k attention score 的 token
  定期驱逐 score 最低的 token

比 StreamingLLM 更智能（保留重要内容而不是只保留最近的）
但实现更复杂，overhead 更大
```

---

## Part 6: Chunked Prefill 在长上下文中的作用

```
128K 的 prefill 如果一次性做:
  耗时 ~5-10 秒（取决于模型和硬件）
  → 其他所有 decode 请求被阻塞 5-10 秒
  → 所有用户感受到卡顿

Chunked Prefill (chunk_size=4096):
  128K / 4096 = 32 个 chunk
  每个 chunk 在一个 iteration 中和 decode 请求一起执行
  → decode 请求的 TPOT 保持稳定
  → 长 prompt 的 TTFT = 32 个 iteration 的时间
```

---

## Part 7: 长上下文场景的实践建议

| 上下文长度 | 推荐策略 | 理由 |
|---|---|---|
| < 4K | 标准配置 | 不需要特殊优化 |
| 4K - 32K | prefix caching + chunked prefill | 管理 TTFT 和 KV Cache |
| 32K - 128K | FP8 KV + prefix caching + chunked prefill + 限制并发 | KV Cache 压力大 |
| 128K+ | Ring Attention 或 KV offload + sparse attention | 需要分布式或近似方案 |

中国模型的长上下文支持：

| 模型 | 声明的最大上下文 | 推理实用建议 |
|---|---|---|
| Qwen2.5-72B | 128K | 实际使用建议 < 32K 以保证并发 |
| DeepSeek-V3 | 128K | MLA 使得长上下文更可行 |
| GLM-4-9B | 128K | 模型小，KV Cache 压力小 |
| MiniMax-Text-01 | 1M+ (号称) | Lightning Attention，O(1) KV |

---

## 交付物

| 文件 | 描述 |
|---|---|
| `long-context-inference-notes.md` | 长上下文挑战 + KV 管理策略 + 按长度的配置建议 |

## 自检问题

1. 128K context 的 KV Cache 对 Qwen2.5-72B 需要多少显存？
2. Ring Attention 如何将 KV Cache 分布到多个 GPU？
3. StreamingLLM 的 attention sink 是什么？为什么保留前几个 token 很重要？
4. Chunked prefill 对长 prompt 的 TTFT 有什么影响？
5. DeepSeek 的 MLA 为什么特别适合长上下文推理？
