---
title: "Day 10: KV Cache 优化"
type: concept
tags: [day10, kv-cache, prefix-caching, chunked-prefill, quantization, eviction]
created: 2026-06-01
updated: 2026-06-01
---

# Day 10: KV Cache 优化

## Part 1: 为什么 KV Cache 是优化重点

KV Cache 是推理系统中**最重要的可优化资源**：
- 它的大小直接决定了并发能力（能同时处理多少请求）
- 它影响 TTFT（通过 prefix caching 减少重复 prefill）
- 它和显存形成零和博弈：KV Cache 用多了 → 留给计算缓冲的空间少了

今天覆盖四种 KV Cache 优化技术：
1. Prefix Caching（前缀缓存）
2. Chunked Prefill（分块预填充）
3. KV Cache Quantization（KV 缓存量化）
4. Eviction / Sparse Attention（淘汰/稀疏策略）

---

## Part 2: Prefix Caching (Automatic Prefix Caching, APC)

### 场景

大量请求共享相同前缀（system prompt / RAG context / few-shot examples）：

```
Request 1: [System: You are a helpful assistant.] + [User: What is AI?]
Request 2: [System: You are a helpful assistant.] + [User: Tell me a joke.]
Request 3: [System: You are a helpful assistant.] + [User: Summarize this doc...]
```

每个请求都重复计算相同 system prompt 的 prefill → 浪费计算 + 浪费 KV Cache 空间。

### 原理

Prefix caching 的核心思想：**如果多个请求有相同的 token 前缀，它们的 KV Cache 也完全相同，可以复用。**

```
没有 prefix caching:
  Req 1: compute_prefill([system_prompt] + [user_msg_1]) → KV Cache 1
  Req 2: compute_prefill([system_prompt] + [user_msg_2]) → KV Cache 2
  Req 3: compute_prefill([system_prompt] + [user_msg_3]) → KV Cache 3
  ← system_prompt 的 prefill 做了 3 次

有 prefix caching:
  Req 1: compute_prefill([system_prompt]) → cached_KV_prefix
         compute_prefill([user_msg_1], reuse cached_KV_prefix) → KV Cache 1
  Req 2: reuse cached_KV_prefix → skip prefill for system_prompt!
         compute_prefill([user_msg_2]) → KV Cache 2
  Req 3: reuse cached_KV_prefix → skip prefill for system_prompt!
         compute_prefill([user_msg_3]) → KV Cache 3
  ← system_prompt 的 prefill 只做了 1 次
```

### vLLM 中的实现

vLLM 的 prefix caching 通过 **hash-based block matching** 实现：

1. 将 token sequence 按 block_size 分成若干 block
2. 对每个 block 的 token 内容以及此前前缀信息计算 hash
3. 如果 hash 匹配 → 直接复用已有的物理 block（不需要重新计算）

```
Block 0: hash("You are a helpful") → 0xABCD → 物理 block 42
Block 1: hash(" assistant.\nUser:") → 0x1234 → 物理 block 17

下一个请求的 Block 0 也是 "You are a helpful" → hash = 0xABCD → 命中！复用物理 block 42
```

### 启用和测试

```bash
# 启用 prefix caching
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-prefix-caching
```

```python
# 测试脚本：对比有无 prefix caching 的 TTFT
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

system_prompt = "You are an expert AI assistant. " * 200  # 长 system prompt

for i in range(5):
    t0 = time.time()
    resp = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question {i}: What is machine learning?"}
        ],
        max_tokens=32,
    )
    print(f"Request {i}: E2E latency = {(time.time()-t0)*1000:.0f}ms")
```

预期结果：
```
Request 0: 850ms   ← 第一次，完整 prefill
Request 1: 120ms   ← 命中 prefix cache，跳过 system prompt 的 prefill
Request 2: 115ms
Request 3: 118ms
Request 4: 112ms
```

### 适用场景分析

| 场景 | 效果 | 原因 |
|---|---|---|
| 固定 system prompt 的 chatbot | 非常好 | 所有请求共享相同前缀 |
| RAG (相同 context + 不同问题) | 好 | context 部分可复用 |
| Few-shot prompting (固定 examples) | 好 | examples 部分可复用 |
| 每个请求完全不同 | 无效果 | 没有共享前缀 |
| 多轮对话 (累积历史) | 部分有效 | 前几轮历史可复用，但每轮增加新内容 |

---

## Part 3: Chunked Prefill

### 问题回顾（Day 9）

长 prompt 的 prefill 会阻塞同一 iteration 中的 decode 请求，导致 TPOT 尖峰。

### 原理

将一次长 prefill 拆分为多个小 chunk，每个 chunk 在一个 iteration 中执行。

```
不分块：
  Iteration 1: [PREFILL: 8192 tokens全部处理] + [decode_A] + [decode_B]
  ↑ 这个 iteration 非常重，decode_A 和 decode_B 要等很久

分块（chunk_size=2048）：
  Iteration 1: [prefill_chunk1: 2048 tokens] + [decode_A] + [decode_B]
  Iteration 2: [prefill_chunk2: 2048 tokens] + [decode_A] + [decode_B]
  Iteration 3: [prefill_chunk3: 2048 tokens] + [decode_A] + [decode_B]
  Iteration 4: [prefill_chunk4: 2048 tokens] + [decode_A] + [decode_B]
  ↑ 每个 iteration 的计算量更均匀
```

### 启用与调参

vLLM V1 在模型支持时通常默认启用 chunked prefill。学习重点不是只会加开关，而是通过 `max_num_batched_tokens` 控制长 prefill 与 decode 请求的混排粒度。

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching
```

### Trade-off

| 方面 | 不分块 | 分块 |
|---|---|---|
| 长 prompt 的 TTFT | 低（一步完成） | 略高（多步完成） |
| decode 请求的 TPOT | 不稳定（有尖峰） | 稳定（均匀分摊） |
| ITL 方差 | 大 | 小 |
| GPU 利用率 | 可能有空隙 | 更均匀 |

---

## Part 4: KV Cache Quantization

### 思路

KV Cache 默认使用和模型权重相同的精度（FP16/BF16）。如果将 KV Cache 量化到更低精度，可以在相同显存下存更多 token。

```
FP16 KV Cache: 每个 element 2 bytes
FP8 KV Cache:  每个 element 1 byte → 同样显存下能存 2x 的 KV Cache
INT8 KV Cache: 同上
INT4 KV Cache: 每个 element 0.5 bytes → 4x 的 KV Cache
```

### 实际效果

以 Qwen2.5-7B 为例（KV per token = 56 KB in FP16）：

| KV 精度 | Per token | 80GB GPU 可缓存 token 数 (扣除权重后) |
|---|---:|---:|
| FP16 | 56 KB | ~1,071,000 |
| FP8 | 28 KB | ~2,142,000 |

KV Cache 量化的难点：
- KV 的数值范围在不同 layer 和不同 head 之间差异大
- 量化带来的精度损失可能影响生成质量（尤其长序列末端）
- 不是所有模型都能容忍 KV 量化

### 在 vLLM 中使用

```bash
# FP8 KV Cache（H100 及以上支持）
vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype fp8
```

---

## Part 5: Eviction 和 Sparse Attention 策略

当 KV Cache 实在不够用时，还有一些更激进的策略：

### StreamingLLM (Attention Sink)

观察：在生成很长序列时，attention score 分布有一个规律——**第一个 token（BOS）总是获得很高的 attention score**，即使它没有语义意义。这被称为 "attention sink"。

StreamingLLM 的策略：
```
保留 KV Cache:
  [前几个 token (attention sink)] + [最近 N 个 token (sliding window)]
  丢弃中间的 token

例：保留前 4 个 + 最近 1024 个，丢弃中间
  seq_len=10000 时 KV Cache 只需存 1028 个 token
```

局限：丢弃的 token 信息不可恢复。如果后续生成需要引用丢弃部分的内容，质量会下降。

### H2O (Heavy-Hitter Oracle)

观察：attention score 分布中，只有少数 token 是"重要的"（获得高 attention score）。

H2O 的策略：
```
每个 attention head 独立维护 top-k 重要 token
定期淘汰 attention score 低的 token 的 KV Cache
```

### Sliding Window Attention

一些模型在训练时就使用了 sliding window attention（如 Mistral）：

```
标准 attention：每个 token attend to 所有之前的 token
Sliding window：每个 token 只 attend to 最近 W 个 token

KV Cache 大小：固定为 W，不随序列长度增长
```

Mistral-7B 的 sliding window size = 4096。

### 各策略对比

| 策略 | KV Cache 大小 | 适用场景 | 质量影响 |
|---|---|---|---|
| 全量缓存 | O(seq_len) | 通用 | 无 |
| Prefix Caching | 节省共享部分 | 重复前缀 | 无 |
| KV Quantization | 减半/4x | 通用 | 轻微 |
| StreamingLLM | O(1) | 超长生成 | 中等 |
| H2O | O(k) | 长序列 | 轻微到中等 |
| Sliding Window | O(W) | 训练时已设计 | 取决于训练 |

---

## Part 6: 中国模型的 KV Cache 特性汇总

| 模型 | Attention 类型 | KV Cache 特点 | 优化建议 |
|---|---|---|---|
| Qwen2.5 全系列 | GQA | kv_heads 较少，标准管理 | prefix caching + chunked prefill |
| DeepSeek-V2/V3 | MLA (Multi-head Latent Attention) | 缓存低秩 latent，极度紧凑 | 天然省显存，长上下文友好 |
| GLM-4 | GQA (kv_heads=2) | 极少 kv_heads，非常紧凑 | 同 Qwen 策略 |
| MiniMax-Text-01 | Lightning Attention | 线性注意力，KV Cache 恒定大小 | 不适用传统 PagedAttention |
| Baichuan2 | MHA → GQA (不同版本) | 标准管理 | prefix caching |

DeepSeek MLA 值得特别注意：它本质上是一种"训练时内置的 KV Cache 压缩"，比推理时做 KV 量化更优雅，因为模型训练时就学会了如何在低维空间表达 KV。

---

## Part 7: 实验

### 实验 1：Prefix Caching 效果

```bash
# 无 prefix caching
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 有 prefix caching
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 --enable-prefix-caching
```

发一组有相同 system prompt 的请求到两个端口，对比 TTFT：

预期 prefix caching 在第二次请求开始显著降低 TTFT。

### 实验 2：Chunked Prefill 调参效果

```bash
# 较小 token budget：更偏向稳定 decode ITL/TPOT
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-num-batched-tokens 2048

# 较大 token budget：更偏向 TTFT / 总吞吐
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 --max-num-batched-tokens 8192
```

同时发长 prompt 和短 prompt 的混合请求，对比短请求的 TPOT P99：

预期较小的 `max_num_batched_tokens` 让短请求 TPOT P99 更稳定；较大的值可能让长 prompt TTFT 更好，但更容易拖慢同批 decode。

---

## 交付物

| 文件 | 描述 |
|---|---|
| `kv-cache-optimization-report.md` | prefix caching 和 chunked prefill 的实验数据 + 分析 |

## 自检问题

1. Prefix caching 的 hash 匹配是在什么粒度上做的？
2. 如果 system prompt 长度不是 block_size 的整数倍，prefix caching 还能用吗？
3. Chunked prefill 的 chunk size 设太小会有什么问题？
4. KV Cache 量化到 FP8 会影响生成质量吗？在什么场景下影响更大？
5. DeepSeek-V2 的 MLA 为什么不需要推理时做 KV 量化？
