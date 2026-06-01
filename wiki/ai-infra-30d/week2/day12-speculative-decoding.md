---
title: "Day 12: Speculative Decoding（推测解码）"
type: concept
tags: [day12, speculative-decoding, draft-model, medusa, eagle]
created: 2026-06-01
updated: 2026-06-01
---

# Day 12: Speculative Decoding（推测解码）

## Part 1: 核心问题

Decode 阶段的根本问题：**每步只生成 1 个 token，但要加载全部模型权重。**

```
标准 decode:
Step 1: 读 16GB 权重 → 生成 token_1     → ~5ms
Step 2: 读 16GB 权重 → 生成 token_2     → ~5ms
Step 3: 读 16GB 权重 → 生成 token_3     → ~5ms
...
生成 100 个 token: 100 × 5ms = 500ms, 读了 100 × 16GB = 1.6 TB

问题: 99% 的时间在搬数据，1% 在做计算
```

Speculative decoding 的思路：**能不能每步生成多个 token？**

---

## Part 2: Speculative Decoding 原理

### 基本框架

使用两个模型：
- **Draft model（草稿模型）**：小而快，用来"猜"接下来的 token
- **Target model（目标模型）**：大而准，用来"验证"猜测

```
1. Draft model 快速生成 k 个候选 token:
   draft_tokens = [t1, t2, t3, t4, t5]    ← 用小模型生成，很快

2. Target model 一次性验证所有候选:
   target_probs = target_model.forward([context + t1 + t2 + t3 + t4 + t5])
   ← 一次 forward pass 就能得到 5 个位置的概率分布

3. 从左到右逐个验证:
   位置 1: P_target(t1) ≥ P_draft(t1) * r → 接受 ✓
   位置 2: P_target(t2) ≥ P_draft(t2) * r → 接受 ✓
   位置 3: P_target(t3) < P_draft(t3) * r → 拒绝 ✗
   → 从拒绝位置重新采样
   → 本次接受了 2 个 token + 1 个修正 token = 3 个 token

4. 净收益: 1 次 target forward 生成了 3 个 token（而不是标准的 1 个）
```

### 为什么这是无损的？

关键性质：**speculative decoding 保证和直接用 target model 生成的分布完全一致。**

验证算法使用了一种精巧的 rejection sampling：
- 如果 target model 在某个位置给了更高概率 → 直接接受
- 如果 target model 给了更低概率 → 按比例随机接受/拒绝
- 拒绝后从修正分布中重新采样

结果：最终输出的 token 序列和纯 target model 生成的分布数学上一致。

### Acceptance Rate

**Acceptance rate**（接受率）决定了 speculative decoding 的收益：

```
acceptance_rate = 平均每 k 个候选中被接受的数量 / k

如果 acceptance_rate = 80%，k = 5:
  平均每次接受 4 个 + 1 个修正 = 5 个 token
  用 1 次 target forward + 1 次 draft forward（k 步）
  加速比 ≈ 5 / (1 + draft_cost/target_cost)

如果 acceptance_rate = 30%，k = 5:
  平均每次接受 1.5 个 + 1 个修正 = 2.5 个 token
  加速比很小，可能不值得
```

Acceptance rate 取决于 **draft model 和 target model 的分布相似度**。

### 什么影响 acceptance rate？

| 因素 | 高 acceptance rate | 低 acceptance rate |
|---|---|---|
| Draft 和 target 的相似度 | 同系列模型（如 Qwen-0.5B 对 Qwen-7B） | 不相关模型 |
| 任务类型 | 确定性高的任务（翻译、代码补全） | 创意性任务（写故事、自由对话） |
| Temperature | 低（更确定） | 高（更随机） |
| 输出位置 | 序列前部（更可预测） | 序列中后部（更不确定） |

---

## Part 3: Draft Model 选择

### 同系列小模型

最常见的方案：用同系列的小模型做 draft。

```
Target: Qwen2.5-72B-Instruct
Draft:  Qwen2.5-0.5B-Instruct 或 Qwen2.5-1.5B-Instruct

Target: Llama-3.1-70B-Instruct
Draft:  Llama-3.1-8B-Instruct（可能太大了）
        → 可以用更小的 draft 或 distilled 版本
```

选择 draft model 的 trade-off：
- Draft 太大：推理快不了多少，draft 自身开销大
- Draft 太小：acceptance rate 低，大部分候选被拒绝

经验法则：draft model 应该是 target model 的 ~1/10 大小或更小。

### N-gram Speculative Decoding

不用 draft model，而是从已生成的 token 中匹配 n-gram 模式来猜测：

```
已生成: "The capital of France is Paris. The capital of Germany is"
N-gram 匹配: 最近出现了 "capital of X is Y" 模式
猜测: "Berlin" (基于 n-gram 频率)
```

优势：不需要额外模型，零成本
劣势：只对重复模式有效，acceptance rate 通常较低

### Medusa

不用单独的 draft model，而是在 target model 上加多个 **prediction heads**：

```
原始 LM Head:    hidden_state → 1 个 next token
Medusa Heads:    hidden_state → head_1 → token at position +1
                               → head_2 → token at position +2
                               → head_3 → token at position +3

每步 forward 同时预测未来 3-4 个位置的 token
```

优势：
- 不需要加载额外模型（节省显存）
- Forward pass 几乎不增加计算量（heads 很小）
- 和 target model 紧密耦合，acceptance rate 较高

劣势：
- 需要额外训练 Medusa heads
- 不保证和原始分布完全一致（除非用特殊的 tree attention 验证）

### EAGLE

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) 改进了 Medusa：

```
Medusa:  只用当前 hidden state 预测
EAGLE:   用当前 hidden state + 已预测的 token embedding → 自回归预测多个位置
```

EAGLE 的 acceptance rate 通常高于 Medusa。

---

## Part 4: 在 vLLM 中使用 Speculative Decoding

### Draft Model 方式

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --speculative-model Qwen/Qwen2.5-0.5B-Instruct \
  --num-speculative-tokens 5 \
  --speculative-draft-tensor-parallel-size 1
```

参数说明：
- `--speculative-model`：draft model 名称
- `--num-speculative-tokens`：每次猜测多少个 token（k 值）
- `--speculative-draft-tensor-parallel-size`：draft model 的 TP 大小（通常设 1）

### N-gram 方式

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --speculative-model "[ngram]" \
  --num-speculative-tokens 5 \
  --ngram-prompt-lookup-max 4
```

### 观察效果

```python
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# 确定性任务（高 acceptance rate 预期）
t0 = time.time()
resp = client.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    prompt="Translate to English: 人工智能推理优化是一个非常重要的技术方向，它涉及到模型压缩、分布式推理、显存管理等多个方面。",
    max_tokens=200,
    temperature=0.0,  # 低 temperature → 更确定 → 更高 acceptance rate
)
print(f"Deterministic: {(time.time()-t0)*1000:.0f}ms, {resp.usage.completion_tokens} tokens")

# 创意性任务（低 acceptance rate 预期）
t0 = time.time()
resp = client.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    prompt="Write a creative short story about a robot learning to cook:",
    max_tokens=200,
    temperature=0.9,  # 高 temperature → 更随机 → 更低 acceptance rate
)
print(f"Creative: {(time.time()-t0)*1000:.0f}ms, {resp.usage.completion_tokens} tokens")
```

---

## Part 5: 什么时候不应该用 Speculative Decoding

Speculative decoding 并不总是有效：

| 情况 | 原因 | 建议 |
|---|---|---|
| 高并发 / 大 batch | Draft model 占用额外显存和计算，在高并发时 target model 已经是 compute-bound，加 draft 反而增加开销 | 高并发场景关闭 speculative decoding |
| Draft model 太大 | Draft 本身的推理开销吞噬了加速收益 | 选更小的 draft 或用 n-gram |
| 高 temperature | Acceptance rate 低，大部分候选被拒绝 | 仅在低 temperature / 确定性任务中使用 |
| 短输出 | 加速的是 decode 阶段，如果 output 很短（如 10 token），收益不明显 | 仅对长输出场景使用 |
| 显存紧张 | Draft model 需要额外显存 | 用 n-gram 替代（不需要额外模型） |

### 适用场景总结

```
最佳场景: 低并发 + 长输出 + 低 temperature + 好的 draft model
  例: 单用户代码补全、翻译、长文档生成

不适用: 高并发在线 serving + 短输出 + 高 temperature
  例: 大量用户的短对话
```

---

## Part 6: 各模型的 Speculative Decoding 方案

| Target Model | 推荐 Draft Model | Acceptance Rate 参考 |
|---|---|---|
| Qwen2.5-72B | Qwen2.5-0.5B / 1.5B | ~60-80% (确定性任务) |
| Qwen2.5-7B | Qwen2.5-0.5B 或 n-gram | ~50-70% |
| DeepSeek-V3 | 复杂 (MoE + MLA) | 需要专门的 draft 方案 |
| GLM-4-9B | GLM 小模型或 n-gram | ~50-65% |

DeepSeek-V3 的 speculative decoding 特别复杂：
- MoE 架构意味着 draft model 很难模仿 expert routing 行为
- MLA 的 KV 格式不同于标准 attention
- 实际上 DeepSeek-V3 的每 token 计算量已经很小（只有 37B active），加速空间有限

---

## 交付物

| 文件 | 描述 |
|---|---|
| `speculative-decoding-notes.md` | 原理笔记、acceptance rate 实验、适用场景判断 |

## 自检问题

1. Speculative decoding 为什么是"无损"的？
2. Acceptance rate 取决于什么？如何提高它？
3. 在什么场景下 speculative decoding 反而会降低吞吐？
4. Draft model 应该多大？太大或太小各有什么问题？
5. Medusa 和标准 speculative decoding 的主要区别是什么？
6. N-gram speculative decoding 的优缺点是什么？
