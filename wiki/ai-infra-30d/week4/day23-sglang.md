---
title: "Day 23: SGLang 对照学习"
type: concept
tags: [day23, sglang, radix-attention, structured-generation, comparison]
created: 2026-06-01
updated: 2026-06-01
---

# Day 23: SGLang 对照学习

## Part 1: SGLang 定位

SGLang 是 UC Berkeley 开发的 LLM serving 框架，核心创新是 **RadixAttention** 和对**结构化生成**的优化。

```
vLLM:    PagedAttention → KV Cache 分页管理 → 通用高效 serving
SGLang:  RadixAttention → KV Cache 树状复用 → 结构化/多轮/分支场景更优
TRT-LLM: Engine build   → 极致 kernel 优化 → NVIDIA 硬件上最快
```

### SGLang vs vLLM 快速对比

| 维度 | vLLM | SGLang |
|---|---|---|
| KV Cache 复用 | Prefix caching (hash matching) | RadixAttention (radix tree) |
| 结构化生成 | 基础支持 | 深度优化 (jump-forward decoding) |
| 多轮对话 | 默认会重算历史，prefix cache 命中时可复用 | 在同一生成程序/会话结构中更自然地复用前缀 KV |
| 批量有分支的请求 | 无特殊优化 | 共享公共前缀的分支高效复用 |
| 性能 | 好 | 某些场景更好 |
| 生态 | 最广泛 | 快速增长中 |

---

## Part 2: RadixAttention

### 核心思想

vLLM 的 prefix caching 基于 hash 匹配：将 token block 做 hash，相同 hash 复用。

SGLang 的 RadixAttention 更进一步：用 **radix tree（基数树）** 管理所有 KV Cache。

```
Radix Tree 示例:

假设有三个请求:
  A: "System: You are helpful. User: What is AI?"
  B: "System: You are helpful. User: Tell me a joke."
  C: "System: You are helpful. User: What is AI? More details."

Radix Tree:
                    [root]
                      |
    "System: You are helpful. User: "
                      |
              ┌───────┴───────┐
              |               |
    "What is AI?"       "Tell me a joke."
         |                   |
    " More details."     [leaf B]
         |
      [leaf C]             [leaf A]
```

### 和 vLLM prefix caching 的区别

```
vLLM prefix caching:
  - 基于 block-level hash 匹配
  - 只能匹配完整的 block（block_size 对齐）
  - 不同请求的共享前缀必须严格相同
  - 无法高效处理"部分匹配"

RadixAttention:
  - 基于 token 级别的 tree 结构
  - 可以匹配任意长度的公共前缀
  - 自然支持多轮对话（新一轮在前几轮的 tree 节点上追加）
  - 支持分支（如 beam search、多个 candidate 共享前缀）
```

### RadixAttention 在多轮对话中的优势

```
多轮对话场景:
  Turn 1: [system] + [user_1] → [assistant_1]
  Turn 2: [system] + [user_1] + [assistant_1] + [user_2] → [assistant_2]
  Turn 3: [system] + [user_1] + [assistant_1] + [user_2] + [assistant_2] + [user_3]

vLLM:
  每一轮需要重新 prefill 整个 history（除非 prefix cache 命中）
  Prefix cache 命中条件: 完全相同的 token 序列 → 同一用户多轮才能命中

SGLang:
  Turn 1 的 KV Cache 保留在 radix tree 中
  Turn 2 直接在 Turn 1 的 tree 节点上追加新 token
  → 只需要 prefill user_2 部分，不需要重新计算前面的
  注意：这是在 SGLang runtime 能管理该会话/程序状态的前提下，不等于任意跨 replica 请求都自动命中。

TTFT 对比:
  Turn 3, history = 5000 tokens, new input = 200 tokens
  vLLM (无 cache): prefill 5200 tokens → TTFT ~200ms
  vLLM (prefix cache hit): prefill 200 tokens → TTFT ~20ms
  SGLang (radix tree): prefill 200 tokens → TTFT ~20ms
  
  → 在多轮对话且 prefix cache 命中时差不多
  → SGLang 的优势在于 radix tree 的管理更自然，不受 block_size 对齐限制
```

---

## Part 3: 结构化生成优化

### 什么是结构化生成

要求模型输出符合特定格式（JSON、SQL、正则表达式等）：

```python
# 示例: 要求模型输出 JSON
response = generate(
    prompt="Extract name and age from: John is 30 years old.",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
    }
)
# 期望输出: {"name": "John", "age": 30}
```

### 标准实现的问题

```
标准 constrained decoding:
  每步 decode:
    1. 获取 logits
    2. 根据 JSON schema 的当前状态，mask 掉不合法的 token
    3. 从合法 token 中采样

问题: 很多 token 是确定性的
  当已生成 '{"name": "John", "age":' 时:
    下一个 token 必须是空格或数字
    接下来几个 token 基本确定
  
  但标准实现仍然逐 token 执行完整的 forward pass → 浪费
```

### SGLang 的 Jump-Forward Decoding

```
优化: 当下一个 token（或多个 token）是确定性的，直接跳过 forward pass

例: 生成 JSON 时
  已生成: '{"name": "'
  Schema 约束下一段是 string → 直到遇到 '"'
  
  前缀 '"name": "' 之后的 '"' 是确定的结构 token
  → 可以直接"跳"过去，不做 forward pass
  
实际效果:
  JSON 生成中 ~30-50% 的 token 是结构性的（冒号、引号、括号等）
  这些 token 可以通过 jump-forward 跳过
  → 整体生成速度提升 ~1.3-2x（取决于 schema 复杂度）
```

---

## Part 4: SGLang 的编程接口

SGLang 提供了一种声明式的 prompt 编程接口：

```python
import sglang as sgl

@sgl.function
def multi_step(s, question):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))
    
    # 基于第一步的回答继续
    s += sgl.user("Can you elaborate on that?")
    s += sgl.assistant(sgl.gen("elaboration", max_tokens=512))

# 执行
state = multi_step.run(question="What is attention?")
print(state["answer"])
print(state["elaboration"])
```

这种接口的优势：
- **自动 KV Cache 管理**：前一步的 KV Cache 自动在 radix tree 中保留
- **分支支持**：可以 fork 一个对话的多个分支
- **批量优化**：多个请求的公共前缀自动共享

### 批量分支场景

```python
@sgl.function
def multi_choice(s, question):
    s += sgl.user(question)
    
    # Fork 3 个不同 temperature 的回答
    forks = s.fork(3)
    for i, f in enumerate(forks):
        f += sgl.assistant(
            sgl.gen("answer", max_tokens=256, temperature=0.7 + i * 0.3)
        )
    s += sgl.select_best(forks, key="answer")
```

在这种场景下，3 个 fork 共享相同的前缀 KV Cache → RadixAttention 的优势最大化。

---

## Part 5: SGLang 在不同场景下的表现

| 场景 | SGLang 优势 | 原因 |
|---|---|---|
| 多轮对话 | TTFT 低 | 前几轮 KV Cache 自动保留和复用 |
| JSON/结构化输出 | 吞吐高 | Jump-forward decoding 跳过确定性 token |
| Tree-of-Thought / Best-of-N | 效率高 | 多个分支共享前缀 |
| Agent / Function Calling | 效果好 | 结构化输出 + 多步推理复用 |
| 简单单轮对话 | 和 vLLM 差不多 | 没有特别场景优势 |
| 极致低延迟 | 不如 TRT-LLM | kernel 层面优化不如 NVIDIA 专属 |

---

## Part 6: 三框架选型总结

```
                    易用性
                      ↑
                      |
            vLLM ★    |
                      |    SGLang ★
                      |
        ──────────────┼──────────────→ 性能
                      |
                      |
                      |    TRT-LLM ★
                      |
```

| 选择 | 场景 |
|---|---|
| vLLM | 通用 serving、快速迭代、社区生态 |
| SGLang | 多轮对话、结构化输出、agent 场景 |
| TensorRT-LLM | 极致性能、NVIDIA 专属、稳定生产 |
| vLLM + SGLang 混用 | 按场景路由（短对话 → vLLM，结构化 → SGLang） |

---

## 交付物

| 文件 | 描述 |
|---|---|
| `vLLM-vs-SGLang.md` | RadixAttention 原理 + 结构化生成优化 + 场景选型 |

## 自检问题

1. RadixAttention 和 vLLM 的 prefix caching 有什么本质区别？
2. Jump-forward decoding 在什么条件下有效？
3. 多轮对话场景下 SGLang 的 TTFT 为什么更低？
4. SGLang 的分支 (fork) 功能适合什么业务场景？
5. 如果你只能选一个框架，会选哪个？为什么？
