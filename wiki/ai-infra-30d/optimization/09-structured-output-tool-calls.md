---
title: "优化专项 09: 结构化输出与 Tool Calling 慢或错"
type: concept
tags: [ai-infra, inference-optimization, structured-output, guided-decoding, xgrammar, tool-calling, reasoning-parser]
created: 2026-09-29
updated: 2026-09-29
---

# 优化专项 09: 结构化输出与 Tool Calling 慢或错

> 一句话：JSON schema 约束和 tool calling 让请求变慢或解析出错时，问题要么在 grammar 编译，要么在每步 bitmask，要么在 parser 和模板，这页按这三条路径排查。关联日课：[[day09-scheduler]]、[[day13-flash-attention]]、[[day23-sglang]]。

## 1. 症状与判定

先回忆路径（[[day13-flash-attention]] Part 4）：请求带 schema → EngineCore 把它置为 `WAITING_FOR_FSM`，线程池编译 grammar → 编译完才可调度 → 每步 `schedule()` 末尾生成 vocab 大小的 bitmask → worker 在 sampler 前 apply → `update_from_output()` 用接受的 token 推进 FSM。tool calling 则多一层：模型按 chat template 输出特定格式 → 前端 tool parser 解析成结构化 tool_calls。

| 观察 | 指向 |
|---|---|
| 带 schema 的请求 TTFT 比同 prompt 不带 schema 高几百 ms 到几秒，TPOT 正常 | grammar 编译（2.1） |
| TTFT 正常，TPOT 明显高于不带 schema 的请求，EngineCore CPU 上升 | 每步 bitmask 生成（2.2） |
| 首次某 schema 慢、重复用同一 schema 快 | 编译缓存生效，属正常；无缓存命中时看 2.1 |
| 输出符合 schema 但内容质量下降、重复或提前截断 | schema 过严或 backend 处理空白字符方式（2.3） |
| tool_calls 为空，内容却是一段像 JSON 的文本 | tool parser 与模型模板不匹配（2.5） |
| reasoning 模型输出里 thinking 混进了正文，或 JSON 前面带一段推理 | reasoning parser 未配置或不匹配（2.6） |
| 开了 spec decode 后带 schema 的请求反而更慢 | bitmask 要为每个 draft 位置各算一份（2.4） |
| 并发上去后带 schema 的请求排队时间长 | 编译线程池被占满（2.1） |

指标：`vllm:request_queue_time_seconds` 里 `WAITING_FOR_FSM` 的时间算在排队里；结构化输出相关的专用指标随版本变化，用 `curl /metrics | grep -i -E "structured|guided|grammar"` 看当前版本有什么。

## 2. 根因逐个排查

### 2.1 grammar 编译慢

**确认**：同一 prompt 带 schema 和不带 schema 各发一次，TTFT 差值就是编译时间加排队。schema 里有长正则（`pattern`）、大枚举（几百个 `enum` 值）、深嵌套、`anyOf`/`oneOf` 组合、无界数组时编译时间指数级上升。

**修**：
- 简化 schema：去掉 `pattern`，枚举改成模型自己填字符串再由应用校验，嵌套控制在 3 层内，避免 `anyOf`。
- 复用 schema：编译结果按 schema 字符串缓存，同一业务用完全相同的 schema 字符串（键顺序、空格都要一致，否则 cache key 不同）。
- 换 backend：`--structured-outputs-config '{"backend":"xgrammar"}'`（默认，编译快、运行快，对 JSON schema 覆盖广）；`guidance` 对复杂 grammar 和正则的处理有时更好；`outlines` 和 `lm-format-enforcer` 是更早的实现，特定场景才选。旧版本参数名是 `--guided-decoding-backend`。
- 编译线程池大小随版本有配置项，高并发带 schema 的流量要看它是否成为瓶颈。

**副作用**：schema 放松后应用要自己做校验和重试。

### 2.2 每步 bitmask 开销

**确认**：EngineCore 火焰图里 `grammar_bitmask` 或 FSM 相关帧占比高；带 schema 的请求比例越高、并发越大越明显。bitmask 是 [num_reqs, vocab/32] 的 int32，每步为每个结构化请求填一次，FSM 状态越复杂越贵。

**修**：减少同时进行的结构化请求数（业务上只对确实需要的调用加 schema）；简化 schema 让 FSM 状态少；xgrammar 对常见 JSON 结构有 token 级的快速路径，复杂正则会退化到慢路径。

### 2.3 输出质量下降

**确认**：加 schema 后内容变差、重复、字段值为空。原因：schema 把模型自然会输出的 token 序列（比如带空格的 JSON、换行）判为非法，模型被迫走低概率路径；或 `max_tokens` 不够容纳完整 JSON 被截断成非法 JSON。

**修**：backend 的 whitespace 处理有配置（xgrammar 有 `any_whitespace` 一类选项，随版本变化）；`max_tokens` 留够；prompt 里同时给出格式说明，让模型自然倾向合法输出而不是全靠 mask 硬拉。

### 2.4 spec decode 与结构化输出叠加

**确认**：两者同时开时，每个 draft 位置都要算一份 bitmask，bitmask 数量乘 `num_speculative_tokens`；draft token 被 mask 拒绝会拉低接受率。

**修**：结构化请求比例高的服务不要同时开 spec decode，或降低 `num_speculative_tokens`；看当前版本文档确认两者兼容。

### 2.5 tool parser 解析失败

**确认**：响应里 `tool_calls` 为空但 `content` 是 JSON 文本；或 `finish_reason` 不是 `tool_calls`。原因：`--tool-call-parser` 选错（每个模型家族有自己的格式：hermes、llama3_json、mistral、qwen 等）；chat template 与 parser 不配套（自定义模板改了 tool 标记）；流式模式下 parser 的增量解析对某些模型不稳定；模型本身不擅长该格式。

**修**：`--enable-auto-tool-choice --tool-call-parser <匹配模型的名字>`，必要时 `--chat-template` 指定官方模板；先在非流式下验证，再开流式；模型输出不稳定时改用 `tool_choice` 指定具体函数，或加 schema 约束（回到 2.1 的成本）。

### 2.6 reasoning parser 与 JSON 冲突

**确认**：DeepSeek-R1、Qwen3 一类模型输出 `<think>...</think>` 段，没配 `--reasoning-parser` 时 thinking 混进 content，tool parser 在 thinking 里找 JSON 会误解析；配了 schema 时 grammar 从第一个 token 就生效，把 thinking 段也约束成 JSON，模型无法推理。

**修**：`--reasoning-parser <匹配模型的名字>`；结构化输出与 thinking 的组合支持随版本变化（部分版本会让 grammar 在 thinking 结束后才生效），不确定时用 non-thinking 模式做结构化调用。

### 2.7 用了过重的约束方式

**确认**：只需要"输出是合法 JSON"却用了完整 schema；只需要几个固定选项却用了正则。

**修**：成本梯度从低到高：不约束（prompt 说明 + 应用校验重试）< JSON mode（`response_format: {"type":"json_object"}`，只约束语法）< JSON schema < 自定义 grammar 或正则。按需选最低档。

### 2.8 schema 设计对性能的影响（汇总）

| schema 特征 | 编译成本 | 每步成本 | 替代写法 |
|---|---|---|---|
| `pattern` 正则 | 高，正则转自动机 | 中到高 | 去掉，应用层校验 |
| 大 `enum` | 中，每个值一条路径 | 中 | 改成 string，应用层映射 |
| 深嵌套对象 | 随层数增长 | 中 | 拍平成两层 |
| `anyOf` / `oneOf` | 高，分支相乘 | 高 | 拆成多个请求或用 discriminator 字段 |
| 无界 `string` / `array` | 低 | 低 | 保留，但配 `max_tokens` |
| `additionalProperties: false` | 低 | 低 | 保留，减少歧义 |
| `minLength` / `maxLength` / `minItems` | 中，计数状态 | 中 | 去掉，应用校验 |

原则：schema 只负责"结构合法"，值的合法性交给应用。

### 2.9 SGLang 的 jump-forward 与 vLLM 的差异

SGLang 的结构化输出在 FSM 处于"只有一条合法路径"的状态时（比如 JSON 的固定 key、引号、逗号）直接把这段 token 一次性写入，跳过逐 token 采样，这就是 jump-forward decoding（[[day23-sglang]]）。对固定字段多、自由文本少的 schema，输出 token 数里可能有一半是可跳的，TPOT 层面的收益明显。vLLM 的 bitmask 方案每个 token 仍要走一次 forward。如果业务是大量固定结构的 JSON 输出，这是框架选型的一个实际差异点；如果 JSON 里大头是自由文本字段，差异不大。

### 排查 checklist

1. 同一 prompt 带与不带 schema 各发一次，差值落在 TTFT 还是 TPOT。
2. 看 schema 字符串是否每次完全一致，不一致就没有编译缓存。
3. 数 schema 里的 `pattern`、`enum` 值总数、嵌套层数、`anyOf` 个数，对照 2.8 逐项去掉。
4. tool calling 先关流式跑 50 条，看 `tool_calls` 非空率。
5. reasoning 模型确认 `--reasoning-parser` 已配，且 content 里没有 think 标记。
6. 如果开了 spec decode，关掉重测一次。
7. 并发 32 下看 `request_queue_time_seconds` 是否比并发 1 时多出编译时间量级。

### 常见误判

- 把结构化输出的 TTFT 归给 prefill。同 prompt 不带 schema 一测就知道。
- 认为"schema 越严格越安全"。严格 schema 让模型走低概率路径，语义错误反而更多。
- 用 tool calling 正确率评估 parser 之前没确认模板。多数解析失败是模板与 parser 不配套，不是模型能力问题。

## 3. 决策表

| 症状组合 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| TTFT 高、TPOT 正常 | 简化 schema、复用同一 schema 字符串 | 换 backend | 调 budget |
| TPOT 高、EngineCore CPU 高 | 减少结构化请求比例 | 简化 FSM | 加 GPU |
| 质量下降 | 放宽 whitespace、留够 max_tokens、prompt 给格式 | 降到 JSON mode | 加更严的 schema |
| tool_calls 为空 | 对准 parser 与模板，先非流式 | tool_choice 指定函数 | 加 grammar 硬约束 |
| thinking 混进正文 | 配 reasoning parser | non-thinking 模式 | 用正则截 |
| spec decode 后更慢 | 结构化流量关 spec decode | 降 k | 两者都开着调参 |

## 4. 验证实验

固定：模型、prompt、`max_tokens=256`、temperature 0、并发 1 和 32 各跑一次，warmup 20 个请求。

```bash
vllm serve <m> --enable-auto-tool-choice --tool-call-parser <name> --reasoning-parser <name>

# 三组请求体：A 不带 schema；B 简单 schema（3 个字符串字段）；C 复杂 schema（20 字段、3 层嵌套、2 个 enum 各 50 值、1 个 pattern）
# 每组 100 个请求，记录 TTFT P50/P99、TPOT P50/P99、request_queue_time、EngineCore CPU
# 预期：B 比 A 的 TTFT 略高（首次编译后缓存）、TPOT 略高；C 的 TTFT 明显高且并发 32 下排队上升

# 第二轮：把 C 的 pattern 去掉、enum 改成自由字符串，重跑
# 预期：TTFT 回落到接近 B

# 第三轮：backend 切到 guidance 重跑 C
# 预期：编译时间与运行时开销的分布不同，选对自己 schema 更快的
```

tool calling 正确率实验：50 个需要调用工具的 prompt，非流式与流式各跑一次，统计 `tool_calls` 非空且参数合法的比例；低于 95% 先换 parser 或模板，再考虑加 schema。

### 请求体示例（三组对照）

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
prompt = "Extract the company, ticker and sentiment from: NVIDIA shares rose 5% after earnings."

# A: 不约束
a = c.chat.completions.create(model="<m>", messages=[{"role":"user","content":prompt}], max_tokens=256)

# B: 简单 schema
simple = {"type":"object","properties":{"company":{"type":"string"},"ticker":{"type":"string"},
          "sentiment":{"type":"string"}},"required":["company","ticker","sentiment"],
          "additionalProperties":False}
b = c.chat.completions.create(model="<m>", messages=[{"role":"user","content":prompt}], max_tokens=256,
    response_format={"type":"json_schema","json_schema":{"name":"extract","schema":simple}})

# C: 复杂 schema：在 simple 上加 pattern、大 enum、嵌套，用来复现 2.1 / 2.2
complex_ = dict(simple); complex_["properties"] = dict(simple["properties"])
complex_["properties"]["ticker"] = {"type":"string","pattern":"^[A-Z]{1,5}$"}
complex_["properties"]["sentiment"] = {"type":"string","enum":[f"level_{i}" for i in range(50)]}
complex_["properties"]["details"] = {"type":"object","properties":{"events":{"type":"array","items":{
    "type":"object","properties":{"kind":{"type":"string"},"impact":{"type":"object",
    "properties":{"score":{"type":"number"},"note":{"type":"string"}}}}}}}}
cc = c.chat.completions.create(model="<m>", messages=[{"role":"user","content":prompt}], max_tokens=256,
    response_format={"type":"json_schema","json_schema":{"name":"extract","schema":complex_}})
```

`response_format` 之外，vLLM 还接受 `extra_body` 里的 `structured_outputs` 或旧的 `guided_json` / `guided_regex` / `guided_grammar` 字段，名称随版本变化，以当前版本文档为准。

### tool calling 最小配置示例

```bash
# Qwen 系列（示例；每个模型家族的 parser 名不同，查 vllm serve --help 的 --tool-call-parser 可选值）
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
# 带 thinking 的模型再加 reasoning parser
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
```

验证时看响应的 `choices[0].message.tool_calls` 是否为列表且 `finish_reason == "tool_calls"`。

## 5. 关联

- 总入口：[[01-diagnosis-playbook]]
- Agent 流量整体：[[08-multi-turn-agent-workload]]
- CPU 开销定位：[[06-gpu-util-low-cpu-bound]]
- spec decode 组合：[[11-spec-decode-no-gain]]
- SGLang 的 jump-forward 对比：[[day23-sglang]]
- 机制来源：[[day09-scheduler]]、[[day13-flash-attention]]
