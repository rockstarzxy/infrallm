---
title: "优化专项 08: 多轮对话与 Agent Workload"
type: concept
tags: [ai-infra, inference-optimization, agent, multi-turn, prefix-caching, routing, tool-calling]
created: 2026-09-29
updated: 2026-09-29
---

# 优化专项 08: 多轮对话与 Agent Workload

> 一句话：Agent 流量是"长 prompt、短输出、同一前缀反复出现、一个任务几十次调用"，通用调参对它几乎无效，收益全在 prefix cache 命中率和路由上。关联日课：[[day09-scheduler]]、[[day10-kv-cache]]、[[day13-flash-attention]]、[[day18-data-parallel]]、[[day23-sglang]]。

## 1. 症状与判定

先认清 workload 特征，再看指标：

| 特征 | 对 serving 的含义 |
|---|---|
| 每次调用 prompt 几 k 到几十 k（system prompt + tool schema + 历史），输出几十到几百 token | prefill 主导，decode 占比小，TTFT 就是大部分延迟 |
| 同一 session 内每次调用的前缀是上一次的超集 | 理论上 prefix cache 可以把每次 prefill 压到只算新增部分 |
| 一个用户任务触发 10 到 100 次模型调用 | 单次延迟的长尾会被放大成任务级的长尾 |
| 并发 session 多但每个 session 串行调用 | 瞬时并发不高，KV 却要为大量"暂停中"的 session 保活 |
| 输出多为 tool call JSON 或短决策 | 结构化输出和 tool parser 的开销占比高 |

判定表：

| 观察 | 指向 |
|---|---|
| `vllm:prefix_cache_hits` / `queries` 低于 50%，而业务明明是多轮 | 前缀被破坏、跨 replica 分散、或被驱逐（2.1 到 2.3） |
| 命中率高但 TTFT 仍高 | 新增部分本身长（大 tool 返回），或排队，或 grammar 编译 |
| 任务级 P99 远大于单次 P99 乘步数 | 某几步撞上抢占或驱逐后的完整 prefill（2.3） |
| API server CPU 高 | tool parser、reasoning parser、chat template 渲染长 tool schema（2.5） |
| KV 使用率高但 `num_requests_running` 低 | 大量 session 的 KV 在 free 队列里占位等待复用，属正常，但要看驱逐率 |

## 2. 根因逐个排查

### 2.1 前缀被自己破坏（最常见）

**确认**：把两次连续调用的完整 prompt（渲染后的文本或 token ids）diff，看第一个不同的位置在哪。常见罪魁：system prompt 里的当前时间、随机 request ID、用户 ID 放在开头；每轮把"第 N 轮"写进 system；tool schema 顺序不稳定（dict 无序序列化）；chat template 在多轮时改变前面轮次的渲染（有的模板会给最后一轮加特殊标记，导致上一轮的位置内容变了）。

**修**：所有变化的内容挪到 prompt 末尾；时间戳精确到天而不是秒；tool schema 排序后序列化；换或改 chat template 让历史轮次渲染稳定。hash 链的性质是"第一个不同 block 之后全部失效"（[[day10-kv-cache]]），所以前缀开头哪怕一个字符的变化都是全 miss。

### 2.2 跨 replica 分散

**确认**：多实例部署下，同一 session 的连续请求被轮询到不同 replica，每个 replica 各自 prefill 一遍。单实例测试命中率高、上线后命中率低，基本就是这个。

**修**：网关做 session 亲和（按 session id 或前缀 hash 一致性路由）。更进一步是 prefix-aware routing：网关维护"哪个 replica 有哪些前缀"的近似表，新请求路由到前缀重叠最多的 replica。vLLM 生态里的 production-stack router 和 llm-d 的调度器都有这类能力，功能细节随版本变化，用前看文档；自己在网关做一致性 hash 是最简单可靠的起点（[[day18-data-parallel]]）。

**副作用**：亲和路由和负载均衡冲突，热 session 会让某个 replica 过载，要有溢出策略。

### 2.3 被驱逐后的完整 prefill

**确认**：`vllm:num_preemptions_total` 上升，或 session 空闲几分钟后第一次调用 TTFT 回到冷启动水平。原因是 free 队列里的缓存 block 被新流量挤掉了。

**修**：估算需要保活的 session 数乘平均前缀长度，与 `# GPU blocks` 对比；不够就 fp8 KV、加卡，或 KV offload connector 让驱逐落到 CPU 内存（[[07-long-context]] 2.9）。客户端侧可以在 session 空闲时发一个 `max_tokens=1` 的保活请求刷新 LRU 位置，但要算清楚这笔 prefill 成本是否比重算便宜。

### 2.4 新增部分本身太长

**确认**：命中率高但每次新增 token 数仍有几 k。通常是 tool 返回的原始内容（整页 HTML、大 JSON、长日志）直接塞进对话。

**修**：这是应用层的问题：tool 返回截断、摘要、只保留相关字段。infra 侧无解，但要把"平均新增 token 数"做成指标反馈给应用团队。

### 2.5 tool parser 与 reasoning parser 的 CPU 开销

**确认**：API server 进程 CPU 高，火焰图里 `tool_parser`、`reasoning_parser` 或 chat template 渲染占比大。tool schema 有几十个函数时，每次请求渲染 template 就是几 k token 的字符串拼接和 tokenize。

**修**：`--api-server-count` 分摊；tool schema 精简，只传本步可能用到的工具；流式模式下 tool parser 每个 chunk 都要跑一遍增量解析，非流式或减少 chunk 频率能省一截。不同模型的 parser 实现质量差异大，解析失败会回退成普通文本，见 [[09-structured-output-tool-calls]]。

### 2.6 步数多带来的任务级长尾

**确认**：单次调用 P99 是 2 秒，任务 30 步，任务 P99 却是 90 秒而不是 60 秒。因为每步都有独立的概率撞上抢占、驱逐、grammar 编译或排队。

**修**：给 agent 流量单独的 priority（[[day09-scheduler]]），避免被批量任务抢占；单独实例池隔离；限制单任务最大步数；应用层并行化能并行的工具调用，把串行 30 步变成 10 步。

### 2.7 并行工具调用没有 batch 化

**确认**：模型一次输出多个 tool call，应用逐个执行再逐个回传，每个回传触发一次模型调用。

**修**：应用层把多个 tool 结果合并进同一轮再调模型；模型侧确认 chat template 和 parser 支持 parallel tool calls（不是所有模型都支持）。

### 2.8 结构化输出叠加

**确认**：tool call 用 JSON schema 强约束时，每步都有 grammar 编译和 bitmask 开销，短输出下这部分占比很高。

**修**：tool call 通常靠模型自身的模板格式已经足够稳定，先试不加 grammar；确实要约束时用简单 schema，见 [[09-structured-output-tool-calls]]。

### 2.9 reasoning 模型的 thinking 段

**确认**：用带 thinking 的模型（Qwen3、DeepSeek-R1 类）做 agent，每步先输出几百到几千 token 的推理再给 tool call，输出从"短"变成"长"，TPOT 成为主要延迟，且 thinking 内容通常不回传给下一轮，历史前缀里没有它。

**修**：agent 决策步用 non-thinking 模式或限制 thinking 预算（模型侧参数，如 Qwen3 的 `enable_thinking` 或 `max_thinking_tokens` 类能力，以模型文档为准）；确认 reasoning parser 正确拆分 thinking 与正文，否则 tool parser 会在 thinking 里误解析；见 [[09-structured-output-tool-calls]]。

### 2.10 session 的 KV 在多个 replica 上重复占位

**确认**：亲和路由做好前，同一 session 在 N 个 replica 上各有一份前缀缓存，总 KV 容量被浪费 N 倍。切换到亲和后 `kv_cache_usage_perc` 应明显下降。

**修**：亲和路由；或用共享 KV 存储（LMCache 一类）让多个 replica 共用远端缓存，代价是网络带宽和额外延迟。

### 2.11 benchmark 用错 dataset

**确认**：用 `--dataset-name random` 或 ShareGPT 测出来的数字和线上对不上。random 没有共享前缀，命中率永远接近 0；ShareGPT 是单轮为主。

**修**：见第 4 节，用线上回放集。vLLM bench 的多轮 dataset 支持随版本变化，可用时优先用它加自己的 session 数据。

## 3. 决策表

| 症状组合 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| 单实例命中率低 | diff 连续两次 prompt，修前缀 | 换 chat template | 调 scheduler 参数 |
| 单实例命中率高、多实例低 | 网关 session 亲和 | prefix-aware router | 加更多 replica 轮询 |
| 命中率高、TTFT 仍高 | 看新增 token 数与排队时间 | grammar 开销 | 调 budget |
| 空闲后首次调用慢 | 算 KV 保活容量，offload | fp8 KV | 客户端盲目保活 |
| 任务级长尾 | agent 流量独立 priority 或实例池 | 限制步数、并行工具 | 只优化单次 P50 |
| API server CPU 满 | `--api-server-count`、精简 tool schema | 非流式 tool call | 加 GPU |

### 排查 checklist

1. 从 `/metrics` 差分算出最近 10 分钟的 prefix 命中率，低于 50% 先查前缀。
2. 抓同一 session 连续两次请求的渲染后 prompt 做 diff，找第一个差异位置。
3. 多实例部署确认网关是否有 session 亲和。
4. 统计每次调用的新增 token 数分布，P90 超过 2k 就找应用层压 tool 返回。
5. 抓 API server 火焰图，看 parser 和 template 渲染占比。
6. 看 `num_preemptions_total` 和空闲后首次调用 TTFT，判断 KV 保活容量。
7. 任务级延迟按步分解，找出哪几步是长尾来源。

### 常见误判

- 命中率是按 block 数算的，长 prompt 里哪怕最后 10% 未命中，命中率也显示 90%，但如果那 10% 是 3k token，TTFT 仍然不低。要同时看"未命中 token 数"。
- 把 agent 流量和普通聊天流量混在一个实例上调参，两者的最优 budget 和 `max_num_seqs` 方向相反。
- 用单次调用 P50 评估 agent 体验。用户感知的是任务完成时间，由步数和长尾决定。

## 4. 验证实验

不能用 random dataset。构造一个真实形态的 agent 回放集：从线上抽 200 个 session，每个 session 保留完整的调用序列（每次的 messages 和 tool 结果），benchmark 按 session 串行、session 间并发重放。

```bash
# 服务端对照：单实例 vs 两实例轮询 vs 两实例 session 亲和
vllm serve <m> --port 8000
vllm serve <m> --port 8001
# 网关用简单 hash(session_id) % 2 做亲和，对照组用轮询

# 客户端：自写重放脚本，记录每次调用的 TTFT、新增 token 数、命中率（从 /metrics 差分）
# 预期：亲和路由下命中率接近单实例，TTFT P50 下降到新增 token 对应的水平

# 实验 2：前缀稳定性
# 把 system prompt 里的时间戳从秒改到天，重放同一集合
# 预期：命中率从接近 0 跳到 80% 以上（示例数字）

# 实验 3：保活容量
# 并发 session 数从 50 扫到 500，观察 num_preemptions_total 和空闲 5 分钟后首次调用的 TTFT
# 预期：超过 KV 容量后首次调用 TTFT 阶跃上升，那就是需要 offload 或加卡的点
```

### 回放脚本骨架

```python
# replay.py：按 session 串行、session 间并发重放，记录每步 TTFT 与新增 token
import asyncio, json, time, requests
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="http://gateway:8000/v1", api_key="x")

def hits():
    t = requests.get("http://gateway:8000/metrics").text
    g = lambda k: float([l for l in t.splitlines() if l.startswith(k)][-1].split()[-1])
    return g("vllm:prefix_cache_hits"), g("vllm:prefix_cache_queries")

async def run_session(sess):
    for step in sess["steps"]:                      # 每步是一份完整 messages
        t0 = time.perf_counter(); first = None
        stream = await client.chat.completions.create(
            model=sess["model"], messages=step["messages"], tools=step.get("tools"),
            max_tokens=step.get("max_tokens", 256), stream=True,
            extra_headers={"x-session-id": sess["id"]})   # 网关据此做亲和
        async for ch in stream:
            if first is None: first = time.perf_counter() - t0
        step["ttft"] = first

async def main(path, concurrency):
    sessions = [json.loads(l) for l in open(path)]
    h0, q0 = hits(); sem = asyncio.Semaphore(concurrency)
    async def guarded(s):
        async with sem: await run_session(s)
    await asyncio.gather(*(guarded(s) for s in sessions))
    h1, q1 = hits()
    print("hit rate", (h1 - h0) / max(q1 - q0, 1))

asyncio.run(main("sessions.jsonl", 32))
```

固定变量：同一份 `sessions.jsonl`、同一并发、同一模型和 chat template。改一个变量（路由策略、system prompt 结构、KV 配置）重跑，对比命中率和 TTFT 分布。

## 5. 关联

- 总入口：[[01-diagnosis-playbook]]
- TTFT 与结构化输出：[[02-ttft-high]]、[[09-structured-output-tool-calls]]
- KV 容量与长 session：[[05-oom-preemption]]、[[07-long-context]]
- 路由与多实例：[[13-parallelism-choice]]、[[day18-data-parallel]]
- RadixAttention 对比：[[day23-sglang]]
- 机制来源：[[day09-scheduler]]、[[day10-kv-cache]]
