---
title: "Day 24：Agent Rollout 基础设施：长轨迹、异步循环与成本"
type: concept
tags: [llm-training, agentic-rl, rollout, async, infrastructure]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 24：Agent Rollout 基础设施：长轨迹、异步循环与成本

> 上一课 [[llm-training-30d/week4/day23-agent-environments-gyms]] · 下一课 [[llm-training-30d/week4/day25-agent-reward-credit-assignment]]。相关：单轮 rollout 引擎与权重同步 [[llm-training-30d/week3/day19-rollout-engine-weight-sync]]；异步与 staleness [[llm-training-30d/week3/day20-async-rl-stability]]；推理侧多轮 workload 的优化 [[ai-infra-30d/optimization/08-multi-turn-agent-workload]]；轨迹 schema [[agent-rsi-30d/week1/day04-tracing-replay]]。

## 学习目标

1. 画出 agent rollout 循环在训练系统里的位置：谁驱动环境、谁驱动引擎、样本何时进队列。
2. 解释 agent rollout 为什么比单轮 rollout 更需要异步，并给出并发模型（每条轨迹一个协程、引擎 continuous batching 吸收）。
3. 会做 partial rollout 与断点续跑，说清权重版本如何按段记录。
4. 能核算一条轨迹的成本（模型 token、工具时间、沙箱占用），并据此决定 rollout 与训练的资源比。
5. 动手：实现 128 条轨迹并发的 async agent loop，接入 vLLM，输出 Day 22 格式的样本。

## 工业现状

agent RL 的 rollout 是一个"生成 ↔ 环境"交替的循环，不能像单轮那样一次 `generate()` 完成。公开系统的做法：Kimi K2（T19）与 k1.5（T18）用异步 rollout 加 partial rollout 处理长轨迹；AReaL（T44）把每条轨迹作为独立异步任务，引擎侧靠 continuous batching 合并；verl 的 agent loop（B06）用 asyncio 驱动每条轨迹，SkyRL（B12）与 ROLL（B13）类似。共识是：**rollout 的并发单位是轨迹，不是 batch**；batch 只在训练侧出现。

成本上，agent 任务里模型 token 常不再是大头：SWE 类任务的沙箱 CPU 时间与容器内存可能超过 GPU 成本，报告里开始出现"每条轨迹的总成本"而非只算 token。

公开系统的 rollout 组织方式（以报告为准）：

| 系统 | 并发单位 | 权重更新时的处理 | 引擎 |
|---|---|---|---|
| Kimi k1.5 / K2（T18、T19） | 轨迹 | partial rollout，跨版本续采 | 自研 + 开源引擎 |
| AReaL（T44） | 轨迹，全异步 | 样本带版本，decoupled PPO | SGLang |
| verl agent loop（B06） | 轨迹（asyncio） | 同步为主，async rollout 可选 | vLLM / SGLang |
| SkyRL（B12） | 轨迹 | 异步 | vLLM / SGLang |

## 核心原理

### 循环的位置

```
                    ┌──────────────── rollout 服务 ────────────────┐
任务队列 ──take──▶ │ trajectory task ×N（asyncio）                   │
                    │   for turn in ...:                              │
                    │     text  = await engine.generate(h_t)  ──────┼──▶ vLLM/SGLang（continuous batching）
                    │     obs   = await env.step(parse(text)) ──────┼──▶ 环境服务（Day 23）
                    │   reward = await verifier(env.final)           │
                    │   sample = build_sample(traj)  → 样本队列       │
                    └─────────────────────────────────────────────────┘
                                          │
                                   trainer 取 batch（Day 20）
```

引擎不知道"轨迹"的存在，它只看到源源不断的 prompt；一条轨迹的第 k 轮 prompt 是前 k−1 轮的完整历史，所以 prefix caching 命中率对 agent rollout 吞吐至关重要（推理课优化专项 08 的内容在这里直接生效）。

### 为什么必须异步

同步做法：取 B 条任务，逐轮"所有轨迹一起 generate，再一起 step"。问题：轨迹轮数不同（3 到 50 轮），每轮长度不同，环境 step 时间不同。同步会让每一轮都等最慢的，GPU 与环境交替空转。

异步做法：每条轨迹独立协程，`await` 引擎与环境；引擎的 continuous batching 自动把各轨迹当前处于生成阶段的请求合批。GPU 利用率由"任意时刻处于生成阶段的轨迹数"决定，需要并发轨迹数远大于目标 batch（示例：目标同时 256 条在生成，则并发 512 到 1024 条轨迹，因为一半时间在等环境）。

### partial rollout 在 agent 场景

Day 20 的 partial rollout 是按 token 切；agent 场景按 turn 切更自然：权重更新到来时，正在等环境的轨迹在下一轮用新权重，`gen_version` 按 turn 记录。断点续跑需要保存 `h_t` 与环境快照（Day 23）。

### 并发模型的资源约束

| 资源 | 约束 | 调节 |
|---|---|---|
| 引擎 KV cache | 同时活跃的 prompt 总 token（长历史 × 并发） | 并发上限、max_model_len、prefix cache、FP8 KV |
| 环境实例 | 池大小 | 并发轨迹 ≤ 池大小 + 等待队列 |
| rollout 进程 CPU | asyncio 单线程 + 解析 + HTTP | 多进程分片任务 |
| 样本队列内存 | 长轨迹 × 队列长度 | 落盘或限流 |

### 成本核算

```
cost(traj) = Σ_turns (prompt_tokens_k × c_in + gen_tokens_k × c_gen)     # 模型侧，prefix 命中部分 c_in 打折
           + Σ_steps env_time_k × c_env                                    # 环境 CPU/内存
           + sandbox_wall_time × c_sandbox                                 # 容器占用（含等待）
```

`c_in` 与 `c_gen` 换算自 GPU 小时价与实测 tokens/s（推理课优化专项 15）。长历史使 `prompt_tokens_k` 随 k 线性增长，没有 prefix cache 时模型侧成本随轮数平方增长。

rollout 与训练的卡数比：`R = (平均轨迹 rollout 时间 × batch) / 训练一个 batch 的时间`，agent 场景常见 R 在 3 到 8（示例），比单轮 RLVR 高。

### 训练侧如何消费 agent 样本

长度方差大的轨迹直接组 batch 会让 padding 浪费超过一半。做法：按 token 总长分桶后 packing（Day 3 的 varlen attention 隔离），micro-batch 按 token 数而不是样本数切；observation 段虽不算 loss 但占 forward 算力，所以"每 micro-batch 的 mask=1 token 数"才是有效吞吐。序列超过训练上限的样本：截断 observation 的中间段（保留头尾）优于丢弃，但要与 rollout 时模型看到的一致，否则回到 Day 22 的一致性问题。

### 故障恢复

| 故障 | 影响 | 处理 |
|---|---|---|
| 引擎实例崩溃 | 该实例上所有活跃轨迹的当前轮失败 | 轨迹标记 env_error 丢弃或从上一轮 checkpoint 重试一次；不计入 reward |
| 环境服务不可用 | 大量 timeout | 熔断：暂停取任务，不把 timeout 当模型错误 |
| rollout 进程重启 | 未入队样本丢失 | 轨迹 trace 落盘，重启后从 trace 重建已完成样本 |
| 权重同步失败于部分实例 | 版本混杂 | 同步用两阶段：全部就绪再切版本号 |

### 质量监控

rollout 不只是吞吐问题。每个 step 必须看的分布：轮数、生成 token 占比、工具错误率（parse error、timeout、环境错误分开）、超时终止比例、reward 非零率、prefix cache 命中率、每条轨迹成本。任何一项突变都意味着策略或环境出了问题，且会先于 reward 曲线显现。

## 实现步骤

### 1. 最小 async agent loop

```python
import asyncio, time
from openai import AsyncOpenAI

engine = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="x")   # vLLM，训练时由框架内嵌

async def run_trajectory(task, env_pool, version, max_turns=20):
    env = await env_pool.acquire()
    traj = {"task": task, "turns": [], "version": []}
    try:
        obs = await env.reset(task, seed=task["seed"])
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": obs}]
        for turn in range(max_turns):
            t0 = time.perf_counter()
            resp = await engine.chat.completions.create(model=MODEL, messages=messages,
                        temperature=1.0, top_p=1.0, max_tokens=1024, logprobs=True, tools=TOOLS)
            t_gen = time.perf_counter() - t0
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            traj["version"].append(version.current)
            action = parse_action(msg)                      # tool_call 或 final
            if action["type"] == "final":
                break
            t0 = time.perf_counter()
            obs, _, done, info = await env.step(action)
            t_env = time.perf_counter() - t0
            messages.append({"role": "tool", "content": obs[:8000], "tool_call_id": action["id"]})
            traj["turns"].append({"t_gen": t_gen, "t_env": t_env, "info": info})
            if done: break
        traj["reward"] = await verifier(env, task)
        traj["messages"] = messages
        return traj
    finally:
        await env_pool.release(env)

async def rollout_service(task_queue, sample_queue, env_pool, version, concurrency=512):
    sem = asyncio.Semaphore(concurrency)
    async def worker(task):
        async with sem:
            traj = await run_trajectory(task, env_pool, version)
            if traj.get("env_error"): return               # 环境错误不进训练
            await sample_queue.put(build_sample(traj))     # Day 22 的格式
    while True:
        task = await task_queue.get()
        asyncio.create_task(worker(task))
```

要点：`tools=TOOLS` 让引擎的 tool parser 生成结构化 tool_call，训练侧 mask 要与该模板一致（Day 22 实验 1）；`logprobs=True` 供 TIS；observation 在这里截断，训练侧看到的就是模型看到的。

### 2. 接入 verl agent loop

verl 的 agent loop 把上面的循环实现为 `AgentLoopBase` 子类（类名与方法以 B06 为准）：`run()` 内部调用 `self.server_manager.generate()`（异步引擎）与注册的 tool，返回 `AgentLoopOutput`（prompt_ids、response_ids、response_mask、num_turns、metrics）。自定义环境只需实现 tool 接口；若环境有状态，在 tool 的 `create()` 里 acquire 实例、`release()` 里归还。

### 3. partial rollout 与断点

```python
# 权重版本更新时不打断协程；下一轮 generate 自动用新权重（引擎已同步）
# 需要暂停时（如 colocated 模式要把 GPU 让给训练）：
async def checkpoint(traj, env):
    traj["env_snapshot"] = await env.snapshot()
    await store.put(traj["id"], traj)
# 恢复：env.restore(snapshot)，messages 原样，继续循环
```

colocated 模式下 agent rollout 通常要求"本轮所有轨迹要么完成要么 checkpoint"才能切换到训练，这是它在 agent 场景不如分离模式的原因。

### 4. 引擎侧配置

为 agent rollout 起的 vLLM 实例：开 prefix caching（默认）、`max_model_len` 按 P99 轨迹长度、`max_num_seqs` 大于并发轨迹数中处于生成态的部分、`--enable-auto-tool-choice --tool-call-parser <模型对应>`、`--reasoning-parser`（若为 thinking 模型）。多实例时用 session-affinity 路由让同一轨迹落到同一实例以命中 prefix cache（推理课 Day 18）。

### 5. 指标与 trace

每条轨迹写一条 trace（复用 Agent 课 Day 4 的 schema）：每轮的 `t_gen`、`t_env`、token 数、tool 名、info、版本号。聚合到面板：轮数直方图、`t_env/t_gen` 比、工具错误率、prefix 命中率、成本分布。

### 6. 多实例引擎与路由（知道即可）

rollout 并发超过单实例 KV 容量时起多个引擎实例。要求：同一轨迹的所有轮次落到同一实例（session affinity，按轨迹 id 哈希），否则 prefix cache 全部失效；权重同步要覆盖所有实例并在同一版本号下完成，未完成同步的实例暂停接单。推理课 Day 18 与优化专项 08 讲了路由层的实现选择。

### 7. 记录模板

每次 rollout 运行记录：引擎版本与配置、环境版本哈希、并发数、权重版本范围、轨迹数、平均/P99 轮数与长度、工具错误率、prefix 命中率、总 GPU 时间与环境 CPU 时间、每轨迹成本。Day 30 capstone 的成本报告直接用这张表。

## 实验

资源分层：单卡跑 0.5B 到 1.5B 模型 + 本地代码执行环境即可完成实验 1 到 3；实验 4 需 2 卡以上。

### 实验 1：同步 vs 异步 rollout 吞吐

同一 256 个任务（代码执行环境，3 到 10 轮），同步实现（逐轮合批）与异步实现各跑一次。记录总时间、GPU 利用率时间线、环境实例平均占用率。预期：异步总时间显著更短，利用率曲线平稳。

### 实验 2：并发数扫描

异步实现，并发 64 / 256 / 1024。记录吞吐（轨迹/分钟）、引擎 `num_requests_running` 均值、KV 使用率、环境池等待时间。预期：吞吐先线性上升后被 KV 或环境池卡住；找到你配置的拐点。

### 实验 3：prefix cache 的作用

引擎开 / 关 prefix caching，同一 workload。记录每轮 TTFT 随轮数的变化与总 GPU 时间。预期：关闭时 TTFT 随轮数线性增长，GPU 时间成倍增加。

### 实验 4：rollout 与训练资源比

在 Day 21 的 pipeline 上换成 agent 任务，分离模式，rollout 卡数 1/2/4 对训练 1 卡。记录 trainer 等待时间占比与 staleness 分布。预期：R 不足时 trainer 空等；R 过大时 staleness 尾部变厚。

### 实验 5：partial rollout 与断点续跑

运行中人为触发一次权重版本切换（模拟同步），观察正在等环境的轨迹在下一轮是否用新版本、`version` 数组是否按 turn 正确记录；再人为杀掉 rollout 进程，从 trace 与环境快照恢复，校验恢复后的轨迹与未中断运行在 observation 上一致。预期：两项校验都通过，恢复代价为快照大小与重建时间。

### 实验 6：成本分解

对实验 2 中并发 256 的一次运行，把每条轨迹的成本按模型 prompt、模型生成、环境 CPU、沙箱等待四项拆开，画分布。预期：SWE 类任务里环境与沙箱项占比可与模型项相当；无 prefix cache 时 prompt 项主导。这张图决定 Day 30 capstone 里优先优化哪一项。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| GPU 利用率低但环境池满 | 环境是瓶颈 | `t_env/t_gen` 比 | 扩环境池、加速 reset、服务化 |
| GPU 利用率低且环境空闲 | 并发轨迹数不够 | 处于生成态的轨迹数 | 提高并发上限 |
| 引擎 OOM 或大量抢占 | 长历史 × 并发超过 KV | `num_preemptions_total`、KV 使用率 | 降并发、FP8 KV、截断 observation、缩 max_model_len |
| TTFT 随轮数线性增长 | prefix cache 未命中（多实例无亲和、模板每轮变化、cache_salt） | 引擎 `prefix_cache_hits/queries` | session affinity；固定模板；检查 system prompt 是否含时间戳 |
| 训练侧 mask 与 rollout 不一致 | 引擎 tool parser 输出的 tool_call 与训练模板序列化不同 | 用训练模板重新序列化 rollout 消息，对比 token | 训练侧用引擎返回的原始文本而不是重新序列化；或统一模板 |
| 样本队列内存暴涨 | 长轨迹积压 | 队列长度与平均样本大小 | 限流、落盘 |
| 部分轨迹永远不结束 | 环境 step 挂起无超时 | 最老的活跃轨迹年龄 | 每步超时 + 轨迹总时长上限 |
| 成本失控 | 轮数上升、工具滥用 | 轮数与工具调用次数分布 | 步数上限、成本进 reward（Day 25） |
| 权重同步期间轨迹 logprob 混版本 | partial rollout 未按 turn 记版本 | 检查 version 数组 | 按 turn 记录，训练侧按段修正 |

## 验收标准

- async agent loop 在 128 并发下稳定运行，输出的样本通过 Day 22 的 mask 验证。
- 实验 1 到 3 的数据与结论；能说出你配置的并发拐点及其瓶颈资源。
- 每条轨迹有成本核算，rollout/训练资源比有依据。
- 质量监控面板包含本页列出的全部分布。

## 自测

1. 并发 512 条轨迹、平均一半时间在等环境，引擎实际同时处理多少请求？若目标是 batch 256，KV 该按多少并发规划？
2. partial rollout 按 turn 记版本与按 token 记版本，训练侧修正有什么不同？
3. 为什么 colocated 模式在 agent 场景要求"全部完成或 checkpoint"才能切换？分离模式为什么没有这个问题？
4. 一条轨迹 20 轮、每轮历史平均 6k token，有无 prefix cache 时模型侧 prompt 处理量各是多少？

## 交付物

| 文件 | 内容 |
|---|---|
| `rollout/agent_loop.py` | 异步 rollout 服务，输出训练样本 |
| `rollout-throughput.md` | 实验 1、2、3 数据 |
| `trajectory-cost.md` | 成本模型与实测分布、资源比建议 |
| `rollout-dashboard.md` | 质量监控指标与截图 |

## 参考

- T18 Kimi k1.5、T19 Kimi K2：partial rollout 与大规模 agent rollout 的工程描述。
- T44 AReaL：轨迹级异步的系统设计。
- T42 HybridFlow：verl agent loop 所在的整体架构。
- B06、B12、B13：agent loop / env 接口实现。
- 推理课优化专项 08、15：多轮 workload 的 prefix cache 与成本换算。
