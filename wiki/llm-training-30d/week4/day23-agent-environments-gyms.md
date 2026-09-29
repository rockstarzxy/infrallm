---
title: "Day 23：Agent 环境与 Gym：接口、沙箱规模化与可复现"
type: concept
tags: [llm-training, agentic-rl, environment, sandbox, gym]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 23：Agent 环境与 Gym：接口、沙箱规模化与可复现

> 上一课 [[llm-training-30d/week4/day22-agentic-rl-formalization]] · 下一课 [[llm-training-30d/week4/day24-agent-rollout-infrastructure]]。相关：Agent 课的工具与沙箱 [[agent-rsi-30d/week1/day03-tools-environments-sandbox]]（那边讲单 Agent 的隔离执行，这里讲训练规模下每秒上百个环境实例的工程）。

## 学习目标

1. 用 gymnasium 约定定义一个 LLM agent 环境，并说清每个方法在 RL 训练循环里被谁调用。
2. 能估算一个环境在训练规模下的成本（每步启动时间、内存、并发上限），并给出池化与快照方案。
3. 设计超时、失败、非确定性的处理规则，使 reward 不被环境噪声污染。
4. 把环境版本化并做到轨迹可回放。
5. 动手：实现一个代码执行环境与一个检索环境，压到 64 并发。

## 工业现状

训练规模的 agent 环境是 2025 年各家报告里投入最大的基础设施之一：Kimi K2（T19）描述了大规模合成工具环境；SWE-RL（T45）与 SWE-bench 类任务（T46）依赖每个任务一个可执行仓库容器；GLM-4.5（T20）在代码、搜索、网页任务上做 RL。开源侧，verl 的 agent loop 与 tool 接口、SkyRL 的 gym 抽象（B12）、ROLL（B13）都把环境做成独立服务，训练器通过 RPC 或 HTTP 调用。

共同经验：环境启动成本和非确定性是两大杀手。SWE 类任务每个 episode 需要一个带依赖的容器，冷启动几十秒；网页与搜索任务依赖外部服务，同一动作不同时间返回不同结果。因此工业实现都做了容器池化、镜像预热、结果缓存与环境版本冻结。

公开系统的环境形态（以报告为准）：

| 系统 | 环境类型 | 规模描述 | 关键工程 |
|---|---|---|---|
| SWE-RL（T45） | GitHub 仓库演化数据，规则奖励 | 百万级 PR 来源 | 无需执行测试，奖励为 patch 相似度，规避沙箱成本 |
| SWE-bench 类训练（T46） | 每任务一个可执行容器 | 千到万级任务 | 镜像预构建、测试隔离 |
| Kimi K2（T19） | 合成工具与 MCP 风格工具集 | 数千工具、大规模任务 | 任务与 rubric 合成、模拟器 |
| GLM-4.5（T20） | 代码、搜索、网页 | 未公开规模 | 与推理任务混合课程 |
| SkyRL（B12） | gym 抽象，SWE / 搜索 / 文本游戏 | 开源示例 | 环境与训练解耦 |

## 核心原理

### 接口

```python
class AgentEnv:                      # gymnasium 风格（B15），LLM 场景的最小接口
    def reset(self, task: dict, seed: int) -> Observation:   # 返回初始 observation（任务描述、初始工具结果）
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        # action：模型这一轮的生成（解析后的 tool_call 或最终答案）
        # 返回：observation 文本、step reward（多为 0）、done、info（错误类型、耗时、成本）
    def close(self): ...
    def snapshot(self) -> bytes / restore(self, blob): ...    # 可选，用于分支采样与回放
```

调用关系：

```
rollout worker（Day 24）
   ├─ env = pool.acquire(task) ; obs = env.reset(task, seed)
   ├─ loop:
   │     text = engine.generate(h_t)            ← 模型
   │     action = parse(text)                   ← 工具调用解析（失败也是一个 observation）
   │     obs, r, done, info = env.step(action)  ← 环境
   │     h_{t+1} = h_t + text + obs
   └─ pool.release(env) ; reward = verifier(env.final_state, task)
```

必须理解：环境 `step` 的返回是 observation，不是 reward 的主要来源；终局 reward 由 verifier 在 episode 结束后基于环境最终状态计算（Day 25）。

### 环境类型与成本

| 类型 | 实例 | 每 episode 启动成本 | 并发瓶颈 | 非确定性来源 |
|---|---|---|---|---|
| 无状态工具 | 计算器、单次代码执行 | 毫秒到百毫秒 | CPU | 几乎无 |
| 有状态容器 | SWE 仓库、数据库、文件系统 | 秒到几十秒（冷）；池化后亚秒 | 内存、磁盘 IO | 依赖版本、时间戳 |
| 外部服务 | 搜索、网页、API | 网络 RTT | 速率限制、费用 | 结果随时间变化 |
| 模拟器 | 文本游戏、GUI 模拟 | 取决于实现 | CPU/GPU | 随机种子 |
| LLM 模拟用户 / 工具 | 用另一个模型扮演用户或工具 | 一次推理 | 推理算力 | 采样 |

### 池化与快照

```
镜像预热：所有依赖装进镜像；启动时只做 git checkout 与数据挂载
池化：   预先启动 K 个容器待命；acquire 时 reset 到干净快照；release 时后台恢复
快照：   文件系统层（overlayfs）或进程级（CRIU）快照，reset 从快照恢复而非重装
分支：   同一 h_t 下采 n 条续写时，从同一快照 fork n 个实例（GRPO 的 group 采样需要）
```

容量估算（示例）：每个容器 2 GB 内存、reset 0.5 s、平均 episode 60 s；256 并发需要约 512 GB 内存与 256 个待命实例，reset 吞吐 512/s 远超需求，瓶颈在内存与磁盘。

### 超时、失败与非确定性的规则

| 事件 | 处理 | 理由 |
|---|---|---|
| 工具调用解析失败 | 作为 observation 返回错误信息，episode 继续，计入 `info.parse_error` | 模型应学会修正格式；直接终止会让 reward 与格式强耦合 |
| 单步超时 | observation 返回 "timeout"，计数；连续 N 次终止 | 避免死循环占用实例 |
| 环境崩溃（非模型原因） | 整条轨迹丢弃，不进训练，记录 | 否则 reward 噪声进入梯度 |
| 外部服务失败 | 重试一次；仍失败按环境崩溃处理 | 同上 |
| 越权动作（删库、访问外网） | 拒绝执行并返回策略提示，可记负 step reward | 安全边界 |
| 达到最大轮数 | done=True，终局 reward 按未完成算 | episode 必须有界 |

原则：**模型的错误进 reward，环境的错误不进**。要做到这一点，`info` 里必须能区分两者。

### 版本化与回放

环境版本 = 镜像 digest + 任务集版本 + 工具实现版本 + 外部服务快照（搜索结果缓存）的哈希。每条轨迹记录环境版本；回放时用同一版本重放动作序列，比对 observation 是否一致。不一致的环境不能用于 off-policy 数据复用，也不能做公平的评测。

### 动作解析与动作 schema

环境只接受结构化动作。解析层放在环境之外（rollout worker），失败时生成一条 observation 而不是抛异常：

```python
ACTION_SCHEMA = {"run_python": {"code": str}, "read_file": {"path": str},
                 "edit_file": {"path": str, "diff": str}, "run_tests": {}, "final": {"answer": str}}

def parse_action(msg):
    if msg.tool_calls:
        tc = msg.tool_calls[0]                       # 多个 tool_call 时按框架约定串行或并行
        name, args = tc.function.name, json.loads(tc.function.arguments or "{}")
        if name not in ACTION_SCHEMA or set(args) - set(ACTION_SCHEMA[name]):
            return {"type": "parse_error", "detail": f"bad tool {name} / args {list(args)}", "id": tc.id}
        return {"type": name, "id": tc.id, **args}
    return {"type": "final", "answer": msg.content or ""}
```

设计取舍：

- `step` 是否允许一次多个动作：并行工具调用能省轮数，但 observation 顺序与 reward 归属更复杂；训练初期建议串行。
- 动作参数上限：`code`、`diff` 超过阈值直接拒绝并返回错误，防止模型用超长动作撑爆序列。
- `final` 的判定：没有 tool_call 即视为最终答案，还是要求显式标记？前者简单但会把"模型忘了调工具"当成结束。

### 环境即数据集

任务生成决定 RL 学到什么。三种来源：

- **真实任务**：SWE-bench 风格从真实 PR 构造（T46），质量高、数量有限、需要去污染。
- **合成任务**：用模型生成任务 + 解法 + 测试（Kimi K2 的做法，T19），数量大、需要 verifier 过滤。
- **程序化生成**：从模板与随机种子生成（text-to-SQL、数学工具题），可无限扩、难度可控。

难度分层：用当前策略采样 8 条测 pass rate，按 0 / (0,1) / 1 分桶；训练只用中间桶（DAPO 的 dynamic sampling 在环境侧的等价物），定期重测让桶随策略移动。

## 实现步骤

### 1. 代码执行环境（无状态，先做）

```python
import subprocess, tempfile, resource, json

class PyExecEnv:
    def reset(self, task, seed):
        self.task = task; self.steps = 0
        return task["prompt"]
    def step(self, action):
        self.steps += 1
        if action["type"] != "run_python":
            return "error: unknown tool", 0.0, False, {"parse_error": True}
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(action["code"]); path = f.name
        try:
            out = subprocess.run(["python", path], capture_output=True, text=True, timeout=10,
                                 preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (2**30, 2**30)))
            obs = (out.stdout + out.stderr)[-4000:]          # 截断 observation，训练侧一致
            info = {"rc": out.returncode}
        except subprocess.TimeoutExpired:
            obs, info = "error: timeout", {"timeout": True}
        done = self.steps >= self.task.get("max_steps", 6)
        return obs, 0.0, done, info
```

生产上换成容器或 gVisor/firecracker 隔离，接口不变。

### 2. 有状态容器环境（SWE 类）

```
镜像：base + 仓库依赖，每个任务一个 tag
reset：docker run --rm -d --memory 4g --cpus 2 <image>；docker exec git checkout <commit>
step：docker exec 执行 shell / 编辑 / 测试命令，stdout 截断到 8k 字符
final_state：docker exec git diff 得到 patch，供 verifier 跑 hidden tests
池：预启动 K 个容器，reset 用 git clean + checkout 而不是重建容器
```

### 3. 环境服务化

把环境放到独立进程或机器，暴露 HTTP：`POST /reset`、`POST /step`、`POST /close`，返回 JSON。rollout worker 只做 HTTP 调用，环境机器按 CPU/内存独立扩容。用 `env_id` 做会话，超时自动回收。

### 4. 接入 verl agent loop / SkyRL

verl：实现 tool 类（`create` / `execute` / `calc_reward` / `release`，方法名以 B06 为准），注册到 tool config，agent loop 按 tool_call 分发。SkyRL：实现 `BaseTextEnv` 的 `init` / `step`，返回 `observations`、`reward`、`done`（以 B12 为准）。两者都要求 `step` 可并发、无全局状态。

### 5. 版本与回放工具

```python
def env_version(image_digest, task_set_sha, tool_sha, cache_sha):
    return hashlib.sha256("|".join([image_digest, task_set_sha, tool_sha, cache_sha]).encode()).hexdigest()[:12]

def replay(trace, env):
    obs = env.reset(trace.task, trace.seed)
    for step in trace.steps:
        obs2, r, done, info = env.step(step.action)
        assert obs2 == step.observation, f"divergence at step {step.idx}"
```

## 实验

资源分层：全部实验只需 CPU 与少量模型调用（API 或本地小模型）。

### 实验 1：并发压测

代码执行环境，64 并发，每个 episode 4 步，跑 500 episode。记录 P50/P99 step 延迟、CPU、失败率。预期：无隔离时 P99 由最慢脚本决定；加 rlimit 与 timeout 后 P99 有上界。

### 实验 2：容器池化收益

SWE 类环境，对比"每 episode 新建容器"与"池化 + git clean reset"，16 并发 100 episode。记录每 episode 启动时间与内存峰值。预期：池化后启动时间从秒级降到亚秒，内存峰值由池大小决定。

### 实验 3：非确定性审计

检索环境，同一 20 个查询间隔 1 小时各跑一次，统计结果差异率。启用结果缓存后重跑。预期：无缓存时差异率非零，缓存后为 0，回放通过。

### 实验 4：难度分桶

用当前策略在 200 个任务上采 8 条，统计三个桶的比例。预期：训练前大量在 0 桶；这就是为什么直接在全集上 RL 信号稀疏。

### 实验 5：安全边界测试

用 20 条故意越权的动作（访问外网、读 `/etc/passwd`、fork 炸弹、写 10 GB 文件、sleep 600）打环境，逐条确认被拒绝或被限制，并检查 `info` 是否正确标记。预期：全部被环境层拦下，且没有任何一条影响同时运行的其他 episode。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| rollout 吞吐远低于引擎能力 | 环境 step 慢或串行 | rollout worker 里模型与环境时间分开计时 | 服务化 + 并发；池化 |
| reward 噪声大，同一轨迹重放结果不同 | 环境非确定 | 回放校验 | 冻结外部服务、缓存、固定 seed |
| 训练几步后模型学会触发环境错误 | 环境错误被算成 reward 或终止收益更高 | 看 info 分布随 step 变化 | 环境错误的轨迹丢弃；错误 observation 不给正 reward |
| 内存被容器吃满 | 池太大或泄漏 | 监控 | 池上限 + 回收 |
| 同一任务 n 条轨迹 observation 相同 | 环境 reset 未真正清理 | 比较 reset 后状态哈希 | 快照恢复 |
| 模型输出的 tool_call 格式总是不被识别 | parser 与模板不一致 | parse_error 比例 | 统一模板；容错解析但记录 |
| 回放 divergence | 环境版本变了 | 版本哈希 | 版本化，旧轨迹标记不可复用 |

## 安全边界 checklist

训练规模的环境会执行模型生成的任意代码，必须在环境层而不是 prompt 层设边界：

- 网络：默认断网；需要外网的工具走白名单代理并记录。
- 文件系统：只读挂载基础镜像，工作目录为临时层，episode 结束销毁。
- 资源：CPU、内存、进程数、磁盘配额 rlimit / cgroup；单步与总时长超时。
- 凭据：容器内没有任何真实密钥；测试用的密钥为假值。
- 输出：observation 大小上限，防止模型用超长输出撑爆训练序列。
- 审计：每条越权尝试记录到 info 并进入 Day 25 的 reward hacking 审计。

## 自测

1. 一个环境 `step` 平均 2 秒、reset 0.5 秒、episode 平均 8 步，要支撑引擎 256 并发生成，环境池至少多大？（提示：生成与环境时间的比例决定处于环境态的轨迹比例）
2. 搜索结果随时间变化，为什么会让 GRPO 的 group 归一化失真？
3. 模型生成的代码把 `/tmp` 写满导致后续 episode 全部失败，这些失败轨迹该不该进训练？为什么？

## 验收标准

- 两个环境（代码执行、检索或 SWE）通过 64 并发压测，有 P99 数据。
- 超时/失败/越权的处理规则写成文档并在代码中实现，`info` 能区分模型错误与环境错误。
- 环境版本哈希与回放校验通过。
- 任务难度分桶统计。

## 交付物

| 文件 | 内容 |
|---|---|
| `envs/py_exec_env.py`、`envs/swe_env.py` 或 `envs/search_env.py` | 环境实现 |
| `env-load-test.md` | 实验 1、2 数据 |
| `env-determinism.md` | 实验 3 与回放结果 |
| `task-difficulty.md` | 实验 4 分桶 |

## 参考

- T46 SWE-bench、T45 SWE-RL：真实软件环境的构造与 reward。
- T19 Kimi K2：大规模合成工具环境。
- T37 DAPO：dynamic sampling，对应环境侧的难度分桶。
- B12 SkyRL、B13 ROLL、B06 verl：环境/工具接口。
- B15 gymnasium：接口约定。
