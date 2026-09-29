---
title: "Day 22：Agentic RL 的形式化：多轮 POMDP、动作粒度与 loss mask"
type: concept
tags: [llm-training, agentic-rl, pomdp, multi-turn, credit-assignment]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 22：Agentic RL 的形式化：多轮 POMDP、动作粒度与 loss mask

> 上一课 [[llm-training-30d/week3/day21-p2-project-grpo]] · 下一课 [[llm-training-30d/week4/day23-agent-environments-gyms]]。相关：Agent 课的 RL 数学 [[agent-rsi-30d/week3/day15-agent-rl-foundations]]（那边讲 POMDP 与策略梯度本身，这里讲训练系统怎么实现它）；轨迹 schema [[agent-rsi-30d/week1/day04-tracing-replay]]。

## 学习目标

1. 把一个多轮工具 Agent 任务写成 POMDP，并明确 state、observation、action、reward、episode 边界。
2. 说清 token 级动作与 turn 级动作的区别，以及两者对 loss、advantage、KL 的影响。
3. 能从一条真实轨迹构造训练样本：哪些 token 进 loss、哪些 mask、position 与 attention 怎么拼。
4. 列出 agentic RL 与单轮 RLVR 的关键差异，并解释每个差异对训练系统提出的新要求。
5. 动手：把 Agent 课 Day 7 的一条轨迹转成 verl 多轮样本格式并验证 mask 正确。

## 工业现状

2025 年发布的 agent 模型报告里，RL 阶段的共同做法是把多轮交互作为一个 episode，只对模型生成的 token 算策略损失，工具返回作为 observation 拼进上下文：Kimi K2（T19）在大规模合成 agent 任务上做 RL；GLM-4.5（T20）把 agentic 任务与推理任务混合 RL；SWE-RL（T45）用软件演化数据做 RL 并以规则奖励衡量 patch 相似度；Qwen3 系列（T15）的 agent 能力同样来自后训练阶段的多轮 RL。框架侧 verl 的多轮 / agent loop、SkyRL（B12）、ROLL（B13）都提供了"一条轨迹多次生成、observation mask"的样本表示。


公开系统在表示层的选择（按报告与代码库整理，细节以原文为准）：

| 系统 | 轨迹表示 | 动作粒度 | reward 粒度 | 备注 |
|---|---|---|---|---|
| SWE-RL（T45） | 单序列，patch 生成为主 | token | 终局（patch 相似度） | 轮数少，observation 为仓库上下文 |
| Kimi K2（T19） | 多轮工具轨迹 | token，trajectory advantage | 终局 + 部分可验证 | 大规模合成环境 |
| GLM-4.5（T20） | 多轮，含 expert iteration | token | 终局 | agentic 与推理任务混训 |
| verl agent loop（B06） | 单序列 + response_mask | token | 终局，可扩展 | 默认 GRPO |
| SkyRL（B12） | gym 风格 step 接口 | token | 终局 / step | 环境抽象更完整 |

### 常见误区

- "把工具返回也训一下模型就会更懂工具"：那是 SFT 的做法，RL 里 observation 进 loss 会让策略学会预测环境而不是行动，且破坏 on-policy 假设。
- "多轮就必须 turn 级算法"：trajectory-level GRPO 在多数公开系统里就是主力，先把它跑对。
- "轨迹长就切成多段独立训练"：切断后每段的 reward 归属不明，除非有 step reward。

工业界还没有统一的 credit assignment 方案：多数系统用 trajectory-level reward 均分到该轨迹所有生成 token（GRPO 的自然扩展），turn-level advantage 与 step reward 仍是研究热点。这一天先把表示做对，Day 25 再讨论分配。

## 核心原理

### 形式化

```
环境隐藏状态  s_t        （仓库文件、数据库、网页 DOM）
观测         o_t        （工具返回文本、错误信息、用户消息）
历史         h_t = (o_0, a_0, o_1, a_1, ..., o_t)   模型看到的是 h_t 的文本化
动作         a_t        （一次 assistant 生成：可能含思考、工具调用 JSON、最终回答）
奖励         r_t        （多数任务只有终局 r_T；可加 step reward）
episode 边界 ：任务完成 / 达到最大轮数 / 工具致命错误 / 模型输出终止标记
```

策略 `π_θ(a_t | h_t)` 是对整段生成的 token 级自回归分布。目标仍是 `max E[Σ r_t] − β KL`，但期望是对整条轨迹的。

### 动作粒度

| 粒度 | 定义 | 优点 | 代价 |
|---|---|---|---|
| token 级 | 每个生成 token 一个动作 | 与单轮 RLVR 的实现完全一致，PPO/GRPO 直接用 | 一次工具调用几十 token 全部拿同一 advantage，决策点不突出 |
| turn 级 | 一次 assistant 生成为一个动作 | advantage 可按轮分配；与 POMDP 定义对齐 | 需要 turn 级 value 或 reward，框架支持有限 |
| 混合 | token 级 loss + turn 级 advantage | 当前多数实现的实际形态 | advantage 在 turn 内广播 |

必须理解：不管哪种粒度，**observation token 不是模型的动作，不进策略损失**，但它们必须在上下文里供后续 token 的 attention 使用。

### 从轨迹到训练样本

一条 3 轮轨迹：

```
[system][user: 任务]                       ← prompt，mask=0
[assistant: 思考 + tool_call(search, q)]   ← 生成，mask=1，属于 turn 1
[tool: 搜索结果 ...]                        ← observation，mask=0
[assistant: 思考 + tool_call(read, url)]   ← 生成，mask=1，turn 2
[tool: 页面内容 ...]                        ← observation，mask=0
[assistant: 最终答案]                       ← 生成，mask=1，turn 3
```

两种表示：

**单序列表示**：把整条轨迹拼成一个 token 序列，`loss_mask` 标出生成段，`position_ids` 连续。训练侧一次 forward 算出所有生成 token 的 logprob。前提：轨迹总长 ≤ 训练序列上限；工具返回没有被截断或改写。

**多样本表示**：每一轮生成一个样本，prompt 是当时的 `h_t`。样本数 = 轮数，前缀重复计算，但每个样本长度短，且能处理"中途上下文被压缩"的情况。

工业实现多用单序列表示，因为 prefix 只算一次；当 agent 有上下文管理（截断、摘要）时，每轮看到的 `h_t` 不是简单前缀，必须用多样本表示，否则 logprob 与 rollout 时不一致。

### 与单轮 RLVR 的差异

| 维度 | 单轮 RLVR | Agentic RL | 对系统的要求 |
|---|---|---|---|
| rollout | 引擎一次 generate | 引擎 generate ↔ 环境 step 交替多次 | rollout 循环在引擎外，需要异步与并发（Day 24） |
| 长度 | 一段 response | 多段生成 + 多段 observation，总长可达几十 k | 长序列训练、CP、样本切分 |
| loss mask | prompt 0 / response 1 | 交替的 0/1 段 | 数据格式与 collator 支持 |
| reward | 终局 verifier | 终局 + 可能的 step reward，环境状态相关 | reward 在环境侧计算（Day 25） |
| 非确定性 | 采样温度 | 采样 + 环境（网络、时间、外部服务） | 环境版本化与回放（Day 23） |
| 失败模式 | 答错 | 工具错误、超时、死循环、越权 | episode 终止条件、安全边界 |
| KL / ref | 对 response | 对生成段；observation 段不算 | 同 mask |
| 成本 | 模型 token | 模型 token + 工具/沙箱时间 | rollout 成本核算 |

### advantage 的初级方案

GRPO 的自然扩展：同一任务采 n 条轨迹，终局 reward 做 group 归一化，advantage 广播到该轨迹所有生成 token。这就是 Day 21 的代码不改算法只改数据格式就能跑 agent 任务的原因。它的问题（长轨迹中早期正确决策与后期错误同罚）留到 Day 25。


### 一条轨迹的数字示例

SWE 类任务里一条典型轨迹（示例数字）：

| 段 | 类型 | token 数 | mask |
|---|---|---:|---|
| system + 任务描述 | prompt | 1,200 | 0 |
| turn 1：思考 + `read_file` 调用 | 生成 | 180 | 1 |
| 文件内容 | observation | 3,500 | 0 |
| turn 2：思考 + `edit_file` 调用（含 diff） | 生成 | 420 | 1 |
| 编辑确认 | observation | 40 | 0 |
| turn 3：`run_tests` 调用 | 生成 | 60 | 1 |
| 测试输出 | observation | 2,800 | 0 |
| turn 4：最终说明 | 生成 | 150 | 1 |
| 合计 | | 8,350 | 生成 810（9.7%） |

含义：训练序列上限 8k 会截掉这条轨迹；一次 forward 的算力花在 8,350 个 token 上，梯度只来自 810 个；observation 截断策略对显存与吞吐的影响比任何算法选择都大。

### turn 级 value 与分层方案（知道即可）

若要 turn 级 advantage，需要 `V(h_t)` 的估计：可训练一个 turn 级 value head（只在每轮末尾 token 处取值），或用蒙特卡洛：同一 `h_t` 下多次续采到终局取平均 reward（成本高）。分层方案把高层"选工具"与低层"写参数"分开建模，目前无主流框架支持，工业界仍以 trajectory-level 为主。

### reference policy 与 KL 在多轮里的位置

ref 只在生成段算；每轮的 ref logprob 条件于该轮的 `h_t`。若 SFT 冷启动模型与 RL 起点相同，ref = SFT 模型；若中途换了 ref（如迭代 RL），要保证 rollout 时的 `h_t` 构造与 ref forward 一致，否则 KL 会因模板差异而虚高。

### 多模态 observation

网页截图、图表等图像 observation 通过视觉编码器进入上下文，token 数由分辨率决定，同样 mask=0；训练序列上限要按视觉 token 计算。verl 的多模态多轮支持以当前版本为准。

## 实现步骤

### 1. verl 多轮样本格式（字段名以 B06 当前版本为准）

verl 的 multi-turn / agent loop 把一条轨迹表示为：

```python
{
  "prompt_ids":    [...],                 # 初始 prompt
  "response_ids":  [...],                 # 之后所有 token（生成 + observation 交替）
  "response_mask": [1,1,1,0,0,0,1,1,0,0,1,1],   # 1 = 模型生成，0 = 工具返回
  "reward":        1.0,                   # 终局
  "num_turns":     3,
}
```

训练时 `loss = Σ_t response_mask[t] · L_t / Σ_t response_mask[t]`（token-mean）。ref 与 old logprob 同样只在 mask=1 处使用。

框架间数据格式对照（字段名以各自当前版本为准）：

| 框架 | 轨迹字段 | mask 字段 | reward 字段 | 多轮支持方式 |
|---|---|---|---|---|
| verl | `prompt_ids` / `response_ids` | `response_mask` | `reward`（或 token-level `rm_scores`） | agent loop / multi-turn SFT & RL |
| TRL GRPOTrainer | `prompt` + 生成的 `completion` | 内部按 completion 计算；多轮需自定义 reward func 与 environment 接口 | reward func 返回 | 较新版本有 environment 钩子 |
| SkyRL | `prompt` / `response` 段列表 | `loss_mask` | `reward`（step 或终局） | gym env |
| OpenRLHF | `input_ids` + `action_mask` | `action_mask` | `reward` | 自定义 agent 函数 |

### 2. 用 chat template 生成 mask

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def build_sample(messages):
    """messages: list of {role, content, tool_calls?}，assistant 段为模型生成"""
    ids, mask = [], []
    prev_len = 0
    for i in range(1, len(messages) + 1):
        full = tok.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=False)
        seg = full[prev_len:]
        is_gen = messages[i-1]["role"] == "assistant"
        ids += seg; mask += [1 if is_gen else 0] * len(seg)
        prev_len = len(full)
    return ids, mask
```

验证：逐段 decode，确认 mask=1 的段恰好是 assistant 内容（含 tool_call JSON），不含模板里的角色标记。不同模型的 chat template 在 tool 消息上的格式差异很大，每换模型都要重新验证。

### 2b. episode 终止条件的实现

```python
def is_done(state):
    if state.final_answer is not None: return True, "final"
    if state.turn >= MAX_TURNS:        return True, "max_turns"
    if state.consecutive_errors >= 3:  return True, "error_loop"
    if state.env_fatal:                return True, "env_error"      # 不进训练
    return False, None
```

终止原因进 `info`，训练侧按原因决定 reward 处理：`max_turns` 与 `error_loop` 按未完成给 reward，`env_error` 丢弃。

### 3. 从 Agent 课的 trace 转换

Agent 课 Day 4 的 trace schema 记录了每次模型调用的输入输出与工具结果。转换脚本：按时间顺序把 `model_call.output` 变成 assistant 消息，`tool_result` 变成 tool 消息，最终 verifier 结果变成 reward。若 trace 里存在上下文压缩事件（某次调用的输入不是前缀），切换到多样本表示。

### 4. 单序列 vs 多样本的选择检查

```python
def is_prefix_consistent(trace):
    for k in range(1, len(trace.calls)):
        if trace.calls[k].input_ids[:len(trace.calls[k-1].input_ids)] != trace.calls[k-1].input_ids:
            return False
    return True
```

不一致时用多样本表示，或在 rollout 时禁用上下文压缩（代价是长度上限）。

### 5. 训练侧的序列长度

单序列表示的轨迹长度 = prompt + Σ 生成 + Σ observation。SWE 类任务里 observation（文件内容、测试输出）常占 70% 以上。训练序列上限要按 P99 轨迹长度设，超长轨迹：截断 observation（rollout 时就截，保证一致）、CP（Day 5）、或丢弃并记录比例。

## 实验

资源分层：全部实验单卡可做（表示层实验不需要训练）。

### 实验 1：mask 正确性

对 3 个不同模型（Qwen、Llama、GLM 各一）的 chat template，用第 2 步的函数构造 5 条含 tool 消息的样本，人工核对 mask。记录每个模板里 tool 消息的包裹标记是否被误标为生成。

### 实验 2：单序列与多样本的 logprob 一致性

取一条 5 轮轨迹，分别用单序列表示一次 forward，和多样本表示 5 次 forward，比较每个生成 token 的 logprob。预期：无上下文压缩时逐 token 一致（误差在 BF16 噪声内）；人为在第 3 轮插入一次压缩后，单序列表示的第 4、5 轮 logprob 偏离。

### 实验 3：observation 占比

统计 Agent 课 Day 7 项目的 100 条轨迹：总长、生成 token 占比、轮数分布、observation 最长段。据此定训练序列上限与截断策略。

### 实验 4：advantage 广播的效果（可选，需训练）

在 Day 21 的 pipeline 上跑一个两轮工具任务（先查后答），用 trajectory-level GRPO，观察第一轮（查询）与第二轮（回答）生成 token 的 logprob 变化是否同向。这是 Day 25 的引子。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 训练后模型开始"复述"工具返回 | observation 段被标为生成，进了 loss | 抽样看 mask | 修 mask 构造；换模板验证 |
| logprob 重算与 rollout 相差大 | 上下文压缩导致 prefix 不一致 | `is_prefix_consistent` | 多样本表示或关闭压缩 |
| 序列超长被丢弃比例高 | observation 未截断 | 实验 3 统计 | rollout 时截断 observation 并保持训练一致 |
| 多轮任务 reward 全 0 | episode 终止条件错，未到最终回答就结束 | 看轨迹末尾 | 修终止逻辑；最大轮数放宽 |
| 同一任务 n 条轨迹 reward 全同 | 环境确定且模型温度低 | group reward std | 提高温度、任务分层 |
| KL 异常高 | ref 在 observation 段也算了 | 检查 KL 计算的 mask | 与 loss 用同一 mask |
| tool_call JSON 格式在训练后变坏 | 模板中 tool_call 段的特殊 token 被 mask 掉或 loss 权重不对 | 看生成样本 | 保证 tool_call 全段 mask=1 |

## 验收标准

- 能写出你任务的 POMDP 五元组与 episode 终止条件。
- 三个模型模板的 mask 验证表；单序列/多样本一致性实验数据。
- 一份"何时必须用多样本表示"的判定规则。
- 一张 agentic RL 与单轮 RLVR 的差异表，每行写明对 Day 23 到 25 系统设计的含义。

## 自测

1. 一次 `tool_call` 的 JSON 参数 token 是动作吗？工具返回的 JSON 是吗？
2. 同一条轨迹用单序列表示与多样本表示，训练算力各是多少 token 的 forward？哪种更省？前提是什么？
3. 如果 rollout 时 agent 框架对超长文件做了"只显示前 200 行"的截断，训练侧要怎么保证一致？
4. trajectory-level advantage 广播会把哪类错误信号给到哪类正确决策？

## 交付物

| 文件 | 内容 |
|---|---|
| `agentic-rl-formalization.md` | POMDP 定义、动作粒度选择、差异表 |
| `build_sample.py` + `mask-validation.md` | 样本构造脚本与验证结果 |
| `trajectory-stats.md` | 实验 3 的长度统计与截断策略 |

## 参考

- T47 ReAct：思考 + 行动交替的轨迹形态，是当前 agent 样本格式的来源。
- T19 Kimi K2、T20 GLM-4.5、T45 SWE-RL：公开报告中 agentic RL 的样本与 reward 形态。
- T36 DeepSeekMath：GRPO 的 group 归一化，agent 场景直接扩展。
- B06、B12、B13：verl / SkyRL / ROLL 的多轮样本表示。
- T46 SWE-bench：软件工程 agent 任务的 episode 定义。
