---
title: "Day 26：案例研究：SWE Agent RL 与 Tool-use RL 的公开配方"
type: concept
tags: [llm-training, agentic-rl, case-study, swe-rl, tool-use]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 26：案例研究：SWE Agent RL 与 Tool-use RL 的公开配方

> 上一课 [[llm-training-30d/week4/day25-agent-reward-credit-assignment]] · 下一课 [[llm-training-30d/week4/day27-reasoning-models-long-cot]]。相关：推理课的 MoE 部署 [[ai-infra-30d/week3/day19-moe-inference]]（这些模型大多是 MoE，rollout 引擎就是推理课讲的那套系统）。

## 学习目标

1. 能按"环境规模、数据来源、算法、rollout 基础设施、评测"五个维度复述 SWE-RL、Kimi K2、GLM-4.5、MiniMax-M1、DeepSeek-R1 五份公开报告里的 agentic / 工具 RL 做法，并区分"报告写了"与"外界推测"。
2. 能从五个案例里抽出共性配方，写成一页可执行的 checklist。
3. 能指出每个案例在 reward、环境或算法上的一个关键取舍，以及它为什么在该团队的约束下成立。
4. 能设计一个单机或 8 卡可跑的缩小版复现，并说明它与原报告的差距。

## 工业现状

到 2026 年，agentic RL 已经从"能不能训"变成"环境规模和 rollout 基础设施谁做得大"。公开报告展示的规模量级：

| 报告 | 模型 | agentic RL 的任务域 | 环境/数据规模（报告口径） | 出处 |
|---|---|---|---|---|
| SWE-RL（Meta，2025-02） | Llama 3 70B 基座续训 | 单轮 issue → patch（Agentless 风格，非多轮 agent） | 从 GitHub PR 构造的大规模 seed 数据，具体条数以报告为准 | T45 |
| Kimi K2（Moonshot，2025-07） | 1T 总参 / 32B 激活 MoE | 多轮工具调用、代码、数学、STEM、指令遵循 | 数千真实 MCP 工具加两万量级合成工具，合成 agent 与任务，轨迹经 judge 过滤 | T19 |
| GLM-4.5（智谱，2025-08） | 355B / 32B 激活 MoE | 网页搜索 agent、agentic coding（SWE 类）、推理 | 可执行环境，测试驱动 reward；具体环境数报告未逐一公开 | T20 |
| MiniMax-M1（2025-06） | 456B / 46B 激活，混合 lightning attention | 数学、逻辑、竞赛编程、软件工程（沙箱执行测试） | 软件工程环境基于 SWE 类沙箱；条数以报告为准 | T21 |
| DeepSeek-R1（2025-01） | 671B / 37B 激活 MoE | 非 agentic：推理 RL，但其 reward 与训练流程是后来 agent RL 的模板 | 规则可验证题库 | T17 |
| Qwen3-Coder（阿里，2025-07，博客） | 480B / 35B 激活 MoE | 长程 agent RL（多轮 SWE 类任务） | 博客称使用两万量级并行环境；无技术报告编号，本课只引用博客明确写的内容 | 无 T 编号 |

注意区分：SWE-RL 是**单轮**生成补丁，Kimi K2、GLM-4.5、Qwen3-Coder 是**多轮**工具调用。两者的 rollout 形态、reward 和基础设施差异很大，面试常考。

## 核心原理

### 案例 1：SWE-RL（T45）

- **任务形态**：输入 issue 描述 + 相关代码片段，输出 search/replace 格式的补丁。一次模型调用，没有工具循环。
- **reward**：补丁与真实 PR 补丁的文本相似度（序列匹配比率，连续值 0 到 1）；格式非法记 −1。不跑测试。这是它最大的取舍：用"像不像人类的修法"代替"能不能跑通"，换来无需沙箱、可在海量 PR 上训练。
- **算法**：GRPO。
- **数据**：从 GitHub 公开 PR 抽取 issue、相关文件、oracle patch 三元组，经过过滤。
- **结果**：报告的 Llama3-SWE-RL-70B 在 SWE-bench Verified 上取得约四成的解决率（具体数字以报告为准），且在推理类通用评测上有迁移收益。
- **局限（报告自述或明显）**：相似度 reward 会奖励"形式相近但逻辑错误"的补丁；无法学到多轮探索仓库的行为。
- **抽取的教训**：reward 可以不是"正确性"，只要它便宜、稠密、与最终目标相关；但这种代理 reward 的上限由它与真目标的相关性决定。

### 案例 2：Kimi K2（T19）

- **数据合成流水线**是报告的核心贡献之一：（1）工具规范生成：真实 MCP 工具加合成工具；（2）为工具组合生成 agent 角色与任务，任务带 rubric；（3）多 agent 模拟生成多轮轨迹：用户模拟器、工具模拟器，编码类任务接真实沙箱执行；（4）LLM judge 按 rubric 过滤轨迹。产物既用于 SFT，也用于 RL。
- **reward**：可验证任务（数学、代码、STEM、指令遵循）用 verifier；不可验证任务用**自评 rubric reward**，策略模型自己作为 critic 按 rubric 打分，critic 用可验证任务的信号持续校准。
- **算法**：K1.5 系列的策略优化（在线策略镜像下降类目标），带长度预算控制、辅助 PTX 损失、温度衰减等稳定手段（细节以报告为准）。
- **基础设施**：训练与推理引擎共置，报告描述了参数广播机制，把万亿参数从训练侧同步到推理侧控制在几十秒量级；长运行的 agent 环境与 rollout 解耦；报告提到 partial rollout 类机制处理长轨迹。
- **教训**：agentic 能力的瓶颈在"有多少可执行、可评分的任务"，合成流水线的规模和 judge 过滤质量决定上限。

### 案例 3：GLM-4.5（T20）

- **训练流程**：预训练 → 中期训练（仓库级代码、合成推理、长上下文、agent 轨迹）→ 后训练：先分别训练推理专家、agent 专家、通用对话专家（各自 SFT + RL），再用**自蒸馏**把专家能力合并进一个模型。
- **agentic RL**：网页搜索 agent 用合成的多跳问题训练；agentic coding 在可执行环境里用测试结果作 reward；报告提到迭代式的自蒸馏与 expert iteration（用当前策略采样、筛选、再 SFT）。
- **推理 RL**：单阶段直接在 64K 上下文上 RL，难度课程、动态采样温度、自适应 clip（细节以报告为准）。
- **基础设施**：报告描述了其 RL 框架 slime（B07）：训练用 Megatron，rollout 用 SGLang，支持同步与异步两种模式，agent 任务走异步。
- **教训**：多专家分训再合并，避免不同任务的 RL 互相干扰；异步 rollout 对长 agent 任务几乎是必需。

### 案例 4：MiniMax-M1（T21）

- **算法贡献 CISPO**：不裁剪 token 的更新，而是裁剪重要性采样权重，保留低概率但关键的"反思"token 的梯度。报告称在同等步数下收敛更快。
- **训练与推理不一致**：报告记录了训练侧与推理侧 logprob 不一致导致的 reward 不增长问题，定位到输出头精度，改为高精度计算后解决。这是 Day 19 讲的 train/infer mismatch 的公开实例。
- **软件工程环境**：基于沙箱执行测试的 SWE 类任务，reward 来自测试通过。
- **长度**：输出长度预算逐步放大到数万 token，用长度相关的 shaping 与截断处理。
- **教训**：算法层面的小改动（clip 什么）和系统层面的精度问题，任何一个都能让 RL 完全不工作。

### 案例 5：DeepSeek-R1（T17）作为模板

虽然不是 agent RL，它奠定了后面所有配方的三个共识：规则 reward 优于 RM 与 PRM；GRPO 去掉 value model 使大模型 RL 可负担；多阶段（冷启动 SFT → RL → 拒绝采样 SFT → 全场景 RL）比单阶段稳。Day 27 展开。

### 共性配方

```text
1. 任务必须可执行、可验证；不可验证的用 rubric + judge 并限制权重
2. 数据 = 环境：合成工具/任务/轨迹的流水线规模决定上限
3. 算法：GRPO 家族 + 去 value model + 对 clip/长度/采样的稳定性修补
4. rollout：推理引擎在训练循环内；长任务异步；权重同步是工程重点
5. 多阶段：SFT 冷启动 → RL → 用 RL 策略采样筛选再 SFT（expert iteration / 自蒸馏）
6. 评测：SWE-bench Verified、tau-bench、BFCL 类，报告 avg@n 而非单次
```

### 差异点

| 维度 | SWE-RL | Kimi K2 | GLM-4.5 | MiniMax-M1 |
|---|---|---|---|---|
| 轮数 | 单轮 | 多轮 | 多轮 | 多轮（SWE）+ 单轮（数学） |
| reward | 补丁相似度 | verifier + 自评 rubric | 测试通过 + 搜索答案匹配 | 测试通过 + RM |
| 算法 | GRPO | 在线镜像下降类 | GRPO 变体 + 课程 | CISPO |
| 引擎 | 未强调 | 共置 + 快速广播 | slime：Megatron + SGLang，异步 | 报告未详述框架 |
| 关键取舍 | 不跑测试换规模 | 合成环境换覆盖 | 分专家再合并 | 裁剪 IS 权重保留反思 token |

## 实现步骤：缩小版复现设计

目标：在 8 卡（或单卡 + LoRA）上复现"多轮 SWE 类 agent RL"的最小闭环，与 GLM-4.5 / Qwen3-Coder 的形态对齐，而不是复现规模。

1. **环境**：选公开的可执行 SWE 类任务集（如 SWE-bench Verified 的子集，或 R2E-Gym 类合成任务，以当前可获得的为准），每个任务一个容器镜像，预热池 16 到 32 个。
2. **agent 接口**：Day 23 的 gym 接口；工具限定为 `read_file`、`search`、`edit`、`run_tests`，最多 20 turn。
3. **模型**：7B 到 32B 开源代码模型；LoRA 或全参按资源。
4. **reward**：hidden tests 通过率为 outcome（二值或分级），step 罚分与格式罚分按 Day 25。
5. **算法**：GRPO + DAPO 的 dynamic sampling 与 clip-higher，无 KL 或极小 KL（多份公开复现的选择，需自己消融）。
6. **rollout**：verl 或 slime 的异步 agent loop；每个 prompt 8 到 16 条 rollout；记录 rollout 与 train 的时间占比。
7. **评测**：hidden 子集 avg@4，同时看步数与工具错误率。
8. **与原报告的差距声明**：环境数、模型规模、训练步数、是否有中期训练的 agent 轨迹数据。写清哪些结论不能外推。

### 缩小版复现的配置骨架（verl 风格，字段名以当前版本 B06 为准）

```yaml
data:
  train_files: data/swe_mini/train.parquet     # 每行：task_id, image, issue, hidden_tests_ref
  val_files:   data/swe_mini/val.parquet
  max_prompt_length: 8192
  max_response_length: 16384                   # 多轮累计
actor_rollout_ref:
  model.path: Qwen/Qwen2.5-Coder-7B-Instruct   # 示例
  actor:
    optim.lr: 1e-6
    ppo_mini_batch_size: 64
    use_kl_loss: false                         # 多份公开复现的选择，需自行消融
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28                      # DAPO clip-higher
  rollout:
    name: vllm
    n: 8
    temperature: 1.0
    multi_turn:                                # agent loop 相关字段随版本变化
      enable: true
      max_turns: 20
      tool_config_path: tools/swe_tools.yaml
algorithm:
  adv_estimator: grpo
  filter_groups.enable: true                   # dynamic sampling：过滤全同 reward 的 group
reward_model:
  reward_manager: custom
  custom_reward_function.path: reward/swe_reward.py
trainer:
  n_gpus_per_node: 8
  total_training_steps: 300
  val_before_train: true
```

工具服务侧（Day 23 的接口）要提供 `run_tests` 的 hidden 与 public 两套入口，训练时只把 public 暴露给策略，reward 函数调用 hidden。

### 评测脚本骨架

```python
# eval_avg_at_n.py：hidden 子集 avg@n，同时记录步数与工具错误率
import json, statistics as st
def evaluate(policy, tasks, n=4):
    rows = []
    for task in tasks:
        outcomes = []
        for _ in range(n):
            traj = run_agent(policy, task, max_turns=20)      # Day 24 的 rollout 接口
            outcomes.append(dict(
                passed=hidden_tests_pass(task, traj.final_patch),
                turns=traj.num_turns,
                tool_err=traj.tool_error_count / max(1, traj.tool_calls)))
        rows.append(dict(task=task.id,
                         avg_pass=st.mean(o["passed"] for o in outcomes),
                         avg_turns=st.mean(o["turns"] for o in outcomes),
                         avg_tool_err=st.mean(o["tool_err"] for o in outcomes)))
    print(json.dumps(dict(
        avg_at_n=st.mean(r["avg_pass"] for r in rows),
        turns=st.mean(r["avg_turns"] for r in rows),
        tool_err=st.mean(r["avg_tool_err"] for r in rows)), indent=2))
    return rows
```

报告里必须同时给出 avg@n 的 n、任务数、上下文上限、是否用 test-time scaling，与上面"评测口径"表对齐。

## 实验

### 实验 1：单轮 vs 多轮的 reward 结构（离线分析）

取 200 个 SWE 任务，分别用 SWE-RL 风格的补丁相似度和测试通过率给同一批 rollout 打分，算两者的 Spearman 相关。预期相关性中等偏低，说明相似度 reward 的天花板。

### 实验 2：expert iteration 一轮

用当前策略在训练集采样 8 条 / 任务，保留通过 hidden tests 的轨迹做一轮 SFT，再评测。对比直接 RL 相同算力下的收益。这是 GLM-4.5 与 R1 都用的步骤，成本低于 RL。

### 实验 3：CISPO vs GRPO 的 clip（可选，需 RL 环境）

在 Day 21 的数学任务上把 token clip 换成 IS 权重 clip，比较 200 步内的 reward 曲线与 entropy。预期 CISPO 的 entropy 下降更慢。

## 深入：两种 rollout 基础设施形态

案例里反复出现两种形态，Day 24 讲过原理，这里用案例对照：

```text
共置（Kimi K2 报告描述的形态）
┌───────────── 同一批 GPU ─────────────┐
│ 阶段 A：推理引擎持有权重 → 采样 rollout │
│ 阶段 B：引擎 sleep 释放显存 → 训练更新   │
│ 阶段 C：参数广播到引擎（万亿参数几十秒） │
└──────────────────────────────────────┘
优点：GPU 不闲置分区；缺点：A/B 串行，长尾 rollout 拖住整批

解耦 + 异步（GLM-4.5 的 slime 异步模式、AReaL）
┌── 训练 GPU ──┐        ┌── rollout GPU ──┐
│ 持续消费轨迹  │ ◄──── │ 持续产生轨迹      │
│ 周期性推权重  │ ────► │ 按版本加载        │
└──────────────┘        └──────────────────┘
优点：长 agent 轨迹不阻塞训练；缺点：off-policy 程度需控制（staleness、IS 修正）
```

判断题：一个任务平均 3 turn、P99 8 turn，另一个平均 15 turn、P99 60 turn 且含真实沙箱执行，各该选哪种？答案与理由写进 `case-studies.md`。

## 深入：评测口径

案例报告的数字不可直接横比，因为口径不同。读报告时先找这些字段：

| 评测 | 测什么 | 常见口径差异 |
|---|---|---|
| SWE-bench Verified（T46 的人工核验子集） | 真实仓库 issue 修复，跑 hidden tests | scaffold 不同（Agentless vs 多轮 agent）、是否 test-time scaling（多次采样选优）、上下文长度上限 |
| tau-bench / tau2-bench | 多轮工具调用 + 用户模拟器，零售/航空场景 | pass^k（k 次全过）比 pass@1 严得多 |
| BFCL | 函数调用格式与语义正确性 | 单轮 / 多轮 / 并行调用分项 |
| Terminal-Bench 类 | 终端任务 | 环境镜像版本 |
| AIME / LiveCodeBench | 推理与代码生成 | avg@n 的 n、题目时间窗（防污染） |

报告里 "pass@1" 若是 avg@64 的平均值，与单次采样不是一回事；带 test-time scaling 的数字要单独列。自己的复现报告必须写清同样的字段。

## 深入：从报告里读不到的东西

以下是五份报告都没有完整公开、但复现时必须自己决定的：

- 每个任务采多少条 rollout、group size 随难度是否变化。
- reward 各项系数的具体值与调参过程。
- 环境的失败率、超时率以及如何处理这些样本（丢弃、记 0、重采）。
- 训练与推理引擎之间的 logprob 差异有多大、是否做 IS 修正（MiniMax-M1 是少数写了的）。
- 中期训练里 agent 轨迹数据的确切构成与比例。
- 多阶段之间的模型选择标准（用哪一步的 checkpoint 进入下一阶段）。

写复现设计时把这些列为"假设"，逐条标注你的选择和理由。这比堆参数更接近工业实践。

## 深入：案例阅读法

读技术报告的顺序建议：先看评测表找口径，再看后训练章节找 reward 与算法，再找基础设施章节（通常最短），最后回头看数据章节。每篇用固定模板记录：

```text
报告：
任务形态（单轮/多轮、工具集）：
reward（来源、连续/二值、是否 hidden）：
算法（基线 + 修改点）：
数据（来源、合成流水线、过滤）：
基础设施（共置/解耦、同步/异步、引擎）：
评测口径：
报告明确的失败与教训：
未公开项：
```

五个案例各填一份，是本课交付物的一部分。

## 面试题（5 题，附要点）

1. **SWE-RL 为什么不跑测试就能训出提升？** 相似度 reward 稠密且便宜，可在海量 PR 上训练；上限受代理 reward 与真实正确性的相关性限制。
2. **Kimi K2 的自评 rubric reward 怎么避免自己给自己打高分？** critic 与策略同源但用可验证任务的信号持续校准；rubric 对策略部分隐藏；报告未公开全部细节。
3. **GLM-4.5 为什么分专家训练再合并？** 不同任务的 RL 目标互相干扰；合并用自蒸馏而非简单平均。
4. **CISPO 与 PPO clip 的区别？** PPO 裁剪并截断被裁 token 的梯度；CISPO 裁剪 IS 权重但保留所有 token 的梯度，低概率关键 token 仍能学。
5. **多轮 agent RL 为什么几乎必须异步？** 轨迹长度长尾（工具执行、沙箱），同步模式下整批等最慢的一条；异步换来吞吐但引入 off-policy，需要 staleness 控制。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 复现结果远低于报告 | 缺中期训练的 agent 轨迹数据、环境太少 | 看 SFT 冷启动后的 pass@8 是否接近零 | 先做 expert iteration 或蒸馏数据，再 RL |
| 训练几十步后全部轨迹超时 | 策略学会拖延或工具死循环 | 步数分布与 timeout 率 | max_turns 硬截断记 0；重复调用罚分 |
| 沙箱成为瓶颈 | 每步冷启动容器 | reward 服务 P99 | 容器池、快照、批量执行 |
| 多任务 RL 后单项能力下降 | 任务间干扰 | 分任务评测曲线 | 分专家训练再合并（GLM-4.5 做法） |
| reward 曲线平坦 | train/infer logprob 不一致 | 对比两侧 logprob 差 | 输出头高精度、开启 IS 修正（Day 19） |

## 验收标准

- 能不看笔记复述五个案例的 reward、算法、基础设施要点，并标明哪些是"报告未公开"。
- 缩小版复现设计文档完成，含与原报告的差距声明。
- 实验 1 与 2 完成并给出数据。

## 交付物

| 文件 | 内容 |
|---|---|
| `case-studies.md` | 五案例五维度对比表、共性配方 checklist、差异点 |
| `mini-swe-rl-plan.md` | 缩小版复现设计与差距声明 |
| `expert-iteration-report.md` | 实验 1、2 的数据与结论 |

## 参考

- T45 SWE-RL：单轮代理 reward 的完整案例。
- T19 Kimi K2：agentic 数据合成流水线与共置 RL 基础设施。
- T20 GLM-4.5：多专家训练与自蒸馏、slime 框架。
- T21 MiniMax-M1：CISPO 与 train/infer 精度问题的实录。
- T17 DeepSeek-R1：多阶段配方模板。
- T46 SWE-bench：评测集定义与 Verified 子集的意义。
- B07 slime、B06 verl：复现用的框架文档。
