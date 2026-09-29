---
title: "Agent 自迭代、Agent RL、RSI 与 ASI：30 天高强度学习路径"
type: synthesis
tags: [agent, self-evolution, agentic-rl, auto-research, rsi, asi, evaluation, safety]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Agent 自迭代、Agent RL、RSI 与 ASI：30 天高强度学习路径

课程入口：[[agent-rsi-30d/index]]；每日教材：[[agent-rsi-30d/reading-index]]。

## 课程目标

30 天约 240–300 小时。完成后，你应能从零实现一个可观测、可评测的工具 Agent，让它通过记忆、prompt/workflow 搜索和 Agent RL 改善行为，再将其扩展成带实验验证的 Auto Research 系统，最后在严格沙箱和隐藏评测下完成受控 RSI 实验。

本课程不把反思提示词称为 RSI，也不声称已经存在 ASI：

```text
单次自修正
  → 跨任务记忆与技能
  → prompt / workflow / code 自迭代
  → 模型权重通过 Agent RL 更新
  → 自动提出和验证研究假设
  → 新版本改进下一代改进器（受控 RSI）
  → ASI：仍需独立证据的长期假设
```

## 30 天安排

| 阶段 | 天数 | 内容 | 阶段项目 |
|---|---:|---|---|
| Week 1 | 1–7 | Agent runtime、工具、沙箱、轨迹、评测与 baseline | P0：可回放 Agent 基线 |
| Week 2 | 8–14 | 反思、记忆、技能、自动课程、prompt/workflow/多 Agent 优化 | P1：非参数自迭代 Agent |
| Week 3 | 15–21 | RL 数学、reward、credit assignment、SFT、PPO/GRPO、harnessed RL | P2：端到端 Agent RL |
| Week 4+ | 22–30 | Auto Research、研究诚信、自修改、DGM、受控 RSI、ASI 与安全 | P3：受控 RSI Auto Researcher |

## 每日固定节奏

1. 2 小时：阅读当天教程和必读论文的指定部分。
2. 2 小时：复现最小例子，先确认定义和数据流。
3. 4–5 小时：完成核心实验并保存全部轨迹和 artifact。
4. 1 小时：运行冻结评测，比较当天版本与 baseline。
5. 1 小时：写实验日志，包括失败、费用、随机种子和下一步假设。

## 四个项目

### P0：可回放 Agent 基线（Day 1–7）

实现一个至少有两个工具的 Agent。所有模型调用、工具调用、环境状态、费用和错误必须可回放。建立 train/dev/hidden task split，并给出单次模型调用、固定 ReAct 和 best-of-N 三个 baseline。

### P1：非参数自迭代 Agent（Day 8–14）

在不更新模型权重的情况下，让 Agent 通过抽象经验、技能库、自动课程以及 prompt/workflow 搜索改善。至少运行五代；最终结论只能来自未参与选择的任务，并报告负迁移。

### P2：Agent RL（Day 15–21）

在 text-to-SQL、证据检索或小型代码修复中选择一个可验证任务。完成 reward spec、轨迹采集、SFT baseline、策略优化和 hidden evaluation。必须检测 reward hacking，而不是只报告 reward。

### P3：受控 RSI Auto Researcher（Day 22–30）

构建文献—假设—实验—验证—报告闭环。Agent 只能修改 planner、memory、tool router、context manager 和候选生成策略；grader、hidden tests、权限策略和审计日志不可修改。运行至少五代，并用全新 transfer tasks 判断是否真的累积改进。

## 统一实验协议

- 开始前冻结目标、allowed actions、预算和失败条件。
- train 用于产生经验，dev 用于选择候选，hidden 用于阶段验收，transfer 只在最终使用。
- Agent 只提交 artifact；可信进程独立重算分数。
- 每个候选记录 parent、patch、模型、seed、成本、结果和拒绝原因。
- 同时报 capability、cost、latency、regression、integrity 和 safety。
- 允许负结果；伪造实验、修改 evaluator、泄漏 hidden labels 或隐瞒失败使项目不合格。

## 计算资源分层

| 资源 | 可完成范围 | 调整方式 |
|---|---|---|
| API-only | 全部 Agent 与 Auto Research；Agent RL 用搜索/偏好学习替代在线更新 | 严格记录调用费用，使用小任务集 |
| 单卡 24–48 GB | 1–7B LoRA SFT、偏好优化和短轨迹 RL | 减少 rollout 长度和 group size |
| 4–8 卡 | 多轮 on-policy Agent RL、并行 rollout | 保持相同评测协议，不以算力替代消融 |

## 结业标准

- 四个项目均可从空目录复现。
- Day 1 的 baseline 与 Day 30 的最终系统使用相同的核心任务定义。
- 至少一个优化方法在 hidden/transfer 上无效或退化，并得到诚实分析。
- 最终系统通过 evaluator tampering、data leakage、fabricated result 和 cost explosion 四类红队测试。
- 答辩只使用 E0–E5 证据等级描述能力，不使用“接近 AGI/ASI”等不可检验结论。

参考资料：[[2026-09-27_agent-self-evolution-rsi-asi-course-references]]。

