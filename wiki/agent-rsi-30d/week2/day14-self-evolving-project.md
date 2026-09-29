---
title: "Day 14：P1 非参数自迭代 Agent 项目"
type: synthesis
tags: [agent, self-evolution, project, evaluation]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# Day 14：P1 非参数自迭代 Agent 项目

## 项目问题

在模型权重、任务定义和总预算不变时，记忆、技能、prompt 与 workflow 的多代优化能否在新任务上持续改善？

## 五代协议

Gen 0 是 Day 7 baseline。每代只允许最多 8 个候选；先在 train 产生经验，在 dev 选择，接受版本成为下一代 parent。Gen 5 完成前不得查看 transfer。archive 保留性能、成本和多样性 Pareto 候选。

每代执行：收集失败 → 提出因果假设 → 选择改动层 → 产生最小 diff → contract/regression test → dev evaluation → 接受或拒绝 → 写 generation report。

## 必做消融

- 无 reflection；无 semantic memory；无 skill library；random prompt mutation；禁止 workflow patch。
- 最优系统使用 Gen 0 预算；Gen 0 使用最优系统预算。
- 清空记忆后运行，区分结构进步与记忆查表。

## 报告表

每代报告 dev/hidden（只在预定检查点）success、transfer 最终结果、tokens、steps、wall time、harmful memory rate、integrity flags、accepted/attempted patches。画出代际曲线与谱系，不只展示最优点。

## 通过条件

1. 至少五代，每代完整可回放。
2. hidden 或 transfer 相对 Gen 0 改善，置信区间和成本变化透明。
3. 至少记录一个被选择指标误导的候选。
4. 明确声明这是 L1/L2 自迭代，不是权重学习或 RSI。

提交可复现命令、archive、谱系图、消融、失败候选和两页研究结论。下一课：[[agent-rsi-30d/week3/day15-agent-rl-foundations]]。

