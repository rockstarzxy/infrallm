---
title: "Day 11：Prompt 自动优化与反思式进化"
type: concept
tags: [agent, prompt-optimization, gepa, evolution]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 11：Prompt 自动优化与反思式进化

## 今日目标

实现一个可审计的 prompt 优化器，理解候选生成、评测、选择、变异和过拟合之间的关系。

## 搜索表示

优化对象拆成角色/目标、决策规则、工具说明、停止条件、输出 schema 和 examples。每次 mutation 只改一到两个块，记录 diff 和依据。整段重写难以归因，也容易丢失已验证约束。

采用多目标 archive：最大化 success，同时最小化 tokens、steps 和 safety violations。候选 `a` 支配 `b` 当且仅当所有目标不差且至少一个更好；保留 Pareto frontier，而不是过早固定主观加权。

## GEPA 风格循环

1. 从 archive 选 parent。
2. 采样 train trajectories，找重复失败。
3. 用语言反思形成可检验规则。
4. 生成最小 prompt diff。
5. train smoke test 后用 dev 选择。
6. 将互补候选交叉，保存谱系。

## 实验

运行至少五代，每代 8 个候选。比较 random mutation、只看 scalar reward、trajectory reflection 三组；每组保持候选数和调用预算相同。最终只在 hidden 上评估 archive 中预先选定的三项。

检查 prompt injection 鲁棒性、长度膨胀、规则冲突和对单一模板的硬编码。统计每种 mutation 类型的接受率和真实增益。

## 验收

提交 optimizer、候选 diff、Pareto 图、谱系和 hidden 结果。若 dev 持续上涨而 hidden 不涨，必须将结论写为 prompt overfitting，而不是“优化成功”。

必读：[GEPA](https://arxiv.org/abs/2507.19457)、[TextGrad](https://arxiv.org/abs/2406.07496)。下一课：[[agent-rsi-30d/week2/day12-workflow-code-evolution]]。

