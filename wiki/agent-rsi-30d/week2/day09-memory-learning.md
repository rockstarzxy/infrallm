---
title: "Day 9：记忆学习与经验抽象"
type: concept
tags: [agent, memory, lifelong-learning, retrieval]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# Day 9：记忆学习与经验抽象

## 今日目标

把单任务 reflection 变成可跨任务复用、可撤销的经验，同时测量错误记忆造成的负迁移。

## 三层记忆

| 层 | 内容 | 写入条件 | 读取方式 |
|---|---|---|---|
| episodic | 完整任务和结果 | 所有运行 | 按相似任务检索 |
| semantic | 抽象规则、失败模式 | 多次证据或验证通过 | 条件匹配 + 置信度 |
| procedural | 可执行技能或 workflow | 独立任务验证通过 | 由 router 调用 |

memory item 包含 `claim/evidence/scope/counterexamples/source_runs/version/confidence/expiry`。不要只存“成功轨迹”；失败与反例决定规则边界。

## 写入与治理

候选经验先进入 quarantine，在独立 validation tasks 上验证后升格。相互矛盾的规则不直接覆盖，而是保存条件分支。每条记忆统计使用次数、收益、伤害次数和最后验证时间；低收益或高伤害项降权/过期。

## 实验

1. 从 Day 8 轨迹抽取 30 条 episodic memories。
2. 聚类后生成 10 条 semantic rules，每条附至少两个证据和一个反例。
3. 比较无记忆、原轨迹检索、摘要检索、验证后规则四组。
4. 制造相似但关键约束相反的任务，测错误检索率和负迁移。
5. 做 leave-one-family-out：某任务族完全不参与记忆生成，只用于 transfer。

## 验收

报告 memory hit、useful hit、harmful hit、平均上下文开销和 transfer gain。提交 memory schema、writer/retriever、升格与删除策略，以及三条被拒绝的“看起来合理”经验。

下一课：[[agent-rsi-30d/week2/day10-skills-curriculum]]。

