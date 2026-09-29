---
title: "Day 6：LLM Judge、统计与可信结论"
type: concept
tags: [agent, llm-judge, statistics, evaluation]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# Day 6：LLM Judge、统计与可信结论

## 今日目标

学会在无法完全程序化评分时使用 LLM judge，并避免把噪声、顺序偏差或额外预算误报为提升。

## 1. Judge 设计

把 rubric 拆成可单独判断的维度，每个维度有定义、正反例和 `insufficient_evidence`。pairwise 判断通常比绝对 1–10 分稳定，但要交换 A/B 顺序；judge 不应知道哪个是新版本。引用和事实性必须通过检索或程序检查，不让 judge 凭记忆裁决。

用人工标注校准 30–50 个样本，计算一致率、混淆矩阵或秩相关。judge ensemble 只有在错误不完全相关时才有意义。

## 2. 统计报告

对二元成功率给 bootstrap 或 Wilson 置信区间；版本比较尽量用 paired bootstrap，因为两个版本做的是同一批任务。预先定义主指标；大量切片后只挑最好结果属于多重比较问题。

效率比较使用 Pareto 图，而不是把质量和费用随意加权成一个分数。至少画 success–cost、success–steps 两张图。

## 今日实验（7 小时）

1. 为 50 个开放式回答写四维 rubric：正确性、证据、完整性、表达。
2. 人工双标 30 条，解决分歧并冻结 rubric。
3. 运行单 judge、交换顺序 judge、三 judge ensemble。
4. 测 position bias、自偏好和长度偏好；报告与人工的一致率。
5. 比较 Day 2 三个 baseline，使用 paired bootstrap 和同预算指标。

## 验收

只有当改进的置信区间、逐任务差异和成本变化同时报告时，才允许写“提升”。交付 judge prompt/version、人工校准集、偏差分析、统计 notebook 和 Pareto 图。

下一课：[[agent-rsi-30d/week1/day07-baseline-project]]。

