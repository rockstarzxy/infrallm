---
title: "Day 5：任务集、数据切分与 Verifier"
type: concept
tags: [agent, evaluation, verifier, benchmark]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# Day 5：任务集、数据切分与 Verifier

## 今日目标

建立一个优化算法无法轻易“背答案”或修改评分器的评测系统。

## 1. 四层数据

- train：Agent 可看到反馈，用于产生经验或更新参数。
- dev：优化器用来比较候选，反复使用会逐渐过拟合。
- hidden：阶段验收；Agent 不见内容，只得到汇总信号。
- transfer：课程最后一次使用，任务分布或工具组合发生变化。

按任务模板、来源和时间分组切分，不能随机打散同一模板的轻微变体。保存 dataset manifest、生成脚本、hash 和污染检查。

## 2. Verifier 层级

优先级通常是形式证明/编译/单元测试 > 可重放模拟器 > 独立程序规则 > 人类 rubric > 校准后的 judge ensemble > 单模型自评。强 verifier 也可能不完整：测试通过不代表代码满足未编码需求。

一个 verifier 输出 `score`、`subscores`、`evidence`、`integrity_flags` 和 `verifier_version`。能力分与完整性分分开，发现篡改时不能只扣一点能力分。

## 今日实验（8 小时）

1. 选择课程主任务：text-to-SQL、证据问答或代码修复。
2. 制作至少 100 个任务，按 50/20/20/10 分成 train/dev/hidden/transfer。
3. 实现确定性 verifier；对等价 SQL/答案使用执行语义而非字符串匹配。
4. 写 20 个 verifier tests：正确、部分正确、格式错、超时、硬编码、修改测试。
5. 将 hidden evaluator 放到独立进程或容器，只接收只读 artifact。

## 指标

报告任务成功、部分分、invalid action、超时、cost、steps 和 integrity violation。对有随机性的 Agent 至少三次运行，保留逐任务结果，不能只留平均数。

## 验收

人工构造的 20 个攻击中，grader 篡改、答案泄漏和硬编码不会得到正常高分；同模板不会跨 split；提交 dataset card、manifest、verifier tests 和基线空提交结果。

下一课：[[agent-rsi-30d/week1/day06-judges-statistics]]。

