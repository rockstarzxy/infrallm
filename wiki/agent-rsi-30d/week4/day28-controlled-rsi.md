---
title: Day 28：受控 RSI 循环与跨代证据
type: concept
tags: [agent, rsi, recursive-self-improvement, evaluation]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 28：受控 RSI 循环与跨代证据

RSI 的核心不是“Agent 改了自己的文件”，而是改进后的系统能够更有效地生成下一轮改进，并在外部评测上持续成立。今天运行至少五代受控循环，用跨代、迁移和改进效率指标检验这一点。

## 学习目标

1. 给 RSI 写出操作性定义和可否证判据。
2. 区分目标能力提升与改进能力提升。
3. 控制每代预算，避免把更多算力误认成递归增益。
4. 识别平台期、退化、评价器过拟合和能力–安全权衡。

## 1. 课程中的 RSI 定义

设第 (g) 代系统为 (A_g)，改进算子为 (I_g)，产生 (A_{g+1}=I_g(A_g, D_g))。我们要求：

1. `A_g` 的可持久组件发生可审计变化；
2. 变化由当前代参与提出、实现或选择；
3. 外部 immutable evaluator 显示目标能力提升；
4. 后续代在相同预算内产生有效改进的效率不下降，并最好提升；
5. 结果在 hidden 或 transfer 上保留，安全约束未越界。

只满足前三项可称为 self-improving agent。要主张 RSI，还需展示改进过程本身的跨代增强证据。

## 2. 双层指标

**对象层**衡量任务能力：hidden pass@1、质量、成本、延迟、安全失败率。**元层**衡量改进能力：每 10 个候选带来的净提升、有效 patch 比例、达到阈值所需实验数、失败诊断准确率、改进的 transfer 保留率。

定义每代改进效率：

\[
\eta_g=\frac{S(A_{g+1})-S(A_g)}{\text{model calls}+\lambda\cdot\text{compute cost}}
\]

单次 `η` 为负不必立即判失败，但五代趋势、置信区间和累计收益必须报告。禁止删除退化代。

## 3. 五代协议

1. 冻结 `G0`、任务集、预算、verifier、安全 gate 和停止规则。
2. 每代让当前系统分析失败并生成固定数量候选。
3. 用 Day 27 archive 选择后代；每代只晋升一个主版本，保留旁支。
4. 每代在相同 dev 预算下选择；hidden 只在预定代次运行。
5. 第 5 代冻结后统一跑 hidden、两个 transfer 集和 red-team。
6. 重放完整谱系，确认报告数字和版本对应。

设置停止条件：连续两代对象层无显著提升、元层效率持续下降、安全 gate 失败，或累计预算耗尽。停止是有效实验结果。

## 4. 必做对照

- **Static improver**：始终由 `G0` 生成每代候选，检验改进后的改进器是否有贡献。
- **Random mutation**：相同数量和成本的随机合法 patch。
- **Best-of-N prompt**：只增加采样，不修改持久组件。
- **Frozen verifier transfer**：换任务分布但保持评价标准，检查是否只记住 dev。

若 `A_g` 提高，但与 static improver 没有显著差异，应报告“迭代搜索有效，递归增强证据不足”。

## 5. 今日运行

完成至少五代，每代 8–12 个候选。记录代际得分、η、有效 patch 率、候选多样性、回归数、失败诊断准确率。为每代写一段“系统认为自己为何失败”并与 evaluator 归因对比，测量 self-model 的校准误差。

## 验收标准

- 五代都遵守相同资源上限，任何预算变化均预注册并单列。
- 主版本形成连续谱系，所有父代、候选和晋升决定可回放。
- 至少完成 static improver、random mutation 两个关键对照。
- 结论明确区分 self-improvement、迭代搜索和 RSI，失败时不升级术语。

可延伸阅读：[AIDE²](https://arxiv.org/abs/2609.26457)。下一课：[[agent-rsi-30d/week4/day29-control-oversight]]。
