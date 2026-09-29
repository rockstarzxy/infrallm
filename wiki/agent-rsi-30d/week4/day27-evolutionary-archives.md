---
title: Day 27：开放式搜索、Population Archive 与谱系
type: concept
tags: [agent, evolutionary-search, archive, dgm]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 27：开放式搜索、Population Archive 与谱系

只保留当前最高分候选，会让自修改系统迅速收敛到同一类技巧。今天构建 population archive：同时保存性能、多样性和祖先关系，让已经暂时落后的候选仍有机会产生有价值的后代。

## 学习目标

1. 理解 greedy hill climbing、Pareto archive 和 quality-diversity 的区别。
2. 设计可解释的行为描述符与 novelty score。
3. 保存完整谱系，用祖先消融判断改进来自哪里。
4. 避免 archive 膨胀、重复候选和评测噪声主导选择。

## 1. 候选与 Archive

每个候选记录：`candidate_id`、`parent_ids`、`generation`、`patch_hash`、`behavior_descriptor`、`dev_score`、`cost`、`safety_score`、`status`。行为描述符应描述策略行为，例如规划深度、工具调用数、记忆读取比例、重试方式；不要直接复制最终分数。

三种选择器：

- **Greedy**：只从最高分候选产生后代，利用强但容易早熟收敛。
- **Pareto**：保留质量、成本、安全互不支配的候选。
- **Quality-diversity**：按行为格子保存各区域精英，鼓励探索不同机制。

Novelty 可用候选行为向量到 archive 中 k 个近邻的平均距离。距离定义必须冻结，并对量纲归一化。

## 2. 谱系是证据，不是装饰

谱系图回答三个问题：某个提升由哪个祖先引入？该特征在后代是否稳定？不同谱系是否独立发现同一机制？为每条父子边保存 mutation rationale 和 diff。若候选有两个父代，说明合并规则和冲突处理。

定期做祖先消融：把后代中的某个关键 patch 回退到祖先版本，其余保持不变。如果收益消失，才支持该 patch 的因果贡献。只看整代分数无法排除共同变化。

## 3. DGM 式开放式改进

Darwin Gödel Machine 的关键启发是：不只维护单一路径，而是保留可产生新后代的多样 archive，并让修改后的 Agent 继续提出修改。工程实现仍需外部 evaluator、sandbox 和 immutable audit log。开放式指搜索空间不被单一模板固定，不代表取消边界。

## 4. 选择与资源分配

每一代按如下比例分配父代：50% 来自 Pareto 前沿，30% 来自高 novelty 区域，20% 随机从有效 archive 抽取。先用廉价 proxy 评测所有候选，再让前 30% 进入完整 dev 评测。任何候选首次进入 archive 前至少复跑两个种子，避免幸运噪声成为祖先。

Archive 清理只能去掉完全重复、无产物或确定损坏的条目。低分候选可以冷存储，不能从审计记录中删除。

## 5. 今日实验

用昨天的 patch generator 产生至少 30 个候选、运行 4 代。并行比较 greedy 与 quality-diversity 两个策略，严格使用相同候选总数和完整评测预算。输出：最佳分数曲线、archive 覆盖率、候选两两距离、谱系图、失败类别分布。

选择最终候选后，对其两个最重要的祖先 patch 做回退消融。再在 transfer 集上比较最佳分数候选和一个多样性候选，观察 dev 排名是否外推。

## 验收标准

- 至少 30 个候选全部有父代、diff、行为描述符和评测状态。
- Greedy 与 quality-diversity 使用相同生成和评测预算。
- 谱系图可以定位最终候选的全部祖先，两个关键 patch 有消融证据。
- 报告覆盖 archive 多样性和 transfer 结果，不以单一 dev 最大值下结论。

必读：[Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)。下一课：[[agent-rsi-30d/week4/day28-controlled-rsi]]。
