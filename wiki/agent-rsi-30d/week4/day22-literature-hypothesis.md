---
title: Day 22：文献检索、Claim Graph 与可证伪假设
type: concept
tags: [agent, auto-research, literature-review, hypothesis]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 22：文献检索、Claim Graph 与可证伪假设

今天把“读论文”改造成一条可审计的研究流水线：检索得到候选文献，证据支持原子主张，主张之间形成图，图中的冲突与空白再生成可证伪假设。最终产物不是一篇综述，而是一组能直接交给实验管理器执行的假设卡。

## 学习目标

1. 区分论文、证据、主张、推断和假设，避免把摘要当作实验事实。
2. 建立带来源定位、证据强度和冲突关系的 claim graph。
3. 把宽泛研究问题压缩成带干预、指标、基线和否证条件的假设。
4. 用“新颖性 × 可证伪性 × 成本 × 风险”筛选实验，而不是追逐听起来宏大的题目。

## 1. 五层研究对象

| 层 | 示例 | 必须记录 |
|---|---|---|
| Paper | 一篇 Agent RL 论文 | 标题、作者、版本、日期、永久链接 |
| Evidence | 表 2 的 hidden-set 结果 | 数据集、设置、方差、限制、页码/表号 |
| Claim | 轨迹级训练优于结果级训练 | 主语、条件、比较项、指标、方向 |
| Inference | 长程任务可能受益于密集归因 | 推断链和不确定度 |
| Hypothesis | 加入步骤归因会提高 pass@1 | 可操纵变量和明确否证标准 |

原子主张建议使用结构：`在条件 C 下，方法 A 相对 B，使指标 M 改变 Δ`。一句话如果包含两个指标或两个条件，就拆成两个节点。

## 2. 建立可追溯文献库

建立 `literature/papers.jsonl`，每行至少含 `paper_id`、`canonical_url`、`version`、`retrieved_at`、`topic`、`status`。建立 `literature/claims.jsonl`，每条 claim 包含：

```json
{"claim_id":"C017","paper_id":"P004","location":"Table 2","claim":"step credit improves hidden pass@1","conditions":["same base model","same budget"],"evidence":"direct experiment","confidence":0.7,"reviewer":"human"}
```

`confidence` 不是论文声望分。它只表达当前证据在你的任务设置中支持该 claim 的程度。没有公开代码、只报最佳种子、训练预算不匹配都要下调置信度。

## 3. 构建 Claim Graph

节点分为 claim、method、dataset、metric、failure。边至少包含：`supports`、`contradicts`、`qualifies`、`depends_on`、`tested_by`。按以下流程完成第一版：

1. 围绕“怎样使 Agent 在固定预算内持续改进”写 5 个检索式。
2. 收集 30–50 篇一手论文或官方技术报告，去除重复版本。
3. 从每篇提取不超过 5 个与本课程直接相关的原子 claim。
4. 随机抽 20% 回看原文，检查来源定位和措辞是否越界。
5. 标出互相矛盾、只在单一 benchmark 成立、缺少强基线的区域。

图的价值在冲突和缺口。若两个方法无法在相同预算下比较，应新增“缺少等预算实验”节点，而不是自行判断胜负。

## 4. 假设卡模板

每张卡只能包含一个主要因果问题：

```yaml
hypothesis_id: H03
question: 步骤级 verifier 是否改善长程工具任务？
intervention: result-only -> step-aware credit
mechanism: 降低早期错误动作的优势估计噪声
baseline: 相同模型、数据、rollout 数的结果奖励
primary_metric: hidden pass@1
secondary_metrics: [tokens, tool_errors, wall_time]
falsified_if: 提升小于 2 个百分点或 95% CI 跨 0
budget: 2 GPU-hours
risks: [verifier leakage, task imbalance]
```

禁止使用“效果更好”“更智能”“更接近 ASI”作为假设。这些表达没有可观察量，也没有失败条件。

## 5. 今日实验

从 claim graph 生成至少 20 张假设卡，按四项各 1–5 分评分：信息增益、新颖性、可执行性、失败风险。选出 3 张，其中至少一张必须可能推翻你当前最相信的结论。两人互换卡片进行 adversarial review：审稿人只需找到一个能让结论失效的混淆因素，就退回重写。

## 验收标准

- 文献库含至少 30 个去重来源，90% 以上 claim 能定位到正文、图或表。
- claim graph 至少有 40 个原子 claim、10 条冲突或限定边。
- 3 张入选假设卡均包含等预算基线、主指标、否证阈值和资源上限。
- 随机审计中不得出现把作者推测写成实验结论的情况。

下一课：[[agent-rsi-30d/week4/day23-experiment-manager]]。
