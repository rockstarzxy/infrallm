---
title: Day 24：Auto Research 系统拆解与等预算复现
type: comparison
tags: [agent, auto-research, ai-scientist, alphaevolve]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 24：Auto Research 系统拆解与等预算复现

今天不按产品名称背系统，而是比较三种搜索范式：研究树搜索、角色化研发循环、进化式程序搜索。你将为同一个小问题实现三个最小版本，以相同模型调用、运行时间和评测集比较它们。

## 学习目标

1. 把 Auto Research 分解为 idea、implementation、experiment、review、archive 五个模块。
2. 理解树搜索、Researcher–Developer 循环和 population search 的差异。
3. 建立等预算比较，防止“多花十倍调用量所以更好”的伪结论。
4. 识别系统可以自动化的部分和仍需外部判断的部分。

## 1. 三种架构

| 范式 | 状态 | 扩展方式 | 选择方式 | 代表性系统 |
|---|---|---|---|---|
| Progressive tree search | 研究节点：想法、代码、结果 | 从有希望节点生成子实验 | reviewer 分数与实验反馈 | AI Scientist-v2 |
| R&D loop | 需求、研究方案、实现任务 | Researcher 提方案，Developer 执行 | 评测结果与角色反馈 | R&D-Agent |
| Evolutionary search | 程序种群与 archive | mutate/crossover/LLM edit | 指标、质量与多样性 | AlphaEvolve |

三者都需要外部 evaluator。若研究代理可以改 benchmark、读取 hidden answer 或修改聚合脚本，搜索过程失去意义。

## 2. 统一任务

选择一个 30–60 秒可完成的任务，例如优化一个受测试保护的 Python 函数、改进小型分类器特征管道，或提升工具 Agent 在 20 个固定任务上的成功率。冻结：

- 初始代码和允许修改的目录；
- train/dev/hidden 划分；
- 每种方法最多 30 个候选、60 次模型调用或 90 分钟；
- 单次候选的 CPU/GPU/内存限制；
- hidden evaluator 和最终聚合脚本。

## 3. 实现最小版本

### A. 研究树

根节点是 baseline。每个节点保存假设、diff、dev 指标、评审和父节点。每轮从上置信界最高的节点扩展两个子节点；连续三次无提升则停止该分支。最终选择必须基于 dev，hidden 只运行一次。

### B. Researcher–Developer

Researcher 只写结构化 issue：观察、假设、预期机制、验收。Developer 只能根据 issue 修改代码，Reviewer 检查测试和结果。加入一次“实现失败退回研究者”循环，记录角色通信成本。

### C. 进化搜索

维护 8 个候选的 archive。父代选择同时考虑性能和新颖度；mutation 提示包含失败日志，但不包含 hidden 结果。每代产生 4 个子代，共运行至预算耗尽。

## 4. 比较指标

主指标使用 hidden performance。次指标包括：最佳结果出现所需实验数、有效候选比例、独立失败模式数、总代价、人工介入次数、候选相似度。画两条曲线：`best score vs. experiments` 与 `best score vs. cost`。

至少做一个机制消融：树搜索去掉 reviewer、R&D loop 合并角色，或进化搜索去掉多样性奖励。若完整系统只在更大调用量下占优，结论应写成“计算扩展带来收益”，不能归因于架构。

## 5. 研究记录

每个候选必须记录来源、生成提示、代码 diff、运行状态和选择原因。对三种方法使用同一 experiment manager 和 verifier。完成后写一页决策备忘录：哪种问题结构适合哪种搜索方式，哪些结论仅对当前任务成立。

## 验收标准

- 三种最小系统均能在统一接口下产生、评测和归档候选。
- 每种系统预算偏差小于 5%，hidden 集只在最终冻结后访问。
- 报告同时给出质量、成本、失败率和多样性，不只报告最佳分数。
- 至少一个消融能说明某个架构部件是否真正贡献提升。

必读：[AI Scientist-v2](https://arxiv.org/abs/2504.08066)、[R&D-Agent](https://arxiv.org/abs/2505.14738)、[AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)。下一课：[[agent-rsi-30d/week4/day25-research-integrity]]。
