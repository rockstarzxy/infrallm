---
title: "Day 7：P0 可回放 Agent 基线项目"
type: synthesis
tags: [agent, project, baseline, evaluation]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# Day 7：P0 可回放 Agent 基线项目

## 项目目标

把前六天组合成稳定实验平台。今天不增加“智能”功能，只修复不可测、不可复现和不可审计的问题。

## 必需目录

```text
p0-agent/
  runtime/        # state machine、model adapter、budget
  tools/          # contract 与沙箱执行器
  tasks/          # manifest；hidden 内容不在 Agent 镜像
  evaluator/      # 独立进程或容器
  traces/         # event schema 与 replay
  reports/        # baseline、失败分类、费用
```

## 三个基线

1. single-shot：一次模型调用，不使用工具。
2. fixed ReAct：最多固定步数，使用相同工具和总预算。
3. best-of-N：多个独立候选，由同一 verifier 选择；总 token 不得超过 ReAct 对照。

每个基线至少跑三次 seed。冻结 prompt、模型快照、采样参数、工具版本和容器 digest。评测报告必须包含逐任务记录，防止平均数掩盖模板性失败。

## 今日执行顺序（9 小时）

1. 运行所有 contract tests、replay tests 和 verifier tests。
2. 清空开发缓存，从干净快照运行 train smoke test。
3. 冻结版本，运行 dev 和 hidden；不要根据 hidden 单题修改系统。
4. 人工审查所有 false positive、false negative 和 integrity flag。
5. 建立 Week 2 不可改变的 baseline tag。
6. 写改进 backlog，每项关联具体失败轨迹，而不是凭直觉加功能。

## 项目评分

| 维度 | 权重 | 合格条件 |
|---|---:|---|
| 可复现 | 25% | 干净环境三次结果差异可解释 |
| 可回放 | 20% | 每条轨迹可重建，artifact hash 有效 |
| 评测隔离 | 20% | Agent 无法读取或修改 hidden grader |
| 基线公平 | 20% | 同模型、同任务、预算明确 |
| 失败分析 | 15% | 至少 20 条失败有首次错误分类 |

## 结业检查

如果换一个 Agent 版本就必须手工改评测器，P0 不合格；如果 success 提升却无法定位来自模型、工具还是预算，也不合格。最终提交运行命令、环境锁文件、baseline 表、失败漏斗和三个优先改进假设。

下一课：[[agent-rsi-30d/week2/day08-reflection-feedback]]。

