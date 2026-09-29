---
title: "Day 10：技能库与自动课程"
type: concept
tags: [agent, skills, curriculum, open-endedness]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 10：技能库与自动课程

## 今日目标

让 Agent 把反复出现的多步行为封装成可测试技能，并依据当前能力选择“略高于现有水平”的任务。

## 技能不是聊天摘要

技能应有名称、输入/输出 schema、前置条件、实现、测试、成本、来源版本和失败边界。只有在多个 validation tasks 通过后才能进入 registry。技能组合失败时要能追溯到具体子技能。

自动课程从任务池选择下一批任务。简单的 learning progress 指标为 `LP_i = recent_success_i - past_success_i`；优先选择有正进步、成功率位于 20%–80% 的任务族，同时保留一定 novelty 和随机探索。始终选择最难任务会得到大量无信息失败。

## 实验

1. 从成功轨迹中挖掘重复 action subsequence，提出 5 个技能。
2. 为每个技能生成 contract tests；手工审查副作用。
3. 实现 skill router，并保留不使用技能的 fallback。
4. 建立三档难度、至少六个任务族的任务池。
5. 比较随机课程、固定从易到难、learning-progress 课程。
6. 测新环境中技能复用、组合成功率和 catastrophic forgetting。

## 关键消融

固定总任务数，移除技能库、移除自动课程、只用成功经验、禁止 fallback。结果应回答收益来自更好的经验复用还是仅来自看到更多相似任务。

## 验收

至少三个技能在 unseen tasks 上提高成功或降低成本；自动课程覆盖所有任务族，没有长期饿死；提交 registry、测试、课程选择曲线和被撤销技能案例。

必读：[Voyager](https://arxiv.org/abs/2305.16291)。下一课：[[agent-rsi-30d/week2/day11-prompt-optimization]]。

