---
title: "Day 12：Workflow 与 Agent 代码进化"
type: concept
tags: [agent, workflow, code-evolution, patch]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# Day 12：Workflow 与 Agent 代码进化

## 今日目标

从改 prompt 扩展到改 planner、router、memory policy 和 context manager，同时保证每个 patch 可审查、可回滚。

## 可修改面

先写 allowlist：`agent/strategies/`、`prompts/`、`routing.yaml` 可改；`evaluator/`、`sandbox/`、`secrets/`、`audit/` 不可读写。候选必须是基于固定 parent commit 的 unified diff，并通过 path policy、静态检查、单测、资源检查、dev eval 五道 gate。

把结构参数化：计划深度、是否验证、工具重试、memory top-k、上下文压缩阈值。优化器先尝试配置改动，再尝试代码；小 patch 更容易归因。

## 实验

1. 从 Day 7 的首次错误分布提出 10 个结构改动。
2. 每个改动写“预期改变哪个指标”和“可能伤害什么”。
3. 自动生成 patch，执行五道 gate；拒绝原因进入 archive。
4. 对接受项做 paired hidden evaluation 和 regression suite。
5. 对最优版本执行 git bisect 式消融，验证收益来自哪个 patch。

## 防止伪进步

检查候选是否增加预算、跳过验证、吞掉异常、读取 task ID 映射或修改报告逻辑。运行 mutation tests：故意破坏工具结果，确认 Agent 仍能发现错误。

## 验收

至少 20 个候选 patch、5 种拒绝原因、一个成功回滚案例；所有 accepted patch 都有 parent、测试、资源差异和 transfer 风险。交付 patch pipeline 与 change card。

下一课：[[agent-rsi-30d/week2/day13-multi-agent-coevolution]]。

