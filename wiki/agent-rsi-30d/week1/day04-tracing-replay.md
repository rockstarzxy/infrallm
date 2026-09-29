---
title: "Day 4：轨迹、事件与回放"
type: concept
tags: [agent, tracing, replay, observability]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# Day 4：轨迹、事件与回放

## 今日目标

把 Agent 运行变成可训练、可调试、可审计的数据，而不是一段终端日志。

## 1. Event-sourced trace

轨迹由不可变事件构成。最小字段：`run_id/task_id/event_id/parent_id/type/timestamp/component/version/input_hash/output_hash/token_usage/cost/error`。大对象存 artifact store，事件只保存 hash 和 URI。

事件类型至少包括 `model_request`、`model_response`、`action_parsed`、`tool_started`、`tool_finished`、`state_updated`、`reward_assigned`、`artifact_created`、`run_finished`。

回放有三种：pure replay 直接读旧 observation；fork replay 在某一步换动作；live replay 重新运行工具。三者不能混淆，否则无法判断差异来自策略还是环境漂移。

## 2. 归因方法

先用 taxonomy 标记理解、计划、工具选择、参数、执行、验证和停止错误。再做最小反事实：保留前缀，只替换首次错误动作。如果任务成功，首次错误附近是候选瓶颈；仍失败则继续定位。

## 今日实验（7 小时）

1. 将 Day 2/3 runtime 改成事件写入，不允许直接散落 `print` 作为唯一记录。
2. 实现 JSONL writer 和 DuckDB/SQLite 查询层。
3. 实现 pure replay，确保不调用模型和工具也能重建最终状态。
4. 对 10 条失败轨迹做 fork replay，每条只替换一个动作。
5. 输出漏斗：任务数 → 合法首动作 → 工具成功 → 正确验证 → 正确停止。

## 数据质量检查

- 每个 started 事件必须有 finished/error。
- token 和 cost 汇总等于子事件之和。
- artifact hash 可解析且内容不可变。
- schema version 明确，旧轨迹可迁移。
- 敏感字段在落盘前脱敏，而不是展示时才隐藏。

## 验收

随机抽取五条运行可以完全重建；失败查询能在一分钟内回答“最常见首次错误是什么”；提交 trace schema、replay CLI、失败漏斗和五个反事实案例。

下一课：[[agent-rsi-30d/week1/day05-evaluation-verifiers]]。

