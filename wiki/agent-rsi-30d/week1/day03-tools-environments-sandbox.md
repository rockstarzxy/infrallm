---
title: "Day 3：工具、环境与沙箱"
type: concept
tags: [agent, tools, environment, sandbox, security]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# Day 3：工具、环境与沙箱

## 今日目标

把工具调用当成带副作用的环境动作，建立最小权限、确定性回放和可信边界。

## 1. Tool contract

每个工具必须声明：输入 schema、输出 schema、超时、最大输出、允许的副作用、幂等性和错误码。工具返回值应包含 `ok/data/error/duration/artifact_ids`，而不是把 stdout 直接塞回上下文。

环境负责状态和转移。相同动作未必得到相同结果，例如网页变化或随机训练；因此轨迹同时保存 observation 和 environment version。用于训练的环境应尽量快照化。

## 2. 信任边界

| 区域 | Agent 权限 | 示例 |
|---|---|---|
| workspace | 读写 | 候选代码、实验输出 |
| reference | 只读 | 任务说明、公开数据 |
| evaluator | 不可见、不可写 | hidden tests、评分器 |
| credentials/network | 默认无权 | API key、外部写操作 |

沙箱至少限制工作目录、进程时间、CPU/内存、输出大小和网络。清理环境应通过新建快照，不依赖 Agent 自己撤销改动。

## 今日实验（7 小时）

1. 新增只读搜索工具：对本地文档语料做检索，返回文档 ID 和证据片段。
2. 新增代码执行工具：在临时工作区运行 Python，强制超时和输出截断。
3. 为两个工具各写 10 个 contract tests，包括 schema 错误、路径越界和超时。
4. 构造 prompt injection 文档，验证检索内容只能作为数据，不能改变 runtime 权限。
5. 运行 20 个“检索后计算”任务；区分模型失败、工具失败和环境失败。

## 常见错误

- 把 tool description 当安全边界；模型可以不遵守描述，权限必须由执行器强制。
- 把 hidden test 放在同一可搜索目录。
- 自动重试非幂等工具，造成重复副作用。
- 将完整网页或巨量日志回填上下文，引发成本和注入风险。

## 验收

路径越界、网络访问、超时和超大输出都被拦截并进入轨迹；重放时可选择使用已记录 observation 或重新执行。交付 `tool-contracts.md`、沙箱测试和威胁边界图。

下一课：[[agent-rsi-30d/week1/day04-tracing-replay]]。

