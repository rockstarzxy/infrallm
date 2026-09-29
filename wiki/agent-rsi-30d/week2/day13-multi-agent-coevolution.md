---
title: "Day 13：多 Agent、辩论与共进化"
type: concept
tags: [agent, multi-agent, debate, coevolution]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# Day 13：多 Agent、辩论与共进化

## 今日目标

判断什么时候角色分工真的增加信息，什么时候只是把一次调用变成昂贵的“多人聊天”。

## 设计原则

角色必须有不同信息、工具、目标或检查责任。planner/executor/critic 如果使用相同上下文和模型，错误高度相关。通过权限和证据划分制造有意义的独立性：研究者提出方案，执行者只运行实验，审计者只看 artifact 和规则。

通信使用结构化 contract：claim、evidence、request、decision。限制轮数和消息预算。最终决定者必须说明采用/拒绝哪些证据，不能只取多数票。

共进化可以让 task generator 与 solver 互相提升，但任务必须由独立 verifier 验证；否则生成器会产出模糊或不可解任务，solver 学会利用漏洞。

## 实验

比较单 Agent、actor-critic、planner-executor-critic、三份独立候选 + selector。固定总 token 和工具次数；测 success、错误相关性、分歧解决率和成本。再让 task generator 产生 50 个任务，过滤重复、不可解和泄漏答案的任务。

对 selector 做盲测和顺序交换；检查“强模型做 selector”是否只是引入更强基础模型。至少分析 10 个多 Agent 比单 Agent 更差的任务。

## 验收

只有在同预算或明确成本—质量 Pareto 改善时采用多 Agent。交付角色权限表、通信 schema、相关错误矩阵、成本对照和 task-generator 质量报告。

下一课：[[agent-rsi-30d/week2/day14-self-evolving-project]]。

