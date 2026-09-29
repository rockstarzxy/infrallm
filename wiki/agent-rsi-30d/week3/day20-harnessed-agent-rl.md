---
title: "Day 20：Harnessed Agent RL 系统设计"
type: concept
tags: [agentic-rl, harness, rollout, training-system]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 20：Harnessed Agent RL 系统设计

## 今日目标

把部署时使用的真实 Agent harness 接到 trainer，而不是在训练框架里重写一个失真的环境循环。

## 解耦架构

```text
Agent workers ──LLM requests──> versioned policy server
      │                              │ token IDs/logprobs
      └──events/tool results──> trajectory store
                                     │
verifier workers ──rewards───────────┤
                                     ↓
                              learner → new policy version
```

harness 拥有控制流、tools 和 memory；trainer 只能消费标准化交互。每条 rollout 固定 policy version，避免一次轨迹中途换权重。新版本先进入 canary workers，验证格式和安全后再扩大。

## 系统难点

retokenization drift 会让训练 token 与采样 token 不一致；动态 tool output 使 samples 长度差异巨大；多轮样本合并影响 loss normalization；慢环境造成 rollout straggler；失败任务不能全部丢弃，否则策略学不到恢复。

## 实验

1. 为 Day 2 runtime 加 policy endpoint，不改业务控制流。
2. endpoint 返回 token IDs、logprobs、model/policy version。
3. 并行启动 4 个 workers，采集 200 条多轮轨迹。
4. 实现 verifier queue 和 learner input manifest；重复/过期结果去重。
5. 模拟 learner 更新、worker 超时、策略回滚和部分服务故障。
6. 比较按 token、turn、trajectory 三种 loss normalization。

## 验收

训练和 Agent 运行可独立重启；每条样本能追到精确策略；中途换版本被拒绝；故障不会丢失审计事件。提交架构图、接口协议、负载统计和恢复演练。

必读：[Agent Lightning v1.0](https://arxiv.org/abs/2608.17528)。下一课：[[agent-rsi-30d/week3/day21-agent-rl-project]]。

