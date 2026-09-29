---
title: "Day 15：Agent RL 的 POMDP 与策略梯度基础"
type: concept
tags: [agentic-rl, pomdp, policy-gradient]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 15：Agent RL 的 POMDP 与策略梯度基础

## 学习目标

把多轮工具 Agent 写成 POMDP，理解 return、advantage、on-policy 数据和 KL，而不是只会调用训练框架。

## 形式化

环境有隐藏状态 `s_t`，Agent 只见 observation `o_t`；history 或 memory 形成 belief。动作可以是一串语言 token，也可把一次完整 tool call 视为高层动作。轨迹 `τ=(o₀,a₀,r₀,...,o_T)`，目标：

```text
J(θ)=E[Σ γᵗr_t] - β E[KL(πθ(.|h_t) || πref(.|h_t))]
∇J≈E[Σ ∇logπθ(a_t|h_t) A_t]
```

return-to-go 把未来奖励分给早期动作；baseline/value 减小方差；advantage 表示该动作相对当前状态通常选择的动作好多少。on-policy 意味着数据来自当前或足够接近的策略，旧轨迹直接重复训练会产生偏差。

## 语言 Agent 的特殊问题

tool observation 不是模型动作，不应计算 policy loss；system/user/tool tokens 通常 mask；只有 assistant 生成 token 参与。一次错误 tool call 可能包含几十 token，但真正决策点是工具名和参数，token-level credit 不天然合理。

## 实验

先在 10 状态文本迷宫实现 REINFORCE：比较无 baseline、state baseline、advantage normalization。观察 reward 方差、收敛和 seed 差异。然后把 Day 7 一条轨迹标出 observation、action、environment transition、reward 和 loss mask。

手算两条轨迹的 discounted return；改变 `γ`、终局奖励和 step penalty，解释哪个行为会被鼓励。最后写一页 sequence RLHF 与 agentic RL 的差异。

## 验收

代码能用固定 seed 复现实验；能解释为什么 tool output 不进入策略损失、旧轨迹为何不是无限可复用的 on-policy 数据。交付 notebook、手算表和 POMDP 定义。

下一课：[[agent-rsi-30d/week3/day16-reward-design]]。

