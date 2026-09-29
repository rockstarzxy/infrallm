---
title: "Day 19：PPO、GRPO 与在线策略优化"
type: concept
tags: [agentic-rl, ppo, grpo, online-rl]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# Day 19：PPO、GRPO 与在线策略优化

## 今日目标

理解并运行小规模 on-policy 更新，能从指标判断训练是在学习任务还是利用 reward 漏洞。

## 两类目标

PPO 用 value model 估计 advantage，并裁剪 importance ratio：`min(rA, clip(r,1-ε,1+ε)A)`。相对组方法对同一 prompt 采样多个回答，以组内 reward 标准化作为 advantage，省去独立 value model，但依赖足够多样的候选；组内 reward 全相同时没有学习信号。

KL 约束限制新策略偏离 reference。KL 太小可能无学习，太大可能格式崩坏或 reward hacking。监控 response length、entropy、clip fraction、KL、reward 各分量、真实成功和 invalid action。

## 实验

1. 选 1–3B 或 LoRA 模型，先在单轮可验证子任务跑通。
2. 对每个 prompt 采样 4–8 条候选，保存原 logprobs。
3. 运行短训练：每隔固定 updates 在冻结 dev 上评测。
4. 做三组：无 KL、固定 KL、adaptive KL；再比较 outcome-only/shaped reward。
5. 人工审计 reward 上升最快的 20 条轨迹，查找捷径。

## 停止条件

若真实 success 连续三个评测点不升、integrity 下降、KL/长度突变或 dev–train gap 扩大，停止并回滚。不要用训练 reward 的峰值选择 checkpoint；预先定义 checkpoint rule。

## 验收

提交完整配置、policy/reference 版本、训练曲线、checkpoint 规则、真实指标与 20 条审计。能解释 PPO 与 group-relative 方法各自在长轨迹上的限制。

下一课：[[agent-rsi-30d/week3/day20-harnessed-agent-rl]]。

