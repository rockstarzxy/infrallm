---
title: "Day 17：长轨迹信用分配与训练样本转换"
type: concept
tags: [agentic-rl, credit-assignment, trajectory]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 17：长轨迹信用分配与训练样本转换

## 今日目标

把 runtime 轨迹可靠转换成训练 transitions，解决“最终失败到底怪哪一步”。

## 三种粒度

- trajectory-level：所有 assistant tokens 共用终局 reward，简单但方差大。
- turn-level：每次模型响应有 advantage，需要中间 verifier 或 value estimation。
- action/span-level：只训练工具选择、参数或关键 reasoning span，归因更精细但实现复杂。

信用可以来自 return-to-go、learned value、leave-one-action-out replay、process verifier 或层级 decomposition。反事实 replay 最有解释性，但计算昂贵；过程 judge 更便宜但可能偏。

## 数据转换不变量

原始 token IDs 优先于重新 tokenize；chat template、special tokens 和 tool serialization 必须与 rollout 一致。保存 `prompt_ids/response_ids/loss_mask/logprobs/rewards/advantages/policy_version`。环境 token mask 为 0，assistant action token mask 为 1。

## 实验

1. 将 100 条轨迹转换成 turn-level 样本。
2. 检查 decode 后文本与原 response 一致；任何 drift 单独记录。
3. 对截断、工具异常、用户取消、超时分别定义是否训练和 reward。
4. 比较终局 reward 广播、return-to-go、启发式首次错误分配。
5. 随机抽 20 条可视化 token mask 和 advantage，人工审计。

## 验收

转换具有幂等性；同一轨迹重复转换 hash 相同；没有把 tool output 当模型动作；policy version 可追踪。提交 schema、转换器、100 条数据统计和 20 条可视化审计。

必读：[Agent Lightning](https://arxiv.org/abs/2508.03680)。下一课：[[agent-rsi-30d/week3/day18-sft-offline-learning]]。

