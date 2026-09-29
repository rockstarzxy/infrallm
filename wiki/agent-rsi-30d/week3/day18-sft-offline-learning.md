---
title: "Day 18：SFT、偏好学习与离线 Agent 学习"
type: concept
tags: [agentic-rl, sft, preference-learning, offline]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# Day 18：SFT、偏好学习与离线 Agent 学习

## 今日目标

先建立强、便宜、稳定的离线 baseline，再判断在线 RL 是否必要。

## 数据选择

成功轨迹不等于优质示范：可能步骤冗余、依赖偶然工具结果或包含未被 verifier 检出的错误。过滤格式错误、完整性标记和异常成本；按任务族平衡，避免高频简单任务支配训练。

SFT 学习 `-log π(a|h)`；它复制示范分布，不能直接优化长期 reward。偏好学习用同一状态下 chosen/rejected 对，适合工具选择和修复比较；pair 必须控制信息量，不能让 chosen 总是更长或来自更强工具。

离线数据有 coverage 问题：训练无法学习从未出现的好动作，也容易在新状态分布失效。保留行为策略和 propensity 信息，评估时关注 distribution shift。

## 实验

1. 从 P1 轨迹生成 SFT 数据：成功、修复成功、低成本成功三套。
2. 训练 LoRA 或在 API-only 路线中构建 few-shot distilled policy。
3. 从相同状态的候选建立 preference pairs，做 DPO 类训练或 reranker。
4. 比较 base、SFT、preference、SFT+preference；使用相同 runtime。
5. 测 seen family、held-out template、transfer tool 三类任务。

## 诊断

检查 action validity、工具过度调用、探索减少、长轨迹退化和 catastrophic forgetting。SFT success 上升但新状态恢复能力下降是常见结果。

## 验收

提交 dataset card、过滤漏斗、训练曲线、四组对照和每类 10 个失败案例。只有离线 baseline 冻结后才进入在线策略优化。

下一课：[[agent-rsi-30d/week3/day19-policy-optimization]]。

