---
title: LLM 训练与后训练 30 天课程
type: synthesis
tags: [llm-training, post-training, sft, lora, rlvr, grpo, agentic-rl, rollout, learning-path]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# LLM 训练与后训练：30 天课程

知识库里的第三门课。三门课的分工：

| 课程 | 回答的问题 | 入口 |
|---|---|---|
| AI 推理基础设施 | 训练好的模型怎么部署、压测、优化 | [[ai-infra-30d/reading-index]] |
| Agent 自迭代与 RSI | Agent 怎么测量、怎么不改权重地自迭代、怎么受控地自动科研 | [[agent-rsi-30d/index]] |
| **LLM 训练与后训练（本课）** | 模型权重怎么被训出来：分布式训练、SFT/LoRA、偏好优化、RLVR、rollout 系统、Agentic RL | 本页 |

三门课在两处交汇：RL 训练里要跑推理引擎做 rollout（本课 Week 3 直接用推理课 Week 2 的 vLLM 知识）；Agent RL 的环境、reward 和评测协议沿用 Agent 课 Week 1 和 Week 3。

## 课程目的

30 天后能独立跑通 2025 到 2026 年一线团队公开配方里的后训练流水线，并解释每一步为什么这样做：

- 算清训练显存和吞吐，为给定模型和卡数选对 DP/TP/PP/CP/EP 组合与框架。
- 跑通 SFT → 偏好优化 → RLVR（GRPO 家族）→ Agentic RL，每一段都有 hidden eval 和消融。
- 说清 rollout 在训练系统里的位置：采样、on/off-policy、rollout 引擎与权重同步、异步与 staleness、训练推理数值不一致的修正。
- 诊断训练不稳定（entropy 崩塌、KL 爆、长度爆炸）和 reward hacking。
- 知道 DeepSeek-R1、Qwen3、Kimi K2、GLM-4.5、Tulu 3、Llama 3 各自的后训练配方，能在资源缩小的条件下复现其中的关键环节。

只教实际在用、当前领先的方法。被替代的方法只在对比时提一句。

## 入口

- [[llm-training-30d/learning-path|30 天学习路径]]：目标、阶段、四个项目、统一实验协议、资源分层。
- [[llm-training-30d/reading-index|Day 1 到 30 教程索引]]：每天一篇可直接学习和实践的教程。
- [[2026-09-29_llm-training-course-references|参考资料归档]]：论文与框架编号表（T01 到 T47，B01 到 B15）。

## 四条主线

| 主线 | 天数 | 最终产出 |
|---|---:|---|
| 训练全景与分布式训练基础 | Day 1 到 7 | P0：FSDP2/TorchTitan 训练小模型，显存构成与 MFU 报告 |
| SFT、PEFT、数据工程、偏好优化、奖励模型 | Day 8 到 14 | P1：SFT + on-policy 偏好对 + DPO 完整流水线与评测 |
| RLVR 与 rollout 系统 | Day 15 到 21 | P2：verl GRPO/DAPO 数学或代码任务，含消融与 reward hacking 审计 |
| Agentic RL、推理模型与生产 | Day 22 到 30 | P3：可验证工具任务上的 agentic RL capstone |

后续材料统一维护在 `wiki/llm-training-30d/`。
