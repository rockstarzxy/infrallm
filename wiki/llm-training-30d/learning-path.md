---
title: "LLM 训练与后训练：30 天高强度学习路径"
type: synthesis
tags: [llm-training, post-training, distributed-training, sft, lora, dpo, reward-model, rlvr, grpo, rollout, agentic-rl]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# LLM 训练与后训练：30 天高强度学习路径

课程入口：[[llm-training-30d/index]]；每日教材：[[llm-training-30d/reading-index]]。

## 课程目标

30 天约 250 到 300 小时。完成后，你应能从一个开源基座模型出发，在自己的算力上完成 SFT、偏好优化、可验证奖励 RL 和多轮 Agent RL，每一步有 baseline、消融和 hidden 评测，并能解释训练系统里 rollout、权重同步、异步与数值一致性这些工程决策为什么这样做。

主线不是"学会调一个训练脚本"，而是理解下面这条链上每一环的原理、工业配方和失败模式：

```text
基座模型
  → 中期训练（长上下文、退火、领域数据）
  → SFT（全参或 LoRA；数据来自人工、合成、rejection sampling、蒸馏）
  → 偏好优化（DPO 家族）或奖励模型
  → RLVR：rollout 引擎采样 → verifier 打分 → GRPO 家族更新 → 权重同步回 rollout 引擎
  → Agentic RL：多轮环境、工具、trajectory 级 reward、异步 rollout
  → 评测门禁、量化导出、交给推理系统（推理课）
```

## 设计原则

1. **只教在用的方法。** 每个环节先写 2025 到 2026 年公开配方（T14 到 T22）里的做法，再讲原理。被替代的方法（如 PPO 的 value model 在 RLVR 里的位置）只作对比。
2. **rollout 贯穿全课。** 它在评测、数据合成、RL 和打标里都出现，含义略有不同；每次出现都说清楚。
3. **训练和推理是一个系统。** RL 训练时间通常一半以上花在 rollout 上，所以 Week 3 直接复用推理课的 vLLM V1 知识。
4. **每天有可验证交付物。** 代码、表、曲线、报告，缺一不可。只读不算完成。
5. **资源分层。** 每个实验给 API-only、单卡、多卡三档做法，不因为没有集群就跳过分布式内容。

## 30 天安排

| 阶段 | 天数 | 内容 | 阶段项目 |
|---|---:|---|---|
| Week 1 | 1 到 7 | 训练全景、训练循环与显存数学、数据流水线、DP/ZeRO/FSDP2、TP/PP/CP/EP、框架与效率、profiling | P0：FSDP2 训练小模型与 MFU 报告 |
| Week 2 | 8 到 14 | SFT 工程、LoRA/QLoRA、数据工程与合成、DPO 家族、奖励模型、评测与实验管理 | P1：SFT + DPO 流水线 |
| Week 3 | 15 到 21 | rollout 与 RL 基础、PPO/GRPO/DAPO/GSPO、RLVR 与 verifier、RL 框架架构、rollout 引擎与权重同步、异步 RL 与稳定性 | P2：verl GRPO 数学/代码任务 |
| Week 4 | 22 到 30 | Agentic RL 形式化、环境与 gym、agent rollout 基础设施、agent reward、SWE/tool-use RL 案例、推理模型、安全对齐与打标、大规模训练运维 | P3：agentic RL capstone |

## 每日固定节奏

1. 2 小时：读当天教程与指定论文段落。
2. 2 小时：复现最小例子，先确认数据流和公式。
3. 4 到 5 小时：核心实验，保存全部配置、seed、曲线和 artifact。
4. 1 小时：跑 hidden 评测，与 baseline 比较。
5. 1 小时：写实验日志：失败、费用、下一步假设。

## 四个项目

### P0：FSDP2 训练小模型（Day 1 到 7）

用 FSDP2 或 TorchTitan 继续训练一个 0.5B 到 1.5B 模型（或从头训一个 125M）。交付显存构成表（参数、梯度、优化器状态、activation 的实测与公式对照）、MFU、吞吐随 GPU 数与并行度的变化、checkpoint/resume 一致性验证、loss 曲线。

### P1：SFT + DPO 流水线（Day 8 到 14）

在 1.5B 到 8B 模型上：SFT（LoRA 或全参）→ 用 SFT 模型采样构造 on-policy 偏好对 → DPO → 在内部评测集和至少两个公开基准上对比。必须报告 DPO 后的似然变化和长度变化，说明是否出现长度偏置。

### P2：RLVR（Day 15 到 21）

用 verl（或 TRL）在 GSM8K/MATH 或小型代码任务上跑 GRPO 或 DAPO，1.5B 到 7B。交付 baseline、两个消融（KL 系数、dynamic sampling）、hidden eval、reward hacking 检查、rollout 与训练时间占比分析。

### P3：Agentic RL capstone（Day 22 到 30）

选一个可验证工具任务（text-to-SQL、小型代码修复、检索问答）。SFT 冷启动 → agentic GRPO（多轮、工具观察 mask）→ hidden eval → reward hacking 审计 → 成本报告。环境、verifier 和 hidden tests 复用 Agent 课 Week 1 的协议。

## 统一实验协议

- 训练前冻结：模型、数据版本、评测集、seed、预算、成功标准。
- train / dev / hidden 分离；hidden 只在阶段验收用；transfer 只在最终用。
- 每次实验记录：配置快照、git commit、数据 hash、GPU 小时、rollout 与训练时间、所有曲线。
- 报告同时给：能力、成本、长度、稳定性指标（entropy、KL、梯度范数）、reward hacking 审计。
- 允许负结果。修改 verifier、泄漏 hidden、隐瞒失败使项目不合格。

## 计算资源分层

| 资源 | 可完成范围 | 调整方式 |
|---|---|---|
| API-only | Week 1 的公式与手算、Week 2 的数据工程与评测、Week 3/4 的 rollout 数据分析与算法推导；训练实验用公开 checkpoint 和日志复盘 | 用 0.5B 模型在 CPU 或 Colab 跑通代码路径 |
| 单卡 24 到 48 GB | 0.5B 到 1.5B 全参、7B LoRA/QLoRA SFT 与 DPO、1.5B GRPO（colocated rollout） | 缩小 group size、序列长度、rollout 数 |
| 4 到 8 卡 | 7B 全参 SFT、7B GRPO/DAPO、多轮 agentic RL、TP/PP/CP 实验 | 保持相同评测协议，不用算力替代消融 |

## 结业标准

1. 给定模型规模、序列长度、卡数，写出显存与吞吐估算并选出并行方案。
2. 从零跑通 SFT、DPO、GRPO、agentic GRPO 各一次，每次有 hidden eval。
3. 解释 GRPO、DAPO、GSPO 目标函数的每一项及其解决的问题。
4. 解释 rollout 引擎权重同步、训练推理数值不一致与重要性采样修正、异步 RL 的 staleness 控制。
5. 看到 entropy 崩塌、KL 爆、长度爆炸、reward 升评测降时给出诊断路径。
6. 复述 DeepSeek-R1、Qwen3、Kimi K2、GLM-4.5 的后训练配方并指出差异。
7. 写出一份 agentic RL 项目的完整报告，含 reward hacking 审计与成本。

## 关键参考

按 [[2026-09-29_llm-training-course-references]] 的编号引用。最先读的五篇：T17（DeepSeek-R1）、T36（GRPO）、T37（DAPO）、T42（verl/HybridFlow）、T27（DPO）。
