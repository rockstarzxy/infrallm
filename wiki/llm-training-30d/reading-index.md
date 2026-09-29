---
title: LLM 训练与后训练 30 天教程索引
type: synthesis
tags: [llm-training, post-training, reading-list]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# 30 天教程索引

配套课程：[[llm-training-30d/learning-path]]。每篇教程包含学习目标、工业现状、核心原理、实现步骤、实验、常见失败与诊断、验收标准、交付物、参考。

## Week 1：训练全景与分布式训练基础

| 天 | 教程 | 当天交付物 |
|---:|---|---|
| 1 | [[llm-training-30d/week1/day01-training-landscape|训练全景与工业配方]] | 六家配方对比表、rollout 术语表 |
| 2 | [[llm-training-30d/week1/day02-training-loop-memory-math|训练循环与显存数学]] | 显存公式与实测对照表、MFU 计算 |
| 3 | [[llm-training-30d/week1/day03-data-pipeline-tokenization|数据流水线、chat template 与 packing]] | loss mask 验证、packing 吞吐对比 |
| 4 | [[llm-training-30d/week1/day04-data-parallel-zero-fsdp|数据并行：DDP、ZeRO、FSDP2]] | FSDP2 训练脚本、吞吐随卡数曲线 |
| 5 | [[llm-training-30d/week1/day05-model-parallel-tp-pp-cp-ep|模型并行：TP、PP、CP、EP]] | 并行方案选择表、通信量估算 |
| 6 | [[llm-training-30d/week1/day06-frameworks-efficiency-profiling|框架、效率技术与 profiling]] | profiler 报告、框架选型表 |
| 7 | [[llm-training-30d/week1/day07-p0-project-fsdp2-training|P0 项目：FSDP2 训练与 MFU 报告]] | P0 报告、Week 1 面试题 |

## Week 2：SFT、PEFT、数据工程、偏好优化与奖励模型

| 天 | 教程 | 当天交付物 |
|---:|---|---|
| 8 | [[llm-training-30d/week2/day08-sft-engineering|SFT 原理与工程]] | 多轮 SFT 脚本、超参对照 |
| 9 | [[llm-training-30d/week2/day09-lora-qlora-peft|LoRA、QLoRA 与 PEFT]] | LoRA 与全参对照实验 |
| 10 | [[llm-training-30d/week2/day10-data-engineering-synthetic|数据工程：合成、rejection sampling、蒸馏]] | 数据流水线与质量报告 |
| 11 | [[llm-training-30d/week2/day11-preference-optimization-dpo|偏好优化：DPO 家族]] | DPO 训练与似然/长度分析 |
| 12 | [[llm-training-30d/week2/day12-reward-models|奖励模型与过优化]] | 小 RM 与过优化曲线 |
| 13 | [[llm-training-30d/week2/day13-evaluation-experiment-management|评测体系与实验管理]] | 评测集、置信区间报告 |
| 14 | [[llm-training-30d/week2/day14-p1-project-sft-dpo|P1 项目：SFT + DPO 流水线]] | P1 报告、Week 2 面试题 |

## Week 3：RLVR 与 rollout 系统

| 天 | 教程 | 当天交付物 |
|---:|---|---|
| 15 | [[llm-training-30d/week3/day15-rollout-and-rl-basics|rollout 与 RL 基础]] | 手写 REINFORCE、rollout 数据结构 |
| 16 | [[llm-training-30d/week3/day16-ppo-grpo-family|PPO、GRPO、DAPO、GSPO 家族]] | 目标函数推导、算法选择表 |
| 17 | [[llm-training-30d/week3/day17-rlvr-verifiers-reward-design|RLVR：verifier 与 reward 设计]] | verifier 服务、reward hacking 案例 |
| 18 | [[llm-training-30d/week3/day18-rl-training-frameworks|RL 训练框架架构与选型]] | verl 调用链笔记、选型矩阵 |
| 19 | [[llm-training-30d/week3/day19-rollout-engine-weight-sync|rollout 引擎与权重同步]] | 权重同步实验、TIS 修正对比 |
| 20 | [[llm-training-30d/week3/day20-async-rl-stability|异步 RL 与训练稳定性]] | 稳定性诊断面板 |
| 21 | [[llm-training-30d/week3/day21-p2-project-grpo|P2 项目：GRPO 数学/代码任务]] | P2 报告、Week 3 面试题 |

## Week 4：Agentic RL、推理模型与生产

| 天 | 教程 | 当天交付物 |
|---:|---|---|
| 22 | [[llm-training-30d/week4/day22-agentic-rl-formalization|Agentic RL 形式化]] | 多轮 loss mask 实现 |
| 23 | [[llm-training-30d/week4/day23-agent-environments-gyms|环境与 gym 接口]] | 环境实现与沙箱池 |
| 24 | [[llm-training-30d/week4/day24-agent-rollout-infrastructure|Agent rollout 基础设施]] | 异步 agent loop、rollout 监控 |
| 25 | [[llm-training-30d/week4/day25-agent-reward-credit-assignment|Agent reward 与 credit assignment]] | reward 规范、hacking 检测 |
| 26 | [[llm-training-30d/week4/day26-case-studies-swe-tool-rl|案例：SWE 与 tool-use RL]] | 案例配方对比表 |
| 27 | [[llm-training-30d/week4/day27-reasoning-models-long-cot|推理模型与长 CoT 训练]] | R1 配方复现计划 |
| 28 | [[llm-training-30d/week4/day28-safety-alignment-rlhf-production|安全对齐、RLHF 生产与打标]] | 打标协议、安全评测 |
| 29 | [[llm-training-30d/week4/day29-training-ops-at-scale|大规模训练运维]] | 容错与 checkpoint 方案 |
| 30 | [[llm-training-30d/week4/day30-p3-capstone-agentic-rl|P3 capstone：Agentic RL]] | 结课报告、20 道面试题 |

## 与其他两门课的衔接

| 本课 | 复用 |
|---|---|
| Day 5 并行 | 推理课 [[ai-infra-30d/week3/day15-gpu-topology]] 到 [[ai-infra-30d/week3/day19-moe-inference]] 的通信基础 |
| Day 19 rollout 引擎 | 推理课 [[ai-infra-30d/week2/day08-vllm-architecture]]、[[ai-infra-30d/week2/day10-kv-cache]] |
| Day 22 到 25 agentic RL | Agent 课 [[agent-rsi-30d/week1/day04-tracing-replay]]、[[agent-rsi-30d/week1/day05-evaluation-verifiers]]、[[agent-rsi-30d/week3/day15-agent-rl-foundations]] 到 [[agent-rsi-30d/week3/day17-credit-assignment]] |
| Day 29 发布流程 | 推理课 [[ai-infra-30d/week3/day20-production-deploy]] 与优化专项 [[ai-infra-30d/optimization/14-cold-start-autoscaling]] |
