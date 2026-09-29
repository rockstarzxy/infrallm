---
title: LLM 训练与后训练 30 天课程参考资料
type: link-collection
created: 2026-09-29
---

# LLM 训练与后训练 30 天课程参考资料

课程 [[llm-training-30d/index]] 使用的一手论文、技术报告和框架。编号 Txx 供各天教程引用。arXiv 编号来自记忆整理，使用前以 arXiv 页面为准。

## 分布式训练与效率

| 编号 | 资料 | 出处 |
|---|---|---|
| T01 | Attention Is All You Need | arXiv 1706.03762 |
| T02 | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | arXiv 1909.08053 |
| T03 | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | arXiv 1910.02054 |
| T04 | PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel | arXiv 2304.11277 |
| T05 | FlashAttention-2 | arXiv 2307.08691 |
| T06 | Ring Attention with Blockwise Transformers | arXiv 2310.01889 |
| T07 | DeepSpeed-Ulysses: Sequence Parallelism | arXiv 2309.14509 |
| T08 | Zero Bubble Pipeline Parallelism | arXiv 2401.10241 |
| T09 | TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training | arXiv 2410.06511 |
| T10 | FP8-LM: Training FP8 Large Language Models | arXiv 2310.18313 |
| T11 | Reducing Activation Recomputation in Large Transformer Models（Megatron sequence parallel + selective recompute） | arXiv 2205.05198 |
| T12 | Training Compute-Optimal Large Language Models（Chinchilla） | arXiv 2203.15556 |
| T13 | Switch Transformers（MoE 与负载均衡损失） | arXiv 2101.03961 |

## 模型与配方技术报告

| 编号 | 资料 | 出处 |
|---|---|---|
| T14 | The Llama 3 Herd of Models | arXiv 2407.21783 |
| T15 | Qwen3 Technical Report | arXiv 2505.09388 |
| T16 | DeepSeek-V3 Technical Report | arXiv 2412.19437 |
| T17 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL | arXiv 2501.12948 |
| T18 | Kimi k1.5: Scaling RL with LLMs | arXiv 2501.12599 |
| T19 | Kimi K2: Open Agentic Intelligence | arXiv 2507.20534 |
| T20 | GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models | arXiv 2508.06471 |
| T21 | MiniMax-M1（CISPO） | arXiv 2506.13585 |
| T22 | Tulu 3: Pushing Frontiers in Open Language Model Post-Training | arXiv 2411.15124 |

## SFT、PEFT、偏好优化、奖励模型

| 编号 | 资料 | 出处 |
|---|---|---|
| T23 | InstructGPT: Training language models to follow instructions with human feedback | arXiv 2203.02155 |
| T24 | LoRA: Low-Rank Adaptation of Large Language Models | arXiv 2106.09685 |
| T25 | QLoRA: Efficient Finetuning of Quantized LLMs | arXiv 2305.14314 |
| T26 | DoRA: Weight-Decomposed Low-Rank Adaptation | arXiv 2402.09353 |
| T27 | Direct Preference Optimization | arXiv 2305.18290 |
| T28 | KTO: Model Alignment as Prospect Theoretic Optimization | arXiv 2402.01306 |
| T29 | SimPO: Simple Preference Optimization with a Reference-Free Reward | arXiv 2405.14734 |
| T30 | ORPO: Monolithic Preference Optimization without Reference Model | arXiv 2403.07691 |
| T31 | Self-Instruct | arXiv 2212.10560 |
| T32 | Constitutional AI: Harmlessness from AI Feedback | arXiv 2212.08073 |
| T33 | Let's Verify Step by Step（PRM） | arXiv 2305.20050 |
| T34 | Generative Verifiers: Reward Modeling as Next-Token Prediction | arXiv 2408.15240 |

## RL 算法与 RL 训练系统

| 编号 | 资料 | 出处 |
|---|---|---|
| T35 | Proximal Policy Optimization Algorithms | arXiv 1707.06347 |
| T36 | DeepSeekMath（GRPO 提出） | arXiv 2402.03300 |
| T37 | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | arXiv 2503.14476 |
| T38 | Understanding R1-Zero-Like Training（Dr. GRPO） | arXiv 2503.20783 |
| T39 | Group Sequence Policy Optimization（GSPO） | arXiv 2507.18071 |
| T40 | REINFORCE++ | arXiv 2501.03262 |
| T41 | Back to Basics: Revisiting REINFORCE Style Optimization（RLOO） | arXiv 2402.14740 |
| T42 | HybridFlow: A Flexible and Efficient RLHF Framework（verl） | arXiv 2409.19256 |
| T43 | OpenRLHF | arXiv 2405.11143 |
| T44 | AReaL: A Large-Scale Asynchronous RL System for Language Reasoning | arXiv 2505.24298 |
| T45 | SWE-RL: Advancing LLM Reasoning via RL on Open Software Evolution | arXiv 2502.18449 |
| T46 | SWE-bench | arXiv 2310.06770 |
| T47 | ReAct: Synergizing Reasoning and Acting in Language Models | arXiv 2210.03629 |

## 博客与工程报告（无 arXiv 编号）

| 编号 | 资料 | 说明 |
|---|---|---|
| B01 | Your Efficient RL Framework Secretly Brings You Off-Policy RL Training（2025） | 训练与推理引擎数值不一致导致的隐式 off-policy，以及 truncated importance sampling 修正 |
| B02 | On-Policy Distillation（Thinking Machines Lab 博客，2025） | 用教师 logprob 做 token 级 on-policy 蒸馏 |
| B03 | Defeating Nondeterminism in LLM Inference（Thinking Machines Lab 博客，2025） | batch-invariant kernel 与 train/infer 一致性 |
| B04 | LoRA Without Regret（Thinking Machines Lab 博客，2025） | LoRA 在 SFT 与 RL 中与全参训练的等价条件 |
| B05 | HuggingFace TRL 文档：SFTTrainer / DPOTrainer / GRPOTrainer | https://huggingface.co/docs/trl |
| B06 | verl 文档 | https://verl.readthedocs.io |
| B07 | slime（THUDM）README | https://github.com/THUDM/slime |
| B08 | NVIDIA NeMo-RL 文档 | https://github.com/NVIDIA-NeMo/RL |
| B09 | Megatron-Core 文档 | https://docs.nvidia.com/megatron-core |
| B10 | TorchTitan 仓库 | https://github.com/pytorch/torchtitan |
| B11 | DeepSpeed 文档 | https://www.deepspeed.ai |
| B12 | SkyRL（NovaSky）仓库 | https://github.com/NovaSky-AI/SkyRL |
| B13 | ROLL（阿里）仓库 | https://github.com/alibaba/ROLL |
| B14 | llm-compressor / PEFT / Unsloth 文档 | https://huggingface.co/docs/peft |
| B15 | OpenAI Gym / Gymnasium 接口约定 | https://gymnasium.farama.org |
