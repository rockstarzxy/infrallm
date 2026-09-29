---
title: 30 天推理优化学习 — 内容索引
type: synthesis
tags: [ai-infra, reading-list, learning-path]
sources: []
created: 2026-06-01
updated: 2026-09-27
---

# 30 天推理优化学习 — 内容索引

本目录为 [[ai-inference-learning-path]] 的配套自包含学习材料。每天一篇完整教程，直接阅读即可学习；涉及 vLLM / TensorRT-LLM / SGLang 的具体参数、默认值和指标名时，以当前安装版本的 `--help`、启动日志和官方文档为准。

外部辅助课程见 [[supplementary-courses]]。

Week 2 的 Day 8/9/10/13 是源码级内容，按 vLLM V1（`vllm/v1/`）编写，每天都有改源码的交付物。开始 Week 2 前先从源码安装 vLLM（`VLLM_USE_PRECOMPILED=1 pip install -e .`）。

## Week 1: 推理系统基础与 vLLM 上手

| 天 | 文件 | 主题 | 核心内容 |
|---:|---|---|---|
| 1 | [[day01-inference-pipeline]] | 推理链路与 KV Cache | prefill/decode 区别、KV Cache 计算、MQA/GQA |
| 2 | [[day02-vllm-setup]] | vLLM 部署 | 安装、启动参数、OpenAI API、日志解读 |
| 3 | [[day03-benchmark]] | 指标体系与压测 | TTFT/TPOT/ITL 定义、benchmark 工具、workload 设计 |
| 4 | [[day04-paged-attention]] | PagedAttention | 分页 KV Cache、block table、continuous batching、中国模型 KV 特点 |
| 5 | [[day05-profiling]] | GPU Profiling | nvidia-smi、PyTorch profiler、Nsight Systems、瓶颈判断 |
| 6 | [[day06-param-tuning]] | 参数调优 | gpu_memory_utilization、max_model_len、max_num_seqs 联动分析 |
| 7 | [[day07-week1-review]] | Week 1 复盘 | 故障排查 checklist、面试题、知识图谱 |

## Week 2: vLLM 深入与核心推理优化

| 天 | 文件 | 主题 | 核心内容 |
|---:|---|---|---|
| 8 | [[day08-vllm-architecture]] | vLLM V1 架构（源码级） | 三层进程模型、ZMQ/msgspec、带路径的请求生命周期、CPU 开销分布、加自定义指标 |
| 9 | [[day09-scheduler]] | V1 Scheduler 精读 | token budget 循环、num_computed_tokens、recompute 抢占、priority、async scheduling、自定义调度器 |
| 10 | [[day10-kv-cache]] | KVCacheManager 与 KV Connector | BlockPool/hash 链/LRU 驱逐、hybrid KV、connector 接口、写文件 connector、KV FP8 |
| 11 | [[day11-quantization]] | 量化 | GPTQ/AWQ/SmoothQuant/FP8、训练推理一致性 |
| 12 | [[day12-speculative-decoding]] | 推测解码 | draft model、acceptance rate、Medusa/EAGLE/n-gram |
| 13 | [[day13-flash-attention]] | 模型执行层 | GPUModelRunner、persistent batch、torch.compile/piecewise CUDA graph、attention backend、FlashAttention、sampler/logits processor/结构化输出 |
| 14 | [[day14-tuning-playbook]] | 综合调优 | 优化决策表、场景化配方、常见陷阱 |

## Week 3: 分布式推理与部署

| 天 | 文件 | 主题 | 核心内容 |
|---:|---|---|---|
| 15 | [[day15-gpu-topology]] | GPU 拓扑 | NVLink/PCIe/IB 带宽、NCCL 集合通信、通信量估算 |
| 16 | [[day16-tensor-parallel]] | Tensor Parallelism | Megatron-style TP、column/row parallel、GQA 下的 TP |
| 17 | [[day17-pipeline-parallel]] | Pipeline Parallelism | PP 原理、bubble 分析、TP+PP 组合、多节点部署 |
| 18 | [[day18-data-parallel]] | 多实例与路由 | DP、LB 策略、prefix-aware routing、限流降级 |
| 19 | [[day19-moe-inference]] | MoE 推理 | Expert Parallel、All-to-All、DeepSeek-V3/MiniMax 部署 |
| 20 | [[day20-production-deploy]] | 生产部署 | K8s/Ray、监控告警、rolling update、部署 checklist |
| 21 | [[day21-system-design-v1]] | 系统设计 v1 | 500 QPS 推理平台设计完整案例 |

## Week 4: 高阶优化与最终项目

| 天 | 文件 | 主题 | 核心内容 |
|---:|---|---|---|
| 22 | [[day22-tensorrt-llm]] | TensorRT-LLM | engine build、kernel fusion、和 vLLM 对比选型 |
| 23 | [[day23-sglang]] | SGLang | RadixAttention、jump-forward decoding、三框架选型 |
| 24 | [[day24-disaggregated]] | Prefill-Decode 解耦 | DistServe/Splitwise、KV 传输分析、适用场景 |
| 25 | [[day25-triton-kernel]] | Triton 编程 | CUDA 基础、Triton softmax 实战、kernel fusion |
| 26 | [[day26-roofline]] | Roofline 分析 | AI 计算、compute/memory bound 判断、量化/GQA 对 roofline 影响 |
| 27 | [[day27-long-context]] | 长上下文推理 | KV offload、Ring Attention、sparse attention、StreamingLLM |
| 28 | [[day28-final-experiment]] | 最终实验 | 6 组实验的完整 benchmark 流程 |
| 29 | [[day29-final-design]] | 最终设计 | 10 章节的生产级推理平台设计模板 |
| 30 | [[day30-retrospective]] | 面试复盘 | 20 个高频面试题 + 公式速查 + 进阶方向 |

## 推理优化专项：问题驱动手册

日课按技术点组织，专项按问题组织。线上或压测遇到症状时从 [[optimization/00-index]] 进入，按"判定 → 根因 → 方法 → 副作用 → 验证"走一遍。

| 编号 | 页面 | 症状 |
|---|---|---|
| 01 | [[optimization/01-diagnosis-playbook]] | 通用诊断流程与四象限归类 |
| 02 | [[optimization/02-ttft-high]] | TTFT 过高 |
| 03 | [[optimization/03-tpot-itl-jitter]] | TPOT/ITL 抖动与 P99 长尾 |
| 04 | [[optimization/04-throughput-low]] | 吞吐上不去 |
| 05 | [[optimization/05-oom-preemption]] | OOM 与抢占 |
| 06 | [[optimization/06-gpu-util-low-cpu-bound]] | GPU 利用率低、CPU 瓶颈 |
| 07 | [[optimization/07-long-context]] | 长上下文 |
| 08 | [[optimization/08-multi-turn-agent-workload]] | 多轮对话与 Agent workload |
| 09 | [[optimization/09-structured-output-tool-calls]] | 结构化输出与 tool calling |
| 10 | [[optimization/10-quantization-choice]] | 量化选型与量化后没变快 |
| 11 | [[optimization/11-spec-decode-no-gain]] | 推测解码没有收益 |
| 12 | [[optimization/12-moe-serving]] | MoE 部署 |
| 13 | [[optimization/13-parallelism-choice]] | 并行策略选择 |
| 14 | [[optimization/14-cold-start-autoscaling]] | 冷启动与扩缩容 |
| 15 | [[optimization/15-cost-per-token]] | 单位 token 成本与容量规划 |

建议在 Day 14 和 Day 21 之后各通读一遍，Day 28 最终实验时按专项页的实验做。

## 涉及的中国模型

| 模型 | 涉及章节 | 主要知识点 |
|---|---|---|
| Qwen2.5 / Qwen3 系列 | Day 1-6, 10, 14, 16, 17, 19, 21 | KV 计算、部署、调优、TP、MoE、thinking/non-thinking 模式、系统设计 |
| DeepSeek-V2/V3 | Day 4, 10, 11, 16, 19, 24, 27 | MLA、MoE EP、解耦、长上下文 |
| GLM-4 | Day 4, 6, 11 | GQA、量化、部署 |
| MiniMax-Text-01 | Day 4, 13, 19 | Lightning Attention、MoE |
| Baichuan2 | Day 10 | 量化 |

## 论文参考编号

| 编号 | 论文 | arXiv / 出处 | 涉及章节 |
|---|---|---|---|
| P01 | Attention Is All You Need | 1706.03762 | Day 1 |
| P02 | FlashAttention | 2205.14135 | Day 13 |
| P03 | FlashAttention-2 | 2307.08691 | Day 13 |
| P04 | vLLM / PagedAttention | 2309.06180 | Day 4 |
| P05 | Orca (Continuous Batching) | OSDI 2022 | Day 4 |
| P06 | Megatron-LM | 1909.08053 | Day 16 |
| P07 | GPTQ | 2210.17323 | Day 11 |
| P08 | AWQ | 2306.00978 | Day 11 |
| P09 | SmoothQuant | 2211.10438 | Day 11 |
| P10 | Speculative Decoding | 2302.01318 | Day 12 |
| P11 | Medusa | 2401.10774 | Day 12 |
| P12 | EAGLE | 2401.15077 | Day 12 |
| P13 | DistServe | 2401.09670 | Day 24 |
| P14 | Splitwise | 2311.18677 | Day 24 |
| P15 | SGLang / RadixAttention | 2312.07104 | Day 23 |
| P16 | StreamingLLM | 2309.17453 | Day 10, 27 |
| P17 | DeepSeek-V2 | 2405.04434 | Day 4, 19 |
| P18 | Ring Attention | 2310.01889 | Day 27 |
| P19 | GQA | 2305.13245 | Day 1 |
