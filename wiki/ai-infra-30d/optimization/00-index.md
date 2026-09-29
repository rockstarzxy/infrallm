---
title: 推理优化专项：问题驱动手册
type: synthesis
tags: [ai-infra, inference-optimization, troubleshooting, playbook]
sources: []
created: 2026-09-29
updated: 2026-09-29
---

# 推理优化专项：问题驱动手册

[[ai-infra-30d/reading-index|30 天日课]]按技术点组织，这个专项按**问题**组织：线上或压测中遇到某个症状，从这里进入，沿着"判定 → 根因 → 方法 → 副作用 → 验证"走一遍。每页都假设你已经学过对应的日课，只讲怎么用。

## 使用方法

1. 先读 [[01-diagnosis-playbook]]，用它的四象限把问题归到 prefill-bound、decode-bound、scheduler/KV-bound、CPU/network-bound 之一。
2. 按下面的症状表进入对应专项页。
3. 每页第 3 节的决策表给出首选、次选和"不要做"，第 4 节给出验证实验。改一个变量，测一次，再改下一个。

## 症状入口

| 你看到的症状 | 首先怀疑 | 进入 |
|---|---|---|
| TTFT P99 高，`num_requests_waiting` 长期大于 0 | 排队、prefill 太重、prefix cache 未命中 | [[02-ttft-high]] |
| TPOT / ITL P99 远高于 P50，流式输出一顿一顿 | prefill 分片干扰、抢占、CUDA graph 未命中 | [[03-tpot-itl-jitter]] |
| 加并发吞吐不涨，GPU 没满 | KV 容量、CPU-bound、TP 通信 | [[04-throughput-low]] |
| 启动 OOM、运行时 OOM、`num_preemptions_total` 增长 | 显存构成算错、max_model_len 过大 | [[05-oom-preemption]] |
| SM 利用率 50% 上下，EngineCore 进程 CPU 100% | launch 开销、调度开销、前端 tokenize/detokenize | [[06-gpu-util-low-cpu-bound]] |
| 32k 以上输入时 TTFT 秒级、并发骤降 | attention O(n²)、KV 线性增长 | [[07-long-context]] |
| 多轮对话或 agent 场景 prefix cache 命中率低、E2E 长尾 | 前缀被破坏、路由不亲和、tool 循环 | [[08-multi-turn-agent-workload]] |
| JSON / tool call 请求比普通请求慢很多或解析失败 | grammar 编译、bitmask、parser | [[09-structured-output-tool-calls]] |
| 量化后没变快甚至变慢，或质量掉 | 格式与 batch 不匹配、kernel 回退 | [[10-quantization-choice]] |
| 开了推测解码吞吐反而降 | 接受率低、batch 大、k 过大 | [[11-spec-decode-no-gain]] |
| MoE 模型吞吐远低于同激活参数的 dense 模型 | all-to-all、专家不均、小 batch | [[12-moe-serving]] |
| TP 加大后延迟没降或吞吐降 | 通信占比、拓扑、应该用 DP | [[13-parallelism-choice]] |
| 扩容要十几分钟，滚动更新期间 P99 飙升 | 权重加载、graph capture、排空 | [[14-cold-start-autoscaling]] |
| 单位 token 成本高，GPU 小时买了但没用满 | goodput 低、机型错配、配置浪费 | [[15-cost-per-token]] |

## 专项页清单

| 编号 | 页面 | 一句话 |
|---|---|---|
| 01 | [[01-diagnosis-playbook]] | 通用诊断流程：指标 → 象限 → 工具 → 假设验证 |
| 02 | [[02-ttft-high]] | TTFT 过高的十类根因与修法 |
| 03 | [[03-tpot-itl-jitter]] | TPOT/ITL 抖动与 P99 长尾 |
| 04 | [[04-throughput-low]] | 吞吐上不去：从 roofline 看 batch 与瓶颈 |
| 05 | [[05-oom-preemption]] | OOM 与抢占：显存构成与容量规划 |
| 06 | [[06-gpu-util-low-cpu-bound]] | GPU 利用率低与 CPU 瓶颈 |
| 07 | [[07-long-context]] | 长上下文推理 |
| 08 | [[08-multi-turn-agent-workload]] | 多轮对话与 Agent workload |
| 09 | [[09-structured-output-tool-calls]] | 结构化输出与 tool calling |
| 10 | [[10-quantization-choice]] | 量化选型与"量化后没变快" |
| 11 | [[11-spec-decode-no-gain]] | 推测解码没有收益 |
| 12 | [[12-moe-serving]] | MoE 部署 |
| 13 | [[13-parallelism-choice]] | 并行策略选择 |
| 14 | [[14-cold-start-autoscaling]] | 冷启动与扩缩容 |
| 15 | [[15-cost-per-token]] | 单位 token 成本与容量规划 |

## 与日课的对应

| 专项页 | 依赖的日课 |
|---|---|
| 01 | [[day03-benchmark]]、[[day05-profiling]]、[[day26-roofline]] |
| 02、03、04 | [[day06-param-tuning]]、[[day09-scheduler]]、[[day13-flash-attention]] |
| 05 | [[day04-paged-attention]]、[[day10-kv-cache]] |
| 06 | [[day08-vllm-architecture]]、[[day13-flash-attention]] |
| 07 | [[day27-long-context]]、[[day10-kv-cache]] |
| 08 | [[day18-data-parallel]]、[[day10-kv-cache]] |
| 09 | [[day13-flash-attention]]、[[day23-sglang]] |
| 10 | [[day11-quantization]] |
| 11 | [[day12-speculative-decoding]] |
| 12 | [[day19-moe-inference]] |
| 13 | [[day15-gpu-topology]]、[[day16-tensor-parallel]]、[[day17-pipeline-parallel]] |
| 14 | [[day20-production-deploy]] |
| 15 | [[day21-system-design-v1]]、[[day29-final-design]] |

## 通用原则

- **先看指标再改参数。** 每页第 1 节列的指标组合就是判定依据，不要凭印象调 `max_num_seqs`。
- **一次改一个变量。** 固定模型、dtype、workload、并发模型、warmup，否则"收益"可能只是 workload 变了（[[day03-benchmark]]）。
- **区分 P50 和 P99。** 大多数优化对 P50 有效但对 P99 无效，反之亦然。SLO 通常写在 P99 上。
- **副作用要写进决策。** 每个方法都有代价：prefix caching 占显存、chunked prefill 拉高长 prompt TTFT、量化可能掉质量、spec decode 在高并发下反而慢。
- **版本敏感。** 参数名、默认值、backend 支持随 vLLM 版本漂移，以 `--help` 和启动日志为准。
