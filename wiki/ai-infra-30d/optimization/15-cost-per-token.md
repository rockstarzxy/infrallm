---
title: "优化专项 15: 单位 token 成本优化与容量规划"
type: concept
tags: [ai-infra, inference-optimization, cost, capacity-planning, goodput, slo, hardware-selection]
created: 2026-09-29
updated: 2026-09-29
---
# 优化专项 15: 单位 token 成本优化与容量规划

> 一句话：把 GPU 账单换算成每百万 token 的成本，找出成本结构里最大的一项，用容量规划表决定要多少实例。关联日课：[[day21-system-design-v1]]、[[day26-roofline]]、[[day29-final-design]]、[[day20-production-deploy]]。

## 1. 症状与判定

成本问题的症状不在单个指标里，而在几个指标的比值里。

| 症状 | 关键证据 | 最可能根因 |
|---|---|---|
| GPU 小时账单高，但 tokens per GPU-hour 远低于同型号公开 benchmark | `vllm:generation_tokens_total` 速率 ÷ GPU 数 | 并发不够，decode 没吃满带宽（根因 A） |
| 利用率看着很高，实际 goodput 低 | 满足 SLO 的请求数 ÷ 总请求数 < 0.9 | batch 超过 SLO 允许的上限，跑了很多"没用"的 token（根因 B） |
| 高峰期 SLO 达标，全天平均利用率 < 30% | 按小时的 `num_requests_running` 曲线 | 容量按峰值预留，无缩容（根因 C） |
| prefill 占了大部分 GPU 时间 | `vllm:request_prefill_time_seconds` 之和远大于 decode 时间之和 | prefill-heavy workload 用了 decode 向的配置，或 prefix cache 没命中（根因 D） |
| 显存里 KV 只占一小部分 | 启动日志 `# GPU blocks` 换算后远小于 `gpu_memory_utilization` 留出的空间 | 权重精度过高或 `max_model_len` 设置浪费（根因 E） |
| 同一批卡上跑着多个低流量模型 | 各实例 `num_requests_running` 长期为个位数 | replica 空转（根因 F） |

## 2. 根因逐个排查

### 成本模型

先把公式写下来，所有优化都是在改分子或分母：

```
每百万 token 成本 = GPU 小时单价 × GPU 数 ÷ (goodput tokens/s × 3600) × 1e6
goodput tokens/s = 满足 SLO 的请求贡献的 output tokens/s（不满足的按 0 计）
```

prefill 和 decode 的成本机制不同，要拆开算：

- **prefill token 成本 ∝ 计算量**。每个 token 约 `2 × 参数量` FLOPs，compute-bound，成本随算力单价走。prefix cache 命中的 token 成本接近 0。
- **decode token 成本 ∝ 每步读权重的时间 ÷ 每步的 token 数**。小 batch 时 memory-bound，一步的时间约为 `权重字节数 ÷ 显存带宽`，这个时间被 batch 内所有请求分摊。batch 翻倍，单 token 成本近似减半，直到进入 compute-bound 或撞到 SLO。

这就是为什么 decode 成本优化的核心是"在 SLO 允许范围内把 batch 做大"，而 prefill 成本优化的核心是"少算"（prefix cache、chunk 合理、避免重复 prefill）。

### 根因 A: 并发不够，decode 没吃满带宽

**确认**。看 `vllm:num_requests_running` 的分布。用 roofline 估一下（[[day26-roofline]]）：当前 batch 下每步的算术强度是否还远低于硬件拐点。

**修**。合并流量到更少的实例（提高每实例并发），或降低 TP 把卡分给更多实例（[[13-parallelism-choice]]）。如果流量本身就低，考虑更便宜的卡（根因 C 的硬件选型）。

### 根因 B: batch 超过 SLO 允许的上限

**机制**。TPOT 随 batch 增长的曲线大致是：小 batch 时平坦（memory-bound，加 batch 几乎不增加步时间），到某个点开始线性上升（compute-bound）。SLO 与这条曲线的交点就是 batch 上限，超过它每增加一个请求都会让所有请求的 TPOT 变差，goodput 反而下降。

**确认**。做 batch 扫描实验（第 4 节前置实验），画 TPOT P99 vs 并发曲线，找到与 SLO 的交点。

**修**。`--max-num-seqs` 设在交点附近；网关层限流把超出的请求排队而不是塞进 batch。prefill 混排会抬高 decode 步时间，`--max-num-batched-tokens` 要一起扫。

### 根因 C: 容量按峰值常驻，硬件选型没按 workload

**硬件的定性对比**（数值为量级，随 SKU 和版本变化，用于选型方向而非精确计算）：

| GPU | 显存 | 显存带宽量级 | 低精度支持 | 适合 |
|---|---|---|---|---|
| A100 80GB | 80 GB | 2 TB/s | 无 FP8 | 已有存量时继续用；decode 单位成本高于 H 系 |
| H100 80GB | 80 GB | 3.3 TB/s | FP8 | 通用 |
| H200 | 141 GB | 4.8 TB/s | FP8 | decode-heavy、长上下文、大 KV 需求 |
| B200 | 180 GB 以上 | 8 TB/s 量级 | FP8 / FP4 | 大模型 decode，MoE |
| L40S | 48 GB | 0.86 TB/s | FP8 | 小模型、prefill 向或低并发场景，单价低 |

decode 成本主要跟"权重字节数 ÷ 带宽"走，所以带宽高、能用 FP8/FP4 的卡 decode 便宜；prefill 成本跟算力走。workload 搭配：
- prefill-heavy（RAG、长文档摘要、输入远长于输出）：算力性价比高的卡，prefix cache 命中率是第一杠杆。
- decode-heavy（对话、代码生成、agent 多轮）：带宽和显存大的卡，batch 上限是第一杠杆。
- 两者混合且量大：P/D 解耦，各配各的卡（[[day24-disaggregated]]）。

**修**。按小时统计流量曲线，常驻容量按谷值加 buffer，峰值靠扩容和预测（[[14-cold-start-autoscaling]]）；低流量模型合并到共享实例或用 LoRA。

### 根因 D: prefill 占比过高

**确认**。`vllm:request_prefill_time_seconds` 与 `vllm:request_decode_time_seconds` 直方图的总和之比；`vllm:prefix_cache_hits ÷ vllm:prefix_cache_queries`。

**修**。prefix cache 命中率的成本换算很直接：

```
节省的 prefill 计算 = 命中 token 数 × 2 × 参数量 FLOPs
```

示例：70B 模型、每请求 2000 token 的共享 system prompt、命中率从 0 提到 0.9，每请求省掉约 1800 token 的 prefill，在 H100 上约为几十毫秒的 GPU 时间，QPS 100 时每天省下的 GPU 小时可以直接算出来。手段：固定 system prompt 放最前、cache-aware routing、多轮对话保持前缀稳定（[[08-multi-turn-agent-workload]]）。

### 根因 E: 显存没有花在 KV 上

**机制**。显存 = 权重 + KV + 激活缓冲 + 预留。权重精度每降一档，腾出的显存全部变成 KV，KV 决定并发上限，并发决定 decode 单位成本。

**确认**。启动日志里权重占用与 `# GPU blocks × block 大小` 的比例；`max_model_len` 是否远大于实际请求长度分布的 P99。

**修**。
- 量化权重（FP8 在 H 系几乎无损，INT4 看任务）：[[10-quantization-choice]]。
- KV FP8：`--kv-cache-dtype fp8`，KV 容量翻倍。
- `max_model_len` 按 P99 请求长度设，超长请求单独实例。
- 关掉不用的功能：`logprobs` 请求会让 sampler 多算并回传大张量；`--enforce-eager` 让 decode 慢 10% 到 30%，等于成本涨同样比例。

### 根因 F: replica 空转

**确认**。按实例看 `num_requests_running` 的时间序列，长期个位数的实例就是空转。

**修**。低流量模型合并：同基座不同微调用多 LoRA 单实例；不同基座的小模型用 MIG 切卡或共享卡；夜间缩容到最小副本数。多租户与优先级下的容量分配：给高优先级租户预留的容量用 `priority` 调度保证顺序（[[day09-scheduler]]），但预留本身是成本，预留比例要按其 SLO 违约代价定，不是按租户数平均分。

### 各类优化如何改变成本结构

| 手段 | 改变的是 | 代价 |
|---|---|---|
| 权重量化 | decode 每步读权重的字节数下降，KV 空间增加 | 质量风险，量化 kernel 可能不如 bf16 GEMM 高效（小 batch 时收益明显，大 batch 时可能反转） |
| MoE 模型 | 每 token 激活参数少，prefill 和 decode 计算都降 | 总权重大，显存和 EP 通信成本 |
| speculative decoding | 每步产出多个 token，decode 单位成本下降 | 只在 batch 小、接受率高时有效；大 batch 时反而增加计算（[[11-spec-decode-no-gain]]） |
| P/D 解耦 | prefill 和 decode 各用最合适的卡和 batch 策略 | KV 传输带宽、两套实例的最小规模 |
| prefix caching | prefill 计算直接省掉 | 需要路由配合，KV 被缓存占用 |
| 增大 batch | decode 单位成本下降 | TPOT 上升，撞 SLO 后 goodput 下降 |

## 3. 决策表

| 情况 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| decode-heavy，tokens per GPU-hour 低 | 提高每实例并发到 SLO 交点 | 换带宽更高的卡 | 加 TP |
| prefill-heavy | 提高 prefix cache 命中率 | 算力性价比高的卡或 P/D 解耦 | 只盯 decode 参数 |
| 全天平均利用率低 | 按曲线缩容 + 合并低流量模型 | 更便宜的卡 | 按峰值常驻 |
| SLO 与成本冲突 | 分租户分 SLO，宽松租户放大 batch | 降级路径（小模型兜底） | 全局放宽 SLO |
| 显存被权重占满 | FP8 权重 + FP8 KV | INT4 权重 | 加大 `max_model_len` |
| 低并发、延迟敏感 | spec decode | TP 到 NVLink 上限 | 大 batch 调优 |

## 4. 容量规划算例

### 输入

| 项 | 示例值 |
|---|---|
| 模型 | 70B 级 dense，bf16 权重约 140 GB，FP8 约 70 GB |
| 峰值 QPS | 20 请求/s |
| 输入长度 | P50 1500 token，P99 6000 token，其中 1200 token 为共享 system prompt |
| 输出长度 | P50 300 token，P99 1000 token |
| SLO | TTFT P99 < 1 s，TPOT P99 < 60 ms |
| 硬件 | H100 80GB × 8 一台，TP=4 一实例，每台两实例 |

### 计算步骤

1. **单实例 batch 上限**：先做 batch 扫描（固定 input 1500 / output 300），得到 TPOT P99 vs 并发曲线，假设与 60 ms 的交点在并发 96。这是 `max_num_seqs` 的上限。
2. **KV 需求**：每请求平均驻留 token 数约 输入 P50 + 输出 P50 的一半（decode 期间逐步增长）≈ 1650；并发 96 时约 158k token 的 KV。按模型 KV 每 token 大小（GQA 70B 级约 0.3 MB bf16，TP=4 后每卡 1/4）换算成每卡 KV 显存，与启动日志 `# GPU blocks` 对照，确认 FP8 权重 + bf16 KV 能放下；放不下就 KV FP8 或降并发。
3. **单实例可承载 QPS**：并发 96 ÷ 平均请求驻留时间。驻留时间 ≈ TTFT + 输出 P50 × TPOT ≈ 0.5 s + 300 × 0.04 s = 12.5 s（TPOT 取交点下方的典型值），得到约 7.7 请求/s。
4. **实例数**：20 ÷ 7.7 ≈ 2.6，向上取 3，再加扩容 buffer（扩容 5 分钟、流量 5 分钟内可能涨 30%）取 4 个实例，即 2 台 8 卡机。
5. **prefill 检查**：6000 token 的 P99 输入在 TP=4 的 H100 上 prefill 约几百毫秒，加上排队要在 1 s 内，`max_num_batched_tokens` 不能太小；prefix cache 命中 1200 token 后 P99 输入实际只算 4800。
6. **成本**：4 实例 × 4 卡 = 16 卡。峰值 goodput = 20 × 300 = 6000 output tokens/s，换算每百万 output token 的 GPU 小时 = 16 ÷ (6000 × 3600) × 1e6 ≈ 0.74 GPU 小时，乘单价即成本。谷值时段按曲线缩到 2 实例。

### 表格模板

| 输入 | 值 | 输出 | 值 |
|---|---|---|---|
| 峰值 QPS | | batch 上限（SLO 交点） | |
| 输入长度 P50 / P99 | | 单实例并发 → KV 需求 | |
| 输出长度 P50 / P99 | | 单实例 QPS | |
| 共享前缀长度与命中率 | | 实例数（含 buffer） | |
| TTFT / TPOT SLO | | 每百万 token GPU 小时 | |
| 硬件与并行配置 | | 谷值副本数 | |

## 5. 要采的成本指标

| 指标 | 来源 | 用途 |
|---|---|---|
| output tokens per GPU-hour | `vllm:generation_tokens_total` 速率 ÷ GPU 数 | 核心效率 |
| goodput ratio | 满足 SLO 的请求 ÷ 总请求（从 TTFT/TPOT 直方图算） | 判断 batch 是否过头 |
| prefill 时间占比 | `vllm:request_prefill_time_seconds` ÷ 总 | 判断 workload 类型 |
| prefix cache 命中率 | `vllm:prefix_cache_hits` ÷ `vllm:prefix_cache_queries` | prefill 成本杠杆 |
| KV 使用率分布 | `vllm:kv_cache_usage_perc` | 显存是否花在 KV 上 |
| 抢占次数 | `vllm:num_preemptions_total` | 过载导致的重算浪费 |
| 各实例 running 数时间序列 | `vllm:num_requests_running` | 空转 replica |
| 实例 Ready 时间 | K8s 事件 | buffer 大小依据 |

常见浪费清单：`max_model_len` 远大于实际需求、生产开着 `logprobs`、`--enforce-eager`、TP 大于放下模型所需、低流量模型独占实例、按峰值常驻不缩容、prefix cache 因路由分散而命中率低。

## 6. 关联

- [[01-diagnosis-playbook]]：从整体症状进入这一页的路径。
- [[04-throughput-low]]：吞吐问题是成本问题的上游。
- [[10-quantization-choice]]、[[11-spec-decode-no-gain]]、[[12-moe-serving]]：各自改变成本结构的方式。
- [[13-parallelism-choice]]：TP 过大是最常见的成本浪费之一。
- [[14-cold-start-autoscaling]]：buffer 与扩容时间的关系。
- 日课：[[day21-system-design-v1]]、[[day24-disaggregated]]、[[day26-roofline]]、[[day29-final-design]]。
