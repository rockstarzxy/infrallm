---
title: "Day 14: Week 2 综合调优 Playbook"
type: concept
tags: [day14, tuning, playbook, decision-table, week2-review]
created: 2026-06-01
updated: 2026-06-01
---

# Day 14: Week 2 综合调优 Playbook

## Part 1: 优化技术决策表

学了一周的优化技术后，最重要的产出是一张 **"什么场景该开/不该开某个优化"** 的决策表。

### 总决策表

| 优化技术 | 适合场景 | 不适合场景 | 前提条件 | 核心收益 |
|---|---|---|---|---|
| **Prefix Caching** | 大量请求共享前缀（固定 system prompt、RAG） | 每个请求完全不同 | 版本相关：可能默认开启，也可显式配置 | 降低 TTFT |
| **Chunked Prefill** | 长短 prompt 混合 workload | 全是短 prompt | vLLM V1 支持时通常默认启用；重点调 `max_num_batched_tokens` | 稳定 TPOT |
| **FP8 量化** | H100+，需要提升吞吐 | 非 H100 硬件 | `--quantization fp8` | 吞吐 ~2x |
| **AWQ/GPTQ 4-bit** | 显存受限，需要在小 GPU 上跑大模型 | 对质量极度敏感 | 预量化模型 | 显存 ~4x 节省 |
| **KV Cache FP8** | 需要更多并发 | 对长序列质量敏感 | `--kv-cache-dtype fp8` | KV Cache 减半 |
| **Speculative Decoding** | 低并发 + 长输出 + 低 temperature | 高并发在线 serving | draft model | 降低 TPOT |
| **增大 max_num_seqs** | 高并发场景 | TPOT 已过高 | 足够的 KV Cache | 提升 throughput |
| **减小 max_model_len** | 业务不需要长上下文 | 需要长上下文 | — | 增加并发空间 |

### 场景化配方

**场景 A: 在线 Chatbot（大量短对话）**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --max-num-seqs 128 \
  --enable-prefix-caching       # 共享 system prompt
```

**场景 B: RAG 应用（长 context + 短回答）**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096 # 控制长 prefill 与 decode 的混排粒度
```

**场景 C: 批量代码生成（离线，长输出）**
```bash
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
  --quantization awq \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --speculative-model Qwen/Qwen2.5-0.5B-Instruct \
  --num-speculative-tokens 5
```

**场景 D: 高吞吐 API 服务（成本优先）**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192
```

---

## Part 2: 调优流程方法论

### 步骤 1: 确定目标

```
需要回答的问题:
1. SLA 是什么？
   - TTFT P99 < ___ms
   - TPOT P99 < ___ms
   - 或者只关心 throughput（离线场景）

2. Workload 特征？
   - input 长度分布
   - output 长度分布
   - 是否有共享前缀
   - QPS 预期

3. 硬件约束？
   - GPU 型号/数量
   - 显存大小
```

### 步骤 2: 建立 Baseline

```bash
# 最简配置启动
vllm serve <model> --gpu-memory-utilization 0.9

# 跑 benchmark
python benchmarks/benchmark_serving.py \
  --backend vllm --model <model> \
  --endpoint /v1/completions --dataset-name random \
  --random-input-len <your_avg_input> \
  --random-output-len <your_avg_output> \
  --num-prompts 500 --request-rate <your_target_qps>

# 记录 baseline 数据
```

### 步骤 3: 逐一尝试优化

每次只开一个新优化，测量效果：

```
Baseline → +prefix_caching → +chunked_prefill → +fp8 → +调整 max_num_seqs → ...

每步记录：TTFT P50/P99, TPOT P50/P99, throughput, GPU util, KV cache usage
```

### 步骤 4: 组合最优配置

将有效的优化组合，测试是否有冲突。

### 步骤 5: 压力测试

用 2x 预期 QPS 做压力测试，确认系统在过载时的表现（是优雅降级还是崩溃）。

---

## Part 3: Week 2 知识串联

```
Day 8:  vLLM 架构
        → Engine → Scheduler → Block Manager → Worker → Model Runner
        → 理解代码是如何组织和运行的

Day 9:  Scheduler 深入
        → WAITING/RUNNING/SWAPPED 三队列
        → Preemption 机制（swap / recompute）
        → Chunked prefill 解决 prefill 阻塞 decode

Day 10: KV Cache 优化
        → Prefix caching: 复用共享前缀
        → Chunked prefill: 长 prefill 分片
        → KV 量化: FP8 减半 cache 大小
        → Eviction: StreamingLLM / H2O

Day 11: Quantization
        → GPTQ (二阶优化) vs AWQ (activation-aware) vs SmoothQuant (W8A8)
        → FP8: H100 原生支持，最佳性价比
        → 训练推理一致性: tokenizer / precision / RoPE

Day 12: Speculative Decoding
        → Draft model + verification (无损)
        → Acceptance rate 决定收益
        → 适合低并发 + 长输出 + 低 temperature

Day 13: FlashAttention
        → IO-aware tiling: 不写 N×N 到 HBM
        → Online softmax: 分块计算全局 softmax
        → FlashDecoding: decode 阶段的并行优化
```

---

## Part 4: 常见错误和陷阱

### 陷阱 1: 一次开启太多优化

"同时开 prefix caching + chunked prefill + speculative decoding + FP8，然后发现性能没提升，不知道哪个有问题。"

→ **一次只开一个，逐步叠加。**

### 陷阱 2: 用错误的 workload 测试

"用全是短 prompt 的 workload 测试 chunked prefill → 没效果 → 结论：chunked prefill 没用。"

→ **测试时的 workload 必须匹配生产 workload。**

### 陷阱 3: 只看平均值不看分位数

"平均 TTFT 30ms，很好！但 P99 是 3000ms..."

→ **永远看 P99。**

### 陷阱 4: 忽略量化对质量的影响

"4-bit 量化后吞吐翻倍！但没测过质量..."

→ **量化后必须做质量回归测试。**

### 陷阱 5: 在错误的场景用 speculative decoding

"高并发在线 serving 开了 speculative decoding → 吞吐反而降了。"

→ **高并发时 GPU 已经满载，draft model 只会增加开销。**

---

## Part 5: 面试高频问题

**Q: 如果给你一个 Qwen2.5-72B 的推理任务，预算 8 张 A100，你怎么部署？**

```
分析：
- 72B FP16 = ~144 GB → 至少 2 张 A100 (80GB each)
- 推荐 TP=4: 144/4 = 36 GB/卡，留 ~44 GB 给 KV Cache

配置：
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192

剩余 4 张卡：部署第二个实例，用 load balancer 分流
→ 2 个 TP=4 实例 + 前端路由

如果用 AWQ 4-bit:
- 72B INT4 = ~40 GB → TP=1 就能放下（但紧张）
- TP=2 更稳妥: 20 GB/卡，60 GB KV Cache
- 可以在 8 卡上部署 4 个实例 → 4x throughput
```

**Q: TTFT P99 突然从 200ms 涨到 2s，怎么排查？**

```
1. QPS 变了吗？→ 看监控
2. prompt 变长了吗？→ 看 input token 分布
3. KV Cache 够用吗？→ gpu_cache_usage_perc, preemptions
4. GPU 正常吗？→ 温度、频率、ECC 错误
5. 是否有配置变更？→ 检查最近的部署
```

---

## 交付物

| 文件 | 描述 |
|---|---|
| `vLLM tuning playbook.md` | 完整的优化技术决策表 + 场景化配方 + 调优方法论 |

## Week 3 预览

下周进入分布式推理和部署：
- Day 15: GPU 拓扑和通信基础
- Day 16: Tensor Parallelism
- Day 17: Pipeline Parallelism 和多节点
- Day 18: Data Parallelism、多实例和路由
- Day 19: Expert Parallelism 和 MoE 推理
- Day 20: 生产部署（K8s、监控、扩缩容）
- Day 21: 系统设计演练 v1
