---
title: "Day 7: Week 1 复盘"
type: concept
tags: [day7, review, week1, interview-prep]
created: 2026-06-01
updated: 2026-06-01
---

# Day 7: Week 1 复盘

## Part 1: 知识串联

这一周学了 6 天的内容，现在把它们串成一条线：

```
LLM 推理的本质问题：
  大模型 + 自回归生成 → 计算量大、显存吃紧、延迟敏感

Day 1: 理解推理链路
  prefill (compute-bound) → decode (memory-bound)
  KV Cache 是显存大户，决定了并发能力

Day 2: vLLM 跑通
  部署 serving → 发请求 → 看日志中的 blocks/显存分配

Day 3: 指标体系
  TTFT（prefill 快不快）→ TPOT（decode 稳不稳）→ throughput（总产出高不高）
  workload 矩阵 + 并发梯度 → 找到系统拐点

Day 4: PagedAttention + Continuous Batching
  KV Cache 碎片化 → 分页管理 → 显存利用率 ~96%
  Static batching 浪费 → iteration-level 调度 → GPU 不空转

Day 5: Profiling
  nvidia-smi / metrics / profiler → 判断 compute-bound vs memory-bound
  prefill 阶段大 GEMM，decode 阶段小 GEMV

Day 6: 参数调优
  gpu_memory_utilization ↔ KV Cache 空间 ↔ 并发能力
  max_model_len ↔ 单请求上限 ↔ 并发数
  max_num_seqs ↔ batch size ↔ throughput vs latency
```

### 核心公式速查

```
KV Cache per token = 2 × layers × kv_heads × head_dim × bytes
GPU blocks = 可用 KV 显存 / (block_size × KV_per_token)
最大并发 ≈ GPU blocks / (avg_seq_len / block_size)
Decode 带宽瓶颈: model_size / HBM_bandwidth ≈ min TPOT (batch=1)
```

---

## Part 2: 线上故障排查思路

如果你是推理系统的 oncall，用户反馈"AI 回复变慢了"，怎么排查？

### Checklist: P99 TTFT 变差

```
1. 看 vllm:num_requests_waiting
   └── > 0 且持续增长？→ 系统过载
       ├── QPS 是否突增？→ 限流 / 扩容
       └── 单请求 input 是否变长？→ 检查上游是否传了更长的 prompt

2. 看 vllm:gpu_cache_usage_perc
   └── > 0.95？→ KV Cache 快满了
       ├── 有 preemption 吗？→ 有请求被驱逐重做
       ├── max_model_len 是否设太大？
       └── 是否可以用 prefix caching 减少重复？

3. 看 prefill 计算时间
   └── input 变长了？→ 考虑 chunked prefill
       └── 检查 max_num_batched_tokens 设置

4. GPU 有故障吗？
   └── nvidia-smi 看温度、ECC 错误、时钟频率
       └── GPU throttling（降频）会让一切变慢
```

### Checklist: TPOT 不稳定

```
1. ITL 方差大？
   └── 有长 prompt 的 prefill 穿插在 decode 步骤中
       └── 开启 chunked prefill

2. preemption 发生了？
   └── 请求被驱逐后重做，TPOT 出现尖峰
       └── 增大 KV Cache / 降低并发

3. CUDA Graph 是否启用？
   └── --enforce-eager 会增加 decode 每步的 CPU 开销
       └── 确认没有意外开启 eager mode
```

---

## Part 3: 模拟面试题

### 基础题

**Q1: vLLM 为什么比 naive HuggingFace generate() 快？**

A: 三个核心优势：
1. **PagedAttention**：将 KV Cache 分页管理，消除显存碎片，相同显存下能处理 2-4x 更多并发请求
2. **Continuous Batching**：iteration 级调度，请求完成即可释放资源并加入新请求，GPU 不空转
3. **工程优化**：CUDA Graph 减少 kernel launch 开销、FlashAttention 加速 attention 计算、优化的内存管理

HuggingFace generate() 是 static batching，一次只能处理一个 batch，且 KV Cache 按最大长度预分配。

**Q2: PagedAttention 解决的是什么问题？**

A: 解决 KV Cache 的**显存碎片化**问题。传统系统为每个请求预分配连续的显存块（按最大长度），导致 60-80% 的显存浪费。PagedAttention 借鉴 OS 虚拟内存分页，将 KV Cache 切成小 block 按需分配，物理上不需要连续，利用率提升到 ~96%。

**Q3: prefill 和 decode 的瓶颈为什么不同？**

A:
- Prefill：处理整个 prompt，是大 batch 的矩阵乘法 → FLOPS 密集 → compute-bound
- Decode：每步只处理 1 个 token 的矩阵-向量乘法 → 计算量小但要读全量权重 → memory-bandwidth-bound

直觉：decode 时 GPU 大部分时间在"搬数据"而不是"做计算"。

### 进阶题

**Q4: max_num_batched_tokens 调大有什么副作用？**

A: 调大意味着单次 iteration 可以包含更多 token：
- 好处：长 prompt 可以一次性完成 prefill（TTFT 更低）
- 副作用：单次 iteration 耗时更长 → 正在 decode 的请求需要等这次 iteration 完成 → TPOT 可能出现尖峰
- 如果同时开启 chunked prefill，prefill 会被分片并和 decode 混排，可以缓解这个问题

**Q5: 什么时候 prefix caching 效果好？什么时候不好？**

A:
- 效果好：大量请求共享前缀（如相同 system prompt、RAG 中的共同 context 片段）
- 效果不好：每个请求的 prompt 完全不同（无共享前缀）→ 缓存命中率低，反而浪费管理开销
- 关键指标：缓存命中率。可以从 vLLM metrics 中观察

**Q6: 如果线上 P99 TTFT 突然从 200ms 飙到 2000ms，你会怎么排查？**

A: 按层排查：
1. **是否 QPS 突增**？→ 看监控面板的请求量
2. **是否 prompt 变长**？→ 看 input token 分布的变化
3. **是否 KV Cache 不够**？→ 看 gpu_cache_usage_perc 和 preemption 数
4. **是否有 GPU 硬件问题**？→ 看温度、ECC 错误、时钟频率（降频）
5. **是否模型/配置变更**？→ 检查是否有发布
6. **是否其他进程抢占 GPU**？→ 检查 nvidia-smi 中是否有其他进程

---

## Part 4: Week 1 知识图谱

```
                    ┌──────────────────────┐
                    │  LLM Inference       │
                    │  生命周期             │
                    └──────┬───────────────┘
                           │
              ┌────────────┼────────────────┐
              ↓            ↓                ↓
        ┌──────────┐ ┌──────────┐    ┌──────────────┐
        │ Prefill  │ │  Decode  │    │   Sampling   │
        │compute-  │ │memory-   │    │ top-k/top-p  │
        │bound     │ │bandwidth │    │ temperature  │
        └────┬─────┘ │bound     │    └──────────────┘
             │       └────┬─────┘
             ↓            ↓
        ┌──────────────────────┐
        │     KV Cache         │
        │ 显存大户，决定并发     │
        └──────────┬───────────┘
                   │
         ┌─────────┼──────────┐
         ↓         ↓          ↓
    ┌────────┐ ┌────────┐ ┌────────────┐
    │ Paged  │ │  MQA/  │ │ KV Cache   │
    │Attn    │ │  GQA   │ │ Quantize   │
    │分页管理 │ │减少KV头│ │ 压缩精度   │
    └────────┘ └────────┘ └────────────┘
         │
    ┌────┴─────┐
    ↓          ↓
┌────────┐ ┌──────────────┐
│Contin. │ │  Scheduler   │
│Batching│ │ 调度策略      │
│请求随时 │ │ preemption   │
│进出batch│ │ priority     │
└────────┘ └──────────────┘
```

---

## Part 5: Week 2 预览

下周的重点是从"会用 vLLM"到"理解 vLLM 的内部机制"和"掌握核心优化技术"：

- Day 8: vLLM 架构源码阅读
- Day 9: Scheduler 深入
- Day 10: KV Cache 优化（prefix caching, chunked prefill）
- Day 11: Quantization（量化）
- Day 12: Speculative Decoding（推测解码）
- Day 13: FlashAttention
- Day 14: 综合调优 playbook

---

## 交付物

| 文件 | 描述 |
|---|---|
| `week-1-review.md` | Week 1 核心知识总结、故障排查 checklist、面试题自答 |
