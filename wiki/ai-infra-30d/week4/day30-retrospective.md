---
title: "Day 30: 面试级复盘和知识图谱"
type: concept
tags: [day30, retrospective, interview, knowledge-map]
created: 2026-06-01
updated: 2026-06-01
---

# Day 30: 面试级复盘和知识图谱

## Part 1: 30 天知识图谱

```
                        ┌─────────────────────────┐
                        │    LLM Inference         │
                        │    推理系统               │
                        └──────────┬──────────────┘
                                   │
         ┌─────────────────────────┼──────────────────────────┐
         ↓                         ↓                          ↓
┌─────────────────┐    ┌───────────────────┐     ┌──────────────────┐
│ 推理链路         │    │   核心优化         │     │  系统工程         │
│ (Week 1)        │    │   (Week 2)        │     │  (Week 3-4)      │
├─────────────────┤    ├───────────────────┤     ├──────────────────┤
│ prefill/decode  │    │ FlashAttention    │     │ TP/PP/DP/EP      │
│ KV Cache        │    │ PagedAttention    │     │ NCCL/NVLink      │
│ MQA/GQA         │    │ Quantization      │     │ 多实例/路由       │
│ Benchmark       │    │ Spec. Decoding    │     │ MoE 推理         │
│ Profiling       │    │ Prefix Caching    │     │ 生产部署         │
│ 参数调优         │    │ Chunked Prefill   │     │ 系统设计         │
└─────────────────┘    └───────────────────┘     └──────────────────┘
         │                         │                          │
         └─────────────────────────┼──────────────────────────┘
                                   ↓
                        ┌─────────────────────────┐
                        │   高阶主题 (Week 4)     │
                        ├─────────────────────────┤
                        │ TRT-LLM / SGLang        │
                        │ Prefill-Decode 解耦      │
                        │ Triton Kernel            │
                        │ Roofline 分析            │
                        │ 长上下文推理             │
                        │ 训练推理一致性           │
                        └─────────────────────────┘
```

---

## Part 2: 20 个高频面试题

### 基础层 (必答)

**1. vLLM 为什么能提升吞吐？**
PagedAttention 消除 KV Cache 碎片（利用率从 ~30% → ~96%），continuous batching 让请求在任意 iteration 进出 batch，CUDA Graph 减少 kernel launch 开销。相同显存下能处理 2-4x 更多并发请求。

**2. PagedAttention 和 OS 虚拟内存的类比是什么？**
逻辑 KV 地址 → 物理 KV block 映射（类比页表），按需分配 block（类比按需分页），Copy-on-Write（共享前缀时），Swap（GPU → CPU 交换，类比 page fault + swap）。

**3. Prefill 和 Decode 的瓶颈为什么不同？**
Prefill 处理整个 prompt，大矩阵乘法，compute-bound。Decode 每步只处理 1 个 token，矩阵向量乘法，需要读全部权重但计算量极小，memory-bandwidth-bound。

**4. TTFT 和 TPOT 分别反映什么？**
TTFT = 排队时间 + prefill 时间，反映系统负载和 prefill 效率。TPOT = decode 每步耗时，反映 decode 效率和 batch 内干扰。

**5. Continuous batching 如何比 static batching 更好？**
Static batching 中最短的请求要等最长的完成才能释放资源。Continuous batching 每个 iteration 后检查，完成的请求立即释放，新请求立即加入，GPU 不空转。

### 优化层 (重点)

**6. FlashAttention 为什么快？**
标准 attention 需要将 N×N 矩阵写到 HBM，IO 开销大。FlashAttention 通过 tiling + online softmax 在 SRAM 内完成所有计算，减少了数倍到数十倍的 HBM 读写。

**7. 量化为什么能加速推理？**
两个原因：(1) 权重更小，从 HBM 读取更快，缓解 decode 阶段的 bandwidth 瓶颈。(2) 低精度硬件（如 H100 FP8 Tensor Core）算力更高。在 bandwidth-bound 的 decode 阶段，FP8 接近 2x 加速。

**8. Prefix caching 什么时候有效？什么时候没用？**
有效：大量请求共享相同前缀（固定 system prompt、RAG 共同 context、few-shot examples）。无效：每个请求完全不同（无共享前缀），缓存命中率为 0。

**9. Speculative decoding 什么时候不应该用？**
高并发在线 serving（GPU 已饱和，draft model 增加额外开销）；高 temperature（acceptance rate 低）；短输出（加速空间小）；显存紧张（draft model 占额外显存）。

**10. Chunked prefill 解决什么问题？代价是什么？**
问题：长 prompt 的 prefill 阻塞同 iteration 内所有 decode 请求，导致 TPOT 尖峰。方案：将 prefill 分片，每个 chunk 和 decode 请求一起执行。代价：长 prompt 自身的 TTFT 略增（多步完成）。

### 分布式层 (进阶)

**11. TP 和 PP 各适合什么场景？**
TP：切矩阵，通信频繁（每层 2 次 AllReduce），需要高带宽互联 → 适合节点内（NVLink）。PP：切层，通信稀少（层间 P2P），但有 pipeline bubble → 适合跨节点（InfiniBand）。

**12. 什么时候应该开多实例(DP)而不是增大TP？**
当 TP 的通信开销导致加速比下降时。例如 TP=8 只有 6x 加速，而 2 个 TP=4 实例能给 2 × 3.5x = 7x 吞吐。如果目标是吞吐而非最低延迟，DP 通常更优。

**13. MoE 推理和 Dense 推理的关键差异？**
MoE 的总参数大（显存需求高）但 active 参数少（计算量小）。需要 Expert Parallel（All-to-All 通信），expert 负载不均可能成为瓶颈。每 token 的推理成本低于同等质量的 dense 模型。

**14. DeepSeek-V3 为什么性价比高？**
671B 总参数提供高质量，但每 token 只激活 37B 参数（MoE），推理成本接近 40B dense 模型。MLA 进一步压缩 KV Cache，长上下文更高效。

### 系统设计层 (综合)

**15. 设计一个 500 QPS 的 72B 模型推理平台，你怎么做？**
(参考 Day 21 的完整设计)

**16. TTFT P99 突然从 200ms 飙到 2s，怎么排查？**
按层排查：QPS 突增？prompt 变长？KV Cache 满（preemption）？GPU 故障/降频？配置变更？

**17. 如何做到模型更新不中断服务？**
Rolling update：启动新实例 → 等待 ready → 切流量 → 下线旧实例。灰度发布：5% → 观察指标 → 逐步放量。回滚：保留旧版本镜像，异常时快速切回。

**18. LLM serving 的 autoscaling 有什么特殊之处？**
冷启动慢（模型加载 30s-5min），GPU 资源稀缺（不能像 CPU 随时弹出），负载不均（一个长请求 = 100 个短请求），应基于 tokens/s 或队列深度而非 QPS。

**19. 训练推理一致性的关键检查点？**
Tokenizer 版本一致、BOS/EOS 处理一致、数值精度匹配（BF16/FP16）、RoPE 参数匹配、attention mask 格式一致。量化后需要在关键任务上做质量回归。

**20. Roofline 模型如何指导推理优化？**
计算操作的 AI（FLOPS/Byte），和 GPU 平衡点比较。Decode batch=1 极度 memory-bound → 量化、增大 batch 有效。Prefill 通常 compute-bound → FlashAttention 和更大 TP 有效。

---

## Part 3: 核心公式速查表

```
KV Cache/token = 2 × layers × kv_heads × head_dim × bytes_per_element
GPU blocks = KV Cache 可用显存 / (block_size × KV_per_token)
最大并发 ≈ GPU blocks × block_size / avg_seq_len
Decode 带宽瓶颈: TPOT ≥ model_size_bytes / HBM_bandwidth
AI = FLOPS / Bytes_transferred
Roofline 平衡点 = peak_FLOPS / peak_bandwidth
AllReduce 通信量 ≈ 2 × data_size × (N-1)/N
Pipeline bubble = (PP-1) / (microbatches + PP-1)
Speculative speedup ≈ avg_accepted / (1 + draft_cost/target_cost)
```

---

## Part 4: 30 天后的进阶方向

```
1. vLLM 源码贡献
   - 修一个 issue 或加一个 feature
   - 理解 CI/CD 和 review 流程

2. Nsight Systems / Nsight Compute 深度使用
   - 分析端到端推理的 GPU timeline
   - 做 roofline 分析

3. 自定义 Triton Kernel
   - 写一个 fused attention kernel
   - 对比和 FlashAttention 的性能

4. TensorRT-LLM Engine Build
   - 实操 build 一个模型
   - 对比和 vLLM 的性能

5. MoE 大规模部署
   - DeepSeek-V3 或类似模型的 multi-node 部署
   - Expert parallel 实操

6. 生产平台建设
   - K8s + GPU Operator 实操
   - 监控 + 告警 + autoscaling 全流程
   
7. 论文精读
   - 按 reading-index.md 中的论文编号顺序深入
```

---

## 交付物

| 文件 | 描述 |
|---|---|
| `AI inference infra 30 天复盘.md` | 知识图谱 + 面试题自答 + 公式速查 + 进阶规划 |
