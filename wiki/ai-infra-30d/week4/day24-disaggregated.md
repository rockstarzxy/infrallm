---
title: "Day 24: Prefill-Decode 解耦"
type: concept
tags: [day24, disaggregated, prefill-decode, distserve, splitwise]
created: 2026-06-01
updated: 2026-06-01
---

# Day 24: Prefill-Decode 解耦

## Part 1: 为什么要解耦

回顾 Day 1 的核心洞察：

```
Prefill:  compute-bound → 需要算力 (TFLOPS)
Decode:   memory-bandwidth-bound → 需要带宽 (TB/s)
```

当 prefill 和 decode 在同一组 GPU 上运行时：

```
问题 1: 资源冲突
  prefill 希望 GPU 做大矩阵乘法（compute-bound）
  decode 希望 GPU 快速读取权重（bandwidth-bound）
  → 同时做两者，两边都不是最优

问题 2: 调度干扰
  长 prefill 会阻塞 decode 请求的下一步
  → TPOT 出现尖峰（Day 9 讲过）
  → chunked prefill 缓解但不消除

问题 3: 硬件不匹配
  prefill: 用高算力 GPU 更好（如 H100 Tensor Core 利用率高）
  decode: 用高带宽 GPU 更好（如 H100 HBM 带宽更关键）
  → 同一 GPU 的 prefill 和 decode 利用率都不是最优
```

解耦的思路：**用不同的 GPU 组分别处理 prefill 和 decode。**

---

## Part 2: 解耦架构

### 基本架构

```
                    ┌──────────────────┐
  Request ────────→ │   Router/LB      │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ↓                             ↓
    ┌───────────────────┐        ┌───────────────────┐
    │  Prefill Workers  │        │  Decode Workers   │
    │  (计算密集型)      │        │  (带宽密集型)      │
    │                   │        │                   │
    │  GPU 0-3          │        │  GPU 4-7          │
    │  高 SM utilization │        │  高 HBM bandwidth  │
    └────────┬──────────┘        └────────┬──────────┘
             │                            ↑
             │    KV Cache Transfer       │
             └────────────────────────────┘
```

### 请求流程

```
1. 请求到达 Router
2. Router 发给 Prefill Worker:
   - Prefill Worker 处理完整 prompt
   - 生成 KV Cache
   
3. KV Cache Transfer:
   - Prefill Worker 将 KV Cache 传输给 Decode Worker
   - 通过 NVLink / PCIe / InfiniBand
   
4. Decode Worker 接管:
   - 用传过来的 KV Cache 做 decode
   - 逐 token 生成直到完成

5. Decode Worker 将结果返回
```

### KV Cache 传输的挑战

```
传输量 = KV Cache 大小 = 2 × layers × kv_heads × head_dim × seq_len × bytes

Qwen2.5-72B, input_len=2000, FP16:
  KV/token = 320 KB
  传输量 = 2000 × 320 KB = 640 MB

通过 NVLink (900 GB/s): 640 MB / 900 GB/s = 0.7 ms → 几乎免费
通过 PCIe (64 GB/s):    640 MB / 64 GB/s = 10 ms → 可接受
通过 IB NDR (50 GB/s):  640 MB / 50 GB/s = 13 ms → 可接受但有开销
通过 IB HDR (25 GB/s):  640 MB / 25 GB/s = 26 ms → 开始有压力

对于长上下文 (input_len=32K):
  传输量 = 32000 × 320 KB = 10 GB
  通过 IB NDR: 10 GB / 50 GB/s = 200 ms → 非常大的开销！
```

结论：**解耦方案在长上下文场景下的 KV 传输开销是核心瓶颈。**

---

## Part 3: DistServe 和 Splitwise

### DistServe

```
核心论点:
  prefill 和 decode 有不同的 SLO（Service Level Objective）:
  - prefill SLO: TTFT < X ms
  - decode SLO: TPOT < Y ms

  如果放在一起，两个 SLO 互相干扰
  分开后可以独立优化

DistServe 的策略:
  - Prefill cluster: 优化 TTFT → 更大 TP → 更快 prefill
  - Decode cluster: 优化 TPOT → 更大 batch → 更高 throughput
  - Goodput-driven 资源分配: 根据 workload 动态调整 prefill/decode 的 GPU 比例
```

### Splitwise

```
核心论点:
  prefill 和 decode 的最优硬件配置不同

Splitwise 的观察:
  - Prefill 适合高算力 GPU（compute-bound）
  - Decode 适合高带宽/低成本 GPU（bandwidth-bound）

  可能的异构配置:
    Prefill: 少量 H100 (高算力)
    Decode:  多量 A100/L40S (较低成本但带宽够)

  甚至更极端:
    Decode 在较弱的 GPU 上（如 L4/T4），用量化进一步降低带宽需求
```

---

## Part 4: vLLM 和 TensorRT-LLM 的解耦支持

### vLLM Disaggregated Prefilling

```
vLLM 目前的支持（实验性质）:
  - 通过 connector 机制将 prefill 和 decode 分离
  - prefill worker 完成后通过 NCCL/socket 传输 KV Cache
  - decode worker 接管后续生成

配置:
  # 需要特殊配置，参考 vLLM 文档中的 disaggregated prefilling 章节
  # 目前还在快速迭代中

限制:
  - KV Cache 传输格式和效率还在优化
  - 不是所有模型都支持
  - 多节点场景的 KV 传输性能取决于网络
```

### TensorRT-LLM Disaggregated Serving

```
TRT-LLM 的支持更成熟:
  - 官方文档中有专门的 disaggregated serving 章节
  - 使用 NVIDIA NIXLA 或类似机制做高效 KV 传输
  - 支持 NVLink/IB 两种传输方式

优势:
  - NVIDIA 端到端优化了 KV 传输路径
  - 和 TRT engine 深度集成
  - 可以利用 GPU Direct RDMA 减少 CPU 开销
```

---

## Part 5: 解耦的适用场景分析

### 适合解耦的场景

```
1. 大规模在线 serving (高 QPS):
   - prefill 和 decode 的资源需求差异大
   - 解耦后可以独立扩缩容
   - 例: prefill QPS 突增时只扩 prefill workers

2. 严格的 SLO 要求:
   - TTFT 和 TPOT 都有严格 P99 要求
   - 解耦后两者互不干扰

3. 异构硬件环境:
   - 有不同代/型号的 GPU
   - 高算力 GPU 给 prefill，高带宽/低成本 GPU 给 decode
```

### 不适合解耦的场景

```
1. 小规模部署 (少量 GPU):
   - GPU 数量不够分成两组
   - 解耦的管理复杂度不值得

2. 长上下文场景:
   - KV Cache 传输量太大
   - 传输时间可能超过 prefill 本身
   - 除非有 NVLink 直连

3. chunked prefill 已经够好:
   - 如果 TPOT 稳定性要求不是极致
   - chunked prefill 已经能解决 prefill 阻塞问题
   
4. 低 QPS:
   - GPU 本身不饱和
   - 分开反而降低每组 GPU 的利用率
```

### 判断是否需要解耦

```
需要回答:
1. TPOT P99 是否有明显尖峰？
   → 是 → 先试 chunked prefill → 不够再考虑解耦

2. prefill 和 decode 的 GPU 利用率差异大吗？
   → prefill 时 GPU util 100%，decode 时 GPU util 30%
   → 差异大 → 解耦可能有收益

3. 有异构 GPU 需要利用吗？
   → 是 → 解耦天然适合

4. KV Cache 传输代价能接受吗？
   → 估算传输时间 vs prefill/decode 时间
   → 传输时间 < 10% → 值得尝试
```

---

## Part 6: DeepSeek-V3 的解耦优势

```
DeepSeek-V3 使用 MLA，KV Cache 极其紧凑:

MLA latent 维度 = 512 (vs 标准 KV dim = 128 × 8 = 1024)
KV Cache/token ≈ 2 × 80 layers × 512 × 2 bytes = 160 KB (MLA)
vs 标准:  2 × 80 × 8 × 128 × 2 = 320 KB

→ KV 传输量减半
→ 更适合 prefill-decode 解耦
→ 长上下文场景的传输瓶颈也更小
```

---

## 交付物

| 文件 | 描述 |
|---|---|
| `prefill-decode-disaggregation.md` | 解耦原理 + 传输代价分析 + 适用场景判断 |

## 自检问题

1. Prefill 和 decode 分别是什么 bound？为什么分开可能更好？
2. KV Cache 传输量怎么估算？在什么互联条件下传输代价可接受？
3. 解耦和 chunked prefill 各解决什么层面的问题？
4. 什么时候不应该解耦？
5. DeepSeek-V3 的 MLA 为什么对解耦方案特别友好？
