---
title: "Day 15: GPU 拓扑和通信基础"
type: concept
tags: [day15, gpu-topology, nvlink, nccl, communication, pcie, infiniband]
created: 2026-06-01
updated: 2026-06-01
---

# Day 15: GPU 拓扑和通信基础

## Part 1: 为什么通信很重要

大模型超过单 GPU 显存 → 必须分布式推理 → GPU 之间需要通信。

通信带宽和延迟直接决定了分布式推理的效率。如果通信是瓶颈，增加 GPU 反而可能不提速。

```
理想情况: 8 GPU → 8x 吞吐
现实情况: 8 GPU → 4-6x 吞吐（通信开销吃掉了一部分）

通信瓶颈程度取决于:
1. 互联带宽（NVLink vs PCIe）
2. 并行策略（TP 需要多通信，PP 和 DP 需要少通信）
3. 模型大小和 batch size
```

---

## Part 2: 互联硬件

### PCIe (Peripheral Component Interconnect Express)

```
PCIe Gen4: ~32 GB/s (双向)
PCIe Gen5: ~64 GB/s (双向)

用途: CPU ↔ GPU, GPU ↔ GPU (如果没有 NVLink), GPU ↔ NIC
特点: 所有 GPU 都有 PCIe, 但带宽相对较低
```

### NVLink / NVSwitch

```
NVLink (A100):   600 GB/s (双向，单卡)
NVLink (H100):   900 GB/s (双向，单卡)
NVLink (B200):   1,800 GB/s

NVSwitch: 全连接交换机，让同一节点内所有 GPU 通过 NVLink 互联

HGX 平台:
  8 × A100/H100 通过 NVSwitch 全连接
  → 任意两个 GPU 之间都有 NVLink 直连
  → 比 PCIe 快 10-30x
```

### InfiniBand / RoCE (节点间)

```
InfiniBand HDR:   200 Gbps = ~25 GB/s
InfiniBand NDR:   400 Gbps = ~50 GB/s
RoCE:             类似带宽，基于以太网

用途: 节点之间的 GPU 通信
特点: 比节点内的 NVLink 慢 10-20x
```

### 带宽对比总览

```
SRAM (on-chip):    ~19 TB/s
HBM:               ~3.35 TB/s (H100)
NVLink:            ~900 GB/s (H100)
PCIe Gen5:         ~64 GB/s
InfiniBand NDR:    ~50 GB/s

每一层差距约 5-20x
```

---

## Part 3: GPU 拓扑

### 查看拓扑

```bash
nvidia-smi topo -m
```

输出示例（8xA100 HGX 系统）：
```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0     X    NV12  NV12  NV12  NV12  NV12  NV12  NV12
GPU1    NV12   X    NV12  NV12  NV12  NV12  NV12  NV12
GPU2    NV12  NV12   X    NV12  NV12  NV12  NV12  NV12
...

NV12 = NVLink, 12 条 link
SYS  = 通过 CPU/PCIe 连接（很慢）
NODE = 同一 NUMA node
```

### 拓扑对性能的影响

```
TP=2 时选哪两张卡？

方案 A: GPU0 + GPU1 (NVLink) → AllReduce 带宽 ~600 GB/s
方案 B: GPU0 + GPU4 (PCIe)  → AllReduce 带宽 ~32 GB/s

→ 方案 A 的 TP 通信快 ~20x

vLLM 会自动选择拓扑最优的 GPU 组合
但如果手动指定 CUDA_VISIBLE_DEVICES 可能选到不好的组合
```

---

## Part 4: NCCL 集合通信

NCCL (NVIDIA Collective Communications Library) 是 GPU 集合通信的标准库。

### 核心 Collective 操作

**AllReduce**
```
功能: 所有 GPU 各有一份数据 → 聚合（如求和）→ 结果广播到所有 GPU
用途: Tensor Parallel 中合并各部分计算结果

GPU0: [1, 2, 3]     AllReduce(SUM)     GPU0: [10, 20, 30]
GPU1: [4, 5, 6]    ──────────────→     GPU1: [10, 20, 30]
GPU2: [5, 13, 21]                      GPU2: [10, 20, 30]
```

**AllGather**
```
功能: 每个 GPU 有一部分数据 → 收集后每个 GPU 都有完整数据
用途: 收集分布在各 GPU 上的结果

GPU0: [A]     AllGather     GPU0: [A, B, C]
GPU1: [B]    ──────────→   GPU1: [A, B, C]
GPU2: [C]                  GPU2: [A, B, C]
```

**ReduceScatter**
```
功能: AllReduce 的一半——聚合后将结果分片，每个 GPU 只得到自己那份
用途: 和 AllGather 配对使用，优化通信量

GPU0: [1,2,3]  ReduceScatter  GPU0: [10]
GPU1: [4,5,6] ──────────────→ GPU1: [20]
GPU2: [5,13,21]                GPU2: [30]
```

**All-to-All**
```
功能: 每个 GPU 向每个其他 GPU 发送不同的数据
用途: MoE 模型的 Expert Parallelism

GPU0: [a0,a1,a2]  All2All  GPU0: [a0,b0,c0]
GPU1: [b0,b1,b2] ────────→ GPU1: [a1,b1,c1]
GPU2: [c0,c1,c2]           GPU2: [a2,b2,c2]
```

### 通信量计算

| 操作 | 每 GPU 通信量 | 总通信量 |
|---|---|---|
| AllReduce | 2 × data_size × (n-1)/n | ~2 × data_size |
| AllGather | data_size × (n-1)/n | ~data_size × n |
| ReduceScatter | data_size × (n-1)/n | ~data_size |
| All-to-All | data_size × (n-1)/n | ~data_size × n |

n = GPU 数量。AllReduce 的通信量不随 GPU 数量增加（ring 算法）。

---

## Part 5: 通信时间估算

### Tensor Parallel 中的 AllReduce

以 Llama-3-8B, TP=4, FP16 为例：

每个 Transformer layer 有 2 个 AllReduce 点（attention 后 + FFN 后）：
```
每次 AllReduce 的数据量 = hidden_size × batch_tokens × 2 bytes
                        = 4096 × 1 × 2 = 8 KB (decode, batch=1)
                        = 4096 × 2048 × 2 = 16 MB (prefill, seq_len=2048)

NVLink 带宽 = 600 GB/s (A100)

AllReduce 时间:
  decode (8 KB):   ~0.001 ms → 可忽略
  prefill (16 MB): ~0.03 ms → 很小

32 layers × 2 AllReduce/layer = 64 次
  decode 总通信: 64 × 0.001 ms = 0.064 ms → 可忽略
  prefill 总通信: 64 × 0.03 ms = 1.9 ms → 可忽略
```

结论：在 NVLink 连接下，TP 的通信开销很小。这就是为什么 TP 适合节点内高速互联的场景。

如果换成 PCIe (32 GB/s)：
```
decode 总通信: 64 × ~0.02 ms = 1.3 ms → 可能是问题（TPOT 只有 5ms）
```

---

## Part 6: 节点间通信挑战

### 为什么跨节点通信是瓶颈

```
节点内 (NVLink): 600-900 GB/s
节点间 (IB NDR): 50 GB/s

差距 ~12-18x

TP 需要频繁的 AllReduce → 跨节点 TP 的通信延迟不可接受
→ 这就是为什么 TP 限制在节点内，节点间用 PP 或 DP
```

### 通信拓扑设计原则

```
节点内（NVLink 连接）:
  → Tensor Parallelism (TP)
  → 高频率、小数据量、延迟敏感

节点间（InfiniBand 连接）:
  → Pipeline Parallelism (PP) — 层间 P2P
  → Data Parallelism (DP) — 独立请求，无模型通信
  → Expert Parallelism (EP) — All-to-All，对带宽有要求

典型部署:
  2 节点 × 8 GPU:  TP=8 (节点内), PP=2 (节点间)
  或:               TP=4 (半节点内), PP=2, DP=2
```

---

## Part 7: 实操

### 查看你的硬件拓扑

```bash
# GPU 拓扑
nvidia-smi topo -m

# GPU 详细信息
nvidia-smi -q

# NVLink 状态
nvidia-smi nvlink --status

# PCIe 带宽
nvidia-smi -q | grep -A 5 "PCI"
```

### 带宽测试（如果有多卡）

```bash
# 安装 NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make

# AllReduce 带宽测试
./build/all_reduce_perf -b 8 -e 128M -f 2 -g <num_gpus>

# 输出解读:
#  size(B)   time(us)  algbw(GB/s)  busbw(GB/s)
#  8         10.2      0.001        0.001
#  ...
#  134217728 580.1     231.4        404.9     ← busbw 是关键指标
```

busbw (bus bandwidth) 反映了实际可用的通信带宽。

---

## 交付物

| 文件 | 描述 |
|---|---|
| `hardware-topology-notes.md` | GPU 拓扑信息 + 通信带宽数据 + 拓扑对并行策略的指导 |

## 自检问题

1. NVLink 和 PCIe 的带宽差多少？为什么这个差距对 TP 很重要？
2. AllReduce 的通信量随 GPU 数量增加吗？
3. 为什么 TP 适合节点内而 PP 适合节点间？
4. All-to-All 在什么场景下使用？
5. 如果 nvidia-smi topo 显示两张卡之间是 SYS 而不是 NV，意味着什么？
