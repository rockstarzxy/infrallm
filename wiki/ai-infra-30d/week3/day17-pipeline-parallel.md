---
title: "Day 17: Pipeline Parallelism 和多节点"
type: concept
tags: [day17, pipeline-parallel, multi-node, ray, bubble]
created: 2026-06-01
updated: 2026-06-01
---

# Day 17: Pipeline Parallelism 和多节点

## Part 1: Pipeline Parallelism 原理

PP 按**层**切分模型，不同的层放在不同 GPU 上。

```
TP: 切矩阵（每个 GPU 算每一层的一部分）
PP: 切层（每个 GPU 算一部分层）

PP=4, 80 层模型:
  GPU 0: Layer 0-19
  GPU 1: Layer 20-39
  GPU 2: Layer 40-59
  GPU 3: Layer 60-79
```

### 通信模式

PP 只需要相邻 stage 之间的 **P2P 通信**（点对点）：

```
GPU 0 (layers 0-19) ──→ activation [batch, seq_len, hidden_dim] ──→ GPU 1 (layers 20-39)
                                                                      ──→ ...

通信量 = hidden_dim × seq_len × batch × bytes_per_element
       = 4096 × 2048 × 1 × 2 = 16 MB (一次 P2P)

vs TP 的 AllReduce:
  PP 的通信量和通信频率都更低
  → 适合跨节点（InfiniBand）
```

### Pipeline Bubble

PP 的主要开销不是通信带宽，而是 **pipeline bubble**（流水线空转）。

```
Naive PP (无流水线):

时间 →
GPU 0: [██ Layer 0-19 ██] ............ [██ Layer 0-19 ██] ............
GPU 1: ............ [██ Layer 20-39 ██] ............ [██ Layer 20-39 ██]
GPU 2: ........................ [██ Layer 40-59 ██] ........................
GPU 3: .................................. [██ Layer 60-79 ██] ................

→ 大部分时间只有 1 个 GPU 在工作，其他 3 个空转
→ GPU 利用率 = 1/PP = 25% （PP=4时）
```

### Microbatching 减少 Bubble

将一个 batch 拆成多个 microbatch，形成流水线：

```
Pipeline PP=4, 4 microbatches:

时间 →
GPU 0: [mb1] [mb2] [mb3] [mb4]
GPU 1:       [mb1] [mb2] [mb3] [mb4]
GPU 2:             [mb1] [mb2] [mb3]  [mb4]
GPU 3:                   [mb1] [mb2]  [mb3]  [mb4]

Bubble = startup + drain = 2 × (PP-1) 个 microbatch 时间
利用率 ≈ num_microbatches / (num_microbatches + PP - 1)
         = 4 / (4 + 3) = 57%

如果 16 个 microbatch: 16 / (16 + 3) = 84%
```

microbatch 越多 → bubble 越小 → 但 latency 越高。

---

## Part 2: PP 在推理中的使用

### 推理 vs 训练的 PP 差异

训练中 PP 配合 microbatching 减少 bubble，很成熟。但推理场景不太一样：

```
推理的 decode 阶段:
  每步只生成 1 个 token → 没有"大 batch 可以切成 microbatch"
  → pipeline bubble 问题更严重
  → 每步的 latency = PP 个 stage 串行执行

推理的 prefill 阶段:
  有大的 input → 可以做 microbatching
  → bubble 可以被缓解
```

### 什么时候用 PP

```
必须用 PP 的场景:
  模型太大，即使 TP=8（单节点最大 GPU 数）也放不下
  → 需要跨节点，跨节点不适合 TP → 用 PP

例：Qwen2.5-72B FP16 = ~144 GB
  单节点 8×A100 80GB: TP=2 就够 (144/2 = 72 GB/GPU)
  但如果显存只有 40GB: TP=4 才能放下 (144/4 = 36 GB/GPU)
  如果只有 4×24GB GPU: 144/4 = 36 GB → 放不下 → 需要更多 GPU 或量化

例：DeepSeek-V3 671B FP16 = ~1.3 TB
  单节点 8×H100 80GB = 640 GB → 不够
  → 至少 2 节点，节点内 TP=8，节点间 PP=2
```

---

## Part 3: TP + PP 组合

大模型推理的典型并行策略：

```
2 节点 × 8 GPU/节点:

方案 A: TP=8, PP=2
  节点 1: 8 GPU 做 TP (layers 0-39, 每个 GPU 存 1/8 参数)
  节点 2: 8 GPU 做 TP (layers 40-79, 每个 GPU 存 1/8 参数)
  节点间: PP 通信 (P2P, 每步 1 次)

方案 B: TP=4, PP=2, 每节点 2 个 TP 组
  节点 1: GPU 0-3 做 TP 组 1 (layers 0-39)
          GPU 4-7 做 TP 组 2 (layers 0-39) ← 这是 DP
  节点 2: GPU 0-3 做 TP 组 1 (layers 40-79)
          GPU 4-7 做 TP 组 2 (layers 40-79)
  → 2 个独立的推理流水线，每个 TP=4, PP=2
  → 更高吞吐（2x），但每个流水线的延迟和方案 A 相同
```

### 选择原则

| 目标 | 策略 |
|---|---|
| 最低延迟 | 最大 TP，不用 PP |
| 最高吞吐 | 适度 TP + DP（多实例） |
| 模型放不下 | 增加 TP 或 PP |
| 跨节点 | 节点内 TP + 节点间 PP |

---

## Part 4: 在 vLLM 中使用 PP

```bash
# 单节点 PP
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2

# TP=4 × PP=2 = 使用 8 个 GPU
```

### 多节点部署

vLLM 多节点常用 Ray 编排，也可以使用 multiprocessing 形式的多节点启动。Ray 是常见选择，但不是唯一选择。

```bash
# 节点 1 (head node):
ray start --head --port=6379

# 节点 2:
ray start --address='<head_node_ip>:6379'

# 在 head node 上启动 vLLM:
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2
# vLLM 通过 Ray runtime 获取集群资源并分配 worker
```

---

## Part 5: 多节点部署的实际挑战

### 模型权重分发

```
问题: 1.3 TB 的模型权重需要到达每个节点
方案:
  1. 共享存储 (NFS/Lustre): 所有节点读同一份 → 网络瓶颈
  2. 预分发: 部署前将权重复制到每个节点本地 SSD → 启动更快
  3. 模型分片: 每个节点只下载自己需要的层 → 最优但需要工具支持
```

### 网络配置

```
必须确保:
  - GPU Direct RDMA 开启（允许 GPU 直接通过 IB 读写远端 GPU 显存）
  - NCCL 环境变量正确配置:
    NCCL_IB_DISABLE=0
    NCCL_NET_GDR_LEVEL=SYS
    NCCL_IB_HCA=mlx5_0,mlx5_1,...
```

### 容错

```
问题: 16 GPU 部署中一个 GPU 故障 → 整个推理管线不可用

方案:
  1. 冗余实例: 部署 2 套，故障时切换
  2. 快速重启: 检测到故障后自动重启受影响的 worker
  3. 模型降级: 故障时自动切换到小模型
```

---

## Part 6: 中国大模型的部署方案

### Qwen2.5-72B

```
FP16 (144 GB):
  最小配置: 2×A100-80GB, TP=2
  推荐:     4×A100-80GB, TP=4 (更多 KV Cache 空间)
  高吞吐:   8×A100-80GB, TP=4, DP=2 (2 个实例)

AWQ 4-bit (~40 GB):
  最小配置: 1×A100-80GB, TP=1
  推荐:     2×A100-80GB, TP=2
  高吞吐:   4×A100-80GB, TP=1, DP=4 (4 个实例)
```

### DeepSeek-V3 (671B MoE, 37B active)

```
FP16 (~1.3 TB 总参数):
  最小配置: 4 节点 × 8×H100 = 32 GPU
  推荐:     节点内 TP=8, 节点间 EP + PP
  特殊考量: expert 数量 = 256, 需要精心设计 EP 方案

FP8:
  ~650 GB → 2 节点 × 8×H100 可能勉强
```

### MiniMax-Text-01 (456B MoE)

```
类似 DeepSeek 的 MoE 部署思路
额外: Lightning Attention 改变长上下文 attention/KV 状态的成本结构
→ 不能直接套用标准 Transformer 的 KV Cache 估算，需看具体框架实现
```

---

## 交付物

| 文件 | 描述 |
|---|---|
| `pipeline-parallel-and-multinode-design.md` | PP 原理、TP+PP 组合策略、多节点部署方案 |

## 自检问题

1. PP 的 pipeline bubble 怎么计算？如何减少？
2. 推理中 PP 的 bubble 为什么比训练更严重？
3. 什么时候选 TP=8 PP=1 vs TP=4 PP=2？
4. 多节点部署中模型权重怎么分发？
5. DeepSeek-V3 为什么不能简单地用 TP+PP 部署？
