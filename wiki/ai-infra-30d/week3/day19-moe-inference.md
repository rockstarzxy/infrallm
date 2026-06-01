---
title: "Day 19: Expert Parallelism 和 MoE 推理"
type: concept
tags: [day19, moe, expert-parallel, deepseek, minimax, all-to-all]
created: 2026-06-01
updated: 2026-06-01
---

# Day 19: Expert Parallelism 和 MoE 推理

## Part 1: MoE 架构回顾

Mixture of Experts (MoE) 将 FFN 层替换为多个独立的 "expert" 网络 + 一个 router：

```
Dense Model (标准):
  FFN: input → gate_proj + up_proj → SiLU → down_proj → output
  每个 token 都经过完整 FFN

MoE Model:
  Router: input → softmax → 选择 top-k experts
  Expert 0: input → FFN_0 → output_0
  Expert 1: input → FFN_1 → output_1
  ...
  Expert N: input → FFN_N → output_N
  
  output = sum(router_weight_i × output_i) for selected experts
```

### 核心特点

```
总参数量: 很大（所有 expert 的参数之和）
Active 参数: 很小（每个 token 只激活 top-k 个 expert）

例: DeepSeek-V3
  总参数: 671B
  Active 参数: ~37B (每 token)
  Experts: 256 个 FFN experts + 1 个 shared expert
  Top-k: 8 (每个 token 激活 8 个 expert)

例: Mixtral-8x7B
  总参数: 46.7B
  Active 参数: ~12.9B (每 token)
  Experts: 8 个
  Top-k: 2
  
例: MiniMax-Text-01
  总参数: 456B
  MoE 架构 + Lightning Attention
```

### MoE 推理的特殊性

```
1. 显存大: 即使 active 参数少，所有 expert 权重都需要在显存中
   DeepSeek-V3 FP16: ~1.3 TB → 需要大量 GPU

2. 计算小: 每 token 只用 top-k experts → 实际 FLOPS 远小于同规模 dense model
   → 推理更快但显存需求不变

3. 负载不均: 不同 expert 的被选中频率不同
   → 某些 GPU（负责热门 expert）负载更重
   → 训练时用 load balancing loss 缓解，但推理时仍可能不均

4. 动态路由: 每个 token 可能选择不同的 expert
   → 通信模式是 All-to-All（每个 token 发给不同 GPU）
```

---

## Part 2: Expert Parallelism

### EP 原理

每个 GPU 负责不同的 experts：

```
EP=8, 64 experts:
  GPU 0: Expert 0-7
  GPU 1: Expert 8-15
  GPU 2: Expert 16-23
  ...
  GPU 7: Expert 56-63

推理流程:
1. Router 决定每个 token 应该去哪个 expert
2. All-to-All: 将 token 发送到持有对应 expert 的 GPU
3. 每个 GPU 处理自己负责的 expert
4. All-to-All: 将结果发回原来的 GPU
```

### All-to-All 通信

```
All-to-All 示例（4 GPU, 每个 GPU 有 4 个 token）:

Before (tokens 按 GPU 分布):
  GPU 0: [t0→E0, t1→E3, t2→E1, t3→E2]   ← 每个 token 要去不同 expert
  GPU 1: [t4→E1, t5→E0, t6→E3, t7→E1]
  GPU 2: [t8→E2, t9→E2, t10→E0, t11→E3]
  GPU 3: [t12→E3, t13→E1, t14→E2, t15→E0]

All-to-All (按 expert 重新分配):
  GPU 0 (E0): [t0, t5, t10, t15]
  GPU 1 (E1): [t2, t4, t7, t13]
  GPU 2 (E2): [t3, t8, t9, t14]
  GPU 3 (E3): [t1, t6, t11, t12]

各 GPU 对自己的 token 执行对应 expert 计算

Reverse All-to-All (结果发回):
  GPU 0: [result_t0, result_t1, result_t2, result_t3]
  ...
```

### EP 的通信开销

```
每个 token 的通信量 = hidden_dim × bytes_per_element × 2 (去和回)

DeepSeek-V3 (hidden_dim=7168, FP16):
  每 token: 7168 × 2 × 2 = 28.7 KB
  1000 tokens: 28.7 MB

All-to-All 带宽需求:
  batch=128, seq_len=1:
  128 × 28.7 KB = 3.6 MB → 在 NVLink 下 <0.01ms
  
  但跨节点 (InfiniBand 50 GB/s):
  3.6 MB / 50 GB/s = 0.072 ms → 仍然很快

问题在于 All-to-All 的 latency 而不是带宽:
  NCCL All-to-All 的固定开销（barrier + setup）可能有数十 us
  在 decode 阶段（每步都要做 All-to-All），累积起来不可忽略
```

---

## Part 3: MoE 推理的并行策略组合

### DeepSeek-V3 的部署方案

```
架构: 
  - Attention layers: dense (标准 GQA/MLA)
  - FFN layers: MoE (256 experts + 1 shared)

并行策略:
  Attention: Tensor Parallel (TP)
  FFN MoE:   Expert Parallel (EP)

典型部署（官方推荐参考）:
  8 节点 × 8 H100:
    - 节点内: attention TP=8, MoE EP 在节点间分布
    - expert 分配: 256 experts / 64 GPU = 4 experts/GPU

通信模式:
  每个 layer:
    Attention: AllReduce (TP, 节点内 NVLink)
    Router:    broadcast routing decisions
    MoE FFN:   All-to-All (EP, 跨节点 IB)
    Combine:   All-to-All (EP, 跨节点 IB)
```

### Expert 负载均衡

```
问题: 
  如果 Expert 0 被 50% 的 token 选中，Expert 63 只被 1% 选中
  → 持有 Expert 0 的 GPU 成为瓶颈
  → 其他 GPU 等待，GPU 利用率低

缓解方案:
  1. 训练时的 load balancing loss (已在模型训练时处理)
  2. Expert capacity: 限制每个 expert 每步最多处理多少 token
     → 超出的 token 被 drop 或发给 shared expert
  3. 智能 expert 放置: 将热门 expert 放在更快的 GPU 上
```

---

## Part 4: Expert Offloading

当 GPU 显存不够放所有 expert 时，可以将不活跃的 expert offload 到 CPU/NVMe：

```
基本思路:
  - 热门 expert: 常驻 GPU 显存
  - 冷门 expert: 放在 CPU 内存，需要时 swap 到 GPU

DeepSeek-V3 有 256 个 expert:
  FP16: 256 experts × ~5 GB/expert ≈ 1.3 TB
  如果 GPU 显存只有 640 GB (8×80GB):
    放不下所有 expert → 需要 offload

Offloading 的代价:
  PCIe Gen5: 64 GB/s
  Swap 一个 5 GB expert: 5/64 = 78 ms
  → decode 阶段不可接受（每步需要 <10ms）
  
  → 需要预测下一步哪些 expert 会被用到 → prefetch
  → 或者用 expert cache（LRU）+ speculative loading
```

### 实际方案

```
方案 1: 全 GPU 内存（推荐，但贵）
  足够的 GPU 放下所有 expert
  DeepSeek-V3: 至少 8×H100 + FP8 量化

方案 2: GPU + CPU offload
  热门 expert 在 GPU，冷门在 CPU
  适合离线推理、低 QPS 场景
  llama.cpp 的 MoE offload 做得比较好

方案 3: GPU + NVMe offload
  类似方案 2 但更慢
  适合极端显存受限场景
```

---

## Part 5: 中国 MoE 模型总览

| 模型 | 总参数 | Active | Experts | Top-k | 特殊设计 |
|---|---:|---:|---:|---:|---|
| DeepSeek-V2 | 236B | 21B | 160 | 6 | MLA attention, shared expert |
| DeepSeek-V3 | 671B | 37B | 256 | 8 | MLA, auxiliary-loss-free balancing |
| MiniMax-Text-01 | 456B | ~46B | 32 | 2 | Lightning Attention (线性) |
| Qwen3-30B-A3B / 235B-A22B | 30B / 235B | 3B / 22B | — | — | Qwen3 已公开提供 dense 和 MoE 系列 |

### DeepSeek-V3 的推理优势

```
1. MLA: KV Cache 极小 → 长上下文友好
2. Active 参数少 (37B): decode 速度接近 40B dense model
3. 总参数多 (671B): 质量接近甚至超过更大的 dense model

→ 性价比极高：以 40B 级别的推理成本获得 700B 级别的质量
```

### MiniMax-Text-01 的特殊性

```
Lightning Attention:
  - 线性注意力: attention 复杂度从 O(n²) 降到 O(n)
  - 长上下文状态和计算成本结构不同于标准 softmax attention
  - 不能简单套用标准 Transformer 的 KV Cache 公式
  - 是否需要 paged/cache 管理或专用 kernel，取决于具体推理引擎实现

推理框架支持:
  - 标准框架（vLLM/SGLang）对 Lightning Attention 支持有限
  - 通常需要用 MiniMax 自己的推理引擎或社区适配
```

---

## Part 6: MoE 推理优化技巧

### Expert 计算 Fusion

```
如果同一个 GPU 上多个 expert 需要处理 token:
  不是逐个 expert 串行执行
  而是将所有 token 按 expert 分组后 batch 执行

例: GPU 0 有 Expert 0-3, 本步有 100 个 token:
  Expert 0: 30 tokens → batch GEMM
  Expert 1: 25 tokens → batch GEMM
  Expert 2: 20 tokens → batch GEMM
  Expert 3: 25 tokens → batch GEMM
```

### Communication-Computation Overlap

```
Pipeline All-to-All 和 Expert 计算:
  1. 发送第一批 token 到目标 GPU (通信)
  2. 同时计算已到达的 token (计算)
  3. 发送第二批... 
  → 通信和计算 overlap
```

---

## 交付物

| 文件 | 描述 |
|---|---|
| `moe-inference-notes.md` | MoE 推理特点、EP 通信分析、DeepSeek-V3 部署方案 |

## 自检问题

1. MoE 模型的总参数量和 active 参数量有什么区别？对推理有什么影响？
2. Expert Parallel 的核心通信原语是什么？
3. Expert 负载不均衡会导致什么问题？怎么缓解？
4. Expert offloading 在什么场景下可行？
5. DeepSeek-V3 为什么性价比高？
6. Lightning Attention 和标准 softmax attention 在推理框架支持上有什么区别？
