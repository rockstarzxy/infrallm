---
title: "Day 5：模型并行：TP、PP、CP、EP 与 Megatron-Core"
type: concept
tags: [llm-training, tensor-parallel, pipeline-parallel, context-parallel, expert-parallel, megatron]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 5：模型并行：TP、PP、CP、EP 与 Megatron-Core

> 上一课 [[llm-training-30d/week1/day04-data-parallel-zero-fsdp]] · 下一课 [[llm-training-30d/week1/day06-frameworks-efficiency-profiling]]。相关：推理课 [[ai-infra-30d/week3/day16-tensor-parallel]]、[[ai-infra-30d/week3/day17-pipeline-parallel]]、[[ai-infra-30d/week3/day19-moe-inference]]，优化专项 [[ai-infra-30d/optimization/13-parallelism-choice]]。训练侧比推理多了反向传播、优化器状态和长序列的 activation。

## 学习目标

1. 推导 TP 每层的通信次数与通信量（含 sequence parallel 变体），说清训练与推理的差异（反向多一次）。
2. 计算 PP 的 bubble 比例，解释 1F1B、interleaved、zero-bubble 各减少了什么。
3. 说清 CP 为什么是长上下文训练的必需品，Ring Attention 与 Ulysses 的通信模式差异。
4. 说清 MoE 的 EP、all-to-all 通信、负载均衡损失与 aux-loss-free 平衡。
5. 给定模型规模、序列长度、集群拓扑，写出并行组合方案并用 Megatron-Core 参数表达。

## 工业现状

公开配方里的并行组合（典型形态，细节以报告为准）：

| 配方 | 规模 | 并行 | 备注 |
|---|---|---|---|
| Llama 3 405B（T14） | dense，16K H100 | TP 8（节点内）× CP（长上下文阶段）× PP 16 × DP（FSDP） | 4D 并行；顺序 TP-CP-PP-DP 按带宽需求由内到外 |
| DeepSeek-V3（T16） | MoE 671B，2048 H800 | EP 64（跨 8 节点）× PP 16 × DP（ZeRO-1）；不用 TP | DualPipe 重叠 all-to-all；FP8 |
| Qwen3 235B-A22B（T15） | MoE | Megatron 类，EP + PP + DP | 报告未给细节 |
| TorchTitan（T09） | 参考实现 | FSDP2 + TP + PP + CP 任意组合 | 学习用最清晰 |

趋势："dense 用 TP + PP，MoE 用 EP + PP，长上下文加 CP，最外层 DP"。TP 因为通信频繁只放在节点内（NVLink）；EP 的 all-to-all 跨节点是 MoE 训练的主要瓶颈，DeepSeek 用定制通信内核和流水重叠解决。

## 核心原理

### Tensor Parallel

Megatron 风格（T02）：MLP 的第一个线性层按列切、第二个按行切；attention 按 head 切。前向每层两次 allreduce（attention 输出一次、MLP 输出一次），反向同样两次，合计每层 4 次 allreduce，每次数据量 `b × s × h × 2 字节`。

```text
前向：  x ──[col-parallel W1，各卡算 h/n 列]──▶ GeLU ──[row-parallel W2]──▶ allreduce ──▶ y
反向：  dy ──[row-parallel 反向]──▶ ──[col-parallel 反向]──▶ allreduce（对 dx）
```

**Sequence parallel（T11）**：TP 区域之外的 LayerNorm、Dropout 沿序列维切到各卡，把 allreduce 拆成 reduce-scatter + all-gather（通信量相同），换来 activation 显存按 TP 度数下降。Megatron-Core 用 `--sequence-parallel` 打开，现在几乎总是和 TP 一起用。

TP 的限制：通信量与 b·s·h 成正比，每层都通信，所以只能在 NVLink 内；TP 度数超过 KV head 数时 GQA 的 KV 被复制（推理课讲过）；TP 不减少每卡的 batch 维度，micro batch 仍需要能放下。

### Pipeline Parallel

把层切成 P 段放不同卡。朴素做法 bubble 比例 `(P-1)/P`。

| 调度 | bubble | 显存（activation） | 机制 |
|---|---|---|---|
| GPipe | (P-1)/(m+P-1)，m 为 micro batch 数 | 存 m 个 micro batch 的 activation | 全部前向再全部反向 |
| 1F1B | 同上 | 只存 P 个 | 稳态时一前向一反向交替 |
| interleaved 1F1B（Megatron） | 除以 v（每卡 v 个不连续 stage） | 略增 | 每卡持多个小 stage，通信次数增加 |
| zero-bubble（T08） | 接近 0 | 略增 | 把反向拆成对输入的梯度（B）和对权重的梯度（W），用 W 填空隙 |
| DualPipe（T16） | 接近 0 且重叠 all-to-all | 参数复制一份 | 双向流水，前向反向的通信与计算互相掩盖 |

"必须理解"：PP 的通信量小（只传层间 activation，每个 micro batch 边界一次），所以适合跨节点；代价是 bubble 和 m 必须远大于 P，这要求 global batch 足够大。这也是 RL 训练里 PP 不受欢迎的原因之一：RL 的 batch 往往不大，且序列长度差异大导致流水不均。

### Context Parallel

序列长度 s 到 32K 以上时，activation（34·s·b·h 每层）和 attention 的 s² 计算都要切。CP 把序列切到 c 张卡，每卡持 s/c 个 token 的 Q、K、V：

- **Ring Attention（T06）**：K、V 块在环上逐卡传递，每卡对每个到来的 KV 块算局部 attention 并用 online softmax 累积。通信量 `2 × s × h_kv × 2 字节` 每层每卡，能与计算重叠。因果 mask 下要做负载均衡（把序列分成 2c 块交错分配，否则后面的卡算得多）。
- **DeepSpeed-Ulysses（T07）**：在 attention 前对 Q、K、V 做 all-to-all，把"按序列切"变成"按 head 切"，attention 完整算，再 all-to-all 切回来。通信量小但 CP 度数不能超过 head 数。
- 工业里两者常组合（Megatron-Core 的 `--context-parallel-size` 是 Ring 风格，NeMo 与 DeepSpeed 支持 Ulysses），长上下文 SFT 与 RL（长 CoT 的 response 也可以到 32K）都需要它。

### Expert Parallel

MoE 层的 E 个专家分到 e 张卡，每卡 E/e 个专家。每层两次 all-to-all（dispatch token 到专家所在卡、combine 回来），通信量取决于每 token 激活的专家数 k 和 token 数。

```text
token ──router(top-k)──▶ all-to-all dispatch ──▶ 本卡专家计算 ──▶ all-to-all combine ──▶ 加权求和
```

负载均衡：router 天然会偏向少数专家，导致某些卡过载、其他卡空闲。两种办法：

- **辅助损失（T13，Switch）**：加一项鼓励各专家负载均匀的 loss，系数要调，会轻微伤害主任务。
- **aux-loss-free（T16，DeepSeek-V3）**：给每个专家一个可调偏置加在 router 分数上，按负载动态增减偏置，不进梯度。这是当前更受欢迎的做法。

此外还有容量因子（capacity factor，超过容量的 token 被 drop）、共享专家、细粒度专家（更多更小的专家）等设计，"知道即可"。EP 与 TP 的取舍：专家内部再切 TP 通信更多；DeepSeek-V3 完全不用 TP，用 EP 64 + PP 16。

### 并行组合的决策顺序

```text
1. 序列长度 > 32K？          → 加 CP，度数 = 使 activation 放得下
2. MoE？                     → EP，度数 ≤ 专家数，尽量节点内；否则跨节点并优化 all-to-all
3. 单层参数放得下但整模型放不下？ → PP（跨节点友好）；层内太大 → TP（节点内）
4. 剩下的卡                  → DP（FSDP / ZeRO-1 分布式优化器）
5. 校验：micro batch × 所有并行度的 activation 是否放得下；global batch = micro × accum × DP × (PP 的 m)
```

Megatron-Core 参数对应：`--tensor-model-parallel-size`、`--pipeline-model-parallel-size`、`--context-parallel-size`、`--expert-model-parallel-size`、`--sequence-parallel`、`--use-distributed-optimizer`；DP 度数 = world_size / (TP × PP × CP)（EP 在 Megatron 里与 DP 共享卡，细节以文档为准）。

## 实现步骤

### 1. TorchTitan 上组合并行（最快看到效果）

```toml
# llama3_8b.toml 片段
[parallelism]
data_parallel_shard_degree = -1     # 剩余卡全给 FSDP
tensor_parallel_degree = 2
pipeline_parallel_degree = 1
context_parallel_degree = 2
[activation_checkpoint]
mode = "selective"                  # "full" | "selective" | "none"
selective_ac_option = "op"          # 只重算 matmul 之外的算子
```

8 卡上 TP 2 × CP 2 × FSDP 2，观察日志里的 tokens/s、MFU 与显存。

### 2. Megatron-Core 上的 TP + PP

```bash
torchrun --nproc_per_node 8 pretrain_gpt.py \
  --tensor-model-parallel-size 4 --pipeline-model-parallel-size 2 --sequence-parallel \
  --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
  --num-layers 32 --hidden-size 4096 --num-attention-heads 32 --seq-length 4096 \
  --micro-batch-size 1 --global-batch-size 64 --bf16 \
  --recompute-activations   # selective
```

参数名随版本变化，以 `--help` 为准。Megatron 的 HF checkpoint 转换用其 `tools/checkpoint` 脚本，RL 框架（verl、slime）里 Megatron 后端的权重与 vLLM 的对接就是这一转换的在线版（Day 19）。

### 3. 手写 TP 的一层（理解用）

```python
# 列并行：每卡持 W1[:, rank*h4/n:(rank+1)*h4/n]；行并行：W2[rank*h4/n:(rank+1)*h4/n, :]
h1 = torch.nn.functional.silu(x @ W1_shard) * (x @ W3_shard)   # SwiGLU，各卡独立
y_partial = h1 @ W2_shard
dist.all_reduce(y_partial)                                      # 前向一次 allreduce
```

用 4 卡验证与单卡完整 MLP 输出一致，再用 autograd 看反向的 allreduce 在哪。

### 4. CP 的最小验证

用 Ring Attention 的开源实现（如 `ring-flash-attention` 包）或 TorchTitan 的 CP，把 s=32K 的 batch 切到 4 卡，对比单卡 s=32K（可能 OOM）与 CP 4 的显存与 loss 一致性。

## 实验

### 实验 1：TP 度数与吞吐

固定：8B 模型（或 1.5B 放大 hidden），s=4096，8 卡。变量：TP 1/2/4/8（其余给 FSDP）。记录 tokens/s、每卡显存、nsys 中 NCCL 占比。预期：TP 提高到 4 之后吞吐下降，因为通信占比上升而每卡计算变小；显存随 TP 线性下降（sequence parallel 开着时 activation 也下降）。

### 实验 2：PP 的 micro batch 数与 bubble

固定 PP 4，global batch 固定。变量 m = 4/8/16/32。记录吞吐，拟合 `(P-1)/(m+P-1)` 曲线。有条件时对比 1F1B 与 interleaved（`--num-layers-per-virtual-pipeline-stage`）。

### 实验 3：CP 对长序列的作用

s = 8K/16K/32K/64K，CP 1/2/4/8。记录能否放下、吞吐、attention 时间占比。预期 s² 项在 32K 以上主导，CP 带来接近线性的 attention 加速。

### 实验 4：MoE 负载均衡

用一个小 MoE（如 Qwen1.5-MoE-A2.7B 或自建 8 专家模型），记录训练前 200 步每个专家的 token 占比。对比：无平衡、辅助损失 0.01、aux-loss-free 偏置。预期无平衡时少数专家吃掉大部分 token，EP 卡间时间差异大。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| TP 开启后 loss 与单卡不一致 | RNG 未按 TP rank 同步（dropout）；LayerNorm 在 SP 下统计错误 | 关 dropout 复测 | Megatron 的 `model_parallel_cuda_manual_seed`；确认 SP 实现 |
| PP 吞吐远低于预期 | m 太小；stage 划分不均（embedding 和 lm_head 在首尾 stage） | 各 stage 计时 | 增大 m；手动均衡层数（`--pipeline-model-parallel-layout` 类参数） |
| CP 下因果 mask 负载不均 | 序列顺序切分 | 各卡 attention 时间 | 用 zigzag 分块（Megatron 默认已做） |
| MoE 训练 all-to-all 占 40% 以上 | EP 跨节点；专家不均 | nsys | EP 缩到节点内；平衡；DeepEP 类内核（以框架支持为准） |
| MoE loss 突然发散 | router 崩塌到单专家；容量 drop 过多 | 专家占比日志 | 平衡机制；router 用 FP32；降 lr |
| 4D 并行启动即 OOM | micro batch × CP 后仍放不下；PP 首 stage 存了 P 份 activation | 各 rank 显存 | 减 micro；full recompute；调 stage 划分 |
| 跨节点 TP | 误把 TP 放到节点间 | 拓扑与并行度对照 | TP ≤ 节点内卡数 |

## 思考题

1. 推理课里 TP 每层前向 2 次 allreduce；训练为什么是 4 次？sequence parallel 把 allreduce 拆成 reduce-scatter + all-gather 后通信量变了吗？
2. RL 训练里 response 长度差异很大（100 到 30K token），这对 PP 和 CP 各有什么影响？框架一般怎么处理（提示：按长度分桶、dynamic batching、序列打包）？
3. DeepSeek-V3 为什么不用 TP？如果它用 TP 8，EP 会变成多少？通信模式怎么变？
4. 把一个 TP 8 × PP 2 训练出来的 Megatron checkpoint 部署到 vLLM TP 4，权重需要怎样重排？

## 验收标准

- 能对 8B dense、s=32K、64 卡 H100 和 200B MoE、s=4K、512 卡两个场景各给出并行方案并算出每卡显存与主要通信开销。
- 实验 1 到 3 至少完成两个，能解释吞吐拐点。
- 能读懂 TorchTitan 或 Megatron 的并行配置并改动。

## 交付物

| 文件 | 内容 |
|---|---|
| `parallelism-plan.md` | 两个场景的并行方案与推导 |
| `tp-ablation.md` / `cp-ablation.md` | 实验数据 |
| `tp_mlp_demo.py` | 手写 TP 一层与正确性验证 |

## 参考

- T02：Megatron-LM，TP 的原始设计。
- T11：sequence parallel 与 selective recompute。
- T08：zero-bubble PP。
- T06、T07：Ring Attention 与 Ulysses，两种 CP。
- T13、T16：MoE 负载均衡的两代做法；T16 还有 DualPipe 与 EP 64 的工程细节。
- T14：Llama 3 的 4D 并行与顺序选择。
- B09、B10：Megatron-Core 与 TorchTitan 文档。
