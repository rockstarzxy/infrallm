---
title: "Day 4：数据并行、ZeRO 与 FSDP2"
type: concept
tags: [llm-training, data-parallel, zero, fsdp2, hsdp]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 4：数据并行、ZeRO 与 FSDP2

> 上一课 [[llm-training-30d/week1/day03-data-pipeline-tokenization]] · 下一课 [[llm-training-30d/week1/day05-model-parallel-tp-pp-cp-ep]]。相关：推理课 [[ai-infra-30d/week3/day15-gpu-topology]]（NCCL 集合通信与拓扑）、[[ai-infra-30d/week3/day18-data-parallel]]（推理侧 DP 只复制参数，没有梯度同步）。

## 学习目标

1. 算出 DDP、ZeRO-1/2/3 每步的通信量与每卡显存，能解释"ZeRO-3 通信量为什么只是 DDP 的 1.5 倍"。
2. 说清 FSDP2 相对 FSDP1 的变化（per-parameter sharding、DTensor），以及它和 ZeRO-3 的对应关系。
3. 配置 HSDP（节点内分片、节点间复制）并解释什么时候用它。
4. 用 FSDP2 或 TorchTitan 训练 1B 模型，测吞吐随 GPU 数的变化并解释非线性。

## 工业现状

- **FSDP2 是 PyTorch 原生路线的默认**。TorchTitan（T09）、TRL、verl 的 FSDP 后端、Tulu 3 训练脚本都用它。FSDP1 已进入维护模式。
- **DeepSpeed ZeRO-3 仍广泛用于 HF 生态**（Accelerate + DeepSpeed 配置），功能上和 FSDP2 等价，差别在生态与 offload 成熟度。
- **Megatron-Core 的数据并行是 ZeRO-1 风格的分布式优化器**（`--use-distributed-optimizer`），配合 TP/PP 使用，不切参数本身，Day 5 讲。
- **大规模训练用 HSDP**：Llama 3（T14）在节点内 FSDP 分片、节点间复制，因为跨节点 all-gather 参数太贵。
- **RL 训练框架里 FSDP 是主流训练后端**（verl 的 FSDP worker），因为它和 HF 模型代码兼容、改动小；Megatron 后端用于超大模型和 MoE。

## 核心原理

### DDP：只切数据

每卡持完整参数、梯度、优化器状态；每步 backward 后对梯度做 allreduce（ring：每卡发送和接收各 2·(n-1)/n · |grad| 字节）。显存不省，通信量约 2× 参数字节（BF16 梯度就是 2N 字节的 2 倍）。适合参数能放进单卡的模型。

### ZeRO：切状态

设 n 张卡，参数 N，混合精度每参数 16 字节（Day 2）：

| 级别 | 切什么 | 每卡显存（静态） | 每步通信量（相对 DDP） | 机制 |
|---|---|---|---|---|
| ZeRO-1 | 优化器状态（12 字节） | 4N + 12N/n | 1× | 梯度 reduce-scatter 而不是 allreduce，每卡只更新自己那片参数，再 all-gather 参数 |
| ZeRO-2 | + 梯度（2 字节） | 2N + 14N/n | 1× | 梯度在 backward 中逐层 reduce-scatter |
| ZeRO-3 | + 参数（2 字节） | 16N/n | 1.5× | forward 与 backward 各 all-gather 一次参数，backward 后 reduce-scatter 梯度 |

ZeRO-3 的 1.5× 来自：forward all-gather（1 份参数）+ backward all-gather（1 份）+ reduce-scatter 梯度（1 份，等价于 allreduce 的一半），合计 3 份对比 DDP allreduce 的 2 份。代价是通信必须和计算重叠好，否则吞吐掉得很快。

```text
ZeRO-3 一层的时序（n 卡）：
  forward:  all-gather W_l → 计算 → 释放 W_l 的非本地分片
  backward: all-gather W_l → 算 dW_l、dx → reduce-scatter dW_l → 释放
  prefetch: 在算第 l 层时提前 all-gather 第 l+1 层（forward）或 l-1 层（backward）
```

"必须理解"：ZeRO 不改变数学，每卡看到的仍是完整模型的前向反向，只是存储被分片。所以它和 DDP 一样"数据并行"，loss 曲线应完全一致。

### FSDP2

FSDP1 把一组参数 flatten 成一个大 buffer 分片；FSDP2 改为 **per-parameter sharding**，每个参数是一个 `DTensor`，在 dim-0 上按卡切分。带来的实际好处：

- 参数、梯度、优化器状态都是 DTensor，可以和 TP（也是 DTensor）自然组合。
- 可以对不同参数用不同 dtype / requires_grad（LoRA 冻结基座时不再需要 workaround）。
- 分布式 checkpoint（DCP）直接按 DTensor 存，与并行度无关。
- 显存更可预测，没有 flat buffer 的 padding。

```python
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
mesh = init_device_mesh("cuda", (world_size,))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for layer in model.model.layers:                 # 每个 Transformer 层一个分片单元
    fully_shard(layer, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
fully_shard(model, mesh=mesh, mp_policy=mp)      # 根模块（embedding、lm_head）
```

`reshard_after_forward=True` 对应 ZeRO-3（forward 后释放全参数），`False` 对应 ZeRO-2（保留到 backward，省一次 all-gather，多占显存）。`reduce_dtype=float32` 让梯度 reduce 在 FP32 做，是稳定性的常见选择。

### HSDP：两级 mesh

```python
mesh = init_device_mesh("cuda", (num_nodes, gpus_per_node), mesh_dim_names=("replicate", "shard"))
fully_shard(layer, mesh=mesh, ...)   # 在 shard 维分片，在 replicate 维做 DDP 式 allreduce
```

节点内 NVLink 上做 all-gather（带宽高），节点间只 allreduce 梯度（量小）。当模型能放进一个节点（8 卡 × 80 GB ≈ 640 GB，对应约 40B 参数全状态）时，HSDP 比全局 FSDP 吞吐高得多。超过一个节点就必须全局分片或加 TP/PP。

### 梯度累积、micro batch 与通信

FSDP 下每个 micro batch 的 backward 都会触发 reduce-scatter；开梯度累积时，可以用 `set_requires_gradient_sync(False)` 在非最后一个 micro batch 跳过同步（FSDP2 API，名字以文档为准），只在最后一步通信。代价是要累积完整梯度（ZeRO-2 语义），显存回升。这是"通信换显存"的又一个旋钮。

### CPU offload

把参数和优化器状态放主机内存，只在需要时搬到 GPU。PCIe 带宽（约 25 到 50 GB/s）远低于 HBM，通常只在"必须在少量卡上跑大模型"时用，吞吐可能掉一个数量级。DeepSpeed 的 ZeRO-Offload 与 ZeRO-Infinity 比 FSDP 的 offload 成熟。

## 实现步骤

### 1. 最小 FSDP2 训练脚本

```python
# train_fsdp2.py  用 torchrun --nproc_per_node=4 启动
import os, torch, torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM
dist.init_process_group("nccl"); rank = dist.get_rank(); torch.cuda.set_device(rank)
mesh = init_device_mesh("cuda", (dist.get_world_size(),))
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float32)
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for layer in model.model.layers: fully_shard(layer, mesh=mesh, mp_policy=mp)
fully_shard(model, mesh=mesh, mp_policy=mp)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
for step in range(50):
    x = torch.randint(0, 150000, (2, 2048), device="cuda")     # 每卡不同数据（真实训练用 DistributedSampler）
    loss = model(input_ids=x, labels=x).loss; loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # FSDP2 下对 DTensor 直接可用
    opt.step(); opt.zero_grad(set_to_none=True)
    if rank == 0 and step % 10 == 0: print(step, loss.item(), torch.cuda.max_memory_allocated()/1e9)
dist.destroy_process_group()
```

### 2. TorchTitan 路线（推荐用于 P0）

```bash
git clone https://github.com/pytorch/torchtitan && cd torchtitan && pip install -r requirements.txt
# 配置在 torchtitan/models/llama3/train_configs/*.toml：data_parallel_shard_degree、
# data_parallel_replicate_degree（HSDP）、tensor_parallel_degree、activation_checkpoint.mode 等
CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.toml ./run_train.sh
```

TorchTitan 内置 MFU 与 tokens/s 打点、DCP checkpoint、torch.compile 与 float8 开关，是理解并行组合最干净的代码库（约 1 万行）。

### 3. DeepSpeed ZeRO-3 路线（HF 生态）

```json
{"zero_optimization": {"stage": 3, "overlap_comm": true, "contiguous_gradients": true,
  "stage3_prefetch_bucket_size": "auto", "stage3_param_persistence_threshold": "auto"},
 "bf16": {"enabled": true}, "gradient_clipping": 1.0,
 "train_micro_batch_size_per_gpu": 2, "gradient_accumulation_steps": 8}
```

`accelerate launch --config_file ds.yaml train.py` 或 `TrainingArguments(deepspeed="ds.json")`。

### 4. 分布式 checkpoint

```python
import torch.distributed.checkpoint as dcp
dcp.save({"model": model.state_dict(), "optim": opt.state_dict()}, checkpoint_id="ckpt/step50")
dcp.load(state, checkpoint_id="ckpt/step50")     # 可在不同 world_size 下加载
```

导出为 HF 格式给推理引擎：先 `dcp` 加载到 rank 0 的完整 state_dict（`get_model_state_dict(..., full_state_dict=True)`），再 `save_pretrained`。RL 训练里每次权重同步本质上就是这个操作的内存版（Day 19）。

## 实验

### 实验 1：ZeRO 级别与显存、吞吐

固定：Qwen2.5-1.5B，4 卡，micro batch 2 × 2048，50 步。变量：DDP、FSDP2 `reshard_after_forward=False`（ZeRO-2）、`True`（ZeRO-3）。记录每卡峰值显存与 tokens/s。预期显存按 Day 2 公式递减；ZeRO-3 吞吐略低于 ZeRO-2，差距在 NVLink 机器上小于 10%。

### 实验 2：扩展性

固定 micro batch，GPU 数 1/2/4/8。画 tokens/s 与理想线性的差距，用 nsys 或 torch profiler 看 NCCL kernel 占比。预期 8 卡效率 85% 以上（NVLink），PCIe 机器明显更低。

### 实验 3：HSDP vs 全局 FSDP（需要 2 节点）

模型放得进单节点时对比两种 mesh 的吞吐。没有多节点就用 `mesh=(2, 4)` 在单机模拟，观察通信模式的变化（nsys 里 all-gather 的 group size）。

### 实验 4：数学等价性

DDP 与 FSDP2 用同一 seed、同一数据顺序跑 20 步，对比 loss 到小数点后 4 位。预期一致；不一致时检查 `reduce_dtype`、梯度裁剪是否在同一位置、dropout 的随机数是否按 rank 同步。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 8 卡吞吐只有 1 卡的 4 倍 | 通信未与计算重叠；PCIe 拓扑；micro batch 太小 | nsys 看 NCCL 与计算是否串行；`nvidia-smi topo -m` | 增大 micro batch；确认 prefetch 开启；节点内用 NVLink |
| 显存没有按 n 减少 | 分片单元太粗（整模型一个单元）；embedding 未分片 | 打印每个 DTensor 的 local shape | 每层 `fully_shard`；根模块也 wrap |
| NCCL timeout | 某 rank 卡住（数据长度不一致、条件分支不同） | `NCCL_DEBUG=INFO`；`TORCH_DISTRIBUTED_DEBUG=DETAIL` | 保证所有 rank 执行相同集合操作；数据用相同 max_len |
| loss 与单卡不一致 | 每卡数据相同（没用 DistributedSampler）；lr 未随 global batch 调 | 打印每 rank 的 batch hash | 正确分发数据；固定 global batch |
| checkpoint 加载后 loss 跳变 | 优化器状态未保存或 dtype 变化 | 对比 save/load 后第一步 loss | DCP 同时存 model 与 optim |
| FSDP2 + LoRA 报错 | 冻结参数与可训练参数混在一个分片单元 | 看报错的参数名 | 升级 PyTorch；FSDP2 原生支持混合 requires_grad，确认版本 |
| offload 后极慢 | PCIe 带宽瓶颈 | profiler 看 H2D/D2H 占比 | 只 offload 优化器状态；或加卡 |

## 思考题

1. ZeRO-3 通信量 1.5× 但为什么实际吞吐损失可能远超 50%？（提示：通信是否在关键路径、消息大小、all-gather 与 reduce-scatter 的带宽利用率）
2. 推理课的 vLLM DP 是复制参数、各自服务；训练的 DP 是复制数据、同步梯度。RL 训练里 rollout 引擎的 DP 和训练框架的 FSDP 分片布局不同，权重同步时要做什么？（Day 19）
3. HSDP 的 replicate 维度上为什么是 allreduce 梯度，而不是 allreduce 参数？
4. FSDP2 下 `clip_grad_norm_` 需要全局范数，它是怎么跨卡算的？

## 验收标准

- 能填出 DDP / ZeRO-1/2/3 的显存与通信量表并推导 1.5×。
- FSDP2 脚本在 4 卡上跑通，实验 1 与 2 有数据，能解释扩展性损失来源。
- 能用 DCP 存取 checkpoint 并导出 HF 格式。

## 交付物

| 文件 | 内容 |
|---|---|
| `train_fsdp2.py` | 最小 FSDP2 训练脚本，含 DCP checkpoint |
| `zero-ablation.md` | 实验 1、2 的显存与吞吐表 |
| `dp-comm-cheatsheet.md` | 一页纸：各级别切什么、通信量、适用场景、HSDP 判定 |

## 参考

- T03：ZeRO 原论文，显存与通信量分析的出处。
- T04：FSDP 论文，理解 flat-parameter 设计与 FSDP2 改动的动机。
- T09：TorchTitan，FSDP2 + TP + PP + CP 组合的参考实现。
- T14：Llama 3 的训练基础设施章节，HSDP 与故障处理的工业实践。
- B10、B11：TorchTitan 与 DeepSpeed 文档。
