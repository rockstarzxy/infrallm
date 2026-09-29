---
title: "Day 7：P0 项目：FSDP2 训练小模型并报告 MFU"
type: concept
tags: [llm-training, fsdp2, torchtitan, mfu, checkpoint, project]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 7：P0 项目：FSDP2 训练小模型并报告 MFU

> 上一课 [[llm-training-30d/week1/day06-frameworks-efficiency-profiling]] · 下一课 [[llm-training-30d/week2/day08-sft-engineering]]。相关：推理课的 profiling 方法 [[ai-infra-30d/week1/day05-profiling]]、roofline [[ai-infra-30d/week4/day26-roofline]]。

## 学习目标

1. 独立用 FSDP2（或 TorchTitan）把一个 0.5B 到 1.5B 的开源 checkpoint 继续训练起来，或从头训练一个 125M 模型，跑满 2 小时不崩。
2. 交出一份"显存构成 + MFU + 吞吐随并行度变化"的实测报告，每个数字都能追溯到命令和日志。
3. 验证 checkpoint 保存和 resume 后 loss 曲线严格接续，说明分布式 checkpoint 的格式和一致性条件。
4. 用 Week 1 学的公式预测显存和吞吐，再和实测对比，解释偏差来源。
5. 能回答 Week 1 面试题，为 Week 2 的 SFT 做好环境准备。

## 工业现状

一线团队的预训练与继续训练几乎都落在两条路线上：Megatron-Core / NeMo（T02、B09，NVIDIA 生态，TP/PP/CP/EP 全套，Llama 3 T14 和多数国内大模型采用 Megatron 派生系统）和 PyTorch 原生 FSDP2 + TorchTitan（T04、T09，Meta 用于 Llama 系列的复现与实验，DTensor 抽象让 TP/PP/CP 可组合）。DeepSpeed ZeRO（T03、B11）在中小团队和 RLHF 框架里仍广泛使用。

P0 选 FSDP2 的理由：它是 PyTorch 原生，代码可读，per-parameter sharding 与后面 RL 框架（verl 的 FSDP 后端，T42）直接衔接。学会 FSDP2 之后读 Megatron 配置不会陌生，反过来不成立。

工业界评价一次训练跑得好不好，看的不是 loss 而是三件事：MFU（通常 40% 到 55% 是 dense 模型的合格线，示例数字）、稳定性（loss spike 次数、重启次数）、可恢复性（checkpoint 频率与 resume 时间）。P0 就按这三件事验收。

## 核心原理

### 显存预算（回顾 Day 2）

```
每参数字节数（BF16 训练，AdamW，混合精度主权重 FP32）：
  权重 BF16        2
  梯度 BF16        2
  主权重 FP32      4
  Adam m FP32      4
  Adam v FP32      4
  合计            16 字节 / 参数

1.5B 模型：1.5e9 × 16 = 24 GB（不含 activation）
FSDP2 ZeRO-3 风格切 N 卡：每卡 24 / N GB + activation + 临时 all-gather 的一层全量权重
```

activation 的量级（无 recompute，示例）：

```
per layer ≈ s × b × h × (34 + 5 × a × s / h) 字节   （T11 的估算式，BF16）
s 序列长度，b micro-batch，h hidden，a 头数
```

selective recompute 把括号里第二项（attention score 相关）去掉，full recompute 只留输入。

### MFU

```
每 token 训练 FLOPs ≈ 6 × N_params（前向 2N，反向 4N）
                    + attention 项 12 × L × h × s（长序列时不可忽略）
MFU = 实测 tokens/s × 每 token FLOPs ÷ (GPU 数 × 单卡峰值 FLOPs)
```

H100 SXM BF16 dense 峰值按 989 TFLOPs 算（示例，以你的卡为准）。MFU 低于 30% 通常说明通信没被隐藏、micro-batch 太小或 recompute 过多。

### FSDP2 的执行模型

```
每个 transformer block 是一个 FSDP unit
forward:  all-gather 该 block 参数 → 计算 → 释放分片外的参数
backward: all-gather 参数 → 反向 → reduce-scatter 梯度 → 释放
优化器 step 只更新本卡持有的分片
```

prefetch（提前 all-gather 下一个 block）是隐藏通信的关键，FSDP2 默认开启 forward/backward prefetch，但 micro-batch 过小时计算太短，通信仍露出来。

### 分布式 checkpoint

`torch.distributed.checkpoint`（DCP）按 DTensor 分片保存，每个 rank 写自己的分片，resume 时可以换 GPU 数（重新分片）。要保存的不只是权重：优化器状态、学习率调度器步数、数据加载器位置（已消费样本数）、RNG 状态。少了数据位置，resume 后会重复训练同一批数据，loss 曲线会出现"回到过去"的台阶。

## 实现步骤

### 1. 环境

```bash
pip install torch>=2.5 torchtitan datasets tokenizers wandb   # 版本以当前文档为准
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

### 2. 两条路径任选

**路径 A：TorchTitan（推荐 4 卡以上）**

```bash
git clone https://github.com/pytorch/torchtitan && cd torchtitan
# 用 llama3 debug 配置改成 ~125M 或加载 Llama-3.2-1B 权重继续训练
cp torchtitan/models/llama3/train_configs/debug_model.toml my_p0.toml
```

关键配置项（名字随版本变化，以仓库为准）：

```toml
[training]
batch_size = 8            # micro-batch per rank
seq_len = 2048
steps = 2000
mixed_precision_param = "bfloat16"
mixed_precision_reduce = "float32"

[parallelism]
data_parallel_shard_degree = -1   # 全部卡做 FSDP
tensor_parallel_degree = 1

[activation_checkpoint]
mode = "selective"        # none / selective / full
selective_ac_option = "op"

[checkpoint]
enable_checkpoint = true
interval = 500
async_mode = "async"
```

```bash
CONFIG_FILE=my_p0.toml ./run_train.sh
```

**路径 B：纯 FSDP2 脚本（单卡到 8 卡都可，代码全在自己手里）**

```python
import torch, torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import AutoModelForCausalLM

dist.init_process_group("nccl")
rank = dist.get_rank(); torch.cuda.set_device(rank)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16)
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for layer in model.model.layers:
    fully_shard(layer, mp_policy=mp)          # 每层一个 unit
fully_shard(model, mp_policy=mp)              # 根 unit 管 embedding / lm_head
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.95), weight_decay=0.1)
```

训练循环要点：梯度累积时只在最后一个 micro-batch 让 FSDP 做 reduce-scatter（`model.set_requires_gradient_sync(is_last)`）；每步记录 `torch.cuda.max_memory_allocated()`、tokens/s、grad norm。

### 3. 数据

用 Day 3 的 packing 流水线，固定 tokenizer 与 seq_len=2048。语料可用 FineWeb-Edu 子集或 wikitext（从头训练）；继续训练用 1 到 2 GB 的通用文本即可。记录总 token 数，后面算 6ND。

### 4. 实验矩阵

| 实验 | 变量 | 固定 | 记录 |
|---|---|---|---|
| E1 显存构成 | recompute none / selective / full | 1 卡、b=4、s=2048 | max_memory、tokens/s |
| E2 扩展性 | GPU 数 1 / 2 / 4 / 8 | 每卡 b、s 不变 | 全局 tokens/s、MFU、通信占比（profiler） |
| E3 micro-batch | b = 1 / 2 / 4 / 8 | 4 卡 | MFU 随 b 的曲线 |
| E4 checkpoint | 第 500 步保存，杀进程，resume | 同上 | resume 前后 10 步 loss 对比、resume 耗时 |

### 5. 报告模板

```
p0-report.md
1. 环境：GPU 型号/数量、CUDA、torch、框架版本、模型、数据
2. 预测 vs 实测显存表（Day 2 公式 → 实测）
3. E1-E4 数据表与曲线
4. MFU 计算过程（写出每个数）
5. profiler 截图：一个 step 的 timeline，标出 all-gather / compute / reduce-scatter
6. 失败记录：OOM、NCCL 超时、resume 台阶，各自原因与修法
7. 结论：这套配置在你的机器上的最佳 b / recompute / 卡数组合
```

## 实验

### 实验 1：预测与实测显存的偏差

单卡、0.5B 模型、b=4、s=2048、无 recompute。先用公式算参数 + 优化器 + activation，再看 `max_memory_allocated`。预期偏差 10% 到 30%，来源：CUDA context、临时 buffer、logits（vocab 15 万 × s × b × 4 字节的 FP32 logits 往往是最大单项）。把 logits 分块 cross-entropy 打开后再测一次。

### 实验 2：通信隐藏

4 卡 FSDP2，b=1 与 b=8 各跑 50 步并抓 torch profiler。看 `nccl:all_gather` 是否和计算 kernel 重叠。预期 b=1 时 MFU 明显低，timeline 上通信裸露；b=8 时重叠。给出"每卡 micro-batch 至少多大通信才被隐藏"的经验值。

### 实验 3：resume 一致性

保存第 500 步 checkpoint，用 `kill -9` 结束训练，resume 后对比第 501 到 510 步 loss。如果 dataloader 位置没保存，loss 会明显低于原曲线（数据重复）。再试换卡数 resume（4 卡存，2 卡读），验证 DCP 重分片。

资源分层：API-only 无法完成本项目，改为阅读 TorchTitan 的 MFU 计算代码并手算一组数字；单卡 24 到 48 GB 用 0.5B 模型完成 E1、E3、E4；4 到 8 卡完成全部。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 第一步就 OOM | logits FP32 太大，或 recompute 关闭 | 看 OOM 时的 allocated 与 reserved | 分块 CE、selective recompute、降 b |
| 多卡吞吐不随卡数增长 | 通信裸露，b 太小 | profiler 看 all-gather 与 compute 是否重叠 | 增大 b 或梯度累积、确认 prefetch 开启 |
| MFU 计算出来超过 70% | 每 token FLOPs 算错（漏掉 recompute 的额外前向） | 重新按 6N（+recompute 2N）算 | full recompute 时用 8N |
| NCCL timeout | 某 rank 卡在数据加载或 shape 不一致 | `NCCL_DEBUG=INFO`，看哪 rank 没进 collective | 各 rank 数据长度一致、drop_last |
| resume 后 loss 台阶下降 | dataloader 位置未保存 | 对比 resume 前后消费的样本 id | 保存 sampler 状态或已消费 step 数 |
| loss 为 NaN | lr 太大、没有 grad clip、FP8 溢出 | 看 grad norm 突增的步 | clip 1.0、warmup、降 lr |
| 8 卡比 4 卡每卡显存反而不降 | 根 unit 没切，embedding 全量 | 打印每个 unit 的分片形状 | 根模型也 `fully_shard` |

## 验收标准

- 训练跑满 2000 步或 2 小时，无人工干预。
- 报告里每个 MFU 数字附计算过程；E2 给出扩展效率（8 卡 tokens/s ÷ 单卡 tokens/s ÷ 8）。
- resume 前后 loss 连续，换卡数 resume 成功。
- 能口头解释：为什么 AdamW 每参数 16 字节；FSDP2 每层 forward 发生了哪两次通信；MFU 低于 30% 先查什么。

## 交付物

| 文件 | 内容 |
|---|---|
| `p0-report.md` | 上面的报告模板填满 |
| `train_fsdp2.py` 或 `my_p0.toml` | 可复现的训练入口 |
| `profiles/step_trace.json` | 一个 step 的 profiler trace |
| `week1-interview.md` | 下面面试题的自答 |

## Week 1 面试题

1. **AdamW 训练一个 7B 模型至少要多少显存（不含 activation）？** 16 字节/参数 → 112 GB，所以单卡放不下，必须 ZeRO/FSDP 切分或 8 位优化器。
2. **ZeRO-1/2/3 分别切什么？通信量怎么变？** 优化器状态 / +梯度 / +参数；ZeRO-1、2 通信量与 DDP 相同，ZeRO-3 多一次参数 all-gather，约 1.5 倍。
3. **TP 为什么要求节点内高速互联？** 每层两次 allreduce，数据量 s×b×h，每步几十次，PCIe 会成为瓶颈。
4. **PP 的 bubble 比例公式？** (p−1)/m，p 为 stage 数，m 为 micro-batch 数；interleaved 与 zero-bubble 进一步降低。
5. **序列长度翻倍，activation 显存变多少？** 线性项翻倍，attention 相关项四倍；用 FlashAttention 后 score 矩阵不落地，接近线性。
6. **MFU 与 HFU 的区别？** MFU 只算有效 FLOPs（6N），HFU 把 recompute 的重复前向也算进去；报告要写清用的是哪个。
7. **packing 时为什么要重置 position_ids 并用 varlen attention？** 否则样本之间互相可见，loss 被污染，模型学到跨样本依赖。
8. **FP8 训练主要省的是什么？** 矩阵乘的算力与激活显存带宽，权重主副本仍是 FP32/BF16；需要 per-tensor 或细粒度 scaling（T16 用 128×128 块级）。
9. **checkpoint 要保存哪些东西才能严格 resume？** 权重、优化器状态、调度器、数据位置、RNG、混合精度 scaler（如有）。
10. **训练与推理系统的关系是什么？** 后训练的 RL 阶段要在训练循环里调用推理引擎做 rollout，训练速度往往被 rollout 卡住（Week 3）。

## Week 2 预览

Week 2 从"能训练"转到"训练出对的东西"：SFT 的 loss mask 与超参（Day 8）、LoRA 何时等价于全参（Day 9）、数据工程与 rejection sampling（Day 10）、DPO 家族（Day 11）、奖励模型与过优化（Day 12）、评测与实验管理（Day 13），最后 P1 把 SFT 和 DPO 串成流水线（Day 14）。P0 的训练脚本和 profiler 习惯会一直用到 Week 4。

## 参考

- T04 FSDP：per-parameter sharding 的设计动机与通信隐藏。
- T09 TorchTitan：生产级 FSDP2/TP/PP 组合的参考实现，MFU 计算代码在此。
- T11：activation 显存估算式与 selective recompute。
- T03 ZeRO：理解 FSDP 三种分片等级的原始出处。
- T14 Llama 3：工业规模训练的 MFU、故障率与 checkpoint 实践数据。
- B10 TorchTitan 仓库、B09 Megatron-Core 文档：配置项以此为准。

## 附录：手算示例（Qwen2.5-0.5B，单卡，b=4，s=2048，示例数字）

```
参数量 N ≈ 0.49e9（含 embedding，vocab 151936，h=896，24 层）
静态显存 = 0.49e9 × 16 B ≈ 7.9 GB

activation（无 recompute，T11 估算式，BF16）：
  每层 ≈ s·b·h·(34 + 5·a·s/h)
       = 2048·4·896·(34 + 5·14·2048/896)
       = 7.34e6 · (34 + 160) ≈ 1.42e9 B ≈ 1.4 GB
  24 层 ≈ 34 GB  ← 24GB 卡放不下
  用 FlashAttention 后去掉 5·a·s/h 项：每层 ≈ 0.25 GB，24 层 ≈ 6 GB
  selective recompute 再降约一半

logits：s·b·vocab·4 B（FP32）= 2048·4·151936·4 ≈ 5.0 GB  ← 单项最大，必须分块 CE

预测峰值（FlashAttention + 分块 CE + 无 recompute）≈ 7.9 + 6 + 1（分块后 logits）+ 1 到 2（临时）≈ 16 到 17 GB
实测请填：__________ GB，偏差 ____%，主要来源：__________

吞吐：若实测 12k tokens/s（示例），每 token FLOPs ≈ 6 × 0.49e9 ≈ 2.9e9
  实际算力 = 12e3 × 2.9e9 = 3.5e13 FLOP/s = 35 TFLOPs
  MFU（H100 BF16 989 TFLOPs）≈ 3.5%  ← 小模型在大卡上 MFU 天然很低，
  这是 kernel 太小、launch 占比高导致的；换 1.5B 或增大 b 会明显上升。
  报告里必须写明这一点，不要把小模型的低 MFU 误判为配置错误。
```

把这一页的空白填完，就是 P0 报告的第 2 节。
