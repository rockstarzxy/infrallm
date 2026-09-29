---
title: "Day 9：LoRA、QLoRA 与 PEFT：何时等价于全参"
type: concept
tags: [llm-training, lora, qlora, peft, dora]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 9：LoRA、QLoRA 与 PEFT：何时等价于全参

> 上一课 [[llm-training-30d/week2/day08-sft-engineering]] · 下一课 [[llm-training-30d/week2/day10-data-engineering-synthetic]]。相关：推理侧多 LoRA serving 在 [[ai-infra-30d/week3/day20-production-deploy]] 有提及；RL 中用 LoRA 见 [[llm-training-30d/week3/day18-rl-training-frameworks]]。

## 学习目标

1. 推导 LoRA 的前向与参数量，解释 rank、alpha、target modules、初始化各自影响什么。
2. 说清 QLoRA 的三个组成（NF4、双重量化、paged optimizer）分别省了什么、代价是什么。
3. 复述 "LoRA Without Regret"（B04）的核心结论：什么条件下 LoRA 与全参训练结果等价，什么时候不等价，学习率该怎么换算。
4. 在 SFT 和 RL 两种场景下用 PEFT / TRL / verl 跑通 LoRA，并把 adapter 合并回权重供 vLLM 部署。
5. 给出"LoRA 还是全参"的决策表。

## 工业现状

- **单卡与小团队**：LoRA/QLoRA 是默认选项。Unsloth、PEFT、TRL 都以 LoRA 为一等公民；7B 到 14B 模型 QLoRA 可在 24 GB 卡上做 SFT。
- **一线实验室的 SFT**：主流仍是全参（T14、T15、T22 都是全参 SFT）。原因不是 LoRA 效果差，而是他们不缺显存，且全参少一个变量。
- **RL 阶段**：LoRA 重新变得重要。RL 每步只更新很少的信息（B04 指出每 episode 约 1 bit 量级的信息），LoRA 容量足够；LoRA 让 rollout 引擎与训练共享同一份 base 权重，权重同步只传 adapter（几十 MB 而不是几十 GB），colocated 模式下显存压力大幅下降。verl、TRL GRPOTrainer 都支持 LoRA RL。
- **多租户与快速迭代**：一个 base 挂几十个 LoRA adapter 的推理形态（vLLM 的多 LoRA serving）让"每个客户一个 adapter"成为常见产品形态。
- **DoRA（T26）** 在部分任务上比 LoRA 更接近全参，PEFT 已内置，但工业界采用度不高，作为可选项。

## 核心原理

### LoRA 的前向

```
原线性层：   y = W x,           W ∈ R^{d_out × d_in}
LoRA：       y = W x + (α / r) · B A x
             A ∈ R^{r × d_in}（高斯初始化），B ∈ R^{d_out × r}（零初始化）
可训练参数： r · (d_in + d_out)     对 4096×4096 的层，r=16 → 131k，占原层的 0.8%
```

- **B 零初始化**保证训练开始时模型等于 base。
- **α/r 缩放**：α 固定、r 变化时等效学习率变化。常见做法 α = 2r 或使用 rsLoRA（缩放 α/√r）让不同 r 的最优 lr 接近。
- **target modules**：只挂 q、v 是早期做法；现在默认挂全部线性层（q、k、v、o、gate、up、down），B04 的结论是 MLP 层挂 LoRA 比 attention 层更重要，只挂 attention 明显欠拟合。
- **rank**：SFT 常用 16 到 128；RL 用 8 到 32 已足够（B04 甚至 r=1 在 RL 上接近全参）。rank 决定的是容量上限，不是学习速度。

### 显存对比（7B 模型，BF16，示例）

| 项目 | 全参 AdamW | LoRA r=32 | QLoRA r=32 |
|---|---|---|---|
| base 权重 | 14 GB | 14 GB（冻结，BF16） | 3.9 GB（NF4） |
| 梯度 | 14 GB | 约 0.2 GB | 约 0.2 GB |
| 优化器状态 | 84 GB（FP32 主权重 + m + v） | 约 0.6 GB | 约 0.6 GB |
| activation | 相同 | 相同（仍要反向穿过 base） | 相同 + 反量化临时 |
| 合计（不含 activation） | 112 GB | 15 GB | 5 GB |

LoRA 省的是梯度和优化器状态，不省 activation；序列长度长时 activation 仍是主要显存，要配 gradient checkpointing。

### QLoRA 三件套（T25）

1. **NF4**：4 位 NormalFloat，按正态分布分位数设计的量化格点，比均匀 INT4 误差小；base 权重以 NF4 存储，前向时逐块反量化到 BF16 计算。
2. **双重量化**：量化常数（每 64 个权重一个 FP32 scale）再量化成 8 位，每参数再省约 0.4 位。
3. **paged optimizer**：优化器状态放在 CUDA 统一内存，显存峰值时自动换出到 CPU，防止长序列 OOM。

代价：反量化让每步慢 20% 到 40%（示例）；NF4 base 的质量损失在 7B 以上模型上通常可忽略，但 1B 以下小模型会明显。

### LoRA 何时等价于全参（B04 的结论）

```
等价条件：
  1. 数据集包含的信息量 < LoRA 容量（SFT 的中小数据集、几乎所有 RL 任务）
  2. LoRA 挂在所有线性层，尤其 MLP
  3. 学习率约为全参最优 lr 的 10 倍（LoRA 的参数化让梯度尺度不同）
  4. batch size 不要太大：LoRA 在大 batch 下的 loss 比全参差，差距随 batch 增大而扩大

不等价：
  - 大规模 SFT（数百万样本，接近继续预训练）：LoRA 容量不够，loss 曲线在后期明显落后
  - 需要改变大量事实知识的任务
```

工程含义：单卡做 SFT 和几乎所有 RL 实验，LoRA 不是"妥协"，是等价方案；只有大数据 SFT 才必须全参。

### DoRA（T26）

把权重更新分解成幅值与方向：`W' = m · (W + BA) / ||W + BA||`，额外训练每列的幅值 m。在部分 benchmark 上比 LoRA 更稳，参数量几乎不变；PEFT 里 `use_dora=True` 即可，代价是前向多一次范数计算。

## 实现步骤

### 1. PEFT + TRL 做 LoRA SFT

```python
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

lora = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_rslora=True,            # α/√r 缩放，换 r 时 lr 更稳
    task_type="CAUSAL_LM",
)
cfg = SFTConfig(..., learning_rate=1e-4,   # 约为全参 1e-5 的 10 倍
                per_device_train_batch_size=4, gradient_accumulation_steps=4)
trainer = SFTTrainer(model=model, args=cfg, peft_config=lora, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.model.save_pretrained("out/lora-adapter")     # 只存 adapter，几十 MB
```

### 2. QLoRA

```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map={"": 0})
# 其余同上；optim="paged_adamw_8bit" 打开 paged optimizer
```

### 3. 合并与部署

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, "out/lora-adapter").merge_and_unload()
merged.save_pretrained("out/merged"); tok.save_pretrained("out/merged")
```

QLoRA 的 adapter 要合并到 BF16 的 base 上，不要合并到 NF4 权重（会把量化误差固化）。vLLM 也可以不合并，直接 `--enable-lora --lora-modules name=path` 动态挂载。

### 4. RL 中用 LoRA（Week 3 会详讲，这里先看入口）

TRL GRPOTrainer 传 `peft_config` 即可；verl 在 `actor_rollout_ref.model.lora_rank` 等参数开启，rollout 侧 vLLM 加载同一 base，每步只同步 adapter 权重。以两者当前文档为准。

### 5. 多 adapter 并行训练

同一 base 上按任务/租户训练多个 adapter 时，可以把 base 权重共享在显存里、按 adapter 分批切换，PEFT 的 `add_adapter` / `set_adapter` 支持；Unsloth 与 LoRAX 类框架做了更深的批处理。

## 实验

### 实验 1：LoRA vs 全参的 lr 换算

1.5B 模型，2 万条 SFT，全参 lr {5e-6, 1e-5, 2e-5} 与 LoRA r=32 lr {5e-5, 1e-4, 2e-4, 5e-4}，记录 dev loss 最优值。预期 LoRA 最优 lr 约为全参的 10 倍，最优 dev loss 差距 < 1%。单卡 24 GB 全参用 0.5B 替代。

### 实验 2：rank 与 target modules

固定 lr=1e-4，网格 r {4, 16, 64} × modules {attention only, all linear}。预期 all linear r=16 优于 attention only r=64。

### 实验 3：容量不足的边界

用 20 万条数据（接近继续预训练）重复实验 1，预期 LoRA 曲线在后半段落后全参，差距随 r 减小而扩大。这验证 B04 的不等价条件。4 到 8 卡。

### 实验 4：QLoRA 的速度与质量

7B 模型，同一数据，LoRA（BF16 base）vs QLoRA（NF4 base），记录 tokens/s、峰值显存、dev loss、IFEval。预期 QLoRA 慢 20% 到 40%，显存 1/3，质量差距 < 0.5 点。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| LoRA 训练 loss 几乎不动 | lr 沿用全参的 1e-5 | 对比 lr 扫描 | lr × 10 |
| 训练后模型和 base 一样 | adapter 没加载 / 没 merge / 挂错 modules | 打印 `print_trainable_parameters()` | 检查 target_modules 名字与模型层名一致 |
| 合并后推理输出乱码 | 合并到量化 base 上 | 检查合并时 base 的 dtype | 用 BF16 base 合并 |
| LoRA 明显比全参差 | 只挂 attention、r 太小、数据太大 | 实验 2/3 | 挂 MLP、增大 r、或改全参 |
| QLoRA OOM 在长序列 | activation 与反量化临时 buffer | 看 max_memory 随 seq_len | gradient checkpointing、paged optimizer |
| 多 adapter 互相污染 | `set_adapter` 没切换或 merge 后再训 | 检查 active adapter 名 | 训练前显式切换 |
| RL 中 LoRA 权重同步后 rollout 不变 | adapter 没传到推理引擎 | 对比同步前后 rollout logprob | 检查框架的 LoRA sync 路径 |
| vLLM 挂载 adapter 报 rank 超限 | 服务端 `--max-lora-rank` 小于 adapter | 看启动参数 | 提高上限 |

## 验收标准

- 能手推 LoRA 参数量与显存对比表，说明为什么 activation 不省。
- 实验 1 的 lr 换算结论与实验 3 的边界结论各有数据支持。
- 产出一个合并后的模型在 vLLM 上正常推理，以及一个不合并的 adapter 通过 `--lora-modules` 挂载成功。
- 能给出"LoRA 还是全参"的决策：数据量、任务类型、显存、是否 RL 四个维度。

## 交付物

| 文件 | 内容 |
|---|---|
| `lora-vs-full.md` | 实验 1 到 3 的表与曲线，附 lr 换算结论 |
| `qlora-benchmark.md` | 实验 4 |
| `out/lora-adapter/`、`out/merged/` | adapter 与合并模型 |
| `peft-decision-table.md` | 决策表 |

## 决策表

| 场景 | 选择 | 理由 |
|---|---|---|
| 单卡 SFT，≤ 10 万条 | LoRA r=32，all linear，lr 1e-4 | 等价全参，省显存 |
| 单卡 SFT，7B+ 模型 | QLoRA | 显存 1/3，质量损失可忽略 |
| 大规模 SFT（百万级）或继续预训练 | 全参 | LoRA 容量不够 |
| RL（GRPO/PPO） | LoRA r=8 到 32 优先 | 容量足够，权重同步快，colocated 显存友好 |
| 多租户定制 | LoRA 不合并，推理侧多 adapter | 一份 base 服务多个客户 |
| 需要改变大量事实知识 | 全参 + 继续预训练 | LoRA 不擅长知识注入 |

## 参考

- T24 LoRA：原始推导与初始化。
- T25 QLoRA：NF4、双重量化、paged optimizer 的定义与实验。
- T26 DoRA：幅值方向分解。
- B04 LoRA Without Regret：等价条件、lr 10 倍规则、MLP 优先、batch size 效应。这是本课最重要的参考。
- B14 PEFT 文档：参数名与多 adapter API 以此为准。

## 附录 A：RL 中 LoRA 的数据流（为 Week 3 预热）

```
训练进程（FSDP，base 冻结 + LoRA 可训练）
   │  每个 RL step 结束
   │  只导出 adapter 张量（A、B 矩阵，示例 7B r=32 约 80 MB）
   ▼
rollout 引擎（vLLM，同一份 base 权重常驻）
   │  热更新 adapter：--enable-lora，或框架内部直接写入 LoRA 权重槽
   │  采样下一批 rollout
   ▼
训练进程：用新 rollout 计算 loss，更新 adapter
```

对比全参 RL：每步要把几十 GB 的完整权重从训练布局（FSDP 分片）重排成推理布局（TP 分片）再广播，Week 3 Day 19 专门讲这个 resharding。LoRA 把这一步的数据量缩小两到三个数量级，是单机 RL 实验能跑起来的关键。

为什么容量够：一次 RL episode 给策略的信息量约为 log2(可能结果数) 比特量级，一个 group 16 条 rollout 的 GRPO step 也只有几十比特；几千步 RL 累积的信息远小于 r=8 adapter 的参数量。B04 用 r=1 在数学 RL 上复现全参曲线，就是这个道理。SFT 不同：一条高质量长回复就携带上千比特，数据集大了容量就会不够。

## 附录 B：LoRA 超参速查

| 超参 | SFT 建议 | RL 建议 | 备注 |
|---|---|---|---|
| r | 16 到 128 | 8 到 32 | 数据越大越大；RL 小即可 |
| alpha | 2r（或 rsLoRA） | 同左 | 换 r 时保持 alpha/r 或 alpha/√r 不变 |
| lr | 全参最优 × 10，常 1e-4 到 3e-4 | 1e-5 到 1e-4（RL 的全参 lr 本来就低） | 先扫 lr 再扫 r |
| dropout | 0.05 | 0 | RL 不要 dropout，会破坏 logprob 一致性 |
| target_modules | 全部线性层 | 全部线性层 | 只挂 attention 是常见错误 |
| embedding / lm_head | 一般不挂 | 不挂 | 新增 special token 时要解冻 embedding 对应行 |
| batch | 与全参相同或略小 | 同全参 | 大 batch 下 LoRA 劣势扩大 |
| 权重衰减 | 0 | 0 | 对 adapter 意义不大 |

## 附录 C：新增 special token 时的处理

工具调用格式常要新增 `<tool_call>` 之类的 token。LoRA 默认不训 embedding，新 token 的 embedding 是随机的，模型学不会用。做法：

```python
tok.add_special_tokens({"additional_special_tokens": ["<tool_call>", "</tool_call>"]})
model.resize_token_embeddings(len(tok))
lora = LoraConfig(..., modules_to_save=["embed_tokens", "lm_head"])   # 这两层全参训练并随 adapter 保存
```

`modules_to_save` 会让 adapter 文件变大（embedding 是 vocab × h，几百 MB），合并时也要一起合并。另一种做法是只解冻新增行（自定义 hook 把其余行梯度置零），PEFT 较新版本提供 `trainable_token_indices`，以当前文档为准。

## 附录 D：多 LoRA 推理时的显存

vLLM 多 LoRA serving 为每个 adapter 预留 `max_loras × max_lora_rank` 的权重槽，并用 Punyu/SGMV 类 kernel 在一个 batch 里同时计算不同 adapter 的请求。显存开销 = 槽数 × 每 adapter 大小，与并发无关；延迟开销通常在 5% 到 15%（示例）。合并后的模型没有这个开销，但每个客户一份完整权重。选择依据：adapter 数量多、切换频繁 → 不合并；单一 adapter 长期服务 → 合并。推理课 Day 20 有部署侧参数。

## 附录 E：自检问题

1. r=32、挂全部 7 个线性层的 Qwen2.5-7B LoRA 有多少可训练参数？占比多少？（按每层 h=3584、intermediate=18944 算。）
2. 为什么 LoRA 的最优 lr 比全参高一个数量级？如果你用 rsLoRA，换 r 时 lr 还需要重新扫吗？
3. 同一份 SFT 数据，LoRA 的 dev loss 在训练后期落后全参，这说明哪个条件被违反？先增大 r 还是先换全参？
4. QLoRA 训出的 adapter 合并到 NF4 base 上会发生什么？正确做法是什么？
5. RL 中为什么 dropout 要设 0？（提示：rollout 时的 logprob 与训练时的 logprob 必须来自同一函数。）
6. 新增 `<tool_call>` token 后只训 LoRA，模型输出这个 token 的概率会怎样？怎么修？
7. 一个 base 挂 50 个 adapter 在线服务，显存开销取决于什么？什么情况下应该合并？

答案要点写进 `peft-decision-table.md` 末尾，Day 14 的 P1 项目会用到 LoRA SFT 与合并流程。
