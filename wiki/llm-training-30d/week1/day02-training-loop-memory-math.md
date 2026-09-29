---
title: "Day 2：训练循环剖析与显存、算力数学"
type: concept
tags: [llm-training, memory, mixed-precision, mfu, activation-checkpointing]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 2：训练循环剖析与显存、算力数学

> 上一课 [[llm-training-30d/week1/day01-training-landscape]] · 下一课 [[llm-training-30d/week1/day03-data-pipeline-tokenization]]。相关：推理课 [[ai-infra-30d/week4/day26-roofline]]（roofline 与算术强度）、[[ai-infra-30d/week1/day01-inference-pipeline]]（KV Cache 显存公式）。

## 学习目标

1. 徒手算出任意规模 dense 模型在 AdamW + BF16 混合精度下的参数、梯度、优化器状态显存，误差在 10% 以内。
2. 解释 activation 显存与序列长度、micro batch、层数、hidden size 的关系，知道 activation checkpointing 换回多少显存、多付多少算力。
3. 用 6ND 估算训练算力，定义并计算 MFU，能判断一次训练"跑得快不快"。
4. 用 1B 模型实测显存与吞吐，和公式对上。

## 工业现状

所有公开配方都在 BF16 混合精度下训练，FP32 只保留主权重和优化器状态；DeepSeek-V3（T16）把主要 GEMM 推到 FP8，用细粒度（block-wise）缩放保持精度，是目前最激进的公开做法，Blackwell 上 FP8 训练会更普遍。activation checkpointing 在预训练里几乎必开（selective recompute，T11），在 SFT 短序列里常关掉换速度。MFU 是各家报告训练效率的统一口径：Llama 3（T14）报告 405B 在 16K H100 上 38% 到 43%；DeepSeek-V3 没有直接报 MFU，但给出了每万亿 token 的 GPU 小时。业内把 dense 模型 40% 以上的 MFU 视为合格，MoE 因为 all-to-all 通常更低。

## 核心原理

### 训练循环的每一步在做什么

```python
for step in range(num_steps):
    optimizer.zero_grad(set_to_none=True)
    for micro in range(grad_accum):                       # 梯度累积
        batch = next(loader)                              # [micro_bs, seq_len]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch.input_ids)               # forward：保存 activation
            loss = ce(logits, batch.labels) / grad_accum
        loss.backward()                                   # backward：用 activation 算梯度，释放 activation
    clip_grad_norm_(model.parameters(), 1.0)              # 梯度裁剪，训练稳定性的第一道防线
    optimizer.step()                                      # AdamW：读参数、梯度、m、v，写参数、m、v
    scheduler.step()
```

"必须理解"的三个事实：

- **forward 保存 activation，backward 消费它**。activation 的峰值在 forward 结束、backward 开始那一刻。这就是为什么显存峰值和 micro batch、序列长度强相关，而与 global batch 无关（梯度累积不增加 activation）。
- **BF16 不需要 loss scaling**。BF16 的指数位和 FP32 一样，不会像 FP16 那样下溢。混合精度的意思是：前向反向用 BF16 算，主权重、优化器状态、梯度累积用 FP32 存。
- **optimizer.step 是一次纯带宽操作**。每个参数读写约 16 到 20 字节，7B 模型就是 100 多 GB 的显存流量，在多卡下被 ZeRO/FSDP 切分（Day 4）。

### 静态显存：参数、梯度、优化器状态

设参数量 N，AdamW，BF16 混合精度，每个参数：

| 项 | 精度 | 字节 |
|---|---|---|
| 主权重（master weights） | FP32 | 4 |
| 计算用权重副本 | BF16 | 2 |
| 梯度 | BF16 或 FP32（框架相关） | 2 或 4 |
| Adam 一阶矩 m | FP32 | 4 |
| Adam 二阶矩 v | FP32 | 4 |
| 合计 | | 16 到 18 |

所以经典说法"AdamW 混合精度每参数 16 字节"来自 4 + 2 + 2 + 4 + 4。7B 模型静态显存约 112 GB（示例），单张 80 GB 卡放不下，必须切分或用 LoRA / 8-bit 优化器。推理只需要 2 字节每参数，这是训练与推理显存差 8 倍的根源。

### 动态显存：activation

以标准 Transformer 层（无 FlashAttention）为例，每层每 token 保存的 activation 约为（T11 的推导，单位字节，s 序列长度、b micro batch、h hidden、a 头数）：

```text
每层 activation ≈ s·b·h·(34 + 5·a·s/h)
                  └──┬──┘   └──┬──┘
                线性层、norm、残差   attention score 矩阵（s² 项）
```

用 FlashAttention 后 s² 项消失（不物化 score 矩阵），剩下约 34·s·b·h 字节每层。示例：Llama-3-8B（h=4096，32 层，s=4096，b=1）：

```text
每层 ≈ 34 × 4096 × 1 × 4096 ≈ 570 MB
32 层 ≈ 18 GB
```

再加上 logits：s × vocab × 4 字节（FP32 交叉熵）= 4096 × 128256 × 4 ≈ 2 GB，这就是为什么大词表模型要用分块交叉熵（chunked CE）。

### activation checkpointing 的取舍

| 策略 | 保存什么 | 显存 | 额外算力 |
|---|---|---|---|
| 不 checkpoint | 全部 activation | 100% | 0 |
| full recompute | 只存每层输入 | 约 1/34 每层 | 约 +33%（多一次 forward） |
| selective recompute（T11） | 存线性层输出，重算 attention 和 norm | 约 1/3 到 1/2 | 约 +5% 到 10% |
| offload 到 CPU | 全存，搬到主机内存 | 显存几乎为 0 | 受 PCIe 带宽限制 |

工业默认是 selective：重算那些"算力便宜、显存贵"的部分。FlashAttention 本身内部就在重算 softmax，是同一思想。

### 算力：6ND 与 MFU

一次训练 step 的 FLOPs ≈ 6 × N × D（N 参数量，D 本步 token 数）：forward 2ND，backward 4ND（对参数和对输入各一次）。attention 的 s² 项在长序列时要加上 `12 · L · h · s²`（L 层数），s 超过 8K 时不可忽略。开 full recompute 变成 8ND。

```text
MFU = 实测 tokens/s × 6N / (GPU 数 × 单卡峰值 FLOPS)
```

H100 SXM 的 BF16 dense 峰值约 989 TFLOPS（示例，以规格表为准）。7B 模型单卡跑到 3000 tokens/s 就是 6 × 7e9 × 3000 / 989e12 ≈ 12.7%，说明还有很大优化空间；预训练级别的 40% 对应约 9400 tokens/s/GPU。

"知道即可"：HFU（硬件利用率，含重算）会比 MFU 高，报告时要说清口径。

### 三个常见修正

**长序列**：s 到 32K 以上时，即使用 FlashAttention，34·s·b·h 的线性项也会超过静态显存（示例：8B 模型 s=128K b=1 时每层约 17 GB，32 层 550 GB），所以长上下文训练必须用 context parallel 把序列切到多卡（Day 5），或者极小的 micro batch 加 full recompute。

**MoE**：参数量 N 用总参数（决定静态显存），算力用激活参数（决定 6ND）。DeepSeek-V3 总参数 671B、激活 37B，静态显存按 671B 算，每 token FLOPs 按 37B 算。这也是 MoE 训练 MFU 口径混乱的来源：分母用总参数会把 MFU 算得很低。

**LoRA**：只有 LoRA 参数有梯度和优化器状态，静态显存降到"基座 BF16 2 字节每参数 + LoRA 参数 16 字节每参数"，但 activation 不变，因为反向传播仍要穿过整个网络。所以 LoRA 省的是静态显存，长序列 LoRA 依然会 OOM（Day 9）。

### 显存构成一览

```text
80 GB H100 上训练 7B（示例，混合精度，FSDP 前）
┌──────────────────────────────────────────────────────┐
│ 主权重 FP32 28 GB │ BF16 副本 14 │ 梯度 14 │ m 28 │ v 28 │  = 112 GB  ← 已超单卡
├──────────────────────────────────────────────────────┤
│ activation（s=4K, b=1, FlashAttention）≈ 18 GB        │
│ logits FP32 ≈ 2 GB │ 临时 buffer、碎片 ≈ 3 到 5 GB    │
└──────────────────────────────────────────────────────┘
```

推理课 Day 1 的 KV Cache 在训练里不存在（训练不做增量 decode），取而代之的是 activation。两者都随序列长度线性增长，但 activation 系数大得多。

## 实现步骤

### 1. 用公式算一遍再实测

```python
# memory_estimate.py
def static_gb(n_params, bytes_per_param=16): return n_params * bytes_per_param / 1e9
def act_gb(layers, s, b, h, flash=True, a=None):
    per_layer = 34 * s * b * h if flash else s * b * h * (34 + 5 * a * s / h)
    return layers * per_layer / 1e9
def logits_gb(s, b, vocab): return s * b * vocab * 4 / 1e9
# Qwen2.5-1.5B：n=1.54e9, layers=28, h=1536, vocab=151936
print(static_gb(1.54e9), act_gb(28, 4096, 1, 1536), logits_gb(4096, 1, 151936))
```

### 2. 实测脚本（单卡）

```python
import torch, time
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.bfloat16,
                                         attn_implementation="flash_attention_2").cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-5, fused=True)
x = torch.randint(0, 150000, (1, 4096), device="cuda")
torch.cuda.reset_peak_memory_stats()
for i in range(5):
    t0 = time.perf_counter()
    loss = m(input_ids=x, labels=x).loss; loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if i >= 2: print(f"step {i}: {time.perf_counter()-t0:.2f}s, peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
```

注意：HF 的 AdamW 默认梯度是 BF16 还是 FP32、主权重是否 FP32，取决于加载 dtype。上面用 BF16 加载意味着主权重也是 BF16（纯 BF16 训练，精度略差），静态显存是每参数 2 + 2 + 4 + 4 = 12 字节。要做标准混合精度就用 FP32 加载配 autocast，或者用 FSDP2 的 `MixedPrecisionPolicy`（Day 4）。

### 3. 开关 activation checkpointing 对比

```python
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

记录峰值显存和 step 时间的变化。预期显存降到原来的一半以下，时间增加 20% 到 35%。

### 4. 计算 MFU

```python
tokens_per_s = 4096 / step_time
mfu = tokens_per_s * 6 * 1.54e9 / 989e12     # H100 示例；A100 用 312e12
```

### 5. 分块交叉熵

大词表模型的 logits 显存可以用 Liger-Kernel 或 torch.compile 的 fused CE 消掉；TRL 与 Unsloth 都集成了。`pip install liger-kernel` 后 `AutoLigerKernelForCausalLM`，对比 logits 显存变化。

## 实验

### 实验 1：显存公式与实测对表

固定：Qwen2.5-1.5B，s=4096，b=1，FlashAttention，AdamW。变量：加载 dtype（BF16 / FP32 + autocast）、checkpointing 开关、序列长度 1K/2K/4K/8K。记录峰值显存，与公式预测画在同一张图。预期：静态部分与公式差 5% 以内；activation 随 s 线性增长（FlashAttention）；关掉 FlashAttention 换 eager attention 后出现 s² 增长。

### 实验 2：micro batch 与梯度累积的等价性

固定 global batch 16 条 × 2048 token。对比 (micro=1, accum=16)、(micro=4, accum=4)、(micro=16, accum=1)。记录显存、step 时间、loss 曲线前 50 步。预期：loss 曲线重合（数值上几乎一致），显存随 micro 线性增长，吞吐先升后平（GPU 饱和后 micro 再大没收益）。

### 实验 3：MFU 分解

用 torch profiler 抓一个 step，把时间分成 GEMM、attention、norm 与逐元素、优化器、数据加载、空闲。算出当前 MFU 并回答"最大的非 GEMM 开销是什么"。单卡小模型的典型答案是逐元素 kernel 和优化器 step 占比过高，这是 Day 6 fused kernel 的动机。

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    loss = m(input_ids=x, labels=x).loss; loss.backward(); opt.step(); opt.zero_grad()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

### 实验 4（多卡可选）：优化器状态的带宽成本

在 1 卡上用 7B 模型只跑 optimizer.step（参数和梯度随机初始化），测每次 step 的时间，换算成显存带宽利用率。预期接近 HBM 带宽上限，证明 optimizer step 是 memory-bound，这是 ZeRO 切优化器状态既省显存又省时间的原因。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 实测显存比公式大很多 | PyTorch 缓存分配器碎片；logits 未分块；eager attention | `torch.cuda.memory_summary()` 看 reserved 与 allocated 差；检查 attn_implementation | 用 FlashAttention；分块 CE；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| OOM 发生在 backward 而不是 forward | activation 峰值在 backward 开头 | 在 forward 后打印 allocated | 开 checkpointing 或减 micro batch |
| 开了 checkpointing 显存没降 | `use_reentrant=True` 与某些模型不兼容；只对部分层生效 | 对比开关前后 allocated | 用 `use_reentrant=False`；确认 `model.is_gradient_checkpointing` |
| MFU 只有个位数 | micro batch 太小、序列太短、CPU 数据加载慢、eager 模式 | profiler 看 GPU 空闲比例 | 增大 micro batch 到显存上限；预取数据；torch.compile |
| loss 为 NaN | FP16 溢出；lr 过大；梯度未裁剪 | 打印梯度范数 | 用 BF16；clip 1.0；降 lr |
| 纯 BF16 训练 loss 比混合精度差 | 主权重 BF16 累积误差（小更新被舍入） | 对比 FP32 主权重版本 | 用 FP32 master weights 或 Kahan 求和优化器 |

## 思考题

1. 8-bit AdamW（bitsandbytes）把 m、v 压到 1 字节，静态显存降到多少？为什么它对预训练不常用，对 SFT 常用？
2. 梯度累积 16 步和 global batch 直接开 16 倍，数值上有什么细微差别？（提示：LayerNorm 统计量、dropout）
3. 如果 profiler 显示 GPU 空闲 30%，但 GEMM 已经是 BF16 Tensor Core，下一步先查什么？
4. 用 6ND 算，训练 7B 模型 1T token 需要多少 H100 小时（MFU 40%）？和 DeepSeek-V3 报告的 2.788M H800 小时训练 14.8T token 的 671B（激活 37B）MoE 相比，谁的单位 token 效率高？

## 验收标准

- 给定任意 N、L、h、s、b，能在 5 分钟内给出静态显存、activation 显存、6ND 算力和目标 MFU 下的 tokens/s。
- 实验 1 的公式与实测误差在 10% 以内，能解释误差来源。
- 能说明梯度累积不改变 activation 峰值、checkpointing 换显存的代价、optimizer step 为什么 memory-bound。

## 交付物

| 文件 | 内容 |
|---|---|
| `memory_estimate.py` | 显存与算力估算函数 |
| `memory-vs-formula.md` | 实验 1 对表图与误差分析 |
| `mfu-breakdown.md` | 实验 3 的时间分解与 MFU |
| `training-math-cheatsheet.md` | 一页纸：16 字节每参数、34·s·b·h、6ND、MFU 公式与典型数值 |

## 参考

- T11：activation 显存公式与 selective recompute 的原始推导，本页公式的出处。
- T10、T16：FP8 训练的做法与精度保持手段。
- T14：Llama 3 报告的 MFU 数据与训练基础设施章节，建立"合格 MFU"的参照。
- T09：TorchTitan 的实现里有干净的混合精度与 checkpointing 配置，Day 4 与 Day 6 会用。
