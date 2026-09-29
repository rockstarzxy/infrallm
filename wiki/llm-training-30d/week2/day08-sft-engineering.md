---
title: "Day 8：SFT 原理与工程"
type: concept
tags: [llm-training, sft, loss-mask, packing, cold-start]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 8：SFT 原理与工程

> 上一课 [[llm-training-30d/week1/day07-p0-project-fsdp2-training]] · 下一课 [[llm-training-30d/week2/day09-lora-qlora-peft]]。相关：数据流水线 [[llm-training-30d/week1/day03-data-pipeline-tokenization]]；推理侧 chat template 的处理见 [[ai-infra-30d/week2/day08-vllm-architecture]]。

## 学习目标

1. 写出 SFT 的目标函数，说清 loss mask 为什么只覆盖 assistant token，多轮样本怎么构造。
2. 给出一组全参 SFT 的默认超参并解释每个数的来源，知道模型规模变化时哪些要跟着变。
3. 用 TRL SFTTrainer 和 verl 的 SFT 入口各跑通一次，能对照两者的数据格式与 mask 实现。
4. 解释 SFT 在整条后训练流水线里的角色（cold start、格式学习、能力注入的边界），以及过拟合与遗忘的表现。
5. 处理 packing、长上下文、多轮三种场景下的工程细节。

## 工业现状

SFT 是所有公开配方的第一个后训练阶段，但它的定位在 2024 到 2026 年发生了变化：

- **Llama 3（T14）**：SFT 用大量人工 + 合成数据，与 rejection sampling 和 DPO 迭代六轮；SFT 承担格式、工具调用、多语言等能力的主体注入。
- **DeepSeek-R1（T17）**：R1-Zero 证明可以不做 SFT 直接 RL，但最终 R1 仍用几千条长 CoT 做"cold start SFT"来解决可读性与语言混杂，之后 RL，再用 RL 模型 rejection sampling 产生 80 万条 SFT 数据做第二轮 SFT。SFT 的角色变成"给 RL 一个好起点"和"把 RL 学到的能力蒸馏回通用格式"。
- **Qwen3（T15）**：四阶段：长 CoT cold start SFT → 推理 RL → thinking/non-thinking 融合 SFT → 通用 RL。SFT 出现两次，第二次专门做模式融合。
- **Tulu 3（T22）**：完全开源的 SFT 数据配方（约 100 万条，按能力分桶配比），证明数据配比对 SFT 结果的影响大于超参。
- **Kimi K2（T19）**：SFT 数据里大量是 agentic 工具调用轨迹的合成数据，SFT 直接教工具调用格式。

共识：SFT 学的是分布与格式，不是新知识；能力上限由预训练决定，SFT 数据质量决定下限；SFT 过多会损害 RL 的探索空间（entropy 太低）。

## 核心原理

### 目标函数

给定对话 `x = (system, user_1, assistant_1, user_2, assistant_2, ...)`，SFT 最大化 assistant token 的条件似然：

```
L_SFT(θ) = − Σ_{t ∈ M} log π_θ(x_t | x_<t)
M = assistant 生成的 token 位置集合（不含 system、user、tool 返回、模板控制符里"由模板决定"的部分）
```

mask 的边界由 chat template 决定。以 Qwen/ChatML 为例：

```
<|im_start|>system\n...<|im_end|>\n            ← mask 掉
<|im_start|>user\n...<|im_end|>\n              ← mask 掉
<|im_start|>assistant\n                        ← mask 掉（提示模型该说话了）
Hello, ...<|im_end|>\n                         ← 算 loss（含 <|im_end|>，模型要学会停）
```

`<|im_end|>` 必须算 loss，否则模型学不会结束，推理时一直生成到 max_tokens。这是最常见的 SFT bug。

### 多轮样本

一条 5 轮对话可以变成：

1. **一条样本、多段 mask**（推荐）：整段对话一次前向，所有 assistant 段都算 loss。每个 assistant 段看到的上下文和推理时一致。
2. **拆成 5 条前缀样本**：第 k 条只对第 k 轮 assistant 算 loss。计算量是方案 1 的多倍，只在需要对不同轮加权时用。

工具调用轨迹同理：tool 返回是 observation，不算 loss；模型发出的 tool call JSON 算 loss。

### 超参的来源

| 超参 | 全参 SFT 典型值（示例） | 为什么 |
|---|---|---|
| 学习率 | 7B: 1e-5 到 2e-5；70B: 5e-6 到 1e-5 | 预训练末期 lr 的量级；太大破坏预训练分布 |
| 调度 | cosine 到 10% 峰值，warmup 3% | 与预训练一致 |
| epochs | 1 到 3 | 高质量小数据 2 到 3 轮，大数据 1 轮；看 dev loss 拐点 |
| 全局 batch（token 数） | 0.5M 到 4M token | 太小噪声大，太大步数少 |
| 序列长度 | 覆盖数据 P99 长度 | 截断会丢掉 assistant 尾部（含结束符） |
| weight decay | 0 到 0.1 | 影响小 |
| grad clip | 1.0 | 防 spike |

### SFT 与后续 RL 的关系

```
SFT 太少 / 数据差  → RL 起点格式不稳，reward 稀疏，训练慢
SFT 太多 / 过拟合  → 策略 entropy 低，rollout 多样性差，GRPO 的 group 内全对或全错，没有梯度
合适               → 格式稳定，pass@k 有梯度（pass@1 低但 pass@16 高的题目多）
```

这就是为什么 R1 只用几千条做 cold start。Week 3 会用 pass@k 分布来判断 SFT 模型是否适合进入 RL。

### 过拟合与遗忘的表现

- dev loss 上升而 train loss 继续下降：epoch 过多。
- 通用 benchmark（MMLU 类）下降超过 1 到 2 点：数据分布太窄，或 lr 太大。
- 回答变短、拒答变多、风格单一：数据里某类样本占比过高。
- 模型复读训练集里的具体句子：数据重复未去重。

## 实现步骤

### 路径 A：TRL SFTTrainer（单机、快速迭代）

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

model_id = "Qwen/Qwen2.5-1.5B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16", attn_implementation="flash_attention_2")

ds = load_dataset("allenai/tulu-3-sft-mixture", split="train[:50000]")   # messages 格式

cfg = SFTConfig(
    output_dir="out/sft-1.5b",
    max_length=4096,
    packing=True,                      # 见下文 packing 注意点
    assistant_only_loss=True,          # 只对 assistant 段算 loss；参数名随版本变化
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=2,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    eval_strategy="steps", eval_steps=200,
    report_to="wandb",
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, eval_dataset=ds.select(range(500)), processing_class=tok)
trainer.train()
```

`assistant_only_loss` 依赖 chat template 里的 `{% generation %}` 标记（较新 transformers 版本）；如果模型模板没有这个标记，要自己写 collator 生成 labels。验证 mask 的方法：取一条样本，把 `labels != -100` 的 token 解码出来看是不是恰好等于所有 assistant 回复加结束符。

### 路径 B：verl 的 SFT trainer（与后续 RL 同一套代码）

```bash
torchrun -m verl.trainer.fsdp_sft_trainer \
  data.train_files=$HOME/data/sft/train.parquet \
  data.val_files=$HOME/data/sft/val.parquet \
  data.multiturn.enable=true \
  data.multiturn.messages_key=messages \
  data.max_length=4096 \
  model.partial_pretrain=Qwen/Qwen2.5-1.5B \
  optim.lr=1e-5 \
  trainer.total_epochs=2 \
  trainer.default_local_dir=out/verl-sft
```

参数名以 verl 当前版本为准（B06）。verl 的多轮 mask 实现和 TRL 独立，两边跑同一条样本对比 `loss_mask` 是 Day 8 实验 1。

### packing 的三个注意点

1. 必须用 varlen attention（`flash_attn_varlen_func`）或 position_ids 重置 + 正确的 attention mask，否则样本之间互相可见。TRL 较新版本的 packing 默认走 varlen；旧版本要检查。
2. packing 后"batch size"变成 token 数，梯度累积按 token 数配。
3. 长样本会被截断跨两个 pack，截断处的 assistant 尾部丢失结束符。设置 `max_length` 覆盖 P99 长度，超长样本直接丢弃而不是截断。

### 长上下文 SFT

序列 32k 以上时 activation 主导显存，做法：CP（Day 5）、序列打包成大 pack、`gradient_checkpointing`、分块 cross-entropy。verl 与 TorchTitan 都支持 CP，TRL 需要外接 DeepSpeed-Ulysses（B11）。

## 实验

### 实验 1：mask 正确性交叉验证

同一条 3 轮工具调用对话，分别用 TRL collator 和 verl 的 `multiturn` 处理，把两边的 `loss_mask` 解码后 diff。预期完全一致；不一致处通常在结束符和 `assistant\n` 提示符上。资源：CPU 即可。

### 实验 2：结束符不算 loss 的后果

用 500 条数据训两个 0.5B 模型，一个 mask 掉 `<|im_end|>`，一个不 mask。推理时统计 `finish_reason == "length"` 的比例。预期前者接近 100%。单卡 20 分钟。

### 实验 3：epoch 与 lr 扫描

1.5B 模型，5 万条数据，网格 {lr 5e-6, 1e-5, 2e-5} × {epoch 1, 2, 3}，记录 dev loss、IFEval、MMLU 子集。预期 lr 2e-5 + 3 epoch 的 MMLU 下降最明显。单卡 24GB 用 LoRA 替代（Day 9），4 卡全参。

### 实验 4：SFT 强度对 RL 起点的影响（为 Week 3 准备）

对 GSM8K 训练集用不同 SFT 强度（epoch 1 vs 5）的模型采样 16 条 rollout / 题，统计 pass@1、pass@16 和 group 内全对全错的比例。预期 epoch 5 的 pass@1 略高但全对全错比例更高，进入 GRPO 后有效样本更少。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 推理不停生成 | 结束符没算 loss，或模板与训练不一致 | 解码 labels 看有无结束符 | mask 含结束符；推理用同一模板 |
| loss 一开始就很低（<0.5） | user/system token 也算了 loss（大量可预测的模板 token） | 看 labels 中 -100 比例 | 修 mask |
| loss 突然降到接近 0 | 数据重复或答案泄漏在 prompt 里 | 去重，检查 prompt/response 重叠 | 去重、去污染（Day 10） |
| 多卡训练 loss 曲线锯齿 | packing 后各 rank token 数差异大 | 打印每 rank 的 token 数 | 按 token 数均衡 pack |
| 通用能力下降 | lr 大、epoch 多、数据窄 | benchmark 对比 base | 降 lr、混入通用数据、减 epoch |
| 工具调用 JSON 格式错 | tool 返回被算了 loss 或 schema 变动 | 检查 tool 段 mask | tool observation 全 mask |
| 显存随步数增长 | 动态 padding 到 batch 内最长 | 记录每步 seq_len | packing 或按长度分桶 |
| 长样本训练后短回答变差 | 长样本占比过高 | 长度分布直方图 | 重新配比 |

## 验收标准

- 能对任意 chat template 手工标出 loss mask 边界并写代码验证。
- 给出一份含依据的 SFT 超参表，并能说出模型从 1.5B 换到 70B 时哪三个要改。
- TRL 与 verl 两条路径各产出一个 checkpoint，且实验 1 的 mask diff 为空。
- 能解释 SFT 强度如何影响 RL 起点，并用实验 4 的数据支持。

## 交付物

| 文件 | 内容 |
|---|---|
| `sft-mask-check.py` | 解码 labels 验证 mask 的脚本与输出 |
| `sft-hparam-table.md` | 超参表 + 每项依据 |
| `sft-sweep-results.md` | 实验 3 的网格结果与结论 |
| `sft-rl-readiness.md` | 实验 4 的 pass@k 分布 |

## 参考

- T23 InstructGPT：SFT → RM → PPO 三阶段的原始定义。
- T14 Llama 3 第 4 章：工业规模 SFT 数据与迭代轮次。
- T17 DeepSeek-R1 第 2.3 节：cold start SFT 的数量与目的。
- T15 Qwen3 第 4 章：两次 SFT 各自解决什么。
- T22 Tulu 3：开源 SFT 数据配比与消融。
- B05 TRL 文档、B06 verl 文档：参数名以此为准。

## 附录 A：自己写 loss mask collator（模板没有 generation 标记时）

```python
import torch

def build_labels(tok, messages, max_length):
    """逐轮渲染模板，assistant 段（含结束符）标 label，其余 -100。"""
    input_ids, labels = [], []
    for i, m in enumerate(messages):
        # 渲染到第 i 轮为止的前缀，与到第 i-1 轮的前缀做差得到本轮 token
        prefix_prev = tok.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=False) if i else []
        prefix_now = tok.apply_chat_template(messages[:i+1], tokenize=True, add_generation_prompt=False)
        seg = prefix_now[len(prefix_prev):]
        if m["role"] == "assistant":
            # 本轮开头的 "<|im_start|>assistant\n" 属于提示，不算 loss
            head = tok.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=True)[len(prefix_prev):]
            seg_labels = [-100] * len(head) + seg[len(head):]
        else:
            seg_labels = [-100] * len(seg)
        input_ids += seg; labels += seg_labels
    input_ids, labels = input_ids[:max_length], labels[:max_length]
    return torch.tensor(input_ids), torch.tensor(labels)

# 验证
ids, lab = build_labels(tok, sample["messages"], 4096)
print(tok.decode(ids[lab != -100]))   # 应恰好是所有 assistant 回复 + 结束符
assert (lab != -100).sum() > 0
```

前缀相减的做法依赖模板渲染是"前缀稳定"的：加一轮不会改写前面的内容。多数模板满足，少数模板会在最后一轮加特殊标记（比如只在末轮加 `<|eot|>`），要单独处理。tool 角色的消息走 else 分支，自然被 mask。

## 附录 B：样本格式约定

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with tools."},
    {"role": "user", "content": "北京今天天气？"},
    {"role": "assistant", "content": null,
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\": \"北京\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 21, \"cond\": \"晴\"}"},
    {"role": "assistant", "content": "北京今天晴，21 度。"}
  ],
  "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"...": "..."}}}],
  "source": "synthetic-v3", "quality_score": 0.91
}
```

- `tools` 字段进 system 段由模板渲染，训练和推理必须用同一份 schema 渲染方式。
- `source` 和 `quality_score` 不进模型，用于 Day 10 的配比与过滤。
- 一个文件一个 schema 版本，schema 变更时升版本号，避免新旧混训。

## 附录 C：SFT 上线前 checklist

1. 随机抽 20 条样本解码 labels，肉眼确认 mask。
2. 训练前用 base 模型算一次 dev loss 作为起点；SFT 后 dev loss 应明显下降但不接近 0。
3. 训练后用与训练相同的 chat template 在 vLLM 上推理 50 条，检查 `finish_reason` 分布和格式。
4. 跑 3 个通用 benchmark 子集与 base 对比，下降超过 2 点要回查数据配比。
5. 记录：数据版本、模板版本、tokenizer hash、超参、seed、commit。缺一项就不算可复现。
