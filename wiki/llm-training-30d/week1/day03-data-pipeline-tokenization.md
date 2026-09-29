---
title: "Day 3：数据流水线、tokenization、chat template 与 packing"
type: concept
tags: [llm-training, data-pipeline, tokenizer, chat-template, packing, loss-mask]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 3：数据流水线、tokenization、chat template 与 packing

> 上一课 [[llm-training-30d/week1/day02-training-loop-memory-math]] · 下一课 [[llm-training-30d/week1/day04-data-parallel-zero-fsdp]]。相关：推理课 [[ai-infra-30d/week2/day08-vllm-architecture]]（推理侧的 tokenize 与 chat template 在同一份 tokenizer 配置里，训练与推理不一致是常见 bug 来源）。

## 学习目标

1. 说清一条 SFT 样本从 JSON 到 `input_ids / labels / position_ids / attention` 的完整变换，并能验证 loss mask 只覆盖 assistant token。
2. 实现 sequence packing，并证明样本之间没有互相看见（attention 隔离）。
3. 量化 padding 浪费与 packing 收益，理解长度分布如何决定 batch 策略。
4. 知道预训练与后训练数据流水线的差异，以及数据配比、shuffle、epoch 对训练的影响。

## 工业现状

后训练数据流水线的三个共识：

- **只对 assistant token 算 loss**。system、user、tool 返回的 token 全部 mask。Llama 3（T14）、Tulu 3（T22）、Qwen3（T15）都是这样做的；TRL 的 `SFTTrainer` 通过 `assistant_only_loss`（新版）或 `DataCollatorForCompletionOnlyLM`（旧版）实现，具体参数名以当前版本文档为准。
- **packing 是默认**。把多条短样本拼成一条固定长度序列，配合 FlashAttention 的 varlen 接口做块对角 attention，避免跨样本污染。TRL、Axolotl、verl 的 SFT 都支持；不隔离的"naive packing"会让模型学到跨样本的伪相关，在多轮数据上尤其明显。
- **chat template 是模型的一部分**。它写在 `tokenizer_config.json` 的 `chat_template` 字段（Jinja），训练用什么模板，推理就必须用同一份。工具调用、reasoning 段（`<think>`）、多轮的 special token 都由它定义。

预训练数据流水线（去重、质量过滤、配比、tokenize 成二进制分片、多 epoch 策略）规模大得多，本课只讲与后训练相关的部分，"知道即可"。

## 核心原理

### 从消息列表到训练张量

```text
messages = [system, user, assistant, user, assistant]
   │ apply_chat_template（Jinja，插入 <|im_start|>role\n ... <|im_end|> 等 special token）
   ▼
text ──tokenize──▶ input_ids [T]
   │
   ▼ 按 role 边界生成
labels [T]：assistant 段 = input_ids，其余 = -100（PyTorch CE 忽略）
   │ 右移一位在模型内部完成（labels[t] 预测的是 input_ids[t]，HF 内部 shift）
   ▼
attention：因果 mask；packing 时改为块对角
position_ids：每条样本从 0 开始（packing 时重置）
```

关键细节：

- **labels 的边界要包含结束 token**。assistant 段末尾的 `<|im_end|>` 必须在 loss 里，否则模型学不会停，推理时会一直生成到 max_tokens。这是最常见的 SFT bug。
- **多轮样本的 mask**：同一条对话里所有 assistant 轮都算 loss（除非数据质量策略只保留最后一轮）。工具调用轮里 assistant 的 tool call 文本算 loss，tool 返回不算。
- **`-100`** 是 PyTorch `CrossEntropyLoss(ignore_index=-100)` 的约定，HF 沿用。

### packing 与 attention 隔离

```text
不 packing（padding 到 max_len=4096）：
  样本 A（900 token）+ 3196 pad   ← 78% 浪费
  样本 B（1500）     + 2596 pad

packing：
  [A(900) | B(1500) | C(1200) | D(396)]  = 4096，无 pad
  attention 必须是块对角：
      A  B  C  D
   A  ■
   B     ■
   C        ■
   D           ■
  实现：flash_attn_varlen_func(q, k, v, cu_seqlens=[0, 900, 2400, 3600, 3996], max_seqlen=1500)
  position_ids = [0..899, 0..1499, 0..1199, 0..395]
```

HF Transformers 在 `attn_implementation="flash_attention_2"` 下，如果传入的 `position_ids` 中出现从 0 重启，会自动推导 `cu_seqlens` 走 varlen 路径（版本相关，以当前实现为准）。SDPA 后端则需要显式的块对角 mask，效率低。这是"packing 必须配 FlashAttention"的原因。

"naive packing"（拼接后用普通因果 mask）的危害：样本 B 能看到样本 A，模型学到"前文无关内容会影响回答"，在多轮和长上下文任务上表现为幻觉式引用前面样本。预训练常用 naive packing 是因为文档间本来就有分隔符且数据量大，后训练不可以。

### 长度分布决定策略

```text
统计 token 长度分布：P50、P90、P99、max
- P99 远小于 max_len：packing 收益大，pad 浪费 = 1 - mean/max_len
- 长尾样本（超过 max_len）：截断（丢尾巴会切掉答案）还是丢弃？后训练通常丢弃或拆分
- 长度分桶（length grouping）：不 packing 时把长度相近的样本放一个 batch，减少 pad
```

### 预训练 vs 后训练数据流

| 项 | 预训练 | 后训练 |
|---|---|---|
| 单位 | 文档，拼接后切固定长度 | 对话，整条不切 |
| loss | 所有 token | assistant token |
| packing | naive 拼接 + 分隔符 | 隔离 packing |
| 去重 | MinHash 文档级 | 精确 + 语义去重，还要与评测集去污染 |
| 配比 | 按领域比例采样，多 epoch 时高质量数据 upsample | 按任务类型配比，通常 1 到 3 epoch |
| 存储 | tokenize 成 uint16/uint32 二进制分片，mmap 读取 | JSONL 即可，训练时在线 tokenize |
| shuffle | 分片级 + 分片内 | 全局 shuffle，固定 seed |

"知道即可"：预训练的 tokenize 是离线一次性的，因为 15T token 在线 tokenize 会成为瓶颈；后训练几十万条在线做就行。

### 数据配比与 curriculum

后训练常见做法：把各来源（数学、代码、通用对话、安全、多语言、agent 轨迹）按比例混合后全局 shuffle。curriculum（先易后难、先通用后领域）在 SFT 里收益不稳定，在 RL 里更有用（Day 17）。多 epoch SFT 的经验：1 到 2 epoch 通常最好，3 epoch 以上小数据集会过拟合，表现为 eval loss 上升但训练 loss 继续下降。

## 实现步骤

### 1. 检查 chat template 与 loss mask

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
msgs = [{"role":"system","content":"You are helpful."},
        {"role":"user","content":"1+1=?"},
        {"role":"assistant","content":"2"},
        {"role":"user","content":"再加 1"},
        {"role":"assistant","content":"3"}]
print(tok.apply_chat_template(msgs, tokenize=False))       # 看 special token 的位置
ids = tok.apply_chat_template(msgs, tokenize=True)
# 新版 transformers 支持 return_assistant_tokens_mask=True（需要模板有 generation 标记）
enc = tok.apply_chat_template(msgs, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
labels = [i if m else -100 for i, m in zip(enc["input_ids"], enc["assistant_masks"])]
print(tok.decode([i for i in labels if i != -100]))         # 应只有 "2<|im_end|>" 和 "3<|im_end|>"
```

如果模板没有 `{% generation %}` 标记，`assistant_masks` 全 0，就要自己按 role 边界的 token offset 生成 mask。写一个通用函数：逐轮渲染前缀，取长度差作为该轮 token 区间。

### 2. 用 TRL SFTTrainer 做 packing

```python
from trl import SFTTrainer, SFTConfig
cfg = SFTConfig(
    output_dir="out", max_length=4096,
    packing=True,                      # 新版默认走 varlen 隔离 packing（bfd 策略），以文档为准
    assistant_only_loss=True,          # 仅 assistant token 算 loss（参数名随版本变化）
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    bf16=True, learning_rate=1e-5, num_train_epochs=2,
    dataset_num_proc=8,
)
trainer = SFTTrainer(model="Qwen/Qwen2.5-1.5B", train_dataset=ds, args=cfg)  # ds 有 "messages" 列
```

训练前先 `trainer.train_dataset[0]` 看一条打包后的样本，确认 `position_ids` 有重启、`labels` 有 -100。

### 3. 手写 varlen packing（理解用）

```python
import torch
from flash_attn import flash_attn_varlen_func
def pack(samples, max_len):
    ids, labels, pos, cu = [], [], [], [0]
    for s in samples:                       # 贪心 first-fit，或先按长度排序
        if len(ids) + len(s["ids"]) > max_len: break
        ids += s["ids"]; labels += s["labels"]; pos += list(range(len(s["ids"])))
        cu.append(len(ids))
    return dict(input_ids=torch.tensor(ids), labels=torch.tensor(labels),
                position_ids=torch.tensor(pos), cu_seqlens=torch.tensor(cu, dtype=torch.int32))
```

验证隔离：把样本 A 的内容换成随机 token，检查样本 B 上每个位置的 logits 是否完全不变（应完全一致到 BF16 精度）。

### 4. 长度统计

```python
import numpy as np
lens = np.array([len(tok.apply_chat_template(m, tokenize=True)) for m in ds["messages"]])
print(np.percentile(lens, [50, 90, 99]), lens.max(), (lens > 4096).mean())
print("pad waste without packing:", 1 - lens.clip(max=4096).mean() / 4096)
```

## 实验

### 实验 1：loss mask 正确性

固定：Qwen2.5-1.5B-Instruct 的模板，50 条多轮样本。检查：解码 labels 里非 -100 的 token，应恰好是各轮 assistant 内容加结束 token。统计 assistant token 占总 token 的比例（典型 30% 到 60%）。故意把结束 token 从 labels 里去掉训练 200 步，推理时观察是否停不下来。

### 实验 2：naive packing vs 隔离 packing

固定：同一数据、同一 seed、500 步。对比 naive（拼接 + 普通因果 mask）与 varlen 隔离。指标：训练 loss 曲线、held-out 单条样本（不 packing）的 eval loss、一个多轮任务的准确率。预期：naive 的训练 loss 更低（能"偷看"），eval loss 更高。

### 实验 3：packing 的吞吐收益

固定 global batch token 数。对比：padding 到 4096、length grouping、packing。记录 tokens/s（只算非 pad token）和显存。预期 packing 吞吐提升与 pad 浪费比例一致。

### 实验 4：训练与推理模板一致性

用训练时的 tokenizer 渲染一条对话，再用 vLLM 的 `/v1/chat/completions` 发同一对话，开 `echo` 或对比 prompt token 数。预期完全一致；不一致的常见原因是训练时手工拼接了模板或 system prompt 默认值不同。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 推理时模型不停止 | 结束 token 未进 loss | 解码 labels 看末尾 | mask 边界包含 `<|im_end|>` / eos |
| 模型回答里出现别的样本内容 | naive packing | 检查 attention 是否块对角 | FlashAttention varlen + position_ids 重置 |
| 训练 loss 异常低 | labels 没 mask，模型在学 user 文本；或标签泄漏 | 统计 -100 比例 | 修 mask |
| eval loss 上升而训练 loss 下降 | epoch 过多、数据太少 | 画两条曲线 | 1 到 2 epoch；加数据 |
| 训练与推理输出风格不一致 | 模板不一致、system prompt 默认值不同 | 实验 4 | 统一 tokenizer 与模板文件 |
| GPU 利用率低、数据加载慢 | 在线 tokenize 单进程 | profiler 看 dataloader 等待 | `dataset_num_proc`、预 tokenize、`num_workers` |
| 长样本被截断丢掉答案 | 截断策略从右截 | 检查被截样本 | 丢弃或按轮拆分，不要截断 assistant |
| 多轮数据里工具返回被算 loss | mask 只按 assistant/user 区分，tool role 处理错 | 解码 labels | tool 角色一律 -100 |

## 思考题

1. 一条 8 轮对话，每轮 assistant 都算 loss，和把它拆成 8 条"前缀 + 最后一轮"样本，训练效果和算力有什么区别？
2. packing 后一条序列里有 4 个样本，梯度累积的"样本数"该怎么算？loss 应该按 token 平均还是按样本平均？这对长短样本的权重有什么影响？
3. RL 的 rollout 数据（prompt + response）进训练时也需要 loss mask 吗？mask 的是什么？
4. 如果 tokenizer 的 chat template 被推理服务改过（比如加了默认 system prompt），训练好的模型上线后会怎样？

## 验收标准

- 能对任意模型的 chat template 生成正确的 assistant mask 并验证。
- 手写 packing 通过隔离测试（换掉相邻样本，logits 不变）。
- 能根据长度分布给出 max_len、packing、截断策略的决定并说明理由。

## 交付物

| 文件 | 内容 |
|---|---|
| `build_sft_dataset.py` | messages → input_ids / labels / position_ids 的转换与 mask 验证 |
| `packing.py` | varlen packing 与隔离测试 |
| `packing-ablation.md` | 实验 2、3 的数据 |
| `data-pipeline-checklist.md` | 训练前数据检查清单（模板、mask、长度、去重、去污染） |

## 参考

- T14、T22：Llama 3 与 Tulu 3 的 SFT 数据处理与 loss mask 描述。
- T05：FlashAttention-2 的 varlen 接口是隔离 packing 的基础。
- B05：TRL 的 SFTTrainer 文档，packing 与 assistant-only loss 的当前参数名以它为准。
- T31：Self-Instruct，理解合成数据的最初形态，Day 10 展开。
