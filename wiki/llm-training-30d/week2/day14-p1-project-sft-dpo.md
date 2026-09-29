---
title: "Day 14：P1 项目：SFT + DPO 完整流水线"
type: concept
tags: [llm-training, sft, dpo, project, model-merging]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 14：P1 项目：SFT + DPO 完整流水线

> 上一课 [[llm-training-30d/week2/day13-evaluation-experiment-management]] · 下一课 [[llm-training-30d/week3/day15-rollout-and-rl-basics]]。本周内容：[[llm-training-30d/week2/day08-sft-engineering]]、[[llm-training-30d/week2/day09-lora-qlora-peft]]、[[llm-training-30d/week2/day10-data-engineering-synthetic]]、[[llm-training-30d/week2/day11-preference-optimization-dpo]]、[[llm-training-30d/week2/day12-reward-models]]。

## 学习目标

1. 独立跑通"基座 → SFT → on-policy 偏好对构造 → DPO → 评测"的完整流水线，每一步有可复现的配置与产物。
2. 用 Day 13 的评测协议给出 SFT 与 DPO 各自的净收益及置信区间，而不是单点分数。
3. 至少完成两个消融，并能解释结果的机制。
4. 交付一份可以交给下一周 RL 阶段直接使用的 SFT checkpoint 和数据版本记录。
5. 知道模型合并与 continual pretraining 在流水线中的位置，能说清什么时候用。

## 工业现状

SFT 加偏好优化仍然是所有公开配方的前两步。Tulu 3（T22）的流水线是 SFT → DPO（用 on-policy 采样构造的偏好对）→ RLVR，并且报告了每一步在各 benchmark 上的增量。Llama 3（T14）做多轮 SFT 与 DPO 迭代，每一轮用最新模型重新采样偏好数据。Qwen3（T15）在长 CoT 冷启动后走推理 RL，再用 rejection sampling 数据做第二轮 SFT。共同点有三个：偏好数据尽量 on-policy（用当前模型采样，而不是拿别人的数据集）；SFT 数据经过 rejection sampling 或 judge 过滤；每一步都有独立评测。

这个项目按同样的骨架缩小到单卡可完成：1.5B 到 8B 模型、几万条 SFT 样本、几千对偏好数据。规模缩小不改变流程和踩坑点。

## 核心原理

流水线的数据流：

```text
基座模型 M0
   │  SFT 数据 D_sft（过滤后，含 loss mask）
   ▼
SFT 模型 M1  ──采样 rollout（每个 prompt n=4 到 8）──▶ 候选回答集合
   │                                                    │ RM / verifier / judge 打分
   │                                                    ▼
   │                                        偏好对 (prompt, chosen, rejected)，来自同一 prompt 的 rollout
   ▼
DPO 模型 M2（reference = M1）
   │
   ▼
评测：M0 / M1 / M2 在 dev + hidden 上的 avg@n、配对区间
```

rollout 在这里的含义：SFT 模型对同一 prompt 采样多条回答，用来构造偏好对。这是 on-policy 偏好数据的来源，也是 Week 3 RL 中 rollout 的雏形。区别是这里只用它筛数据，Week 3 用它直接算梯度。

三个必须理解的点：

1. **DPO 的 reference 是 M1 不是 M0**。DPO 的隐式奖励是 β·log(π/π_ref)，reference 换了，优化目标就换了。用 M0 做 reference 会让 DPO 同时在学 SFT 的内容和偏好，损失解释不清。
2. **偏好对必须来自同一 prompt 的 rollout**。chosen 与 rejected 的差异应该只在质量，不在话题和长度分布。跨 prompt 配对会引入长度偏置，DPO 会学成"越长越好"。
3. **评测在 SFT 后和 DPO 后各做一次**。DPO 常见的结果是对话质量（Arena-Hard 类）上升而数学、代码持平或略降；只看一个集合会误判。

## 实现步骤

### 项目规格

| 项 | 要求 |
|---|---|
| 模型 | Qwen2.5-1.5B（单卡 24GB，全参 SFT 可行）或 Qwen2.5-7B / Llama-3.1-8B（LoRA SFT，QLoRA 可在 24GB 上做） |
| SFT 数据 | 20k 到 50k 条，来自 Day 10 的过滤流水线：至少两个来源混合，经过去重、去污染、judge 过滤 |
| 偏好数据 | 3k 到 8k 对，全部由 M1 采样构造，打分器用 Day 12 训练的 RM 或 verifier（数学/代码）加 judge（开放题） |
| 评测 | Day 13 的 eval-protocol：dev 集每 epoch 一次；hidden 集只在 M0、M1、M2 各跑一次 |
| 资源 | 单卡 24 到 48GB 全流程可完成；多卡时用 FSDP2 全参 |
| 时间 | 一天 10 小时：SFT 3 小时，采样与打分 2 小时，DPO 2 小时，评测与报告 3 小时 |

### 步骤 1：SFT

TRL 路径（以当前版本文档为准）：

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
cfg = SFTConfig(
    output_dir="runs/p1-sft", num_train_epochs=2, learning_rate=1e-5,   # 全参；LoRA 用 1e-4 量级
    per_device_train_batch_size=4, gradient_accumulation_steps=8,
    max_length=4096, packing=True, bf16=True, warmup_ratio=0.03,
    lr_scheduler_type="cosine", logging_steps=10, save_strategy="epoch",
    assistant_only_loss=True,     # 只对 assistant token 算 loss；字段名随版本变化
    seed=0, report_to="wandb",
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                     peft_config=LoraConfig(r=64, lora_alpha=128, target_modules="all-linear") if use_lora else None)
trainer.train()
```

检查项：packing 开启时确认 attention 没有跨样本（看 `position_ids` 是否在样本边界重置）；loss mask 生效（打印一条样本的 labels，user 部分应为 -100）；第一个 epoch 结束时在 dev 集上跑评测。

### 步骤 2：on-policy 采样与打分

```python
from vllm import LLM, SamplingParams
llm = LLM("runs/p1-sft/final", dtype="bfloat16")
sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048, n=6, seed=0)
outs = llm.generate([apply_chat_template(p) for p in pref_prompts], sp)
# 每个 prompt 6 条 rollout → 打分 → 取最高分与最低分组成一对；分差小于阈值的 prompt 丢弃
```

打分规则：数学、代码 prompt 用 verifier 给 0/1，再用长度做 tie-break（更短优先，抑制长度偏置）；开放题用 RM 分数，分差低于 RM 分数标准差的 0.5 倍丢弃。记录丢弃比例，它反映 M1 在这批 prompt 上的区分度。

### 步骤 3：DPO

```python
from trl import DPOTrainer, DPOConfig
cfg = DPOConfig(
    output_dir="runs/p1-dpo", beta=0.1, learning_rate=5e-7,   # DPO 学习率比 SFT 低一个量级以上
    num_train_epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=16,
    max_length=4096, max_prompt_length=1024, bf16=True, logging_steps=10,
    loss_type="sigmoid",          # 标准 DPO；可切 ipo / kto 等做消融
    rpo_alpha=None,               # 加 SFT 正则项时设为 1.0 量级，缓解 chosen 似然下降
    seed=0,
)
trainer = DPOTrainer(model=m1, ref_model=None, args=cfg, train_dataset=pref_ds, processing_class=tok)
trainer.train()
```

`ref_model=None` 时 TRL 会复制当前模型作 reference，也就是 M1。LoRA 训练时 reference 是关闭 adapter 的同一模型，省一份显存。

训练中盯三条曲线：`rewards/margins`（应上升）、`rewards/chosen`（不应大幅下降，下降说明似然坍缩）、`logps/chosen`。

### 步骤 4：评测与报告

M0、M1、M2 各跑 dev 与 hidden；用 Day 13 的配对 bootstrap 给 M1 vs M0、M2 vs M1 的区间。报告表：

| 模型 | GSM8K avg@4 | IFEval strict | 代码 pass@1 | Arena-Hard 风格 judge 胜率 vs M0 | 平均输出长度 |
|---|---|---|---|---|---|
| M0 | | | | 50% | |
| M1 | | | | | |
| M2 | | | | | |

### 步骤 5：两个消融（必做）

- **消融 A：偏好对来源**。用同样规模的公开偏好数据集（off-policy）训练 DPO，与 on-policy 版本对比。预期 on-policy 更好，且长度增长更小。
- **消融 B：β**。β = 0.05 / 0.1 / 0.3，看 margin、KL（用 logps 差估计）与评测的关系。预期 β 小时 margin 大但评测可能下降。

可选消融：LoRA vs 全参 SFT（学习率按 Day 9 的等价条件调整）；SFT epoch 1 vs 3。

### 步骤 6：交接给 Week 3

保存 `runs/p1-sft/final` 作为 Week 3 RL 的起点，附上数据版本 hash、评测结果与 seed。Week 3 的 GRPO 会以它为初始策略和 reference。

## 模型合并与 continual pretraining（知道即可）

**模型合并**：把多个同基座的微调模型按权重合并，常见方法线性平均、TIES（剪掉小幅更新并解决符号冲突）、DARE（随机丢弃更新再放大）。工业上用于合并不同能力方向的 SFT 分支，或把 DPO 与 SFT 模型插值以控制风格漂移。用 mergekit 之类工具一行配置可做。合并后必须重新评测，收益不保证。

**continual pretraining**：在后训练之前，用领域语料（代码、医学、多语言）继续做预训练目标训练，学习率比原预训练低、混入一定比例通用数据防遗忘。它属于"中期训练"，Qwen3、Llama 3 的长上下文扩展与领域增强都在这一阶段做。本课程不做实验，但流水线设计时要知道领域知识应在这里注入，而不是靠 SFT 硬灌。

## 实验

项目本身就是实验。额外要求记录：

- SFT 期间每 epoch 的 dev 曲线与训练 loss，判断是否过拟合（dev 下降而 loss 继续降）。
- 采样阶段每个 prompt 6 条 rollout 的分数分布，报告"全对/全错/有区分"三类 prompt 的比例。
- DPO 训练前后 M1 与 M2 在 200 条 prompt 上的输出长度分布图。

资源分层：API-only 学员用 API 模型做打分器和 judge，SFT 与 DPO 用 0.5B 模型在 CPU 或免费 GPU 上缩小规模完成，流程不变。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| SFT 后 dev 分数低于 M0 | 数据质量差；lr 过大；loss mask 没生效 | 看 labels；对比 lr 一半 | 重跑过滤；降 lr；修 mask |
| SFT loss 很低但评测不涨 | packing 跨样本泄漏；数据与评测分布不符 | 检查 position_ids；抽样看数据 | 修 packing；调整数据配比 |
| 偏好对丢弃比例超过 70% | M1 在这批 prompt 上没有区分度（太难或太简单） | 看 6 条 rollout 分数分布 | 换 prompt 难度；提高 n |
| DPO 后输出长度暴涨 | 偏好对里 chosen 系统性更长 | 统计 chosen/rejected 长度差 | 长度 tie-break；SimPO 的长度归一化；限制长度差 |
| `rewards/chosen` 持续下降 | 似然坍缩，DPO 在同时压低 chosen 和 rejected | 看 logps/chosen 曲线 | 加 SFT 正则（rpo_alpha）；降 lr；减 epoch |
| DPO 后数学分数下降 | 偏好数据缺少推理类 prompt；β 太小 | 按类别分解评测 | 混入 verifier 打分的推理偏好对；提 β |
| M2 与 M1 无显著差异 | 偏好对太少或噪声大 | margin 曲线是否上升；判分器一致性 | 扩大数据；用更强判分器 |
| 显存 OOM | 序列长度 4096 全参 + DPO 同时算两条序列 | 看 max_memory_allocated | 减 batch、开 gradient checkpointing、LoRA |

## 验收标准

- 三个模型的评测表带置信区间，且 M2 vs M1 至少在一个维度显著提升、无维度显著下降超过区间。
- 两个消融有数据和机制解释。
- 数据版本、配置、seed、checkpoint 一一对应，他人能按文档复现。
- SFT checkpoint 已归档并附交接说明。

## 交付物

| 文件 | 内容 |
|---|---|
| `p1-report.md` | 流水线图、数据统计、三模型评测表、消融结果、失败记录与费用 |
| `configs/` | SFT、采样、DPO 的完整配置 |
| `data/` | 数据版本记录（hash、来源、过滤日志、污染检测结果） |
| `runs/p1-sft/final` | 交接给 Week 3 的 checkpoint |

## Week 2 面试题

1. **SFT 为什么只对 assistant token 算 loss？** 用户与系统 token 不是模型要学的输出；对它们算 loss 会让模型学会复述 prompt，浪费容量并干扰 chat 行为。
2. **LoRA 和全参在什么条件下等价？** 任务需要的更新秩不超过 LoRA 容量、学习率按 LoRA 约 10 倍于全参调整、batch 不过大时，SFT 与 RL 的效果接近；大规模多任务 SFT 或需要注入大量新知识时全参更好（B04）。
3. **QLoRA 省的是哪部分显存？** 基座权重以 4-bit 存储，省的是权重显存；梯度和优化器状态只在 LoRA 参数上，本来就小。activation 不省。
4. **rejection sampling 生成 SFT 数据时最大的风险是什么？** verifier 或 judge 的盲区被放大：模型会学到"骗过判分器"的模式。要用 hidden 判分器抽检并做去污染。
5. **DPO 的隐式奖励是什么？为什么需要 reference？** β·log(π(y|x)/π_ref(y|x))；没有 reference 时目标退化为最大化 chosen 与 rejected 的似然差，会无限压低 rejected，SimPO/ORPO 用长度归一化或 odds ratio 来替代 reference 的约束。
6. **on-policy 偏好数据为什么效果更好？** 偏好对处于当前策略的分布内，梯度信号直接作用在模型会生成的内容上；离线数据的 chosen 可能是模型根本不会生成的风格，学到的是分布外的差异。
7. **Bradley-Terry RM 的过优化怎么发现？** 用 RM 分数与 hidden judge 或人工评估画曲线，RM 分数持续上升而后者下降的拐点就是过优化点；KL 到 reference 的距离是常用横轴。
8. **PRM 与 ORM 分别适合什么？** ORM 只看最终结果，训练数据便宜，适合可验证任务；PRM 给每步打分，用于搜索和 credit assignment，但标注贵且容易被步骤格式游戏。
9. **一个 benchmark 提升 2 分，什么时候可以相信？** 当 2 分超过该集合在当前 n 下的 2 倍标准误，且配对 bootstrap 区间不跨 0，且该集合已做污染检测。
10. **什么时候用 DPO，什么时候直接上 RLVR？** 有可验证 reward（数学、代码、格式）时 RLVR 更直接、上限更高；主观质量（风格、帮助性、安全）没有 verifier 时用 RM + DPO 或 RM + PPO；工业配方通常两者都用，DPO 在前处理通用偏好，RLVR 在后强化推理。

## Week 3 预览

Week 3 把 rollout 从"筛数据"变成"直接产生梯度"：Day 15 讲 rollout 与策略梯度的精确定义，Day 16 讲 PPO/GRPO/DAPO/GSPO 的目标函数，Day 17 讲 verifier 与 reward 设计，Day 18 到 20 讲 RL 训练系统：框架架构、rollout 引擎与权重同步、异步 RL 与稳定性。Day 21 用 verl 在数学任务上跑 GRPO。本周的 SFT checkpoint、评测协议和 RM 都会被直接复用。

## Week 2 知识图谱

```text
                    ┌──────────────── 数据工程（Day 10）────────────────┐
                    │ 来源 → 去重/去污染 → judge/verifier 过滤 → 配比 │
                    └───────────────┬─────────────────────────────────┘
                                    ▼
   Day 8 SFT ──────────── 全参 / LoRA（Day 9）──────────▶ M1
   loss mask · packing     rank/alpha · 等价条件            │
                                                            │ 采样 rollout（n 条/prompt）
                                                            ▼
   Day 12 RM / verifier ──打分──▶ 偏好对 (chosen, rejected) 同 prompt
   BT · generative · PRM/ORM                                │
   过优化 / Goodhart                                        ▼
   Day 11 DPO 家族 ────── reference = M1 ──────────────▶ M2
   隐式奖励 · β · 长度偏置                                   │
                                                            ▼
   Day 13 评测 ──── avg@n · 配对区间 · 污染检测 · dev/hidden 分离
```

## 提交前自查清单

- [ ] SFT 样本打印检查过 labels，user/system 位置为 -100
- [ ] packing 下 position_ids 在样本边界重置
- [ ] 偏好对全部来自 M1 的 rollout，chosen/rejected 长度差分布已记录
- [ ] DPO 的 reference 是 M1
- [ ] `rewards/chosen` 曲线没有持续下降
- [ ] hidden 集只跑了三次（M0、M1、M2）
- [ ] 每个分数带 n、温度、判分器版本与置信区间
- [ ] 两个消融各自只改一个变量
- [ ] 费用与 GPU 小时已记录
- [ ] 交接 checkpoint 附数据版本 hash

## 参考

- T22 Tulu 3：SFT → DPO → RLVR 的完整开源配方与每步增量。
- T14 Llama 3：多轮 SFT/DPO 迭代与 on-policy 偏好数据。
- T27 DPO、T29 SimPO：目标函数与长度偏置的处理。
- T15 Qwen3：rejection sampling SFT 在推理 RL 之后的用法。
- B05 TRL 文档：SFTTrainer 与 DPOTrainer 的参数。
