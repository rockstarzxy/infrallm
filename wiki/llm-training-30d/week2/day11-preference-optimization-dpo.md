---
title: "Day 11：偏好优化：DPO 及其变体"
type: concept
tags: [llm-training, dpo, preference-optimization, simpo, kto, orpo]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 11：偏好优化：DPO 及其变体

> 上一课 [[llm-training-30d/week2/day10-data-engineering-synthetic]] · 下一课 [[llm-training-30d/week2/day12-reward-models]]。相关：RLVR 与 GRPO 在 [[llm-training-30d/week3/day16-ppo-grpo-family]]，DPO 与 RL 的分工在本页末尾。

## 学习目标

1. 从 RLHF 的 KL 约束目标出发推导 DPO 的损失，解释"隐式奖励"和 β 的含义。
2. 列出 IPO、KTO、SimPO、ORPO 与 DPO 的差异（是否需要参考模型、是否需要成对数据、长度归一化、目标函数），并说明各自适用场景。
3. 构造 on-policy 偏好对，并用实验说明它比离线偏好数据集效果好的原因。
4. 诊断 DPO 的四类常见失败：chosen 似然下降、长度膨胀、过拟合、β 选错。
5. 用 TRL DPOTrainer 跑通一轮，并能说清 2025 到 2026 年工业配方中 DPO 与 RLVR 各承担什么。

## 工业现状

- **Llama 3（T14）**：SFT 之后用 DPO 而不是 PPO 做偏好对齐，理由是稳定、省算力、效果相当；每轮用最新模型 on-policy 采样构造偏好对，共迭代六轮；对 DPO 做了两个修改：mask 掉格式 token 的 loss、加 NLL 正则项防止 chosen 似然下降。
- **Tulu 3（T22）**：SFT → DPO（用 on-policy 与离线混合的偏好数据，length-normalized DPO）→ RLVR。DPO 负责通用偏好，RLVR 负责可验证任务。
- **Qwen3（T15）、DeepSeek-R1（T17）**：推理能力主要靠 RL，通用对齐阶段仍有偏好数据参与的 RL（用 RM）；DPO 在这些报告里不是主角。
- **DeepSeek-V3（T16）**：后训练用 SFT + GRPO，偏好信号来自 RM，没有 DPO。
- **小团队与开源微调**：DPO/SimPO/ORPO 是最常用的对齐手段，因为一张卡、一个脚本就能跑，且 SimPO 不需要参考模型。

共识：DPO 适合"偏好由人或 RM 判、无法程序化验证"的对齐（风格、安全、有用性）；可验证任务（数学、代码、agent 成功率）用 RLVR 更强；两者在同一条流水线上串联而不是二选一。

## 核心原理

### 从 RLHF 目标到 DPO

RLHF 的目标（T23）：

```
max_π  E_{x~D, y~π(·|x)} [ r(x, y) ]  −  β · KL( π(·|x) || π_ref(·|x) )
```

这个目标有闭式最优解：

```
π*(y|x) = π_ref(y|x) · exp( r(x,y) / β ) / Z(x)
⇒  r(x, y) = β · log [ π*(y|x) / π_ref(y|x) ] + β log Z(x)
```

把奖励代入 Bradley-Terry 偏好模型 `P(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l))`，`Z(x)` 消掉，得到 DPO 损失（T27）：

```
L_DPO = − E [ log σ( β · ( log π_θ(y_w|x)/π_ref(y_w|x)  −  log π_θ(y_l|x)/π_ref(y_l|x) ) ) ]
```

- **隐式奖励** `r̂(x,y) = β log π_θ(y|x)/π_ref(y|x)`：训练后可以直接用它给回答打分。
- **β**：越大越贴近参考模型（KL 小），越小越激进。常用 0.05 到 0.3（示例）。
- 梯度形式：`−β · σ(r̂_l − r̂_w) · [∇log π(y_w) − ∇log π(y_l)]`，权重 `σ(r̂_l − r̂_w)` 在模型已经分对时趋近 0，这是 DPO 自带的"难例加权"。

### 变体对比

| 方法 | 参考模型 | 数据形式 | 目标 | 长度归一化 | 适用 |
|---|---|---|---|---|---|
| DPO（T27） | 需要 | 成对 (w, l) | 上式 | 无（有 length-normalized 变体） | 默认 |
| IPO | 需要 | 成对 | 把 σ 换成平方损失，目标间隔为 1/(2β) | 无 | 偏好数据噪声大、DPO 过拟合时 |
| KTO（T28） | 需要 | 非成对，每条只有"好/坏"标签 | 前景理论效用，好样本拉高、坏样本压低，相对一个 KL 基线 | 无 | 只有点赞/点踩反馈，没有成对数据 |
| SimPO（T29） | 不需要 | 成对 | 平均每 token logprob 之差减去间隔 γ | 有（内建） | 省显存、抗长度偏置 |
| ORPO（T30） | 不需要 | 成对 | SFT NLL + odds ratio 偏好项，一步完成 SFT+对齐 | 无 | 从 base 直接对齐、省一个阶段 |

### 为什么 on-policy 偏好数据更好

离线数据集（他人模型生成的 chosen/rejected）和当前模型分布相差很远：rejected 往往是当前模型根本不会生成的东西，压低它不改变当前行为；chosen 可能超出当前模型能力，抬高它的似然要付出破坏其他分布的代价。on-policy 构造：

```
对 prompt x 用当前 π_θ 采 n 条 → RM / judge / verifier 打分 → 取最高为 y_w，最低（或随机低分）为 y_l
```

这样的偏好对都在模型分布附近，DPO 的梯度直接作用于模型会产生的行为。Llama 3 每轮重新采样就是为了保持 on-policy。

### 四类失败的机理

1. **chosen 似然下降**：DPO 只约束 `log π(y_w) − log π(y_l)` 的差，两者一起降也能减小损失；结果模型把概率质量挪到训练集外的序列上，生成质量下降。修法：加 NLL 正则（`L = L_DPO + λ · NLL(y_w)`，Llama 3 的做法，TRL 的 `rpo_alpha`）、降 lr、早停。
2. **长度膨胀**：偏好数据里 chosen 平均更长，DPO 学到"长即好"；序列 logprob 之和也天然偏向短序列被压低。修法：长度平衡的数据、SimPO 的平均 logprob、加长度惩罚。
3. **过拟合**：偏好对少（几千条）、epoch 多，训练集准确率 100% 而 dev 下降。修法：1 到 2 epoch、β 增大、IPO。
4. **β 选错**：β 太小则 KL 爆、通用能力下降；太大则几乎不动。看隐式奖励间隔 `r̂_w − r̂_l` 的分布与 dev 上的 KL。

## 实现步骤

### 1. 构造 on-policy 偏好对

```python
# 采样：Day 10 的 vLLM 批量采样，n=4，temperature 0.8
# 打分：RM（Day 12）或 pairwise judge（Day 10 附录 A）
pairs = []
for r in rollouts:
    scored = sorted([(rm_score(r["prompt"], y), y) for y in r["responses"]], reverse=True)
    if scored[0][0] - scored[-1][0] > margin:            # 分差太小的对子噪声大，丢弃
        pairs.append({"prompt": r["prompt"], "chosen": scored[0][1], "rejected": scored[-1][1]})
```

长度平衡：统计 chosen 与 rejected 的长度差分布，如果 chosen 系统性更长，按长度分桶重采样或对超长 chosen 降权。

### 2. TRL DPOTrainer

```python
from trl import DPOTrainer, DPOConfig
cfg = DPOConfig(
    output_dir="out/dpo",
    beta=0.1,
    loss_type="sigmoid",          # "ipo" / "kto_pair" / "simpo"（部分版本叫 cpo/simpo）以文档为准
    rpo_alpha=1.0,                # NLL 正则，防 chosen 似然下降；不需要时设 None
    learning_rate=5e-7,           # DPO 的 lr 比 SFT 低一个数量级
    num_train_epochs=1,
    per_device_train_batch_size=2, gradient_accumulation_steps=16,
    max_length=4096, max_prompt_length=2048,
    bf16=True, gradient_checkpointing=True,
    precompute_ref_log_probs=True,   # 参考模型 logprob 预计算，省一半显存
    logging_steps=10, report_to="wandb",
)
trainer = DPOTrainer(model=model, ref_model=None,   # None 则用 PEFT 的 base 或复制一份
                     args=cfg, train_dataset=pairs_ds, processing_class=tok)
trainer.train()
```

用 LoRA 时 `ref_model=None` 让 TRL 通过禁用 adapter 得到参考模型，不用第二份权重。全参时参考模型要常驻或预计算 logprob。

要盯的日志：`rewards/chosen`、`rewards/rejected`、`rewards/margins`、`rewards/accuracies`、`logps/chosen`（绝对值持续下降就是失败 1）。

### 3. SimPO / ORPO

TRL 的 `CPOTrainer(loss_type="simpo")` 与 `ORPOTrainer`；参数以当前版本文档为准。SimPO 要调 `simpo_gamma`（间隔）与 β（此处含义是缩放，常 2 到 10）。

### 4. 迭代 DPO

```
for round in 1..3:
    用 π_round 采样 → 打分 → 偏好对 → DPO（参考模型用 π_round 还是 π_0？）
```

两种选择：参考模型固定为 SFT 模型（KL 约束累计，更保守）或每轮更新为上一轮模型（更激进，Llama 3 的做法）。做实验比较。

## 实验

### 实验 1：on-policy vs 离线偏好对

1.5B SFT 模型。A 组：UltraFeedback 类离线数据 1 万对；B 组：同样 prompt 用当前模型采样 + judge 构造 1 万对。相同 DPO 配置，评测 Arena-Hard 风格 pairwise 胜率（用 judge 对 base）与 IFEval。预期 B 组胜率更高，A 组 `logps/chosen` 下降更明显。单卡 LoRA 可完成。

### 实验 2：β 扫描与 KL

β ∈ {0.02, 0.05, 0.1, 0.3}，记录 dev 上对 SFT 模型的 KL（用采样 200 条估计）、胜率、MMLU 子集。预期 β 小 KL 大、胜率先升后降、MMLU 下降。画一张 KL-胜率曲线。

### 实验 3：NLL 正则的作用

`rpo_alpha` ∈ {None, 0.5, 1.0}，看 `logps/chosen` 曲线与最终胜率。预期无正则时 chosen logp 持续下降且生成偶现乱码。

### 实验 4：长度偏置

把偏好对按 len(chosen) − len(rejected) 分成"chosen 更长"与"平衡"两组各训一个模型，比较生成平均长度与胜率（judge 加长度控制）。再用 SimPO 重跑"chosen 更长"组。预期 DPO 长度膨胀，SimPO 缓解。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 训练准确率 100% 但生成变差 | 过拟合 / chosen 似然下降 | `logps/chosen` 曲线 | 1 epoch、加 NLL、增大 β |
| 回答越来越长 | 数据长度偏置 | chosen/rejected 长度差分布 | 长度平衡、SimPO、长度惩罚 |
| `rewards/margins` 一直接近 0 | lr 太小或 β 太大 | 扫 lr | lr 5e-7 到 1e-6，β 0.05 到 0.1 |
| 通用 benchmark 掉 3 点以上 | β 太小、KL 爆 | 估 KL | 增大 β、混入 SFT 数据（ORPO/NLL） |
| 显存翻倍 | 参考模型常驻 | 看进程显存 | `precompute_ref_log_probs` 或 LoRA |
| 偏好对里 chosen 与 rejected 几乎一样 | 采样温度低、同一模型 | 计算对子的编辑距离 | 提高温度、丢弃分差小的对子 |
| 迭代 DPO 第三轮反而下降 | 参考模型每轮更新导致漂移 | 对 SFT 模型的累计 KL | 固定参考模型或降 lr |
| 安全类偏好训练后拒答率暴涨 | 安全对占比高且 rejected 都是"回答了" | 拒答率统计 | 平衡有用性对，Day 28 |

## 验收标准

- 手推 DPO 损失，并能解释 σ 权重为什么是难例加权、β 与 KL 的关系。
- 实验 1 与 3 有数据支持 on-policy 与 NLL 正则的结论。
- 交付一个 DPO 后的 checkpoint，胜率高于 SFT 起点，MMLU 子集下降 < 1 点。
- 能给出"DPO 还是 RLVR"的判断标准：奖励是否可程序化验证、是否需要探索、算力预算。

## 交付物

| 文件 | 内容 |
|---|---|
| `dpo-derivation.md` | 推导与变体对比表 |
| `pref-pairs/v1/` | on-policy 偏好对 + 长度统计 + manifest |
| `dpo-sweep.md` | 实验 1 到 4 |
| `out/dpo-1.5b/` | checkpoint |

## DPO 与 RLVR 的分工（2025 到 2026 的典型流水线）

```
SFT（格式与能力注入）
  → DPO / 偏好 RL（风格、有用性、安全：奖励来自人或 RM，不可程序化验证）
  → RLVR / GRPO（数学、代码、agent：奖励来自 verifier，可大规模探索）
  → 通用 RL 收尾（混合 RM 与 verifier，Qwen3 / DeepSeek-V3 的最后阶段）
```

DPO 便宜、稳定但不探索：它只在已采样的对子之间移动概率；RLVR 靠 rollout 探索新解法，能突破 SFT 分布。两者的数据都来自 rollout 采样，Week 3 讲的 rollout 系统对两者通用。

## 参考

- T27 DPO：推导、隐式奖励、与 RLHF 的等价。
- T23 InstructGPT：RLHF 目标的定义与 β KL 项。
- T28 KTO、T29 SimPO、T30 ORPO：各变体原文，重点读损失函数与消融。
- T14 Llama 3 第 4.1.4 节：DPO 的两个工程修改与迭代方案。
- T22 Tulu 3 第 5 章：DPO 数据配方与 length-normalized DPO。
- B05 TRL 文档：DPOTrainer / CPOTrainer / ORPOTrainer 的参数名。

## 附录 A：DPO 梯度的手算例子（示例数字）

一对样本，参考模型下 `log π_ref(y_w) = −50`，`log π_ref(y_l) = −48`（rejected 反而更"自然"）。当前模型 `log π_θ(y_w) = −49`，`log π_θ(y_l) = −49`，β = 0.1：

```
r̂_w = 0.1 × (−49 − (−50)) = +0.1
r̂_l = 0.1 × (−49 − (−48)) = −0.1
margin = r̂_w − r̂_l = 0.2
loss = −log σ(0.2) ≈ 0.598
梯度权重 σ(r̂_l − r̂_w) = σ(−0.2) ≈ 0.45   ← 还没分开，权重大

训练到 margin = 3 时：权重 σ(−3) ≈ 0.047，这对样本几乎不再贡献梯度
```

所以 DPO 的训练集准确率（margin > 0 的比例）很快到 100%，之后梯度主要来自少数 margin 小的困难对；epoch 多了就是在这些对上过拟合。

## 附录 B：隐式奖励当 RM 用

训练好的 DPO 模型可以给任意回答打分 `r̂(x,y) = β [log π_θ(y|x) − log π_ref(y|x)]`，这就是一个 RM。用途：

- 下一轮 rejection sampling 或偏好对构造的打分器（省一个独立 RM）。
- 检查长度偏置：对同一 prompt 的不同长度回答算 r̂，看是否与长度线性相关。
- 局限：只在训练分布附近可靠；两次前向（θ 与 ref）比独立 RM 贵。

## 附录 C：偏好数据质量检查清单

1. 抽 100 对人工看：chosen 真的更好吗？一致率低于 80% 的来源整体丢弃。
2. 长度差分布：|len_w − len_l| 的中位数与符号比例。
3. chosen 与 rejected 的编辑距离：过小（几乎相同）的对子对训练无信息，过大的对子往往是格式差异而非质量差异。
4. prompt 去重与去污染（Day 10）。
5. 每个 prompt 的对子数：多对同 prompt 会过拟合该 prompt。
6. 来源标注：人工 / RM / judge / verifier，训练时可按来源加权。

## 附录 D：自检问题

1. 为什么 DPO 不需要显式训练 RM？`Z(x)` 是怎么消掉的？
2. `logps/chosen` 和 `logps/rejected` 同时下降、margin 上升，模型在做什么？后果是什么？
3. SimPO 为什么不需要参考模型？它用什么代替 KL 约束？
4. 离线偏好对里 rejected 是当前模型从不会生成的句子，DPO 对它的梯度有效果吗？
5. 迭代 DPO 时参考模型固定与更新各有什么风险？
6. 什么任务应该跳过 DPO 直接 RLVR？什么任务反过来？
