---
title: "Day 16：PPO、GRPO、DAPO、GSPO 与 REINFORCE 家族"
type: concept
tags: [llm-training, ppo, grpo, dapo, gspo, rl-algorithms]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 16：PPO、GRPO、DAPO、GSPO 与 REINFORCE 家族

> 上一课 [[llm-training-30d/week3/day15-rollout-and-rl-basics]] · 下一课 [[llm-training-30d/week3/day17-rlvr-verifiers-reward-design]]。相关：Agent 课的策略优化 [[agent-rsi-30d/week3/day19-policy-optimization]]；推理课 MoE 推理 [[ai-infra-30d/week3/day19-moe-inference]]（GSPO 的动机与 MoE 路由有关）。

## 学习目标

1. 写出 PPO、GRPO、DAPO、Dr. GRPO、GSPO、REINFORCE++、RLOO、CISPO 的目标函数，每个符号有定义，并能指出相邻算法之间只改了哪一项。
2. 说清 value model 为什么在 LLM 上难训、group baseline 为什么能替代它、以及替代的代价。
3. 解释 DAPO 四项改进各自针对的失败现象。
4. 解释序列级重要性比（GSPO）为什么对 MoE 训练更稳。
5. 给定任务类型、模型规模、是否 MoE、是否异步，选出算法与超参起点。

## 工业现状

2025 年以来公开配方的主流是 GRPO 及其修正版：DeepSeek-R1（T17）用 GRPO；DAPO（T37）在 Qwen2.5-32B 上把 AIME 从 GRPO 基线大幅提升并开源全部代码，其四项改进成为事实标准；Qwen3（T15）的推理 RL 用 GRPO 变体；Kimi k1.5（T18）用无 value 模型的 online policy mirror descent，思想接近；GSPO（T39）是 Qwen3 系列 MoE 模型训练稳定性的解法；MiniMax-M1（T21）提出 CISPO 处理长 CoT 中关键 token 被 clip 掉的问题；Tulu 3（T22）的 RLVR 仍用 PPO 带 value model。REINFORCE++（T40）与 RLOO（T41）证明去掉 value 模型后 REINFORCE 风格也能稳定。

现在的共识：可验证 reward 场景下不需要 value 模型；token 级 clip 需要不对称（clip-higher）；全对或全错的 group 要过滤；损失按 token 平均；长度失控需要显式处理；MoE 要用序列级比值。这些结论都来自 2025 年的公开消融，本页把它们串成一条演化线。

## 核心原理

统一记号：

```text
x            prompt；D 是 prompt 数据集
y_i          第 i 条 rollout，长度 |y_i|，i = 1..G（G 是 group size）
y_{i,t}      第 i 条的第 t 个 token
π_θ          当前策略；π_old 采样该 batch 时的策略；π_ref 参考模型
r_i          第 i 条的标量 reward（verifier 或 RM）
A_{i,t}      第 i 条第 t 个 token 的 advantage
ρ_{i,t}      = π_θ(y_{i,t} | x, y_{i,<t}) / π_old(y_{i,t} | x, y_{i,<t})   token 级重要性比
clip(ρ, 1−ε, 1+ε)   把 ρ 截到区间内
```

### PPO（T35，应用于 LLM 见 T23）

```text
A_{i,t}  由 GAE 计算：δ_t = r_t + γ V(s_{t+1}) − V(s_t)，A_t = Σ_l (γλ)^l δ_{t+l}
         LLM 里 r_t 通常只在最后一个 token 非零（r_T = r_i − β·KL），中间 token r_t = −β·KL_t 或 0
L_PPO(θ) = E [ (1/|y|) Σ_t  min( ρ_t · A_t,  clip(ρ_t, 1−ε, 1+ε) · A_t ) ]   （最大化）
L_value(φ) = E [ (V_φ(s_t) − R_t)^2 ]                                             R_t 为 return
总损失 = −L_PPO + c_v · L_value − c_e · H(π_θ)                                    H 为熵（可选）
```

四个模型同时在显存里：policy、value、reference、reward（或 verifier）。value 模型与 policy 同规模，初始化自 RM 或 SFT 模型。LLM 上 value 难训的原因：每个 token 位置的 return 几乎相同（reward 只在末尾），value 要从部分序列预测最终对错，本质上是一个和策略一样难的问题；训练早期 value 估计差，GAE 给出的 advantage 噪声大。

### GRPO（T36）

去掉 value 模型，用组内统计做 baseline：

```text
Â_i     = ( r_i − mean(r_1..r_G) ) / ( std(r_1..r_G) + ε_std )        整条序列一个 advantage
A_{i,t} = Â_i   对所有 t
L_GRPO(θ) = E [ (1/G) Σ_i (1/|y_i|) Σ_t  min( ρ_{i,t} Â_i,  clip(ρ_{i,t}, 1−ε, 1+ε) Â_i ) ]
            − β · E [ (1/G) Σ_i (1/|y_i|) Σ_t  KL_t ]
KL_t（k3 估计器）= π_ref(y_{i,t}|·)/π_θ(y_{i,t}|·) − log( π_ref(y_{i,t}|·)/π_θ(y_{i,t}|·) ) − 1
```

与 PPO 的差别：advantage 是序列级常数广播到每个 token；KL 从 reward 移到 loss；损失先按序列内平均再按 group 平均（sequence-level）。

### Dr. GRPO（T38）

指出 GRPO 两个偏置并删掉两项：

```text
Â_i     = r_i − mean(r)                    ← 去掉 / std：难题（std 小）不再被放大权重
L       = E [ (1/G) Σ_i Σ_t  min(...) ]    ← 去掉 1/|y_i|：长的错误回答不再被稀释惩罚
```

结论是 GRPO 的 "响应越来越长" 有一部分是目标函数造成的，不是能力提升。

### DAPO（T37）

四项改进，每项对应一个可观察的失败：

```text
1. clip-higher：      clip(ρ, 1−ε_low, 1+ε_high)，ε_low=0.2，ε_high=0.28
   针对：熵坍塌。对称 clip 下低概率 token 的概率上限被压得很紧（0.2 × 小概率），探索消失。
2. dynamic sampling： 过滤 r 全 0 或全 1 的 group，持续采样直到 batch 里全是有梯度的 group
   针对：advantage 全 0 的 group 浪费算力并让有效 batch 变小、梯度噪声上升。
3. token-level loss： L = E [ (1 / Σ_i |y_i|) Σ_i Σ_t min(...) ]
   针对：sequence-level 下长序列每个 token 权重低，高质量长推理学不到，低质量长废话惩罚不足。
4. overlong reward shaping：超过 L_max − L_cache 的部分线性扣分，超过 L_max 的截断样本 mask 掉不计
   针对：截断样本被 verifier 判错引入的噪声 reward。
DAPO 同时设 β = 0（无 KL），因为推理 RL 希望策略走远。
```

### GSPO（T39）

把重要性比从 token 级改为序列级并做长度归一化：

```text
s_i(θ) = ( π_θ(y_i|x) / π_old(y_i|x) )^{1/|y_i|} = exp( (1/|y_i|) Σ_t log ρ_{i,t} )
L_GSPO(θ) = E [ (1/G) Σ_i  min( s_i Â_i,  clip(s_i, 1−ε, 1+ε) Â_i ) ]
```

动机：MoE 模型每次 forward 的专家路由可能变化，同一 token 在 π_old 与 π_θ 下可能走了不同专家，token 级 ρ 的噪声极大，clip 会随机地丢掉大量 token 的梯度，训练崩溃。序列级比值把噪声平均掉，且 clip 的单位与 reward 的单位（整条序列）一致。ε 要比 token 级小得多（论文量级 1e-3 到 1e-2，以论文为准）。

### REINFORCE++（T40）与 RLOO（T41）

```text
RLOO：     Â_i = r_i − (1/(G−1)) Σ_{j≠i} r_j       leave-one-out baseline，无偏
REINFORCE++：不用 group，advantage = r_i − β·KL_i 后做全局 batch 归一化，配 PPO clip；
             REINFORCE++-baseline 变体加回 group mean
```

它们说明"PPO 的核心贡献是 clip 的稳定性，而不是 value 模型"。

### CISPO（T21）

不 clip 梯度，而是 clip 重要性比的权重并停止其梯度：

```text
L_CISPO = E [ (1/Σ|y_i|) Σ_i Σ_t  sg( clip(ρ_{i,t}, 1−ε_low, 1+ε_high) ) · Â_i · log π_θ(y_{i,t}|·) ]
sg = stop-gradient
```

针对的失败：长 CoT 里"However"、"Wait" 这类低概率但关键的反思 token，在 PPO clip 下一旦 ρ 超出区间梯度就为 0，永远学不到；CISPO 保留梯度只限制权重。

### 一张演化表

| 算法 | baseline | 重要性比 | 损失平均 | KL | 主要针对 |
|---|---|---|---|---|---|
| PPO | value 模型 + GAE | token clip | token | reward 内 | 通用 |
| GRPO | group mean/std | token clip | sequence | loss 内 k3 | 去 value |
| Dr. GRPO | group mean | token clip | 无归一化 | 可无 | 长度与难度偏置 |
| DAPO | group mean/std | token 非对称 clip | token | 无 | 熵坍塌、无效 group、截断 |
| GSPO | group mean/std | 序列级 clip | sequence | 可无 | MoE 稳定性 |
| RLOO | leave-one-out | 可无 clip | sequence | reward 内 | 无偏 baseline |
| REINFORCE++ | 全局归一化 | token clip | token | reward 内 | 无 group 场景 |
| CISPO | group | 权重 clip、梯度不 clip | token | 无 | 关键 token 被 clip |

### 超参起点（示例量级，以各框架默认与论文为准）

| 超参 | 起点 | 说明 |
|---|---|---|
| lr | 1e-6（全参） | 比 SFT 低一个量级；LoRA 1e-5 |
| G（group size） | 8 到 16；难任务 32 到 64 | 越大 baseline 越准，rollout 成本线性增长 |
| batch（prompt 数） | 128 到 512 | 乘 G 得 rollout 数 |
| mini-batch 数 / PPO epochs | 每 batch 2 到 8 次更新 | 越多越 off-policy |
| ε | 0.2；clip-higher 上界 0.28 | GSPO 另算 |
| β（KL） | 0（RLVR）或 1e-3 到 1e-2（RM 场景） | |
| 采样温度 | 1.0 | 低温减少探索 |
| max response length | 4k 到 32k | 推理任务需要长 |

## 实现步骤

### 1. 在一个函数里实现所有变体

以 verl 的 `compute_policy_loss` 与 advantage 估计器为参照（`verl/trainer/ppo/core_algos.py`，函数名随版本变化），自己写一个纯 PyTorch 版本：

```python
import torch

def group_advantage(rewards, group_ids, normalize_std=True):
    # rewards: [N]，group_ids: [N]，返回每条序列的 Â_i
    adv = torch.zeros_like(rewards)
    for g in group_ids.unique():
        m = group_ids == g
        r = rewards[m]
        base = r.mean()
        a = r - base
        if normalize_std:
            a = a / (r.std(unbiased=False) + 1e-6)
        adv[m] = a
    return adv

def policy_loss(logp, old_logp, adv, mask, variant="dapo",
                eps_low=0.2, eps_high=0.28):
    # logp/old_logp/mask: [N, T]；adv: [N]（序列级，广播到 token）
    A = adv[:, None].expand_as(logp)
    log_ratio = logp - old_logp
    if variant == "gspo":
        # 序列级比值：先对有效 token 平均 log ratio
        seq_log_ratio = (log_ratio * mask).sum(-1) / mask.sum(-1)
        s = seq_log_ratio.exp()                                  # [N]
        obj = torch.min(s * adv, s.clamp(1 - eps_low, 1 + eps_high) * adv)
        return -obj.mean()
    ratio = log_ratio.exp()
    if variant == "cispo":
        w = ratio.clamp(1 - eps_low, 1 + eps_high).detach()
        obj = w * A * logp
    else:
        obj = torch.min(ratio * A, ratio.clamp(1 - eps_low, 1 + eps_high) * A)
    if variant in ("dapo", "cispo"):                # token-level
        return -(obj * mask).sum() / mask.sum()
    if variant == "grpo":                           # sequence-level
        return -((obj * mask).sum(-1) / mask.sum(-1)).mean()
    if variant == "drgrpo":                         # 无长度归一化
        return -(obj * mask).sum() / mask.shape[0]
    raise ValueError(variant)
```

用 Day 15 的 toy 数据验证：全对 group 的 advantage 为 0；GSPO 在 `logp == old_logp` 时 s 恒为 1，损失等于 `-adv.mean()`。

### 2. 在 verl 配置里切换算法

```yaml
# verl GRPO 训练的关键配置片段（字段名以当前版本文档为准）
algorithm:
  adv_estimator: grpo          # gae / grpo / rloo / reinforce_plus_plus ...
  use_kl_in_reward: false
actor_rollout_ref:
  actor:
    use_kl_loss: false         # DAPO 风格：无 KL
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28      # clip-higher
    loss_agg_mode: token-mean  # token-level；seq-mean-token-mean 为 GRPO 原始
    ppo_epochs: 1
    ppo_mini_batch_size: 64
  rollout:
    n: 16                      # group size
    temperature: 1.0
data:
  train_batch_size: 256
  max_response_length: 8192
# dynamic sampling 与 overlong shaping 在 verl 里由 filter_groups / reward 配置控制
```

### 3. 从曲线判断该换哪个变体

训练日志里至少看这几条：`reward/mean`、`response_length/mean` 与分位数、`actor/entropy`、`actor/pg_clipfrac`（被 clip 的 token 比例）、`actor/kl`。判断规则写进 Day 20。

## 实验

资源分层：实验 1 任何机器；实验 2 单卡（0.5B 到 1.5B，GSM8K）；实验 3 需要 MoE 小模型与多卡，没有条件时只做推导。

### 实验 1：token-level 与 sequence-level 的权重

用实现步骤 1 的函数，构造长度 50 与 500 的正确/错误 rollout 各一条，打印四种变体下每条序列对损失的贡献。预期与 Day 15 实验 2 一致。

### 实验 2：clip-higher 与熵

Qwen2.5-1.5B-Instruct 在 GSM8K 上用 TRL 或 verl 跑 GRPO 200 步，两组：对称 clip 0.2 与 clip-higher 0.2/0.28。记录熵曲线与 avg@4。预期：对称 clip 下熵在 100 步内快速下降到很低并停滞，clip-higher 下熵下降更慢，最终分数更高或相当。

### 实验 3：dynamic sampling 的有效 batch

同一配置，统计每步全对/全错 group 的比例，打开与关闭过滤各跑 200 步。预期：不过滤时有效 rollout 比例随训练推进上升（全对增多），梯度噪声变大，分数曲线更抖；过滤后每步 rollout 成本上升但曲线更平滑。

### 实验 4（可选）：GSPO 与 MoE

用一个小 MoE（如 Qwen1.5-MoE-A2.7B 或 OLMoE）跑 GRPO 与 GSPO 各 100 步，记录 `pg_clipfrac`。预期：GRPO 下 clipfrac 显著高于 dense 模型，GSPO 下稳定。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 熵在几十步内降到接近 0，reward 停滞 | 对称 clip 压制低概率 token | 看 entropy 与 clipfrac | clip-higher；温度 1.0；entropy bonus 小量 |
| reward 涨、长度持续涨、评测不涨 | sequence-level 长度偏置 | 按对错分组看长度 | token-level；Dr. GRPO；overlong shaping |
| 梯度范数尖峰、loss NaN | 重要性比爆炸；MoE 路由噪声 | 看 ratio 的 max | 降 lr；GSPO；减 ppo_epochs |
| clipfrac 超过 30% | 太 off-policy（epochs 多或异步 staleness 大） | 看 epochs 与 staleness | 减更新次数；Day 20 的异步修正 |
| 很多 group advantage 全 0 | 任务太易或太难 | 统计全对全错比例 | dynamic sampling；调整题目难度分布 |
| PPO value loss 不降 | value 初始化差、return 尺度 | 看 value/explained_var | 用 RM 初始化 value；reward 归一化；或换 GRPO |
| KL 快速增大到几十 | β 为 0 且 lr 大 | KL 曲线 | 若评测同步上升可接受；否则加 β 或降 lr |
| 反思词（wait/however）消失 | 关键低概率 token 被 clip | 统计这些 token 的频率 | clip-higher 或 CISPO |

## 验收标准

- 闭卷写出 PPO、GRPO、DAPO、GSPO 四个目标函数并标出彼此的差异项。
- 实现步骤 1 的函数通过三个 sanity check（全对 group 零梯度、GSPO 恒等情形、token 与 sequence 权重差异）。
- 实验 2 的熵曲线对比与解释。
- 一张"任务类型 × 模型类型 → 算法与超参起点"的决策表。

## 交付物

| 文件 | 内容 |
|---|---|
| `policy_loss_variants.py` | 八个变体的统一实现与测试 |
| `algo-notes.md` | 目标函数、演化表、超参表、实验 2/3 曲线与结论、决策表 |

## 参考

- T35 PPO、T23 InstructGPT：clip 与 LLM 上的 PPO 形态。
- T36 GRPO、T38 Dr. GRPO：group baseline 与它的偏置。
- T37 DAPO：四项改进的消融，Week 3 最重要的一篇。
- T39 GSPO：MoE 训练稳定性。
- T40 REINFORCE++、T41 RLOO：无 value 模型路线。
- T21 MiniMax-M1：CISPO 与长 CoT 的 clip 问题。
- B06 verl 文档：`core_algos` 与配置字段。
