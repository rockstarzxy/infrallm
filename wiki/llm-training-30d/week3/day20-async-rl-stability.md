---
title: "Day 20：异步 RL、off-policy 修正与训练稳定性"
type: concept
tags: [llm-training, async-rl, off-policy, stability, entropy]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 20：异步 RL、off-policy 修正与训练稳定性

> 上一课 [[llm-training-30d/week3/day19-rollout-engine-weight-sync]] · 下一课 [[llm-training-30d/week3/day21-p2-project-grpo]]。相关：Agent 场景的异步 rollout 见 [[llm-training-30d/week4/day24-agent-rollout-infrastructure]]；算法细节见 [[llm-training-30d/week3/day16-ppo-grpo-family]]。

## 学习目标

1. 说清同步 RL 的时间浪费在哪里，异步 RL 用什么换掉了它，代价是什么。
2. 能定义 staleness、partial rollout、版本号，并写出 off-policy 修正在 loss 里的位置。
3. 能读一张 RL 训练面板，在 entropy 崩塌、KL 爆、长度爆炸、loss spike、reward 与评测背离五类问题里判断是哪一种。
4. 对每类不稳定给出首选修法和不要做的事。
5. 动手：在 verl/TRL 上复现至少两种不稳定并修好。

## 工业现状

同步 RL（每步：同步权重 → rollout 全部完成 → 训练）在长 CoT 和 agent 任务上撞到长尾墙：一个 batch 里最长的 rollout 决定 step 时间，GPU 在末尾大量空转。2025 年的主要系统都走向了异步：

| 系统 | 异步方式 | 修正 |
|---|---|---|
| Kimi k1.5（T18） | partial rollout：长 rollout 跨 step 分段，未完成部分下一步接着采 | 分段处的 logprob 由新旧策略拼接，训练时按版本修正 |
| AReaL（T44） | 全异步：rollout worker 与 trainer 解耦，样本带版本号进队列 | 限制最大 staleness，decoupled PPO 目标 |
| DAPO（T37） | 同步为主，但 dynamic sampling 过滤全对/全错 group | 用 token-level loss 与 clip-higher 稳定 |
| MiniMax-M1（T21） | CISPO：对 IS 权重做 clip 而不是对 token 的更新做 clip | 保留低概率 token 的梯度 |
| GSPO（T39） | 序列级重要性比 | MoE 下逐 token 比值噪声大，序列级更稳 |
| verl / slime（B06、B07） | 提供 async rollout、staleness 上限、rollout IS 修正等选项 | 以当前版本文档为准 |

稳定性方面的公开经验集中在 DAPO、Dr. GRPO（T38）、GSPO 三篇：clip-higher 防 entropy 崩塌，去掉长度归一化防长度偏置，序列级比值防 MoE 崩。

## 核心原理

### 同步 RL 的浪费

```
step k:  [sync][ rollout ........................ 长尾 .... ][ train ]
                ▲ GPU 利用率 90%      ▲ 30%     ▲ 5%
```

rollout 阶段 GPU 利用率随完成样本数下降。长 CoT 任务里 P50 长度 2k、P99 长度 16k（示例）时，末尾一半时间只有几条样本在跑。

### 异步 RL 的三个概念

**版本号**：每次权重更新后 `version += 1`；每条 rollout 样本记录生成它的版本 `v_gen`。

**staleness**：`s = v_train − v_gen`，训练时当前版本与样本生成版本之差。同步 RL 的 `s = 0`（首个 mini-batch）。异步系统设上限 `s_max`（示例：1 到 4），超过的样本丢弃或降权。

**partial rollout**：一条 rollout 采到 `L` 个 token 时权重更新了，不丢弃已生成部分，下一步用新权重接着采。得到的序列前半来自 `π_{v}`，后半来自 `π_{v+1}`，logprob 按段记录。

```
样本队列（带版本）:  [v=7, len 512] [v=7, len 4096 未完成→续] [v=8, len 900] ...
trainer 取满一个 batch 就更新，不等所有 rollout 结束
rollout worker 持续采，用它当时拿到的最新权重
```

### off-policy 修正在 loss 里的位置

对每个 token 有三个分布：`π_gen`（生成它的旧版本，含推理引擎差异）、`π_old`（训练侧当前 step 开始时的策略）、`π_θ`（正在更新的）。完整形式：

```
w_t   = π_old(a_t) / π_gen(a_t)              # 采样修正（staleness + 引擎不一致）
r_t   = π_θ(a_t) / π_old(a_t)                # PPO 比值
L_t   = clip_or_truncate(w_t) · min(r_t A_t, clip(r_t, 1−ε_low, 1+ε_high) A_t)
```

三种处理方式：

| 方式 | 对 `w_t` 的处理 | 特点 |
|---|---|---|
| TIS（B01） | `min(w_t, C)` | 简单；大偏差 token 仍保留有限权重 |
| MIS | `w_t > C` 的 token mask 掉 | 更保守，丢样本 |
| decoupled PPO（T44） | 把 `π_old` 换成行为策略做 clip 参照，`π_prox` 单独控制更新步长 | 允许更大 staleness |

必须理解：`w_t` 只在 staleness > 0 或引擎不一致时偏离 1。同步 RL 且已做 Day 19 的 TIS，就是 `s = 0` 的特例。

### 五类不稳定的机制

**entropy 崩塌**：策略分布迅速尖锐化，group 内 response 趋同，advantage 归零，学习停止。机制：PPO 对称 clip 让低概率 token 的上调空间比高概率 token 小（`1+ε` 对小概率是绝对值很小的增量），高概率 token 持续被强化。修法：clip-higher（`ε_high > ε_low`，DAPO 用 0.28/0.2 示例）、entropy 项（慎用，易乱码）、提高温度、过滤全对 group。

**KL 爆**：策略偏离 reference 太远，输出退化或 reward hacking。机制：KL 系数太小、lr 太大、reward 有可 hack 的捷径。修法：先查 reward 是否被 hack（看样本），再调 KL 系数或 lr；DAPO 类方法干脆去掉 KL 项而靠 clip 控制步长，此时要看的是与 `π_old` 的比值分布而不是与 ref 的 KL。

**长度爆炸 / 坍缩**：response 平均长度单调上升到 max_tokens 或掉到几十 token。机制：GRPO 的 1/|o| 归一化让长错误答案每 token 惩罚小（T38 指出），或截断样本被给了 0 reward（模型学到少说）。修法：token-level loss（不按样本长度归一化）、overlong soft penalty、把截断样本的 reward 单独处理。

**loss spike**：单步 loss 或梯度范数跳变几个量级。机制：某个 mini-batch 里极端 IS 比值、混合精度溢出、数据里的异常样本（超长、乱码）。修法：梯度裁剪（1.0 常见）、IS 截断、检查该 step 的样本。

**reward 涨评测不涨**：reward hacking 或 reward 与目标不对齐。机制见 Day 17。判断：抽样看高 reward 样本；换一个不同来源的 verifier 复核；hidden set 评测。

### KL 项的两种放法（必须理解）

```
放在 reward 里：r'_t = r_t − β · log(π_θ(a_t)/π_ref(a_t))     # PPO-RLHF 传统做法，进入 advantage
放在 loss 里：  L = L_pg + β · KL_est(π_θ || π_ref)              # GRPO 的做法，用 k3 估计量
```

放 reward 里会让 KL 影响 advantage 的排序，放 loss 里只做正则；DAPO 与 Dr. GRPO 去掉 KL 后靠 clip 控制步长，此时 reference model 也可省掉，省一份显存与一次 forward。异步系统里 KL 的参照应是固定 ref，不是 `π_behav`。

### 面板必看指标

| 指标 | 健康形态 | 异常含义 |
|---|---|---|
| `reward/mean`、`reward/std` | 缓慢上升，std 不归零 | std → 0 表示 group 内无差异，学习停 |
| `entropy` | 缓降 | 断崖式下降 = 崩塌；上升 = 乱码 |
| `kl` 或 `ratio` 分布 | KL 缓升；ratio 集中在 1 附近 | 尾部变厚 = 更新过猛 |
| `response_length/mean, max, clip_ratio` | 平稳或按任务缓变 | clip_ratio 上升 = 长度爆炸 |
| `grad_norm` | 平稳 | spike |
| `pg_clipfrac` | 0.1 到 0.3（示例） | 接近 0 = 没学；接近 1 = 步子太大 |
| `staleness` 直方图（异步） | 集中在 0 到 s_max | 长尾说明 rollout 跟不上训练 |
| `is_weight` 分布（做修正时） | 均值 1 附近 | 均值偏离 = 系统性不一致 |
| eval on hidden set | 与 reward 同向 | 背离 = hacking |

### 异步系统的组件（知道即可）

```
┌────────────┐  prompts   ┌──────────────────┐  样本(带版本)  ┌────────────┐
│ 任务采样器  │──────────▶│ rollout workers   │──────────────▶│ 样本队列    │
│ (难度分层)  │           │ 推理引擎 ×N       │               │ (按版本/长度)│
└────────────┘           └────────▲─────────┘               └──────┬─────┘
                                  │ 权重 v+1                        │ 取满 batch
                          ┌───────┴─────────┐                ┌──────▼─────┐
                          │ 权重发布 (版本号) │◀───────────────│ trainer    │
                          └─────────────────┘                └────────────┘
```

设计要点：

- **队列出队策略**：按版本优先（先训旧样本，减少 staleness 溢出）还是按到达顺序，AReaL 选前者。
- **reward 计算的位置**：放在 rollout worker 侧（verifier 与采样并行）而不是 trainer 侧，否则 trainer 成为瓶颈。
- **rollout 与训练的卡数比**：由 `t_rollout_per_sample × batch / t_train_per_batch` 决定，长 CoT 任务常见 rollout 卡数是训练卡数的 2 到 4 倍（示例）。
- **权重发布频率**：每步发布会让 rollout 端频繁 reload；每 k 步发布则平均 staleness 增大。AReaL 报告在 `s_max` 内两者对最终效果影响小于噪声，吞吐差异大。

### decoupled PPO 的目标（知道即可）

标准 PPO 的 clip 参照是 `π_old`（rollout 策略）。异步下 `π_old` 是几个版本前的行为策略 `π_behav`，若仍用它做 clip 参照，clip 区间会把大量正常更新截掉。T44 把两者分开：

```
r_t = π_θ / π_prox              # π_prox：当前训练 step 开始时的策略，控制单步步长
w_t = π_prox / π_behav           # 行为策略修正，截断到 C
L   = w_t · min(r_t A_t, clip(r_t) A_t)
```

`π_prox` 每 step 更新，`π_behav` 随样本携带。这就是 Day 19 三分布公式在异步下的一般形式。

## 实现步骤

### 1. verl 里的异步与修正选项

字段名随版本变化，以 B06 为准，下面是要找的开关和它们对应的概念：

```yaml
actor_rollout_ref:
  rollout:
    mode: async                    # 异步 rollout（部分版本叫 async_rollout / agent loop）
    calculate_log_probs: true      # 引擎 logprob，供 IS
  actor:
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28          # clip-higher
    loss_agg_mode: token-mean      # token-level loss（对比 seq-mean-token-mean）
    use_kl_loss: false             # DAPO 式去 KL；保留时 kl_loss_coef ~1e-3（示例）
    entropy_coeff: 0.0
    # rollout 重要性采样修正：搜 "tis" / "rollout_is" / "importance"
algorithm:
  adv_estimator: grpo
  filter_groups:                   # dynamic sampling：过滤全对全错 group
    enable: true
    metric: acc
    max_num_gen_batches: 10
trainer:
  max_staleness: 2                 # 异步时的 staleness 上限（若存在）
```

### 2. TRL GRPOTrainer 的对应项

```python
GRPOConfig(
    epsilon=0.2, epsilon_high=0.28,        # clip-higher
    loss_type="dapo",                      # 或 "grpo" / "dr_grpo"，决定长度归一化方式
    beta=0.0,                              # KL 系数
    mask_truncated_completions=True,       # 截断样本不进 loss
    num_generations=8,
    use_vllm=True, vllm_mode="colocate",
    # 版本较新时有 vllm_importance_sampling_correction 一类选项
)
```

### 3. 手写 partial rollout 的数据结构

```python
@dataclass
class RolloutSample:
    prompt_ids: list[int]
    response_ids: list[int]
    gen_logprobs: list[float]      # 引擎返回，按 token
    gen_version: list[int]         # 按 token 的权重版本（partial rollout 时不同段不同）
    finished: bool
    reward: float | None

def resume(sample, engine, version):
    out = engine.generate(sample.prompt_ids + sample.response_ids, max_tokens=remaining)
    sample.response_ids += out.token_ids
    sample.gen_logprobs += out.logprobs
    sample.gen_version += [version] * len(out.token_ids)
    sample.finished = out.finish_reason == "stop"
```

训练侧按 `gen_version` 分段算 `w_t`；staleness 超限的段可以整段 mask。

### 4. 面板

W&B 或 TensorBoard 上把 Part "面板必看指标" 里的 9 项放在同一页，按 step 对齐。异步系统额外加 rollout 队列长度和 trainer 等待时间。每 50 步 dump 10 条最高 reward 与 10 条最低 reward 的样本到文本文件，人工看。

### 5. 实验记录模板

每次运行记录：配置 diff（相对主配置）、seed、`s_max`、采样参数、每 20 步的 reward/entropy/KL/长度/评测五元组、异常 step 的样本 dump 路径。写成一张表放进交付物，Day 21 项目直接复用。

## 实验

资源分层：单卡可完成实验 1、2（1.5B 模型、GSM8K）；实验 3、4 需 4 卡以上。

### 实验 1：复现 entropy 崩塌并修复

GRPO，对称 clip 0.2，温度 0.6，不过滤 group。跑 300 步，记录 entropy 与 reward std。预期 100 步内 entropy 断崖、reward std 趋 0。修复组：clip-higher 0.28 + 温度 1.0 + 过滤全对 group。对比曲线。

### 实验 2：长度偏置

同一任务，`loss_agg_mode` 取 seq-mean（GRPO 原版归一化）与 token-mean 两组，max_tokens 2048。记录平均长度与错误答案的平均长度。预期 seq-mean 下错误答案越来越长（T38 现象），token-mean 下长度平稳。

### 实验 3：staleness 上限

异步模式，`max_staleness` 取 0、2、8。记录每步墙钟时间、GPU 利用率、最终评测。预期：吞吐随上限上升，评测在上限过大时下降；找到你任务上的拐点。

### 实验 4：partial rollout 的收益

max_tokens 8192 的长 CoT 任务，开关 partial rollout。记录 rollout 阶段 GPU 利用率曲线与 step 时间。预期：关闭时末尾利用率长时间低于 20%，开启后平稳。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| entropy 100 步内跌到接近 0 | 对称 clip、低温、全对 group 占比高 | 看 `pg_clipfrac` 与 reward std | clip-higher、温度 1.0、dynamic sampling |
| response 长度单调涨到上限 | 样本级长度归一化、截断样本 reward 为 0 | 分对错看长度趋势 | token-level loss、overlong soft penalty |
| response 坍缩到几十 token | 截断被重罚、format reward 权重过大 | 看短样本内容 | 降 format 权重、截断样本 mask 而非负 reward |
| KL 单调上升伴随乱码 | lr 太大或 KL 系数 0 且 clip 不够 | 看 ratio 尾部 | 降 lr；加 KL 或用 GSPO 序列级比值 |
| MoE 模型训练几十步后崩 | token 级 IS 比值在专家路由变化下噪声极大 | 看 ratio 方差随 step 的变化 | GSPO（T39）；固定路由（rollout 与训练同一路由）若框架支持 |
| grad_norm 偶发尖峰 | 极端 IS 权重、超长样本、混合精度溢出 | dump 该 step 样本 | 梯度裁剪、IS 截断、过滤异常长度 |
| 异步下评测比同步差 | staleness 过大或修正没开 | staleness 直方图 | 降上限、开 IS 修正 |
| reward 上升评测下降 | hacking | 抽样看高 reward 样本，换 verifier 复核 | 修 reward（Day 17），加 hidden set 门禁 |
| rollout 队列堆积 | trainer 慢于 rollout | 队列长度曲线 | 减 rollout 卡或加训练卡；调 batch |
| trainer 空等 | rollout 慢，长尾 | GPU 利用率时间线 | partial rollout、降 max_tokens、更多 rollout 卡 |

## 自测

1. staleness=0 但引擎不一致时，`w_t` 的均值应接近多少？均值系统性偏离说明什么？
2. entropy 崩塌时 `pg_clipfrac` 通常怎么变？为什么？
3. 去掉 KL 项后，用什么指标替代 KL 来判断"步子太大"？

## 验收标准

- 能在一张面板截图上指出五类不稳定各自的证据指标。
- 能写出带 staleness 修正的 loss，并解释 clip-higher、token-level loss、GSPO 各自解决什么。
- 实验 1、2 的对照曲线；若有多卡，实验 3 的拐点。
- 能说明你的任务应该同步还是异步、staleness 上限取多少、依据是什么。

## 交付物

| 文件 | 内容 |
|---|---|
| `rl-stability-dashboard.md` | 指标清单、每项的健康区间、你实验里的截图 |
| `instability-repro.md` | 实验 1、2 的配置差异与曲线 |
| `async-config-notes.md` | 异步选项、staleness 上限与实验 3、4 结果 |

## 参考

- T37 DAPO：clip-higher、dynamic sampling、token-level loss、overlong shaping 四项的实证。
- T38 Dr. GRPO：长度与难度偏置的来源。
- T39 GSPO：MoE 稳定性与序列级比值。
- T44 AReaL、T18 Kimi k1.5：异步系统与 partial rollout 的设计。
- T21 MiniMax-M1：CISPO 的 IS 权重裁剪思路。
- B01：引擎不一致导致的隐式 off-policy。
- T35 PPO：clip 的原始动机。
