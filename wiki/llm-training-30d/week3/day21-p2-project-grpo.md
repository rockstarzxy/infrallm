---
title: "Day 21：P2 项目：用 verl 跑通 GRPO/DAPO 并做消融"
type: concept
tags: [llm-training, project, grpo, verl, rlvr]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 21：P2 项目：用 verl 跑通 GRPO/DAPO 并做消融

> 上一课 [[llm-training-30d/week3/day20-async-rl-stability]] · 下一课 [[llm-training-30d/week4/day22-agentic-rl-formalization]]。相关：算法 [[llm-training-30d/week3/day16-ppo-grpo-family]]、reward [[llm-training-30d/week3/day17-rlvr-verifiers-reward-design]]、框架 [[llm-training-30d/week3/day18-rl-training-frameworks]]、评测 [[llm-training-30d/week2/day13-evaluation-experiment-management]]。

## 学习目标

1. 独立完成一次端到端 RLVR 训练：数据准备、verifier、配置、训练、评测、报告。
2. 用单变量消融回答两个具体问题，并给出带置信区间的结论。
3. 做一次 reward hacking 审计，证明 reward 上升对应真实能力提升。
4. 给出 rollout 与训练的时间分解，说明下一步该优化系统的哪一段。
5. 通过 Week 3 面试题自测。

## 工业现状

这一天做的是 DeepSeek-R1（T17）、DAPO（T37）、Tulu 3（T22）RLVR 阶段的缩小版：可验证任务 + GRPO 家族 + 独立推理引擎 rollout。公开复现里最常见的入门组合是 Qwen2.5-1.5B/3B/7B 基座或 Instruct 版 + GSM8K/MATH + verl，几百步内能看到明显的准确率提升，单机 8 卡一天内可完成，单卡 24 GB 用 1.5B 与 LoRA 也能跑出趋势。项目不追求刷分，追求把 Week 3 每一天的概念都在自己的日志里对上号。

## 核心原理

项目里要用到的公式集中回顾：

```
GRPO advantage:  A_i = (r_i − mean(r_1..n)) / (std(r_1..n) + δ)      # Dr. GRPO 建议去掉 std 归一化
DAPO loss:       L = −(1/Σ|o_i|) Σ_i Σ_t min(r_{i,t} A_i, clip(r_{i,t}, 1−ε_low, 1+ε_high) A_i)
dynamic sampling: 丢弃 r 全相同的 group，重新采样直到 batch 满
overlong shaping: 截断样本 reward = r − penalty(len)，penalty 在软区间内线性
```

必须在报告里体现的因果链：采样参数 → group 内多样性 → advantage 非零比例 → 学习信号强度 → reward 曲线斜率。

## 实现步骤

### 1. 环境与数据

```bash
pip install verl  # 或从源码安装，版本以 B06 为准；确认 vllm 版本兼容
python -m verl.utils.data.preprocess.gsm8k --local_dir ~/data/gsm8k   # 示例脚本名以仓库为准
```

数据格式（verl parquet）每行：`prompt`（chat 格式）、`reward_model.ground_truth`、`data_source`。自己的任务按此格式写转换脚本。对 MATH 类任务，ground truth 用规范化后的答案字符串。

### 2. verifier

```python
# reward_fn.py
import re
from math_verify import parse, verify   # 或自写等价判定

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    m = re.search(r"\\\\boxed\\{(.+?)\\}", solution_str)
    if m is None:
        return 0.0                       # 无格式 → 0；不要给负分，避免坍缩
    ok = verify(parse(ground_truth), parse(m.group(1)))
    return 1.0 if ok else 0.0
```

先在 200 条人工核对样本上测 verifier 的假阳/假阴率，写进报告。

verifier 边界案例（至少覆盖这些再上线）：

| 输入 | 期望 | 常见错误 |
|---|---|---|
| `\boxed{1/2}` vs 真值 `0.5` | 等价 | 字符串比较判错 |
| `\boxed{2}` 与 `\boxed{3}` 同时出现 | 0（多答案） | 取第一个或最后一个给了分 |
| 答案在 boxed 外、boxed 内为空 | 0 | 正则匹配到空串 |
| `\boxed{x=2}` vs `2` | 视任务定 | 未规范化 |
| 超长输出被截断、无 boxed | 0，且标记 truncated | 与"答错"混在一起 |
| 单位、百分号、逗号分隔千位 | 规范化后等价 | 判错 |

### 3. 配置

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=~/data/gsm8k/train.parquet data.val_files=~/data/gsm8k/test.parquet \
  data.train_batch_size=256 data.max_prompt_length=512 data.max_response_length=1024 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  custom_reward_function.path=reward_fn.py custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 trainer.total_epochs=1 \
  trainer.val_before_train=true trainer.test_freq=20 trainer.save_freq=50 \
  trainer.logger='["console","wandb"]'
```

单卡替代：`Qwen2.5-0.5B-Instruct` 或 1.5B + LoRA（`actor_rollout_ref.model.lora_rank=32`，字段以版本为准），`train_batch_size=64`，`n=4`，`gpu_memory_utilization=0.4`。

### 4. baseline 与 seed

三组必跑：训练前评测（val_before_train）；GRPO 主配置 2 个 seed；每个消融 2 个 seed。评测用 `avg@4`，温度与训练一致，再补一组 greedy。

### 5. 消融（二选二或自选）

| 消融 | 变量 | 预期 |
|---|---|---|
| A. KL 系数 | 0 vs 1e-3 vs 1e-2 | 0 时上升快但后期可能漂；1e-2 时学不动 |
| B. dynamic sampling | 开 vs 关 | 开时每步有效样本多，同步数下 reward 更高，但每步采样时间上升 |
| C. group size | 4 vs 8 vs 16 | 大 group 方差小，rollout 时间线性增加 |
| D. loss_agg_mode | seq-mean vs token-mean | 长度趋势不同（Day 20 实验 2） |

### 6. reward hacking 审计

- 每 50 步 dump 20 条最高 reward 样本，人工看是否有"多个 boxed 答案""复读题目""答案猜测穷举"等模式。
- 用第二个独立 verifier（不同解析规则）复核最终 checkpoint 的 val 集，报告两者不一致率。
- hidden set：从 MATH 或另一来源抽 300 题训练期间不碰，最后评。

### 7. 时间分解

从日志取每步的 `timing/gen`、`timing/reward`、`timing/old_log_prob`、`timing/update_actor`（字段名以版本为准），画堆叠图。计算 rollout 占比，以及 rollout 内长尾（`response_length/max` 与 mean 之比）。

### 7b. 导出与端到端验证

训练结束后把 FSDP/Megatron checkpoint 合并为 HF 格式（verl 提供 model merger 脚本，名字以版本为准），用 vLLM 加载做最终评测，并与训练日志里最后一次 val 分数对比。两者差异超过 1 个标准误说明导出或采样参数有问题。这一步是 Day 29 发布流程的雏形。

### 8. 资源分层适配

| 资源 | 模型 | 改动 | 能得到什么 |
|---|---|---|---|
| API-only | 任意可采样的 API 模型 | 用 API 采样 n=8 → verifier 过滤 → 只保留对的样本 SFT 一个开源小模型（rejection sampling fine-tuning） | 数据侧的全部经验；没有 on-policy 更新曲线 |
| 单卡 24 GB | Qwen2.5-0.5B/1.5B-Instruct + LoRA | `n=4`、`train_batch_size=64`、`max_response_length=512`、`gpu_memory_utilization=0.4`、gradient checkpointing | 主配置 + 1 个消融，趋势可见 |
| 单卡 48 GB | 1.5B 全参或 3B LoRA | 上述放宽一倍 | 主配置 + 2 个消融 |
| 4 到 8 卡 | 3B 到 7B 全参，vLLM TP=1 每卡一个 rollout 实例 | 完整配置 | 全部 + 异步实验 |

LoRA 做 RL 时注意 B04 的结论：学习率要比全参高约一个数量级（示例 1e-5 对 1e-6），rank 32 以上，所有 linear 层都挂；否则会误以为"LoRA 学不动"。

### 9. 常见配置错误速查

- `ppo_mini_batch_size` 必须整除 `train_batch_size × n`，否则最后一个 mini-batch 大小异常。
- `max_prompt_length + max_response_length` 不能超过引擎的 `max_model_len`。
- chat template：Instruct 模型要用 chat 格式 prompt，base 模型用纯文本加格式说明；混用会让 verifier 非零率异常。
- 评测温度与训练一致，否则 val 曲线与 reward 曲线不可比。
- 训练与 vLLM 的 dtype 都用 BF16；rollout 里不要开 FP8 KV（Day 19 实验 1 的原因）。

## 实验

本日的"实验"就是项目本身，按下面的顺序执行并记录。资源分层：API-only 学员用 Day 10 的 rejection sampling 流程替代 RL 更新（采样 → verifier 过滤 → SFT），其余步骤相同；单卡完成主配置 + 1 个消融；4-8 卡完成全部。

### 阶段 0：计划（1 小时）

写一页计划：任务与为什么选它、模型与资源、主配置、两个消融及各自预期方向、评测集与 hidden set 来源、成功标准。计划先于实验，事后与结果对照，写出偏差原因。

### 阶段 1：跑通（半天）

50 步 smoke test，确认：verifier 在 rollout 样本上非零率合理（10% 到 80% 之间，太低说明任务太难，太高说明没学习空间）；`pg_clipfrac` 在 0.05 到 0.3；每步时间稳定；checkpoint 能保存并被 vLLM 加载评测。

### 阶段 2：主训练（半天到一天）

按配置跑满 1 epoch 或 300 步。每 20 步评测。记录 Day 20 面板的全部指标。

### 阶段 3：消融（一天）

选两个消融各 2 个 seed。固定其他所有变量，包括数据顺序（固定 seed）和评测采样参数。

### 阶段 4：审计与报告（半天）

hacking 审计、hidden set、时间分解、报告。

### 统计要求

每个对比给 mean ± std（2 个 seed 至少给两点），评测集 300 题以上，报告 avg@4 的标准误。两组差异小于 2 倍标准误的写"未观察到显著差异"。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| reward 从第一步就 0 | verifier 解析失败、chat template 让模型不输出 boxed | 手动跑 5 条 rollout 看输出 | 修 prompt 要求格式；verifier 容错 |
| reward 第一步就 0.9 | 任务对模型太简单 | val_before_train 分数 | 换 MATH 或提高难度筛选 |
| 20 步后 reward 掉到 0 | lr 太大、KL 0 且 clip 太松 | KL/ratio 面板 | lr 1e-6、加 KL 或 clip 收紧 |
| OOM 在 rollout 阶段 | `gpu_memory_utilization` 与训练显存冲突 | 切换点显存 | 降到 0.4 到 0.5；`free_cache_engine` |
| OOM 在 old_log_prob 阶段 | micro batch 过大、序列过长 | 报错栈 | 降 `log_prob_micro_batch_size`，开 gradient checkpointing |
| 每步时间波动大 | 长尾 rollout | `response_length/max` | 降 max_response_length；overlong shaping |
| 评测分数抖动大 | 评测样本少或温度采样方差 | 算标准误 | 300 题以上，avg@4 |
| wandb 曲线与 console 不一致 | logger 统计口径（token-mean vs seq-mean） | 看代码 | 报告里注明口径 |
| 两个 seed 结论相反 | 效应小于噪声 | 标准误 | 加 seed 或加大评测集，如实写 |

## 验收标准

- 报告包含：任务与 verifier 描述及其误判率；主配置与 baseline 的评测对比（含标准误）；两个消融的曲线与结论；hacking 审计结果与 hidden set 分数；时间分解图与系统层面的下一步建议。
- 能口头回答：为什么 rollout 温度用 1.0；为什么 token-mean；你的 clip-higher 是否生效的证据；你的实验里 rollout 占比多少、长尾多大。
- 代码、配置、seed、数据版本全部可复现。

## 交付物

| 文件 | 内容 |
|---|---|
| `p2-report.md` | 上述报告 |
| `reward_fn.py` + `verifier-audit.md` | verifier 与其误判率、hacking 审计样本 |
| `configs/*.sh` | 主配置与每个消融的启动脚本 |
| `timing-breakdown.png` | 时间分解 |

## 评分 rubric

| 维度 | 权重 | 4 分（优秀） | 2 分（合格） | 0 分 |
|---|---:|---|---|---|
| 可复现性 | 20% | 配置、seed、数据版本、依赖版本齐全，他人能一键复跑 | 缺 1 到 2 项但能补 | 无法复跑 |
| verifier 质量 | 15% | 误判率有测量且 < 2%（示例），有第二 verifier 复核 | 有测量 | 未测 |
| 训练结果 | 15% | 主配置相对 baseline 提升显著（> 2 倍标准误）且 hidden set 同向 | 有提升但显著性不足 | 无提升或未评 |
| 消融设计 | 20% | 单变量、2 seed、结论与机制解释一致 | 单变量但 1 seed | 多变量混杂 |
| hacking 审计 | 10% | 样本审查 + 独立 verifier + hidden set 三项齐全 | 两项 | 无 |
| 系统分析 | 10% | 时间分解 + 长尾分析 + 可执行的下一步 | 有分解 | 无 |
| 报告表达 | 10% | 结论先行，图表可读，负结果如实 | 完整但冗长 | 缺关键数据 |

总分 = Σ 权重 × 得分 / 4。低于 60% 不通过 Week 3。

## 报告模板

```
# P2：<任务> 上的 GRPO/DAPO 实验
## 1. 结论（3 句话）
## 2. 任务、数据、verifier（含误判率表）
## 3. 主配置与 baseline 对比（表 + 曲线，含标准误）
## 4. 消融 A / B（各一张图，一段机制解释）
## 5. 稳定性面板摘要（entropy / KL / 长度 / clipfrac 关键截图）
## 6. Hacking 审计与 hidden set
## 7. 时间分解与系统建议
## 8. 未做的、失败的、下一步
附录：配置 diff、seed、版本
```

## Week 3 知识串联

```
Day 15 rollout 定义、on/off-policy、advantage
   │
Day 16 PPO → GRPO → DAPO / Dr.GRPO / GSPO：目标函数与超参
   │
Day 17 verifier 与 reward：训练信号从哪来，怎么被 hack
   │
Day 18 框架：控制流放哪、colocated 还是分离
   │
Day 19 rollout 引擎：权重同步、引擎不一致 → TIS
   │
Day 20 异步：staleness、partial rollout；五类不稳定与面板
   │
Day 21 项目：把以上每个概念在自己的日志里找到对应指标
```

## Week 3 面试题

1. **rollout 在 RLVR 里指什么？和评测里的采样有什么区别？** 一个 prompt 用当前策略采 n 条完整 response，用于产生训练信号；评测采样是为了估计 pass@k，不参与梯度，通常固定温度与 seed。
2. **GRPO 相比 PPO 去掉了什么，代价是什么？** 去掉 value model，用 group 内 reward 的均值做 baseline；代价是需要每个 prompt 采多条，且 advantage 只在 group 内比较，跨 prompt 难度差异不进入信号。
3. **clip-higher 为什么能缓解 entropy 崩塌？** 对称 clip 下低概率 token 的上调空间是绝对值很小的 `ε·π`，高概率 token 持续被强化；放宽上界让探索性 token 有更大上调空间。
4. **为什么训练侧要重算 logprob 而不用引擎返回的？** 引擎 kernel 与训练 kernel 数值不同，直接用会污染 PPO 比值；引擎 logprob 只用于 TIS 修正。
5. **TIS 修正了什么？** `π_old/π_infer` 的采样偏差，截断到常数 C 防止极端权重。
6. **异步 RL 中 staleness 如何进入 loss？** 样本带版本号，训练侧用 `π_old/π_behav` 做重要性权重，超过上限的样本丢弃或 mask。
7. **response 长度单调上升的两个常见原因？** 样本级长度归一化让长错答案每 token 惩罚小；截断样本被给 0 reward 让模型学到少说。
8. **colocated 与 disaggregated 怎么选？** 卡少、模型小、同步 RL 选 colocated；长 CoT/agent、需要异步、卡多选分离。判据是 rollout 长尾造成的空转是否超过权重跨机传输成本。
9. **reward 涨但 hidden set 不涨，先查什么？** 抽高 reward 样本看是否 hack，用独立 verifier 复核，检查 val 与训练集是否污染。
10. **GSPO 解决什么问题？** MoE 下 token 级重要性比因路由变化噪声极大，序列级比值更稳。

## Week 4 预览

Week 4 把 rollout 从"一个 prompt 一条 response"扩展到"一条多轮轨迹含多次工具调用"。Day 22 先把 agent 任务写成 POMDP 并定义哪些 token 进 loss；Day 23、24 讲环境与 rollout 基础设施；Day 25 讲 agent 的 reward 与 credit assignment；Day 26 拆解 SWE-RL、Kimi K2、GLM-4.5 的公开配方；Day 27 讲推理模型；Day 28、29 讲安全对齐与大规模运维；Day 30 是 agentic RL capstone。

## 参考

- T17 DeepSeek-R1、T37 DAPO：本项目复现的原型。
- T38 Dr. GRPO：advantage 归一化与长度偏置的选择依据。
- T22 Tulu 3：RLVR 作为后训练最后一阶段的整体位置。
- T36 DeepSeekMath：GRPO 原始定义。
- B06 verl 文档：配置字段与数据格式。
