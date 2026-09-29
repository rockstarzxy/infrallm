---
title: "Day 15：Rollout 与语言模型 RL 基础"
type: concept
tags: [llm-training, rollout, policy-gradient, on-policy, advantage]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 15：Rollout 与语言模型 RL 基础

> 上一课 [[llm-training-30d/week2/day14-p1-project-sft-dpo]] · 下一课 [[llm-training-30d/week3/day16-ppo-grpo-family]]。相关：Agent 课的 POMDP 与策略梯度 [[agent-rsi-30d/week3/day15-agent-rl-foundations]]（那边侧重多轮 agent 形式化，这里侧重训练系统中的数据结构与实现）；推理课的采样与 logprobs [[ai-infra-30d/week2/day13-flash-attention]]。

## 学习目标

1. 给出 rollout 的精确定义，能画出一条 rollout 在 RL 训练系统里的数据结构，并说清每个字段由谁产生、被谁消费。
2. 从 RL 目标推导 REINFORCE 与带 baseline 的策略梯度，能解释 advantage、baseline、KL 惩罚各自解决什么问题。
3. 说清 on-policy 与 off-policy 在语言模型 RL 中的具体含义，以及"用旧 rollout 多更新几步"为什么需要重要性采样。
4. 区分 token-level 与 sequence-level 目标，能写出两者的梯度表达式并解释长度带来的偏差。
5. 用 numpy 实现 REINFORCE 并复现方差随 baseline 变化的曲线。

## 工业现状

所有当前的后训练 RL 系统都围绕 rollout 组织：DeepSeek-R1（T17）、Qwen3（T15）、Kimi k1.5（T18）在推理 RL 阶段对每个 prompt 采样一组 rollout（group size 通常 8 到 64），用 verifier 打分，再做 GRPO 类更新。Tulu 3（T22）的 RLVR 用 PPO。差别在算法细节，不在 rollout 这一层。

rollout 的成本主导整个训练：一次 GRPO 迭代中采样通常占 60% 到 80% 的墙钟时间，因为响应长度可达上万 token 且分布长尾。这就是 Week 3 后半段（Day 18 到 20）讲训练系统、rollout 引擎与异步的原因。理解 rollout 的数据结构，是读懂 verl、slime、AReaL 这些框架的前提。

术语在工业界的三种用法（Day 1 提过，这里固定下来）：

| 场景 | rollout 指 | 产物 |
|---|---|---|
| RL 训练 | 当前策略对一个 prompt 采样的一条完整 response（或多轮轨迹） | token、logprob、reward，进梯度 |
| 评测 | 对一题的多次独立采样 | 只用于估计 avg@n / pass@k |
| 数据合成 / 打标 | 模型采样出的候选输出 | 经筛选进 SFT 数据，或交给标注员评价 |

本课只讲第一种。

## 核心原理

### rollout 的定义与数据结构

给定 prompt x，策略 π_θ 自回归采样得到 response y = (y_1, ..., y_T)。一条 rollout 是元组：

```text
rollout = {
  prompt_ids:        [x_1 ... x_m]            # 由数据集给出
  response_ids:      [y_1 ... y_T]            # 由 rollout 引擎（vLLM/SGLang）采样
  old_logprobs:      [log π_old(y_t | x, y_<t)]  # 采样时策略的 logprob，T 个
  ref_logprobs:      [log π_ref(y_t | ...)]   # reference 模型，用于 KL
  reward:            r                        # 由 verifier / RM 给出，通常整条一个标量
  response_mask:     [1 ... 1, 0 ... 0]       # 有效 token 为 1，padding 与工具返回为 0
  finish_reason:     "stop" | "length"        # 截断的 rollout 需要特殊处理
}
```

一个 group 是同一 prompt 的 n 条 rollout。一个 batch 是 B 个 prompt 的 group，共 B·n 条 rollout。训练系统里的"一步"是：采样一个 batch 的 rollout → 打分 → 计算 advantage → 对 rollout 做若干次 mini-batch 梯度更新。

必须理解：`old_logprobs` 在采样时记录还是训练前重算，直接决定 on-policy 的严格程度（Day 19 展开）；`response_mask` 决定哪些 token 参与损失，多轮 agent 里工具返回的 token 必须为 0（Day 22）。

### RL 目标

语言模型 RL 是一个单步（bandit）或多步（agent）决策问题。单轮情况下，目标是：

```text
J(θ) = E_{x ~ D, y ~ π_θ(·|x)} [ r(x, y) ]  −  β · E_x [ KL( π_θ(·|x) || π_ref(·|x) ) ]
```

第一项要 reward 高，第二项限制策略不要离 reference（通常是 SFT 模型）太远。KL 项存在的原因有两个：reward 模型或 verifier 有盲区，策略走远后会找到 reward 高但质量差的区域（reward hacking）；保持生成分布接近 SFT 模型能防止能力遗忘。RLVR 场景里 DAPO（T37）等工作把 β 设为 0，因为 verifier 不容易被 hack，且希望策略能走得远；这是算法层面的取舍，Day 16 讨论。

### REINFORCE 与策略梯度

对第一项求梯度，用 log-derivative trick：

```text
∇_θ J = E_{y ~ π_θ} [ r(x, y) · ∇_θ log π_θ(y | x) ]
      = E [ r(x, y) · Σ_{t=1}^{T} ∇_θ log π_θ(y_t | x, y_<t) ]
```

这就是 REINFORCE：采样一条 rollout，把整条的 reward 乘到每个 token 的 log-prob 梯度上。问题是方差极大：reward 的绝对值大小与"这条比平均好多少"无关，但会整体放大或缩小梯度。

### baseline 与 advantage

减去一个与 y 无关的 baseline b(x) 不改变梯度期望，但能降方差：

```text
∇_θ J = E [ (r(x, y) − b(x)) · ∇_θ log π_θ(y | x) ]
A(x, y) = r(x, y) − b(x)      ← advantage：这条 rollout 比该 prompt 的"平均水平"好多少
```

baseline 的三种来源，对应三类算法：

| baseline | 算法 | 代价 |
|---|---|---|
| 学一个 value 模型 V_φ(x, y_<t) | PPO | 多一个与策略同规模的模型，训练不稳定 |
| 同一 prompt 其他 rollout 的平均 reward | GRPO、RLOO | 需要 group size n ≥ 2，无额外模型 |
| 移动平均等全局统计 | REINFORCE++ 的部分变体 | 最便宜，方差降得最少 |

group baseline 是当前推理 RL 的主流选择：同一 prompt 采 n 条，advantage 用组内均值（和标准差）归一化。直觉是"同一道题，这条比其他几条好"，天然对齐了 prompt 难度差异。

### on-policy 与 off-policy

策略梯度公式里的期望是在当前策略 π_θ 下取的。如果 rollout 是 π_old 采的，而更新已经让 θ 变了，梯度估计就有偏。修正方式是重要性采样：

```text
∇_θ J = E_{y ~ π_old} [ (π_θ(y|x) / π_old(y|x)) · A · ∇_θ log π_θ(y|x) ]
```

比值 π_θ/π_old 在序列级是 T 个 token 比值的乘积，会爆炸，所以实践中用 token 级比值并裁剪（PPO clip），或用序列级比值加长度归一化（GSPO）。Day 16 详细讲。

语言模型 RL 里的 off-policy 来源有三种，程度递增：

1. 一个 batch 的 rollout 用来做多次 mini-batch 更新（PPO 的 epochs > 1）：轻微 off-policy，用 clip 修正。
2. 异步训练，rollout 由几步前的权重采样：中度 off-policy，需要 staleness 上限与重要性采样（Day 20）。
3. 训练引擎与推理引擎数值不一致，`old_logprobs` 与训练侧重算的不一样：隐式 off-policy，B01 的发现，用 truncated importance sampling 修正（Day 19）。

### token-level 与 sequence-level

同一条 rollout 的损失可以按 token 平均，也可以按序列平均后再按 batch 平均：

```text
token-level:    L = (1 / Σ_i T_i) · Σ_i Σ_t  ℓ_{i,t}      # 长序列贡献更多 token，权重更大
sequence-level: L = (1 / B) · Σ_i (1 / T_i) · Σ_t ℓ_{i,t} # 每条序列权重相同，长序列每个 token 被稀释
```

两者的差异在响应长度分布不均时很大。GRPO 原始实现是 sequence-level，Dr. GRPO（T38）指出这会让长的错误回答受到更小的每 token 惩罚，从而鼓励错误回答变长；DAPO（T37）改用 token-level。这是 Week 3 反复出现的"长度偏置"问题的根源之一。

### KL 的两种放法

```text
放进 reward：  r' = r − β · log( π_θ(y|x) / π_ref(y|x) )    # PPO 常用，KL 进 advantage，被 value 模型拟合
放进 loss：    L = L_pg + β · KL_est                          # GRPO 常用，用 k3 估计器 π_ref/π_θ − log(π_ref/π_θ) − 1
```

放进 loss 的好处是 KL 不影响 advantage 的归一化；放进 reward 的好处是 value 模型能学到它。选择跟算法绑定。

### 知道即可

- 熵正则、entropy bonus 在 LLM RL 中偶尔用于防止熵坍塌（Day 20）。
- 多轮 agent 中 reward 在最后一步给出，credit assignment 靠 return-to-go 或 turn-level advantage（Day 22、25）。

## 实现步骤

### 1. 手写 REINFORCE（numpy，无框架）

用一个 toy 问题：策略是 10 个候选序列上的 softmax（参数 θ ∈ R^10），reward 是预设向量。目的是看方差，不是学语言。

```python
import numpy as np
rng = np.random.default_rng(0)
K = 10
r = np.array([0,0,0,0,0,0,0,0,0.9,1.0])       # 只有两条序列有 reward
theta = np.zeros(K)

def sample(theta, n):
    p = np.exp(theta - theta.max()); p /= p.sum()
    return rng.choice(K, size=n, p=p), p

def grad_reinforce(theta, n=8, baseline="none"):
    ys, p = sample(theta, n)
    rewards = r[ys]
    if baseline == "none":     A = rewards
    elif baseline == "group":  A = rewards - rewards.mean()
    elif baseline == "group_norm": A = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    g = np.zeros(K)
    for y, a in zip(ys, A):
        onehot = np.zeros(K); onehot[y] = 1
        g += a * (onehot - p)                    # ∇ log softmax
    return g / n

for baseline in ["none", "group", "group_norm"]:
    theta = np.zeros(K); grads = []
    for step in range(300):
        g = grad_reinforce(theta, n=8, baseline=baseline)
        grads.append(g); theta += 0.5 * g
    grads = np.array(grads)
    print(baseline, "final p(best)=", sample(theta, 1)[1][-1].round(3),
          "grad var=", grads.var(axis=0).sum().round(4))
```

预期：三种 baseline 最终都收敛到最优序列，但梯度方差依次下降，group_norm 收敛最快。把 n 从 8 改成 2 和 64 再跑，观察 group baseline 在 n=2 时的退化。

### 2. 用 HF 模型算一条 rollout 的全部字段

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16).cuda()
msgs = [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}]
prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").cuda()
with torch.no_grad():
    out = model.generate(prompt_ids, do_sample=True, temperature=1.0, max_new_tokens=32,
                         output_scores=True, return_dict_in_generate=True)
response_ids = out.sequences[0, prompt_ids.shape[1]:]
# 采样时的 logprob（old_logprobs）
old_logprobs = torch.stack([torch.log_softmax(s[0].float(), -1)[t] for s, t in zip(out.scores, response_ids)])
# 训练侧重算（同一模型，理论上应相等；差异来自 kernel 与 batch 形状）
with torch.no_grad():
    logits = model(out.sequences).logits[0, prompt_ids.shape[1]-1:-1].float()
recomputed = torch.log_softmax(logits, -1).gather(-1, response_ids[:, None]).squeeze()
print((old_logprobs - recomputed).abs().max())     # 记下这个数，Day 19 会解释它为什么不是 0
reward = float(tok.decode(response_ids, skip_special_tokens=True).strip() == "391")
```

把 `prompt_ids / response_ids / old_logprobs / ref_logprobs / reward / response_mask` 六个字段写成一个 dataclass，作为 Day 21 项目的数据格式。

### 3. 读 verl 的 rollout 数据结构

在 verl 源码里找 `DataProto`（`verl/protocol.py`），对照上面的字段：`input_ids`、`responses`、`old_log_probs`、`ref_log_prob`、`token_level_rewards`、`response_mask`、`advantages`。写一张映射表。字段名随版本变化，以当前源码为准。

## 实验

资源分层：实验 1 任何机器；实验 2、3 单卡。

### 实验 1：baseline 与方差

运行实现步骤 1，对 n ∈ {2, 4, 8, 16, 64} 与三种 baseline 画"收敛步数 vs n"和"梯度方差 vs n"两张图。记录：group baseline 需要的最小 n；group_norm 在 reward 全相同的 group 里会发生什么（std=0，advantage 全 0，这条 group 没有梯度，DAPO 的 dynamic sampling 正是针对它）。

### 实验 2：token-level 与 sequence-level 的长度效应

构造 4 条人工 rollout：两条正确（长 50、长 200 token），两条错误（长 50、长 200）。reward 1/0，group 归一化 advantage。分别用 token-level 与 sequence-level 计算每条序列在总损失里的权重。预期：sequence-level 下长的错误回答每 token 惩罚只有短回答的 1/4。写一段话解释这如何导致"错误回答越来越长"。

### 实验 3：old_logprobs 与重算 logprobs 的差异

用实现步骤 2 的脚本，分别在 batch size 1 与 batch size 8（padding 后）下重算 logprobs，比较与采样时 logprob 的最大差。再换成 vLLM 采样（`SamplingParams(logprobs=1)`）与 HF 重算比较。预期：同一框架内差异在 1e-3 量级，跨框架可达 1e-1。这个数就是 Day 19 讲的训练推理不一致。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 梯度方差大、reward 曲线剧烈抖动 | 没有 baseline 或 group size 太小 | 对比 n=2 与 n=16 的曲线 | 增大 n；用 group 归一化 |
| 某些 group 完全没有贡献 | reward 全相同，归一化后 advantage 为 0 | 统计每 batch 全对/全错 group 比例 | 过滤这类 prompt（dynamic sampling）或调整难度 |
| 错误回答越来越长 | sequence-level 平均稀释了长错误回答的惩罚 | 按正确性分组画长度曲线 | token-level 损失；overlong 惩罚 |
| 截断（finish_reason=length）的 rollout 拿到低 reward | verifier 把未完成当错误 | 统计截断比例 | 截断样本 mask 掉或给中性 reward；提高 max_tokens |
| old_logprobs 与训练侧差异大 | 训练与推理引擎数值不一致 | 实验 3 | Day 19 的重要性采样修正 |
| reward 上升但 KL 暴涨 | β 太小或没有 KL | 看 KL 曲线 | 加 KL 或 clip |
| 多轮数据里工具返回被算进 loss | response_mask 未正确置 0 | 打印一条轨迹的 mask | 修 mask 构造（Day 22） |

## 验收标准

- 能不看资料写出 rollout 的六个字段与 REINFORCE、advantage、重要性采样三个公式。
- 实验 1 的两张图与结论。
- 实验 2 的权重计算表与解释。
- 实验 3 的差异数值，并能说明它意味着什么。
- 一个 dataclass 定义的 rollout 数据格式，Day 21 直接用。

## 交付物

| 文件 | 内容 |
|---|---|
| `reinforce_toy.py` | 实现步骤 1 与实验 1 |
| `rollout_fields.py` | 实现步骤 2，含 dataclass |
| `rollout-notes.md` | 三种 rollout 含义、数据结构、公式推导、verl 字段映射表、实验 2/3 结论 |

## 参考

- T35 PPO：clip 与重要性采样在 LLM 之前的原始形式。
- T36 DeepSeekMath：GRPO 的 group baseline 提出。
- T38 Dr. GRPO：token-level 与 sequence-level 的长度偏置分析。
- T41 RLOO：leave-one-out baseline 与 REINFORCE 在 LLM 上的复兴。
- T42 HybridFlow：rollout 在训练系统中的数据流。
- B01：训练推理数值不一致导致的隐式 off-policy。
