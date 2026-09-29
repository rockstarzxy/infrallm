---
title: "Day 12：奖励模型：BT RM、生成式 RM、PRM 与过优化"
type: concept
tags: [llm-training, reward-model, bradley-terry, generative-rm, prm, goodhart]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 12：奖励模型：BT RM、生成式 RM、PRM 与过优化

> 上一课 [[llm-training-30d/week2/day11-preference-optimization-dpo]] · 下一课 [[llm-training-30d/week2/day13-evaluation-experiment-management]]。相关：RLVR 的 verifier 在 [[llm-training-30d/week3/day17-rlvr-verifiers-reward-design]]，agent reward 在 [[llm-training-30d/week4/day25-agent-reward-credit-assignment]]，Agent 课的 reward 设计 [[agent-rsi-30d/week3/day16-reward-design]]。

## 学习目标

1. 训练一个 Bradley-Terry 奖励模型，说清损失、数据、初始化与评测方法。
2. 区分四类奖励来源：BT 标量 RM、生成式 RM（LLM-as-judge 形式）、PRM/ORM、程序化 verifier，给出各自的适用边界。
3. 解释 RM 过优化（Goodhart）的机理，并用实验画出"KL 越大、RM 分数越高、真实质量先升后降"的曲线。
4. 说明 KL 惩罚、RM 集成、rubric 与不确定性各自如何缓解过优化。
5. 判断一个任务是否需要 RM：可验证的用 verifier，不可验证的用 RM，agent 任务为什么两者都难。

## 工业现状

- **InstructGPT（T23）** 确立 BT RM + PPO 的范式；RM 从 SFT 模型初始化，最后一层换成标量头。
- **Llama 3（T14）** 训练 RM 用于 rejection sampling 打分而不是 PPO；偏好数据带"明显更好 / 更好 / 略好 / 几乎一样"四档，训练时去掉"几乎一样"的对子；加了"edited response"作为第三档让 RM 学到细粒度差异。
- **DeepSeek-V3（T16）** 与 **Qwen3（T15）**：可验证任务用规则 verifier，不可验证任务用"带参考答案的生成式 RM"（模型读题、参考、回答后输出判断与理由），认为生成式 RM 比标量 RM 更抗 hacking、更可解释。DeepSeek-R1（T17）明确不用 PRM 做 RL 信号，理由是步骤定义困难、PRM 本身易被 hack。
- **Generative Verifiers（T34）**：把 RM 训练成"输出 Yes/No 的下一 token 预测"，可以用 CoT、多数投票，效果随推理算力增长。
- **PRM（T33）**：逐步骤打分，用于 best-of-N 与搜索时效果好，但作为 RL reward 会诱导"多写正确的废话步骤"。
- **rubric-based reward**（2025 起多家报告）：对开放任务写逐条评分标准，judge 按条给分；比整体打分稳定，和 Agent 课 Day 16 的做法一致。
- **RewardBench 类评测**：RM 在 chat、safety、reasoning 子集的偏好准确率，是选 RM 的第一参考，但高分 RM 在 RL 中不一定好（过优化难度不同）。

## 核心原理

### Bradley-Terry RM

```
数据：(x, y_w, y_l)
模型：r_φ(x, y) = 标量头(最后一个 token 的 hidden state)
损失：L = − E log σ( r_φ(x, y_w) − r_φ(x, y_l) )
```

- 初始化用 SFT 模型或与策略同架构的 instruct 模型（RM 需要理解回答）。
- 标量头接在序列末尾 token（通常是结束符）上；packing 时每条回答单独取末位。
- 分数没有绝对尺度，只有差值有意义；常加一个 `r(x, y_w) + r(x, y_l)` 的中心化正则防止漂移。
- 多档偏好可加 margin：`σ(r_w − r_l − m)`，Llama 3 的做法。

### 生成式 RM

```
输入：题目 x、（可选）参考答案、待评回答 y、评分说明
输出：推理过程 + 最终判断（Yes/No 或 1 到 10 或 rubric 逐条）
reward = P(Yes) 或 解析出的分数
```

优点：能用 CoT、能读参考答案、可解释、可用多数投票提升；缺点：每个样本一次生成（比标量 RM 贵 10 倍以上，示例）、输出格式需要解析。训练方式：SFT 到判断数据上（T34），或用 verifier 结果做 RL 训练 judge 自身。

### PRM 与 ORM

```
ORM：整条回答一个分数（正确/错误）
PRM：每个步骤一个分数，训练标签来自人工（T33）或蒙特卡洛估计（从该步继续采样的正确率）
```

PRM 用于推理时的 best-of-N（取步骤分最小值或乘积）与树搜索效果明显；作为 RL reward 时，模型会学会拆出更多"正确但无用"的步骤刷分，且步骤边界定义模糊（T17 的判断）。工业界 2025 后的推理 RL 主要用 outcome verifier，PRM 退回到推理时使用。

### 过优化（Goodhart）

RM 是真实偏好 `r*` 的有噪声估计，只在训练分布附近准确。策略 π 在 RM 上优化时离开这个区域：

```
d = KL(π || π_ref) 增大时：
  RM 分数 r_φ 单调上升
  真实质量 r*：先升（RM 准确区域）后降（策略找到 RM 的漏洞：长度、格式、套话、自信语气）
```

经验规律：真实质量峰值出现在某个 KL（与 RM 规模、数据量有关），RM 越大、数据越多，峰值越靠后。这就是 RLHF 要加 KL 惩罚、要早停、要用 hidden 评测的原因。

缓解手段：

| 手段 | 机理 | 代价 |
|---|---|---|
| KL 惩罚 / 早停 | 限制离开 RM 可靠区域 | 限制提升幅度 |
| RM 集成（多个 RM 取最小或均值） | 不同 RM 的漏洞不重合 | 多倍 RM 推理成本 |
| 不确定性惩罚 | 分歧大的样本降权 | 需要集成或 dropout 估计 |
| 迭代重训 RM | 用新策略的样本补 RM 的盲区（Llama 3 每轮重训） | 需要持续标注 |
| rubric / 生成式 RM | 逐条判定，难以用表面特征骗过 | 贵、需解析 |
| 长度与格式的显式惩罚 | 堵住最常见的漏洞 | 需要调权重 |
| 换 verifier | 可验证任务根本不用 RM | 只适用于可验证任务 |

### 什么时候不需要 RM

```
可程序化验证（数学答案、单元测试、格式、工具调用结果）  → verifier（Day 17）
部分可验证 + 主观（代码风格、解释质量）                  → verifier 硬约束 + RM/rubric 软分
完全主观（写作、对话）                                  → RM / 生成式 RM / DPO
agent 任务                                             → outcome verifier 为主，过程用 rubric 辅助（Day 25）
```

## 实现步骤

### 1. 用 TRL 训练 BT RM

```python
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification
rm = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", num_labels=1, torch_dtype="bfloat16")
rm.config.pad_token_id = tok.pad_token_id
cfg = RewardConfig(output_dir="out/rm", learning_rate=1e-5, num_train_epochs=1,
                   per_device_train_batch_size=4, gradient_accumulation_steps=8,
                   max_length=4096, bf16=True, gradient_checkpointing=True,
                   center_rewards_coefficient=0.01,     # 中心化正则；参数名以文档为准
                   logging_steps=10, eval_strategy="steps", eval_steps=200)
trainer = RewardTrainer(model=rm, args=cfg, train_dataset=pairs, eval_dataset=pairs_dev, processing_class=tok)
trainer.train()
```

数据格式 `{"chosen": messages, "rejected": messages}`，两边 prompt 相同。评测看 dev 偏好准确率，1.5B RM 在通用偏好上 70% 到 80% 是常见水平（示例）。

### 2. 生成式 RM（最小版本）

```python
JUDGE = """题目：{q}
参考答案：{ref}
待评回答：{y}
请逐步核对待评回答是否与参考答案等价且推理无误，最后一行只输出 Yes 或 No。"""
def gen_rm_score(q, ref, y, n=4):
    outs = llm.generate([JUDGE.format(q=q, ref=ref, y=y)] * n, SamplingParams(temperature=0.7, max_tokens=512))
    votes = [o.outputs[0].text.strip().splitlines()[-1].startswith("Yes") for o in outs]
    return sum(votes) / n           # 多数投票概率作 reward
```

用 verifier 能判定的题目生成 (q, ref, y, label) 数据，SFT 这个 judge，再拿到不可验证的题上用。

### 3. 过优化实验的搭建

```
π_0 = SFT 模型；RM_train = 上一步的 1.5B RM；RM_gold = 更强的 RM 或 judge（7B+）当"真实偏好"代理
用 best-of-N 模拟优化：N ∈ {1, 4, 16, 64, 256}，对每个 prompt 取 RM_train 最高分回答
记录：RM_train 平均分、RM_gold 平均分、平均长度
KL 的代理：log N − (N−1)/N（best-of-N 的 KL 上界）
```

best-of-N 不需要训练就能画出过优化曲线，是最便宜的复现方式；之后 Week 3 用 GRPO + RM 再画一次真实训练版。

### 4. RM 评测

- 偏好准确率：内部 dev 集 + RewardBench 类公开集。
- 长度相关性：同一 prompt 不同长度回答的分数与长度的 Spearman 相关，绝对值 > 0.3 说明有长度偏置（示例阈值）。
- 一致性：同一回答改写措辞后分数变化。
- 对抗集：套话、自信语气但错误、格式漂亮但空洞的回答，RM 应给低分。

## 实验

### 实验 1：训练 BT RM 并评测

2 万对通用偏好数据训练 1.5B RM，报告 dev 准确率、长度相关性、对抗集表现。单卡。

### 实验 2：best-of-N 过优化曲线

按实现步骤 3 跑 N 到 256（每 prompt 256 条采样，200 个 prompt，用 vLLM 约 1 小时，示例）。画 RM_train、RM_gold 随 log N 的曲线。预期 RM_gold 在某个 N 后下降，平均长度持续上升。

### 实验 3：生成式 RM vs 标量 RM 的抗 hacking

构造 300 条对抗回答（正确答案 + 大量废话；错误答案 + 自信措辞；格式模板正确但内容空）。比较 1.5B BT RM、生成式 RM（带参考答案）、verifier 的判定准确率。预期生成式 RM 明显优于标量 RM，verifier 最准但只覆盖可验证题。

### 实验 4：RM 集成

训练 3 个不同 seed / 数据子集的 RM，用 min 与 mean 集成重跑实验 2。预期集成推迟 RM_gold 的峰值。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| RM dev 准确率 ~50% | 数据噪声、prompt 两边不同、标量头没接对 | 抽检数据；打印 chosen/rejected 分数分布 | 清洗数据，检查末位 token |
| RM 分数绝对值漂移到几百 | 没有中心化正则 | 看分数均值随步数 | 加正则 |
| RM 偏好长回答 | 数据长度偏置 | 长度相关性 | 长度平衡数据、加长度惩罚 |
| RL 中 reward 上升评测下降 | 过优化 | hidden 评测、KL | KL 惩罚、早停、重训 RM |
| 生成式 RM 输出无法解析 | 格式不稳 | 解析失败率 | 约束输出格式或取 Yes token 概率 |
| PRM 做 RL 后回答步骤暴增 | 步骤刷分 | 平均步骤数曲线 | 改 outcome reward |
| RM 在新领域完全失效 | 分布外 | 对新领域抽检 | 补数据重训 |
| RM 与策略共享 tokenizer 不一致 | 模板差异 | 检查渲染 | 统一模板 |

## 验收标准

- 交付一个 BT RM，dev 准确率高于 70%（示例），并附长度相关性与对抗集结果。
- 实验 2 的过优化曲线明确显示 RM_gold 峰值位置。
- 能解释 KL 惩罚、集成、生成式 RM 各自缓解过优化的机理。
- 给出一份"任务 → 奖励来源"的判断表，覆盖数学、代码、对话、写作、agent。

## 交付物

| 文件 | 内容 |
|---|---|
| `out/rm-1.5b/` | RM checkpoint |
| `rm-eval.md` | 准确率、长度相关性、对抗集 |
| `overoptimization.md` | 实验 2/4 曲线与解释 |
| `reward-source-table.md` | 任务与奖励来源对照 |

## 参考

- T23 InstructGPT：BT RM 的训练细节与 PPO 中的 KL。
- T14 Llama 3 第 4.1.2 节：多档偏好、edited response、RM 每轮重训。
- T33 PRM：步骤级标注与 best-of-N 效果。
- T34 Generative Verifiers：把 RM 变成 next-token 预测。
- T16 DeepSeek-V3 第 5.1.2 节、T17 R1 第 2.2.2 节：规则 reward 与生成式 RM 的分工，不用 PRM 的理由。
- T15 Qwen3 第 4 章：通用 RL 阶段的 RM 使用。

## 附录 A：best-of-N 与 KL 的关系（实验 2 的横轴怎么来）

从 π_ref 采 N 条取 RM 最高的那条，得到的分布 π_BoN 相对 π_ref 的 KL 有解析上界：

```
KL(π_BoN || π_ref) ≤ log N − (N − 1) / N
N = 4   → 1.39 − 0.75 = 0.64 nats
N = 16  → 2.77 − 0.94 = 1.83
N = 64  → 4.16 − 0.98 = 3.17
N = 256 → 5.55 − 1.00 = 4.55
```

把 RL 训练得到的策略也按对 π_ref 的 KL 放到同一横轴上，可以直接比较"RL 走了多远"与"best-of-N 走了多远"时各自的真实质量。经验上 RL 在相同 KL 下过优化更严重，因为 RL 会系统性地找漏洞，而 best-of-N 只是重加权。

## 附录 B：奖励来源判断表

| 任务 | 首选奖励 | 辅助 | 不要用 |
|---|---|---|---|
| 数学（有标准答案） | 规则 verifier（答案等价判定） | 格式 reward | 标量 RM 作主信号 |
| 代码（有测试） | 沙箱测试通过率 | 生成式 RM 判风格 | 只看编译通过 |
| 数学/代码证明（无标准答案） | 生成式 RM + 参考解法 | 多数投票 | 单次标量分 |
| 指令遵循 | 可程序化检查的约束（IFEval 式） | RM | 纯 judge |
| 对话有用性、写作 | BT RM 或 rubric 生成式 RM | DPO 直接用偏好对 | PRM |
| 安全 | 分类器 + rubric | 人工抽检 | 单一 RM（易 over-refusal） |
| 工具调用 / agent | 任务完成 verifier（hidden tests） | 过程 rubric（步数、冗余调用） | 过程分作主信号 |
| 长文档摘要 | 事实一致性检查（NLI/引用核对）+ RM | 长度约束 | 长度相关的 RM |

## 附录 C：自检问题

1. BT RM 的分数为什么没有绝对意义？两个不同 RM 的分数能直接比较吗？
2. RM 从 SFT 模型初始化还是从 base 初始化更好？为什么？
3. 生成式 RM 比标量 RM 贵多少？什么情况下值得？
4. 为什么 R1 不用 PRM 做 RL 信号？PRM 还有什么用？
5. 过优化曲线的峰值位置由什么决定？怎么把峰值推后？
6. 一个 RM 在 RewardBench 上 90% 准确率，在 RL 中一定好用吗？
