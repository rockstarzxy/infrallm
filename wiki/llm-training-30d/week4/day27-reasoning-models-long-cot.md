---
title: "Day 27：推理模型训练：长 CoT、R1 配方与蒸馏"
type: concept
tags: [llm-training, reasoning, long-cot, rlvr, distillation]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 27：推理模型训练：长 CoT、R1 配方与蒸馏

> 上一课 [[llm-training-30d/week4/day26-case-studies-swe-tool-rl]] · 下一课 [[llm-training-30d/week4/day28-safety-alignment-rlhf-production]]。相关：推理课的长上下文与推测解码 [[ai-infra-30d/week4/day27-long-context]]、[[ai-infra-30d/week2/day12-speculative-decoding]]（长 CoT 模型的 serving 成本主要在 decode，训练侧的长度控制直接决定推理侧成本）。

## 学习目标

1. 能完整复述 DeepSeek-R1 的多阶段配方，说明每一阶段解决上一阶段的什么问题。
2. 能解释 RL 训练中 response 长度为什么会自发增长、何时是能力提升何时是退化，并给出三种长度控制手段的机制。
3. 能配置 Qwen3 式 thinking / non-thinking 融合训练的数据格式与 thinking budget 机制。
4. 能设计从大推理模型蒸馏到 1.5B 到 8B 小模型的数据流水线，并说明为什么蒸馏在小模型上通常优于直接 RL。
5. 能用 AIME 类评测的 avg@n 口径正确报告推理模型的结果。

## 工业现状

2025 年推理模型的公开配方（T17、T15、T18、T21）收敛为同一骨架：

```text
基座（预训练 + 中期训练含长上下文）
  → [可选] R1-Zero 式纯 RL 探索：验证 RL 能否从基座直接激发长 CoT
  → 冷启动 SFT：少量高质量长 CoT 数据，解决可读性与语言混杂
  → 推理 RL（RLVR）：数学、代码、逻辑等可验证任务，规则 reward + 语言一致性 reward
  → 拒绝采样 SFT：用 RL 策略采样、过滤，混入通用数据（写作、问答、安全）再 SFT
  → 全场景 RL：可验证任务继续规则 reward，通用任务用 RM
  → 蒸馏：用最终模型的输出 SFT 小模型（R1-Distill 系列）
```

各家的差异：Qwen3（T15）把 thinking 与 non-thinking 融合进一个模型，并提供 thinking budget；Kimi k1.5（T18）强调长上下文 RL 与 long-to-short 蒸馏；MiniMax-M1（T21）把输出预算分阶段放大到数万 token 并用 CISPO；GLM-4.5（T20）直接在 64K 上下文做单阶段推理 RL 并加难度课程。共识：**推理能力主要来自 RL 阶段，SFT 阶段决定格式与可读性，蒸馏是小模型的最优路径**。

## 核心原理

### 为什么 RL 会让 CoT 变长

RLVR 只奖励最终答案。对难题，更长的推理（验证、回溯、分支）提高正确率，策略梯度自然增加这类 token 的概率。R1-Zero 报告中 response 长度随训练步数持续增长，并出现"反思"类语句，这是能力提升的信号。但长度增长有两个退化形态：

- **无效冗长**：正确率不再提升，长度仍增长。常见原因是长度与 reward 弱相关（长回答略高的通过率被放大），或 GRPO 的长度偏置（T38 指出对错误回答的 token 级平均会鼓励更长的错误回答）。
- **截断塌陷**：长度撞到 max_tokens 被截断记 0 分，策略学到"短一点"，正确率下降。

必须理解的量化关系（示例，非普适）：把 max_response_length 从 8K 提到 32K，训练每步的 rollout 时间和显存都随最长样本增长，长尾几条样本决定一步的墙钟。这是推理 RL 的核心工程矛盾。

### 长度控制的三种机制

| 手段 | 机制 | 副作用 | 出处 |
|---|---|---|---|
| overlong reward shaping | 接近上限时线性扣分，超过记惩罚而非直接 0 | 需要选拐点；对短任务无影响 | T37 |
| 长度惩罚 / 预算奖励 | reward 减去 α·len 或按预算内外分段给分 | α 过大直接塌陷 | T18、T21 |
| Dr. GRPO 去长度归一化 | 不对 token 数做平均，避免错误回答"越长越便宜" | 需重新扫 lr | T38 |
| 分阶段放大预算 | 先 16K 训稳，再 32K、48K… | 每阶段要重新观察长度分布 | T21 |
| budget forcing（推理时） | 到预算强制输出结论，或插入"再想想"延长 | 训练侧不改，属推理策略 | 多篇 test-time scaling 工作 |

### thinking / non-thinking 融合（Qwen3，T15）

数据格式：thinking 样本带 `<think>…</think>` 段，non-thinking 样本带空的 think 段，用对话里的开关（如 `/think`、`/no_think`）控制。融合阶段先 SFT 混合数据，再做 RL；thinking budget 由推理时在达到预算后强制关闭 think 段实现，模型在 SFT 阶段见过"被截断的思考后直接作答"的样本。训练侧要点：两类样本的 loss mask 相同；RL 阶段要防止 non-thinking 模式偷偷变长。

### 蒸馏为什么在小模型上更好

T17 的实验：直接对 32B 做大规模 RL，不如用 R1 输出蒸馏 32B。原因：小模型的 pass@k 太低，RLVR 的 group 大多零梯度；蒸馏直接提供高质量长 CoT 的分布。工业实践：小模型先蒸馏（离线 SFT，或 B02 的 on-policy 蒸馏用教师 logprob 逐 token 监督），再用少量 RL 精修。on-policy 蒸馏的优势是学生在自己的采样分布上学，避免 exposure bias，且每 token 有稠密信号，比 RL 的稀疏 reward 样本效率高一个量级以上（B02 的报告口径）。

### 评测口径

AIME 24/25 只有 30 题，单次采样方差极大。必须报 avg@n（n 通常 16 到 64）和标准误；pass@k 只用于分析上限。题目时间窗要晚于训练数据截止以防污染（Day 13）。

## 实现步骤

### 1. 冷启动 SFT 数据

- 来源：用现有推理模型对题库采样，过滤答案正确、格式规范、语言一致的样本；或少量人工整理。
- 规模：R1 报告为数千条量级（以报告为准）；目的不是教会推理，而是固定格式与可读性。
- 格式：`<think>` 推理 `</think>` 答案，答案用固定标记（如 `\boxed{}`）便于 verifier 抽取。
- 训练：Day 8 的 SFT 流程，1 到 2 epoch，lr 小。

### 2. 推理 RL 配置（verl 风格，字段名以 B06 当前版本为准）

```yaml
data:
  train_files: data/math_rlvr/train.parquet      # 题目 + 标准答案，已去污染
  max_prompt_length: 1024
  max_response_length: 16384                     # 分阶段放大：8K → 16K → 32K
actor_rollout_ref:
  actor:
    optim.lr: 1e-6
    use_kl_loss: false                           # R1 类配方常不用 KL 或用极小系数
    clip_ratio_high: 0.28
    loss_agg_mode: token-mean                    # 或 seq-mean-token-sum，配合 Dr.GRPO 消融
  rollout:
    n: 16
    temperature: 1.0
    top_p: 1.0                                   # 不截断分布，保持 on-policy
algorithm:
  adv_estimator: grpo
  filter_groups.enable: true
reward_model:
  reward_manager: rule                           # 答案抽取 + 等价判定 + 格式 + 语言一致性
  overlong_buffer: {enable: true, len: 4096, penalty_factor: 1.0}   # DAPO 式
```

reward 函数：抽取 `\boxed{}` 内容，用符号等价（sympy）或字符串归一化比较；格式不合法记 −1；语言混杂按目标语言比例扣分（T17 的语言一致性 reward）。

### 3. 拒绝采样 SFT

用 RL 后的策略对推理题库采样 n 条，保留正确且可读的；对通用任务用 RM 或 judge 打分保留高分；按比例混合（推理数据占比示例 60% 到 80%），再 SFT 一次。这一步把 RL 学到的推理能力"固化"并恢复通用能力。

### 4. 蒸馏到小模型

```text
教师（最终推理模型）对题库采样 → 过滤正确 + 长度上限 → 学生 SFT（离线蒸馏）
可选：on-policy 蒸馏（B02）：学生采样，教师给每个 token 的 logprob，loss = 反向 KL 的逐 token 近似
```

学生规模 1.5B 到 8B；SFT 后再做 100 到 300 步 RLVR 精修，观察是否继续提升。

### 5. 规则 reward 的实现细节

```python
import re
from sympy import simplify, sympify

BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")

def extract_answer(text: str):
    m = BOX.findall(text)
    return m[-1].strip() if m else None      # 取最后一个 boxed，防止思考段里的中间结果

def normalize(s: str) -> str:
    return s.replace(" ", "").replace("\\left", "").replace("\\right", "").rstrip(".")

def equivalent(pred: str, gold: str) -> bool:
    if normalize(pred) == normalize(gold):
        return True
    try:                                      # 符号等价：1/2 与 0.5、x^2+2x+1 与 (x+1)^2
        return simplify(sympify(pred) - sympify(gold)) == 0
    except Exception:
        return False

def lang_consistency(text: str, target="zh") -> float:
    # 目标语言字符比例；示例实现，按任务替换
    zh = sum("\u4e00" <= c <= "\u9fff" for c in text)
    alpha = sum(c.isalpha() for c in text)
    return zh / max(1, zh + alpha) if target == "zh" else 1 - zh / max(1, zh + alpha)

def compute_score(response: str, gold: str, max_len: int, buffer: int = 4096):
    if "<think>" not in response or "</think>" not in response:
        return -1.0                                          # 格式罚
    pred = extract_answer(response.split("</think>")[-1])
    r = 1.0 if (pred is not None and equivalent(pred, gold)) else 0.0
    r -= 0.1 * (1 - lang_consistency(response))              # 语言一致性
    n = len(response)                                        # 实际用 token 数
    if n > max_len - buffer:                                 # DAPO overlong shaping
        r -= min(1.0, (n - (max_len - buffer)) / buffer)
    return r
```

三个必须测试的坑：思考段里出现 `\boxed` 导致误抽取；`sympify` 对形如 `x=3` 的输入会抛异常；带单位的答案（`3 cm`）需要专门归一化。verifier 的单元测试要和训练代码一起提交。

### 探索与 entropy

推理 RL 里 entropy 是最需要盯的量之一：

```text
entropy 快速下降 → 策略过早确定 → pass@k 上限收窄 → 后续难题无法解
entropy 不降甚至上升 → 采样质量差 → reward 噪声大
```

公开配方的做法：clip-higher（放宽正向 clip，让低概率 token 有机会被提升，T37）；不加 KL 或极小 KL（让策略能离开参考分布）；温度 1.0 保持采样多样性；dynamic sampling 保证每步都有非零梯度。DAPO 报告了 clip-higher 对 entropy 崩塌的缓解，作为对照实验值得复现（Day 20 的稳定性面板同样适用）。

### test-time scaling 与训练的关系

推理时多采样、投票、或用 PRM 选择（best-of-n）能提高正确率，但这是"消费"训练出的 pass@k 分布，不产生新能力。训练侧的目标是抬高 pass@1 同时不压低 pass@k；RL 若把 entropy 压得太低，pass@1 升而 pass@64 降，test-time scaling 的收益就没了。报告结果时同时给 pass@1（avg@n）与 pass@k，能看出这一点。

### 数据去污染（推理题库专用）

- n-gram 重叠：题干 13-gram 与评测集重叠即剔除。
- 语义去重：embedding 相似度阈值（示例 0.9）剔除改写题。
- 时间窗：评测用训练数据截止后发布的题（AIME 25、LiveCodeBench 新窗口）。
- 记录每个评测集的污染检查结果，写进报告。

## 面试题（5 题，附要点）

1. **R1 为什么先做 R1-Zero？** 证明基座可直接经 RL 激发长 CoT，同时暴露可读性与语言混杂问题，于是引入冷启动 SFT。
2. **长度增长什么时候是好事？** 正确率同步上升且截断率低；反之是无效冗长或长度偏置。
3. **蒸馏为什么在小模型上优于直接 RL？** 小模型 pass@k 低，RLVR 零梯度 group 多；蒸馏提供稠密、高质量分布；on-policy 蒸馏还避免 exposure bias。
4. **Qwen3 的 thinking budget 是训练出来的还是推理时的？** 主要是推理时强制关闭 think 段，训练侧通过融合 SFT 让模型见过截断后作答的样本。
5. **为什么推理 RL 常不用 KL？** 目标是让策略显著偏离基座去学新行为，KL 会限制探索；用 clip 与 entropy 监控替代稳定性保障；但通用任务 RL 仍常保留 KL。

## 实验

资源分层：API-only 可做实验 1（用公开推理模型的输出分析长度与正确率关系）；单卡 24 到 48 GB 用 1.5B 做实验 2、4；8 卡做实验 3。

### 实验 1：长度与正确率的关系

对 500 道数学题用同一模型采样 16 条，按长度分桶统计正确率。预期：中等长度桶正确率最高，最长桶因截断或循环下降。这是设定长度惩罚拐点的依据。

### 实验 2：长度控制手段对比

1.5B 模型、同一 RL 配置各训 300 步：无控制、overlong shaping、Dr. GRPO 去归一化。记录 avg@16、平均长度、截断率。预期：无控制的长度持续增长且截断率上升；两种控制手段长度平稳，avg@16 接近或更高。

### 实验 3：分阶段放大预算

8K 训 200 步 → 16K 再 200 步，与直接 16K 训 400 步对比。记录每步墙钟与最终 avg@16。预期：分阶段总墙钟更短、结果相当。

### 实验 4：蒸馏 vs 直接 RL

1.5B 学生：a) 直接 RLVR 500 步；b) 教师蒸馏 SFT 后 RLVR 200 步。同算力预算比较 avg@16。预期 b 显著更好，与 T17 结论一致。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 长度持续增长，正确率不涨 | 长度偏置、无效冗长 | 实验 1 的分桶 | overlong shaping；Dr. GRPO；降 top-p 不是办法（会 off-policy） |
| 长度骤降，正确率下降 | 长度惩罚过强或截断记 0 | 看截断率与惩罚项均值 | 用 shaping 代替硬 0；放大预算 |
| 输出中英混杂 | 无语言一致性 reward | 统计非目标语言 token 比例 | 加语言一致性 reward（T17） |
| RL 后通用能力（写作、对话）下降 | 只训推理任务 | 通用评测曲线 | 拒绝采样 SFT 混入通用数据；全场景 RL |
| think 段与答案矛盾 | 冷启动数据质量低 | 抽样人工看 | 提高冷启动过滤标准；答案标记规范 |
| non-thinking 模式也变长 | 融合 RL 阶段未约束 | 分模式长度分布 | 对 non-thinking 样本加长度预算 reward |
| AIME 波动 ±10 分 | 样本太少 | 看标准误 | avg@64；同时报多个题集 |
| 蒸馏学生格式对但答案错 | 教师输出未过滤正确性 | 过滤前后对比 | 只保留 verifier 通过的教师样本 |

## 验收标准

- 能画出 R1 配方的阶段图并说明每阶段的输入、输出和目的。
- 实验 1 与 2 完成，能用数据说明选择的长度控制手段。
- 交付的蒸馏流水线能产出一个 1.5B 学生，avg@16 相对基座有可测提升，并报告标准误。

## 交付物

| 文件 | 内容 |
|---|---|
| `reasoning-recipe.md` | 多阶段配方图、每阶段数据与超参、与 Qwen3/K1.5/M1 的差异 |
| `length-control-report.md` | 实验 1 到 3 的数据、长度分布图、截断率曲线 |
| `distill/` | 教师采样与过滤脚本、学生 SFT 配置、评测结果（avg@n + 标准误） |

## 参考

- T17 DeepSeek-R1：多阶段配方、语言一致性 reward、蒸馏优于直接 RL 的实验。
- T38 Dr. GRPO：长度与难度偏置的来源与修正。
- T37 DAPO：overlong reward shaping 与 dynamic sampling。
- T15 Qwen3：thinking / non-thinking 融合与 thinking budget。
- T18 Kimi k1.5：长上下文 RL 与 long-to-short。
- T21 MiniMax-M1：分阶段放大输出预算。
- B02 on-policy 蒸馏：逐 token 教师监督的做法与样本效率。
