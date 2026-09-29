---
title: "Day 25：Agent Reward 设计与 Credit Assignment"
type: concept
tags: [llm-training, agentic-rl, reward-design, credit-assignment, reward-hacking]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 25：Agent Reward 设计与 Credit Assignment

> 上一课 [[llm-training-30d/week4/day24-agent-rollout-infrastructure]] · 下一课 [[llm-training-30d/week4/day26-case-studies-swe-tool-rl]]。相关：Agent 课的 reward 设计 [[agent-rsi-30d/week3/day16-reward-design]] 与 credit assignment [[agent-rsi-30d/week3/day17-credit-assignment]] 讲的是"该奖励什么"；本课讲训练系统里 reward 怎么算、怎么分到轨迹的哪一段、怎么防止被钻空子。

## 学习目标

1. 能为一个多轮工具任务写出 outcome、process、rubric 三层 reward 的规范，并说明每一层的计算成本和可被 hack 的方式。
2. 能把一条 agent rollout（多次模型调用 + 工具返回）的 reward 分配到 turn 级和 token 级，并写出对应的 advantage 计算代码。
3. 能设计 hidden tests 与审计流程，在训练日志里识别 reward hacking 的四种典型形态。
4. 能解释稀疏 reward 下 GRPO group 归一化为什么在 agent 任务上容易退化，以及三种缓解方法的取舍。
5. 交付一份可执行的 reward spec 和一个 reward 审计脚本。

## 工业现状

2025 到 2026 年公开的 agentic RL 配方在 reward 上高度收敛于**可验证的 outcome reward**，process reward 只作辅助：

| 配方 | 主 reward | 辅助信号 | 防 hack 手段 | 出处 |
|---|---|---|---|---|
| SWE-RL | 补丁与真实 PR 补丁的相似度（连续值），后续工作改为测试通过 | 格式惩罚 | 只用 GitHub 真实 issue/PR 对 | T45 |
| Kimi K2 | 可验证任务用 verifier，开放任务用自评 rubric（self-critique rubric reward） | 长度、格式 | 大规模合成 agentic 环境，reward 与训练解耦 | T19 |
| GLM-4.5 | agentic 任务测试通过率，配合 expert iteration 迭代数据 | 步骤成功 | 多轮 hidden 环境 | T20 |
| DeepSeek-R1 | 规则 reward（答案正确 + 格式） | 语言一致性 | 明确不用 PRM，理由是 PRM 易被 hack 且难标注 | T17 |
| Tulu 3 | RLVR：可验证正确性 | 无 | 与 RM 训练分开 | T22 |

共同点：**能程序化验证的就不用模型判**；必须用模型判的（开放写作、rubric），用生成式 RM（T34）并限制它的权重。PRM（T33）在数学推理上有价值，但在 agent 任务上因为"中间步骤对不对"没有 ground truth，主流配方几乎不用于训练，只用于分析。

## 核心原理

### 三层 reward

```text
outcome reward  R_o ∈ {0,1} 或 [0,1]     任务终态：测试通过、答案正确、SQL 结果匹配
process reward  r_t 每个 turn           工具调用是否合法、是否重复、是否推进
rubric reward   R_r ∈ [0,1]             LLM judge 按 rubric 打分（代码质量、安全、可读性）
R = R_o + λ_p Σ_t r_t + λ_r R_r - λ_len f(steps) - λ_fmt 1[格式违规]
```

必须理解的三条：

- **outcome 是锚**。没有 outcome 的任务不要上 RL，先做 SFT 或换任务。process 和 rubric 的系数 λ 要小到不能单独把一条失败轨迹拉成"好"轨迹。经验上 λ_p Σ r_t 的上限设在 outcome 满分的 20% 到 30%（示例，按任务调）。
- **process reward 只惩罚不奖励**是更安全的形态：重复调用、非法参数、超时各扣分，但不给"调用了工具"加分，否则策略学会刷工具调用。
- **rubric reward 必须有 hidden rubric**。judge 看到的 rubric 和策略在 prompt 里看到的不能完全一样，否则策略直接对着 rubric 写。

### Credit assignment：reward 落到哪一段

一条 agent rollout：

```text
turn 1: [assistant tokens a_1] → tool → [obs o_1]
turn 2: [assistant tokens a_2] → tool → [obs o_2]
...
turn T: [assistant tokens a_T] → 终态 → R_o
```

三种分配方式：

| 方式 | advantage | 优点 | 缺点 | 何时用 |
|---|---|---|---|---|
| trajectory-level | 所有 assistant token 共享 A = R - baseline | 实现最简单，verl/slime 默认 | 早期正确决策与晚期错误决策同奖同罚 | 轨迹短（< 10 turn）、reward 稠密 |
| turn-level | A_t = G_t - b，G_t = 未来 reward 折扣和 | 区分 turn 贡献 | 需要每 turn 有 r_t 或 value 估计 | 有 process reward 或 PRM |
| token-level（value model） | PPO 风格 GAE | 最细 | value model 在长轨迹上难训练，多数配方已放弃 | 短序列单轮 |

工业主流是 trajectory-level + GRPO group 归一化。要理解它在 agent 上的退化：

```text
group 内 n 条 rollout 的 R 全是 0（任务太难）或全是 1（太简单）
→ 归一化后 advantage 全为 0 → 该 prompt 贡献零梯度
agent 任务 pass rate 常在 5% 以下 → 大部分 group 无梯度 → 有效 batch 骤减
```

缓解：DAPO 的 dynamic sampling（过滤全对全错 group，补采样到 batch 满，T37）；按难度分桶做 curriculum；把 outcome 从二值改成分级（通过的测试比例）。分级 reward 要小心：测试比例可被"删测试"hack，必须用 hidden tests。

### loss mask 与 reference

observation token 不进 policy loss，也不进 KL：

```python
# 简化：构造 agent rollout 的 loss mask
mask = torch.zeros(len(tokens))
for span in assistant_spans:          # 每个 turn 的 assistant 生成区间
    mask[span.start:span.end] = 1
# tool 返回、system、user 全为 0
kl = (logp_policy - logp_ref) * mask
```

多轮里 reference 模型的 logprob 要在**同样的上下文**（含所有工具返回）下重算，不能复用单轮缓存。

### reward 的尺度与归一化（必须理解）

GRPO 的 group 归一化让 reward 的绝对尺度不重要，但**组内相对结构**决定一切：

```text
outcome 二值 + process 小罚分：组内成功轨迹之间靠 process 排序，失败轨迹之间也靠 process 排序
→ 策略同时学"成功"和"少犯错"，两者不冲突
outcome 分级（测试通过比例）：组内可能出现 0.6 vs 0.7 的细微差别被放大成 ±1 的 advantage
→ 噪声测试（flaky test）会产生随机梯度，必须先修 flaky
```

Dr. GRPO（T38）指出 std 归一化会放大"全组都接近同分"的 prompt 的梯度，建议去掉 std 只减 mean；在 agent 任务上这个问题因 reward 稀疏更明显。两种都跑一遍再决定。

另一个尺度问题是 KL 项：reward 在 [−0.5, 1] 时 KL 系数 0.001 与 0.01 的含义完全不同。改 reward 尺度后要重新扫 KL 系数，不能沿用数学 RLVR 的默认值。

### 与 Agent 课的分工

| 问题 | Agent 课 Day 16/17 | 本课 Day 25 |
|---|---|---|
| 奖励什么 | 任务目标、可验证性、reward spec 的产品含义 | 作为输入 |
| 怎么算 | 概念 | verifier 服务、hidden tests、缓存、超时 |
| 怎么分 | 折扣、baseline 的数学 | trajectory/turn/token 三种实现与框架接口 |
| 怎么防 hack | 原则 | 审计脚本、分叉实验、日志指标 |

## 实现步骤

### 1. 写 reward spec（先于代码）

```yaml
task: text2sql
outcome:
  type: exec_match            # 执行结果集匹配
  hidden_db_variants: 3       # 同 schema 不同数据，防止硬编码答案
  score: {match: 1.0, partial_columns: 0.3, error: 0.0}
process:
  penalties:
    invalid_tool_args: -0.05
    repeated_identical_call: -0.05
    exceed_max_turns: -0.2
  cap: -0.3
rubric:
  enabled: false              # text2sql 不需要
length:
  max_turns: 8
  per_turn_over_4: -0.02
format:
  final_answer_tag_required: true
  violation: -0.2
```

### 2. verl 里挂 reward function

```python
# verl 的 custom reward：接收 data_source, solution_str, ground_truth, extra_info
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    meta = extra_info or {}
    trace = meta["trace"]                       # 由 agent loop 写入的轨迹
    r_o = exec_match(trace.final_sql, ground_truth, meta["hidden_dbs"])
    r_p = max(-0.3, process_penalties(trace))
    r_len = -0.02 * max(0, trace.num_turns - 4)
    r_fmt = -0.2 if not has_final_tag(solution_str) else 0.0
    return float(r_o + r_p + r_len + r_fmt)
```

配置 `reward_model.reward_manager` 指向自定义函数（参数名随版本变化，以当前 verl 文档 B06 为准）。多轮 agent loop 的 trace 如何进入 `extra_info` 取决于框架的 agent 接口，Day 23 已建立。

### 3. turn-level advantage（可选）

```python
def turn_level_advantage(turn_rewards, gamma=1.0):
    # turn_rewards: [r_1..r_T]，最后一项含 outcome
    G, out = 0.0, []
    for r in reversed(turn_rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    b = sum(out) / len(out)           # 轨迹内 baseline，或换成 group 内同 turn 均值
    return [g - b for g in out]
```

把每个 turn 的 advantage 广播到该 turn 的 assistant token 上，再进 GRPO/PPO 的 clip 目标。

### 4. hidden tests 与审计

- 训练用的 verifier 只暴露公开测试；hidden tests 只在评测与审计时跑。
- 每 N 步抽 100 条 reward 最高的 rollout 人工或 judge 审计：是否修改了测试文件、是否绕过工具直接输出、是否在 final answer 里塞多个候选。
- 记录 `reward_o`、`reward_p`、`steps`、`tool_error_rate` 的分布随训练步数的变化。

## 实验

资源分层：API-only 可做实验 1 的离线分析；单卡 24 到 48 GB 可用 1.5B 模型跑实验 2、3；4 到 8 卡跑完整对比。

### 实验 1：reward 分解对训练信号的影响（离线）

固定 Day 24 采集的 500 条 rollout。分别用 outcome-only、outcome + process、outcome + process + length 三种 spec 计算 reward，统计每个 prompt 的 group 内 reward 方差和"零梯度 group"比例。预期：加入 process 惩罚后零梯度比例下降，但要检查是否出现"失败轨迹因少调工具反而分数更高"的反转。

### 实验 2：trajectory-level vs turn-level

同一任务、同一模型（1.5B）、同一 seed，各训 200 步。记录 hidden eval 成功率、平均步数、每 turn 的工具错误率。预期：turn-level 在步数控制上更好，成功率差异不一定显著；报告置信区间。

### 实验 3：reward hacking 注入

故意把 hidden tests 关掉，只用公开测试训练 300 步，观察 reward 曲线与 hidden eval 的分叉点，并抽样看策略学到了什么。这是理解"reward 上升不等于能力上升"最直接的方法。

### 实验 4：手算一个 group 的 advantage（必做，纸上）

一个 prompt 采 8 条 agent rollout，outcome 与 process 罚分如下（示例）：

| rollout | 步数 | R_o | process 罚 | 总 R |
|---|---:|---:|---:|---:|
| 1 | 3 | 1.0 | 0 | 1.00 |
| 2 | 6 | 1.0 | -0.05 | 0.95 |
| 3 | 8 | 0.0 | -0.20 | -0.20 |
| 4 | 2 | 0.0 | 0 | 0.00 |
| 5 | 5 | 0.0 | -0.05 | -0.05 |
| 6 | 7 | 1.0 | -0.10 | 0.90 |
| 7 | 8 | 0.0 | -0.20 | -0.20 |
| 8 | 4 | 0.0 | 0 | 0.00 |

GRPO advantage：A_i = (R_i - mean) / std。算出 mean ≈ 0.30，std ≈ 0.50，于是 rollout 4 和 8（两步就放弃）的 A ≈ -0.6，rollout 3 和 7（八步失败）的 A ≈ -1.0。问题：策略从这组数据学到"少尝试"还是"多尝试"？把 process 罚分去掉再算一遍，答案会反转吗？把这个推演写进报告。这是 λ 系数为什么必须小的直观证据。

## 补充：多目标与 reward 服务化

### 多目标 reward 的两种做法

线性加权（上面的 R 公式）简单但系数难调；另一种是**约束式**：只优化 outcome，把步数、成本作为硬约束进 rollout（超过即截断并判失败）。约束式不会出现"用步数换成功率"的隐式交易，工业配方里长度控制多用约束（max_turns、max_tokens 截断记 0 分或 overlong shaping，T37）。

### reward 服务化

agent 任务的 outcome 计算通常要跑测试或查数据库，耗时从百毫秒到分钟。训练侧要求：

```text
rollout worker ──HTTP/gRPC──► reward service（无状态、可水平扩展）
                                ├─ 沙箱池：预热容器，按任务快照 reset
                                ├─ 超时：单条 reward 上限（示例 60 s），超时记失败并打标 timeout
                                ├─ 幂等：同 (task_id, trajectory_hash) 缓存结果
                                └─ 审计日志：输入摘要、输出分数、耗时、沙箱 stdout 尾部
```

三个必须监控的量：reward 计算 P99 延迟（它直接决定一步 RL 的墙钟）、timeout 率（高了说明 reward 分布被截断污染）、缓存命中率（高了说明 rollout 多样性不足）。

### PRM 为什么不进训练

T33 的 PRM 在数学上能给每一步打分，但 agent 任务的"步"是工具调用，标注"这一步好不好"需要知道后续所有可能路径，人工标注不可靠，模型标注则是又一个可被 hack 的 RM。T17 明确记录了这一判断。实践里 PRM 只用于离线分析（找出常见失败 turn）和 best-of-n 选择，不进 policy loss。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| reward 上升，hidden eval 不动或下降 | reward hacking | 抽高分 rollout 看行为；对比公开测试与 hidden 通过率 | 加 hidden tests；去掉可被游戏的分级项 |
| 有效梯度 batch 很小、loss 接近 0 | group 内 reward 全同 | 统计零方差 group 比例 | dynamic sampling；难度分桶；分级 outcome |
| 步数持续增加 | process reward 奖励了调用 | 看 tool call 次数分布随步数 | process 只罚不奖；加步数惩罚 |
| 步数坍缩到 1，直接猜答案 | 长度惩罚过强或 outcome 可猜 | 看首 turn 就输出 final 的比例 | 降 λ_len；outcome 加 hidden 变体 |
| rubric 任务分数虚高 | judge 被 prompt 注入或 rubric 泄漏 | 用另一 judge 复评；检查 response 里是否出现 rubric 关键词 | hidden rubric；judge 输入去掉策略的自评段 |
| KL 快速上升 | 稀疏 reward 下少数高分轨迹主导更新 | 看 advantage 分布的峰度 | advantage clip；降 lr；增大 group size |

## 验收标准

- 能对任一多轮任务在 30 分钟内写出上面格式的 reward spec，并说明每一项的可 hack 面。
- 能把 trajectory-level 改成 turn-level advantage 并跑通一次训练。
- 实验 3 的分叉图与行为抽样报告完成，能指出策略具体钻了哪个空子。

## 交付物

| 文件 | 内容 |
|---|---|
| `reward-spec.yaml` | 任务的三层 reward 规范与系数 |
| `reward_fn.py` + `audit.py` | 可挂进 verl 的 reward 函数；每 N 步的高分 rollout 审计脚本 |
| `credit-assignment-report.md` | 实验 1 到 3 的数据、图与结论 |

## 参考

- T37 DAPO：dynamic sampling 解决零梯度 group，agent 任务上尤其重要。
- T17 DeepSeek-R1：为什么放弃 PRM 与 MCTS 式搜索，只用规则 reward。
- T33 / T34：PRM 与生成式 verifier 的原理，理解 rubric reward 的实现方式。
- T45 SWE-RL：连续相似度 reward 的设计与局限。
- T19 Kimi K2：rubric self-critique reward 在开放任务上的用法。
- B06 verl 文档：reward manager 与 agent loop 接口。
