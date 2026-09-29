---
title: "Day 10：数据工程：合成、拒绝采样、蒸馏与去污染"
type: concept
tags: [llm-training, data-engineering, synthetic-data, rejection-sampling, distillation, decontamination]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 10：数据工程：合成、拒绝采样、蒸馏与去污染

> 上一课 [[llm-training-30d/week2/day09-lora-qlora-peft]] · 下一课 [[llm-training-30d/week2/day11-preference-optimization-dpo]]。相关：评测污染检测 [[llm-training-30d/week2/day13-evaluation-experiment-management]]；rollout 采样用推理引擎的批量推理见 [[ai-infra-30d/week1/day02-vllm-setup]]；Agent 课的轨迹合成 [[agent-rsi-30d/week1/day04-tracing-replay]]。

## 学习目标

1. 画出一条现代后训练数据流水线：种子 → 合成 → rollout 采样 → 过滤/打分 → 去重去污染 → 配比 → 版本化。
2. 实现 rejection sampling：用 verifier 或 RM 从 rollout 中筛出正确/高分回答作为 SFT 数据，并解释它为什么是 R1/Qwen3 配方的核心环节。
3. 区分离线蒸馏与 on-policy 蒸馏，说清后者为什么在推理模型上更有效。
4. 用 MinHash 去重和 n-gram/embedding 去污染处理一份 10 万条数据集，给出污染率报告。
5. 设计数据配比与难度分桶方案，并用 pass@k 定义"对当前模型有价值的数据"。

## 工业现状

数据是后训练里投入最大、公开最少的部分。从公开报告能确认的做法：

- **Llama 3（T14）**：SFT 数据大部分来自"用当前最好的模型对 prompt 采样多条回答，RM 打分选最优"的 rejection sampling，人工数据只占少数；代码、数学、多语言各有专门的合成流水线（执行反馈过滤、翻译回译）。
- **DeepSeek-R1（T17）**：推理 RL 收敛后，对 60 万条推理 prompt 采样并用规则 verifier + 模型判读筛选，加 20 万条非推理数据，做第二轮 SFT。R1-Distill 系列直接用这 80 万条数据 SFT 小模型，效果超过对小模型直接做 RL。
- **Qwen3（T15）**：thinking 与 non-thinking 数据混合 SFT；用 rejection sampling 保证推理数据正确性；"strong-to-weak distillation" 用大模型的 on-policy 输出蒸馏小模型。
- **Tulu 3（T22）**：完全公开的数据配比与去污染流程（对评测集做 8-gram 重叠检查，去掉污染样本），并证明去污染后分数会下降，说明很多公开数据集本身就被污染。
- **Kimi K2（T19）**：大规模 agentic 数据合成：生成工具 schema、模拟环境、多轮轨迹，再用 LLM judge 过滤。
- **on-policy 蒸馏（B02）**：学生自己采样，教师对学生每个 token 给 logprob 作为 dense reward。比离线蒸馏（学教师的输出）在推理任务上更省算力，因为学生只学自己分布附近的修正。

共识：合成数据必须有过滤器（verifier、执行、RM、judge），无过滤的合成数据会放大模型偏差；去污染是发布前的硬要求；数据配比的消融比模型结构的消融更值得做。

## 核心原理

### 流水线全景

```
种子 prompt（人工、爬取、模板扩展、Self-Instruct T31）
   │
   ▼
prompt 增广与难度分层（改写、组合、按 pass rate 分桶）
   │
   ▼
rollout 采样：对每个 prompt 用当前策略/教师模型采 n 条完整回答      ← rollout 在这里指"生成候选回答"
   │
   ▼
过滤：verifier（数学答案、代码测试）/ RM 打分 / LLM judge / 规则（长度、格式、语言）
   │
   ▼
去重（MinHash / 精确）→ 去污染（对所有评测集）→ 质量抽检
   │
   ▼
配比（按能力桶、来源、难度）→ 版本化（数据集 hash、每条样本的 source 与 score）
   │
   ▼
SFT / DPO / RL 各取所需
```

### Rejection sampling 的统计含义

对 prompt x 采 n 条 `y_1..y_n ~ π(·|x)`，保留 `verifier(x, y_i) = 1` 的样本，就是从 `π(y|x) · 1[correct]` 归一化后的分布采样。用这些样本做 SFT，等价于把策略往"正确回答的条件分布"推，是一步近似的 policy iteration（expert iteration）。R1 的第二轮 SFT、Llama 3 的每轮迭代都是这个循环：

```
π_0 --采样+过滤--> D_1 --SFT--> π_1 --采样+过滤--> D_2 --SFT--> π_2 ...
```

它比 RL 稳定（没有 KL、没有 advantage），但每轮只利用"对/错"一比特信息，效率低于 GRPO；工业界两者都用：RS 做数据沉淀，RL 做能力推进。

### 难度分桶与"有价值的数据"

对每个 prompt 用当前模型采 k 条，记 `p = pass@1 估计 = 正确数 / k`：

| p | 含义 | 用途 |
|---|---|---|
| 0 | 当前模型完全不会 | SFT 用教师答案（蒸馏）；RL 无梯度，先放一边 |
| (0, 1) | 会但不稳 | RL 最有价值（group 内有对有错） |
| 1 | 已掌握 | 从 RL 数据里剔除，节省 rollout；SFT 也无需 |

DAPO（T37）的 dynamic sampling 就是在训练时动态剔除 p=0 和 p=1 的 prompt。数据工程阶段提前做一遍能省 30% 以上的 rollout（示例）。

### 离线蒸馏 vs on-policy 蒸馏

```
离线蒸馏：  D = {(x, y_teacher)}，学生做 SFT
            学生学的是教师分布下的序列，与学生自己会犯的错无关（exposure bias）

on-policy： y ~ π_student(·|x)
            reward_t = log π_teacher(y_t | x, y_<t) − log π_student(y_t | x, y_<t)   （逐 token）
            用 RL 或直接加权最大似然更新学生
            学生只在自己会走的路径上被纠正，样本效率高；教师只需算 logprob（prefill），不需生成
```

B02 报告 on-policy 蒸馏在推理任务上以数倍更少的算力达到离线蒸馏 + RL 的效果。实现上教师是一个推理服务（vLLM 的 prompt_logprobs 接口），Week 3 的 rollout 系统可以直接复用。

### 去重与去污染

- **精确去重**：对 prompt+response 的规范化文本（小写、去空白、去标点）做 hash。
- **近似去重**：MinHash + LSH，shingle 用 5-gram 字或 word，Jaccard 阈值 0.8 左右（示例）；datasketch 或 text-dedup 库。
- **去污染**：对每个评测集（MMLU、GSM8K、MATH、HumanEval、AIME、SWE-bench 等）的题目取 8 到 13-gram，训练样本命中即标记；再用 embedding 相似度补漏改写题。Tulu 3 用 8-gram 重叠 ≥ 50% 判污染。改写后的评测题（数字变化）需要模板匹配。
- **反方向**：不要用评测集的训练 split 做 SFT 后又拿测试 split 报分，除非明确声明。

### 配比

按能力桶（通用对话、指令遵循、数学、代码、工具调用、多语言、安全）定比例，每桶内按来源和难度再分。原则：

1. 目标能力桶占比高，但通用桶不低于 30%（示例），防止遗忘。
2. 每个桶单独做消融：去掉它看哪些 benchmark 掉。
3. 长度分布与推理时的输入分布匹配。
4. 每次改配比升版本，评测对比同一 checkpoint 起点。

## 实现步骤

### 1. 用 vLLM 批量采样 rollout

```python
from vllm import LLM, SamplingParams
llm = LLM("Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size=1, gpu_memory_utilization=0.9)
sp = SamplingParams(n=8, temperature=1.0, top_p=1.0, max_tokens=2048, seed=0)
outs = llm.chat([[{"role":"user","content":p}] for p in prompts], sp)
rollouts = [{"prompt": p, "responses": [o.text for o in out.outputs]} for p, out in zip(prompts, outs)]
```

温度 1.0、不截 top-p 是为了得到策略的真实分布；筛 SFT 数据时可以再用较低温度补采一轮"最优样本"。

### 2. verifier 过滤（数学示例）

```python
import re
from math_verify import parse, verify      # 或自己写答案抽取 + 符号等价判定
def is_correct(resp, gold):
    m = re.search(r"\\boxed\{(.+?)\}", resp)
    return m is not None and verify(parse(gold), parse(m.group(1)))
kept = [(r["prompt"], y) for r in rollouts for y in r["responses"] if is_correct(y, gold[r["prompt"]])]
```

代码任务用沙箱执行单元测试（Day 17 会服务化），通用任务用 RM（Day 12）或 LLM judge（pairwise，带 rubric）。

### 3. 去重去污染

```python
from datasketch import MinHash, MinHashLSH
lsh = MinHashLSH(threshold=0.8, num_perm=128)
def mh(text):
    m = MinHash(num_perm=128)
    for i in range(len(text) - 4): m.update(text[i:i+5].encode())
    return m
uniq = []
for i, s in enumerate(samples):
    m = mh(s["prompt"] + s["response"])
    if not lsh.query(m):
        lsh.insert(str(i), m); uniq.append(s)

# 去污染：评测集 n-gram 集合
eval_ngrams = set(ngrams(q, 8) for q in eval_questions for ngrams in [make_ngrams])
contaminated = [s for s in uniq if len(set(make_ngrams(s["prompt"], 8)) & eval_ngrams) > 0]
```

输出一份报告：原始条数、精确去重后、近似去重后、每个评测集的污染条数与比例。

### 4. 难度分桶

对 RL 候选 prompt 集用当前 SFT 模型采 k=8，记录 pass rate，写入样本元数据 `difficulty = 1 - pass_rate`；RL 数据只取 `0 < pass_rate < 1`，SFT 蒸馏数据优先取 `pass_rate == 0` 且教师能做对的。

### 5. 版本化

每个数据集目录含 `manifest.json`：来源列表、过滤器版本、去污染的评测集列表、配比、样本数、内容 hash、生成日期、生成模型。训练配置引用 manifest hash 而不是路径。

## 实验

### 实验 1：rejection sampling 一轮迭代

GSM8K 训练集 2000 题，1.5B SFT 模型采 8 条，verifier 过滤，SFT 一轮，再评测。预期 GSM8K 提升 3 到 8 点（示例），第二轮提升递减。记录每轮采样成本（token 数）。单卡可完成。

### 实验 2：过滤器强弱对比

同一批 rollout 用三种过滤：无过滤、只看格式、verifier。各训一个模型评测。预期无过滤反而下降（学到错误推理），格式过滤持平，verifier 明显提升。

### 实验 3：去污染的代价

用一份公开 SFT 数据集训练两个模型：去污染前 / 后，评测 GSM8K 与 MATH。预期去污染后分数下降 1 到 5 点，这个差值就是原数据集的污染带来的虚高。

### 实验 4：on-policy 蒸馏 vs 离线蒸馏（4 卡以上）

教师 7B、学生 1.5B，数学任务。离线：教师生成 1 万条 SFT。on-policy：学生采样，教师算 token logprob 作 reward，用 Week 3 的 GRPO 代码或 TRL 的 GKD trainer。相同教师算力下比较学生 MATH 分数。预期 on-policy 更高。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 合成数据训完模型变差 | 无过滤或过滤器太弱，错误样本被放大 | 抽检 100 条正确率 | 加 verifier / RM 阈值 |
| verifier 通过率异常高 | 答案抽取宽松（匹配到中间步骤）或 gold 泄漏进 prompt | 人工核对 50 条 | 收紧抽取，检查 prompt |
| RL 阶段大量 group 全对 | 难度分桶没做，简单题占比高 | pass rate 直方图 | 剔除 p=1 |
| benchmark 分数高但线上差 | 评测污染 | n-gram 检查 | 去污染后重评 |
| 模型输出与教师风格一致但推理错 | 离线蒸馏的 exposure bias | 看错误分布是否集中在长链后半段 | on-policy 蒸馏 |
| 去重后数据量骤降 | 模板化 prompt 导致近似去重误杀 | 看被去掉样本的 Jaccard 分布 | 只对 response 去重或提高阈值 |
| 同一 prompt 多条回答全被保留 | 没做 prompt 级去重 | 按 prompt 分组计数 | 每 prompt 保留最多 m 条 |
| 数据配比消融结果不可比 | 起点 checkpoint 或步数不同 | 检查配置 | 固定 checkpoint 与 token 数 |

## 验收标准

- 交付一份带 manifest 的数据集，包含去重去污染报告与配比表。
- 实验 1 完成一轮 rejection sampling 并有提升；实验 2 说明过滤器的必要性。
- 能解释 rejection sampling 与 RL 的关系、on-policy 蒸馏为什么更高效、去污染为什么会让分数下降。
- 给出 RL 候选 prompt 的难度分桶统计。

## 交付物

| 文件 | 内容 |
|---|---|
| `data-pipeline.md` | 流水线图与每步工具 |
| `dedup-decontam-report.md` | 去重去污染统计 |
| `rs-round1.md` | 实验 1、2 结果 |
| `difficulty-buckets.json` | 每条 RL 候选 prompt 的 pass rate |
| `datasets/v1/manifest.json` | 版本化清单 |

## 参考

- T31 Self-Instruct：合成指令数据的起点。
- T14 Llama 3 第 4.2 到 4.3 节：rejection sampling 与各领域合成流水线。
- T17 DeepSeek-R1 第 2.3.3 节：RL 后的 rejection sampling SFT 与蒸馏。
- T15 Qwen3 第 4 章：strong-to-weak distillation。
- T22 Tulu 3 第 3 章：数据配比与去污染流程。
- T19 Kimi K2 第 3 章：agentic 数据合成。
- T37 DAPO：dynamic sampling 与难度过滤的训练时版本。
- B02 On-Policy Distillation：逐 token 教师 logprob 作 reward 的做法。

## 附录 A：LLM judge 过滤的最小可用协议

通用对话没有 verifier，只能用 RM 或 judge。judge 用错比不用更糟，最小协议：

1. **pairwise 而不是绝对打分**：让 judge 比较同一 prompt 的两条回答，位置随机交换，两次结果不一致的对子丢弃。绝对打分的一致性通常低于 pairwise（Agent 课 Day 6 有校准方法）。
2. **带 rubric**：给 judge 明确的维度（正确性、遵循指令、简洁、安全），每个维度单独判，最后合成。
3. **校准集**：抽 200 对人工标注，算 judge 与人工的一致率；低于 80%（示例）不要用它过滤。
4. **judge 与被过滤模型不同源**：用被过滤模型自己当 judge 会偏好自己的风格（self-preference bias）。
5. **长度去偏**：judge 偏好长回答，对长度差异大的对子加惩罚或只比较长度相近的对子。

```python
def judge_pair(prompt, a, b, rubric):
    # 两个顺序各问一次
    r1 = ask_judge(prompt, a, b, rubric)   # 返回 "A" / "B" / "tie"
    r2 = ask_judge(prompt, b, a, rubric)
    r2 = {"A": "B", "B": "A", "tie": "tie"}[r2]
    return r1 if r1 == r2 else None        # 不一致丢弃
```

## 附录 B：agentic 轨迹的合成（Kimi K2 式，缩小版）

```
1. 工具 schema 生成：从真实 API 文档或 LLM 生成 100 个工具，每个带参数 schema 与模拟实现
2. 任务生成：给定工具子集，让 LLM 生成需要 2 到 5 次调用才能完成的任务及其验收条件
3. 轨迹采样：agent 模型在模拟环境里跑（rollout = 多轮模型调用 + 工具返回）
4. 过滤：验收条件程序化判定 + judge 判轨迹质量（无冗余调用、参数正确）
5. 入库：messages 格式（Day 8 附录 B），tool 返回 mask，记录工具集版本
```

这套流水线与 Week 4 的 agent 环境是同一套基础设施，Day 23 会把模拟环境升级成可训练的 gym。

## 附录 C：数据集 manifest 示例

```json
{
  "name": "sft-mix", "version": "v3", "created": "2026-09-29",
  "content_hash": "sha256:...",
  "num_samples": 412000,
  "mixture": {"general_chat": 0.35, "math": 0.20, "code": 0.20, "tool_use": 0.15, "safety": 0.05, "multilingual": 0.05},
  "sources": [
    {"name": "tulu3-subset", "n": 150000, "filter": "judge-pairwise-v2"},
    {"name": "math-rs-round2", "n": 80000, "filter": "math_verify", "generator": "qwen2.5-7b-sft-v2", "n_per_prompt": 8},
    {"name": "code-rs-round1", "n": 80000, "filter": "unit-tests-sandbox-v1"}
  ],
  "dedup": {"exact": 18211, "minhash_0.8": 9312},
  "decontamination": {"eval_sets": ["gsm8k", "math", "humaneval", "mbpp", "aime24", "mmlu", "ifeval"], "removed": 1204, "method": "8gram+embed0.9"},
  "template_version": "chatml-v2", "tokenizer_sha": "..."
}
```

训练日志里只需要记 `sft-mix@v3` 和 hash，任何人都能回溯这份数据怎么来的。这是 Day 13 实验管理的一部分。

## 附录 D：自检问题

1. rejection sampling 保留的样本服从什么分布？为什么它是一步近似的 policy iteration？
2. 一个 prompt 当前模型 pass@8 = 0，应该进 SFT 还是 RL 数据？为什么？
3. on-policy 蒸馏里教师需要做生成吗？它的算力主要花在哪？
4. 为什么去污染之后 benchmark 分数会下降，这个下降说明了什么？
5. judge 过滤时为什么要交换位置问两次？不一致的对子丢弃会引入什么偏差？
6. 数据配比里通用桶为什么不能太低？怎么用消融证明？
