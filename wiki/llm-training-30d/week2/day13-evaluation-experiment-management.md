---
title: "Day 13：评测体系与实验管理"
type: concept
tags: [llm-training, evaluation, contamination, experiment-management]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 13：评测体系与实验管理

> 上一课 [[llm-training-30d/week2/day12-reward-models]] · 下一课 [[llm-training-30d/week2/day14-p1-project-sft-dpo]]。相关：Agent 课的任务集与 judge 统计 [[agent-rsi-30d/week1/day05-evaluation-verifiers]]、[[agent-rsi-30d/week1/day06-judges-statistics]]；推理课的 benchmark 方法学 [[ai-infra-30d/week1/day03-benchmark]]。

## 学习目标

1. 能按"测什么能力、答案怎么判、采样方差多大"三个维度给出主流 benchmark 的一张表，并说清每个 benchmark 的失效方式。
2. 能对一个 SFT 或 RL 实验设计单变量消融，算出需要多少个样本或多少次采样才能区分两个配置。
3. 能实现 n-gram 与 embedding 两种污染检测，并解释为什么 benchmark 分数上涨不等于能力上涨。
4. 能搭起一套实验管理流程：配置快照、seed、指标日志、评测结果与 checkpoint 一一对应。
5. 能在预算约束下裁剪评测集，并说清裁剪后置信区间变宽多少。

## 工业现状

后训练团队的评测分三层。第一层是公开 benchmark，用来和外部比较：Qwen3（T15）、DeepSeek-R1（T17）、Llama 3（T14）、Tulu 3（T22）的报告都以 MMLU-Pro、GPQA-Diamond、MATH-500、AIME、LiveCodeBench、SWE-bench Verified、IFEval、Arena-Hard、tau-bench 这类集合报数。第二层是内部 hidden set，用来做真正的模型选择，因为公开集一旦进了训练数据就失效。Tulu 3 明确把开发集和 unseen 集分开，并报告两者差异。第三层是在线 A/B 与人工评估，用来验证离线指标是否与用户体验一致。

推理模型让采样方差成了核心问题。AIME 只有 30 题，单次采样的分数在 ±10 分内抖动，所以 R1、Qwen3 都报 avg@n（例如 avg@64），并且固定温度、top-p 和最大长度。这意味着一次评测的成本可能是几十次完整推理，评测预算成了实验设计的一部分。

实验管理方面，主流做法是训练配置、数据版本、代码 commit、seed 和评测结果全部绑定到一个 run id。Tulu 3 公开了全部配置和评测脚本，其可复现性正是靠这套绑定。

Agent 类评测（SWE-bench Verified、tau-bench）还有第三个维度：scaffold。同一个模型换一个 agent 框架，resolve rate 可以差 10 分以上，所以报告必须固定 scaffold、工具集和最大步数。tau-bench 提出的 pass^k（k 次全部成功的概率）比 pass@k 更贴近生产要求，因为用户需要的是每次都对，而不是多试几次总有一次对。

## 核心原理

### benchmark 一张表

| benchmark | 测什么 | 判分方式 | 主要失效方式 | 建议报法 |
|---|---|---|---|---|
| MMLU-Pro | 知识 + 多步推理，选择题 | 精确匹配选项 | 污染；提示格式敏感 | 0-shot CoT，单次 |
| GPQA-Diamond | 研究生级科学题 | 选项匹配 | 题少（198），方差大 | avg@8 以上 |
| MATH-500 / AIME | 数学推理 | 答案抽取 + 等价判定 | 抽取器漏判；AIME 只 30 题 | avg@n，报 n |
| LiveCodeBench | 代码生成 | 沙箱测试用例 | 版本随时间滚动，需注明日期区间 | pass@1，注明版本 |
| SWE-bench Verified | 仓库级修 bug | 隐藏测试 | 环境非确定性；agent 框架差异大 | 固定 scaffold，报 resolve rate |
| IFEval | 指令遵循 | 程序化约束检查 | 只测可验证约束 | strict / loose 两种 |
| Arena-Hard | 开放式对话质量 | LLM judge 对比 baseline | judge 偏好长度和风格 | 报 judge 与 baseline 版本 |
| tau-bench | 工具调用多轮任务 | 状态检查 | 用户模拟器不稳定 | pass^k |
| RewardBench 类 | RM 判别能力 | 偏好对准确率 | 与下游 RL 效果相关性弱 | 仅作 RM 初筛 |

必须理解的是"判分方式"这一列：判分器本身是代码，有 bug 和盲区。数学答案抽取器把 `\boxed{1/2}` 和 `0.5` 判成不同答案时，模型分数的变化反映的是格式而不是能力。

### 采样方差与样本量

单题通过率 p 的 n 次采样均值方差为 p(1-p)/n。一个 30 题的集合，每题 avg@n，总体标准误约为：

```text
SE ≈ sqrt( mean_i[ p_i (1 - p_i) / n ] / N_questions )
示例：p_i 平均 0.5，n = 16，N = 30  →  SE ≈ sqrt(0.0156 / 30) ≈ 0.023，即 ±2.3 分（1σ）
```

两个配置差 3 分时，在这个方差下不显著。要区分 3 分差异且 2σ 置信，需要 SE 降到 1.5 以下，即 n 提到 40 以上或题数翻倍。这就是为什么小集合上"提升 2 分"几乎没有信息量。

pass@k 的无偏估计（Chen et al. 的公式，各评测框架通用）：

```python
def pass_at_k(n, c, k):
    # n 次采样中 c 次正确，估计 k 次采样至少一次正确的概率
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
```

pass@k 随 k 增长反映的是"模型分布里有没有正确答案"，avg@n 反映的是"随便采一次多大概率对"。RL 通常提升 avg@1 而不明显提升 pass@k 的上界，这是 Dr. GRPO（T38）等工作讨论的现象，评测时要分开报。

### 污染检测

两种主流方法，都要在训练数据入库前跑：

```text
n-gram 重叠：对 benchmark 每道题取 13-gram（Llama 3 用的量级），在训练语料里查精确命中；命中比例超过阈值（如 50%）的题标为污染。
embedding 相似：对题干做 embedding，与训练样本做近邻检索，余弦相似度高于阈值再人工抽查；能抓到改写后的泄漏。
```

污染的后果不是分数虚高这么简单：污染题上的分数会掩盖真实的回归，让消融结论反向。Llama 3（T14）报告了每个 benchmark 的污染比例和去污染后的分数，这是当前的最佳实践。

### 单变量消融与统计检验

消融的原则：一次只改一个变量，其余全部固定（数据、seed、步数、评测配置）。结果比较用配对检验，因为两个模型答的是同一批题：

```python
# 按题配对的 bootstrap，比 t 检验更适合非正态的 0/1 数据
import numpy as np
def paired_bootstrap(a, b, iters=10000, seed=0):
    # a, b: 每题的 avg@n 得分数组，长度相同
    rng = np.random.default_rng(seed)
    diffs = []
    idx = np.arange(len(a))
    for _ in range(iters):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(a[s].mean() - b[s].mean())
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5])
```

如果 95% 区间跨过 0，就不能宣称有差异。这条规则在 Week 3 的 RL 消融里同样适用。

## 实现步骤

### 1. 搭评测 harness

用 lm-evaluation-harness 或 lighteval 做公开 benchmark，用 vLLM 后端加速采样；数学和代码用专门的判分器（以当前版本文档为准）：

```bash
pip install lm-eval vllm
lm_eval --model vllm \
  --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct,tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.8 \
  --tasks gsm8k_cot,ifeval \
  --batch_size auto --num_fewshot 0 \
  --gen_kwargs temperature=0,max_gen_toks=2048 \
  --output_path results/run-001 --log_samples
```

`--log_samples` 必开：没有逐题输出就无法做配对检验和错误分析。

### 2. 自己写 avg@n 评测

对 AIME 这类小集合，harness 的单次采样不够用。写一个基于 vLLM 的脚本：

```python
from vllm import LLM, SamplingParams
llm = LLM("path/to/ckpt", dtype="bfloat16")
sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=8192, n=16, seed=0)
outs = llm.generate(prompts, sp)             # 每题 16 条 rollout
# rollout 在评测里的含义：对同一题的多次独立采样，用来估计通过率而不是取最好的一条
scores = [[verify(o.text, gold) for o in out.outputs] for out, gold in zip(outs, golds)]
avg_at_n = np.mean([np.mean(s) for s in scores])
```

温度、top-p、max_tokens、seed 写进结果文件名，否则两次评测不可比。

### 3. 污染检测脚本

```python
from datasketch import MinHash, MinHashLSH
def ngrams(text, n=13):
    toks = text.split()
    return {" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)}
# 对训练集建 LSH，对每道 benchmark 题查询
lsh = MinHashLSH(threshold=0.5, num_perm=128)
for i, doc in enumerate(train_docs):
    m = MinHash(num_perm=128)
    for g in ngrams(doc): m.update(g.encode())
    lsh.insert(str(i), m)
contaminated = [q for q in bench if lsh.query(minhash_of(q))]
```

数据入库前跑一次，结果作为数据版本的元信息保存。

### 4. 实验管理

每个 run 固定产出：

```text
runs/<run_id>/
  config.yaml          # 训练全部参数，含数据版本 hash 与代码 commit
  seeds.json           # 数据 shuffle、初始化、采样 seed
  metrics.jsonl        # 每步 loss、lr、grad_norm、吞吐；RL 还有 reward、KL、entropy、response_len
  eval/<ckpt_step>/    # 每个评测 checkpoint 的逐题输出与汇总
  ckpt/                # 或指向对象存储的路径
```

用 W&B 或 MLflow 记录 metrics 与 config，但逐题评测输出留在文件里，因为后续错误分析需要原文。

## 实验

资源分层：API-only 可用现成模型输出做统计部分；单卡完成全部。

### 实验 1：采样方差

用 Qwen2.5-1.5B-Instruct 在 MATH-500 的前 100 题上跑 n=1、4、16、64 的 avg@n，各重复 3 个 seed。记录每种 n 下 3 个 seed 的分数极差。预期：n=1 时极差可达 5 分以上，n=64 时降到 1 分内。把极差与上面的 SE 公式对比。

### 实验 2：判分器敏感性

同一批模型输出，用两个答案抽取器（严格 `\boxed{}` 抽取 vs 宽松最后一个数字）判分。预期：分数差 2 到 8 分，差异集中在格式不规范的输出。结论写进 P1 项目的评测协议。

### 实验 3：污染注入

把 GSM8K 测试集的 5% 题目改写后混进 SFT 数据训练一个小模型，对比污染题与非污染题的分数变化。预期：污染题分数明显上涨，非污染题不变或略降。用 n-gram 与 embedding 两种检测各自能抓回多少。

### 实验 4：消融样本量

取两个只差学习率的 SFT checkpoint，在 GSM8K 上用配对 bootstrap 计算区间。逐步减少题数（1319 → 500 → 200 → 50），记录区间何时跨 0。

### 5. 训练过程中的评测曲线

评测不是训练结束才做一次。SFT 每个 epoch、RL 每 N 步在一个小而稳定的 dev 集上跑一次，画成曲线：

```text
step    gsm8k_avg@4   ifeval_strict   response_len   kl_to_ref
  0        0.61          0.72            310            0.00
 50        0.66          0.71            352            0.02
100        0.70          0.68            431            0.05   ← ifeval 开始跌，长度在涨
150        0.71          0.63            560            0.09   ← 典型的单目标 RL 副作用
```

（示例数字）曲线能暴露单点评测看不到的东西：reward 单调上升但 IFEval 下降，说明策略在牺牲指令遵循换数学分。Week 3 的稳定性诊断（[[llm-training-30d/week3/day20-async-rl-stability]]）就建立在这类曲线上。dev 集要小到每次评测几分钟内完成，又要覆盖所有关心的能力维度，通常每个维度 50 到 100 题、n=4。

### 6. 评测预算分配

给定总预算 B（以采样 token 数计），按每个集合的方差分配 n：

```text
目标：最小化 Σ_j w_j · SE_j²，约束 Σ_j N_j · n_j · L_j ≤ B
w_j：该集合在决策中的权重；N_j 题数；L_j 平均输出长度
近似解：n_j ∝ sqrt(w_j · p_j(1-p_j) / (N_j · L_j))
```

结论是长输出、题多的集合（如 LiveCodeBench）用小 n，短输出、题少的集合（如 AIME）用大 n。把这个分配写进评测协议，而不是所有集合统一 n。

### 知道即可：在线评测与人工评估

离线指标与线上体验的相关性需要定期校准。做法是每次发布前抽一批真实流量 prompt，用新旧模型各生成一次，做双盲人工对比或 judge 对比，把胜率和离线指标的差值记录下来。当离线涨、线上平时，说明评测集已经偏离真实分布，需要更新 hidden set。这部分在 Day 29 的数据飞轮里再展开。

### 实验 5：pass@k 与 avg@n 的分离

对 SFT 前后的两个 checkpoint，在 MATH-500 前 100 题上跑 n=64，同时算 avg@1（=avg@64 的均值）和 pass@8、pass@64。预期：SFT 或 RL 后 avg@1 上升明显，pass@64 上升很少甚至持平。写一段解释：这说明训练主要是把已有的正确答案采样概率提高，还是引入了新的解法。这个区分在 Week 3 判断 RL 是否真的扩展了能力时会反复用到。

### 实验 6：dev 集设计

从 GSM8K、IFEval、一个代码集各抽 60 题组成 dev 集，n=4，测一次完整评测耗时与 SE。调整题数直到单次评测在 5 分钟内且每个维度 SE 小于 3 分。把最终 dev 集与 SE 写进评测协议。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 同一 checkpoint 两次评测分数差 3 分以上 | 采样温度不为 0 且 n 太小；vLLM 版本或 batch 变化影响数值 | 固定 seed 重跑；比较逐题输出 | 提高 n；温度 0 做确定性基线；记录引擎版本 |
| benchmark 涨、hidden set 跌 | 污染或过拟合公开集 | 跑污染检测；看涨幅是否集中在少数题 | 去污染；以 hidden set 做模型选择 |
| 数学分数低但答案看起来对 | 抽取器或等价判定失效 | 抽 50 条人工核对 | 换 sympy 等价判定；统一输出格式的 SFT |
| 代码分数在两次评测间不同 | 沙箱超时、依赖版本、非确定性测试 | 固定镜像与超时；重跑失败用例 | 固化环境；报 3 次取中位 |
| judge 评测偏爱某一模型 | 长度偏置、位置偏置 | 交换对比顺序重跑；控制长度 | 双向对比取平均；报 judge 版本 |
| 消融结论在换 seed 后反转 | 差异小于噪声 | 配对 bootstrap 区间 | 增加 n 或题数；否则报"无显著差异" |
| 评测太贵跑不完 | 全量 avg@64 | 算每个集合的 SE | 大集合降 n，小集合保 n；按 SE 分配预算 |

## 验收标准

- 能对课程涉及的 10 个 benchmark 说出判分方式和一种失效方式。
- 交付的评测脚本能对同一 checkpoint 复现分数（温度 0 下完全一致，采样下在 SE 内）。
- 能给出一个消融的样本量估算和配对检验结果。
- 训练数据入库流程包含污染检测记录。

## 交付物

| 文件 | 内容 |
|---|---|
| `eval-protocol.md` | 课程后续所有项目统一使用的评测协议：集合、n、温度、判分器版本、报告格式 |
| `eval_avg_n.py` | 基于 vLLM 的 avg@n 与 pass@k 脚本 |
| `contamination_check.py` | n-gram + embedding 污染检测 |
| `variance-report.md` | 实验 1、2、4 的数据与结论 |

## 参考

- T22 Tulu 3：开源后训练里最完整的评测框架与 unseen 集实践。
- T14 Llama 3：污染检测方法与去污染后分数的报法。
- T17 DeepSeek-R1、T15 Qwen3：avg@n 报告方式与采样参数。
- T38 Dr. GRPO：avg@1 与 pass@k 分离的讨论。
- B05 TRL 文档：训练与评测回调的集成方式。
