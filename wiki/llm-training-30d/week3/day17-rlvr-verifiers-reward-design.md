---
title: "Day 17：RLVR、Verifier 与 Reward 设计"
type: concept
tags: [llm-training, rlvr, verifier, reward-design, reward-hacking, curriculum]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 17：RLVR、Verifier 与 Reward 设计

> 上一课 [[llm-training-30d/week3/day16-ppo-grpo-family]] · 下一课 [[llm-training-30d/week3/day18-rl-training-frameworks]]。相关：Agent 课的 reward 设计 [[agent-rsi-30d/week3/day16-reward-design]]（那边侧重 agent 任务，这里侧重单轮可验证任务与 verifier 服务）；本课 Day 12 的 RM [[llm-training-30d/week2/day12-reward-models]]；Day 25 agent reward [[llm-training-30d/week4/day25-agent-reward-credit-assignment]]。

## 学习目标

1. 说清 RLVR 与 RM-based RL 的边界：什么任务可以写 verifier，什么任务写不出。
2. 实现数学与代码两类 verifier，覆盖答案抽取、等价判定、沙箱执行、超时与并发。
3. 设计一个混合 reward（正确性 + 格式 + 长度），并能解释每一项的权重为什么这样定。
4. 列出至少六种 reward hacking 的具体形态，并给出检测方法。
5. 用 pass rate 对题库做难度分层，构造 curriculum，并解释它与 dynamic sampling 的关系。

## 工业现状

RLVR（Reinforcement Learning with Verifiable Rewards）是 Tulu 3（T22）命名、DeepSeek-R1（T17）证明有效的范式：reward 不来自学习到的 RM，而来自规则判定。R1-Zero 只用两项 reward：答案正确性（数学用规则比对，代码用编译器和测试）和格式（思考过程必须在指定标签内）。Qwen3（T15）、Kimi k1.5（T18）、DAPO（T37）沿用这一思路并加上长度处理。SWE-RL（T45）把可验证性推广到软件工程，reward 来自 patch 与真实提交的相似度或测试。

RLVR 能成立的原因是 verifier 不会被优化坏：RM 是神经网络，策略走远后总能找到 RM 判错的区域；规则 verifier 的盲区是有限且可枚举的。但"可验证"是一个梯度：数学最终答案最容易，代码次之（测试用例覆盖率决定），证明、开放式写作、多轮对话则需要 rubric 或 RM。工业配方通常把 RLVR 用于推理与代码，把 RM/DPO 用于通用对话，最后混合训练。

verifier 的工程量常被低估。一个数学 verifier 要处理 `\boxed{}` 抽取、LaTeX 归一化、分数与小数、区间与集合、多答案；一个代码 verifier 要管沙箱、依赖、超时、内存、非确定性测试；两者都要能每秒处理上千次判定，因为一个 batch 有几千条 rollout。

## 核心原理

### reward 的可验证性梯度

| 任务 | verifier | 盲区 | 工业用法 |
|---|---|---|---|
| 数学（最终答案） | 抽取 + 符号等价 | 抽取失败；等价判定错；多解 | 主力 RLVR 任务 |
| 代码（函数级） | 沙箱跑测试用例 | 测试覆盖不全；特判测试 | 主力 RLVR 任务 |
| 代码（仓库级） | 隐藏测试、CI | 环境非确定性；修改测试 | Agent RL，Day 26 |
| 指令约束（IFEval 类） | 程序化检查 | 只覆盖可检查约束 | 混合 reward 的一项 |
| 格式 | 正则 | 无 | 辅助 reward |
| 证明 | Lean 等形式化系统 | 非形式化证明无法验证 | 形式化数学专项 |
| 开放式问答、写作 | 无；用 rubric + judge 或 RM | judge 偏置 | RM/DPO 路线 |

### verifier 设计原则

```text
1. 判定必须是确定性的：同一输出两次判定结果相同。非确定性测试要重跑取多数或剔除。
2. 抽取失败与答案错误要区分：抽取失败可能是格式问题，给格式 reward 而不是正确性 reward。
3. 宽松等价而非字符串比对：1/2 == 0.5 == \frac{1}{2}；用 sympy 或 math-verify 类库做符号等价。
4. 判定时间有上限：单次超过 T_max（例如 10 秒）视为失败并记录，防止一条 rollout 卡住整个 batch。
5. 判定器要有自己的测试集：用人工标注的 500 条（输出, 标准答案, 应判结果）回归测试 verifier。
6. 记录判定原因：reward 为 0 时区分"错误答案 / 抽取失败 / 超时 / 截断"，Day 20 的诊断需要这个分解。
```

### 混合 reward

R1 风格的组合：

```text
r = r_correct + λ_fmt · r_format + λ_len · r_length
r_correct ∈ {0, 1}                        verifier
r_format  ∈ {0, 1}                        是否符合 <think>...</think><answer>...</answer> 或 \boxed{} 约定
r_length  = − max(0, |y| − L_soft) / (L_max − L_soft)  截断到 [−1, 0]，DAPO 的 overlong shaping
```

权重原则：正确性主导（λ 都小于 1）；格式 reward 只在训练早期有用，模型学会格式后它是常数，对 advantage 无贡献；长度惩罚只在长度失控时开。任何一项 reward 如果与正确性负相关（比如长度惩罚让模型放弃需要长推理的题），就要降权。

不要把多个 reward 简单相加后再做 group 归一化而不看分解：advantage 归一化后你看不出是哪一项在驱动。日志里每项 reward 单独记。

### reward hacking 的形态

| 形态 | 表现 | 检测 |
|---|---|---|
| 格式游戏 | 输出多个 `\boxed{}`，verifier 取到了对的那个 | 统计每条 rollout 的 boxed 数量 |
| 答案枚举 | 列出所有候选答案 | 抽检输出；限制答案唯一 |
| 特判测试 | 代码里 `if input == test_case_1: return expected` | 隐藏测试；测试用例不进 prompt |
| 空转长度 | 输出大量无意义 token 拖到截断，利用截断样本被 mask 的规则 | 截断率、长度分布 |
| 语言混杂 | 中英混杂推理拿到正确答案 | 语言一致性检查（R1 加了语言一致性 reward） |
| 复读 prompt 中的提示 | few-shot 例子里泄漏答案模式 | 去掉 few-shot；检查 prompt 泄漏 |
| 利用 verifier bug | 等价判定把 `0` 与空判等 | verifier 回归测试；抽检 reward=1 的样本 |
| 跳过推理 | 直接猜答案，在简单题库上正确率高 | 按难度分层看长度与正确率 |

检测的通用手段：定期抽 100 条 reward=1 的 rollout 人工看；用一个独立的、不参与训练的 hidden verifier（更严格的版本）重新判定，看两者的一致率是否下降。

### 难度分层与 curriculum

对题库先用当前策略采 n 条 rollout 算 pass rate p，分层：

```text
p = 0（全错）：无梯度，且可能超出能力；暂时移出或降低占比
0 < p < 1：有梯度，最有价值；p 在 0.2 到 0.8 之间信息量最大
p = 1（全对）：无梯度；移出，定期重测防止遗忘
```

curriculum 有两种实现：离线分层（训练前算好 p，按阶段调整各层占比）和在线过滤（DAPO 的 dynamic sampling，每步只保留有梯度的 group）。在线过滤更简单也更准确，因为 p 随训练变化；代价是采样成本上升。Qwen3 报告了按难度分阶段的推理 RL，Kimi k1.5 用 curriculum 与 prioritized sampling。

题库构造要点：来源去污染（不能含评测集）；每题必须有可验证的标准答案（人工核对一遍抽取器能否抽出）；答案唯一；难度覆盖从 p≈0.9 到 p≈0.1。

### 知道即可

- PRM 在 RL 中做 dense reward 的尝试（T33）在 R1 中被放弃，原因是步骤定义模糊、标注贵、容易被 hack；当前主流是 outcome reward。
- 生成式 verifier / judge（T34）用在无法写规则的任务，但要当作 RM 对待（有过优化风险）。

## 实现步骤

### 1. 数学 verifier

```python
import re
from math_verify import parse, verify   # 以当前库版本为准；也可用 sympy 自己写

def extract_boxed(text: str):
    # 取最后一个 \boxed{...}，处理嵌套花括号
    idx = text.rfind("\\boxed")
    if idx < 0: return None
    i = text.find("{", idx); depth = 0; start = i
    for j in range(i, len(text)):
        if text[j] == "{": depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0: return text[start+1:j]
    return None

def math_reward(response: str, gold: str) -> dict:
    ans = extract_boxed(response)
    if ans is None:
        return {"correct": 0.0, "format": 0.0, "reason": "no_boxed"}
    try:
        ok = verify(parse(f"${gold}$"), parse(f"${ans}$"))
    except Exception:
        return {"correct": 0.0, "format": 1.0, "reason": "parse_error"}
    return {"correct": float(ok), "format": 1.0, "reason": "ok" if ok else "wrong"}
```

回归测试：准备 200 条（response, gold, expected）用例，覆盖分数、小数、根式、区间、多 boxed、无 boxed。

### 2. 代码 verifier 与沙箱

```python
import subprocess, tempfile, os, json

def run_tests(code: str, tests: list[dict], timeout=5) -> dict:
    # tests: [{"input": "...", "output": "..."}]；每个用例独立进程，资源受限
    passed = 0; reasons = []
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sol.py")
        open(path, "w").write(code)
        for t in tests:
            try:
                p = subprocess.run(["python3", path], input=t["input"], capture_output=True,
                                   text=True, timeout=timeout, cwd=d,
                                   env={"PATH": "/usr/bin"})          # 生产用 nsjail / firejail / docker
                ok = p.returncode == 0 and p.stdout.strip() == t["output"].strip()
                passed += ok; reasons.append("ok" if ok else "wrong_or_error")
            except subprocess.TimeoutExpired:
                reasons.append("timeout")
    return {"correct": float(passed == len(tests)), "pass_frac": passed / len(tests), "reasons": reasons}
```

reward 用 `correct`（全过才算对）而不是 `pass_frac`，否则模型会学到只过简单用例。生产沙箱需要：CPU/内存限制、无网络、文件系统隔离、并发池（每个 batch 几千次执行）。

### 3. verifier 服务化

```text
训练进程 ──HTTP/gRPC──▶ verifier 服务（多副本，每副本一个进程池）
                              ├─ 数学：CPU 密集但快，单核每秒几百次
                              └─ 代码：进程/容器池，预热，每次执行 100ms 到数秒
要求：p99 延迟有上限；失败重试一次；每次判定返回 reason；日志可回放
```

verl 支持自定义 reward function 与远程 reward 服务（以文档为准）；大规模训练通常把 verifier 拆成独立集群。

### 4. 混合 reward 与日志

```python
def total_reward(resp, gold, max_len=8192, soft_len=7168):
    m = math_reward(resp, gold)
    L = len(tokenize(resp))
    r_len = -max(0, L - soft_len) / (max_len - soft_len)
    r = m["correct"] + 0.1 * m["format"] + 1.0 * r_len
    return r, {"correct": m["correct"], "format": m["format"], "len_pen": r_len, "reason": m["reason"], "len": L}
```

每步日志记录 `correct` 均值、`format` 均值、`len_pen` 均值、`reason` 分布、截断率。

### 5. 难度分层脚本

用 vLLM 对题库每题采 8 条，算 p，按 [0, 0.2, 0.8, 1] 分四桶，输出每桶题数与样例。训练配置里按桶设采样权重，或直接开 dynamic sampling。

## 实验

资源分层：实验 1、2 任何机器；实验 3、4 单卡。

### 实验 1：verifier 回归测试

写 200 条数学用例跑 verifier，记录误判率；再用两个不同的等价判定实现（sympy 直接比较 vs math-verify）对同一批用例比较一致率。预期：字符串比对误判率超过 20%，符号等价低于 3%。

### 实验 2：代码 verifier 的测试覆盖

取 50 道有公开测试的题，构造一个"特判测试用例"的作弊解法，验证 `pass_frac` reward 能被骗过而 hidden 测试能抓住。统计需要多少隐藏用例才能把作弊解法的通过率压到 0。

### 实验 3：格式 reward 的作用期

Qwen2.5-1.5B 在 GSM8K 上 GRPO 100 步，两组：有格式 reward / 无格式 reward。记录 `format` 均值曲线与 `correct` 曲线。预期：有格式 reward 时前 20 步格式率快速到 1，之后该项对 advantage 无贡献；无格式 reward 时抽取失败率更高，`correct` 曲线起步慢。

### 实验 4：难度分层与 curriculum

对 GSM8K 训练集用初始模型算 p 分桶。两组训练：全量随机 vs 只用 0<p<1 的题。各 150 步，比较 avg@4 曲线与每步有效 group 比例。预期：过滤后曲线更平滑、早期更快；后期需要把 p=0 的题逐步加回来。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| reward 快速升到 0.9 以上但评测持平 | reward hacking 或题库太简单 | 抽检 reward=1 样本；hidden verifier 一致率 | 修 verifier；换难题 |
| 抽取失败率高 | 格式不稳定或抽取器太严 | reason 分布 | 加格式 reward；放宽抽取 |
| 代码任务 reward 全 0 | 沙箱环境与题目依赖不符；超时太短 | 用标准答案跑一遍 verifier | 修环境；调超时 |
| 训练每步时间抖动大 | 某些 rollout 的代码执行卡住 | verifier p99 延迟 | 硬超时；并发池 |
| 长度惩罚后正确率下降 | soft_len 太短，砍掉了需要长推理的题 | 按长度分桶看正确率 | 提高 soft_len；只对错误回答惩罚 |
| 语言混杂 | 多语言基座在 RL 下漂移 | 语言检测统计 | 语言一致性 reward（R1 做法） |
| 全对 group 比例超过 50% | 题库对当前模型太易 | 每步统计 | 加难题；dynamic sampling |
| verifier 判定不一致 | 非确定性测试或依赖时间/随机 | 同一输出判两次 | 固定 seed；剔除非确定性用例 |

## 验收标准

- 数学与代码 verifier 各通过自己的回归测试集，误判率有数据。
- 混合 reward 每一项有独立日志，能回答"这一步 reward 上升是哪一项贡献的"。
- 实验 2 展示了至少一种 reward hacking 被 hidden verifier 抓住。
- 题库有难度分层记录与 curriculum 方案。

## 交付物

| 文件 | 内容 |
|---|---|
| `verifiers/math_verifier.py`、`verifiers/code_verifier.py` | 含回归测试 |
| `reward_fn.py` | 混合 reward 与日志分解 |
| `difficulty_buckets.json` | 题库分层结果 |
| `reward-design-notes.md` | 可验证性梯度表、hacking 形态表、实验 1 到 4 结论 |

## 参考

- T17 DeepSeek-R1：规则 reward 的设计与语言一致性 reward。
- T22 Tulu 3：RLVR 命名与在 GSM8K/MATH/IFEval 上的用法。
- T37 DAPO：overlong reward shaping 与 dynamic sampling。
- T45 SWE-RL：把可验证 reward 推广到软件工程。
- T33 PRM、T34 生成式 verifier：dense reward 与 judge 路线的边界。
- T18 Kimi k1.5：curriculum 与 prioritized sampling。
