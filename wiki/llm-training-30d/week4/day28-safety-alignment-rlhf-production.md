---
title: "Day 28：安全对齐与生产 RLHF：数据标注流程中的 rollout"
type: concept
tags: [llm-training, safety, rlhf, annotation, constitutional-ai]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 28：安全对齐与生产 RLHF：数据标注流程中的 rollout

> 上一课 [[llm-training-30d/week4/day27-reasoning-models-long-cot]] · 下一课 [[llm-training-30d/week4/day29-training-ops-at-scale]]。相关：Agent 课的 judge 与统计 [[agent-rsi-30d/week1/day06-judges-statistics]]（标注一致性与置信区间的方法通用）。

## 学习目标

1. 能说明安全数据在 SFT、偏好优化、RL 三个阶段各自的位置与作用，以及"有用性 vs 无害性"冲突在 reward 上如何表达。
2. 能设计一条生产级 RLHF 标注流水线：从 prompt 采样、rollout 生成、标注任务分发、一致性检查到数据入库，并解释标注员看到的 rollout 是什么。
3. 能实现 Constitutional AI / RLAIF 式的模型自标注流程，并知道它相对人工标注的可靠性边界。
4. 能用拒答率、越狱基准、over-refusal 基准三类指标评测安全对齐，并解释它们之间的权衡。
5. 交付一份标注协议与一个标注质量监控脚本。

## 工业现状

公开报告里安全对齐的共同做法：

| 环节 | 做法 | 出处 |
|---|---|---|
| SFT 阶段 | 混入安全示范数据（拒绝有害请求、安全地回答边界问题），比例小但覆盖分类体系 | T14、T23 |
| 偏好阶段 | 安全偏好对：同一 prompt 下"安全且有用"优于"过度拒绝"优于"有害" | T14、T22 |
| RL 阶段 | 安全 RM 与有用性 RM 分开训练，reward 组合（加权或"安全优先"门控） | T23、T14 的多 RM 描述 |
| AI 反馈 | Constitutional AI：用一组原则让模型自我批评与修改，再用 AI 偏好训练 RM（RLAIF） | T32 |
| 评测 | 有害请求拒答率、越狱攻击成功率、良性请求的 over-refusal 率三者同时报 | 各报告 |

R1（T17）的全场景 RL 阶段对"有用性"用 RM，对"无害性"对整条 response 含推理过程打分；Llama 3（T14）描述了按风险分类体系收集对抗 prompt、用多轮红队迭代补数据。生产里人工标注仍是安全数据的主要来源，AI 反馈用于放大覆盖。

## 核心原理

### 有用性与无害性的冲突

```text
单一 reward R = w_h · R_helpful + w_s · R_safe
问题：w_s 大 → over-refusal；w_s 小 → 有害输出
门控形式：R = R_helpful  若 R_safe ≥ τ
          R = R_safe - c 否则
```

门控形式让安全成为硬约束、有用性成为优化目标，更接近产品要求。τ 与 c 要用 over-refusal 基准和越狱基准同时扫。

### 安全数据的三层

| 层 | 内容 | 用途 |
|---|---|---|
| 分类体系 | 风险类别（暴力、隐私、违法、自伤……）× 严重度 × 场景 | 保证覆盖，追踪每类的指标 |
| 对抗 prompt | 红队人工编写、模型生成的越狱变体、多轮诱导 | SFT 示范与偏好对 |
| 边界 prompt | 表面敏感实则良性（医学、法律、安全研究） | 治 over-refusal |

多轮与多语言是常见盲区：单轮安全的模型在多轮铺垫后可能失守；训练数据要包含多轮对抗轨迹，这就是 agent rollout 形态的安全数据。

### Constitutional AI / RLAIF（T32）

```text
阶段 1（SL-CAI）：模型生成回答 → 按原则自我批评 → 修改 → 用修改后的回答 SFT
阶段 2（RL-CAI）：模型对成对回答按原则选优 → AI 偏好数据 → 训练 RM → RL
```

可靠性边界：AI 标注在"明显有害 vs 明显安全"上与人一致率高，在边界案例上偏保守（倾向拒绝），需要人工抽检校准；原则的措辞直接影响分布。

## 打标任务中的 rollout

生产 RLHF 的数据来自标注任务。标注员看到的每条"回答"就是一条 **rollout**：策略模型在某个 checkpoint 对某个 prompt 的一次采样输出（单轮是一段文本，agent 任务是一整条含工具调用的轨迹）。理解这一点决定了标注协议怎么设计。

### 流水线

```text
prompt 池（分类体系覆盖、去重、脱敏）
  → 采样 rollout：当前策略 checkpoint × 采样参数 × n 条 / prompt
      （温度、top-p、max_tokens 记录进元数据；n 常为 2 到 8）
  → 任务打包：pairwise（两条比较）/ listwise（n 条排序）/ pointwise（单条打分或多维评分）
  → 分发：每个任务 k 个标注员（k = 2 到 3），带黄金题（gold）与重复题（dup）
  → 回收：一致性检查（Cohen/Fleiss κ、Krippendorff α）、黄金题准确率、耗时异常
  → 仲裁：分歧任务由资深标注或专家裁决
  → 入库：偏好对 / 评分 / 拒绝理由 / 标签，附 rollout 元数据与标注员 ID（匿名化）
```

### 标注员实际做什么

| 任务类型 | 标注员看到 | 输出 | 用于 |
|---|---|---|---|
| pairwise 偏好 | 同一 prompt 的 2 条 rollout | 更好的一条 + 置信度 + 理由标签 | RM 训练、DPO |
| listwise 排序 | n 条 rollout | 全序或部分序 | RM 训练（转成成对） |
| pointwise 评分 | 1 条 rollout | 多维分数（正确、有用、安全、格式） | 过滤 SFT 数据、RM 回归目标 |
| 安全标注 | 1 条 rollout + 分类体系 | 风险类别、严重度、是否应拒绝 | 安全 RM、评测集 |
| 轨迹标注（agent） | 整条多轮 rollout | 终态成功与否、每 turn 是否合理、失败 turn 定位 | outcome/process 数据、错误分析 |
| 修改（edit） | 1 条 rollout | 改写成理想回答 | SFT 示范数据 |

### 协议要点

- **rollout 元数据不可省**：checkpoint、采样参数、生成时间。同一 prompt 不同 checkpoint 的 rollout 混标会让 RM 学到"版本"而非"质量"。
- **随机化顺序**：pairwise 的左右位置随机，消除位置偏差。
- **长度控制**：标注员偏好长回答是已知偏差；协议里明确"同等质量不奖励长度"，并在 RM 训练时做长度平衡。
- **拒绝理由标签化**：只给"哪个更好"不够，理由标签（事实错误、不安全、不遵循指令、格式）让数据可用于分析与定向补数据。
- **黄金题比例**：5% 到 10%（示例），准确率低于阈值的标注员数据整批复核。
- **一致性目标**：κ 或 α 低于 0.6（示例）的任务类型说明协议本身不清晰，先改协议再标。
- **agent 轨迹标注**要给标注员回放工具（Agent 课 Day 4 的 trace 回放），否则无法判断工具返回是否被正确使用。

### 一致性与质量监控（脚本骨架）

```python
import itertools, collections
def fleiss_kappa(labels_per_item, categories):
    # labels_per_item: list[list[label]]，每项 k 个标注
    n_items = len(labels_per_item); k = len(labels_per_item[0])
    P_bar, p_cat = 0.0, collections.Counter()
    for labels in labels_per_item:
        cnt = collections.Counter(labels)
        P_bar += (sum(c*c for c in cnt.values()) - k) / (k*(k-1))
        p_cat.update(cnt)
    P_bar /= n_items
    total = n_items * k
    P_e = sum((p_cat[c]/total)**2 for c in categories)
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0

def annotator_report(tasks):
    # tasks: 含 gold 标记、annotator_id、label、duration
    by_a = collections.defaultdict(lambda: dict(gold=0, gold_ok=0, n=0, fast=0))
    for t in tasks:
        a = by_a[t["annotator_id"]]; a["n"] += 1
        if t.get("is_gold"):
            a["gold"] += 1; a["gold_ok"] += int(t["label"] == t["gold_label"])
        if t["duration_s"] < 5: a["fast"] += 1          # 示例阈值：疑似未阅读
    return {k: dict(v, gold_acc=v["gold_ok"]/max(1, v["gold"]), fast_rate=v["fast"]/v["n"]) for k, v in by_a.items()}
```

### 从标注到训练的数据格式

标注结果要能直接喂给 Day 11 / Day 12 的训练器，字段设计示例：

```json
{
  "prompt_id": "p_000123",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "rollouts": [
    {"rollout_id": "r_a", "checkpoint": "policy-step-1200", "sampling": {"temperature": 0.8, "top_p": 1.0, "max_tokens": 2048},
     "content": "...", "turns": null},
    {"rollout_id": "r_b", "checkpoint": "policy-step-1200", "sampling": {"temperature": 0.8, "top_p": 1.0, "max_tokens": 2048},
     "content": "...", "turns": null}
  ],
  "labels": [
    {"annotator": "a17", "preferred": "r_a", "confidence": 0.8, "reasons": ["factual_error_in_b"]},
    {"annotator": "a42", "preferred": "r_a", "confidence": 0.6, "reasons": ["b_over_refusal"]},
    {"annotator": "a03", "preferred": "r_b", "confidence": 0.5, "reasons": ["a_too_long"]}
  ],
  "resolved": {"chosen": "r_a", "rejected": "r_b", "agreement": 0.67, "adjudicated": false},
  "safety": {"category": "medical_boundary", "severity": 1, "should_refuse": false}
}
```

转换规则：`resolved.chosen/rejected` 直接生成 DPO 的 chosen/rejected 对；`agreement` 低于阈值的对可降权或剔除；`safety.should_refuse` 与实际是否拒绝的组合用于安全 RM 与 over-refusal 评测集；agent 任务把 `turns` 填为每 turn 的标签数组（合理 / 不合理 / 失败点），供 Day 25 的 process 数据使用。

### 规模与成本估算（示例口径）

| 项 | 估算方式 |
|---|---|
| 每条 pairwise 任务人工时长 | 2 到 6 分钟，长回答与 agent 轨迹成倍增加 |
| 有效数据率 | 回收 × 一致性通过率 × 黄金题通过率，示例 0.85 × 0.8 × 0.95 ≈ 0.65 |
| 每轮 RLHF 的偏好对量级 | 公开报告从数万到数十万不等（以报告为准），按分类体系分配 |
| rollout 生成成本 | 推理侧 token 数 × 单价；采样 n 条 / prompt 时随 n 线性 |
| 迭代周期 | 采样 → 标注 → 训练 → 评测一轮通常以周计；AI 反馈把周期压到天 |

规划标注任务时先算"每类需要多少有效对"，再反推发放量，而不是发放后再看够不够。

### 多轮与 agent 安全轨迹的采样

单轮 prompt 池采 rollout 直接用推理引擎批量生成；多轮对抗轨迹需要**对手模型**扮演攻击者，与策略模型交互若干轮生成整条轨迹，再交标注。这条轨迹的每一 turn 都是策略的一次 rollout，标注员标"哪一 turn 失守"。生成配置（对手模型、最大轮数、攻击策略集）同样入元数据。

## 面试题（5 题，附要点）

1. **标注任务里的 rollout 是什么？为什么元数据必须入库？** 策略 checkpoint 对 prompt 的一次采样输出；缺 checkpoint 与采样参数会让 RM 学到版本差异，且无法复现。
2. **pairwise 与 pointwise 各适合什么？** pairwise 一致性高、适合训练 RM/DPO；pointwise 提供多维分数、适合过滤与评测，但绝对分数一致性差。
3. **有用性与无害性怎么在 reward 里组合？** 加权易失衡；门控让安全成硬约束，阈值用 over-refusal 与越狱两类评测扫。
4. **AI 反馈能替代人工吗？** 明显案例可以，边界案例偏保守，需人工抽检校准；原则措辞决定分布。
5. **κ 很低怎么办？** 先怀疑协议而不是标注员：加判定顺序、示例、理由标签；主观维度改 pointwise 多维。

## 实现步骤

1. 建分类体系与 prompt 池：每类至少百条量级（示例），含多轮与多语言。
2. 采样 rollout：用当前策略 checkpoint，pairwise 任务每 prompt 采 2 到 4 条，温度 0.7 到 1.0（示例），记录元数据。
3. 写标注协议：定义"更好"的判定顺序（安全 → 正确 → 有用 → 风格）、理由标签集、边界案例示例。
4. 发放与回收：k=3，黄金题 8%，回收后跑一致性与标注员报告。
5. 训练安全 RM：Day 12 的 BT RM，输入含完整对话历史；对长度做平衡采样。
6. 进 RL：门控 reward；每 N 步在三类安全评测上跑一次。
7. AI 反馈补充：用 T32 的流程对分类体系里数据稀少的类别生成候选，人工抽检 10% 后入库。

## 实验

资源分层：API-only 可做实验 1、2；单卡可训小 RM 做实验 3。

### 实验 1：标注协议对一致性的影响

同一批 200 个 pairwise 任务，用两版协议（无判定顺序 vs 有"安全 → 正确 → 有用"顺序）各让 3 人标注，比较 κ。预期有顺序的协议 κ 明显更高。

### 实验 2：AI 标注与人工标注的一致率

用一个强模型按同一协议标注实验 1 的任务，与人工多数票比较；按"明显案例 / 边界案例"分层报告一致率。预期边界案例一致率显著更低且 AI 偏向拒绝。

### 实验 3：over-refusal 与越狱率的权衡曲线

训练安全 RM 后，在 RL 中扫门控阈值 τ 三个值，各训 100 步，在良性边界集与越狱集上测拒答率，画权衡曲线。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| RM 偏好长回答 | 标注数据长度偏差 | 按长度分桶看 RM 分数 | 协议明确；训练时长度平衡 |
| over-refusal 上升 | 安全权重过大或边界数据缺 | 良性边界集拒答率 | 门控 τ 调低；补边界数据 |
| 多轮越狱成功 | 只有单轮安全数据 | 多轮对抗评测 | 加多轮对抗轨迹 |
| κ 很低 | 协议模糊或任务本身主观 | 分任务类型看 κ | 改协议；主观类改 pointwise 多维 |
| RM 学到 checkpoint 特征 | rollout 元数据混杂 | 用 checkpoint 作标签训练分类器能否预测 | 同 prompt 同 checkpoint 配对 |
| AI 反馈数据让模型更保守 | 原则措辞偏拒绝 | 分层一致率 | 改原则；人工校准边界案例 |
| 标注员黄金题准确率骤降 | 疲劳或作弊 | 耗时分布 | 限时长；复核 |

## 验收标准

- 标注协议文档完成，含判定顺序、理由标签、黄金题与一致性阈值。
- 监控脚本能输出 κ、黄金题准确率、疑似未阅读率。
- 实验 1 到 3 至少完成两个，并能解释 rollout 元数据为何必须入库。

## 交付物

| 文件 | 内容 |
|---|---|
| `annotation-protocol.md` | 任务类型、判定顺序、理由标签、质量控制规则 |
| `annotation_qc.py` | 一致性与标注员报告脚本 |
| `safety-eval-report.md` | 三类安全指标与权衡曲线 |

## 参考

- T23 InstructGPT：RLHF 标注流程的原型（标注员、比较数据、RM）。
- T32 Constitutional AI：AI 反馈替代部分人工标注的流程。
- T14 Llama 3：安全分类体系、红队迭代、多 RM 组合的描述。
- T22 Tulu 3：开源后训练里安全数据的位置。
- T17 DeepSeek-R1：全场景 RL 阶段无害性 reward 的做法。
