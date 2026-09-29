---
title: "Day 30：P3 Capstone：Agentic RL 全流程与结课复盘"
type: concept
tags: [llm-training, capstone, agentic-rl, retrospective, interview]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 30：P3 Capstone：Agentic RL 全流程与结课复盘

> 上一课 [[llm-training-30d/week4/day29-training-ops-at-scale]] · 课程入口 [[llm-training-30d/index]]。相关：Agent 课的 P2 项目 [[agent-rsi-30d/week3/day21-agent-rl-project]]（同一任务域，本课要求训练系统层面的完整交付）；推理课的最终设计 [[ai-infra-30d/week4/day29-final-design]]（训练产出的模型要能在那套系统上跑）。

## 学习目标

1. 在一个可验证的工具任务上独立完成 SFT 冷启动 → agentic GRPO → hidden 评测 → reward hacking 审计 → 成本报告的全流程。
2. 能用一张知识图谱把 30 天内容串成"数据 → 训练系统 → 算法 → rollout → 评测"的闭环。
3. 能回答 20 道覆盖全课的面试题，每题在两分钟内讲清机制和取舍。
4. 能写出自己的进阶路线：下一步补哪块、用什么资源。

## 工业现状

这份 capstone 对应的是 2025 到 2026 年一线团队 agentic 后训练的最小闭环：Kimi K2（T19）、GLM-4.5（T20）、Qwen3-Coder 博客描述的流程都是"合成或收集可执行任务 → SFT 冷启动 → 多轮 RL → 拒绝采样再 SFT → 评测门禁"。你在单机到 8 卡上做的是同一形态的缩小版，差距在环境数量、模型规模和迭代轮数，而不是流程。

## P3 项目规格

### 任务选择（三选一）

| 任务 | 环境 | 工具 | outcome verifier | 推荐模型 |
|---|---|---|---|---|
| text-to-SQL（多轮） | SQLite 数据库集（Spider 类 schema，自造 hidden 数据变体） | `list_tables`、`describe`、`run_sql`、`submit` | 执行结果集匹配（3 个 hidden 数据变体） | 1.5B 到 7B |
| 小型代码修复 | 函数级 bug 修复任务集，每任务一个容器 | `read_file`、`edit`、`run_tests`、`submit` | hidden tests 通过 | 7B 代码模型 |
| 检索问答 | 本地文档库 + 检索服务 | `search`、`open`、`submit` | 答案精确匹配或 judge（带 hidden rubric） | 1.5B 到 7B |

选择依据：单卡选 text-to-SQL（环境最轻）；8 卡可选代码修复。

### 流程与验收点

```text
阶段 0  环境与评测集（Day 23）
        ├─ 任务集 ≥ 500，train/dev/hidden = 70/10/20，hidden 只在验收用
        ├─ 工具服务化，超时与失败处理，trace 可回放
        └─ 验收点：随机策略与"直接猜答案"策略的 hidden 分数（作为下限）

阶段 1  SFT 冷启动（Day 8-10）
        ├─ 用强模型或人工构造 200 到 1000 条成功轨迹（示例），过滤 verifier 通过者
        ├─ loss mask 只算 assistant token；多轮格式与工具调用格式固定
        └─ 验收点：dev avg@4 高于阶段 0 下限；工具格式错误率 < 5%（示例）

阶段 2  agentic GRPO（Day 15-25）
        ├─ reward spec（Day 25）：outcome + process 罚分 + 步数约束 + hidden 变体
        ├─ verl 或 TRL 的多轮 agent loop；n = 8；dynamic sampling；clip-higher
        ├─ 稳定性面板（Day 20）：entropy、KL、长度、零梯度 group 比例、train/infer logprob 差
        ├─ 至少 2 组消融：KL 系数（0 vs 小值）、process 罚分（有 vs 无）
        └─ 验收点：dev avg@4 相对 SFT 提升且 hidden 同步提升；成本记录完整

阶段 3  拒绝采样 SFT（可选，Day 27）
        └─ 用 RL 策略采样，保留 verifier 通过的轨迹再 SFT 一次，比较是否继续提升

阶段 4  审计与报告（Day 25、28、29）
        ├─ reward hacking 审计：高分轨迹抽样 100 条人工或 judge 检查
        ├─ 导出 safetensors，用 vLLM 加载做 hidden 评测，报告 logprob 差
        └─ 成本报告：GPU 小时按 SFT / rollout / 训练更新 / 评测拆分
```

### 交付清单

| 文件 | 内容 |
|---|---|
| `env/` | 环境接口、工具服务、任务集与 split、trace 回放 |
| `sft/` | 冷启动数据构造脚本、SFT 配置、dev 结果 |
| `rl/` | reward spec 与函数、训练配置、消融配置、训练日志（含面板截图或导出） |
| `eval/` | hidden 评测脚本（avg@n + 标准误）、阶段 0 到 3 的结果表 |
| `audit.md` | reward hacking 审计结论、发现的问题与修复 |
| `cost.md` | GPU 小时拆分、rollout 占比、每次评测成本 |
| `report.md` | 全流程报告：每阶段做了什么、为什么、数据、失败与教训 |

### 评分标准（100 分）

| 项 | 分值 | 满分要求 |
|---|---:|---|
| 环境与评测 | 15 | hidden 从未用于选择；下限基线齐全；trace 可回放 |
| SFT 冷启动 | 10 | 数据来源与过滤清楚；格式错误率达标 |
| RL 训练 | 25 | 面板完整；消融有结论；提升在置信区间外 |
| 稳定性与诊断 | 15 | 至少诊断并修复一个真实问题（entropy、长度、mismatch 之一） |
| 审计 | 10 | 有具体的 hacking 检查方法与结果，哪怕结论是"未发现" |
| 导出与一致性 | 10 | 推理侧评测与训练侧一致；logprob 差有量级说明 |
| 成本与报告 | 15 | 成本可追溯；报告能让他人复现；诚实记录负结果 |

不合格条件：hidden 泄漏进训练或选择；reward 函数在训练中被改而未记录；报告与日志不一致。

## 全课知识图谱

```text
                         ┌──────────── Week 1：训练系统基础 ────────────┐
                         │ 显存与算力数学(D2) → 数据与 packing(D3)       │
                         │ DP/ZeRO/FSDP2(D4) → TP/PP/CP/EP(D5)          │
                         │ 框架与效率/profiling(D6) → P0 FSDP2 训练(D7) │
                         └──────────────────────┬───────────────────────┘
                                                │ 提供：能跑、能算、能测的训练循环
        ┌───────────────────────────────────────▼──────────────────────────────────┐
        │ Week 2：监督与偏好                                                        │
        │ SFT(D8) ─ LoRA/PEFT(D9) ─ 数据工程/合成/蒸馏(D10)                          │
        │ DPO 家族(D11) ─ Reward Model(D12) ─ 评测与实验管理(D13) ─ P1 SFT+DPO(D14) │
        └───────────────────────────────────────┬──────────────────────────────────┘
                                                │ 提供：冷启动模型、RM、评测协议
        ┌───────────────────────────────────────▼──────────────────────────────────┐
        │ Week 3：RLVR 与 rollout 系统                                              │
        │ rollout 与 RL 基础(D15) ─ PPO/GRPO/DAPO/GSPO(D16) ─ verifier(D17)          │
        │ RL 框架架构(D18) ─ rollout 引擎与权重同步(D19) ─ 异步与稳定性(D20)         │
        │ P2 GRPO(D21)                                                              │
        └───────────────────────────────────────┬──────────────────────────────────┘
                                                │ 提供：可运行的 RL 循环与诊断能力
        ┌───────────────────────────────────────▼──────────────────────────────────┐
        │ Week 4：Agentic RL 与生产                                                 │
        │ 多轮形式化(D22) ─ 环境/gym(D23) ─ agent rollout 基础设施(D24)              │
        │ reward 与 credit(D25) ─ 案例(D26) ─ 推理模型(D27)                          │
        │ 安全与标注(D28) ─ 运维与飞轮(D29) ─ P3 capstone(D30)                       │
        └──────────────────────────────────────────────────────────────────────────┘

横向主线：rollout 这个词在 D1 定义，D15 精确化，D19/D24 是它的系统实现，D25/D28 是它被评分与标注的方式。
外部接口：rollout 引擎 = 推理课的 vLLM/SGLang；环境与 trace = Agent 课 Week 1。
```

## 20 道面试题（附要点答案）

**训练系统**

1. **AdamW 全参训练每个参数占多少字节？从哪来？** BF16 权重 2 + BF16 梯度 2 + FP32 主权重 4 + 两个 FP32 动量 8 = 16 字节；ZeRO/FSDP 把后三项按 DP 数切分。
2. **FSDP2 与 ZeRO-3 的关系？** 同一思想（参数、梯度、优化器状态全分片，按需 all-gather）；FSDP2 是 PyTorch 原生的 per-parameter DTensor 实现，与 TP/PP 组合更自然。
3. **TP 为什么要放在节点内？** 每层两次 allreduce、通信量与激活大小同量级，需要 NVLink 带宽；跨节点用 PP 或 DP 通信量小得多。
4. **MFU 怎么算？多少算健康？** 实测 tokens/s × 6N（近似 FLOPs/token）÷ GPU 峰值算力；密集模型在好的框架上通常几十个百分点（以硬件与序列长度为准），MoE 更低。
5. **packing 为什么要隔离注意力？** 否则一个样本能看到同一 pack 里前面样本的 token，训练分布被污染；用 varlen attention 或重置 position_ids。

**监督与偏好**

6. **SFT 的 loss mask 应该 mask 什么？** 只对 assistant 生成的 token 算 loss；system、user、tool 返回不算；多轮里每个 assistant turn 都算。
7. **LoRA 何时与全参等价、何时不等价？** 容量足够（rank 与数据量匹配）且学习率按比例放大时，SFT 与 RL 上接近全参（B04）；数据量大、需要学新知识的预训练式任务不等价。
8. **DPO 的隐式奖励是什么？为什么会出现 chosen 似然下降？** β·log(π/π_ref)；DPO 只优化 chosen 与 rejected 的差，两者似然可同时下降；用 on-policy 偏好对与 SFT 正则缓解。
9. **RM 过优化怎么检测？** 固定 hidden 评测或人工评估随 RM 分数上升的曲线出现拐点；KL 与 RM 分数的关系图。
10. **合成数据里 rejection sampling 与蒸馏的区别？** rejection sampling 用 verifier 过滤自己或强模型的采样；蒸馏是学教师分布，on-policy 蒸馏在学生采样上用教师 logprob 逐 token 监督。

**RLVR 与 rollout**

11. **rollout 的精确定义？on-policy 的含义？** 一个 prompt 用当前策略采样的一条完整 response 或轨迹；on-policy 指用于更新的 rollout 来自当前（或版本差在允许范围内的）策略。
12. **GRPO 相对 PPO 去掉了什么、代价是什么？** 去掉 value model，用 group 内 reward 归一化做 advantage；代价是每个 prompt 要采多条、全同 reward 的 group 无梯度、有长度与难度偏置（T38）。
13. **DAPO 的四项改进各解决什么？** clip-higher 治 entropy 崩塌；dynamic sampling 治零梯度 group；token-level loss 治长回答梯度被稀释；overlong shaping 治截断记 0 的噪声。
14. **训练与推理引擎的 logprob 不一致为什么让 RL 失效？怎么修？** 采样分布与更新分布不同，等价于隐式 off-policy，梯度有偏；用截断重要性采样修正（B01）、统一关键算子精度（T21 的输出头）、batch-invariant kernel（B03）。
15. **权重同步在共置模式下怎么做？** 训练侧按推理侧的 TP 布局重分片，NCCL broadcast 或 CUDA IPC 传到引擎；引擎 sleep 释放显存给训练，唤醒后加载。
16. **异步 RL 的 staleness 怎么控制？** 限制轨迹的策略版本差（示例 ≤ 2 到 4 个版本）；超限丢弃或用 IS 权重修正；监控 off-policy 比例。

**Agentic RL 与生产**

17. **多轮 agent RL 的 loss mask 与单轮有什么不同？** 工具返回 token 不进 loss 也不进 KL；每个 assistant turn 的 advantage 可用 trajectory-level 广播或 turn-level 折扣。
18. **agent 任务上 GRPO 为什么容易零梯度？怎么办？** pass rate 极低使 group 全 0；dynamic sampling、难度分桶课程、分级 outcome（配 hidden tests）。
19. **打标任务里的 rollout 是什么？标注协议最重要的三条？** 策略 checkpoint 对 prompt 的一次采样；元数据入库、位置随机化、判定顺序明确并用黄金题与 κ 控质量。
20. **发布一个后训练模型前必须做的数值检查是什么？** 导出后在推理引擎上对同一批 prompt 比较逐 token logprob 差，量级与 dtype、kernel 相关；差异大时 RL 侧要开 IS 修正、推理侧要查量化与 backend。

## 常见失败与诊断（capstone 专用）

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| SFT 后 dev 分数与随机策略接近 | 冷启动数据格式与环境不一致、工具名不匹配 | 抽 20 条轨迹看格式错误 | 修数据构造脚本；先跑格式验证 |
| RL 第一步就大量零梯度 | 任务太难 | 零梯度 group 比例 | 先做 expert iteration；难度分桶 |
| dev 升 hidden 不升 | 用 dev 选了太多超参、或 hacking | 对比 dev/hidden 曲线 | 减少选择次数；审计 |
| rollout 占比 > 85% | 长尾、沙箱串行 | 最长样本时间、超时率 | 异步、容器池、max_turns |
| 推理侧评测低于训练侧 | 导出错误或 chat template 不一致 | logprob 差、模板对比 | 修导出；固定模板文件 |
| 成本失控 | 评测过频、n 过大 | cost.md | 分级评测 |

## 验收标准

- 交付清单齐全，评分自评 ≥ 70 且每项有证据。
- 20 道面试题能口头作答，其中至少 15 题能画图或写公式。
- 进阶路线写出并与自己的资源匹配。

## 进阶方向

| 方向 | 下一步 | 资源 |
|---|---|---|
| 训练系统 | 读 Megatron-Core 的 TP/PP 实现与 TorchTitan 的 FSDP2 + TP 组合；实现一次 CP | T02、T09、T11、B09、B10 |
| RL 算法 | 复现 DAPO 与 GSPO 的消融；研究 MoE 上的 RL 稳定性 | T37、T39、T21 |
| RL 系统 | 读 verl 的 HybridFlow 与 slime 的异步实现；自己实现权重同步 | T42、B06、B07、T44 |
| Agentic | 扩大环境数量（合成任务流水线）；多环境混合训练 | T19、T20、T45 |
| 推理模型 | 长 CoT 的长度控制与蒸馏系统化；on-policy 蒸馏实现 | T17、T38、B02 |
| 数据与评测 | 建自己的 hidden 评测集与污染检测；生成式 RM 训练 | T22、T34、T33 |
| 安全 | 多轮红队流水线；Constitutional 式自动数据 | T32、T14 |
| 与推理课衔接 | 把训练产出的模型按推理课 Day 29 的设计部署并压测；用推理优化专项排查 serving 问题 | [[ai-infra-30d/reading-index]]、[[ai-infra-30d/optimization/00-index]] |
| 与 Agent 课衔接 | 把 capstone 的 agent 放进 Agent 课 P3 的受控 RSI 循环 | [[agent-rsi-30d/index]] |

## 交付物

见"交付清单"。另加 `retrospective.md`：30 天里每周最重要的一个认知、最大的一个错误、下一步计划。

## 参考

- T19、T20、T45：capstone 对标的工业流程。
- T37、T38、T39：RL 训练配置与消融依据。
- T42、B06、B07：训练框架文档。
- B01、B03：train/infer 一致性检查。
- T22 Tulu 3：开源后训练全流程的可复现范本。
