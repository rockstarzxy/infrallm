---
title: Day 30：P3 结业项目、RSI 答辩与 ASI 证据标准
type: synthesis
tags: [agent, rsi, asi, capstone, evaluation]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 30：P3 结业项目、RSI 答辩与 ASI 证据标准

最后一天把四周的组件集成为一个受控 Auto Research + self-improvement 系统。项目的目标不是宣布“做出 ASI”，而是准确回答：系统改进了什么、证据有多强、哪些替代解释已排除、离 RSI 或 ASI 的主张还缺什么。

## 项目目标

系统接收一个研究问题，完成文献主张整理、假设生成、实验编排、候选修改、独立评测和研究报告；随后在固定预算内运行至少五代改进。所有关键决定、产物和权限事件可回放。

最低架构包含：

1. Agent runtime、工具沙箱、trace/replay。
2. train/dev/hidden/transfer 四层任务集和 immutable verifier。
3. claim graph、hypothesis cards、experiment DAG 和 artifact registry。
4. prompt/workflow/code 至少两种持久修改面。
5. population archive、谱系、canary、rollback。
6. 独立 monitor、审计日志和红队任务。

## 1. 最终冻结协议

在查看最终 hidden 结果前提交 `final-manifest.yaml`：代码 commit、数据哈希、环境哈希、五代谱系、主指标、安全阈值、停止规则和预计运行预算。冻结后只允许修复基础设施故障；任何行为性修改都使当前 hidden 运行失效，并生成新版本。

最终评测顺序：baseline → G1–G5 → static improver 对照 → transfer A/B → red-team → 独立复现。随机化任务顺序，保存逐样本输出。若复现失败，答辩仍可进行，但证据等级下调。

## 2. E0–E5 证据梯度

| 等级 | 可以主张 | 最低证据 |
|---|---|---|
| E0 | 演示可运行 | 单次任务和完整 trace |
| E1 | 组件优化有效 | 对照、多个种子、dev 改进 |
| E2 | Agent 自改进有效 | 持久修改、hidden 提升、回归通过 |
| E3 | 多代自改进 | 至少五代、等预算、谱系与迁移保留 |
| E4 | 递归改进证据 | 改进器效率优于 static improver，跨任务成立 |
| E5 | 广泛递归能力证据 | 多领域、长周期、独立复现、强控制下持续扩展 |

本课程预期优秀项目达到 E2–E3；E4 需要很强的元层对照；E5 只是研究标准。即使达到 E5，也不能单凭课程 benchmark 宣称 ASI。

## 3. ASI 主张需要什么

“ASI”没有被单一 benchmark 操作化。严肃主张至少需要：广泛领域表现、超过顶尖人类或团队的清晰比较、长期自主性、真实环境中的稳健迁移、自主研究与工程能力、可重复的自我改进，以及在对抗评测和控制下仍成立的证据。还需排除测试污染、人工隐性介入、计算量暴增和 evaluator 被利用等解释。

因此最终报告必须包含“不支持的主张”一节。例如：系统只在代码优化任务提升，就不能外推为科学能力；五代得分提高但元效率未提升，就不能称为递归加速；代理能设计实验但无法独立复现，就不能称为自主科研闭环。

## 4. 评分量表（100 分）

| 维度 | 分值 | 满分要求 |
|---|---:|---|
| 系统与可复现性 | 20 | 一键复现、版本冻结、完整 registry |
| 实验与统计 | 20 | 强基线、多种子、区间、消融、失败分析 |
| 自改进证据 | 20 | 五代谱系、等预算、对象层与元层指标 |
| Auto Research | 15 | 可证伪假设、DAG、独立复现、诚信检查 |
| 控制与安全 | 15 | 最小权限、监控、红队、回滚、损失上限 |
| 论证质量 | 10 | 结论不过界，替代解释和限制完整 |

硬性不通过项：读取 hidden answer、候选可修改 verifier、删除失败运行、无法对应代码与结果、用不同预算比较后声称算法优势。

## 5. 答辩问题

答辩者需要现场回答并用 artifact 证明：

1. 哪个具体组件发生了持久变化？
2. 最强替代解释是什么，你用哪个对照排除了它？
3. 第几代开始平台化，为什么？
4. 改进器本身变强的证据在哪里？
5. 哪个结果在 transfer 上失败？
6. 红队最接近成功的攻击是什么？
7. 若扩大 100 倍预算，首先失效的控制是什么？
8. 你最高能主张 E 几，为什么不能再高一级？

## 6. 最终交付目录

```text
capstone/
├── README.md              # 一键运行、范围与限制
├── final-manifest.yaml    # 冻结版本与预注册规则
├── literature/            # papers、claims、hypotheses
├── experiments/           # specs、DAG、registry、artifacts
├── agent/                 # runtime 与可变组件
├── evaluator/             # 独立、只读评测器
├── archive/               # 候选、谱系、代际指标
├── safety/                # threat model、监控与红队结果
└── report.md              # 研究报告和 E0–E5 自评
```

## 最终验收

- 新机器或干净环境能按 README 复现至少一个 baseline 和最终结果。
- 五代运行、对照、hidden/transfer、红队和独立复现都有原始产物。
- 报告中的每个表格数字能反查到 run 和逐样本记录。
- 结论使用 E0–E5 证据等级，明确列出失败、限制和仍需的证据。

完成后回到 [[agent-rsi-30d/index|课程主页]] 整理作品集，并依据 [[agent-rsi-30d/learning-path|学习路径]] 的统一协议复核四个项目。
