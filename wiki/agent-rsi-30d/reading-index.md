---
title: Agent 自迭代、Agent RL、RSI 与 ASI 30 天教程索引
type: synthesis
tags: [agent, agentic-rl, auto-research, rsi, asi, reading-list]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# 30 天教程索引

配套课程：[[agent-rsi-30d/learning-path]]。每篇教程都包含学习目标、核心原理、实现步骤、实验、常见失败和验收标准。

## Week 1：先做出可测量的 Agent

| 天 | 教程 | 当天交付物 |
|---:|---|---|
| 1 | [[agent-rsi-30d/week1/day01-self-improvement-map|自改进地图与证据边界]] | 系统分层图、术语测试 |
| 2 | [[agent-rsi-30d/week1/day02-minimal-agent-runtime|最小 Agent Runtime]] | 显式状态机 Agent |
| 3 | [[agent-rsi-30d/week1/day03-tools-environments-sandbox|工具、环境与沙箱]] | 两个工具、隔离执行器 |
| 4 | [[agent-rsi-30d/week1/day04-tracing-replay|轨迹、事件与回放]] | trace schema、replay runner |
| 5 | [[agent-rsi-30d/week1/day05-evaluation-verifiers|任务集与 Verifier]] | 四层数据集、程序化 verifier |
| 6 | [[agent-rsi-30d/week1/day06-judges-statistics|LLM Judge 与统计]] | judge 校准、置信区间报告 |
| 7 | [[agent-rsi-30d/week1/day07-baseline-project|P0 基线项目]] | 可回放 Agent baseline |

## Week 2：非参数 Agent 自迭代

| 天 | 教程 | 当天交付物 |
|---:|---|---|
| 8 | [[agent-rsi-30d/week2/day08-reflection-feedback|反思与外部反馈]] | critique-revise 对照实验 |
| 9 | [[agent-rsi-30d/week2/day09-memory-learning|记忆学习]] | episodic/semantic memory policy |
| 10 | [[agent-rsi-30d/week2/day10-skills-curriculum|技能库与自动课程]] | skill registry、curriculum loop |
| 11 | [[agent-rsi-30d/week2/day11-prompt-optimization|Prompt 自动优化]] | GEPA 风格候选 archive |
| 12 | [[agent-rsi-30d/week2/day12-workflow-code-evolution|Workflow 与代码进化]] | patch evaluator、回滚机制 |
| 13 | [[agent-rsi-30d/week2/day13-multi-agent-coevolution|多 Agent 与共进化]] | 单/多 Agent 成本对照 |
| 14 | [[agent-rsi-30d/week2/day14-self-evolving-project|P1 自迭代项目]] | 五代非参数自迭代报告 |

## Week 3：Agent RL

| 天 | 教程 | 当天交付物 |
|---:|---|---|
| 15 | [[agent-rsi-30d/week3/day15-agent-rl-foundations|Agent RL 基础]] | POMDP 与 policy-gradient notebook |
| 16 | [[agent-rsi-30d/week3/day16-reward-design|Reward 与反作弊]] | reward spec、攻击用例 |
| 17 | [[agent-rsi-30d/week3/day17-credit-assignment|长轨迹信用分配]] | trajectory-to-transition pipeline |
| 18 | [[agent-rsi-30d/week3/day18-sft-offline-learning|SFT 与离线学习]] | SFT/preference baseline |
| 19 | [[agent-rsi-30d/week3/day19-policy-optimization|PPO、GRPO 与策略优化]] | 小规模 on-policy 训练 |
| 20 | [[agent-rsi-30d/week3/day20-harnessed-agent-rl|Harnessed Agent RL 系统]] | rollout/trainer 解耦架构 |
| 21 | [[agent-rsi-30d/week3/day21-agent-rl-project|P2 Agent RL 项目]] | hidden eval 与消融报告 |

## Week 4：Auto Research、RSI 与 ASI

| 天 | 教程 | 当天交付物 |
|---:|---|---|
| 22 | [[agent-rsi-30d/week4/day22-literature-hypothesis|文献与可证伪假设]] | claim graph、hypothesis cards |
| 23 | [[agent-rsi-30d/week4/day23-experiment-manager|实验管理器]] | experiment DAG、artifact registry |
| 24 | [[agent-rsi-30d/week4/day24-auto-research-systems|Auto Research 系统拆解]] | 三种架构复现与比较 |
| 25 | [[agent-rsi-30d/week4/day25-research-integrity|科研评测与诚信]] | 复现实验、伪造检测 |
| 26 | [[agent-rsi-30d/week4/day26-self-modification|Agent 自修改]] | self-model、patch boundary |
| 27 | [[agent-rsi-30d/week4/day27-evolutionary-archives|开放式搜索与谱系]] | population archive、谱系树 |
| 28 | [[agent-rsi-30d/week4/day28-controlled-rsi|受控 RSI 循环]] | 至少五代递归改进运行 |
| 29 | [[agent-rsi-30d/week4/day29-control-oversight|控制、监督与 Reward Hacking]] | threat model、红蓝队报告 |
| 30 | [[agent-rsi-30d/week4/day30-capstone-asi-evidence|结业项目与 ASI 证据]] | P3 完整系统、E0–E5 答辩 |

