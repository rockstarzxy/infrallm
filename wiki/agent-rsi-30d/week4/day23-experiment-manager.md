---
title: Day 23：实验管理器与可复现执行
type: concept
tags: [agent, auto-research, experiment-manager, reproducibility]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 23：实验管理器与可复现执行

Auto Research 的瓶颈常常不是提出想法，而是让几十个实验在失败、重试、并发和资源限制下仍可解释。今天实现一个小型实验操作系统：规范输入、生成 DAG、隔离运行、登记产物、聚合结果，并允许第三方从零复现。

## 学习目标

1. 将研究计划编译成不可歧义的 experiment spec。
2. 使用 DAG 表达数据准备、训练、评测和分析之间的依赖。
3. 让配置、代码、数据、环境和结果通过内容哈希绑定。
4. 正确处理失败重试、早停、并发污染和选择性汇报。

## 1. Experiment Spec

每个实验先写 spec，后运行代码：

```yaml
id: H03-step-credit-seed1
hypothesis: H03
code_commit: <git-sha>
dataset_version: <content-hash>
environment: <container-or-lock-hash>
seed: 1
budget: {gpu_hours: 2, max_rollouts: 500}
command: python train.py --config configs/h03.yaml
primary_metric: hidden_pass_at_1
stop_rule: no_gain_3_evals
parent_runs: [baseline-seed1]
```

运行开始后冻结 spec。任何修改都生成新的 `run_id`，不能在原记录上覆盖。这样才分得清“重跑”和“换了实验”。

## 2. Experiment DAG

把一个假设拆成：`prepare_data → train{seed} → evaluate{split} → aggregate → report`。调度器只执行依赖已成功的节点。节点状态至少包括 `queued/running/succeeded/failed/cancelled/invalid`。`failed` 表示工程失败；`succeeded` 但指标下降仍是有效科研结果。

实现时应有三个接口：

```text
submit(spec) -> run_id
heartbeat(run_id, usage, step)
finalize(run_id, status, artifacts, metrics)
```

任务必须在隔离工作目录运行。并发 run 不能共享可写 checkpoint、缓存键或临时文件名。

## 3. Artifact Registry

产物按内容哈希登记，不按文件名猜测版本。至少保存：stdout/stderr、完整配置、环境锁、训练曲线、checkpoint、逐样本预测、verifier 输出、资源用量。manifest 示例：

```json
{"run_id":"r_104","artifact":"predictions.jsonl","sha256":"...","producer":"eval-hidden","size":183042}
```

报告中的每个数字都应能反查到某个 artifact 和聚合脚本。聚合脚本也属于产物，不能在 notebook 中手工复制数字。

## 4. 失败与预算策略

- 基础设施错误可在相同 spec 下自动重试，最多 2 次，并记录全部失败。
- OOM 要生成新 spec 调整 batch size；不能默默改变训练参数。
- 早停规则必须在运行前写入，避免看见差结果后临时终止。
- 每个假设预先分配种子和预算；某方法失败时不能把剩余预算只给喜欢的方法。
- 超预算的 run 立即进入 `invalid`，其指标不得用于主结论。

## 5. 今日实现与故障注入

实现本地队列和 SQLite/JSONL registry，提交一个包含 12–30 个节点的 DAG。主动注入四类故障：进程退出、产物损坏、心跳超时、重复提交。验证调度器能区分重试、失效和成功，并能从中断处恢复。

然后选择昨天的一张假设卡，运行 1 个基线 × 3 seeds 与 1 个处理组 × 3 seeds。自动生成含均值、置信区间、成本和失败数的 Markdown 表。

## 验收标准

- 从空目录使用一条命令可完成小实验并重建报告。
- 相同 spec 的 `run_id`、配置哈希和数据哈希可追踪；不同 spec 不会覆盖。
- 四类故障均被正确标记，失败样本没有从分母中消失。
- 随机选择报告中的一个数字，可在 3 分钟内追溯到逐样本结果。

下一课：[[agent-rsi-30d/week4/day24-auto-research-systems]]。
