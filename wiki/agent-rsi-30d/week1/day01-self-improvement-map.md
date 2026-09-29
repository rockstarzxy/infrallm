---
title: "Day 1：Agent 自改进地图与证据边界"
type: concept
tags: [agent, self-improvement, rsi, asi, evaluation]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 1：Agent 自改进地图与证据边界

## 今日目标

建立全课程共同语言。完成后，你应能看到一个“自我改进”系统时，准确指出谁在改、改什么、用什么反馈、改动是否持久、是否真的递归。

## 1. Agent 不是单个模型

把系统写成六元组 `A = (M, H, T, E, μ, V)`：模型 `M` 产生候选决策；harness `H` 管理控制流；工具 `T` 执行动作；环境 `E` 返回观察；记忆 `μ` 保存跨步或跨任务状态；verifier `V` 给出反馈。很多“模型进步”其实只是 H、T 或 μ 变了。

对每个系统做四问：优化对象是什么？反馈来自哪里？更新能否跨任务保留？更新后的版本是否负责下一轮更新？只有最后一问为真，才出现递归性。

## 2. 六级证据

| 级别 | 改动 | 合理声明 | 不能推出什么 |
|---|---|---|---|
| L0 | 重采样、搜索、反思当前答案 | inference-time improvement | 模型学会了 |
| L1 | 写入记忆或技能 | lifelong/context learning | 权重变强了 |
| L2 | 改 prompt、workflow、tools、代码 | compound system optimization | 通用智能提升 |
| L3 | SFT/RL 更新策略参数 | policy learning | 改进过程也变强 |
| L4 | 新版本继续修改自己的改进器 | bounded RSI | 开放世界持续增长 |
| L5 | 跨域持续超越人类并改进方向选择 | ASI 假设 | 当前没有公认实例 |

## 3. 一个改进声明需要哪些对照

`新版分数 > 旧版分数` 不够。至少控制模型、推理预算、工具、任务集和随机种子；使用未参与优化的 hidden set；同时报告成功率、费用、步骤、失败率和安全回归。若新版只是调用次数更多，应该比较 `success@same_budget`。

## 今日实验（6 小时）

1. 选择三个系统：Reflexion、Agent Lightning、Darwin Gödel Machine。
2. 分别填 `M/H/T/E/μ/V` 表，标出实际被更新的组件。
3. 为每个系统写一条“论文能支持的最强声明”和一条“不能支持的宣传式声明”。
4. 设计一个反例：dev 分数连续增长，但 transfer 分数下降。
5. 画出本课程最终系统的信任边界：Agent 可写区、只读区、不可见区。

## 验收

- 能解释“反思三次后答对”为什么不是 RSI。
- 能解释“Agent 修改自身代码”为什么仍可能不是 RSI：修改可能未被接受、未累积，或下一代改进器仍是原版本。
- 提交 `self-improvement-map.md`，含六元组表、L0–L5 判定和三条可证伪课程假设。

下一课：[[agent-rsi-30d/week1/day02-minimal-agent-runtime]]。

