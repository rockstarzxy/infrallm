---
title: Day 26：Agent 自修改、Self-model 与 Patch Boundary
type: concept
tags: [agent, self-modification, rsi, safety]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 26：Agent 自修改、Self-model 与 Patch Boundary

今天让 Agent 修改自己的可变组件，但不允许它控制裁判、权限边界和最终发布。重点是建立 self-model、明确 patch boundary，并让每次修改经过静态检查、单元测试、沙箱评测、回归测试和人工可读审计。

## 学习目标

1. 区分运行时适应、组件优化、代码自修改和递归自改进。
2. 用 machine-readable self-model 描述系统组件、依赖和权限。
3. 约束可修改面，使候选变更可比较、可回滚、可追责。
4. 实现从提案到 canary 发布的五道 gate。

## 1. 什么算自修改

修改当前上下文、检索一条记忆属于运行时适应；更新 prompt、workflow、skill、工具选择器属于持久组件优化；修改 optimizer 或生成下一代改进器才开始接近递归自改进。所有层次都可能提高成绩，但证据等级不同，报告中必须分别标注。

建立 `self_model.yaml`：

```yaml
components:
  planner: {path: agent/planner.py, mutable: true, tests: [planner_unit]}
  prompts: {path: prompts/, mutable: true, tests: [prompt_schema]}
  verifier: {path: evaluator/, mutable: false}
  sandbox: {path: runtime/sandbox.py, mutable: false}
  release_policy: {path: governance/, mutable: false}
capabilities: [read_repo, edit_mutable_paths, run_dev_eval]
denied: [read_hidden_answers, edit_audit_log, network_by_default]
```

Self-model 不要求系统具有人类式自我意识。它是一个用于推理和验证的工程模型：我由哪些模块构成、哪些可以修改、修改会影响什么。

## 2. Patch Proposal

Agent 不直接改主分支，只输出候选包：

```yaml
proposal_id: p27
parent: generation-2
observed_failure: tool retry loops after timeout
mechanism: distinguish retryable and terminal errors
files: [agent/tool_policy.py, tests/test_tool_policy.py]
predicted_effect: lower tool errors without token increase
rollback: restore parent artifact hash
```

候选包必须包含 diff、测试、生成轨迹和预期副作用。无机制说明的“大规模重写”降低优先级，因为失败后难以归因。

## 3. 五道 Gate

1. **Boundary gate**：拒绝越界路径、二进制、不透明下载和权限提升。
2. **Static gate**：语法、类型、依赖和危险调用检查。
3. **Unit gate**：受影响模块测试；新增行为必须有至少一个测试或 verifier。
4. **Behavior gate**：固定 dev 集、成本上限和安全探针。
5. **Regression gate**：旧能力、transfer 集、资源和拒绝行为不能越过阈值。

通过后只发布到 canary。Canary 在少量任务上出现异常就自动回滚；候选永远不能删除 parent checkpoint。

## 4. Gödel Agent 与 SEAL 的启发

Gödel Agent 研究代理修改自身逻辑或代码的开放式框架；SEAL 强调模型生成自编辑数据并通过训练吸收。它们都比“让模型重写 prompt”更接近持久自修改，但仍需外部任务、训练基础设施和验证器。阅读时要追问：修改了哪个状态？收益是否跨代保留？改进器本身是否变强？

## 5. 今日实验

从 P1 或 P2 收集 20 条失败轨迹，让 Agent 生成 8 个 patch proposal。至少包含 prompt、workflow 和代码三类候选。运行五道 gate，保留最多 2 个进入 canary；人为加入一个越界修改 verifier 的候选和一个能通过 dev 但破坏 transfer 的候选，验证 gate 能拒绝它们。

## 验收标准

- self-model 覆盖全部关键组件、依赖、可变性和权限。
- 任意候选可从 proposal 追踪到 diff、测试、评测和发布决定。
- 越界候选在执行前被拒绝，transfer 回归候选不能进入正式版本。
- 回滚命令能在 2 分钟内恢复 parent，并通过 smoke test。

必读：[Gödel Agent](https://arxiv.org/abs/2410.04444)、[SEAL](https://arxiv.org/abs/2506.10943)。下一课：[[agent-rsi-30d/week4/day27-evolutionary-archives]]。
