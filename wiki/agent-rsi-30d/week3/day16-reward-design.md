---
title: "Day 16：Reward、Verifier 与反作弊"
type: concept
tags: [agentic-rl, reward, verifier, reward-hacking]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 16：Reward、Verifier 与反作弊

## 学习目标

写一份可执行 reward specification，并在训练前主动找出它鼓励的错误行为。

## Reward 分解

```text
R = R_task + λp R_process - λc cost - λi integrity_violation
```

`R_task` 来自最终可验证结果；`R_process` 只奖励与目标一致且难以投机的中间进展；cost 控制无止境思考；integrity violation 对 grader 篡改、数据泄漏等应直接终止，而非轻微扣分。

奖励塑形会改变最优策略。除非能证明 potential-based shaping，过程奖励都需用最终真实指标复核。不要奖励“调用了搜索”“写了解释”这类表面行为；奖励可验证证据或测试通过。

## Reward threat modeling

列出 Agent 可观察什么、可修改什么、评分器信任什么。攻击包括硬编码 task ID、读测试、改 metric、吞掉失败、伪造日志、用超时规避扣分、生成超长答案骗 judge。为每个攻击写 detector 和可信重算路径。

## 实验

1. 为 P2 任务写 reward spec，包含成功、部分进展、成本和完整性。
2. 人工写 20 个 adversarial outputs/patches，确认不会高分。
3. 用随机策略、规则策略和强 baseline 检查 reward 动态范围。
4. 比较 outcome-only 与 shaped reward 的行为；关注真实成功而非训练 reward。
5. 对 LLM judge 做答案顺序交换、引用核验和盲化。

## 验收

reward 与人工任务成功在校准集高度一致；所有高危篡改被独立 detector 捕获；提交 reward card、攻击表、单元测试和“已知但未解决的漏洞”。

下一课：[[agent-rsi-30d/week3/day17-credit-assignment]]。

