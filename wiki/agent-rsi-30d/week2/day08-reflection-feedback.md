---
title: "Day 8：反思、批评与外部反馈"
type: concept
tags: [agent, reflection, feedback, self-correction]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 8：反思、批评与外部反馈

## 目标与边界

反思是把反馈转成下一次决策的上下文，不更新权重。它只有在获得新证据、执行错误或独立批评时才可能增加信息；让同一个模型重复“再想想”常常只是增加计算。

## 反馈管线

```text
attempt → external verifier → failure evidence
        → critic: diagnosis + causal hypothesis + repair
        → actor retries under same total budget
```

批评输出固定字段：首次错误步骤、证据、错误类型、最小修复、适用条件、置信度。禁止只有“更仔细”“重新检查”一类不可执行建议。critic 不得看到 hidden answer，只看可公开的 verifier evidence。

## 实验

在 40 个失败任务上比较：无重试；同 prompt 重试；自我反思；独立 critic；外部 verifier + critic。每组限制相同 token 和工具调用。记录修复率、破坏原本正确答案的比例、每次修复成本、critic 诊断与人工标签一致率。

为反思做消融：移除工具错误、移除测试输出、移除旧动作，看哪种证据真正贡献修复。抽样检查“最终正确但诊断错误”的偶然修复。

## 实现要点

- 反思与原轨迹分开存，不覆盖历史。
- retry 从失败点 fork，也做从头运行的对照。
- 每条 reflection 带 task scope，不能直接当全局规则。
- 连续两轮无新证据或动作重复即停止。

## 验收

提交五组同预算结果、20 条人工诊断、反思模板和停止规则。只有 hidden 修复率提升且破坏率可接受，才将反思接入主 Agent。

必读：[Reflexion](https://arxiv.org/abs/2303.11366)。下一课：[[agent-rsi-30d/week2/day09-memory-learning]]。

