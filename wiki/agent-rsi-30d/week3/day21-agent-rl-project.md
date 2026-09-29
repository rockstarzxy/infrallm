---
title: "Day 21：P2 端到端 Agent RL 项目"
type: synthesis
tags: [agentic-rl, project, evaluation]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# Day 21：P2 端到端 Agent RL 项目

## 研究问题

相对冻结的 base/SFT policy，在线 Agent RL 是否在未见任务上提高真实成功率，并保持成本、格式和完整性？

## 完整流水线

冻结 dataset/verifier → base eval → 收集高质量轨迹 → SFT → SFT eval → on-policy rollouts → reward/credit → policy updates → 预定 checkpoint → hidden eval → 红队。

至少做四项消融：SFT only vs RL；outcome vs shaped reward；trajectory vs turn credit；有/无 KL。资源不足可缩小模型和任务数，不能省略对照与 hidden set。

## 必报指标

task success、partial score、reward、tokens、steps、wall time、invalid action、tool error recovery、integrity violation、KL 和每 100 个成功任务的总成本。报告 per-family 和 overall，避免简单族掩盖难任务。

## 红队

在隐藏副本上加入 task-ID shortcut、可修改 metric 文件、诱导泄漏文档、超时漏洞和错误但高 judge 风格答案。比较 base/SFT/RL 的攻击利用率，分析 RL 是否放大了投机。

## 项目通过条件

- 训练和评测从 manifest 可复现。
- RL 相对 SFT 的结论来自 hidden，而非训练 reward。
- 报告至少 30 条版本差异轨迹和一个策略退化 checkpoint。
- 未以更高 hack rate、更多预算或任务泄漏换取能力分。

提交 model/data cards、配置、曲线、消融、红队和限制。下一课：[[agent-rsi-30d/week4/day22-literature-hypothesis]]。

