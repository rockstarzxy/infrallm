---
title: "Day 2：最小 Agent Runtime"
type: concept
tags: [agent, runtime, state-machine, react]
sources: [2026-09-27_agent-self-evolution-rsi-asi-course-references.md]
created: 2026-09-27
updated: 2026-09-27
---

# Day 2：最小 Agent Runtime

## 今日目标

不用大型框架，写出一个控制流完全可见的工具 Agent。重点是“状态如何改变”，不是做漂亮聊天界面。

## 1. 状态机

最小状态包含 `task`、`messages`、`step`、`budget`、`pending_action`、`artifacts`、`status`。合法状态转移为：

```text
INIT → MODEL_CALL → ACTION_VALIDATION → TOOL_EXECUTION
          ↑                              ↓
          └──────── OBSERVATION ←────────┘
MODEL_CALL → FINAL → VERIFY → DONE/FAILED
```

模型只能提出结构化动作；runtime 决定动作是否合法、是否执行、何时停止。把自然语言中的“我将运行测试”与真实工具事件分开。

## 2. 核心接口

```python
class Tool:
    name: str
    input_schema: dict
    def run(self, args, ctx) -> "ToolResult": ...

class AgentState:
    task: str
    messages: list
    step: int
    tokens_left: int
    status: str

def policy(state) -> Action: ...
def transition(state, action, observation) -> AgentState: ...
```

所有异常都转成显式 observation，例如 `INVALID_JSON`、`UNKNOWN_TOOL`、`TIMEOUT`，不能静默重试。设置最大步数、token 和 wall-clock 三重预算。

## 今日实验（7 小时）

1. 定义 `respond`、`call_tool`、`finish` 三种动作的 JSON Schema。
2. 实现模型适配器；保存原始 response 和解析后的 action。
3. 实现循环和三重预算；连续两次相同无效动作时终止。
4. 先接一个确定性计算器，完成 20 个多步算术任务。
5. 对比 single-shot、ReAct loop、best-of-3；固定总 token budget。
6. 注入非法 JSON、未知工具、工具异常、无限循环，确认状态机可控退出。

## 应记录的指标

任务成功、有效动作率、平均步骤、重复动作率、token、wall time、失败类型。不要用最终回答字符串是否相同判断成功，应由任务 verifier 判定。

## 验收与交付

- `runtime.py` 不依赖 Agent 框架，控制循环不超过约 200 行。
- 20 个任务均有最终状态和失败原因；异常不会导致未记录退出。
- 提交状态转移图、接口说明、三种 baseline 的同预算结果。

下一课：[[agent-rsi-30d/week1/day03-tools-environments-sandbox]]。

