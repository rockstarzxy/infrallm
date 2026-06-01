---
title: "Day 9: Scheduler 深入"
type: concept
tags: [day9, scheduler, vllm, preemption, priority, chunked-prefill]
created: 2026-06-01
updated: 2026-06-01
---

# Day 9: Scheduler 深入

## Part 1: Scheduler 的职责

每次 `LLMEngine.step()` 调用时，Scheduler 需要回答一个核心问题：**本次 iteration 让哪些请求上 GPU？**

这个决策要平衡多个约束：
1. GPU 显存限制：KV Cache block 是有限的
2. 计算预算：max_num_batched_tokens 限制单次 iteration 的计算量
3. 并发上限：max_num_seqs 限制同时处理的请求数
4. 公平性：不能让某些请求永远等待
5. 效率：尽量让 GPU 满载

---

## Part 2: 三个队列

Scheduler 管理三个队列：

```
┌─────────────┐     schedule()      ┌─────────────┐
│   WAITING   │ ──────────────────→  │   RUNNING   │
│  等待首次    │   分配 KV block     │  正在执行    │
│  prefill    │ ←────────────────── │  prefill/   │
└─────────────┘   preemption        │  decode     │
                  (recompute)       └──────┬──────┘
                                          │
                                  block 不够
                                  需要腾空间
                                          │
                                          ↓
                                   ┌─────────────┐
                                   │   SWAPPED   │
                                   │  KV Cache   │
                                   │  已换到 CPU  │
                                   └─────────────┘
```

**WAITING**：新请求进入的地方，等待首次 prefill。
**RUNNING**：当前在 GPU 上有 KV Cache 的请求。可能正在做 prefill，也可能正在做 decode。
**SWAPPED**：KV Cache 被移到 CPU 内存的请求。GPU 上没有它的数据，需要 swap in 后才能继续。

---

## Part 3: schedule() 的决策逻辑

### 简化的调度流程

```python
def schedule(self):
    # 阶段 1: 处理已在运行的 decode 请求
    # 检查 RUNNING 队列中的请求是否还有 block 可用
    running_scheduled = []
    for seq_group in self.running:
        if can_allocate_block(seq_group):
            running_scheduled.append(seq_group)
        else:
            # block 不够了 → preemption
            self.preempt(seq_group)  # 移到 swapped 或 recompute

    # 阶段 2: 尝试 swap in 之前被驱逐的请求
    for seq_group in self.swapped:
        if enough_blocks_for_swap_in(seq_group):
            swap_in(seq_group)
            running_scheduled.append(seq_group)

    # 阶段 3: 从 WAITING 队列调入新请求
    for seq_group in self.waiting:
        if (len(running_scheduled) < max_num_seqs and
            total_tokens < max_num_batched_tokens and
            can_allocate_blocks(seq_group)):
            allocate_blocks(seq_group)
            running_scheduled.append(seq_group)

    return SchedulerOutput(
        scheduled_seq_groups=running_scheduled,
        blocks_to_swap_in=...,
        blocks_to_swap_out=...,
        blocks_to_copy=...,
    )
```

### 调度优先级

默认顺序：**RUNNING > SWAPPED > WAITING**

这个顺序的逻辑：
1. **RUNNING 优先**：已经在 GPU 上有 KV Cache 的请求应该优先完成，否则之前的计算白做了
2. **SWAPPED 次之**：已经做过 prefill 的请求，swap 回来比重做便宜
3. **WAITING 最后**：新请求还没有投入任何计算，等待成本最低

### Preemption 策略

当 RUNNING 请求继续 decode 需要新 block，但 block 已经不够了：

**策略 1: Swap（默认）**
```
1. 选择优先级最低的 RUNNING 请求
2. 将其 KV Cache 从 GPU 复制到 CPU 内存
3. 释放 GPU block
4. 请求移入 SWAPPED 队列
5. 之后有空闲 block 时再 swap 回来继续
```

代价：GPU ↔ CPU 数据复制有延迟，且占用 CPU 内存和 PCIe 带宽。

**策略 2: Recompute**
```
1. 选择优先级最低的 RUNNING 请求
2. 直接丢弃其 GPU 上的 KV Cache
3. 释放 GPU block
4. 请求移回 WAITING 队列
5. 之后需要重新做 prefill
```

代价：浪费之前的 prefill 计算。适合 CPU 内存不足或序列不长的情况。

### 被 preempt 的请求如何选择？

通常按 **FCFS 的逆序**（First-Come-First-Served 反过来）：最晚到达的请求先被驱逐。

逻辑：越早到达的请求已经做了更多计算，驱逐它的浪费更大。

---

## Part 4: Prefill 和 Decode 的混排

这是 scheduler 设计中最精细的部分之一。

### 问题：Prefill 会干扰 Decode

```
Step N:   [A_decode] [B_decode] [C_decode] [D_decode]     ← 纯 decode，每步 ~10ms
Step N+1: [A_decode] [B_decode] [C_decode] [E_PREFILL!!!] ← E 的 prefill 加入
          ↑ 这些 decode 请求要等 E 的 prefill 做完这个 iteration
          ↑ 如果 E 的 input=4096，这步可能要 100ms
          ↑ A/B/C/D 的这一步 TPOT 从 10ms 飙到 100ms → ITL 尖峰
```

### 解决方案 1：Chunked Prefill

将长 prompt 的 prefill 分成多个 chunk，每个 chunk 在一个 iteration 中和 decode 请求一起执行：

```
不用 chunked prefill:
Step N:   [A_d] [B_d] [C_d] [E_prefill_全部4096tokens]  ← iteration 很重

用 chunked prefill (chunk_size=512):
Step N:   [A_d] [B_d] [C_d] [E_prefill_chunk1_512tokens]
Step N+1: [A_d] [B_d] [C_d] [E_prefill_chunk2_512tokens]
...
Step N+7: [A_d] [B_d] [C_d] [E_prefill_chunk8_512tokens]
Step N+8: [A_d] [B_d] [C_d] [E_decode]   ← E 的 prefill 完成，开始 decode
```

效果：
- E 的 TTFT 略增（从 1 步变成 8 步完成 prefill）
- A/B/C/D 的 TPOT 稳定性大幅改善（每步计算量更均匀）
- 整体 ITL 方差降低

### 解决方案 2：Prefill/Decode 分离调度

一些系统（如 DistServe）将 prefill 和 decode 分配到不同的 GPU 上，彻底消除干扰。vLLM 的 disaggregated prefilling 也是这个思路（Day 24 详解）。

---

## Part 5: Priority Scheduling

在生产环境中，不同请求可能有不同优先级：

```
高优先级：VIP 用户、实时对话
低优先级：批量任务、内部测试
```

vLLM 支持通过 API 传入 priority 参数（或通过自定义 scheduler policy）。

优先级影响：
- WAITING 队列的排序：高优先级请求先 prefill
- Preemption 的选择：低优先级请求先被驱逐
- 资源分配：为高优先级请求预留一定比例的 block

---

## Part 6: 实验——观察调度行为

### 实验 1：长短 prompt 混合

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

async def send(prompt, label):
    import time
    t0 = time.time()
    resp = await client.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt=prompt,
        max_tokens=32,
        stream=False,
    )
    elapsed = time.time() - t0
    print(f"[{label}] TTFT+gen = {elapsed*1000:.0f}ms, "
          f"tokens={resp.usage.completion_tokens}")

async def main():
    # 先发 10 个短请求
    short_tasks = [send("Hello! " * 10, f"short-{i}") for i in range(10)]
    # 同时发 2 个长请求
    long_prompt = "Analyze this: " + "data " * 4000
    long_tasks = [send(long_prompt, f"long-{i}") for i in range(2)]

    await asyncio.gather(*short_tasks, *long_tasks)

asyncio.run(main())
```

观察：
- 短请求的延迟是否被长 prompt 的 prefill 影响？
- 如果开启 chunked prefill 后重跑，短请求的延迟变化如何？

### 实验 2：触发 preemption

```bash
# 启动一个 KV Cache 较小的 vLLM
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.5 \
  --max-model-len 4096 \
  --max-num-seqs 32

# 发大量并发请求，耗尽 KV Cache
python benchmarks/benchmark_serving.py \
  --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
  --endpoint /v1/completions --dataset-name random \
  --random-input-len 1024 --random-output-len 512 \
  --num-prompts 100 --request-rate inf

# 同时监控 preemption
watch -n 1 'curl -s http://localhost:8000/metrics | grep preemption'
```

观察：
- 何时开始出现 preemption？
- preemption 对 throughput 和 TTFT 的影响？

---

## Part 7: Scheduler 调优建议

| 场景 | 建议 |
|---|---|
| 短对话，低延迟要求 | max_num_seqs 适中（32-64），不开 chunked prefill |
| RAG（长 input），延迟敏感 | 开 prefix caching + chunked prefill |
| 长短混合 workload | 开 chunked prefill，保护 decode 请求的 TPOT |
| 批量离线生成 | max_num_seqs 设大，不关心 TTFT，最大化 throughput |
| 多租户 | 配置 priority scheduling，VIP 用户高优先级 |

---

## 交付物

| 文件 | 描述 |
|---|---|
| `scheduler-analysis.md` | Scheduler 状态机、preemption 机制、长短混合实验结果 |

## 自检问题

1. WAITING → RUNNING → SWAPPED 各代表什么状态？
2. 为什么 RUNNING 队列的请求优先级最高？
3. Preemption 的 Swap 和 Recompute 策略各适合什么场景？
4. Chunked prefill 如何解决"长 prefill 阻塞 decode"的问题？代价是什么？
5. 如果 preemption 频繁发生，说明什么问题？怎么解决？
