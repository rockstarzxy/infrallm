---
title: "Day 9: vLLM V1 Scheduler 逐段精读"
type: concept
tags: [day9, scheduler, vllm, vllm-v1, preemption, priority, chunked-prefill, async-scheduling]
created: 2026-06-01
updated: 2026-09-27
---

# Day 9: vLLM V1 Scheduler 逐段精读

> 文件：`vllm/v1/core/sched/scheduler.py`（`Scheduler` 类），配套 `sched/output.py`（`SchedulerOutput`、`NewRequestData`、`CachedRequestData`）、`sched/request_queue.py`（FCFS / priority 队列）、`sched/interface.py`（`SchedulerInterface`，自定义调度器要实现的接口）。
>
> V0 的 scheduler 有 `_schedule_prefills` / `_schedule_running` / `_schedule_swapped` 三段逻辑和 WAITING/RUNNING/SWAPPED 三个队列。V1 把这些全部删掉，换成一个统一的 token budget 循环。理解这一设计变化，比背旧的三队列图有价值得多。

## 今日目标

1. 读懂 `schedule()` 的完整控制流，能徒手写出简化版。
2. 说清 V1 为什么不区分 prefill 和 decode，chunked prefill 为什么"天然存在"。
3. 说清 preemption 只剩 recompute 的原因，以及 prefix cache 如何降低 recompute 代价。
4. 理解 priority scheduling、async scheduling、spec decode 和结构化输出各自在调度器里的接入点。
5. 动手：改一行代码改变抢占顺序，用实验验证；实现一个自定义 `--scheduler-cls`。

---

## Part 1: 核心抽象：num_computed_tokens

V1 里每个 `Request` 只有两个关键计数：

```
request.num_tokens           # prompt token 数 + 已生成 token 数
request.num_computed_tokens  # 已经算过 KV 的 token 数
```

调度器每一步对每个请求只问一个问题：**这一步给它算多少个新 token？**

```
num_new_tokens = request.num_tokens - request.num_computed_tokens
```

- 新请求：`num_computed_tokens = 0`（或 prefix cache 命中的长度），`num_new_tokens` 是整段 prompt，这就是 prefill。
- 老请求（decode）：上一步生成了 1 个 token，`num_tokens` 加 1，`num_new_tokens = 1`。
- 长 prompt 被 token budget 截断：`num_new_tokens = min(剩余, budget)`，这就是 chunked prefill。下一步接着算剩下的。
- spec decode：`num_new_tokens = 1 + 上一步的 draft token 数`（要验证的 token 一起算）。

所以 V1 里 **prefill、decode、chunked prefill、spec decode 验证是同一段代码路径**，区别只是 `num_new_tokens` 的值。这也是为什么 V1 里 chunked prefill 是默认行为而不是开关：调度器根本没有"不可分割的 prefill"这个概念。`--max-num-batched-tokens` 就是每步的 token budget。

---

## Part 2: schedule() 控制流

以下是按真实结构简化的伪代码。读的时候对照源码，把每一段的行号记下来。

```python
def schedule(self) -> SchedulerOutput:
    token_budget = self.max_num_scheduled_tokens        # = max_num_batched_tokens
    scheduled_new_reqs, scheduled_running_reqs, preempted_reqs = [], [], []
    req_to_new_blocks = {}

    # ---------- 第一段：先服务 RUNNING 队列 ----------
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
        num_new_tokens = min(num_new_tokens, token_budget)
        # long_prefill_token_threshold：限制单个长 prefill 每步最多吃多少 budget
        # encoder 输入（多模态）也要检查 encoder cache 预算

        while True:
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)
            if new_blocks is not None:
                break
            # ---- 分不到 KV block：抢占 ----
            if self.policy == SchedulingPolicy.PRIORITY:
                preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
                self.running.remove(preempted_req)
            else:
                preempted_req = self.running.pop()        # 最后加入 running 的先被踢
            self.kv_cache_manager.free(preempted_req)
            preempted_req.status = RequestStatus.PREEMPTED
            preempted_req.num_computed_tokens = 0         # 下次重新 prefill（recompute）
            self.waiting.prepend_request(preempted_req)   # 放回 waiting 队头
            preempted_reqs.append(preempted_req)
            if preempted_req == request:
                break                                     # 自己被踢了，退出
        if new_blocks is None:
            break                                          # 已无法为任何 running 请求分配

        scheduled_running_reqs.append(request)
        req_to_new_blocks[request.request_id] = new_blocks
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        req_index += 1
        # spec decode：记录本步要验证的 draft token；结构化输出：记录需要 bitmask 的请求

    # ---------- 第二段：如果这一步没发生抢占，才从 WAITING 拉新请求 ----------
    if not preempted_reqs:
        while self.waiting and token_budget > 0:
            if len(self.running) == self.max_num_running_reqs:   # = max_num_seqs
                break
            request = self.waiting.peek_request()

            # 结构化输出：grammar 还没编译好 → 跳过（WAITING_FOR_FSM）
            # KV connector：远端 KV 还在传 → 跳过（WAITING_FOR_REMOTE_KVS）

            # prefix cache 查询（Day 10）
            new_computed_blocks, num_new_local_computed_tokens = \
                self.kv_cache_manager.get_computed_blocks(request)
            # KV connector 可以额外报告"远端已有多少 token 的 KV"
            num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens

            num_new_tokens = request.num_tokens - num_computed_tokens
            if num_new_tokens == 0:
                # 整个 prompt 都命中缓存：退一个 token 出来，保证至少算 1 个（要拿 logits）
                num_new_tokens = 1; num_computed_tokens -= 1
            if not self.enable_chunked_prefill and num_new_tokens > token_budget:
                break                                     # 关了 chunked prefill 就整段等
            num_new_tokens = min(num_new_tokens, token_budget)

            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, num_new_local_computed_tokens, new_computed_blocks)
            if new_blocks is None:
                break                                     # KV 不够，本步不再拉新请求
            self.waiting.pop_request()
            self.running.append(request)
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
            scheduled_new_reqs.append(request)
            token_budget -= num_new_tokens

    # ---------- 第三段：打包输出 ----------
    # 新请求 → NewRequestData（带完整 prompt token ids、sampling params、block ids）
    # 老请求 → CachedRequestData（只带 request_id、新 token、新增 block ids）
    # 结构化输出 → grammar_bitmask（StructuredOutputManager 生成）
    # KV connector → connector 元数据
    # 释放上一步已完成请求的 KV（延迟到这里，是为了让 connector 有机会先保存）
    return SchedulerOutput(...)
```

### 读完必须能回答的三个问题

**为什么 running 优先，而且发生抢占后不再拉新请求？**
running 请求的 KV 已经在显存里，不让它们继续就等于浪费已算的 prefill；如果这一步都腾不出 block 给老请求，再拉新请求只会立刻触发下一轮抢占，造成抖动。

**RUNNING 队列本身的顺序是什么？**
`self.running` 是按加入顺序的 list。FCFS 下 `pop()` 踢的是最晚进 running 的请求。这就是"最晚到的先被抢占"，因为它已投入的计算最少。priority 模式下按 `(priority, arrival_time)` 挑数值最大的，也就是优先级最低、最晚到的。vLLM 里 **priority 数值越小优先级越高**，默认 0。

**`max_num_seqs` 和 `max_num_batched_tokens` 分别卡住哪一段？**
`max_num_seqs` 只在拉新请求时检查（第二段），限制 running 队列长度。`max_num_batched_tokens` 是两段共用的 budget。decode 请求每个只消耗 1 个 budget，所以 256 个 decode 请求只用 256 budget，剩下的 budget 全给 prefill 分片。这就是 chunked prefill 让 decode 和 prefill 混排的机制。

---

## Part 3: Preemption 只有 recompute

V0 有 swap（KV 搬到 CPU）和 recompute 两种。V1 删掉了 swap，原因：

1. **swap 的收益被 prefix cache 吃掉了**。被抢占的请求 `num_computed_tokens` 清零，但它的 KV block 只是回到 free 队列，没有被清空，hash 还在（Day 10）。只要没被别人挤掉，重新调度时 `get_computed_blocks()` 会命中这些 block，实际重算的只有最后一个不完整 block。代价接近于零拷贝的 swap-in。
2. **swap 需要 CPU 内存池、PCIe 拷贝、额外状态**，复杂度高，而且 PCIe 带宽在高并发时本身是稀缺资源。
3. **CPU offload 现在走 KV connector 做**（`OffloadingConnector` / LMCache），是可插拔的，不在调度器核心路径里。

抢占频率是最重要的健康指标之一：

```bash
curl -s localhost:8000/metrics | grep -E "num_preemptions|num_requests_(running|waiting)|kv_cache_usage"
```

`vllm:num_preemptions_total` 持续增长意味着 KV 不够：要么降 `max_num_seqs`，要么降 `max_model_len`，要么加显存（`gpu_memory_utilization`、KV 量化、更多卡）。

---

## Part 4: 调度策略与队列实现

`sched/request_queue.py`：

| policy | 队列类型 | 出队顺序 | 抢占顺序 |
|---|---|---|---|
| `fcfs`（默认） | deque | 到达顺序 | running 尾部 |
| `priority` | heap，key = `(priority, arrival_time)` | priority 小的先 | priority 大的先 |

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --scheduling-policy priority
```

```python
# 请求侧：OpenAI 兼容 API 的扩展字段
client.chat.completions.create(model=..., messages=..., extra_body={"priority": 0})   # VIP
client.chat.completions.create(model=..., messages=..., extra_body={"priority": 10})  # 批量任务
```

注意 priority 只影响 waiting 出队顺序和抢占选择，不会给高优先级请求预留 KV 或 budget。想做真正的租户隔离要在网关层限流（Day 18）。

### 长 prefill 相关参数

```
--max-num-partial-prefills N        # 同一步最多有几个请求处于"部分 prefill"状态
--max-long-partial-prefills N       # 其中最多几个是"长"请求
--long-prefill-token-threshold T    # 多长算"长"（默认按 max_model_len 比例）
```

用途：多个长 prompt 同时到达时，避免它们一起把 budget 吃光，让短请求饿死。

---

## Part 5: 调度器的四个扩展点

### 1. Async scheduling（`--async-scheduling`）

默认流程是串行的：`schedule()` → GPU 执行 → 等结果 → `update_from_output()` → 下一步 `schedule()`。GPU 执行期间 CPU 在等，CPU 调度期间 GPU 在等。

async scheduling 让调度器在 step N 的 GPU 结果回来之前就调度 step N+1：假设每个 decode 请求一定会生成 1 个 token（先占位），KV block 提前分配。结果回来后再补 token id、判停。代价是判停晚一步（可能多算一个 token），以及部分功能不兼容（历史上 spec decode、结构化输出、PP 都有过限制，以当前版本文档为准）。

效果：decode 主导的 workload 下 CPU 开销被隐藏，吞吐提升几个到十几个百分点。实验对比时记得固定其他参数。

### 2. Speculative decoding 接入

`--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4}'` 之后：

- 调度器为 running 请求预留 `num_lookahead_tokens` 个额外 slot（draft token 的 KV 也要占位）。
- `num_new_tokens` 包含待验证的 draft token。
- `update_from_output()` 里按 rejection sampler 的结果只接受前 k 个，被拒的 token 位置的 KV block 需要回退（`num_computed_tokens` 相应调整）。
- 指标：`vllm:spec_decode_num_draft_tokens_total`、`vllm:spec_decode_num_accepted_tokens_total`、按位置的接受率。Day 12 做实验。

### 3. 结构化输出接入

grammar 编译在 `StructuredOutputManager` 的线程池里异步做，请求状态 `WAITING_FOR_FSM`，编好前调度器跳过它。每一步 `schedule()` 末尾调用 `grammar_bitmask()` 为本步所有结构化输出请求生成 vocab 大小的 bitmask，随 `SchedulerOutput` 发给 worker，采样前 mask 掉非法 token。`update_from_output()` 里用接受的 token 推进 FSM。Day 13 讲 sampler 侧。

### 4. 自定义调度器（`--scheduler-cls`）

```bash
vllm serve ... --scheduler-cls my_pkg.my_sched.MyScheduler
```

`MyScheduler` 继承 `vllm.v1.core.sched.scheduler.Scheduler` 或实现 `SchedulerInterface`。常见用法：改抢占策略、按租户配额限流、给长请求打折、实验性的 SLO-aware 调度。这是把课程里"改源码"的能力落到生产可用形态的正规路径，不需要 patch vLLM 本体。

---

## Part 6: 动手实验

### 实验 1: 追踪一个请求的 num_computed_tokens

延续 Day 8 实验 4 的单进程模式，用 0.5B 模型、`max_num_batched_tokens=64`、发一个 200 token 的 prompt，`max_tokens=5`。在 `schedule()` 末尾打印每个请求的 `(request_id, num_computed_tokens, num_scheduled_tokens)`。

预期：前 4 步是 64/64/64/8 的 prefill 分片，之后每步 1。把这个表画出来，就是 chunked prefill 的真实时序。

### 实验 2: 长短混合，看 budget 的作用

```bash
# 服务端两组配置分别跑
vllm serve Qwen/Qwen2.5-7B-Instruct --max-num-batched-tokens 512
vllm serve Qwen/Qwen2.5-7B-Instruct --max-num-batched-tokens 8192
```

```python
import asyncio, time
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="x")

async def one(prompt, label, max_tokens):
    t0 = time.perf_counter(); first = None; n = 0
    stream = await client.completions.create(model="Qwen/Qwen2.5-7B-Instruct",
        prompt=prompt, max_tokens=max_tokens, stream=True, temperature=0)
    itl = []; last = t0
    async for ch in stream:
        now = time.perf_counter()
        if first is None: first = now - t0
        else: itl.append(now - last)
        last = now; n += 1
    print(f"{label:8s} TTFT={first*1000:6.0f}ms  ITL_p99={sorted(itl)[int(len(itl)*0.99)]*1000 if itl else 0:5.0f}ms")

async def main():
    shorts = [one("Hi " * 20, f"short-{i}", 128) for i in range(8)]
    await asyncio.sleep(0.5)          # 让短请求先进入 decode
    longs = [one("data " * 6000, f"long-{i}", 16) for i in range(2)]
    await asyncio.gather(*shorts, *longs)
asyncio.run(main())
```

对比两组配置下短请求的 ITL P99 和长请求的 TTFT。预期：budget 小时短请求 ITL 平稳、长请求 TTFT 高；budget 大时相反。这就是 Day 6 那条"取舍"背后的机制。

### 实验 3: 触发并观察抢占

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.5 --max-model-len 4096 --max-num-seqs 128
vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --num-prompts 200 --request-rate inf
watch -n1 'curl -s localhost:8000/metrics | grep -E "num_preemptions_total|num_requests_running|kv_cache_usage_perc"'
```

记录：抢占开始时 `kv_cache_usage_perc` 是多少？抢占后吞吐掉多少？把 `--max-num-seqs` 降到 32 再跑，抢占是否消失、吞吐是否反而更高？

### 实验 4: 改抢占顺序（改源码）

把 `self.running.pop()` 改成 `self.running.pop(0)`（踢最早的），重跑实验 3，对比 E2E 延迟分布和总吞吐。预期：吞吐下降，因为踢掉的是投入最多的请求。写一段分析说明为什么"最晚到的先被抢占"是对的。

### 实验 5: 自定义调度器

写一个 `TenantQuotaScheduler(Scheduler)`，重载 `schedule()`：在调用父类之前，把 waiting 队列里同一 `tenant`（用 `cache_salt` 或 request_id 前缀模拟）已经在 running 的请求数超过 8 的请求暂时跳过。用 `--scheduler-cls` 加载，发两个租户的请求验证配额生效。

---

## 交付物

| 文件 | 描述 |
|---|---|
| `scheduler-analysis.md` | `schedule()` 三段控制流的自己版本伪代码（带行号）、实验 1 的 num_computed_tokens 时序表、实验 2/3 的对比数据 |
| `preempt-order.diff` + 分析 | 实验 4 |
| `tenant_quota_scheduler.py` | 实验 5 |

## 自检问题

1. V1 里一个 8192 token 的 prompt 在 `max_num_batched_tokens=2048` 下需要几步完成 prefill？这几步里 decode 请求各分到多少 budget？
2. 为什么 V1 不需要 `--enable-chunked-prefill` 开关也能分片？什么情况下会显式关闭它？
3. 被抢占的请求重新调度时一定要从头 prefill 吗？什么条件下几乎不用重算？
4. priority scheduling 能保证高优先级请求的 TTFT 吗？为什么不能？
5. async scheduling 为什么会让请求多生成一个 token？这对 `max_tokens=1` 的请求意味着什么？
6. 如果你要实现"某租户最多占 30% KV cache"，应该改 `schedule()` 的哪一段？为什么不能只改 waiting 队列的排序？
