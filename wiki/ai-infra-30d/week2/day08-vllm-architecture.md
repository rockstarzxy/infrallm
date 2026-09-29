---
title: "Day 8: vLLM V1 架构与请求生命周期（源码级）"
type: concept
tags: [day8, vllm, vllm-v1, architecture, source-code, engine-core, async-llm]
created: 2026-06-01
updated: 2026-09-27
---

# Day 8: vLLM V1 架构与请求生命周期（源码级）

> 本页按 vLLM V1 引擎（`vllm/v1/` 目录，0.10/0.11 时代的代码）编写。V1 从 0.8 开始成为默认引擎，V0 代码已在 2025 年下半年移除。网上大量教程仍在讲 V0 的 `SequenceGroup`、`core/block_manager.py`、`SWAPPED` 队列和 swap 抢占，这些在当前源码里已经不存在。读到这些词就要意识到资料过时。
>
> 文件路径会随版本漂移。先定位你安装版本的源码根目录，所有路径以它为准：
>
> ```bash
> VLLM_ROOT=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
> echo $VLLM_ROOT && ls $VLLM_ROOT/v1
> ```

## 今日目标

1. 说清 vLLM V1 由哪几个进程组成、进程之间靠什么通信、为什么要这样拆。
2. 能带着真实文件路径讲完一个请求从 HTTP 到第一个 token 流回客户端的完整路径。
3. 知道 CPU 侧开销在哪里（tokenize、detokenize、序列化、调度），为什么 V1 比 V0 快的很大一部分来自 CPU 侧。
4. 动手：从源码安装、用 py-spy 看到 EngineCore 忙循环、给 vLLM 加一个自定义 Prometheus 指标。

---

## Part 1: 进程模型

V0 是"一个 Python 进程里跑 API server + 引擎 + 调度 + GPU worker"，GPU 等 CPU 的时间很长。V1 把它拆成三层进程：

```
┌──────────────────────────────────────────────────────────────────────┐
│ 进程 A: API Server（FastAPI / uvicorn）                              │
│   vllm/entrypoints/openai/api_server.py                              │
│   serving_chat.py / serving_completion.py  ← chat template、参数校验  │
│                                                                      │
│   AsyncLLM            vllm/v1/engine/async_llm.py                    │
│    ├─ Processor       vllm/v1/engine/processor.py   ← tokenize、构造  │
│    │                                                  EngineCoreRequest│
│    ├─ EngineCoreClient (AsyncMPClient)  vllm/v1/engine/core_client.py │
│    │       ↓ ZMQ socket（ipc://），msgspec/msgpack 序列化              │
│    └─ OutputProcessor vllm/v1/engine/output_processor.py             │
│                       ← detokenize、stop string、logprobs、stats      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ ZMQ（请求下行 / 输出上行，两条 socket）
┌───────────────────────────────▼──────────────────────────────────────┐
│ 进程 B: EngineCoreProc         vllm/v1/engine/core.py                 │
│   run_busy_loop():                                                   │
│     输入线程：从 socket 收请求 → 放入 input_queue                     │
│     主循环：  step() = scheduler.schedule()                           │
│                        → executor.execute_model(scheduler_output)    │
│                        → scheduler.update_from_output(...)           │
│     输出线程：把 EngineCoreOutputs 序列化后推回 socket                 │
│                                                                      │
│   Scheduler            vllm/v1/core/sched/scheduler.py   （Day 9）    │
│   KVCacheManager       vllm/v1/core/kv_cache_manager.py  （Day 10）   │
│   StructuredOutputManager  vllm/v1/structured_output/    （Day 13）   │
│   Executor             vllm/v1/executor/{uniproc,multiproc,ray_*}.py │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ TP=1: 同进程直接调用
                                │ TP>1: MultiprocExecutor，每个 rank 一个进程，
                                │       共享内存 MessageQueue 广播 + ZMQ 返回
┌───────────────────────────────▼──────────────────────────────────────┐
│ 进程 C..: Worker（每个 GPU 一个）  vllm/v1/worker/gpu_worker.py        │
│   GPUModelRunner       vllm/v1/worker/gpu_model_runner.py （Day 13）  │
│    ├─ InputBatch（persistent batch）  gpu_input_batch.py             │
│    ├─ attention backend    vllm/v1/attention/backends/               │
│    ├─ model forward（torch.compile + piecewise CUDA graph）            │
│    ├─ Sampler              vllm/v1/sample/sampler.py                 │
│    └─ drafter（spec decode）vllm/v1/spec_decode/                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 为什么这样拆

- **API server 和 EngineCore 分进程**：HTTP 解析、tokenize、detokenize、JSON 序列化全是 Python CPU 工作。放在同一个进程里，GIL 会让它们和调度循环互相抢。拆开后 EngineCore 的循环只做"调度 + 发 GPU 任务 + 处理采样结果"，GPU 空转时间明显下降。
- **EngineCore 和 Worker 分进程**：TP>1 时每个 rank 必须是独立进程（NCCL 要求）。TP=1 时 `UniProcExecutor` 直接在 EngineCore 进程里调用 worker，少一次进程间通信。
- **多 API server**：`--api-server-count N` 可以起多个 API server 进程共用一个 EngineCore，用来扛住高 QPS 下的前端 CPU 瓶颈。
- **数据并行**：`--data-parallel-size N` 会起 N 个 EngineCore，由 `DPCoordinator`（`vllm/v1/engine/coordinator.py`）协调。MoE 模型下各 DP rank 必须同步 step（没请求的 rank 跑 dummy batch），否则 expert all-to-all 会死锁。Day 18 再展开。

### 调试开关

```bash
# 把 EngineCore 放回 API server 进程（InprocClient），方便单步调试和打断点
VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve Qwen/Qwen2.5-7B-Instruct

# 日志级别
VLLM_LOGGING_LEVEL=DEBUG vllm serve ...
```

---

## Part 2: 请求生命周期（带文件路径）

以一次 `POST /v1/chat/completions`（stream=true）为例，逐步跟：

### 阶段 1: HTTP 入口（进程 A）

1. `api_server.py` 的路由 → `OpenAIServingChat.create_chat_completion()`（`entrypoints/openai/serving_chat.py`）。
2. 应用 chat template（HF tokenizer 的 `apply_chat_template`），得到 prompt 字符串。工具调用（tool calling）的解析、reasoning 内容的拆分也在这层。
3. 构造 `SamplingParams`（`vllm/sampling_params.py`），校验 `max_tokens`、`logprobs`、结构化输出参数。
4. 调用 `AsyncLLM.generate(prompt, sampling_params, request_id)`。

### 阶段 2: AsyncLLM 前端处理（进程 A）

`AsyncLLM.add_request()` → `Processor.process_inputs()`（`v1/engine/processor.py`）：

- tokenize（HF tokenizer，纯 CPU）
- 多模态输入的预处理和 hash（`mm_hashes`，供 encoder cache 和 prefix cache 用）
- 生成 block hash 链的初始信息（Day 10 讲）
- 结构化输出请求：校验 grammar 参数
- 打包成 `EngineCoreRequest`（`v1/engine/__init__.py`，一个 msgspec Struct：`request_id`、`prompt_token_ids`、`sampling_params`、`mm_*`、`lora_request`、`arrival_time`、`priority`、`cache_salt` 等）

然后 `OutputProcessor.add_request()` 登记一个 `RequestState`（保存 detokenizer 增量状态、已生成 token、stats），并创建 asyncio queue 供 HTTP handler 等待。

最后 `EngineCoreClient.add_request_async()` 把 `EngineCoreRequest` 用 msgspec 编码后通过 ZMQ 发给 EngineCore。序列化在 `v1/serial_utils.py`，大 tensor（比如多模态 embedding）走零拷贝 buffer 而不是 pickle。

### 阶段 3: EngineCore 忙循环（进程 B）

`EngineCoreProc.run_busy_loop()`（`v1/engine/core.py`）：

```python
# 简化，保留真实结构
def run_busy_loop(self):
    while True:
        self._process_input_queue()     # 输入线程收到的 add/abort/utility 请求
        self._process_engine_step()     # = self.step()

def step(self):
    if not self.scheduler.has_requests():
        return {}, False
    scheduler_output = self.scheduler.schedule()                       # Day 9
    model_output = self.model_executor.execute_model(scheduler_output) # Worker
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output)                                # 追加 token、判停、释放 KV
    return engine_core_outputs
```

`add_request` 在 EngineCore 侧做的事：把 `EngineCoreRequest` 转成 `Request`（`v1/request.py`），状态 `WAITING`，进 `scheduler.add_request()`。如果是结构化输出请求，状态先是 `WAITING_FOR_FSM`，等 grammar 在线程池里编译完才可调度。

`Request` 的状态枚举（`RequestStatus`）：

```
WAITING → WAITING_FOR_FSM / WAITING_FOR_REMOTE_KVS（可选中间态）
        → RUNNING
        → PREEMPTED（被抢占，回到 waiting 队列头部，num_computed_tokens 清零）
        → FINISHED_STOPPED / FINISHED_LENGTH_CAPPED / FINISHED_ABORTED / FINISHED_IGNORED
```

注意：没有 `SWAPPED`。V1 抢占只有 recompute 一种，Day 9 讲原因。

### 阶段 4: Worker 执行（进程 C）

`Executor.execute_model(scheduler_output)` → `Worker.execute_model()` → `GPUModelRunner.execute_model()`（`v1/worker/gpu_model_runner.py`）：

1. `_update_states(scheduler_output)`：把 SchedulerOutput 里的 `NewRequestData` / `CachedRequestData` 应用到 **persistent batch**（`InputBatch`）。只处理增量：新请求加一行，完成的请求删一行，其余行只更新 `num_computed_tokens` 和 block table。
2. `_prepare_inputs()`：拼出这一步的 `input_ids`、`positions`、`query_start_loc`、`seq_lens`、`slot_mapping`，构造 attention metadata。
3. 模型 forward（在 `set_forward_context` 下，按 token 数选择对应的 CUDA graph）。
4. `compute_logits` → `Sampler.forward(logits, sampling_metadata)`：logits processor、penalties、temperature、top-k/top-p、采样、logprobs。
5. 如果开了 spec decode：drafter 提出下一批 draft token。
6. 返回 `ModelRunnerOutput`（`sampled_token_ids`、`logprobs`、`prompt_logprobs`、spec token）。TP>1 时只有 rank 0 返回。

Day 13 逐段读这个文件。

### 阶段 5: 回到 EngineCore，再回到前端

`scheduler.update_from_output()`：把新 token 追加到 `Request`，检查 EOS / `max_tokens` / stop token ids（stop **string** 不在这里，在前端 detokenize 之后判）；完成的请求释放 KV block；结构化输出请求推进 FSM；生成 `EngineCoreOutputs`（每个请求的新 token ids、finish reason、stats）。

输出线程序列化后推回 ZMQ。进程 A 的 `AsyncLLM.output_handler()` 协程收到后调 `OutputProcessor.process_outputs()`：

- 增量 detokenize（`v1/engine/detokenizer.py`，处理 BPE 拼接和不完整 UTF-8）
- 判 stop string，命中则截断并向 EngineCore 发 abort
- 组装 `RequestOutput`，放进该请求的 asyncio queue
- 更新 `IterationStats` / `RequestStateStats`，喂给 Prometheus logger（`v1/metrics/`）

HTTP handler 从 queue 取到 `RequestOutput`，序列化成 SSE chunk 写回。这就是客户端看到的第一个 token。TTFT 覆盖了以上 5 个阶段加上排队时间。

### 一句话版本

```
HTTP → chat template → tokenize → EngineCoreRequest --ZMQ--> Request(WAITING)
  → schedule() → execute_model() → update_from_output() --ZMQ--> detokenize → SSE
```

---

## Part 3: CPU 开销在哪里

V1 的设计目标之一是把 GPU 每步之间的 CPU 空隙压到最小。把开销拆开看：

| 开销 | 位置 | 进程 | 怎么观察 |
|---|---|---|---|
| chat template + tokenize | Processor | A | py-spy 看 `apply_chat_template` / `encode` 占比 |
| 请求序列化 / 反序列化 | serial_utils | A/B | 大多模态 embedding 时明显 |
| schedule() | Scheduler | B | 请求数很多（几百）时 Python 循环开销可见 |
| _update_states / _prepare_inputs | GPUModelRunner | C | py-spy 或 torch profiler 看 CPU 段 |
| kernel launch | 模型 forward | C | 不开 CUDA graph 时每层几十次 launch |
| sampling 的 CPU 部分 | Sampler / InputBatch 更新 | C | 混合采样参数（有的 greedy 有的 top-p）时更贵 |
| detokenize + stop string | OutputProcessor | A | 高并发流式输出时 A 进程 CPU 打满 |

V1 的对策：persistent batch（不重建 tensor）、piecewise CUDA graph（消 launch 开销）、async scheduling（`--async-scheduling`，让 step N+1 的调度和 step N 的 GPU 执行重叠，Day 9 讲）、多 API server 进程、前端和引擎分离。

实践判断：如果 `nvidia-smi` 的 SM 利用率在高并发下仍然只有 50% 到 60%，而 EngineCore 进程 CPU 接近 100%，瓶颈就在 CPU 侧，加 GPU 没用。

---

## Part 4: Executor 与分布式接线

| Executor | 文件 | 什么时候用 |
|---|---|---|
| `UniProcExecutor` | `v1/executor/uniproc_executor.py` | TP=1，单 GPU |
| `MultiprocExecutor` | `v1/executor/multiproc_executor.py` | 单机 TP/PP>1（默认） |
| `RayDistributedExecutor` | `v1/executor/ray_distributed_executor.py` | 多节点，或显式 `--distributed-executor-backend ray` |

`MultiprocExecutor` 的关键机制：每个 rank 一个 `WorkerProc`，EngineCore 通过共享内存 `MessageQueue`（`vllm/distributed/device_communicators/shm_broadcast.py`）把 `SchedulerOutput` 一次广播给所有 rank，避免 N 份序列化。rank 之间的 NCCL 通信在模型 forward 内部（Day 16）。

`collective_rpc(method, args)` 是 EngineCore 调用所有 worker 任意方法的通道，`/reset_prefix_cache`、profile 启停、LoRA 加载、sleep/wake_up 都走它。写自定义功能时优先复用这个通道而不是自己开 socket。

---

## Part 5: 动手实验

### 实验 1: 从源码安装并定位关键文件

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
# 只改 Python 不重编 kernel 的安装方式（用预编译 wheel 的 CUDA 部分）
VLLM_USE_PRECOMPILED=1 pip install -e .

VLLM_ROOT=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
grep -n "def run_busy_loop\|def step" $VLLM_ROOT/v1/engine/core.py
grep -n "class Scheduler" $VLLM_ROOT/v1/core/sched/scheduler.py
grep -n "class GPUModelRunner\|def execute_model" $VLLM_ROOT/v1/worker/gpu_model_runner.py
grep -n "class RequestStatus" -A 15 $VLLM_ROOT/v1/request.py
```

交付：一张表，列出本页提到的每个类在你版本里的真实路径和行号。路径对不上就说明版本变了，记录差异。

### 实验 2: 用 py-spy 看三层进程

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 &
# 找到进程树：api server、EngineCore、（TP>1 时）worker
pstree -p $(pgrep -f "vllm serve" | head -1)   # macOS 用 ps -ef | grep vllm
# 压测时分别抓
pip install py-spy
py-spy dump --pid <EngineCore pid>
py-spy record -o core.svg --pid <EngineCore pid> --duration 30 &
vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --dataset-name random \
  --random-input-len 512 --random-output-len 256 --num-prompts 300 --request-rate 20
```

回答：EngineCore 的时间花在 `schedule`、`execute_model`、`update_from_output` 各占多少？API server 进程里 detokenize 占多少？把 `--request-rate` 翻倍后哪个进程先到 100% CPU？

### 实验 3: 加一个自定义 Prometheus 指标

目标：暴露"每次 step 调度到的请求数"直方图 `vllm:num_scheduled_reqs_per_step`。

步骤：
1. 在 `v1/metrics/stats.py` 的 `SchedulerStats` 里加字段 `num_scheduled_reqs: int = 0`。
2. 在 `v1/core/sched/scheduler.py` 的 `make_stats()`（或 `schedule()` 末尾构造 stats 的地方）填入 `len(scheduler_output.scheduled_new_reqs) + len(cached_reqs)`。
3. 在 `v1/metrics/loggers.py` 的 `PrometheusStatLogger` 里注册一个 `Histogram`，在 `record()` 里 `observe`。
4. 重启后 `curl localhost:8000/metrics | grep num_scheduled_reqs`。

交付：diff 和一张压测期间该直方图的分布截图或文本。做完你就知道指标从哪来、经过哪几层。

### 实验 4: 单进程断点跟踪一个请求

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 python -c "
from vllm import LLM, SamplingParams
llm = LLM('Qwen/Qwen2.5-0.5B-Instruct', enforce_eager=True)
import pdb; pdb.set_trace()
print(llm.generate(['hello'], SamplingParams(max_tokens=8)))
"
```

在 `Scheduler.schedule`、`GPUModelRunner.execute_model`、`Scheduler.update_from_output` 各下一个断点，走完一个请求的 prefill 步和前两个 decode 步，记录每一步 `request.num_computed_tokens` 和 `num_new_tokens` 的值。这是 Day 9 的预习。

---

## 交付物

| 文件 | 描述 |
|---|---|
| `vllm-v1-request-lifecycle.md` | 带真实路径和行号的请求生命周期图、三层进程职责表、CPU 开销分布表 |
| `custom-metric.diff` | 实验 3 的补丁和验证输出 |

## 自检问题

1. V1 为什么要把 EngineCore 放到独立进程？如果放回同一进程，哪类 workload 最先变慢？
2. stop token id 和 stop string 分别在哪个进程、哪一步判定？为什么要分开？
3. `EngineCoreRequest` 和 `Request` 的区别是什么？各自活在哪个进程？
4. TP=4 时 `SchedulerOutput` 如何到达 4 个 worker？为什么不用 ZMQ 发 4 次？
5. 一个请求被 abort（客户端断连）时，从 HTTP handler 到 KV block 释放经过哪几步？
6. V0 的 `SWAPPED` 状态在 V1 里去哪了？（Day 9 揭晓，先自己猜）
