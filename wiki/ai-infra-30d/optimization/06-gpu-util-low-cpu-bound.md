---
title: "优化专项 06: GPU 利用率低与 CPU 瓶颈"
type: concept
tags: [ai-infra, inference-optimization, cpu-bound, cuda-graph, async-scheduling, profiling]
created: 2026-09-29
updated: 2026-09-29
---

# 优化专项 06: GPU 利用率低与 CPU 瓶颈

> 一句话：并发已经很高、KV 没满、却跑不满 GPU 时，瓶颈几乎都在 CPU 侧，这页教你把它定位到具体进程和具体函数，再按机制修。关联日课：[[day05-profiling]]、[[day08-vllm-architecture]]、[[day09-scheduler]]、[[day13-flash-attention]]、[[day18-data-parallel]]。

## 1. 症状与判定

先回忆 [[day08-vllm-architecture]] 的三层进程：API server（tokenize / detokenize / SSE）、EngineCore（schedule / update_from_output）、Worker（_prepare_inputs / forward / sampler）。CPU 瓶颈一定落在其中一个进程的某段 Python 代码上，判定就是找出是哪个。

| 观察组合 | 指向 |
|---|---|
| `nvidia-smi dmon -s u` 的 SM 利用率在高并发下只有 40% 到 60%，`vllm:kv_cache_usage_perc` 未满，`vllm:num_requests_waiting` 接近 0 | 不是显存也不是排队，是每步之间 GPU 在等 CPU |
| EngineCore 进程（或 TP=1 时合并的进程）CPU 接近一个核的 100% | 调度或 model runner 的 Python 开销 |
| API server 进程 CPU 接近 100%，EngineCore 反而不满 | 前端进程：tokenize、detokenize、stop string、SSE 序列化 |
| nsys 时间线上每个 forward 之间有几毫秒到十几毫秒的空白 | kernel launch 或 CPU 准备输入的空隙 |
| decode 步的 GPU 时间很短但步间隔长，prefill 步反而饱满 | 典型的 launch 开销或调度开销，decode 每步计算太少盖不住 CPU |
| 并发从 64 提到 256，吞吐几乎不涨，TPOT 线性变差 | 每步固定 CPU 开销随请求数增长 |
| TP>1 时 rank 0 CPU 高、其他 rank GPU 在等 | rank 0 准备输入或广播的时间 |

三个必备工具（[[day05-profiling]]）：

```bash
nvidia-smi dmon -s u -d 1                      # SM 与显存带宽利用率逐秒
py-spy dump --pid <pid>                        # 某一瞬间各线程在哪个函数
py-spy record -o out.svg --pid <pid> -d 30     # 30 秒火焰图
nsys profile -t cuda,nvtx,osrt -o run --capture-range=cudaProfilerApi ...   # 配合 /start_profile /stop_profile 或 VLLM_TORCH_PROFILER_DIR
```

判定顺序：先看哪个进程 CPU 高，再对那个进程做火焰图，找到占比最大的 Python 帧，对照下面的根因表。

## 2. 根因逐个排查

### 2.1 kernel launch 开销：eager 或 CUDA graph 桶没覆盖到

**确认**：nsys 里 decode 步有几百个很短的 kernel，launch 间隔大于 kernel 本身；或启动日志显示 `--enforce-eager`；或本步 token 数超过最大 graph 桶（默认最大桶通常在几百，超过就退化到 eager，版本不同）。

**修**：生产不用 `--enforce-eager`。调 `--compilation-config` 的 `cudagraph_capture_sizes` 让桶覆盖常见 batch 大小；decode 主导时尝试 `cudagraph_mode` 为 `FULL_AND_PIECEWISE`（需 attention backend 支持，见 [[day13-flash-attention]]）。

**副作用**：桶多则录制时间和显存增加；FULL 模式对 backend 有要求，换 backend 要重新验证。

### 2.2 schedule() 的 Python 开销

**确认**：EngineCore 火焰图里 `Scheduler.schedule` 和 `update_from_output` 占比超过 forward 的 CPU 部分。几百并发时这两段是纯 Python 循环，每步都要遍历 running 队列。

**修**：`--async-scheduling` 让 step N+1 的调度和 step N 的 GPU 执行重叠（[[day09-scheduler]]）。降低 `max_num_seqs` 到吞吐不再增长的拐点，多余的并发只增加调度开销。

**副作用**：async scheduling 与某些特性的兼容性随版本变化（spec decode、结构化输出、PP 都出现过限制），先看当前版本文档。

### 2.3 _prepare_inputs 与 InputBatch 更新

**确认**：Worker 侧火焰图里 `_prepare_inputs`、`_update_states`、`condense` 占比高。请求进出频繁（短输出、高 QPS）时最明显。

**修**：这部分已经是 persistent batch（[[day13-flash-attention]]），能做的是减少请求进出频率：合并短请求、避免 `max_tokens=1` 的探测型请求。混合采样参数会让 `SamplingMetadata` 每步重建，统一采样参数能省一截。

### 2.4 前端进程：tokenize、chat template、detokenize、stop string

**确认**：API server 进程 CPU 满，火焰图里 `apply_chat_template`、`encode`、`detokenize_incrementally`、stop string 匹配占比大。长 prompt 高 QPS 或多 tool schema 的 agent 流量最容易撞上。

**修**：`--api-server-count N` 起多个前端进程分摊。请求侧尽量传 token ids 而不是文本（网关预先 tokenize），或至少缓存 chat template 渲染结果。流式响应减少 `stop` 字符串数量，stop token id 在 EngineCore 判、stop string 在前端判，后者贵得多。

**副作用**：多前端进程各自有 tokenizer 内存；网关侧 tokenize 要保证 tokenizer 版本与服务一致。

### 2.5 logprobs 与 prompt_logprobs

**确认**：请求带 `logprobs` 或 `prompt_logprobs`，sampler 返回的 tensor 从 [num_reqs] 变成 [num_reqs, k]，`prompt_logprobs` 更是整段 prompt 的 vocab 维输出，序列化和 detokenize 成本都上去。火焰图里 `LogprobsProcessor` 或 logprobs 相关帧明显。

**修**：默认不返回；只对需要的评测流量开。`prompt_logprobs` 和 prefix caching 会互相影响（有缓存的 token 没有 logits），版本不同处理不同。

### 2.6 结构化输出 bitmask

**确认**：请求带 JSON schema 或 grammar，EngineCore 火焰图里 `grammar_bitmask`、FSM 推进占比高。每步要为每个结构化请求生成一份 vocab 大小的 bitmask。

**修**：见 [[09-structured-output-tool-calls]]：简化 schema、换 backend、用 JSON mode 替代完整 schema。

### 2.7 多模态预处理

**确认**：前端进程里图像解码、resize、processor 调用占比高；或 encoder cache 频繁 miss。

**修**：客户端预处理成模型要求的分辨率；`--mm-processor-kwargs` 限制最大像素；利用多模态 hash 让相同图片命中 encoder cache 和 prefix cache。

### 2.8 大对象序列化

**确认**：`serial_utils` 或 msgspec 编解码在火焰图里可见。多模态 embedding、超长 prompt、大批 logprobs 都会放大。

**修**：减少跨进程传输量（同上几条）；多模态 embedding 走零拷贝路径（V1 已支持，确认版本）。

### 2.9 SSE 写回

**确认**：前端进程里 uvicorn / starlette 的 write 占比高，客户端多且每 token 一个 chunk。

**修**：`--api-server-count`；客户端合并 chunk 的需求可以在网关做；确认没有开调试级别日志（每 chunk 打日志是常见坑）。

### 2.10 Python GC

**确认**：火焰图里 `gc.collect` 周期性出现，对应 ITL 周期性尖峰。

**修**：vLLM 在关键路径上已经 `gc.freeze` 或调整阈值，版本不同做法不同。可以试 `PYTHONGC` 相关环境变量或在启动脚本里调高 `gc.set_threshold`，先量再改。

### 2.11 单 API server 进程

**确认**：QPS 上去后前端进程一个核跑满，EngineCore 还有余量。

**修**：`--api-server-count`；再不够就 DP 多 EngineCore（[[day18-data-parallel]]），或者多实例加网关。

### 2.13 日志与指标采集本身

**确认**：`--disable-log-stats` 关掉后吞吐上升，或日志级别是 DEBUG，或每请求打完整 prompt 日志（`--enable-log-requests` 一类开关，名称随版本变化）。

**修**：生产用 INFO 以上，关掉逐请求日志，Prometheus 采集间隔不低于 5 秒。

### 2.12 TP 下 rank 同步等待

**确认**：nsys 多 rank 视图里非 0 rank 在 NCCL 等待，rank 0 在 CPU 准备输入。

**修**：TP 不是越大越好，单卡放得下的模型不要为了"更快"开 TP；CPU 亲和把每个 rank 进程绑到对应 NUMA 节点（`numactl --cpunodebind`），避免跨 socket 内存访问。

## 3. 决策表

| 症状组合 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| EngineCore 满、decode 主导、`--enforce-eager` 在 | 去掉 eager，调 graph 桶 | `FULL_AND_PIECEWISE` | 加并发 |
| EngineCore 满、火焰图 schedule 占大头 | `--async-scheduling` | 降 `max_num_seqs` 到拐点 | 开更多可选特性 |
| API server 满、EngineCore 有余 | `--api-server-count` | 网关预 tokenize、减 stop 字符串 | 给 GPU 加卡 |
| 两者都不满、SM 仍低、KV 未满 | 看 nsys 步间空隙：launch 或 rank 同步 | 检查 logprobs / 结构化输出比例 | 猜 |
| 并发上去吞吐不涨 | 找每步固定开销（2.2 / 2.3） | DP 拆成多 EngineCore | 继续加并发 |
| 周期性 ITL 尖峰 | 查 GC 与 graph 桶边界 | 查日志级别 | 归咎网络 |

## 4. 验证实验

固定：模型、dtype、`max_model_len`、输入输出长度分布、请求速率模式（用 `--request-rate` 固定速率而不是 `inf`），warmup 60 秒。

```bash
# 基线
vllm serve Qwen/Qwen2.5-7B-Instruct --max-num-seqs 256
vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --dataset-name random \
  --random-input-len 256 --random-output-len 256 --num-prompts 2000 --request-rate 40

# 变量 1：eager 对照，预期 TPOT 明显变差、SM 利用率下降
vllm serve ... --enforce-eager

# 变量 2：async scheduling，预期高并发下 output tokens/s 提升、EngineCore CPU 下降
vllm serve ... --async-scheduling

# 变量 3：多前端进程，预期只在 API server 原本打满时有效
vllm serve ... --api-server-count 4

# 变量 4：并发扫描，找拐点
for n in 32 64 128 256 512; do vllm serve ... --max-num-seqs $n; done
```

每组同时记录：output tokens/s、TPOT P50/P99、SM 利用率均值、EngineCore 和 API server 进程的 CPU 均值（`pidstat -p <pid> 1`）。判定标准是"哪个变量让 SM 利用率和吞吐同步上升"，只涨吞吐不涨 SM 的通常是测量噪声。

### 排查 checklist（按顺序做，每步 5 分钟内）

1. `nvidia-smi dmon -s u` 看 SM 利用率；同时 `curl /metrics` 看 `kv_cache_usage_perc` 和 `num_requests_waiting`。SM 低且 KV 未满才继续，否则去 [[05-oom-preemption]] 或 [[04-throughput-low]]。
2. `ps -eo pid,pcpu,comm | grep -i vllm` 或 `pidstat` 找出哪个进程 CPU 高。进程树见 [[day08-vllm-architecture]] 实验 2。
3. 对该进程 `py-spy dump` 三次，看主线程反复停在哪个函数；再 `py-spy record` 30 秒出火焰图。
4. 火焰图最宽的帧对照 2.1 到 2.12。同时出现多个时先修排在前面的。
5. 改一个变量，重跑同一 workload，记录 SM 利用率和吞吐是否同步变化。没同步变化就回到第 3 步，说明找错了。
6. 检查启动参数里有没有 `--enforce-eager`、调试日志级别、不必要的 `logprobs`，这三项是最常见的"自己造成的" CPU 瓶颈。

### 常见误判

- SM 利用率高不等于 GPU 有效利用：小 batch 下 memory-bound 的 decode 也能让 SM 显示 90%。这页解决的是 SM 明显偏低的情况，decode 带宽瓶颈去 [[04-throughput-low]] 和 [[day26-roofline]]。
- `nvidia-smi` 首页的 GPU-Util 是粗粒度采样，只要采样窗口内有 kernel 在跑就算 100%。要用 `dmon -s u` 的 SM 列或 DCGM 的 `DCGM_FI_PROF_SM_ACTIVE`。
- API server 进程 CPU 高但吞吐正常，可能只是流式输出正常工作。只有当它成为吞吐上限时才需要处理。

## 5. 关联

- 总入口：[[01-diagnosis-playbook]]
- 表现为延迟：[[02-ttft-high]]、[[03-tpot-itl-jitter]]
- 表现为吞吐：[[04-throughput-low]]
- 结构化输出专项：[[09-structured-output-tool-calls]]
- 多实例分摊：[[13-parallelism-choice]]、[[day18-data-parallel]]
- 机制来源：[[day08-vllm-architecture]]、[[day09-scheduler]]、[[day13-flash-attention]]、[[day05-profiling]]
