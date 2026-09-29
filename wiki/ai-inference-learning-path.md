---
title: AI 推理基础设施与优化 — 30 天高强度学习路径
type: synthesis
tags: [ai-infra, inference, vllm, distributed-systems, optimization, learning-path]
sources: []
created: 2026-06-01
updated: 2026-09-27
---

# AI 推理基础设施与优化 — 30 天高强度学习路径

目标：在 1 个月内快速建立 AI inference infra 的可用能力，能部署和压测 vLLM，理解主流推理优化技术，能做多 GPU / 多实例部署方案，能定位吞吐、延迟、显存和调度瓶颈。

学习强度：每天 10+ 小时，30 天约 300 小时。

核心判断：原版本是 16-20 周的系统学习路线，适合长期打底，但不是 1 个月冲刺的最优路径。30 天版本应以 **vLLM serving 实战 → 指标体系 → 优化手段 → 分布式部署 → kernel/底层补强 → 系统设计** 为主线，而不是先花数周深挖 CUDA 和论文。

---

## 当前课程设计评估

### 优点

- 覆盖面完整：包括 vLLM、PagedAttention、FlashAttention、量化、speculative decoding、KV Cache、continuous batching、分布式推理、Triton/CUDA、MoE 和 prefill-decode 解耦。
- 知识层次正确：从 Transformer / GPU 基础到 serving system，再到 kernel 和前沿系统设计。
- 有论文、代码和实操意识，适合沉淀成长期知识体系。

### 不适合 1 个月快速掌握的地方

- 时间太长：原计划 16-20 周，与 30 天目标冲突。
- 前置基础过重：CUDA GEMM、CUTLASS、roofline 可以帮助理解极致优化，但不应在第一阶段占用太多时间。
- vLLM 实战太靠后：如果目标是 AI infra / 推理优化，第一周就应跑通 serving、压测、指标和核心参数。
- 缺少每日产出：快速学习需要每天有可验证交付物，例如 benchmark 报告、参数对比表、源码阅读笔记、部署拓扑图。
- 缺少优先级：FlashAttention、PagedAttention、continuous batching、prefix caching、chunked prefill、quantization、speculative decoding、TP/PP/DP 的收益场景不同，需要按工程高频度排序。

### 优化原则

1. 先建立系统手感，再回补原理。
2. 每 2-3 天完成一个可演示实验。
3. vLLM 是主线，TensorRT-LLM / SGLang 是对照组。
4. 指标驱动学习：TTFT、TPOT、E2E latency、throughput、P50/P95/P99、GPU utilization、KV cache usage、batch size、queue time。
5. 只深挖能改变工程决策的底层知识。

### 准确性修正与补漏

以下是对本课程内容的审校修正，学习时以这些表述为准：

1. **prefill 通常更偏 compute-bound，decode 通常更偏 memory-bandwidth-bound，但不是绝对规律。** 实际瓶颈取决于 batch size、context length、attention backend、GQA/MQA/MLA、量化格式、GPU 型号和调度策略。判断时必须看 profiling 和指标，而不是只按阶段下结论。
2. **KV Cache 显存公式需要按 rank 和模型结构解释。** 更准确的近似是：`2 * local_layers * batch_size * sequence_length * local_kv_heads * head_dim * bytes_per_element`。其中 `2` 是 K/V；`local_layers` 与 PP 有关；`local_kv_heads` 与 GQA/MQA/TP/实现有关；MLA、sliding window、KV quantization、paged/block 管理都会改变实际占用。
3. **vLLM V1 中 chunked prefill 在支持时通常默认启用。** 学习重点不是只会加 `--enable-chunked-prefill`，而是理解它如何用 `max_num_batched_tokens` 在 TTFT、ITL/TPOT 和吞吐之间做权衡。小 `max_num_batched_tokens` 往往更利于 ITL，大值更利于 TTFT 和总吞吐，但要用 workload 验证。
4. **prefix caching / automatic prefix caching 的默认行为随 vLLM 版本和模型支持变化。** 旧资料常写“手动开启”，新版本可能在支持时默认开启。实验时必须记录 vLLM 版本、启动参数、是否命中 prefix cache，以及 workload 是否真的有共享前缀。
5. **`max_model_len` 不应简化成“越大 KV block 越少”。** 更稳妥的说法是：更大的 `max_model_len` 会提高单请求最大上下文能力，但可能降低可承载并发、增加内存压力、影响 profiling / CUDA graph / scheduling 配置；实际可用 block 数要看 vLLM 启动日志和 metrics。
6. **vLLM 分布式不只 Ray。** 单节点默认可用 multiprocessing；多节点常用 Ray，也支持 multiprocessing 形式的多节点启动。生产部署还要关注 Ray/vLLM 节点间网络安全，因为控制和数据流量不应暴露到不可信网络。
7. **Data Parallelism 不只是外部多实例负载均衡。** vLLM 也有 data parallel deployment / multi-API server 相关能力；课程中仍保留“多实例 + gateway”作为最通用生产模式，但学习时应同时理解框架原生 DP 与外部 replica 路由的差异。
8. **Speculative decoding 的收益边界要讲清。** 它主要适合中低 QPS、decode memory-bound、target model 每步验证能并行吃下多个 draft token 的场景；高负载大 batch、acceptance rate 低、draft model 过重、任务输出分布难预测时可能收益小甚至变慢。
9. **结构化输出和工具调用是 inference infra 的必修补充。** 生产 serving 不只是吐自然语言，还要支持 JSON schema / constrained decoding / tool calling / reasoning outputs。这会影响 sampler、latency、CPU overhead 和框架选择，尤其影响 vLLM vs SGLang 对比。
10. **观测体系应落到 Prometheus 指标。** 至少记录 `time_to_first_token`、`inter_token_latency`、`e2e_request_latency`、`request_queue_time`、`request_prefill_time`、`request_decode_time`、running/waiting/swapped requests、KV cache usage，以及 speculative decoding accepted/draft token 指标。
11. **Benchmark 必须控制变量。** 固定模型、dtype、tokenizer、chat template、输入/输出 token 分布、并发模型（固定 concurrency 还是固定 request rate）、warmup、采样参数和随机种子；否则很多“优化收益”只是 workload 改了。
12. **模型加载、冷启动和权重分发也属于 inference infra。** 生产里需要理解 safetensors、tensorizer/model streamer、容器镜像、权重缓存、启动时间、CUDA graph capture 和 rolling update 对可用性的影响。
13. **vLLM 源码学习必须以 V1 引擎为准（2026-09-27 修订）。** V1 从 0.8 成为默认，V0 已在 2025 年下半年移除。V0 的 `SequenceGroup`、`core/block_manager.py`、`worker/model_runner.py`、WAITING/RUNNING/SWAPPED 三队列、swap 抢占、copy-on-write 在当前代码里都不存在。V1 的关键事实：API server 与 EngineCore 分进程（ZMQ + msgspec）；调度器是单一 token budget 循环，用 `num_computed_tokens` 统一 prefill/decode/chunked prefill/spec decode；抢占只有 recompute，靠保留 hash 的 free block 让重算代价接近零；prefix caching 内建且默认开启；模型执行用 persistent batch + torch.compile + piecewise CUDA graph；扩展点是 `--scheduler-cls`、`--logits-processors`、KV connector、`--speculative-config`。Week 2 的 Day 8/9/10/13 已按此重写，并要求每天有改源码的交付物。

---

## 30 天课程总览

| 阶段 | 天数 | 主题 | 核心交付物 |
|---|---:|---|---|
| Week 1 | Day 1-7 | LLM 推理系统基础 + vLLM 跑通 + 指标体系 | 单卡 vLLM 服务、压测报告、指标解释文档 |
| Week 2 | Day 8-14 | vLLM 深入 + 推理优化技术 | 参数调优矩阵、PagedAttention / scheduler 源码笔记 |
| Week 3 | Day 15-21 | 分布式推理与部署 | 多 GPU / 多实例部署方案、TP/PP/DP 对比报告 |
| Week 4 | Day 22-30 | 高阶优化 + 系统设计项目 | 生产级 inference platform 设计文档和最终复盘 |

---

## Week 1: 推理系统基础与 vLLM 上手

目标：不要先陷入论文。先把一个真实 LLM serving 系统跑起来，知道每个指标在反映什么瓶颈。

### Day 1: 推理系统全景和 Transformer 推理路径

上午：
- 梳理 LLM inference 生命周期：request → tokenize → prefill → decode loop → sampling → stream response。
- 区分 prefill 和 decode：
  - prefill：处理完整 prompt，通常更接近 compute-bound。
  - decode：逐 token 生成，通常更接近 memory-bandwidth-bound。
- 复习 KV Cache 的形状和显存公式：
  - 近似：`2 * local_layers * batch_size * sequence_length * local_kv_heads * head_dim * bytes_per_element`。
  - 注意：TP/PP、GQA/MQA/MLA、sliding window、KV quantization 都会改变实际占用。

下午：
- 阅读 Transformer attention、MQA、GQA、KV Cache 基础材料。
- 做一个小脚本估算 7B / 70B 在不同上下文长度下的 KV Cache 显存。

晚上交付物：
- `推理链路图.md`
- `KV Cache 显存估算表.md`

### Day 2: vLLM 单卡部署和 OpenAI-compatible API

上午：
- 安装并启动 vLLM。
- 跑通一个 7B/8B 模型，例如 Llama/Qwen/Mistral 系列。
- 用 OpenAI-compatible API 发起 chat/completions 请求。

下午：
- 学习 vLLM 基本参数：
  - `--gpu-memory-utilization`
  - `--max-model-len`
  - `--max-num-seqs`
  - `--max-num-batched-tokens`
  - `--enable-prefix-caching`
  - `--enable-chunked-prefill`
  - `--kv-cache-dtype`
  - `--distributed-executor-backend`
  - `--disable-log-stats` / metrics 相关配置

晚上交付物：
- `vLLM 单卡启动命令清单.md`
- 成功请求截图或 curl 输出记录

### Day 3: Benchmark 和指标体系

上午：
- 使用 vLLM benchmark 工具或自写压测脚本测试不同输入/输出长度。
- 建立指标词典：
  - TTFT: time to first token
  - TPOT: time per output token
  - ITL: inter-token latency
  - E2E latency
  - output tokens/s
  - requests/s
  - P50/P95/P99
- 建立 benchmark 规范：
  - 固定模型、dtype、量化格式、tokenizer、chat template、sampling params。
  - 区分固定 concurrency 与固定 request rate。
  - 记录 warmup、输入/输出 token 分布、失败率、OOM、重试。

下午：
- 设计 4 组 workload：
  - short input / short output
  - short input / long output
  - long input / short output
  - long input / long output
- 对比不同并发数下 TTFT、TPOT、throughput 的变化。

晚上交付物：
- `baseline-benchmark-report.md`
- 一张 workload 对比表

### Day 4: PagedAttention 和 continuous batching

上午：
- 读 vLLM/PagedAttention 论文的核心部分。
- 重点理解：
  - KV Cache 为什么会碎片化。
  - block table 如何把逻辑 token block 映射到物理 block。
  - continuous batching 为什么比 static batching 更适合在线 serving。

下午：
- 从 benchmark 结果反推调度行为：并发升高时 TTFT/TPOT 如何变化。
- 画出 vLLM 中 request lifecycle。

晚上交付物：
- `PagedAttention 一页纸解释.md`
- `Continuous batching 调度时序图.md`

### Day 5: profiling 基础

上午：
- 学习 GPU 利用率、显存、SM utilization、HBM bandwidth 的基本含义。
- 使用 `nvidia-smi`, DCGM, PyTorch profiler 或 Nsight Systems 做一次粗粒度 profile。

下午：
- 分析 prefill 和 decode 的 kernel 行为差异。
- 找到一次服务中的主要耗时阶段：排队、prefill、decode、采样、网络返回。

晚上交付物：
- `profile-notes.md`
- 解释当前瓶颈更像 compute-bound、memory-bound 还是 scheduler-bound。

### Day 6: vLLM 参数调优第一轮

上午：
- 系统测试以下参数：
  - `gpu_memory_utilization`
  - `max_num_seqs`
  - `max_num_batched_tokens`
  - `max_model_len`
  - `enable_prefix_caching`
  - `kv_cache_dtype`

下午：
- 在相同 workload 下记录吞吐、TTFT、TPOT、OOM 情况。
- 总结参数之间的取舍：
  - `max_num_batched_tokens` 小时通常更利于 ITL/TPOT；大时通常更利于 TTFT 和吞吐，但取决于 workload。
  - 更高 GPU memory utilization 给 KV Cache 更多空间，但也提高 OOM 风险。
  - 更大的 `max_model_len` 提高最大上下文能力，但可能降低并发、增加内存压力，并影响 profiling / CUDA graph / scheduling。

晚上交付物：
- `vLLM 参数调优矩阵 v1.md`

### Day 7: Week 1 复盘

上午：
- 整理 Week 1 所有实验。
- 写一页总结：如果线上服务 P99 TTFT 变差，应该先看哪些指标。

下午：
- 进行一次模拟面试：
  - vLLM 为什么快？
  - PagedAttention 解决什么问题？
  - prefill/decode 的瓶颈为什么不同？
  - max_num_batched_tokens 调大有什么副作用？

晚上交付物：
- `week-1-review.md`

---

## Week 2: vLLM 深入与核心推理优化

目标：能解释常见优化为何有效，并能通过实验验证收益边界。

### Day 8: vLLM V1 架构与请求生命周期（源码级）

先定位源码：`python -c "import vllm, os; print(os.path.dirname(vllm.__file__))"`，所有路径以 `vllm/v1/` 为准。V0 的 `SequenceGroup`、`core/block_manager.py`、`SWAPPED` 队列已经不存在。

阅读顺序：
1. `entrypoints/openai/`：chat template、参数校验、tool calling 解析。
2. `v1/engine/async_llm.py`、`processor.py`、`output_processor.py`：前端进程做 tokenize / detokenize / stop string。
3. `v1/engine/core.py`：EngineCore 独立进程的 busy loop，`step()` 三段。
4. `v1/engine/core_client.py`、`serial_utils.py`：ZMQ + msgspec 的进程间通信。
5. `v1/executor/`：UniProc / Multiproc / Ray，共享内存广播 SchedulerOutput。
6. `v1/request.py`：`RequestStatus` 状态机（没有 SWAPPED）。

实验：从源码安装；py-spy 分别抓 API server 和 EngineCore 进程；给 vLLM 加一个自定义 Prometheus 指标并在 `/metrics` 看到。

交付物：
- `vllm-v1-request-lifecycle.md`（带真实路径和行号）
- `custom-metric.diff`

### Day 9: V1 Scheduler 逐段精读

文件：`v1/core/sched/scheduler.py`、`sched/output.py`、`sched/request_queue.py`。

重点问题：
- `num_computed_tokens` 如何把 prefill、decode、chunked prefill、spec decode 统一成一条代码路径。
- `schedule()` 的三段：先服务 running、无抢占时才拉 waiting、打包输出。`max_num_seqs` 和 `max_num_batched_tokens` 各卡在哪一段。
- 抢占为什么只剩 recompute，被抢占请求为什么通常不需要重算全部 prefill。
- priority 队列、长 prefill 保护参数、async scheduling、结构化输出的 `WAITING_FOR_FSM`、KV connector 的 `WAITING_FOR_REMOTE_KVS` 各在哪里接入。

实验：
- 单进程模式打印每步 `num_computed_tokens`，画出 chunked prefill 时序。
- 长短混合 workload 对比 `max_num_batched_tokens=512/8192`。
- 触发抢占并观察 `num_preemptions_total`。
- 改一行抢占顺序看吞吐变化；用 `--scheduler-cls` 实现租户配额调度器。

交付物：
- `scheduler-analysis.md`、`preempt-order.diff`、`tenant_quota_scheduler.py`

### Day 10: KVCacheManager、Prefix Caching 与 KV Connector（源码级）

文件：`v1/core/kv_cache_manager.py`、`block_pool.py`、`kv_cache_utils.py`、`single_type_kv_cache_manager.py`、`distributed/kv_transfer/kv_connector/v1/`。

学习内容：
- block 生命周期：分配 → 写满登记 hash → 释放回 free 队列（保留 hash）→ 命中复用或 LRU 惰性驱逐。
- hash 链（含 parent hash、多模态 hash、LoRA、`cache_salt`），为什么只有写满的 block 才缓存。
- V1 为什么不需要 swap 和 copy-on-write。
- hybrid KV cache：full attention / sliding window / Mamba 层各自的 manager 和 `KVCacheGroup`。
- KV connector 的 scheduler 侧和 worker 侧回调，逐层 save/load 如何和计算重叠；NIXL / LMCache / SharedStorage / Offloading 各自用途。
- KV FP8；StreamingLLM / H2O 类有损方法只做了解。
- prefix-aware routing：让共享前缀请求尽量落到同一 replica（Day 18 展开）。

实验：
- 用 `prefix_cache_hits/queries` 指标验证命中、hash 链和 `cache_salt` 隔离。
- 小 KV 实例观察驱逐；抢占恢复时 `num_computed_tokens` 恢复到多少。
- 参照 `SharedStorageConnector` 写一个文件 KV connector，两实例共享前缀。
- bf16 vs fp8 KV 的 block 数和质量对比。

交付物：
- `kv-cache-manager-notes.md`、`prefix-cache-experiments.md`、`my_file_connector.py`

### Day 11: Quantization

学习内容：
- FP16 / BF16 / FP8 / INT8 / INT4 的适用场景。
- GPTQ、AWQ、SmoothQuant 的差异。
- 权重量化 vs KV Cache 量化。
- 量化对吞吐、显存、质量、冷启动的影响。

实验：
- 同模型对比 FP16/BF16 与 AWQ/GPTQ/FP8 中至少一种。
- 记录显存、吞吐、TTFT、TPOT、质量样例。

交付物：
- `quantization-tradeoff-report.md`

### Day 12: Speculative Decoding

学习内容：
- draft model + target model 的验证机制。
- acceptance rate 决定收益上限。
- batch size、输出长度、任务类型对收益的影响。
- Medusa、EAGLE、n-gram speculative decoding 的大致区别。

实验：
- 在 vLLM 或 TensorRT-LLM 文档环境中跑一个 speculative decoding 示例。
- 记录 acceptance rate、吞吐和 latency。

交付物：
- `speculative-decoding-notes.md`

### Day 13: 模型执行层：GPUModelRunner、CUDA Graph、Attention Backend 与 Sampler

文件：`v1/worker/gpu_model_runner.py`、`gpu_input_batch.py`、`v1/attention/backends/`、`compilation/`、`v1/sample/`、`v1/structured_output/`。

学习内容：
- `execute_model()` 五阶段：persistent batch 增量更新、扁平输入和 `slot_mapping`、forward context、compute_logits、sampler。
- torch.compile 和 piecewise CUDA graph 的分工，attention 为什么被切出 graph，`cudagraph_mode` 各模式，`--enforce-eager` 实际关掉了什么，编译缓存与冷启动。
- attention backend 选择逻辑；FA2/FA3/FlashInfer/Triton/MLA 的适用条件；FlashAttention 的 tiling、online softmax、FlashDecoding split-KV。
- Sampler 的处理顺序，batch 内混合采样参数的代价，V1 `LogitsProcessor` 接口。
- 结构化输出从 `WAITING_FOR_FSM` 到 bitmask apply 的完整路径及开销来源。

实验：
- 四种 CUDA graph 配置对比并数 kernel launch 次数。
- 切换三个 attention backend，长 prefill 和长 decode 两种 workload。
- 采样参数混合的 TPOT 代价；写并加载一个自定义 logits processor。
- 简单/复杂 JSON schema 的结构化输出开销，切换 xgrammar / guidance。

交付物：
- `model-runner-notes.md`、`attention-backend-report.md`、`sampler-and-structured-output.md`

### Day 14: Week 2 综合调优实验

任务：
- 选定一个模型和 workload，依次开启/调整：
  - prefix caching
  - chunked prefill
  - kv cache dtype
  - quantization
  - max_num_batched_tokens
  - max_num_seqs
  - speculative decoding
  - guided / structured output 开销

最终交付物：
- `vLLM tuning playbook.md`
- 包含“什么场景该开/不开某个优化”的决策表。

---

## Week 3: 分布式推理与部署

目标：能设计单机多卡、多节点、多实例 serving 方案，知道 TP/PP/DP/EP 的通信代价和适用场景。

### Day 15: GPU 拓扑和通信基础

学习内容：
- PCIe、NVLink、NVSwitch、InfiniBand/RoCE 的作用。
- NCCL collective：AllReduce、AllGather、ReduceScatter、All-to-All。
- 为什么 tensor parallel 对节点内高速互联敏感。

实验：
- 查看机器 GPU topo。
- 如有多卡，跑 NCCL tests 或等价带宽测试。

交付物：
- `hardware-topology-notes.md`

### Day 16: Tensor Parallelism

学习内容：
- Megatron-style column parallel / row parallel linear。
- attention 和 MLP 中 TP 如何切矩阵。
- TP 的通信点和通信量。

实验：
- vLLM `--tensor-parallel-size 2/4/8` 对比。
- 记录 latency、throughput、显存和失败情况。

交付物：
- `tensor-parallel-report.md`

### Day 17: Pipeline Parallelism 和多节点

学习内容：
- PP 按层切分。
- pipeline bubble 和 microbatch。
- vLLM 多节点通常结合 TP + PP：节点内 TP，节点间 PP。
- vLLM 多节点 runtime：Ray 常用，multiprocessing 也可用于指定 `nnodes/node-rank/master-addr` 的多节点启动。
- 多节点安全：Ray/vLLM 内部网络必须放在可信私网，不要暴露到公网。

实验：
- 如果有多节点：使用 Ray 启动多节点 vLLM。
- 如果没有：画出 2 node x 8 GPU 的部署方案和通信路径。

交付物：
- `pipeline-parallel-and-multinode-design.md`

### Day 18: Data Parallelism、多实例和路由

学习内容：
- DP 复制模型，按请求分流。
- 多 vLLM instance + load balancer 的生产架构。
- vLLM 原生 data parallel deployment 与外部多 replica gateway 的差异。
- 模型版本路由、灰度、限流、队列隔离。
- 何时应该 scale out 多实例，而不是继续加 TP。
- cache-aware routing / prefix-aware routing：共享上下文请求优先路由到已有 KV cache 的 replica。

实验：
- 本机起多个 vLLM 实例或模拟多实例路由。
- 设计一个 gateway：按模型名、租户、优先级路由。

交付物：
- `multi-instance-serving-design.md`

### Day 19: Expert Parallelism 和 MoE 推理

学习内容：
- MoE 推理中 active parameters 与 total parameters 的区别。
- expert parallel 的 All-to-All 通信。
- DeepSeek / Mixtral 类模型为什么需要特殊并行策略。
- attention DP + expert TP/EP 的组合思想。

交付物：
- `moe-inference-notes.md`

### Day 20: Kubernetes / Ray Serve / 生产部署

学习内容：
- 镜像、模型权重挂载、健康检查、rolling update。
- 模型加载与冷启动：权重格式、下载/挂载、缓存、tensorizer/model streaming、CUDA graph capture 时间。
- GPU 调度、MIG、节点亲和性。
- autoscaling 指标：QPS、queue depth、tokens/s、GPU utilization、KV cache pressure。
- 监控和告警：TTFT/TPOT/P99、OOM、preemption、swap、CUDA error。
- Prometheus / Grafana：关注 `time_to_first_token`、`inter_token_latency`、`e2e_request_latency`、`request_queue_time`、`request_prefill_time`、`request_decode_time`、running/waiting/swapped requests、KV cache usage。
- CPU 侧瓶颈：tokenization、chat template、JSON/schema constrained decoding、网络流式返回。

交付物：
- `production-deployment-checklist.md`

### Day 21: Week 3 系统设计演练

题目：
- 设计一个支持 7B、70B、MoE 模型的在线推理平台。
- SLA：P99 TTFT < 800ms，P99 TPOT < 80ms。
- 需求：多租户、灰度、限流、成本可控、单实例故障可恢复。

交付物：
- `inference-platform-design-v1.md`

---

## Week 4: 高阶优化、框架对比与最终项目

目标：从“会用 vLLM”升级到“知道不同框架和优化路线的边界”，并形成能面试/落地的最终作品。

### Day 22: TensorRT-LLM 对照学习

学习内容：
- TensorRT-LLM 的定位：NVIDIA GPU 上的极致性能路线。
- 重点概念：
  - in-flight batching
  - paged KV cache
  - quantization
  - CUDA Graph
  - speculative decoding
  - disaggregated serving

交付物：
- `vLLM-vs-TensorRT-LLM.md`

### Day 23: SGLang 对照学习

学习内容：
- SGLang 的定位：结构化 generation、agentic workflow、多轮/分支 prompt 场景。
- RadixAttention：复用 prompt/KV 前缀，减少重复计算。
- 与 vLLM prefix caching 的联系和差异。
- 结构化输出、constrained decoding、tool calling 对 latency 和 sampler 的影响。

交付物：
- `vLLM-vs-SGLang.md`

### Day 24: Prefill-Decode 解耦

学习内容：
- prefill 和 decode 的资源瓶颈不同。
- 解耦 serving：prefill worker + decode worker + KV transfer。
- DistServe / Splitwise 思路。
- vLLM disaggregated prefilling 当前更偏 experimental / connector-based。
- 设计重点：KV transfer 带宽、跨节点延迟、cache ownership、故障恢复、decode worker 饥饿问题。

交付物：
- `prefill-decode-disaggregation.md`

### Day 25: Triton 和 kernel 优化入门

学习内容：
- CUDA grid/block/thread 基础。
- memory coalescing、shared memory、register、occupancy。
- Triton block programming 模型。

实验：
- 写一个 Triton softmax 或 matmul tutorial。
- 用 profiler 看 kernel 数量和耗时。

交付物：
- `triton-kernel-notes.md`

### Day 26: Roofline 和瓶颈判断

学习内容：
- arithmetic intensity。
- compute-bound vs memory-bound。
- 为什么 decode 常被 HBM bandwidth 限制。
- 为什么 batch、GQA/MQA、quantization 能改变瓶颈。

交付物：
- `roofline-inference-notes.md`

### Day 27: 长上下文推理

学习内容：
- 长上下文下 KV Cache 显存线性增长。
- attention 计算、prefill TTFT、cache eviction 的压力。
- sliding window、sparse attention、KV offloading、Ring Attention、chunked prefill。
- 长上下文 benchmark 必须覆盖 cache 命中、首轮长 prefill、多轮复用、长输入短输出和长输入长输出。

交付物：
- `long-context-inference-notes.md`

### Day 28: 最终项目实验日

任务：
- 选一个模型，完成从部署到优化的完整实验。
- 至少包含：
  - baseline
  - benchmark methodology：固定变量、workload、warmup、失败率
  - 2 个 vLLM 参数调优
  - 1 个 KV 优化
  - 1 个量化或 speculative decoding 实验
  - 1 个分布式或多实例方案

交付物：
- `final-project-benchmark-data.md`

### Day 29: 最终系统设计文档

文档结构：
1. 目标和 SLA。
2. 模型和 workload 假设。
3. 硬件选型。
4. vLLM 参数配置。
5. 并行策略。
6. KV Cache 策略。
7. 量化和 speculative decoding 策略。
8. 路由、限流、监控、扩缩容。
9. 成本和风险。
10. 下一步优化路线。

交付物：
- `inference-platform-design-final.md`

### Day 30: 面试级复盘和知识图谱

任务：
- 整理 20 个高频问题：
  - vLLM 为什么能提升吞吐？
  - PagedAttention 和 OS paging 像在哪里？
  - continuous batching 如何影响 TTFT/TPOT？
  - 为什么 decode 是 memory-bound？
  - TP/PP/DP/EP 如何选择？
  - 什么情况下 speculative decoding 不加速？
  - prefix caching 和 RadixAttention 解决什么问题？
  - 量化为何可能降低 latency，也可能不明显？
- 把所有交付物汇总成一份 portfolio。

最终交付物：
- `AI inference infra 30 天复盘.md`

---

## 推理优化专项（2026-09-29 新增）

日课之外增设问题驱动的优化专项 [[ai-infra-30d/optimization/00-index]]，15 页，覆盖 TTFT 高、TPOT 抖动、吞吐低、OOM 与抢占、CPU 瓶颈、长上下文、多轮与 Agent workload、结构化输出、量化选型、推测解码无收益、MoE 部署、并行策略、冷启动与扩缩容、单位 token 成本。每页按"症状判定 → 根因排查 → 决策表 → 验证实验"组织。Day 14、Day 21、Day 28 各安排一次通读。

## 每日时间分配模板

| 时间 | 内容 |
|---|---|
| 2 小时 | 原理阅读：论文、官方文档、源码导读 |
| 3 小时 | 实验：部署、压测、参数对比、profiling |
| 2 小时 | 源码：vLLM / framework 关键路径 |
| 2 小时 | 笔记：写成可复用文档、图和表 |
| 1 小时 | 复盘：回答面试题、总结取舍和故障场景 |

原则：每天必须有一个可检查产出。只看资料不算完成。

---

## 必学知识点优先级

### P0: 1 个月内必须掌握

- vLLM serving、OpenAI-compatible API、benchmark。
- TTFT、TPOT、throughput、P99、GPU utilization、KV cache usage。
- prefill vs decode。
- PagedAttention、continuous batching。
- prefix caching、chunked prefill。
- `max_num_batched_tokens`、`max_num_seqs`、`gpu_memory_utilization`、`max_model_len`。
- FP16/BF16/FP8/INT4/AWQ/GPTQ 的基本取舍。
- speculative decoding 的收益条件。
- TP、PP、DP、EP 的适用场景和通信代价。
- 多实例 serving、load balancing、autoscaling、monitoring。
- benchmark 方法学：固定变量、workload 设计、warmup、P99、失败率。
- 结构化输出 / guided decoding / tool calling 的 serving 成本。

### P1: 应该理解，但不必一开始深挖

- FlashAttention / FlashDecoding 原理。
- TensorRT-LLM 与 SGLang 的定位和差异。
- MoE 推理和 expert parallel。
- prefill-decode disaggregation。
- 长上下文推理策略。
- Nsight Systems / Nsight Compute 基础使用。
- 模型加载、冷启动、权重分发、CUDA graph capture。
- LoRA adapter serving、多模态输入、embedding / rerank 模型 serving。

### P2: 1 个月后继续深入

- 手写 CUDA GEMM。
- CUTLASS。
- 自定义 attention kernel。
- 复杂 roofline 分析。
- 调度器二次开发和生产级 patch。

---

## 推荐阅读顺序

1. vLLM quickstart、serving、benchmark、parallelism/scaling 官方文档。
2. vLLM optimization/tuning、metrics、prefix caching、quantization、speculative decoding 官方文档。
3. vLLM paper: Efficient Memory Management for Large Language Model Serving with PagedAttention。
4. Orca: A Distributed Serving System for Transformer-Based Generative Models。
5. FlashAttention / FlashAttention-2。
6. GPTQ、AWQ、SmoothQuant 中各选一篇精读，其余读摘要和工程实现。
7. Speculative Decoding、Medusa / EAGLE。
8. Megatron-LM parallelism 相关章节。
9. TensorRT-LLM docs：Paged Attention、IFB、KV Cache、Quantization、Speculative Decoding、Parallelism。
10. SGLang docs / paper：RadixAttention、structured outputs、constrained decoding。
11. DistServe / Splitwise：prefill-decode disaggregation。

---

## 实验环境建议

最低可行：
- 1 张 24GB+ GPU。
- 7B/8B 模型。
- 能完成 vLLM serving、benchmark、prefix caching、chunked prefill、量化对比。

推荐：
- 单机 4-8 张 A100/H100/H200 或同级 GPU。
- 可以完成 TP、多实例、NCCL/topology、长上下文和较大模型实验。

如果没有多 GPU：
- 仍然按计划学习 TP/PP/DP/EP，但把实验替换为部署设计、通信量推导和已有 benchmark 复盘。
- 分布式能力的重点是“知道为什么这么切、什么时候这么切、代价是什么”，不是必须拥有完整集群。

---

## 结业标准

30 天结束时，应能完成以下任务：

1. 从零启动一个 vLLM online serving。
2. 设计并执行 benchmark，解释 TTFT、TPOT、throughput、P99 的变化。
3. 根据 workload 选择 vLLM 关键参数。
4. 解释 PagedAttention、continuous batching、prefix caching、chunked prefill。
5. 判断何时使用量化、speculative decoding、FlashAttention 类优化。
6. 设计单机多卡、多节点或多实例 serving 架构。
7. 解释 TP/PP/DP/EP 的差异和通信瓶颈。
8. 写出一份生产级推理平台设计文档。
9. 面对 latency 变差、OOM、吞吐低、长 prompt 拖慢短请求等问题，给出排查路径。

---

## 长期进阶路线

如果 30 天后继续深挖，建议按这个顺序：

1. vLLM scheduler / block manager 二次开发。
2. Nsight Systems / Nsight Compute 体系化 profiling。
3. Triton kernel：softmax、layernorm、matmul、attention。
4. CUDA / CUTLASS：GEMM、Tensor Core、warp-level primitives。
5. TensorRT-LLM engine build、plugin、quantization pipeline。
6. MoE 大规模部署：expert parallel、load balancing、All-to-All 优化。
7. 长上下文：KV offload、sparse attention、disaggregated KV cache。
8. 生产平台：multi-tenant scheduler、cost-aware routing、cache-aware routing、capacity planning。

---

## 关键参考

- vLLM official docs: parallelism and scaling, disaggregated prefilling, serving and benchmarking.
- vLLM official docs: optimization and tuning, automatic prefix caching, quantization, speculative decoding, structured outputs, production metrics.
- vLLM paper: Efficient Memory Management for Large Language Model Serving with PagedAttention.
- TensorRT-LLM official docs: Paged Attention, in-flight batching, KV Cache, quantization, speculative decoding, disaggregated serving, parallelism.
- SGLang docs: RadixAttention, structured generation, continuous batching.
- FlashAttention and FlashAttention-2 papers.
- Orca paper.
- Megatron-LM paper.
- GPTQ, AWQ, SmoothQuant papers.
