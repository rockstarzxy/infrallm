---
title: "优化专项 03: TPOT/ITL 抖动与 P99 长尾"
type: concept
tags: [ai-infra, inference-optimization, tpot, itl, tail-latency, cuda-graph, scheduling]
created: 2026-09-29
updated: 2026-09-29
---
# 优化专项 03: TPOT/ITL 抖动与 P99 长尾

> 一句话：decode 每步的耗时本应稳定，抖动说明有别的东西挤进了这一步；本页按"谁挤进来了"分类排查。关联日课：[[day03-benchmark]]、[[day05-profiling]]、[[day06-param-tuning]]、[[day09-scheduler]]、[[day13-flash-attention]]、[[day15-gpu-topology]]、[[day12-speculative-decoding]]。

先分清两个指标：

- **TPOT**（time per output token）：一个请求的总生成时间除以 token 数，是平均值，抹平了抖动。
- **ITL**（inter-token latency）：相邻两个 token 之间的间隔，每个请求有几十到几百个样本。抖动和长尾看 ITL 的 P99，不看 TPOT。

一次 decode 步的耗时 = 这一步 batch 里所有请求的 forward 时间 + 采样 + CPU 侧调度。任何一个请求的特殊情况都会拖慢同一步里的所有请求，这是抖动的根本来源：**decode 是共享一条时间线的**。

## 1. 症状与判定

| 指标 / 观察 | 指向 |
|---|---|
| `vllm:inter_token_latency_seconds` P99 远高于 P50（示例：5 倍以上） | 有周期性或偶发的重步 |
| ITL 尖峰与 `vllm:num_requests_running` 新增时刻重合 | prefill 分片挤进 decode 步 |
| ITL 尖峰与 `vllm:num_preemptions_total` 增长重合 | 抢占，被抢占请求重做 prefill |
| ITL 与 batch 大小呈阶梯关系 | CUDA graph 桶边界或 eager fallback |
| 开了 penalties / logit_bias / logprobs 的请求出现后整体变慢 | batch 内混合采样参数 |
| 尖峰周期性出现，与请求无关 | Python GC、日志刷盘、metrics 采集 |
| TP > 1 时尖峰，TP=1 无 | NCCL 抖动、PCIe 拓扑、卡间不均 |
| 引擎 ITL 正常，客户端 ITL 抖 | 前端 detokenize 排队、SSE flush、网络 |

判定顺序：先对比引擎侧直方图和客户端测量，确定抖动在引擎内还是引擎外；再看尖峰是否与 running 数变化、抢占计数、batch 大小相关。

## 2. 根因逐个排查

### 2.1 prefill 分片挤进 decode 步

V1 调度器每步的 token budget（`--max-num-batched-tokens`）先给 running 请求，剩余给新请求的 prefill。budget 越大，一步里能塞进的 prefill token 越多，这一步就越重，同批 decode 请求的这个 token 就越晚出来。

确认：ITL 尖峰高度接近"一个 budget 大小的 prefill 步耗时"，且尖峰时刻有新请求进入 running。单进程模式下打印每步 `num_scheduled_tokens` 之和，看尖峰是否对应 budget 被吃满的步。

修：
- 调小 budget（示例：从 8192 到 2048），代价是长 prompt 的 TTFT 上升，见 [[02-ttft-high]]。
- 长 prefill 保护参数限制同时分片的长请求数。
- 延迟敏感和吞吐优先的流量分到不同实例，各自配不同 budget。
- 彻底解决是 P/D 解耦，decode 实例上没有 prefill，见 [[day24-disaggregated]]。

### 2.2 抢占与 recompute

KV 分不出新 block 时，最晚进入 running 的请求被踢回 waiting，`num_computed_tokens` 清零。它本身的 ITL 出现一个巨大的空洞（等待重新调度加重做 prefill），其他请求也会因为它的重做 prefill 而被拖慢一步。

确认：`vllm:num_preemptions_total` 增长，`kv_cache_usage_perc` 接近 1。

修：本质是 KV 容量问题，见 [[05-oom-preemption]]。短期止血是降 `max_num_seqs` 让并发在 KV 能承受的范围内，长尾会立刻消失，吞吐略降。

### 2.3 CUDA graph 桶未命中或 eager fallback

piecewise CUDA graph 按 token 数分桶捕获，一步的 token 数向上取整到最近的桶。桶之间跳变时 pad 浪费不同，超过最大捕获桶的步走 eager，每层几十次 kernel launch 全部回来。

确认：ITL 与 `num_requests_running` 呈阶梯关系；日志里有形状未捕获的提示；nsys 里某些步 kernel 之间的 CPU 空隙明显变大。

修：`--compilation-config '{"cudagraph_capture_sizes":[...]}'` 按实际 batch 分布覆盖，尤其把 `max_num_seqs` 附近的桶补上。较新版本的 `cudagraph_mode` 设为 `FULL_AND_PIECEWISE` 让纯 decode 步整图回放。参数名随版本变化，以 `--help` 为准。代价是启动时捕获时间和显存池增加。

### 2.4 batch 内混合采样参数

Sampler 只在 batch 里存在某个特性时才走对应分支：一个请求要 `repetition_penalty`，整个 batch 多跑一遍 penalty kernel（需要每个请求的 token 计数）；一个请求要 `logprobs`，整个 batch 多算一次 log_softmax 和 topk；`logit_bias`、`min_p`、`bad_words` 同理。

确认：把带这些参数的请求单独隔离到另一个实例，看剩余流量的 ITL P99 是否下降。或 py-spy / torch profiler 看 sampler 占比。

修：产品侧统一采样参数；不需要的 `logprobs` 不要开；必须支持的话按参数把流量分到不同实例池。

### 2.5 spec decode 接受率波动

推测解码每步验证 k 个 draft token，接受几个就前进几个。接受率高时 ITL 很低，接受率低时一步只前进 1 个 token 但仍付了验证 k 个的代价，ITL 分布天然更宽。

确认：`vllm:spec_decode_num_accepted_tokens_per_pos` 按位置的接受率，以及关掉 spec decode 后 P99 是否收敛。

修：降 `num_speculative_tokens`，换更匹配的 draft（EAGLE-3 或 MTP），或在高并发时关掉，见 [[11-spec-decode-no-gain]]。

### 2.6 Python GC 与 CPU 抖动

EngineCore 是 Python 进程，每步都在分配对象。GC 的 gen2 回收会造成几十毫秒的停顿，与请求无关，周期性出现。日志刷盘、metrics 序列化也会占用主循环。

确认：尖峰周期固定且与流量无关；`py-spy record` 采样到 `gc` 相关帧；`PYTHONTRACEMALLOC` 不需要，直接在 EngineCore 进程里 `gc.set_debug(gc.DEBUG_STATS)` 看回收时间。

修：`--async-scheduling` 让调度与 GPU 执行重叠，CPU 抖动被 GPU 时间掩盖一部分；关闭不必要的请求级日志（`--disable-log-requests`）；降低 metrics 抓取频率。在自定义启动脚本里 `gc.freeze()` 冻结启动后的长期对象可减少 gen2 扫描量，这是通用 Python 手段，vLLM 是否内置以源码为准。

### 2.7 TP 下的 NCCL 抖动与 PCIe 拓扑

TP 每层两次 all-reduce，一步 decode 有几十次小规模集合通信。任何一张卡慢一点，所有卡都等它。PCIe 直连而非 NVLink 的拓扑、跨 NUMA 的卡、和其他进程共享 PCIe 带宽，都会让通信时间出现长尾。

确认：TP=1 时无尖峰，TP>1 有；`nvidia-smi topo -m` 看卡间是 NV 还是 PIX/PHB/SYS；nsys 里 all-reduce kernel 时长分布是否有长尾；`nccl-tests` 的 all-reduce 延迟是否稳定。

修：TP 组内只用 NVLink 全互联的卡；能用 DP 就不用 TP（7B 到 14B 单卡装得下就 DP，见 [[13-parallelism-choice]]）；自定义 all-reduce（vLLM 对小消息默认启用，看日志）；固定 NUMA 绑定。

### 2.8 KV 接近满时的分配失败重试

KV 使用率长期在 0.95 以上时，调度器每步都在"能不能给 running 请求再分一个 block"的边缘，分配失败就抢占、下一步再试。抖动表现为周期性小尖峰而不是一次大空洞。

确认：`kv_cache_usage_perc` 长期高位，`num_preemptions_total` 缓慢增长。

修：留出余量。经验上让稳态 KV 使用率在 0.8 到 0.9，方法同 [[05-oom-preemption]]。

### 2.9 前端进程排队：detokenize 与 stop string

EngineCore 每步产出所有请求的新 token，前端进程逐个 detokenize、判 stop string、组装 SSE。高并发流式输出下单个 API server 进程被打满，token 在前端排队，客户端看到的 ITL 抖而引擎直方图正常。

确认：引擎侧 `inter_token_latency` P99 正常，客户端 P99 高；API server 进程 CPU 接近 100%；py-spy 看到 detokenize 和 JSON 序列化占比高。

修：`--api-server-count N`；减少每 token 响应体的字段（不需要 `logprobs` 就不要）；客户端按 chunk 而不是按 token 消费。

### 2.10 网络与 SSE flush

反向代理的响应缓冲会把多个 SSE chunk 合并后一次发出，客户端看到的是"卡一下然后一串"。

确认：直连引擎端口和经过网关分别测 ITL。

修：nginx `proxy_buffering off`、`X-Accel-Buffering: no`；HTTP/2 或长连接复用；跨机房时接受基线延迟，但不应有抖动。

## 3. 决策表

| 症状组合 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| 尖峰与新请求进入 running 重合 | 调小 budget，加长 prefill 保护参数 | 分流量到不同实例 | 调大 max_num_seqs |
| 尖峰与抢占计数重合 | 降 max_num_seqs 止血，再扩 KV 容量 | KV FP8、缩 max_model_len | 调 budget |
| ITL 随 batch 大小阶梯变化 | 补 cudagraph_capture_sizes | FULL_AND_PIECEWISE 模式 | enforce-eager（更慢） |
| 少数请求带 penalties/logprobs 后整体变慢 | 统一采样参数 | 按参数分实例池 | 调引擎参数 |
| 开 spec decode 后 P99 变宽 | 降 num_speculative_tokens | 高并发时关闭 | 增大 k |
| 周期性尖峰与流量无关 | async scheduling、关请求日志 | gc 调优 | 加 GPU |
| TP>1 才有尖峰 | 检查拓扑，改 DP | 自定义 all-reduce、NUMA 绑定 | 加大 TP |
| 引擎正常客户端抖 | api-server-count、网关关缓冲 | 精简响应字段 | 调引擎参数 |

## 4. 验证实验

固定：模型、dtype、`max_model_len`、`temperature=0`、`--num-prompts`、种子。每组 3 次取中位数，报告 ITL P50 / P99 / max。

实验 A，budget 对 ITL 的影响（长短混合 workload）：

```bash
for B in 1024 2048 4096 8192; do
  vllm serve Qwen/Qwen2.5-7B-Instruct --max-num-batched-tokens $B --port 8000 &
  # 短请求持续流 + 长 prompt 周期性注入
  vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --dataset-name random \
    --random-input-len 256 --random-output-len 512 --num-prompts 200 --request-rate 8 &
  vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --dataset-name random \
    --random-input-len 8192 --random-output-len 32 --num-prompts 20 --request-rate 0.5
done
```

预期：短请求 ITL P99 随 B 增大而上升，长请求 TTFT 随 B 增大而下降。

实验 B，CUDA graph 桶：`--request-rate` 固定使稳态 running 数在某个桶边界附近（示例：running 在 60 到 70 之间，默认桶有 64），对比默认桶与把 72、80 加进 `cudagraph_capture_sizes` 后的 ITL 分布。预期加桶后阶梯消失。

实验 C，混合采样参数：同一流量，三组：全 greedy；全 `temperature=0.7,top_p=0.9`；10% 请求带 `repetition_penalty=1.2` 和 `logprobs=5`。预期第三组所有请求的 ITL P50 都上升，不只是那 10%。

实验 D，async scheduling：`--async-scheduling` 开关对比，decode 主导 workload（input 128 / output 1024 / concurrency 64）。预期 ITL P50 下降、P99 收窄、吞吐上升几个百分点；同时确认与当前版本的 spec decode 或结构化输出是否兼容。

实验 E，引擎内外分离：同一压测分别直连 8000 端口和经过网关，比较客户端 ITL P99 与引擎 `inter_token_latency` P99。差值稳定为基线延迟，差值抖动看网关。

## 6. 一步 decode 的时间预算

把一步拆开，才知道抖动的量级是否合理。以 7B 模型、单张 H100、BF16 为例（数字是示意，用你机器的 profile 替换）：

| 段 | 典型耗时 | 抖动来源 |
|---|---|---|
| 调度 `schedule()` | 0.5 到 2 ms，随 running 数线性 | GC、请求数突增 |
| SchedulerOutput 广播到 worker | 0.1 到 0.5 ms | TP 下共享内存队列 |
| `_prepare_inputs` | 0.5 到 1 ms | 新请求加入时重建 block table |
| forward（CUDA graph 回放） | 8 到 15 ms，随 batch 缓慢上升 | 桶未命中、prefill 混入 |
| sampler | 0.3 到 2 ms | 混合采样参数 |
| 输出回前端 + detokenize | 0.5 到 3 ms | 前端进程排队 |

稳态 ITL 应接近这几段之和。P99 比 P50 高出的部分，对照上表找是哪一段被放大了。放大十倍以上的只有三种可能：prefill 混入、抢占重做、eager fallback。

## 7. 快速 checklist

1. 用 ITL 直方图而不是 TPOT 看抖动？
2. 引擎侧 P99 和客户端 P99 差多少？差值抖动先看前端和网关（2.9、2.10）。
3. 尖峰时刻 `num_requests_running` 是否有新增？是则调 budget（2.1）。
4. `num_preemptions_total` 是否增长？是则先做 KV 容量（2.2）。
5. ITL 随 batch 大小是否阶梯变化？是则补 capture sizes（2.3）。
6. 流量里有没有 penalties、logit_bias、logprobs？有则隔离测一次（2.4）。
7. 开了 spec decode？关掉对比一次（2.5）。
8. 尖峰周期固定？看 GC 和日志（2.6）。
9. TP>1？用 TP=1 的小模型复现一次排除通信（2.7）。

## 8. 关联

- 总入口：[[01-diagnosis-playbook]]
- 相邻：[[02-ttft-high]]（budget 的另一侧）、[[05-oom-preemption]]（抢占）、[[06-gpu-util-low-cpu-bound]]（CPU 抖动）、[[11-spec-decode-no-gain]]、[[13-parallelism-choice]]（TP vs DP）
- 日课：[[day03-benchmark]]、[[day05-profiling]]、[[day06-param-tuning]]、[[day09-scheduler]]、[[day12-speculative-decoding]]、[[day13-flash-attention]]、[[day15-gpu-topology]]
