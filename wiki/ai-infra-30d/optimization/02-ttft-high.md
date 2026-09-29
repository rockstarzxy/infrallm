---
title: "优化专项 02: TTFT 过高"
type: concept
tags: [ai-infra, inference-optimization, ttft, prefill, prefix-caching, scheduling]
created: 2026-09-29
updated: 2026-09-29
---
# 优化专项 02: TTFT 过高

> 一句话：把"首 token 慢"拆成排队、prefill 计算、缓存未命中、CPU 侧、冷启动五类根因，逐个用指标确认再修。关联日课：[[day03-benchmark]]、[[day06-param-tuning]]、[[day09-scheduler]]、[[day10-kv-cache]]、[[day13-flash-attention]]、[[day18-data-parallel]]、[[day24-disaggregated]]。

TTFT = 客户端发出请求到收到第一个 token 的时间。它不是一个阶段，而是一条链：

```
HTTP 解析 → chat template → tokenize → 进 EngineCore 队列 → 等待被调度（排队）
→ prefill（可能分多步）→ 采样 → 回前端 detokenize → SSE 第一个 chunk
```

链上每一段都可能是主因。先测再改，不要一上来就调 `max_num_batched_tokens`。

## 1. 症状与判定

先看 `/metrics` 里这几组（指标名随版本略有不同，用 `grep` 确认）：

| 指标 | 含义 | 指向 |
|---|---|---|
| `vllm:time_to_first_token_seconds` | TTFT 直方图 | 症状本身，看 P50 和 P99 差距 |
| `vllm:request_queue_time_seconds` | 从进队到首次被调度 | 高 → 排队 |
| `vllm:request_prefill_time_seconds` | 首次调度到 prefill 完成 | 高 → prefill 计算或分片 |
| `vllm:num_requests_waiting` | 等待队列长度 | 持续 > 0 → 容量不足 |
| `vllm:kv_cache_usage_perc` | KV 使用率 | 接近 1 且 waiting > 0 → KV 卡住并发 |
| `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` | 命中率 | 低于预期 → 缓存未命中 |
| `vllm:num_preemptions_total` | 抢占计数 | 增长 → 被抢占请求重排队，TTFT 翻倍 |

判定表：

| P50 TTFT | P99 TTFT | queue_time | prefill_time | 首选怀疑 |
|---|---|---|---|---|
| 高 | 高 | 低 | 高 | prefill 本身慢（长 prompt、backend、TP） |
| 低 | 高 | 高 | 低 | 排队，容量或突发流量 |
| 低 | 高 | 低 | 高 | 少数长 prompt 被 budget 切成多步，或多模态 |
| 高 | 高 | 低 | 低 | CPU 侧（tokenize、template、grammar 编译）或前端排队，引擎指标看不到 |
| 忽高忽低 | 极高 | 低 | 低 | 冷启动、CUDA graph 未捕获形状、KV connector 远端加载 |

`queue_time + prefill_time` 与端到端 TTFT 的差值就是 CPU 侧和网络的开销。差值超过 50 ms 就值得看前端进程。

## 2. 根因逐个排查

### 2.1 排队：容量不足或流量突发

确认：`num_requests_waiting` 持续大于 0，`request_queue_time` P99 占 TTFT 大头。再看 `kv_cache_usage_perc`：接近 1 说明 KV 容量卡住了并发；远低于 1 但 `num_requests_running` 等于 `max_num_seqs`，说明是 `max_num_seqs` 卡住。

修：
- KV 卡住：缩 `--max-model-len` 到业务实际需要、`--kv-cache-dtype fp8`、权重量化腾显存，或加 replica。见 [[05-oom-preemption]]。
- `max_num_seqs` 卡住：调大，前提是 KV 有余量，否则只会换成抢占。
- 流量突发：网关限流和排队隔离，让排队发生在网关而不是引擎，至少 P99 可控。加 replica 或 `--data-parallel-size`（见 [[13-parallelism-choice]]）。

副作用：`max_num_seqs` 调大后 decode batch 变大，TPOT 上升，见 [[03-tpot-itl-jitter]]。

### 2.2 prefill 本身长

确认：`request_prefill_time` 高且与 prompt 长度强相关。用 `vllm bench serve` 固定 `--random-input-len` 扫 1k/4k/16k，看 TTFT 是否近似平方增长（attention 部分 O(n²)，线性层 O(n)）。`nsys` 里 prefill 步的 attention kernel 占比是关键证据。

修：
- attention backend：Hopper 上确认日志用的是 FA3，不是 Triton 兜底。`VLLM_ATTENTION_BACKEND=FLASHINFER` 对比一次。FP8 KV 会改变 backend 选择，看日志。
- TP：TP 过大时每层两次 all-reduce 的通信占比上升，prefill 也受影响。单机 8 卡跑 7B 用 TP=8 通常比 DP=8 的 TTFT 差。
- 长 prompt 本身：prompt 压缩、把不变部分放前面吃 prefix cache、RAG 场景减少 context 数量。

### 2.3 prefix cache 未命中

V1 默认开启 prefix caching，所以问题通常不是"没开"，而是"命中不了"。确认：`prefix_cache_hits / prefix_cache_queries` 远低于 workload 的理论共享比例。

常见原因：
- system prompt 里有动态内容（时间戳、用户名、session id）放在最前面，hash 链从第一个 block 就断了。修：把动态内容挪到 prompt 尾部。
- chat template 每次渲染结果不同（工具列表顺序、默认字段）。修：固定模板输出，确认 token 级别一致。
- 多租户用了 `cache_salt` 但本意不是隔离。修：只在真正需要隔离时传。
- 多 replica 下请求被轮询到没有该前缀的实例。修：网关做 prefix-aware routing（按 system prompt 或会话 id 一致性哈希），见 [[day18-data-parallel]] 和 [[08-multi-turn-agent-workload]]。
- 前缀长度不到一个 block_size（默认 16）的整数倍部分不缓存，短 system prompt 收益天然有限。
- 缓存被驱逐：KV 太小、并发太高，free 队列里的缓存 block 很快被回收。看 `kv_cache_usage_perc` 是否长期接近 1。

副作用：prefix-aware routing 会让负载不均，需要配合上限和溢出策略。

### 2.4 token budget 太小，长 prefill 被切成多步

V1 里 chunked prefill 是默认行为：`--max-num-batched-tokens` 是每步的 token budget，decode 请求每个占 1，剩下给 prefill。一个 8k prompt 在 budget 2048 下至少 4 步才能完成 prefill，每步还要等同批 decode 一起跑完。

确认：`request_prefill_time` 高，但 nsys 里单步 prefill kernel 并不长；`prefill_time / 单步耗时` 约等于 `prompt_len / budget`。

修：调大 budget（示例：8192 或 16384）。代价是每步更重，decode 请求的 ITL 上升，见 [[03-tpot-itl-jitter]]。折中方法是配合长 prefill 保护参数：`--max-num-partial-prefills`、`--max-long-partial-prefills`、`--long-prefill-token-threshold`，限制同时处于分片状态的长请求数，避免多个长 prompt 一起把 budget 吃光。参数名以当前版本 `--help` 为准。

### 2.5 多个长 prefill 同时到达

确认：TTFT P99 尖峰与流量里长 prompt 的簇发时刻重合。日志里同一时刻多个请求进入 running 且 `num_scheduled_tokens` 都被压到很小。

修：上面的长 prefill 保护参数；网关按 prompt 长度分桶限流；极端情况下 P/D 解耦，让 prefill 有独立算力，见 [[day24-disaggregated]]。

### 2.6 CPU 侧：tokenize、chat template、多模态预处理、grammar 编译

引擎指标看不到这段。确认方法：端到端 TTFT 减去 `queue_time + prefill_time` 的残差；对 API server 进程 `py-spy dump`，看 `apply_chat_template`、`encode`、多模态 processor、`xgrammar` 编译的占比。

- tokenize 和 chat template：高 QPS 下单个 API server 进程会被打满。修：`--api-server-count N` 起多个前端进程。
- 结构化输出：grammar 首次编译几十毫秒到秒级，请求状态停在 `WAITING_FOR_FSM`。修：schema 稳定就靠编译缓存命中；复杂 schema 简化；对比 `xgrammar` 与 `guidance` backend。见 [[09-structured-output-tool-calls]]。
- 多模态：图片解码和 resize 在前端进程做，大图和多图请求会拖慢所有请求。修：客户端预先缩放，前端进程数增加，`--limit-mm-per-prompt` 限制数量。

### 2.7 多模态 encoder

vision encoder 在 worker 上跑，但它不走 KV cache，每张图都是一次完整 forward。确认：nsys 里 prefill 步前有明显的 encoder kernel 段。修：encoder cache 命中依赖相同图片的 hash；减少图片分辨率和数量；部分模型支持把 encoder 放到独立进程或独立卡。

### 2.8 冷启动与 CUDA graph 未捕获的形状

确认：服务刚启动或刚扩容后的前几分钟 TTFT 极高；或某些特定 batch 大小的请求慢，日志出现 eager fallback。

- 冷启动：权重加载、torch.compile、CUDA graph 捕获加起来几分钟。修：编译缓存目录预热进镜像，健康检查通过后再挂流量，见 [[14-cold-start-autoscaling]]。
- 形状未捕获：`cudagraph_capture_sizes` 只覆盖到某个上限，超出的 token 数走 eager 或 piecewise 之外的路径。修：按实际 batch 分布调整捕获桶，见 [[day13-flash-attention]]。

### 2.9 KV connector 远端加载

P/D 解耦或远端 KV 缓存时，请求状态会停在 `WAITING_FOR_REMOTE_KVS`。确认：connector 的传输指标和日志；断开 connector 对比 TTFT。修：传输带宽（RDMA 而不是 TCP）、逐层流水是否生效、远端命中率是否值得。见 [[day10-kv-cache]] 和 [[day24-disaggregated]]。

## 3. 决策表

| 症状组合 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| queue_time 高，KV 使用率接近 1 | 缩 max_model_len、KV FP8 | 加 replica | 调大 max_num_seqs（只会变成抢占） |
| queue_time 高，KV 使用率低，running = max_num_seqs | 调大 max_num_seqs | 加 replica | 调大 budget（无关） |
| prefill_time 高，与 prompt 长度平方相关 | 确认 backend 是 FA3/FlashInfer | 减 TP 改 DP | 缩 budget |
| prefill_time 高，单步不长 | 调大 budget + 长 prefill 保护参数 | P/D 解耦 | 关 chunked prefill（decode 会被整段 prefill 阻塞） |
| 命中率低于预期 | 动态内容后移、固定模板 | prefix-aware routing | 调大 block_size（不是主因） |
| P99 尖峰与长 prompt 簇发重合 | 长 prefill 保护参数 | 网关按长度分桶限流 | 只调 max_num_seqs |
| 端到端与引擎指标差值大 | py-spy 前端进程，加 api-server-count | 简化 schema、客户端预处理图片 | 调引擎参数 |
| 启动后几分钟慢 | 预热编译缓存、就绪探针 | 减少 capture sizes 缩短启动 | 提前挂流量 |

## 4. 验证实验

固定：模型、dtype、`max_model_len`、采样参数（temperature=0）、`--num-prompts`、随机种子。每组跑 3 次取中位数。

实验 A，budget 与 prefill 分片：

```bash
for B in 2048 8192 16384; do
  vllm serve Qwen/Qwen2.5-7B-Instruct --max-num-batched-tokens $B --port 8000 &
  vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --dataset-name random \
    --random-input-len 8192 --random-output-len 64 --num-prompts 64 --request-rate 4
done
```

预期：TTFT 随 B 增大而下降，长输出场景下 ITL P99 随 B 增大而上升。记录两条曲线的交点，就是这个 workload 的 budget 取值。

实验 B，prefix cache 命中：同一 system prompt 2000 token，两组 100 个请求，一组动态内容在头部，一组在尾部。比较 `prefix_cache_hits/queries` 和 TTFT P50。预期尾部组命中率接近 1，TTFT 下降到接近只算 user 部分的时间。

实验 C，排队与容量：`--request-rate` 从 2 扫到 32，记录 `request_queue_time` P99 和 `kv_cache_usage_perc`。找到 queue_time 开始非线性上升的拐点，这是单实例容量上限，用于 [[15-cost-per-token]] 的容量规划。

实验 D，CPU 侧残差：`--request-rate 32` 下分别用 `--api-server-count 1` 和 `4`，比较端到端 TTFT 与 `queue_time + prefill_time` 之和的差值。

## 6. 测量陷阱

- **非流式请求没有 TTFT。** 客户端拿到的是整条响应，"TTFT" 只能从服务端直方图看。压测必须 `stream=True`，否则测的是 E2E。
- **服务端 TTFT 不含网络和网关。** 客户端测到的 TTFT 减服务端直方图的差值，就是网关、TLS、跨机房的开销。差值稳定就不用管，差值抖动看网关。
- **首个 SSE chunk 可能是空 delta。** 一些 chat 实现先发 role chunk 再发内容，客户端按"第一个非空 content"计时才准。
- **prefix cache 让 TTFT 分布双峰。** 命中和未命中是两个分布，只看 P50 会误判。按命中与否分组统计，或者压测时明确控制命中率。
- **warmup 请求要排除。** 第一批请求承担 CUDA graph 首次回放、编译缓存加载，压测前先打 50 到 100 个请求预热。

## 7. 快速 checklist

按顺序过一遍，每一步有明确的"是/否"：

1. `stream=True` 且排除了 warmup？
2. `queue_time` 占 TTFT 多少？超过一半先看容量（2.1）。
3. `kv_cache_usage_perc` 接近 1？是则先做 KV 容量（[[05-oom-preemption]]）。
4. `prefix_cache_hits/queries` 符合 workload 预期？不符合先修前缀结构（2.3）。
5. `prefill_time` 与 prompt 长度的关系是平方还是与 `prompt_len / budget` 成正比？前者看 backend 和 TP（2.2），后者调 budget（2.4）。
6. 端到端与引擎指标残差超过 50 ms？py-spy 前端进程（2.6）。
7. 问题只在启动后几分钟或特定 batch 大小出现？看冷启动和 capture sizes（2.8）。
8. 以上都正常但 P99 仍高？看长 prompt 簇发（2.5）和抢占（[[05-oom-preemption]]）。

## 8. 关联

- 总入口：[[01-diagnosis-playbook]]
- 相邻：[[03-tpot-itl-jitter]]（budget 的另一侧代价）、[[04-throughput-low]]、[[05-oom-preemption]]、[[06-gpu-util-low-cpu-bound]]、[[08-multi-turn-agent-workload]]、[[09-structured-output-tool-calls]]、[[14-cold-start-autoscaling]]
- 日课：[[day03-benchmark]]、[[day06-param-tuning]]、[[day09-scheduler]]、[[day10-kv-cache]]、[[day13-flash-attention]]、[[day18-data-parallel]]、[[day24-disaggregated]]
