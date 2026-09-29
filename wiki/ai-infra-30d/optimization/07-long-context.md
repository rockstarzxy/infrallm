---
title: "优化专项 07: 长上下文推理"
type: concept
tags: [ai-infra, inference-optimization, long-context, kv-cache, chunked-prefill, kv-offload, sparse-attention, rope]
created: 2026-09-29
updated: 2026-09-29
---

# 优化专项 07: 长上下文推理

> 一句话：上下文从 8k 拉到 32k 以上后，TTFT 飙升、并发骤降、显存告急，这页把三个问题拆到各自的根因和对应手段。关联日课：[[day01-inference-pipeline]]、[[day09-scheduler]]、[[day10-kv-cache]]、[[day13-flash-attention]]、[[day24-disaggregated]]、[[day27-long-context]]。

## 1. 症状与判定

长上下文的三个问题来源不同，先分清是哪一个：

| 观察 | 指向 | 本质 |
|---|---|---|
| TTFT 随输入长度超线性增长（32k 是 8k 的 6 倍以上而不是 4 倍） | prefill attention 的 O(n²) 部分开始主导 | 计算问题 |
| `# GPU blocks` 除以每请求 block 数只有个位数；`vllm:num_requests_waiting` 高、`kv_cache_usage_perc` 常年 95% 以上 | KV 显存线性增长把并发压没了 | 显存问题 |
| `vllm:num_preemptions_total` 持续增长，长请求反复被抢占 | 少数长请求把 block 吃光又吐出 | 显存问题的抢占表现，见 [[05-oom-preemption]] |
| 短请求 ITL 在长请求到达时出现尖峰 | 长 prefill 分片仍太大 | 调度问题，见 [[03-tpot-itl-jitter]] |
| 长文档 RAG 场景，同一文档反复出现但 TTFT 不降 | prefix cache 未命中或被驱逐 | 缓存问题 |
| 超过训练长度后输出质量崩 | RoPE 扩展配置或模型本身不支持 | 模型问题，不是 infra 能修的 |
| 模型声称 128k，服务启动就 OOM 或 block 数为 0 | `max_model_len` 与 profiling 的 dummy run 占满显存 | 配置问题 |

先做一件事：统计真实流量的输入长度分布（P50/P90/P99）。绝大多数"长上下文"服务的 P90 远低于 `max_model_len`，很多手段只要针对尾部就够了。

## 2. 根因逐个排查

### 2.1 KV 显存线性增长（最常见）

**确认**：用 [[day01-inference-pipeline]] 的公式算每 token KV 大小，乘以 `max_model_len`，对比 `# GPU blocks` 乘 block_size。示例：Qwen2.5-7B（GQA，4 个 KV head）bf16 下每 token 约 56 KB，128k 上下文单请求 7 GB，24 GB 卡扣掉权重后只能放 1 个请求。

**修**（按代价从小到大）：
1. `--max-model-len` 设为 P99 实际长度而不是模型上限，多余的部分只是给 profiling 和抢占添麻烦。
2. `--kv-cache-dtype fp8`，KV 减半，需要 backend 支持（FA3 / FlashInfer / Triton，[[day13-flash-attention]]），长序列末端质量要抽查。
3. 换 MLA 或更少 KV head 的模型：DeepSeek 系列的 MLA 每 token KV 是 GQA 模型的几分之一到十几分之一（[[day10-kv-cache]]）。
4. KV offload：KV connector 把不活跃请求的 KV 放到 CPU 内存或远端（`OffloadingConnector`、LMCache），用 PCIe 或网络带宽换显存。适合多轮长 session，不适合每个请求都是新文档。
5. 更多卡做 TP 分摊 KV（每卡只存自己那份 KV head），或 PP 分摊层，见 [[13-parallelism-choice]]。

**副作用**：fp8 KV 的质量损失在长序列末端更明显；offload 增加 TTFT 尾部；TP 引入通信。

### 2.2 prefill 的 O(n²) 计算

**确认**：nsys 里 attention kernel 占 prefill 时间的比例随长度上升。经验上 GQA 模型在 32k 左右 attention 开始和 GEMM 平分秋色，再往上 attention 主导（示例数字，和 head 数、backend 有关）。

**修**：
- 确认用了 FA3（Hopper）或 FlashInfer，不是 Triton fallback，prefill 差距可达数倍。
- prefix caching 把重复前缀的 prefill 直接省掉，见 2.4。
- chunked prefill 不减少总计算，只改变分片，对 TTFT 本身没帮助，甚至略增。
- 稀疏注意力只在模型侧支持时可用（2.6）。
- 上下文并行 / 序列并行把一个长 prefill 的 attention 切到多卡。vLLM 对 DeepSeek 类模型有 decode context parallel（`--decode-context-parallel-size`，主要针对 MLA decode 时的 KV 分摊），通用 prefill 的序列并行在不同版本和框架里支持程度不同，用前查当前文档。
- P/D 解耦：长输入短输出的 workload 本质是 prefill-bound，把 prefill 放到专门的实例池，decode 池不受影响，见 [[day24-disaggregated]]。

### 2.3 chunked prefill 分片过大或过小

**确认**：`max_num_batched_tokens` 太大则一个长 prefill 分片占满一步，同步的 decode 请求 ITL 尖峰；太小则长请求 TTFT 被拉长很多步（[[day09-scheduler]]）。

**修**：按 SLO 定。ITL 敏感就把 budget 降到 2k 到 4k；TTFT 敏感就升到 8k 以上，并用 `--max-num-partial-prefills` 和 `--long-prefill-token-threshold` 限制同时进行的长 prefill 数，防止几个长请求把 budget 全吃掉。参数名以当前版本 `--help` 为准。

### 2.4 prefix caching 在长文档场景不命中

**确认**：`vllm:prefix_cache_hits` 除以 `queries` 很低。原因通常是：文档前面有会变的内容（用户 ID、时间戳、对话轮次），把长文档放在 prompt 末尾而不是开头，或者缓存被后续请求驱逐（block 数太少、LRU）。

**修**：prompt 结构改成"固定 system 前缀 + 文档 + 变化的问题"，文档必须在变化内容之前。同一文档的请求路由到同一 replica（[[08-multi-turn-agent-workload]]）。KV 不够时用 offload connector 让驱逐的 block 落到 CPU 而不是消失。

### 2.5 RoPE 扩展配置错误

**确认**：超过模型原生训练长度（比如 32k）后输出乱码或重复，但更短时正常。或者模型卡片说明需要 YaRN 一类扩展才能到 128k，而 vLLM 默认没启用。

**修**：按模型卡片用 `--hf-overrides '{"rope_scaling": {...}}'` 或 `--rope-scaling`（旧参数名）配置，注意某些模型（如 Qwen 系列）开启静态 YaRN 后短序列质量会略降，官方建议只在确实需要长上下文时开。这是模型问题，infra 侧只能保证配置正确。

### 2.6 稀疏注意力与窗口注意力

**确认**：模型是否原生带 sliding window（Mistral、Gemma 部分层）、混合注意力（Qwen3-Next、MiniMax 的线性层）或稀疏注意力（DeepSeek V3.2 的 DSA 一类）。这些由模型决定，vLLM 通过 hybrid KV manager 和专用 backend 支持（[[day10-kv-cache]]）。

**修**：能换模型时优先选原生长上下文友好的架构。StreamingLLM / H2O 类推理时丢 KV 的方法 vLLM 主线不支持，需要改 backend，且有损，不建议生产。

### 2.7 max_model_len 导致启动失败

**确认**：启动时 profiling 用 `max_model_len` 长度做 dummy run，activation 峰值把显存占满，报 "No available memory for the cache blocks" 或 block 数极少。

**修**：降 `max_model_len`；降 `max_num_batched_tokens`（profiling 的 dummy batch 大小和它有关）；用 fp8 KV；`--gpu-memory-utilization` 适当上调但留余量。

### 2.9 多轮长 session 的 KV 生命周期

**确认**：session 越聊越长，每轮都把整段历史重新发来。命中 prefix cache 时只算新增部分，但一旦 session 的 block 被 LRU 驱逐（用户几分钟没说话，期间别的请求把 free 队列冲掉了），下一轮就是一次完整的长 prefill，用户感受是"突然卡了几秒"。

**修**：估算"活跃 session 数 × 平均历史长度"的 KV 总量，和 `# GPU blocks` 对比，差太多就上 offload connector 让驱逐的 block 落到 CPU 内存；或在网关做 session 亲和路由，避免同一 session 在多个 replica 上各存一份。客户端可以做历史摘要压缩，把老轮次替换成摘要，减少前缀长度，但这会让前缀变化、旧缓存失效，要和产品一起权衡。

### 2.10 长输出与长输入叠加

**确认**：输入 32k 且输出几千 token 的请求，decode 阶段每步都要读整段 KV，单请求 decode 也变慢，同时它长期占用大量 block 让别的请求排队。

**修**：`max_tokens` 上限按业务收紧；这类请求单独限流或走独立实例；FlashDecoding 的 split-KV 在长 KV 下对 decode 帮助明显，确认 backend 版本支持（[[day13-flash-attention]]）。

### 2.8 长上下文评测不对

**确认**：只测了 needle-in-a-haystack 就宣布支持 128k。needle 只测检索，RULER 一类多任务基准和真实业务的长文档 QA 差距很大。

**修**：性能评测覆盖四种 workload（长入短出、长入长出、多轮复用、首轮长 prefill），质量评测至少跑 RULER 或业务自己的长文档集，并在每个长度档位分别看。

## 3. 决策表

| 症状组合 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| KV 满、并发个位数、P99 长度远低于 max_model_len | 降 `max_model_len` | fp8 KV | 加 TP |
| KV 满、确实需要 128k、GQA 模型 | fp8 KV + 更多卡分摊 | 换 MLA 模型 | 降 `gpu_memory_utilization` |
| 多轮长 session、KV 反复驱逐 | KV offload connector | 增大实例 | 每轮重算 |
| 长入短出、TTFT 是唯一指标 | FA3/FlashInfer + prefix cache + 大 budget | P/D 解耦 | 小 budget |
| 长入短出 + 同实例有短对话流量 | P/D 解耦或分实例 | 小 budget + 长 prefill 数限制 | 混在一起调参 |
| 超训练长度质量崩 | 按模型卡片配 RoPE 扩展 | 换模型 | 调 infra 参数 |
| 长文档 RAG 命中率低 | 改 prompt 结构，文档前置 | 前缀亲和路由 | 关 prefix caching |

### 常见误判

- 把 TTFT 高全归给 O(n²)。8k 到 32k 这一段多数模型仍是 GEMM 主导，TTFT 线性增长是正常的，先看排队时间 `vllm:request_queue_time_seconds` 是不是才是大头。
- 把 fp8 KV 当成无损。在 64k 以上的序列末端做抽查，尤其是需要精确引用前文数字的任务。
- 把"模型支持 128k"当成"服务应该开 128k"。绝大多数流量用不到，开了只会让 profiling 和并发规划都按最坏情况走。

## 4. 验证实验

固定：模型、backend、`gpu_memory_utilization`、输出长度 64、并发 8，warmup 30 秒。

```bash
# 实验 1：TTFT 随长度的曲线，判断 O(n²) 从哪里开始主导
for L in 2048 8192 32768 65536 131072; do
  vllm bench serve --model <m> --dataset-name random --random-input-len $L \
    --random-output-len 64 --num-prompts 32 --max-concurrency 8
done
# 预期：TTFT/L 的比值在某个长度后开始上升，那就是 attention 主导点

# 实验 2：KV 手段对并发上限的影响，记录 # GPU blocks 和最大稳定并发
vllm serve <m> --max-model-len 131072
vllm serve <m> --max-model-len 131072 --kv-cache-dtype fp8
vllm serve <m> --max-model-len 32768
# 预期：fp8 让 block 数约翻倍；降 max_model_len 不改 block 数但改每请求需求

# 实验 3：prefix cache 在长文档 RAG 的收益
# 同一 30k 文档 + 20 个不同问题，文档在前 vs 文档在后各跑一次
# 预期：文档在前时第 2 个请求起 TTFT 降到几百 ms 量级；文档在后时无收益

# 实验 4：长短混合下 budget 的取舍
vllm serve <m> --max-num-batched-tokens 2048
vllm serve <m> --max-num-batched-tokens 16384
# 短请求 ITL P99 与长请求 TTFT 反向变化，选 SLO 允许的点
```

### 快速估算表（示例数字，按自己模型的 config 重算）

每 token KV 字节数 = 2 × layers × kv_heads × head_dim × bytes。下面是常见架构在 bf16 下的量级，用来判断"我的卡能放几个长请求"：

| 架构类型 | 代表 | 每 token KV（bf16） | 128k 单请求 KV | 80 GB 卡扣 16 GB 权重后可放请求数 |
|---|---|---|---:|---:|
| MHA | 早期 Llama 2 7B | 约 512 KB | 约 64 GB | 不到 1 |
| GQA，8 个 KV head | Llama 3 8B | 约 128 KB | 约 16 GB | 约 4 |
| GQA，4 个 KV head | Qwen2.5-7B | 约 56 KB | 约 7 GB | 约 9 |
| GQA，2 个 KV head | GLM-4-9B | 约 40 KB | 约 5 GB | 约 12 |
| MLA | DeepSeek-V3（TP 后每卡） | 约 70 KB / 全模型 | 视 TP 而定 | 视 TP 而定 |

fp8 KV 把每一行除以 2。这张表回答的是显存问题，不回答计算问题：MLA 省显存但 prefill 的 attention FLOPs 并没有变少。

### 排查 checklist

1. 拉 7 天输入长度分布，记下 P50/P90/P99 和最大值。
2. 用估算表算 P99 长度下每请求 KV，除进 `# GPU blocks`，得到理论并发。低于业务并发就是显存问题，先走 2.1。
3. 单请求跑 P99 长度，看 TTFT 是否可接受。不可接受走 2.2，先确认 backend 是 FA3 或 FlashInfer。
4. 看 `prefix_cache_hits/queries`。业务有重复文档但命中率低于 30%，走 2.4。
5. 长短混合流量看短请求 ITL P99，尖峰走 2.3。
6. 超过模型原生长度的请求，抽 20 条人工看质量，崩了走 2.5。
7. 以上都没问题但成本太高，评估 P/D 解耦或换 MLA 模型。

### 与 P/D 解耦的关系

长输入短输出的 workload 里，一个请求的 GPU 时间几乎全在 prefill。把它和短对话流量放在同一实例，等于让 compute-bound 的 prefill 和 memory-bound 的 decode 抢同一块 GPU，任何 budget 设置都是折中。[[day24-disaggregated]] 的 P/D 解耦在这类 workload 上收益最明确：prefill 池用大 budget 追求 TTFT 和吞吐，decode 池用小 batch 保 ITL，中间用 KV connector 传 KV。代价是 KV 传输带宽（128k 请求的 KV 是 GB 级，需要 RDMA 或 NVLink 级别的链路）和两个池的容量规划。单机小规模不值得，多机长文档服务值得。

## 5. 关联

- 总入口：[[01-diagnosis-playbook]]
- 显存与抢占：[[05-oom-preemption]]
- 延迟表现：[[02-ttft-high]]、[[03-tpot-itl-jitter]]
- 多轮 session 的 KV 复用：[[08-multi-turn-agent-workload]]
- 并行与 P/D：[[13-parallelism-choice]]、[[day24-disaggregated]]
- 机制来源：[[day09-scheduler]]、[[day10-kv-cache]]、[[day13-flash-attention]]、[[day27-long-context]]
