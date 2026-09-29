---
title: "优化专项 01: 通用诊断流程"
type: concept
tags: [ai-infra, inference-optimization, diagnosis, profiling, metrics]
created: 2026-09-29
updated: 2026-09-29
---

# 优化专项 01: 通用诊断流程

> 一句话：任何"推理服务慢/贵/挂"的问题，先用这一页把它归到四个象限之一，再进对应专项页。关联日课：[[day03-benchmark]]、[[day05-profiling]]、[[day08-vllm-architecture]]、[[day26-roofline]]。

## 1. 先固定问题的定义

优化前先把"慢"翻译成指标，否则后面每一步都在换目标。

| 用户说的 | 翻译成 | 指标 |
|---|---|---|
| 首字慢 | TTFT P50 / P99 | `vllm:time_to_first_token_seconds` |
| 打字一顿一顿 | ITL P99 与 P50 的比值 | `vllm:inter_token_latency_seconds`（旧版 `time_per_output_token_seconds`） |
| 整体慢 | E2E P99，拆成排队 + prefill + decode | `vllm:e2e_request_latency_seconds`、`request_queue_time_seconds`、`request_prefill_time_seconds`、`request_decode_time_seconds` |
| 扛不住并发 | 固定 SLO 下的最大 request/s 或 output tokens/s | `vllm:generation_tokens_total` 的速率、`num_requests_running/waiting` |
| 太贵 | 每百万 token 的 GPU 小时 | 见 [[15-cost-per-token]] |
| 报错 | OOM / 抢占 / 超时 / 429 | `num_preemptions_total`、日志、网关指标 |

同时固定 workload：输入/输出长度分布、并发模型（固定并发还是固定到达率）、是否流式、采样参数、是否结构化输出。这些任何一个变了，后面的对比都无效。

## 2. 四象限归类

把 `/metrics` 和 `nvidia-smi dmon` 各看 60 秒，按下表归类：

```
                     GPU SM 利用率高                    GPU SM 利用率低
                 ┌──────────────────────────┬──────────────────────────┐
 waiting 队列长  │  A. 算力/带宽真的不够       │  C. 调度或 KV 卡住          │
 或 KV 使用率高  │  prefill-bound / decode-   │  KV 容量、抢占、budget、     │
                 │  bound，见 02/04/07         │  长 prefill 独占，见 03/05   │
                 ├──────────────────────────┼──────────────────────────┤
 waiting 为 0    │  B. 正常满载               │  D. CPU / 网络 / 客户端      │
 KV 使用率不高   │  想再快只能换硬件、量化、     │  EngineCore 或前端 CPU 100%、│
                 │  spec decode，见 10/11      │  tokenize、SSE、LB，见 06    │
                 └──────────────────────────┴──────────────────────────┘
```

### A 象限再细分：prefill-bound 还是 decode-bound

| 看什么 | prefill-bound | decode-bound |
|---|---|---|
| workload | 长输入短输出（RAG、摘要、agent） | 短输入长输出（写作、代码生成） |
| `request_prefill_time` vs `request_decode_time` | prefill 占大头 | decode 占大头 |
| 加并发的效果 | TTFT 线性变差，吞吐几乎不涨 | TPOT 缓慢变差，吞吐明显涨 |
| roofline 位置 | 算术强度高，靠算力 | 算术强度低，靠 HBM 带宽 |
| 主要方法 | prefix caching、更多算力（DP）、P/D 解耦、FA3 | 更大 batch、KV/权重量化、spec decode、MTP |

### C 象限的三种情况

1. **KV 容量**：`kv_cache_usage_perc` 长期 90% 以上且 `num_preemptions_total` 增长。进 [[05-oom-preemption]]。
2. **budget 太小**：长 prompt 被切成很多步，`request_prefill_time` 高但 GPU 不满。进 [[02-ttft-high]]。
3. **单个长 prefill 独占**：ITL 尖峰与长请求到达时间重合。进 [[03-tpot-itl-jitter]]。

### D 象限的判定

```bash
# EngineCore 进程 CPU
top -p $(pgrep -f "EngineCore" | head -1)          # 接近 100% 即 CPU-bound
# 前端进程 CPU（tokenize / detokenize / SSE）
top -p $(pgrep -f "vllm serve" | head -1)
# 客户端侧：对比 vllm bench 的 TTFT 和网关记录的 TTFT，差值就是网络与 LB
```

## 3. 工具与用法

| 层次 | 工具 | 看什么 | 用在哪个象限 |
|---|---|---|---|
| 服务指标 | `/metrics` + Prometheus/Grafana | 上面所有 `vllm:*` | 全部，第一步 |
| GPU 粗粒度 | `nvidia-smi dmon -s pucm` | SM%、显存带宽%、功耗 | 分 A/C 与 B/D |
| 进程 | `py-spy dump/record` | Python 栈：schedule、_prepare_inputs、detokenize | D |
| 时间线 | `nsys profile -t cuda,nvtx,osrt` 或 vLLM `--profiler-config` / `VLLM_TORCH_PROFILER_DIR` | kernel 之间的空隙、NCCL 占比、launch 次数 | A、D、13 |
| kernel | `ncu` | 单 kernel 的算术强度与带宽利用 | 只在写 kernel 时 |
| 网络 | `nvidia-smi topo -m`、`nccl-tests`、`ib_write_bw` | 拓扑、集合通信带宽 | 13 |
| 应用 | 网关 access log、trace id | 排队在网关还是引擎 | D |

`nsys` 最小可用命令，抓 20 秒压测期间的时间线：

```bash
nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o vllm-trace python -m vllm.entrypoints.openai.api_server --model ... &
# 另一个终端跑 vllm bench serve；结束后
nsys stats --report cuda_gpu_kern_sum vllm-trace.nsys-rep | head -30
```

看三件事：GPU 时间线上有没有周期性空隙（有则 D 或 C）、NCCL kernel 占比（超过 20% 进 [[13-parallelism-choice]]）、attention kernel 名字（确认 backend）。

## 4. 假设验证循环

```
1. 写下假设："TTFT 高是因为 prefix cache 未命中"
2. 找一个能证伪的指标：prefix_cache_hits / prefix_cache_queries
3. 最小改动：只改一个参数或只换一个 workload
4. 固定 warmup 后跑 3 次取中位数
5. 结果进表：配置 | 指标 | 变化 | 结论
6. 假设成立 → 进专项页看副作用；不成立 → 回到象限表换下一个假设
```

常见的错误顺序：先改一堆参数再压测。这样即使变好了也不知道是哪个起作用，下次换 workload 就复现不了。

## 5. 一张总表：症状 → 首选方法

| 象限 | 症状 | 首选 | 次选 | 不要做 |
|---|---|---|---|---|
| A prefill | TTFT 随并发线性恶化 | prefix caching + prefix-aware routing | 加 DP replica、P/D 解耦 | 加 TP（通信涨、算力没多） |
| A decode | TPOT 随并发缓慢恶化、吞吐仍在涨 | 加 `max_num_seqs`，直到 TPOT 碰 SLO | KV FP8、权重 FP8、MTP/spec decode | 减小 budget（对 decode 无关） |
| C KV | 抢占增长 | 降 `max_model_len` 到真实分布 P99 | KV FP8、加显存/卡 | 调 `gpu_memory_utilization` 到 0.98 |
| C budget | 长 prompt TTFT 高、GPU 不满 | 调大 `max_num_batched_tokens` | 长 prefill 保护参数 | 关 chunked prefill |
| D CPU | EngineCore 100% | CUDA graph 全开、async scheduling | `--api-server-count`、DP | 加 GPU |
| D 网络 | 引擎 TTFT 正常、客户端 TTFT 高 | 检查 LB 与 SSE flush | 就近部署 | 调引擎参数 |
| B | 一切正常但想更快 | 量化、spec decode、更新硬件 | 换 attention backend | 无根据地调参 |

## 6. 验证实验：建立自己的基线

在任何优化前先做一次，保存为对照：

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct
for conc in 1 4 16 64; do
  vllm bench serve --model $MODEL --dataset-name random \
    --random-input-len 1024 --random-output-len 256 \
    --num-prompts $((conc*10)) --max-concurrency $conc \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --save-result --result-filename baseline-c$conc.json
done
```

同时抓一份 `/metrics` 快照和 `nvidia-smi dmon` 60 秒。把四个并发档的 TTFT/TPOT P50/P99 和 tokens/s 画成表，这就是你的 roofline 曲线的经验版：吞吐在哪个并发档停止增长，TPOT 在哪个档碰到 SLO。后面每个专项页的实验都和这张表比。

## 7. 关联

下一页按症状选：[[02-ttft-high]]、[[03-tpot-itl-jitter]]、[[04-throughput-low]]、[[05-oom-preemption]]、[[06-gpu-util-low-cpu-bound]]。总入口 [[00-index]]。
