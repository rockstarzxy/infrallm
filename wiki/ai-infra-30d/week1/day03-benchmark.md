---
title: "Day 3: Benchmark 和指标体系"
type: concept
tags: [day3, benchmark, metrics, ttft, tpot, throughput, latency]
created: 2026-06-01
updated: 2026-06-01
---

# Day 3: Benchmark 和指标体系

## Part 1: 推理指标词典

做推理优化之前，必须先建立精确的指标体系。不同指标反映不同瓶颈，优化目标不同时关注的指标也不同。

### 延迟指标

**TTFT (Time to First Token)**
- 定义：从客户端发出请求到收到第一个生成 token 的时间
- 组成：网络延迟 + 排队等待 + prefill 计算
- 反映：prefill 性能 + 系统负载
- 重要性：用户感知的"响应速度"，在流式场景下直接决定用户体验
- 典型范围：几十 ms（短 prompt）到数秒（长 prompt / 高并发）

**TPOT (Time Per Output Token)**
- 定义：生成阶段每个 token 的平均耗时
- 计算：(最后一个 token 时间 - 第一个 token 时间) / (output_tokens - 1)
- 反映：decode 性能
- 重要性：决定流式输出的"打字速度"
- 典型范围：10-50 ms/token（取决于模型大小和并发）

**ITL (Inter-Token Latency)**
- 定义：相邻两个 token 之间的时间间隔
- 和 TPOT 的区别：TPOT 是平均值，ITL 是每一步的实际值
- 重要性：ITL 的方差反映 decode 稳定性
- 如果 ITL 波动大（有些 10ms，有些 200ms），说明有干扰（如其他请求的 prefill 插入）

**E2E Latency (End-to-End Latency)**
- 定义：从请求发出到完整响应返回的总时间
- 计算：TTFT + TPOT × output_tokens
- 重要性：非流式场景的核心指标

### 吞吐指标

**Throughput (tokens/s)**
- 定义：系统每秒生成的 token 总数（所有请求合计）
- 反映：系统的整体处理能力
- 重要性：直接关联成本（GPU 利用率）
- 注意区分：output tokens/s vs total tokens/s（是否包含 prompt tokens）

**QPS (Queries/Requests Per Second)**
- 定义：每秒处理完成的请求数
- 和 throughput 的关系：QPS × avg_output_tokens ≈ throughput

### 分位数指标

**P50 / P95 / P99**
- P50：50% 的请求在这个延迟以下（中位数）
- P95：95% 的请求在这个延迟以下
- P99：99% 的请求在这个延迟以下
- 为什么关注 P99：SLA 通常定义在 P99 级别。如果 P50=30ms 但 P99=500ms，说明有严重的尾部延迟问题

### 系统指标

| 指标 | 含义 | 采集方式 |
|---|---|---|
| GPU Utilization | GPU SM 使用率 | `nvidia-smi`, DCGM |
| GPU Memory Used | 已用显存 | `nvidia-smi` |
| KV Cache Usage | KV Cache 已用 block 占总 block 比例 | vLLM metrics |
| Requests Running | 当前正在处理的请求数 | vLLM metrics |
| Requests Waiting | 排队中的请求数 | vLLM metrics |
| Preemption Count | 被驱逐的请求数 | vLLM metrics |

---

## Part 2: 指标之间的关系

### TTFT 变差的常见原因

```
TTFT 升高
├── prompt 太长 → prefill 计算量大 → 解决：chunked prefill, prefix caching
├── 并发太高 → 排队等待 → 解决：加实例, 限流
├── KV Cache 不够 → preemption/swap → 解决：调大 gpu_memory_utilization, 量化
└── 长 prefill 阻塞 → 其他请求的 decode 被推迟 → 解决：chunked prefill
```

### Throughput vs Latency 的 trade-off

这是推理优化中最核心的权衡：

```
                高吞吐
                  ↑
                  |     ← 增大 batch size
                  |        更多请求共享 GPU
                  |        权重加载的带宽均摊
                  |
低延迟 ←──────────┼──────────→ 高延迟
                  |
                  |     ← 减小 batch size
                  |        更少排队
                  |        更少干扰
                  ↓
                低吞吐
```

- 增大 batch：throughput 上升（decode 阶段效率更高），但 TTFT 和 TPOT 都可能升高
- 减小 batch：每个请求的延迟更低，但 GPU 利用率下降，throughput 降低
- 生产中需要根据 SLA 找到平衡点

### decode 阶段 batch size 对效率的影响

回顾 Day 1 的分析：decode 阶段是 memory-bandwidth-bound。

```
batch=1:  读 16 GB 权重，做 1 个 token 的计算 → 极低效率
batch=32: 读 16 GB 权重，做 32 个 token 的计算 → 计算量增加 32x，带宽开销不变
```

所以 decode 阶段的 throughput 随 batch size 几乎线性增长（直到变成 compute-bound）。这就是 continuous batching 的价值所在——尽量让 batch 里塞满请求。

---

## Part 3: vLLM Benchmark 工具使用

### benchmark_throughput.py — 离线吞吐测试

这个工具测试的是"不考虑在线延迟，纯粹看 GPU 能跑多快"。

```bash
# 基本用法
python -m vllm.entrypoints.openai.run_batch \
  # 或直接用 benchmarks 目录下的脚本：
python benchmarks/benchmark_throughput.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 200
```

输出关键指标：
- Throughput: XX.XX requests/s, YYYY.YY tokens/s

### benchmark_serving.py — 在线 serving 压测

这个工具模拟真实的在线请求场景，最重要的 benchmark 工具。

```bash
# 前提：先启动 vLLM server
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 压测
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 200 \
  --request-rate 10
```

**关键参数说明：**

| 参数 | 含义 |
|---|---|
| `--request-rate 10` | 每秒发 10 个请求（Poisson 分布）。设 `inf` 表示尽快发完（closed-loop） |
| `--random-input-len` | 随机生成的 prompt 长度 |
| `--random-output-len` | 期望的 output 长度（实际可能不完全相同） |
| `--num-prompts` | 总共发多少个请求 |
| `--dataset-name` | `random`（随机生成）或真实数据集名 |

**输出解读：**

```
============ Serving Benchmark Result ============
Successful requests:                     200
Benchmark duration (s):                  25.32
Total input tokens:                      102400
Total generated tokens:                  25600
Request throughput (req/s):              7.90
Output token throughput (tok/s):         1011.07
Total Token throughput (tok/s):          5054.28
---------------Time to First Token----------------
Mean TTFT (ms):                          45.23
Median TTFT (ms):                        38.12
P99 TTFT (ms):                          156.78
-----------Time per Output Token------------------
Mean TPOT (ms):                          12.45
Median TPOT (ms):                        11.89
P99 TPOT (ms):                          28.34
-----------Inter-token Latency--------------------
Mean ITL (ms):                           12.45
Median ITL (ms):                         11.67
P99 ITL (ms):                           35.21
==================================================
```

---

## Part 4: Workload 矩阵实验设计

### 四组标准 workload

| 组 | Input Len | Output Len | 典型场景 | 预期特征 |
|---|---:|---:|---|---|
| A | 128 | 64 | 短对话、简单问答 | TTFT 低，TPOT 低 |
| B | 128 | 1024 | 短问长答、代码生成 | TTFT 低，E2E 长 |
| C | 4096 | 64 | RAG 检索 + 短回答 | TTFT 高（prefill 重），TPOT 低 |
| D | 4096 | 1024 | 长文档分析 | TTFT 高，E2E 最长 |

运行命令示例：
```bash
# Workload A
python benchmarks/benchmark_serving.py \
  --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
  --endpoint /v1/completions --dataset-name random \
  --random-input-len 128 --random-output-len 64 \
  --num-prompts 200 --request-rate 10

# Workload B
# ...把 --random-output-len 改为 1024

# Workload C
# ...把 --random-input-len 改为 4096, output 改为 64

# Workload D
# ...input 4096, output 1024
```

### 预期观察

1. **Workload C vs A**：input 变长 32x，TTFT 应该显著增加（prefill 计算量增大），但 TPOT 变化不大（decode 不受 input 长度影响太多——不过 attention 计算会增加）
2. **Workload B vs A**：output 变长 16x，E2E latency 增加约 16x，但 TTFT 不变
3. **Workload D**：TTFT 和 E2E 都很长，是系统压力最大的场景

### 并发梯度实验

固定 workload（input=512, output=128），变化 request-rate：

```bash
for rate in 1 5 10 20 50 100; do
  python benchmarks/benchmark_serving.py \
    --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
    --endpoint /v1/completions --dataset-name random \
    --random-input-len 512 --random-output-len 128 \
    --num-prompts 200 --request-rate $rate
done

# 再跑一次 inf（closed-loop，尽快发完）
python benchmarks/benchmark_serving.py \
  --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
  --endpoint /v1/completions --dataset-name random \
  --random-input-len 512 --random-output-len 128 \
  --num-prompts 200 --request-rate inf
```

**预期观察：**
- request-rate 较低时（如 1-5/s）：TTFT 和 TPOT 都很稳定，系统没有压力
- request-rate 接近系统吞吐上限时：TTFT 开始升高（请求排队），P99 和 P50 差距拉大
- request-rate 超过上限时：TTFT 急剧升高，可能出现 preemption
- 拐点所在的 QPS 就是当前配置下的系统容量

---

## Part 5: 简易自写压测脚本

如果不方便用 vLLM 的 benchmark 脚本，可以自己写一个简单的：

```python
import asyncio
import time
import aiohttp
import json
import statistics

async def send_request(session, url, model, input_len, output_len):
    prompt = "x " * input_len  # 简单的 padding prompt
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
    }

    t_start = time.perf_counter()
    t_first_token = None
    token_times = []

    async with session.post(url, json=payload) as resp:
        async for line in resp.content:
            line = line.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            if line == "data: [DONE]":
                break

            now = time.perf_counter()
            if t_first_token is None:
                t_first_token = now
            else:
                token_times.append(now)

    t_end = time.perf_counter()

    ttft = (t_first_token - t_start) * 1000 if t_first_token else None
    e2e = (t_end - t_start) * 1000
    num_tokens = len(token_times) + 1
    tpot = ((t_end - t_first_token) * 1000 / len(token_times)) if token_times else None

    return {"ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": e2e, "tokens": num_tokens}


async def benchmark(url, model, input_len, output_len, num_requests, qps):
    connector = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(num_requests):
            task = asyncio.create_task(
                send_request(session, url, model, input_len, output_len)
            )
            tasks.append(task)
            if qps != float('inf'):
                await asyncio.sleep(1.0 / qps)

        results = await asyncio.gather(*tasks)

    # 汇总
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"]]
    tpots = [r["tpot_ms"] for r in results if r["tpot_ms"]]

    print(f"TTFT  P50={statistics.median(ttfts):.1f}ms  "
          f"P95={sorted(ttfts)[int(0.95*len(ttfts))]:.1f}ms  "
          f"P99={sorted(ttfts)[int(0.99*len(ttfts))]:.1f}ms")
    print(f"TPOT  P50={statistics.median(tpots):.1f}ms  "
          f"P95={sorted(tpots)[int(0.95*len(tpots))]:.1f}ms")


if __name__ == "__main__":
    asyncio.run(benchmark(
        url="http://localhost:8000/v1/completions",
        model="Qwen/Qwen2.5-7B-Instruct",
        input_len=512,
        output_len=128,
        num_requests=100,
        qps=10,
    ))
```

---

## Part 6: 读懂 vLLM Prometheus Metrics

在压测过程中同时观察 vLLM 的实时指标：

```bash
# 持续监控（每 2 秒刷新）
watch -n 2 'curl -s http://localhost:8000/metrics | grep -E "vllm:(num_requests|kv_cache|gpu_cache|time_to_first|inter_token|e2e|tokens|preemption)"'
```

关键指标：

```
# 当前活跃/排队请求
vllm:num_requests_running    ← 正在 GPU 上执行的
vllm:num_requests_waiting    ← 排队等待的（KV Cache 不够或达到 max_num_seqs）

# KV Cache 压力
vllm:kv_cache_usage_perc     ← 新版本常见名称
vllm:gpu_cache_usage_perc    ← 旧版本常见名称

# 延迟
vllm:time_to_first_token_seconds
vllm:inter_token_latency_seconds
vllm:e2e_request_latency_seconds

# Preemption（请求被驱逐）
vllm:num_preemptions_total 或相关 preemption 指标
```

**读指标的思路：**
- `waiting > 0` 且持续增长 → 系统过载，需要降低 QPS 或加实例
- `kv_cache_usage_perc/gpu_cache_usage_perc > 0.95` → KV Cache 快满了，可能触发 preemption
- `preemptions > 0` → 有请求被驱逐重做，严重浪费计算

---

## 交付物

| 文件 | 描述 |
|---|---|
| `baseline-benchmark-report.md` | 4 组 workload 的 TTFT/TPOT/throughput 数据 + 分析 |
| `并发拐点分析` | request-rate 梯度实验的数据表 + 拐点 QPS 判断 |
| `指标监控截图` | 压测过程中 vLLM metrics 的关键数值 |

## 自检问题

1. TTFT 包含哪几部分时间？哪部分最可能成为瓶颈？
2. throughput (tokens/s) 和 latency (TPOT) 能同时优化吗？为什么？
3. 为什么 P99 和 P50 差距越大越值得警惕？
4. request-rate=inf 和 request-rate=10 测的是不同的什么？
5. 如果 `vllm:num_preemptions_total` 在涨，你会怎么排查？
