---
title: "Day 2: vLLM 单卡部署和 OpenAI-compatible API"
type: concept
tags: [day2, vllm, deployment, api, serving]
created: 2026-06-01
updated: 2026-06-01
---

# Day 2: vLLM 单卡部署和 OpenAI-compatible API

## Part 1: vLLM 是什么

vLLM 是一个高性能的 LLM 推理和 serving 引擎。它的核心创新是 **PagedAttention**（Day 4 深入），但作为工程系统它还集成了：

- OpenAI-compatible API server
- Continuous batching（持续批处理）
- 多种 attention backend（FlashAttention, FlashInfer 等）
- 量化支持（GPTQ, AWQ, FP8 等）
- Tensor Parallel / Pipeline Parallel 分布式推理
- Prefix caching（前缀缓存）
- Speculative decoding（推测解码）

**vLLM 的两种使用模式：**

1. **Offline inference**（离线推理）：批量处理一组 prompt，不需要 server
2. **Online serving**（在线服务）：启动一个 HTTP server，对外提供 API

本课程主要使用 online serving 模式，因为它更接近生产环境。

---

## Part 2: 安装和环境准备

### 前置条件

- NVIDIA GPU，CUDA 12.1+
- Python 3.9+
- 推荐显存：24 GB 起步（可跑 7B/8B 模型）

### 安装

```bash
# 方式 1：pip（推荐）
pip install vllm

# 方式 2：Docker
docker run --runtime nvidia --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct

# 验证
python -c "import vllm; print(vllm.__version__)"
```

### 模型选择

对于学习用途，推荐以下模型（按显存需求排序）：

| 模型 | 显存需求 (FP16) | KV heads | 特点 |
|---|---:|---:|---|
| Qwen2.5-3B-Instruct | ~6 GB | 2 | 显存不够时的备选 |
| Qwen2.5-7B-Instruct | ~14 GB | 4 | 适合 24GB GPU |
| Llama-3.1-8B-Instruct | ~16 GB | 8 | 业界标准，适合 24GB GPU |
| Mistral-7B-Instruct-v0.3 | ~14 GB | 8 | 带 sliding window attention |

---

## Part 3: 启动 vLLM Serving

### 最简启动

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
```

首次启动会从 Hugging Face 下载模型（需要网络），之后会使用本地缓存。

### 带参数启动

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-seqs 64
```

### 启动日志解读

启动时 vLLM 会打印关键信息，学会读这些日志很重要：

```
# 模型加载信息
INFO: Loading model weights took X.XX GB

# KV Cache 分配信息 ← 非常重要
INFO: GPU KV cache memory: X.XX GiB
INFO: Maximum number of KV cache tokens: XXXXX
INFO: # GPU blocks: XXXX, # CPU blocks: XXXX

# 这告诉你：
# 1. 显存中除了模型权重，还剩多少给 KV Cache
# 2. 能存多少 token 的 KV Cache → 直接决定了并发能力
```

**GPU blocks 数量的意义：**
- vLLM 把 KV Cache 显存切成固定大小的 block（类似操作系统的内存页）
- 每个 block 存放若干个 token 的 KV
- blocks 数量决定了系统同时能处理的总 token 数上限
- 当 blocks 用完时，新请求会排队或触发 preemption（驱逐已有请求）

---

## Part 4: 核心启动参数详解

### `--gpu-memory-utilization` (默认 0.9)

控制 vLLM 使用 GPU 显存的比例。

```
总 GPU 显存
├── 模型权重（固定）
├── KV Cache（由这个参数控制上限）
├── 临时计算缓冲（activation memory）
└── 预留空间（1 - gpu_memory_utilization）
```

- 设 0.9：用 90% 显存，留 10% 给系统和其他进程
- 设太高（如 0.95）：可能因碎片导致 OOM
- 设太低（如 0.7）：KV Cache 空间不足，并发能力下降

**实际影响**：这个值主要决定 KV Cache 的大小，进而决定能同时处理多少请求。

### `--max-model-len`（默认值取决于模型 config 和 vLLM 版本）

单个请求允许的最大上下文长度（input + output tokens 之和）。

- 设太大：单请求最大上下文能力变强，但会增加内存压力，可能降低可承载并发，并影响 profiling、CUDA graph 和调度配置
- 设太小：长 prompt 会被拒绝
- 建议：根据实际业务需求设置，不要盲目用模型最大值

例如模型支持 128K 上下文，但业务最长只需要 8K，则设 `--max-model-len 8192`，通常能降低内存压力并提升可用并发。实际收益要看启动日志里的 KV block 数和压测结果。

### `--max-num-seqs`（默认值取决于 vLLM 版本、后端和使用场景）

单次 scheduler iteration 中最多并发处理多少个请求。

- 设太大：单次 iteration 计算量增大，每个请求的延迟可能增加
- 设太小：无法充分利用 GPU 算力（尤其在 decode 阶段，batch 越大越能摊薄带宽开销）
- 和 KV Cache 空间有联动：即使设了 256，如果 block 不够，实际也无法达到

### `--max-num-batched-tokens`

单次 iteration 中最多处理多少 token（包括 prefill 和 decode 的 token 总和）。

- 主要影响 prefill 阶段：一个长 prompt 的 prefill 如果超过这个值，需要用 chunked prefill 分片
- 不设此参数时，vLLM 有内部默认逻辑
- 设太小：长 prompt 的 TTFT 可能增加（需要多步完成 prefill）
- 设太大：单次 iteration 耗时增加，影响所有请求的 TPOT

### `--enable-prefix-caching`

开启或显式配置自动前缀缓存（Automatic Prefix Caching, APC）。注意：新版本 vLLM 在模型支持时可能默认开启；实验时要记录版本、启动日志和 cache hit 行为。

- 当多个请求有共同前缀（如相同的 system prompt）时，缓存共享部分的 KV Cache
- 后续请求可以跳过已缓存前缀的 prefill 计算，大幅降低 TTFT
- 适合：RAG 场景（大量请求用相同 context）、固定 system prompt 的 chatbot
- Day 10 会详细测试

### `--enable-chunked-prefill`

开启或显式配置分块 prefill。注意：vLLM V1 在支持时通常默认启用 chunked prefill；真正要调的是 `max_num_batched_tokens` 如何影响 TTFT、ITL/TPOT 和吞吐。

- 将长 prompt 的 prefill 拆成多个 chunk，每个 chunk 和 decode 请求一起调度
- 好处：避免一个超长 prompt 的 prefill 阻塞所有 decode 请求（改善短请求的 TPOT）
- 代价：长 prompt 自身的 TTFT 可能略增
- Day 10 会详细测试

### `--dtype` (默认 auto)

模型权重的数据类型：
- `auto`：使用模型 config 中指定的类型（通常是 BF16 或 FP16）
- `float16` / `bfloat16`：显式指定
- `float32`：几乎不用，显存翻倍

### `--enforce-eager`

同时禁用 torch.compile 和 CUDA Graph，使用 eager mode 执行。

- CUDA Graph：将一系列 GPU 操作"录制"成一个 graph，后续回放跳过 CPU launch 开销
- torch.compile：算子融合和 Triton kernel 生成
- V1 默认两者都开，用 piecewise CUDA graph 加速 decode（Day 13 讲机制）
- `--enforce-eager` 适合调试或遇到兼容性问题时使用；只想关其中一个用 `--compilation-config`
- 性能影响：CUDA Graph 通常带来 10-30% 的 decode 加速，启动时多几十秒的编译和录制时间

### V1 时代的其他关键参数（Week 2 逐个展开）

| 参数 | 作用 | 展开 |
|---|---|---|
| `--compilation-config '{"cudagraph_mode":...}'` / `-O3` | torch.compile 级别、CUDA graph 模式和分桶 | Day 13 |
| `--async-scheduling` | 调度和 GPU 执行重叠，降低 CPU 空隙 | Day 9 |
| `--scheduling-policy fcfs|priority` | 请求出队和抢占顺序 | Day 9 |
| `--speculative-config '{"method":"ngram",...}'` | 推测解码（V1 统一入口，替代旧的 `--speculative-model`） | Day 12 |
| `--structured-outputs-config '{"backend":"xgrammar"}'` | 结构化输出后端 | Day 13 |
| `--kv-transfer-config '{"kv_connector":...}'` | KV connector：P/D 解耦、offload | Day 10 / 24 |
| `--data-parallel-size N` | 引擎原生数据并行 | Day 18 |
| `--api-server-count N` | 多个前端进程分担 tokenize/detokenize | Day 8 |
| `--no-enable-prefix-caching` | V1 默认开 prefix caching，这是关闭开关 | Day 10 |
| `--logits-processors` / `--scheduler-cls` | 插件式扩展点 | Day 13 / 9 |

不要求今天记住，知道有这些开关、知道它们在哪一天讲即可。

---

## Part 5: 发送请求

### curl 调用

```bash
# Chat Completions API（推荐）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain KV Cache in transformer inference."}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
  }'
```

```bash
# Streaming 模式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Count from 1 to 20."}],
    "max_tokens": 128,
    "stream": true
  }' --no-buffer
```

### Python OpenAI SDK 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# 非流式
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "What is vLLM?"}],
    max_tokens=256,
    temperature=0.7,
)
print(response.choices[0].message.content)
print(f"Usage: {response.usage}")  # prompt_tokens, completion_tokens, total_tokens

# 流式
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Explain PagedAttention."}],
    max_tokens=256,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 响应格式解读

```json
{
  "id": "cmpl-xxx",
  "object": "chat.completion",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"  // "stop" | "length" | null(streaming)
  }],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170
  }
}
```

- `finish_reason: "stop"` → 模型自然结束（遇到 EOS token）
- `finish_reason: "length"` → 达到 max_tokens 限制被截断

---

## Part 6: 有用的管理接口

vLLM serving 还暴露了一些管理接口：

```bash
# 查看当前加载的模型
curl http://localhost:8000/v1/models

# 健康检查
curl http://localhost:8000/health

# 查看 metrics（Prometheus 格式）
curl http://localhost:8000/metrics
```

Metrics 中的关键指标（Day 3 会详细学）：
- `vllm:num_requests_running` — 当前正在运行的请求数
- `vllm:num_requests_waiting` — 排队等待的请求数
- `vllm:kv_cache_usage_perc` 或旧版 `vllm:gpu_cache_usage_perc` — KV Cache 使用率
- `vllm:inter_token_latency_seconds` / `vllm:time_to_first_token_seconds` / `vllm:e2e_request_latency_seconds` — 延迟指标
- 吞吐相关指标名随版本变化，必要时直接在 `/metrics` 中搜索 `tokens`、`throughput`、`generation`

---

## 交付物

| 文件 | 描述 |
|---|---|
| `vLLM 单卡启动命令清单.md` | 记录：GPU 型号、显存、CUDA 版本、vLLM 版本、使用的模型、启动参数、日志中的 block 数量 |
| `请求测试记录` | 非流式和流式请求各一个成功返回的记录 |

## 自检问题

1. `--gpu-memory-utilization 0.9` 意味着什么？模型权重算在这 90% 里面吗？
2. 日志中 "# GPU blocks: 2000" 这个数字意味着什么？如果 block_size=16，能同时缓存多少 token？
3. 为什么不建议把 `--max-model-len` 设成模型最大值（如 128K）？
4. `--enforce-eager` 关掉 CUDA Graph 后性能会变化多少？在什么场景下需要开启它？
5. Streaming 和非 Streaming 模式对 TTFT 有影响吗？
