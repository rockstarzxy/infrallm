---
title: "Day 5: Profiling 基础"
type: concept
tags: [day5, profiling, gpu, nvidia-smi, nsight, pytorch-profiler]
created: 2026-06-01
updated: 2026-06-01
---

# Day 5: Profiling 基础

## Part 1: GPU 硬件指标基础

在做推理优化之前，你需要理解 GPU 的几个关键硬件指标。

### GPU 架构速览

以 NVIDIA A100 和 H100 为例：

| 指标 | A100 (80GB) | H100 (80GB) | 意义 |
|---|---:|---:|---|
| FP16 算力 (TFLOPS) | 312 | 989 | 能做多少浮点运算/秒 |
| BF16 算力 (TFLOPS) | 312 | 989 | 同上，BF16 格式 |
| FP8 算力 (TFLOPS) | — | 1,979 | H100 新增，FP8 翻倍 |
| HBM 带宽 (TB/s) | 2.0 | 3.35 | 显存读写速度 |
| HBM 容量 (GB) | 80 | 80 | 显存总量 |
| NVLink 带宽 (GB/s) | 600 | 900 | GPU 间通信速度 |
| SM 数量 | 108 | 132 | 流多处理器个数 |
| Tensor Core | 3rd Gen | 4th Gen | 矩阵计算专用硬件 |

### 关键概念

**SM (Streaming Multiprocessor) Utilization**
- GPU 由多个 SM 组成，每个 SM 能并行执行多个 warp（32 个线程一组）
- SM Utilization 表示"有多少 SM 在干活"
- 100% 不代表 GPU 被用满了——可能每个 SM 都没有跑满

**GPU Utilization (nvidia-smi 显示的)**
- nvidia-smi 显示的 "GPU-Util" 表示"在采样周期内 GPU 是否有 kernel 在执行"
- 这是一个非常粗糙的指标：只要有一个 kernel 在跑就可能显示 100%
- 不能用来判断 GPU 是否被高效利用

**HBM Bandwidth Utilization**
- 实际 HBM 读写速度 / 峰值 HBM 带宽
- 在 decode 阶段（memory-bound），这个指标反映了 GPU 有多"满载"
- 如果 bandwidth utilization 很高但 throughput 不增加，说明已经到了带宽天花板

**Occupancy**
- 每个 SM 上实际活跃的 warp 数 / SM 最大 warp 数
- 低 occupancy 意味着 GPU 的并行能力没有被充分使用
- 但高 occupancy 不一定代表高性能（还要看每个 warp 的效率）

---

## Part 2: nvidia-smi 和 DCGM

### nvidia-smi 基本用法

```bash
# 一次性查看 GPU 状态
nvidia-smi

# 持续监控（每秒刷新）
nvidia-smi -l 1

# 只看关键指标
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 1

# 查看 GPU 拓扑（多卡时很重要，Day 15 会用到）
nvidia-smi topo -m
```

### nvidia-smi 输出解读

```
+-------------------------------------------+
| GPU  Name        | Util% | Memory-Usage   |
|   0  A100-SXM4   |  95%  | 72000/81920MiB |
+-------------------------------------------+

Util%: GPU 计算利用率（粗糙指标）
Memory-Usage: 72000 MB 已用 / 81920 MB 总显存
```

注意：`Memory-Usage` 包含了模型权重 + KV Cache + 临时计算缓冲 + 框架开销。

### DCGM (Data Center GPU Manager)

DCGM 提供比 nvidia-smi 更细粒度的指标：

```bash
# 安装（如果有）
# 查看 SM 活跃度、Tensor Core 利用率、内存带宽利用率等
dcgmi dmon -e 1001,1002,1003,1004,1005 -d 1000
```

DCGM 指标编号：
- 1001: GPU Utilization
- 1002: Memory Utilization
- 1003: SM Activity
- 1004: SM Occupancy
- 1005: Tensor Active（Tensor Core 利用率）

在推理场景下：
- prefill 阶段：Tensor Active 高（大矩阵乘法），SM Activity 高
- decode 阶段：Tensor Active 低（矩阵向量乘），Memory Bandwidth 高

---

## Part 3: PyTorch Profiler

PyTorch Profiler 可以看到每个 kernel 的耗时和显存变化。

### 在 vLLM 中使用 profiler

vLLM 支持通过环境变量或代码启用 profiling：

```bash
# 方式 1：使用 VLLM 内置的 profiling 支持
# 查看 vLLM 文档中 profiling 相关章节

# 方式 2：用 torch.profiler 包装推理代码（适合 offline inference）
```

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# 示例：对一个简单的 attention 操作做 profiling
import torch.nn.functional as F

seq_len = 2048
d_model = 4096
n_heads = 32
d_head = d_model // n_heads
batch = 1

Q = torch.randn(batch, n_heads, seq_len, d_head, device='cuda', dtype=torch.float16)
K = torch.randn(batch, n_heads, seq_len, d_head, device='cuda', dtype=torch.float16)
V = torch.randn(batch, n_heads, seq_len, d_head, device='cuda', dtype=torch.float16)

# Warmup
for _ in range(3):
    _ = F.scaled_dot_product_attention(Q, K, V)
torch.cuda.synchronize()

# Profile
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    with record_function("attention"):
        output = F.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 导出 Chrome trace（可在 chrome://tracing 中可视化）
prof.export_chrome_trace("attention_trace.json")
```

### Profiler 输出解读

```
---------------------------------  --------  --------  --------
                             Name  CPU Time  CUDA Time  # Calls
---------------------------------  --------  --------  --------
                        attention    1.2ms     0.8ms         1
       aten::scaled_dot_product_   0.1ms     0.7ms         1
          flash_fwd_kernel          0.05ms    0.65ms        1
---------------------------------  --------  --------  --------
```

关键看：
- **CUDA Time**：GPU 上的实际执行时间（不含 CPU 调度）
- **CPU Time**：包含 CPU 端调度开销
- **kernel 名称**：`flash_fwd_kernel` 说明底层用了 FlashAttention

---

## Part 4: Nsight Systems（粗粒度 GPU Timeline）

Nsight Systems 是 NVIDIA 的系统级 profiler，能看到完整的 GPU timeline。

### 基本用法

```bash
# 对 vLLM offline inference 做 profiling
nsys profile -o vllm_profile \
  python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-3B-Instruct')
outputs = llm.generate(['Hello, how are you?'] * 4, SamplingParams(max_tokens=64))
"

# 生成 .nsys-rep 文件，用 Nsight Systems GUI 打开查看 timeline
```

### Timeline 解读

在 Nsight Systems GUI 中能看到：

```
CPU Thread ─────────────────────────────────────────────
  │ Python │ Scheduler │ Tokenizer │ Python │ ...
  └────────┘           └───────────┘

CUDA Stream ────────────────────────────────────────────
  │  GEMM  │ GEMM │ softmax │ GEMM │ layernorm │ GEMM │ ...
  └────────┘                                     

  ← prefill kernels (大 GEMM) →  ← decode kernels (小 GEMM) →
```

观察重点：
1. **Prefill vs Decode 的 kernel 差异**：prefill 的 GEMM kernel 比 decode 的大很多
2. **CPU-GPU 同步点**：CPU 等 GPU 的间隙说明有 sync 开销
3. **Kernel launch gap**：kernel 之间的空隙说明 CPU 调度跟不上（CUDA Graph 可以消除）
4. **内存拷贝**：H2D / D2H 拷贝如果频繁说明有性能问题

---

## Part 5: 推理阶段的瓶颈判断

### 一次完整推理请求的耗时分解

```
总延迟 (E2E Latency)
├── 排队时间 (Queue Time)           ← scheduler-bound
├── Prefill 时间                    ← compute-bound（通常）
│   ├── QKV Projection (GEMM)
│   ├── Attention (FlashAttention)
│   ├── FFN (GEMM × 3)
│   └── × n_layers
├── Decode 时间 × output_tokens     ← memory-bandwidth-bound（通常）
│   ├── QKV Projection (GEMV)
│   ├── Attention (对所有 KV Cache)
│   ├── FFN (GEMV × 3)
│   ├── × n_layers
│   └── Sampling
├── 网络传输时间                     ← 通常可忽略
└── Tokenize/Detokenize             ← 通常可忽略
```

### 判断瓶颈的方法

**方法 1：看 CUDA Time vs 理论计算时间**

Prefill 阶段理论 FLOPS（近似，忽略 attention 中 O(n²) 部分）：
```
FLOPS ≈ 2 × model_params × seq_len
Llama-3-8B, seq_len=2048:
  ≈ 2 × 8B × 2048 = 32.8 TFLOPS

H100 FP16 峰值: 989 TFLOPS
理论 prefill 时间: 32.8 / 989 ≈ 33 ms

如果实际 prefill 时间远大于 33 ms，说明还没有做到 compute-bound。
```

**方法 2：看 bandwidth utilization**

Decode 阶段（batch=1）的理论带宽需求：
```
每步需读取的数据 ≈ model_params × bytes_per_element + KV_cache
Llama-3-8B FP16: ≈ 16 GB + KV_cache

H100 HBM 带宽: 3.35 TB/s
理论 decode 时间/token: 16 GB / 3.35 TB/s ≈ 4.8 ms

如果实际 TPOT ≈ 5-6 ms（接近理论值），说明确实是 bandwidth-bound。
```

**方法 3：看 batch size 对 throughput 的影响**

```
如果 batch 从 1 增加到 8，throughput 接近 8x → memory-bound（带宽被摊薄）
如果 batch 从 32 增加到 64，throughput 不到 2x → 开始转为 compute-bound
```

### 各模型在不同硬件上的典型瓶颈

| 模型 | 硬件 | Prefill 瓶颈 | Decode 瓶颈 (batch=1) | Decode 瓶颈 (batch=64) |
|---|---|---|---|---|
| Qwen2.5-7B | A100 | Compute | Memory BW | 接近 Compute |
| Qwen2.5-72B (TP=8) | 8×A100 | Compute | Memory BW | Memory BW |
| DeepSeek-V3 (EP) | 多节点 H100 | Compute | Memory BW + 通信 | 通信 |

---

## Part 6: 实操——分析一次 vLLM 推理

### Step 1: 基础监控

在运行 vLLM serving 时，开一个终端持续监控：

```bash
# 终端 1：GPU 硬件指标
nvidia-smi -l 1

# 终端 2：vLLM metrics
watch -n 2 'curl -s http://localhost:8000/metrics | grep -E "(num_requests|cache_usage|throughput|preemption)"'

# 终端 3：发压测请求（Day 3 的 benchmark 脚本）
```

### Step 2: 分析 prefill vs decode

发两种不同的请求：
```bash
# 长 prefill 请求（观察 prefill 耗时）
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "'$(python -c "print('hello ' * 2000)")'", "max_tokens": 10}'

# 短 prefill + 长 decode（观察 decode 行为）
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "Count from 1 to 1000:", "max_tokens": 2000}'
```

### Step 3: 记录观察

创建一个表格记录：

| 请求类型 | Input Len | Output Len | TTFT | TPOT | GPU Util | Memory Used | KV Cache % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 长 prefill | 2000 | 10 | ? | ? | ? | ? | ? |
| 短 prefill + 长 decode | 10 | 2000 | ? | ? | ? | ? | ? |
| 混合并发 | mix | mix | ? | ? | ? | ? | ? |

---

## 交付物

| 文件 | 描述 |
|---|---|
| `profile-notes.md` | 记录 profiling 过程、观察到的数据、瓶颈判断 |
| `瓶颈分析` | 解释当前模型+硬件配置下 prefill 和 decode 分别是什么 bound |

## 自检问题

1. nvidia-smi 显示 GPU Util=100% 能说明 GPU 被充分利用了吗？
2. prefill 阶段和 decode 阶段的主导 kernel 有什么不同？
3. 如何判断当前推理是 compute-bound 还是 memory-bandwidth-bound？
4. CUDA Graph 是如何减少 decode 阶段开销的？
5. Llama-3-8B 在 H100 上 decode batch=1 时理论最快能做到多少 tokens/s？
