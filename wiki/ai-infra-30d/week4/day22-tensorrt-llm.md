---
title: "Day 22: TensorRT-LLM 对照学习"
type: concept
tags: [day22, tensorrt-llm, nvidia, comparison, vllm]
created: 2026-06-01
updated: 2026-06-01
---

# Day 22: TensorRT-LLM 对照学习

## Part 1: TensorRT-LLM 定位

TensorRT-LLM 是 NVIDIA 官方的 LLM 推理引擎，核心定位：**在 NVIDIA GPU 上追求极致推理性能。**

```
vLLM:
  - 社区驱动，开源生态丰富
  - 通用性好，支持多种模型和硬件
  - Python 为主，易于二次开发
  - PagedAttention + continuous batching

TensorRT-LLM:
  - NVIDIA 官方，深度优化 NVIDIA GPU
  - 性能极致（定制 kernel + CUDA Graph + 算子融合）
  - C++ 为主，需要"编译"模型为 TRT engine
  - 学习曲线陡峭，灵活性不如 vLLM
```

### 适用场景对比

| 维度 | vLLM | TensorRT-LLM |
|---|---|---|
| 性能 | 好 | 极致（通常快 10-30%） |
| 易用性 | 高（pip install 即用） | 中（需要 build engine） |
| 模型支持 | 广（HuggingFace 生态） | 较广（NVIDIA 重点适配的模型） |
| 灵活性 | 高（Python，易改源码） | 低（C++ engine，难修改） |
| 社区 | 活跃 | NVIDIA 主导 |
| 新模型适配速度 | 快（社区贡献） | 较慢（NVIDIA 官方适配） |

---

## Part 2: TensorRT-LLM 核心概念

### Engine Build

与 vLLM 最大的区别：TensorRT-LLM 需要先将模型"编译"为 TRT engine。

```
HuggingFace 权重 → TensorRT-LLM convert → TRT Engine 文件
                                            ├── 固定了 TP/PP 配置
                                            ├── 固定了数据类型
                                            ├── 固定了 max_batch_size
                                            └── 针对目标 GPU 优化

vs vLLM: 直接加载 HuggingFace 权重，运行时动态适配
```

### In-Flight Batching (IFB)

TensorRT-LLM 的 continuous batching 实现叫 in-flight batching：

```
和 vLLM 的 continuous batching 概念相同:
  - iteration 级调度
  - 请求可以随时加入和退出 batch
  - 不同请求可能处于 prefill 或 decode 阶段

TensorRT-LLM 的实现特点:
  - 更激进的 kernel fusion
  - 和 TRT engine 深度整合
  - CUDA Graph 覆盖更多执行路径
```

### Paged KV Cache

TensorRT-LLM 也支持分页 KV Cache（类似 PagedAttention）：

```
原理相同: block table + 物理 block 按需分配
实现差异: 
  - 定制的 C++ paged attention kernel
  - 和 TRT engine 的内存管理深度集成
  - 支持 FP8/INT8 KV Cache
```

---

## Part 3: TensorRT-LLM 的性能优势来源

### 1. Kernel Fusion

```
vLLM: 可能需要多次 kernel launch
  [RMSNorm kernel] → [QKV GEMM kernel] → [Attention kernel] → ...

TensorRT-LLM: 将多个操作融合为一个 kernel
  [fused_rmsnorm_qkv_kernel] → [attention_kernel] → ...

减少:
  - kernel launch overhead (~5us per launch)
  - 中间结果的 HBM 读写
  - GPU idle time between kernels
```

### 2. 定制 CUDA Kernel

```
NVIDIA 工程师为常见模型编写了高度优化的 kernel:
  - GEMM: 使用 CUTLASS 模板库，针对特定矩阵形状优化
  - Attention: 定制的 fused multi-head attention
  - LayerNorm: 和前后操作融合
  - SwiGLU: gate + up + SiLU 融合

这些 kernel 比通用库（如 cuBLAS/FlashAttention）更快
但只适用于特定配置（特定矩阵大小、数据类型）
```

### 3. 编译时优化

```
Build engine 时，TensorRT 会:
  1. 分析计算图，找到可以融合的操作
  2. 对每个 kernel 做 auto-tuning（尝试多种实现，选最快的）
  3. 确定最优的内存布局
  4. 生成针对目标 GPU 的 SASS 代码

→ 运行时没有解释执行的开销
→ 但 engine 和 GPU 型号绑定（A100 engine 不能在 H100 上跑）
```

### 4. 量化支持

```
TensorRT-LLM 的量化支持非常全面:
  - FP8 (H100+): 原生支持，性能几乎翻倍
  - INT8 SmoothQuant: W8A8，高吞吐
  - INT4 GPTQ/AWQ: 显存节省
  - FP4 (Blackwell): 最新一代

量化后的 engine 性能通常优于 vLLM:
  因为量化的 dequantize + GEMM 被融合为一个 kernel
```

---

## Part 4: TensorRT-LLM 的工作流程

```
Step 1: 准备模型
  下载 HuggingFace 权重

Step 2: 转换和量化 (optional)
  python convert_checkpoint.py \
    --model_dir ./Qwen2.5-7B-Instruct \
    --output_dir ./trt_ckpt \
    --dtype float16 \
    --tp_size 2

Step 3: Build Engine
  trtllm-build \
    --checkpoint_dir ./trt_ckpt \
    --output_dir ./trt_engine \
    --gemm_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --paged_kv_cache enable \
    --use_paged_context_fmha enable

Step 4: Serving
  # 使用 Triton Inference Server + TRT-LLM backend
  # 或使用 TRT-LLM 的 Python API

Step 5: 发请求
  类似 vLLM 的 OpenAI-compatible API
```

### 和 vLLM 的工作流对比

```
vLLM:
  pip install vllm → vllm serve model_name → done
  总时间: ~2 分钟（加上模型下载）

TensorRT-LLM:
  安装 → 转换权重 → build engine (可能需要 10-60 分钟) → 配置 Triton → 启动
  总时间: ~30 分钟到数小时
```

---

## Part 5: Disaggregated Serving

TensorRT-LLM 支持 prefill 和 decode 分离部署：

```
传统: 同一组 GPU 做 prefill + decode
分离: prefill 在一组 GPU，decode 在另一组

Prefill Worker (计算密集型):
  - 可以用高算力 GPU (H100)
  - 处理完 prefill 后，将 KV Cache 传输给 decode worker

Decode Worker (带宽密集型):
  - 可以用高带宽 GPU (或量化后使用)
  - 只做 decode，TPOT 更稳定

KV Cache Transfer:
  prefill 完成后通过 NVLink/IB 将 KV Cache 传给 decode worker
  传输量 = 全量 KV Cache（可能很大）
  → 需要高速互联
```

这个概念在 vLLM 中也开始支持（disaggregated prefilling），Day 24 会详细讨论。

---

## Part 6: vLLM vs TensorRT-LLM 选择指南

```
选 vLLM 当:
  ✓ 需要快速原型和实验
  ✓ 需要频繁更换模型
  ✓ 需要二次开发（修改调度器、自定义 attention 等）
  ✓ 团队 Python 为主
  ✓ 需要支持非 NVIDIA 硬件（AMD/Intel GPU，TPU 等）

选 TensorRT-LLM 当:
  ✓ 性能是最高优先级
  ✓ 模型和配置稳定（不频繁更换）
  ✓ 全 NVIDIA 硬件环境
  ✓ 有专人维护推理基础设施
  ✓ 需要最大化 GPU ROI
```

实际上很多团队的做法：**用 vLLM 做开发和实验，用 TensorRT-LLM 做生产部署。**

---

## 交付物

| 文件 | 描述 |
|---|---|
| `vLLM-vs-TensorRT-LLM.md` | 两者的核心差异 + 选型指南 |

## 自检问题

1. TensorRT-LLM 的 engine build 做了什么？为什么它能更快？
2. 为什么 engine 和 GPU 型号绑定？
3. In-flight batching 和 vLLM 的 continuous batching 有什么异同？
4. TensorRT-LLM 的 kernel fusion 具体省了什么开销？
5. 什么场景下 TensorRT-LLM 的性能优势不明显？
