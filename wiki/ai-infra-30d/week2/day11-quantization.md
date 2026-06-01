---
title: "Day 11: Quantization（量化）"
type: concept
tags: [day11, quantization, gptq, awq, fp8, smoothquant, int4]
created: 2026-06-01
updated: 2026-06-01
---

# Day 11: Quantization（量化）

## Part 1: 量化基础

### 什么是量化

量化是将模型中的浮点数参数从高精度（FP32/FP16/BF16）转换为低精度（INT8/INT4/FP8）的过程。

```
FP16:  16 bits per parameter → Llama-3-8B ≈ 16 GB
INT8:   8 bits per parameter → Llama-3-8B ≈  8 GB
INT4:   4 bits per parameter → Llama-3-8B ≈  4 GB
FP8:    8 bits per parameter → Llama-3-8B ≈  8 GB
```

### 量化的收益

1. **显存节省**：模型权重占更少显存 → 留更多空间给 KV Cache → 更高并发
2. **推理加速**：
   - 更小的数据量 → 更少的 HBM 读取 → 缓解 memory-bandwidth 瓶颈
   - 低精度计算硬件支持 → 更高 TFLOPS（如 H100 的 FP8 算力是 FP16 的 2x）
3. **成本降低**：可以在更小的 GPU 上部署更大的模型

### 量化的代价

1. **质量损失**：低精度必然损失信息，可能导致输出质量下降
2. **模型特异性**：不同模型对量化的容忍度不同
3. **额外步骤**：有些量化方法需要校准数据和预处理时间

---

## Part 2: 数值格式详解

### 常见浮点格式

```
FP32: 1 sign + 8 exponent + 23 mantissa = 32 bits
FP16: 1 sign + 5 exponent + 10 mantissa = 16 bits
BF16: 1 sign + 8 exponent +  7 mantissa = 16 bits  ← 和 FP32 同样的指数范围
FP8 E4M3: 1 sign + 4 exponent + 3 mantissa = 8 bits ← 用于 forward (权重/activation)
FP8 E5M2: 1 sign + 5 exponent + 2 mantissa = 8 bits ← 用于 backward (梯度)
```

BF16 vs FP16：
- BF16 的指数范围和 FP32 一样大（不容易溢出/下溢），但精度更低
- FP16 的精度更高，但容易溢出
- 推理中 BF16 更常用（稳定性好）

### 整数量化格式

```
INT8:  8 bits，范围 [-128, 127]
INT4:  4 bits，范围 [-8, 7]
UINT4: 4 bits，范围 [0, 15]
```

将浮点值映射到整数：
```
Symmetric quantization:
  x_int = round(x / scale)
  scale = max(|x|) / (2^(n-1) - 1)

Asymmetric quantization:
  x_int = round((x - zero_point) / scale)
  scale = (max(x) - min(x)) / (2^n - 1)
```

### 量化粒度

```
Per-tensor:   整个矩阵共享一个 scale
Per-channel:  每一列（或行）一个 scale
Per-group:    每 g 个元素一个 scale（g 通常是 128）
Per-token:    每个 token 一个 scale（用于 activation 量化）
```

粒度越细 → 精度越高 → 存储 scale 的开销越大

---

## Part 3: 权重量化方法

### GPTQ (Post-Training Quantization using Optimal Brain Quantization)

核心思想：逐层量化，用二阶（Hessian）信息指导量化误差最小化。

```
过程：
1. 用校准数据（~128 条样本）跑一遍模型，收集每层的 activation 统计
2. 逐层量化权重：
   for each column in W:
     quantize column → 产生量化误差
     用 Hessian 信息调整其他列来补偿这个误差
3. 保存量化后的权重 + scale/zero-point

特点：
- 只量化权重，activation 仍用 FP16
- 通常量化到 INT4（4-bit）
- per-group quantization（group_size=128）
- 需要校准数据，量化过程可能需要几小时
```

### AWQ (Activation-Aware Weight Quantization)

核心思想：不是所有权重都同等重要。那些和"大 activation"相乘的权重更重要，应该给予更高精度。

```
过程：
1. 用校准数据收集 activation 统计
2. 找出 activation 中的"重要通道"（值大的通道）
3. 对这些重要通道的权重乘以一个 scale（让它们更大，量化后误差相对更小）
4. 对 activation 做相反操作（除以 scale），保持数学等价
5. 在 scaled 后的权重上做标准量化

特点：
- 比 GPTQ 更简单、更快
- 效果通常和 GPTQ 接近或略好
- 4-bit per-group quantization
- 是目前 4-bit 推理最常用的方案之一
```

### SmoothQuant

核心思想：Activation 中有离群值（outlier），直接量化 activation 会导致巨大误差。SmoothQuant 将离群值从 activation "迁移"到 weight。

```
数学等价变换：
  Y = X @ W
    = (X / s) @ (s * W)     # s 是 per-channel scaling factor

原始：X 有离群值（难量化），W 分布均匀（好量化）
变换后：X/s 更平滑（好量化），s*W 略不均匀但还是好量化
→ 两边都可以量化为 INT8
```

特点：
- 主要用于 **W8A8**（权重 INT8 + activation INT8）
- 比 W4A16 的计算效率更高（INT8 GEMM 比 FP16 快）
- 但显存节省不如 4-bit 明显

### 方法对比

| 方法 | 量化位宽 | 量化对象 | 需要校准？ | 典型用途 |
|---|---|---|---|---|
| GPTQ | W4 | 权重 | 是 | 显存受限，需要极致压缩 |
| AWQ | W4 | 权重 | 是 | 和 GPTQ 类似，更简单 |
| SmoothQuant | W8A8 | 权重+激活 | 是 | INT8 加速，吞吐优先 |
| FP8 | W8/A8 | 权重(+激活) | 否/简单 | H100+，开箱即用 |

---

## Part 4: FP8 推理

### 为什么 FP8 特殊

FP8 是 H100（及 Ada Lovelace / Blackwell）原生支持的数据格式：
- 不需要复杂的校准（简单的 per-tensor scale 就行）
- H100 的 FP8 Tensor Core 算力是 FP16 的 2x
- 同时享受显存减半和计算加速

```
FP16 推理 on H100:
  - 权重 16 GB (8B model)
  - 算力 989 TFLOPS
  - 带宽 3.35 TB/s

FP8 推理 on H100:
  - 权重 8 GB (8B model) → 显存减半
  - 算力 1,979 TFLOPS  → 计算翻倍
  - 带宽还是 3.35 TB/s，但要读的数据少了一半 → 等效带宽翻倍

→ 对于 memory-bound 的 decode 阶段，FP8 接近 2x 加速
```

### 在 vLLM 中使用

```bash
# FP8 权重量化
vllm serve Qwen/Qwen2.5-7B-Instruct --quantization fp8

# FP8 KV Cache（和权重量化独立）
vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype fp8

# 两者同时使用
vllm serve Qwen/Qwen2.5-7B-Instruct --quantization fp8 --kv-cache-dtype fp8
```

---

## Part 5: 在 vLLM 中使用量化模型

### 使用预量化模型（推荐）

社区已经提供了大量预量化模型，直接下载使用：

```bash
# AWQ 量化模型
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq

# GPTQ 量化模型
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --quantization gptq

# 自动检测量化方式
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ
# vLLM 会从 config.json 中读取 quantization_config 自动选择
```

### 中国模型的量化版本

大部分主流模型都有社区制作的量化版本：

| 模型 | AWQ 4-bit | GPTQ 4-bit | FP8 | 显存需求 (4-bit) |
|---|:---:|:---:|:---:|---:|
| Qwen2.5-7B-Instruct | ✓ | ✓ | ✓ | ~4 GB |
| Qwen2.5-14B-Instruct | ✓ | ✓ | ✓ | ~8 GB |
| Qwen2.5-72B-Instruct | ✓ | ✓ | ✓ | ~40 GB |
| DeepSeek-V2-Lite | ✓ | — | ✓ | ~9 GB |
| GLM-4-9B-Chat | ✓ | ✓ | ✓ | ~5 GB |
| Baichuan2-13B-Chat | ✓ | ✓ | — | ~7 GB |

---

## Part 6: 量化对性能的影响

### 量化对吞吐的影响

```
Qwen2.5-7B on A100 80GB:

                FP16        AWQ-4bit      FP8
权重显存:       ~14 GB       ~4 GB        ~7 GB
KV Cache可用:   ~60 GB       ~70 GB       ~67 GB
Max并发:        ~64          ~84          ~74
Decode TPOT:    ~12 ms       ~10 ms       ~8 ms
Throughput:     1200 tok/s   1500 tok/s   1800 tok/s
```

为什么 4-bit 比 FP16 快？
1. 权重更小 → 从 HBM 读取更快 → 缓解 memory-bandwidth 瓶颈
2. 更多 KV Cache 空间 → 更高并发 → 更高 throughput

为什么 FP8 通常比 4-bit 更快？
1. FP8 有原生 Tensor Core 支持 → GEMM 本身更快
2. 4-bit 需要 dequantize（解压到 FP16 再计算）→ 额外开销

### 量化对质量的影响

```
一般规律：
FP16 > FP8 > AWQ-4bit > GPTQ-4bit（quality）

但差距通常很小：
- 大模型（70B+）对量化的容忍度好于小模型（7B）
- 编码、数学任务更容易受量化影响
- 日常对话、总结等任务几乎不受影响
```

评估量化质量的方法：
1. 在标准 benchmark 上测（如 MMLU、HumanEval、GSM8K）
2. 用业务特定的 eval set 测
3. 人工对比采样输出

---

## Part 7: 训练推理一致性 (Training-Inference Consistency)

### 问题

量化推理可能导致模型输出和训练时的预期不一致：

```
场景 1: 训练用 BF16 → 推理用 INT4
  训练时: softmax([3.14, 2.71, 1.62]) = [0.50, 0.33, 0.17]
  推理时: softmax([3.12, 2.73, 1.60]) = [0.49, 0.34, 0.17]  ← 量化误差传播
  → 可能导致不同的 token 被采样
```

```
场景 2: Attention 计算精度
  训练: attention scores 在 FP32 中计算 softmax
  推理: 某些 backend 在 FP16 中计算 → 数值差异
  → 在长序列末端累积误差更大
```

### 关键一致性问题

**1. 数值精度差异**
- BF16 vs FP16：不同的精度和数值范围
- Flash Attention 的 online softmax 和标准 softmax 在数值上有微小差异
- 不同硬件（A100 vs H100）的 Tensor Core 精度可能不同

**2. 量化引入的偏差**
- 权重量化后，某些层的行为可能显著改变
- 特别是涉及条件逻辑的场景（如 function calling、JSON 格式化）

**3. Tokenizer 一致性**
- 训练和推理必须使用完全相同的 tokenizer
- 不同版本的 tokenizer 可能有不同的分词结果
- 特殊 token（BOS, EOS, padding）的处理必须一致

**4. Position encoding 一致性**
- RoPE（Rotary Position Embedding）的实现必须和训练一致
- 不同框架的 RoPE 实现可能有 index offset 差异
- 扩展上下文长度时的 RoPE scaling factor 必须匹配

### 如何保障一致性

```
检查清单：
□ Tokenizer 版本和训练一致
□ BOS/EOS token 处理和训练一致
□ RoPE 参数（base, scaling）和训练配置一致
□ Attention mask 格式（causal mask）和训练一致
□ 如果模型用 BF16 训练，推理也用 BF16（除非有意量化）
□ 量化后在关键任务上回归测试
```

DeepSeek-V2 的一致性挑战：
- MLA 的低秩投影矩阵如果有精度误差，会在解压缩时被放大
- 推理框架需要精确复现训练时的 MLA 计算流程

Qwen2.5 的注意点：
- Qwen 系列使用了自定义的 tokenizer（基于 tiktoken）
- 确保 vLLM 中加载的 tokenizer 和训练完全一致

---

## 交付物

| 文件 | 描述 |
|---|---|
| `quantization-tradeoff-report.md` | FP16 vs AWQ/GPTQ vs FP8 的对比数据（显存、吞吐、质量样例） |

## 自检问题

1. GPTQ 和 AWQ 的核心区别是什么？
2. 为什么 H100 上 FP8 推理通常比 INT4 更快？
3. 权重量化为什么能加速 decode 阶段？
4. SmoothQuant 的 "smooth" 是什么意思？
5. 量化对大模型和小模型的质量影响有什么区别？为什么？
6. 训练推理一致性的关键检查点有哪些？
