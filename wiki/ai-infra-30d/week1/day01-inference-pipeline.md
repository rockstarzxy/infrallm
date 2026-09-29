---
title: "Day 1: 推理系统全景和 Transformer 推理路径"
type: concept
tags: [day1, inference-pipeline, kv-cache, prefill, decode]
sources: [2026-09-27_mainstream-llm-weight-references.md]
created: 2026-06-01
updated: 2026-09-27
---

# Day 1: 推理系统全景和 Transformer 推理路径

## Part 1: LLM 推理的完整生命周期

一个 LLM 推理请求从进入系统到返回结果，经过以下阶段：

```
HTTP Request
    ↓
1. API Server 接收请求，解析参数（model, messages, max_tokens, temperature...）
    ↓
2. Tokenizer 编码：将 text 转为 token IDs
    ↓
3. Scheduler 调度：决定这个请求何时上 GPU 执行
    ↓
4. Prefill 阶段：一次性处理所有 input tokens，生成 KV Cache
    ↓
5. Decode 循环：每步生成 1 个 token，更新 KV Cache
    ↓  （重复直到遇到 EOS 或达到 max_tokens）
6. Sampling：从 logits 中采样下一个 token（top-k, top-p, temperature）
    ↓
7. Detokenize：token ID → text
    ↓
8. Stream / 返回完整响应
```

### 关键区分：Prefill vs Decode

这是推理优化中最基础也最重要的概念划分。

**Prefill 阶段（首次前向传播）：**
- 输入：完整的 prompt（所有 input tokens）
- 计算：对所有 input tokens 并行做 attention 计算
- 输出：所有层的 KV Cache + 第一个 output token 的 logits
- 计算特征：大矩阵乘法，**通常是 compute-bound**
- 类比：一次性"理解"整个问题

**Decode 阶段（自回归生成）：**
- 输入：上一步生成的 1 个 token
- 计算：这 1 个 token 和之前所有 token 的 KV Cache 做 attention
- 输出：下一个 token 的 logits
- 计算特征：batch size 很小（通常是 1 个 token），矩阵运算退化为矩阵-向量乘法，**通常是 memory-bandwidth-bound**
- 类比：一个字一个字地"写答案"

**为什么 decode 是 memory-bound？**

在 decode 阶段，每一步需要：
1. 从 HBM 加载本步需要的模型权重（随模型、精度和路由变化，可达数百 GB）
2. 从 HBM 加载 KV Cache（随序列长度增长）
3. 但实际计算量很小：只处理 1 个 token 的矩阵-向量乘法

GPU 的算力远超这点计算量的需求，瓶颈在于"数据从显存搬到计算单元"的速度——即 HBM 带宽。

量化理解（batch=1、稠密模型、权重驻留显存，只计权重读取）：
- [H100 SXM 的 HBM 峰值带宽](https://www.nvidia.com/en-gb/data-center/h100/)：~3.35 TB/s
- Llama-3-8B FP16 权重：~16 GB
- 近似每次 decode 读取一遍主要权重：16 GB / 3.35 TB/s ≈ 4.8 ms/token
- 只计权重读取的理想吞吐上界约 ~208 tokens/s；实际还受有效带宽、KV Cache、计算和 kernel 开销影响，不能当实测速度

这就是为什么 **batching 对 decode 阶段至关重要**：多个请求共享一次权重加载，均摊带宽开销。


### 主流大模型的参数量与权重体积速查

**核对日期：2026-09-27。** 覆盖主流通用、推理、代码和多模态理解模型的代表版本，并保留部分常见部署基线；不是热度排名，也不穷举所有微调版、蒸馏版与 API 快照。每个模型名链接到官方模型卡或文档；GLM-5.2 的参数估算另外注明框架来源。

#### 先统一计算口径

- **B = 十亿参数，T = 一万亿参数。** FP16/BF16 每参数 2 bytes，8-bit 理想值 1 byte，4-bit 理想值 0.5 byte。
- **权重体积 ≈ 存储的参数量 × 每参数字节数。** 下面统一用十进制 GB/TB：1 GB = 10⁹ bytes，1 TB = 1000 GB；1 GiB ≈ 1.074 GB。
- **表中的体积都是推算值，不是下载文件大小，也不是部署显存需求。** 8-bit/4-bit 列不代表每个模型都有可用的对应量化版本；混合精度、scale/zero-point、对齐、未量化层都会增加实际体积。
- **总参数决定完整权重存储量，激活参数更接近单 token 的计算规模。** MoE 的不同 token 会选不同专家；全 GPU 常驻部署通常仍要存全部专家，或使用 CPU/SSD offload。不能用激活参数代替总参数算完整权重。
- **开放权重不等于所有代码、数据、训练流程都开源。** 下表以是否提供权重区分；具体使用条件以各仓库许可证为准。

#### 开放权重：大规模模型

此表按官方公布的近似参数量计算。模型卡若只公布语言主干，表中也只估语言主干；视觉编码器、MTP、额外嵌入表等需按注释另算。

| 模型 | 架构；总参数口径 | 每 token 激活参数 | FP16/BF16 理论体积 | 8-bit 理论体积 | 4-bit 理论体积 |
|---|---|---:|---:|---:|---:|
| [DeepSeek V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | MoE；约 1.6T | 49B | 3.2 TB | 1.6 TB | 800 GB |
| [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | MoE；284B | 13B | 568 GB | 284 GB | 142 GB |
| [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | MoE；552B **主干**，见注① | prefill 8B / decode 16B | 1.104 TB | 552 GB | 276 GB |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | MoE；约 2.8T | 104B | 5.6 TB | 2.8 TB | 1.4 TB |
| [Kimi K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) | MoE；约 1T | 32B | 2 TB | 1 TB | 500 GB |
| [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | MoE；约 2.4T | 95B | 4.8 TB | 2.4 TB | 1.2 TB |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | MoE；125B + 51B + 4B ≈ 180B，见注② | 主干 6B | 360 GB | 180 GB | 90 GB |
| [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) | MoE；397B 语言模型 | 17B | 794 GB | 397 GB | 198.5 GB |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | MoE；约 743B，见注③ | 约 39B | 1.486 TB | 743 GB | 371.5 GB |
| [MiniMax M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) | MoE；约 428B | 约 23B | 856 GB | 428 GB | 214 GB |
| [Mistral Large 3](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) | MoE；675B | 41B | 1.35 TB | 675 GB | 337.5 GB |
| [Llama 4 Maverick](https://huggingface.co/docs/transformers/model_doc/llama4) | MoE；约 400B | 17B | 800 GB | 400 GB | 200 GB |
| [Llama 4 Scout](https://huggingface.co/docs/transformers/model_doc/llama4) | MoE；约 109B | 17B | 218 GB | 109 GB | 54.5 GB |
| [ERNIE 4.5-300B-A47B](https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-PT) | MoE；300B，既有部署基线 | 47B | 600 GB | 300 GB | 150 GB |
| [Hunyuan A13B](https://huggingface.co/tencent/Hunyuan-A13B-Instruct) | MoE；80B，既有部署基线 | 13B | 160 GB | 80 GB | 40 GB |

① **DeepSeek V4.1 的 552B 是 backbone 参数口径。** 官方还列出 196B Engram 条件记忆、视觉编码器和 DSpark 组件。这里仅给主干体积，不能把这一行当作完整 checkpoint 容量；完整预算应核对张量清单、模块是否已计入及各模块精度，避免漏算或重复相加。[官方说明](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)

② **Qwen3.8-Flash-Next 的 125B 不是全部存储。** 官方另外列出 51B n-gram embedding 和 4B MTP，上表将三项合计约 180B；该口径仍不含单列的视觉编码器。嵌入表可 offload，意味着 CPU 内存与 GPU 显存预算需要拆开。[官方参数说明](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

③ GLM-5.2 模型卡未直接列出总参数；此处采用 [vLLM 项目部署配方](https://github.com/vllm-project/recipes/blob/main/models/zai-org/GLM-5.2.yaml) 的约 743B / 39B，作为容量级别估算，不视为模型厂商的精确张量统计。

DeepSeek V4 官方权重使用 FP4/FP8 混合精度；Kimi K3 使用 MXFP4 权重与 MXFP8 激活。因此它们的 BF16 列是统一换算基线，**不是实际发布格式**。Kimi K3 的 104B 激活参数也不意味着只需 208 GB 就能装下 BF16 全模型。[DeepSeek 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)、[Kimi 模型卡](https://huggingface.co/moonshotai/Kimi-K3)

#### 开放权重：中小规模与单机常见模型

| 模型 | 参数口径 | FP16/BF16 理论体积 | 8-bit 理论体积 | 4-bit 理论体积 |
|---|---|---:|---:|---:|
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B 稠密语言模型 | 54 GB | 27 GB | 13.5 GB |
| [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | 27B 稠密语言模型 | 54 GB | 27 GB | 13.5 GB |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 35B 语言模型；MoE 激活 3B | 70 GB | 35 GB | 17.5 GB |
| [gpt-oss-120b](https://developers.openai.com/api/docs/models/gpt-oss-120b) | 实际约 117B；MoE 激活 5.1B | 234 GB | 117 GB | 58.5 GB |
| [gpt-oss-20b](https://developers.openai.com/api/docs/models/gpt-oss-20b) | 实际约 21B；MoE 激活 3.6B | 42 GB | 21 GB | 10.5 GB |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | 5.1B 含嵌入；有效 2.3B | 10.2 GB | 5.1 GB | 2.55 GB |
| [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E2B-it) | 8B 含嵌入；有效 4.5B | 16 GB | 8 GB | 4 GB |
| [Gemma 4 12B](https://huggingface.co/google/gemma-4-E2B-it) | 11.95B；统一多模态 | 23.9 GB | 11.95 GB | 5.975 GB |
| [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-E2B-it) | 25.2B 主干；MoE 激活 3.8B | 50.4 GB | 25.2 GB | 12.6 GB |
| [Gemma 4 31B](https://huggingface.co/google/gemma-4-E2B-it) | 30.7B 稠密主干 | 61.4 GB | 30.7 GB | 15.35 GB |
| [Llama 3.1 8B](https://huggingface.co/meta-llama) | 约 8B 稠密，教学/部署基线 | 16 GB | 8 GB | 4 GB |
| [Llama 3.1 70B](https://huggingface.co/meta-llama) | 约 70B 稠密，部署基线 | 140 GB | 70 GB | 35 GB |

**Gemma 的附加模块要另算。** 官方系列模型卡单列了 E2B/E4B 的约 150M 视觉编码器和 300M 音频编码器，以及 26B/31B 的约 550M 视觉编码器；上表按主干/嵌入口径计算，未计这些附加模块。实际部署请参考 [Google 的推理内存表](https://ai.google.dev/gemma/docs/core)：BF16 约为 E2B 11.4 GB、E4B 17.9 GB、12B 26.7 GB、26B-A4B 57.7 GB、31B 69.9 GB。官方也注明它们随工具和环境变化，不能直接当成任意上下文与并发下的总显存保证。

#### 闭源 / 仅服务形式公开的主流型号

“未公开”在这里表示本次查阅的官方资料未给出可复核的总参数和完整权重；不能用 API 价格、跑分或传闻推算准确 GB 数。

| 厂商 / 系列 | 当前代表型号或服务 | 参数量 / 权重体积 |
|---|---|---|
| [OpenAI GPT](https://developers.openai.com/api/docs/models) | GPT-6 Astra、Sol、Luna | 未公开；不能计算。与开放权重 gpt-oss 分开看 |
| [Anthropic Claude](https://www.anthropic.com/system-cards) | Fable 5.1、Mythos 5.1；另有 Opus、Sonnet、Haiku 产品线 | 未公开；不能计算。Mythos 的访问限制与一般服务不同 |
| [Google Gemini](https://ai.google.dev/gemini-api/docs/models) | 3.8 Flash、3.5 Flash-Lite、3.1 Pro Preview | 未公开；不能计算。与开放权重 Gemma 分开看 |
| [Grok](https://docs.x.ai/developers/models) | Grok 4.7 | 未公开；不能计算 |
| [Qwen3.7 Max](https://docs.modelstudio.console.alibabacloud.com/tc/model-studio/qwen3-7-max) / [Plus](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/qwen3-7-plus) | Qwen3.7-Max、Qwen3.7-Plus | 本次未找到可核实参数量或对应权重仓库；不能沿用 Qwen3.5/3.8 参数 |
| [豆包 Seed](https://docs.volcengine.com/docs/ark/model-list?lang=zh) | Seed 2.0 Pro / Lite / Mini / Code；Seed Evolving | 未公开；不能计算 |

有官方 API 不等于闭源：DeepSeek、Kimi、Qwen、GLM、MiniMax 都可能同时提供 API 和开放权重。应按**具体型号**判断，不能按公司一概归类。上表只记录已核对的代表型号，服务别名和开放状态可能继续变化。

#### 怎么把这个表用到推理预算里

```text
完整权重存储 ≈ Σ(各模块参数量 × 各模块实际存储字节数) + 量化元数据
部署显存 ≈ GPU 常驻权重 + KV Cache/循环状态 + 激活与工作区
           + CUDA Graph/通信缓冲 + 框架与内存碎片余量
```

- 例如 Qwen3.6-35B-A3B：BF16 语言权重约 **70 GB**，不是按激活 3B 算出的 6 GB；完整多模态模型和运行时还要留额外空间。
- Llama 8B 的 **16 GB / 3.35 TB/s** 是稠密、batch=1、只看权重读取的简化估算。MoE 每 token 只访问部分专家，batch 增大又会覆盖更多专家；不能把“总权重 / 单卡带宽”直接套成所有模型的 token 延迟。
- 多 GPU 分片还受专家/张量并行、层划分、复制参数与互联通信影响。总容量除以单卡容量只能给出粗下界，不能直接得出可用部署方案。

配套目录：[[ai-infra-30d/reading-index]]。本次核对的来源链接归档见 [[2026-09-27_mainstream-llm-weight-references]]。

---

## Part 2: Self-Attention 计算流程

### 标准 Attention

给定输入 X（shape: [seq_len, d_model]），计算过程如下：

```
Q = X @ W_Q    # [seq_len, d_model] @ [d_model, d_head * n_heads] → [seq_len, n_heads * d_head]
K = X @ W_K    # 同上
V = X @ W_V    # 同上

# 重塑为多头：[n_heads, seq_len, d_head]

# 对每个 head：
scores = Q @ K^T / sqrt(d_head)   # [seq_len, seq_len]  ← O(n²) 在这里
attn_weights = softmax(scores)     # [seq_len, seq_len]
output = attn_weights @ V          # [seq_len, d_head]

# 合并多头 → 输出投影
output = concat(all heads) @ W_O   # [seq_len, d_model]
```

计算复杂度：O(n² · d) 其中 n 是序列长度，d 是 head 维度。

### KV Cache 机制

在 decode 阶段，每步只新增 1 个 token。如果每次都重新计算所有 token 的 attention，计算量会随序列增长而重复浪费。

KV Cache 的核心思路：**缓存之前所有 token 的 K 和 V，新 token 只需计算自己的 Q、K、V，然后和缓存的 K/V 做 attention。**

```
# Decode step t：
# 新 token x_t 的投影：
q_t = x_t @ W_Q    # [1, d_head * n_heads]
k_t = x_t @ W_K    # [1, d_head * n_heads]
v_t = x_t @ W_V

# 追加到 cache：
K_cache = concat(K_cache, k_t)   # [t, d_head * n_heads]
V_cache = concat(V_cache, v_t)

# Attention：
scores = q_t @ K_cache^T / sqrt(d_head)  # [1, t]
output = softmax(scores) @ V_cache        # [1, d_head]
```

好处：避免对前 t-1 个 token 重复计算 Q、K、V 投影。
代价：需要在 GPU 显存中保存 K 和 V，且显存占用随序列长度线性增长。

---

## Part 3: MQA 和 GQA

标准 Multi-Head Attention (MHA) 中，每个 head 有独立的 Q、K、V。
KV Cache 大小 = 2（K和V） × n_heads × d_head × seq_len × batch × bytes_per_element。

**Multi-Query Attention (MQA)：**
- 所有 Q head 共享 **1 组** K 和 V
- KV Cache 缩小为原来的 1/n_heads
- 代价：质量可能略有下降

**Grouped-Query Attention (GQA)：**
- 将 Q heads 分成 g 组，每组共享 1 组 K/V
- KV Cache 缩小为原来的 g/n_heads
- 是 MHA 和 MQA 之间的折中
- Llama-3 系列使用 GQA

```
MHA:  每个 Q head 有自己的 K/V head    → n_kv_heads = n_q_heads
GQA:  多个 Q head 共享一组 K/V          → n_kv_heads < n_q_heads（如 8）
MQA:  所有 Q head 共享同一组 K/V        → n_kv_heads = 1
```

**Llama-3-8B 的具体参数：**
- n_q_heads = 32, n_kv_heads = 8 (GQA, 4 个 Q head 共享 1 个 KV head)
- d_head = 128
- n_layers = 32

---

## Part 4: KV Cache 显存计算

### 公式

```
KV Cache 显存 = 2 × n_layers × n_kv_heads × d_head × seq_len × batch_size × bytes_per_element
```

- 2：K 和 V 各一份
- n_kv_heads：GQA 下是 KV head 数（不是 Q head 数）
- bytes_per_element：FP16 = 2 bytes, FP8 = 1 byte, INT8 = 1 byte

### 计算示例

**Llama-3-8B (FP16)：**
- 参数：layers=32, kv_heads=8, head_dim=128, dtype=FP16(2 bytes)

| seq_len | batch=1 | batch=8 | batch=32 | batch=128 |
|---:|---:|---:|---:|---:|
| 1,024 | 128 MB | 1.0 GB | 4.0 GB | 16.0 GB |
| 4,096 | 512 MB | 4.0 GB | 16.0 GB | 64.0 GB |
| 8,192 | 1.0 GB | 8.0 GB | 32.0 GB | 128.0 GB |
| 32,768 | 4.0 GB | 32.0 GB | 128.0 GB | 512.0 GB |
| 131,072 | 16.0 GB | 128.0 GB | 512.0 GB | 2.0 TB |

单位换算：2 × 32 × 8 × 128 × 2 = 131,072 bytes per token per batch element = **128 KB/token**

**Llama-3-70B (FP16)：**
- 参数：layers=80, kv_heads=8, head_dim=128, dtype=FP16(2 bytes)
- Per token: 2 × 80 × 8 × 128 × 2 = 327,680 bytes = **320 KB/token**

| seq_len | batch=1 | batch=8 | batch=32 |
|---:|---:|---:|---:|
| 4,096 | 1.25 GB | 10.0 GB | 40.0 GB |
| 8,192 | 2.5 GB | 20.0 GB | 80.0 GB |

**关键结论：**
- KV Cache 显存可以轻松超过模型权重显存
- 长上下文 + 大 batch 时 KV Cache 是显存瓶颈的主要来源
- 这就是 PagedAttention（Day 4）和 KV Cache 量化（Day 10）要解决的问题

### 模型权重 vs KV Cache 显存对比

| 模型 | 权重显存 (FP16) | 1 个 seq_len=4096 的请求 KV Cache |
|---|---:|---:|
| Llama-3-8B | ~16 GB | 0.5 GB |
| Llama-3-70B | ~140 GB | 1.25 GB |

单个请求看起来 KV Cache 不大，但当并发 64 个请求时：
- 8B: 64 × 0.5 GB = 32 GB —— 已经是权重的 2 倍
- 70B: 64 × 1.25 GB = 80 GB —— 超过单卡显存

---

## Part 5: 模型前向传播的完整计算链

以一个标准的 Decoder-only Transformer（如 Llama）为例，一个 layer 的计算：

```
Input x  →  RMSNorm
         →  Self-Attention (QKV projection → attention → output projection)
         →  Residual Add
         →  RMSNorm
         →  FFN (gate_proj + up_proj → SiLU → down_proj)
         →  Residual Add
         →  Output x'
```

**FFN（SwiGLU variant）的计算：**
```
gate = x @ W_gate          # [seq_len, d_model] @ [d_model, d_ffn]
up   = x @ W_up            # 同上
hidden = SiLU(gate) * up   # element-wise
output = hidden @ W_down   # [seq_len, d_ffn] @ [d_ffn, d_model]
```

完整模型：N 个这样的层 + 最终的 RMSNorm + LM Head（vocabulary projection）。

---

## 交付物模板

### 推理链路图.md

画出从 HTTP request 到 token stream 的完整路径，标注每一步的：
- 计算类型（CPU / GPU）
- 数据格式（text / token IDs / tensors）
- 显存操作（allocate / read / write）
- 是否有批处理（single request vs batched）

### KV Cache 显存估算表.md

用上面的公式计算你计划使用的模型在不同配置下的 KV Cache 显存。重点关注：
- 你的 GPU 有多少显存？
- 扣除模型权重后，能容纳多少 KV Cache？
- 这决定了能同时处理多少并发请求？

---

## 自检问题

1. 一个 request 从到达服务到返回第一个 token，经历了哪些步骤？
2. prefill 阶段为什么可以并行处理所有 input token？
3. decode 阶段为什么必须逐 token 生成？
4. Llama-3-8B 在 seq_len=8192、batch=32 时 KV Cache 占多少显存？（答：32 GB）
5. GQA 把 kv_heads 从 32 降到 8，KV Cache 节省了多少？（答：75%）
6. 为什么说 decode 阶段是 memory-bandwidth-bound 而不是 compute-bound？
7. 如果不做 batching，单卡 H100 上 Llama-3-8B 的 decode 吞吐大约是多少 tokens/s？
