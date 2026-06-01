---
title: "Day 8: vLLM 架构源码阅读"
type: concept
tags: [day8, vllm, architecture, source-code]
created: 2026-06-01
updated: 2026-06-01
---

# Day 8: vLLM 架构源码阅读

## Part 1: vLLM 整体架构

```
┌────────────────────────────────────────────────────────────┐
│                    API Layer                                │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ OpenAI-compat    │  │  vLLM native API │               │
│  │ /v1/completions  │  │  LLM.generate()  │               │
│  │ /v1/chat/compl.  │  │  (offline mode)  │               │
│  └────────┬─────────┘  └────────┬─────────┘               │
│           └──────────┬──────────┘                          │
├──────────────────────┼─────────────────────────────────────┤
│                      ↓                                     │
│              ┌───────────────┐                             │
│              │  LLM Engine   │  ← 核心调度器               │
│              │               │                             │
│  ┌───────────┤  - add_request()                            │
│  │           │  - step()     │                             │
│  │           │  - abort()    │                             │
│  │           └───────┬───────┘                             │
│  │                   │                                     │
│  │    ┌──────────────┼──────────────┐                     │
│  │    ↓              ↓              ↓                     │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │ │Scheduler │ │  Block   │ │  Model   │               │
│  │ │          │ │ Manager  │ │ Runner   │               │
│  │ │waiting   │ │          │ │          │               │
│  │ │running   │ │ logical  │ │ forward  │               │
│  │ │swapped   │ │ → physical│ │ execute  │               │
│  │ └──────────┘ │ block map│ │ sample   │               │
│  │              └──────────┘ └────┬─────┘               │
│  │                                │                      │
│  │              ┌─────────────────┼─────────────┐        │
│  │              ↓                 ↓             ↓        │
│  │         ┌─────────┐      ┌─────────┐   ┌────────┐   │
│  │         │ Worker 0│      │ Worker 1│   │Worker N│   │
│  │         │ (GPU 0) │      │ (GPU 1) │   │(GPU N) │   │
│  │         └─────────┘      └─────────┘   └────────┘   │
│  │                                                       │
│  │ ┌──────────────────────────────────────────────────┐  │
│  │ │              Attention Backends                   │  │
│  │ │  FlashAttention │ FlashInfer │ xFormers │ ...    │  │
│  │ └──────────────────────────────────────────────────┘  │
│  │                                                       │
│  │ ┌──────────────────────────────────────────────────┐  │
│  └→│              Tokenizer                            │  │
│    └──────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 核心流程：一个请求的生命周期

```
1. 用户发 HTTP 请求 → API Server 收到
2. API Server → Tokenizer 编码 → 创建 SequenceGroup
3. SequenceGroup 进入 Scheduler 的 waiting 队列
4. Engine.step() 被调用（循环执行）：
   a. Scheduler.schedule() → 决定哪些请求上 GPU
   b. Block Manager → 为被调度的请求分配/释放 KV block
   c. Model Runner → 执行模型前向传播
   d. Sampler → 采样下一个 token
   e. 更新序列状态（append token, check stop condition）
5. 如果请求完成 → 通过 callback 返回结果
6. 循环回到 step 4
```

---

## Part 2: 源码目录结构

以 vLLM 主要目录为例（具体路径可能随版本变化）：

```
vllm/
├── entrypoints/              ← API 层
│   ├── openai/
│   │   ├── api_server.py     ← FastAPI server
│   │   ├── serving_chat.py   ← /v1/chat/completions
│   │   └── serving_completion.py  ← /v1/completions
│   └── llm.py                ← 离线推理入口 (LLM class)
│
├── engine/                   ← 引擎层
│   ├── llm_engine.py         ← LLMEngine: 核心调度循环
│   ├── async_llm_engine.py   ← AsyncLLMEngine: 异步版
│   └── arg_utils.py          ← 参数解析
│
├── core/                     ← 核心调度
│   ├── scheduler.py          ← Scheduler: 请求调度逻辑
│   └── block_manager.py      ← BlockManager: KV Cache 内存管理
│
├── worker/                   ← GPU 执行层
│   ├── worker.py             ← Worker: 单个 GPU 的工作进程
│   └── model_runner.py       ← ModelRunner: 模型前向传播
│
├── model_executor/           ← 模型实现
│   ├── models/               ← 各模型实现 (llama, qwen, deepseek, ...)
│   │   ├── llama.py
│   │   ├── qwen2.py
│   │   ├── deepseek_v2.py
│   │   └── ...
│   └── layers/
│       ├── attention/        ← attention 实现
│       ├── quantization/     ← 量化层实现
│       └── sampler.py        ← 采样逻辑
│
├── attention/                ← Attention Backend
│   ├── backends/
│   │   ├── flash_attn.py     ← FlashAttention backend
│   │   ├── flashinfer.py     ← FlashInfer backend
│   │   └── ...
│   └── selector.py           ← backend 自动选择
│
├── distributed/              ← 分布式通信
│   ├── parallel_state.py     ← TP/PP 状态管理
│   └── communication_op.py   ← AllReduce 等操作
│
└── transformers_utils/       ← tokenizer 等 HF 工具封装
```

---

## Part 3: 关键模块详解

### LLMEngine（引擎核心）

```python
# 简化的 LLMEngine.step() 逻辑：
class LLMEngine:
    def step(self):
        # 1. 调度：决定本次 iteration 处理哪些请求
        scheduler_output = self.scheduler.schedule()
        #   scheduler_output 包含：
        #   - scheduled_seq_groups: 要执行的请求
        #   - blocks_to_swap_in:  从 CPU swap 回 GPU 的 block
        #   - blocks_to_swap_out: 从 GPU swap 到 CPU 的 block
        #   - blocks_to_copy:     CoW 需要复制的 block

        # 2. 执行：模型前向传播 + 采样
        output = self.model_executor.execute_model(scheduler_output)
        #   output 包含每个序列的新 token 和 logprobs

        # 3. 更新：把新 token 加入序列，检查是否完成
        for seq_group, sample in zip(scheduled, output):
            seq_group.append_token(sample.token_id)
            if sample.token_id == eos_id or len(seq) >= max_tokens:
                seq_group.set_finished()

        # 4. 释放已完成请求的资源
        self.scheduler.free_finished_seqs()
```

### 请求状态机

```
  add_request()
       ↓
  ┌─────────┐
  │ WAITING  │ ← 等待 GPU 资源
  └────┬─────┘
       │ schedule() 分配了 block
       ↓
  ┌─────────┐
  │ RUNNING  │ ← 正在 GPU 上执行（prefill 或 decode）
  └────┬─────┘
       │
       ├──── 生成完成 ──→ FINISHED (释放 block)
       │
       ├──── block 不够 ──→ SWAPPED (KV Cache 移到 CPU)
       │                        ↓
       │                    等待 block 空闲
       │                        ↓
       └────────────────── 恢复 RUNNING
```

---

## Part 4: vLLM 中的模型实现

### 以 Qwen2 为例

vLLM 中的模型不是直接用 HuggingFace 的实现，而是重新实现的版本，针对推理做了优化：

```python
# 简化的 Qwen2 模型结构 (vllm/model_executor/models/qwen2.py)

class Qwen2ForCausalLM:
    def __init__(self, config):
        self.model = Qwen2Model(config)
        self.lm_head = VocabParallelEmbedding(...)  # 支持 TP 的 LM head

    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        hidden = self.model(input_ids, positions, kv_caches, attn_metadata)
        logits = self.lm_head(hidden)
        return logits

class Qwen2Model:
    def __init__(self, config):
        self.layers = [Qwen2DecoderLayer(config) for _ in range(config.num_layers)]

class Qwen2DecoderLayer:
    def __init__(self, config):
        self.self_attn = Qwen2Attention(config)   # 包含 QKV projection
        self.mlp = Qwen2MLP(config)               # gate + up + down
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)
```

vLLM 模型实现的关键差异：
1. **QKV 投影合并**：Q、K、V 的 linear layer 合并为一个，减少 kernel launch
2. **支持 TP**：linear layer 使用 `ColumnParallelLinear` / `RowParallelLinear`
3. **KV Cache 接口**：attention 层直接操作 KV Cache（读取/写入 PagedAttention 的 block）
4. **量化支持**：linear layer 可以替换为量化版本

### DeepSeek-V2 的特殊处理

DeepSeek-V2 的 MLA (Multi-head Latent Attention) 实现在 vLLM 中需要特殊适配：

```python
# DeepSeek-V2 attention 简化逻辑
class DeepseekV2Attention:
    # MLA: 缓存的是低维 latent 而不是完整 KV
    # kv_lora_rank << kv_heads * head_dim
    def forward(self, hidden, kv_cache):
        # 1. 计算 compressed KV latent
        kv_compressed = hidden @ W_kv_down   # 降维
        # 2. 存入 KV Cache（比标准 KV 小很多）
        kv_cache.append(kv_compressed)
        # 3. 从 cache 读取时需要"解压"
        k = kv_cache @ W_k_up   # 升维回来
        v = kv_cache @ W_v_up
        # 4. 做标准 attention
```

---

## Part 5: Attention Backend 选择

vLLM 支持多个 attention backend，自动根据硬件和配置选择：

```python
# vllm/attention/selector.py 的选择逻辑（简化）

def get_attn_backend():
    if is_flashinfer_available() and use_flashinfer:
        return FlashInferBackend      # FlashInfer: 高性能，支持 paged KV
    elif is_flash_attn_available():
        return FlashAttentionBackend  # FlashAttention-2: 最常用
    else:
        return XFormersBackend        # xFormers: fallback
```

| Backend | 特点 | 适用场景 |
|---|---|---|
| FlashAttention-2 | 成熟稳定，广泛支持 | 默认选择 |
| FlashInfer | 对 paged KV 优化更好，支持更多特性 | 新版 vLLM 倾向使用 |
| xFormers | 兼容性好，性能略低 | 旧硬件 fallback |

---

## Part 6: Backend Services 架构

当 vLLM 作为生产服务时，外围需要一系列基础设施：

### 完整的推理服务架构

```
┌──────────────────────────────────────────────────────────┐
│                     Client / Gateway                      │
│  ┌──────────────────────────────────────────────────┐    │
│  │  API Gateway / Load Balancer                      │    │
│  │  - 认证鉴权                                       │    │
│  │  - 限流 (Rate Limiting)                           │    │
│  │  - 路由 (Model Routing: 按模型名/版本/租户分发)     │    │
│  │  - 请求队列 / 优先级                               │    │
│  └──────────────┬───────────────────────────────────┘    │
│                 │                                        │
├─────────────────┼────────────────────────────────────────┤
│                 ↓                                        │
│  ┌──────────────────────────────────────────────────┐    │
│  │              Inference Proxy Layer                 │    │
│  │  - 请求预处理 (prompt template, context assembly)  │    │
│  │  - 响应后处理 (detoxification, output filtering)   │    │
│  │  - 请求日志 / 审计 (logging, tracing)              │    │
│  │  - A/B 实验分流                                    │    │
│  │  - 缓存层 (semantic cache: 相似 prompt 直接返回)    │    │
│  └──────────────┬───────────────────────────────────┘    │
│                 │                                        │
├─────────────────┼────────────────────────────────────────┤
│                 ↓                                        │
│  ┌─────────────────────────────────────────────────┐     │
│  │           vLLM Instance Pool                     │     │
│  │                                                  │     │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐  │     │
│  │  │ vLLM #1    │ │ vLLM #2    │ │ vLLM #3    │  │     │
│  │  │ Qwen-72B   │ │ Qwen-72B   │ │ Qwen-7B    │  │     │
│  │  │ TP=4       │ │ TP=4       │ │ TP=1       │  │     │
│  │  │ 4×A100     │ │ 4×A100     │ │ 1×A100     │  │     │
│  │  └────────────┘ └────────────┘ └────────────┘  │     │
│  │                                                  │     │
│  │  ┌────────────┐ ┌────────────┐                  │     │
│  │  │ vLLM #4    │ │ vLLM #5    │                  │     │
│  │  │ DeepSeek   │ │ GLM-4-9B   │                  │     │
│  │  │ MoE, EP    │ │ TP=1       │                  │     │
│  │  └────────────┘ └────────────┘                  │     │
│  └─────────────────────────────────────────────────┘     │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                    Infrastructure                         │
│  ┌─────────┐ ┌───────────┐ ┌──────────┐ ┌───────────┐  │
│  │ K8s /   │ │ Model     │ │ Monitor  │ │ Autoscaler│  │
│  │ Ray     │ │ Registry  │ │ Grafana  │ │ HPA/KEDA  │  │
│  │ Cluster │ │ (weights) │ │ Prometheus│ │           │  │
│  └─────────┘ └───────────┘ └──────────┘ └───────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 关键后端组件

**Model Registry / Weight Storage**
- 模型权重的版本管理和分发
- 通常用 S3/GCS/NAS + 本地 cache
- 模型热更新策略：rolling update，先启新实例再切流量

**Monitoring & Alerting**
- Prometheus 采集 vLLM metrics
- Grafana dashboard 展示：TTFT/TPOT P99、throughput、KV Cache usage、preemption rate
- 告警规则：TTFT P99 > 阈值、preemption rate > 0、GPU 温度过高

**Autoscaling**
- 基于 QPS / 队列深度 / GPU 利用率触发扩缩容
- 注意：LLM serving 的扩容时间长（下载模型 + 加载权重 + 预热），需要预留 buffer

---

## 交付物

| 文件 | 描述 |
|---|---|
| `vLLM request lifecycle 源码笔记.md` | 一个请求从 API 到 token 返回的完整路径，标注关键代码位置 |

## 自检问题

1. LLMEngine.step() 一次调用做了哪些事？
2. vLLM 中的模型实现和 HuggingFace Transformers 中的有什么区别？
3. Scheduler 和 Block Manager 各自负责什么？它们如何交互？
4. vLLM 如何支持新模型？需要实现哪些接口？
5. FlashAttention backend 和 FlashInfer backend 的差异是什么？
