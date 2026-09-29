---
title: "Day 13: 模型执行层：GPUModelRunner、CUDA Graph、Attention Backend 与 Sampler"
type: concept
tags: [day13, vllm-v1, gpu-model-runner, torch-compile, cuda-graph, flash-attention, flashinfer, mla, sampler, logits-processor, structured-output]
created: 2026-06-01
updated: 2026-09-27
---

# Day 13: 模型执行层：GPUModelRunner、CUDA Graph、Attention Backend 与 Sampler

> 文件：`vllm/v1/worker/gpu_model_runner.py`（`GPUModelRunner`）、`gpu_input_batch.py`（`InputBatch`、`CachedRequestState`）、`vllm/v1/attention/backends/`（`flash_attn.py`、`flashinfer.py`、`triton_attn.py`、`mla/`）、`vllm/compilation/`（torch.compile 后端、piecewise CUDA graph）、`vllm/config.py` 里的 `CompilationConfig`、`vllm/v1/sample/`（`sampler.py`、`logits_processor/`、`rejection_sampler.py`）、`vllm/v1/structured_output/`。
>
> Day 8 讲到 SchedulerOutput 到达 worker 就停了。今天把 worker 内部走完：从 SchedulerOutput 到 logits 到采样出的 token id。

## 今日目标

1. 读懂 `execute_model()` 的五个阶段，知道 persistent batch 省掉了什么。
2. 说清 torch.compile 和 piecewise CUDA graph 在 vLLM 里各做什么、为什么 attention 要被切出来、`--enforce-eager` 关掉了什么。
3. 知道 attention backend 怎么选、FA2/FA3/FlashInfer/Triton/MLA 各自的适用条件，能读懂 FlashAttention 的 tiling 和 online softmax。
4. 读懂 Sampler 的处理顺序，能写一个自定义 logits processor，知道结构化输出的 bitmask 在哪一步生效。
5. 动手：对比 CUDA graph 模式、切换 attention backend、写 logits processor、测结构化输出开销。

---

## Part 1: execute_model() 的五个阶段

```python
# gpu_model_runner.py（简化，保留真实调用顺序）
def execute_model(self, scheduler_output):
    self._update_states(scheduler_output)          # 1. 增量更新 persistent batch
    if not scheduler_output.total_num_scheduled_tokens:
        return EMPTY_MODEL_RUNNER_OUTPUT
    attn_metadata, ... = self._prepare_inputs(scheduler_output)   # 2. 拼输入、建 attention metadata
    # 多模态：_execute_mm_encoder() 跑 vision encoder，把 embedding 塞进 inputs_embeds
    with set_forward_context(attn_metadata, self.vllm_config, num_tokens=num_input_tokens):
        hidden_states = self.model(input_ids=..., positions=..., inputs_embeds=...)   # 3. forward
    logits = self.model.compute_logits(hidden_states[sample_indices])               # 4. 只取每个请求最后一个位置
    # 结构化输出：apply_grammar_bitmask(logits, scheduler_output.grammar_bitmask)
    sampler_output = self.sampler(logits, sampling_metadata)                        # 5. 采样
    # spec decode：self.drafter.propose(...) 生成下一步的 draft token
    return ModelRunnerOutput(sampled_token_ids=..., logprobs=..., spec_token_ids=...)
```

### 阶段 1: persistent batch

`InputBatch` 是一个固定 `max_num_seqs` 行的表，每行一个请求，列是预分配的 CPU tensor（pinned）和对应的 GPU tensor：

```
token_ids_cpu       [max_num_reqs, max_model_len]   每个请求的全部 token
num_computed_tokens_cpu [max_num_reqs]
block_table         [max_num_reqs, max_blocks_per_req]（每个 KV group 一张）
temperature / top_p / top_k / min_p / repetition_penalty ... [max_num_reqs]
generators          dict[row → torch.Generator]   （带 seed 的请求）
req_id_to_index     dict
```

`_update_states` 只做增量：完成的请求删行，新请求加行（把 `NewRequestData` 里的 prompt token、sampling params 写进对应行），继续的请求只更新 `num_computed_tokens` 和新增的 block id。然后 `condense()` 把空洞压实。

V0 每步重建一份 `SequenceGroupMetadata` 列表再转 tensor，请求数上百时这段 Python 开销就是 GPU 空转的主因。persistent batch 把它变成几次 tensor 切片赋值。

### 阶段 2: _prepare_inputs

从 batch 表里拼出本步的扁平输入。所有请求的新 token 首尾相接成一维：

```
请求 A 本步 3 个新 token，B 1 个，C 64 个（prefill 分片）
input_ids      = [a1 a2 a3 | b1 | c1 ... c64]                长度 68
positions      = [pA pA+1 pA+2 | pB | pC ... pC+63]
query_start_loc = [0, 3, 4, 68]                             每个请求在扁平数组里的起点
seq_lens       = [A 的总长, B 的总长, C 的总长]                attention 要看到的 KV 长度
slot_mapping   = 每个新 token 的 KV 该写到哪个物理 block 的哪个槽
```

`slot_mapping` 由 block table 算出：`block_ids[pos // block_size] * block_size + pos % block_size`。attention kernel 用它写 KV，用 block table 读 KV。这就是 PagedAttention 在 V1 里的全部落地：**没有单独的 paged kernel 概念，任何支持 block table 输入的 attention kernel 都是 paged 的。**

`AttentionMetadataBuilder`（每个 backend 一个）把这些打包成 backend 需要的格式，比如 FlashInfer 要预先 plan。

### 阶段 3: forward 与 forward context

`set_forward_context` 把 attention metadata 放进一个全局上下文，模型里每个 `Attention` 层通过它拿到自己的 KV cache tensor 和 metadata，而不是层层传参。这样模型代码可以被 torch.compile 整体捕获。

### 阶段 4/5 见 Part 4。

---

## Part 2: torch.compile 与 piecewise CUDA graph

### 两个不同的东西

- **torch.compile（Inductor）**：图级优化，做算子融合（RMSNorm + 量化、SiLU + mul、residual + norm 等）、消除冗余拷贝、生成 Triton kernel。效果是 kernel 数量变少、单个 kernel 更快。
- **CUDA graph**：把一段 GPU 操作序列录制下来，之后一次 launch 回放，消除每个 kernel 的 CPU launch 开销。效果是 CPU 侧时间变少。decode 每步只算 1 个 token，launch 开销占比最高，CUDA graph 对 decode 收益最大。

### 为什么是 piecewise

CUDA graph 要求录制时的 tensor 形状和地址固定。attention kernel 的输入（seq_lens、block table、slot_mapping 内容）每步都变，而且 FlashInfer 这类 backend 还有 plan 阶段的 CPU 逻辑，不能被录进 graph。vLLM 的做法：

```
torch.compile 把模型图在每个 attention 算子处切开（splitting_ops）
  [embed + layer0 前半] → attention0 → [layer0 后半 + layer1 前半] → attention1 → ... → [最后一段 + norm]
每一段 non-attention 子图录成 CUDA graph（按 num_tokens 分桶：1, 2, 4, 8, 16, ..., 512）
attention 段 eager 执行
```

这就是 piecewise CUDA graph。启动日志里 `Capturing CUDA graphs` 那几十秒就是在按桶录制。

较新版本引入 `cudagraph_mode`：

| mode | 含义 | 适用 |
|---|---|---|
| `NONE` | 不用 CUDA graph | 调试 |
| `PIECEWISE` | 上面的方案（长期默认） | 通用 |
| `FULL` | 整个 forward 含 attention 一起录 | attention backend 支持时（如 FA3 decode），decode 更快 |
| `FULL_DECODE_ONLY` | 纯 decode 步用 FULL，含 prefill 的步用 eager/piecewise | 混合 workload |
| `FULL_AND_PIECEWISE` | decode 用 FULL，其余用 PIECEWISE | 较新版本推荐 |

```bash
vllm serve ... -O3                                   # 编译级别快捷方式
vllm serve ... --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256]}'
vllm serve ... --enforce-eager                       # 同时关掉 torch.compile 和 CUDA graph
```

`cudagraph_capture_sizes` 决定桶：本步 token 数向上取整到最近的桶，多出来的位置 pad。桶太少 pad 浪费大，桶太多录制慢、显存占用多（每个 graph 有自己的中间 buffer）。

`--enforce-eager` 常被当作"关 CUDA graph"，实际两个都关了。想只关一个：`--compilation-config '{"cudagraph_mode":"NONE"}'` 只关 graph，`{"level":0}`（新版 `"mode"`）只关编译。

### 编译缓存

首次启动编译要几十秒到几分钟，产物缓存在 `~/.cache/vllm/torch_compile_cache/<hash>/`。hash 包含模型、并行配置、编译配置。生产镜像里预热这个目录能明显缩短冷启动（Day 20）。

---

## Part 3: Attention Backend

### 选择逻辑

`vllm/attention/selector.py` → 平台类的 `get_attn_backend_cls()`，按顺序看：是否 MLA 模型、dtype、`kv_cache_dtype`、head_size、block_size、GPU 架构、环境变量覆盖。

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER vllm serve ...     # 强制指定；新版本也有 --attention-backend
# 启动日志会打印 "Using XXX backend"
```

| Backend | 文件 | 条件 / 特点 |
|---|---|---|
| `FLASH_ATTN`（FA2/FA3） | `backends/flash_attn.py`，kernel 在 `vllm_flash_attn` | 默认。Hopper 上自动用 FA3，支持 fp8 KV；Ampere 用 FA2 |
| `FLASHINFER` | `backends/flashinfer.py` | Blackwell 上倾向默认；decode 有 TRT-LLM kernel 路径；采样也有 FlashInfer 版本；需要 plan 阶段 |
| `TRITON_ATTN` | `backends/triton_attn.py` | 纯 Triton，兼容性最好，性能中等；老卡或 fp8 KV 在 FA2 不支持时的兜底 |
| `FLEX_ATTENTION` | `backends/flex_attention.py` | PyTorch FlexAttention，用于奇特 mask |
| MLA 系列 | `backends/mla/{flashmla,cutlass_mla,triton_mla,flashinfer_mla,flashattn_mla}.py` | DeepSeek V2/V3/R1 专用，各自对应硬件 |
| ROCm 系列 | `backends/rocm_*.py` | AMD |

每个 backend 由三部分组成：`AttentionBackend`（声明能力、KV cache 形状）、`AttentionMetadataBuilder`（把 InputBatch 转成 kernel 参数）、`AttentionImpl`（`forward()` 调 kernel）。加新 backend 就是实现这三个类。

### FlashAttention 原理（读懂就够，不要求手写）

标准 attention 要把 N×N 的 S 和 P 矩阵写到 HBM 再读回。N=8192 时每个 head 256 MB，序列越长越 memory-bound。FlashAttention 的思路：分块，中间结果只在 SRAM 里。

**Tiling**：Q、K、V 沿序列维切块，每块能放进 SRAM。外层循环 Q 块，内层循环 K/V 块。

**Online softmax**：softmax 需要整行的最大值和 exp 和，分块时用两个累积量修正：

```
初始 m = -inf, l = 0, O = 0
每来一个 K/V 块:
  S  = Q_blk @ K_blk^T
  m' = max(m, rowmax(S))
  l' = l * exp(m - m') + rowsum(exp(S - m'))
  O  = O * (l * exp(m - m') / l') + exp(S - m') @ V_blk / l'
  m, l = m', l'
```

每块只需要 O(B_q × B_k) 的中间存储，S 和 P 从不落到 HBM。HBM 读写从 O(N²) 降到 O(N²d²/M)，M 是 SRAM 大小。

**FlashAttention-2**：减少非矩阵乘法运算、把并行从 batch×head 扩展到序列维、warp 之间的分工改成按 K/V 切而不是按 Q 切，减少 shared memory 同步。**FlashAttention-3**：针对 Hopper，用 warp specialization 让 TMA 搬数据、WGMMA 算矩阵、softmax 三者流水，并支持 FP8。

**FlashDecoding**：decode 时 Q 只有 1 行，序列维没法并行，SM 大量空闲。解决：把 KV 沿序列切成多个 split 并行算 partial (O, m, l)，最后用 online softmax 的合并公式归约。这就是 FA 的 `num_splits` 参数和 FlashInfer decode kernel 的做法。长上下文 decode 的吞吐主要靠它。

### 模型结构对 backend 的影响

- **GQA（Qwen2.5/3、Llama 3）**：kv_heads 少，decode 时同一 KV 被多个 Q head 读，kernel 会把共享同一 KV head 的 Q head 打包成一个 tile 提高算术密度。
- **MLA（DeepSeek）**：KV 是压缩 latent，decode 时把 Q 投影到 latent 空间直接和 latent 做 attention（"absorb" 技巧），KV 读取量降一个数量级，但需要专用 kernel，prefill 和 decode 走不同路径。
- **Sliding window（Mistral、Gemma）**：backend 收到 window 参数，只读最近 W 个 block，配合 Day 10 的 `SlidingWindowManager`。
- **线性注意力（MiniMax、Qwen3-Next 的部分层）**：不用 attention backend，走 Mamba/linear attention 专用算子，KV 是固定大小 state。

---

## Part 4: Sampler、Logits Processor 与结构化输出

### Sampler 的顺序

`vllm/v1/sample/sampler.py`，`forward(logits, sampling_metadata)`：

```
1. logits 转 float32
2. 结构化输出 bitmask（在 model runner 里已经 apply，非法 token 置 -inf）
3. allowed_token_ids / bad_words
4. logits processors（内置：min_tokens、logit_bias、min_p；用户自定义的按注册顺序）
5. penalties：repetition / frequency / presence（需要每个请求的 prompt+output token 统计）
6. temperature（0 → greedy 分支直接 argmax）
7. top-k / top-p（可选 FlashInfer 实现）
8. 采样（带 seed 的请求用各自的 generator）
9. logprobs（如果请求了）：在处理后的分布上算
```

关键点：只有 batch 里**存在**某个特性的请求时才走对应分支（`sampling_metadata` 里有 `no_penalties`、`no_top_p` 这类标志）。一个请求要 repetition_penalty 就让整个 batch 多跑一遍 penalty kernel。这就是 Day 8 说"混合采样参数更贵"的原因。生产上统一采样参数是免费的优化。

### 自定义 logits processor

V1 的接口是 `vllm.v1.sample.logits_processor.LogitsProcessor`，按 batch 处理而不是按请求：

```python
from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate, MoveDirectionality
import torch

class BanFirstTokenProcessor(LogitsProcessor):
    """前 3 步禁止输出指定 token（示例：禁止一上来就输出 "Sorry"）"""
    def __init__(self, vllm_config, device, is_pin_memory):
        self.banned = {}       # row_index -> (token_id, steps_left)
    def is_argmax_invariant(self) -> bool:
        return False           # 会改变 argmax → greedy 请求也要跑
    def update_state(self, batch_update: BatchUpdate | None):
        if batch_update is None: return
        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            tid = (params.extra_args or {}).get("ban_token_id")
            if tid is not None: self.banned[index] = [tid, 3]
        for index in batch_update.removed: self.banned.pop(index, None)
        for a, b, direction in batch_update.moved:    # condense() 搬行时同步
            ...
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for row, (tid, left) in list(self.banned.items()):
            if left > 0:
                logits[row, tid] = float("-inf"); self.banned[row][1] -= 1
        return logits
```

```bash
vllm serve ... --logits-processors my_pkg.lp.BanFirstTokenProcessor
# 请求侧：extra_body={"vllm_xargs": {"ban_token_id": 1234}}
```

`update_state` 必须跟着 `InputBatch` 的加行、删行、搬行走，否则 row index 会串。这是 V1 logits processor 和 V0 按请求回调的最大区别，也是它能不拖慢 batch 的原因。

### 结构化输出的完整路径

```
请求带 response_format / guided_json / guided_grammar
  → Processor 校验 → EngineCore: Request 状态 WAITING_FOR_FSM
  → StructuredOutputManager 在线程池编译 grammar（xgrammar 默认；guidance / outlines / lm-format-enforcer 可选）
  → 编译完成 → WAITING → 被调度
  → 每步 schedule() 末尾：grammar_bitmask() 生成 [num_reqs, vocab/32] 的 int32 bitmask
  → 随 SchedulerOutput 到 worker → model runner 在 sampler 前 apply_grammar_bitmask
  → 采样 → update_from_output() 用接受的 token 推进 FSM
```

```bash
vllm serve ... --structured-outputs-config '{"backend":"xgrammar"}'    # 旧参数名 --guided-decoding-backend
```

开销在三处：grammar 编译（首次几十 ms 到秒级，有缓存）、每步 bitmask 生成（CPU，和 FSM 状态数相关）、bitmask apply（GPU，很便宜）。复杂 JSON schema 的 TTFT 会明显变高，这是 Day 23 对比 SGLang 的一个维度。spec decode 和结构化输出同时开时，bitmask 要为每个 draft 位置各算一份。

---

## Part 5: 动手实验

### 实验 1: CUDA graph 模式对比

同一 workload（decode 主导：input 128 / output 512 / concurrency 64）跑四组：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enforce-eager
vllm serve Qwen/Qwen2.5-7B-Instruct --compilation-config '{"cudagraph_mode":"NONE"}'
vllm serve Qwen/Qwen2.5-7B-Instruct                                      # 默认 piecewise
vllm serve Qwen/Qwen2.5-7B-Instruct --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

记录 output tokens/s、TPOT P50、启动时间、显存占用。用 `nsys profile --trace=cuda,nvtx -o run python ...` 或 `--profiler-config` 抓一次 decode step，数 kernel launch 次数。预期 eager 下每层十几次 launch，graph 下整个 forward 只有几次。

### 实验 2: 切 attention backend

```bash
VLLM_ATTENTION_BACKEND=FLASH_ATTN   vllm serve Qwen/Qwen2.5-7B-Instruct
VLLM_ATTENTION_BACKEND=FLASHINFER   vllm serve Qwen/Qwen2.5-7B-Instruct
VLLM_ATTENTION_BACKEND=TRITON_ATTN  vllm serve Qwen/Qwen2.5-7B-Instruct
```

两种 workload：长 prefill（input 8k / output 32）和长 decode（input 512 / output 1k / concurrency 32）。记录 TTFT、TPOT。再加 `--kv-cache-dtype fp8` 看哪个 backend 拒绝启动。写一段"在我这张卡上选哪个、为什么"。

### 实验 3: 采样参数的 batch 代价

concurrency 64，三组请求：全部 greedy；全部 `temperature=0.7, top_p=0.9`；一半 greedy 一半带 `repetition_penalty=1.2`。对比 TPOT。用 py-spy 或 torch profiler 看 sampler 时间占比。

### 实验 4: 自定义 logits processor

实现 Part 4 的 `BanFirstTokenProcessor`，用 `--logits-processors` 加载，验证：带 `ban_token_id` 的请求前 3 个 token 确实没有该 token；不带的请求不受影响；高并发下 TPOT 变化小于 5%。

### 实验 5: 结构化输出开销

同一 prompt，三组：自由文本、简单 JSON schema（3 个字段）、复杂 schema（嵌套 + enum + 数组，20 个字段）。concurrency 1 和 32 各跑一次。记录 TTFT（含 grammar 编译）、TPOT、`vllm:request_queue_time_seconds`。切换 `backend` 为 `guidance` 重跑。结论写进 Day 23 的三框架对比表。

---

## 交付物

| 文件 | 描述 |
|---|---|
| `model-runner-notes.md` | `execute_model` 五阶段图、`_prepare_inputs` 的扁平输入示例、piecewise CUDA graph 示意 |
| `attention-backend-report.md` | 实验 1/2 数据、backend 选择结论、FlashAttention 原理一页纸 |
| `sampler-and-structured-output.md` | 实验 3/5 数据、logits processor 代码 |

## 自检问题

1. persistent batch 相对 V0 每步重建 metadata 省掉了什么？请求数从 8 涨到 256 时两者的 CPU 时间各怎么变？
2. 为什么 attention 要从 CUDA graph 里切出去？`FULL` 模式怎么解决这个问题、有什么前提？
3. `slot_mapping` 和 block table 分别在 attention 的写路径还是读路径上用？
4. decode 阶段 FlashAttention 为什么需要 split-KV？split 数太多会有什么代价？
5. 一个 batch 里只有一个请求用了 `logit_bias`，其他请求的采样会变慢吗？为什么？
6. 结构化输出请求的 TTFT 比普通请求多出的部分主要来自哪里？怎么用指标区分是 grammar 编译还是排队？
7. MLA 模型的 decode 为什么比同规模 GQA 模型更不容易 memory-bound？这对 batch size 选择有什么影响？
