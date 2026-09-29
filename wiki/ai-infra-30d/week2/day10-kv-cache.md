---
title: "Day 10: KVCacheManager、Prefix Caching 与 KV Connector（源码级）"
type: concept
tags: [day10, kv-cache, vllm-v1, prefix-caching, block-pool, kv-connector, hybrid-kv-cache, eviction]
created: 2026-06-01
updated: 2026-09-27
---

# Day 10: KVCacheManager、Prefix Caching 与 KV Connector（源码级）

> 文件：`vllm/v1/core/kv_cache_manager.py`（`KVCacheManager`）、`block_pool.py`（`BlockPool`、`FreeKVCacheBlockQueue`）、`kv_cache_utils.py`（`KVCacheBlock`、block hash、`get_kv_cache_config`）、`single_type_kv_cache_manager.py`（full attention / sliding window / Mamba 各自的 manager）、`kv_cache_coordinator.py`（多种 KV 类型共存时的协调）、`vllm/v1/kv_cache_interface.py`（`KVCacheSpec` / `KVCacheConfig`）。KV connector 在 `vllm/distributed/kv_transfer/kv_connector/v1/`。
>
> Day 4 讲的是 PagedAttention 论文里的 block table 思想。今天讲 V1 的实际实现，它和论文有几处关键差异：没有 swap，没有 copy-on-write，prefix caching 内建在 block 生命周期里而不是外挂功能。

## 今日目标

1. 说清一个 KV block 从分配、写满、被 hash 缓存、释放、被驱逐或被命中复用的完整生命周期。
2. 读懂 prefix caching 的 hash 链设计，知道它为什么是"自动"的、为什么默认开启、命中率怎么看。
3. 理解 hybrid KV cache（全注意力 + sliding window + Mamba 共存）为什么需要多个 manager。
4. 知道 KV connector 的接口，能画出 P/D 解耦和 CPU offload 各自走哪些回调。
5. 动手：验证 prefix cache 命中、观察驱逐、写一个最小 KV connector。

---

## Part 1: 数据结构

### KVCacheBlock 和 BlockPool

```python
# kv_cache_utils.py（简化）
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0            # 被多少个请求引用；0 才能进 free 队列
    _block_hash: BlockHash | None   # 写满并被缓存后才有
    prev_free_block / next_free_block   # 双向链表指针，用于 FreeKVCacheBlockQueue
```

`BlockPool` 持有全部物理 block（数量 = 启动日志里的 `# GPU blocks`），两个核心结构：

```
free_block_queue : FreeKVCacheBlockQueue   # 双向链表，头部是最久没用的（LRU）
cached_block_hash_to_block : dict[BlockHash, dict[block_id, KVCacheBlock]]
```

关键设计：**block 进 free 队列时不清 hash**。它既是"可分配的空闲 block"，又是"可被命中的缓存"。只有当它被从 free 队列头部取走去装别的内容时，才从 `cached_block_hash_to_block` 里删掉。这就是"驱逐"，是惰性的、LRU 的。

### block hash 链

一个 block 的 hash 不只取决于它自己的 16 个 token，还取决于它前面所有 token：

```
hash(block_i) = H( hash(block_{i-1}), token_ids[block_i], extra_keys )
extra_keys 包含：多模态输入的 hash、LoRA id、cache_salt（用户显式传的隔离盐）
```

这样两个请求只要在第 i 个 block 的 hash 相同，就保证从头到第 i 个 block 的全部 token 完全一致，可以安全共享 KV。hash 在 `Request` 侧增量计算（每写满一个 block 算一次），默认用内置 hash，`--prefix-caching-hash-algo sha256` 可切到抗碰撞的版本（多租户共享实例时建议开，防止 hash 碰撞导致跨租户读到别人的 KV）。

只有**写满**的 block 才会被缓存。最后一个不满的 block 永远不缓存，所以命中长度总是 block_size 的整数倍。这回答了 Day 4 的自检题"前缀长度不是 block_size 整数倍怎么办"。

---

## Part 2: KVCacheManager 的三个方法

调度器（Day 9）只调用它的三个方法。逐个读：

### get_computed_blocks(request) → (blocks, num_computed_tokens)

新请求进 running 前调用。按 `request.block_hashes` 逐个查 `cached_block_hash_to_block`，从第一个 miss 处停下。返回命中的 block 列表和对应 token 数。命中的 block 会 `ref_cnt += 1` 并从 free 队列摘出（如果它在里面）。

指标在这里累加：`prefix_cache_stats.queries += 请求的 block 数`，`hits += 命中数`。对应 `/metrics` 里的 `vllm:prefix_cache_queries` 和 `vllm:prefix_cache_hits`，命中率 = hits / queries。

细节：如果整个 prompt 全命中，调度器会故意少算一个 token（Day 9 伪代码里那行），因为至少要跑一次 forward 才有 logits 来采样。

### allocate_slots(request, num_new_tokens, ...) → new_blocks | None

为本步要新算的 token 分配 block：

1. 算需要几个新 block：`ceil((num_computed + num_new + lookahead) / block_size) - 已有 block 数`。
2. 如果 `free_block_queue` 里的数量不够（要扣掉刚被 prefix 命中但还在 free 队列里的那些，避免"借给自己"），返回 `None`，调度器据此抢占。
3. 从 free 队列头部取 block（`BlockPool.get_new_blocks`）：如果这个 block 有 hash，说明它是别人用过的缓存，先从 `cached_block_hash_to_block` 删掉，这就是驱逐。
4. `cache_full_blocks()`：把这个请求中刚被写满的 block 算 hash、登记进缓存字典。

注意分配和真正写 KV 是分开的：这里只是分 block id，KV 数据要等 worker 跑 forward 时按 `slot_mapping` 写入。

### free(request)

请求结束或被抢占时调用。所有 block `ref_cnt -= 1`，归零的**按倒序**放回 free 队列尾部。倒序的原因：请求尾部的 block 复用价值最低（越靠后的前缀越少人共享），让它们更靠近队列头部，先被驱逐。

### 三个方法串起来的生命周期

```
allocate_slots 取出 ──→ 请求写 KV ──→ 写满 → cache_full_blocks 登记 hash
      ↑                                         │
      │ 从 free 队列头取走时删 hash（驱逐）           │ 请求结束 free()：ref_cnt→0
      │                                         ↓
   free 队列头 ←──── LRU 排队 ←──── free 队列尾（仍带 hash，可被命中）
                                                ↑
            get_computed_blocks 命中 → ref_cnt+1，摘出 free 队列 ──┘
```

---

## Part 3: 为什么 V1 不需要 swap 和 CoW

**swap**：Day 9 讲过，被抢占请求的 block 回到 free 队列但保留 hash，重新调度时大概率命中。只要驱逐没发生，recompute 的实际代价是"最后一个不满 block + 重新分配"。这比 PCIe 来回拷贝便宜。

**copy-on-write**：V0 用 CoW 处理 beam search / `n>1` 时多个序列共享前缀再分叉。V1 里 `n>1` 在前端拆成 n 个独立请求（共享 prompt，靠 prefix cache 复用 KV），分叉后各写各的新 block，天然不需要 CoW。beam search 也被移到 `LLM.beam_search()` 这种应用层实现。

---

## Part 4: Hybrid KV Cache 与多 manager

Gemma 2/3、Qwen3-Next、MiniMax、Jamba 这类模型里不同层的 KV 需求不同：全注意力层要存全部 token，sliding window 层只要最近 W 个，Mamba 层只要一个固定大小的 state。V1 的处理：

1. 启动时每个 attention 层向 `get_kv_cache_config()` 报告自己的 `KVCacheSpec`（`FullAttentionSpec` / `SlidingWindowSpec` / `MambaSpec` 等）。
2. 相同 spec 的层归为一个 `KVCacheGroup`，共享一套 block；不同 group 的 page 大小要对齐（sliding window 层可能用更大的 block_size 来凑）。
3. `KVCacheCoordinator` 为每个 group 建一个 `SingleTypeKVCacheManager`，`allocate_slots` 时逐 group 分配。`SlidingWindowManager` 会把滑出窗口的 block 提前 free，这是它省显存的来源。
4. prefix caching 在 hybrid 模型上的支持随版本变化，看启动日志有没有 "prefix caching disabled for hybrid model" 之类提示。

看你模型属于哪种：启动日志里 `KV cache config` 相关行，或 `grep -n "KVCacheSpec" $VLLM_ROOT/model_executor/models/<model>.py`。

---

## Part 5: KV Connector 接口

KV connector 是 V1 里把 KV 搬出本机显存的唯一正规通道，P/D 解耦（Day 24）、CPU/SSD offload、跨实例 prefix 共享全走它。配置：

```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
--kv-transfer-config '{"kv_connector":"SharedStorageConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"/tmp/kv"}}'
```

`KVConnectorBase_V1` 的方法分两侧：

| 侧 | 方法 | 调用时机 | 用途 |
|---|---|---|---|
| Scheduler | `get_num_new_matched_tokens(request, num_computed_tokens)` | 调度新请求前 | 报告远端已有多少 token 的 KV，可要求异步加载（请求进 `WAITING_FOR_REMOTE_KVS`） |
| Scheduler | `update_state_after_alloc(request, blocks, num_external_tokens)` | 分完 block 后 | 记下要把远端 KV 装进哪些 block |
| Scheduler | `build_connector_meta(scheduler_output)` | 每步打包输出时 | 生成 worker 侧需要的元数据（哪些 block 要 load/save） |
| Scheduler | `request_finished(request, block_ids)` | 请求结束 | 决定 block 是否延迟释放（等异步 save 完成） |
| Worker | `register_kv_caches(kv_caches)` | 启动 | 拿到 KV tensor 指针，RDMA 场景要注册内存 |
| Worker | `start_load_kv(forward_context)` | forward 开始前 | 发起加载 |
| Worker | `wait_for_layer_load(layer_name)` | 每层 attention 前 | 逐层等待，允许 load 和计算流水 |
| Worker | `save_kv_layer(layer_name, kv_layer, attn_metadata)` | 每层 attention 后 | 逐层保存 |
| Worker | `wait_for_save()` | forward 结束 | 等所有 save 完成 |
| Worker | `get_finished()` | 每步 | 报告哪些请求的异步传输完成了 |

逐层回调是关键设计：prefill 算第 i+1 层时第 i 层的 KV 已经在往外发，传输和计算重叠。

内置实现：`SharedStorageConnector`（写本地文件，教学用）、`NixlConnector`（RDMA/UCX，P/D 解耦的主力）、`LMCacheConnectorV1`（多级缓存，CPU/磁盘/远端）、`MultiConnector`（串联多个）、`OffloadingConnector`（CPU offload，较新版本）。Mooncake 通过 LMCache 或独立 connector 接入，以当前文档为准。

---

## Part 6: 动手实验

### 实验 1: 看命中率，验证 hash 链

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct   # V1 默认开 prefix caching；对照组加 --no-enable-prefix-caching
```

```python
import time, requests
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
sys_prompt = "You are an expert assistant. " * 300     # 约 2000 token
def m():
    t = requests.get("http://localhost:8000/metrics").text
    q = [l for l in t.splitlines() if l.startswith("vllm:prefix_cache_queries")]
    h = [l for l in t.splitlines() if l.startswith("vllm:prefix_cache_hits")]
    return q[-1].split()[-1], h[-1].split()[-1]
for i in range(5):
    t0 = time.perf_counter()
    c.chat.completions.create(model="Qwen/Qwen2.5-7B-Instruct", max_tokens=8, stream=False,
        messages=[{"role":"system","content":sys_prompt},{"role":"user","content":f"Q{i}"}])
    print(i, f"{(time.perf_counter()-t0)*1000:.0f}ms", "queries/hits =", m())
```

再做两组变体：把 system prompt 的第 1 个词改掉重发（预期全 miss，验证 hash 链）；给请求加 `extra_body={"cache_salt":"tenant-b"}`（预期 miss，验证隔离）。

### 实验 2: 观察驱逐

用 `--gpu-memory-utilization 0.4 --max-model-len 4096` 起一个 KV 很小的实例。先用 20 个不同的长 system prompt 各发一次把缓存填满，再重发第 1 个。它是否还命中？把 20 改成 5 再试。结合 `# GPU blocks` 数算出理论上能缓存多少个 2000 token 的前缀。

### 实验 3: 抢占后的 recompute 到底重算了多少

Day 9 实验 3 的配置下，在 `allocate_slots` 和 `get_computed_blocks` 加打印（或用 `VLLM_LOGGING_LEVEL=DEBUG`），找一个被抢占又恢复的请求，记录它恢复时 `num_computed_tokens` 从 0 变成多少。如果接近原值，说明 prefix cache 兜住了 recompute。

### 实验 4: 写一个最小 KV connector

目标：把每个请求 prefill 完成的 KV 逐层写到本地 `/tmp/kv/<hash>.pt`，下次相同前缀从文件加载。步骤：

1. 读 `SharedStorageConnector` 的实现，它就是这个功能的官方参考。
2. 复制成 `MyFileConnector`，只保留 `get_num_new_matched_tokens`、`update_state_after_alloc`、`build_connector_meta`、`start_load_kv`、`save_kv_layer`、`wait_for_save`。
3. 注册：`KVConnectorFactory.register_connector("MyFileConnector", "my_pkg.connector", "MyFileConnector")`，或通过 vLLM 插件入口点。
4. 起两个实例（不同端口）指向同一目录，实例 A 发过的 prompt 在实例 B 上 TTFT 是否下降。

交付：代码 + 两实例 TTFT 对比。这是 Day 24 P/D 解耦实验的地基。

### 实验 5: FP8 KV 的显存和质量

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype fp8      # 看启动日志 # GPU blocks 翻倍
```

用同一组 20 条长文本摘要 prompt，对比 bf16 KV 和 fp8 KV 的输出差异（可以用 rouge 或让更强模型打分）。记录 `# GPU blocks`、最大并发、质量差异。注意 fp8 KV 需要 attention backend 支持（FA3 / FlashInfer / Triton），日志里会说用了哪个。

---

## Part 7: Eviction 与稀疏策略（模型侧，简述）

以上都是"精确"的 KV 管理：不丢信息。当显存实在不够时还有一类**有损**方法，它们改变模型的注意力范围而不是内存管理：

| 策略 | 思路 | KV 大小 | 代价 |
|---|---|---|---|
| Sliding window（Mistral、Gemma 部分层） | 训练时就只看最近 W 个 token | O(W) | 模型决定，推理端无损 |
| StreamingLLM（attention sink） | 保留开头几个 token + 最近 N 个 | O(1) | 丢中间信息，长依赖任务掉点 |
| H2O / SnapKV 类 | 按 attention score 只留重要 token | O(k) | 需要额外统计，不同 head 不同 |
| MLA（DeepSeek） | 训练时把 KV 压成低秩 latent | 每 token 约为 GQA 的 1/5 到 1/10 | 需要专用 attention kernel（Day 13） |

vLLM 原生只做 sliding window（通过 hybrid manager）和 MLA；StreamingLLM/H2O 类需要改 attention backend，工程上少见。Day 27 长上下文再讨论。

---

## 交付物

| 文件 | 描述 |
|---|---|
| `kv-cache-manager-notes.md` | block 生命周期图（带你版本的行号）、hash 链解释、hybrid manager 说明 |
| `prefix-cache-experiments.md` | 实验 1/2/3/5 的数据 |
| `my_file_connector.py` | 实验 4 |

## 自检问题

1. 一个 block 被 `free()` 之后，什么情况下会被再次命中，什么情况下会被驱逐？两者的判定点在哪个方法里？
2. 为什么 `free()` 要倒序归还 block？正序会怎样？
3. 两个租户共用一个实例，prompt 完全相同但要求 KV 不共享，用哪个参数？它进入 hash 的哪一部分？
4. `allocate_slots` 返回 `None` 的时候，KV 使用率一定是 100% 吗？（提示：free 队列里那些刚被命中的 block）
5. P/D 解耦时 prefill 节点的 `save_kv_layer` 和 decode 节点的 `wait_for_layer_load` 如何配合做到传输和计算重叠？
6. 为什么 MLA 模型的 prefix cache 命中收益比 GQA 模型更大？（提示：每 token KV 大小和 prefill 计算量的比值）
