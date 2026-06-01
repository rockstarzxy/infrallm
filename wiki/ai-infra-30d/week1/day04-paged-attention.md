---
title: "Day 4: PagedAttention 和 Continuous Batching"
type: concept
tags: [day4, paged-attention, continuous-batching, kv-cache, vllm]
created: 2026-06-01
updated: 2026-06-01
---

# Day 4: PagedAttention 和 Continuous Batching

## Part 1: KV Cache 管理的痛点

回顾 Day 1：推理需要为每个请求维护 KV Cache，显存占用随序列长度线性增长。

**在 vLLM 之前，系统怎么管理 KV Cache？**

传统方案：为每个请求预分配连续的显存块，大小 = max_seq_len × kv_size_per_token。

问题：

1. **内部碎片 (Internal Fragmentation)**：如果预分配了 2048 token 的空间，但请求实际只生成了 200 token，剩余 1848 个位置的显存全部浪费。
2. **外部碎片 (External Fragmentation)**：请求结束释放显存后，空闲块不连续，无法分配给需要大块连续显存的新请求。
3. **预分配浪费**：因为不知道一个请求会生成多少 token，只能按最大长度预分配。

实际浪费有多严重？vLLM 论文中的测量：**传统系统的 KV Cache 显存浪费高达 60-80%**。这意味着原本能同时处理 100 个请求的 GPU，实际只能处理 20-40 个。

---

## Part 2: PagedAttention 的核心思想

PagedAttention 借鉴了操作系统的**虚拟内存分页**机制：

### 操作系统类比

| OS 概念 | PagedAttention 对应 |
|---|---|
| 虚拟地址空间 | 请求的逻辑 KV Cache |
| 物理内存页 (page) | 物理 KV Cache block |
| 页表 (page table) | Block table |
| 页大小 (page size) | Block size (如 16 tokens) |
| 按需分页 | 按需分配 block |
| Copy-on-Write | Fork/共享前缀时的 COW |

### 工作原理

**Step 1: 分块存储**

不再为一个请求分配连续的大块显存，而是将 KV Cache 切成固定大小的 **block**（默认 16 个 token 一个 block）。

```
传统方案：
Request A: [████████████████████░░░░░░░░░░░░] ← 预分配 2048，只用了 640，浪费 1408

PagedAttention：
Request A: [block0][block1][block2]...[block39][block40(部分填充)]
            ↓       ↓       ↓
          物理 block 可以在显存中不连续存放
```

**Step 2: Block Table 映射**

每个请求维护一个 block table，记录"第 i 个逻辑 block → 第 j 个物理 block"的映射。

```
Request A 的 Block Table:
  逻辑 block 0 → 物理 block 7
  逻辑 block 1 → 物理 block 23
  逻辑 block 2 → 物理 block 5
  ...

Request B 的 Block Table:
  逻辑 block 0 → 物理 block 12
  逻辑 block 1 → 物理 block 3
  ...
```

**Step 3: 按需分配**

- 请求开始时，只分配 prefill 需要的 block 数
- 每当 decode 填满当前 block（每 16 个 token），才分配一个新的物理 block
- 请求结束后，归还所有物理 block

### 解决了什么

1. **消除内部碎片**：最多浪费 block_size - 1 个 token 的空间（最后一个 block 部分填充）
2. **消除外部碎片**：物理 block 不需要连续，任何空闲 block 都能被任何请求使用
3. **消除预分配浪费**：按需分配，用多少分多少

vLLM 论文的实测结果：KV Cache 利用率从 ~20-40% 提升到 ~96%+。相同显存下能服务 2-4x 更多的并发请求。

---

## Part 3: 更深入理解 Block Manager

### Block 的数据结构

每个物理 block 存储 block_size 个 token 的 K 和 V：

```
一个 block 的显存大小:
= 2(K,V) × n_layers × n_kv_heads × head_dim × block_size × bytes_per_element

Llama-3-8B, block_size=16, FP16:
= 2 × 32 × 8 × 128 × 16 × 2 = 2,097,152 bytes = 2 MB

总 GPU blocks 数 = 可用 KV Cache 显存 / 每个 block 的大小
```

如果可用 KV Cache 显存为 20 GB：
- 20 GB / 2 MB = 10,240 blocks
- 能存 10,240 × 16 = 163,840 个 token 的 KV Cache

### Preemption 机制

当物理 block 全部用完而新请求到来时，vLLM 有两种策略：

1. **Swap**：将低优先级请求的 KV Cache 从 GPU 移到 CPU 内存，腾出 block。该请求暂停，等有空闲 block 时再 swap 回来继续。
2. **Recompute**：直接丢弃被抢占请求的 KV Cache，等资源够时从头重算 prefill。

Preemption 是很昂贵的操作（swap 有数据搬移开销，recompute 有重复计算开销），应该尽量避免。

### Copy-on-Write (COW)

当多个请求共享前缀（如相同的 system prompt）时，可以共享物理 block：

```
Request A: system_prompt + user_msg_A
Request B: system_prompt + user_msg_B

共享的 block table:
  A: [shared_block0] [shared_block1] [A_block2] [A_block3]
  B: [shared_block0] [shared_block1] [B_block2] [B_block3]
```

当需要修改共享 block 时（某个请求要在共享 block 中写入新 token），先复制一份，再修改——即 Copy-on-Write。

---

## Part 4: Continuous Batching

### Static Batching 的问题

传统 batching：收集一组请求，一起处理完毕后再收集下一组。

```
Static Batching:
时间 ──→
Batch 1: [Req A ████████████] [Req B ████] [Req C ████████]
                                    ↑ B 结束了，但必须等 A 和 C 完成
                                      GPU 空闲周期被浪费

Batch 2:                                              [Req D ...] [Req E ...]
                                                       ↑ 必须等 Batch 1 全部结束
```

问题：
- 最短的请求也要等最长的请求完成才能释放资源
- GPU 利用率低：短请求结束后 batch 内有空位但无法填入新请求
- 新请求必须等当前 batch 完成才能开始处理

### Continuous Batching (Iteration-level Scheduling)

Continuous batching 的核心改变：**调度粒度从 request 级别降到 iteration（单步 decode）级别。**

```
Continuous Batching:
时间 ──→
Step 1: [A_decode] [B_decode] [C_prefill]
Step 2: [A_decode] [B_decode] [C_decode]
Step 3: [A_decode] [B_done ✓] [C_decode]  ← B 结束了
Step 4: [A_decode] [D_prefill] [C_decode] ← D 立即填入空位
Step 5: [A_decode] [D_decode] [C_decode]
Step 6: [A_done ✓] [D_decode] [C_decode]  ← A 结束了
Step 7: [E_prefill] [D_decode] [C_decode]  ← E 立即填入
...
```

每一步（iteration）结束后，scheduler 重新决策：
- 检查哪些请求已完成 → 释放资源
- 检查是否有等待中的新请求 → 有空间就加入
- 检查是否需要 preempt 某些请求

### 与 PagedAttention 的协同

Continuous batching + PagedAttention 是 vLLM 的两大支柱，它们互相增强：

- PagedAttention 让 KV Cache 可以灵活分配和回收 → 支持请求随时加入/退出
- Continuous batching 让请求在任意 iteration 进出 → 需要灵活的 KV Cache 管理

如果没有 PagedAttention，continuous batching 的效果大打折扣：新请求加入时可能没有连续的显存块可分配。

---

## Part 5: 中国模型的 KV Cache 特点

不同模型的 KV 结构差异会直接影响 KV Cache 大小和 PagedAttention 的效果：

### Qwen2.5 系列

| 模型 | n_layers | n_q_heads | n_kv_heads | head_dim | KV Cache/token (FP16) |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-3B | 36 | 16 | 2 | 128 | 36 KB |
| Qwen2.5-7B | 28 | 28 | 4 | 128 | 56 KB |
| Qwen2.5-14B | 48 | 40 | 8 | 128 | 192 KB |
| Qwen2.5-72B | 80 | 64 | 8 | 128 | 320 KB |

Qwen2.5-3B 使用 GQA 且 kv_heads=2，KV Cache 极其紧凑。

### DeepSeek 系列

DeepSeek-V2/V3 使用了独特的 **Multi-head Latent Attention (MLA)**：

- 传统 GQA：缓存 K 和 V 的完整表示
- MLA：缓存的是 KV 的**压缩低秩表示**（latent），推理时再解压
- 效果：KV Cache 大小可以压缩到 GQA 的 ~1/5 甚至更少
- 代价：decode 时需要额外的解压计算

这意味着 DeepSeek-V2/V3 在长上下文场景下的 KV Cache 显存优势非常大。不过 MLA 的实现对推理框架有特殊要求，vLLM 和 SGLang 都做了专门适配。

### GLM / ChatGLM 系列

- ChatGLM3-6B：使用 Multi-Query Attention (MQA)，kv_heads=2
- GLM-4-9B：使用 GQA，kv_heads=2
- KV Cache 较小，适合在显存有限的环境下部署

### MiniMax 系列

- MiniMax-Text-01 (456B)：MoE 架构
  - 使用了 Lightning Attention（线性注意力变体）
  - 线性注意力的 KV Cache 不随序列长度增长（constant size）
  - 这和标准 softmax attention 的 KV Cache 管理逻辑完全不同
  - vLLM 对此类模型的支持需要特殊适配

---

## Part 6: 从 Benchmark 反推调度行为

用 Day 3 的 benchmark 数据来验证对 PagedAttention 和 continuous batching 的理解：

### 实验 1：并发升高时观察 TTFT

```
并发数:      1      10     50     100    200
TTFT P50:   20ms   25ms   80ms   200ms  600ms   ← 为什么升高？
TTFT P99:   25ms   40ms   150ms  500ms  2000ms  ← 为什么 P99 飙升？
```

解释：
- 并发少时：新请求立即获得 GPU 和 KV Cache block，TTFT ≈ prefill 计算时间
- 并发高时：部分请求需要排队（等待 block 释放或等待 scheduler 安排）
- P99 飙升：少数请求恰好碰上系统高负载时刻，排队时间特别长

### 实验 2：长 prompt + 短 prompt 混合

```bash
# 同时发长短混合请求
# 50% 请求 input=128, 50% 请求 input=4096
```

观察：
- 长 prompt 的 prefill 占用大量计算资源
- 如果没有 chunked prefill：短请求的 TPOT 在长 prompt prefill 期间会显著升高
- 如果开启 chunked prefill：长 prompt 的 TTFT 略增，但短请求的 TPOT 更稳定

### 实验 3：KV Cache 压力

```bash
# 监控 KV Cache 使用率
watch -n 1 'curl -s http://localhost:8000/metrics | grep gpu_cache_usage'
```

观察：
- 并发升高时 `gpu_cache_usage_perc` 逐渐上升
- 接近 1.0 时新请求开始排队或触发 preemption
- 这就是 PagedAttention 在管理的资源边界

---

## 交付物

| 文件 | 描述 |
|---|---|
| `PagedAttention 一页纸解释.md` | 用自己的话解释：问题 → 方案 → 效果，包含 block table 示意图 |
| `Continuous batching 调度时序图.md` | 画 5-6 步的调度时序，展示请求进出 batch 的过程 |

## 自检问题

1. PagedAttention 解决的是什么层面的问题？（计算？显存？调度？）
2. Block size 设大或设小各有什么影响？
3. 什么时候会触发 preemption？有几种 preemption 策略？
4. Continuous batching 和 static batching 在哪一步有本质区别？
5. PagedAttention 的 Copy-on-Write 在什么场景下触发？
6. DeepSeek 的 MLA 对 KV Cache 管理有什么不同？
