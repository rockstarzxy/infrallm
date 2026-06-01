---
title: "Day 21: Week 3 系统设计演练 v1"
type: concept
tags: [day21, system-design, architecture, week3-review]
created: 2026-06-01
updated: 2026-06-01
---

# Day 21: Week 3 系统设计演练 v1

## 设计题目

设计一个在线 LLM 推理平台，满足以下需求：

```
模型:
  - 主力模型: Qwen2.5-72B-Instruct
  - 轻量模型: Qwen2.5-7B-Instruct (降级 / 低成本场景)
  - 可选: DeepSeek-V3 (高质量场景)

SLA:
  - P99 TTFT < 800ms
  - P99 TPOT < 80ms
  - 可用性 > 99.9%

规模:
  - 峰值 QPS: 500
  - 平均 input_len: 1000 tokens
  - 平均 output_len: 300 tokens
  - 80% 请求有共同 system prompt (~500 tokens)

硬件:
  - 2 个机房，每个机房可用 4 台 8×H100 服务器
  - 共 64 块 H100 GPU

约束:
  - 单 GPU 故障不影响服务
  - 模型更新不中断服务
  - 成本可控: GPU 利用率 > 60%
```

---

## 参考设计方案

### Step 1: 容量估算

```
Qwen2.5-72B FP16 权重: ~144 GB
  TP=4: 36 GB/GPU, KV Cache ≈ 44 GB/GPU

每请求 KV Cache (FP16):
  72B: KV/token = 320 KB
  (input + output) × 320 KB = 1300 × 320 KB = 416 MB/请求

单实例 (TP=4) KV Cache 总量: 44 GB × 4 = 176 GB
最大并发: 176 GB / 416 MB ≈ 423 个请求
  → 实际上受 max_num_seqs 和调度器限制，有效并发可能 ~200

单实例吞吐估算:
  decode TPOT: ~30ms (TP=4, batch ~100)
  throughput: ~100 × (1000/30) ≈ 3300 tokens/s
  request throughput: 3300 / 300 ≈ 11 req/s

满足 500 QPS:
  500 / 11 = ~46 个实例？→ 太多了

优化空间:
  - FP8 量化: 权重 72 GB, KV 减半 → 并发翻倍
  - prefix caching: 80% 请求共享 500 token 前缀 → TTFT 大幅降低
  - chunked prefill: TPOT 更稳定

FP8 + prefix caching 估算:
  权重 72 GB, TP=4: 18 GB/GPU, KV Cache ≈ 62 GB/GPU (FP8 KV)
  KV/token (FP8) = 160 KB
  实际 KV 需求 (考虑 prefix cache 命中 80%):
    未命中: 1300 × 160 KB = 208 MB
    命中:   800 × 160 KB = 128 MB (跳过前 500 token)
    加权: 0.2 × 208 + 0.8 × 128 = 144 MB/请求
  最大并发: 248 GB / 144 MB ≈ 1722 请求
  → 足够

吞吐 (FP8):
  TPOT ~15ms (FP8 加速), batch ~200
  throughput: 200 × (1000/15) ≈ 13333 tokens/s
  request throughput: 13333 / 300 ≈ 44 req/s

需要实例数: 500 / 44 ≈ 12 个 TP=4 实例
GPU 数量: 12 × 4 = 48 GPU

实际留 buffer (冗余 + 扩容):
  15 个实例 × 4 GPU = 60 GPU → 合理，在 64 GPU 预算内
  剩余 4 GPU: 部署 Qwen2.5-7B 作为降级
```

### Step 2: 架构设计

```
                    ┌─────────────────────────────────┐
   Internet ──────→ │         API Gateway              │
                    │  - 认证 / 限流 / 路由             │
                    │  - model routing                 │
                    │  - prefix-aware LB               │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼─────────────────┐
              ↓                ↓                 ↓
    ┌─────────────────┐ ┌──────────┐    ┌────────────────┐
    │  72B Pool        │ │ 7B Pool  │    │ DeepSeek Pool  │
    │  (主力)          │ │ (降级)   │    │ (高质量,可选)   │
    │                  │ │          │    │                │
    │  机房 A:         │ │ 2 实例   │    │ 按需启用        │
    │  8 实例 (TP=4)   │ │ TP=1     │    │                │
    │  机房 B:         │ │ GPU×2    │    │                │
    │  7 实例 (TP=4)   │ │          │    │                │
    │                  │ │          │    │                │
    │  共 60 GPU       │ │ 2 GPU    │    │ 需要更多 GPU   │
    └─────────────────┘ └──────────┘    └────────────────┘
```

### Step 3: 关键配置

```bash
# Qwen2.5-72B 实例配置
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 4 \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  --max-num-seqs 200 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192

# Qwen2.5-7B 降级实例
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --quantization fp8 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --max-num-seqs 128 \
  --enable-prefix-caching
```

### Step 4: 高可用设计

```
单 GPU 故障:
  → 影响 1 个 TP=4 实例（4 GPU 中有 1 个故障）
  → 该实例不可用，LB 自动摘除
  → 其他 14 个实例继续服务
  → 容量损失 ~7%，在 buffer 范围内

单节点故障:
  → 影响 2 个 TP=4 实例（8 GPU / 4 = 2 实例）
  → 容量损失 ~13%
  → 剩余实例可能接近满载
  → 触发告警，手动/自动启用备用节点

机房故障:
  → 影响一半容量
  → 另一个机房的实例承担全部流量
  → 可能需要降级（部分请求发到 7B 模型）
  → 流量 > 容量时启用限流
```

### Step 5: 监控和告警

```
SLI 定义:
  TTFT_P99 < 800ms    → 当前 baseline ~300ms，有 buffer
  TPOT_P99 < 80ms     → 当前 baseline ~30ms，有 buffer
  Error rate < 0.1%
  Availability > 99.9% → 月停机 < 43 分钟

告警:
  P1 (立即处理): TTFT_P99 > 600ms 持续 5 分钟
  P1: GPU 温度 > 85°C
  P1: 实例可用数 < 10
  P2: KV Cache > 90% 持续 10 分钟
  P2: Preemption rate > 0
  P3: GPU 利用率 < 40% 持续 30 分钟（可能浪费资源）
```

### Step 6: 成本分析

```
64 × H100 GPU

利用率目标 > 60%:
  15 个 72B 实例 × 4 GPU = 60 GPU → 93.75% 硬件分配率
  实际 GPU 利用率取决于 QPS:
    峰值 500 QPS: ~75% 利用率 ✓
    低峰 100 QPS: ~15% 利用率 ✗ → 可缩容
    
缩容策略:
  低峰期关闭 5 个 72B 实例 → 40 GPU → 省电 + 降温
  保留 10 个实例足够处理 ~200 QPS
```

---

## 设计总结

| 维度 | 决策 | 理由 |
|---|---|---|
| 量化 | FP8 权重 + FP8 KV | H100 原生支持，最优性价比 |
| 并行 | TP=4 per instance | 平衡延迟和吞吐 |
| 实例数 | 15 × 72B + 2 × 7B | 满足 500 QPS + 冗余 + 降级 |
| KV 优化 | prefix caching + chunked prefill | 80% 共享前缀 + 长短混合 |
| 路由 | prefix-aware LB | 最大化 cache 命中率 |
| 高可用 | 双机房 + 降级 + 限流 | 单点故障不中断服务 |

---

## 交付物

| 文件 | 描述 |
|---|---|
| `inference-platform-design-v1.md` | 完整的系统设计文档（架构图 + 容量估算 + 配置 + 高可用 + 监控） |

## 自检

这个设计还有哪些可以改进的地方？
- 是否考虑了 burst traffic？
- 如果用户 workload 突然变为长上下文（32K），系统会怎样？
- DeepSeek-V3 如果要加入这个平台，需要多少额外资源？
- 成本优化还有哪些空间？（如 spot instance、分时调度）
