---
title: "Day 18: Data Parallelism、多实例和路由"
type: concept
tags: [day18, data-parallel, multi-instance, load-balancer, routing]
created: 2026-06-01
updated: 2026-06-01
---

# Day 18: Data Parallelism、多实例和路由

## Part 1: Data Parallelism 基本概念

DP 是最简单的并行策略：**每个 GPU（或 GPU 组）持有完整的模型副本，独立处理不同的请求。**

```
TP/PP: 一个请求的计算分散到多个 GPU → 降低单请求延迟
DP:    不同请求分配到不同 GPU → 提升总吞吐量

DP 没有模型层面的通信（每个副本独立工作）。
唯一需要的是请求分发（routing / load balancing）。
```

### DP 的两种形式

**形式 1: 框架内 DP**
```
vLLM 的 data_parallel_size 参数（如果支持）
→ vLLM 内部管理多个模型副本
→ scheduler 统一调度所有副本上的请求
```

**形式 2: 多实例 + 外部 LB（更常见）**
```
启动多个独立的 vLLM 进程
每个进程是一个完整的 serving 实例
前面放一个 load balancer 分发请求

这是生产环境中最主流的做法
```

---

## Part 2: 多实例部署架构

### 基本架构

```
                ┌──────────────────┐
  Clients ────→ │  Load Balancer   │
                │  (Nginx / Envoy  │
                │   / 自研 Gateway)│
                └────┬────┬────┬───┘
                     │    │    │
              ┌──────┘    │    └──────┐
              ↓           ↓           ↓
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ vLLM #1  │ │ vLLM #2  │ │ vLLM #3  │
        │ GPU 0-1  │ │ GPU 2-3  │ │ GPU 4-5  │
        │ TP=2     │ │ TP=2     │ │ TP=2     │
        └──────────┘ └──────────┘ └──────────┘
```

### 实例规划

```
8×A100 80GB, 部署 Qwen2.5-72B (FP16):

方案 A: 1 实例, TP=8
  延迟最低（单请求利用所有 GPU）
  吞吐有限（只有一个调度器）

方案 B: 2 实例, 每实例 TP=4
  延迟略高（每个请求只用 4 GPU）
  吞吐约 2x（两个独立调度器）
  
方案 C: 4 实例, 每实例 TP=2
  延迟更高
  吞吐约 3-4x

选择取决于 SLA:
  延迟优先 → 方案 A
  吞吐优先 → 方案 B 或 C
  通常方案 B 是最佳平衡点
```

---

## Part 3: Load Balancing 策略

### 策略 1: Round Robin（轮询）

```
Request 1 → Instance 1
Request 2 → Instance 2
Request 3 → Instance 3
Request 4 → Instance 1
...
```

优点：简单
缺点：不考虑请求差异。一个长 prompt 和一个短 prompt 被等价对待 → 实例负载不均。

### 策略 2: Least Connections（最少连接）

```
哪个实例当前处理的请求最少 → 新请求发给它
```

优点：比 round robin 更均衡
缺点：不考虑请求处理时间的差异

### 策略 3: 基于队列深度（Queue-Aware）

```
查询每个 vLLM 实例的 metrics:
  waiting_requests = vllm:num_requests_waiting
  running_requests = vllm:num_requests_running
  kv_cache_usage  = vllm:gpu_cache_usage_perc

→ 将新请求发给 waiting 最少的实例
```

优点：考虑了实际负载
实现：需要 gateway 定期拉取 vLLM metrics

### 策略 4: Prefix-Aware Routing（前缀感知路由）

```
如果开启了 prefix caching:
  相同 system prompt 的请求应该路由到同一个实例
  → 最大化 prefix cache 命中率

实现:
  对 system prompt 做 hash → hash mod num_instances → 路由到固定实例
  或: consistent hashing（一致性哈希）
```

这是性能最优的策略，但实现复杂度也最高。

### 策略对比

| 策略 | 实现复杂度 | 负载均衡度 | Prefix Cache 友好 |
|---|---|---|---|
| Round Robin | 低 | 一般 | 不友好 |
| Least Connections | 低 | 较好 | 不友好 |
| Queue-Aware | 中 | 好 | 不友好 |
| Prefix-Aware | 高 | 好 | 非常友好 |

---

## Part 4: 多模型路由

生产环境通常需要同时服务多个模型：

```
┌──────────────────────┐
│     API Gateway       │
│                       │
│  /v1/chat/completions │
│  model: qwen-72b ─────┼──→ Pool A (Qwen-72B, 2 实例)
│  model: qwen-7b  ─────┼──→ Pool B (Qwen-7B, 4 实例)
│  model: deepseek ─────┼──→ Pool C (DeepSeek, 1 实例)
│  model: glm-9b   ─────┼──→ Pool D (GLM-4-9B, 2 实例)
└──────────────────────┘
```

### 路由维度

1. **按模型名**：最基础，不同模型到不同实例池
2. **按租户**：VIP 用户到专属实例，普通用户到共享池
3. **按优先级**：高优先级请求到预留实例，低优先级可排队
4. **按任务类型**：短对话到低延迟实例，批量任务到高吞吐实例

### 实现示例（Nginx 简单路由）

```nginx
upstream qwen72b {
    server 10.0.0.1:8000;
    server 10.0.0.2:8000;
}

upstream qwen7b {
    server 10.0.0.3:8000;
    server 10.0.0.4:8000;
    server 10.0.0.5:8000;
}

server {
    listen 80;

    location /v1/ {
        # 根据请求体中的 model 字段路由
        # 需要 Lua 或自研 gateway
        proxy_pass http://qwen72b;
    }
}
```

更常见的做法是用自研 gateway 或 Envoy + WASM filter 来解析请求体做路由。

---

## Part 5: 限流和降级

### 限流 (Rate Limiting)

```
全局限流: 总 QPS < 系统容量 × 0.8（留 20% headroom）
租户限流: 每个租户的 QPS/TPM（tokens per minute）上限
模型限流: 大模型的并发上限比小模型低

限流方式:
  1. 直接拒绝 (429 Too Many Requests)
  2. 排队等待 (202 Accepted + async polling)
  3. 降级到小模型 (72B → 7B)
```

### 降级策略

```
级别 1 (正常):
  所有请求正常处理

级别 2 (高负载):
  - 限制 max_tokens
  - 关闭 speculative decoding
  - 新请求排队

级别 3 (过载):
  - 低优先级请求拒绝
  - 部分请求降级到小模型
  - 减少并发 (降低 max_num_seqs)

级别 4 (故障):
  - 切换到备用实例
  - 返回缓存的默认回复
```

---

## Part 6: Autoscaling

### 基于指标的扩缩容

```
Scale Up 触发条件:
  - num_requests_waiting > 阈值 (持续 2 分钟)
  - gpu_cache_usage_perc > 0.9 (持续 5 分钟)
  - QPS > capacity × 0.8

Scale Down 触发条件:
  - GPU utilization < 20% (持续 10 分钟)
  - num_requests_running < max_num_seqs × 0.2 (持续 10 分钟)
```

### LLM Serving Autoscaling 的特殊挑战

```
1. 冷启动慢:
   模型加载需要 30s-5min（取决于模型大小和权重获取方式）
   → 需要预热实例或维护 warm pool

2. GPU 资源稀缺:
   不像 CPU 可以随时弹性伸缩
   → 通常预留 GPU 而不是按需分配

3. 负载突变:
   LLM 的请求长度差异大（一个长请求 = 100 个短请求的负载）
   → 基于 QPS 的 autoscaling 不准确
   → 应该基于 tokens/s 或队列深度
```

---

## Part 7: 实操

### 本机模拟多实例

```bash
# 实例 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 &

# 实例 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 &

# 简单的 round-robin 压测
for i in $(seq 1 100); do
  port=$((8000 + i % 2))
  curl -s http://localhost:$port/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "Hello", "max_tokens": 32}' &
done
wait
```

---

## 交付物

| 文件 | 描述 |
|---|---|
| `multi-instance-serving-design.md` | 多实例架构图 + LB 策略选择 + 限流/降级方案 |

## 自检问题

1. DP 和 TP 的本质区别是什么？
2. 什么时候应该增大 TP 而不是开更多实例？
3. Prefix-Aware Routing 为什么能提升性能？
4. LLM serving 的 autoscaling 和普通 web 服务有什么不同？
5. 如果一个实例 OOM 了，gateway 应该怎么处理？
