---
title: "Day 20: 生产部署"
type: concept
tags: [day20, production, kubernetes, monitoring, autoscaling, health-check]
created: 2026-06-01
updated: 2026-06-01
---

# Day 20: 生产部署

## Part 1: 部署架构选型

### 方案 1: 裸机 + Systemd / Docker

```
适合: 小规模、单机、快速验证
组件: Docker + docker-compose + Nginx

docker-compose.yml:
  services:
    vllm:
      image: vllm/vllm-openai:latest
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: all
                capabilities: [gpu]
      command: --model Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.9
      ports: ["8000:8000"]
    
    nginx:
      image: nginx
      ports: ["80:80"]
      # 配置反向代理到 vllm
```

### 方案 2: Kubernetes + GPU Operator

```
适合: 中大规模、需要弹性、多模型
组件: K8s + NVIDIA GPU Operator + KServe/Ray Serve

关键 K8s 资源:
  - GPU Device Plugin: 让 K8s 识别和调度 GPU
  - GPU Operator: 自动安装 GPU 驱动和运行时
  - Node labels: gpu-type=a100, gpu-count=8 等
  
Pod spec:
  resources:
    limits:
      nvidia.com/gpu: 4   # 请求 4 个 GPU
```

### 方案 3: Ray Serve

```
适合: 多模型、需要动态扩缩容、和 Ray 生态集成
组件: Ray Cluster + Ray Serve

优势:
  - 可与 vLLM 集成；vLLM 多节点常用 Ray，但单节点默认可用 multiprocessing，多节点也有非 Ray 启动方式
  - 内置 autoscaling
  - 多模型可以共享 GPU 集群
```

---

## Part 2: 模型权重管理

### 权重存储方案

```
方案 1: Hugging Face Hub
  - 直接从 HF 下载
  - 首次启动慢，后续有本地缓存
  - 适合开发和小规模部署

方案 2: 对象存储 (S3 / GCS / OSS / MinIO)
  - 权重上传到对象存储
  - Pod 启动时拉取到本地 SSD
  - 适合生产环境

方案 3: 共享文件系统 (NFS / Lustre / GPFS)
  - 所有节点挂载同一个文件系统
  - 权重只存一份
  - 注意: 多节点同时读取时的 IO 带宽

方案 4: 预烘焙镜像
  - 将模型权重打包进 Docker 镜像
  - 启动最快（无需下载）
  - 缺点: 镜像巨大（几十到几百 GB）
```

### 权重加载优化

```
问题: 大模型加载时间长
  Qwen2.5-72B FP16: ~144 GB → 从 SSD 读取约 30s-60s
  DeepSeek-V3: ~1.3 TB → 分钟级

优化:
  1. Tensor Parallel 加载: 每个 GPU 只读自己那部分权重
  2. 异步加载: 边加载边初始化
  3. 预热: 加载完后发几个 warmup 请求（编译 CUDA Graph 等）
  4. 保持 warm pool: 总有预热好的实例待命
```

---

## Part 3: 健康检查和探针

### K8s 探针配置

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 120    # 模型加载时间
  periodSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 120
  periodSeconds: 5
  failureThreshold: 2

startupProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 30        # 允许最长 5 分钟启动
```

### 自定义健康检查

```python
# 除了 /health，还应该检查：

# 1. GPU 是否正常
nvidia-smi  # 返回码非 0 → GPU 故障

# 2. 模型是否能正常推理
curl http://localhost:8000/v1/completions \
  -d '{"model": "...", "prompt": "test", "max_tokens": 1}'
# 超时或错误 → 模型卡住

# 3. KV Cache 是否健康
curl http://localhost:8000/metrics | grep gpu_cache_usage
# > 0.99 持续 → 需要告警
```

---

## Part 4: 监控和告警

### Prometheus + Grafana 监控

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm-instance-1:8000', 'vllm-instance-2:8000']
    metrics_path: /metrics
    scrape_interval: 5s
```

### 关键 Dashboard 指标

```
═══ 服务质量 (SLI/SLO) ═══
  TTFT P50 / P95 / P99
  TPOT P50 / P95 / P99
  E2E Latency P99
  Error Rate (4xx, 5xx)

═══ 系统负载 ═══
  QPS (requests/s)
  Output Throughput (tokens/s)
  num_requests_running
  num_requests_waiting ← 排队数 > 0 持续 → 需要扩容
  
═══ 资源利用 ═══
  GPU Utilization (per GPU)
  GPU Memory Used
  kv_cache_usage_perc / gpu_cache_usage_perc ← KV Cache 使用率（名称随版本变化）
  preemption 相关指标 ← preemption 数量（名称随版本变化）
  
═══ 硬件健康 ═══
  GPU Temperature
  GPU Power Draw
  ECC Error Count
  Clock Frequency (是否降频)
```

### 告警规则示例

```yaml
groups:
  - name: vllm_alerts
    rules:
      # TTFT P99 超过 SLA
      - alert: HighTTFT
        expr: histogram_quantile(0.99, vllm_e2e_request_latency_seconds_bucket) > 2
        for: 5m
        labels:
          severity: warning

      # 请求排队
      - alert: RequestsQueued
        expr: vllm_num_requests_waiting > 10
        for: 2m
        labels:
          severity: warning

      # KV Cache 快满
      - alert: KVCacheFull
        expr: vllm:kv_cache_usage_perc > 0.95 or vllm:gpu_cache_usage_perc > 0.95
        for: 5m
        labels:
          severity: critical

      # Preemption 频繁
      - alert: FrequentPreemptions
        expr: rate(vllm:num_preemptions_total[5m]) > 1
        for: 5m
        labels:
          severity: warning

      # GPU 温度过高
      - alert: GPUOverheat
        expr: nvidia_gpu_temperature_celsius > 85
        for: 2m
        labels:
          severity: critical
```

---

## Part 5: Rolling Update 和灰度发布

### 模型更新策略

```
问题: 更新模型版本（如 Qwen2.5-7B → Qwen2.5-7B-v2）不能中断服务

Rolling Update:
  1. 启动新版本实例（新 Pod）
  2. 等待新实例 ready（通过 readinessProbe）
  3. 将流量逐步切到新实例
  4. 确认新实例正常后下线旧实例

灰度:
  1. 新实例接 5% 流量
  2. 观察 TTFT/TPOT/质量指标
  3. 确认无异常 → 逐步放量到 100%
  4. 有异常 → 立即回滚到旧版本

K8s 配置:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1       # 允许多 1 个 Pod
      maxUnavailable: 0 # 不允许少 Pod
```

### A/B Testing

```
同时运行两个模型版本:
  版本 A: Qwen2.5-72B (原版)
  版本 B: Qwen2.5-72B (新 checkpoint / 新量化方案)

Gateway 按用户 ID hash 分流:
  50% 用户 → A
  50% 用户 → B

比较:
  - 质量指标 (用户满意度、任务成功率)
  - 性能指标 (TTFT, TPOT)
  - 资源消耗 (GPU 利用率、显存)
```

---

## Part 6: 部署 Checklist

### 上线前

```
□ 模型权重版本确认（和训练侧确认 checkpoint）
□ Tokenizer 版本确认（训练推理一致性）
□ vLLM 版本锁定（写死版本号，不用 latest）
□ GPU 驱动和 CUDA 版本确认
□ 启动参数确认（gpu_memory_utilization, max_model_len 等）
□ 量化方案确认（如果用量化，确认质量回归测试通过）
□ Benchmark 通过（在目标硬件上跑过 benchmark，指标达标）
□ 健康检查配置
□ 监控和告警配置
□ 限流和降级策略
□ 回滚方案（如何快速回到上一个版本）
□ 权重安全（模型权重不应暴露在公网）
```

### 上线后

```
□ 观察 10 分钟无告警
□ 抽样检查输出质量
□ 确认指标在基线范围内
□ 确认没有 preemption / OOM
□ 确认 autoscaling 正常工作（如果启用）
```

---

## 交付物

| 文件 | 描述 |
|---|---|
| `production-deployment-checklist.md` | 部署方案选型 + 监控配置 + 上线 checklist |

## 自检问题

1. K8s 部署 LLM serving 和普通 web 服务有哪些额外考虑？
2. 模型权重更新时如何保证服务不中断？
3. 哪些 vLLM metrics 应该设置告警？阈值怎么定？
4. LLM serving 的 autoscaling 应该基于什么指标？为什么不能只看 QPS？
5. 如果 GPU 温度持续升高，可能的原因和处理方式？
