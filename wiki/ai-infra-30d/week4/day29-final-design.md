---
title: "Day 29: 最终系统设计文档"
type: concept
tags: [day29, system-design, final-project, architecture]
created: 2026-06-01
updated: 2026-06-01
---

# Day 29: 最终系统设计文档

## 文档结构

用你在过去 28 天学到的知识，独立完成一份生产级 LLM 推理平台设计文档。

### 1. 目标和 SLA

```
定义:
  - 支持的模型列表和版本
  - TTFT P99 要求
  - TPOT P99 要求
  - 可用性要求
  - 吞吐要求（峰值 QPS）
```

### 2. Workload 分析

```
分析:
  - input/output 长度分布
  - 是否有共享前缀
  - 峰值/低谷 QPS 比
  - 是否有多租户需求
  - 是否需要结构化输出
```

### 3. 硬件选型

```
考虑:
  - GPU 型号（A100 vs H100 vs L40S）
  - 内存配置（显存大小，NVLink 拓扑）
  - 网络配置（节点内 NVLink，节点间 IB/RoCE）
  - 存储（模型权重存储方案）
```

### 4. 模型部署配置

```
每个模型:
  - 并行策略: TP / PP / DP
  - 量化方案: FP16 / FP8 / AWQ / GPTQ
  - vLLM 核心参数
  - 实例数量
```

### 5. KV Cache 策略

```
  - Prefix caching 是否启用
  - Chunked prefill 是否启用
  - KV Cache dtype (FP16 / FP8)
  - 是否需要 disaggregated prefilling
```

### 6. Serving 架构

```
  - API Gateway 设计
  - Load balancing 策略
  - 模型路由逻辑
  - 限流和降级方案
```

### 7. 监控和告警

```
  - 关键 metrics 和 dashboard
  - 告警规则和处理流程
  - 日志和审计
```

### 8. 高可用和容灾

```
  - 单卡/单节点故障处理
  - 机房故障处理
  - 模型更新的 rolling update 策略
  - 回滚方案
```

### 9. 成本分析

```
  - GPU 硬件成本
  - GPU 利用率预估
  - 和按需 API (如 OpenAI/Claude) 的成本对比
  - 成本优化空间
```

### 10. 演进路线

```
  - 短期优化（1-3 月）
  - 中期演进（3-6 月）
  - 长期方向（6-12 月）
```

## 评估标准

一份好的系统设计文档应该做到：

1. **有数字**：容量估算要有具体计算过程，不是凭感觉
2. **有取舍**：每个决策要说明为什么选这个方案，不选其他方案
3. **有边界**：明确这个设计能支撑的规模上限和已知局限
4. **可执行**：看完文档后一个新人能按步骤部署起来
5. **可演进**：说明未来怎么扩展

## 交付物

| 文件 | 描述 |
|---|---|
| `inference-platform-design-final.md` | 完整的生产级推理平台设计文档（10 个章节） |
