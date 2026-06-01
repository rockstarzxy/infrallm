---
title: "Day 28: 最终项目实验日"
type: concept
tags: [day28, experiment, benchmark, final-project]
created: 2026-06-01
updated: 2026-06-01
---

# Day 28: 最终项目实验日

## 任务说明

选一个模型，完成从部署到优化的完整实验闭环。至少包含 6 个实验组。

## 推荐模型选择

| 你的硬件 | 推荐模型 | 理由 |
|---|---|---|
| 1× 24GB GPU | Qwen2.5-7B-Instruct | 能体验大部分优化 |
| 1× 80GB GPU | Qwen2.5-14B-Instruct | 更大模型，空间够量化对比 |
| 4× 80GB GPU | Qwen2.5-72B-Instruct | 能体验 TP |
| 无 GPU | 回顾所有实验设计，写理论分析报告 | 用已有 benchmark 数据 |

## 实验矩阵

### 实验 1: Baseline

```bash
vllm serve <model> \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192

python benchmarks/benchmark_serving.py \
  --backend vllm --model <model> \
  --endpoint /v1/completions --dataset-name random \
  --random-input-len 512 --random-output-len 256 \
  --num-prompts 300 --request-rate 20
```

记录：TTFT P50/P99, TPOT P50/P99, throughput, GPU blocks

### 实验 2: 参数调优 A — max_num_seqs

测试 max_num_seqs = 16, 32, 64, 128, 256

### 实验 3: 参数调优 B — max_model_len

测试 max_model_len = 2048, 4096, 8192, 16384

### 实验 4: KV Cache 优化

开启 prefix caching + chunked prefill，用有共享 system prompt 的 workload 测试

### 实验 5: 量化对比

对比 FP16/BF16 baseline 和至少一种量化方案（AWQ/GPTQ/FP8）

### 实验 6: 分布式或多实例

如果有多卡：TP=2 vs TP=4 对比
如果只有 1 卡：用不同配置模拟 2 个场景的对比

## 结果记录模板

| 实验 | 配置 | TTFT P50 | TTFT P99 | TPOT P50 | TPOT P99 | Throughput | GPU Blocks | Preemptions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1. Baseline | default | | | | | | | |
| 2a. seqs=16 | | | | | | | | |
| 2b. seqs=64 | | | | | | | | |
| 2c. seqs=256 | | | | | | | | |
| 3a. len=2048 | | | | | | | | |
| 3b. len=16384 | | | | | | | | |
| 4. prefix+chunk | | | | | | | | |
| 5. quantized | | | | | | | | |
| 6. TP/multi | | | | | | | | |

## 分析要求

每个实验写 2-3 句分析：
- 这个变化为什么产生了这样的效果？
- 结果是否符合预期？如果不符合，可能的原因是什么？
- 这个优化在什么场景下最有价值？

## 交付物

| 文件 | 描述 |
|---|---|
| `final-project-benchmark-data.md` | 完整实验数据表 + 每组分析 |
