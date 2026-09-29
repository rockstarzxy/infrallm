---
title: "优化专项 11: 推测解码没有收益或变慢"
type: concept
tags: [ai-infra, inference-optimization, speculative-decoding, eagle, mtp, ngram, acceptance-rate]
created: 2026-09-29
updated: 2026-09-29
---

# 优化专项 11: 推测解码没有收益或变慢

> 一句话：把"开了 spec decode 但 TPOT 没降 / 吞吐反而掉"拆成接受率、batch 规模、draft 开销、特性冲突四类根因，各给确认方法和修法。关联日课：[[day12-speculative-decoding]]、[[day09-scheduler]]、[[day13-flash-attention]]、[[day26-roofline]]。

## 0. 机制回顾与收益公式

一步 spec decode 在 V1 里的流程（[[day09-scheduler]] Part 5）：

1. drafter 提出 k 个 draft token（ngram 查表、EAGLE 小网络、MTP head、或独立 draft model）。
2. 调度器给该请求分配 `1 + k` 个 token 的 budget 和 `num_lookahead_tokens` 个 KV slot。
3. target 模型一次 forward 验证 k+1 个位置（这是一次"prefill 形状"的计算，不是 k 次 decode）。
4. rejection sampler 从左到右接受，遇到第一个被拒的位置停下，再从该位置的修正分布采 1 个 token。
5. 被拒位置之后的 KV slot 作废，`num_computed_tokens` 回退。

每步产出的 token 数是随机变量。记单位置接受率为 α（简化为各位置独立），则期望接受数：

```
E[tokens/step] = (1 - α^(k+1)) / (1 - α)        # 含最后一个修正 token
示例：α=0.8, k=3 → 2.95；α=0.5, k=3 → 1.88；α=0.3, k=3 → 1.42
```

每步时间：

```
T_step(spec) = T_draft(k) + T_verify(batch, k+1)
T_step(base) = T_decode(batch, 1)
收益 = E[tokens/step] × T_step(base) / T_step(spec)
```

收益大于 1 需要同时满足：α 够高、`T_verify(k+1)` 接近 `T_decode(1)`（也就是 target 仍处于 memory-bound，多算 k 个位置几乎免费）、`T_draft` 远小于 `T_decode`。三个条件各对应一类根因。

## 1. 症状与判定

先看三个指标（名称随版本，`curl /metrics | grep spec_decode` 核对）：

```
vllm:spec_decode_num_drafts_total             # 提了多少次 draft（≈ 步数）
vllm:spec_decode_num_draft_tokens_total       # 提了多少个 draft token（≈ 步数 × k）
vllm:spec_decode_num_accepted_tokens_total    # 接受了多少个
vllm:spec_decode_num_accepted_tokens_per_pos  # 按位置的接受计数（第 0 位、第 1 位……）
```

```
acceptance rate      = accepted_tokens / draft_tokens
mean accepted length = accepted_tokens / drafts        # 每步平均多产出的 token 数
per-position rate    = per_pos[i] / drafts             # 第 i 位的接受概率，应单调下降
```

| 症状 | 指标组合 | 根因方向 | 跳到 |
|---|---|---|---|
| TPOT 没降，接受率低于 0.5 | mean accepted length 小于 1.5 | draft 质量差：任务分布、temperature、draft 与 target 不匹配 | 2.1 |
| 低并发有收益，并发升到 32 以上反而吞吐下降 | 接受率正常，SM 利用率高 | target verify 已 compute-bound，多算 k 个位置不再免费 | 2.2 |
| 接受率高但 per-position 从第 2 位起骤降 | per_pos[2] / per_pos[0] 小于 0.4 | k 设太大，后几位白算 | 2.3 |
| 单请求 TPOT 也没降 | 接受率正常，nsys 里 draft 阶段占比高 | draft 本身太慢：draft_model 过大、TP 配置、EAGLE 多层 | 2.4 |
| 开了 spec decode 后某些请求变慢或报错 | 结构化输出、async scheduling、PP 相关日志 | 特性冲突走了回退路径 | 2.5 |
| 抢占变多、`kv_cache_usage_perc` 上升 | `num_preemptions_total` 增长 | lookahead slot 占 KV，并发能力下降 | 2.6 |
| ngram 在对话任务上几乎零收益 | 接受率低于 0.2 | ngram 只在输出重复输入片段时有效 | 2.1 / 3 |

## 2. 根因逐个排查

### 2.1 接受率低

**任务分布**：ngram 依赖输出中出现输入里的片段（代码补全、摘要、改写、RAG 抄原文高；开放式对话、创作低）。EAGLE/MTP 依赖 draft 学到 target 的分布，训练语料和业务不一致时接受率下降。

**采样温度**：温度高、top_p 大时 target 分布更平，同一个 draft token 被接受的概率下降。示例：同一 EAGLE head 在 temperature 0 下 α≈0.8，temperature 1.0 下可能掉到 0.5 到 0.6。

**draft 与 target 不匹配**：EAGLE/EAGLE-3 head 是针对特定 target 权重训练的，换了 target 的微调版本、量化版本（[[10-quantization-choice]]）、甚至 chat template，接受率都会变。draft_model 方式要求 tokenizer 完全一致。

**确认**：按业务分组打 acceptance rate（可以在 gateway 按任务类型分流到不同实例后分别看指标）；固定 prompt 集在 temperature 0 / 0.7 / 1.0 下各测一次。

**修**：按任务分流，只对高接受率的任务开；用为该 target 训练的 EAGLE-3 head（社区有 Llama、Qwen 系列的现成 head，看 HF）；MTP 模型（DeepSeek-V3/R1、Qwen3-Next）优先用自带 MTP；温度高的流量不开 spec decode。

### 2.2 大 batch 下 verify 不再免费

**机制**：并发 N 的 decode 步是 N 个 token 的 forward。开 k=3 后变成 4N 个 token。N 小时 4N 仍在 memory-bound 区间（步时间几乎不变），N 大到 4N 越过 roofline 拐点后，步时间随 token 数线性增长，spec decode 的收益消失，还多付了 draft 时间和被拒 token 的算力。

**确认**：concurrency 梯度 1 / 4 / 16 / 64 分别开关 spec decode 测吞吐，画两条曲线找交叉点。示例量级：8B 模型 H100 上交叉点常在并发 16 到 64。

**修**：高并发实例不开 spec decode，或降 k 到 1 到 2；把低并发、延迟敏感的流量单独一个实例开 spec decode。这是 [[15-cost-per-token]] 里"按 SLO 分池"的典型案例。

### 2.3 k 太大

per-position 接受率是递减的：第 i 位接受要求前 i 位都被接受。k 越大，尾部位置几乎白算，但 verify 的 token 数和 lookahead KV 都按 k 计。

**确认**：看 `per_pos` 曲线，找到接受率跌破 0.3 到 0.4 的位置。

**修**：k 设为该位置减 1。EAGLE 常用 3 到 5，ngram 3 到 5，MTP 通常 1 到 2（DeepSeek 只训了 1 个 MTP head，多层要看模型）。

### 2.4 draft 本身太慢

- **draft_model**：独立小模型每步一次完整 forward。7B target 配 0.5B draft，draft 步时间约为 target 的 1/8 到 1/5，k=4 就要付 4 次；draft 走 `draft_tensor_parallel_size=1` 时集中在一张卡，其他卡空等。
- **EAGLE**：单层轻量网络，开销小，但 EAGLE-3 要拿 target 多层的 hidden state，实现上有额外拷贝。
- **确认**：`--profiler-config` 或 nsys 看一步里 draft 段与 verify 段的时间比；draft 段超过 verify 的 30% 就不划算。
- **修**：draft_model 换成 EAGLE-3；draft 用 CUDA graph（默认应包含，看日志 capture 是否覆盖 drafter）。

### 2.5 特性冲突（版本相关，以启动日志和文档为准）

| 组合 | 状态（写作时） | 处理 |
|---|---|---|
| spec decode + async scheduling | 历史上互斥，新版本逐步支持；不支持时启动会报错或静默关掉其中一个 | 看日志，二选一测收益 |
| spec decode + 结构化输出 | 支持，bitmask 要对每个 draft 位置各算一份，CPU 开销增加；被 grammar 拒的 draft 直接不接受 | JSON 场景接受率会更低，单独测 |
| spec decode + prefix caching | 兼容 | 无 |
| spec decode + chunked prefill | 兼容，prefill 分片期间不提 draft | 无 |
| spec decode + PP | 部分版本不支持 | 用 TP 替代 |
| spec decode + 量化 target | 兼容，接受率要重测 | 见 2.1 |
| ngram + 多模态 | 兼容 | 无 |

**确认**：grep 启动日志里的 "speculative" 和 "not supported / disabled"。

### 2.6 lookahead 占用 KV

每个 running 请求要预留 k 个 KV slot 给 draft token，等价于每个请求多占 k 个 token 的 KV。示例：256 并发、k=5、block_size 16，最多多占约 256 个 block（每个请求跨 block 边界时）。数量不大，但在 KV 本来就满的实例上会把抢占触发得更早。

**修**：与 [[05-oom-preemption]] 同一套：降 `max_num_seqs` 或开 KV FP8。

## 3. 决策表

方法选型：

| 方法 | 需要额外权重 | 适合 | 不适合 | 配置 |
|---|---|---|---|---|
| ngram | 否 | 输出大量复述输入：代码编辑、摘要、RAG、改写 | 开放式生成 | `{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}` |
| EAGLE-3 | 是（对应 target 的 head） | 通用对话、低到中并发、有现成 head | 没有 head、微调后的 target | `{"method":"eagle3","model":"<head>","num_speculative_tokens":3}` |
| MTP | 模型自带 | DeepSeek-V3/R1、Qwen3-Next 等 | 无 MTP head 的模型 | `{"method":"mtp","num_speculative_tokens":1}` |
| draft_model | 是（同 tokenizer 小模型） | 有现成小模型且 EAGLE 不可用 | draft 与 target 比例小于 1:8 | `{"method":"draft_model","model":"<small>","num_speculative_tokens":3}` |
| Medusa | 是 | 历史方案 | 有 EAGLE 可选时 | 不推荐新部署 |

症状到动作：

| 症状 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| 接受率低于 0.5 | 换为该 target 训练的 EAGLE-3 / MTP | 按任务分流只对高接受率流量开 | 加大 k |
| 高并发吞吐下降 | 高并发实例关掉，低并发实例开 | k 降到 1 到 2 | 全局开着不分池 |
| per-position 骤降 | 降 k | — | 换方法 |
| draft 段占比高 | draft_model 换 EAGLE-3 | 缩小 draft_model | 提高 draft TP |
| 与 async scheduling 冲突 | 分别测两者收益，decode 主导高并发选 async，低并发选 spec | — | 强行同时开 |
| JSON 场景收益差 | 接受 ngram 或关掉 | — | 期待 EAGLE 在 grammar 下保持接受率 |

## 4. 验证实验

固定：模型、temperature（分别测 0 和 0.7）、input 512 / output 512、warmup、300 条、同一 GPU、prefix caching 关闭。

```bash
for MODE in off ngram eagle3; do
  case $MODE in
    off)    SPEC="";;
    ngram)  SPEC='--speculative-config {"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}';;
    eagle3) SPEC='--speculative-config {"method":"eagle3","model":"<head-repo>","num_speculative_tokens":3}';;
  esac
  vllm serve <target> --port 8000 --no-enable-prefix-caching $SPEC &
  sleep 120
  for C in 1 4 16 64; do
    vllm bench serve --model <target> --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 300 --max-concurrency $C --temperature 0 --save-result --result-filename spec_${MODE}_c${C}.json
    curl -s localhost:8000/metrics | grep spec_decode >> spec_${MODE}_c${C}.metrics
  done
  pkill -f "vllm serve"; sleep 10
done
```

记录每组：TPOT P50、output tokens/s、acceptance rate、mean accepted length、per-position 前 4 位。预期方向：

- 并发 1：EAGLE-3 的 TPOT 明显低于 off；ngram 在 ShareGPT 上收益小。
- 并发 64：off 的吞吐反超或持平。
- 用代码数据集（如 HumanEval prompt）重跑 ngram 组，接受率应显著高于 ShareGPT。

把交叉并发写进 [[15-cost-per-token]] 的分池策略。

## 5. 关联

- 诊断入口：[[01-diagnosis-playbook]]、[[03-tpot-itl-jitter]]
- 相关专项：[[04-throughput-low]]、[[05-oom-preemption]]、[[09-structured-output-tool-calls]]、[[10-quantization-choice]]、[[12-moe-serving]]、[[15-cost-per-token]]
- 日课：[[day12-speculative-decoding]]、[[day09-scheduler]]、[[day13-flash-attention]]、[[day26-roofline]]
