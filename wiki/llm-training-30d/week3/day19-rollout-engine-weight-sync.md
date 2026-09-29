---
title: "Day 19：Rollout 引擎集成与权重同步"
type: concept
tags: [llm-training, rollout, weight-sync, vllm, off-policy]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 19：Rollout 引擎集成与权重同步

> 上一课 [[llm-training-30d/week3/day18-rl-training-frameworks]] · 下一课 [[llm-training-30d/week3/day20-async-rl-stability]]。相关：推理引擎内部见 [[ai-infra-30d/week2/day08-vllm-architecture]] 与 [[ai-infra-30d/week2/day13-flash-attention]]；prefix cache 与 sleep 见 [[ai-infra-30d/week2/day10-kv-cache]]。

## 学习目标

1. 说清 RL 训练循环里 rollout 引擎的位置：谁调用它、什么时候调用、它拿到的权重是哪一步的。
2. 能画出一次权重同步的完整路径：训练侧分片布局 → 重分片 → 传输 → 推理侧 TP 布局 → 重建 CUDA graph，并估算每一步的时间。
3. 能解释"训练与推理数值不一致会让 on-policy 变成隐式 off-policy"，并写出 TIS/MIS 修正公式。
4. 能用一组指标定位 rollout 阶段的瓶颈是长尾样本、权重同步、显存切换还是引擎本身。
5. 动手：在 verl 或 TRL 里切换 colocated / 非 colocated，测同步时间与 rollout 吞吐。

## 工业现状

2025 年以来所有公开的大规模 RL 系统都把采样交给独立推理引擎，而不是用训练框架的 `generate()`：verl 用 vLLM 或 SGLang（T42）、OpenRLHF 用 vLLM（T43）、slime 用 SGLang 配 Megatron（B07）、AReaL 用自研 + SGLang（T44）、NeMo-RL 用 vLLM（B08）。原因是同一件事：RL 一步里 rollout 常占 60% 到 80% 的墙钟时间，推理引擎的 continuous batching、PagedAttention 和 CUDA graph 能把这段压缩数倍，而 HF `generate()` 做不到。

两种部署形态并存：

| 形态 | 做法 | 谁在用 | 取舍 |
|---|---|---|---|
| colocated（共置） | 训练进程与推理引擎轮流占同一批 GPU，靠 sleep/wake 或显存 offload 切换 | verl 默认、OpenRLHF 的 colocate 模式、TRL vLLM colocate | GPU 利用率高、同步走本地；但每步要两次显存切换，rollout 与训练串行 |
| disaggregated（分离） | 一部分 GPU 专跑推理引擎，一部分专跑训练，权重跨机传输 | AReaL、slime 的部分配置、Kimi 的异步系统（T18） | rollout 与训练可重叠、天然支持异步；但需要更多卡，权重要走网络 |

权重同步的工业做法也趋同：不再落盘再加载（分钟级），而是训练侧把参数按推理引擎的 TP 布局重分片后，通过 NCCL broadcast 或 CUDA IPC 直接写进推理进程的权重张量（秒级）。DeepSeek-R1（T17）、Kimi k1.5（T18）、DAPO（T37）等报告都默认这条路径。

第三个共识来自 B01 和 B03：训练框架与推理引擎的 kernel 不同，同一权重算出的 logprob 有系统性差异，PPO/GRPO 的重要性比会被污染。2025 年下半年起主流框架都提供了 truncated importance sampling 一类的修正开关。

## 核心原理

### 一次 RL step 的数据流

```
             ┌──────────────────────────────────────────────────┐
             │ 1. 权重同步  θ_train(t) ──resharding──> θ_infer(t) │
             └──────────────────────────────────────────────────┘
                              │
             ┌────────────────▼─────────────────┐
             │ 2. rollout：推理引擎对 batch 里每个 │  rollout = 一个 prompt 采样 n 条完整 response
             │    prompt 采样 n 条 response，返回   │  这里的"完整"指到 EOS 或 max_tokens
             │    token ids（可选 logprob）        │
             └────────────────┬─────────────────┘
                              │
             ┌────────────────▼─────────────────┐
             │ 3. reward：verifier / RM 打分       │
             └────────────────┬─────────────────┘
                              │
             ┌────────────────▼─────────────────┐
             │ 4. 训练侧重算 logprob：π_old(t) 与   │  用训练框架的 forward，不是引擎返回的
             │    π_ref，算 advantage、loss、更新    │
             └────────────────┬─────────────────┘
                              │  θ_train(t+1)
                              ▼ 回到 1
```

必须理解的三点：

- **rollout 用的权重永远是上一次同步的**。同步在 step 开头做，rollout 中途不更新。如果一个 step 内有多个 mini-batch 更新（PPO epochs > 1），后面的 mini-batch 相对 rollout 已经是 off-policy，这是 PPO clip 存在的原因。
- **训练侧重算 logprob 而不是信任引擎返回的**。引擎的 logprob 来自不同 kernel、不同精度路径（FP8 KV、CUDA graph 下的 padding、不同 reduction 顺序），和训练侧 forward 的差异可达 1e-2 量级（示例）。重算的是 `π_old`，引擎的 logprob 只在做 TIS 修正时使用。
- **rollout 的长尾决定 step 时间**。group 里最长的那条 response 结束前，整个 batch 不能进入训练；一条 8k token 的 response 会让其他 255 条等它。

### 权重同步路径

训练侧布局（FSDP2 每参数分片、或 Megatron TP×PP×DP）和推理侧布局（vLLM TP，通常无 PP）不同，同步不是 memcpy：

```
FSDP2 rank i 持有每个参数的 1/N 分片
  → all_gather 得到完整参数（按参数逐个或分桶，避免峰值显存）
  → 按推理 TP 布局切：列并行权重按输出维切、行并行按输入维切、
    QKV 合并权重要按 head 重新排列（vLLM 的 QKVParallelLinear 布局）
  → 传给推理 rank j 的 model_executor.driver_worker.model_runner.model.load_weights()
```

传输通道按形态选：

| 通道 | 适用 | 典型耗时（示例，7B BF16 = 14 GB） |
|---|---|---|
| CUDA IPC 句柄（同机同卡） | colocated，训练与推理在同一 GPU 上时零拷贝 | 亚秒级 |
| NCCL broadcast（同机或跨机，训练 rank 0 → 所有推理 rank） | 最通用 | 同机 NVLink 数秒；跨机 IB 十几秒到几十秒 |
| 落盘 + `load_weights` | 调试或框架不支持时 | 分钟级，生产不用 |

verl 的实现要点（以当前版本文档 B06 为准）：`ShardingManager` 在进入 rollout 前做 FSDP → vLLM 的重分片，退出时释放；参数分桶 all_gather 控制峰值显存；vLLM 侧调用 `load_weights` 而不是重建引擎。slime 走 Megatron → SGLang 的同类路径。

### colocated 下的显存切换

同一张卡上训练进程持有参数 + 梯度 + 优化器状态 + activation，推理引擎持有权重 + KV cache + CUDA graph 池。两者加起来放不下，所以轮流：

```
进入 rollout：训练侧 offload 优化器状态/梯度到 CPU（或释放 activation），
             推理引擎 wake_up()：重新分配 KV cache 与权重 buffer
退出 rollout：推理引擎 sleep(level)：释放 KV（level 1）或连权重一起释放（level 2），
             训练侧把优化器状态搬回 GPU
```

vLLM 的 `sleep_mode` 与 `wake_up` 就是为这个场景加的（推理课 Day 8 提到它走 `collective_rpc`）。代价：level 2 唤醒后权重要重新写入；KV 释放意味着 prefix cache 清空，下一步的 rollout 没有跨 step 的缓存可用；CUDA graph 池在某些版本下也要重新 capture，几十秒的启动成本每步都会出现。这是 colocated 形态在小模型上反而不划算的原因之一。

### 训练与推理数值不一致：隐式 off-policy

PPO/GRPO 的目标是

```
L = E[ min( r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t ) ],   r_t = π_θ(a_t) / π_old(a_t)
```

理论上 rollout 由 `π_old` 采样，训练侧算 `π_θ` 时第一个 mini-batch 有 `r_t = 1`。实际上 rollout 由推理引擎的 `π_infer` 采样，训练侧重算的 `π_old` 是训练 kernel 下的值，`π_infer ≠ π_old`。这就是 B01 指出的隐式 off-policy：采样分布与你以为的 `π_old` 有偏差，偏差在长序列上累积。

修正（B01 的 TIS，truncated importance sampling）：

```
w_t = min( π_old(a_t) / π_infer(a_t), C )        # C 常取 2 到 5（示例）
L = E[ w_t · min( r_t · A_t, clip(r_t) · A_t ) ]
```

需要引擎返回采样时的 logprob（vLLM 的 `logprobs` 参数），代价是引擎多做一次 gather。MIS（masked IS）则把偏差过大的 token 直接 mask。verl、slime、OpenRLHF 在 2025 年下半年都加了对应开关，名字随版本变化，查文档里的 "rollout importance sampling" 或 "tis"。

根治方向是 B03 的 batch-invariant kernel：让推理引擎在任意 batch 大小下给出逐位一致的结果，训练侧用同一套 kernel 做 forward。目前只有部分算子可用，主流仍是 TIS 修正。

### 采样参数

| 参数 | 训练时建议 | 原因 |
|---|---|---|
| temperature | 1.0（GRPO/DAPO 默认） | 低温会缩小 group 内多样性，advantage 归零；高温增加乱码 |
| top_p / top_k | 关闭（1.0 / -1） | 截断会让 `π_infer` 与 `π_old` 不一致的部分变成硬 0，IS 修正失效 |
| max_tokens | 按任务定，配合 overlong shaping（T37） | 截断样本要标记，reward 单独处理 |
| n（group size） | 8 到 16 常见（示例） | 与 GRPO advantage 方差直接相关 |
| logprobs | 开（做 TIS 时） | 否则无法修正 |

### SGLang 侧与分离形态的权重更新（知道即可）

SGLang 提供 `update_weights_from_distributed`（NCCL 组内广播）和 `update_weights_from_tensor`（同机张量直传）两类接口，slime 与 verl 的 SGLang 后端都走它们；名字与参数以当前 SGLang 文档为准。分离形态下推荐流程：

```
训练 rank 0 发起 → 建立训练侧 + 全部推理 rank 的 NCCL 组（一次性）
每步：按参数分桶 broadcast → 推理侧逐桶 load → flush cache → 通知就绪
```

分桶的意义：7B 以上模型整体 all_gather 会让训练 rank 0 出现一份完整权重的峰值显存；按 1 到 2 GB 一桶流水化传输可以把峰值压到桶大小（示例）。异步系统（Day 20）还会给每份权重打版本号，rollout 样本带上它生成时的版本，训练侧据此判断 staleness。

## 实现步骤

### 1. verl：colocated 与非 colocated 的切换

```yaml
# verl 配置片段（字段名以当前版本为准）
actor_rollout_ref:
  rollout:
    name: vllm                 # 或 sglang
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.6   # colocated 时给引擎的显存比例
    free_cache_engine: true       # 退出 rollout 时释放 KV
    enforce_eager: false          # 生产关掉，否则 CUDA graph 收益没了
    temperature: 1.0
    top_p: 1.0
    n: 8
    calculate_log_probs: true     # 返回引擎 logprob，供 TIS
  hybrid_engine: true             # colocated；false 时训练与 rollout 分池
```

非 colocated 需要在 Ray 资源池里为 rollout 单独指定 GPU 数（`resource_pool` 配置），并接受权重跨进程 broadcast。

### 2. 读一次同步的代码路径

```bash
VERL=$(python -c "import verl, os; print(os.path.dirname(verl.__file__))")
grep -rn "class .*ShardingManager" $VERL/workers/sharding_manager/
grep -rn "load_weights\|update_weights" $VERL/workers/rollout/vllm_rollout/ | head
grep -rn "sleep\|wake_up" $VERL/workers/rollout/vllm_rollout/ | head
```

把 `__enter__`（进入 rollout）和 `__exit__`（退出）里做的事按顺序列成表：all_gather、重排、load_weights、wake_up、释放。这张表就是你的同步路径图。

### 3. TRL 的两种 vLLM 接法

```bash
# 方式 A：独立 vLLM server，训练进程通过 HTTP 取样本并推权重
trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --tensor-parallel-size 1
# 训练侧 GRPOConfig(use_vllm=True, vllm_mode="server", vllm_server_host=...)

# 方式 B：colocate，训练进程内嵌 vLLM
# GRPOConfig(use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3)
```

方式 A 的权重同步走 NCCL（TRL 起一个通信组把训练 rank 的权重广播到 server），方式 B 走同进程内存。参数名以 TRL 文档（B05）为准。

### 4. 手写最小同步（理解用）

```python
# 单卡教学版：训练模型与 vLLM LLM 对象在同一进程
from vllm import LLM
llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.3, enable_sleep_mode=True)
model_runner_model = llm.llm_engine.model_executor.driver_worker.model_runner.model  # 路径随版本变化

def sync(train_model):
    llm.wake_up()
    weights = ((name, p.detach()) for name, p in train_model.named_parameters())
    model_runner_model.load_weights(weights)   # vLLM 各模型类实现了名字映射与 QKV 合并
    llm.reset_prefix_cache()                   # 旧权重的 KV 不能再命中

def rollout(prompts, n):
    outs = llm.generate(prompts, SamplingParams(n=n, temperature=1.0, top_p=1.0,
                                                max_tokens=1024, logprobs=0))
    llm.sleep(level=1)
    return outs
```

生产框架做的就是把这段拆到多进程、多卡、分片布局上。

### 5. 指标埋点

每个 step 记录：`t_sync`、`t_rollout`、`t_reward`、`t_train`、rollout 里 `max_len / mean_len`（长尾比）、引擎 `num_requests_running` 随时间的曲线（末尾拖尾多久）、`mean |logπ_infer − logπ_old|`（不一致程度）。

## 实验

资源分层：单卡 24-48 GB 可完成实验 1、3；实验 2 需 2 卡以上。

### 实验 1：不一致有多大

0.5B 或 1.5B 模型，同一 prompt 采 64 条 response，引擎返回 logprob，训练侧（HF 模型 BF16）重算。统计每 token 的 `|Δ logp|` 分布和按位置的累积，分别在 `enforce_eager=True/False`、`kv_cache_dtype=auto/fp8` 下重复。预期：FP8 KV 与 CUDA graph 都会增大差异；差异随序列位置累积而不是均匀分布。

### 实验 2：同步时间随模型与拓扑

verl 或 TRL server 模式，1.5B 与 7B，TP=1 与 TP=2，记录 `t_sync`。再把训练和推理放到不同机器（若有）。预期：同机 NVLink 数秒内，跨机受 IB 带宽约束线性增长；TP 越大重排开销越明显。

### 实验 3：colocated 的切换成本

同一配置下对比 `free_cache_engine=true/false`、sleep level 1/2，记录每步 `t_sync + 唤醒` 和峰值显存。预期：level 2 每步多出权重重写时间；关掉释放会 OOM 或迫使 `gpu_memory_utilization` 降到无法放下 batch。

### 实验 4：TIS 开关的 A/B

GSM8K 上 GRPO 200 步，开关 TIS（或等价选项）各跑 2 个 seed。记录 reward 曲线、KL、评测分数。预期：小模型短序列差异不明显；长 CoT（max_tokens 4k 以上）时不修正的组更容易在后期崩。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 同步后 rollout 输出乱码 | 权重名映射错、QKV 交织顺序不对、dtype 不匹配 | 同步后用固定 prompt 贪心解码对比训练侧 | 用框架自带的 `load_weights`，不要手写 state_dict 拷贝 |
| 每步开头卡几十秒 | 每步重新 capture CUDA graph 或重新 profile KV | 看引擎日志是否每步打印 capture | 用 sleep level 1；升级到支持保留 graph 的版本 |
| reward 曲线正常但评测不涨 | 引擎采样时开了 top_p/top_k 或 repetition penalty | 检查 rollout 采样参数 | 关闭截断；训练与评测用同一套采样 |
| 训练 loss 出现 NaN 或 KL 跳变 | `π_infer` 与 `π_old` 差异大且未修正；长序列累积 | 画 `|Δ logp|` 按位置曲线 | 开 TIS/MIS；检查 FP8 KV 是否在训练 rollout 里被打开 |
| rollout 阶段 GPU 利用率先高后长时间低 | 长尾样本，batch 末尾只剩几条在跑 | 引擎 `num_requests_running` 时间线 | 降 max_tokens、overlong shaping、partial rollout（Day 20）、动态补样 |
| colocated 下训练 OOM | 引擎没真正释放显存，或优化器状态没 offload | 切换点前后 `torch.cuda.memory_allocated` | `free_cache_engine=true`、sleep level 2、降 `gpu_memory_utilization` |
| 多卡同步后各推理 rank 权重不一致 | broadcast 组构建错误或 rank 映射错 | 各 rank 对同一层做 checksum | 用框架的 sharding manager；检查 TP rank 顺序 |
| prefix cache 命中率异常高 | 同步后没 reset，命中的是旧权重的 KV | `vllm:prefix_cache_hits` 在同步后是否清零 | 同步后 `reset_prefix_cache()` |

## 自测

1. 同步 RL 且 PPO epochs=1、mini-batch=1，`r_t` 是否恒为 1？考虑引擎不一致后呢？
2. 7B 模型 BF16 权重同步走 NCCL 跨机 IB（示例 400 Gb/s），理论下限多少秒？实际为什么更长？
3. `sleep(level=2)` 后唤醒，除了权重还要恢复什么才能保证 rollout 行为与上一步一致？

## 验收标准

- 能画出你所用框架里一次 step 的完整数据流，标出权重同步的时刻和 rollout 用的权重版本。
- 能解释 TIS 修正的每个符号，并说清它为什么需要引擎返回 logprob。
- 实验 1 的 `|Δ logp|` 分布图与实验 2 的同步时间表。
- 能在 colocated 与分离两种形态间做出选型并说明理由。

## 交付物

| 文件 | 内容 |
|---|---|
| `rollout-sync-path.md` | 同步路径表（带你版本的代码位置）、显存切换时序图 |
| `train-infer-mismatch.md` | 实验 1、4 的数据与结论 |
| `rollout-metrics.json` | 实验 2、3 的每步指标 |

## 参考

- T42 HybridFlow：verl 的单控制器架构与 sharding manager 设计动机。
- T43 OpenRLHF：Ray + vLLM 的早期形态，对比理解 colocated。
- T44 AReaL、T18 Kimi k1.5：分离形态与异步同步的工程细节。
- B01：隐式 off-policy 与 TIS 的来源，必读。
- B03：batch-invariant kernel，理解不一致的根源。
- T37 DAPO：overlong reward shaping 与采样参数设置。
- B06、B07、B08：各框架 rollout 配置的当前字段名。
