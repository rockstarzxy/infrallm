---
title: "优化专项 13: 并行策略选错导致的性能问题"
type: concept
tags: [ai-infra, inference-optimization, tensor-parallel, pipeline-parallel, data-parallel, expert-parallel, nccl, topology]
created: 2026-09-29
updated: 2026-09-29
---
# 优化专项 13: 并行策略选错导致的性能问题

> 一句话：模型放得下却跑不快、加了卡吞吐不涨、多节点延迟爆炸，多半是 TP/PP/DP/EP 选错或拓扑不匹配。本页给出通信量估算、判定方法和切换实验。关联日课：[[day15-gpu-topology]]、[[day16-tensor-parallel]]、[[day17-pipeline-parallel]]、[[day18-data-parallel]]、[[day19-moe-inference]]、[[day26-roofline]]。

## 1. 症状与判定

并行策略问题的特征是"单卡指标看起来都健康，但整体效率随卡数不线性"。先看这张表：

| 症状 | 关键证据 | 最可能根因 |
|---|---|---|
| TP 从 2 加到 4，decode 吞吐几乎不涨，TPOT 只降一点 | nsys 里 NCCL / custom allreduce kernel 时间占 step 的 30% 以上 | 小 batch 下 TP 通信占比反超计算（根因 A） |
| 同样 TP=4，在另一台机器上慢 2 到 5 倍 | `nvidia-smi topo -m` 显示 GPU 间是 PHB/SYS 而不是 NV# | 拓扑不匹配，TP 跑在 PCIe 上（根因 B） |
| 单机 8 卡 TP=8 跑 7B 模型，吞吐低于 8 个 TP=1 实例之和 | 每步 token 数少、KV 占用率低 | 模型放得下时不该加 TP，应 DP（根因 C） |
| 多节点 TP，TTFT 和 TPOT 同时高一个数量级 | `NCCL_DEBUG=INFO` 日志显示 `NET/Socket` 而非 `NET/IB` | 跨节点 TP 走了 TCP，或应改 PP（根因 D） |
| PP 开着，GPU 利用率在各 stage 之间交替空转 | 每个 stage 的 SM 利用率呈周期性 0 | pipeline bubble，microbatch 不足（根因 E） |
| TP 加大后每卡 KV block 数不再增长 | 启动日志 `# GPU blocks` 与 TP=kv_heads 时相同 | TP 超过 kv_heads，KV 被复制（根因 F） |
| MoE 模型 TP=8 慢于 attention DP + expert EP | expert 权重被切碎，每步 all-to-all 之外还有 allreduce | MoE 应该用 EP/DP 而非纯 TP（根因 G） |
| vLLM 原生 DP 下某些 rank 空转 | 日志里 dummy batch 频繁 | 前端负载不均，或 MoE lockstep 特性（根因 H） |

指标来源：`vllm:num_requests_running`、`vllm:kv_cache_usage_perc`、`vllm:time_per_output_token_seconds`，加上 nsys 或 torch profiler 的 kernel 时间分布。

## 2. 根因逐个排查

### 根因 A: 小 batch 下 TP 通信占比反超计算

**机制**。Megatron 风格 TP 在每个 transformer 层有两次 all-reduce：attention 输出投影之后一次，MLP down projection 之后一次。每次 all-reduce 的张量形状是 `[本步 token 数, hidden_size]`。

每卡每步通信量的估算公式（ring all-reduce）：

```
bytes_per_allreduce = tokens_this_step × hidden_size × bytes_per_elem
per_gpu_traffic     = 2 × (N-1)/N × bytes_per_allreduce      # N = TP size
per_step_traffic    = layers × 2 × per_gpu_traffic
```

示例（仅作量级说明）：70B 级模型 hidden 8192、80 层、bf16、decode 步 64 个 token、TP=8：单次 all-reduce 1 MiB，每步 160 次，每卡传输约 280 MB。在 NVLink 上带宽不是问题，问题是每次 all-reduce 都有固定几十微秒的启动延迟，160 次叠加就是几毫秒，而 decode 一步本身可能只有十几毫秒。batch 越小，计算越少，这几毫秒的占比越高。

**确认**。

```bash
nsys profile -t cuda,nvtx -o tp_profile python -m vllm.entrypoints.openai.api_server ... &
# 压测一段时间后停止，看统计
nsys stats --report cuda_gpu_kern_sum tp_profile.nsys-rep | grep -iE "nccl|cross_device_reduce|all_reduce"
```

`cross_device_reduce_1stage` / `cross_device_reduce_2stage` 是 vLLM 的 custom allreduce kernel，`ncclDevKernel_AllReduce*` 是 NCCL 的。两者时间之和除以总 kernel 时间就是通信占比。超过 25% 就说明 TP 对当前负载太大。

**修**。三个方向按优先级：
1. 降 TP，用 DP 多实例补吞吐（见根因 C）。
2. 提高每步 token 数：放宽 `--max-num-seqs`、确认 chunked prefill 把 prefill 分片和 decode 混排。通信量随 token 数线性增长，但启动延迟不变，摊薄后占比下降。
3. 确认 custom allreduce 在用。vLLM 在单机 NVLink 全连接时默认对小消息用 custom allreduce；`--disable-custom-all-reduce` 是关闭开关，排查时可以对比开关前后。

**副作用**。降 TP 意味着每卡要放下更多权重，可能需要量化（[[10-quantization-choice]]）。

### 根因 B: 拓扑不匹配，TP 跑在 PCIe 上

**确认**。

```bash
nvidia-smi topo -m
```

判读矩阵里 GPU 对之间的标记：`NV#` 表示有 # 条 NVLink，TP 应该只在这些 GPU 之间；`PIX` 同一 PCIe switch；`PXB` 跨多个 PCIe switch；`PHB` 经过 CPU 的 PCIe host bridge；`NODE` 同 NUMA 节点但跨 host bridge；`SYS` 跨 NUMA，最慢。PCIe 版 GPU 常见的是 PIX/PHB，此时 TP=2 勉强可用，TP=4 以上通信会明显拖慢 decode。

`nvidia-smi nvlink -s` 看 NVLink 是否全部 active。有过 NVLink 部分断开导致 TP 慢的案例。

**修**。用 `CUDA_VISIBLE_DEVICES` 把 TP 组限制在 NVLink 互联的 GPU 内；如果机器整体没有 NVLink，把 TP 降到 2 或 1，用 DP 补吞吐。

### 根因 C: 模型放得下时加 TP 而不是 DP

**机制**。TP 的收益是把权重读取分摊到多卡，让 decode 的显存带宽瓶颈变宽；代价是每步通信。当单卡能放下权重和足够的 KV 时，多个独立实例（DP）没有通信开销，总吞吐接近线性，只是单请求延迟不如 TP。

判定口诀：**延迟优先用 TP，吞吐优先用 DP，两者都要则 TP 取能满足 TPOT SLO 的最小值，剩下的卡做 DP。**

**确认**。做 TP 梯度实验（第 4 节实验 1），对比 `8 × TP1` 与 `4 × TP2`、`2 × TP4`、`1 × TP8` 的总 tokens/s 和 TPOT P99。

**修**。两种 DP 形态：
- 外部多实例 + 网关：每个实例独立 `vllm serve`，前面放 LB。最通用，实例间无耦合，支持 prefix-aware routing（[[day18-data-parallel]]）。
- vLLM 原生 DP：`--data-parallel-size N`，一个前端进程带 N 个 EngineCore，由 DPCoordinator 分发。多节点时配合 `--data-parallel-size-local` 和 `--data-parallel-address`。MoE 模型下各 DP rank 必须同步 step，这是它存在的主要理由（根因 G）。具体 LB 模式的参数随版本变化，以 `vllm serve --help` 为准。

**副作用**。DP 下 prefix cache 分散在各实例，命中率下降，需要 cache-aware routing。

### 根因 D: 跨节点 TP 或 NCCL 走了错误网络

**机制**。TP 的 all-reduce 对延迟极其敏感，跨节点即使走 InfiniBand 也比 NVLink 慢一个量级以上。跨节点应优先 PP（每层边界只传一次激活）或 DP（不传）。

**确认**。

```bash
NCCL_DEBUG=INFO vllm serve ... 2>&1 | grep -E "NET/|NCCL INFO Using|Connected"
```

期望看到 `NET/IB` 或 `NET/IBext`，如果是 `NET/Socket` 说明在走 TCP。用 `nccl-tests` 的 `all_reduce_perf -b 1M -e 256M` 看 busbw，IB HDR 单卡应到几十 GB/s 量级。

**修**。
- 指定网卡：`NCCL_IB_HCA=mlx5_0,mlx5_1`，`NCCL_SOCKET_IFNAME=eth0`（控制面网卡）。
- 多节点用节点内 TP + 节点间 PP：`--tensor-parallel-size 8 --pipeline-parallel-size 2`。
- 排查时可以用 `NCCL_P2P_DISABLE=1` 或 `NCCL_IB_DISABLE=1` 做对照，确认问题在哪一层，排查完记得去掉。

**副作用**。PP 引入 bubble（根因 E）。

### 根因 E: PP 的 bubble 没有被 microbatch 填满

**机制**。PP 把层切成 S 段，一个 batch 依次流过各 stage。如果同时只有一个 batch 在流水线里，任意时刻只有一个 stage 在算，利用率是 1/S。bubble 比例的近似：

```
bubble ≈ (S - 1) / (M + S - 1)      # M = 同时在流水线里的 microbatch 数
```

vLLM V1 的 PP 实现会让多个调度批次同时在流水线中（in-flight 批次数与 PP size 相关），所以负载高时 bubble 自动变小，负载低时（比如只有一两个请求）bubble 接近 (S-1)/S。

**确认**。`nvidia-smi dmon -s u` 看各卡 SM 利用率是否周期性交替为 0；或者 torch profiler 的 timeline 上各 rank 的 forward 是否错开且有空白。

**修**。
- 保证并发足够：PP 不适合低 QPS 场景。
- 能用 TP 就不用 PP：单机内 NVLink 全连接时 TP 几乎总是优于 PP。
- 层数不能整除 stage 数时，vLLM 会不均分配，尾部 stage 可能成为瓶颈，看各 rank 的 step 时间。

### 根因 F: TP 超过 kv_heads，KV 被复制

**机制**。GQA 模型的 KV head 数远小于 Q head 数。TP 切分 attention 时按 KV head 切，当 TP > kv_heads，vLLM 会在多卡上复制同一个 KV head，KV cache 每卡占用不再随 TP 下降，attention 计算也有重复。

示例：Qwen2.5-7B 有 4 个 kv_heads，TP=8 时每个 KV head 被复制 2 份；Llama-3-70B 有 8 个 kv_heads，TP=8 刚好一卡一个，TP=16 开始复制。

**确认**。对比 TP=kv_heads 与更大 TP 的启动日志 `# GPU blocks`，如果没有按比例增长就是被复制了。

**修**。TP 取值不超过 kv_heads；再需要吞吐用 DP。MLA 模型（DeepSeek）没有多 KV head 的概念，TP 对 KV 的切分方式不同，以 [[day19-moe-inference]] 为准。

### 根因 G: MoE 模型用纯 TP

**机制**。MoE 的 expert 权重本来就是按 expert 天然可切的。纯 TP 把每个 expert 再切成 N 份，每层除了 all-to-all 路由还要 allreduce，通信翻倍且每个 expert 的 GEMM 变小效率变差。主流做法是 attention 用 DP（或小 TP），expert 用 EP（每卡放一部分完整 expert），层间用 all-to-all 交换 token。

**修**。`--enable-expert-parallel` 配合 `--data-parallel-size` 或 `--tensor-parallel-size`，具体组合和 DeepEP 等通信库的启用方式见 [[12-moe-serving]]。

**副作用**。EP 下 expert 负载不均会造成某些卡等其他卡，需要看 expert 命中分布。

### 根因 H: DP rank 间负载不均或 lockstep 空转

**机制**。原生 DP 在 MoE 模型下要求所有 rank 每步同时 forward（因为 all-to-all 需要所有 rank 参与），没请求的 rank 跑 dummy batch。如果前端分发不均，一部分 rank 满载一部分空转，整体吞吐被最忙的 rank 决定。

**确认**。按 rank 拆看 `vllm:num_requests_running`（指标带 rank 标签的版本），或日志里 dummy batch 计数。

**修**。请求长度分布差异大时，优先外部网关做 least-loaded 路由；原生 DP 更适合请求分布均匀的批量场景。

## 3. 决策表

| 情况 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| 模型单卡放得下，要吞吐 | DP 多实例，TP=1 | 原生 DP | TP=8 跑 7B |
| 模型单卡放得下，要低 TPOT | TP=2（NVLink） | 量化后 TP=1 | 跨 PCIe 的 TP |
| 模型需要 2 到 8 卡才放下 | TP = 放下所需最小值，且 ≤ kv_heads | TP + 量化再降一档 | TP 超过 kv_heads |
| 单机放不下 | 节点内 TP，节点间 PP | 量化后压回单机 | 跨节点 TP |
| MoE 大模型 | attention DP + expert EP | TP + EP 混合 | 纯 TP |
| 低 QPS、单请求延迟敏感 | TP 到 NVLink 上限 | spec decode 叠加 | PP |
| 高 QPS、SLO 宽松 | DP 优先，TP 最小化 | 原生 DP（MoE） | 为了显存开 PP 而不加 microbatch |

## 4. 验证实验

### 实验 1: TP 梯度

固定：模型、dtype、`max_model_len`、workload（input 512 / output 256）、总 GPU 数 8。变量：并行组合。

```bash
# 组合 a: 8 × TP1（8 个端口），组合 b: 4 × TP2，组合 c: 2 × TP4，组合 d: 1 × TP8
for tp in 1 2 4 8; do
  n=$((8 / tp))
  for i in $(seq 0 $((n-1))); do
    CUDA_VISIBLE_DEVICES=$(seq -s, $((i*tp)) $((i*tp+tp-1))) \
    vllm serve MODEL --tensor-parallel-size $tp --port $((8000+i)) &
  done
  wait_ready
  # 用一个简单轮询 LB 或多进程 bench 打到 n 个端口，固定总并发 128
  vllm bench serve --model MODEL --base-url http://LB:8000 --dataset-name random \
    --random-input-len 512 --random-output-len 256 --num-prompts 1000 --max-concurrency 128
  pkill -f "vllm serve"
done
```

预期：总 tokens/s 随 TP 增大而下降（模型放得下时），TPOT P50 随 TP 增大而下降。选 TPOT 满足 SLO 的最小 TP。

### 实验 2: 通信占比

固定 TP=4，变量为并发 1 / 8 / 64。用 nsys 抓 30 秒，算 allreduce kernel 时间占比。预期：并发 1 时占比最高，并发 64 时明显下降。如果并发 64 仍超过 25%，检查拓扑（根因 B）和 custom allreduce 是否生效。

### 实验 3: 拓扑对照

同一 TP=2 配置，分别选 NVLink 直连的 GPU 对和 PHB 的 GPU 对（用 `CUDA_VISIBLE_DEVICES`）。预期：PHB 组合 TPOT 高 30% 以上，差距随并发降低而扩大。

### 实验 4: 多节点网络

`nccl-tests` 的 `all_reduce_perf` 在 2 节点 16 卡上跑，分别用默认配置和 `NCCL_IB_DISABLE=1`。busbw 差距应该是数量级级别。然后用两种配置各起一次 TP=8 PP=2 的 vLLM，对比 TTFT。

## 5. 关联

- [[01-diagnosis-playbook]]：从整体症状进入这一页的路径。
- [[04-throughput-low]]：吞吐问题里并行策略只是原因之一。
- [[05-oom-preemption]]：降 TP 后显存不够时的处理。
- [[10-quantization-choice]]：用量化换更小的 TP。
- [[12-moe-serving]]：EP 与 DP 的组合细节。
- [[15-cost-per-token]]：并行策略对单位 token 成本的影响。
- 日课：[[day15-gpu-topology]]、[[day16-tensor-parallel]]、[[day17-pipeline-parallel]]、[[day18-data-parallel]]、[[day19-moe-inference]]。
