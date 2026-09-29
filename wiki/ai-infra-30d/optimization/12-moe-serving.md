---
title: "优化专项 12: MoE 模型部署与性能问题"
type: concept
tags: [ai-infra, inference-optimization, moe, expert-parallel, all-to-all, deepseek, qwen3-moe, wide-ep, eplb]
created: 2026-09-29
updated: 2026-09-29
---

# 优化专项 12: MoE 模型部署与性能问题

> 一句话：MoE 的瓶颈不在算力而在"专家权重怎么放、token 怎么路由到专家所在的卡、各卡负载是否均衡"，这页给出并行方案选择、通信与负载诊断、以及 DeepSeek / Qwen3 / MiniMax 的典型部署形态。关联日课：[[day19-moe-inference]]、[[day16-tensor-parallel]]、[[day15-gpu-topology]]、[[day18-data-parallel]]、[[day12-speculative-decoding]]、[[day26-roofline]]。

## 0. 先算清楚三个数

| 量 | 决定什么 | 示例（DeepSeek-V3，671B 总参 / 37B 激活，256 路由专家 + 1 共享，top-8） |
|---|---|---|
| 总参数 × 每参数字节 | 权重显存，决定最少几张卡 | FP8 约 671 GB，8 卡 H200（141 GB）刚好，8 卡 H100（80 GB）放不下 |
| 激活参数 × 每参数字节 | 每个 token 的权重读取量，决定 decode 的 memory-bound 下限 | FP8 约 37 GB/token，比同显存的 dense 模型少一个量级 |
| 每步被激活的专家总数 | 实际读多少专家权重 | batch 小时每步只激活少数专家，batch 大到覆盖全部专家后，读取量趋于"整个模型" |

第三条是 MoE 特有的：**MoE 在小 batch 下比 dense 更 memory-bound，在大 batch 下读取量向总参数收敛**。这是所有 MoE 性能问题的底层原因，也是为什么 MoE serving 需要比 dense 更大的 batch 才能把 GPU 用满。

## 1. 症状与判定

| 症状 | 伴随指标 / 现象 | 根因方向 | 跳到 |
|---|---|---|---|
| 吞吐远低于按激活参数估算的值 | nsys 里 all-to-all / NCCL kernel 占步时间 30% 以上 | 通信主导：EP 跨节点、没用高性能 all-to-all、拓扑不匹配 | 2.1 |
| 各 GPU 利用率差异大，步时间由最慢的卡决定 | 某几张卡 SM 利用率高、其他卡在等 | 专家负载不均（热门专家扎堆在少数卡） | 2.2 |
| 低并发下 TPOT 远高于 dense 同激活量模型 | 并发 1 到 8，SM 利用率低 | fused MoE kernel 在小 batch 下效率低；专家权重读取碎片化 | 2.3 |
| DP 部署下有的 rank 没请求也在跑 | 日志里 dummy batch / DP 同步 | attention DP + EP 要求各 rank 同步 step | 2.4 |
| 显存 OOM 或 KV 极小 | 启动日志 KV 只有几 GB | 专家权重占满显存；没用 EP 或没量化 | 2.5 |
| 启动日志提示 fused_moe 没有 tuned config | "Using default MoE config" 类提示 | kernel tile 参数没为该 shape / GPU 调过 | 2.6 |
| TP=8 跑 MoE，吞吐随并发增长很慢 | all-reduce 占比高 | TP 切专家：每层两次 all-reduce，且每卡都要参与所有专家 | 2.1 / 3 |

诊断命令：

```bash
# 通信占比
nsys profile -t cuda,nvtx -o moe python -c "..."     # 看 ncclKernel_* / all_to_all 时间
# 专家负载（需要打开 vLLM 的专家统计，版本相关；或自己在 fused_moe 前 hook 统计 topk_ids）
# DP 同步等待
VLLM_LOGGING_LEVEL=DEBUG  # 看 DP coordinator 日志里的 step 同步与 dummy batch
# 拓扑
nvidia-smi topo -m
```

## 2. 根因逐个排查

### 2.1 通信主导：EP 与 TP 的选择

MoE 层的并行有两种切法：

| 方式 | 权重放置 | 每层通信 | 优点 | 缺点 |
|---|---|---|---|---|
| TP 切专家（默认，不加 `--enable-expert-parallel`） | 每个专家的矩阵按列/行切到所有 rank | 两次 all-reduce（与 dense 相同） | 实现简单，节点内 NVLink 下够用 | 每卡都要持有全部专家的一片，小 batch 下每卡的 GEMM 极碎；专家数多时效率差 |
| EP（`--enable-expert-parallel`） | 专家整个放在某个 rank 上，`experts / EP size` 个每卡 | dispatch all-to-all（token 发到专家所在卡）+ combine all-to-all（结果发回） | 每卡 GEMM 完整、专家权重只存一份 | 通信是 all-to-all，对跨节点带宽和延迟敏感；负载不均直接变成等待 |

**经验边界**：8 卡以内、单节点 NVLink、专家数不多（Mixtral 8 专家）用 TP 通常够；专家数上百（DeepSeek 256、Qwen3-235B 128）或跨节点部署，EP 是主流。vLLM 里 EP size = TP size × DP size，通过 `--tensor-parallel-size` 和 `--data-parallel-size` 组合得到。

**all-to-all 实现**：默认走 NCCL 或 PyTorch 的 all-to-all。DeepSeek 开源的 DeepEP 提供为 MoE 定制的高性能 all-to-all（节点内 NVLink、节点间 RDMA，decode 有低延迟模式），vLLM 通过 `--all2all-backend`（名称随版本，如 `deepep_high_throughput` / `deepep_low_latency` / `pplx`）接入，需要单独安装 DeepEP 和对应网络驱动。DeepGEMM 是配套的 FP8 GEMM 库，也是可选加速项。

**确认**：nsys 看 all-to-all 占比；跨节点部署时看 IB 带宽是否被打满（`ibstat`、`nvidia-smi nvlink`）。

**修**：单节点先试 TP；多节点或大专家数用 EP + DeepEP；跨节点 EP 必须有 RDMA（IB 或 RoCE），走 TCP 的 EP 几乎不可用。

### 2.2 专家负载不均

路由是数据驱动的，某些专家在特定任务上被频繁选中。EP 下这些专家所在的卡成为瓶颈，其他卡在等。

**确认**：统计一段时间内每个专家的 token 数（vLLM 的专家负载统计随版本提供，或在 `fused_moe` 入口 hook `topk_ids` 做直方图）。最热专家 / 平均值超过 2 到 3 倍就要处理。

**修**：
- **EPLB（Expert Parallel Load Balancer）**：按统计结果重新分配专家到卡，并给热门专家建冗余副本（redundant experts）。vLLM 支持 `--enable-eplb` 及相关参数（窗口大小、重平衡步数、冗余专家数，名称随版本），DeepSeek 的 EPLB 算法是参考实现。
- 冗余专家会多占显存，权衡副本数与 KV。
- 任务分池：代码流量和对话流量的热门专家不同，分实例后各自的负载更均匀。

### 2.3 小 batch 下 MoE 的低效

每步只有少量 token，分到每个专家的 token 可能只有个位数。fused MoE kernel 按专家分组做 grouped GEMM，每组极小，Tensor Core 利用率低；同时每个被激活的专家权重都要完整读一遍。

**确认**：并发 1 到 8 时 SM 利用率低于 30%，TPOT 与"读一遍激活专家权重的时间"接近。

**修**：
- 用 attention DP 把多个 DP rank 的 token 汇聚到同一批专家计算里（wide-EP 思路，见 2.4），让每个专家看到更多 token。
- 提高并发；MoE 实例的经济 batch 通常比 dense 大数倍。
- 开 spec decode（[[11-spec-decode-no-gain]]）：MTP 一步验证多个位置，等价于放大 batch，DeepSeek-V3/R1 的 MTP 是天然搭配。
- CUDA graph 覆盖 MoE 路径（默认应该覆盖，看 capture 日志）。

### 2.4 attention DP + expert EP（wide-EP）

思路：attention 部分 KV 大、权重小，用 DP 每 rank 一份；MoE 部分权重大，用 EP 切到所有 rank。一个 token 在自己的 DP rank 做 attention，再 all-to-all 到专家所在 rank 做 FFN。

```bash
vllm serve deepseek-ai/DeepSeek-R1 --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel   # 单节点示例
# 多节点：各节点 --data-parallel-size-local、--data-parallel-address、--data-parallel-rpc-port，名称随版本
```

**代价**：所有 DP rank 必须同步 step。某个 rank 没有请求时也要跑 dummy batch，否则其他 rank 在 all-to-all 里永远等不到它。低负载时这是浪费；高负载时各 rank 的 batch 不均衡会让步时间由最大 batch 决定。

**确认**：DP coordinator 日志里的 dummy batch 频率；各 rank 的 `num_requests_running` 差异。

**修**：前端负载均衡要按 rank 均分（`--data-parallel-hybrid-lb` 或外部 LB 按 rank 路由，名称随版本）；负载低时缩 DP size；MLA 模型的 attention 在 DP 下每 rank 一份完整 KV 计算，比 TP 切 head 更划算，这是 DeepSeek 推荐 DP attention 的原因。

### 2.5 显存放不下

**修法按顺序**：EP（专家只存一份）→ FP8（DeepSeek 原生 FP8，Qwen3 MoE 有 FP8 版本）→ 更大显存的卡（H200 / B200）→ NVFP4（Blackwell）。注意 EP 下每卡只放 `experts / EP` 个专家，但 attention 和共享专家仍每卡一份。

MoE 量化的特殊性见 [[10-quantization-choice]]：专家权重是显存大头，量化收益最大；fused MoE kernel 对 FP8 blockwise、W4A16 的支持随版本，看启动日志用的是哪个 kernel。

### 2.6 fused MoE kernel 未调优

vLLM 的 Triton fused MoE kernel 按（专家数、隐藏维、中间维、dtype、GPU 型号）查一份 tile 配置 JSON（仓库里 `fused_moe/configs/` 目录，文件名含 E、N、device name）。没有匹配文件时用默认配置，可能慢 20% 到 50%。

**确认**：启动日志有 "Using default MoE config" 或类似提示。

**修**：用仓库里的 `benchmark_moe.py` 脚本为你的 shape 和 GPU 生成配置文件放到对应目录；Blackwell 上另有 CUTLASS / FlashInfer 的 MoE kernel 路径，看日志确认走了哪条。

## 3. 决策表

典型部署形态（写作时的常见做法，不含精确吞吐数字）：

| 模型 | 硬件 | 典型形态 |
|---|---|---|
| DeepSeek-V3 / R1（FP8 原生，MLA，MTP） | 8 × H200 单节点 | TP=8 或 DP=8 + EP，MTP k=1，FP8 KV 可选 |
| 同上 | 2 节点 × 8 H100 | TP=8 节点内 + PP=2，或 DP=16 + EP=16 + DeepEP（需 IB） |
| 同上 | 多节点 B200 | wide-EP（DP attention + EP 32 以上）+ DeepEP + EPLB，P/D 解耦（[[day24-disaggregated]]） |
| Qwen3-235B-A22B（128 专家，GQA） | 8 × H100 | TP=8（FP8 版本）或 TP=4 × DP=2 + EP |
| Qwen3-30B-A3B | 1 × H100 或 2 × A100 | TP=1 / 2，不需要 EP |
| MiniMax-M1 / Text-01（Lightning attention + MoE） | 8 × H100 | TP=8；线性注意力层 KV 固定，长上下文友好 |
| Mixtral 8x7B / 8x22B | 2 到 4 × A100 | TP，不需要 EP |

症状到动作：

| 症状 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| all-to-all 占比高，单节点 | 回到 TP 切专家 | EP + DeepEP 节点内模式 | 跨 TCP 做 EP |
| all-to-all 占比高，多节点 | EP + DeepEP + RDMA | 节点内 TP + 节点间 PP | 多节点 TP |
| 专家负载不均 | EPLB + 冗余专家 | 按任务分池 | 手动固定专家映射后不再更新 |
| 低并发 TPOT 高 | MTP / spec decode | DP attention 聚合 batch | 加 TP |
| DP rank 空转 | 缩 DP size 或改 LB 策略 | — | 关掉同步（会死锁） |
| 显存不够 | EP + FP8 | H200 / B200 | 二次量化原生 FP8 权重 |
| kernel 未调优提示 | benchmark_moe 生成 config | 换 CUTLASS/FlashInfer 路径 | 忽略 |

## 4. 验证实验

固定：模型、并行配置、input 1024 / output 256、temperature 0、300 条、warmup。

实验 1：TP 切专家 vs EP（单节点 8 卡，Qwen3-235B-A22B-FP8）

```bash
vllm serve Qwen/Qwen3-235B-A22B-FP8 --tensor-parallel-size 8 --port 8000 &                              # A
vllm serve Qwen/Qwen3-235B-A22B-FP8 --tensor-parallel-size 8 --enable-expert-parallel --port 8001 &     # B
for C in 8 32 128 256; do
  vllm bench serve --model Qwen/Qwen3-235B-A22B-FP8 --port 8000 --dataset-name random --random-input-len 1024 --random-output-len 256 --num-prompts 300 --max-concurrency $C
  # 同样对 8001
done
```

预期：低并发两者接近，高并发 EP 领先；nsys 里 A 的 all-reduce 时间与 B 的 all-to-all 时间对比。

实验 2：负载均衡

在 EP 配置下，先用纯代码 prompt 跑 300 条，再用纯对话 prompt 跑 300 条，各自统计专家直方图和步时间方差。开 EPLB 后重跑，看最热专家比例和 P99 TPOT 变化。

实验 3：MTP 对 MoE 的收益

DeepSeek-R1 在 8 × H200 上，开关 `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`，并发 1 / 8 / 32 对比 TPOT 和接受率。预期：低并发收益最明显，接受率通常高于通用 EAGLE。

实验 4：DP 空转代价

DP=8 + EP，用 request rate 从 1 到 50 QPS 扫，记录 dummy batch 次数和 GPU 平均利用率。找到 dummy batch 占比低于 10% 的最小 QPS，这就是这个形态的经济负载下限。

## 5. 关联

- 诊断入口：[[01-diagnosis-playbook]]、[[04-throughput-low]]
- 相关专项：[[10-quantization-choice]]、[[11-spec-decode-no-gain]]、[[13-parallelism-choice]]、[[15-cost-per-token]]、[[07-long-context]]
- 日课：[[day19-moe-inference]]、[[day16-tensor-parallel]]、[[day15-gpu-topology]]、[[day18-data-parallel]]、[[day24-disaggregated]]、[[day26-roofline]]

## 6. 速答

- **"MoE 的显存按激活参数算就行吧"**：不行，全部专家都要常驻显存，按总参数算；激活参数只决定每 token 的算力和带宽。
- **"EP 一定比 TP 快"**：不一定，单节点小专家数时 TP 常更快；EP 的优势在专家数多、跨节点、以及配合 DP attention 聚合 batch。
- **"共享专家怎么放"**：共享专家每个 token 都过，通常每卡一份复制（或 TP 切），不参与 EP 分配。
- **"MoE 能开 prefix caching / chunked prefill 吗"**：都能，与 dense 无差异；差异只在 FFN 层。
- **"为什么 DeepSeek 推荐 DP attention"**：MLA 的 KV 是共享 latent，TP 切 head 后每卡仍要读同一份 latent，收益小；DP 让每 rank 独立算 attention，再用 EP 把 FFN 摊开。
