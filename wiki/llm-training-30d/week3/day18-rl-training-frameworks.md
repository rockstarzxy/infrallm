---
title: "Day 18：RL 训练框架架构与选型"
type: concept
tags: [llm-training, rl-frameworks, verl, openrlhf, slime, nemo-rl, areal]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 18：RL 训练框架架构与选型

> 上一课 [[llm-training-30d/week3/day17-rlvr-verifiers-reward-design]] · 下一课 [[llm-training-30d/week3/day19-rollout-engine-weight-sync]]。相关：推理课的 vLLM 架构 [[ai-infra-30d/week2/day08-vllm-architecture]]（rollout 引擎的内部）；本课 Day 4 到 6 的训练并行与框架 [[llm-training-30d/week1/day06-frameworks-efficiency-profiling]]；Agent 课的 harnessed RL [[agent-rsi-30d/week3/day20-harnessed-agent-rl]]。

## 学习目标

1. 画出一个 RL 训练系统的通用组件图：数据、rollout 引擎、reward 服务、actor/critic/reference、权重同步、调度器，并说清每个组件的显存与时间占比。
2. 解释 verl 的 HybridFlow 设计：单控制器 + 多控制器、colocated 与 disaggregated 放置、训练后端与 rollout 后端的组合。
3. 能读完 verl 一次 GRPO step 的调用链，指出 rollout、reward、advantage、update 四段各在哪个文件。
4. 对比 verl、OpenRLHF、slime、NeMo-RL、AReaL、SkyRL、ROLL、TRL 的架构取舍，按模型规模、MoE、异步、agent 支持给出选型。
5. 估算一次 GRPO 迭代的时间分解，并指出瓶颈在哪一段。

## 工业现状

后训练 RL 系统在 2024 到 2025 年经历了一次架构收敛。早期 RLHF 系统（DeepSpeed-Chat、TRL 早期）把 rollout 用 HF generate 做，慢一个数量级。现在所有主流框架都把推理引擎（vLLM 或 SGLang）嵌进训练循环做 rollout，训练用 FSDP 或 Megatron，中间靠权重同步与 resharding 连接，用 Ray 或自研调度做多角色编排。

verl（T42，字节）是当前最广泛使用的开源框架：DAPO（T37）、许多学术复现和不少工业团队基于它。slime（B07，智谱）是 GLM-4.5（T20）的训练框架，Megatron + SGLang，为大 MoE 和 agentic 数据生成设计。OpenRLHF（T43）是最早把 Ray + vLLM 组合起来的框架。NeMo-RL（B08，NVIDIA）走 Megatron-Core / DTensor 后端。AReaL（T44）是全异步系统，用 interruptible rollout 与解耦 PPO 目标解决 staleness。SkyRL（B12）与 ROLL（B13）侧重 agent 场景的环境接口。TRL（B05）的 GRPOTrainer 适合单机与快速原型，也支持 vLLM colocate。

选型的主要维度是：模型规模（决定训练后端 FSDP 还是 Megatron）、是否 MoE（Megatron 的 EP 支持更成熟）、是否需要异步（长尾 rollout 严重时）、agent 环境接口是否原生、团队对 Ray 的接受度。

## 核心原理

### 通用组件图

```text
                       ┌─────────────────────────────────────────────┐
  prompt 数据集 ─────▶ │  Controller / Trainer（单进程驱动，Ray driver）│
                       └──────┬──────────────┬───────────────┬────────┘
                              │ 1 采样        │ 2 打分         │ 3-4 计算与更新
                              ▼              ▼               ▼
                  ┌──────────────────┐ ┌──────────────┐ ┌──────────────────────────┐
                  │ Rollout 引擎      │ │ Reward       │ │ Actor（策略，训练后端）    │
                  │ vLLM / SGLang     │ │ verifier /   │ │ Reference（只 forward）   │
                  │ TP 布局的权重副本  │ │ RM 模型      │ │ Critic（PPO 才有）        │
                  └────────┬─────────┘ └──────────────┘ └─────────┬────────────────┘
                           │  ◀──────── 权重同步（每步一次）──────────┘
                           │  colocated：同一批 GPU 轮流用；disaggregated：各自独占 GPU
```

一次迭代的时间分解（示例，7B dense，GRPO，G=16，响应 4k 到 8k）：

| 阶段 | 占比 | 瓶颈 |
|---|---|---|
| rollout 采样 | 60% 到 80% | 最长的几条 response 决定这一步的时长（长尾） |
| reward 计算 | 1% 到 10% | 代码沙箱可能很慢 |
| old/ref logprob 重算 | 5% 到 10% | 两次 forward |
| actor 更新 | 10% 到 20% | 与 ppo_epochs 成正比 |
| 权重同步 | 1% 到 5% | 大模型与跨节点时上升 |

这张表决定了后面三天的重点：rollout 长尾（Day 19、20）与异步（Day 20）。

### verl 的 HybridFlow

两层控制：

```text
单控制器（driver，Python 进程）：
  描述整个 RL 数据流：rollout → reward → advantage → update，用 DataProto 在各角色间传数据
多控制器（每个角色一个 WorkerGroup，每个 worker 一个进程/GPU）：
  角色内部用 SPMD 方式执行分布式计算（FSDP/Megatron 的 TP/PP/DP 对 driver 透明）
```

driver 调用 `actor_rollout_wg.generate_sequences(batch)` 时，WorkerGroup 把 batch 按 DP 切分发给每个 worker，worker 内部走 vLLM，结果收集回 driver。这就是"单控制器"让算法代码像单机脚本一样短、"多控制器"让每个角色能用任意并行策略的原因。

角色与放置：

```text
ActorRolloutRefWorker：actor（训练）+ rollout（vLLM/SGLang）+ reference 三合一
  hybrid engine：训练与推理共享同一批 GPU，通过 resharding 在 FSDP/Megatron 布局与 vLLM TP 布局间切换
  rollout 时训练态权重 offload 或 sleep，推理态权重加载；训练时反过来（Day 19）
CriticWorker、RewardModelWorker：PPO / RM 场景才有
资源池：可以把不同角色放到不同 GPU 组（disaggregated），也可以全部 colocated
```

训练后端：FSDP（默认，改模型方便）或 Megatron（大模型、MoE、PP）。rollout 后端：vLLM 或 SGLang。四种组合都支持，以当前版本文档为准。

### 一次 GRPO step 的调用链（verl）

```text
RayPPOTrainer.fit()                                  verl/trainer/ppo/ray_trainer.py
  for batch in dataloader:
    gen_batch = actor_rollout_wg.generate_sequences(batch)      # 1. rollout（vLLM）
    batch = batch.union(gen_batch)                              # 把 responses 合并进 DataProto
    reward_tensor = reward_fn(batch)                            # 2. reward（自定义函数或 RM worker）
    batch = actor_rollout_wg.compute_log_prob(batch)            # 3a. old_log_probs（训练侧重算）
    batch = ref_policy_wg.compute_ref_log_prob(batch)           # 3b. ref_log_prob
    batch = compute_advantage(batch, adv_estimator="grpo")      # 3c. group advantage，core_algos.py
    actor_output = actor_rollout_wg.update_actor(batch)         # 4. mini-batch 更新，policy loss 在 core_algos.py
    # 权重同步发生在下一次 generate_sequences 之前（hybrid engine 内部）
```

文件名与函数名随版本变化，读的时候用 `grep -n "def generate_sequences\|def update_actor\|def compute_advantage" -r verl/` 定位。

### 其他框架的架构要点

| 框架 | 训练后端 | rollout | 编排 | 放置 | 异步 | agent 接口 | 适合 |
|---|---|---|---|---|---|---|---|
| verl | FSDP / Megatron | vLLM / SGLang | Ray | colocated 或 disaggregated | 部分（agent loop 与 async rollout 视版本） | 有 agent loop / 多轮工具 | 通用首选，7B 到数百 B |
| OpenRLHF | DeepSpeed ZeRO | vLLM | Ray | 可 colocate | 有 async 模式 | 有限 | 中等规模，代码简单 |
| slime | Megatron | SGLang | 自研 + Ray | 训练与 rollout 可分离 | 支持 | 数据生成为中心，自定义 rollout 函数 | 大 MoE、GLM 系 |
| NeMo-RL | Megatron-Core / DTensor | vLLM | Ray | 可分离 | 视版本 | 有 | NVIDIA 栈、大模型 |
| AReaL | 自研（Megatron 类） | SGLang / vLLM | 自研 | 分离 | 全异步，interruptible rollout | 有 | 长尾严重、追求吞吐 |
| SkyRL | FSDP | vLLM / SGLang | Ray | 可分离 | 有 | SkyRL-gym 原生 | agent RL 研究 |
| ROLL | Megatron / DeepSpeed | vLLM / SGLang | Ray | 可分离 | 有 | 有 | 大规模 agentic |
| TRL GRPOTrainer | HF / DeepSpeed / FSDP | vLLM colocate 或 server | 无 | 单机为主 | 无 | 弱 | 原型、单机、教学 |

必须理解的三个取舍：

1. **colocated vs disaggregated**：colocated 省 GPU（同一批卡训练与采样轮流），但采样时训练卡闲、训练时采样卡闲；disaggregated 两边可流水但需要更多卡且权重同步跨机。异步系统本质上是 disaggregated 加上放宽同步。
2. **FSDP vs Megatron 后端**：FSDP 改模型代码零成本，适合 70B 以下 dense；Megatron 有 TP/PP/EP 与更好的 MoE 支持，大模型必需，但模型要按 Megatron 方式实现且权重转换麻烦。
3. **Ray 的角色**：只做进程编排与 RPC，不在数据路径上；大对象通过 Ray object store 或 NCCL 传，别让 DataProto 走 pickle 序列化几个 GB。

### 知道即可

- 多数框架的 reference 模型可以用 LoRA 技巧省掉：actor 用 LoRA 训练时，关闭 adapter 就是 reference。
- 某些框架支持把 reward model 也用 vLLM 跑（只 forward），比训练后端快。

## 实现步骤

### 1. 装 verl 并跑通最小 GRPO

```bash
git clone https://github.com/volcengine/verl && cd verl
pip install -e .            # 需要与 vLLM 版本匹配，看 docs 的版本矩阵
# 数据预处理成 parquet（prompt + ground_truth）
python examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
# 单卡 0.5B GRPO（以 examples/grpo_trainer 下脚本为准，字段随版本变化）
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=~/data/gsm8k/train.parquet data.val_files=~/data/gsm8k/test.parquet \
  data.train_batch_size=64 data.max_prompt_length=512 data.max_response_length=1024 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.total_epochs=1 \
  trainer.logger='["console","wandb"]'
```

`rollout.gpu_memory_utilization=0.5` 是 colocated 模式下给 vLLM 的显存比例，剩下给训练态；OOM 时先调它。

### 2. 读调用链并打标

在 `ray_trainer.py` 的 fit 循环里，用 `time.perf_counter` 或 verl 自带的 timing 记录（日志里 `timing_s/gen`、`timing_s/update_actor` 等，字段以版本为准）画出一次 step 的时间分解饼图。

### 3. 同一实验在 TRL 上跑

```python
from trl import GRPOTrainer, GRPOConfig
cfg = GRPOConfig(output_dir="runs/trl-grpo", learning_rate=1e-6, num_generations=8,
                 max_completion_length=1024, per_device_train_batch_size=8,
                 use_vllm=True, vllm_mode="colocate", bf16=True, logging_steps=5)
trainer = GRPOTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", reward_funcs=[gsm8k_reward],
                      args=cfg, train_dataset=ds)
trainer.train()
```

比较两个框架每步时间与每步 rollout 数，理解 TRL 在单机上的便利与在多机上的局限。

### 4. 切换放置模式（有多卡时）

在 verl 里把 rollout 与 actor 放到不同资源池（disaggregated），对比 colocated 下每步时间与 GPU 利用率。观察 colocated 下训练阶段 rollout 卡的闲置。

## 实验

资源分层：实验 1、2 单卡；实验 3 需要 2 到 4 卡；没有条件时用日志分析替代。

### 实验 1：时间分解

单卡 0.5B GRPO 跑 20 步，记录每步 gen / reward / logprob / update 时间。把 `max_response_length` 从 1024 提到 4096 再跑，看 gen 占比的变化。预期：gen 占比从约 50% 升到 70% 以上。

### 实验 2：长尾对 gen 时间的影响

在实验 1 的日志里取每步 response 长度的 p50 与 max。画 gen 时间与 max 长度的相关性。预期：gen 时间跟 max 走而不是 p50，这是 Day 20 partial rollout 的动机。

### 实验 3：colocated vs disaggregated

2 卡：colocated（两卡都做训练和采样）vs disaggregated（1 卡训练 1 卡采样）。记录每步时间与两卡的平均利用率。预期：小模型下 colocated 更快（两卡一起采样），但利用率有周期性空洞。

### 实验 4：读源码

写出 verl 中 `DataProto` 在一步里字段的增长过程（进 generate 前有哪些字段，出来后多了哪些，reward 后、advantage 后又多了哪些），交一张表。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 启动即 OOM | vLLM 与训练态显存加起来超过卡 | 看启动日志中 vLLM KV blocks 与训练 peak | 降 `gpu_memory_utilization`；开 offload；减 micro batch |
| 第二步开始 OOM | 训练态与推理态切换时权重没释放 | 看 resharding 日志 | 升级版本；开 sleep mode（Day 19） |
| gen 时间远大于预期 | vLLM 用了 eager 或 TP 配置不当；长尾 | 看 vLLM 启动日志与长度分布 | 关 enforce_eager；调 TP；partial rollout |
| Ray worker 起不来 | 版本不匹配、端口、NCCL 环境 | ray status；worker 日志 | 固定版本矩阵；设 NCCL_SOCKET_IFNAME |
| 数据传输慢 | DataProto 走 pickle 传大 tensor | py-spy 看序列化时间 | 减少 driver 上的大对象；用框架推荐的传输路径 |
| reward 全 0 | reward_fn 的 ground_truth 字段名不对 | 打印一条 batch | 对齐数据预处理字段 |
| 多机训练 hang | 权重同步的 NCCL group 跨机没建好 | NCCL_DEBUG=INFO | 检查 IB 与拓扑（Day 29） |

## 验收标准

- 一张标注了每个组件显存与时间占比的系统图。
- verl 一次 step 的调用链表（函数、文件、输入输出字段）。
- 实验 1、2 的数据与"瓶颈在 rollout 长尾"的结论。
- 一张选型表：给定"7B dense 单机 / 70B dense 多机 / 200B MoE / agent 多轮长尾"四个场景，各选什么框架与后端并说明理由。

## 交付物

| 文件 | 内容 |
|---|---|
| `rl-system-notes.md` | 组件图、时间分解、verl 调用链、框架对比与选型表 |
| `verl-minimal-grpo.sh` | 跑通的命令与关键日志 |
| `timing-breakdown.csv` | 实验 1、2 数据 |

## 参考

- T42 HybridFlow：verl 的设计论文，单/多控制器与 hybrid engine。
- T43 OpenRLHF：Ray + vLLM 的早期形态。
- T44 AReaL：全异步系统的设计动机与解耦目标。
- T20 GLM-4.5 与 B07 slime：大 MoE 与 agentic 数据生成的框架需求。
- B06 verl 文档、B08 NeMo-RL、B12 SkyRL、B13 ROLL、B05 TRL：各框架的配置与版本矩阵。
