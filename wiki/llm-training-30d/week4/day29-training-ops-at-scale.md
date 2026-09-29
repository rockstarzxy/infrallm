---
title: "Day 29：大规模训练运维：容错、checkpoint、观测、成本与数据飞轮"
type: concept
tags: [llm-training, training-ops, fault-tolerance, checkpoint, observability, data-flywheel]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 29：大规模训练运维：容错、checkpoint、观测、成本与数据飞轮

> 上一课 [[llm-training-30d/week4/day28-safety-alignment-rlhf-production]] · 下一课 [[llm-training-30d/week4/day30-p3-capstone-agentic-rl]]。相关：推理课的生产部署 [[ai-infra-30d/week3/day20-production-deploy]] 与冷启动专项 [[ai-infra-30d/optimization/14-cold-start-autoscaling]]（训练产出的 checkpoint 最终要变成推理侧能加载的权重）；Week 1 的 checkpoint 基础 [[llm-training-30d/week1/day06-frameworks-efficiency-profiling]]。

## 学习目标

1. 能列出千卡级训练中的主要故障类型、检测信号和恢复动作，并算出 checkpoint 频率与故障率之间的最优点。
2. 能设计训练与 RL 任务的观测面板：哪些指标必须采、各自的告警阈值怎么定。
3. 能把 GPU 小时拆到预训练、SFT、rollout、训练更新、评测五项，并识别 RL 任务里 rollout 占比过高的原因。
4. 能设计从线上数据回流到周期性再训练的数据飞轮，并写出发布门禁。
5. 能把分布式 checkpoint 转换成推理引擎可加载的格式并验证数值一致。

## 工业现状

Llama 3（T14）报告了大规模训练期间的中断统计：多数中断来自硬件（GPU、HBM、网络），团队通过自动检测、快速 checkpoint 与重启把有效训练时间维持在 90% 以上（具体数字以报告为准）。DeepSeek-V3（T16）强调了训练框架的稳定性与 FP8 的数值验证。RL 系统方面，Kimi K2（T19）描述了训练与推理引擎的共置和快速参数广播；GLM-4.5 的 slime（T20、B07）与 AReaL（T44）都把异步 rollout 的容错（单条轨迹失败不影响批次）作为设计目标。共识：**大规模训练的主要工程量在容错与观测，而不是模型代码**。

## 核心原理

### 故障类型与检测

| 故障 | 典型信号 | 检测方式 | 恢复 |
|---|---|---|---|
| GPU 硬件错误（Xid、ECC） | 进程崩溃或 NCCL 超时 | 节点健康检查、dmesg/DCGM | 隔离节点，从 checkpoint 重启 |
| 慢节点（straggler） | 某 rank 的 step 时间持续高于中位数 | 每 rank 的 step 计时分布 | 替换节点；短期用 timeout 容忍 |
| NCCL 超时 / 网络抖动 | `NCCL timeout` 日志、allreduce 耗时尖峰 | 通信耗时监控、IB 错误计数 | 重启通信组；检查网卡与交换机 |
| 存储慢 | 数据加载或 checkpoint 写入耗时突增 | dataloader 等待时间、写入吞吐 | 本地缓存、异步写 |
| loss 异常（NaN、spike） | loss 或梯度范数突变 | 梯度范数与 loss 的滑动窗口 | 回退到上一 checkpoint 并跳过该批数据 |
| RL 环境失败 | 沙箱超时率、reward 服务 5xx | reward 服务指标 | 单条轨迹记失败并重采，不阻塞批次 |

### checkpoint 频率的选择

设故障平均间隔 MTBF，checkpoint 写入耗时 c，间隔 T，则期望浪费时间约为 `T/2 + c`（每次故障平均丢失半个间隔，加每次保存的开销分摊）。最优间隔近似 `T* ≈ sqrt(2 · c · MTBF)`（示例：c = 2 分钟、MTBF = 24 小时 → T* ≈ 1.4 小时）。异步 checkpoint 把 c 压到近零，间隔可以更短。

必须理解：checkpoint 不只是权重，还包括优化器状态（AdamW 是权重的两倍）、数据加载器位置、RNG 状态、学习率调度状态；RL 还要包括 rollout 引擎的版本号和异步队列里未消费的轨迹。

### 分布式 checkpoint 与格式转换

```text
训练布局（FSDP2 分片 / Megatron TP×PP 分片）
  → 分布式 checkpoint（每 rank 写自己的分片 + 元数据，PyTorch DCP 或 Megatron dist-ckpt）
  → 合并/重分片（resharding）：改变并行度后继续训练
  → 导出：合并成 HF safetensors 格式 → 推理引擎加载（vLLM/SGLang）
  → 数值验证：同一 prompt 在训练侧与推理侧的 logits / logprob 对比
```

FP8 或 BF16 训练的主权重通常保留 FP32 master copy，导出时转 BF16；量化（Day 10 推理课）在导出后另做。

### 观测面板

| 类别 | 指标 | 告警思路 |
|---|---|---|
| 训练健康 | loss、梯度范数、lr、有效 batch token 数 | 梯度范数超过历史 P99 的 3 倍；loss 不降超过 N 步 |
| 吞吐 | tokens/s/GPU、MFU、step 时间分布（按 rank） | MFU 低于基线 20%；单 rank 显著慢 |
| 通信 | allreduce / all-to-all 耗时占比 | 占比突增 |
| 资源 | 显存峰值、GPU 温度与功耗、CPU 与内存 | 显存接近上限；温度异常 |
| RL 专项 | rollout 时间占比、reward 均值与方差、entropy、KL、response 长度分布、零梯度 group 比例、train/infer logprob 差 | entropy 骤降；KL 超阈；长度撞上限比例升高 |
| 环境 | 沙箱超时率、工具错误率、reward 服务 P99 | 超时率超阈 |
| 数据 | 每类数据的消费量、重复率、被跳过的批次 | 某类数据耗尽 |

RL 任务的观测比预训练多一倍指标，Day 20 的稳定性面板是这里的子集。

### 成本拆分

```text
总 GPU 小时 = 预训练 + 中期训练 + SFT + RL(rollout + 训练更新 + 评测) + 失败重跑
RL 阶段：rollout 常占 60% 到 80%（示例）
  原因：长尾样本、推理引擎在共置模式下与训练串行、沙箱等待
  降本：异步 rollout、partial rollout、缩短 max_tokens、推理侧优化（推理课全部内容）
评测：avg@n 的 n 与评测集大小直接乘进成本，按阶段分级（快速集每 N 步，全量集里程碑）
```

### 数据飞轮与持续后训练

```text
线上流量 → 脱敏与合规过滤 → 采样与去重 → 标注（Day 28）或自动 verifier
  → 新数据进入 SFT / 偏好 / RL 数据池（带版本）
  → 周期性再训练（示例：双周）从上一发布模型出发
  → 回归评测（能力、安全、格式、延迟）
  → 发布门禁通过 → 量化与导出 → 推理侧灰度（推理课 Day 20）
```

要点：回流数据的分布会随产品变化，每轮记录数据构成；再训练从发布模型出发要防止"遗忘"，用固定回归集把关；每轮保留可复现的配置快照。

### 发布门禁（示例）

| 门 | 标准 |
|---|---|
| 能力 | 关键评测不低于上一版 − 显著性阈值 |
| 安全 | 越狱率不高于上一版；over-refusal 不高于阈值 |
| 格式与工具 | 工具调用格式正确率、JSON 合法率 |
| 数值一致 | 训练侧与推理侧 logprob 差在阈值内 |
| 延迟与成本 | 平均输出长度未显著增长（影响推理成本） |
| 可复现 | 配置、数据版本、seed、checkpoint 哈希齐全 |

## 实现步骤

1. **健康检查脚本**：训练前对每个节点跑 NCCL all-reduce 带宽测试与 GPU 自检；训练中每 rank 上报 step 时间。
2. **checkpoint**：配置异步分布式 checkpoint（TorchTitan / Megatron 均支持，以当前版本文档为准）；间隔按上面的公式；保留最近 k 个和每个里程碑。
3. **自动重启**：作业调度器（Slurm / K8s）配置失败重试；启动脚本自动定位最新完整 checkpoint（写入完成标记文件后才算完整）。
4. **观测**：把训练指标与 RL 指标统一进一个面板（W&B / Prometheus + Grafana）；告警规则按上表。
5. **导出与验证**：写导出脚本把分布式 checkpoint 合并为 safetensors，用 vLLM 加载，对 50 条 prompt 比较 logprob 差。
6. **成本报表**：从调度器日志汇总 GPU 小时，按阶段与任务打标签。
7. **飞轮**：定义回流数据的 schema 与版本号，建立回归评测集与门禁清单。

### 关键脚本骨架

**导出与数值验证**

```python
# export_and_verify.py（骨架；DCP / Megatron 的 API 以当前版本为准）
import torch, json
from torch.distributed.checkpoint import state_dict_loader   # 示例入口
from safetensors.torch import save_file

def export_to_hf(dcp_dir, out_dir, name_map):
    sd = load_full_state_dict(dcp_dir)                       # 合并分片，得到完整权重
    hf_sd = {name_map(k): v.to(torch.bfloat16).contiguous() for k, v in sd.items()
             if not k.startswith("optimizer.")}
    # tied embedding、qkv 合并/拆分、RoPE inv_freq 等按模型架构处理
    save_file(hf_sd, f"{out_dir}/model.safetensors", metadata={"format": "pt"})

def verify(train_model, out_dir, prompts, tol=1e-2):
    from vllm import LLM, SamplingParams
    llm = LLM(out_dir, dtype="bfloat16", enforce_eager=True)   # 先关 CUDA graph 排除干扰
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0)
    diffs = []
    for p in prompts:
        infer_lp = llm.generate([p], sp)[0].prompt_logprobs     # 推理侧每 token logprob
        train_lp = train_side_logprobs(train_model, p)          # 训练侧同一 prompt
        diffs.append(max_abs_diff(infer_lp, train_lp))
    print(json.dumps(dict(mean=sum(diffs)/len(diffs), max=max(diffs), tol=tol)))
    return max(diffs) < tol
```

差异量级的经验（示例）：同 dtype、同 kernel 家族时逐 token logprob 差在 1e-3 到 1e-2；开 FP8 KV 或量化后会到 1e-1 量级，这时 RL 必须开 IS 修正（Day 19）。

**checkpoint 完成标记与自动定位**

```bash
# 保存端：先写分片，再写完成标记
save_ckpt "$CKPT_DIR/step_$STEP" && touch "$CKPT_DIR/step_$STEP/.complete"
# 启动端：只认带标记的最新目录
LATEST=$(ls -d $CKPT_DIR/step_* | while read d; do [ -f "$d/.complete" ] && echo "$d"; done | sort -V | tail -1)
```

**Slurm 自动重试（示例）**

```bash
#SBATCH --requeue
#SBATCH --signal=B:USR1@120        # 抢占前 120 s 发信号，触发一次紧急 checkpoint
trap 'save_ckpt_now; scontrol requeue $SLURM_JOB_ID' USR1
srun python train.py --resume "$LATEST" &
wait
```

K8s 用 Job 的 `backoffLimit` 加 PyTorchJob / 类似 operator 的重启策略，原则相同：进程可被随时杀、启动时自动找最新完整 checkpoint。

### checkpoint 策略计算示例

| 集群 | MTBF（示例） | 同步保存耗时 c | 最优间隔 T* | 每次故障期望损失 | 异步保存后 |
|---|---|---|---|---|---|
| 64 卡 | 72 h | 1 min | ≈ 1.5 h | ≈ 46 min | 间隔可缩到 15 到 30 min |
| 256 卡 | 12 h | 3 min | ≈ 1.1 h | ≈ 36 min | 同上 |
| 1024 卡 | 4 h | 5 min | ≈ 0.8 h | ≈ 30 min | 必须异步，否则保存开销占比过高 |

MTBF 随卡数近似反比下降，这是千卡训练必须做异步 checkpoint 与快速重启的原因。

### RL 任务恢复的额外状态

同步 RL 恢复相对简单；异步 RL 恢复要处理：

- rollout 引擎持有的权重版本号（恢复后引擎必须重新加载 checkpoint 对应版本，否则轨迹 staleness 失控）。
- 队列里未消费的轨迹：丢弃（简单、安全）或按版本差重新计算 IS 权重后保留。
- 环境池状态：沙箱快照可复用，进行中的任务作废。
- reward 缓存：按 (task, trajectory hash) 键的缓存可跨重启复用。

恢复后前几十步要对比 reward、entropy、长度分布与故障前是否连续，不连续说明某个状态没恢复对。

### 面试题（5 题，附要点）

1. **checkpoint 间隔怎么定？** `T* ≈ sqrt(2·c·MTBF)`；异步保存把 c 压到近零，间隔按丢失容忍度定。
2. **恢复后 loss 跳变最常见的原因？** 优化器状态、lr 调度或 dataloader 位置未恢复；用恢复前后的 lr 与样本 ID 序列确认。
3. **RL 里 rollout 占比为什么高、怎么降？** 长尾样本与串行共置；异步、partial rollout、缩预算、推理侧优化。
4. **导出模型如何证明和训练模型是同一个？** 同 prompt 的逐 token logprob 对比；差异量级与 dtype、kernel 相关。
5. **发布门禁为什么要看平均输出长度？** 长度直接乘进推理成本与延迟，RL 后长度上涨是常见的隐性成本。

## 实验

资源分层：单卡可做实验 1、3；4 到 8 卡做实验 2。

### 实验 1：checkpoint 完整性与 resume 一致性

训练 200 步保存 checkpoint，从中恢复再训 50 步，与不间断训练 250 步比较 loss 曲线与最终权重差（应在数值噪声内）。故意在保存中途杀进程，验证"完成标记"机制能识别不完整 checkpoint。

### 实验 2：故障注入

多卡训练中随机杀掉一个 rank 的进程，测量从故障到恢复训练的时间，拆成检测、调度、加载三段。再模拟慢节点（对某 rank 加 sleep），看 step 时间分布能否定位。

### 实验 3：导出与数值一致性

把 Week 3 的 RL checkpoint 导出为 safetensors，用 vLLM 加载，比较训练侧与推理侧 50 条 prompt 的 logprob 差；再开启 FP8 KV 或量化重测，记录差异量级。这也是 Day 19 train/infer mismatch 的度量方法。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 恢复后 loss 跳变 | 优化器状态或 lr 调度未恢复 | 对比恢复前后 lr 与优化器统计量 | checkpoint 包含全部状态 |
| 恢复后数据重复 | dataloader 位置未保存 | 检查样本 ID 序列 | 保存 sampler 状态 |
| 一个 rank 慢拖全局 | 硬件降频、网卡问题 | per-rank step 时间 | 替换节点 |
| 导出模型推理结果异常 | 权重名映射错、tied embedding、RoPE 配置丢失 | logprob 对比、抽样生成 | 修导出脚本；对照 HF config |
| RL 训练吞吐远低于预期 | rollout 长尾、沙箱等待 | rollout 时间占比、最长样本时间 | 异步、partial rollout、缩预算 |
| 再训练后旧能力退化 | 回流数据分布偏移 | 回归集曲线 | 混入固定比例旧数据；门禁拦截 |
| 成本超预算 | 评测过频、失败重跑多 | 成本报表按阶段 | 分级评测；提高 checkpoint 频率 |

## 验收标准

- 能给出一个假设集群（示例 256 卡、MTBF 12 小时、checkpoint 3 分钟）的 checkpoint 策略与预期有效训练时间。
- 实验 1 与 3 完成，导出模型在推理引擎上的 logprob 差在可解释范围内。
- 观测面板与告警规则文档完成；发布门禁清单完成。

## 交付物

| 文件 | 内容 |
|---|---|
| `training-ops-runbook.md` | 故障类型表、检测与恢复步骤、checkpoint 策略计算 |
| `export_and_verify.py` | 分布式 checkpoint → safetensors → 推理侧 logprob 对比 |
| `observability.md` + `release-gates.md` | 面板指标、告警阈值、发布门禁 |
| `cost-report.md` | 本课程所有实验的 GPU 小时按阶段拆分 |

## 参考

- T14 Llama 3：大规模训练中断统计与容错实践的公开描述。
- T16 DeepSeek-V3：训练框架稳定性与 FP8 数值验证。
- T09 TorchTitan、B09 Megatron-Core：分布式 checkpoint 与并行配置。
- T44 AReaL、B07 slime、T19 Kimi K2：RL 系统的异步容错与参数同步。
- B06 verl：RL 任务的 checkpoint 与恢复接口。
