---
title: "Day 1：训练全景、流水线与 rollout 的三重含义"
type: concept
tags: [llm-training, landscape, post-training, rollout]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 1：训练全景、流水线与 rollout 的三重含义

> 课程入口 [[llm-training-30d/index]] · 下一课 [[llm-training-30d/week1/day02-training-loop-memory-math]]。相关：推理课 [[ai-infra-30d/week1/day01-inference-pipeline]]（推理链路），Agent 课 [[agent-rsi-30d/week1/day01-self-improvement-map]]（自改进分层）。

## 学习目标

1. 画出一张从预训练到 agentic RL 的完整训练流水线图，并说清每个阶段的输入、输出和"为什么要有这个阶段"。
2. 对比 Llama 3、Qwen3、DeepSeek-V3/R1、Kimi K2、GLM-4.5、Tulu 3 六个公开配方，能指出它们在阶段划分和算法选择上的共性与差异。
3. 准确区分 rollout 在 RL 训练、评测、数据合成三个语境里的含义，并能解释"训练系统里为什么要嵌一个推理引擎"。
4. 完成环境准备与自评，确定自己走 API-only、单卡还是多卡路线。

## 工业现状

2025 到 2026 年公开技术报告里的训练流水线已经收敛成一个相当一致的形态。写成一行：

```text
预训练（万亿 token，next-token）
  → 中期训练 / 退火（长上下文扩展、高质量数据升采样、领域数据）
  → 后训练
       SFT（冷启动 / 指令遵循）
       → 偏好优化（DPO 家族）或 RM + RL
       → RLVR（可验证奖励的 RL，数学 / 代码 / 逻辑）
       → agentic RL（多轮工具环境）
       → 安全与全场景 RL、蒸馏到小模型
  → 评测门禁 → 发布（量化、推理格式）
```

六个公开配方的对照（写"典型形态"，量级来自各报告，细节以原文为准）：

| 配方 | 预训练 | 中期训练 | 后训练主干 | RL 算法 | 特点 |
|---|---|---|---|---|---|
| Llama 3（T14） | 15T 级 token，dense | 长上下文退火到 128K | 多轮 SFT → 拒绝采样 → DPO，迭代 6 轮 | DPO 为主 | 大规模拒绝采样数据；不用在线 RL |
| Qwen3（T15） | 36T 级，dense 与 MoE | 长上下文、推理数据 | 长 CoT 冷启动 → 推理 RL → thinking 融合 → 通用 RL | GRPO 类 | thinking / non-thinking 一体化；蒸馏小模型 |
| DeepSeek-V3（T16） | 14.8T，MoE 671B，FP8 | 长上下文 | SFT → GRPO | GRPO | FP8 训练；aux-loss-free 负载均衡；MTP |
| DeepSeek-R1（T17） | 基于 V3 | — | R1-Zero 纯 RL → 冷启动 SFT → 推理 RL → 拒绝采样 SFT → 全场景 RL | GRPO | 证明纯 RL 可涌现长 CoT；蒸馏系列 |
| Kimi K2（T19） | 15.5T，MoE 1T | — | SFT → 大规模 agentic 数据合成 → 联合 RL（可验证 + 自评） | 策略梯度改进版 | 万级工具环境的合成 agent 数据；MuonClip 优化器 |
| GLM-4.5（T20） | 23T，MoE | 中期训练含 repo 级代码与长上下文 | 专家模型（推理 / agent / 通用）分别 RL → 自蒸馏合并 | GRPO 类 | expert iteration + 蒸馏合并 |
| Tulu 3（T22） | 用 Llama 3 底座 | — | SFT → DPO → RLVR | PPO / GRPO | 开源完整数据与配方；RLVR 命名来源 |

三个共同结论：

- **后训练已是主战场**。基座模型能力差距在缩小，能力差异主要来自后训练数据和 RL。
- **RLVR 是通用组件**。数学和代码这类有 verifier 的任务先用 RL 拉高推理能力，再把能力迁移到通用场景。
- **训练里嵌着推理**。拒绝采样、RL rollout、蒸馏、评测全部依赖高吞吐采样，所以 vLLM / SGLang 在训练集群里和训练框架一样重要。

## 核心原理

### 三段流水线各解决什么

| 阶段 | 目标函数 | 数据 | 解决的问题 | 不解决的问题 |
|---|---|---|---|---|
| 预训练 | next-token 交叉熵 | 网页、代码、书籍 | 知识与语言能力 | 不会按指令回答 |
| 中期训练 | 同上，数据分布切换 | 长文档、高质量子集 | 上下文长度、领域深度 | 行为对齐 |
| SFT | 只对 assistant token 的交叉熵 | 指令 - 回答对 | 格式、遵循指令 | 超越标注者的能力 |
| 偏好优化 | DPO 类或 RM + RL | 偏好对 | 风格、安全、细粒度取舍 | 可验证的正确性 |
| RLVR | 策略梯度 + verifier reward | 有标准答案的题 | 推理正确率 | 无法验证的任务 |
| agentic RL | 多轮策略梯度 | 环境 + 工具 | 长程任务完成 | 环境覆盖不到的场景 |

"必须理解"的一点：**SFT 的上限是数据，RL 的上限是 reward**。SFT 只能模仿分布里的行为，RL 能在 reward 信号下探索出数据里没有的行为，代价是 reward 一旦有漏洞就会被利用。Week 3 到 4 反复回到这一点。

### rollout 的三重含义

rollout 这个词来自经典 RL："让策略在环境里从头跑到结束，得到一条轨迹"。在 LLM 训练里它出现在三个地方，含义略有不同：

```text
1. RL 训练中的 rollout
   prompt ──策略模型采样──▶ n 条完整 response（或多轮轨迹）
   每条带：token ids、每个 token 的 logprob、reward、loss mask
   用途：计算 advantage 和策略梯度。on-policy 要求它来自当前策略。

2. 评测中的 rollout
   prompt ──模型采样──▶ k 条 response ──verifier──▶ pass@k / avg@k
   用途：估计能力，不回传梯度。采样温度和 k 决定方差。

3. 数据合成中的 rollout
   prompt ──模型采样──▶ 多条候选 ──RM / verifier 过滤──▶ SFT 或偏好数据
   用途：拒绝采样、蒸馏、偏好对构造。这是 Llama 3 后训练的主要数据来源。
```

三处的共同点是"用推理引擎批量采样"，差别在于采样后的数据去了哪里。所以一个 RL 训练系统本质上是"训练框架 + 推理引擎 + 数据流"，Day 18 到 20 讲这三者怎么拼在一起。打标任务里标注员看到的"rollout"属于第 3 类：采样出来的候选输出。

### 训练系统与推理系统的关系

```text
                 ┌────────────────────────────────┐
                 │ 训练框架（Megatron / FSDP）     │
                 │  参数 + 梯度 + 优化器状态       │
                 └────────┬───────────────▲───────┘
      权重同步（每步或每 N 步）│               │ rollout 数据（tokens、logprob、reward）
                 ┌────────▼───────────────┴───────┐
                 │ 推理引擎（vLLM / SGLang）       │
                 │  只存参数 + KV cache            │
                 └────────┬───────────────▲───────┘
                          │ response       │ reward
                 ┌────────▼───────────────┴───────┐
                 │ verifier / 环境 / RM            │
                 └────────────────────────────────┘
```

推理课讲的是右下角这个引擎怎么快；本课讲的是它如何被嵌进训练循环，以及为什么它的数值必须和训练框架一致（Day 19）。

### 一次 RL 训练 step 的时间线

把 Week 3 的内容先压成一张图，读技术报告时对照它定位每个术语：

```text
t0  同步权重：训练框架 → 推理引擎（Day 19）
t1  rollout：  推理引擎对 B 个 prompt 各采 G 条 response          ← 通常占总时间 60% 以上
t2  reward：   verifier / 沙箱 / RM 给每条 response 打分            ← 代码任务可能比 rollout 还慢
t3  advantage：按 group 归一化（GRPO）或 GAE（PPO）
t4  logprob：  用训练框架重算 old / ref logprob（或复用推理引擎的）
t5  update：   若干个 mini-batch 的策略梯度更新
t6  日志：     reward 均值、response 长度、entropy、KL、梯度范数、rollout 与 train 时间占比
```

异步系统（AReaL、Kimi 的做法）把 t1 到 t2 与 t4 到 t5 放在不同 GPU 上流水，代价是 rollout 数据来自旧策略，需要修正（Day 20）。

### 术语与量级

| 术语 | 含义 | 典型量级（示例） |
|---|---|---|
| global batch | 一次优化器更新看到的样本数（tokens 或 sequences） | 预训练 4M 到 16M tokens；SFT 64 到 512 条；RL 256 到 1024 个 prompt |
| micro batch | 单卡一次前向的样本数，受显存限制 | 1 到 8 条 |
| 梯度累积步数 | global / (micro × 数据并行度) | 由上两者算出 |
| step | 一次优化器更新 | SFT 数百到数千步；RL 数百步 |
| epoch | 遍历数据一次 | SFT 1 到 3；RL 通常按 step 计 |
| group size G | RL 中每个 prompt 的 rollout 条数 | 4 到 32 |
| KL 系数 β | 约束策略偏离 reference 的强度 | 0 到 0.01 量级，RLVR 常设 0 |

优化器：AdamW 仍是默认；Muon（Kimi K2 用的 MuonClip）在大规模预训练中开始出现，"知道即可"。学习率：预训练 1e-4 到 3e-4 量级带 cosine；SFT 1e-5 量级；RL 1e-6 量级。这些数字随模型规模变化，只用于建立数量级感觉。

### 六个配方的数据流差异（知道即可）

- **离线迭代型**（Llama 3、Tulu 3 的 DPO 阶段）：用当前模型大量采样，人或 RM 选偏好对，训练 DPO，再采样。每轮几天，可控性强。
- **在线 RL 型**（R1、Qwen3、GLM-4.5 的推理阶段）：每步采样立即用于更新。收益上限高，对基础设施要求高。
- **合成环境型**（Kimi K2 的 agentic 数据）：先构造万级工具与任务，用模型批量生成轨迹，过滤后 SFT，再 RL。数据规模由环境规模决定。

三种类型不互斥，同一模型的不同阶段常混用。

## 实现步骤

### 环境准备

```bash
# 建议独立环境，训练与推理依赖版本冲突常见
conda create -n llmtrain python=3.11 -y && conda activate llmtrain
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 以本机 CUDA 为准
pip install transformers datasets accelerate peft trl vllm             # 版本以各项目文档为准
pip install flash-attn --no-build-isolation                           # 需要匹配的 CUDA / torch
# RL 框架 Week 3 再装：verl 建议用其官方 Docker 镜像
nvidia-smi && python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

### 资源分层与路线选择

| 资源 | 能做什么 | 本课如何调整 |
|---|---|---|
| API-only（无 GPU） | 数据工程、评测、reward 设计、读框架源码 | 训练实验用 0.5B 模型在 Colab / 租卡上做，或用 TRL 的 CPU 小例子跑通逻辑 |
| 单卡 24 到 48 GB | 0.5B 到 8B 的 LoRA SFT / DPO；1.5B 全参 GRPO（短 response） | P0 用 0.5B 到 1.5B；P2 用 TRL GRPOTrainer 或 verl 单卡 colocated |
| 4 到 8 卡 | 7B 全参 SFT、7B GRPO、多轮 agentic RL | 按大纲原样执行，可加并行度对比 |

### 自评清单

回答以下问题，不会的就是本课要补的：

1. AdamW 训练一个 7B 模型，参数、梯度、优化器状态各占多少显存？（Day 2）
2. SFT 为什么只对 assistant token 算 loss？packing 时如何避免样本间互相看见？（Day 3）
3. ZeRO-3 和 FSDP 切的是什么？TP 每层通信几次？（Day 4 到 5）
4. DPO 的 loss 里为什么有一个 reference model？（Day 11）
5. GRPO 相比 PPO 去掉了什么？为什么能去掉？（Day 16）
6. RL 训练里 rollout 引擎和训练框架的数值不一致会造成什么？（Day 19）
7. 多轮 agent 训练里工具返回的 token 要不要进 loss？（Day 22）

## 实验

### 实验 1：读三份技术报告的后训练章节，填对照表

固定变量：只读 T14（Llama 3 第 4 节）、T17（R1 第 2 节）、T22（Tulu 3 第 3 到 6 节）的后训练部分。产出一张表：阶段顺序、每阶段数据来源与量级、算法、评测集、作者声称的每阶段收益。预期：三者都有"采样 → 过滤 → 再训练"的循环，差别在 RL 是否在线。

### 实验 2：用推理引擎做一次"评测 rollout"

单卡即可。用 vLLM 对 Qwen2.5-1.5B-Instruct 在 GSM8K 前 200 题做 k=8 采样（温度 0.6 到 1.0），写一个答案抽取与等价判断函数，计算 pass@1、pass@8、avg@8。记录：采样吞吐（tokens/s）、每题平均 response 长度、温度对 pass@8 的影响。这段代码 Day 17 会复用为 verifier。

```python
from vllm import LLM, SamplingParams
llm = LLM("Qwen/Qwen2.5-1.5B-Instruct", gpu_memory_utilization=0.8)
sp = SamplingParams(n=8, temperature=0.8, top_p=1.0, max_tokens=1024, seed=0)
outs = llm.generate(prompts, sp)          # prompts 用 chat template 渲染
# 每题 8 条 rollout；抽取 \boxed{} 或最后一个数字与标准答案比对
```

### 实验 3：估算一次 RL step 的 rollout 成本

假设 batch 512 个 prompt、group size 8、平均 response 800 token、7B 模型在单张 H100 上 decode 吞吐约 2000 tokens/s（示例数字）。算出一次 rollout 需要多少 GPU 秒，再与一次训练 step 的 6ND 算力做对比。预期：rollout 时间是训练时间的数倍，这就是 Day 19 到 20 存在的理由。

### 实验 4：把一份 SFT 数据集里的"rollout 痕迹"找出来

下载一个开源后训练数据集（例如 Tulu 3 的 SFT mixture 或任一 distilled 推理数据集），随机抽 50 条，判断每条是人工写的、模型采样后过滤的、还是教师模型蒸馏的。依据：格式一致性、思维链风格、是否带 `<think>` 标签、错误类型。写 20 行结论：这个数据集里模型生成占比大约多少，过滤信号可能是什么。预期：现代 SFT 数据里模型生成的比例远高于人工，这决定了 Day 10 数据工程的重点。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| 环境装好后 flash-attn 报错 | torch / CUDA 版本不匹配 | `python -c "import flash_attn"` 看报错 | 用 flash-attn 的预编译 wheel 对应版本，或先用 SDPA |
| vLLM 与训练框架的 torch 版本冲突 | 两者钉死不同 torch | pip check | 分环境，或用 verl 官方镜像 |
| 读技术报告时分不清阶段 | 各家命名不同（SFT / 冷启动 / 拒绝采样微调） | 按"目标函数 + 数据来源"归类 | 用本页的六阶段表重新标注 |
| 实验 2 的 pass@8 远低于报告 | prompt 没用 chat template、答案抽取不鲁棒、max_tokens 截断 | 打印几条完整 response 看结尾 | 用 tokenizer.apply_chat_template；抽取 \boxed 与最后数字双路径 |
| 采样吞吐很低 | 没开 prefix caching、n=8 未用同一 prompt 共享 prefill | 看 vLLM 日志的 prefix cache 命中 | 同一 prompt 的 n 条 rollout 天然共享前缀，检查 `SamplingParams(n=8)` 而不是复制 prompt 8 次 |

## 思考题

1. 如果一个团队只有 SFT 数据、没有 verifier，也没有人力标偏好，它能做 RL 吗？用什么当 reward？（提示：generative RM、rubric、self-consistency）
2. R1-Zero 跳过 SFT 直接 RL 成功了，为什么 R1 最终还是加回了冷启动 SFT？
3. 拒绝采样微调（RFT）和 RL 都用 rollout，一个是 off-policy 的监督学习，一个是 on-policy 的策略梯度。什么情况下前者够用？
4. 训练集群里推理引擎的吞吐决定了 RL 的迭代速度。如果推理课学到的 prefix caching、CUDA graph 在 rollout 阶段开着，会不会影响 on-policy 的正确性？（Day 19 给答案）
5. 打标平台让标注员给"每个 prompt 的 4 条 rollout 排序"，这份数据能直接训 DPO 吗？还需要什么处理？

## 验收标准

- 能在白板上画出六阶段流水线，并对每个阶段说出一个公开配方里的具体做法。
- 能用一句话区分三种 rollout，并解释打标任务里"标注 rollout"标的是什么。
- 实验 2 的脚本能跑通，pass@k 与吞吐有记录。
- 完成自评清单，标出自己缺的部分。

## 交付物

| 文件 | 内容 |
|---|---|
| `training-pipeline-map.md` | 六阶段流水线图 + 六个配方对照表 |
| `rollout-glossary.md` | 三种 rollout 的定义、数据结构、用途 |
| `eval_rollout.py` + `gsm8k-passk.md` | 实验 2 脚本与结果 |
| `self-assessment.md` | 自评清单答案与路线选择 |

## 参考

- T14、T15、T16、T17、T19、T20、T22：六个公开配方的技术报告，本页对照表的来源。
- T23：InstructGPT，SFT → RM → PPO 三段式的起点，理解为什么后来演化成 DPO 与 RLVR。
- T12：Chinchilla，理解预训练阶段的算力与数据配比。
- B06：verl 文档，先看架构页，Day 18 精读。
