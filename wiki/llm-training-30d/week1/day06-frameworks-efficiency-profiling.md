---
title: "Day 6：训练框架选型、效率技术、profiling 与 checkpoint"
type: concept
tags: [llm-training, frameworks, fp8, fused-kernels, torch-compile, profiling, checkpoint]
sources: [2026-09-29_llm-training-course-references.md]
created: 2026-09-29
updated: 2026-09-29
---

# Day 6：训练框架选型、效率技术、profiling 与 checkpoint

> 上一课 [[llm-training-30d/week1/day05-model-parallel-tp-pp-cp-ep]] · 下一课 [[llm-training-30d/week1/day07-p0-project-fsdp2-training]]。相关：推理课 [[ai-infra-30d/week1/day05-profiling]]（nsys 与 torch profiler 用法相同）、[[ai-infra-30d/week2/day13-flash-attention]]（FlashAttention 与 fused kernel 在推理侧的形态）。

## 学习目标

1. 按模型规模、并行需求、生态和团队情况选出训练框架，并说出每个选项的硬边界。
2. 说清 FlashAttention、fused kernel、torch.compile、FP8 训练、selective recompute、通信重叠各自消除的是哪类开销。
3. 用 torch profiler 与 nsys 把一个训练 step 分解到 kernel 级，找出前三个非 GEMM 开销。
4. 设计 checkpoint 策略（频率、异步、格式、resume 一致性、导出到推理格式）。

## 工业现状

| 框架 | 定位 | 并行 | 谁在用（公开） | 硬边界 |
|---|---|---|---|---|
| Megatron-Core / NeMo | 最大规模预训练、MoE | TP/PP/CP/EP/DP 全套，FP8（Transformer Engine） | NVIDIA、多家国内厂商的预训练；slime 与 verl 的 Megatron 后端 | 模型定义要按 Megatron 方式写，HF 模型需转换 |
| TorchTitan | PyTorch 原生参考实现 | FSDP2/TP/PP/CP，float8 | Meta 内部路线，学习首选 | 模型种类少 |
| FSDP2 原生 + HF 模型 | 后训练主力 | FSDP2（+ TP 实验性） | TRL、verl FSDP worker、Tulu 3、大量 SFT/RL | 单层放不进单卡时无解（需 TP） |
| DeepSpeed | HF 生态的 ZeRO | ZeRO-1/2/3、offload、Ulysses | HF Trainer 用户、OpenRLHF | 与 Megatron 的 TP/PP 组合较弱 |
| HF Trainer / Accelerate | 快速 SFT/DPO | 通过 FSDP 或 DeepSpeed | 大多数中小团队 | 灵活性低，RL 需 TRL |
| Unsloth | 单卡/少卡 LoRA 加速 | 单卡为主 | 个人与小团队 | 多卡支持有限 |

后训练的主流是"FSDP2 + HF 模型代码"，因为改动最小、与推理引擎权重格式一致；超过 70B 或 MoE 上 Megatron。RL 框架的后端选择直接继承这个格局（Day 18）。

效率技术在公开配方里的状态：FlashAttention 与 fused kernel 是标配；torch.compile 在 TorchTitan 和 TRL 里可开，收益 5% 到 20%；FP8 训练 DeepSeek-V3 已在生产，Llama 3 未用，Blackwell 上会更普遍；selective recompute 是预训练标配。

## 核心原理

### 开销分类与对应技术

| 开销类型 | 表现 | 技术 | 原理 |
|---|---|---|---|
| attention 的 HBM 读写 | 长序列时 attention 占比高 | FlashAttention（T05） | tiling + online softmax，不物化 s² 矩阵 |
| 逐元素 kernel 多 | profiler 里大量小 kernel，GPU 空隙 | fused kernel（RMSNorm、SwiGLU、RoPE、CE）、torch.compile | 合并访存，一次读写完成多个算子 |
| 交叉熵 logits 显存与带宽 | 大词表 OOM | chunked / fused CE（Liger、compile） | 分块计算不物化完整 logits |
| GEMM 本身 | 已是 Tensor Core | FP8（T10、T16） | 半字节存储与计算，吞吐翻倍，需缩放与精度保护 |
| activation 显存 | 长序列 OOM | selective recompute（T11） | 重算便宜的算子 |
| 通信等待 | NCCL 与计算串行 | 重叠（FSDP prefetch、Megatron overlap 选项、DualPipe） | 通信放独立 stream，与下一层计算并行 |
| 优化器 step | memory-bound | fused AdamW、分布式优化器 | 单 kernel 更新；分片到多卡 |
| 数据加载 | GPU 等 CPU | 预 tokenize、多进程、预取 | 让 loader 领先于计算 |

### FP8 训练怎么保持精度

BF16 有 8 位指数，FP8 的 E4M3 只有 4 位指数、动态范围小得多。做法：

- **缩放**：每个 tensor（或每块）乘一个缩放因子把数值移到 FP8 可表示范围，反量化时除回。Transformer Engine 用"延迟缩放"（用历史 amax 估计），DeepSeek-V3 用细粒度 block-wise（128×128 权重块、1×128 activation 块）缩放，减少离群值影响。
- **保留高精度的部分**：主权重、梯度累积、优化器状态、norm、softmax、残差相加仍是 BF16/FP32；只有 GEMM 的输入输出用 FP8。
- **累加精度**：FP8 GEMM 在 Tensor Core 内部累加精度有限，DeepSeek-V3 每 128 次累加提升到 FP32 寄存器一次。

"知道即可"：FP8 在小模型、短训练上收益不明显且有精度风险，先在 GEMM 占比高、规模大的场景用。

### torch.compile 在训练里的边界

- 收益来自算子融合与减少 Python 开销；对 attention 无效（已经是单 kernel）。
- 动态形状（packing 导致每步 cu_seqlens 不同）会触发重编译，要用 `dynamic=True` 或固定桶。
- 与 FSDP2 组合：先 compile 每个 Transformer 层再 `fully_shard`（TorchTitan 的做法）。
- 首次编译几分钟，多机要保证编译缓存一致。

### checkpoint 策略

| 决策 | 选项 | 考虑 |
|---|---|---|
| 频率 | 按步数或按时间（如每 30 分钟） | 故障平均间隔、写入耗时（几百 GB 到 TB 级） |
| 同步 vs 异步 | DCP 异步保存 | 异步把 GPU→CPU 拷贝后由后台线程写盘，训练只停顿几秒 |
| 内容 | 模型、优化器、lr scheduler、数据加载器状态、RNG | 缺任何一项 resume 后曲线不重合 |
| 格式 | DCP（并行度无关）→ HF safetensors（推理） | RL 每步都要做"训练格式 → 推理格式"的内存版转换 |
| 存储 | 本地 NVMe 暂存 + 对象存储 | 写本地再异步上传，避免网络抖动阻塞 |

resume 一致性检查：从 checkpoint 恢复后前 10 步 loss 与原训练日志逐步对比，允许 BF16 级别误差。

## 实现步骤

### 1. 打开效率开关（HF + FSDP2 路线）

```python
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2")
# Liger fused kernels（RMSNorm、SwiGLU、RoPE、fused CE）
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
apply_liger_kernel_to_qwen2()
# compile 每层
for i, layer in enumerate(model.model.layers):
    model.model.layers[i] = torch.compile(layer, dynamic=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
```

### 2. TorchTitan 的 float8 与 compile

```toml
[training]
compile = true
[float8]
enable_fsdp_float8_all_gather = true
enable_float8_linear = true       # 需要 torchao，Hopper 以上
```

### 3. torch profiler 抓训练 step

```python
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler
prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
               schedule=schedule(wait=2, warmup=2, active=3),
               on_trace_ready=tensorboard_trace_handler("./prof"), record_shapes=True, with_stack=True)
prof.start()
for step in range(8):
    train_step(); prof.step()
prof.stop()
# 用 tensorboard 或 perfetto 打开 ./prof 下的 trace；看 GPU 时间线的空隙与 kernel 分布
```

### 4. nsys（多卡通信问题首选）

```bash
nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o train_rank0 torchrun --nproc_per_node 4 train.py   # 代码里用 torch.cuda.cudart().cudaProfilerStart()
nsys stats --report cuda_gpu_kern_sum train_rank0.nsys-rep | head -30
```

看 `ncclKernel_*` 与计算 kernel 是否在时间线上重叠。

### 5. 异步 DCP checkpoint

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict
model_sd, optim_sd = get_state_dict(model, opt)
future = dcp.async_save({"model": model_sd, "optim": optim_sd, "step": step,
                         "sched": sched.state_dict(), "rng": torch.get_rng_state()},
                        checkpoint_id=f"ckpt/step{step}")
# 下一次保存前 future.result()，确保上一次写完
```

导出到 HF：`get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))` 在 rank 0 得到完整权重后 `model.save_pretrained`（需要未分片的模型实例或直接写 safetensors）。

## 实验

### 实验 1：效率技术逐项叠加

固定：Qwen2.5-1.5B，4 卡 FSDP2，s=4096，100 步。依次叠加：baseline（SDPA）→ FlashAttention → Liger fused → compile → fused AdamW → selective recompute 关闭（显存允许时）。记录 tokens/s、显存、MFU。预期每项 5% 到 30%，合计可能翻倍；把哪一项收益最大写进结论（通常是 fused CE 与 FlashAttention）。

### 实验 2：profiler 分解

对 baseline 与全开配置各抓一个 step，按 kernel 类别汇总时间：GEMM、attention、norm/逐元素、CE、NCCL、优化器、空闲。画两张饼图。预期全开后 GEMM 占比显著上升。

### 实验 3：FP8（Hopper 以上）

TorchTitan 开 float8 对比 BF16：吞吐、loss 曲线 500 步的差异。预期小模型吞吐提升有限（GEMM 占比低），loss 差异在噪声内。

### 实验 4：checkpoint resume 一致性

训练 100 步存 checkpoint，继续训 20 步记录 loss；从 checkpoint 恢复再训 20 步。两条曲线逐步对比。故意不存 RNG 或数据加载器状态，观察差异出现在哪一步。

## 常见失败与诊断

| 症状 | 可能原因 | 确认方法 | 修法 |
|---|---|---|---|
| compile 后每步都在重编译 | 动态形状（packing、变长 batch） | `TORCH_LOGS=recompiles` | `dynamic=True`；固定长度桶；`torch._dynamo.config.cache_size_limit` 调大 |
| Liger / fused kernel 后 loss 变化 | kernel 数值实现差异或版本 bug | 与 baseline 前 50 步逐步对比 | 只开经过验证的 kernel；升级版本 |
| FP8 loss 发散 | 缩放因子失效、离群值 | 监控 amax 与溢出计数 | 延迟缩放窗口调整；block-wise；关键层退回 BF16 |
| profiler 显示 GPU 空闲 20% 以上 | 数据加载、Python 开销、同步点（`.item()`、日志） | trace 里找 CPU 段 | 去掉每步 `.item()`；异步日志；预取 |
| checkpoint 写入让训练停顿几分钟 | 同步写、单节点写盘 | 计时 | 异步 DCP；各 rank 并行写；本地 NVMe |
| resume 后 loss 跳变 | 缺优化器/scheduler/RNG/数据位置 | 实验 4 | 存全状态；数据加载器支持 seek |
| 多机编译缓存不一致导致 hang | 部分 rank 在编译 | 日志时间戳 | 共享编译缓存目录或先单机预热 |

## 思考题

1. FlashAttention 在训练里除了省显存还省时间，在推理 decode 里为什么主要靠 FlashDecoding 而不是 FA 本身？
2. fused CE 把 logits 分块计算，对 RL 训练里需要完整 logprob 的场景有什么影响？（提示：只需要采样 token 的 logprob，仍可分块）
3. 如果 checkpoint 每 30 分钟一次、写入 5 分钟、故障平均每 6 小时一次，训练有效时间比例是多少？异步保存后呢？
4. RL 训练每步都要把权重给推理引擎，这个"checkpoint"应该走内存还是走盘？两者各在什么规模下合理？

## 验收标准

- 能为三个场景（单卡 LoRA SFT、8 卡 7B 全参 SFT、64 卡 MoE 预训练）各给出框架选择与理由。
- 实验 1 的叠加表完成，能指出每项技术消除的开销类型。
- 能用 profiler 找出并解释前三个非 GEMM 开销。
- 有一个通过一致性验证的异步 checkpoint 实现。

## 交付物

| 文件 | 内容 |
|---|---|
| `framework-selection.md` | 选型矩阵与三个场景的决定 |
| `efficiency-stack.md` | 实验 1、2 的数据与饼图 |
| `checkpoint.py` | 异步 DCP 保存、加载、HF 导出 |
| `resume-consistency.md` | 实验 4 记录 |

## 参考

- T05、T11：FlashAttention-2 与 selective recompute。
- T10、T16：FP8 训练的两代做法。
- T09：TorchTitan 的 compile、float8、DCP 实现。
- T14：Llama 3 的故障与 checkpoint 实践（16K 卡上的中断统计）。
- B09、B10、B11：三个框架文档。
