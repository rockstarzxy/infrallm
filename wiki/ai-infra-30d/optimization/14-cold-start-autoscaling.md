---
title: "优化专项 14: 冷启动慢与扩缩容问题"
type: concept
tags: [ai-infra, inference-optimization, cold-start, autoscaling, kubernetes, model-loading, cuda-graph, lora]
created: 2026-09-29
updated: 2026-09-29
---
# 优化专项 14: 冷启动慢与扩缩容问题

> 一句话：实例从拉起到能接流量要几分钟到十几分钟，导致扩容跟不上流量、滚动更新掉请求、成本上要长期预留。本页拆解启动时间的每一段和对应缩短手段，以及扩缩容指标怎么选。关联日课：[[day20-production-deploy]]、[[day21-system-design-v1]]、[[day29-final-design]]、[[day08-vllm-architecture]]。

## 1. 症状与判定

| 症状 | 关键证据 | 最可能根因 |
|---|---|---|
| Pod 从 Pending 到 Ready 超过 10 分钟 | `kubectl describe pod` 的事件时间戳：Pulling image 占大头 | 镜像太大且节点无缓存（根因 A） |
| 容器起来后日志停在 `Loading weights` 很久 | 日志里 `Loading model weights took` 的秒数 | 权重从远端下载或单线程加载（根因 B） |
| 权重加载完到 `Application startup complete` 之间还有几分钟 | 日志顺序：memory profiling → `torch.compile` → `Capturing CUDA graphs` | 编译和 graph 录制没有缓存（根因 C） |
| 扩容触发了，但新实例 Ready 时流量高峰已经过去 | HPA/KEDA 事件时间 vs 实例 Ready 时间 | 扩容延迟 > 流量变化时间尺度（根因 D） |
| 滚动更新期间出现 5xx 或流式请求中断 | 网关日志里的 connection reset 集中在旧 Pod 终止时刻 | 没有排空，SIGTERM 直接杀掉在飞请求（根因 E） |
| 新实例上线后前几分钟 TTFT 明显高于老实例 | 新实例 `vllm:prefix_cache_hits` 为 0，CUDA graph 首次命中 | 冷缓存（根因 F） |
| GPU 利用率指标很高但队列一直在涨 | `vllm:num_requests_waiting` 持续增长而 autoscaler 不动 | 扩容指标选错（根因 G） |
| 多个小模型共享一张卡时互相拖慢或 OOM | 时间片共享下 vLLM 各自预占显存 | 共享方式选错（根因 H） |

## 2. 根因逐个排查

### 启动时间的构成

先在自己环境里测一次完整启动，把每段时间填进表。没有这张表的优化都是猜。

| 阶段 | 典型量级（示例） | 日志锚点 |
|---|---|---|
| 调度 + 镜像拉取 | 1 到 10 分钟（镜像 10 GB 以上） | K8s 事件 `Pulled` |
| Python 进程启动、import | 10 到 30 秒 | 第一行 vLLM 日志 |
| 权重下载 | 0 到 10 分钟（取决于来源） | HF 下载进度条 |
| 权重加载到 GPU | 10 秒到几分钟 | `Loading model weights took X GB / Y s` |
| KV cache profiling run | 几秒到几十秒 | `GPU KV cache size` / `# GPU blocks` |
| torch.compile | 30 秒到几分钟（无缓存） | `torch.compile takes X s` 或 compilation 相关日志 |
| CUDA graph capture | 10 秒到 1 分钟 | `Capturing CUDA graphs` |
| 健康检查通过 | 探针周期 | `/health` 返回 200 |

### 根因 A: 镜像拉取

**确认**。`kubectl describe pod` 里 `Pulling` 到 `Pulled` 的间隔。

**修**。
- 镜像预热：DaemonSet 提前在节点拉取，或节点镜像里预装。
- 拆分：基础镜像（CUDA + vLLM）稳定不变，模型不进镜像。
- 用镜像加速（registry 就近、lazy pulling 如 stargz/nydus）。

### 根因 B: 权重下载与加载

**确认**。日志里 HF 下载进度条时间 + `Loading model weights took`。

**修**，按效果排序：
1. 权重放本地 NVMe 或节点共享的高速存储（PVC / hostPath），不要每次从 HF 或对象存储拉。`--download-dir` 指定缓存目录。
2. 用 safetensors 格式，vLLM 默认 `--load-format auto` 会优先 safetensors，mmap 加载比 pickle 的 `.bin` 快得多。
3. 直接从对象存储流式加载：`--load-format runai_streamer`（需要安装 vllm 的 runai extra，支持 S3 路径），或 `--load-format tensorizer`（需要先用 tensorizer 序列化权重，配 `--model-loader-extra-config`）。两者的价值是把"下载 + 反序列化 + 拷贝到 GPU"变成一条流水线。
4. 网络带宽：对象存储到节点的带宽决定下限，70B bf16 约 140 GB，10 Gbps 网络要 2 分钟以上。
5. 排查阶段可以用 `--load-format dummy` 跳过权重加载，验证其他阶段的时间。

**副作用**。共享存储上的 mmap 在 NFS 上可能比本地慢，需要实测。

### 根因 C: 编译与 CUDA graph 录制

**机制**。V1 默认开 torch.compile 和 piecewise CUDA graph（[[day13-flash-attention]]）。编译产物缓存在 `~/.cache/vllm/torch_compile_cache/<hash>/`，hash 包含模型、并行配置、编译配置；根目录可用 `VLLM_CACHE_ROOT` 环境变量改。CUDA graph 每次启动都要重新录制，不能缓存。

**修**。
- 编译缓存预热：构建镜像或准备节点时用相同配置启动一次，把缓存目录打进镜像或放到共享存储，运行时挂载。配置有任何变化 hash 就变，要和启动参数一起管理。
- 减少 capture 桶数：`--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128]}'`，桶越少录制越快，但大 batch 时 pad 浪费增加。
- 应急：`--enforce-eager` 跳过两者，启动最快，但 decode 性能下降，只用于调试或极短生命周期的实例。
- 不要在 KV profiling 阶段浪费时间：较新版本支持 `--kv-cache-memory-bytes` 直接指定 KV 显存，跳过 profiling run；参数是否存在以 `--help` 为准。

### 根因 D: 扩容延迟大于流量变化尺度

**机制**。LLM 实例从触发到 Ready 通常 2 到 15 分钟，而流量突增常在 1 分钟内发生。反应式扩容注定追不上，只能靠预留。

**修**。
- 容量规划留 buffer：常驻容量 = 峰值预测 × (1 + 安全系数)，安全系数由扩容时间和流量增长速率决定。扩容需要 5 分钟，流量 5 分钟内可能涨 30%，buffer 就不能低于 30%。
- 预测式扩容：按日周期提前扩。
- 缩短启动时间（根因 A 到 C）是唯一能从根上降低 buffer 的办法。
- 热备：保留已加载权重但不接流量的实例。vLLM 的 sleep mode（`--enable-sleep-mode`，配合 `/sleep` 和 `/wake_up` 接口，level 1 把权重挪到 CPU 内存并丢弃 KV，level 2 丢弃权重）可以让实例在几秒内唤醒，代价是占着 GPU。接口是否需要开发模式开关随版本变化。

### 根因 E: 滚动更新没有排空

**机制**。vLLM 收到 SIGTERM 会退出，在飞请求被中断，流式请求的客户端会看到连接断开。KV cache 随进程消失。

**修**。
- K8s 上：`preStop` hook 先让实例从 LB 摘除（比如让 `/health` 返回非 200，或直接 sleep N 秒等网关摘除），`terminationGracePeriodSeconds` 设成大于最长请求时间。
- 网关侧：滚动时先停止向旧实例派发新请求，等 `vllm:num_requests_running` 归零再终止。
- 更新策略 `maxUnavailable: 0`，先起新再杀旧，容量不下降。
- 长请求（agent 多轮、长输出）要么设上限，要么接受被中断后客户端重试。

### 根因 F: 新实例冷缓存

**机制**。新实例的 prefix cache 是空的，CUDA graph 各桶首次执行也有开销，前几分钟指标偏差。

**修**。Ready 之后、接流量之前发一轮 warm-up：覆盖常用 system prompt（填 prefix cache）、覆盖各 batch 桶（走一遍 CUDA graph）。readiness 探针可以做成"warm-up 完成后才通过"，而不是 `/health` 一返回 200 就通过。

`/health` 的语义只是"进程活着且引擎初始化完成"，不代表性能已稳定。

### 根因 G: 扩缩容指标选错

**机制**。`nvidia-smi` 的 GPU utilization 是"过去采样窗口内有 kernel 在跑的时间比例"，一个请求在 decode 也能把它顶到 90% 以上，完全不反映还能接多少并发。用它做扩容指标会导致过早扩容或永不扩容。

**修**。按适用性排序的指标：

| 指标 | 反映什么 | 适合触发 |
|---|---|---|
| `vllm:num_requests_waiting` | 已经排队、还没被调度的请求数 | 扩容（最直接的过载信号） |
| `vllm:kv_cache_usage_perc` | KV 显存用了多少，接近 1 就会抢占 | 扩容 |
| TTFT P99（`vllm:time_to_first_token_seconds` 直方图） | 用户可感知的排队 + prefill 延迟 | 扩容（SLO 驱动） |
| `vllm:generation_tokens_total` 的速率 | 实际吞吐，与容量上限对比得到利用率 | 缩容 |
| `vllm:num_requests_running` | 当前并发，与 `max_num_seqs` 对比 | 缩容 |
| GPU utilization | 有没有 kernel 在跑 | 不用于扩缩容 |

KEDA 的 Prometheus scaler 可以直接用这些指标。扩容阈值宁可低一点（比如 waiting > 0 持续 30 秒），缩容阈值要高延迟（比如利用率 < 40% 持续 10 分钟），避免抖动。

### 根因 H: 多模型共享 GPU 的方式

**机制**。vLLM 启动时按 `--gpu-memory-utilization` 预占显存，两个实例在同一张卡上时要各自把这个值调低，且互相争抢 SM。MIG 把一张卡硬切成独立实例，隔离好，但 MIG 实例之间没有 NVLink，TP 不可能跨 MIG。时间片（time-slicing）共享显存不隔离，vLLM 的预占模式下几乎不可用。

**修**。
- 小模型多实例：MIG（A100/H100 支持），每个 MIG 实例跑一个 TP=1 的 vLLM。
- 同一张卡跑两个 vLLM：显式设 `--gpu-memory-utilization` 之和小于 0.95，接受互相干扰。
- 多个 LoRA 变体：不要起多个实例，用一个实例的多 LoRA 服务（下一节）。

### LoRA 多租户的启动模型

`--enable-lora --max-loras N --max-lora-rank R` 让一个基座实例同时服务多个 adapter，请求里用 `model` 字段指定 adapter 名。启动时用 `--lora-modules name=path` 预加载；运行时热加载需要 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` 后调用 `/v1/load_lora_adapter` 和 `/v1/unload_lora_adapter`。

对冷启动的意义：新增一个租户不再需要拉起新实例，只需要加载几十 MB 的 adapter，秒级。代价是 `max_loras` 越大每步的 LoRA kernel 开销越高，且所有租户共享一个 KV 池。

## 3. 决策表

| 情况 | 首选 | 次选 | 不要做 |
|---|---|---|---|
| 启动 10 分钟以上，镜像拉取占大头 | 节点镜像预热 + 拆分基础镜像 | lazy pulling | 把权重打进镜像 |
| 权重加载占大头 | 本地 NVMe 缓存 + safetensors | runai_streamer / tensorizer 直读对象存储 | 每次从 HF 下载 |
| 编译和录制占大头 | 预热编译缓存目录并挂载 | 减少 capture sizes | 生产用 `--enforce-eager` |
| 流量突增追不上 | 预留 buffer + 预测式扩容 | sleep mode 热备 | 单纯降低 HPA 阈值 |
| 滚动更新掉请求 | preStop 摘流 + 足够的 grace period + maxUnavailable 0 | 网关层排空 | 直接 rolling restart |
| 新实例前几分钟慢 | warm-up 后才 Ready | 网关对新实例限流预热 | 无 |
| 扩容不触发或乱触发 | waiting 数 + KV usage + TTFT P99 | tokens/s 利用率 | GPU util |
| 多个小模型共享卡 | MIG | 调低各自 memory utilization | time-slicing |
| 多个微调变体 | 单实例多 LoRA + 热加载 | 每变体一实例（仅当 adapter 大或延迟敏感） | 无 |

## 4. 验证实验

### 实验 1: 启动时间分解

固定：模型、镜像、节点。记录三次冷启动，把日志时间戳按第 2 节的表分段。然后依次应用：本地权重缓存、编译缓存预热、减少 capture sizes，每改一项测一次。预期：权重和编译两项加起来通常占 60% 以上，处理后总时间下降一半量级。

```bash
# 抓取时间戳的简易方法
vllm serve MODEL 2>&1 | while IFS= read -r line; do echo "$(date +%s.%N) $line"; done | tee startup.log
grep -E "Loading model weights|KV cache|compile|Capturing|startup complete" startup.log
```

### 实验 2: 编译缓存有效性

启动一次，记录编译时间；不改任何参数再启动一次，编译时间应接近 0；改 `--max-num-seqs` 再启动，看编译缓存是否失效（hash 是否包含这个参数随版本变化，实测为准）。

### 实验 3: 滚动更新掉请求

用 `vllm bench serve --request-rate 5 --num-prompts 600` 持续打流量，中途执行一次滚动更新。对比有无 preStop 排空时的失败请求数和流式中断数。预期：排空后失败数为 0，代价是更新时间变长。

### 实验 4: 扩容指标对比

用 KEDA 或手动脚本，分别以 GPU util > 80% 和 `num_requests_waiting > 0` 作为触发条件，施加阶梯式增长的流量。记录触发时刻与 TTFT P99 越过 SLO 的时刻之差。预期：GPU util 要么早触发（浪费）要么不触发；waiting 数触发点与 SLO 越界点接近。

### 实验 5: warm-up 效果

新实例 Ready 后立即打流量 vs 先发 50 条覆盖常用 system prompt 和各 batch 桶的 warm-up 请求再打流量。对比前 2 分钟的 TTFT P99。

## 5. 关联

- [[01-diagnosis-playbook]]：从整体症状进入这一页的路径。
- [[02-ttft-high]]：新实例冷缓存造成的 TTFT 抖动。
- [[05-oom-preemption]]：扩容不及时导致的 KV 压力。
- [[06-gpu-util-low-cpu-bound]]：为什么 GPU util 不能当容量指标。
- [[15-cost-per-token]]：预留 buffer 的成本换算。
- [[13-parallelism-choice]]：MIG 与 TP 的冲突。
- 日课：[[day08-vllm-architecture]]（编译缓存、进程模型）、[[day13-flash-attention]]（CUDA graph 桶）、[[day20-production-deploy]]、[[day21-system-design-v1]]、[[day29-final-design]]。
