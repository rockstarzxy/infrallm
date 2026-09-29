---
title: 辅助学习课程：Coursera 与站外资源（推理基础设施 + Agent RSI）
type: reference
tags: [coursera, learning-resources, ai-infra, agent-rsi, supplementary]
sources: []
created: 2026-09-27
updated: 2026-09-27
---

# 辅助学习课程：Coursera 与站外资源

为 [[ai-infra-30d/reading-index]]（课程 A，30 天推理基础设施）和 [[agent-rsi-30d/index]]（课程 B，30 天 Agent 自迭代 / RSI）配套的外部课程清单。2026-09-27 用无头浏览器在 coursera.org 做了约 35 次站内搜索并逐一打开 44 个课程页核对，下面每一门都实际看过详情页（只见过搜索结果的会标注）。时长、级别、评分为当日页面所示。

## 先说结论

- Coursera 上**没有**以 vLLM、SGLang、TensorRT-LLM、speculative decoding、Triton kernel、FlashAttention、roofline 为主题的课。搜 "SGLang" 出来的是编程语言课。
- Coursera 上**没有**任何涉及 RSI、AI Scientist、自动化研究、scalable oversight、alignment 研究的课。搜 "recursive self-improvement" 出来的是个人成长课。
- 课程 A 有两门意外贴合：Board Infinity 的 Deploying Deep Learning（大纲明写 vLLM PagedAttention、AWQ/GPTQ、KV cache 吞吐上限）和 Red Hat 的 AI Inference Technical Overview（vLLM、tensor parallelism、llm-d、speculative decoding）。
- 课程 B 只能在 Coursera 上覆盖 Week 1 到 3（Agent 设计模式、RL 基础、RLHF/PPO）；Week 4 全靠站外资源。

相关度标记：**直接** = 内容与我们的某几天重合；**部分** = 覆盖前置或相邻知识；**基础** = 只作背景。

## 1. LLM 推理与部署（课程 A）

| 课程 | 机构 | 时长 / 级别 | 对应天数 | 相关度 / 说明 |
|---|---|---|---|---|
| [Deploying Deep Learning: Quantization, Serving, and Edge AI](https://www.coursera.org/learn/deploying-deep-learning-quantization-serving-and-edge-ai) | Board Infinity | 约 20h / Advanced | A Day 2、3、4、10、11、20、22 | **直接**。含 Running a vLLM Server、Why KV Cache Limits Throughput、PagedAttention Deep Dive、AWQ/GPTQ 量化、When Triton Makes Sense、ONNX/llama.cpp、GPU 集群扩缩容。Capstone 是量化 → vLLM 服务 → benchmark → Docker。Coursera 上最贴近课程 A 的一门 |
| [Red Hat AI Inference Technical Overview](https://www.coursera.org/learn/red-hat-ai-inference-technical-overview) | Red Hat Training | 约 1h / Intermediate，2026-08 更新 | A Day 8、10、12、16、20、24 | **直接但极简**。vLLM runtime、KV cache 管理、parallelism、llm-d 分布式推理、speculative decoding。1 小时概览，不能替代动手 |
| [Efficiently Serving LLMs](https://www.coursera.org/projects/efficiently-serving-llms) | DeepLearning.AI × Predibase | 约 1h / Intermediate | A Day 1、3、9、10 | **直接**。用代码实现 KV caching、continuous batching、量化并 benchmark；LoRA 多 adapter 服务 |
| [Quantization in Depth](https://www.coursera.org/projects/quantization-in-depth) | DeepLearning.AI × Hugging Face | 约 1h / Intermediate | A Day 11 | **直接**。per-tensor / per-channel / per-group 线性量化，手写 PyTorch 量化器，2-bit 打包 |
| [Quantization Fundamentals with Hugging Face](https://www.coursera.org/projects/quantization-fundamentals) | DeepLearning.AI × Hugging Face | 约 1h / Beginner | A Day 11 | **部分**。入门版，先学这个再学上一门 |
| [Open Source LLMOps Solutions](https://www.coursera.org/learn/open-source-llmops-solutions) | Duke University | 约 36h / Beginner，4.8 分 | A Day 2、20 | **部分**。llamafile / whisper.cpp 本地运行，最后用 LoRAX 和 vLLM 容器化部署。工具使用向 |
| [LLMOps Specialization](https://www.coursera.org/specializations/large-language-model-operations) | Duke University | 6 门约 181h / Beginner | A Day 20、21 | **基础**。Azure / AWS / Databricks 平台向 |
| [Optimize TensorFlow Models with TensorRT](https://www.coursera.org/projects/tensorflow-tensorrt) | Coursera 引导项目 | 1.5h / Intermediate | A Day 22 | **部分**。TF-TRT 的 FP32/FP16/INT8 对比。是 TensorRT 不是 TensorRT-LLM |
| [Optimize AI Inference Speed & Accuracy](https://www.coursera.org/learn/optimize-ai-inference-speed--accuracy) | Starweaver | 4h / Intermediate | A Day 5、11 | **部分**。PyTorch Profiler、剪枝、量化，非 LLM 专用 |
| [Generative AI with LLMs](https://www.coursera.org/learn/generative-ai-with-llms) | DeepLearning.AI × AWS | 约 17h / Intermediate，4.8 分 | A Day 1；B Day 15 到 19 | 对 A **基础**，对 B Week 3 **部分**（RLHF、PPO、reward hacking、ReAct） |
| [AI Infrastructure: Deployment, Networking, and Storage](https://www.coursera.org/specializations/ai-infrastructure-deployment-networking-and-storage) | Google Cloud | 5 门约 18h / Intermediate | A Day 15、20 | **部分**。GKE GPU 集群、网络存储、GPU vs TPU。子课 [Cloud GPUs](https://www.coursera.org/learn/ai-infrastructure-cloud-gpus) 2h |
| [GPU Clusters & Containers](https://www.coursera.org/learn/gpu-clusters--containers) | Coursera 自制 | 2h / Intermediate | A Day 15、20 | **部分** |
| [AI Infrastructure and Operations Fundamentals](https://www.coursera.org/learn/ai-infrastructure-operations-fundamentals) | NVIDIA Training | 约 10h / Beginner | A Day 15 | **基础**。NVIDIA 官方入门概念课 |
| [NVIDIA: LLMs and Generative AI Deployment](https://www.coursera.org/learn/nvidia-large-language-models-and-generative-ai-deployment) | Whizlabs（考证课，非 NVIDIA 制作） | 4h，3.8 分 | A Day 20 | **基础**，评分偏低 |
| [Designing Production LLM Architectures](https://www.coursera.org/learn/designing-production-llm-architectures1) | Coursera 自制 | 约 10h / Intermediate | A Day 21、29 | **部分**。self-host vs API 取舍、多区域延迟、容错成本。不进引擎内部 |
| [Machine Learning in Production](https://www.coursera.org/learn/introduction-to-machine-learning-in-production) | DeepLearning.AI | 约 1 周，4.8 分 | A Day 20 | **基础**。通用 MLOps |
| [Gradient to Production: MLOps & Model Serving](https://www.coursera.org/specializations/gradient-to-production-mlops-model-serving) | Coursera 自制 | 15 门微课约 35h | A Day 20 | **基础**。FastAPI / Docker / K8s 推理 API |

## 2. GPU / CUDA / 性能（课程 A Week 4）

| 课程 | 机构 | 时长 / 级别 | 对应天数 | 相关度 / 说明 |
|---|---|---|---|---|
| [GPU Programming Specialization](https://www.coursera.org/specializations/gpu-programming) | Johns Hopkins | 4 门约 97h / Intermediate | A Day 13、25、26 | **部分**。CUDA C/C++、内存管理、cuBLAS/Thrust。写 Triton/CUDA kernel 前的底子，不讲 attention kernel。子课 [Intro to Parallel Programming with CUDA](https://www.coursera.org/learn/introduction-to-parallel-programming-with-cuda)；[CUDA Advanced Libraries](https://www.coursera.org/learn/cuda-advanced-libraries) 评分仅 3.4 |
| [GPU Programming with C++ and CUDA](https://www.coursera.org/learn/packt-gpu-programming-with-c-and-cuda) | Packt | 约 10h / Intermediate | A Day 25 | **部分**。CUDA streams、自定义 kernel、Python 集成，更短更实操 |
| Multicore and GPGPU Programming；High-Performance and Parallel Computing | 未核实 | 未打开详情页 | A Day 25 | 仅见搜索结果标题，未核实 |

## 3. Agent / RL（课程 B）

| 课程 | 机构 | 时长 / 级别 | 对应天数 | 相关度 / 说明 |
|---|---|---|---|---|
| [Agentic AI](https://www.coursera.org/learn/dlai-agentic-ai) | DeepLearning.AI（Andrew Ng） | 约 19h / Intermediate | B Day 2、3、5、6、8、13 | **直接**（对 Week 1 到 2）。reflection / tool use / planning / multi-agent 四大模式，量化 reflection 增益，运行时信号接入 reflection loop，评测与错误分析 |
| [Reinforcement Learning Specialization](https://www.coursera.org/specializations/reinforcement-learning) | University of Alberta | 4 门约 75h / Intermediate，4.7 分 | B Day 15、17 | **基础**但质量最高。MDP、TD、函数逼近、规划。不涉及 LLM |
| [Deep RL: From Theory to Practice](https://www.coursera.org/learn/deep-reinforcement-learning-from-theory-to-practice) | CU Boulder | 约 20h / Intermediate | B Day 17、19 | **部分**。GAE、Actor-Critic 到 PPO、TRPO/DDPG/SAC。经典环境，非 GRPO |
| [RLHF](https://www.coursera.org/projects/reinforcement-learning-from-human-feedback-project) | DeepLearning.AI × Google Cloud | 1h / Intermediate，4.7 分 | B Day 16、18、19 | **部分**。preference 数据集、Vertex Pipeline 对 Llama 2 做 RLHF |
| [Multi AI Agent Systems with crewAI](https://www.coursera.org/projects/multi-ai-agent-systems-with-crewai) | DeepLearning.AI × crewAI | 1h / Beginner，4.7 分 | B Day 13 | **部分**。多 Agent 编排入门，无共同进化 |
| [Advanced Agentic AI: Self-Improving Systems & Frameworks](https://www.coursera.org/learn/packt-advanced-agentic-ai-self-improving-systems-frameworks) | Packt | 6h / Intermediate | B Day 12（勉强） | **基础，名不副实**。"Self-Improving" 指企业成熟度路线图和 ROI，不是 RSI。不建议为课程 B 购买 |

## 4. AI 安全与监督（课程 B Week 4）

| 课程 | 机构 | 时长 / 级别 | 对应天数 | 相关度 / 说明 |
|---|---|---|---|---|
| [Red Teaming LLM Applications](https://www.coursera.org/projects/red-teaming-llm-applications) | DeepLearning.AI × Giskard | 1h / Beginner，4.8 分 | B Day 5、6、29 | **部分**。自动化红队测试 LLM 应用 |
| [Secure AI: Red-Teaming & Safety Filters](https://www.coursera.org/learn/secure-ai-red-teaming--safety-filters) | Starweaver | 4h / Intermediate | B Day 29 | **部分**。应用安全视角 |
| [Responsible AI for Developers](https://www.coursera.org/specializations/responsible-ai-for-developers) | Google Cloud | 3 门约 10h / Intermediate | B Day 25、29 | **基础**。公平性、可解释性、隐私 |
| [AI Safety Fundamentals: Protecting Users and Organizations](https://www.coursera.org/learn/ai-safety-fundamentals-protecting-users-and-organizations) | SkillUp | 4h / Beginner | B Day 29 | **基础**。法规与 guardrails，非对齐研究 |
| [Advanced Techniques and Interpretability in LLMs](https://www.coursera.org/learn/packt-advanced-techniques-and-interpretability-in-llms) | Packt | 9h / Intermediate | 无 | **基础**。实为微调 + RAG，可解释性占比很小 |

搜 "AI alignment" 返回的全是 AI governance / ISO 42001 课程。Coursera 上没有 scalable oversight、debate、weak-to-strong、AI control 相关内容。

## 5. 自动化研究（课程 B Week 4）

| 课程 | 机构 | 时长 / 级别 | 对应天数 | 相关度 / 说明 |
|---|---|---|---|---|
| [AI for Research & Analysis](https://www.coursera.org/learn/ai-for-research--analysis) | AI CERTs | 8h / Beginner | B Day 22 | **基础**。Elicit 文献综述案例，商业研究工具向 |
| [AI for Scientific Research](https://www.coursera.org/specializations/artificial-intelligence-scientific-research) | LearnQuest | 4 门约 48h，3.3 分 | 无 | **不相关**。用 ML 分析科研数据，不是自动化研究 Agent |
| [Getting Started with AutoML](https://www.coursera.org/learn/automated-machine-learning-automl) | Edureka | 约 10h / Beginner | B Day 23（作类比） | **基础**。H2O AutoML、HPO |
| [AutoML Using AutoGluon](https://www.coursera.org/projects/automl-using-autogluon) | 引导项目 | 2h / Beginner | B Day 23 | **基础** |

## 6. Coursera 覆盖不到的主题与已核实的站外替代

| 主题 | 已核实存在的替代资源 |
|---|---|
| vLLM 内部（A Week 2） | vLLM 官方文档 docs.vllm.ai（本次访问被 bot 验证拦截，未读到内容）；Stanford [CS336 Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)（含 Inference、Parallelism 讲次） |
| SGLang / RadixAttention（A Day 23） | [SGLang 官方文档](https://docs.sglang.ai/)，2026-08 仍在更新 |
| Triton / CUDA kernel、roofline（A Day 25、26） | [GPU MODE lectures](https://github.com/gpu-mode/lectures)（100 多讲，含 Triton kernel、CUDA C++、CuTe，2026-09 仍在更新）；[MIT 6.5940 TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)（Efficient Inference 章节，2024 秋视频在线，页面注明 2025 秋停开） |
| 量化 / 推理服务短课（A Week 2） | [DeepLearning.AI 短课目录](https://www.deeplearning.ai/short-courses/) 有 LLM Serving 6 门、Compression and Quantization 3 门；Attention in Transformers: Concepts and Code in PyTorch 对应 A Day 13 |
| NVIDIA 官方培训（A Day 22） | [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/)（目录页已确认，具体推理课程未逐一核实） |
| GRPO / post-training（B Day 18、19） | DLAI [Reinforcement Fine-Tuning LLMs With GRPO](https://www.deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo/)（Predibase，约 2h）；DLAI [Post-training of LLMs](https://www.deeplearning.ai/short-courses/post-training-of-llms/)（SFT / DPO / online RL）；[Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)（Unit 8 PPO） |
| Agent runtime / 评测 / 多 Agent（B Week 1、2） | [Berkeley LLM Agents MOOC](https://llmagents-learning.org/f24)（AutoGen、SWE-agent、"LLMs Cannot Self-Correct Reasoning" 等阅读）；[Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction)（含 Agent Observability and Evaluation 单元） |
| Auto Research / AI Scientist（B Day 24） | [Sakana AI-Scientist](https://github.com/SakanaAI/AI-Scientist) 代码库 |
| 自修改 / DGM / 受控 RSI（B Day 26 到 28） | [Darwin Gödel Machine](https://github.com/jennyzzt/dgm) 代码库 |
| scalable oversight / alignment（B Day 28、29） | [BlueDot Impact AI Safety Fundamentals](https://aisafetyfundamentals.com/)（免费） |

## 推荐的最小组合

- **课程 A**：Board Infinity 的 Deploying Deep Learning 为主，配 DLAI 的 Efficiently Serving LLMs 和 Quantization in Depth（各 1h），Red Hat 概览 1h。CUDA 底子补 JHU 的 Intro to Parallel Programming with CUDA。Week 2 源码级内容和 Week 4 kernel 内容 Coursera 帮不上，用 GPU MODE 和 CS336。
- **课程 B**：DLAI 的 Agentic AI 覆盖 Week 1 到 2，Alberta 的 RL 专项或 CU Boulder 的 Deep RL 补 Week 3 数学，DLAI 的 RLHF 项目 1h。Week 4 完全靠站外：GRPO / Post-training 短课、AI-Scientist、DGM、BlueDot。
