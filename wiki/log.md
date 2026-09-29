---
title: Log
updated: 2026-09-27
---

# Knowledge Base Log

## [2026-09-29] synthesis | Created LLM training and post-training 30-day course

User asked for a third course covering rollout, pretraining, post-training, training frameworks, RL and agentic RL, LoRA and SFT, designed to be practical and industry-current. Created [[llm-training-30d/index]], [[llm-training-30d/learning-path]] and [[llm-training-30d/reading-index]] following the agent-rsi-30d paradigm (four projects, unified experiment protocol, compute tiers) with ai-infra-30d depth per day. 30 daily tutorials in `wiki/llm-training-30d/week1-4/`: Week 1 distributed training (memory math, data pipeline, ZeRO/FSDP2, TP/PP/CP/EP, frameworks, profiling), Week 2 SFT/LoRA/data engineering/DPO family/reward models/evaluation, Week 3 rollout and RLVR (PPO/GRPO/DAPO/GSPO, verifiers, verl-class frameworks, rollout engine weight sync and train-infer mismatch, async RL stability), Week 4 agentic RL (formalization, environments, rollout infra, reward, SWE/tool-use case studies, reasoning models, safety alignment and labeling, training ops, capstone). Added [[2026-09-29_llm-training-course-references]] (T01 to T47, B01 to B15) and updated the wiki index.

## [2026-09-29] synthesis | Added problem-driven inference optimization track to ai-infra-30d

User judged the course too simple and asked for an inference-optimization track organized by common problems and their fixes. Created `wiki/ai-infra-30d/optimization/` with [[ai-infra-30d/optimization/00-index]] (symptom entry table), [[ai-infra-30d/optimization/01-diagnosis-playbook]] (four-quadrant triage, tools, hypothesis loop) and 14 topic pages: TTFT high, TPOT/ITL jitter, throughput low, OOM and preemption, GPU util low / CPU bound, long context, multi-turn and agent workloads, structured output and tool calls, quantization choice, spec decode no gain, MoE serving, parallelism choice, cold start and autoscaling, cost per token. Each page follows 症状判定 → 根因排查 → 决策表 → 验证实验 and links back to the day pages. Linked from [[ai-infra-30d/reading-index]], [[ai-inference-learning-path]], Day 14 and the wiki index.

## [2026-09-27] reference | Surveyed Coursera and external courses as supplementary material

Ran about 35 Coursera site searches and opened 44 course pages for vLLM, LLM serving, quantization, CUDA, RL, agents, AI safety and automated research. Coursera has no course on vLLM/SGLang/TensorRT-LLM internals, Triton, RSI, AI Scientist or scalable oversight; closest matches are Board Infinity's Deploying Deep Learning (vLLM, PagedAttention, AWQ/GPTQ), Red Hat's AI Inference Technical Overview, DeepLearning.AI's Efficiently Serving LLMs / Quantization in Depth / Agentic AI, and the Alberta RL specialization. Filed [[supplementary-courses]] with day-level mappings to both [[ai-infra-30d/reading-index]] and [[agent-rsi-30d/index]], plus verified off-platform alternatives (CS336, GPU MODE, MIT 6.5940, DLAI GRPO/post-training, Berkeley LLM Agents, HF Agents, AI-Scientist, DGM, BlueDot). Linked from the wiki index and the ai-infra reading index.

## [2026-09-27] synthesis | Completed detailed 30-day Agent self-iteration, Agent RL and RSI course

Replaced the previous 16-week outline with the standalone [[agent-rsi-30d/index]] intensive course and wrote 30 complete daily tutorials under `wiki/agent-rsi-30d/`. The curriculum now contains executable learning objectives, implementation contracts, experiments, controls, failure modes and acceptance criteria for measurable Agent runtimes, reflection and memory, prompt/workflow evolution, Agent RL, Auto Research, self-modification, population archives, controlled RSI, scalable oversight and ASI evidence standards. Added four cumulative projects (P0–P3), a unified reproducibility protocol, hidden/transfer evaluation, red-team gates and an E0–E5 evidence rubric. Updated [[agent-rsi-30d/reading-index]], the course boundary with AI Infra, and the wiki index.

## [2026-09-27] rewrite | Upgraded Week 2 vLLM content from V0 to source-level V1

User flagged the vLLM material as too basic. Audit found Day 8/9 described the removed vLLM V0 architecture (SequenceGroup, block_manager.py, WAITING/RUNNING/SWAPPED, swap preemption, copy-on-write) and no coverage of EngineCore, KVCacheManager, torch.compile/CUDA graph, async scheduling, logits processors, structured output internals or KV connectors. Rewrote [[ai-infra-30d/week2/day08-vllm-architecture]] (three-process model, ZMQ/msgspec, request lifecycle with `vllm/v1/` paths, CPU overhead, custom metric exercise), [[ai-infra-30d/week2/day09-scheduler]] (token-budget loop, `num_computed_tokens`, recompute-only preemption, priority, async scheduling, `--scheduler-cls` exercise), [[ai-infra-30d/week2/day10-kv-cache]] (BlockPool/hash chain/LRU eviction, hybrid KV, KV connector API, file-connector exercise) and [[ai-infra-30d/week2/day13-flash-attention]] (GPUModelRunner, persistent batch, piecewise CUDA graph, attention backends, compressed FlashAttention theory, sampler/logits processor/structured output). Patched V0 remnants in Day 2/4/6/7/12/14/30, added V1 parameter table to Day 2 and `--speculative-config` to Day 12, added correction #13 to [[ai-inference-learning-path]], updated [[ai-infra-30d/reading-index]] and the index.

## [2026-09-27] refactor | Separated Agent RSI course from AI Infra course

Moved the Agent self-evolution, Agent RL, Auto Research, RSI and ASI curriculum into a standalone course directory. This course was subsequently compressed and completed as [[agent-rsi-30d/index]]. Added a dedicated course landing page and explicit scope boundary with [[ai-infra-30d/reading-index]].

## [2026-09-27] synthesis | Designed Agent self-evolution, RSI and ASI research course

Created the initial 16-week research curriculum spanning measurable agent systems, reflection and memory, prompt/workflow evolution, agentic reinforcement learning, automated research, controlled recursive self-improvement, scalable oversight, reward hacking and ASI evidence standards. It was subsequently replaced by the complete [[agent-rsi-30d/learning-path|30-day course]]. Added four progressive projects, a unified experiment protocol, compute-tier alternatives, safety gates, grading criteria and current primary references. Added [[2026-09-27_agent-self-evolution-rsi-asi-course-references]] and updated the wiki index.

## [2026-06-01] review | Audited 30-day AI inference course for accuracy and gaps

Reviewed `wiki/ai-inference-learning-path.md` and the generated `wiki/ai-infra-30d/` daily course against current vLLM, TensorRT-LLM, SGLang, DeepSeek, MiniMax, and Qwen references. Corrected vLLM V1 chunked prefill defaults, prefix caching version caveats, KV Cache memory formulas, `max_model_len` tradeoffs, metrics naming drift, Ray vs multiprocessing distributed runtime wording, DeepSeek/MLA and MiniMax/Lightning Attention overclaims, Qwen3 MoE status, and benchmark methodology gaps. Added missing coverage for structured outputs/tool calling, cold start/model loading, cache-aware routing, Prometheus metrics, and version-specific documentation checks.

## [2026-06-01] synthesis | Generated 30-day self-contained learning materials

Created `wiki/ai-infra-30d/` with 30 self-contained daily tutorials covering: inference pipeline, vLLM deployment/architecture/scheduling, benchmarking, PagedAttention, profiling, parameter tuning, FlashAttention, quantization (GPTQ/AWQ/FP8), speculative decoding, KV Cache optimization, distributed inference (TP/PP/DP/EP), MoE inference (DeepSeek-V3/MiniMax), production deployment, TensorRT-LLM/SGLang comparison, prefill-decode disaggregation, Triton kernel programming, roofline analysis, long context inference, training-inference consistency, and system design exercises. Includes Chinese model examples (Qwen, DeepSeek, GLM, MiniMax) throughout. Updated reading-index.md.

## [2026-06-01] synthesis | Revised AI inference learning path into 30-day crash course

Updated `wiki/ai-inference-learning-path.md` from a 16-20 week broad curriculum into a 30-day high-intensity course plan for learning AI inference infrastructure and optimization. Added an explicit evaluation of the previous design, daily 10+ hour schedule, weekly deliverables, vLLM-first learning sequence, distributed deployment focus, optimization experiments, and final system design project. Updated index description.

## [2026-06-01] synthesis | Created AI inference learning path

Created `wiki/ai-inference-learning-path.md` — a 6-phase, 16-20 week curriculum covering GPU fundamentals, inference optimization (FlashAttention, quantization, speculative decoding, KV Cache), vLLM deep dive, distributed deployment (TP/PP/EP), Triton/CUDA kernel optimization, and frontier topics (MoE, long context, prefill-decode disaggregation). Updated index.

## [2026-05-10] init | Knowledge base initialized

Set up project structure: `raw/`, `wiki/`, CLAUDE.md schema. Ready for first source ingest.

## [2026-09-27] update | Expanded mainstream model weight reference in Day 1

Updated [[ai-infra-30d/week1/day01-inference-pipeline]] with dated open-weight and closed-service model tables, official source links, total versus active parameter distinctions, and calculated BF16/8-bit/4-bit storage. Covered DeepSeek V4/V4.1, Kimi K3/K2.5, Qwen3.8/3.7/3.6/3.5, Gemma 4, GLM, MiniMax, Mistral, Llama, gpt-oss, ERNIE, Hunyuan, GPT, Claude, Gemini, Grok and Doubao. Flagged backbone-only counts, additional embedding/encoder modules, unavailable proprietary parameter counts, and quantization overhead. Corrected the H100 example to specify SXM and its weight-bandwidth-only assumptions. Added [[2026-09-27_mainstream-llm-weight-references]] as a new link source and updated the wiki index.
