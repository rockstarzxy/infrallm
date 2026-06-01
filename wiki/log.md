---
title: Log
updated: 2026-06-01
---

# Knowledge Base Log

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
