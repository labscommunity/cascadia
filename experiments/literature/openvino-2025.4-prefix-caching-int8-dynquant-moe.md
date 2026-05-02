# OpenVINO: Prefix caching + INT8 dynamic quantization + MoE on GPU (2025.4.0)

**Released:** 2025-12 (2025-12-01)
**What changed:** GPU plugin gains *prefix caching* for chat-history workloads (reuse KV blocks across turns). INT8 dynamic quantization on GPU (was FP16 only on dGPU before; trades off some accuracy gains over INT4). Multi-token generation accelerated via new GPU kernels with smarter KV-cache reuse. MoE preview on CPU+GPU (Qwen3-30B-A3B). Encrypted blob format. NNCF ONNX backend gets INT8 PTQ + INT8/INT4 weight-only + SmoothQuant. Mistral-Small-24B + Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B. NPU batch via reshape→1.
**Headline perf claim (if any):** "Improved performance with prefix caching for chat history scenarios and enhanced LLM accuracy with dynamic quantization support for INT8" on GPU. "Accelerated multi-token generation … faster inference, smarter KV-cache reuse, and scalable LLM performance."
**How to use it from optimum-intel / OV runtime:**
```python
# Prefix caching is part of the scheduler — turn on explicitly:
sched = ov_genai.SchedulerConfig()
sched.enable_prefix_caching = True
sched.cache_size = 4
pipe = ov_genai.LLMPipeline(model_dir, "GPU", scheduler_config=sched)

# Use the chat API to actually benefit from prefix caching:
pipe.start_chat()
print(pipe.generate("First question..."))
print(pipe.generate("Follow-up that shares the system prompt..."))
pipe.finish_chat()

# INT8 dynamic quant on GPU (more accurate than INT4, faster than FP16)
core.set_property("GPU", {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
                         "INFERENCE_PRECISION_HINT": "f16"})
# Combined with --quant-mode int8 at export:
# optimum-cli export openvino -m qwen2.5-7b --quant-mode int8 --sym ./qwen2.5-7b-int8
```
**Intel GPU applicability:** HIGH for both Arc 140V and Arc B390. Prefix caching is a game-changer for any chat or RAG workload with shared system prompts; INT8 dyn-quant lets us serve 7B-class models with INT4-like speed and INT8-like accuracy.
**Open hypothesis it generates for us:** On charlie (B390) measure TTFT for a 5-turn chat with a 2000-token system prompt + 50-token user turns, with prefix caching ON vs OFF. Hypothesis: turn-2-onwards TTFT drops by ≥85% (only the new user turn reprocessed).

Sources:
- https://github.com/openvinotoolkit/openvino/releases/tag/2025.4.0
- https://medium.com/openvino-toolkit/openvino-2025-4-faster-models-smarter-agents-3709e6437a08
