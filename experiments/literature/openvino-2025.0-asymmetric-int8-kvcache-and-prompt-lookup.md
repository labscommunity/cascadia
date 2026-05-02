# OpenVINO: Asymmetric INT8 KV cache + Prompt Lookup decoding (2025.0.0)

**Released:** 2025-02 (2025-02-06)
**What changed:** Asymmetric KV cache compression for INT8 enabled on CPUs (must be explicitly opted in, not default). Prompt Lookup decoding added to GenAI for 2nd-token latency improvement (no draft model needed — it grabs n-gram matches from the prompt itself). DeepSeek-R1-Distill family + Qwen 2.5 + FLUX support. Whisper improved on CPU + iGPU + dGPU via GenAI. NPU `torch.compile` preview. Triton Inference Server backend.
**Headline perf claim (if any):** "Lower memory consumption and improved 2nd token latency, especially when dealing with long prompts" for the INT8 KV cache. Prompt Lookup "improves 2nd token latency for LLMs by effectively utilizing predefined prompts that match the intended use case."
**How to use it from optimum-intel / OV runtime:**
```python
# CPU asymmetric INT8 KV cache (opt-in)
core.set_property("CPU", {"KV_CACHE_PRECISION": "u8"})

# Prompt Lookup decoding via GenAI
config = ov_genai.GenerationConfig()
config.num_assistant_tokens = 5    # n-grams to look up
config.assistant_confidence_threshold = 0.4
pipe = ov_genai.LLMPipeline(model_dir, "GPU", prompt_lookup=True)
out = pipe.generate(prompt, config)
```
**Intel GPU applicability:** MEDIUM. Asymmetric INT8 KV cache is *CPU-only* in 2025.0 — but prompt-lookup decoding works on CPU and GPU. On GPU U8 KV cache was already default since 2024.6.
**Open hypothesis it generates for us:** On alpha (Arc 140V) and charlie (B390) run Llama-3-8B-INT4 with a long-context summarization task using Prompt Lookup vs. greedy. Hypothesis: ≥1.5x speedup for inputs where output reuses ≥30% of input n-grams (RAG-style summarization).

Sources:
- https://github.com/openvinotoolkit/openvino/releases/tag/2025.0.0
- https://medium.com/openvino-toolkit/enhancing-llm-inference-with-prompt-lookup-decoding-and-openvino-genai-e15b69aeaeab
