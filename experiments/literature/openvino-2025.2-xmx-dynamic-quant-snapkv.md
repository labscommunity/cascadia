# OpenVINO: XMX dynamic quant + SnapKV + GGUF reader (2025.2.0)

**Released:** 2025-06 (2025-06-18)
**What changed:** Dynamic quantization + GEMM/Conv optimization for XMX systolic platforms (Core Ultra Series 2 iGPU, Arc B-series). VLM uses continuous batching by default. KV cache compression INT8 *default on CPU* (not GPU yet). Further LoRA optimization on built-in GPU (fused-kernel approach). Preview: GGUF Reader for llama.cpp models loaded directly into OpenVINO graphs (DeepSeek-R1-Distill-Qwen, Qwen2.5-Instruct, Llama-3.2-Instruct). Text-to-Speech pipeline (SpeechT5). RAG backend with reduced memory. SnapKV (selective KV cache compression with clustered retention) on CPU + GPU when KV eviction is on. NPU FP16-NF4 precision for ≤8B models. INT4 data-free weights for ONNX in NNCF.
**Headline perf claim (if any):** "Enhance the performance of VLM models and hybrid quantized image generation models, as well as improve first-token latency for LLMs through dynamic quantization" on XMX platforms.
**How to use it from optimum-intel / OV runtime:**
```python
# Dynamic quant on GPU (XMX)
core.set_property("GPU", {
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
    "INFERENCE_PRECISION_HINT": "f16",
})

# GGUF reader path (no optimum-cli export needed!)
pipe = ov_genai.LLMPipeline("./qwen2.5-1.5b.Q4_K_M.gguf", "GPU")  # one shot

# SnapKV (selective compression with cache eviction enabled)
sched.cache_eviction_config = ov_genai.CacheEvictionConfig(
    start_size=32, recent_size=128, max_cache_size=2048,
    aggregation_mode=ov_genai.AggregationMode.NORM_SUM)
```
**Intel GPU applicability:** HIGH for Arc 140V (XMX iGPU on Series 2) and Arc B390 (B-series XMX dGPU). The XMX kernels target *exactly* these two SKUs. GGUF reader is GPU-validated.
**Open hypothesis it generates for us:** On alpha (140V) and charlie (B390), benchmark Qwen2.5-7B-Instruct loaded from GGUF Q4_K_M directly via GenAI vs. exported via optimum-cli to INT4 OV-IR. Hypothesis: GGUF path is within 10% of native OV-IR INT4 on tokens/sec but slashes the conversion step from minutes to seconds — making the GGUF path the right default for an Ollama-killer UX.

Sources:
- https://github.com/openvinotoolkit/openvino/releases/tag/2025.2.0
- https://medium.com/openvino-toolkit/announcing-openvino-2025-2-new-models-generative-ai-pipelines-and-performance-improvements-9e4e46335db3
