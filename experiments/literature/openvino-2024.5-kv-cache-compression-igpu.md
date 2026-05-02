# OpenVINO: KV cache compression + dynamic quantization on iGPU (2024.5.0)

**Released:** 2024-11 (2024-11-20)
**What changed:** KV cache compression for LLMs landed on built-in GPUs (Core Ultra Series 1, Arc Graphics). Dynamic quantization for first-token latency on built-in GPUs (Series 1) without accuracy loss; also helps 2nd-token latency at large batch. SDPA + PagedAttention extended to FP16. Model load time optimizations to improve TTFT. Speculative decoding added to GenAI API. LoRA adapters preview in GenAI API. NPU LLMPipeline support (Llama 3 8B, Llama 2 7B, Mistral-v0.2-7B, Qwen2-7B-Instruct, Phi-3 Mini).
**Headline perf claim (if any):** No specific %, but explicit "memory reduction" from KV cache compression and "first token latency improvement" on integrated GPU.
**How to use it from optimum-intel / OV runtime:**
```python
# Enable KV cache compression and dynamic quantization on GPU
import openvino as ov
core = ov.Core()
core.set_property("GPU", {
    "KV_CACHE_PRECISION": "u8",                  # quantize KV cache to U8 to save mem
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",     # 32/64/128; activation quant group
    "INFERENCE_PRECISION_HINT": "f16",
})
```
GenAI speculative decoding:
```python
pipe = ov_genai.LLMPipeline(target_model_dir, "GPU",
    draft_model=ov_genai.draft_model(draft_model_dir, "CPU"))
```
**Intel GPU applicability:** HIGH for Arc 140V (Lunar Lake iGPU). HIGH for Arc B390 (dGPU got it later in 2024.4 release for dynamic quant on dGPU).
**Open hypothesis it generates for us:** Set `KV_CACHE_PRECISION=u8` + `DYNAMIC_QUANTIZATION_GROUP_SIZE=32` on alpha (Arc 140V) for Phi-3-mini-INT4 at 2048-context; expect ≥30% memory reduction on KV cache and ≤5% accuracy degradation on a small lm-eval suite.

Sources:
- https://github.com/openvinotoolkit/openvino/releases/tag/2024.5.0
