# OpenVINO: Per-channel key-cache compression + sparse-attention prefill (2025.3.0)

**Released:** 2025-09 (2025-09-03)
**What changed:** Per-channel KEY cache compression (in addition to existing per-token), restoring accuracy when running int8/u8 KV cache on built-in and discrete GPUs. NPU plugin gains 8K context, dynamic prompts, dynamic LoRA, and dynamic batch (via reshape→1 + concurrent inference requests). TextRerankPipeline + Structured Output (XGrammar) in GenAI. Arc Pro B-series (B50, B60) supported. GGUF for OVMS preview (DeepSeek Distill, Qwen2/2.5, Llama 3). int4 data-aware compression for ONNX in NNCF. Sparse attention PREFILL implemented in GenAI.
**Headline perf claim (if any):** "Accuracy improvements for GenAI models on both built-in and discrete graphics achieved through the implementation of the key cache compression per channel technique." Sparse-attention prefill: targeted at long-context prefill (no number).
**How to use it from optimum-intel / OV runtime:**
```python
# KEY cache per-channel compression — automatic when KV_CACHE_PRECISION=u8 on GPU
core.set_property("GPU", {"KV_CACHE_PRECISION": "u8"})  # gets the new per-channel scheme

# Sparse attention prefill via SchedulerConfig
sparse_cfg = ov_genai.SparseAttentionConfig(
    mode=ov_genai.SparseAttentionMode.TRISHAPE,
    num_last_dense_tokens_in_prefill=128,
    num_retained_start_tokens_in_cache=32,
    num_retained_recent_tokens_in_cache=128,
)
sched.sparse_attention_config = sparse_cfg

# NPU dynamic prompts (no need to recompile per length anymore)
core.set_property("NPU", {"NPUW_LLM_PREFILL_HINT": "DYNAMIC"})
```
**Intel GPU applicability:** HIGH for both Arc 140V (Lunar Lake iGPU) and Arc B390 (Battlemage dGPU). Per-channel key-cache directly addresses the U8-KV accuracy regression we'd otherwise see at long context.
**Open hypothesis it generates for us:** On Arc B390, run Llama-3-8B-INT4 at 4K context with `KV_CACHE_PRECISION=u8` on OV 2025.2 vs 2025.3. Measure accuracy (perplexity on a 1k-prompt sample) and tokens/sec. Hypothesis: 2025.3 closes ≥50% of the U8-vs-FP16 perplexity gap with no measurable throughput cost.

Sources:
- https://github.com/openvinotoolkit/openvino/releases/tag/2025.3.0
- https://github.com/openvinotoolkit/openvino.genai/pull/2299 (sparse attention prefill)
- https://github.com/openvinotoolkit/openvino.genai/pull/2067 (SnapKV)
