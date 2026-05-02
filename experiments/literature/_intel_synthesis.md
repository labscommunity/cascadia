# Intel inference perf synthesis: top-8 low-hanging fruit for Tahoma on Intel iGPU/dGPU

**Survey period:** 2024-01 → 2026-04 (today: 2026-05-01)
**Targets:** alpha = Arc 140V (Lunar Lake iGPU), charlie = Arc B390 (Battlemage dGPU)
**Selection criteria:** ranked by (a) expected impact, (b) ease of test, (c) likelihood of working out-of-the-box on both targets, (d) being upstream-supported (not a research patch).

---

## Top 8 perf knobs to test FIRST

### #1. Use `LLMPipeline` from openvino_genai, NOT `OVModelForCausalLM`
The single biggest decision. `LLMPipeline` exposes SchedulerConfig (continuous batching), prefix caching, KV eviction, sparse attention, speculative decoding, and prompt-lookup — `OVModelForCausalLM` exposes none of those. On chat workloads it can be 1.4-2.0x faster *with no other changes*. It's also the path Intel actively optimizes.

```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline(model_dir, "GPU")
print(pipe.generate("hi", ov_genai.GenerationConfig(max_new_tokens=128)))
```

### #2. Pin GPU plugin to U8 KV cache + dynamic quantization group=32
Default since OV 2024.6 (U8 KV) and 2025.2 (XMX dynamic quant) but worth setting explicitly to make sure the path is engaged on both targets. INT8 KV cache cuts KV memory ~2x; dynamic quant of activations (group=32) is the killer feature on XMX silicon.

```python
plugin_config = {
    "KV_CACHE_PRECISION": "u8",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
    "INFERENCE_PRECISION_HINT": "f16",
    "CACHE_DIR": "./ov_cache",          # huge cold-start TTFT win
    "CACHE_MODE": "OPTIMIZE_SPEED",
}
pipe = ov_genai.LLMPipeline(model_dir, "GPU", **plugin_config)
```

### #3. Turn ON SchedulerConfig with prefix caching + dynamic split-fuse
Continuous batching is on by default since OV 2025.1, but `enable_prefix_caching=True` and `dynamic_split_fuse=True` are NOT — explicitly enabling them is critical for chat / RAG / agentic workloads. Prefix caching alone can cut turn-N TTFT by ≥85% in multi-turn chat with shared system prompts.

```python
sched = ov_genai.SchedulerConfig()
sched.cache_size = 4                    # GB
sched.max_num_batched_tokens = 4096
sched.dynamic_split_fuse = True
sched.enable_prefix_caching = True
pipe = ov_genai.LLMPipeline(model_dir, "GPU", scheduler_config=sched)

pipe.start_chat()  # actually USE the prefix cache
```

### #4. Export INT4 with `--awq --scale-estimation --dataset wikitext2`
Default INT4 export is data-free; data-aware AWQ + scale estimation typically recovers 1-2 perplexity points at the cost of a slower export. For a model you'll deploy thousands of times, always pay this once. Group-size 128 is the standard sweet spot; 64 for smaller models per optimum-intel's per-model config map.

```bash
optimum-cli export openvino \
  -m meta-llama/Meta-Llama-3-8B-Instruct \
  --weight-format int4 --group-size 128 \
  --awq --scale-estimation --dataset wikitext2 \
  --task text-generation-with-past \
  ./Meta-Llama-3-8B-int4-ov
```

### #5. Speculative decoding with a small draft model
Available since OV 2024.5 on CPU/GPU and OV 2026.0 on NPU. A small draft (Llama-3.2-1B for an 8B target, or Phi-3-mini-FastDraft-50M for Phi-3) gives 1.5-3x decode speedup on tasks where draft acceptance is high (code, summarization, chat with predictable patterns).

```python
pipe = ov_genai.LLMPipeline("Llama-3-8B-int4-ov", "GPU",
    draft_model=ov_genai.draft_model("Llama-3-2-1B-int4-ov", "GPU"))
config.num_assistant_tokens = 5
config.assistant_confidence_threshold = 0.4
```

### #6. Prompt Lookup decoding for RAG / summarization
For workloads where the output reuses n-grams from the input (RAG, summarization, code completion-in-context) — Prompt Lookup is FREE acceleration with no draft model. Available since OV 2025.0 on CPU/GPU; extended to VLMs in 2026.1.

```python
pipe = ov_genai.LLMPipeline(model_dir, "GPU", prompt_lookup=True)
config = ov_genai.GenerationConfig(num_assistant_tokens=5,
                                    assistant_confidence_threshold=0.4)
```

### #7. Always set `CACHE_DIR` on GPU
The most overlooked free win: kernel JIT compile on Intel GPU costs 5-20 seconds per fresh model load. With `CACHE_DIR` set, second+ runs of the same model+device combo skip the compile entirely. This is the difference between a 20-second cold start and a 2-second one.

```python
core.set_property("GPU", {"CACHE_DIR": "./ov_gpu_cache"})
```

### #8. KV cache eviction + sparse attention prefill for long context
For workloads with >2K input contexts, set `CacheEvictionConfig` to bound KV memory and `SparseAttentionConfig` for prefill. Together they let Phi-3-mini run 16K context on Arc 140V and 8B-class models at 32K context on B390 without OOM.

```python
sched.cache_eviction_config = ov_genai.CacheEvictionConfig(
    start_size=32, recent_size=128, max_cache_size=2048,
    aggregation_mode=ov_genai.AggregationMode.NORM_SUM)
sched.sparse_attention_config = ov_genai.SparseAttentionConfig(
    mode=ov_genai.SparseAttentionMode.TRISHAPE,
    num_last_dense_tokens_in_prefill=128,
    num_retained_start_tokens_in_cache=32,
    num_retained_recent_tokens_in_cache=128)
```

---

## "Maybe-also" knobs (worth a single test each)

- **GPU streams=2**: sometimes helps THROUGHPUT mode on dGPU (charlie); rarely on iGPU (alpha).
- **GPU_QUEUE_THROTTLE=HIGH + GPU_HOST_TASK_PRIORITY=HIGH**: ~5-15% latency win at higher power on Lunar Lake.
- **MoE quant via `--quant-mode int4_f8e4m3`**: cb4 / FP8 LUT codebook compression (OV 2026.0); needs MoE model.
- **GGUF reader path (OV 2025.2)**: skip optimum-cli entirely for prototypes — `LLMPipeline("model.gguf", "GPU")` works.
- **HETERO:GPU,NPU**: rarely best (host-transfer overhead), but worth one test on alpha for power-efficiency.
- **NPU acceleration for short prompts**: lower power, sometimes lower TTFT than iGPU on Lunar Lake.

---

## Knobs to AVOID

- **`--disable-stateful`**: kills GPU LLM perf. Only use for legacy code that needs raw KV inputs/outputs.
- **`KV_CACHE_PRECISION=f16`**: undoes the 2024.6+ default. Only use to debug a U8-accuracy regression (and try OV 2025.3+ first since per-channel key-cache fixed most U8 accuracy gaps).
- **NPU for >8K context** until OV 2025.3+ (longer context support landed there).
- **`GGML_SYCL_F16=ON` on Battlemage** without `GGML_SYCL_DISABLE_OPT=1` (corrupted output bug — issue #21893).
- **Manual graph rewriting**: rely on the OV plugin's automatic SDPAToPagedAttention + make_stateful + fuse_cache_reorder transforms.

---

## Suggested first campaign (1 day budget)

For each target (alpha, charlie):
1. Baseline: `OVModelForCausalLM` + Llama-3-8B-INT4 (default OV 2025.4, no plugin config)
2. Switch to `LLMPipeline` (no other changes) — measure delta
3. Add KV=u8 + DynQuant=32 + CACHE_DIR — measure delta
4. Add SchedulerConfig (CB+prefix caching + dynamic_split_fuse) — measure delta
5. Add speculative decoding with Llama-3.2-1B draft — measure delta
6. Try Prompt Lookup on RAG-style prompts — measure delta

Expected cumulative impact at end: 2-4x speedup vs baseline on charlie, 1.5-3x on alpha. Report which knobs contributed most so we know the marginal value of each.

---

## Files in this literature survey

| File | Topic |
|------|-------|
| `openvino-2024.4-paged-attention-discrete-gpu.md` | First PA on dGPU; XMX kernels |
| `openvino-2024.5-kv-cache-compression-igpu.md` | KV compress + dyn quant on iGPU; spec decode |
| `openvino-2024.6-battlemage-and-u8-kvcache-default.md` | Battlemage support; U8 KV becomes default |
| `openvino-2025.0-asymmetric-int8-kvcache-and-prompt-lookup.md` | INT8 KV (CPU) + Prompt Lookup |
| `openvino-2025.1-paged-attention-default-on-gpu.md` | PA + CB enabled by default on GPU |
| `openvino-2025.2-xmx-dynamic-quant-snapkv.md` | XMX dyn quant; GGUF reader; SnapKV |
| `openvino-2025.3-key-cache-per-channel-and-snapkv-prefill.md` | Per-channel KV; sparse attention prefill; NPU 8K |
| `openvino-2025.4-prefix-caching-int8-dynquant-moe.md` | Prefix caching + INT8 dyn quant on GPU; MoE |
| `openvino-2026.0-2026.1-fp8-lut-llamacpp-backend.md` | FP8 LUT; NPU spec decode; OV-as-llama.cpp-backend |
| `optimum-intel-cli-export-flags.md` | Complete `optimum-cli export openvino` flag map |
| `optimum-intel-v1.21-xpu-pagedattention.md` | IPEX PA in optimum |
| `openvino-gpu-plugin-properties.md` | All GPU plugin properties + ov::set_property syntax |
| `openvino-genai-llmpipeline-vs-ovmodel-and-cb-config.md` | Runtime API choice + scheduler / decoding |
| `openvino-sdpa-pagedattention-stateful-transformations.md` | The graph passes that actually make GPU LLMs fast |
| `ipex-2.5-2.8-xpu-llm-stack.md` | IPEX XPU evolution and EOL |
| `llamacpp-sycl-intel-arc-2025-2026.md` | Competitor llama.cpp SYCL state |
| `intel-npu-acceleration-library-eol-and-genai.md` | NPU lib superseded by OV NPU plugin |
| `intel-neural-compressor-3x-fp8-lut-mxfp4.md` | INC FP8/MXFP4/NVFP4 path |
| `vllm-2025-2026-xpu-features.md` | vLLM techniques relevant to OV / port-watch |
| `exo-explore-recent-state-2026.md` | Competitor exo: maintenance mode |
