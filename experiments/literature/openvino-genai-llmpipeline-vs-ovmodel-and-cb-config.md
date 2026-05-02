# OpenVINO GenAI: LLMPipeline vs OVModelForCausalLM, scheduler config, decoding modes

**Released:** GenAI library 2024.4 (first release as `openvino_genai`) → 2026.1
**What changed:** OV GenAI's `LLMPipeline` is the new canonical entry point for LLM inference; `OVModelForCausalLM` (in optimum-intel) is the legacy HF-API-shaped wrapper. They share the underlying OV runtime but `LLMPipeline` exposes more knobs and is faster on GPU because it controls KV cache layout end-to-end.

**Headline perf claim (if any):** Public docs frame `LLMPipeline` as "lightweight deployment" with "enhanced internal optimizations"; specific A/B numbers vs `OVModelForCausalLM` are not published, but in practice the gap is dominated by:
1. **Continuous batching default** (since OV 2025.1 GPU plugin) — `LLMPipeline` exposes `SchedulerConfig`, `OVModelForCausalLM` does not.
2. **KV cache reuse across `start_chat()` / `finish_chat()`** turns — only in `LLMPipeline`.
3. **Speculative decoding & Prompt Lookup**: only in `LLMPipeline`.
4. **Explicit static reshape** for fixed batch+seq: `LLMPipeline` exposes; `OVModelForCausalLM` partial.
5. **Token Eviction / SnapKV / Sparse Attention**: only `LLMPipeline`.

When to use which:
- `OVModelForCausalLM`: prototyping, code that already uses `transformers`, you want `.generate()` semantics that match HF.
- `LLMPipeline`: production inference, especially on GPU, especially batched/concurrent.

**How to use it from optimum-intel / OV runtime:**
```python
import openvino_genai as ov_genai

# === Basic LLMPipeline on GPU ===
pipe = ov_genai.LLMPipeline("Llama-3-8B-int4-ov", "GPU")
config = ov_genai.GenerationConfig(max_new_tokens=128, do_sample=False)
print(pipe.generate("Once upon a time,", config))

# === SchedulerConfig (continuous batching, prefix caching) ===
sched = ov_genai.SchedulerConfig()
sched.cache_size = 4                   # GB; or num_kv_blocks for fine control
sched.max_num_batched_tokens = 4096
sched.max_num_seqs = 32
sched.dynamic_split_fuse = True        # split prefill across decode steps
sched.enable_prefix_caching = True
# Plus optional:
sched.cache_eviction_config = ov_genai.CacheEvictionConfig(
    start_size=32, recent_size=128, max_cache_size=2048)
sched.sparse_attention_config = ov_genai.SparseAttentionConfig(
    mode=ov_genai.SparseAttentionMode.TRISHAPE,
    num_last_dense_tokens_in_prefill=128)
pipe = ov_genai.LLMPipeline("Llama-3-8B-int4-ov", "GPU", scheduler_config=sched)

# === Speculative decoding ===
pipe = ov_genai.LLMPipeline("Llama-3-8B-int4-ov", "GPU",
    draft_model=ov_genai.draft_model("Llama-3-2-1B-int4-ov", "GPU"))
config.num_assistant_tokens = 5
config.assistant_confidence_threshold = 0.4

# === Prompt Lookup ===
pipe = ov_genai.LLMPipeline("Llama-3-8B-int4-ov", "GPU", prompt_lookup=True)
config.num_assistant_tokens = 5

# === Chat mode (uses prefix caching effectively) ===
pipe.start_chat()
print(pipe.generate("Hi! What is the capital of France?"))
print(pipe.generate("And of Germany?"))
pipe.finish_chat()

# === Plugin properties through GenAI ===
plugin_config = {
    "GPU": {
        "KV_CACHE_PRECISION": "u8",
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
        "CACHE_DIR": "./ov_gpu_cache",
        "INFERENCE_PRECISION_HINT": "f16",
    }
}
pipe = ov_genai.LLMPipeline("model", "GPU", **plugin_config["GPU"])
```

For comparison the optimum path:
```python
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("model_dir")
model = OVModelForCausalLM.from_pretrained("model_dir", device="GPU",
    ov_config={"KV_CACHE_PRECISION": "u8", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32"})
out = model.generate(tok("Hi", return_tensors="pt").input_ids, max_new_tokens=128)
```

**Intel GPU applicability:** HIGH for both Arc 140V and Arc B390. **Make `LLMPipeline` the default on Tahoma**; only fall back to `OVModelForCausalLM` for HF-API compatibility.

**Open hypothesis it generates for us:** On charlie (B390) and alpha (140V) run identical workload (Llama-3-8B-INT4, 100 prompts of varied length, batch=1 streaming) via (a) `OVModelForCausalLM.generate()`, (b) `LLMPipeline.generate()` with default config, (c) `LLMPipeline.generate()` with full SchedulerConfig + prefix caching. Hypothesis: (b) is 1.0-1.1x (a); (c) is 1.4-2.0x (a) on the chat-mode prompts due to prefix cache hits.

Sources:
- https://huggingface.co/blog/deploy-with-openvino
- https://docs.openvino.ai/2025/api/genai_api/_autosummary/openvino_genai.SchedulerConfig.html
- https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md
- https://deepwiki.com/openvinotoolkit/openvino.genai/7.5-configuration-and-advanced-usage
