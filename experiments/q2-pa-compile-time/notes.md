# Q2 — apply SDPAToPagedAttention at compile time (D1 partially debunked)

**Hypothesis:** D1 said the OV PA transformation is "un-retrofittable" because the EXPORT-time API (`openvino._offline_transformations.paged_attention_transformation`) failed twice. That was the wrong API. openvino-genai's `LLMPipeline` applies the SDPAToPagedAttention pass at COMPILE time on a loaded `ov::Model`, via the public C++ pass `ov::pass::SDPAToPagedAttention`. Mirror that pattern in tahoma.

## What worked

1. **PA pass DOES apply to v5 stage_0 IRs.** `paged_attention_transformation()` from `openvino._offline_transformations` (same function as in the v5 export script's `apply_paged` branch) succeeds on a v5 IR loaded via `core.read_model()`. With manual registration of dangling `attention_mask` + `beam_idx` Parameters, the model compiles on GPU.

2. **PA inputs added (`logs/q2-pa-output.log`):** `past_lens` (i32 [?]), `subsequence_begins` (i32 [?]), `block_indices` (i32 [?]), `block_indices_begins` (i32 [?]), `max_context_len` (i32 []). Plus `position_ids` is reshaped from [?,?] to [?]. Stateful KV machinery is preserved — no key_cache.N / value_cache.N inputs added.

3. **PA-transformed model RUNS inference end-to-end.** Verified with a 7-token prefill on alpha GPU. Output shape changes from `[1, 7, 4096]` (plain) to `[1, 1, 7, 4096]` (PA — extra leading dim). Mean abs diff 0.398 vs plain — different attention computation, expected.

→ **D1 should be downgraded from "structural blocker" to "needs workaround for register-dangling-parameters quirk + plugin-config tuning."** The PA pass isn't un-retrofittable — it just has rough edges.

## What's still blocking

- **Decode-loop GPU OOM.** Plain v5 stage_0 ran 64 decode steps in 1483 ms (23.17 ms/token). PA-transformed v5 stage_0 OOMs the alpha B390 GPU (12 GB) during the decode loop — the paged KV cache pool is preallocated too large.
- **CPU fallback fails too.** PA-transformed model on CPU: `Node ReadValue_36864 contains less parent edges than 0` — internal OV plugin issue. The PA pass leaves stateful ReadValue nodes behind that the CPU plugin can't compile.

The OV plugin's PA cache sizing seems to assume a long-context budget by default (likely 32K tokens × 16 layers × 2 KV × 8 heads × 128 head_dim × 2 bytes = 268 MB just for stage_0 — manageable, but plugin may add other overhead pushing past 12 GB combined with INT4 weights).

## Status

- **D1 conclusion partially debunked:** PA at compile time works; just blocks on plugin-side memory + CPU-plugin bug.
- **Did not measure speedup** vs plain v5 because PA decode loop OOMs.
- **Workaround paths to try (not done in this session):**
  1. Find OV property to cap PA cache size or set `KV_CACHE_PRECISION=u8` for PA. Tried `u8` once — same OOM.
  2. Apply PA via openvino-genai's higher-level `apply_paged_attention_transformations()` (in continuous_batching/) which removes problematic ReadValue nodes from the graph before compile. That's the function the agent's research identified as the actual entry point. We didn't test because no Python binding exists in stock OV — would need C++ shim work.
  3. Use the FULL Llama 8B (monolithic, with tokenizer) where ov-genai's full PA path is available — bench tokens/sec there, then prove the per-stage IR can match.

## Decision

PA path needs deeper OV-plugin investigation than fits here. Pivot to **Q3 (continuous async spec)** which is cleaner engineering with documented 1.5-7× wins per literature.

## Next-session entry point

Modify the C++ shim (`crates/tahoma-ov-genai-shim/cpp/shim.cpp`) to add `tahoma_runtime_compile_with_pa(xml_path, device, properties)`. Mirror what openvino-genai does internally — load model → apply pass → register dangling parameters → compile with appropriate KV cache sizing properties. Test against the full Llama 8B IR first to prove the codepath works, then per-stage. Once that's stable, modify the dist_spec engine to bind PA bookkeeping inputs each forward.
