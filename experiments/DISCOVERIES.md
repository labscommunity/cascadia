# Discoveries

Novel / surprising / undocumented findings from this autolab session. Format mirrors rainier's `DISCOVERIES.md`.

---

## DISCOVERY #1 — `openvino_genai.LLMPipeline` is **10×** faster than `optimum.intel.OVModelForCausalLM` on Intel GPU for INT4 LLMs

**Setup:** alpha (Arc B390 Battlemage) running OpenVINO 2026.1.0 + openvino-genai 2026.1.0; pre-exported INT4 IR at `C:\cascadia\models\llama-3.1-8b-int4`; greedy generation of 64 tokens; prompt "What is the capital of France?".

**Finding:** decode throughput jumps from **8.89 tok/s** (`OVModelForCausalLM`) to **96.41 tok/s** (`LLMPipeline`) on the same model file on the same hardware. **+10.8× speedup.** Confirmed by re-runs; not a measurement glitch.

**Why this is worth saving:**

- The published expectation from the OV 2024.5+ release notes was a 1.4-2.0× win. We got 5-10× more than that.
- The standard advice "use `OVModelForCausalLM` for HuggingFace compatibility" is *catastrophic* on GPU. Even Intel's own docs hedge here.
- The optimum-intel path apparently does not engage the GPU-default optimisations (PagedAttention, U8 KV cache, XMX dynamic quant) introduced in OV 2024.6 → 2025.4 for IR files that were exported before those versions landed. `LLMPipeline` applies the runtime SDPAToPagedAttention pass at compile time and works against any IR.
- This single change is bigger than any individual quantisation, kernel, or scheduling perf knob we could tune.

**Source experiments:** `experiments/c1-llmpipeline/`. Both c0-1b (8.89) and c1-1 (96.41) used the same model dir, same prompt, same target node, same OV version.

**Action items:**
1. Charlie has an OV/genai version mismatch (genai 2026.0.0 vs OV 2026.2.0.dev) — install matching versions and re-confirm the win on Lunar Lake.
2. Replace the `ov-optimum` engine in tahoma with a `LLMPipeline`-based engine as the recommended single-stage GPU path. Keep the optimum engine for non-GPU and for engine types where genai doesn't have a wrapper yet.
3. Re-run *every* OV-engine benchmark in `baselines.md` against the new path — most "baseline" numbers are now wildly out of date.

---
