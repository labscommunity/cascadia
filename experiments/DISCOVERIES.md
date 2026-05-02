# Discoveries

Novel / surprising / undocumented findings from this autolab session. Format mirrors rainier's `DISCOVERIES.md`.

---

## DISCOVERY #1 — `openvino_genai.LLMPipeline` is **10×** faster than `optimum.intel.OVModelForCausalLM` on Intel GPU for INT4 LLMs

**Setup:** alpha (Arc B390 Battlemage) running OpenVINO 2026.1.0 + openvino-genai 2026.1.0; pre-exported INT4 IR at `C:\cascadia\models\llama-3.1-8b-int4`; greedy generation of 64 tokens; prompt "What is the capital of France?".

**Finding:** decode throughput jumps from **8.89 tok/s** (`OVModelForCausalLM`) to **96.41 tok/s** (`LLMPipeline`) on the same model file on the same hardware. **+10.8× speedup.** Confirmed by re-runs on charlie too (8.83×).

**Why this is worth saving:**

- The published expectation from the OV 2024.5+ release notes was a 1.4-2.0× win. We got 5-10× more than that.
- The standard advice "use `OVModelForCausalLM` for HuggingFace compatibility" is *catastrophic* on GPU.
- The optimum-intel path apparently does not engage GPU-default optimisations (PagedAttention, U8 KV cache, XMX dynamic quant) for IRs exported pre-2025.1. `LLMPipeline` applies the runtime SDPAToPagedAttention pass at compile time and works against any IR.

**Source experiments:** `experiments/c1-llmpipeline/`, `experiments/c3-ov-genai-engine/`.

---

## DISCOVERY #2 — Intel's official FastDraft 150M companion gives **+24% on top of LLMPipeline** for Llama 3.1 8B INT4 short responses

**Setup:** alpha (Arc B390 Battlemage), Llama 3.1 8B INT4 target (same dir as #1), `OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov` as draft (155 MB, INT8 OV-format), `LLMPipeline(draft_model=...)`, greedy K=5, 64-token output, factual prompt.

**Finding:** **119.24 tok/s** vs plain LLMPipeline's 96.41 — **+23.7%**. **+13.4× over the original `ov-optimum` baseline of 8.89.**

This is the new tahoma single-GPU leaderboard high for Llama 3.1 8B INT4.

**Why this is worth saving:**

- Our previous spec-decode tests with a 1B-INT4 draft (Llama 3.2 1B) gave only +5% at K=10 (within noise) on the same workload — the 1B draft's compute cost cancelled the savings.
- Intel publishes FastDraft companions specifically *trained* for spec decode against a target. The 150M draft is the right size for an 8B target on Battlemage: small enough that draft compute is cheap (~1 ms per round), large enough that accept rate stays high.
- The 1B draft equivalence breaks down only at long-gen (256+ tok), where per-token target cost dominates and draft size becomes irrelevant.
- The previous `ov-spec` engine in tahoma (35 tok/s on main's saved baseline) is now strictly worse than `ov-genai + FastDraft`. ov-spec is no longer the recommended spec-decode path.

**Source experiments:** `experiments/c18-fastdraft/`.

**Action items:**
1. Wire `--draft-model` into the `ov-genai` tahoma engine and document FastDraft as the recommended pairing for Llama 3.1 8B.
2. Search for FastDraft equivalents for other model families (Phi-3, Qwen, Gemma) — Intel publishes a few.
3. Mark `ov-spec` engine as deprecated in tahoma's engine listing.

---
