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

**Finding:** **119.24 tok/s** vs plain LLMPipeline's 96.41 — **+23.7%**. Through the tahoma `ov-genai --draft-model` engine: **134.90 tok/s** (within 13% noise of raw, all warmer-cache effects). **+15.2× over the original `ov-optimum` baseline of 8.89.**

This is the new tahoma single-GPU leaderboard high for Llama 3.1 8B INT4.

**Generalisation:** validated across model families. Phi-3-mini-128k INT4 + 50M FastDraft K=5: **43.90 tok/s** vs 32.18 plain (+36%).

**K-sweep findings:**
- Short factual (64 tok): K=5-10 best (~119 tok/s on alpha)
- Long creative (256 tok): K=3 best (27.12 tok/s) — over-speculation hurts
- `assistant_confidence_threshold` (dynamic K): -22% vs fixed K=5 — too conservative

**Why this is worth saving:**

- Our previous spec-decode tests with a 1B-INT4 draft (Llama 3.2 1B) gave only +5% at K=10 (within noise) on the same workload — the 1B draft's compute cost cancelled the savings.
- Intel publishes FastDraft companions specifically *trained* for spec decode against a target. The 150M draft is the right size for an 8B target on Battlemage: small enough that draft compute is cheap (~1 ms per round), large enough that accept rate stays high.
- The previous `ov-spec` engine in tahoma (35 tok/s on main's saved baseline) is now strictly worse than `ov-genai + FastDraft`.

**Source experiments:** `experiments/c18-fastdraft/`.

---

## DISCOVERY #3 — Prompt Lookup decoding gives **+58-65%** on RAG / summarization workloads at zero draft cost

**Setup:** Llama 3.1 8B INT4 on alpha B390 GPU and charlie 140V GPU. LLMPipeline with `prompt_lookup=True`, `num_assistant_tokens=5`, `max_ngram_size=3`. 128-token output. Prompt: a passage about distributed inference followed by "Summarize the passage above in 2 short sentences." (~50% of output tokens reuse input vocabulary.)

**Finding:**

| Hardware | No lookup | + prompt_lookup | Δ |
|---|---|---|---|
| alpha B390 | 57.69 tok/s | **91.57** | **+58.7%** |
| charlie 140V | 66.16 tok/s | **108.82** | **+64.5%** |

**Why this is worth saving:**

- Prompt Lookup works by hash-table-matching the last N-gram of the generated sequence against substrings of the input prompt. When the output reuses input text (RAG, summarization, code completion-in-context, "rewrite this"), accept rate is high.
- Unlike FastDraft, **there is no draft-model compute cost** — the lookup is a constant-time hash table operation. The amortisation against the target's per-token cost is therefore very favourable.
- For RAG / summarization workloads, this is a strictly better choice than FastDraft. The synthesis literature mentioned this but quantified the win as "free acceleration" without specifying the magnitude on Intel GPUs; we now have measured 58-65%.
- The flag is a single `prompt_lookup=True` on `LLMPipeline()` plus `max_ngram_size=N` on the GenerationConfig — three lines of code change, no model download.

**Source experiments:** `experiments/c21-prompt-lookup/`.

**Action items for tahoma:**
1. Add `--prompt-lookup` flag to the `ov-genai` engine (mutually exclusive with `--draft-model`).
2. Document the three drafting modes in `docs/engines/ov-genai.md`:
   - chat / short factual → FastDraft + K=5
   - long creative gen → FastDraft + K=3
   - RAG / summarization → prompt_lookup + max_ngram_size=3
3. Auto-detect mode from request? (e.g. `RAG: True` flag on the API).

---
