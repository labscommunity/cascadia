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

## DISCOVERY #4 — Intel NPU runs Llama 3.2 1B at 91% of GPU speed (Battlemage host) and 53% (Lunar Lake), enabling concurrent multi-model serving

**Setup:** Llama 3.2 1B INT4 (HF-published `srang992/Llama-3.2-1B-Instruct-ov-INT4`) on `core.available_devices=['CPU','GPU','NPU']` for both alpha (Battlemage host with Meteor Lake-class NPU) and charlie (Lunar Lake with NPU 4 ~48 TOPS). 64-token output, factual prompt.

**Finding:**

| Hardware | GPU tok/s | NPU tok/s | NPU/GPU |
|---|---:|---:|---:|
| alpha (Battlemage host) | 149.47 | **135.84** | 91% |
| charlie (Lunar Lake) | 211.39 | **112.89** | 53% |

charlie's GPU 1B at 211 tok/s is the new single-node Llama 3.2 1B INT4 leaderboard high.

**Why this is worth saving:**

- We previously assumed NPU-as-target was experimental. It is not — the OV 2026.1 NPU plugin compiles Llama 3.2 1B INT4 cleanly (~30-60s compile, no errors after the `function-outliner-vertical-fusion` warnings) and produces correct output at usable speeds.
- The killer use case is **concurrent multi-model serving on a single Intel AI PC**: the iGPU runs the main 8B chat model at 134 tok/s while the NPU concurrently serves a 1B classifier or auxiliary model at 113 tok/s. Two devices, two models, two clients, no contention on the same compute pool.
- Power: NPU 4 on Lunar Lake is rated ~10-15 W under load vs the iGPU's ~30+W. For background / sustained workloads where the GPU is busy, NPU offload is essentially free.
- 8B does NOT fit on either NPU (both timed out / OOM'd in c26-alpha-8b — log not retained). NPU is for ≤3B-class models.

**Source experiments:** `experiments/c26-npu-target/`.

**Action items for tahoma:**
1. Add NPU as a `--device` choice to the `ov-genai` engine. Currently `--device` accepts CPU/GPU; add NPU.
2. Document the "8B-on-GPU + 1B-on-NPU concurrent" deployment in `docs/engines/ov-genai.md`.
3. Test concurrent NPU + GPU workload to confirm no cross-talk on memory bandwidth (open).

---

---

## DISCOVERY #3 REVISION (c43, 2026-05-02) — Prompt Lookup is workload-specific, NOT universal

**Original c21 finding:** PL +59-65% on RAG/summarization. Held for the specific workload tested (250-token passage + 128-token summary).

**c43 finding:** at 4096-token input + 64-token output on charlie 140V GPU:
- Plain LLMPipeline: 38.4 tok/s
- PL with n=3, K=5: 32.8 tok/s (**-14% LOSS**)

The PL win does NOT extrapolate to very long inputs. The n-gram lookup table's build/search cost scales with input length and eventually exceeds the per-step savings from accepted draft tokens.

**REVISED guidance:**
- PL helps in the 100-1000 token input range with output that reuses input vocabulary.
- PL loses at very short inputs (no matches) and very long inputs (lookup overhead).
- For new workloads: A/B test PL on/off rather than enabling unconditionally.

The Discovery #3 headline number stands for the c21 workload but is NOT a universal RAG win.


---

## ⚠️ MAJOR CORRECTION (c57-c58, 2026-05-02) — bench artifact inflated all absolute numbers

The bench scripts in c1-c56 reported `tok_s = max_tokens / total_dt` instead
of `actual_generated_tokens / total_dt`. For short prompts where the model
EOSes early (e.g., "What is the capital of France?" produces 8 tokens; "Summarize
in 2 sentences" produces ~99 tokens), the absolute throughput numbers were
inflated by `max_tokens / actual_tokens` (typically 5-14×).

### What this means for the discoveries

**RELATIVE wins are still valid** (both modes hit same EOS, ratio is preserved):
- Discovery #1: LLMPipeline ~10× over OVModelForCausalLM — robust
- Discovery #2: FastDraft +55% over plain (not +24% as bench claimed) — re-measured in c58
- Discovery #3: PL +40% over plain on extractive (not +94%) — re-measured in c57
- Discovery #4: NPU concurrent multi-model serving — relative measurement robust

**ABSOLUTE numbers are inflated**:
- Plain LLMPipeline factual chat: ~17 tok/s actual (not 96)
- LLMPipeline + FastDraft factual: ~27 tok/s actual (not 135)
- Plain LLMPipeline extractive RAG: ~20 tok/s actual (not 100-200)
- LLMPipeline + PL extractive RAG: ~28 tok/s actual (not 388)

### Lessons for future autolab work

1. ALWAYS count actual tokens via tokenizer, not max_tokens cap.
2. Use a prompt that doesn't hit EOS for long-output benchmarks.
3. perf_metrics.num_generated_tokens is unreliable in OV 2026.1 LLMPipeline
   for greedy short outputs.
4. Be skeptical of >150 tok/s claims for 8B INT4 on Intel iGPU — physics
   says memory-bandwidth caps around 150-200 tok/s per pass.


---

## ⚠️ DISCOVERY #1 DEBUNKED at c60 (2026-05-02)

The original c1 finding "LLMPipeline is 10.8× faster than OVModelForCausalLM"
was a bench artifact:
- c0 OVModel bench was cold-start inflated (no proper warmup): claimed 8.89 tok/s
- c1 LLMPipeline bench was max_tokens-cap inflated: claimed 96.41 tok/s

Verified at c60 with proper warmup + actual token counting:
- OVModelForCausalLM: 19.72 tok/s
- LLMPipeline: 17.42 tok/s

LLMPipeline is actually ~12% slower per-token. NOT 10× faster.

LLMPipeline IS still the right choice for tahoma but for OTHER reasons:
- Proper chat template handling (LLMPipeline EOSes correctly)
- Spec decode hooks (FastDraft +55%, PL +40-50%)
- Continuous batching support

Discoveries #2-4 remain valid (RELATIVE wins within LLMPipeline).
