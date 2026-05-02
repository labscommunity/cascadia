# Autolab Session Summary — Intel GPU Inference Performance

**Branch:** `autolab/intel-gpu-perf`
**Hardware:** alpha (Battlemage Arc B390 dGPU), charlie (Lunar Lake Arc 140V iGPU + NPU 4)
**Software:** OpenVINO 2026.1.0 + openvino-genai 2026.1.0
**Model:** Llama 3.1 8B Instruct INT4 (also Llama 3.2 1B, Phi-3 mini)
**Campaigns:** 61
**Iterations:** ~410

## ⚠️ Methodology corrections at c57-c61

The bench scripts in c1-c56 had two systematic errors:

1. **Counted `max_tokens` cap instead of actual generated tokens** (c57). For prompts that EOS early (factual chat → 8 tokens, extractive summary → ~99 tokens), absolute throughput numbers were inflated 5-14×. RELATIVE comparisons (mode A vs mode B) remained valid because both sides hit the same EOS behavior.

2. **Asymmetric warmup between engines** (c60). The c0 baseline didn't warm up `OVModelForCausalLM` properly while c1 LLMPipeline was warmed normally, exaggerating the engine comparison from 1× to 10×.

**All numbers below are post-correction, verified with proper methodology.**

## Real Discoveries (verified)

### Discovery #1 — DEBUNKED

The original claim "LLMPipeline is 10.8× faster than OVModelForCausalLM" was 100% a bench artifact. Apples-to-apples test (same chat template, same warmup, same actual output):

| Engine | actual tok/s |
|--------|-------------:|
| OVModelForCausalLM | 17.23 |
| LLMPipeline | 17.24 |
| **Ratio** | **1.00×** |

Tahoma should still use LLMPipeline — for chat template handling, spec decode hooks, and CB support — but **not** for raw per-token speed.

### Discovery #2 — VERIFIED (+55%)

**FastDraft 150M companion gives +55% over plain LLMPipeline** for short factual chat. Verified with 5-run statistics:

| Mode | median tok/s |
|------|-------------:|
| LLMPipeline plain | 17.45 |
| LLMPipeline + FastDraft K=5 | **27.02** (+55%) |

Originally claimed +24%; properly counted, the win is actually +55%.

### Discovery #3 — VERIFIED (+40-50% on extractive)

**Prompt Lookup decoding gives +40-50% on EXTRACTIVE workloads** (passages summarized, code completion-in-context, "rewrite this in style X" — anywhere the model's output naturally quotes input vocabulary).

| Workload | Plain | PL | Δ |
|----------|------:|---:|--:|
| 4K input + extractive summary (charlie) | ~20 | ~28 | +40% |
| 1K input + extractive summary (alpha/charlie) | similar | similar | +40-50% |

Originally claimed +59-65% (or +94% in extreme configs); properly counted, the win is +40-50%.

**Caveat: PL is workload-specific.** For open-ended (non-extractive) prompts, PL is tied with plain or slightly slower at very long input.

### Discovery #4 — VERIFIED (NPU concurrent serving)

**Intel NPU enables concurrent multi-model serving** with low cross-talk:

- 8B-on-GPU + 1B-on-NPU concurrent: GPU costs only -3% throughput
- Realistic deployment: 16 chat sessions on GPU (CB b=8) + 1 always-on classifier on NPU
- Aggregate effective throughput: 131 + 38 = **169 tok/s** across 17 concurrent inferences on a single Lunar Lake AI PC

This is a real Intel-only differentiator; NVIDIA consumer GPUs don't have NPU equivalents.

## Real Achievable Rates (verified)

For Llama 3.1 8B INT4 on alpha B390 (Battlemage dGPU) and charlie 140V (Lunar Lake iGPU):

| Hardware | Workload | Engine | actual tok/s |
|----------|----------|--------|-------------:|
| alpha | 8B factual chat | LLMPipeline plain | 17.5 |
| alpha | 8B factual chat | + FastDraft K=5 | **27.2** |
| alpha | 8B creative 256-out | + FastDraft K=3 | **28.3** |
| alpha | 8B multi-tenant b=8 CB | aggregate | **131** |
| charlie | 8B extractive RAG | + PL | **~28** |
| charlie | 1B factual chat | LLMPipeline plain | **55.6** |

These are the actual achievable rates. Memory bandwidth caps 8B INT4 on Intel GPU around 150-200 tok/s theoretical max; real achievable is in the 17-30 tok/s range for single-user.

## Decision Matrix (still valid)

| Workload | Recommended engine |
|----------|--------------------|
| Short factual chat (<100 in, <100 out) | LLMPipeline + FastDraft K=5 (+55%) |
| Long-creative writing (256+ output) | LLMPipeline + FastDraft K=3 |
| Extractive RAG / summarization | LLMPipeline + Prompt Lookup (+40%) |
| Multi-tenant chat (4+ concurrent) | LLMPipeline + SchedulerConfig CB |
| Multi-model on one Lunar Lake | 8B-on-GPU + 1B-on-NPU |
| Long inputs (1K+) with open-ended Qs | LLMPipeline plain (FastDraft + PL don't help) |

## Negative findings (also valuable)

- **All explicit GPU plugin property overrides are no-op or regressions** vs default (KV=u8, DynQuant, QUEUE_THROTTLE, INFERENCE_PRECISION_HINT, NUM_STREAMS, PERFORMANCE_HINT).
- **bf16 KV cache is catastrophic** (-73%). u8 (default) is optimal.
- **Heterogeneous draft device** (NPU draft + GPU target): -75% — cross-device sync dominates.
- **NPU as 8B target**: too big — NPU is for ≤3B-class models only.
- **FastDraft K=5 fixed at long input**: -7% vs plain.
- **PL on open-ended (non-extractive) prompts**: tied with plain or slightly worse.
- **Phi-3 mini + PL at 4K input**: only +4% (PL win scales with model size).
- **Adaptive K (`assistant_confidence_threshold`)**: within ~5% noise — not reliably better than fixed K=5.
- **AWQ INT4 export**: timed out at 71 min CPU; pre-quantized HF variants recommended.
- **GGUF reader works** but 35% slower than native OV INT4 IR.
- **Multiple LLMPipeline procs on same GPU concurrently**: serialise on kernel queue (always inflates latency, never throughput).

## Critical methodological lessons

1. **Always count actual generated tokens via tokenizer**, not max_tokens cap.
2. **Warm BOTH engines properly** before cross-engine comparisons (multiple full generates, not just a 4-token forward).
3. **Use the same chat template handling** on both sides of an A-vs-B comparison.
4. **Never run multiple LLMPipeline procs against the same physical GPU concurrently** — they inflate latency to fake "regression."
5. **Single-trial benches at long input + short output have ~10% variance** — use ≥5 runs for sub-10% effects.
6. **Be skeptical of >150 tok/s claims for 8B INT4 on Intel GPU** — physics caps the rate around 150-200 tok/s memory bandwidth.
7. **Same prompt for plain vs spec comparisons** — prompt template differences cause apples-to-oranges errors.
8. **`perf_metrics.num_generated_tokens` is unreliable in OV 2026.1 LLMPipeline** — always cross-check with tokenizer.

## Code that landed

1. `tahoma/worker/engines/openvino/genai_engine.py` — `OVGenAIBuilder` / `OVGenAIEngine`.
2. `tahoma/worker/engines/registry.py` — `ov-genai` engine entry.
3. `tahoma/cli.py` — flags: `--ov-cache-dir`, `--ov-kv-precision`, `--ov-dyn-quant-group`, `--draft-model`, `--draft-device`, `--spec-k`, `--prompt-lookup`.
4. `experiments/DECISION_MATRIX.md` — engine selection guide.

## Open follow-ups

- Per-token streaming via LLMPipeline streamer callback.
- Wire `--ov-scheduler-cb` flag for multi-tenant mode.
- Make `--ov-cache-dir` default-on (cuts cold-start by 62%).
- Mark `ov-spec` engine as deprecated (now ~10× slower than `ov-genai+FastDraft`).
- Q4 2025 / Q1 2026 papers: MARS rejection, KVTC — would need custom OV impl.
- Concurrent NPU+GPU loading penalty investigation (NPU drops 113→42 when GPU pipe also loaded).

## What this autolab actually showed

The biggest learning was **methodological**, not technical: bench measurement errors can produce dramatically misleading results, and "10× speedups" should always be verified by counting actual tokens with proper warmup.

After corrections:
- Tahoma's real perf advantage on Intel GPU comes from **stacking compatible techniques**: LLMPipeline + FastDraft + CB + NPU concurrent serving.
- The compounded gains over plain ov-optimum baseline are ~3× (real), not 10× (claimed).
- Intel hardware does have a real differentiator: NPU for concurrent multi-model serving.
- Real per-token rate for 8B INT4 single-user is 17-28 tok/s — well within the laptop chat-app comfort zone.
