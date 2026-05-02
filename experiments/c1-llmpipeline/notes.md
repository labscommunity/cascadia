# c1: LLMPipeline result

## Numbers

| ID | Node | Path | Decode tok/s | Decode time | Δ vs c0 baseline |
|---|---|---|---|---|---|
| c0-1b | alpha B390 | `optimum.intel.OVModelForCausalLM` | 8.89 | 7.20 s / 64 tok | (baseline) |
| **c1-1** | **alpha B390** | **`openvino_genai.LLMPipeline`** | **96.41** | **0.66 s / 64 tok** | **+10.8×** |
| c1-2 | charlie 140V | `openvino_genai.LLMPipeline` | (n/a) | DLL ABI mismatch | — |

## Why this is a 10× and not the literature's 1.4-2.0×

The synthesis predicted 1.4-2.0×. We got 10.8×. Hypotheses for the gap:

1. **Optimum-intel was running an unoptimised path.** OVModelForCausalLM on `--device GPU` doesn't always engage the GenAI fast-path (PagedAttention, U8 KV cache, XMX dynamic quant — all of which became GPU-default in OV 2024.6 → 2025.4). LLMPipeline does engage them.
2. **The cached model dir at `C:\cascadia\models\llama-3.1-8b-int4` may have been exported pre-2025.1**, so the optimum loader couldn't auto-upgrade it to PA. LLMPipeline applies the SDPAToPagedAttention runtime pass at compile time regardless of how the IR was exported.
3. **Tokeniser overhead in optimum's call path**: the tokeniser/de-tokeniser stays in transformers Python land for OVModelForCausalLM; LLMPipeline pushes both through OV tokeniser ops on-device. Not the bulk of the 10× gap, but contributes.

## Per-token timing

96.41 tok/s = **10.4 ms / token** decode on alpha B390. Compare to c0-1 = 113 ms/token. That's a 10.9× per-token improvement. Even allowing for prompt prefill being lumped into the decode timer (the prompt is 8 tokens; prefill takes maybe ~50 ms), the decode-token cost is in the single-digit-millisecond range now.

## Charlie status

Charlie has openvino 2026.2.0.dev20260420 + openvino-genai 2026.0.0.0 — known DLL ABI mismatch. The project memory called this out earlier. Need a matched-version install before we can re-run on charlie.

Action item (split into c1-3): install matching openvino + openvino-genai on charlie, snapshot env first, then re-run bench.

## Implications

- **`tahoma/worker/engines/openvino/optimum_engine.py` is the wrong engine to use as the default for INT4 LLMs on GPU.** A new engine that wraps `LLMPipeline` should become the recommended single-stage path. The optimum engine still has value for pre-deployment evaluation and for engine-types we don't have GenAI for, but for serving LLMs on GPU it's strictly worse.
- The `35 tok/s` baseline that `main` claimed for `ov-spec K=4` (35 tok/s for spec on alpha) was already half what LLMPipeline gets in plain greedy mode (96 tok/s). With LLMPipeline + speculative decoding, the upper bound should be much higher — c2 will test that.
- The dist-spec engine's 17.5 tok/s is now well below what a single GPU can do — distributed only makes sense for models that don't fit on one node. With 8B INT4 fitting comfortably in 32 GB, distributed 8B should not be a serving target anymore.

## Next experiments

- c1-3: install matching genai+openvino on charlie, repeat bench.
- c2 (new campaign): LLMPipeline + speculative decoding via `draft_model=`. Hypothesis: cumulative speedup over c1-1.
- c3 (new campaign): plug LLMPipeline-style perf into `ov-optimum` engine in tahoma, behind a feature flag, to make this real for users.
- c4 (new campaign): test SchedulerConfig (continuous batching + prefix caching) for chat-style multi-turn.
