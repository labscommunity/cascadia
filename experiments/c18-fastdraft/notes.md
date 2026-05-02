# c18: Intel's FastDraft 150M companion for Llama 3.1 8B

## Setup

- **Target:** Llama 3.1 8B INT4 at `C:\cascadia\models\llama-3.1-8b-int4`.
- **Draft:** `OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov`
  — Intel's *official* FastDraft companion model, 150M parameters,
  INT8 OV-format, trained specifically as a speculative-decode partner
  for Llama 3.1 8B. Downloads to ~155 MB.
- **Hardware:** alpha B390 GPU.
- **Engine:** `openvino_genai.LLMPipeline` with `draft_model=`.

## Results

| Output | K | tok/s | vs plain LLMPipeline (96.4 / 21.5) |
|---|---|---|---|
| 64 tok | 5 | **119.24** | **+23.7%** |
| 64 tok | 10 | 118.22 | +22.6% |
| 256 tok | 5 | 24.79 | +15.1% |
| 256 tok | 10 | 18.88 | -12.3% (over-speculates) |

## DISCOVERY (v2)

**The 150M FastDraft + plain LLMPipeline + Llama 3.1 8B INT4 hits 119 tok/s on alpha B390 — the new tahoma high.** That's +13.4× over the original ov-optimum baseline (8.89) and +23.7% over plain LLMPipeline (96.4).

The size of the draft matters MORE than its accuracy for short-output workloads:
- The 1B-INT4 Llama-3.2 draft (c2-4): K=10 → 100.9 tok/s. Same target, same K, but the 1B draft adds ~3-4 ms of compute per spec round.
- The 150M FastDraft: K=5 → 119.2 tok/s. Smaller draft = less per-round overhead = faster amortisation per accepted token.

For long-gen (256+ tokens), per-token cost on the 8B is so high (~46 ms) that the draft's per-round cost is a much smaller fraction. Both drafts are equivalent there (~25 tok/s).

## Recommendations

| Workload | Engine config |
|---|---|
| Short factual / chat (≤128 tok) | LLMPipeline + FastDraft 150M K=5 |
| Long-form (≥256 tok) | LLMPipeline + any draft K=5 (smaller = no benefit) |
| Open-ended at any length | K=5 always; K=10 over-speculates on creative content |

## Implications for tahoma

The `ov-genai` engine should accept a `--draft-model` arg and pre-pull the FastDraft. We should also document FastDraft as the recommended draft for Llama 3.1 8B in the engine docs.

The `ov-spec` engine baseline (35 tok/s on main, 13.83 today) is now 8.6× slower than `ov-genai` with FastDraft. ov-spec should be deprecated as a recommended engine; `ov-genai --draft-model fastdraft` replaces it.

## Open follow-ups

- Run FastDraft on charlie 140V GPU.
- Search for FastDraft equivalents for Phi-3, Qwen, Gemma (Intel publishes a few).
- Wire `--draft-model` into the tahoma `ov-genai` engine.
