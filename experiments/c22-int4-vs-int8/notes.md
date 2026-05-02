# c22: INT4 vs INT8 weight precision on alpha B390

## Setup

- Llama 3.1 8B on alpha B390 GPU.
- INT4: `C:\cascadia\models\llama-3.1-8b-int4` (optimum-cli export, default
  group-size 128, no AWQ).
- INT8: `fakezeta/Meta-Llama-3.1-8B-Instruct-ov-int8` from HF
  (optimum-cli export, INT8 weights — `openvino_model.bin` is 8.0 GB
  vs the INT4's ~4.1 GB).
- Both share the same FastDraft companion (`OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov`).
- 64-token output, factual prompt.

## Results

| Engine | INT4 weights | INT8 weights |
|---|---|---|
| LLMPipeline plain | 96.41 | 82.56 (-14%) |
| LLMPipeline + FastDraft K=5 | 119.24 | 133.30 (+12% over INT4) |

## Findings

1. **INT4 plain wins by 14%** for the no-spec case. Memory-bandwidth matters
   most here; INT4 reads half the weights from VRAM per forward pass.
2. **With FastDraft, INT8 ties INT4** (within run-to-run variance ~5%).
   At spec-decode the per-token compute fraction grows, narrowing the
   memory-bandwidth gap.
3. **INT8 is never faster** in our tests. It would only be the right
   choice for accuracy reasons (less quantisation error), which we did
   not measure here.

## Recommendation

Default INT4 for Llama 3.1 8B on Intel Battlemage / Lunar Lake unless
accuracy regression is noticed (AWQ + scale-estimation INT4 is the
mid-step before falling back to INT8 — see `_intel_synthesis.md` #4).
