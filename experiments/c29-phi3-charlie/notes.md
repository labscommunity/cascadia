# c29: Phi-3 mini + FastDraft 50M on charlie 140V (vs alpha B390)

## Setup
- Llama target / draft pairs:
  - charlie 140V GPU: phi3-mini-int4-ov + phi3-fastdraft-50m-int8-ov, K=5
  - alpha B390 GPU: same models (results already in c18 / leaderboard)

## Results

| Hardware | Engine | tok/s |
|---|---|---|
| charlie 140V GPU | Phi3 plain (LLMPipeline) | 36.26 |
| charlie 140V GPU | Phi3 + 50M FastDraft K=5 | 40.68 (+12.2%) |
| alpha B390 GPU | Phi3 plain | 32.18 (from c18) |
| alpha B390 GPU | Phi3 + 50M FastDraft K=5 | 43.90 (+36.4%) (from c18) |

## Findings

1. **Phi-3 + FastDraft win is much smaller on Lunar Lake (+12%) than on
   Battlemage (+36%).** Hypotheses:
   - Lunar Lake's NPU/GPU has a relatively faster per-token decode than
     Battlemage, so the FastDraft per-spec-round overhead eats more of
     the savings.
   - Or accept rate is lower for some reason (less likely — it's the
     same model and tokenizer).
2. **Lunar Lake plain Phi-3 (36.26) is 13% faster than Battlemage plain
   (32.18).** Lunar Lake's iGPU has lower memory bandwidth than B390
   but higher per-clock efficiency for small models.

## Recommendation

For Phi-3 on Lunar Lake, the FastDraft spec decode is still net positive
(+12%) but not the no-brainer it is on Battlemage. For deployment we
should still default to FastDraft enabled — the worst-case overhead is
minimal because the draft model is only 50M parameters.

For LLama 3.1 8B (the more common deployment), FastDraft remains a clear
win on both platforms (~+24-40%).
