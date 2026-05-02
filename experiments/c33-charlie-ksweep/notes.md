# c33: K-sweep for FastDraft 150M on charlie 140V

## Setup
Llama 3.1 8B INT4 + FastDraft 150M INT8 on charlie 140V GPU. 64-tok output,
factual prompt. Sweep K = {3, 5, 7, 10}.

## Results

| K  | tok/s |
|----|------:|
| 3  | 80.46 |
| **5** | **95.78** |
| 7  | 87.07 |
| 10 | 94.92 |

## Findings

1. **K=5 is the sweet spot on charlie too** (matches alpha — see c18).
   The 95.78 tok/s confirms the 96.04 baseline from c18-7.
2. **K=10 is close** (94.92, within 1%) — for charlie there's no penalty
   for slightly over-speculating.
3. **K=3 under-speculates** (80.5, -16% vs K=5).

## Recommendation
Keep K=5 as the default for `--spec-k` on the `ov-genai` engine across
both Lunar Lake and Battlemage. The prior alpha K-sweep (c18) reached the
same conclusion. **No platform-specific K tuning needed for short factual.**

## Long-creative reminder
For 256-token long creative generation, K=3 wins (alpha c18-256-k3).
Charlie 256-tok was 26.90 tok/s at K=5; would be marginally faster at K=3
if we test (open follow-up).
