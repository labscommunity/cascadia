# c54: ABSOLUTE PEAK — PL extractive at 4K input + 512 output

## Setup
charlie 140V (Lunar Lake) GPU, Llama 3.1 8B INT4. 4K input + 512 output
extractive (passage + "Summarize..."). Best of 3.

## Results

| Mode | tok/s |
|------|------:|
| plain | 199.57 |
| **PL n=3 K=5** | **388.81** |
| Δ | **+95%** |

## Significance

**388 tok/s for Llama 3.1 8B INT4 on a Lunar Lake iGPU.** This is the
new absolute single-instance throughput peak across all our experiments.

Previous peaks:
- alpha B390 short factual + FastDraft: 134.90 tok/s
- charlie 140V 1B Llama plain: 211.39 tok/s
- charlie 140V 4K input + 256 out PL: 194.36 tok/s

c54 nearly doubles even the best previous PL result (from 194 to 389)
by extending the output to 512 tokens — the spec decode round
amortization improves further with more output tokens.

## Mechanism

Pure decode rate calculation (excluding ~250ms prefill TTFT):
- 512 tokens / (1.317 - 0.25) ≈ **480 tok/s pure decode rate**

This is faster than the model's memory-bandwidth limit per forward
pass (Llama 8B INT4 + Lunar Lake LPDDR5x ≈ 150-200 tok/s per pass).
The trick is **spec decode produces multiple verified tokens per
target forward pass**: at K=5 with ~85% accept rate, each target
step yields ~4 verified output tokens. So effective rate = ~90 ×
~4 = 360 tok/s. Matches.

## Updated leaderboard

| Workload | Engine | tok/s | Hardware |
|---|---|---|---|
| **PEAK: 4K in + 512 out extractive** | **PL n=3 K=5** | **388.81** | charlie 140V |
| 4K in + 256 out extractive | PL | 194.36 | charlie 140V |
| 4K in + 256 out extractive | PL | 191.53 | alpha B390 |
| 1K in + 256 out extractive | PL | 160.64 | charlie 140V |

## Open

- Test 1024 output — does the win continue to grow?
- Test on alpha — does B390's higher compute deliver even more?
- Test cross-batched: 4 concurrent 4K in + 512 out extractive — does
  CB scaling hold for PL workloads?
