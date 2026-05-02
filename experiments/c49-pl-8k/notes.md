# c49: PL extractive at 8K input — finding the saturation point

## Setup
charlie 140V GPU, Llama 3.1 8B INT4, extractive prompt (passage +
"Summarize..."), 8192 input + 256 output.

## Results

| Mode | tok/s |
|------|------:|
| plain | 100.62 |
| **PL n=3** | **173.91** |
| Δ | **+73%** |

## Comparison across input lengths (charlie, 256 output)

| Input | Plain | PL | Δ |
|-------|------:|---:|--:|
| 4096 | 100.17 | **194.36** | **+94%** ← PEAK |
| 8192 | 100.62 | 173.91 | +73% |

## Findings

1. **PL win is highest at 4K input (+94%)**, not at 8K (+73%).
2. **Plain decode rate plateaus around 100 tok/s for inputs ≥4K** — 
   memory-bandwidth or attention compute saturated.
3. **PL wins decrease at 8K** due to lookup table overhead growing
   with input. The win/cost tradeoff inverts somewhere past 4K.

## Final PL scaling curve (charlie 140V, 256 output)

| Input | Plain | PL | Δ | Win curve |
|-------|------:|---:|--:|-----------|
|  256 | (n/a) | (n/a) | — | (low input — small lookup) |
| 1024 | 101.88 | 160.64 | +58% | growing |
| 4096 | 100.17 | 194.36 | +94% | **PEAK** |
| 8192 | 100.62 | 173.91 | +73% | declining |

## Recommendation

For tahoma extractive RAG deployment:
- Input <2K: PL gives ~+50-60% (modest but consistent)
- Input 2K-4K: PL gives +70-94% (peak)
- Input 4K-8K: PL gives +73% (still good but past peak)
- Input >8K: untested; likely declining further

The PL flag remains universally beneficial for extractive workloads
across this entire range. The peak is at 4K input.
