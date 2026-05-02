# c47: PL extractive cross-platform input sweep validation

## Setup
alpha B390 GPU, Llama 3.1 8B INT4. Extractive prompt (passage +
"Summarize the passage above in 2 short sentences"). 128-token output.
Sweep input ∈ {256, 2048}. Mirrors c45 charlie sweep.

## Results (alpha vs charlie)

| Input | alpha plain | alpha PL | alpha Δ | charlie plain | charlie PL | charlie Δ |
|-------|-----------:|---------:|--------:|--------------:|-----------:|----------:|
|  256  | 52.17 | 77.98 | **+49%** | 56.94 | 85.13 | +50% |
| 1024  | 47.11 | 75.34 | +60%  (from c45) | 51.70 | 80.76 | +56% |
| 2048  | 48.21 | 85.76 | **+78%** | 52.15 | 89.37 | +71% |

## Findings

1. **PL win grows monotonically with input length for extractive workloads** — confirmed on BOTH alpha (49→60→78%) and charlie (50→56→71%).
2. **Cross-platform parity**: both platforms show similar absolute speedups and similar growth rate.
3. **The longer the passage, the bigger the win** — at 2048 input, alpha hits +78% PL win (best so far).

## Why does PL win grow with input?

For extractive workloads (model output quotes input vocabulary):
- More input = more candidate n-grams in the lookup table.
- More candidates = higher accept rate per spec round.
- Higher accept rate = bigger savings vs fixed-cost target verify.

The lookup table build/search cost scales with input (O(n_input)), but
the per-step savings scales faster than linearly because each accepted
draft token saves a target forward pass at the now-larger KV.

## Updated leaderboard line

For RAG / extractive summary at 2K input:
- alpha B390 GPU: 85.76 tok/s
- charlie 140V GPU: 89.37 tok/s

For RAG / extractive summary at 1K input + 256 output:
- charlie 140V GPU: **160.64 tok/s** (peak)
- alpha B390 GPU: 150.71 tok/s

## Final Discovery #3 statement

Prompt Lookup decoding gives **+49-78% on EXTRACTIVE workloads** for
Llama 3.1 8B INT4 across both Lunar Lake and Battlemage Intel GPUs.
The win grows monotonically with input length (256 → 2K input: +49% →
+78% on alpha) because the n-gram lookup table grows with input but
the per-step savings grow faster.

For non-extractive (open-ended) workloads, PL is at best tied with
plain (c43, c44).
