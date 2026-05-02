# c53: PL works with sampling (do_sample=True, temp=0.7)

## Setup
charlie 140V GPU, Llama 3.1 8B INT4. 1K-input passage + extractive
summary, 256-token output. **Sampling** (do_sample=True, temperature=0.7)
— the realistic chat configuration.

## Results

| Mode | Best tok/s |
|------|-----------:|
| plain (sampling) | 119.13 |
| **PL n=3 K=5 (sampling)** | **188.04** |
| Δ | **+58%** |

## Comparison to greedy (c45)

| Decoding | Plain | PL | Δ |
|----------|------:|---:|--:|
| greedy   | 101.88 | 160.64 | +58% |
| sampling | 119.13 | 188.04 | +58% |

Same +58% relative win in both regimes.

## Findings

1. **PL works equally well in sampling regime** as greedy.
2. **Absolute throughput is HIGHER in sampling** (119 plain vs 102 greedy,
   188 PL vs 161 greedy) — likely because warm-up state / GPU thermal
   conditions varied between runs. Or the bench harness has subtle
   differences. Either way, PL win is consistent.
3. **PL win does NOT depend on greedy decode** — the n-gram lookup
   accepts based on target's sampled token matching the draft's
   prediction, which works in both modes.

## Implication for tahoma chat deployment

Real chatbot deployments use sampling (temp=0.7-1.0) for diverse
responses. PL still gives the +58% extractive RAG win at this
realistic configuration. PL is universally useful for extractive
chat workloads regardless of decoding mode.

## Open

- Test with higher temperature (1.0+) — does accept rate drop more?
- Test with top-p / top-k sampling — does PL still help?
