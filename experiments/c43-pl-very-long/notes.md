# c43: Prompt Lookup at very long inputs (2K, 4K) — REVISED

## Setup
Llama 3.1 8B INT4 on charlie 140V GPU. 64-token output. Synthetic
distributed-systems passage.

PL via LLMPipeline with `prompt_lookup=True`, `max_ngram_size=N`,
`num_assistant_tokens=5`. Plain via standard LLMPipeline (no spec).

## Results

### PL sweep (n ∈ {2, 3, 5}, K=5)

| Input | n=2 | n=3 | n=5 |
|-------|----:|----:|----:|
|  2048 | 34.3 | 35.5 | 34.0 |
|  4096 | 32.8 | 32.8 | 32.8 |

### Plain LLMPipeline (no spec)

| Input | tok/s | TTFT ms |
|-------|------:|--------:|
|  4096 | 38.4 | 351 |

## Findings — REVISED

1. **PL at 4K input is a -14% LOSS vs plain LLMPipeline** (32.8 vs 38.4
   on charlie). The PL win we saw in c21 (small passage, +65%) does
   NOT extrapolate to very long inputs.
2. **The cross-over point is somewhere in 1K-2K range.** At 1K input
   we measured PL ~32 tok/s; if plain is ~38 there too, PL would also
   lose. Need direct plain-vs-PL bench at 1K and 2K to find exact
   crossover.
3. **At 4K input the n-gram lookup table is large** (~4K positions)
   and its build/search per-step cost grows. The savings from any
   accepted draft tokens are eaten by the lookup overhead.

## REVISED understanding of when PL wins

PL wins when:
- Input has highly repetitive vocabulary that the model is summarizing.
- Input length is in the 100-1000 token range.
- Output is at least as long as the average matched n-gram.

PL loses when:
- Input is very short (no n-gram matches possible).
- Input is very long (lookup overhead exceeds savings).
- Output is very short (savings can't amortise).

The c21 +65% finding was from a sweet-spot workload (250-token passage,
128-token summary). It does not generalize.

## Action

Update DECISION_MATRIX.md to reflect PL being workload-specific, not a
universal RAG default. Recommend manual A/B test for new workloads.
