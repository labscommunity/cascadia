# Q3.X — better draft model: Llama 3.2 1B INT4 vs FastDraft 150M

**Hypothesis:** FastDraft 150M's 38% acceptance was the bottleneck. Per the prior python autolab d3, larger drafts (1B) give higher accept but slower per-call. The MATH on whether 1B nets out positive depends on whether draft cost can be HIDDEN behind charlie wait via async overlap (Q3.2).

## Result (factual workload, 256 tokens, alpha+charlie/TB4)

| Draft model | K | tok/s | accept | tok/round | target_ms/round | draft_ms/round |
|-------------|--:|------:|-------:|----------:|----------------:|---------------:|
| FastDraft 150M | 1 | 15.81 | 0.378 | 1.38 | ~70 | ~17 |
| **Llama 3.2 1B INT4** | 1 | 15.84 | **0.808** | 1.82 | 71.9 | 42.3 |
| **Llama 3.2 1B INT4** | 2 | 17.70 | 0.685 | 2.37 | 75.3 | 57.8 |
| **Llama 3.2 1B INT4** | **3** | **18.42** | 0.628 | 2.88 | 78.7 | 76.5 |
| Llama 3.2 1B INT4 | 4 | 17.47 | 0.543 | 3.16 | 83.3 | 96.3 |
| Llama 3.2 1B INT4 | 5 | 17.33 | 0.515 | 3.56 | 87.3 | 116.3 |

**Sweet spot: K=3 with 1B draft = 18.42 tok/s** — a free **+16% over the prior leaderboard high (15.81 with FastDraft K=1)**. Just by changing the draft model + tuning K. No engine code changes.

## What this enables

The current per-round time is `target + draft = 78.7 + 76.5 ≈ 155 ms` (sequential).

The relevant ratio for async overlap: **draft compute is now ~the same magnitude as charlie wait**. If we hide drafts behind the 43 ms charlie stage_1 + ~30 ms alpha stage_0 sit-idle:
- Best case (full parallelism): per-round = max(target = 78.7, draft = 76.5) + 5 = **83.7 ms → 34.4 tok/s** (BEATS BAR 28)
- Realistic (alpha GPU serializes target stage_0 + draft): alpha total = 27 + 76.5 = 103.5 ms; charlie wait hidden behind alpha drafts → per-round = 103.5 + 5 = **108.5 ms → 26.5 tok/s** (close to bar)
- Pessimistic (no benefit): 155 ms → 18.4 tok/s (current measured)

Either of the first two clears or comes very close to the bar. **Async overlap is now the single highest-value engineering work item.**

## New leaderboard high

Distributed alpha+charlie/TB4 (factual workload): **18.42 tok/s** (1B draft K=3, no engine changes). +16% over previous best.

## Next: implement async overlap (Q3.2)

The change in `dist_spec.rs` `spec_decode_greedy`:
1. Refactor target.feed to expose async send + recv halves
2. After sending target verify, run draft.feed for next round in parallel
3. Reconcile when target returns

Engineering effort: 4-6 hours. Risk: alpha GPU may serialize draft + target stage_0. Mitigation: configure OV with `NUM_STREAMS=2` to allow concurrent inference on the GPU.
