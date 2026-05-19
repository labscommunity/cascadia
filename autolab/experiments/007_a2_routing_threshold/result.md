# 007 — A2 routing-threshold expert pruning — neutral vs A3

**Verdict:** **neutral** (works, but A3 fixed-K=4 dominates the Pareto)
**Date:** 2026-05-17 ~18:57 PT

## Result

| Config | tok/s | Δ vs K=8 | Quality |
|--------|------:|---------:|---------|
| K=8 baseline | 0.0797 | (ref) | 3/3 |
| `--routing-threshold 0.05` | 0.0645 | -19% (noise) | 3/3 — drops 0 experts (output identical to K=8) |
| `--routing-threshold 0.2` | 0.1043 | +31% | 3/3 — drops ~2 experts (output matches K=6) |
| **A3 `--top-k-override 4` (006 leader)** | **0.1667** | **+109%** | **3/3** |

A2 sigmoid-weight thresholding works mechanically — at threshold=0.05
the routing weights for K2.6 top-8 are all >0.05 so nothing is dropped
(output identical to K=8 baseline). At threshold=0.2, ~2 experts are
dropped per token on average (output matches the K=6 A3 result).

**The Pareto-frontier-leading config is still A3 with K=4.** Variable
per-token K from A2 doesn't beat fixed-K cap of 4 on K2.6 sparse-MoE
single-stage on miner. Likely because K2.6's sigmoid weights are
relatively uniform across top-8 — there's no "obviously bad" expert
to drop conditionally; just dropping the bottom 4 (A3 K=4) wins.

## Implementation notes

The `--routing-threshold f32` CLI flag composes with `--top-k-override
u32` — applied AFTER the top-K cap, so e.g. `--top-k-override 6
--routing-threshold 0.1` is "top 6 of routed, filtered to weight>=0.1".
Both default to None/0 = no behavior change.

Could be useful for:
- Mixed-workload K2.6 inference where some prompts route confidently
  (a few high-weight experts) and others route diffusely (many low
  weights). A2 adapts; A3 doesn't.
- Future quality-eval-driven sweep — try thresholds 0.3/0.5 to find a
  combo that beats K=4. Single 3-prompt eval is too narrow to draw
  strong conclusions.

## What NOT to pursue right now

- Threshold > 0.5: probably hits the K=2-equivalent quality cliff
  (router weights for top-1 are typically 0.3-0.5 range based on
  observed behavior; threshold 0.5 might drop almost all experts).
- Pure A2 (without --top-k-override) at multiple thresholds — A3 K=4
  is strictly better in the measured regime.

## Linked

- `bench_thr05.jsonl` / `bench_thr2.jsonl` — raw outputs
- Builds on 005/006 (A3 K=4 leader)
- Commit f37100b — A2 patch
