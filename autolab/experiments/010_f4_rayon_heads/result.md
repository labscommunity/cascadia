# 010 — F4 rayon over heads — NEUTRAL on miner (-2.7%)

**Verdict:** **negative (mild)** on miner single-stage substrate.
Same quality as K=4 alone, very slightly slower (within bench noise).
**Date:** 2026-05-17 ~20:28 PT
**Magnitude class:** — (within noise)

## Result

| Config | tok/s | quality | Δ |
|--------|------:|---------|---|
| K=4 alone (009 leader) | 0.2100 | 9/10 | (ref) |
| **K=4 + F4 (rayon over heads)** | **0.2044** | **9/10** | **-2.7%** |
| K=8 baseline | 0.0853 | 9/10 | -59% (vs K=4) |

## Why F4 didn't help on miner

Per the q1 breakdown (iteration 003):
- Shell attention bucket: 14.5% of decode time (728ms rank-0 + 578ms rank-1)
- Per-layer attention: ~24ms / layer = ~0.4ms per head (64 heads)
- Rayon task spawn overhead: ~10µs/task × 64 tasks = ~640µs
- Overhead as % of attention bucket: ~3% (per shell call)

Even at perfect parallelism (24 cores on miner), the attention bucket
shrinks from 14.5% → ~1.2% of decode. That's a 13% throughput gain
upper bound. The measured -2.7% says rayon overhead ATE the gain
because:
1. Miner is I/O-bound (cold expert pages). Reducing CPU compute on
   attention doesn't help when the bottleneck is disk paging in expert
   weights.
2. The 24-core CPU is likely already saturated by expert dispatch
   work, leaving no idle cores for parallel attention.
3. The per-head work (~0.4ms) is so small that rayon's task scheduling
   dwarfs the compute.

## When F4 might still pay off

- On matias 2-box pipeline (compute-bound, not disk-bound) — but
  matias-02 Tailscale is currently blocked, so unmeasured.
- On a smaller model that fits in RAM (no I/O bottleneck) — but K2.6
  is too big for any single-box RAM in this fleet.
- Tighter per-head loops (manual SIMD, simdeez, or in a tighter
  inner-product kernel) — bigger code change.

## Patch is kept, not reverted

The `rayon::par_chunks_mut` patch is small (~5 LOC delta), composes
cleanly, and doesn't regress quality. Keeping it on the autolab branch
in case future infra-changes (compute-bound regime, faster disk)
flip the verdict.

## Linked

- `bench_k4_f4_10p.jsonl`
- builds on 009 (K=4 robust leader at 0.2100 tok/s)
- commit 38caab2 — F4 patch
