# e2 — K sweep on ov-dist-spec + FastDraft (creative workload)

**Hypothesis:** The `K=3` chosen in Phase 14 was tuned for short factual prompts where FastDraft acceptance is high (~0.83). On the creative workload, acceptance collapsed to 0.054 in e1. Sweep K to find the optimal value when acceptance is low — possibly K=1 wins because we waste less work per round.

**Setup:** Same as e1 (alpha+charlie/TB4, ov-dist-spec, v5 16/16, FastDraft 150M, 256-tok creative prompt), 3 trials per K. Charlie restarted between trials.

## Result

| K | trials | med tok/s | med accept | med steps | med elapsed (s) |
|---|-------:|----------:|-----------:|----------:|----------------:|
| 1 | 3 | **11.78** | 0.154 | 221 | 21.72 |
| 2 | 3 | 10.83 | 0.079 | 221 | 23.63 |
| 3 | 5 (e1) | 9.88 | 0.055 | 220 | 25.91 |
| 4 | 3 | 8.90 | 0.047 | 215 | 28.76 |
| 5 | 3 | 9.04 | 0.058 | 199 | 28.31 |
| 6 | 3 | 8.61 | 0.055 | 192 | 29.74 |

**K=1 wins by 19%** over the prior K=3 default. **K=4 is the worst.** Beyond K=4, the engine reduces step count (fewer rounds) but per-step cost grows faster.

## Why K scales like this

For independent draft tokens with single-token accept probability `p` ≈ 0.15:
- Per-round expected accepted prefix length = Σ p^j for j=1..K = `p(1-p^K)/(1-p)`
  - K=1: 0.15
  - K=2: 0.17
  - K=3: 0.18
  - K=4: 0.18
- Per-round target verify cost grows ~linearly with K (more tokens in the verify forward).
- Net: bigger K wastes more target compute for diminishing draft amortization.

This is the canonical low-acceptance regime: spec decode is **negative-value** at high K when the draft can't keep up with the target. FastDraft 150M was Intel-trained for chat/short-factual prompts — creative content is out-of-distribution.

## Conclusion

- Best distributed config so far: **K=1 = 11.78 tok/s** (51% of e0 monolithic 23.01).
- Distance to bar (27.6 tok/s × 0.51 / target) = **need 2.34× current K=1 to hit bar**.
- **The win path is NOT K-tuning** — K=1 already minimizes spec-decode waste. We need:
  1. A draft that actually matches creative content (or self-spec via top-K of target's own logits)
  2. Or async pipeline overlap that hides the wait for the slow distributed target.feed
  3. Or per-stage compute optimization (paged-attention re-export — addressed in a later campaign)
  4. Or layer rebalance so charlie isn't the bottleneck (e4 — shards in flight)

Per-K tok/s plot data ready for follow-up plotting.

## Update LEADERBOARD

Distributed alpha+charlie/TB4 best is now `ov-dist-spec K=1 + FastDraft 150M, v5 16/16` = **11.78 tok/s** on 256-tok creative.

## Engineering note

Current `--spec-k` defaults to whatever the user passes. Should we change the engine's default, or expose an "auto" mode? Defer — the right answer depends on workload and per-deployment FastDraft accept rate. Document the K-sensitivity in the engine docs as a follow-up.
