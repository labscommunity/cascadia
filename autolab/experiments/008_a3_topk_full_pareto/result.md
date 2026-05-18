# 008 — A3 K-sweep full Pareto — K=3 LEADER (+208%)

**Verdict:** **WIN.** K=3 is the new leader. K=5 is the modest middle ground.
**Date:** 2026-05-17 ~19:07 PT
**Magnitude class:** **L** (>2× delta)

## Full Pareto (miner single-stage K2.6, 3-prompt substring eval)

| K | tok/s | Δ vs K=8 | Quality | Source iteration |
|--:|------:|---------:|---------|------------------|
| 8 | 0.0797 | (ref) | 3/3 | 005 (baseline) |
| 6 | 0.1116 | +40.0% | 3/3 | 005 |
| 5 | 0.1547 | +94.1% | 3/3 | **008** |
| 4 | 0.1667 | +109.2% | 3/3 | 006 |
| **3** | **0.2455** | **+208.1%** | **3/3** | **008 (LEADER)** |
| 2 | 0.2716 | +240.8% | 2/3 | 006 (cliff) |

## Curve interpretation

- K=8→K=6: ~25% expert reduction → +40% tok/s. Sub-linear gain (cache
  state effects).
- K=6→K=5: marginal +30% additional. Still cache-bound.
- K=5→K=4: +8%. Diminishing returns — K=5 was already in
  bottom-of-cache regime.
- **K=4→K=3: +47%.** Non-linear bump! At K=3, expert dispatch time
  has shrunk enough that I/O cost dominates less. Likely the OS page
  cache holds a higher fraction of the active expert weights when only
  3 are dispatched per layer per token. **K=3 is the sweet spot.**
- K=3→K=2: +11% throughput but breaks quality (sampling format issue
  on "four" prompt — gave digit "4" instead of word).

## K=3 per-prompt

| Prompt | wall (s) | tok/s | content |
|--------|---------:|------:|---------|
| Paris   | 30.52 | 0.2621 | " Paris. The capital of the United States" ✓ |
| Pacific | 36.95 | 0.2165 | " Pacific Ocean. It covers about 30" ✓ |
| four    | 30.28 | 0.2642 | " four, and the square of the square" ✓ |
| **AGG** | 97.75 | **0.2455** | **3/3** |

Output coherent and contains expected substring on all three prompts.

## K=5 per-prompt

| Prompt | wall (s) | tok/s | content |
|--------|---------:|------:|---------|
| Paris   | 42.28 | 0.1892 | " Paris. What is the capital of Germany" ✓ |
| Pacific | 58.92 | 0.1358 | " Pacific Ocean. It covers about 30" ✓ |
| four    | 53.92 | 0.1484 | " four. Yes, that is correct." ✓ |
| **AGG** | 155.13 | **0.1547** | **3/3** |

## Productionization recommendation

**Default `--top-k-override` to 3 for K2.6 single-stage on Intel CPU.**

- +208% throughput (3× faster) over manifest top_k=8 default.
- 3/3 quality eval passes on Paris/Pacific/four.
- No behavior change if flag is omitted (default = manifest top_k = 8).
- Spinout PR off main: add the flag (commits db85e74 + fe31d7c + f37100b)
  with docs/A3_TOPK_REDUCTION.md showing the full Pareto and
  recommending K=3.

**Caveats:**
- Single-stage on miner is disk-bound; the cliff at K=2 might not
  reproduce on the 2-box matias setup (memory-bound vs disk-bound).
  Need to re-run on matias once Tailscale infra unblocks.
- 3-prompt substring eval is narrow. Iteration 009 = multi-prompt
  robustness check to validate K=3 across a wider input distribution
  before recommending as default.
- K=3 output diverges from K=8 in non-trivial ways (each generates
  different text starting from the same prompt). Substring pass
  doesn't imply semantic equivalence; a real quality metric
  (perplexity, MMLU) would be needed for full validation.

## Linked

- `bench_k3.jsonl`, `bench_k5.jsonl` — raw outputs
- builds on 005/006 (K=6, K=4, K=2)
- commit f37100b — A2/A3 patch
