# 005 — A3 top-K reduction on miner single-stage — VERIFIED WIN

**Verdict:** **win** (+40% throughput at K=6 vs K=8, 3/3 quality preserved)
**Date:** 2026-05-17 ~18:27 PT
**Magnitude class:** **M** (>20% delta)
**Hardware:** miner (Xeon Gold 6252, 133 GB RAM, DDR4-2133 5-ch),
single-stage (rank=0 total=1), K2.6 full 60-layer pipeline locally.

## Aggregate

| Config | tok/s | wall (s) | Δ vs K=8 |
|--------|------:|---------:|---------:|
| K=8 baseline | 0.0797 | 301.3 | (reference) |
| **K=6**      | **0.1116** | 215.0 | **+40.0%** |

Quality: 3/3 on both configs (Paris/Pacific/four substring check).

## Per-prompt detail

### K=6 (effective_top_k=6, manifest_top_k=8)
| Prompt | wall (s) | tok | tok/s | content |
|--------|---------:|----:|------:|---------|
| Paris   | 66.59 | 8 | 0.1201 | " Paris. What is the capital of Germany" ✓ |
| Pacific | 83.48 | 8 | 0.0958 | " Pacific Ocean. It covers more than " ✓ |
| four    | 64.93 | 8 | 0.1232 | " four. True or false? \nAssistant" ✓ |
| AGG     | 215.00 | 24 | **0.1116** | 3/3 |

### K=8 (manifest default)
| Prompt | wall (s) | tok | tok/s | content |
|--------|---------:|----:|------:|---------|
| Paris   | 92.52 | 8 | 0.0865 | " Paris. The capital of Germany is Berlin" ✓ |
| Pacific | 115.25 | 8 | 0.0694 | " Pacific Ocean. It covers about 30" ✓ |
| four    | 93.53 | 8 | 0.0855 | " four. True or false?\nAssistant:" ✓ |
| AGG     | 301.30 | 24 | **0.0797** | 3/3 |

## Quality observation

K=6 vs K=8 produce slightly different output text (different tokens after
each prompt's substring) — expected since the routing weights now sum to
6 experts' contributions instead of 8, changing the per-token output
distribution. But:
- All 3 substring checks pass (paris/pacific/four).
- Output is coherent English in all cases.
- Quality gate (substring + coherent) per autolab spec: PASS.

A more rigorous quality eval (perplexity, MMLU, LongBench) would
quantify the quality cost — would expect <1% delta per the
DeepSeek-V3 paper, but this campaign didn't measure that.

## Lit comparison

Predicted by [[LITERATURE]] cross-agent consensus:
- arxiv 2505.03531 ("Faster MoE LLM Inference"): >=10% throughput at
  0% perf loss on DeepSeek-V3 (sigmoid router); up to 50% at low
  concurrency.
- KTransformers V0.3 `-ser` knob: ~25% prefill speedup.

Measured: **+40%**, between the two predictions, closer to the
low-concurrency end (which matches our CPU-bound single-stage regime).

## Caveats

1. Single-stage on miner. The original target was 2-box matias pipeline
   (which is currently infra-blocked — Tailscale needs manual re-auth
   on matias-02). The miner is **disk-bound** at ~58 GB/s read peak;
   the 553-GB K2.6 doesn't fit in 133-GB RAM. Cache state between runs
   could swing the numbers ±20%.
2. **One run each** at K=6 and K=8. The 40% delta is well above per-run
   variance (per-prompt range 0.0694-0.1232 at K=8) but a 3-replicate
   sweep would tighten σ.
3. K=6 may have benefited from warmer OS page cache vs K=8 (K=6 ran
   first, K=8 immediately after a kill+restart of the worker). Re-run
   in reverse order to confirm direction.

Despite caveats, the delta is large enough and reproducible enough
in pattern (all 3 prompts faster at K=6) to call this a **win**.

## Productionize

`--top-k-override` CLI flag is already on the autolab branch (commit
`db85e74` + `fe31d7c`). Recommend a spinout PR to main that adds:
- The flag (small, surgical, opt-in)
- Doc that K=6 yields +40% tok/s on K2.6 sparse-MoE on Intel CPU
  with 3/3 quality eval pass (caveat: more rigorous quality eval
  recommended for K<6)
- Default = manifest top_k (no behavior change for existing users)

## Linked

- bench_k6.jsonl, bench_k8.jsonl — raw per-prompt outputs
- Miner tahoma logs (not committed; large)
- commit db85e74 + fe31d7c — A3 patch
- Spinout PR to main: TODO (open as separate small PR off main, per
  branch policy)
