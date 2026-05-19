# 013 — Apples-to-apples K=4 vs K=8 at max_tokens=64

**Verdict:** **win confirmed at higher rigor.** K=4 is +210% faster
AND equal-or-better quality than K=8 baseline at long context.
**Date:** 2026-05-17 ~21:01 PT
**Magnitude class:** **L** (confirms K=4 productionization)

## Head-to-head (10-prompt, max_tokens=64)

| Config | tok/s | quality | total wall (s) |
|--------|------:|---------|---------------:|
| K=4 (iter 011) | **0.3253** | **9/10** | 1968 |
| K=8 (this iter) | 0.1048 | 8/10 | 5498 |
| **Δ (K=4 vs K=8)** | **+210%** | **+1 prompt passing** | **2.8× faster wall** |

## Why this surprises

The original K=8 baseline measurement at max_tokens=16 (iter 009) showed
K=8 = 9/10 quality. At max_tokens=64, K=8 dropped to 8/10. Why?

Looking at the K=8 output: when given more tokens, the model goes
off-task more often. Examples:
- "km" prompt → "How many times the speed of sound..." (math
  derivation instead of direct answer)
- One other prompt also went off-task (576 total tokens means at
  least one prompt EOSed early or output less than 64)

K=4 at max_tokens=64 also has 1 failure ("celsius" → multi-choice
format) but the model stays more on-task with the smaller expert
budget — possibly because fewer experts contribute makes the output
distribution sharper.

This is an **unexpected positive result**: K=4 isn't just faster, it's
also slightly MORE consistent on the substring eval at long context.

## Productionization recommendation (FIRM)

**Default `--top-k-override 4` for K2.6 sparse-MoE on Intel CPU
disk-bound substrates.**

- 3× throughput vs K=8 baseline at chat-realistic output lengths
- 9/10 vs 8/10 quality (slight edge to K=4)
- No retrain, no requant, no architectural change
- Opt-in flag (default = manifest top_k = no behavior change)

Spinout PR off main: ready to open. Add the flag (commits db85e74 +
fe31d7c + f37100b) + a docs/A3_TOPK_REDUCTION.md page with the full
Pareto and the K=4 recommendation.

## Caveats unchanged

- Single-stage on miner (disk-bound). On compute-bound substrates
  (matias 2-box when Tailscale is fixed), the throughput delta may
  shrink but the quality picture should hold.
- 10-prompt eval is still narrow. A proper MMLU/LongBench eval would
  give a final say.

## Linked

- `bench_k8_mt64.jsonl`
- compares to iter 011 (K=4 mt=64 = 0.3253 tok/s 9/10)
