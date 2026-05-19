# 003 — q1 instrumentation — RESULT

**Verdict:** complete. q1 hypothesis VERIFIED (expert dispatch dominates per-token time, even more than the 60% predicted — actual 82%).
**Date:** 2026-05-17 ~14:34 PT
**Bench tok/s:** 0.0550 (-0.5% vs baseline 0.0553, within noise)
**Quality:** 3/3 pass (Paris/Pacific/four)
**Instrumentation overhead:** <1% (well within 5% budget)

## Per-token decode breakdown

| Stage | ms | % |
|-------|---:|--:|
| Rank-0 layer 0 | 81 | 0.9% |
| Rank-0 shell attention (30L) | 728 | 8.1% |
| **Rank-0 shell expert dispatch (30 × top-8)** | **3,229** | **35.9%** |
| Rank-0 combine | <1 | <0.1% |
| Pure wire (Tailscale DERP) | 60 | 0.7% |
| Rank-1 shell attention (30L) | 578 | 6.4% |
| **Rank-1 shell expert dispatch (30 × top-8)** | **4,151** | **46.1%** |
| Rank-1 head | 139 | 1.5% |
| **TOTAL DECODE PER TOKEN** | **9,005** | **100%** |

Variance: rank-1 experts 1.5s–9.0s (5.9× range). Disk-page-in dominated.

## Tier-S re-rank (loop output, supersedes lit-only Tier-S in MOONSHOTS.md)

1. **A3 top-K reduction** — attacks 82%
2. **D4 async pipeline overlap** — hides 54% (rank-1 compute)
3. **F4 multi-thread per shell** — attacks 14.5% (attention)
4. **A2 sigmoid threshold pruning** — variant of A3
5. ~~**D1 BF16 wire**~~ — DROPPED (wire is 0.7%)

## Linked artifacts

- `bench.jsonl` — 3 prompts + aggregate from `k26-bench-iter003-v3.jsonl`
- Rank-0 worker log: `/tmp/rank0-bench-v3.log` (138 stage_timing events, not committed; large)
- Rank-1 worker log: `/tmp/rank1-bench-v3.log` (92 stage_timing events, not committed; large)
- Parse logic: inline python in JOURNAL 003 commit

## What's next

Iteration 004 = A3 top-K reduction. Implement `--top-k-override` CLI flag, plumb through SparseMoEBuilderConfig → Runner, modify forward_shells inner loop to iterate over only `min(top_k, override)` experts. Build, deploy, bench at K=6 and K=4. Aim: 15-25% throughput gain at minimal quality cost.
