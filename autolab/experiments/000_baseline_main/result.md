# 000_baseline_main — result

**Verdict:** baseline (anchor for downstream moonshots)
**Date:** 2026-05-17 ~13:36 PT

## Per-prompt

| Prompt | wall (s) | tokens | tok/s | quality |
|--------|---------:|-------:|------:|---------|
| Paris   | 123.06 | 8 | **0.0650** | ✓ |
| Pacific | 170.09 | 8 | **0.0470** | ✓ |
| four    | 140.93 | 8 | **0.0568** | ✓ |
| **AGG** | **434.09** | **24** | **0.0553** | **3/3** |

## Notes

- `completion_tokens` and `tok_per_sec` from the bench JSONL are 0 due to
  PS 5.1 / Invoke-RestMethod JSON snake_case parsing quirk. wall_ms is
  correct; tok/s recomputed by hand using `max_tokens=8` (which all 3
  prompts reached without EOS). Rank-0 internal tracing log confirms.
- This is a single-run baseline; 3 independent runs would tighten the σ.
  Acceptable for an anchor; subsequent moonshots will do ≥3 runs.
- Variance: 0.047-0.065 across prompts. Pacific (longest expected
  generation context) was slowest. Per-token cost creeps with KV size
  even with pre-allocated geometric buffers because attention QK·dot
  is O(N).
- Workers stayed warm during the run (rank-0 23.5 GB RSS, ~420s CPU
  time across the 3 prompts). No restarts mid-run.

## Linked artifacts

- `bench.jsonl` — raw 4 lines (3 prompts + 1 aggregate) pulled from `matias-02:k26-bench-000-baseline.jsonl`
- Rank-0 inference log lines at `tahoma-rank0.log` on matias-02 (not committed; large)

## What's next

Iteration 002 = q1 (instrumentation). Patch `runner.rs::step()` and
`forward_shells()` to emit per-stage timing per token (layer0 / shells_attn /
experts / wire send-recv / head). Re-run the same 3-prompt eval against
the instrumented build; verify ≤5% overhead vs this baseline; produce a
breakdown that's published-style "where the time goes" rather than
estimated.
