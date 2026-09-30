# 021 verdict: REVERTED. The plugin's decode kernel kills the worker on Linux too.

Rank 5 with `OV_GPU_MOE_BATCHED_GEMV_THRESHOLD=32`: it loaded and warmed up (the warm-up uses expert ids 0-7), then
died at the first real frames; the chain kept re-forming and every request failed ("downstream closed before
StreamTokens") until the overrides were rolled back by hand (~12 minutes of outage, my `--force` kept the harness
sending into it). Lessons built into 022: a canary must undo itself, and the harness runs without `--force`.
