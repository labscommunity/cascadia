# 021: the plugin's decode kernel for the fused experts, on one rank

**Why.** See 020: decode frames run through the fused-MoE op's PREFILL path because of a crash seen on Windows.
Intel's PR #35901 built the batched-GEMV path for exactly our case (a few tokens, many experts, "about 20 %").
At 15 streams the MoE calls are 25 of a 41 ms frame, ~9 ms of it a fixed cost per frame.

**Method.** Overrides only. Rank 5 sets `OV_GPU_MOE_BATCHED_GEMV_THRESHOLD=32`; everything else as 015c (the 2-row
padding of one-row frames stays). Per-rank A/B against ranks 1-4 and 6-7 (same hardware). Gates first: a different
kernel may move f16 rounding, so the side-by-side gate and the known answers decide.

**Risk.** If the kernel crashes on Linux too, rank 5's worker dies at its first decode frame, the chain restarts in a
loop, and the fleet is down until the next release (~5 minutes). A host-side crash, as on Windows, does not wedge the
GPU; if it did, a one-shot reboot of that rank through the overrides is the fallback.

**Prediction.** Rank 5's MoE time per 1.36-row frame falls from 25 ms to 17-19 ms. Kill: crash, gate failure, or no gain.
