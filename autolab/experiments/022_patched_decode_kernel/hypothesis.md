# 022: the fused-MoE decode kernel with its 32-bit expert offset patched (canary: rank 5)

**Finding (research agent, read in the 2026.3.1 source; confirmed in the shipped library's embedded kernel text).**
The batched-GEMV decode kernels compute `gate_weight_addr + expert_id * expert_wei_size` in 32-bit ints. One expert
is 3072 x 6144 / 2 = 9,437,184 bytes, so ids >= 228 overflow and read out of bounds. This model has 258 experts per
layer and every token lists the shared ones (ids 256, 257): every real call faults, on any OS (021 on Linux, the
earlier crash on Windows, and the "1-row call crashes" rule: threshold 0 is coerced to 1, so one-row calls took this
kernel too). Models Intel validates stay under 2 GiB per matrix. The kernel file is unchanged in 2026.4 and master.

**Fix.** `const int expert_wei_size=` -> `const long expert_wei_size=` at the 4 sites, same byte length (one space
dropped), in the plugin library's embedded OpenCL text; sha256 checked before (6f8bba74...) and after (a2f70bbb...),
original kept as `.orig`. `bench/patch_gpu_plugin.py`, embedded in the overrides.

**Method.** Rank 5 only: patch, then `OV_GPU_MOE_BATCHED_GEMV_THRESHOLD=32`. Self-healing: if that worker dies by a
signal or restarts more than 8 times, its next start restores the original library and the prefill path.
Correctness = the greedy gates (a wrong expert read gives garbage text at once). Per-rank A/B for speed.

**Prediction.** Rank 5's fused-MoE time per 1.36-row frame: 25 ms -> 16-19 ms (three kernels and no host waits per call
instead of ~8 host synchronisations and 516 sub-buffer objects). Kill: gate failure or a reverted canary.
