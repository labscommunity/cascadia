# 023 verdict: NO. Two threads of device calls barely overlap on this iGPU: the split stage is not worth building.

| device calls of one synthetic frame, ms | one thread | 2-thread pipeline | 3-thread pipeline |
|---|---|---|---|
| rank 5 (25 W), 1 row | 27.7 | 25.2 (1.10x) | 25.5 |
| rank 5, 2 rows | 38.6 | 36.7 (1.05x) | 36.6 |
| rank 9 (higher limit), 1 row | 27.0 | 25.1 (1.08x) | 24.9 |
| rank 9, 2 rows | 38.5 | 36.9 (1.04x) | 36.7 |

Decision rule was >= 1.25x. The "fixed cost per call" is DEVICE time (or serialised inside the driver), not host time
during which the GPU idles. What a split stage could still hide is the 4-6 ms of CPU work per frame (~12 %); not
enough for that refactor now. So at one or two rows per frame the fused-MoE op's prefill path is simply inefficient
ON the device (a grouped GEMM doing GEMV-sized work: ~83 GB/s for the first row, ~130 GB/s for every further one),
and the op's decode kernel, built for this case, is the one that crashes here (021, 022).

Also in this build: every fused layer on every rank agrees with the host kernels on its highest expert ids
(cosine 0.999995-0.999996, 64 layers): the load-time check now has a record. 15 streams: 23.6 tok/s steady, gates pass.
