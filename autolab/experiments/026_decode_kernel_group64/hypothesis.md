# 026: the GPU plugin's fused-MoE decode kernels on ONE layer of ONE rank, next to the prefill path

Goal (re-set 2026-09-20): interactive speed for up to 15 streams, ideally 60 tok/s aggregate. At 15 streams a frame
carries 1.36 rows and a middle rank spends 25 of its 41 ms in six fused-experts calls, all through the plugin's
PREFILL path (grouped GEMM: host waits, sub-buffer objects per expert), because its DECODE path (batched GEMV, three
kernels, no host waits) threw on this fleet (021, 022, 025).

025's reading of the OpenVINO 2026.3.1 sources: the decode kernels refuse int4 group 32 on Xe2 and newer (they want
group >= 2 x sub-group 32); the refusal is swallowed at compile time and `infer()` throws. So:

* the generator can write a layer re-quantised to group 64 (`--group 64`; 7.8 % rms away from the group-32 weights,
  12 % of the block's output, cosine 0.992: measured on the miner, layer 30, NOT exact),
* `run.sh` writes such layers into `moe_ov_g64/` (`CASCADIA_FUSE_DECODE_LAYERS`),
* the engine takes a layer from there with the decode threshold ON (`CASCADIA_INKLING_OV_MOE_DECODE_DIR`), every other
  layer keeps the prefill path, and `CASCADIA_INKLING_LAYER_BENCH=1` times one call per layer at 1, 2 and 3 rows at
  load, with nothing else running.

Canary: rank 5, layer 30 only. The 32-bit expert offset is patched in the plugin as in 022, the load check on the
highest expert ids is enforced, the host expert cache is capped at 1 GiB on every rank (025: a rank whose device
calls fail must get slow, not OOM-killed), and the block reverts itself on a signal death or more than 8 starts.

Prediction: the decode path saves 1-1.5 ms per layer per frame (7-9 ms of a 41 ms frame if rolled out).
Kill: any gate failure, rank 5 restarting, 15-stream steady below 23 tok/s.
