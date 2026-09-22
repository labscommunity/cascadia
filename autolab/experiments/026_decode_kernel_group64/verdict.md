# 026 verdict: the decode kernels work (with group 64), read at the bus limit, and are worth less than predicted

**What ran.** Rank 5 re-quantised layer 30 to int4 group 64 on the box (140 s; weight error 7.77 / 7.79 % rms) and ran
it through the plugin's DECODE path (threshold 32 for that layer only, 32-bit expert offset patched as in 022); layers
31-35 stayed on the prefill path. Binary e8c3c744.

**It works.** No exception, no restart, both gates pass. The load check on the HIGHEST expert ids (250-255 + shared)
measured cosine **0.99206** against the group-32 host kernels; numpy predicted 0.9921 for correctly applied group-64
weights (miner, same layer, same ids), so the device computes what was asked, ids >= 228 included. 025's reading of
the plugin source (group 32 is refused on Xe2+, the refusal surfaces as the failed cast) is confirmed.

**Side by side, one call per layer, rank 5 idle (median of 33, microseconds):**

| layer | path | "1 row" (padded to 2, same experts twice) | 2 rows (14 experts) | 3 rows (padded to 8) |
|---|---|---|---|---|
| 30 | decode (GEMV) | 3758 | 4899 | 15287 |
| 31-35 | prefill (grouped GEMM) | 2914-3263 | 4743-5301 | 7087-7933 |

* The decode path reads at the bus limit: 16 expert reads (470 MB at group 64) in 3.76 ms = 125-136 GB/s, 64 reads in
  15.3 ms = 123-133 GB/s. The prefill path moves 85-92 GB/s (8 experts 2.98 ms, 14 experts 4.83 ms).
* The decode path never shares an expert between rows, and the engine pads one row to two, so in this canary a
  one-row call cost MORE on the decode path (the pad row re-reads the same eight experts). Unpadded, one row is
  ~0.4 + 1.75 ms = 2.1-2.2 ms against 2.98: **-0.8 ms per layer for one-row frames, nothing at two rows, a loss beyond**.

**Worth.** -5 ms per one-row frame. At 15 streams 7 of 11 frames carry one row: -3.2 ms per average frame (+8 %);
a lone stream's trip -55 ms (+13 %). Predicted was 7-9 ms per frame: the prefill path's fixed cost is ~1 ms a call,
not 1.5, and the rest of the gap was the pad row.

**Not rolled out.** Group 64 moves every MoE block's output by 12 % (cosine 0.992): a different model. The exact
route is the plugin rebuilt with sub-group 16 for group 32 (plus the 64-bit offset), one-row calls unpadded and a
threshold of 1-2 rows. Parked behind the stage-balance work (027, 028), which is exact and larger. The canary leaves
the fleet with 028 (library restored; `moe_ov_g64/layer_30`, 8 GB, stays on rank 5's disk until someone removes it).

15 streams, canary in place: 24.6 / 24.7 tok/s steady (019: 22.5-23.4; 024: 24.2-24.6). Per rank: rank 0 51.9 ms a
frame and 96 % busy, ranks 1-7 39-42 ms, ranks 8-9 35.5 ms, rank 10 35.1 + 11.6 ms of head.
