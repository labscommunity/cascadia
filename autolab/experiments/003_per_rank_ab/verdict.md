# 003 verdict: the iGPU wins on the 25 W boxes. Fused MoE at f16 is exact now and halves a layer's time.

Gates: single PASS (exact), 8 side by side PASS (exact), with rank 6's layers 36-38 on the iGPU at f16.
(The run's totals are depressed on purpose: rank 3 ran 8 threads and became the bottleneck.)

Seven identical 25 W boxes, identical work, one run (decode windows):

| rank | setting | ms/frame @4.1 rows | ms/frame @14 rows | single stream ms/frame | package W @14 rows |
|---|---|---|---|---|---|
| 1, 7 | control (16 threads) | 132, 129 | 329, 296 | 57.9, 59.4 | 19.6, 18.5 |
| 2 | 12 threads | 152 (+16 %) | 352 | 67.6 | 20.1 |
| 3 | 8 threads | 187 (+43 %) | 455 | 79.4 | 22.1 |
| 4 | 12 threads pinned to CPUs 0-11 | 146 (+12 %) | 334 | 62.5 | 20.3 |
| 5 | PM QoS, no deep C-states | 133 (same) | 298 | **53.7 (-8 %)** | 24.1 |
| **6** | **3 of 6 MoE layers fused on the iGPU, f16 + weight rescale** | **99.7 (-24 %)** | **254 (-19 %)** | **50.0 (-15 %)** | **14.1** |

- Rank 6: 0 fallbacks, 0 non-finite outputs in 1300 frames; a fused layer takes 23.7 ms at 15 rows
  against ~46 ms for a CPU layer on the same box, at 5 W less package power. With three fused
  layers a 25 W box is as fast as the 60 W boxes on the CPU path (254 vs 195-203 ms).
  The earlier "fused is slower than CPU" was f32; "fused gives NaN" was the routing weights
  (sum 8 x global_scale, ~100 each on deep layers) overflowing f16 in the weighted sum.
- Fewer threads lose, also with the low-power E cores excluded: all 16 cores earn their watts.
- Holding the CPUs out of deep C-states buys 8 % of a stage in single-stream latency and nothing
  under load, for +5 W per box at idle. Not adopted.
Next (007): fused f16 on every rank's three IR layers.
