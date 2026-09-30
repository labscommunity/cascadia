# 009 verdict: KEEP. 57.9 tok/s steady at 176 streams; more streams stop helping; memory is at the edge.

Gates: single PASS (2 exact, the third departs at character 81 as in 007: equivalent wording),
8 side by side PASS. Rank 0's dense layers compiled and run on the iGPU (no fallbacks).

| phase | 008b | 009 |
|---|---|---|
| single stream, unseen prompts | 3.00, 3.39 | **3.09, 3.59** (TTFT 5.3 s) |
| 176 streams: steady / aggregate | 55.6 / 32.7 | **57.9 / 34.8** |
| 352 streams (all 352 completed through the wide forward) | - | 54.9 / 35.4 |

- Rank 0: 229.5 -> 186.5 ms/frame at 16 rows. It is no longer the slowest stage.
- The fused path costs about 10 ms per row per rank for six layers at 12, 16 and 24 rows per
  frame alike: on the device rows do not share work the way the CPU kernel's rows did, so more
  streams only add memory. At 352 streams 3.5-4 GiB stay available (102 MiB on rank 10) and five
  ranks swap 1800-3100 pages/s in spite of swappiness 1; those ranks are the slow stages (attention
  79-89 ms per frame against 47-55 on the ranks that do not swap, single frames up to 0.75 s).
- Throughput = 1 / (ms per row of the slowest stage) x utilization = 1 / 13.9 ms x 0.79 = 57.

The aggregate target is 4 % away. What is left: attention (26-54 ms of a frame; its per-row part
runs row after row on a CPU that is otherwise idle now), utilization (79 %), memory headroom.
