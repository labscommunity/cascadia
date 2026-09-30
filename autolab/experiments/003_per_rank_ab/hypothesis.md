# 003 per-rank A/B inside one run

Ranks 1-7 are identical boxes (25 W platform limit) with identical work (6 MoE layers), so one
run compares settings across them; ranks 1 and 7 stay as controls (their spread is the noise).

| rank | change | question |
|---|---|---|
| 2 | 12 rayon threads | under a power cap, do 16 threads waste watts stalled on DRAM? |
| 3 | 8 threads | same, further |
| 4 | 12 threads pinned to CPUs 0-11 | are the last four CPUs (low-power E cores) stragglers in row-parallel GEMV? |
| 5 | PM QoS: no deep C-states (/dev/cpu_dma_latency = 0 held open by the worker) | frame receive is 3.5 ms on these boxes vs 0.8 ms on ranks 9-10: wake-up latency? |
| 6 | 3 fused MoE layers on the iGPU at f16 with the weight rescale | is f16 now finite on deep layers (36-38), how many calls fall back, and is the iGPU faster than 16 capped cores? |

Read from the stage profiles: ms per row, attention / MoE split, recv per frame, ov_moe calls /
fallbacks / non-finite, package watts and clocks. The gate decides whether rank 6's output is sane.
