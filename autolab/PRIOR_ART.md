# What is already known (do not re-measure without a reason)

Measured on the venue fleet, 2026-09-19/20, warm unless noted.

| config | single tok/s | TTFT | 16 streams agg | 48 streams agg |
|---|---|---|---|---|
| pure CPU | 1.81 | 12.9 s | 4.88 | - |
| attention + head on iGPU, experts CPU (in force) | 1.68 | 9.5 s | 5.5 | 7.1 (TTFT mean 79 s) |
| + 3 fused MoE layers/box on iGPU at f32 | 1.66 | - | 4.2 | - |
| fused MoE at the plugin's f16 | NaN logits (`!!!!`) | | | |

Stage profile (exp 000, 2026-09-20, profiling build):

- single stream: every rank idle 91 %; per frame 53 ms on the 25 W boxes (attention 19, MoE 34),
  38 ms on the 60 W boxes (12 + 26); head 11 ms; frame receive 3.5 ms; round trip 562 ms = sum of stages.
- 48 streams: ranks idle 65-70 %; 6.9 rows/frame where 4.4 were expected; MoE 33 ms/row at
  6.9 rows/frame vs 34 ms/row at 1 (no sharing at all, not even the 2 shared experts).
- streams per group at 48 streams: 15, 10, 8, 6, 4, 4, 2, 0, 0, 0, 0. At 11 streams: 5, 3, 3.
- prefill: 37-41 ms per prompt row per rank (same as decode): a 24-token prompt is 0.9 s per
  rank, 11 ranks in series = TTFT 9.5 s. 530 tokens: 22 s per rank.
- expert cache misses 0.3-1 % a few minutes after a restart; first request after a restart
  TTFT 34-37 s, 0.8 tok/s.
- LAN ping 2-3.5 ms (USB NIC on `cdc_ncm`); clocks of the boxes differ by up to a day.
- power: ranks 0-7 psys PL1/PL2 = 25/31 W (package peaks at 30 W, ~2.2 GHz under load);
  ranks 8-10 PL2 = 148 W (package 60 W, 2.6 GHz, 96 C peaks on rank 10).
- RAM per rank: 61 GiB; worker anon 30-42 GiB (the expert cache is `EXPERT_CACHE_MIB=8000`
  PER LAYER, i.e. the resident copy of the experts), page cache holds a second copy of part of it.

Negative / closed:

- iGPU cannot beat the CPU on a single row: same memory bus (see PHYSICS.md).
- f32 fused MoE is slower than CPU experts at 1-2 rows/frame.
- Prompts > 256 tokens crash the chain (one `StreamOpen` > `MAX_STREAM_ROWS`): fixed in exp 001.
- Raising the 25 W platform limit from software is NOT done by this loop: the limit protects
  hardware nobody can reach if it browns out. Reported to the user instead.
