# 004 verdict: KEEP everything. MoE time now equals (distinct experts) x (one read); admission is lock-starved.

Gates: single PASS (3/3 exact), 8 side by side PASS (the three reference prompts exact while
decoding next to five others: multi-row kernel, batched dense MLP and batched admission are exact).

| phase | 002a/002b | 004 | note |
|---|---|---|---|
| single stream | 1.52-1.77 tok/s, TTFT 9.6 s | **2.82 tok/s, TTFT 5.8 s** | pre-warm (no misses) + speculation + faster prefill; this prompt was seen before in this process, so the drafter had memorised part of it: see 005 for unseen prompts |
| gate prompts | 1.77-2.13 | 2.17-2.59 | |
| 11 streams steady | 12.9 | 13.4 | one row per frame: nothing to share |
| 48 streams steady / aggregate | 18.4 / 11.2 | **20.7 / 12.6** | |
| 96 streams steady / aggregate | - | **24.5 / 13.9** | TTFT mean 150 s: see below |

Per rank at 48 streams (decode windows): rank 0 181.7 -> 129.6 ms/frame (65 % busy: no longer the
bottleneck), ranks 1-7 160-166 -> 128-148 (95-99 % busy: they are), ranks 8-10 104-112 -> 84-91
(63 % busy). Prefill cost per prompt row fell 2.5x (34-37 -> 14-15 ms on the 25 W boxes, 24.7 -> 9.6
on the others).

The model that now fits every rank and batch size:

    t_frame(R) = ~17 ms + attention(R) + 6 layers x U(R) x t_expert
    U(R) = 256 (1 - (250/256)^R) + 2 distinct experts per layer;  t_expert = 0.69 ms (25 W), 0.46 ms (60 W)

R = 6.4: predicted 211 ms, measured 205. So decode is one memory read per distinct expert, and
more rows per frame is the remaining software lever for aggregate throughput:
R = 16 -> ~38 rows/s, 32 -> ~46, 64 -> ~55-60 on the 25 W boxes (x 0.82 utilization today).
With t_expert = 0.46 ms everywhere (no platform power limit) the same points are 52 / 64 / 80.

Burst TTFT did not fall as predicted (80 s at 48, 150 s at 96) although prefill work per rank fell
from 40 s to 17 s: requests reach the engine's queue a few per round, because a submit needs the
engine lock and one engine step (= one lock hold) was a whole round of 11 group turns. Fixed in
binary 7994e0c4 (a step ends at the first productive group turn) -> exp 005.
