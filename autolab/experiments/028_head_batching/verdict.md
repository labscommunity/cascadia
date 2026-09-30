# 028 verdict: the head calls are shared and the ring got 2.5 % slower; what paces it is its two-row frames

Binary 8baebd3c (the first rollout, 7561ffb8, was hit with traffic before it had settled: operator error, see the
journal; rolled back, re-run). `CASCADIA_STREAMS_HEAD_BATCH=2`, 026's canary removed.

| 15 streams x 128 tokens | 027 | 028 |
|---|---|---|
| steady tok/s (two phases) | 24.89 / 24.69 | **24.08 / 24.16** |
| rank 0's round trip | 544 ms | **569 ms** |
| rank 10: layers + head per frame, waiting | 35.9 + 11.6 ms, 11.5 % | not reliable: rank 10's clock jumped a day ahead and `analyze.py` double-counts its records from here on (see 029) |
| head calls / frames served (rank 10, HB line) | 1 / 1 | 1024 / 1555 (52 % of frames shared a call) |
| ranks 1-9 waiting | 23-33 % | 25-36 % |
| one fresh stream, gates | 4.9 tok/s, pass | 3.5-3.8 (another prompt), pass |

The mechanism works (test `inkling_streams_head_batch`, HB counter), rank 10's head time per frame fell, yet every
other rank waits MORE. No stage is saturated (the sum of all ranks' work per frame is 422 ms; the round takes 569), so the
ring is not paced by a slow stage's average at all:

**the convoy.** Fifteen streams in eleven frames = four frames of two rows and seven of one. A two-row frame takes
~51-54 ms per stage, a one-row frame ~35; in a ring with FIFO stages the short frames catch up with the long ones
and then travel at their pace: the round is ~11 x T(2 rows) = 557-594 ms, not 11 x T(1.36) = 451. Measured 544-569.
Deferring a reply by one frame's layers (+36 ms for half the frames) adds to that; saving head time on a rank that
is not the constraint returns nothing.

Consequences, in order:
1. Frames of EQUAL size. `CASCADIA_STREAMS_INFLIGHT=15` gives fifteen one-row frames: the round becomes 15 x the
   slowest stage's one-row time (rank 0: ~37 ms -> 555 ms: no gain yet) ...
2. ... and one-row frames are exactly where the plugin's decode kernels win (026: -0.8 ms per layer, -5 ms per
   frame): 15 x ~32 = 480 ms = 31 tok/s (+15 %). That is 029 (exact route) -> 030.
3. With fifteen frames on eleven stages four frames always queue, mostly at the last rank: head sharing then has a
   standing queue to work on and costs no extra wait. Kept in the binary, off (=1) until then.
