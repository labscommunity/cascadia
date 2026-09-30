# 006 verdict: KEEP. Burst TTFT 73 -> 44 s, 34.9 tok/s steady at 264 streams, the 25 W limit is measured.

Gates: single PASS (exact), 8 side by side PASS (exact).

| phase | 005 | 006 |
|---|---|---|
| 48-request burst: TTFT mean (max) / aggregate / steady | 73 s (93) / 15.2 / 25.6 | **44 s (55) / 18.7 / 26.4** |
| 264 streams (19 rows/frame) | - | **34.9 steady / 20.6 aggregate** (251 of 264 completed: 13 client connections were reset in the tunnel, not on the fleet) |
| single stream, unseen prompt | 1.92 | 1.99 |
| gate prompts, single stream | 2.2-2.6 | **8.5, 6.5, 5.3 tok/s** |

- The gate prompts' speed is the drafter table from 005, now kept on disk: it has seen these exact
  prompts and answers many times, so almost every guess is right. That is not a general number,
  but it is the measured top of the mechanism on this fleet: with a draft that is right about nine
  times in ten one stream runs at 8.5 tok/s, where the time model says 8.6.
- Platform watts (new telemetry), 264 streams: ranks 0-7 psys 23.7-25.2 W (limit PL1 = 25 W),
  package 21.7-23.4 W, clocks 1.8-2.0 GHz; ranks 8-10 psys 42-48 W, package 31-36 W, P cores up to
  3.2 GHz, 68-79 C. The platform limit is what holds eight of eleven boxes: their time per distinct
  expert is 0.69 ms against 0.46 ms (the memory bus limit) on the other three.
- Frame time at 19 rows: predicted 475 ms on the 25 W boxes, measured 396-472.
