---
id: fifteen-one-row-frames-in-the-ring-cascadia-streams-inflight
title: fifteen one-row frames in the ring (CASCADIA_STREAMS_INFLIGHT=15) with the decode kernels on every layer
status: dropped
outcome: 
priority: 2
target: 15 streams
exact: yes
needs: dropped: its gain rested on the decode kernels being faster for one-row frames (029: they are not); fifteen one-row frames alone put 15 head calls a round on rank 10 (648 ms)
proposed_by: autolab session 2026-09-20
owner: 
experiment: 
created: 2026-09-20
updated: 2026-09-20
---

## Hypothesis
028 showed the ring is paced by its LONGEST frames (convoy): 15 streams in 11 frames = four two-row frames (~52 ms
a stage) that the seven one-row frames (~35 ms) queue behind; the round is ~11 x T(2 rows) = 544-569 ms. With fifteen
one-row frames the round is 15 x the slowest stage's ONE-row time, and one-row frames are exactly where the plugin's
decode kernels win (026: -0.8 ms a layer, -5 ms a frame).

## Method
Two steps, one release each: (a) 029's patch and `CASCADIA_INKLING_OV_MOE_DECODE_LAYERS=all` on every rank (load
check enforced everywhere, self-reverting per rank); (b) `CASCADIA_STREAMS_INFLIGHT=15` on rank 0. Measure
mix15 x 2, one stream, 8 streams, 176 streams.

## Prediction
(a) alone: +3-5 % at 15 streams (only 7 of 11 frames are one-row and the two-row frames still pace the ring), a lone
stream's trip -55 ms (+13 %). (a)+(b): round 15 x ~32 ms = 480 ms = **~31 tok/s (+25 % on 24.8)**.

## Kill
Any load check under 0.995, a gate failure, 15-stream steady below 24.5, 176 streams below 60.

## Result
