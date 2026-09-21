---
id: frames-in-flight-11-10-at-15-streams-cascadia-streams-inflig
title: frames in flight 11 -> 10 at 15 streams (CASCADIA_STREAMS_INFLIGHT=10 on the box that plays rank 0)
status: running
outcome: 
priority: 4
target: 15 streams
exact: yes
needs: overrides only (start from the last published overrides)
proposed_by: autolab session 2026-09-20
owner: autolab-continuation-20260921
experiment: 034_inflight10
created: 2026-09-20
updated: 2026-09-20
---

## Hypothesis
PHYSICS.md, "The 15-stream regime, as measured": a round is the larger of the last rank's work per round
(25 F + 255 ms) and one frame's trip (179 + 3135/F ms). F = 11 gives 532 ms, F = 10 gives 507 ms.

## Prediction
15 streams 24.8 -> ~26 tok/s (+5 %); 176 streams unchanged (check it); one stream unchanged.

## Kill
15-stream steady below 24.5 in two phases, or 176 streams below 60.

## Result
