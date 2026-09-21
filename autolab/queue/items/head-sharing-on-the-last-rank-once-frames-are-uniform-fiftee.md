---
id: head-sharing-on-the-last-rank-once-frames-are-uniform-fiftee
title: head sharing on the last rank once frames are uniform (fifteen frames on eleven stages always queue somewhere)
status: dropped
outcome: 
priority: 3
target: 15 streams
exact: yes
needs: dropped with the uniform-frames item (028: -2.5 % with mixed frames)
proposed_by: autolab session 2026-09-20
owner: 
experiment: 
created: 2026-09-20
updated: 2026-09-20
---

## Hypothesis
With more frames than stages, frames queue at the slowest stage. If that is rank 10 (layers + 11.6 ms of head per
call), a shared head call costs the waiting frame nothing extra and halves the head's cost. 028 measured the
mechanism (52 % of frames shared) but with mixed frame sizes it lost 2.5 %.

## Prediction
Rank 10's one-row frame 31 + 11.6 -> 31 + ~6 ms; worth it only if rank 10 is the slowest one-row stage (+5-8 %).

## Kill
No gain over the uniform-frames result within two phases.

## Result
