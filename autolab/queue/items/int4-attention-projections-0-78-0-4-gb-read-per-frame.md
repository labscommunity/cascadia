---
id: int4-attention-projections-0-78-0-4-gb-read-per-frame
title: int4 attention projections (0.78 -> 0.4 GB read per frame)
status: proposed
outcome: 
priority: 25
target: 15 streams, single stream
exact: no
needs: USER DECISION (changes numerics); IR regeneration on the boxes; a wider quality gate than twelve questions
proposed_by: autolab session 2026-09-20
owner: 
experiment: 
created: 2026-09-20
updated: 2026-09-20
---

## Hypothesis
Attention projections are int8 and read once per frame whatever the row count: 9.7 ms of a 41 ms frame. Caveat
from 027: this GPU's generic int4 MatMul path reads at 26-31 GB/s against int8's 80, so int4 through the ordinary
FC kernel may be SLOWER; measure one layer on one rank first.

## Prediction
-4 ms a frame (+10 %) if an int4 path at >= 80 GB/s exists; otherwise negative.
