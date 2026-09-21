---
id: re-score-the-model-s-own-mtp-head-on-states-captured-from-th
title: re-score the model's own MTP head on states captured from the fleet (f16 fused experts, int8 attention), same 36 prompts as 033
status: running
outcome: 
priority: 7
target: single stream, 15 streams
exact: n/a
needs: collecting 36 original rendered prompts from the settled and gated fleet, then score on miner
proposed_by: autolab session 2026-09-20 (user: use the fleet, not the Mac Pro)
owner: autolab-continuation-20260921
experiment: 039_mtp_fleet_rescore
created: 2026-09-20
updated: 2026-09-21
---

## Hypothesis
033's a1 = 0.726 came from the CPU path's states and text. The head was trained on bf16 states; the fleet's are f16-fused
and its text is its own. The number that prices the fleet wiring is the one measured on the fleet's states.

## Prediction
a1 within 0.02 of 0.726 overall; if it is lower by more than 0.05 the wiring's gain shrinks accordingly.

## Result
