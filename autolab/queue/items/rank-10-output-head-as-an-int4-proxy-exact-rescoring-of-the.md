---
id: rank-10-output-head-as-an-int4-proxy-exact-rescoring-of-the
title: rank 10 output head as an int4 proxy + exact rescoring of the top candidates
status: dropped
outcome: 
priority: 90
target: 15 streams
exact: yes
needs: dropped: this GPU reads int4 through the generic FC path 3x slower per byte than int8 (027: 8.1 ms for 255 MB), so a 0.69 GB int4 head would take longer than the 1.24 GB int8 one; and 028 showed the head is not what paces the ring
proposed_by: autolab session 2026-09-20
owner: 
experiment: 
created: 2026-09-20
updated: 2026-09-20
---

## Hypothesis

## Method

## Prediction

## Kill

## Result
