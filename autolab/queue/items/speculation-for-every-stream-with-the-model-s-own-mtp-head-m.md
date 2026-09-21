---
id: speculation-for-every-stream-with-the-model-s-own-mtp-head-m
title: offline: the model's own MTP head (8 dense draft modules) names the next token 0.726 of the time (0.63-0.69 prose, 0.87 arithmetic); deeper modules need their context kept current; ~5-6 tok/s for one stream if wired
status: done
outcome: measurement
priority: 5
target: 15 streams, interactive speed at 3-8 streams, single stream
exact: yes
needs: DONE offline (033): hidden-state dump on the Mac Pro's CPU path + scoring on the build host. Follow-ups use the fleet state capture instead (queue)
proposed_by: teammate (spec.md E4/E1) + autolab
owner: background research agent
experiment: 033_mtp_offline_study
created: 2026-09-20
updated: 2026-09-20
---

## Hypothesis
A guess row costs a full set of expert reads, so at 15 streams speculation pays only with a drafter that is right
>= 0.8 of the time. The checkpoint ships one: 8 chained MTP modules (embed_norm, hidden_norm, input_proj
[6144, 12288], one dense Inkling block each; 10.5 GB bf16), dropped by the exporter. DeepSeek-V3-style heads accept
0.8-0.9 of first drafts.

## Method
Teammate's E4 with a reduced E1: final residuals at every position + rank-boundary residuals over the 013 corpus
(Mac Pro, CPU path), MTP forward in PyTorch from the reference modeling code, a1..a8 by prompt family, teeth check.

## Prediction
a1 0.75-0.85 on prose. On the fleet: +12-15 % at 15 streams, ~2x per stream at 3-8 streams, 3.4 -> 5-6 tok/s alone.

## Kill
a1 < 0.7 on prose, or > 25 ms per draft on one bus.

## Result
