---
id: speculation-for-every-stream-with-the-model-s-own-mtp-head-m
title: speculation for every stream with the model's own MTP head (mtp.safetensors, 8 dense draft blocks): offline acceptance first
status: running
outcome: 
priority: 5
target: 15 streams, interactive speed at 3-8 streams, single stream
exact: yes
needs: offline: Mac Pro dump + miner scoring (background agent running); fleet part blocked on a1 >= 0.7
proposed_by: teammate (spec.md E4/E1) + autolab
owner: background research agent (offline part)
experiment: 
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
