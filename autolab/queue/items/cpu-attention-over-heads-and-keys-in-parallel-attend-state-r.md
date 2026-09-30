---
id: cpu-attention-over-heads-and-keys-in-parallel-attend-state-r
title: CPU attention over heads and keys in parallel (attend_state runs one row's 64 heads on one core): up to ~12x on decode and prefill at long context
status: ready
outcome: 
priority: 2
target: context length, single stream
exact: yes
needs: engine change in attn.rs (rayon over heads / key blocks); 047 measured 11 GFLOPS effective per box against 16 cores
proposed_by: autolab session 2026-09-30
owner: 
experiment: 
created: 2026-09-30
updated: 2026-09-30
---

## Why
047: at 64k context a token costs 193 ms of attention per box (2.1 GFLOP = 11 GFLOPS), and a 64k prompt's prefill
spends ~90 of its 110 minutes in the same loop (67 TFLOP at 11 GFLOPS). Every box had exactly one core busy.
`attend_state` computes one row's 64 query heads one after another; only rows are parallel (011), and a lone
stream's frame has one row. Sixteen cores are idle.

## Method
Parallelise over query heads (64 independent dot-product/softmax/accumulate passes over the same keys) with the
existing rayon pool, then over key blocks with a two-pass softmax for very long contexts. Exact: the same sums in
another order (bf16 rounding points unchanged); gate with the reference outputs and the long-prompt tests.

## Prediction
Decode at 64k: 0.4 -> ~2.5 tok/s per stream (11 x (45 + 16) ms); prefill of 64k: 110 -> ~15 min; 128k reachable in
under an hour.

## Kill
Any gate departure; less than 4x on the 64k decode probe.
