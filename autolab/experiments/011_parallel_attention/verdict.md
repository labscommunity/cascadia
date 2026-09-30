# 011 verdict: KEEP. 64.2 tok/s steady at 176 streams: the aggregate target is met in steady decode.

Gates: single PASS (2 exact, the third as in 007/009), 8 side by side PASS.

| phase | 009 | 011 |
|---|---|---|
| 176 streams: steady / aggregate | 57.9 / 34.8 | **64.2 / 37.0** |
| 264 streams | - | **60.9 / 38.4** (all 264 completed) |

Attention per frame at 16 rows: 33-54 -> 25-30 ms. Slowest stage: rank 1 at 197 ms (93.5 % busy),
the others 157-193; fleet utilization 83 %.
"steady" is the sum of the streams' own decode rates; "aggregate" divides all tokens by the
phase's wall time, and a 32-token phase spends a third of it admitting 176 prompts. 011b repeats
the measurement with 128 tokens per stream and with the server's own token counter.
