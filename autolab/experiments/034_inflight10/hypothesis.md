# 034: ten frames in flight at fifteen streams

The measured round model is max(25 F + 255, 179 + 3135/F) ms. Reducing
frames F from 11 to 10 predicts roughly 5% higher combined decode rate
(24.8 to 26 tokens/s) without changing tokens. Only rank 0's
CASCADIA_STREAMS_INFLIGHT changes; preserve experiment 032's role swap.

Measure identical phase names and experiment tag before and after: one
stream, two mixed fifteen-stream phases, and 176-stream throughput.
Keep if both fifteen-stream phases improve without regression elsewhere.
Kill if fifteen streams fall below 24.5 tokens/s twice, 176 streams fall
below 60 tokens/s, either correctness gate fails, or any worker fails.
The revert is the currently published 032 overrides and 8baebd3c binary.
