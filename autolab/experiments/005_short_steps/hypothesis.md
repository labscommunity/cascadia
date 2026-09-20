# 005 short engine steps (admission no longer lock-starved), drafter persistence, 176 streams

Binary 7994e0c4 (sha 600e47d1). Overrides = 004 + CASCADIA_STREAMS=192, API concurrency 512,
CASCADIA_STREAMS_SPEC_TABLE on rank 0's disk.

Predictions:
- burst TTFT: 48 requests enter the queue within a few stage times and go down as 6 prefill
  frames back to back: mean 80 s -> 25-35 s, and aggregate tok/s (which includes the ramp) rises
  toward the steady rate: 12.6 -> ~17 at 48 streams.
- 176 streams (16 rows/frame): t_frame ~ 17 + 60 + 6 x 84 x 0.69 = 425 ms -> 37 rows/s x ~0.85 = ~31 tok/s steady.
- single stream on UNSEEN prompts ("fresh" phases): the honest figure for speculation with the
  cross-request table after ~100 requests of traffic; expected 1.9-2.3 tok/s (a = 0.15-0.3).
