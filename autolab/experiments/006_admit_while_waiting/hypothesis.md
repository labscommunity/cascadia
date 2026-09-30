# 006 rank 0 admits while it waits; 264 streams; platform watts in the telemetry

Binary d5128fcc (sha 60d09d29), beacon with psys watts / P- and E-core clocks (rig 53/53),
overrides = 005 + CASCADIA_STREAMS=272 + file-backed head table on the last rank.

Predictions:
- 48-request burst: six prefill frames back to back (17 s of rank 0's time), then the train crosses
  the pipeline: last first-token at ~50 s, mean TTFT ~30 s (005: 73 s); aggregate 15 -> ~19 tok/s.
- 264 streams (24 rows/frame): t_frame = 17 + ~80 + 6 x 113 x 0.69 = 565 ms -> 42 rows/s x 0.8 = ~34 tok/s.
- psys watts on the 25 W boxes sit at ~25 W under load while the package shows 14-17 W: the
  platform domain is the limiter. On the three other boxes psys follows the package.
