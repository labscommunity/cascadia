# 001 balanced groups + windowed prompts

Binary 066c1728 (sha 71973619). Overrides unchanged from the baseline (attention + head on the
iGPU, experts on the CPU, 48 slots, 11 groups in flight, stage profile every 10 s).

Hypothesis: with new streams joining the emptiest group, 11 streams occupy 11 groups (1 row per
frame, 11 frames in flight) instead of 3 groups; expected aggregate at 11 streams: from 3.4 tok/s
toward 11 x 1.6 x utilization = 10-14 tok/s. At 48 streams (4-5 rows per frame in 11 groups instead
of 15/10/8/6/4/4/2): from 7 toward 15-20 tok/s. A 400-word prompt (about 530 tokens) must complete
instead of taking the chain down; expected TTFT about 22 s + 10 x 5 s = 70-80 s.
