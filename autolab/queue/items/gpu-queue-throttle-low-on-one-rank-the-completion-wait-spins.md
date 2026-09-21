---
id: gpu-queue-throttle-low-on-one-rank-the-completion-wait-spins
title: GPU queue throttle LOW on one rank: the completion wait spins a core that the 25 W budget could give the iGPU
status: running
outcome: 
priority: 1
target: 15 streams
exact: yes
needs: overrides only, rank 6 A/B (rides on 029)
proposed_by: autolab session 2026-09-20
owner: autolab session
experiment: 029_decode_kernels_exact
created: 2026-09-20
updated: 2026-09-20
---

## Hypothesis
Every rank shows exactly 1.00 busy cores while its iGPU works and 18 W of package power under a 25 W platform limit;
ranks 8-10 with a higher limit are 12 % faster. A spinning P-core is 2-3 W. `OV_GPU_QUEUE_THROTTLE=LOW` lets the
driver sleep on the interrupt.

## Prediction
Rank 6: cores 1.0 -> < 0.5, ms per frame -3 to -5 %, or +3-6 ms if every host wait now costs a wake-up (60+ waits
per frame on the prefill path). Either way it is one release to know.

## Kill
Rank 6 slower than ranks 5 and 7 by more than 1 ms a frame.

## Result
