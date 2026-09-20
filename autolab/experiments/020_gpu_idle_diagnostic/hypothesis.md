# 020: is the iGPU idle during the fixed per-call cost? (diagnostic + per-rank A/B)

**Why.** At 15 streams a frame has 1-2 rows and costs a middle rank 35-51 ms, of which ~13 ms do not scale with rows:
12 attention calls at ~0.3 ms and 6 fused-MoE calls at ~1.5 ms of fixed cost each (019). If that time is HOST work
(OpenVINO's dynamic-shape flow: shape inference, kernel selection, argument setting per primitive per call, plus this
engine's tensor handling), the GPU sits idle for a third of every frame, and a second frame in flight inside the box
could use it. If it is device time, that idea is dead and only fewer/larger graphs help.

**Method.** One binary (19bc7207). Rank 5 compiles with `PERF_COUNT` and reports, per 10 s window, wall time, time
inside `infer()` and summed device execution time for the attention and MoE calls. Ranks 1-4 write inputs into the
request's own (USM host) tensors instead of binding a fresh tensor per call; ranks 6-7 keep the old path: a per-rank
A/B on identical 25 W boxes. Load: 15 mixed streams, and 11 streams (exactly one row per frame).

**Prediction.** Device time is 60-75 % of wall time in the MoE calls and ~65 % in the attention calls. Input reuse
saves 0.05-0.15 ms per call (1-2.5 ms per frame).
