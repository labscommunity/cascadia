# 023: would two frames in flight INSIDE a box use the iGPU better? (load-time micro-benchmark, no serving change)

**Why.** At 15 streams a frame has 1-2 rows and takes 35-51 ms per stage; the fused-MoE op runs its prefill path
(8 host synchronisations, 516 sub-buffer objects and ~15 oneDNN wrappers per call, read in the 2026.3.1 source) and
the engine adds 4-6 ms of CPU work per frame. If the GPU idles during that host work, cutting a box's six layers in
two halves with disjoint state (one thread each, frame f+1 in the first half while frame f is in the second) raises
the box's frame rate without touching the model. That refactor is large, so the premise is measured first.

**Method.** `split_bench` at load on ranks 5 and 9: 48 synthetic frames (q/k/v/r, o, fused experts per layer, varying
expert ids, 1 and 2 rows) on one thread vs. pipelined over 2 and 3 threads. Microseconds per frame, relayed by the beacon.

**Prediction.** One thread ~31 ms per one-row frame (device calls only); two threads 21-24 ms; three threads ~20 ms.
**Decision rule.** Build the split stage if two threads give >= 1.25x.
