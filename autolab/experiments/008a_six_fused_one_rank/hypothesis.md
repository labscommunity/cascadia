# 008a one rank with all six MoE layers on the iGPU

Binary unchanged (d5128fcc). New run.sh (generates missing fused IRs on the box, can raise ttm's
pages_limit; dry-run with a stub generator: existing layers skipped, new ones moved into place only
when complete, a failing layer marked and tolerated, the worker starts regardless). Overrides = 007
+ rank 1 drops layer 8 from the device + rank 6 fuses 36-41 with the iGPU allowed 52 GiB.

Predictions for rank 6 (from its own 003/007 numbers: a fused layer 23.7 ms at 15 rows, a CPU layer ~46):
- 14-19 rows/frame: MoE 3 x 24 + 3 x 46 = 210 ms -> 6 x 24 = 144 ms; frame ~255 -> ~190 ms.
- single stream: 50 -> ~40 ms per frame.
- memory: 6 x 8.3 = 50 GiB on the device, no resident CPU copy (pre-warm skips fused layers):
  ~5 GiB available. The risky moment is compiling the sixth layer (8.4 GiB host copy in the shim).
Failure modes and what they look like: pages_limit not writable or too low -> layers 4-6 fail to
compile, stay on the CPU path, ov_moe calls per frame stay at 3; generation fails -> same; the
worker is OOM-killed while compiling -> rank 6 restart loop, fleet down until the next publish
(overrides without the rank 6 block).
