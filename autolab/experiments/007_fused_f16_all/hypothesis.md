# 007 fused f16 MoE on every rank's three IR layers

Binary d5128fcc (unchanged), beacon unchanged, overrides = 006 + OV_MOE on with each rank's own layers at f16.
Pre-warm stays on (the binary skips layers served by the device).

Predictions from rank 6 in exp 003: the 25 W ranks go from ~130 to ~100 ms/frame at 4 rows and from
~310 to ~255 at 14 rows; the slowest stage then is a 25 W rank at ~255 ms -> 176 streams: 28-30 ->
~36 tok/s steady, 264 streams: 35 -> ~42. Single stream: stage 58 -> 50 ms on eight ranks:
1.69 -> ~1.9 without guesses, unseen-prompt speculation ~2.2.
Risks watched in telemetry: ov_moe fallbacks / non-finite per rank (layer 8's shared expert once
reached -94909 inside the expert: those calls would fall back to the CPU path, which now reads
experts on demand on fused layers), memory on rank 10 (head + attention + 25 GB of fused layers
on the iGPU), first-request compile time (fused layers compile while loading).
