# 008b every MoE layer of every rank on the iGPU

Binary 0c5ee1ee (sha 5e2d86b6: honors cascadia_moe.json), run.sh with per-layer regeneration,
overrides: all MoE layers fused, layer 8 regenerated with exponent 4, 52/54 GiB for the iGPU, 360 slots.

Predictions from rank 6 in 008a: every 25 W rank ~170 ms/frame at 14 rows (now 230-260), the 60 W
ranks ~125 (now 165-185); slowest stage ~170-180 ms -> 176 streams ~60 tok/s steady x utilization
~0.85 = ~50; 352 streams (32 rows/frame) higher still. Single stream: every stage ~36 ms ->
round trip ~420 ms: 2.4 tok/s without guesses, ~3.2 on unseen prompts, > 10 on memorised ones.
Risks: rank 10 (head + six layers on the device, ~3 GiB left), layer 8's attenuated scales
(quality check decides), generation failing on some box (that layer stays on the CPU path and
the rank is slower, nothing breaks).
