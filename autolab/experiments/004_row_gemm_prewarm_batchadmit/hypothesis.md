# 004 multi-row expert kernel, batched dense MLP, batched admission, pre-warm, 96 slots

Binary 1157d7eb (sha c246580d). Overrides = 002b + CASCADIA_INKLING_PREWARM=1, CASCADIA_STREAMS=96.
New in the binary: rows share expert reads (shared experts once per block, routed experts with
>= 2 rows once), dense MLP takes all rows in one pass, up to 8 waiting prompts per prefill frame,
cross-request drafter table, resident expert copy filled at load.

Predictions (002a: rank 0 46.7 ms/row at 3.9 rows/frame, ranks 1-7 40-43, fleet 18.4 tok/s steady @48):
- rank 0: two dense layers stop costing per row (2 x 3.9 x ~7.5 ms -> ~15 ms) and the shared
  experts of its 4 MoE layers are read once: 182 ms/frame -> ~120.
- ranks 1-7: shared experts 6 x 2 x R reads -> 6 x 2: 166 ms/frame -> ~140 at R = 4.
- so at 48 streams the slowest stage moves to ranks 1-7 at ~34 ms/row: ~26 tok/s steady.
- at 96 streams (8.7 rows/frame) routed experts start to repeat (70 pairs -> ~50 distinct per
  layer): ~30 ms/row -> ~30 tok/s.
- burst TTFT: 48 requests in ~6 prefill frames instead of 48: mean 91 s -> ~30 s.
- first request after the restart at full speed (pre-warm): gate tok/s >= 1.6 right away.
Gate now also runs 8 requests side by side, because every new kernel only runs with >= 2 rows.
