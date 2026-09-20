# Journal

## 2026-09-20, iteration 000: instrument first

Question: where does the time go? Nothing on the fleet could answer it (every rank says
"serving"). Built a per-rank stage profile into the engine and a telemetry packet into the
beacon, rolled both out, ran 48/1/11/16/48 streams plus long prompts.

Found: (1) the ranks idle 65-70 % at 48 streams because streams pile into the first groups;
(2) experts are not shared between rows at all; (3) eight boxes are power-capped at 25 W;
(4) prompts over 256 tokens crash the chain; (5) LAN ping is 2-3.5 ms; (6) prefill costs as much
per row as decode. The long-prompt phases took the chain down four times; Tailscale on rank 0
went away mid-run, so the recording of the last phases stayed on rank 0.

Reasoning for the next step: the two targets need different things. Aggregate: fill the
pipeline (M1, M4), then make rows share expert reads (M2) with the iGPU doing the GEMM (M3).
Single stream: the sequential bandwidth ceiling is 2.4 tok/s; only several positions in flight
(S1) can pass it. First release batches M1, T1, the telemetry door and the profile fix.
