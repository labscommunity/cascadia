# 008b verdict: KEEP. Every MoE layer on the iGPU: 55.6 tok/s steady at 176 streams, unseen prompts 3.0-3.4 tok/s, >10 tok/s on a memorised prompt.

Gates: single PASS (3/3 character-exact, 12.3 / 9.4 / 7.6 tok/s on these memorised prompts),
8 side by side PASS (exact). Answers 12/12 (with layer 8 regenerated at up-scale exponent 4).
Every rank generated its missing IRs on the box and compiled all of its MoE layers: CPU expert
cache 0 MiB everywhere, 0 non-finite calls on rank 1 (layer 8 attenuated), 4 on rank 10.

| phase | 007 (3 fused layers per rank) | 008b (all fused) |
|---|---|---|
| single stream, unseen prompts | 2.59, 3.04 | **3.00, 3.39** (TTFT 5.1-5.4 s) |
| 48 streams: steady / aggregate / TTFT mean | 33.8 / 23.2 / 37 s | **38.3 / 23.8** / 43 s |
| 176 streams | 44.6 / 26.8 | **55.6 / 32.7** |
| 352 streams | - | invalid: only 251 connections got through (the Mac-side tunnel agent has 256 descriptors) |

Stage times at 14-16 rows/frame: ranks 1-9 160-203 ms (the 25 W and the 60 W boxes are now the
same speed: the iGPU does the work), rank 10 129 ms, **rank 0 230 ms and 96.8 % busy**: its two
dense layers are still on the CPU (about 60 ms of the frame). Swap-ins of 800-4200 pages/s on
several ranks with 6-7 GiB "available": the kernel swaps driver-owned pages at swappiness 60;
latency spikes of 0.5-1.8 s in single frames come from there.
Next (009): dense layers on the iGPU (binary ea80c2a8), swappiness 1, 352 streams through a
forward with enough descriptors.
