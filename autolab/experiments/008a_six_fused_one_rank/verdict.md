# 008a verdict: six fused layers fit and make a 25 W box as fast as a 60 W one. Roll out to every rank.

Gates PASS (as 007). Rank 6 generated its three missing IRs on the box (run.sh), the iGPU was given
52 GiB, all six layers compiled: 0 fallbacks, no resident CPU expert copy (cache 0 MiB, process
anon 1.8-4 GiB), 5.9-8.5 GiB available, no swapping.

| rank 6 vs the other 25 W ranks (three fused layers) | single stream | 14-15 rows/frame |
|---|---|---|
| ms per frame | **36.4** vs 45.9-49.0 | **168.7** vs 230-261 |
| the 60 W boxes (three fused layers) | 36.1-36.5 | 165-185 |

Rank 1 with layer 8 back on the CPU (2 fused + 4 CPU layers) had no non-finite calls and no memory
trouble, but is the slowest stage (313 ms/frame), so the fleet total (42.9 tok/s at 176 streams)
is a little under 007's 44.6. 008b: every rank fuses every MoE layer; rank 1's layer 8 is
regenerated with its up scales x 2^-4 (binary 0c5ee1ee multiplies the output back).
