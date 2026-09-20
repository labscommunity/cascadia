# 002a verdict: KEEP. The pipeline is full now; the slowest stage sets the pace.

Gate: PASS (3/3 character-identical), single-stream gate requests 1.60-1.68 tok/s.

| phase | 001 | 002a | change |
|---|---|---|---|
| single, steady | 1.43 | 1.52 (not fully warm: expert cache 27-42 GiB and still growing, misses 0.4-1.4 %) | - |
| 11 streams: aggregate / steady | 5.45 / 8.92 | 8.37 / 12.87 | +54 % / +44 % |
| 48 streams: aggregate / steady | 8.98 / 14.69 | 11.21 / 18.35 | +25 % / +25 % |
| vs the original fleet (exp 000) at 48 streams | 7.1 / 9.7 | 11.2 / 18.4 | +58 % / +89 % |

Stage profiles, 48 streams, decode windows (first per-rank view through /api/fleet/telemetry):

- utilization: rank 0 **96 %**, ranks 1-7 84-90 %, ranks 8-10 62-71 % (they wait for the 25 W
  boxes). Fleet mean 82 % (exp 000: 30 %). Replies relayed: 0.
- ms per row: rank 0 46.7, ranks 1-7 40-43, ranks 8-10 30. Rank 0 is the slowest stage although
  it holds only 4 MoE layers: its two dense layers run row by row.
- MoE per row per rank is still flat in the batch size (33-39 ms/row at 3.9 rows/frame).
- hypothesis check: group turn / stage time went from ~1.7 to ~1.05. Confirmed.

Consequence: scheduling is no longer where the time goes. From here throughput = rows per frame /
time per frame of the slowest stage, so the next gains must come from (1) the multi-row expert
kernel and batched dense MLP (rank 0 first), (2) more rows per frame, (3) the 25 W limit.
TTFT under a burst is unchanged (91 s mean): batched admission ships in the next binary.
