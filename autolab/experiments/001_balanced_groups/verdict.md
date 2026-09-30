# 001 verdict: KEEP (both changes)

Gate: PASS, 3/3 prompts character-identical to the reference.

| phase | baseline (exp 000) | 001 | change |
|---|---|---|---|
| single, steady tok/s | 1.59 | 1.43 (caches not fully warm: gate ran at 1.27-1.35 two minutes after the restart) | n/a |
| 11 streams: aggregate / steady | 3.39 / 4.19 | 5.45 / 8.92 | +61 % / +113 % |
| 16 streams | 5.5 / 7.4 | 6.42 / 9.61 | +17 % / +30 % |
| 48 streams | 6.1-7.1 / 8.3-9.7 | 8.98 / 14.69 | +27-47 % / +51-77 % |
| 508-token prompt | chain rebuild (2 min outage) | completes, TTFT 65 s, 4 windows | fixed |

"steady" = sum of the streams' own decode rates (excludes the admission ramp); "aggregate" =
all tokens over the phase's wall time. TTFT under a burst is unchanged (mean 87 s at 48
streams): admissions still go one at a time, each about 1 s on every rank.

Read: hypothesis confirmed in direction, not in size. 11 streams in 11 groups reach 8.9 tok/s
where the stage times allow ~16: rank 0's group turn takes ~1.7x a stage time at 11, 16 and 48
streams alike. Next: the reply path (002a).

Method note: the first window of the long prompt cost 132 ms/row on rank 0 and the next three
39 ms/row: first touches of cold experts. Timed phases need a many-stream warm-up after every
restart (added to lab.py: 16 streams x 24 tokens).
