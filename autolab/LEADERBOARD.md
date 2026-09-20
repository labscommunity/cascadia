# Leaderboard (gate-passing configurations only)

| exp | date | config | single tok/s | TTFT s | agg tok/s @streams | note |
|---|---|---|---|---|---|---|
| 000 | 09-20 | attention+head iGPU, experts CPU, 48 slots, 11 groups | 1.59-1.69 | 9.4 | 3.4@11, 6.1@48 cold, 7.1@48 warm | baseline with profile build |
| 001 | 09-20 | + emptiest-group admission, StreamFeed windows (binary 71973619) | 1.43 (not warm) | 9.7 | 8.9 steady@11 (5.45 agg), 9.6@16 (6.4), **14.7@48 (8.98)** | 508-token prompt: TTFT 65 s instead of a chain rebuild |
| 002a | 09-20 | + direct reply link last rank -> rank 0, telemetry door (binary 12f998b8, beacon 7e3012f2) | 1.52-1.68 | 9.6 | 12.9 steady@11 (8.4 agg), **18.4@48 (11.2)** | fleet mean utilization 82 % (was 30 %); rank 0 96 % busy = the bottleneck |
| 002b | 09-20 | + pipelined speculation, per-request n-gram drafts | **1.77-2.13** free text, **2.79** on a copy task | 9.6 | (as 002a) | output character-identical; acceptance 10 % on reasoning text, 70 % on copying |
| 004 | 09-20 | + multi-row int4 expert kernel, batched dense MLP, batched admission, pre-warm, cross-request drafter, 96 slots (binary c246580d) | **2.82** (seen prompt), gate 2.2-2.6 | **5.8** | 13.4@11, 20.7@48 (12.6 agg), **24.5@96 (13.9 agg)** | both gates exact; MoE time = distinct experts x one read |
| 005 | 09-20 | + short engine steps, drafter table on disk, 192 slots (binary 600e47d1) | **1.9 on unseen prompts**, 3.15 copy task | 6.2 | 25.6@48 (15.2 agg), **29.6@176 (19.4 agg)** | frame-time model confirmed at 14 rows/frame; tokenizer = o200k |
