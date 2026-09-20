# Leaderboard (gate-passing configurations only)

| exp | date | config | single tok/s | TTFT s | agg tok/s @streams | note |
|---|---|---|---|---|---|---|
| 000 | 09-20 | attention+head iGPU, experts CPU, 48 slots, 11 groups | 1.59-1.69 | 9.4 | 3.4@11, 6.1@48 cold, 7.1@48 warm | baseline with profile build |
| 001 | 09-20 | + emptiest-group admission, StreamFeed windows (binary 71973619) | 1.43 (not warm) | 9.7 | 8.9 steady@11 (5.45 agg), 9.6@16 (6.4), **14.7@48 (8.98)** | 508-token prompt: TTFT 65 s instead of a chain rebuild |
