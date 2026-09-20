# Leaderboard (gate-passing configurations only)

| exp | date | config | single tok/s | TTFT s | agg tok/s @streams | note |
|---|---|---|---|---|---|---|
| 000 | 09-20 | attention+head iGPU, experts CPU, 48 slots, 11 groups | 1.59-1.69 | 9.4 | 3.4@11, 6.1@48 cold, 7.1@48 warm | baseline with profile build |
