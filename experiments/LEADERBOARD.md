# Leaderboard

Best measured tok/s per (model × workload × topology). Updated after each campaign that sets a new high.

## Llama 3.1 8B INT4

| Topology | Engine | Config | tok/s | Source |
|----------|--------|--------|------:|--------|
| Single-node alpha | `ov-genai` | + FastDraft K=5, 256-tok creative | **23.01** | e0 |
| Distributed alpha+charlie/TB4 | `ov-dist-spec` | K=3, FastDraft 150M, v5 16/16 | _pending e1_ | _pending_ |
