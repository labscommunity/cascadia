# Leaderboard

Best measured tok/s per (model × workload × topology). Updated after each campaign that sets a new high.

## Llama 3.1 8B INT4

| Topology | Engine | Config | tok/s | Source |
|----------|--------|--------|------:|--------|
| Single-node alpha | `ov-genai` | + FastDraft K=5, 256-tok creative | **23.01** | e0 |
| Distributed alpha+charlie/TB4 | `ov-runtime` | **no spec, v3 16/16** | **12.15** | e3 |
| Single-node alpha (factual) | `ov-genai` | + FastDraft K=5 | **23.30** | e7 (single trial) |

## Mixtral 8x7B Instruct INT4 (28 GB OV format)

| Topology | Engine | Config | tok/s | Notes |
|----------|--------|--------|------:|-------|
| Single-node alpha B390 (12 GB GPU) | `ov-genai` | LLMPipeline | **0.54** | Spilled to shared system memory — usable but glacial. Distributed has huge upside here once shards exist. |
| Distributed alpha+charlie/TB4 (factual workload) | `ov-dist-spec` | **K=1**, FastDraft 150M, v5 16/16 | **15.81** | e8 |
| Distributed alpha+charlie/TB4 | `ov-dist-spec` | K=1, FastDraft 150M, v5 16/16 | 11.78 | e2 |
