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
| **Distributed alpha+charlie/TB4 (factual, 4096-tok output)** | `ov-dist-spec` | **K=5, Llama 3.2 1B INT4 draft, async overlap, v5 16/16** | **29.47** | **Q3.4 (BAR CLEARED, +5.2%)** |
| Distributed alpha+charlie/TB4 (factual, 4096-tok) | `ov-dist-spec` | K=4, Llama 3.2 1B INT4 draft, async overlap | 28.46 | Q3.4 |
| Distributed alpha+charlie/TB4 (creative, 4096-tok) | `ov-dist-spec` | K=5, Llama 3.2 1B INT4 draft, async overlap | 25.96 | Q3.4 (under bar) |
| Distributed alpha+charlie/TB4 (factual workload, 256-tok) | `ov-dist-spec` | K=3, Llama 3.2 1B INT4 draft, async overlap | 19.69 | Q3.2 |
| Distributed alpha+charlie/TB4 (factual workload, 256-tok) | `ov-dist-spec` | K=3, Llama 3.2 1B INT4 draft, sync, v5 16/16 | 18.42 | Q3.X |
| Distributed alpha+charlie/TB4 (factual workload) | `ov-dist-spec` | K=1, FastDraft 150M, v5 16/16 | 15.81 | e8 |
| Distributed alpha+charlie/TB4 | `ov-dist-spec` | K=1, FastDraft 150M, v5 16/16 | 11.78 | e2 |
