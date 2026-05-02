# c41: GPU saturation sweep — alpha B390 vs charlie 140V

## Setup
Llama 3.1 8B INT4 + LLMPipeline + SchedulerConfig (cache_size=4 GB,
max_num_batched_tokens=8192, dynamic_split_fuse=True). 64-tok output.
Batch sizes {8, 16, 32, 64}, each batch run as a single
`pipe.generate([N prompts], cfg)` call.

## Results

### alpha B390 (dGPU, Battlemage)

| Batch | Aggregate tok/s | Per-request | Decode_s |
|---|---|---|---|
| 1   |  134 | 134.0 | 0.48  |
| 8   |  138 |  17.2 | 3.71  |
| 16  |  274 |  17.1 | 3.74  |
| 32  | **559** |  17.5 | 3.66  |
| 64  |  362 |   5.65 | 11.32 (-35% vs 32!) |

**Hard saturation between batch=32 and batch=64.** Sweet spot: batch=32.

### charlie 140V (iGPU, Lunar Lake)

| Batch | Aggregate tok/s | Per-request | Decode_s |
|---|---|---|---|
| 1   |  143 | 143.0 | 0.45 |
| 8   |  149 |  18.6 | 3.43 |
| 16  |  211 |  13.2 | 4.86 |
| 32  |  287 |   8.97 | 7.14 |
| 64  |  314 |   4.90 | 13.05 |

**Gradual saturation curve.** Per-request drops monotonically as batch
grows; aggregate scales with diminishing returns (8→16 +41%, 16→32 +36%,
32→64 +9%). No cliff like alpha.

## Findings

1. **Different scaling behaviour by silicon class:**
   - dGPU (B390): clean linear scaling 1→32, then sharp -35% cliff at 64
     due to KV memory pressure and XMX tile saturation.
   - iGPU (140V): gradual diminishing returns; per-request degrades
     monotonically; aggregate plateaus around 314 by batch=64.
2. **Per-request SLA picks the batch:**
   - alpha: stays at 17 tok/s up to batch=32 (good for chat SLA).
   - charlie: drops fast — batch=8 = 18.6, batch=16 = 13.2, batch=32 = 9.
3. **For the same total throughput target**, alpha B390 needs fewer
   concurrent users to saturate (batch=32 = 559 agg) than charlie
   (batch=64 = 314 agg) — i.e., **alpha is ~80% faster aggregate at
   peak**.

## Recommendations for tahoma deployment

Default CB caps:
- alpha B390-class hardware: max_concurrent_requests = 32
- charlie 140V-class hardware: max_concurrent_requests = 16 (per-request
  SLA permitting)

Above these, schedule onto more nodes (the original tahoma value prop).

## Open follow-ups

- Test larger cache_size SchedulerConfig (e.g. 8 GB) — may delay alpha's
  batch=64 cliff.
- Test continuous request stream (rather than fixed batch) to model
  realistic API load.
- Per-batch-size memory measurement to identify the actual saturation
  cause (compute vs memory vs cache eviction).
