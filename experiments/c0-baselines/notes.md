# Campaign 0 — re-confirmed baselines + first surprises

## Measured numbers (this branch, 2026-05-02)

| ID | Node | Engine | Model | tok/s (decode) | accept | tokens | Notes |
|---|---|---|---|---|---|---|---|
| c0-1  | alpha B390 GPU | ov-optimum | Llama 3.1 8B INT4 | **8.85** | — | 64 | Cold OV cache. Load 26.8s. |
| c0-1b | alpha B390 GPU | ov-optimum | Llama 3.1 8B INT4 | **8.89** | — | 64 | Re-run; same speed. Load 23.6s. |
| c0-2  | charlie 140V GPU | ov-optimum | Llama 3.1 8B INT4 | **10.33** | — | 64 | Faster than alpha! Load 29.3s. |
| c0-3  | alpha B390 GPU | ov-spec K=4 | Llama 3.1 8B INT4 + 1B INT4 draft | **13.83** | 0.50 | 64 | accept far below baseline 0.91. |
| c0-4  | alpha+charlie / TB | ov-runtime | v5 shards | **failed** | — | — | Shape mismatch — engine is v3-only, doesn't accept v5 attention_mask input. |
| c0-5  | alpha+charlie / TB | ov-dist-spec K=4 | v5 shards | **17.59** | 0.62 | 64 | Matches main baseline 17.36. |

## Surprises that turn into experimental threads

### 1. alpha B390 dGPU is *slower* than charlie 140V iGPU on the same INT4 model

Saved baselines from main said alpha = 16.7 vs charlie = 17.0 (essentially tied). Today we measure alpha = 8.85 vs charlie = 10.33 — both ~halved, alpha clearly worse. Expected the discrete Battlemage to crush the iGPU for this workload; it doesn't.

Hypotheses for follow-up:
- alpha is in a low-power state (battery? thermal throttle? driver default?).
- Different OV plugin version on alpha vs charlie.
- The compiled blob is being recompiled every run (no `CACHE_DIR` set; alpha has 22 GB of cached blobs at `C:\cascadia\ov_cache` from rainier work that aren't being used).
- Battlemage drivers may be in an immature state vs the Lunar Lake Xe2 path (similar-arch but the 140V path is more battle-tested in OV).
- Power limit or `xpu-smi` settings.

### 2. ov-spec K=4 accept rate dropped from 0.91 → 0.50

Same prompt ("What is the capital of France?"), same target (Llama 3.1 8B INT4 at `C:\cascadia\models\llama-3.1-8b-int4`), same draft (`unsloth/Llama-3.2-1B-Instruct` autoexported to INT4 at `~/.cache/tahoma/ov_exports/`). Under those conditions the math should be deterministic.

Hypotheses:
- The INT4 quantizations (target and/or draft) re-exported with a different group size.
- A newer optimum-cli / nncf produced different rounding.
- Different `--task` flag at export time changed the IR (e.g. text-generation vs text-generation-with-past).

### 3. ov-runtime engine is broken on v5 shards

`ov-runtime` doesn't pass `attention_mask` into the per-stage IR — it sends only hidden_states between stages. v5 shards have 4 inputs; the second slot is `attention_mask`. The driver is sending the hidden state into the wrong input port; OV rejects with a shape mismatch.

This is a documented limitation but it deserves to be either:
- Fixed properly (extend `ov-runtime` to send attention_mask + position_ids alongside hidden_states), OR
- Documented + the engine should fail-fast at load time when handed a v5 shard layout.

### 4. ov-dist-spec is the only currently-correct distributed pathway

Both ov-runtime and any code expecting v3 layout will break against the v5 shards (which we want for the future TP / KV-mask-rewind work). The existing `ov-dist-spec` engine is the only distributed engine that handles v5 correctly. That's also the engine that matched its baseline number cleanly today.

## Session-current baselines (replaces the carried-from-main numbers)

These are what subsequent experiments compare against:

| Hardware | Engine | Config | tok/s |
|---|---|---|---|
| alpha B390 GPU | ov-optimum | greedy 64 | **8.85** |
| charlie 140V GPU | ov-optimum | greedy 64 | **10.33** |
| alpha B390 GPU | ov-spec K=4 | + 1B draft | **13.83** |
| alpha+charlie TB | ov-dist-spec K=4 | v5 shards | **17.59** |
