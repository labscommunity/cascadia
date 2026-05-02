# `ov-dist-spec` — distributed speculative decoding

Combines pipeline parallelism with speculative decoding: the **driver** holds a small fast draft model plus stage 0 of the target; **workers** hold downstream stages and respond to FORWARD frames over TCP.

## When to use

- The target model fits across `N` machines but not on one.
- You have a draft model that shares the target's tokenizer (e.g. Llama-3.2-1B for a Llama-3.1 target).
- You want the spec-decode token-rate boost without losing the ability to run a model bigger than one node.

## Shard format

Requires **v5 shards** (canonical optimum-style inputs: `input_ids|hidden_states, attention_mask, position_ids, beam_idx`). v3 shards are rejected at load time. Generate via rainier's `scripts/export_cached_shards_v5.py`.

Driver expects the full pipeline directory:
```
<pipeline_dir>/
  pipeline_config.json
  tokenizer/                 (optional; falls back to model_id's HF cache)
  stage_0/
    openvino_model.xml
    openvino_model.bin
    stage_config.json
  stage_1/...
```

Workers can hold just their own stage as a flat dir:
```
<worker_model_dir>/
  openvino_model.xml
  openvino_model.bin
  stage_config.json
```

## Wire protocol

`tahoma/worker/engines/openvino/dist_spec_protocol.py` defines three frames:

- **FORWARD** (driver → worker → next worker): `[kind=1][logical_pos_start]` + `attention_mask` (int64 `[1, total_seq_len]`) + `hidden_states` (float16 `[1, new_tokens, hidden_size]`)
- **RESET** (driver → workers, propagated): `[kind=3]`
- **LOGITS_RESPONSE** (last worker → upstream): `[kind=4]` + `logits` (float16 `[1, new_tokens, vocab_size]`)

Mask-based rewind is **driver-side only** — the worker just sees a new `attention_mask` on the next FORWARD. There is no REWIND frame.

## Topology

```
driver (rank 0, has draft + stage_0 + tokenizer)
       ⇣ FORWARD
worker (rank 1, has stage_1)
       ⇡ LOGITS_RESPONSE
```

## Example

```bash
# Worker (last stage, listens on TB/LAN port):
tahoma worker --rank 1 --total 2 --engine ov-dist-spec --device GPU \
              --model /shards/shards_2stage_v5_beam_stage_1 \
              --listen 10.10.10.2:9100

# Driver (rank 0, holds draft, serves the API or stdin):
tahoma worker --rank 0 --total 2 --engine ov-dist-spec --device GPU \
              --model /shards/shards_2stage_v5_beam \
              --next 10.10.10.2:9100 --api :8000 \
              --draft-model unsloth/Llama-3.2-1B-Instruct \
              --spec-k 4
```

## Picking K

K is the draft length per spec round. Higher K = more parallel target verifications but more wasted work when the draft is wrong.

Measured on Llama-3.1-8B INT4 target + Llama-3.2-1B INT4 draft, alpha+charlie via TB:

| K | 64-tok factual | accept | 256-tok creative | accept |
|---|----------------|--------|------------------|--------|
| 3 | 15.84 tok/s    | 0.76   | 14.68 tok/s      | 0.55   |
| **4** | **16.11**  | 0.62   | **15.52**        | 0.50   |
| 5 | **16.87**      | 0.65   | 13.91            | 0.42   |
| 6 | 15.59          | 0.60   | 12.70            | 0.38   |

**K=4 is the recommended default.** K=5 wins narrowly on short factual prompts but loses ~11% on long creative prompts. Push K higher only if your accept rate is consistently ≥ 0.7.

## Performance notes

- Mask-based rewind (v5) is ~free vs the ~40 ms-per-call physical rewind that v3 shards forced. The win shows up most on long generations and low-accept prompts.
- Per-FORWARD network cost dominates on slow links. Thunderbolt 4/5 between hosts on AC power gives a 25% throughput bump over LAN; on battery the link drops under load.
- `--device GPU` resolves to whatever Intel iGPU/dGPU OpenVINO sees; `tahoma engines` does not currently list devices.
