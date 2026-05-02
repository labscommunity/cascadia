# `ov-runtime` — multi-stage stateful KV cache

Pipeline-parallel inference using pre-exported per-stage OpenVINO IRs with internal stateful KV cache (ReadValue/Assign ops).

## When to use

- The target model is too big for one machine.
- You have already exported per-stage shards (no auto-export support yet).
- You don't need speculative decoding (use `ov-dist-spec` for that).

## Shard format

Pre-export with rainier's `scripts/export_cached_shards_v5.py` — produces:

```
<pipeline_dir>/
  pipeline_config.json
  tokenizer/
  stage_<i>/
    openvino_model.xml
    openvino_model.bin
    stage_config.json
```

Each stage IR has `(input_ids|hidden_states, attention_mask, position_ids, beam_idx)` inputs and stateful KV cache.

## Example

```bash
# Last stage (rank 1):
tahoma worker --rank 1 --total 2 --engine ov-runtime --device GPU \
              --model /shards/shards_2stage_v5_beam \
              --listen 10.10.10.2:9100

# First stage (rank 0):
tahoma worker --rank 0 --total 2 --engine ov-runtime --device GPU \
              --model /shards/shards_2stage_v5_beam \
              --next 10.10.10.2:9100 --api :8000
```

## Performance notes

- Cross-stage activation transfer is hidden_states float16 — small enough that LAN/TB is rarely the bottleneck.
- Same shards file path on every node simplifies launch (use `--model <same-path>` everywhere).
- Tokenizer is loaded from the model_id's HF snapshot if the bundled `tokenizer/` dir uses a class the local `transformers` install can't import (common with rainier exports).
