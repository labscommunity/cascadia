# `ov-runtime` — multi-stage stateful KV cache

Pipeline-parallel inference using pre-exported per-stage OpenVINO IRs with internal stateful KV cache (ReadValue/Assign ops).

## When to use

- The target model is too big for one machine.
- You have already exported per-stage shards (no auto-export support yet).
- You don't need speculative decoding (use `ov-dist-spec` for that).

## Shard format

Pre-export with `cascadia shard` (the exporter emits the v5
`v5_canonical_inputs` layout; the engine also accepts legacy v3
shards):

```bash
cascadia shard \
    --model /path/to/Llama-3.1-8B-Instruct \
    --output-dir /shards/shards_2stage \
    --num-stages 2 --layer-split 16 --quantization int4
```

Produces:

```
<pipeline_dir>/
  pipeline_config.json
  config.json              # copy from the source HF model dir
  tokenizer/
  stage_<i>/
    openvino_model.xml
    openvino_model.bin
    stage_config.json
```

Current (v5) stage IRs use the canonical `(input_ids|hidden_states,
attention_mask, position_ids, beam_idx)` inputs; legacy v3 stage IRs
have `(input_ids|hidden_states, cos, sin)` positional inputs instead —
the engine auto-detects the layout from the stage IR's input names
(`export_version` in `stage_config.json` is informational). Both use stateful
KV cache. For v3 the Rust port sends `cos` / `sin` / `hidden_states` as
f16 to match the export's default dtype — v3 shards exported with
`--default-dtype fp32` are not currently supported.

`config.json` (the source model's HF config) is required at the
pipeline root or under `tokenizer/` so the Rust runtime can derive
`head_dim`, `rope_theta`, and `rope_scaling` for its on-the-fly RoPE.

For `ov-dist-spec` you need v5 shards instead — see [ov-dist-spec.md](ov-dist-spec.md).

## Example

```bash
# Last stage (rank 1):
cascadia worker --rank 1 --total 2 --engine ov-runtime --device GPU \
              --model /shards/shards_2stage_v3 \
              --listen 10.10.10.2:9100

# First stage (rank 0):
cascadia worker --rank 0 --total 2 --engine ov-runtime --device GPU \
              --model /shards/shards_2stage_v3 \
              --next 10.10.10.2:9100 --api :8000
```

## Performance notes

- Cross-stage activation transfer is hidden_states float16 — small enough that LAN/TB is rarely the bottleneck.
- Same shards file path on every node simplifies launch (use `--model <same-path>` everywhere).
- Tokenizer is loaded from the model_id's HF snapshot if the bundled `tokenizer/` dir uses a class the local `transformers` install can't import (common with older shard exports).
