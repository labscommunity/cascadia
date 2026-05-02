# `ov-spec` — single-stage speculative decoding

Runs a target + draft model on one machine, using mask-based KV-cache rewind to skip rejected drafts cheaply.

## When to use

- Model fits comfortably on one node.
- You have a draft model with a matching tokenizer.
- You want roughly 2× the throughput of the plain `ov-optimum` engine.

## Shard format

Both target and draft must be optimum-cli-exported INT4 IRs (canonical inputs `input_ids, attention_mask, position_ids, beam_idx`). The engine auto-exports on first run via `optimum-cli export openvino`.

## Example

```bash
tahoma worker --rank 0 --total 1 --engine ov-spec --device GPU \
              --model unsloth/Meta-Llama-3.1-8B-Instruct \
              --draft-model unsloth/Llama-3.2-1B-Instruct \
              --spec-k 4 \
              --api :8000
```

## Picking K

Measured on Arc B390 (alpha) with Llama-3.1-8B + Llama-3.2-1B:

| K | tok/s | accept |
|---|-------|--------|
| 3 | 24.6  | 0.81   |
| **4** | **35.0** | **0.91** |
| 5 | 28.1  | 0.83   |
| 7 | 26.2  | 0.72   |

K=4 is optimal here because the 1B draft is unusually well-aligned with the 8B target. For weaker drafts the sweet spot drops to K=3.

## Implementation notes

- Bypasses optimum-intel's `_assisted_decoding` (transformers 4.57's path calls `outputs.past_key_values.crop()` which optimum-intel 1.27 doesn't implement).
- Mask-based rewind: `attention_mask[i] = 0` for rejected drafts instead of physically trimming KV cache. ~40 ms per physical trim on Intel iGPU per rainier DISCOVERY #20.
- Greedy only (no temperature sampling); add sampling support if you need it.
