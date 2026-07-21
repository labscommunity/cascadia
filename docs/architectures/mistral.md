# Mistral family

The Mistral / Mistral-Nemo / Mistral-Small line is one of the
reference paths in `cascadia shard`. The text backbone is plain
GQA Llama-shape with an extra `bias=False` on the projections,
RMSNorm, RoPE, and (on the older 7B v0.1 only) sliding-window
attention.

## Variants

| Model | `model_type` | Hidden layers | Hidden | Heads / KV | RoPE θ | INT4 size | Notes |
|-------|--------------|--------------:|-------:|-----------:|-------:|-----------|-------|
| `Mistral-7B-Instruct-v0.1` | `mistral` | 32 | 4096 | 32 / 8 | 10k | ~4 GB | Sliding window 4096 (disabled in export — see below) |
| `Mistral-7B-Instruct-v0.2` | `mistral` | 32 | 4096 | 32 / 8 | 10k | ~4 GB | Sliding window removed |
| `Mistral-7B-Instruct-v0.3` | `mistral` | 32 | 4096 | 32 / 8 | 1M | ~4 GB | New tokenizer (`tokenizer.model.v3`), function-calling tokens, no sliding window |
| `Mistral-Nemo-Instruct-2407` | `mistral` | 40 | 5120 | 32 / 8 | 1M | ~7 GB | 12B params; Tekken tokenizer (131k vocab) |
| `Mistral-Small-24B-Instruct-2501` | `mistral` | 40 | 5120 | 32 / 8 | 1M | ~13 GB | 24B params; Tekken tokenizer |
| `Mistral-Small-3.1-24B-Instruct-2503` | `mistral3` (wrapper) → `mistral` (text inner) | 40 | 5120 | 32 / 8 | 1M | ~13 GB | Pixtral vision encoder on top; detects as `mistral` via the `text_config` unwrap (#69) — text-backbone export not yet validated e2e |
| `Pixtral-12B-2409` | `mistral3` | 40 | 5120 | 32 / 8 | 1M | ~7 GB | Same — detects as `mistral`; not yet validated e2e |

All accessible via `cascadia shard --model mistral-7b` /
`mistral-nemo-12b` aliases (see `tools/model_aliases.py`).

## What "Mistral 3.x text-only" means

`Mistral-Small-3.1` and `Pixtral` ship with a vision encoder. The
HuggingFace config has `model_type: "mistral3"` and looks like:

```json
{
  "model_type": "mistral3",
  "architectures": ["Mistral3ForConditionalGeneration"],
  "vision_config": { ... },
  "text_config": {
    "model_type": "mistral",
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    ...
  }
}
```

Cascadia's `_text_config` helper (added in #69) recurses into
`config.text_config` before architecture detection, so
`detect_architecture` sees the inner `model_type: "mistral"` and returns
`"mistral"`; `main()` likewise reads the per-stage fields off the
unwrapped text config. The text backbone is then built by the generic
stage builder. Note: a multimodal wrapper's weight keys can be nested
(e.g. under `language_model.*`), so validate a `mistral3` export against
the HF reference before trusting it — only the plain `mistral` models
above are exercised today.

(The cascadia `Rust` runtime's `Rotary::from_config_json` does the
same unwrap independently — `text_config` keys take precedence over
outer wrapper keys.)

If you want the multimodal capability, that's tracked separately —
the vision tower is not exported.

## Sliding window — what we drop

Mistral 7B v0.1 used sliding-window attention with a hard window
of 4096 tokens (later versions dropped it). The generic
`cached_layer_forward_sdpa` in `tools/export_shards.py` uses standard
SDPA across the full KV cache, so the sliding window is not
enforced — every token attends to every previous token. For chat
or short-context workloads this is fine and arguably better. For
long-document workloads (>8k tokens) on v0.1 specifically the
output may drift from the HF reference more than on shorter
prompts.

If you care about exact parity with HF's reference Mistral
implementation on long context, prefer v0.2 / v0.3 / NeMo, which
all dropped sliding window.

## End-to-end recipe (pipeline-parallel on iGPU AI PCs)

Same shape as the [R1-Distill recipe](./r1-distill.md). Substitute:

```bash
ssh <export-host> "cascadia shard \
  --model mistral-nemo-12b \
  --output-dir /tmp/mistral_nemo_int4 \
  --num-stages 2 --quantization int4"
```

NeMo 12B INT4 is ~7 GB total; comfortably 2-stage across two Lunar
Lake iGPUs. Mistral-Small-3.1-24B INT4 is ~13 GB; tight on a
single iGPU's 8 GB envelope so always 2-stage or 3-stage. Both
ride the `mistral` arch path through the generic stage builder.

## Known gotchas

- `Mistral-Small-3.1` uses the **Tekken** tokenizer (131k vocab,
  Apache-2.0). The exporter copies `tokenizer.model.v3` (or
  `tokenizer.json` depending on revision) into the shard tree;
  make sure the shard worker reads the right one.
- `Mistral-Nemo` uses Tekken too (different revision). Both
  tokenizers are loaded correctly by the `tokenizers` crate via
  the shard's `tokenizer/` directory.
- `Pixtral-12B` repository has the text-only `consolidated.safetensors`
  + a separate `mistral_inference`-compatible layout. Use the HF
  Transformers-compatible `Pixtral-12B-2409` mirror
  (`mistral-community/pixtral-12b`) for the text-tower export to
  match the layer names cascadia expects.
- Chat templates: NeMo and the Small models use the same
  `[INST]...[/INST]` template as 7B v0.1. They're read from the
  HF tokenizer's `tokenizer_config.json::chat_template` and applied
  by cascadia's HTTP API handler automatically; no work needed.
