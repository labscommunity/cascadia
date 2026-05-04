# Sharding models for tahoma

`tahoma shard` slices a HuggingFace causal-LM into per-stage OpenVINO IRs
that the worker engines (`ov-runtime`, `ov-dist-spec`) can load. Each
stage holds a contiguous slab of decoder layers + (on the first stage)
the embedding + (on the last stage) the final norm + lm_head.

## Quick start

```bash
# One-time pip install (export-time only, not needed at runtime):
pip install torch transformers openvino safetensors huggingface_hub nncf

# Two-stage shard from HF:
tahoma shard --model unsloth/Meta-Llama-3.1-8B-Instruct \
             --output-dir ~/tahoma/llama-8b-2stage \
             --num-stages 2 \
             --quantization int4
```

Output:

```
~/tahoma/llama-8b-2stage/
  pipeline_config.json    # global metadata: model_id, layer count, vocab, etc.
  tokenizer/              # tokenizer.json + config.json + special tokens
  stage_0/
    openvino_model.xml    # OpenVINO IR (FP16 graph + INT4 weights)
    openvino_model.bin    # quantized weights
    stage_config.json     # per-stage metadata: layer_start, layer_end, has_embed, has_head
  stage_1/
    openvino_model.xml
    openvino_model.bin
    stage_config.json
```

This directory is portable. Copy it to every worker node (or re-run
`tahoma shard` separately on each, depending on which is faster on your
network).

## Flags

| Flag | Meaning |
|------|---------|
| `--model` | HF repo id (`unsloth/Meta-Llama-3.1-8B-Instruct`) OR local path to a directory with `config.json` + `*.safetensors`. HF repos auto-download into `~/.cache/tahoma/models/`. |
| `-o`, `--output-dir` | Where to write the shard tree. |
| `--num-stages N` | Pipeline stages to split into. 2 is the common case for 2-machine setups; use 3+ for larger clusters. |
| `--quantization` | `int4` (default — typical), `int4_asym`, `int8`, or `fp16`. INT4 needs nncf installed. |
| `--layer-split a,b,...` | Override the uniform split. With `--num-stages 3 --layer-split 16,24` on a 32-layer model: stage 0 = layers 0..15, stage 1 = 16..23, stage 2 = 24..31. Useful for asymmetric hardware (e.g. give the bigger node more layers). |
| `--stage N` | Re-export only stage N (debugging). |
| `--python /path/to/python` | Override Python interpreter. Defaults to `python3` then `python`. |
| `--skip-check` | Skip the dependency probe — pass if you know your env is already set up. |

## Supported architectures

Tahoma's exporter uses HuggingFace's standard decoder-layer classes,
so anything whose layer exposes the conventional names
(`self_attn.{q_proj,k_proj,v_proj,o_proj}`, `mlp`, `input_layernorm`,
`post_attention_layernorm`) and uses RoPE rotary embeddings works:

| Family | Status | Notes |
|--------|--------|-------|
| **Llama** (1, 2, 3, 3.1, 3.2) | ✅ tested | Reference path. Llama 3.2 1B/3B have tied embeddings — handled. |
| **Mistral** (7B v0.1+) | ✅ tested | Sliding-window attention is disabled in the export (uses standard SDPA). |
| **Qwen2 / Qwen2.5** | ✅ tested | Uses Qwen2-specific decoder layer with bias on q/k/v projections. |
| **Phi-3** (Mini, Medium) | ⚠️ best-effort | Standard Phi-3 layouts work; LongRope variants may need `rope_theta` override. |
| **Gemma 2** | ⚠️ best-effort | Gemma2DecoderLayer is used; logit softcapping is NOT applied (the export skips it for performance) — quality may regress by ~1% on benchmarks. |
| **Mixtral / MoE** | ❌ not supported | The exporter assumes one MLP per layer; MoE routing requires a different export path. |
| **Multimodal (Llava, etc.)** | ❌ not supported | Vision encoder is not exported; text-only path may work but isn't tested. |

For unknown architectures the script falls back to `LlamaDecoderLayer`
and warns on stderr. If the resulting shard's outputs match the HF
reference (you can spot-check with `tahoma worker --engine ov-genai`
single-stage on the same model), it's good. If they diverge, file an
issue with the model's `model_type` and `architectures` from `config.json`.

## Picking `--num-stages`

The right answer is determined by **memory** more than throughput. A
shard's GPU memory footprint at runtime is roughly:

```
shard_size ≈ (layers_in_shard / total_layers) × model_size_int4
           + KV_cache_per_token × max_context_len
           + workspace (~500 MB OV plugin)
```

For Llama 3.1 8B INT4 (~4 GB) on a 12 GB Battlemage Arc B390 with
4096-token max context: a 16/16 2-stage split fits comfortably.

For Llama 3.1 70B INT4 (~35 GB) on the same B390: you'd need 4–5 stages
to keep each shard under 8 GB of weights + leave room for KV cache.
Practical layout: `--num-stages 4 --layer-split 20,40,60` (uneven, with
the most layers on the GPU/CPU node that has the most free RAM).

For Mixtral 8x7B INT4 (~28 GB): same story — split 3-4 ways.

There's a per-stage overhead (network round-trip + OV plugin init), so
**don't** over-shard a model that fits on fewer nodes. 2 stages costs
~1ms / token of network latency on TB4, ~5ms on 1 GbE.

## Picking `--quantization`

| Mode | Weight bits | Quality loss | Speed | Recommended for |
|------|------------:|--------------|-------|-----------------|
| `int4` | 4 (sym, group=128) | ~1-2% on standard benchmarks | fastest | **Default.** Almost always the right choice. |
| `int4_asym` | 4 (asym, group=128) | similar to int4 | similar | When INT4 sym shows numerical issues on a specific model. |
| `int8` | 8 (asym, per-channel) | ~0.5% | slower than int4 | When int4 quality is unacceptable. |
| `fp16` | 16 | none | slowest, biggest | Debugging or when nncf isn't installed. |

INT4 with group_size=128 is the sweet spot on Intel Arc / iGPU because
the weight unpacking happens in the same SIMD pass as the matmul.

## Re-exporting one stage

If you change a config knob (rope_theta, quantization mode) and want to
avoid the full multi-minute re-export:

```bash
tahoma shard --model <same args> --stage 1
```

This re-runs only stage 1 against the same source weights. The
`pipeline_config.json` is overwritten each invocation but should be
identical.

## Troubleshooting

**"NNCF not installed" warning on INT4 export**: the script falls back to
FP16 weights silently. Install nncf (`pip install nncf>=2.13`) and re-run.

**"tied embeddings" log line on Llama 3.2 1B/3B**: not an error. Llama
3.2 small models share `lm_head.weight` with `embed_tokens.weight`. The
exporter detects `config.tie_word_embeddings == True` and reuses the
same tensor.

**Export hangs after "Quantization OK"**: NNCF on tied-weight models
sometimes loops in an internal scan. Use `--quantization fp16` and let
the runtime do dynamic int4 unpacking instead, or use `--quantization
int8` which doesn't hit the same path.

**"unknown model_type, falling back to Llama" warning**: the model
isn't in the explicit support list. The export will produce a working
IR for any model whose layer interface matches Llama's, but you should
verify the outputs (run `tahoma worker --engine ov-genai` against the
same source model and compare the first 10 generated tokens).

**OOM during INT4 quantization**: NNCF holds the full FP16 model in
RAM during compression. For 70B+ models, run on a machine with ≥ 64 GB
RAM, or use `--quantization fp16` and quantize per-shard later via
`nncf.compress_weights` directly.
