# MoE family support

The generic `cascadia shard` exporter assumes one MLP per decoder
layer. Mixture-of-Experts (MoE) families replace that with a router +
N expert MLPs and a per-token top-k gating. `tools/export_shards.py`'s
`cached_layer_forward_sdpa` calls `layer.mlp(hidden_states)` directly;
on an MoE layer that field is usually named `block_sparse_moe` or
`experts`, so the call either raises `AttributeError` or silently runs
the wrong path.

`is_moe_config` (#60) rejects MoE configs config-first — before the
download — with a clear error pointing to either `cascadia-engine-sparse-moe`
or this doc. It matches both the known MoE `model_type`s (mixtral, dbrx,
deepseek_v2/v3, qwen2_moe, qwen3_moe, phimoe, jamba, granitemoe, olmoe,
grok, llama4, …) and the structural signals (`num_local_experts` /
`n_routed_experts` / `num_experts_per_tok` / nested `ffn_config`).

## Why a generic MoE exporter is not trivial

Each MoE family routes differently:

| Family | Experts / layer | Top-k | Shared expert | Routing function | Notes |
|--------|----------------:|------:|---------------|-----------------|-------|
| **Mixtral 8x7B / 8x22B** | 8 | 2 | none | softmax | Hugging Face has `MixtralForCausalLM` |
| **Qwen3-MoE** (30B-A3B, 235B-A22B) | 128 | 8 | none | softmax | dense/MoE per-layer is configurable via `mlp_only_layers` |
| **Llama 4 Scout** | 16 | 1 | yes | softmax | `interleave_moe_layer_step = 1` (every layer is MoE) + iRoPE NoPE-every-4 |
| **Llama 4 Maverick** | 128 | 1 | yes | softmax | |
| **gpt-oss-20b** | 32 | 4 | none | **sigmoid** on top-k | MXFP4 native quant; alternating sliding/full per-layer; YARN RoPE |
| **gpt-oss-120b** | 128 | 4 | none | sigmoid | same |
| **DeepSeek-V3 / R1** | 256 routed + 1 shared | 8 | 1 | sigmoid + bias balancing | MLA attention (separate concern) |
| **GraniteMoE** | varies | varies | none | dropless routing | |
| **HunYuan-Large / A13B** | many | small | yes | shared + specialised | |
| **Gemma 4 26B-A4B** | (unknown; see config) | — | — | — | wrapper around `Gemma4ForConditionalGeneration` with `enable_moe_block: True` |

Each variant needs its own export branch — there is no "one MoE forward"
that fits them all. The choices that differ:

1. Whether to **stack expert weights** into a single `[num_experts,
   hidden, intermediate]` tensor (memory-efficient, einsum-traceable)
   or keep them as N separate Linear modules (faithful to HF reference
   but explodes graph size).
2. **Routing op**: `softmax(logits).topk(k)` vs `sigmoid(logits).topk(k)`
   vs DeepSeek's bias-balanced variant.
3. **Shared expert**: present-or-not, weighted-how.
4. **`norm_topk_prob`**: whether the top-k weights are re-normalised
   to sum to 1 after selection.

## What cascadia has today

`crates/cascadia-engine-sparse-moe/` implements a specialised MoE
runtime for **Kimi K2.6 only** (60-layer Mixtral-style with 384 experts
per layer, top-8 routing). It uses:

- per-expert OV IRs (each expert is exported as its own IR)
- a router pass that picks top-k experts per token
- the hand-rolled AVX-512 INT4 GEMM kernel for the expert matmul
- a bounded LRU cache of compiled experts (most aren't in memory at
  once on a 133-GB-RAM box)

This is NOT wired into `cascadia shard`. The expert-bin export
pipeline is not part of this repo; the engine is invoked via
`--engine sparse-moe` directly against the pre-built artefacts.

Known-working export paths exist for Mixtral, Kimi K2.6 sparse-MoE
(plus a batched-expert variant), and the Gemma 4 26B-A4B MoE, but
none ship in this repo yet. If/when cascadia grows generic MoE
support, each family would land as its own `export_<family>.py`
beside the generic exporter (the Gemma 4 dense exporter sets the
pattern).

## Until then

`cascadia shard --model mistralai/Mixtral-8x7B-Instruct-v0.1 ...` will
fail with:

```
error: model mistralai/Mixtral-8x7B-Instruct-v0.1 reports model_type
       "mixtral" with num_local_experts=8 and num_experts_per_tok=2.
       cascadia's generic sharding path does not support MoE routing.

       Options:
       - For Kimi K2.6-style sparse-MoE (one model only today), use
         the standalone `cascadia worker --engine sparse-moe` against
         pre-built artefacts.
       - To add generic MoE support, see
         docs/architectures/moe.md#what-cascadia-has-today.
```

Same will apply to: Qwen3-MoE, Llama 4, gpt-oss, GraniteMoE, Hunyuan,
Gemma 4 26B-A4B, DeepSeek-V3.
