# Gemma 4 support

Google released Gemma 4 on **2026-04-02**. Four variants ship under
the same `model_type: "gemma4"` outer wrapper (the text backbone is
`model_type: "gemma4_text"` nested under `text_config`):

| Variant | Hidden layers | Hidden size | head_dim (l/g) | num_kv_heads (l/g) | PLI | KV-share | Multimodal |
|---------|--------------:|------------:|----------------|--------------------|-----|----------|------------|
| **E2B-it** | 35 | 1536 | 256 / 512 | 1 / —   | 256 | 20 / 35 | yes |
| **E4B-it** | 42 | 2560 | 256 / 512 | 2 / —   | 256 | 18 / 42 | yes |
| **26B-A4B-it** (MoE) | 48 | — | — | — | — | — | yes |
| **31B-it** | 60 | 5376 | 256 / 512 | 16 / 4  |   0 |  0 / 60 | yes |

## What ships

`tools/export_gemma4.py` — a Gemma-4-specific exporter. `cascadia
shard` auto-dispatches to it on detection of `model_type ∈
{"gemma4", "gemma4_text"}`.

Tested on a Linux/Xeon export host (OpenVINO 2026.1):

```
$ python tools/export_shards.py \
    --model google/gemma-4-E2B-it \
    --output-dir /tmp/test_gemma4_e2b \
    --num-stages 2 --quantization fp16
[…]
Detected Gemma 4 — dispatching to tools/export_gemma4.py
Loading config...
  35 layers, hidden=1536, heads=8, kv_heads=1,
  head_dim=256, global_head_dim=512, pli_dim=256,
  num_kv_shared=20, softcap=30.0

Stage plan (2 stages):
  Stage 0: layers [0, 18) + embed
  Stage 1: layers [18, 35) + norm/head

Loaded + patched 942 scalar buffers in 132s
STAGE 0: layers [0, 18) | embed=True | head=False
  KV sharing: 15 own + 3 shared, 2 cross-stage sources out, 0 external sources in
  Rotary: local hd=256 theta=10000.0 prf=1.0 | global hd=512 theta=1000000.0 prf=0.25
  Wrapper built / Trace OK (1s) / Converted (27s); 32 inputs, 35 outputs
  apply_make_stateful_transformation (15 KV pairs)...
  Self-verify on CPU... Prefill OK / Decode OK
  Saved: /tmp/test_gemma4_e2b/stage_0 (6713 MB)
STAGE 1: layers [18, 35) | embed=False | head=True
  KV sharing: 0 own + 17 shared, 0 cross-stage sources out, 2 external sources in
  Head stage will apply final_logit_softcapping=30.0
  Self-verify on CPU... Prefill OK: shape=(1, 3, 262144) / Decode OK: shape=(1, 1, 262144)
  Saved: /tmp/test_gemma4_e2b/stage_1 (2882 MB)
GEMMA 4 EXPORT COMPLETE  Total: 9596 MB
```

Each stage compiles to an OpenVINO IR that runs through
`openvino.Core().compile_model()` and returns logits of correct shape
on prefill + decode. The pre-existing `--quantization {int4,int4_asym,
int8}` knobs are wired through (nncf compress_weights), though INT4
on the per-layer-scalar buffers sometimes fails — start with `fp16`
and try INT4 once you've confirmed parity.

## What the exporter handles (vs the generic export_shards.py)

* **Per-layer-type asymmetric attention** — `head_dim=256` (sliding)
  vs `global_head_dim=512` (full). `head_dims` array in the stage
  config records per-layer dims so the runtime can allocate KV cache
  of the right shape.
* **Per-layer-type RoPE** — two `GemmaTracedRotaryEmbedding`
  instances per stage (one local 10k-θ, one global 1M-θ with the
  Gemma-4 invention `partial_rotary_factor=0.25` for the
  "proportional" rope_type). Per-layer-type cos/sin precomputed once
  per forward and dispatched off `layer_types[i]`.
* **KV sharing across layers** (E2B/E4B) — `num_kv_shared_layers`
  trailing layers reuse their most-recent matching-type layer's KV.
  Cross-stage shared sources stay as **regular** I/O ports
  (`cross_kv.N.{key,value}` out of stage A, `external_kv.M.{key,
  value}` into stage B) while own KV becomes stateful.
* **Per-layer embeddings (PLI)** — E2B/E4B have
  `embed_tokens_per_layer` + a projection / norm / input-scale stack
  that modulates every layer's MLP output. The exporter concatenates
  the downstream-stage PLI onto `hidden_states` so the wire format
  stays a single tensor; the receiving stage slices it back out.
* **`final_logit_softcapping=30.0`** — applied in the head stage:
  `logits = softcap * tanh(logits / softcap)`.
* **Q/K/V RMSNorm per head before rotary** — handled by the manual
  cached_gemma4_layer_forward (HF's Gemma 4 path has a FP16/FP32
  autocast issue that breaks tracing; the cached forward keeps
  dtypes consistent).
* **Patched ov_utils.torch_tensor_to_ov_const** — reshapes 0-dim
  scalar tensors to (1,) so the per-layer scalar buffers don't crash
  OV's PyTorch frontend.

## What it does NOT handle (intentionally)

* **26B-A4B MoE variant** — `enable_moe_block=True` in its text
  config; runs into the same MoE blocker as the rest of the family.
  Tracked in [`moe.md`](./moe.md).

The exported IRs use a custom I/O contract (cross/external KV,
downstream-PLI side channel, per-layer head_dim, optional final
softcap), served by the dedicated `gemma4` engine in
`crates/cascadia-engine-openvino/` (`--engine gemma4`; Phase B below,
now shipped).

## Port plan — status

**Phase A — exporter (shipped, #48):**

- [x] Detect Gemma 4 (`model_type ∈ {gemma4, gemma4_text}`).
- [x] Per-layer-type rotary in TracedRotaryEmbedding (handles
  `partial_rotary_factor` with zero-padded `inv_freq`).
- [x] KV sharing resolution: cross-stage source identification,
  external-shared-source tracking, stateful-vs-regular I/O split.
- [x] PLI side-channel concatenated onto wire (for stages without
  embed).
- [x] `final_logit_softcapping` in the head stage.
- [x] Manual `cached_gemma4_layer_forward` (consistent dtype,
  q/k/v_norm + GQA + KV concat + attention + post-attention norm +
  MLP + optional PLI + optional layer_scalar).
- [x] Self-verify each stage on CPU after export.
- [x] Skip 26B-A4B MoE variant cleanly.
- [x] Tested end-to-end on a Xeon export host with Gemma-4-E2B-it.

**Phase B — Rust runtime (`crates/cascadia-engine-openvino`) — shipped
(`--engine gemma4`, `gemma4.rs`):**

- [x] New `Gemma4Stage` engine that:
  - Reads the extended `stage_config.json` schema (per-layer
    `head_dims`, `layer_types`, `is_shared`, `own_kv_head_dims`,
    `cross_stage_sources_local`, `external_shared_sources`,
    `pli_dim`, `downstream_pli_count`, `final_logit_softcapping`).
  - Allocates per-layer KV cache with per-layer head_dim.
  - Skips KV allocation for `is_shared[i] = true` layers.
  - Sends/receives `external_kv.N.key/value` pairs over the wire
    alongside `hidden_states`.
  - On stage 0 (embed), emits `hidden_states + downstream_PLI`
    concatenated on the last dim. On middle stages, slices off the
    head-stage PLI before sending.
  - On the head stage, samples directly from the post-softcap
    `logits` output.
- [x] Stage-config schema versioning (`export_version:
  "gemma4_cached_v1"`).

**Phase C — End-to-end testing:**

- [ ] 2-stage pipeline-parallel across two nodes (rank 0 + rank 1)
  via cascadia worker. Verify Gemma-4-E2B-it produces sensible
  tokens for a chat prompt.
- [ ] 3-stage shard for Gemma-4-31B-it across three nodes. Verify
  single-box runtime memory stays under the per-box envelope.

## Why Gemma 4 needs its own exporter

The generic `tools/export_shards.py` makes three assumptions that
all break on Gemma 4:

1. **One `head_dim` per stage.** Used to size every per-layer KV
   cache tensor uniformly. Gemma 4 has two — see `head_dims` list
   in stage config.
2. **One rotary per stage.** Used to precompute cos/sin once.
   Gemma 4 needs two (sliding vs full), and the full-attention one
   uses the `proportional` rope_type (partial-rotary on global
   layers only).
3. **All KV is stateful, all layers compute own KV.** The KV-share
   pattern in E2B/E4B trades memory for an explicit cross-stage
   handoff, and the head stage may need to receive KV from an
   earlier stage's source layer.

A separate code path is cleaner than threading these as flags
through the generic stage builder, and we expect the same pattern
when Llama 4 / Qwen3-MoE / gpt-oss eventually land — each gets its
own `export_<family>.py`.
