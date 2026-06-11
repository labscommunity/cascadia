# Qwen3.6-35B-A3B support — design spec (draft)

Status: **draft for review** — design decisions marked `DECISION`, open items
marked `TODO(review)`.

Goal: run Qwen3.6-35B-A3B (hybrid Gated-DeltaNet + sparse-MoE) sharded
across cascadia mesh nodes. Whole-model serving on one box already works via
OVMS 2026.2 in the enterprise OVMS backend; this spec covers the **sharded
engine path only**.

## 1. Model facts (verified against `Qwen/Qwen3.6-35B-A3B` config.json)

- `model_type: qwen3_5_moe` — Qwen3.6 ships on the Qwen3.5 architecture
  (`Qwen3_5MoeForConditionalGeneration`, `text_config.model_type:
  qwen3_5_moe_text`). There is no `qwen3_6` model_type.
- 40 layers, `hidden_size 2048`, vocab 248320, 262K native context.
  `layer_types` = 3× `linear_attention` : 1× `full_attention`
  (`full_attention_interval: 4`) → **30 DeltaNet layers + 10 full-attention
  layers**, every layer followed by MoE.
- MoE: **256 experts, top-8 routed + 1 shared expert** (sigmoid
  `shared_expert_gate`), `moe_intermediate_size 512`,
  `shared_expert_intermediate_size 512`. 35B total / ~3B active.
  `norm_topk_prob` absent from config — falls back to the transformers
  `Qwen3_5MoeTextConfig` default. TODO(review): pin the default from the
  transformers version we export with.
- Linear attention (Gated DeltaNet): 32 V-heads / 16 K-heads, head_dim 128,
  conv kernel 4. No RoPE in linear layers.
- Full attention: 16 Q-heads / 2 KV-heads, head_dim 256,
  `attn_output_gate: true`, `partial_rotary_factor: 0.25`.
- **Vision encoder present**; no text-only repo published. v1 is text-only —
  export drops the vision tower (the standard `--language-model-only`
  treatment).
- MTP fields present (`mtp_num_hidden_layers`) — ignored in v1 (relevant
  later for speculative decoding; see genai PR #3938 "DFlash").

## 2. Why the current stack rejects it

- `tools/export_shards.py` `is_moe_config()` rejects on
  `num_experts`/`num_experts_per_tok` (and would reject `qwen3_5_moe` by
  structure even though the model_type isn't in the name list).
- `cached_layer_forward_sdpa` assumes one SDPA + KV per layer; 30/40 layers
  here carry a recurrent matrix state + conv tail instead of KV.
- `cascadia-engine-sparse-moe` is Kimi-K2.6-shaped: softmax top-8 over 384
  experts, **no shared expert**, MLA-free SDPA shell, KV-only caches.

## 3. Prior art (Intel, OV 2026.x) — what we reuse

- **OV core has a native fused `GatedDeltaNet` internal op** with CPU + GPU
  kernels (`fuse_gated_delta_net` transformation). optimum-intel exports the
  HF module as a `RecurrentAttentionCell` → `ov::Loop` → fused op. Inputs:
  q/k `[B,T,H,dk]`, v `[B,T,Hv,dv]`, recurrent_state `[B,Hv,dk,dv]`, gate
  (log-space), beta. We do **not** hand-roll a DeltaNet kernel.
- **State representation** (genai PR #3359 "linear state in SDPA pipeline"):
  stateful OV models with `ReadValue`/`Assign` variables; shape-based
  classification (4D dynamic = KV, fixed-size = linear). Semantics we must
  copy: **linear state cannot be trimmed — any rollback is a full reset**
  (recompute from token 0).
- **State sizes per linear layer per sequence**: recurrent S
  `[32,128,128]` ≈ 1 MiB fp16 + conv tail `[8192,4]` ≈ 64 KiB. All 30
  linear layers ≈ 32 MiB fp16 per live sequence.
- **Reference IR**: `OpenVINO/Qwen3.6-35B-A3B-int4-ov` on HF — inspect
  before writing the exporter.
- Caveat to avoid copying: OVMS 2026.2 prefix caching for linear-attention
  models checkpoints full state snapshots per interval and blows up memory.
  Our `kv_prefix_cache` must simply **not apply** to linear layers in v1.

## 4. Design

### 4.1 Wire protocol — no change (the load-bearing simplification)

DeltaNet state is **layer-local**. With layer-range sharding, steady-state
decode ships exactly what it ships today: hidden activations
`[B, T, 2048]` between stages. Each stage owns the S/conv state for its own
linear layers and the KV for its own full-attention layers, exactly like
gemma4's stateful-KV stages.

DECISION: v1 makes **no wire-format or ShardSpec change**. Frozen wire pins
in enterprise stay frozen. Consequences accepted for v1:

- No mid-sequence shard migration/failover (state is not reconstructible
  from the wire; shipping it = ~32 MiB + KV). Same limitation gemma4
  stateful stages already have.
- Task boundary = full state reset per stage (existing gemma4 reset
  pattern; matches PR #3359 trim-is-reset semantics).

### 4.2 Engine — extend `gemma4`-style staged OV engine, not sparse-moe (v1)

Two candidate homes were considered:

- **(a) Staged stateful OV engine** (gemma4 pattern): per-stage OV IRs
  containing DeltaNet (fused op) + full-attn + **dense MoE** (all 256
  experts via stacked bmm, as optimum-intel exports it), state as OV
  `ReadValue`/`Assign` inside each stage.
- **(b) sparse-moe engine extension**: OV shell for attention/DeltaNet +
  routed top-8 expert dispatch through the INT4 GEMM expert cache (true
  3B-active compute, Kimi pattern).

DECISION (v1): **(a)**. Rationale: the entire DeltaNet/state problem is
solved by OV's own kernels and the stateful-stage pattern we already run for
gemma4; the exporter can closely follow the official optimum-intel graph.
The cost is dense-MoE compute (~all-expert FLOPs instead of 3B active).
With `moe_intermediate_size 512` the per-layer expert GEMM is
256×(3×2048×512) ≈ 0.8 GFLOP/token — wasteful but well inside what the
gemma4 stages already sustain per token on AI-PC GPUs, and int4 weights keep
the working set ≈ 18–20 GB total across the mesh.

v2 (separate effort, only if v1 throughput disappoints): port routing to
the sparse-moe expert cache — needs shared-expert support, sigmoid shared
gate, 512-wide GEMM shapes, and a DeltaNet-capable shell; tracked as
follow-up, not specced here.

TODO(review): confirm (a)'s dense-MoE decode latency on target hardware
during M2 below before building stage export for all 40 layers.

### 4.3 Exporter — `tools/export_qwen36_moe.py`

Follows `export_gemma4.py` shape: detect `model_type ∈ {qwen3_5_moe,
qwen3_5_moe_text}` in `export_shards.py` and dispatch (alongside the
existing MoE rejection, which keeps rejecting everything else). Per stage:

- Slice `layer_start..layer_end` of the 40-layer stack; embed on first
  stage, head on last (existing `has_embed`/`has_head`).
- Reuse optimum-intel's machinery (RecurrentAttentionCell ModuleExtension +
  conversion extension, `patched_qwen3_next_sparse_moe_block` dense-MoE
  patch, `patch_stateful_hybrid_ssm`) rather than reimplementing — the
  exporter wraps a layer-range submodule and runs the same patches.
  TODO(review): optimum-intel's qwen3_5 path requires transformers 5.2.x;
  the published config says 4.57.1 — pin the working combination in the
  exporter docstring.
- `stage_config.json` extensions (gemma4 schema precedent):
  `layer_types` for the local range, `linear_state_dims`, partial-rotary
  params for full-attn layers, MoE arity (for sanity checks only — graph is
  self-contained).
- Quantization: int4 channel-wise weights (match the official
  `-int4-ov` export recipe). Mark `export_version: ov-qwen36-v1`.

### 4.4 Runtime — `crates/cascadia-engine-openvino/src/qwen36.rs`

New `EngineKind::Qwen36Moe` + builder, structured like `gemma4.rs`:

- Stateful stages; reset all variables at task start (existing pattern).
- No cross-stage KV pairing needed (unlike gemma4): Qwen3.6 has no
  cross-layer KV sharing — stage I/O is hidden states + position only.
  Positions matter only to full-attn layers (partial RoPE); cos/sin
  computed stage-locally from the position counter, as the generic path
  already does.
- `kv_prefix_cache`: disabled for this engine in v1 (see §3 caveat).

### 4.5 Enterprise integration (separate branch `pawan/qwen36-moe` there)

- New `EmbeddedEngine` variant dispatch once the engine crate lands; rev
  bump of the git pin. No wire-pin change (per 4.1).
- OVMS whole-model path documented as the already-working alternative
  (`source_model = "OpenVINO/Qwen3.6-35B-A3B-int4-ov"`, `target_device =
  "GPU"`) — worth a `mesh-qwen36.toml` example independent of this spec.

## 5. Milestones

- **M1 — feasibility probe**: load `OpenVINO/Qwen3.6-35B-A3B-int4-ov`
  whole-model through plain OV runtime on the miner; inspect IR (fused
  GatedDeltaNet present? state variable names/shapes match §3?). Exit: facts
  confirmed or spec amended.
- **M2 — 2-layer slice**: export a 4-layer stage (one full DeltaNet:attn
  block) with the exporter skeleton; verify logits vs HF reference;
  measure dense-MoE per-token cost. Exit: go/no-go on decision 4.2.
- **M3 — engine**: `qwen36.rs` runner, 2-stage split on one box, greedy
  decode parity with M1.
- **M4 — mesh**: 2-node split, soak, perf profile, docs. Enterprise
  integration starts here.

## 6. Out of scope (v1)

Vision tower, MTP/speculative decoding, prefix caching on linear layers,
mid-sequence migration/failover, continuous batching, sparse routed
dispatch (v2 candidate), Qwen3.5/Qwen3-Next variants (should fall out of
the same path — verify in M2, don't promise).
