# Qwen3.6-35B-A3B support — design spec (draft, rev 2)

Status: **draft for review** — design decisions marked `DECISION`, open items
marked `TODO(review)`. Rev 2 incorporates adversarial + feasibility review
(2026-06-11): dense-MoE cost re-gated to M1, exporter pipeline corrected,
§4.1 preconditions pinned, enterprise gossip-bump priced. Rev 3 aligns with
tracking issue #77: this spec is **Part B Path 2 (pipeline-parallel,
deferred)**; the shipped-first path is single-stage `ov-genai`
(docs/architectures/qwen3.6.md), and #77's Path-1 validation doubles as
this spec's M1 throughput measurement.

Goal: run Qwen3.6-35B-A3B (hybrid Gated-DeltaNet linear attention +
sparse-MoE) sharded across cascadia mesh nodes. Whole-model serving on one
box already works via OVMS 2026.2 in the enterprise OVMS backend; this spec
covers the **sharded engine path only**.

## 1. Model facts (verified against `Qwen/Qwen3.6-35B-A3B` config.json)

- `model_type: qwen3_5_moe` — Qwen3.6 ships on the Qwen3.5 architecture
  (`Qwen3_5MoeForConditionalGeneration`, `text_config.model_type:
  qwen3_5_moe_text`). There is no `qwen3_6` model_type. The expert/MoE
  fields live under the nested `text_config`.
- 40 layers, `hidden_size 2048`, vocab 248320. `layer_types` = 3×
  `linear_attention` : 1× `full_attention` (`full_attention_interval: 4`)
  → **30 DeltaNet layers + 10 full-attention layers**, every layer followed
  by MoE.
- MoE: **256 experts, top-8 routed + 1 shared expert** (sigmoid
  `shared_expert_gate`), `moe_intermediate_size 512`,
  `shared_expert_intermediate_size 512`. 35B total / ~3B active.
  `norm_topk_prob` absent from config — resolved in M1 by reading the
  official `-int4-ov` IR, not by trusting a library default (the default
  can differ across the transformers 4.57/5.2 divide; a wrong value is
  silently miscorrect, not a crash).
- Linear attention (Gated DeltaNet): 32 V-heads / 16 K-heads, head_dim 128,
  conv kernel 4. No RoPE in linear layers.
- Full attention: 16 Q-heads / 2 KV-heads, head_dim 256,
  `attn_output_gate: true`, `partial_rotary_factor: 0.25` with **mRoPE**
  (`rope_parameters.mrope_section [11,11,10]`, `mrope_interleaved: true`,
  `rope_theta 1e7`) — multimodal RoPE, not plain partial rotary. M2 parity
  must verify the text-only rotary contract end-to-end (#77 flags this as
  exactly the kind of thing that matches intermediates but diverges
  applied).
- Native context 262K, but the staged-engine pattern materializes
  full-sequence logits at the head stage — at vocab 248320 a 10K-token
  prefill is a ~10 GB f32 logits tensor. **v1 practical context is
  prompt-length-bound**; the exporter must slice last-token logits in the
  head stage (M2 exit criterion), and even then long-context soak is an M4
  measurement, not a §1 promise.
- **Vision encoder present**; no text-only repo published. v1 is text-only —
  export drops the vision tower (the standard `--language-model-only`
  treatment).
- MTP head present (`mtp_num_hidden_layers: 1`) — ignored in v1 (relevant
  later for speculative decoding; see genai PR #3938 "DFlash"). Note v1
  fences off ALL spec-decode regardless (§4.1 invariant 2).

## 2. Why the current stack rejects it

- `tools/export_shards.py`: the pre-download `is_moe_config()` gate runs on
  the **outer** config; Qwen3.6's expert fields are nested under
  `text_config`, so the structural check likely does NOT fire there — the
  model is instead caught later by `check_export_quirks` (which unwraps
  `_text_config`). Either way it aborts today. The new dispatch (§4.3)
  hooks in config-first **before** `is_moe_config`, mirroring the gemma4
  dispatch ordering (export_shards.py:1737-1765), so the rejection-gate gap
  is moot for us.
- `cached_layer_forward_sdpa` assumes one SDPA + KV per layer; 30/40 layers
  here carry a recurrent matrix state + conv tail instead of KV.
- `cascadia-engine-sparse-moe` is Kimi-K2.6-shaped: softmax top-8 over 384
  experts, **no shared expert**, KV-only caches.

## 3. Prior art (Intel, OV 2026.x) — what we reuse

- **OV core has a native fused `GatedDeltaNet` internal op** with CPU + GPU
  kernels (`fuse_gated_delta_net` transformation). optimum-intel exports the
  HF module as a `RecurrentAttentionCell` (ModuleExtension) → `ov::Loop` →
  fused op. Inputs: q/k `[B,T,H,dk]`, v `[B,T,Hv,dv]`, recurrent_state
  `[B,Hv,dk,dv]`, gate (log-space), beta. We do **not** hand-roll a
  DeltaNet kernel — but see §4.3: reaching the fused op from a layer-range
  export is the central exporter risk, not a free reuse.
- **State representation** (genai PR #3359 "linear state in SDPA pipeline"):
  stateful OV models with `ReadValue`/`Assign` variables; shape-based
  classification (4D dynamic = KV, fixed-size = linear). Semantics we must
  copy: **linear state cannot be trimmed — any rollback is a full reset**
  (recompute from token 0).
- **State sizes per linear layer per sequence**: recurrent S
  `[32,128,128]` ≈ 1 MiB fp16 + conv tail `[8192,4]` ≈ 64 KiB. All 30
  linear layers ≈ 32 MiB fp16 per live sequence.
- **Reference IR**: `OpenVINO/Qwen3.6-35B-A3B-int4-ov` on HF — M1 inspects
  it before any exporter work.
- Caveat to avoid copying: OVMS 2026.2 prefix caching for linear-attention
  models checkpoints full state snapshots per interval and blows up memory.
  (No prefix cache exists on our staged OV path — see §4.4 — but this
  binds the v2 sparse-moe port, where `kv_prefix_cache` is real.)

## 4. Design

### 4.1 Wire protocol — no activation-wire change, under pinned preconditions

DeltaNet state is **layer-local**. With layer-range sharding, steady-state
decode ships exactly what the gemma4 engine ships today: hidden activations
plus the existing position frame between stages. Each stage owns the S/conv
state for its own linear layers and the KV for its own full-attention
layers.

DECISION: v1 makes **no activation wire-format or ShardSpec change**. This
claim is valid **only** under the following v1 invariants, each load-bearing
(violating any one silently corrupts recurrent state, which — unlike stale
KV — has no position mask to bound the damage):

1. **Batch = 1, one live task per stage at a time.** Matches today's
   gemma4 engine: one InferRequest per stage
   (cascadia-ov-genai-shim/src/lib.rs:524), `active: Option<ActiveTask>`
   serialization. OV stateful variables are per-infer-request; concurrent
   sequences on a stage would interleave into one S matrix.
2. **Greedy decode only; no speculative drafting.** Draft-rejection
   rollback = state trim = impossible for linear state (PR #3359
   semantics). The qwen36 engine must be fenced off from `dist_spec`
   speculative paths and the sparse-moe `spec_decode`/`ngram_draft`
   machinery.
3. **Position-0 reset is the only state-recovery mechanism.** First stage
   resets on task pickup; relay stages reset on a position-0 frame
   (gemma4.rs:767, 946-959). A missed reset is unrecoverable garbage for
   DeltaNet, so error paths must guarantee the next task always enters at
   position 0.

Consequences accepted for v1: no mid-sequence shard migration/failover
(state is not reconstructible from the wire; shipping it = ~32 MiB + KV);
task boundary = full state reset per stage.

Cross-node transport is already solved: the gemma4 engine pipelines stages
over TCP via `cascadia_transport::{ActivationClient, ActivationServer}`
(gemma4.rs:38-40, 1131-1161; CLI `--listen`/`--next`). M4 inherits this,
not new networking.

### 4.2 Engine — staged stateful OV engine (gemma4 pattern), dense-MoE v1 — **gated at M1, with a threshold**

Two candidate homes:

- **(a) Staged stateful OV engine** (gemma4 pattern): per-stage OV IRs
  containing DeltaNet (fused op) + full-attn + **dense MoE** (all 256
  experts via stacked bmm, as optimum-intel exports it), state as OV
  `ReadValue`/`Assign` inside each stage.
- **(b) sparse-moe engine extension**: routed top-8 expert dispatch through
  the INT4 GEMM expert cache (true 3B-active compute) — needs shared-expert
  support, sigmoid shared gate, 512-wide GEMM shapes, and a
  DeltaNet-capable shell.

DECISION (provisional, M1-gated): **(a)**, because the DeltaNet/state
problem is solved by OV's own kernels and the stateful-stage pattern already
runs cross-node. **The honest cost model is bandwidth, not FLOPs**: dense
MoE touches all ~16–18 GB of int4 expert weights **per decode token**, and
pipeline stages are serial for one sequence, so mesh sharding divides
per-node traffic but latency is additive. At 60–90 GB/s effective AI-PC
memory bandwidth the naive ceiling is **~3–5 tok/s**, vs ~25–40 tok/s for
true 3B-active. Decision (a) survives only if one of these holds, measured
in M1 *before any exporter work*:

- OV 2026.x fuses the traced dense-MoE block into an internal MoE op with
  **sparse execution** (it does exactly this for GatedDeltaNet — M1
  inspects the official IR for it), making "dense" a graph-level fiction; or
- measured whole-model decode on the existing OVMS 2026.2 deployment (zero
  new code — §0) meets the threshold anyway.

**Threshold:** v1 ships only if M1 measures ≥ T decode tok/s at 1K context
on target hardware. TODO(review): set T — proposal: T = 10 tok/s
(interactive floor); below that v1 has no user, regardless of how clean the
engineering is.

**No-go path (priced):** if M1 fails the threshold, decision flips to (b)
and this spec gets a rev-3 §4.2b before further work — sparse-moe port
scope: shared-expert + sigmoid gate in the router, 2048×512 expert GEMM
shapes in `cascadia-int4-gemm`, DeltaNet shell via OV fused op, manifest
extensions. Materially larger; if that's also rejected, the model is
declared unsupported on the sharded path and the OVMS whole-model story
stands alone.

### 4.3 Exporter — `tools/export_qwen36_moe.py` (a NEW pipeline, not an export_gemma4 variant)

Review killed the "follows export_gemma4.py shape" framing: export_gemma4
is `torch.jit.trace` → `ov.convert_model(traced)` and never used
optimum-intel. The optimum-intel machinery this model needs
(RecurrentAttentionCell ModuleExtension + conversion extension,
`patched_qwen3_next_sparse_moe_block`, `patch_stateful_hybrid_ssm`) only
functions when `ov.convert_model` receives the **live nn.Module** — a
pre-traced ScriptModule has erased the module boundaries the extensions
hook, and tracing would unroll the DeltaNet recurrence over the example
sequence length anyway.

DECISION: the exporter is a new convert-with-extensions pipeline:

- Dispatch: config-first on raw config.json `model_type ∈ {qwen3_5_moe,
  qwen3_5_moe_text}` (outer or `text_config`), **before** `is_moe_config`,
  mirroring the gemma4 dispatch ordering in export_shards.py:1737-1765.
- Per stage: wrap a layer-range submodule of the text model and run
  `ov.convert_model(live_module, extensions=...)` with the optimum-intel
  patches applied. Stateful conversion + `fuse_gated_delta_net` pattern
  matching must be verified to survive the layer-range wrapper — they
  operate on full-model export conventions (input naming, beam_idx), and a
  silent fusion miss leaves an unfused `ov::Loop` DeltaNet that is
  catastrophically slow and would corrupt the M2 measurement.
- **M2 exit criterion: the fused `GatedDeltaNet` op is present in OUR
  stage IR** (not just the official whole-model IR), verified by graph
  inspection.
- **Fallback** if layer-range wrapping defeats the patches: post-hoc graph
  surgery — split the official whole-model optimum-intel IR into stage IRs
  by cutting at layer boundaries. Different and larger work; named here so
  a M2 failure reroutes instead of stalls.
- Stage outputs: head stage slices **last-token logits** (see §1 context
  bound). `stage_config.json` extensions (gemma4 schema precedent):
  `layer_types` for the local range, `linear_state_dims`, partial-rotary
  params, MoE arity (sanity checks only). `export_version: ov-qwen36-v1`.
- Quantization: int4 channel-wise (match the official `-int4-ov` recipe).
  gemma4 precedent warns int4 nncf sometimes fails → fp16 fallback — fp16
  is NOT viable here (70 GB doesn't fit the mesh), so **"int4 compresses
  cleanly on a DeltaNet+stacked-MoE stage graph" is an M2 exit criterion**,
  not an assumption.
- Export host: the rainier/miner class box (133 GB RAM — the Kimi pipeline
  precedent); a 35B bf16 checkpoint + convert copies ≈ 70 GB+ does not fit
  smaller hosts.
- Version triple: optimum-intel's qwen3_5 path requires transformers
  5.2.x; the published config is stamped 4.57.1; export_gemma4 documents
  transformers < 5.5 quirks. **M1 exit criterion: one pinned
  transformers + optimum-intel + OV triple that loads the config, runs the
  patches, and coexists with the gemma4 exporter (or gets its own venv),
  recorded in the exporter docstring.** The M2 logit-parity reference must
  pin its transformers version independently of the export path.

### 4.4 Runtime — `crates/cascadia-engine-openvino/src/qwen36.rs`

New `EngineKind::Qwen36Moe` + builder in the tahoma CLI, structured like
`gemma4.rs`:

- Stateful stages; reset all variables at task start / position-0 frame
  (existing pattern, §4.1 invariant 3).
- No cross-stage KV pairing (gemma4's cross-KV machinery is E2B/E4B
  specific); stage relay is pure hidden states + the existing position
  frame — the cross-KV header can be dropped entirely.
- **RoPE: baked into the IR, gemma4-style.** Partial rotary (factor 0.25)
  is computed inside the stage graph from `position_ids` fed off the
  existing wire position frame (gemma4.rs:444-446, 1266-1268). No
  host-side `rotary.rs` involvement — that mechanism belongs to the
  generic v3 path this engine does not use.
- Prefix caching: **not applicable** — `kv_prefix_cache` is sparse-moe-only
  (nothing exists to disable on the staged OV path). The §3 OVMS caveat
  binds the v2 sparse-moe port, where the cache is real.

### 4.5 Enterprise integration — **deferred to v1.1; gossip bump required**

Review finding (must-fix): adding an `EmbeddedEngine` variant in enterprise
adds a `cascadia_protocol::engines::EngineKind` wire tag, which is pinned
(`wire_pins.rs::pin_engine_kind_variant_tags`) and per ADR-001 §1 requires
a **`/cascadia/gossip/N+1` protocol bump** — §4.1's "no wire change" covers
the activation wire only, not gossip. Additionally, enterprise has **no
staged-pipeline engine precedent**: the embedded backend dispatches only
OvGenai/OvRuntime/OvDistSpec/SparseMoe, and its loopback-splice/PeerLayout
bridging is written around `OvRuntimeBuilder`.

DECISION: v1 ships tahoma-only (`--engine qwen36-moe` against exported
stage artefacts, like sparse-moe today). Enterprise sharded integration is
v1.1 with its own scope: gossip protocol bump + wire-pin update, new
`EngineKind` tag, first staged-pipeline `build_*` path through the embedded
backend. The enterprise v1 story for this model stays the OVMS whole-model
backend (`source_model = "OpenVINO/Qwen3.6-35B-A3B-int4-ov"`,
`target_device = "GPU"`) — a `mesh-qwen36.toml` example is worth landing
independently of this spec.

## 5. Milestones (reordered rev 2: riskiest claims measured first, no code)

- **M1 — feasibility measurements (no exporter code).**
  (1) Benchmark whole-model Qwen3.6 decode tok/s on the existing OVMS
  2026.2 deployment at ~1K context on target hardware → §4.2 threshold T.
  (2) Pull `OpenVINO/Qwen3.6-35B-A3B-int4-ov`; inspect IR: fused
  GatedDeltaNet present? any fused/sparse MoE op? state variable names and
  shapes vs §3? `norm_topk_prob` ground truth.
  (3) Pin the transformers + optimum-intel + OV version triple (§4.3).
  Exit: go/no-go on decision 4.2 with measured numbers, or rev-3 of this
  spec.
- **M2 — stage-export probe.** Export one 4-layer stage (one full
  DeltaNet:attn block) via the §4.3 pipeline. Exit criteria: fused
  GatedDeltaNet present in our stage IR; int4 nncf compresses cleanly;
  last-token-logits head slicing works; logit parity vs pinned-version HF
  reference on the same layer range.
- **M3 — engine.** `qwen36.rs` runner, 2-stage split on one box, greedy
  decode parity with the M1 whole-model output.
- **M4 — mesh.** 2-node split over the existing gemma4-style TCP transport,
  soak, perf profile vs the M1 threshold, docs.
- **M5 / v1.1 — enterprise** (separate spec amendment): gossip `N+1` bump,
  `EngineKind` wire tag + wire-pin update, embedded-backend staged-pipeline
  build path.

## 6. Out of scope (v1)

Vision tower, MTP/speculative decoding (hard-fenced — §4.1 invariant 2),
batching / multiple live tasks per stage (§4.1 invariant 1), mid-sequence
migration/failover, prefix caching, continuous batching, sparse routed
dispatch (the priced v2/no-go path, §4.2), enterprise embedded integration
(v1.1, §4.5), Qwen3.5/Qwen3-Next variants (plausibly fall out of the same
path — verify in M2, don't promise).
