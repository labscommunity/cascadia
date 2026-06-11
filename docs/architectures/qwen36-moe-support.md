# Qwen3.6-35B-A3B support — design spec (draft, rev 5)

Status: **draft for review** — design decisions marked `DECISION`, open items
marked `TODO(review)`. Rev 2 incorporates adversarial + feasibility review
(2026-06-11): dense-MoE cost re-gated to M1, exporter pipeline corrected,
§4.1 preconditions pinned, enterprise gossip-bump priced. Rev 3 aligns with
tracking issue #77: this spec is **Part B Path 2 (pipeline-parallel,
deferred)**; the shipped-first path is single-stage `ov-genai`
(docs/architectures/qwen3.6.md), and #77's Path-1 validation doubles as
this spec's M1 throughput measurement. Rev 4 (2026-06-11, post-M1):
M1 measured the dense-MoE no-go — decision flipped to the sparse-moe
routed-dispatch port. Rev 5 (same day, post-adversarial-review of rev 4):
the port itself is **conditional, not decided** — review surfaced an ISA
precondition (no AVX-512 on Lunar Lake; the int4 GEMM crate is
AVX-512-only), a corrected bandwidth model (~1.7 GB/token, not 570 MB),
and the single-stream sharding truth (pipeline adds capacity, not
bandwidth). §4.2 rewritten accordingly; M2' now begins with an ISA/device
probe whose kill condition is explicit.

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

### 4.2 Engine — sparse-moe routed-dispatch port (rev 4: M1-decided)

**M1 verdict (2026-06-11, pawan-01, Lunar Lake GPU, int4 whole-model via
Intel's own GenAI pipeline):** greedy 1.27 tok/s, prompt-lookup 2.46
tok/s. The staged dense-MoE engine (rev 2's provisional decision (a))
inherits this graph and adds pipeline serialization — it can only be
slower. T = 10 tok/s threshold failed by 4x at best. **Decision (a) is
dead by measurement; (b) — the sparse-moe engine port — is now the
design.**

#### 4.2a Bandwidth model (rev 5: corrected) and the ISA precondition

**Per-token traffic is the ACTIVE parameter set, not just routed
experts.** Itemized (int4, short context): routed experts 566 MB +
shared experts 63 MB + attention/DeltaNet/router weights ~800 MB +
lm_head GEMV ~254 MB + int4 group-scales ~+12.5% on weights + DeltaNet
state R/W ~60 MB + KV reads (grows with context) ≈ **~1.7-1.8 GB/token**
— consistent with "A3B" (~3B active × 0.5 B). At 60-90 GB/s ideal that
is a **33-50 tok/s ceiling; 15-35 tok/s at realistic kernel efficiency**.
Above T=10, but not the "two orders" rev 4 claimed. Still ~10x better
than dense (17 GB/token): the pivot survives the corrected math —
*if the silicon can consume the bytes*.

**ISA precondition (review Finding 1 — can invalidate the port):**
every SIMD path in `cascadia-int4-gemm` is gated
`avx512f/avx512bw/avx512vl` with a pure-scalar fallback and **no AVX2
path**. Lunar Lake (Lion Cove/Skymont) has **no AVX-512**. On the very
box that motivated this design, the expert GEMV would run scalar —
likely below even the dense GPU number. The Kimi precedent ran on an
AVX-512 Xeon; that substrate assumption was silent in rev 4.

Resolution options, decided by **M2'-0 (ISA/device probe)**:

- **(i) OV-IR expert backend** — already implemented in the engine
  (`experts_format = "ov_ir"`, per-expert OV CPU-plugin calls); oneDNN
  has AVX2/AVX-VNNI kernels. Cost: per-call overhead × 8 experts × 40
  layers/token — the probe measures whether launch overhead eats the
  bandwidth win.
- **(ii) AVX2/AVX-VNNI port of `cascadia-int4-gemm`** — real kernel
  work, priced only if (i) fails.
- **(iii) GPU expert execution** — does not exist today; largest build.
- **(iv) Re-target to AVX-512 hosts** (Xeon miners) — abandons the
  AI-PC story for this model.

**Device placement (unstated in rev 4, must be decided in M2'-0):**
shells (DeltaNet/attention/router) on GPU with experts on CPU means ~80
cross-device transitions/token — at 0.25-1 ms each that alone caps
12-50 tok/s. All-CPU placement avoids ping-pong but puts shell compute
on the same bandwidth-limited cluster. The probe measures one full
shell→experts→shell layer round-trip per placement, not the GEMM in
isolation.
#### 4.2b Engine scope (delta vs the Kimi-only engine)

`cascadia-engine-sparse-moe` changes:

- **Router**: softmax top-8 (Kimi-compatible) PLUS **1 shared expert
  with sigmoid gate** summed alongside routed output (new), per
  `Qwen3_5MoeSparseMoeBlock` semantics. Top-k renormalization is
  hardcoded in this family (no `norm_topk_prob` knob — verified
  transformers 5.4.0).
- **Expert GEMM shapes**: 2048x512 (gate/up) + 512x2048 (down) int4
  paths in `cascadia-int4-gemm` (Kimi's are wider; kernels are
  shape-generic but efficiency at 512 needs measuring).
- **Shell**: per-layer-range OV IR shells containing embeddings/
  DeltaNet/full-attention/router — DeltaNet via the OV fused
  `GatedDeltaNet` op (compile-time fusion; never hand-rolled). Shells
  are stateful (`ReadValue`/`Assign`) with the §4.1 reset semantics.
  This inherits §4.3's exporter risk: fused-op survival in a layer-range
  convert is M2''s exit criterion, unchanged.
- **Manifest**: `arch = "qwen3_5_moe"`, shared-expert fields,
  `layer_types` (3:1 linear:full), linear-state dims.
- **engine_args knobs** (`top_k_override`, `routing_threshold`,
  `ffn_sparsity_threshold`) carry over unchanged. They trade quality for
  speed and do NOT address the §4.2a failure modes (ISA, non-expert
  traffic, device transitions) — not an escape hatch for a missed
  ceiling.
- **Expert bin format**: the int4 GEMM crate consumes group_size=32
  symmetric zero_point=8 bf16-scale bins — NOT the official channel-wise
  `-int4-ov` recipe. The exporter converts; format-roundtrip parity is
  an M2' criterion.

#### 4.2c What sharding actually buys (rev 5: single-stream honesty)

int4 Qwen3.6 fits one 32 GB box; the single-box problem is bandwidth.
**Pipeline sharding does not add single-stream bandwidth**: with batch=1
greedy (the v1 invariants), per-token time = sum of stage times +
network ≥ single-node time. The rev-4 "2 nodes → ~2x" claim was wrong.
Sharding buys: (a) per-node memory headroom (KV/state/OS comfort on
32 GB boxes), (b) **multi-stream throughput** — N concurrent sequences
keep stages busy — which contradicts §4.1 invariant 1 and is therefore
a priced future spec change (per-task state isolation across stages),
not a v1 deliverable.

Bigger family members (Qwen3.5-235B-A22B etc.) are a **possible future
direction, unpriced, separate spec** — note A22B's active set (~11 GB
int4/token) caps single-stream at ~7 tok/s on this class of node
*regardless of node count*, and its bf16 checkpoint (~470 GB) exceeds
the only export host we have. Sharding gets big models to *fit*, not to
*speed*, without multi-stream. Qwen3.6-35B must justify this port on
its own numbers.

DECISION (conditional): proceed to M2'-0; the port is built only if the
probe clears the kill condition below.

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

## 5. Milestones (rev 4)

- **M1 — feasibility measurements: DONE 2026-06-11.** Dense whole-model
  GPU: 1.27 tok/s greedy / 2.46 prompt-lookup (no-go vs T=10). Official
  IR inspected: 30x `Loop` DeltaNet (fusion is compile-time, not
  serialized), 80 state variables matching §3 exactly, dense bmm MoE, no
  fused MoE op. Version triple: transformers 5.4.0 + OV/GenAI 2026.2
  works; `norm_topk_prob` not a config axis. Single-stage serving via
  VLMPipeline landed separately (Path 1, docs/architectures/qwen3.6.md).
- **M2'-0 — ISA/device probe (Lunar Lake box, no export host needed).**
  Measure on TARGET silicon, kernel path logged and asserted
  (avx512/avx2/scalar): (1) `cascadia-int4-gemm` scalar-fallback GEMV
  rate on LNL; (2) OV-IR per-expert-call overhead (8×40 calls/token
  envelope) via the existing `ov_ir` backend; (3) one synthetic
  shell→experts→shell layer round-trip per device placement (all-CPU vs
  GPU-shell+CPU-experts). **Kill condition: if no §4.2a resolution
  option extrapolates ≥ T on target silicon, decision (b) is dead like
  (a) was, and the model is declared unsupported on the sharded path for
  AI-PC-class nodes.**
- **M2' — expert-path probe (needs 133 GB-RAM export host; gated on
  M2'-0).** Export one full-attention layer AND one DeltaNet layer's
  experts + shells (fused GatedDeltaNet present in OUR IR); reconcile
  the §4.2a per-token budget term-by-term against measurement
  (including lm_head and state I/O); int4 format-roundtrip parity vs
  reference with a numeric threshold (greedy token-match ≥ N tokens).
- **M3' — engine.** Router (+shared expert), manifest, full 40-layer
  single-box sparse run; greedy parity vs the M1 whole-model output;
  measure vs T.
- **M4' — mesh.** 2-node split over the existing transport. Measures
  single-stream rate (expected: ≈ M3' minus network overhead — NOT 2x;
  see §4.2c) + memory headroom; multi-stream scaling only if the §4.2c
  spec change is taken; soak; docs.
- **M5 / v1.1 — enterprise** (unchanged): gossip N+1 bump, `EngineKind`
  wire tag + wire-pin update, embedded-backend staged-pipeline path.

## 6. Out of scope (v1)

Vision tower, MTP/speculative decoding (hard-fenced — §4.1 invariant 2),
batching / multiple live tasks per stage (§4.1 invariant 1), mid-sequence
migration/failover, prefix caching, continuous batching, dense-MoE staged
execution (measured dead at M1), enterprise embedded integration
(v1.1, §4.5), Qwen3.5/Qwen3-Next variants (plausibly fall out of the same
path — verify in M2, don't promise).
