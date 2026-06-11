# Qwen3.6-35B-A3B sharded support — design spec (rev 6)

Status: **M2'-0 PASSED (2026-06-11) — strategy chosen: all-CPU OV-IR
experts.** Measured full-layer-loop envelope ~17 tok/s on Lunar Lake
(≥ T=10); the sharded design proceeds to M2'. Details in §5. Single-stage (nonsharded) serving already works
and is documented separately (`qwen3.6.md`).

Review lineage: rev 1 draft → rev 2 (adversarial+feasibility review:
dense-MoE cost re-gated, exporter corrected, preconditions pinned) →
rev 3 (issue #77 alignment) → rev 4 (M1 measured dense-MoE dead; pivot
to sparse routed dispatch) → rev 5 (review: ISA precondition, corrected
bandwidth, single-stream honesty) → **rev 6** (4-angle review incl.
external: "port" framing replaced with new-engine reality, ov_ir
call-overhead prior added, alternatives section added, benchmark
protocol defined, stale §2 fixed, M4' cut to conditional). Rev 5's
detailed exporter/runtime sections live in git history (`b593466`) and
return only if M2'-0 passes.

## 1. Model facts (verified against config.json + the official IR)

- `model_type: qwen3_5_moe` (`Qwen3_5MoeForConditionalGeneration`;
  expert fields nested under `text_config`). No `qwen3_6` type exists.
- 40 layers, hidden 2048, vocab 248320: 3× Gated-DeltaNet (linear
  attention, recurrent S `[32,128,128]` + conv `[8192,4]` state, no KV)
  : 1× full attention (GQA 16/2, head_dim 256, `attn_output_gate`,
  partial-rotary 0.25 with mRoPE `[11,11,10]` interleaved).
- MoE per layer: 256 experts top-8 + 1 shared expert (sigmoid gate),
  `moe_intermediate_size 512`. ~3B active of 35B total.
  `norm_topk_prob` is not a config axis in this family (transformers
  5.4.0 verified).
- Official IR (`OpenVINO/Qwen3.6-35B-A3B-int4-ov`): 30× `Loop` DeltaNet
  (fused `GatedDeltaNet` op materializes at compile time), 80 state
  variables (30 ssm + 30 conv + 10 K + 10 V), dense-bmm MoE, no fused
  MoE op. VLM-layout; vision tower out of scope (text-only).
- MTP head present; ignored (spec-decode is fenced off — §4.1).

## 2. Where the repo stands today (rev 6: corrected)

- `is_moe_config()` **now** rejects this model correctly config-first —
  nested `text_config` unwrap + `qwen3_5_moe` model_type landed on this
  branch (`e01faf4`, TDD'd) with a rejection message pointing at the
  single-stage path. (Earlier revs described the pre-fix gap; stale.)
- Single-stage serving works end-to-end via VLMPipeline in the
  `ov-genai` engine (validated 2026-06-11, `qwen3.6.md`).
- The generic shard exporter and the staged engines have no notion of
  DeltaNet state, mRoPE, or routed MoE — unchanged.

## 3. Measured + code-verified priors (what any design must beat/respect)

- **M1 (2026-06-11, pawan-01 = Lunar Lake 258V, 32 GB UMA, GPU,
  single-run — see §5 protocol for why future gates need better):**
  dense whole-model int4 via Intel GenAI: 1.27 tok/s greedy, 2.46
  prompt-lookup. Dense-MoE staged execution is dead (stages of this
  graph are strictly slower).
- **Active-set bandwidth model** (corrected rev 5): ~1.7–1.8 GB/token
  (routed 566 MB + shared 63 MB + attention/DeltaNet/router ~800 MB +
  lm_head 254 MB + int4 scales ~12.5% + state R/W ~60 MB + KV(ctx)).
  Ideal ceiling at 60–90 GB/s: 33–50 tok/s; realistic 15–35. ~10× over
  dense — IF the silicon can consume the bytes.
- **ISA wall:** every SIMD path in `cascadia-int4-gemm` is
  AVX-512-gated with a SILENT pure-scalar fallback and no AVX2 path
  (verified; the fallback logs nothing). Lunar Lake has no AVX-512.
- **ov_ir call-overhead prior: OVERTURNED by M2'-0.** The manifest's
  ~5 ms/call (miner-Xeon era) measured as **~0.106 ms/call on Lunar
  Lake** (OV 2026.2 CPU plugin, oneDNN AVX2): 320 single-expert
  calls/token = 33.9 ms median. Call overhead is negligible —
  batching 8 experts/call was *slower* (41.7 ms). Strategy (C) is
  viable, not dead.
- **The sparse-moe engine is NOT a portable base today** (feasibility
  review, file:line in review record): the OV shell path is vestigial
  (`manifest.shell_xml` has zero callers); all in-range layers run
  hard-coded Kimi-shaped Rust shells (MLA constants, `HIDDEN=7168`);
  `format.rs` structurally rejects non-Kimi expert bins (size check);
  the expert-bin exporter lives in rainier, not this repo; `LayerState`
  is KV-only with no layer-type concept. **Shared-expert plumbing DOES
  already exist** (`shared_expert_out` in shell.rs/runner.rs) — the
  Qwen delta there is gate algebra, not new machinery.
- DeltaNet state semantics: linear state cannot be trimmed — any
  rollback is a full reset (genai PR #3359). State is layer-local;
  steady-state pipeline wire = hidden activations + position frame only.

## 4. The open design question

**What runtime executes Qwen3.6's experts fast enough on the target
hardware, and is any multi-node arrangement worth it for this model?**
Nothing below is decided; M2'-0 (§5) decides.

### 4.1 Invariants any candidate must hold (carried from rev 2, still valid)

1. Batch = 1, one live task per stage (OV stateful vars are
   per-infer-request; DeltaNet state has no position mask to bound
   corruption).
2. Greedy only; all speculative/draft paths fenced (rollback = full
   reset for linear state).
3. Position-0 reset is the only state-recovery mechanism; error paths
   must guarantee next-task entry at position 0. **Rev 6 addition
   (external review): cancellation, client disconnect, timeout, and
   mid-token downstream failure are explicit test cases for whichever
   engine is built — the gemma4 precedent inherits the no-op
   `Engine::cancel` default and never tested these against
   non-recoverable state.**
4. No activation-wire or ShardSpec changes; gossip `EngineKind` bump
   only at enterprise integration (ADR-001).

### 4.2 Candidate execution strategies (decided by M2'-0 + §5 protocol)

- **(A) CPU routed dispatch, AVX2/AVX-VNNI kernel port** — new kernel
  work in `cascadia-int4-gemm` (no AVX2 path exists) + de-Kimi-fying
  the bin container + a new heterogeneous-layer execution loop with a
  DeltaNet state type. Honest name: **a new engine that reuses the
  expert-cache/router concepts**, not a port.
- **(B) GPU expert execution** — does not exist in any form; largest
  build; sidesteps the ISA wall entirely and shares the device with the
  shells (no ping-pong).
- **(C) OV-IR experts (oneDNN AVX2)** — presumptively dead on the 5 ms
  prior; only alive if per-layer expert batching (one OV call per layer
  for all 8 experts) collapses the call count 8×.
- **(D) Re-target to AVX-512 hosts (Xeon miners)** — abandons the
  AI-PC story for this model; zero kernel work.

Cross-device placement note: GPU shells + CPU experts = ~80
transitions/token; at 0.25–1 ms each that alone caps 12–50 tok/s.
All-CPU contends with the (lower) CPU-cluster bandwidth. Placement is a
measured M2'-0 axis, not a default.

### 4.3 Alternatives to layer-pipeline sharding (rev 6: decision-grade, previously unexamined)

- **Expert parallelism** — shard by EXPERT OWNERSHIP, not layers: every
  node holds the shells + a subset of experts; per layer, the 2048-dim
  hidden state (~4 KB bf16) travels to expert owners instead of
  ~13 MB/layer of expert weights being re-read locally. For a
  bandwidth-bound MoE this is a fundamentally better exchange rate, it
  aggregates the mesh's total memory bandwidth for ONE stream (which
  layer-pipelining cannot do — §3), and DeltaNet state stays wholly on
  the shell owner. Cost: per-layer network round-trips (latency-bound;
  ~40 × RTT/token floor) + a new wire pattern (= protocol work the
  pipeline approach avoids). On LAN RTTs (~0.2–0.5 ms) the floor is
  8–20 ms/token network — potentially compatible with 30+ tok/s.
  **This is the only examined arrangement where multi-node helps a
  single stream; it must be priced in M2'-0's paper analysis before
  any pipeline work.**
- **Tensor parallelism** — the repo already documents it
  (`docs/architecture/tensor-parallelism.md`) as the latency-reducing
  design on high-bandwidth fabric; per-layer all-reduce of 2048-dim
  activations. Same network-bound character as expert parallelism;
  examine together.
- **Runtime reuse (llama.cpp/GGUF)** — ggml has AVX2 int4 kernels and
  qwen3-MoE support today; a GGUF side-engine contradicts the
  OV-centric stack but is the cheapest path to "fast on AI-PC CPU".
  Named for honesty; likely rejected on stack-coherence grounds — but
  rejected explicitly, not silently.
- **Wait for vendor** — Intel's CB/paged path (OVMS) gains Qwen3.x MoE
  perf every release; the B60-class GPU answer may simply arrive.
  Zero work; zero differentiation.

### 4.4 Exporter and runtime (conditional placeholders)

Rev 5 specified a convert-with-extensions layer-range exporter and a
gemma4-style runtime in detail; both are **suspended** until M2'-0
picks a strategy (they presume strategy A/B shapes). See `b593466` for
the suspended text. One scope-review note carried forward: **IR surgery
on the official whole-model int4 IR (cutting at layer boundaries) may
be cheaper than re-conversion** — it inherits fused ops and
quantization, eliminating two of the three exporter risks; it must be
costed against convert-with-extensions when the exporter returns.

### 4.5 Enterprise

Unchanged from rev 2/5: any embedded-engine variant = gossip
`/cascadia/gossip/N+1` bump + wire-pin update (ADR-001); enterprise has
no staged-pipeline precedent; enterprise v1 story for this model stays
the OVMS whole-model backend. All enterprise sharded work is v1.1+.

## 5. Milestones (rev 6)

**Benchmark protocol (applies to every gate; rev 6, external review):**
fixed prompt set + context lengths, ≥3 warmup + ≥5 measured runs,
report median + p10/p90, end-to-end tok/s (not per-stage), kernel/ISA
path logged AND asserted (new instrumentation — the current fallback is
silent), correctness check per run (greedy token-match vs reference,
N = 64 tokens), power/thermal state noted. M1's single-run numbers are
indicative only; any kill/ship decision re-measures under this
protocol. T = 10 tok/s decode at 1K context (TODO(review): confirm).

- **M1 — DONE 2026-06-11** (single-run; see §3 for results and caveat).
- **M2'-0 — strategy probe: DONE 2026-06-11 (pawan-01), PASSED.**
  Protocol: 3 warmup + 5 runs, medians. Results:
  (1) scalar Rust GEMV, Qwen shapes, `avx512=false` asserted, rayon×8:
  gate/up 0.109 ms, down 0.099 ms → ~0.32 ms/expert → ~10 tok/s
  expert-only floor on the SCALAR path (the scalar-doom prior was
  wrong: 0.5 MB matrices are cache-friendly).
  (2) OV per-expert call: 0.106 ms (320 calls = 33.9 ms/token →
  29.5 tok/s expert ceiling); batched-8 slower (41.7 ms) — launch
  overhead negligible.
  (3) Placement: full 40-layer shell+experts loop — all-CPU 58.9 ms
  (**17.0 tok/s**), GPU-shell+CPU-experts 70.8 ms (14.1 tok/s);
  ping-pong costs ~12 ms/token, all-CPU wins.
  **Chosen strategy: (C) all-CPU, OV-IR experts** (Rust AVX2 kernel
  port demoted to optional optimization). Caveats carried to M2':
  synthetic f32 shells (real = int4 + fused GatedDeltaNet), no real
  lm_head/KV-growth modeled, one box. Note: single-box sparse ~17 tok/s
  vs 1.27 whole-model makes **M3' the user-value milestone** (13×)
  independent of any mesh.
- **M2' — IN PROGRESS via IR surgery (no export host needed).** The
  official IR's expert stacks are directly sliceable: per layer,
  gate/up `[256,512,32x64] u4` + down `[256,2048,8x64] u4` (+ zp/scale),
  expert index = leading axis, contiguous 512 KB strides; offsets in the
  XML. **Brick 1 DONE (2026-06-11, pawan-01): one-expert slice
  (layer 0, expert 3) byte-sliced from the .bin, dequantized, rebuilt as
  an OV model — numpy-vs-OV parity max_rel 7.8e-7**
  (`tools/qwen36_surgery/probe_expert_slice.py`). Eliminates the 133 GB
  export host, re-quantization, and the bin-format conversion entirely
  (strategy C keeps OV's group layout). **Brick 2 DONE: semantic
  parity vs the full model's layer-0 MoE output — max_rel 5.9e-3
  (f16 noise), top-8 indices exact, all conventions proven from the
  graph: softmax→top8→renorm, silu, shared expert × sigmoid(gate
  [1,2048] u4-quantized)** (`probe_moe_parity.py`, staged-tap
  comparison). **Brick 3 DONE: shell extraction
  proven BIT-EXACT (rel 0.0e0)** — layer 0 (DeltaNet) cut from the full
  IR as a standalone stateful stage (boundary Parameter rewire; ssm.0 +
  conv.0 preserved as ReadValue/Assign sinks; only beam_idx needed as
  aux input; saved + compiled + ran standalone vs full-model tap;
  `probe_shell_extract.py`). The hard half of shard creation is
  de-risked; **full-attention layer cut (layer 3) also BIT-EXACT (0.0e0)** —
  key.0/value.0 KV state preserved, position_ids/attention_mask ride
  along (`probe_shell_extract_l3.py`; exporter cleanup noted: rewire
  upstream ShapeOf chains off inputs_embeds onto stage_hidden). BOTH
  layer types proven. **EXPORTER WORKING**
  (`export_qwen36_moe.py`): official IR → N stage shards + manifest;
  2-stage chained validation vs full model passes token-level
  (top1 match, top5 5/5, logits rel 2.8e-2 — f16 fusion-order noise
  injected per full-attn layer, DeltaNet layers bit-exact; criterion is
  token agreement per the §5 protocol, with ≥64-token greedy parity at
  M3'). Debug findings encoded in the tool: state vars are numbered by
  LAYER-TYPE SEQUENCE (conv/ssm 0..29, kv 0..9), and global past-length
  bookkeeping reads early layers' caches — non-owning stages rewire
  those onto owned same-kind caches. Validation uses a real token via
  the text-embeddings IR (random embeds make logits degenerate).
  Remaining: `cascadia shard` dispatch arm;
  `cascadia shard` dispatch arm for `qwen3_5_moe` (same-command UX is an
  exit criterion).
- **M3' prototype DONE (2026-06-11): 64/64 greedy token parity** over
  the 2-stage chain vs whole model (f16 drift never flips greedy).
  Device map measured: CPU decode 7.2 tok/s short-ctx / 4.9 @1K (chain
  4.8) but CPU prefill 123 s @1K; GPU prefill 6-10 s but decode
  1.3-2.5. **No single device clears T — the M3' engine design is
  heterogeneous: GPU prefill + CPU decode**, with routed-dispatch
  externalization (the ~17 tok/s envelope; MoE cut points proven in
  brick 2) as the decode optimization. `proto_m3_decode.py` is the
  decode-loop reference for the Rust engine.
- **M3' engine E2E (2026-06-12): 5/5 API suite green** on pawan-01
  (`cascadia run <shards> --engine qwen36-moe --device CPU --api`):
  /v1/models, two sequential chats (state-reset invariant), greedy
  determinism (identical 13-token outputs), 96-token generation at
  4.7-8.8 tok/s decode (matches the M2' chain envelope). An earlier
  2/5 run (empty contents on requests 2+) did NOT reproduce after a
  server restart with `RUST_LOG=info` → node-side log; OV-level reset
  was ruled out as the cause (probe_reset_state.py: 40 states on
  stage0, bit-exact position-0 reproduction after reset). The wedged
  instance had no logging; keep `RUST_LOG=info` + log file on every
  serve so a recurrence is attributable. Known gap found while
  testing: client disconnect on the non-streaming API path does NOT
  cancel the in-flight task (it runs to completion; server stays
  healthy) — robustness item, not a wedge.
- **M3' acceptance (2026-06-12): engine ≡ Python chain EXACTLY** —
  64/64 generated tokens identical on a fresh prompt through the API
  (`probe_engine_parity.py` whole-model reference +
  `probe_chain_vs_full_prompt.py` 3-way discriminator). The Rust
  engine is a bit-faithful port of the validated decode loop. Chain vs
  whole-model greedy is prompt-dependent: 64/64 on the prototype's
  reference prompt, first divergence at token 37 on the parity-probe
  prompt (f16 fusion-order near-tie flip, both continuations coherent
  — same property already documented at exporter validation; engine
  inherits it from the shards, not from the port). Engine tok/s @64
  tokens short-ctx: 8.7 (log-reported), envelope 4.7-8.8 across the
  E2E suite.
- **M3' robustness (2026-06-12): cancel/disconnect validated** after
  rewriting `step()` to be incremental (one decode token per call;
  full prefill in the first call — the runner closes streams after 3
  empty steps). The monolithic step() held the engine mutex for the
  whole generation, making cancel unreachable and streaming emit one
  blob. Now: per-token SSE chunks at the engine's real cadence
  (~150-250 ms); `/v1/cancel/:id` interrupts mid-decode (29/200
  tokens, "cancelled; resetting state" in log); SSE client disconnect
  cancels via ChunkStream::Drop (29/200); follow-up requests coherent;
  E2E suite re-passed 5/5. Known minor: a cancelled task's deferred
  final chunk is buffered for a stream nobody drains (Drop removed the
  cancelled-flag first) — one map entry per cancelled task, runner-side
  cleanup candidate; and engine finalization of a cancelled task waits
  for the next poll (no poller after stream close), reset still
  precedes next admission so the invariant holds.
- **M3' engine build / M4' — remaining, proceed from the prototype.** Shapes (export probe, engine,
  mesh) return from `b593466` rewritten around the chosen strategy.
  Standing corrections whenever they return: M3' is the user-value
  milestone for single-box; a mesh milestone must name a concrete
  scenario a single box cannot serve (multi-stream = a §4.1 invariant-1
  spec change, priced separately) — otherwise it is cut, not deferred.
- **M5 / v1.1 — enterprise** (unchanged, conditional on all of the
  above).

## 6. Out of scope (v1, regardless of strategy)

Vision tower; MTP/speculative decoding; batching/multi-stream (priced
spec change, not a default); prefix caching on linear layers;
mid-sequence migration/failover; dense-MoE staged execution (measured
dead at M1); Qwen3.5-235B-A22B and other family members (unpriced
future direction — note its ~11 GB/token active set caps single-stream
at ~7 tok/s on this node class at ANY node count, and its checkpoint
exceeds our export host; capacity ≠ speed).
