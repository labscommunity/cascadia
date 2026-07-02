# Qwen3.6-35B-A3B sharded support

Single-stage (nonsharded) serving works and is hardware-validated
(`qwen3.6.md`), and the staged engine (`--engine qwen36-moe`) is
shipped — single-box and N-rank pipeline. Strategy: all-CPU OV-IR
experts. Measured on Lunar Lake: 4.7-8.8 tok/s single-box short-context
(4.1-4.8 tok/s in 2-node pipeline mode); the ~17 tok/s figure quoted in
the early strategy probe (§5) was a synthetic full-layer-loop upper
bound, not an end-to-end number.

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

## 2. Where the repo stands today

- `is_moe_config()` rejects this model correctly config-first —
  nested `text_config` unwrap + `qwen3_5_moe` model_type — with a
  rejection message pointing at the single-stage path.
- Single-stage serving works end-to-end via VLMPipeline in the
  `ov-genai` engine (hardware-validated, `qwen3.6.md`).
- `cascadia shard` dispatches `qwen3_5_moe` to a dedicated IR-surgery
  exporter (§5), and the in-process `qwen36-moe` staged engine serves
  the resulting stage shards.
- The generic shard exporter and the other staged engines have no
  notion of DeltaNet state, mRoPE, or routed MoE — unchanged.

## 3. Measured + code-verified priors (what any design must beat/respect)

- **Dense whole-model baseline (Intel Lunar Lake 258V, 32 GB UMA, GPU,
  single-run — see §5 protocol for why decisions need better):**
  dense whole-model int4 via Intel GenAI: 1.27 tok/s greedy, 2.46
  prompt-lookup. Dense-MoE staged execution is dead (stages of this
  graph are strictly slower).
- **Active-set bandwidth model**: ~1.7–1.8 GB/token
  (routed 566 MB + shared 63 MB + attention/DeltaNet/router ~800 MB +
  lm_head 254 MB + int4 scales ~12.5% + state R/W ~60 MB + KV(ctx)).
  Ideal ceiling at 60–90 GB/s: 33–50 tok/s; realistic 15–35. ~10× over
  dense — IF the silicon can consume the bytes.
- **ISA wall:** every SIMD path in `cascadia-int4-gemm` is
  AVX-512-gated with a SILENT pure-scalar fallback and no AVX2 path
  (verified; the fallback logs nothing). Lunar Lake has no AVX-512.
- **ov_ir call-overhead prior: OVERTURNED by the strategy probe.** The
  ~5 ms/call prior (measured on an older Xeon export host) measured as
  **~0.106 ms/call on Lunar Lake** (OV 2026.2 CPU plugin, oneDNN AVX2):
  320 single-expert calls/token = 33.9 ms median. Call overhead is
  negligible — batching 8 experts/call was *slower* (41.7 ms).
  Strategy (C) is viable, not dead.
- **The sparse-moe engine is NOT a portable base today** (feasibility
  review at the time; since then MiniMax-M2 support activated the
  OV-IR shell backend — see `manifest.rs`): the OV shell path was
  vestigial (`manifest.shell_xml` had zero callers); all in-range
  layers ran hard-coded Kimi-shaped Rust shells (MLA constants,
  `HIDDEN=7168`); `format.rs` structurally
  rejects non-Kimi expert bins (size check); the expert-bin exporter
  is not part of this repo; `LayerState` is KV-only with no layer-type
  concept. **Shared-expert plumbing DOES already exist**
  (`shared_expert_out` in shell.rs/runner.rs) — the Qwen delta there
  is gate algebra, not new machinery.
- DeltaNet state semantics: linear state cannot be trimmed — any
  rollback is a full reset (genai PR #3359). State is layer-local;
  steady-state pipeline wire = hidden activations + position frame only.

## 4. The design question

**What runtime executes Qwen3.6's experts fast enough on the target
hardware, and is any multi-node arrangement worth it for this model?**
The strategy probe in §5 decided this by measurement.

### 4.1 Invariants any candidate must hold

1. Batch = 1, one live task per stage (OV stateful vars are
   per-infer-request; DeltaNet state has no position mask to bound
   corruption).
2. Greedy only; all speculative/draft paths fenced (rollback = full
   reset for linear state).
3. Position-0 reset is the only state-recovery mechanism; error paths
   must guarantee next-task entry at position 0. Cancellation, client
   disconnect, timeout, and mid-token downstream failure are explicit
   test cases for whichever engine is built — the gemma4 precedent
   inherits the no-op `Engine::cancel` default and never tested these
   against non-recoverable state.
4. No activation-wire or ShardSpec changes.

### 4.2 Candidate execution strategies (decided by the §5 probe)

- **(A) CPU routed dispatch, AVX2/AVX-VNNI kernel port** — new kernel
  work in `cascadia-int4-gemm` (no AVX2 path exists) + de-Kimi-fying
  the bin container + a new heterogeneous-layer execution loop with a
  DeltaNet state type. Honest name: **a new engine that reuses the
  expert-cache/router concepts**, not a port.
- **(B) GPU expert execution** — does not exist in any form; largest
  build; sidesteps the ISA wall entirely and shares the device with the
  shells (no ping-pong).
- **(C) OV-IR experts (oneDNN AVX2)** — presumptively dead on the
  5 ms/call prior; only alive if that prior fails to hold (it did not
  hold — see §3 and the probe results in §5).
- **(D) Re-target to AVX-512 hosts (Xeon-class servers)** — abandons
  the AI PC story for this model; zero kernel work.

Cross-device placement note: GPU shells + CPU experts = ~80
transitions/token; at 0.25–1 ms each that alone caps 12–50 tok/s.
All-CPU contends with the (lower) CPU-cluster bandwidth. Placement is a
measured axis (§5), not a default.

### 4.3 Alternatives to layer-pipeline sharding

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
  single stream** — but it lives or dies on RTT (see the wire-latency
  measurement in §5).
- **Tensor parallelism** — the repo already documents it
  (`docs/architecture/tensor-parallelism.md`) as the latency-reducing
  design on high-bandwidth fabric; per-layer all-reduce of 2048-dim
  activations. Same network-bound character as expert parallelism;
  examine together.
- **Runtime reuse (llama.cpp/GGUF)** — ggml has AVX2 int4 kernels and
  qwen3-MoE support today; a GGUF side-engine contradicts the
  OV-centric stack but is the cheapest path to "fast on AI PC CPU".
  Named for honesty; likely rejected on stack-coherence grounds — but
  rejected explicitly, not silently.
- **Wait for vendor** — Intel's CB/paged path (OVMS) gains Qwen3.x MoE
  perf every release; the B60-class GPU answer may simply arrive.
  Zero work; zero differentiation.

### 4.4 Exporter approach

An earlier draft specified a convert-with-extensions layer-range
exporter (re-tracing the model per stage). That was superseded: **IR
surgery on the official whole-model int4 IR (cutting at layer
boundaries) is cheaper than re-conversion** — it inherits fused ops
and quantization, eliminating two of the three exporter risks
(re-quantization and format conversion). The shipped exporter (§5)
takes the IR-surgery path.

## 5. Measurements and validation

**Benchmark protocol (applies to every performance gate):**
fixed prompt set + context lengths, ≥3 warmup + ≥5 measured runs,
report median + p10/p90, end-to-end tok/s (not per-stage), kernel/ISA
path logged AND asserted (the current fallback is silent), correctness
check per run (greedy token-match vs reference, N = 64 tokens),
power/thermal state noted. Single-run numbers are indicative only; any
kill/ship decision re-measures under this protocol. Target: 10 tok/s
decode at 1K context.

All measurements below were taken on Intel Lunar Lake nodes
(Core Ultra 7 258V, 32 GB UMA) unless noted.

### Strategy probe (3 warmup + 5 runs, medians)

- (1) Scalar Rust GEMV, Qwen shapes, `avx512=false` asserted, rayon×8:
  gate/up 0.109 ms, down 0.099 ms → ~0.32 ms/expert → ~10 tok/s
  expert-only floor on the SCALAR path (the scalar-doom prior was
  wrong: 0.5 MB matrices are cache-friendly).
- (2) OV per-expert call: 0.106 ms (320 calls = 33.9 ms/token →
  29.5 tok/s expert ceiling); batched-8 slower (41.7 ms) — launch
  overhead negligible.
- (3) Placement: full 40-layer shell+experts loop — all-CPU 58.9 ms
  (**17.0 tok/s**), GPU-shell+CPU-experts 70.8 ms (14.1 tok/s);
  ping-pong costs ~12 ms/token, all-CPU wins.

**Chosen strategy: (C) all-CPU, OV-IR experts** (Rust AVX2 kernel port
demoted to optional optimization). Caveats: synthetic f32 shells
(real = int4 + fused GatedDeltaNet), no real lm_head/KV-growth
modeled, one box. Note: single-box sparse ~17 tok/s vs 1.27
whole-model makes the staged engine the user-value deliverable (13×)
independent of any mesh.

### IR surgery (no export host needed)

The official IR's expert stacks are directly sliceable: per layer,
gate/up `[256,512,32x64] u4` + down `[256,2048,8x64] u4` (+ zp/scale),
expert index = leading axis, contiguous 512 KB strides; offsets in the
XML. Proven step by step:

- **One-expert slice** (layer 0, expert 3) byte-sliced from the .bin,
  dequantized, rebuilt as an OV model — numpy-vs-OV parity max_rel
  7.8e-7 (`tools/qwen36_surgery/probe_expert_slice.py`). Eliminates
  the 133 GB export host, re-quantization, and the bin-format
  conversion entirely (strategy C keeps OV's group layout).
- **Semantic parity** vs the full model's layer-0 MoE output —
  max_rel 5.9e-3 (f16 noise), top-8 indices exact, all conventions
  proven from the graph: softmax→top8→renorm, silu, shared expert ×
  sigmoid(gate [1,2048] u4-quantized) (`probe_moe_parity.py`,
  staged-tap comparison).
- **Shell extraction proven BIT-EXACT (rel 0.0e0)** — layer 0
  (DeltaNet) cut from the full IR as a standalone stateful stage
  (boundary Parameter rewire; ssm.0 + conv.0 preserved as
  ReadValue/Assign sinks; only beam_idx needed as aux input; saved +
  compiled + ran standalone vs full-model tap;
  `probe_shell_extract.py`). The **full-attention layer cut (layer 3)
  is also BIT-EXACT (0.0e0)** — key.0/value.0 KV state preserved,
  position_ids/attention_mask ride along
  (`probe_shell_extract_l3.py`). BOTH layer types proven.
- **Exporter working** (`tools/qwen36_surgery/export_qwen36_moe.py`):
  official IR → N stage shards + manifest; 2-stage chained validation
  vs full model passes token-level (top1 match, top5 5/5, logits rel
  2.8e-2 — f16 fusion-order noise injected per full-attn layer,
  DeltaNet layers bit-exact; the acceptance criterion is token
  agreement per the protocol above, with ≥64-token greedy parity at
  the engine gate). Debug findings encoded in the tool: state vars are
  numbered by LAYER-TYPE SEQUENCE (conv/ssm 0..29, kv 0..9), and
  global past-length bookkeeping reads early layers' caches —
  non-owning stages rewire those onto owned same-kind caches.
  Validation uses a real token via the text-embeddings IR (random
  embeds make logits degenerate).

### Prototype and engine validation

- **Prototype: 64/64 greedy token parity** over the 2-stage chain vs
  whole model (f16 drift never flips greedy). Device map measured: CPU
  decode 7.2 tok/s short-ctx / 4.9 @1K (chain 4.8) but CPU prefill
  123 s @1K; GPU prefill 6-10 s but decode 1.3-2.5. No single device
  clears the target — the engine design is heterogeneous: GPU prefill
  + CPU decode, with routed-dispatch externalization (the ~17 tok/s
  envelope; MoE cut points proven above) as the decode optimization.
  `proto_m3_decode.py` is the decode-loop reference for the Rust
  engine.
- **Engine E2E: 5/5 API suite green**
  (`cascadia run <shards> --engine qwen36-moe --device CPU --api`):
  /v1/models, two sequential chats (state-reset invariant), greedy
  determinism (identical 13-token outputs), 96-token generation at
  4.7-8.8 tok/s decode (matches the chain envelope). An earlier 2/5
  run (empty contents on requests 2+) did NOT reproduce after a
  server restart with `RUST_LOG=info` → node-side log; OV-level reset
  was ruled out as the cause (probe_reset_state.py: 40 states on
  stage0, bit-exact position-0 reproduction after reset). The wedged
  instance had no logging; keep `RUST_LOG=info` + log file on every
  serve so a recurrence is attributable.
- **Acceptance: engine ≡ Python chain EXACTLY** — 64/64 generated
  tokens identical on a fresh prompt through the API
  (`probe_engine_parity.py` whole-model reference +
  `probe_chain_vs_full_prompt.py` 3-way discriminator). The Rust
  engine is a bit-faithful port of the validated decode loop. Chain vs
  whole-model greedy is prompt-dependent: 64/64 on the prototype's
  reference prompt, first divergence at token 37 on the parity-probe
  prompt (f16 fusion-order near-tie flip, both continuations coherent
  — same property already documented at exporter validation; the
  engine inherits it from the shards, not from the port). Engine tok/s
  @64 tokens short-ctx: 8.7 (log-reported), envelope 4.7-8.8 across
  the E2E suite.
- **Robustness: cancel/disconnect validated** after rewriting `step()`
  to be incremental (one decode token per call; full prefill in the
  first call — the runner closes streams after 3 empty steps). The
  monolithic step() held the engine mutex for the whole generation,
  making cancel unreachable and streaming emit one blob. Now:
  per-token SSE chunks at the engine's real cadence (~150-250 ms);
  `/v1/cancel/:id` interrupts mid-decode (29/200 tokens, "cancelled;
  resetting state" in log); SSE client disconnect cancels via
  ChunkStream::Drop (29/200); follow-up requests coherent; E2E suite
  re-passed 5/5. Known minor: a cancelled task's deferred final chunk
  is buffered for a stream nobody drains (Drop removed the
  cancelled-flag first) — one map entry per cancelled task, runner-side
  cleanup candidate; and engine finalization of a cancelled task waits
  for the next poll (no poller after stream close), reset still
  precedes next admission so the invariant holds.
- **TTFT: chunked batched prefill shipped; single-box GPU-prefill
  split RULED OUT.** The stage IRs are fully dynamic in T (no
  re-export), so prefill runs in 256-token chain passes
  (`probe_batched_prefill.py`: 54s → 12.8s @313 tokens, 4.2x;
  engine-measured 290-token prompt ~21s prefill ≈ 13.8 tok/s vs ~5.3
  at T=1 — chunk-boundary and copy overhead eat part of the probe
  number). Batched-vs-sequential decode parity 6/8 with a whitespace
  near-tie flip at token 6 — f16 accumulation-order noise, the same
  regime every batched-prefill server (incl. whole-model GenAI)
  operates in. The heterogeneous GPU-prefill design is dead on this
  node class: a ~32 GB node cannot hold two ~18 GB int4 weight copies
  (GPU + CPU compiled); the cross-node variant (GPU prefill node
  streaming states to a CPU decode node) needs the state-transfer FFI
  (get/set variable states) priced in.
- **Same-command shard UX: `cascadia shard` dispatch arm SHIPPED.**
  `cascadia shard --model <int4-ov dir or HF id> --output-dir <out>
  --num-stages N` detects `model_type qwen3_5_moe` config-first and
  dispatches to the IR-surgery exporter (embedded in the binary beside
  the other exporters); --quantization is ignored (stages inherit the
  official int4 IR), --layer-split/--stage rejected. The qwen3_5_moe
  path runs on openvino+numpy alone (torch made lazy in
  export_shards.py — the target env is an inference node without
  torch). Validated E2E: dispatch → stage saves (40 states each, 7
  orphan rewires) → one-token validation EXPORT_VALIDATE_OK with
  numbers identical to the hand export (rel 2.8e-2, top1 match, top5
  5/5). Post-export hint is arch-aware
  (`cascadia run <dir> --engine qwen36-moe`).
- **Exporter polish, all shipped.** (1) The v1 dummy-input wart is
  gone: mid stages rewire their mask/position ShapeOf chains off
  `inputs_embeds` onto `stage_hidden` (4 consumers on the 2-stage
  cut), so stage1's inputs are just
  stage_hidden/attention_mask/position_ids/beam_idx. (2) The last
  stage slices logits to the final position ([1,1,vocab]); manifest
  sets `last_logits_only` and the engine skips its row slicing —
  batched prefill stops materializing [1,T,vocab] (~1 MB/token).
  Pre-slice shard trees still work (flag defaults false). (3)
  `--validate` adds an 8-token greedy chain-vs-full check (accept
  ≥6/8; measured 8/8). Re-validated E2E via `cascadia shard`:
  single-step numbers identical (rel 2.8e-2, top1, top5 5/5),
  MULTI_TOKEN_PARITY 8/8, and the qwen36_parity golden test passes
  against the polished shards (engine integration, token-identical to
  the blessed golden).

### Cross-node feasibility spikes

- **GPU→CPU state handoff — mechanics PASS, numerics quantified**
  (`probe_state_handoff.py`, stage0, single node). All 40
  VariableStates (DeltaNet ssm/conv + KV) are plain f32 tensors:
  export from a GPU request 34.1 MB @32-token ctx in 46 ms, import
  into a CPU request 23 ms; dual residency (stage0 compiled CPU+GPU
  simultaneously) fits a ~32 GB node, GPU compile 16 s. Teacher-forced
  hidden tracking: handoff decode tracks CPU-pure at rel 4.0e-2 on the
  first step (no-state control: 4.3e+1 — three orders apart), drifting
  to ~1.9e-1 by step 8 (mixed GPU/CPU f16 regimes compound). Verdict:
  the heterogeneous design's state handoff is REAL; decode after GPU
  prefill will be coherent but not token-parity with CPU-pure — the
  same per-regime rule already documented for GPU serving. Lesson
  encoded in the probe: a hidden-argmax pseudo-token comparator is an
  attractor (passes its own negative control 7/8) — compare hidden
  VECTORS with a no-state control instead.
- **Cross-node wire latency** — 8 KB ([1,1,2048] f32 hidden state)
  TCP round-trip between two test nodes, TCP_NODELAY: median
  **14.55 ms**, p95 16.4 ms — over a relayed VPN path, which was the
  only route between them (their local subnets did not route to each
  other; verified with a live listener). Implications: (a) 2-stage
  LAYER pipelining pays ~1 RTT/token ≈ 7% of the 200 ms decode budget
  — viable; (b) EXPERT parallelism (§4.3) is DEAD on a relay-linked
  mesh: ~40 layer round-trips/token × 14.5 ms ≈ 580 ms/token network
  floor (<2 tok/s) vs the §4.3 LAN assumption of 0.2–0.5 ms RTTs.
  Revisits only on a routable LAN or a direct (non-relayed) path.
- **Full-chain GPU-prefill handoff — 16/16 token-perfect**
  (`probe_handoff_full_chain.py`). Both stages prefilled on GPU
  (sequentially compiled/freed to fit a ~32 GB node), states exported
  and imported into CPU-compiled stages, 16 greedy tokens decoded:
  token-identical to the blessed CPU-pure golden on the parity prompt.
  The state-handoff spike's hidden-level drift does not flip greedy
  near-ties here.

**Cross-node pre-engine gates: ALL GREEN.** These spikes established
the go/no-go picture: a layer pipeline pays ~1 RTT/token (viable even
over a relayed path at ~7% of the decode budget); expert parallelism
stays dead on relay-linked meshes. The pipeline build shipped — see
"Pipeline mode (multi-node)" below. Still open: per-node
GPU-prefill→CPU-decode handoff inside a rank (state round-trip proven
by the spikes above, but the state get/set FFI in the shim is not
built).

### Pipeline mode (multi-node)

The staged engine runs as an N-rank layer pipeline
(`--rank i --total N`). Rank 0 holds embeddings + stage 0 + tokenizer
and drives decode; middle ranks relay (run their stage, pass the span
downstream, return the token back upstream); the last rank holds the
logits head and answers each FORWARD with the argmax token. Control
frames (HELLO handshake, RESET/RESET_ACK) chain through middle ranks —
one ACK at rank 0 means the whole chain is at position 0. Frames are
lockstep on one transport session per hop (12-byte header
`[kind][epoch][pos]` + body); stale-epoch frames are dropped, and
peer loss mid-task fails the task loud — recovery is restarting the
affected workers, never partial-state serving. The full frame format
and invariants are documented in the `qwen36.rs` module doc.

Launch **highest rank first**, then descending — each rank's listener
must be up before its upstream dials:

```bash
# Node B (rank 1 of 2: stage 1 + logits head):
cascadia worker --rank 1 --total 2 --engine qwen36-moe \
  --model /path/to/qwen36-shards-2stage --device CPU --listen :9100

# Node A (rank 0: embeddings + stage 0 + tokenizer + API):
cascadia worker --rank 0 --total 2 --engine qwen36-moe \
  --model /path/to/qwen36-shards-2stage --device CPU \
  --next <node-b-host>:9100 --api :8000
```

Ranks ≥ 1 need only `manifest.json` and their own `stage<i>/`
directory from the shard tree (`generation_config.json`, embeddings,
and the tokenizer are read on rank 0 only) — no need to copy the full
tree to every node.

**Acceptance gates** (harnesses: `tools/qwen36_surgery/m4_gate_serving.py`
for gates 1–2 + the prompt set, `m4_gate_robustness.py` for gate 3;
gate 4 is read from the rank-0 wire-histogram log line):

1. **Token agreement** — 64-token greedy through the pipeline must be
   char-identical to the single-box engine on the same shard tree. On
   divergence, apply the near-tie coherence judgment (§5 "Prototype
   and engine validation"): report the divergence index plus both
   continuations rather than raw equality.
2. **Decode throughput** — ≥ 4 tok/s short-context (completion tokens
   over decode wall time, cross-checked against the engine's `tok_s`
   log line).
3. **Robustness rows** — cancel mid-decode then a clean follow-up
   task; SSE client disconnect mid-decode then clean follow-up; three
   sequential tasks with identical outputs (no state bleed); kill a
   downstream rank mid-decode → rank 0 fails the task loud and the
   server stays healthy; kill rank 0 mid-decode → downstream survives
   with rate-limited warnings; restart all ranks → next task clean.
4. **Wire histogram** — per-decode-frame RTT minus the peer-reported
   infer time; **p95 > 40 ms blocks**. Measure over a long window:
   relayed paths are bursty and a short window under-samples the tail.

Measured on the validation pair (Intel Lunar Lake nodes, CPU decode,
2-stage): parity exact, 4.1–4.8 tok/s short-context vs a 4.7–8.8
single-box envelope, wire p50 ~21 ms / p95 ~24 ms. A 3-stage chain
with a real middle rank measured per-hop-additive wire (~2× single
hop) and parity exact — 2-node remains the perf-reference config.

## 6. Out of scope (v1, regardless of strategy)

Vision tower; MTP/speculative decoding; batching/multi-stream (priced
spec change, not a default); prefix caching on linear layers;
mid-sequence migration/failover; dense-MoE staged execution (measured
dead — §3); Qwen3.5-235B-A22B and other family members (unpriced
future direction — note its ~11 GB/token active set caps single-stream
at ~7 tok/s on this node class at ANY node count, and its checkpoint
exceeds typical export-host RAM; capacity ≠ speed).
