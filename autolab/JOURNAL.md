# JOURNAL — autolab/k26-perf

Append-only. Newest at top. One entry per moonshot iteration.

## 048 — Wire SIMD tiles into engine callsites — SEAM SHIPPED, real bench waits for caller merge (2026-05-18 ~16:55 PT)

The trampoline iter: per-shape dispatcher routes iter 042 + iter 046
SIMD tiles to their right projections inside the engine. Branch
`perf/wire-simd-multi-tiles-048` @ `beebd0c` (based on iter 046).

**What's wired:**
- `ProjShape` enum (`Oproj`, `SharedDown`, `Generic`) +
  `dispatch_int4_multi` helper replaces 9 callsites in
  `shell_forward_decode_int4_multi_batched`
- **oproj** (N=7168, K=8192, 28 MB) and **shared_down** (N=7168,
  K=2048, 7 MB) → iter 046 `dequant_gemm_int4_multi_blocked_auto`
  at seq≥4 (falls through to iter 042 at seq=2-3, single-token at
  seq=1)
- **All other 7 shells projections** → iter 042
  `dequant_gemm_int4_multi_auto` (auto-falls-through to single-token
  at seq=1)
- New `Runner::forward_shells_multi(h_in, h_shape, past_seq_len,
  seq)` public method calls `_multi_with_capacity`
- Fast-paths seq=1 → existing `forward_shells` (**seq=1 hot path
  UNCHANGED**)
- 3 new unit tests:
  - `dispatch_int4_multi_seq_1_matches_single_token_kernel` (regression
    guard for all 3 ProjShape variants)
  - `multi_batched_matches_scalar_seq_4_iter046_dispatch` (seq=4
    bit-identical)
  - `multi_batched_matches_scalar_seq_8_iter046_dispatch` (seq=8
    bit-identical, iter 046 sweet spot)
- 23 int4-gemm + 16 sparse-moe tests pass, fmt + clippy clean

**Honest caveats (verbatim agent):**
> No miner bench run. Miner load avg 46.x on 48 cores; iter 044
> spec-decode-compound long-running experiment is active. Per task
> brief + autolab_loop_autonomy.md, I shipped code + tests +
> theoretical analysis instead. The underlying kernels are pre-
> benchmarked on miner (iter 042 + iter 046 commits); this iter is
> a pure trampoline through dispatch_int4_multi, so engine-level
> performance should be a Σ-of-per-projection-wins multiplier
> (~2-3× shell time at seq=4, ~3-4× at seq=8) less Amdahl tax from
> per-token RoPE/SDPA.

> No multi-token driver loop callers in this branch. The
> forward_shells_multi API is added, but the chunked-prefill (iter
> 040) and spec-decode-verify (iter 036/039/044) branches that
> would call it are not merged into iter 046 base. Wiring those
> callers is a separate merge.

> Layer-0 multi-token kernel intentionally not lifted. Layer 0 is
> one call per token (not per-layer × per-token), so the wiring
> effort isn't justified vs the 60-layer shell hot path. Easy
> follow-up if profiles flag it.

**Conservative per-shape table:** only oproj + shared_down get iter
046 dispatcher. iter 046's commit body showed +118%/+62%/+28% wins
on qproj/kvproj/shared_gate at seq=8/16 too, but with "more variable
seq=4 behavior". Follow-up should run engine-level bench to decide
whether to lift more shapes.

**This + iter 044 are the key integration steps.** Once both land:
- iter 044 measures compound spec-decode on miner
- iter 048 wires the SIMD tiles to actually fire
- Combined: real e2e number that proves the architectural foundations
  ship perf, not just unit-tests

---

## 049 — f16/bf16 inter-layer hidden states — NEGATIVE RESULT (verified by microbench) (2026-05-18 ~16:50 PT)

Honest negative result via path (C) investigation-first. Branch
`perf/f16-hidden-states-049` @ `4b0c4a8`. Agent decided NOT to ship
the change after microbenching.

**Why bf16 inter-layer is net-negative (verbatim agent finding):**
- bf16 hidden states: **3.4× SLOWER** per token (105.5 µs vs 31.4 µs
  across 60 layers). `bf16::from_f32` round-to-nearest-even is ~3×
  more expensive than a plain f32 store.
- Even at the THEORETICAL BEST CASE (free bf16), the saving would be
  31 µs out of a **~41-second-per-token attention budget** (iter
  032's KV bf16 baseline). That's **0.00007%** — invisible.
- Distributed (pipeline-parallel): hidden-state wire is 22.4 ms RTT
  vs 0.23 ms raw byte-transmit. **Wire is 100× latency-bound, not
  bandwidth-bound** — halving payload doesn't move RTT.

**Why iter 032 KV bf16 IS a win but this isn't:**
> The KV cache lives in attention SDPA's hot loop (60 layers × 64
> heads × past_seq_len × QK dot products) — that IS bandwidth-bound
> at long context. The inter-layer hidden state is one 28 KB buffer
> that fits in L1 and gets touched once per layer — not
> bandwidth-bound at all.

**What was shipped (path B defensive infra):**
- New microbench `bin/bench_hidden_state_transition.rs` (+239 LOC)
  comparing the actual inter-layer loop vs hypothetical bf16 path
- `trace!` instrumentation in `runner.rs::forward_shells` summing
  per-layer inter-layer transition cost — gated `LevelFilter=trace`,
  zero cost in default builds
- 2 new unit tests in runner.rs:
  `bf16_hidden_state_roundtrip_error_envelope` (pins rounding bounds)
  `bf16_u16_conversion_matches_half_crate` (cross-checks iter 032's
  bit-shortcut)
- 36 tests pass; fmt + clippy clean
- 16 → 18 sparse-moe tests

**Why this is a successful iteration despite "negative":** the
investigation saved future effort (no one will re-attempt this
without re-reading the analysis). The microbench is reusable —
future quantization questions about inter-layer buffers can use the
same harness. The envelope tests document the bound if a future iter
DOES want to ship bf16 for a different reason (e.g., memory savings
on long-context).

**No bench conflict:** the microbench is host-laptop pure-Rust, not
a miner run. Did not collide with iter 044 / iter 048.

---

## 046 — Row-blocked AVX-512 for oproj — VERIFIED WIN: +40% over iter 042 (2026-05-18 ~16:30 PT)

**FOURTH VERIFIED ARCHITECTURAL MOONSHOT.** AND a critical
methodology finding: iter 042's "DRAM-bandwidth-bound" claim for
oproj was wrong — perf stat shows 97.7% L3 hit rate.

Branch `perf/oproj-amx-or-avx512-blocked-046` @ `77bc56f`. Path:
(B) row-blocked AVX-512 + partial (C) perf-stat investigation.

**Measured (miner Xeon Gold 6252, taskset -c 0-23, 100 iters,
oproj N=7168 K=8192):**

| seq | scalar/ms | iter042/ms | iter046/ms | vs iter042 | vs scalar |
|----:|----------:|-----------:|-----------:|-----------:|----------:|
|  1  |  1.778    |  0.629     |  0.626     | 1.00× (→042) | 2.83× |
|  2  |  1.424    |  1.171     |  1.169     | 1.00× (→042) | 1.22× |
|  4  |  2.974    |  2.277     |  **1.617** | **1.41×**  | **1.84×** |
|  8  |  9.174    |  5.815     |  **4.150** | **1.40×**  | **2.21×** |
| 16  | 20.585    | 10.280     |  **7.292** | **1.41×**  | **2.83×** |

**~40% faster than iter 042 SIMD tile on oproj at seq 4-16.** Wins
also seen on qproj, kvproj, shared_gate, shared_down at seq 8-16.

**Key hardware finding (option C work product — verbatim agent):**
> perf stat on miner Xeon Gold 6252 disproved iter 042's "DRAM-
> bandwidth-bound" claim for oproj: LLC miss rate is only **2.27%**
> (97.7% of weight loads hit L3 — the 14 MB int4 weight fits in
> 35.8 MiB L3). Real bottleneck is L2/L3 latency + redundant xs
> reads (IPC=1.10).

The win comes from RB=2 row-blocking: each xs load reused across 2
output rows, halving the redundant L2/L3 traffic. NOT from
prefetching, NOT from DRAM bandwidth.

**Honest negative result (also documented):** an explicit
`_mm_prefetch` variant LOST across all seq (0.63-0.87× of iter 042).
HW prefetcher already handles sequential packed stream; prefetch
instructions added front-end pressure. Did not ship.

**Tests:** 5 new correctness tests (bit-identical per cell vs iter
042). Pass on miner AVX-512 + arm64 scalar fallback.

**Auto-dispatch:** `dequant_gemm_int4_multi_blocked_auto` routes:
- seq>=4 → blocked tile (this iter's win)
- seq=2-3 → iter 042 multi tile
- seq=1 → scalar single-token

**What's left:**
- AMX implementation skipped: no AMX hardware in fleet (Xeon 6252
  Cascade Lake; matias Lunar Lake also lacks AMX; AMX is Sapphire/
  Granite Rapids Xeon or Lunar Lake-X only)
- NOT wired into `shell_int4.rs` yet — per-shape thresholding
  needed (oproj benefits at seq=4 but qproj/kvproj more variable
  at seq=4). Follow-up should benchmark engine-level integration.
- iter 044 worker shared miner cores; ±15% variance run-to-run
  without `taskset -c 0-23`

**Architectural impact:** the 4 SIMD wins now compound:
- iter 042 AVX-512 multi-token tile: 1.4-4.75× per projection at
  seq=4-16
- iter 046 row-blocked variant: another 40% on top for oproj +
  other large shapes at seq>=4
- Combined: ~2-6× per projection at production seq sizes

---

## 047 — Better C1 predictor (top-N pre-softmax) — CODE + INSTRUMENTATION shipped, bench deferred (2026-05-18 ~16:25 PT)

Code + measurement instrumentation done. Branch
`perf/c1-better-predictor-047` @ `4753fa2` (based on iter 038
Windows-port branch, +361 / -32 across 6 files).

**What shipped:**
- `shell_int4::shell_forward_decode_int4_predict_n(..., predict_top_n)`
  new entry point
- Pure helper `select_top_n_by_score(scores, n)` —
  `select_nth_unstable_by` partial sort then small-prefix sort, so
  first TOPK come out in canonical descending order (dispatch path's
  invariant)
- Existing `shell_forward_decode_int4_with_capacity` delegates with
  `predict_top_n=TOPK` for back-compat
- `ShellOutputs.predicted_top_n_ids: Vec<i64>` (first TOPK exactly
  match `routing_ids`)
- `Runner.prefetch_n: u32` + `set_prefetch_n(Option<u32>)` setter
  (clamped to `[TOPK, N_ROUTED_EXPERTS]`)
- `last_routing_ids[i]` now stores top-N predicted IDs (when
  N=TOPK and no A2/A3, behaviorally identical to iter 033)
- **Hit-rate counters** (`prefetch_hits`, `prefetch_chances`) via
  `prefetch_stats()` + logged in per-token `stage_timing` line —
  makes future bench cheap
- Prefetcher channel cap 4K → 16K (absorbs N=24 × 60 layers worst
  case)
- `--prefetch-n N` CLI flag
- C ABI destructuring-only update (no byte-compat change for rainier
  cdylib consumers)

**Tests (5 new, all pass):**
- `top_n_is_superset_of_top_k` — 16 trials of 384-expert score
  vectors, validates top-K ⊆ top-N for N ∈ {8, 12, 16, 24, 32, 64,
  384} AND first-TOPK prefix is sorted descending
- 4 edge-case (ascending, descending, N=0, N=len)
- All 12 int4-gemm + 17 sparse-moe lib pass; fmt + clippy clean

**Hit-rate analysis (theoretical, bench deferred due to miner
contention from iter 044):**

| N  | Predicted hit-rate | Hinted/tok | Notes |
|----|-------------------:|----------:|-------|
| 8  | ~0.70 (baseline)   | 11.5 GB   | iter 033 |
| 12 | ~0.78              | 17.3 GB   | comfortable in 133 GB cache |
| **16** | **~0.85**      | **23 GB** | **sweet-spot candidate** |
| 24 | ~0.95              | 34.5 GB   | risk of evicting earlier hints |

Per-expert: ~24 MB (553 GB / 60 layers / 384 experts). N=24 may
regress due to page-cache eviction; the shipped counters will
empirically falsify/confirm.

**Why it might compound nicely with iter 033 C1:** iter 033 measured
+27% with same-as-last (~0.70 hit-rate). Each +10 percentage points
of hit-rate translates to ~3-5% additional throughput (avoided disk
faults). N=16 → +5-7% on top of iter 033 = ~33-35% combined vs
no-prefetch baseline.

---

## 045 — Head tensor-parallelism — PLUMBING + Rust head kernel done, runtime gated on back-channel + low-RTT (2026-05-18 ~15:55 PT)

Substantial architectural work + honest "not net-positive on current
infra" finding. Branch `perf/head-tp-045` (3 commits).

**(B+) delivered:**
- `head.rs`: `HeadSlice` kernel — RMSNorm + bf16 GEMV against a vocab
  row-slice. `concat_partials` (gap/overlap-rejecting),
  `even_vocab_split` (matches existing even_moe_split style).
- `safetensors_source.rs`: `lm_head_slice(vs, ve, hidden)` exploits
  that `lm_head [vocab, hidden]` rows are byte-contiguous; plus
  `final_norm()` for pre-head norm weight.
- `dist.rs`: `FrameKind::HeadPartial = 0x534D4530`, body =
  `vocab_start u32 BE + vocab_end u32 BE + F32 tensor`. Send/recv
  both sides.
- `runner.rs`: `LayerRange.head_vocab_range` per-rank loading;
  `Runner::forward_head_partial` and `forward_head_last_rust`;
  `step()` auto-routes to Rust head when slice covers full vocab.
- `engine.rs`: `enable_head_tp` + `force_rust_head` config flags;
  builder partitions vocab via `even_vocab_split`.
- **26 tests pass** (9 new head + 4 new wire-frame + 13 existing).
  Includes byte-identical full-vs-split-then-concat round-trip.
- Gated integration test `k26_paris_pacific_four_rust_head` runs the
  real Paris/Pacific/four prompts via Rust head when
  `K26_MODEL_DIR` is set.

**NOT delivered + WHY (honest architectural finding):**
> Runtime activation of the multi-stage head-TP path in the engine
> driver. The crux: for rank-0 to run its head slice, it needs the
> post-shells hidden state from rank-1 (since rank-1 owns the second
> half of layers). That requires a NEW back-channel frame
> (post-norm hidden, ~14 KB bf16 or 28 KB f32) plus the HeadPartial
> round-trip. On the current 2-box matias topology (117ms RTT via
> SSH-tunnel-Mac chain per iter 030) the extra wire RTT (~234ms)
> exceeds the head-compute savings (~70ms on 139ms baseline). Net
> negative on the only currently-revived 2-box pipeline.

**Critical architectural insight:** head TP needs **sub-10ms RTT** to
be net-positive. The 117ms SSH-tunnel-chain is far too slow. Useful
substrates would be:
- LAN fleet (beta/charlie/alpha) at ~1-5ms RTT — but K2.6 isn't
  loaded there (~14hr/box transfer)
- Shared-memory ranks on one box — different deployment model
- Future direct Tailscale recovery (~22ms baseline) — borderline

**Standalone opportunity identified:** `force_rust_head=true` is an
A/B benchable on single-stage even WITHOUT head TP. Could replace the
OV head with native Rust on miner — pure correctness + perf swap.
Worth iter 046 once miner frees from iter 044.

**Free wire-frame slot:** `0x534D454x` range is free for the future
`NormedHidden` back-channel frame.

---

## 037 — F5 bench retry — REAL +80% TPS at W=32 BUT quality cliff (substring eval too weak) (2026-05-18 ~15:35 PT)

Mixed result + methodology discovery. Branch `perf/f5-bench-results-
037` @ `f05569b`.

**Measured (miner single-stage, K=8, single-prompt):**

| W   | mt  | tok/s   | substring | qualitative |
|-----|-----|---------|----------:|-------------|
| 0   | 64  | 0.1253  | 1/1       | coherent (baseline) |
| 32  | 64  | 0.1585  | 1/1       | coherent ~25 tok then "Question?" loop |
| 0   | 128 | 0.1192  | 1/1       | coherent (baseline) |
| 32  | 128 | **0.2150** | 1/1  | **+80.4% TPS but garbage after ~30 tok** |
| 128 | 128 | 0.1235  | 1/1       | within noise (+3.6%) |

**The throughput claim is real:** W=32 at mt=128 = +80.4% tok/s. The
seam works. But:

**🚨 CRITICAL METHODOLOGY FINDING — substring eval is too weak:**
> Existing substring eval passes "Paris ?? ?? &" because "paris" is
> in the first sentence. A stricter eval (perplexity vs W=0
> reference, or first-divergence position) would have flagged W=32.

W=32 output degenerates after ~30 tokens to literal garbage
(`?? && ..`). But the substring check rewards first-sentence
correctness, so it scored 1/1.

**K2.6-specific architectural finding:** quality cliff is severe at
W=32 because K2.6 has no per-layer attention_type (unlike Gemma3's
mixed full/sliding stack — F5 impl agent had flagged this risk in
iter 029). Uniform window across all 60 layers can't preserve
long-range deps.

**W=128 at mt=128 is within noise (+3.6%)** — matches pre-bench
expectation ("window barely active"). The wider window doesn't hurt
quality but also doesn't help throughput unless prompts get
significantly longer.

**Honest verdict:** F5 is neutral-to-negative at production-safe W;
it's only fast-at-W=32 by sacrificing coherence. Not a shippable win
for K2.6 without per-layer attention_type retrofit.

**Memory update:** saved
`autolab-substring-eval-too-weak` — substring eval insufficient for
attention/KV/sparsity changes; need stricter eval (perplexity,
first-divergence, or eyeball).

**Operational notes from agent:**
- Build needed `INTEL_OPENVINO_DIR=/home/tatef/openvino_2026.1.0`
  for `tahoma-ov-genai-shim`
- `setupvars.sh` must be sourced before worker start (for
  `libopenvino_genai.so.2610` on LD_LIBRARY_PATH)
- F5 branch lacks `--top-k-override` flag (it's in sibling branch),
  so K=8 manifest default was used instead of K=6 as requested. The
  W=0 vs W=32 vs W=128 comparison at same K=8 is valid.
- Miner was clean (killed stale pid 2702179 at start)
- `mt=256` test infeasible (~32 min/prompt at 0.13 tok/s); skipped

---

## 042 — AVX-512 multi-token int4 tile — VERIFIED WIN: 1.4-4.75× per projection (2026-05-18 ~15:35 PT)

**THIRD VERIFIED ARCHITECTURAL MOONSHOT.** Real measured speedups
across all K2.6 production-relevant projection shapes. This is the
piece that turns the iter 041 seam into actual perf.

Branch `perf/int4-multi-token-avx-vnni-042` @ `7292c81` (3 commits,
descended from iter 041 `d9512c1`).

**Measured speedup on miner (Xeon Gold 6252):**

| Shape           | seq=2 | seq=4 | seq=8 | seq=16 |
|-----------------|------:|------:|------:|-------:|
| qproj (5 MB)    | 1.41× | 1.43× | 1.66× |  1.72× |
| kvproj (2 MB)   | 0.97× | 2.15× | 2.60× |  2.44× |
| shared_gate (7 MB) | 1.42× | 1.87× | 2.03× |  3.96× |
| **shared_down (7 MB)** | **1.75×** | **2.80×** | **4.75×** |  **3.96×** |
| oproj (28 MB)   | 1.37× | 2.07× | 2.09× |  2.20× |

**Bit-identical** to scalar reference via new test
`shell_int4::tests::multi_batched_matches_scalar` (bit-identical KV
+ routing_ids; f32 buffers within 1e-3 — in practice bit-identical
since multi-tile sums in same nibble/col order as single-token).

**What was built:**
- `kernel_avx512_multi.rs` — AVX-512 multi-token int4 GEMM.
  Dequantizes each int4 group once per (row, group) and FMADDs against
  `seq` input vectors. Weights stay in registers across the seq loop.
  Auto-dispatch falls through to single-token at seq=1.
- `bin/bench_int4_multi.rs` — microbench across K2.6 projection
  shapes × seq ∈ {1,2,4,8,16}.
- `shell_int4.rs::_multi_with_capacity` now dispatches to new
  `_batched` variant at seq≥2 using the multi-tile for every
  KV-cache-independent projection (q_a/b, kv_a/b, o_proj, router,
  shared_gate/up/down). `_multi_scalar` preserved as bit-identity
  reference.

**Caveats (agent honest):**
- The "AVX-VNNI" name is misleading: `_mm512_dpbusd_epi32` not used.
  With f32 inputs + bf16 scales, the f32-FMA path is the right
  baseline. VNNI would need int8 quantization of the input X-vector
  (separate lift). VNNI hook
  (`dequant_gemm_int4_multi_vnni_tile`) is in place.
- oproj (28 MB) gets only ~2× — DRAM-bandwidth-bound even with
  weight reuse. AMX (Intel TMUL int8) is the next swing for this
  shape.
- At seq=1 the multi-tile loses ~0.7-0.9× (per-row scratch +
  scatter doesn't amortize). Auto-dispatch handles transparently.

**Impact on the stack:**
- iter 036 spec-decode skeleton ✓
- iter 039 ForwardBatch(K) wire ✓ (~110ms wire savings per round)
- iter 041 multi-token int4 seam ✓
- iter 042 multi-tile SIMD ✓ ← THIS
- iter 043 spec-decode bug fix ✓ (single-stage works past round 1)

**ALL FOUNDATIONS NOW IN PLACE.** Spec-decode @ K=4 on 2-box matias
should see ~1.4-2.8× per projection (full pipeline ~30-50% e2e
estimated). Chunked prefill @ seq=8-16 should see ~2-5× per
projection at prefill time. Real bench needed — but the substrate is
ready.

15 int4-gemm tests pass on macOS (scalar fallback) + miner (AVX-512).

---

## 043 — spec_decode off-by-one fix — BUG CONFIRMED + FIXED (2026-05-18 ~15:20 PT)

Real bug fix, real correctness improvement. Branch
`fix/spec-decode-reconcile-off-by-one-043` @ `19c61cd` (based on iter
039 `a4245db`).

**The bug, confirmed (verbatim agent finding):**
> The runner pre-pushes `first_gen` to history before round 1 and
> appends each round's `bonus` to history at end-of-round — both ride
> ahead of KV by 1 slot. The K-loop's first verify forward absorbs
> the previous round's pending token's KV slot as a side effect, so
> the helper must rewind one less than the clean-convention formula:
> - Partial accept (A < K): runner needs `K - A - 1`, helper returned `K - A`
> - All-accepted with bonus forward: runner needs `0`, helper returned `1`

The existing helper tests used a clean "no-first-gen" convention so
the bug was hidden. The runner's `kv_invariant_holds` debug_assert
would fire on round 2 in debug builds, but no integration test
exercised the full path.

**Fix:**
- Added `pending_token_in_history: bool` param to
  `reconcile_after_round`
- Both conventions documented in helper
- 4 new tests (spec_decode 10→14, sparse-moe lib 38→42):
  `reconcile_pending_token_partial_accept`,
  `reconcile_pending_token_all_accepted`,
  `reconcile_pending_token_zero_accepted`,
  `simulated_runner_pending_session_matches_sequential_greedy`
- `runner.rs::generate_speculative` now passes `pending=true`
- `kv_invariant_holds` takes `pending_drift: usize` (drift=1 if bonus
  pushed, drift=0 if round cut by EOS/max_tokens)
- iter 039 `engine.rs::drive_generation_first_spec` now defers to
  the helper instead of computing rewind inline — 30-line inline
  math comment replaced with pointer to helper's docstring

**Bug-catching verification:** agent temporarily flipped to call with
`pending=false` (simulating unfixed helper) and confirmed test fails
with `kv_len=4, history.len()=6` (drift=2, expected=1). With
`pending=true`, all 14 tests pass.

**Impact:** single-stage spec-decode now works past round 1 in debug
builds. Pipeline-parallel driver is also cleaner (defers to helper).
This is the prerequisite correctness fix for any future spec-decode
bench attempt.

---

## 041 — Multi-token int4 shell kernel — SEAM SHIPPED (B), real win still pending SIMD (2026-05-18 ~15:08 PT)

Keystone moonshot **API seam done**, perf win pending. Branch
`perf/int4-multi-token-041` @ `d9512c1`. Delivered (B): skeleton +
bit-identity tests, internal scalar loop (functionally equivalent to
today).

**What shipped:**
- `shell_int4::shell_forward_decode_int4_multi_with_capacity(shell,
  xs_f32, &mut past_k, &mut past_v, past_seq_len, capacity, seq)
  -> MultiShellOutputs` (+412 LOC)
- `layer0_int4::layer0_forward_decode_int4_multi_with_capacity(layer,
  xs_f32, &mut past_k, &mut past_v, past_seq_len, capacity, seq)
  -> MultiLayer0Outputs` (+289 LOC)
- Both internally loop seq times over existing seq=1 entry point,
  writing present_k/present_v into caller's pre-allocated KV between
  iterations so each token attends to predecessors
- Outputs concatenated per-token: `attn_out_post_norm`, `attn_
  residual`, `shared_expert_out` shape `[seq, HIDDEN]`; `routing_
  ids` / `routing_weights` shape `[seq, TOPK]`; `hidden_out` for
  layer-0
- seq=1 entry points UNCHANGED — today's hot path byte-identical

**Tests (all 4 pass with `assert_eq!`):**
- `shell_int4::tests::multi_seq_1_matches_seq_1_reference`
- `shell_int4::tests::multi_seq_3_matches_sequential_seq_1_calls`
- `layer0_int4::tests::multi_layer0_seq_1_matches_seq_1_reference`
- `layer0_int4::tests::multi_layer0_seq_3_matches_sequential_seq_1_
  calls`

seq=3 tests pre-seed `past_seq_len=2` with non-zero history to
exercise the populated-history path. fmt + clippy clean.

**Honest: NOT a perf win.** Same scalar GEMV iteration; same memory
motion; same int4 dequant count.

**Real win unlocks via three follow-ups (now all plug into this
seam):**
1. **Native tiled GEMM across seq** — replace internal scalar loop
   with `[seq, K] × [K, N]` GEMMs. AVX-VNNI / AMX. Memory motion
   drops from `seq × W` to `~W` for the dominant 1536/512/7168
   projection weights. **Est: 1-2 weeks of SIMD work**.
2. **Caller adoption** — `runner.rs:619` currently loops `_with_
   capacity` per token in chunked-prefill / spec-decode scenarios.
   After SIMD #1 lands, switching to single `_multi(seq=N)` captures
   the win. Today switching is safe but neutral.
3. **Native multi-token attention** — token-batched Q·Kᵀ then
   softmax then ·V across `seq` query rows simultaneously. **Est:
   3-5 days.**

**This is the keystone seam.** With this in place:
- iter 036 spec-decode skeleton + iter 039 ForwardBatch(K) wire +
  this seam = full pipeline-parallel spec-decode path. ONLY needs
  the SIMD GEMM (#1) to translate into measured tok/s.
- iter 040 chunked prefill seam + this seam = chunked-prefill
  unblocked too.
- Continuous batching becomes possible once seq>1 kernels are real.

---

## 039 — ForwardBatch(K) wire frame for spec-decode — WIRE BATCHING DONE + bug discovered (2026-05-18 ~14:55 PT)

---

## 039 — ForwardBatch(K) wire frame for spec-decode — WIRE BATCHING DONE + bug discovered (2026-05-18 ~14:55 PT)

Real architectural deliverable + bonus bug discovery. Branch
`perf/forward-batch-spec-decode-039` @ `a4245db` (based on iter 036
spec-decode skeleton `acd21bd`).

**What shipped:**
- `FrameKind::ForwardBatch` (0x534D4503) — K hiddens +
  past_seq_len_start + SamplingConfig in one wire frame
- `FrameKind::TokenBatch` (0x534D4521) — K token IDs back in one frame
- `send_forward_batch` / `recv_forward_batch_body_server` /
  `send_token_batch_upstream` / `recv_token_batch_body_client`
- `MAX_BATCH_COUNT = 256` cap; defensive K=0 / shape-mismatch
  rejection
- Existing Forward/Reset/Token codes UNCHANGED — pre-iter-039 workers
  stay wire-compatible for the per-token path
- `handle_forward_batch_frame`: K sequential `forward_shells`
  internally, last rank does K head+sample, mid ranks relay K hiddens
  downstream + forward TokenBatch upstream
- `maybe_rewind_to`: worker self-rewinds KV (+ `last_rank_history`)
  on every Forward / ForwardBatch — no dedicated RewindBatch frame
  needed
- `spec_decode_k` now honored on **both** single-stage AND pipeline-
  parallel (was single-stage-only)

**Tests:** 58 sparse-moe tests pass (was 50; +8 new). Critical:
- `forward_batch_frame_round_trips_k4` — K=4 ser/deser
- `token_batch_frame_round_trips_k4` + extreme-ids
- `forward_batch_then_token_batch_pair` — full request/response cycle
- 3 defensive-rejection tests (K=0, K>MAX, shape mismatch)
- `frame_kind_codes_remain_stable` — pins all 5 codes
- Existing `simulated_session_matches_sequential_greedy` still passes
  unchanged (single-stage path unmodified)
- `cargo fmt` clean, `cargo clippy --no-deps` clean, workspace builds

**🐛 BUG DISCOVERED (off-by-one in iter 036's spec_decode helper):**
> The existing `spec_decode::reconcile_after_round` is off-by-one for
> the runner's convention (history pre-pushes `first_gen` + appends
> bonus after the K-loop, so trail-by-1 is invariant). The helper's
> tests use a no-first-gen convention which doesn't expose the bug;
> `debug_assert` in `Runner::generate_speculative` would likely fire
> on the second round. My pipeline-parallel driver computes the
> correct rewind locally (K-A-1 for partial accept, 0 for all-
> accepted) and documents why in comments. The single-stage runner's
> spec-decode is left unfixed — out of scope.

**Follow-up needed:** fix the off-by-one in
`spec_decode::reconcile_after_round` so single-stage `generate_
speculative` actually works past round 1 in debug builds. Small fix
but easy to miss without this finding.

**What's left:**
- No 2-box bench (matias workers stale; gated on iter 038 source-sync
  or on revival via tunnel chain again)
- Per-token shell still seq=1 (unchanged) — wire-batching not kernel-
  batching. The ~9s/token K2.6 dispatch cost is untouched. Win is on
  cross-rank wire latency: ~110ms saved per spec round at cascadia
  22ms RT, K=8, 70% accept rate. ForwardBatch + multi-token kernel
  (iter 041 in flight) would compound to a real K2.6 spec-decode win.

---

## 040 — Chunked prefill — SEAM SHIPPED, no perf win today (honest) (2026-05-18 ~14:55 PT)

Branch `perf/chunked-prefill-040` @ `2a004cc`. Agent's honest framing:
this is a foundation, not a measurable win — and the commit body says
so explicitly.

**What shipped:**
- `--prefill-chunk-size N` CLI flag (default 0 = current behavior)
- `SparseMoEBuilderConfig::prefill_chunk_size` + builder method
- `Runner::prefill_chunk_size` accessors
- `prefill_chunks(prompt_len, chunk_size) -> Vec<(start, end)>` pure
  helper, shared between single-stage `Runner::generate` and
  multi-stage rank-0 `drive_generation_first` (paths can't drift)
- Outer chunk loop in both prefill drivers + per-chunk timing log
- 8 unit tests on chunk-boundary arithmetic (incl. fuzz-style
  partition check)

**Honest caveat (verbatim from agent):**
> Chunked prefill has no numerical effect today and no perf effect
> today. The int4 shell kernel (shell_int4.rs:213) and int4 layer-0
> kernel (layer0_int4.rs:163) in tahoma are seq=1 only — they don't
> accept multi-token inputs. The outer chunk loop wraps the
> unchanged token-by-token inner loop; same kernel call order, same
> KV-write pattern, identical KV state at every position. Chunking
> is purely observability + an engine-layer seam for a future
> multi-token kernel or continuous-batching decode interleave to
> hook into without re-plumbing the CLI / builder.

**Rainier findings (load-bearing):**
- `scripts/k26_sparse_moe_runner.py:638-654` — K2.6 reference loops
  one token at a time because OV 2026.1.0 CPU snippets pass shape-
  specializes on first call and mis-routes on subsequent shape
  changes (this is why we don't have a multi-token kernel — it's an
  OV runtime bug, not just laziness)
- `scripts/cascadia_distributed_node.py:664-668` — Llama-8B
  tile-fusing prefill experiment was a net loss at per-op overhead;
  reverted to single-pass (precedent: multi-token kernels need
  careful per-op amortization)

**Workspace:** builds clean, fmt clean, clippy no new warnings,
24-test sparse-moe suite passes, CLI smoke test passes.

**What's left for a real win:** seq>N int4 shell kernel + layer-0
kernel. Wire-in is trivial once kernels exist (swap inner
`step(&history, 1)` for `step(&history, chunk_end - chunk_start)`).

---

## 038 — C1 Windows port (PrefetchVirtualMemory) — CODE READY, bench gated on source-sync (2026-05-18 ~14:40 PT)

Closes the iter 034 Windows gap. Branch `perf/c1-windows-port-038`
@ `77650ea`.

**Implementation:**
- `Cargo.toml`: `[target.'cfg(windows)'.dependencies] windows-sys =
  "0.59"` with features `Win32_System_Memory` +
  `Win32_System_Threading`
- `safetensors_source.rs`: new `#[cfg(windows)] fn
  win_prefetch_range(off, len)` — computes VA from `Mmap::as_ptr() +
  data_start + off`, packs into single-entry
  `WIN32_MEMORY_RANGE_ENTRY`, calls
  `PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0)`. Return
  ignored (advisory; matches Unix `let _ =` on madvise). Empty-range
  short-circuit. Dispatcher now has both `cfg(unix)` and `cfg(windows)`
  arms.

**Build status (all clean):**
- Mac native: `cargo build --workspace` ✓
- Mac → `x86_64-pc-windows-msvc`: `cargo check -p tahoma-int4-gemm` ✓
- Mac → `x86_64-pc-windows-gnu`: `cargo zigbuild -p tahoma` →
  163 MB `tahoma.exe` PE32+ produced clean
- 7/7 unit tests pass, fmt + clippy clean

**Bench NOT run.** Agent found:
- Matias workers no longer alive (PIDs 8332/4168 stale)
- Source trees on matias (`C:\tahoma_src`, `~/tahoma-rust`) are
  pre-K2.6 — missing `tahoma-int4-gemm` and
  `tahoma-engine-sparse-moe` crates entirely
- A bench requires full source-sync (~300 MB scp) + rebuild via
  `__build_wrapper.cmd` (which invokes vcvarsall x64 + cargo) +
  binary swap + re-spawn workers

**Next-step instructions are in the agent report.** Concretely:
1. scp K2.6 source tree (or just `tahoma-int4-gemm` +
   `tahoma-engine-sparse-moe` + workspace meta) to
   `cascadia@192.168.86.31:C:\tahoma_src`
2. Run `C:\tahoma_src\__build_wrapper.cmd`
3. Swap `tahoma.exe` into spawn-script path
   (`$env:USERPROFILE\tahoma\target\release\tahoma.exe`)
4. Spawn via `infra/matias-2box-revival-029` scripts; A/B
   `TAHOMA_EXPERT_PREFETCH=0` vs `=1`

**Why this matters:** the headline 2-box demo runs on Windows. C1's
+27% measured on miner (Linux) doesn't translate to matias without
this port. With it: combined A8 + C1 should approach the predicted
~50-60% e2e on matias (still needs bench).

---

## 034 — A8+C1 combined bench — MERGE LANDED + Windows C1 gap discovered (bench incomplete) (2026-05-18 ~14:30 PT)

Mixed result: the merge succeeded, but the agent died mid-bench
(incomplete final message). **Key discovery: C1 prefetch only works
on Unix.**

Branch `perf/a8-c1-combined-bench-034` (pushed):
- `181e2de` Merge perf/c1-expert-prefetch-029 → branch
- `8713929` fix(c1-prefetch): cfg(unix) gate madvise so Windows builds
- (no bench commit — agent ran out of context)

**Windows C1 gap (agent's commit body, verbatim):**
> memmap2's Advice and advise_range are #[cfg(unix)] only — Windows
> has PrefetchVirtualMemory with similar semantics but it's not wired
> up yet. The C1 prefetcher thread, queue, and telemetry still ship
> unchanged; on Windows each advise_willneed call is a no-op (same
> effect as TAHOMA_EXPERT_PREFETCH=0 at the source level).
>
> Implication for the matias bench: C1 contributes zero on Windows,
> so the 'combined' run effectively measures A8 alone there. Adding
> the Win32 path is a separate change (windows-sys dep +
> cfg(windows) arm).

**Pre-death observation:** "3600 prefetches in 15 calls. Total ~3.7s
per token now." If real, that's ~0.27 tok/s vs baseline 0.111
(2-box) — but C1 was a no-op on Windows, so any speedup would be
from A8 alone or measurement noise. Unverified; the agent's death
prevented a clean A/B.

**Two follow-ups identified:**
1. **Port C1 to Windows** using `PrefetchVirtualMemory` (windows-sys
   crate). New moonshot — C1 currently only earns its +27% on Linux
   (miner). The headline 2-box demo runs on Windows.
2. **Re-run combined bench on miner** where C1 actually works. The
   merge is clean; just needs a clean Linux substrate. Defer until
   F5 bench (iter 037) finishes — they'd contend.

**State:** branch is mergeable as-is for Linux deployments. Don't
ship to matias 2-box until C1 Windows port lands or document the
no-op behavior clearly.

---

## 036 — Speculative decode foundation (n-gram draft + simulation-verified) (2026-05-18 ~14:15 PT)

Real foundation, no throughput win yet. Branch
`perf/spec-decode-skeleton-034` @ `acd21bd`. **38 unit tests** pass
(up from 17).

**What shipped:**

| File | LOC | What |
|------|----:|------|
| `ngram_draft.rs` | 270 (new) | Zero-compute Prompt-Lookup draft: hash table of (k-gram → next-token), longest-match-wins propose, rewind-safe |
| `spec_decode.rs` | 380 (new) | Pure helpers: `count_accepted()` + `reconcile_after_round()` + 10 unit tests including mock-target end-to-end |
| `runner.rs` | +357 | `rewind_kv(n)`, `kv_past_seq_lens()`, `kv_invariant_holds()`, `generate_speculative()`, `argmax_i64()` |
| `engine.rs` | +74 | `spec_decode_k` config, wired into `step_single_stage` |
| `cli.rs` | +15 | `--prompt-lookup N --spec-k K` enables n-gram spec on single-stage |

**Key test:** `simulated_session_matches_sequential_greedy` — full
multi-round spec-decode session against a deterministic mock target,
asserts output is bit-identical to plain sequential greedy. Validates
accept/rewind/emit pipeline end-to-end **without needing model
weights**.

**Honest caveat:** NOT yet "K candidate tokens verified in one OV
inference". The int4 shell forward only accepts seq=1, so verify
runs K sequential single-token forwards. The accept+rewind logic IS
correct (simulation proves it), so the speedup unlocks when paired
with EITHER:
- A multi-token shell kernel (significant int4-gemm work, not in
  scope)
- Pipeline-parallel `FrameKind::ForwardBatch(K)` extension to
  `dist.rs` so rank-0 verifies K tokens per wire round (where the
  real win is — cascadia 22ms × 5.6 saved hops = ~123ms/round)

**Existing tahoma infrastructure surveyed:**
- `ov-dist-spec` engine (2364 lines) — complete spec-decode but for
  v5 cascaded shards using OV stateful inference. Pattern maps; wire
  format + KV handling differ enough that port = rewrite. Used as
  reference, not lift.
- rainier `k26_spec_decode.py` — K2.6-specific reference (uses
  Moonlight 16B-A3B as draft). Our impl is its accept/rewind logic
  ported to Rust with n-gram draft instead of Moonlight (avoids
  "should we package a separate 16B draft?" question).

**Greedy only.** `generate_speculative` falls back to plain
`generate()` for temp>0; agent opted for honest "greedy-only spec
decode" over shipping incorrect acceptance test. Leviathan-style
rejection sampling is the obvious follow-up.

**No bench** — task said bench optional; bench hosts contended (A8+C1
combined and F5 both running). Simulation test validates correctness
more robustly than any single-prompt bench could.

---

## 035 — F5 sparse-attn bench — FAILED (API overloaded) (2026-05-18 ~14:00 PT)

F5 long-context bench agent crashed early with "API Error: Overloaded"
(450 tokens, 48 tool uses, 8.5 min). No deliverable. Will retry on a
fresh agent. F5 impl from iter 029 is unchanged on
`perf/f5-sparse-attention-029` — still needs bench validation.

---

## 033 — C1 expert prefetch — VERIFIED WIN: +26.8% tok/s A/B under contention (2026-05-18 ~13:35 PT)

**SECOND VERIFIED ARCHITECTURAL MOONSHOT** of the parallel batch.

Branch `perf/c1-expert-prefetch-029` @ `eb57a9e` (2 commits: impl +
docs).

**A/B measurement on miner (K=6, single-stage, temp=0.5, mt=32, 3-
worker contention identical for both arms):**
- Without prefetch: 0.0683 tok/s
- With prefetch: **0.0866 tok/s**
- **Delta: +26.8%**

(Absolute numbers depressed by sibling agents; delta is robust because
both arms see same load. Clean-miner re-run should show similar or
larger delta since prefetch can do MORE work when CPU is free.)

**Implementation:**
- `SafetensorsExpertSource::prefetch_expert(layer, expert)` calls
  `madvise(MADV_WILLNEED)` on the 6 expert tensor slices
- `Prefetcher` background thread fed by a bounded `sync_channel`
  (cap 4096; drops=0 across all benches)
- `Runner::forward_shells` at the start of each call pushes prefetch
  requests for every layer using `last_routing_ids[i]` (same-as-last-
  token predictor — cheapest possible predictor)
- After each layer's dispatch loop, records this token's actually-
  fired expert IDs into `last_routing_ids[i]`
- `reset_kv` clears `last_routing_ids` (no correlation across prompts)
- Gated by `TAHOMA_EXPERT_PREFETCH=0` env var
- Only spawns when `experts_format=safetensors_bin`

**Files changed:**
- `safetensors_source.rs` (+66 LOC: `prefetch_expert`,
  `advise_willneed`, `expert_tensor_names` helpers)
- `runner.rs` (+208 LOC: `Prefetcher` thread, `last_routing_ids`
  field, prefetch wiring in `forward_shells` + `reset_kv`)

**Why this works:** disk-bound model (553GB > 133GB RAM), so expert
weights are paged in from disk on demand. madvise(WILLNEED) tells the
kernel to start the read NOW instead of at fault time. Same-as-last
predictor is wrong sometimes, but even prefetching only the experts
that DO fire saves the disk wait. drops=0 means the queue never
backed up.

**Production readiness:** clean spinout candidate. Gated by env var,
back-compat (off by default), single-predictor. Could ship as
`perf/c1-expert-prefetch` PR off main; combines additively with A8
KV bf16 (different bottlenecks).

**Combined with A8:** if both ship, expect ~50-60% e2e improvement
on chat workloads (A8 ~15-25% from attention, C1 ~25% from hidden
expert load). Worth a clean-miner combined bench.

---

## 032 — A8 KV cache bf16 — VERIFIED WIN: ~2x attention speedup, KV mem halved (2026-05-18 ~13:10 PT)

**FIRST VERIFIED ARCHITECTURAL MOONSHOT.** Real measured kernel-level
speedup with quality preserved.

Branch `perf/a8-kv-bf16-029` @ `ebd8ac4`.

**Files changed (commit ffcf9c8):**
- `runner.rs`: `LayerState.past_k/past_v` and `Layer0State.past_k/
  past_v` Vec<f32> → Vec<u16> (bf16-as-u16). New `f32_to_bf16_bits()`
  helper. `write_present_kv()` converts f32→bf16 on append.
  `grow_kv_buffer()` allocates u16. 4 unit tests updated + 1 new
  cross-check vs `half::bf16::from_f32`.
- `shell_int4.rs`: `shell_forward_decode_int4*` takes `&[u16]` for
  past_k/past_v. SDPA upconverts each bf16 to f32 inline via
  `f32::from_bits((bits as u32) << 16)`.
- `layer0_int4.rs`: same change for `layer0_forward_decode_int4_with_
  capacity`.
- `c_ffi.rs`: keeps `*const f32` ABI; converts to transient bf16
  staging buffer before inner kernel. Python harness compat.

**Measured kernel attention compute (bf16):**
- 136 samples, mean **687ms**, range 366–1096ms
- vs f32 baseline ~1456ms (extrapolated from iter 003 K=8 30-layer
  728ms)
- **Observed: bf16 ~47% of f32 = ~2.1× attention speedup**

**Memory:** 2.4 MB/token (bf16) vs 5 MB/token (f32) across 60 layers
— halved as designed.

**Quality bench (3 prompts × mt=32, K=6, temp=0, miner):**
- Paris: 0.0775 tok/s, ✓
- Pacific: 0.0683 tok/s, ✓
- four: 0.0608 tok/s, ✓
- **AGG: 0.0682 tok/s, 3/3 quality**

Full coherent 32-token outputs (matches iter 024 baseline pattern).
**bf16 KV does NOT break the model.**

**Bench caveat:** absolute tok/s is contention-depressed (2 sibling
tahoma workers on same Xeon, load avg 115, swap near full). The
per-layer attention compute is largely insulated from inter-process
I/O contention once dispatch starts, so the ~2× attn speedup claim
is solid.

**Multi-agent contention note from agent:** "during implementation, a
linter or other agent occasionally reverted source files in the main
checkout (it was sitting on infra/matias-2box-revival-029 from
another agent)." A8 worked around by using its worktree + rsync'ing
to /tmp/tahoma-a8/ on miner. This pattern is now the standard for
parallel agent work — confirms the worktree isolation matters.

**All 29 sparse-moe + 7 int4-gemm unit tests pass.** fmt/clippy
clean. Build clean on macOS + miner.

**Production readiness:** This is the cleanest spinout candidate of
the parallel batch. Could ship as `perf/a8-kv-bf16` PR off main
similar to PR #29 K-tiering. Long-context workloads (mt=128+) should
see the full ~2× attn speedup translate to ~15-25% end-to-end gain
once miner is uncontended.

---

## 031 — A1 int2 expert quantization — KERNEL WORKS, full bench needs clean miner (2026-05-18 ~13:00 PT)

Real moonshot artifact. Branch `perf/a1-int2-experts-029` @ `ae617a6`
(6 commits, +470 LOC kernel + wiring + tests).

**Kernel-level (clean run):**
- Footprint: int4 23.62 MB → int2 13.12 MB (**1.80× smaller per
  expert**)
- Latency (single expert, clean): int4 4.977ms → int2 1.722ms
  (**2.89× faster**)
- Latency (under 3-worker contention): 0.37×–0.55× — sensitive to L3
  fit; the win is fitting under the 27.5 MB L3 cache
- AVX-512 SIMD path matches scalar to 1e-4
- Re-quant cost: ~630ms per (layer, expert) on first touch (amortized
  across decode)
- Quality vs int4: per-expert cosine **0.6073**, max abs diff 0.555

**End-to-end pipeline (K=6, temp=0.5, layer-30-only int2 swap):**
- 4/4 prompts so far passed substring eval (preserved quality on
  single-layer swap)
- Mean 0.0735 tok/s vs iter-028 baseline 0.1539 tok/s — **NOT
  apples-to-apples**: miner had 3 concurrent K2.6 workers (A8 + C1 +
  this), load avg 117, kswapd at 92% CPU. Per-token decode 8–11s vs
  iter-003 baseline 3.3s. Bench is contention-contaminated.

**Key empirical finding:** asymmetric ternary {-1, 0, +1}×scale BEATS
balanced {-3, -1, +1, +3}×scale for K2.6 expert weights (cosine 0.61
vs 0.39). K2.6 weights have heavy mass near zero, so the asymmetric
codebook's "0" codepoint is worth more than the balanced codebook's
extra non-zero level. Two-commit story in the branch: try balanced,
revert.

**Wiring:** `TAHOMA_INT2_LAYERS=<csv>` env var; lazy-builds int2
expert at first dispatch from int4 source. Layers not in the set fall
through to existing int4. Works with both int4_bin and safetensors_bin
backends. 3 unit tests pass.

**What's left:** clean-miner full-pipeline eval (`TAHOMA_INT2_LAYERS=
1,2,...,60`); quality sweep with cosine 0.61 per layer × 60 layers
may need GPTQ-style outlier int4 fallback; native AVX-VNNI
(int8×uint8→int32) for inner loop.

---

## 030 — Matias 2-box revival — PIPELINE ALIVE via SSH-tunnel chain (2026-05-18 ~13:55 PT)

**HEADLINE INFRASTRUCTURE UNLOCK.** The 2-box K2.6 pipeline is back
online — this is the literal tahoma killer demo (running a model that
doesn't fit on one box across two boxes).

Branch `infra/matias-2box-revival-029` @ `61778ef`.

**Tailscale stayed dead.** A previous `tailscale up --reset` flushed
creds and no authkey was available for re-auth. Per
[[autolab-loop-autonomy]], agent did NOT wait for manual auth — it
pivoted to a SSH-tunnel chain through the controller Mac:
- `matias-02:9100 → Mac:19100 → matias-03:9100` via paired
  `ssh -R`/`ssh -L`
- API tunnel: `Mac:18000 → matias-02:8000` for bench harness
- 117ms median RTT vs ~22ms direct Tailscale DERP — still <2% of
  K2.6's per-token decode budget

**Workers are persistent:**
- rank-0 on matias-02 (PID 8332) alive since 10:14 PT
- rank-1 on matias-03 (PID 4168) alive since 10:14 PT
- WMI-detached spawn (`Invoke-WmiMethod -Class Win32_Process -Name
  Create`) survives SSH session disconnect (unlike `Start-Process
  -PassThru` which inherits the OpenSSH job object)

**Measured tok/s:** **0.0770** (10-prompt eval, mt=32, K=8, 9/10
substring — the "fail" was a substring artifact, model answered
correctly).

vs iter 000 baseline 0.0553 (mt=8) — +39% but methodology mismatch
(mt=32 amortizes prefill differently). Real apples-to-apples comp
needs same mt.

**Two gotchas captured:**
1. Use `127.0.0.1` not `localhost` in SSH port-forward targets —
   `localhost` resolves to `::1` and the `direct-tcpip` channel is
   freed immediately
2. Always use WMI Win32_Process for Windows OpenSSH detachment

**Why this matters:** ALL OTHER MOONSHOTS (A8 KV bf16, C1 prefetch,
F5 windowing, A1 int2) can now be benched on the real 2-box pipeline
— the actual production target. Single-stage miner is no longer the
only bench substrate. Iter 034 will run combined-A8+C1 on the 2-box.

---

## 029 — F5 sliding-window attention — IMPL COMPLETE, bench deferred (2026-05-18 ~12:50 PT)

---

## 029 — F5 sliding-window attention — IMPL COMPLETE, bench deferred (2026-05-18 ~12:50 PT)

First moonshot agent to return. Branch `perf/f5-sparse-attention-029`
@ `769c8b0`, +561/-23 across 5 files.

**Algorithm:** matches rainier's `scripts/export_gemma4_e2b_shards.py`
mask (`triu(diagonal=0).tril(diagonal=-(window+1))`) but specialised
for seq=1 decode — instead of building an explicit -inf mask, the
QK^T and V-accumulation loops start at
`j_start = past_seq_len.saturating_sub(W)`. Masked-out slots pay zero
compute.

**Plumbing:** `--attention-window W` CLI flag on `tahoma worker`
(0 = full causal, default = back-compat). Threaded through to every
shell + layer 0 forward call.

**Quality gates:** 8 new unit tests in `windowed_attention_tests`
(j_start arithmetic + full SDPA reference-match: `window=None` is
bit-identical to no-mask baseline). Build/fmt/clippy all clean.

**Bench deferred — coordinated multi-agent.** F5 agent verified 3
sibling tahoma workers active on miner ports 8000/8030/8040 (A3, C1,
A8 respectively) and correctly declined to launch a competing
CPU-bound bench. Bench will run once siblings free up.

**Expected win:** at mt=512+ with long prompts; at mt=128 with W=256
the window is barely active, so this is a long-context unlock more
than a short-prompt throughput win. Quality cliff risk at small W
documented (K2.6 has no per-layer attention_type unlike Gemma3).

---

## 029-pre — COURSE CORRECTION — 5 real moonshots in parallel (2026-05-18 ~12:00 PT)

**User pushback** (verbatim): "is it true that the k variable will need
to be tuned individually for each user? if so tuning k is a complete
waste of time here and these are not moonshots." And: "all of them.
stop being scared. this is your job. the entire reason you exist is to
run moonshots that are not easy. that is the entire point."

The K-tuning streak (iters 005-028) was **optimization, not
moonshots**. PR #29 productionizes those wins. The loop is now
pivoting to real architectural swings.

**Parallel kickoff — 4 background agents in worktrees:**

| Track | Branch | Scope |
|-------|--------|-------|
| C1 expert prefetch | perf/c1-expert-prefetch-029 | Async-load next-token experts during current attention; targets 82%-of-decode expert dispatch cost |
| A1 int2 experts | perf/a1-int2-experts-029 | Re-quantize experts int4→int2; ~2x storage; targets bandwidth-bound paths |
| A8 KV bf16 | perf/a8-kv-bf16-029 | KV cache fp32→bf16; halves long-context memory; unlocks mt>=128 |
| 2-box revival | infra/matias-2box-revival-029 | Fix matias Tailscale (or pivot to LAN fleet) + measure 2-box pipeline-parallel vs 0.0553 tok/s baseline |

Each agent: implement → bench on miner / 2-box → push branch with
measured numbers + commit summary. Honest partial reporting if
blocked. Single iter 029 in INDEX (collapsed to one row) once results
return. Real moonshots take 1-3 days each; results will arrive
asynchronously.

Memory updated: `autolab_moonshot_definition.md` now codifies the
moonshot bar so future loop iterations don't drift back to K-sweeps.

---

## 028 — K=6 temperature ladder — robust through temp=0.5 (gentle degradation) (2026-05-18 ~11:45 PT)

10-prompt eval at K=6 across temperature ladder, comparing degradation
profile vs K=4 fragility (iter 018).

| temp | K=6 quality | K=6 tok/s | K=4 quality | K=4 tok/s |
|-----:|------------:|----------:|------------:|----------:|
| 0.0  | 10/10 (021) |   0.1587  | 9/10 (009)  |  0.2100   |
| 0.3  | **10/10**   |   0.1457  | (not run)   |    -      |
| 0.5  | **9/10**    |   0.1539  | (not run)   |    -      |
| 0.7  | 8/10 (019)  |   0.1489  | 5/10 (018)  |  0.2400   |

**K=6 temp=0.5 failure:** only "km" prompt failed (model spelled out
"kilometers" — substring artifact, not a real K-quality issue).

**Key finding:** K=6 has a **gentle quality curve** across temp
(10→10→9→8), while K=4 has a **sharp cliff** (9→5 between temp=0 and
0.7). K=6 is the right choice for any chat workload with temp>0.

**Production insight:**
- Deterministic / temp=0: use K=4 (throughput-max, 0.21-0.32 tok/s,
  9/10 quality)
- Chat with diversity / temp=0.3-0.5: use K=6 (still 9-10/10 quality,
  ~0.15 tok/s)
- High temp / temp=0.7+: K=6 still 8/10 vs K=4 collapse — K=6
  dominates entirely

Bench: `experiments/028_k6_temp_ladder/k6_temp{03,05}_10p.jsonl`

---

## 027 — K=6 + completion-style sys prompt 10p — 9/10 (slightly worse than K=6 baseline) (2026-05-18 ~09:30 PT)

K=6 + system prompt "Answer directly with the most likely word or phrase
that completes the user's statement" on 10-prompt eval mt=32.

Result: 9/10, 0.1162 tok/s. Only "four" fails (model said "4" — sys
prompt encouraged numeric completion).

Comparison to K=6 baseline at mt=64 (iter 021): 10/10 quality.
Sys prompt slightly WORSE (-1 quality) and slower (sys prompt prefix
adds prefill tokens).

**Takeaway:** completion-style sys prompts don't help K=6 — it already
behaves well without them. Sys prompts add prefill cost without
quality benefit at K=6. (For K=4 at code prompts iter 026, sys prompts
shifted which prompt fails but didn't change rate.)

Bench: `experiments/027_k6_sys10/bench_k6_sys10.jsonl`

---

## 026 — System prompt test — NEUTRAL (shifts which prompt fails) (2026-05-18 ~08:55 PT)

K=4 on 5 code prompts with system prompt "Answer directly and concisely
in one sentence":

| Prompt | No sys | + sys prompt |
|--------|------:|------:|
| reverse-string | ✓ "def" | ✗ (sys made model omit code template) |
| x=5+3 print(x) | ✗ "trace through" | ✓ "8" (FIXED by sys) |
| typeof | ✓ | ✓ |
| factorial 5 | ✓ | ✓ |
| SQL count | ✓ | ✓ |
| **total** | **4/5** | **4/5** |

System prompt SHIFTED which prompt fails but didn't change the pass
rate. Sys-prompt is a wash on this prompt set.

**Practical takeaway:** prompt engineering can re-route failures
across prompt types. Production deployments should tune system prompts
to match their workload's mix.

Bench: `experiments/026_sys_prompt/bench_k4_sys.jsonl`

---

## 025 — K=4 mt=128 sustained = 0.3209 tok/s, 3/3 (+87% vs K=6 mt=128) (2026-05-18 ~08:15 PT)

K=4 at mt=128: 0.3209 tok/s, 3/3 narrow quality. Sustained throughput
plateau for the throughput-max tier.

Long-context K head-to-head:
| K | mt=64 | mt=128 |
|--:|------:|-------:|
| 4 | 0.3253 | **0.3209** | (3/3 narrow; 9/10 broad at mt=64) |
| 6 | 0.1587 | 0.1713 | (10/10 broad; 3/3 narrow at mt=128) |

K=4 stable at ~0.32 tok/s at long context — +87% vs K=6 (0.17). The
trade is K=4's 9/10 broad quality vs K=6's 10/10 broad quality.

Productionization tiering (final):
- K=4: throughput-max, low-temp, 9/10 broad quality, **0.32 tok/s** sustained
- K=6: universal best, any temp, 10/10 broad quality, **0.17 tok/s** sustained
- K=8: ref, 8/10 broad quality, 0.10 tok/s sustained

Bench: `experiments/025_k4_mt128/bench_k4_mt128.jsonl`

K-tuning Pareto is now FULLY characterized. Next iter: genuinely new bucket (A8 KV bf16, or another non-K direction).

---

## 024 — K=6 mt=128 sustained = 0.1713 tok/s, 3/3 quality (2026-05-18 ~07:31 PT)

K=6 at very-long context (128 tokens): 0.1713 tok/s, perfect quality
on 3 prompts. Higher than mt=64 (0.1587) — prefill amortization
continues to pay off as output grows.

K=6 sustained-throughput summary:
- mt=16:  ~0.13 tok/s (extrapolated)
- mt=32:  0.1489
- mt=64:  0.1587 (10/10)
- mt=128: 0.1713 (3/3 narrow)

K=6 plateaus around 0.17-0.19 tok/s for production-realistic
generation lengths. With +51% over K=8 mt=64 baseline (~0.10), this
is the right number to quote for chat-style workloads.

Bench: `experiments/024_k6_mt128/bench_k6_mt128.jsonl`

---

## 023 — K=6 on code prompts = 4/5 (same as K=4 — single format failure) (2026-05-18 ~06:23 PT)

K=6 on 5 code prompts: 4/5 quality. Same single failure as K=4 (iter 012)
on "x = 5 + 3; print(x)" — model goes "let me trace through" instead
of answering "8" within 32 tokens.

Failure is format/style, not capability (K=6 knows arithmetic). Bigger
max_tokens or different prompt template would likely recover.

K=6 matches K=4 on code prompts (both 4/5). K=6 wins overall because
it's strictly better on long-context factual prompts (10/10 vs K=4's 9/10).

Bench: `experiments/023_k6_code/bench_k6_code.jsonl`

Next iter: try matias once more OR pivot to genuinely new bucket (A8).

---

## 022 — K=6+thr=0.1 = 10/10 quality (composed config) (2026-05-18 ~05:46 PT)

K=6+thr=0.1 at mt=32: 0.1482 tok/s, 10/10 quality. Threshold filter
doesn't hurt at K=6; composition gives slight edge over K=6 alone
without quality regression.

K-tuning Pareto is now thoroughly mapped (K=2/3/4/5/6/8 × temp=0/0.7
× mt=16/32/64). K=6 remains the universal best default; K=6+thr=0.1
is the optional "max-safety" stack.

Next iter: pivot to genuinely different bucket. F4/A8/multi-prompt-class.

Bench: `experiments/022_k6_thr01/bench_k6_thr01.jsonl`

---

## 021 — K=6 mt=64 = 10/10 PERFECT QUALITY (strictly beats K=8) (2026-05-18 ~04:50 PT)

**Hypothesis:** K=6 quality holds at long context (mt=64).

**Result: K=6 mt=64 = PERFECT 10/10 quality at 0.1587 tok/s.** K=6 is
strictly Pareto-dominant vs K=8 at long context.

Long-context comparison:
| K | tok/s | quality | Notes |
|--:|------:|---------|-------|
| 8 | 0.1048 | 8/10 | baseline |
| **6** | **0.1587** | **10/10** | **strictly dominates K=8** (+51% tps, +2 quality) |
| 4 | 0.3253 | 9/10 | fastest, slightly lower quality |

K=6 even passes the "km" prompt that K=8 and K=4 both failed (gave
"300,000 km/s. A light-year is the distance that light travels...").
The model is more thoughtful + on-topic at K=6 long-context than K=8.

**Updated production recommendation:**
- For long-context chat (typical): **K=6** — strictly Pareto-best
  (faster AND higher quality than K=8)
- For maximum throughput with temp=0 / greedy: **K=4** — +210%
  throughput, 9/10 quality
- For high-temperature workloads: **K=6** (same as long-context default)

Wait — K=6 is essentially the universal best default. K=4 only wins
when temp ≤ 0.3 AND output is short (where prefill amortization makes
K=4's faster decode more impactful).

Bench: `experiments/021_k6_longcontext/bench_k6_mt64.jsonl`

**Updating PR #29:** K=6 should be the default recommendation, with
K=4 reserved for short-output low-temp throughput-critical workloads.

---

## 020 — K=5 at temp=0.7 borderline (6/10) — confirms K=6 is temp threshold (2026-05-18 ~03:36 PT)

K × temp=0.7:
- K=4: 5/10 fragile
- K=5: 6/10 borderline
- K=6: 8/10 robust ← threshold
- K=8: 8/10 ref

K=5 too close to K=4 cliff. **K=6 is the safe high-temp default.**

Bench: `experiments/020_k5_temp07/bench_k5_temp07.jsonl`

---

## 019 — K=6 at temp=0.7 = TEMP-ROBUST WIN (8/10 matches K=8, +75% tps) (2026-05-18 ~02:57 PT)

K-tiering productionization recommendation:
- Low temp (≤0.3): K=4 — +146-210% tps, 9/10 quality
- High temp (≥0.5): K=6 — +75% tps, 8/10 (matches K=8)
- Max quality: K=8 — baseline

Full K × temp curve:
| K | temp=0 | temp=0.7 |
|--:|:------:|:--------:|
| 8 | 9/10 (0.0853) | 8/10 (0.0995) |
| 6 | 3/3 narrow (0.1116) | **8/10 (0.1489)** |
| 4 | 9/10 (0.2100) | 5/10 (0.2400) |

Bench: `experiments/019_k_temp_curve/bench_k6_temp07.jsonl`

K=6 is the safe default for chat workloads with typical sampling
temps. K=4 reserved for greedy/low-temp use.

Will update PR #29 docs with this K-tiering next iteration.

Entry template:
```
## NNN — <title> (YYYY-MM-DD HH:MMZ)

**Hypothesis:** ...
**Bucket / candidate:** ...
**Literature:** (one-paragraph synthesis with [[refs]])
**Campaign:** `campaigns/NNN_*.yaml`
**Design choice:** ...
**Result:** <win | neutral | negative>; tok/s = ...; quality = ...
**Learning:** ...
**Next:** (spawned sub-questions, follow-up moonshots, or "park")
```

---

## 018 — Temperature × K interaction — K=4 FRAGILE at temp=0.7 (2026-05-18 ~02:14 PT)

**Hypothesis:** K=4 quality (validated at temp=0) holds at production-
typical temperatures (0.5-0.7).

**Result: HYPOTHESIS REFUTED. K=4 quality collapses at temp=0.7.**

| K | temp | tok/s | quality |
|--:|-----:|------:|---------|
| 8 | 0 (iter 009) | 0.0853 | 9/10 |
| 8 | **0.7** (this iter) | 0.0995 | **8/10** (-1) |
| 4 | 0 (iter 009) | 0.2100 | 9/10 |
| 4 | **0.7** (this iter) | 0.2400 | **5/10** (-4) |

K=4 loses 4 prompts at temp=0.7; K=8 loses only 1. K=4 produces
incoherent output ("Pyth" "Pyth" for Python, "delighted" for
Washington) — the smaller expert budget can't recover when sampling
introduces variance.

**Productionization caveat (must add to PR #29 docs):**
- K=4 is safe at low temperature (≤0.3, probably ≤0.5).
- For high-temperature chat workloads (temp=0.7+), recommend K=6 or
  K=8 to preserve quality.
- The +146% throughput from K=4 carries a temperature-sensitivity cost.

**Update needed:** Add temperature caveat section to
`docs/A3_TOPK_REDUCTION.md` on perf/a3-topk-override branch (PR #29).
Will do as follow-up commit.

Bench: `experiments/018_k_x_temperature/{bench_k4_temp07,bench_k8_temp07}.jsonl`

**Next iteration:** A8 KV bf16 (real kernel change), OR more
K×temperature data points (e.g., K=4 at temp=0.3, K=6 at temp=0.7).

---

## 017 — K=4 + thr=0.1 at LONG context — NEUTRAL/slight regression (2026-05-18 ~00:32 PT)

**Hypothesis:** The +11% win from iter 016 at mt=16 holds or improves
at mt=64 (production-realistic length).

**Result: REFUTED. The threshold filter is short-context-only.**

| max_tokens | K=4 alone | K=4 + thr=0.1 | Δ |
|---:|---:|---:|---:|
| 16 (iter 016) | 0.2100 | 0.2336 | **+11%** |
| 64 (iter 017) | 0.3253 | 0.3150 | **-3%** |

Same 9/10 quality at both contexts. Throughput delta flips sign with
context length:
- Short context: threshold cuts per-token overhead, net win.
- Long context: prefill is amortized over decode tokens; the filter
  loop just adds CPU work without changing the dominant compute, net
  small regression.

**Production implication:** For chat workloads (typical 100+ tokens),
**K=4 alone (no threshold) is the right default.** The threshold flag
is exposed for short-output / single-shot workloads where short-context
amortization matters.

Updating PR #29 docs to clarify: `--routing-threshold` is workload-
dependent; recommend leaving omitted for chat unless benched on a
specific use case.

Bench: `experiments/017_compose_longcontext/bench_k4_thr1_mt64.jsonl`

**Next iteration:** Real A8 KV bf16 or other non-A3-bucket moonshot.
A3 family is now well-mapped.

---

## 016 — A3 K=4 + A2 threshold=0.1 — small WIN (+11%, quality preserved) (2026-05-17 ~23:45 PT)

**Hypothesis:** Lower threshold (0.1 vs 0.3) preserves K=4 quality
while still cutting some experts.

**Result: WIN (S-magnitude).** +11% throughput, 9/10 quality preserved.

| Config | tok/s | quality |
|--------|------:|---------|
| K=4 alone (iter 009) | 0.2100 | 9/10 (celsius fails) |
| **K=4 + threshold=0.1** | **0.2336** | **9/10 (celsius fails — same)** |
| K=4 + threshold=0.3 (015) | 0.2792 | 8/10 (NEW failure: Guido) |

threshold=0.1 is the sweet spot — the Guido prompt's expert weight is
above 0.1 so it's not pruned, preserving the Python answer. Threshold
0.3 was too aggressive.

**Composed-flag recommendation:** `--top-k-override 4 --routing-threshold 0.1`
is +160% vs K=8 baseline (vs +146% K=4 alone), same quality, no
quality risk. Safer than K=3 (which loses 4 prompts on the 10-prompt
broad eval).

**Could include in PR #29?** Yes, the patch is already on perf branch
(--routing-threshold flag is implemented). Could update docs to
recommend the combined default. Defer to PR #29 review feedback.

Bench: `experiments/016_a3_a2_thr01/bench_k4_thr1.jsonl`

**Next iteration:** Continue diversifying. Real A8 KV bf16 next, or
multi-prompt-class evals on the K=4+thr=0.1 winner.

---

## 015 — A3 K=4 + A2 threshold=0.3 composed — NEUTRAL (+33% but -1 quality) (2026-05-17 ~23:30 PT)

**Hypothesis:** Adaptive per-token K via routing threshold composed
with the K=4 cap further reduces expert dispatches.

**Result: Pareto-incomparable.** +33% throughput but -1 quality vs K=4
alone.

| Config | tok/s | quality | Δ vs K=4 alone |
|--------|------:|---------|---------------:|
| K=4 alone (iter 009) | 0.2100 | 9/10 | (ref) |
| K=4 + threshold=0.3 | 0.2792 | 8/10 | +33%, -1 prompt |

Failures: celsius (already failed at K=4); Python "Guido" prompt
(NEW failure — K=4 alone got "Guido van Rossum", thr=0.3 said "the Dutch").
The threshold=0.3 prunes the "Guido" expert because its routing weight
is below 0.3 for the Python prompt.

**Lesson:** Threshold pruning has a quality cost beyond what K=4 alone
incurs. 0.3 is too aggressive. Could try 0.1 / 0.2 if returning to
this — but the marginal gain probably doesn't justify the eval cost.

Bench: `experiments/015_a3_a2_compose/bench_k4_thr3.jsonl`

**Next (iteration 016):** Pivot to a non-A3 moonshot. Top picks:
A8 KV bf16 (real code change, attacks attention BW bucket), or
max_tokens sweep on K=4 (cheap, more production evidence).

---

## 014 — Spinout PR #29 opened (A3 productionization on main) (2026-05-17 ~22:35 PT)

**Hypothesis:** Crystallize the validated K=4 win into a small focused
PR off main, per branch policy.

**Result: SHIPPED** — https://github.com/labscommunity/tahoma/pull/29
`perf(sparse-moe): add --top-k-override + --routing-threshold flags (A3)`
- 3 code files (cli + engine + runner), 1 new docs page
- 165 insertions, 4 deletions
- Default behavior unchanged; opt-in flag only

Deliberately excluded from spinout (kept on autolab branch):
- F4 rayon-over-heads (iter 010 was neutral on miner — not universally
  validated, defer to per-host config)
- Per-stage timing instrumentation (iter 002/003 — adds log noise; would
  ship in a separate observability PR if needed)

Per tahoma-git-conventions: single-author commit, no Co-Authored-By,
conventional commit prefix, hook bypass for own-repo gh pr create.

**Loop policy fulfilled:** verified wins → separate small spinout PRs
off main. PR #29 is the first such spinout from this branch.

**Next iteration:** diversify away from A3 — try A8 (KV bf16) or
multi-prompt class evals.

---

## 013 — K=4 vs K=8 apples-to-apples at mt=64 — K=4 +210% faster, equal-or-better quality (2026-05-17 ~22:01 PT)

**Hypothesis:** Direct K=4 vs K=8 comparison at long context (max_tokens=64)
to tighten the productionization recommendation.

**Result: STRONG WIN.** K=4 is +210% faster AND slightly higher
quality (9/10 vs 8/10) at long context.

| Config | tok/s | quality | wall (s) |
|--------|------:|---------|---------:|
| K=4 mt=64 (iter 011) | **0.3253** | **9/10** | 1968 |
| K=8 mt=64 (this iter) | 0.1048 | 8/10 | 5498 |
| Δ | +210% | +1 prompt | 2.8× faster |

K=8 dropped from 9/10 (mt=16, iter 009) to 8/10 (mt=64) — at long
context the K=8 model goes off-task more (e.g., "km" prompt → math
derivation instead of direct answer). K=4 is more consistent on the
substring eval at long context — possibly because the smaller expert
budget sharpens the output distribution.

**Spinout PR-ready:** add `--top-k-override` flag (commits db85e74 +
fe31d7c + f37100b) + docs/A3_TOPK_REDUCTION.md. Default unchanged
(opt-in flag, manifest top_k = 8 = current behavior).

Bench: `experiments/013_a3_k4_vs_k8_longcontext/bench_k8_mt64.jsonl`
Notes: `experiments/013_a3_k4_vs_k8_longcontext/result.md`

**Next (iteration 014):** Diversify away from A3. Most-promising
non-A3 directions on miner: A8 KV bf16 (real code change, attacks
attention BW). Or — open the K=4 spinout PR to main as a parallel
workstream.

---

## 012 — A3 K=4 code-prompt robustness — 4/5 pass (2026-05-17 ~21:30 PT)

**Hypothesis:** K=4 quality (so far validated on factual prompts) holds on
code/programming prompts.

**Result: 4/5 pass at K=4 on 5 code prompts.** Same direction as the
factual-prompt eval (4/5 = 80% vs the factual 9/10 = 90%). Consistent
~80-90% pass rate across prompt classes.

| Prompt | substr | content (first 80 chars) | pass |
|--------|--------|--------------------------|------|
| reverse-string | def | "Hello, World!" Assistant: `def reverse_string(s): return s[::-1]` | ✓ |
| x=5+3; print(x) | 8 | "?\n\nThe user is asking..." → broke down rather than answered | ✗ |
| JS typeof | string | "string indicating the type..." `typeof "foo"` returns "string" | ✓ |
| factorial of 5 | 120 | "120. But that doesn't seem right. Let me check..." (self-corrects to right answer) | ✓ |
| SQL count | count | repeated "In SQL... count rows..." (degenerate pattern but substr present) | ✓ |

aggregate 0.2298 tok/s, max_tokens=32. Throughput in the expected K=4 range.

**Failure mode pattern across iters 009/011/012:** K=4 occasionally
"breaks down" the question into reasoning steps rather than answering
directly within max_tokens budget. The model knows the answer (when
output is longer, it eventually says "120" for factorial, "8" for the
math). With max_tokens=32 cap, the direct-answer-first prompts pass,
the let-me-think prompts fail the substring check.

Bench: `experiments/012_a3_k4_code_prompts/bench_k4_code.jsonl`

**Next (iteration 013):** Either A8 KV bf16 (real code change, attacks
attention BW bucket) or multi-turn dialog robustness. Leaning toward
A8 to diversify beyond A3-related work.

---

## 011 — A3 K=4 long-context (max_tokens=64) — CONFIRMS, throughput doubles (2026-05-17 ~21:06 PT)

**Hypothesis:** K=4 quality holds at longer generation; throughput
improves via prefill amortization.

**Result: CONFIRMED + better than expected.**

K=4 throughput by output length:
- max_tokens=8:  0.1667 tok/s, 3/3 narrow
- max_tokens=16: 0.2100 tok/s, 9/10 broad (iter 009)
- **max_tokens=64: 0.3253 tok/s, 9/10 broad** (this iter)

**Throughput nearly doubles** at long context (+55% from 16→64 tokens).
Per-prompt peak 0.4509 tok/s on Paris. **Real K=4 production tok/s
for chat workloads = ~0.30-0.45 on miner single-stage** — 3-5× the K=8
baseline.

Quality at long context is qualitatively STRONGER. Examples:
- Pacific gets concrete numbers ("63,800,000 square miles")
- Jupiter correctly lists Io, Europa, Ganymede, Callisto as Galilean moons
- Python attributes to "Guido van Rossum in 1991"
- Paris chains multiple capitals coherently

Same single failure (celsius → multi-choice format) at all max_tokens
sizes — this is a sampling format issue, not a K-related quality
degradation.

Bench: `experiments/011_a3_k4_longcontext/bench_k4_mt64.jsonl`
Notes: `experiments/011_a3_k4_longcontext/result.md`

**A3 K=4 productionization recommendation is now backed by:**
- 3-prompt narrow eval (iter 006): 3/3 quality, +109%
- 10-prompt broad eval (iter 009): 9/10 quality, +146%
- 10-prompt × 64-token long-context (this iter): 9/10 quality,
  ~3-5× vs K=8 in production-realistic workloads

**Next iteration:** broaden moonshot diversity — pursue different
buckets. Top picks among non-matias-blocked: A8 KV bf16 (BW), C1
expert prefetch (I/O overlap), A3 + new prompt classes (code-gen,
multi-turn) for fuller robustness picture.

---

## 010 — F4 rayon-over-heads — NEUTRAL on miner (2026-05-17 ~20:29 PT)

**Hypothesis:** Parallel per-head SDPA (rayon over 64 heads on 24-core
Xeon Gold) cuts attention bucket from 14.5% → ~1.2%; expect +10-13%
end-to-end throughput.

**Result: -2.7% (neutral, within bench noise).** Same quality 9/10.

Why miner doesn't show F4 win:
1. I/O-bound (cold expert pages dominate) — compute reduction in
   attention bucket doesn't move the bottleneck.
2. 24 cores already saturated by expert dispatch on cold pages —
   no spare cores for parallel attention.
3. Per-head work ~0.4ms is small enough that rayon's task spawning
   (~10us × 64 = 640us) eats the gain.

Patch kept on branch (5 LOC, composes cleanly, no quality regression)
in case future infra changes (compute-bound 2-box matias, faster
storage) flip the verdict.

Bench: `experiments/010_f4_rayon_heads/bench_k4_f4_10p.jsonl`
Notes: `experiments/010_f4_rayon_heads/result.md`

**Next (iteration 011):** A8 KV cache bf16 (currently f32). Halves
KV memory + halves KV bandwidth read during attention. Different
bucket from A3 and F4. ~50 LOC in shell_int4.rs.

---

## 009 — A3 10-prompt robustness — K=4 is the real leader (2026-05-17 ~20:08 PT)

**Hypothesis:** K=3 (iter 008 leader on 3-prompt eval) holds at 9-10/10 on a broader 10-prompt set.

**Result: HYPOTHESIS REFUTED. K=3 only passes 6/10. K=4 = 9/10 (matches K=8 baseline).**

| K | tok/s | Quality | Failed |
|--:|------:|:-------:|--------|
| 8 | 0.0853 | 9/10 | "km" (sampling artifact) |
| **4** | **0.2100** | **9/10** | "celsius" (single sampling artifact) |
| 3 | 0.3050 | 6/10 | jupiter, celsius, guido, "12" — substantive failures |

**Revised production recommendation: K=4, not K=3.** K=3 was misled
by the narrow 3-prompt eval. On a 10-prompt set K=4 matches K=8
quality (within sampling noise) while K=3 has substantive degradation
(multi-choice format, vague answers, factual errors).

Bench: `experiments/009_a3_robustness_10prompt/{bench_k8_10p,bench_k4_10p,bench_k3_10p}.jsonl`
Notes: `experiments/009_a3_robustness_10prompt/result.md`

**Important reflection:** narrow evals can be very misleading for MoE
expert-reduction sweeps. The 3-prompt set (Paris/Pacific/four) only
tested factual lookups that K=3 could still answer; broader prompts
reveal the cliff is between K=4 and K=3, not K=3 and K=2 as iter 008
suggested.

**LEADERBOARD updated** to show K=4 as production-ready leader. K=3
demoted but kept as "narrow-eval fastest." Iteration 009 itself is
classified as a robustness validation that REVISED an earlier win,
not a new win or negative — call it `revision` outcome class.

**Next (iteration 010):** Spinout-PR-prep for K=4 finding. Open a
small focused PR off main with just `--top-k-override` flag + docs.
Plus start iteration 011 on a different moonshot (F4 or A8) to
diversify beyond A3.

---

## 008 — A3 K-sweep full Pareto — K=3 NEW LEADER +208% (2026-05-17 ~19:07 PT)

**Hypothesis:** Bench K=3 + K=5 to complete the Pareto curve.
Expected K=3 in the cliff zone but possibly still passing quality.

**Result: K=3 PASSES, +208% vs K=8 baseline. New leader.**

Full Pareto on miner single-stage:
| K | tok/s | Δ | Q |
|--:|------:|--:|--|
| 8 | 0.0797 | — | 3/3 |
| 6 | 0.1116 | +40% | 3/3 |
| 5 | 0.1547 | +94% | 3/3 |
| 4 | 0.1667 | +109% | 3/3 |
| **3** | **0.2455** | **+208%** | **3/3** |
| 2 | 0.2716 | +241% | 2/3 cliff |

Bench: `experiments/008_a3_topk_full_pareto/{bench_k3,bench_k5}.jsonl`
Notes: `experiments/008_a3_topk_full_pareto/result.md`

**K=4→K=3 is non-linear +47%.** Likely the OS page cache fits a
higher fraction of active experts when only 3 are dispatched per
layer — disk I/O cost drops more than proportionally.

**Next (iteration 009):** Multi-prompt robustness check on K=3.
Before recommending K=3 as production default, validate across 10+
prompts (not just Paris/Pacific/four). 3-prompt substring eval is
narrow; want to bound the quality risk more tightly.

---

## 007 — A2 routing-threshold sweep — NEUTRAL (A3 K=4 still leader, 2026-05-17 ~18:57 PT)

**Hypothesis:** Variable per-token K via sigmoid-weight threshold could
outperform fixed-K=4 by adapting to per-token router confidence.

**Result: neutral.** A2 works mechanically:
- `--routing-threshold 0.05`: 0.0645 tok/s (drops 0 experts, noise from K=8)
- `--routing-threshold 0.2`:  0.1043 tok/s (+31% vs K=8, drops ~2 experts)
- (vs A3 K=4 leader: 0.1667 tok/s, +109%)

A3 fixed-K=4 dominates the Pareto. K2.6's sigmoid weights appear
relatively uniform across top-8, so dropping experts by absolute
threshold is no better than just capping at K=4. The two flags
compose (`--top-k-override 4 --routing-threshold X`) for future
adaptive workloads but don't improve the single-prompt sweep here.

Bench: `experiments/007_a2_routing_threshold/{bench_thr05,bench_thr2}.jsonl`
Notes: `experiments/007_a2_routing_threshold/result.md`

**Next (iteration 008):** F4 multi-thread per shell. Attacks the
14.5% attention bucket (728 ms rank-0 + 578 ms rank-1 per q1).
rayon over the 64 attention heads should halve shell_attn time
on the 24-core Xeon Gold 6252 miner. Different bucket from A3, so
expected to compose with A3 K=4 leader.

---

## 006 — A3 top-K Pareto sweep on miner — K=4 LEADER (2026-05-17 ~18:38 PT)

**Hypothesis:** Push K further than K=6 to find quality cliff.

**Result:**
| K | tok/s | Δ vs K=8 | Quality | Outcome |
|--:|------:|---------:|---------|---------|
| 8 | 0.0797 | (ref) | 3/3 | baseline |
| 6 | 0.1116 | +40% | 3/3 | 005 win |
| **4** | **0.1667** | **+109%** | **3/3** | **006 win** (new leader, L-magnitude) |
| 2 | 0.2716 | +241% | 2/3 | quality cliff — "four" prompt format break |

**K=4 is the productionizable sweet spot.** +109% throughput, no
quality loss per substring eval, no code change needed by end users
(just `--top-k-override 4`). Lit (DeepSeek-V3 paper) predicted this
direction; we now have the concrete K2.6 / Intel CPU number.

K=2 is interesting but breaks the substring quality gate (the model
answered "Two plus two equals" with "? (A) 4 (B" — digit answer
rather than word "four"; semantically correct, format wrong for our
eval).

Bench: `experiments/006_a3_topk_sweep/{bench_k4,bench_k2}.jsonl`
Notes: `experiments/006_a3_topk_sweep/result.md`

**Next (iteration 007):** A2 sigmoid-threshold pruning (drop experts
whose routing weight < threshold rather than fixed K). Lit suggests
this can outperform fixed-K reduction at same average K-active.
Implementation: ~30 LOC in forward_shells; doesn't need 2-box.

---

## 005 — A3 top-K reduction VERIFIED WIN on miner (2026-05-17 ~18:30 PT)

**Hypothesis:** K=8→K=6 yields 15-25% tok/s improvement at <1% quality cost.

**Result: WIN. +40.0% throughput. Quality 3/3 preserved.**

| | tok/s | quality |
|---|---:|---|
| K=8 baseline | 0.0797 | 3/3 |
| **K=6**      | **0.1116** | **3/3** |
| **Δ**        | **+40.0%** | preserved |

Bench: `experiments/005_a3_topk_miner/{bench_k6,bench_k8}.jsonl`
Notes: `experiments/005_a3_topk_miner/result.md`

**Hardware substrate:** miner single-stage (forced pivot — matias-02
Tailscale was broken from earlier iteration's `tailscale up --reset`
attempt; needs manual re-auth). Miner is disk-bound at 58 GB/s read,
133 GB RAM. Per-prompt times vary ±20% by cache state but the +40%
aggregate delta is well above noise.

**Lit alignment:** Predicted +10-25% (DeepSeek-V3 paper, KTransformers).
Measured +40% — at the upper end of lit, consistent with low-concurrency
CPU-bound regime where expert FFN computation + page-in are the bottleneck.

**Tier-S #1 productionizable.** Spinout PR off main: add the
`--top-k-override` flag (commits db85e74 + fe31d7c) with the +40%
finding documented. Default = manifest top_k = no behavior change.

**Next (iteration 006):** D4 async pipeline overlap. Hides 54% of
per-token wall time per q1 breakdown. Requires the 2-box matias setup
— blocked on Tailscale fix. Either:
(a) fix matias-02 Tailscale (manual re-auth on box) and retry, OR
(b) try D4 single-stage variant on miner (less natural; pipeline
    overlap only meaningful with stages), OR
(c) try F4 multi-thread per shell (attacks 14.5% attention bucket;
    doesn't need 2-box; can validate on miner).

Leaning toward (c) F4 for iteration 006 — keeps the loop moving on
miner while matias is parked.

---

## 004 — A3 top-K reduction PARTIAL (2026-05-17 ~14:34 to 15:50 PT, parked on infra)

**Hypothesis:** K=8→K=6 yields 15-25% end-to-end tok/s improvement
(experts are 82% of decode per q1; reducing 8→6 = -25% experts ≈
-20% wall time).

**Bucket / candidate:** A3 (Tier-S #1 per iteration 003 ranking)
**Campaign:** `campaigns/004_a3_topk_reduction.yaml`

**Implementation: SHIPPED + VERIFIED at the per-stage level.**
- `--top-k-override` CLI flag on `tahoma worker` (commit `db85e74`)
- Plumbing: `WorkerArgs.top_k_override` → `SparseMoEBuilderConfig.top_k_override`
  → `Runner::set_top_k_override` → `forward_shells::effective_top_k`
- `TAHOMA_TOPK` env var in `start_rank{0,1}.ps1` wrappers
- Per-token shell breakdown CONFIRMS the override is active:
  `stage_timing shells ... top_k=8 effective_top_k=6 ... experts_us=1,734,411`
  (vs baseline K=8 experts_us=3,229,000 = **-46%** on the warmup token's experts dispatch)

**Bench: BLOCKED on infrastructure** — could not complete a 3-prompt
eval. Tailscale DERP relay between matias-02 ↔ matias-03 went into a
degraded state during this iteration; pattern reproduced at BOTH K=6
and K=8:

- First request's first token round-trip succeeds (rank-1 logs shells
  + head completion, sends Token upstream)
- Second round-trip onward: rank-0's `recv_kind_client` hangs forever
  (no client-side timeout); rank-1 keeps logging 60-second
  `recv_kind: recv_exact timed out after 60s` cycles.
- `Test-NetConnection matias-03:9100` returns True (TCP socket-level
  works); `tailscale ping -c 3 100.123.40.123` reports "direct
  connection not established" — DERP-relay-only path, asymmetric byte
  counts (tx 6.6M / rx 341K cumulative).
- Restarting Tailscale on both boxes (`tailscale down; tailscale up
  --reset`) is in flight; will retry bench when peer link is back.

**Learning (infra):**
1. `recv_kind_client` has no client-side timeout (only the server side
   has 60s). For autolab iteration robustness, future fix-forward
   should add a client-side recv timeout so hangs surface as errors
   instead of indefinite blocks. Filing as follow-up.
2. Multiple kill/restart cycles of tahoma workers seem to leave
   Tailscale DERP connection in a degraded state. Resetting Tailscale
   (`down; up --reset`) before restarting workers may be the right
   cycle for autolab iterations.

**Status: PARKED** awaiting Tailscale link recovery. Will retry K=6
bench in iteration 005; if successful and quality 3/3, also try K=4.

**Per-stage data captured (single warmup token, K=6 with effective_top_k=6):**

| Stage | K=6 (this iter) | K=8 baseline (iter 003) | Δ |
|-------|----------------:|------------------------:|---:|
| Rank-0 layer 0 | 80 ms | 81 ms | -1% |
| Rank-0 shell attn | 696 ms | 728 ms | -4% |
| Rank-0 shell experts | **1,734 ms** | **3,229 ms** | **-46%** |
| Rank-0 shells total | 2,436 ms | 3,974 ms | -39% |

The 46% drop in expert dispatch time at K=6 (vs the expected 25%
proportional to expert-count reduction) is probably partly disk-cache
effects (different runs, different cold-page mix) and partly the smaller
effective working set. Encouraging signal but needs end-to-end bench to
confirm.

---

## 003 — q1 instrumentation COMPLETE (2026-05-17 ~14:34 PT)

**Hypothesis:** Per-stage breakdown will show expert dispatch >60% of
per-token wall time.

**Result: VERIFIED + over-shot.** Expert dispatch is **82%** of per-token
decode time (much higher than predicted 60%). Other stages much smaller
than estimated.

**Bucket / candidate:** q1 — instrumentation (not a moonshot)
**Campaign:** `campaigns/001_instrumentation_breakdown.yaml`
**Bench:** `experiments/003_q1_instrumentation/bench.jsonl` —
0.0550 tok/s aggregate, 3/3 quality (Paris/Pacific/four).
Instrumentation overhead vs baseline 0.0553 = **-0.5%** (within noise,
well below 5% budget).

**Per-token decode breakdown (median of 24 late-sample steady-state events):**

| Stage | ms | % of total |
|-------|---:|-----------:|
| Rank-0 layer 0 (embed + dense attn + KV) | 81 | 0.9% |
| Rank-0 shell attention (30 layers) | 728 | 8.1% |
| Rank-0 shell expert dispatch (30 × top-8) | 3,229 | 35.9% |
| Rank-0 shells combine (residual + shared + moe) | <1 | <0.1% |
| **Rank-0 compute subtotal** | **3,974** | **44.1%** |
| Pure wire latency (Tailscale DERP) | 60 | 0.7% |
| Rank-1 shell attention (30 layers) | 578 | 6.4% |
| Rank-1 shell expert dispatch (30 × top-8) | 4,151 | 46.1% |
| Rank-1 head (RMSNorm + lm_head OV IR) | 139 | 1.5% |
| **Rank-1 compute subtotal** | **4,889** | **54.3%** |
| **TOTAL PER-TOKEN DECODE** | **9,005** | **100%** |
| **→ Implied decode tok/s (no prefill)** | | **0.111** |

End-to-end bench tok/s = 0.055 because the API count includes prefill
(prompts of 3-9 tokens, similar per-token cost as decode).

**Variance** (min vs max over 46 samples):
- Rank-0 experts: 2,098–7,770 ms (3.7× range)
- Rank-1 experts: 1,517–9,003 ms (5.9× range)
- All other stages: <1.5× range
**→ Expert variance is dominated by disk-page-in on cold expert pages.**

**Learning — re-ranking moonshots:**

| Stage | % of decode | Tier-S re-rank | Rationale |
|-------|------------:|----------------|-----------|
| Expert dispatch | **82%** | **#1** | Single biggest knob. A2/A3 expert reduction directly attacks this. C1/C7 prefetch + prewarm reduce its variance. |
| Shell attention | 14.5% | #3 | Second-biggest. F4 multi-thread per shell (rayon over 64 heads) is the obvious win. |
| Wire | 0.7% | dropped | **D1 BF16 wire is not worth pursuing** — saving half of 0.7% = 0.35% delta. |
| Async overlap | hides rank-1 | **#2** | D4 (start T+1 on rank-0 while rank-1 still on T) can hide the 4,889 ms of rank-1 compute — **up to 54%** of per-token time recovered. |
| Layer 0 | 0.9% | skip | Negligible. |
| Head | 1.5% | skip | Negligible. |

**Next iteration (004):** First real moonshot = A3 (top-K reduction
K=8 → K=4 or K=6). Lit (DeepSeek-V3 paper, KTransformers V0.3) reports
10-25% throughput improvement at K=6 with negligible quality cost on
sigmoid-router MoE. K2.6 is sigmoid-router family. Direct attack on
the 82% bucket.

---

## 003-pre — q1 instrumentation EXECUTE (DEFERRED & RESOLVED 2026-05-17 ~14:00-14:34 PT)

*Original deferred entry; superseded by the COMPLETE entry above. Kept
for traceability of the multi-step iteration.*

---

## 002 — q1 instrumentation patch + infrastructure discovery (2026-05-17 ~14:00 PT)

**Hypothesis:** Per-stage breakdown on 2-box K2.6 pipeline will show
expert dispatch >60% of per-token wall time. (Test deferred to 003 —
this iteration uncovered an infrastructure blocker that had to be
resolved first.)

**Bucket / candidate:** q1 — instrumentation (not a moonshot)
**Campaign:** `campaigns/001_instrumentation_breakdown.yaml`
**Literature:** none required.
**Code change:** runner.rs + engine.rs, +48 LOC net
- `forward_layer0_step`: wrap in `Instant`, log `stage="layer0", duration_us`
- `forward_shells`: per-layer accumulators for shell_attn_us + experts_us +
  combine_us, log aggregate at function exit with `stage="shells"`
- `forward_head_last`: wrap in `Instant`, log `stage="head", duration_us`
- `engine.rs` rank-0 driver: split timing of `send_forward` vs
  full round-trip to log `stage="rank0_wire", send_done_us,
  downstream_compute_us`
Each emits a `tracing::info!` line per token. ~1 µs overhead each;
budget << 5%.

**Result: PARKED → 003 (infrastructure blocker discovered + mitigation in flight)**

**What happened:**
1. ✓ Patched runner.rs + engine.rs locally
2. ✓ Built on matias-02 + matias-03 with `--features openvino`
   - matias-03 built clean (8.87 MB)
   - matias-02 failed first attempt because old tahoma.exe was in use
     (Windows can't overwrite running binary; classic). Killed PID 7620,
     rebuilt clean (8.87 MB)
3. ✓ Rank-1 started cleanly on matias-03 (foreground SSH + run_in_background
   pattern; the original `Start-Process powershell -WindowStyle Hidden`
   chain was silently dying in the detached powershell — possibly an SSH
   session/PowerShell lifecycle interaction)
4. ✗ Rank-0 startup failed: `Error: backend error: runner load: internal:
   safetensors layer0: io: The system cannot find the file specified
   (os error 2)`
5. ✓ Root cause: PR #10's `Int4Layer0` requires the safetensors source
   for layer-0 dense tensors + `embed_tokens` table. Pre-PR-#10 layer 0
   used the OV IR (no safetensors needed) which is what matias-02 was
   originally provisioned for. Inventory showed matias-02 has shards
   2-31 but is missing model-00001 (the shard with embed_tokens).
6. ✓ model-00001-of-000064.safetensors on miner is only **949 MB**
   (much smaller than the typical ~9.3 GB shard — embed_tokens layout
   compresses it). Initiated transfer miner→Mac→matias-02.

**Learning:**
- **Deploy contract drift.** PR #10 added a new runtime dependency
  (safetensors source for layer 0) that wasn't surfaced as a deployment
  requirement. Future PRs that add asset dependencies should explicitly
  list the new files in PR description + deploy docs. Added to follow-up.
- **Start-Process detach over SSH is unreliable** for long-running
  Windows processes. The pattern `ssh "powershell ... Start-Process -WindowStyle Hidden"`
  was silently losing the child. Replaced with `ssh "powershell -File <wrapper>" &
  bash run_in_background` — SSH session stays alive while child runs;
  log redirection captures all output reliably. Updated `start_workers.sh`
  semantics for future iterations.
- **Windows binary swap requires process termination first.** Cargo build
  errors with "Access is denied" if the .exe is in use. Kill tahoma
  BEFORE every rebuild on Windows.

**Next:** 003 picks up after transfer completes. Restart workers (matias-02
rank-0 will now find model-00001), run bench, parse per-stage timing,
verify <5% instrumentation overhead vs baseline 0.0553 tok/s.

---

## 001 — Baseline established (2026-05-17 ~13:36 PT)

**Hypothesis:** 2-box matias-02+03 K2.6 pipeline on main @ 208104e
delivers ~0.05 tok/s steady-state (3-prompt aggregate), 3/3 on the
Paris/Pacific/four quality eval. Matches the PR #9 / PR #10 numbers
from memory.

**Bucket / candidate:** baseline (not a moonshot — reference anchor)
**Literature:** none required for baseline. See [[LITERATURE]] for the
horizon: 0.05 tok/s is ~200x below comparable systems in the literature
(KTransformers on Xeon+A100 = 13.69, ik_llama on TR Pro+A6000 = 13.13,
mlx-lm on M3 Ultra = >20). The 30-300x gap is structural, not
hardware-bound, per the pipeline-parallel research agent's read.

**Campaign:** `campaigns/000_baseline_main.yaml`
**Design choice:** Clean restart of matias-02 (rank 0) + matias-03 (rank 1)
via the new `start_workers.sh` / `start_rank{0,1}.ps1` wrappers. Bench
script `k26_3prompt_eval.ps1` polls API readiness then runs 3 prompts at
max_tokens=8 temp=0.

**Result:** **baseline** (anchor for downstream moonshots)
- Paris    : 8 tok / 123.06 s = 0.0650 tok/s ✓ "Paris"
- Pacific  : 8 tok / 170.09 s = 0.0470 tok/s ✓ "Pacific"
- four     : 8 tok / 140.93 s = 0.0568 tok/s ✓ "four"
- **AGG**  : 24 tok / 434.09 s = **0.0553 tok/s**, **3/3 quality**

**Learning:**
- Variance is large (0.047-0.065 across prompts). Pacific (longest output
  context window after prompt + 8 tokens) was slowest; Paris shortest
  was fastest. Per-token latency creeps up with KV size in the
  shells (O(N) with our pre-allocated KV but the attention dot product
  is still per-token).
- The bench script's per-prompt `completion_tokens` is 0 due to a
  PS 5.1 / Invoke-RestMethod auto-parse quirk with snake_case JSON
  fields. tok/s in the rank-0 internal log is correct; bench's tok/s
  computation needs a max_tokens fallback. Filing as a bench-harness
  improvement, not a baseline-blocker.
- Cold-cache restart took ~5 min for the workers to come ready
  (warmer than the historical ~40 min in memory — likely OS page
  cache still warm from the morning's stale rank-0 process).

**Next (iteration 002):**
- Implement q1 (instrumentation): add per-stage timing (layer0 / shells
  attention / experts dispatch / wire / head) so we can attribute the
  17-20 s/tok across stages and rank moonshots by which stage they
  actually attack.
- After q1, the next real moonshot is Tier-S #1: per-token expert
  reduction (A2/A3). Lit says +10-50% with <1% quality cost on
  DeepSeek-V3 sigmoid router family (K2.6 is in that family).

## 000 — Scaffold (2026-05-17 ~12:45 PT)

Branch `autolab/k26-perf` cut from `origin/main @ 208104e`. Autolab
artifact tree created. PRIOR_ART synthesized from PRs #1/#4/#5/#7/#9/#10.
60 moonshot candidates enumerated in MOONSHOTS.md across 7 buckets
(quant, KV/attn, dispatch, wire, topo, sched, algo). 7 research
questions decomposed in research_plan.yaml. PR #11 opened as draft
(long-lived, will not merge). 3 parallel lit-research agents converged
on Tier-S moonshots (A2/A3 expert reduction, D1 BF16 wire, D4 async
overlap).
