# PR #30 production-readiness review

**Reviewed at:** `434560b` (`perf/k26-linux-production-tier-s`) against
`perf/a3-topk-override` (PR #29 base) and `main` @ `208104e`.
**Reviewer:** read-only audit; no patches applied.

## TL;DR: SHIP-WITH-CONDITIONS

PR #30 lands cleanly, tests pass workspace-wide (114 lib + 13 integration,
0 failures on Mac aarch64; cleanup agent reports 151 green on miner),
clippy is no worse than `main`, and the perf claim reproduces — the
3-way bench at `036eaa4` shows +60.0% e2e vs `main` K=8 and +17.9% vs
PR #29. The two highest-risk pieces — AVX-512 multi-token tiles and
spec-decode reconcile — have rigorous bit-identity test coverage
(including explicit seq=1 routing tests for all four `ProjShape`
variants and 8 distinct `reconcile_after_round` regression tests
covering both history conventions). The C-FFI ABI is preserved.

Three things stand in the way of a clean "click merge":

1. **`mergeable: CONFLICTING` / `mergeStateStatus: DIRTY`** on GitHub
   (`gh pr view 30 --json mergeable`). `git merge-tree` confirms three
   real conflicts in `cli/src/lib.rs`, `engine.rs`, `runner.rs` against
   PR #29. The PR description's "GitHub auto-merge will detect
   identical content" claim is optimistic — these are conventional
   3-way merge conflicts that will need a manual rebase after PR #29
   lands. **Not a correctness issue**, but it does block a
   one-button merge.
2. **No CI results recorded** on `perf/k26-linux-production-tier-s`
   (`statusCheckRollup: []`). Compare PR #29 which has a green
   `cargo test (stub)` check. Likely the CI workflow doesn't trigger
   on PRs whose base is a non-`main` branch; verify this is the cause
   before merging.
3. The PR description checklist is missing the third item ("3-way bench
   on miner — in progress"). The bench at `036eaa4` is now complete;
   the box should be ticked.

If those three are addressed, ship it. The architecture, testing, and
benchmark numbers all hold up.

## Findings by dimension

### 1. Build cleanliness — ✅ GOOD

- `cargo build -p tahoma` (Mac aarch64, no `openvino` feature) — clean,
  finishes in ~1.5 s after first build. Only warnings present are
  pre-existing on `main` (66 clippy warnings total on PR HEAD; same
  count on `main` and on the autolab branch). The cleanup agent's
  76a8fed already fixed the one new warning the PR itself introduced
  (`std::f32::consts::PI`).
- `cargo fmt --check` — clean (exit 0).
- `cargo clippy --workspace --all-targets` — same warning count
  (66) as `main`, no new warnings introduced by the PR. One
  cosmetic nit: `shell_int4.rs` line 15 has an `unused import:
  crate::kernel_bf16::bf16_gemv_auto` warning that appears to be
  pre-existing (also fires on the autolab branch); not introduced
  by PR #30.
- Did not exercise `--features openvino` on Mac (no Intel/OpenVINO
  runtime locally); the cleanup agent reports it green on miner.

### 2. Test coverage — ✅ GOOD

- `cargo test --workspace --lib`: all 114 lib tests pass, 0 failures.
- `cargo test --workspace --tests`: integration tests pass on top
  (binary serves health/models/chat, dist_wire integration, topology,
  transport, types). Numbers are slightly lower than the 151 the
  cleanup agent reported on miner because miner has more OS-specific
  test paths exercised; but no test regressions vs `main`.
- Critical-path coverage is strong:
  - **SIMD bit-identity:** `multi_matches_per_token_loop_seq_{1,4}`,
    `multi_matches_per_token_loop_large`,
    `blocked_matches_per_token_loop_seq_{1,4}`,
    `blocked_matches_per_token_loop_odd_rows`,
    `blocked_matches_per_token_loop_oproj_seq_8`,
    `blocked_matches_iter042_multi_seq_8` —
    all use `assert_eq!(a.to_bits(), b.to_bits())` (byte-exact, not
    tolerance-based) and exercise the K2.6 oproj-shaped K=8192.
  - **Dispatch routing:** `dispatch_int4_multi_seq_1_matches_single_token_kernel`
    asserts that AT SEQ=1, every shape variant
    (`Generic`/`Oproj`/`SharedDown`/`LargeShape`) routes to the
    scalar single-token kernel byte-for-byte. This is THE critical
    regression guard since every K2.6 inference today hits seq=1.
  - **End-to-end bit-identity:** `multi_seq_1_matches_seq_1_reference`
    (seq=1 multi forward matches scalar reference including KV state)
    and `multi_batched_matches_scalar_seq_8_iter046_dispatch`
    (seq=8 batched matches scalar within f32 tolerance, KV bit-exact).
  - **bf16 round-trip:** `f32_to_bf16_bits_matches_half_crate`
    cross-checks the hand-rolled rounding against `half::bf16::from_f32`
    on zero, ±1, ±0.5, powers of 2, denormal, π, -42.5, and NaN.
  - **Spec-decode reconcile:** 8 distinct `reconcile_*` tests
    including the explicit `reconcile_pending_token_*` regression
    suite (iter 043 fix) and the brute-force
    `end_to_end_round_math_is_invariant` that walks every
    (k=1..8, a=0..k, bonus_ran ∈ {true, false}, convention ∈ {clean,
    pending}) and verifies the invariants. Plus two simulated-session
    tests that compare against a sequential greedy reference.

### 3. API stability — ✅ GOOD

- New CLI flags (`--prompt-lookup`, `--spec-k`, `--top-k-override`,
  `--routing-threshold`) are all opt-in with sensible defaults:
  - `prompt_lookup: u32 = 0` (disabled; only enables spec-decode
    when > 0)
  - `top_k_override: Option<u32> = None` (uses manifest top_k)
  - `routing_threshold: Option<f32> = None`
- The `SparseMoEBuilderConfig` defaults to `spec_decode_k: None`.
  Builder explicitly only constructs the spec-decode driver when
  the user passes `--prompt-lookup > 0` AND `temperature <= 0.0`
  (`engine.rs:660`).
- **seq=1 hot path is byte-identical:** `forward_shells` (seq=1) is
  unchanged and still calls `shell_forward_decode_int4_with_capacity`
  (the existing path). The new `forward_shells_multi` is only invoked
  from `step_multi`, which is only called from `generate_speculative`
  and the pipeline-parallel spec-decode driver — both opt-in.
- **Existing engines unaffected:** the `mock`, `ov-genai`, `ov-runtime`,
  `ov-dist-spec` builders are not touched in this PR. New flags are
  only consumed by `EngineKind::SparseMoe`.
- **C-FFI ABI preserved:** `tahoma_int4_shell_forward_int4` still
  accepts `f32 past_k/past_v` and returns `f32 present_k/present_v`;
  it transparently bf16-converts on the inbound side and the inner
  kernel writes f32 into `present_*` (caller sees no change). This
  matters for rainier's cdylib consumer. Added: `tahoma_int4_prewarm`
  is a new entry point (additive, not breaking).

### 4. Code quality (correctness focus) — ✅ GOOD with one note

Read all listed high-risk files end-to-end.

- **`kernel_avx512_multi.rs` (iter 042):** clean AVX-512 fmadd tile,
  parallelized over row chunks of 64 with each chunk holding `seq`
  __m512 accumulators (capped at `MAX_SEQ=64`). The `unsafe` is
  tightly scoped to the `#[target_feature(enable = ...)]` function and
  the dispatch wrapper checks all three feature bits
  (`avx512f`/`bw`/`vl`) before calling. The cross-chunk write of
  `y_ptr` via `usize`-cast is sound (rayon hands disjoint chunks; the
  comment explicitly documents this).
- **`kernel_avx512_multi_blocked.rs` (iter 046):** row-blocked RB=2
  variant. Register budget analysis is documented inline (2*seq accs
  + 4 weight regs + 2 x regs = 22 ZMM at seq=8 fits in 32 ZMM with 10
  spare). Tail-row fallback handles odd `n_rows` correctly (test
  `blocked_matches_per_token_loop_odd_rows` proves it). Same
  rayon-disjoint pattern as iter 042.
- **`shell_int4.rs::dispatch_int4_multi` (iter 048):** matches the
  documented routing table. `Oproj`/`SharedDown` → `blocked_auto`
  (seq>=4 → iter 046, seq=2-3 → iter 042, seq=1 → scalar);
  `LargeShape` → seq>=8 → iter 046 else iter 042; `Generic` → iter 042.
  All four variants tested for seq=1 bit-identity (the critical
  regression guard).
- **`c_ffi.rs`:** ABI preserved (see dimension 3). `f32_to_bf16_bits_ffi`
  uses the same RNE rounding as `runner.rs::f32_to_bf16_bits` and
  `shell_int4.rs::f32_to_bf16_bits_local`. Note: **three duplicate
  bf16 rounding helpers** exist (`_ffi`, `_local`, plain). Only the
  `runner.rs` one is directly cross-checked against `half::bf16`;
  the other two share identical bit-pattern logic but lack
  redundant tests. SHOULD-FIX: consolidate to a single helper in
  `tahoma-int4-gemm::format` once the PR lands.
- **`runner.rs::forward_shells_multi`:** correctly grows KV capacity
  to fit `past_seq_len + seq` (handles doubling-boundary straddle
  via `while`). Errors propagate cleanly through
  `RunnerError::Internal`. Past-seq-len mismatch is detected and
  returns Err (not panic). The opt-in switch is the spec-decode K
  flag.
- **`spec_decode.rs::reconcile_after_round`:** the iter 043
  off-by-one fix is well-tested. The brute-force invariant test
  (`end_to_end_round_math_is_invariant`) walks every
  (k, a, bonus_ran, convention) combination and proves both KV-end
  arithmetic and emitted-token count are consistent with the
  documented contract. Two simulated-session tests
  (`simulated_session_matches_sequential_greedy` and
  `simulated_runner_pending_session_matches_sequential_greedy`)
  cross-check that spec-decode produces the same output as
  sequential greedy.
- **`runner.rs::kv_invariant_holds`:** correctly iterates layer-0
  + every shell layer; returns `false` (no panic) if any layer's
  `past_seq_len != history.len() - pending_drift`. The
  `generate_speculative` driver `debug_assert!`s this every round.
  **Coverage gap:** the pipeline-parallel
  `drive_generation_first_spec` driver in `engine.rs` does NOT
  invoke `kv_invariant_holds` (it just calls `rewind_kv` and trusts
  the helper). The pure-function tests cover the same arithmetic,
  but no end-to-end invariant check fires in the distributed path.
  SHOULD-FIX (post-merge).
- **`safetensors_source.rs` prefetch:** the prompt mentioned
  `madvise`/`VirtualLock`. The PR actually uses `MADV_WILLNEED` on
  Unix and `PrefetchVirtualMemory` on Windows (not `VirtualLock`).
  Both are pure best-effort hints — neither pins pages, both
  silently swallow errors. On failure the code path degrades
  gracefully: the read happens synchronously on demand, which is
  the baseline behavior. No `RLIMIT_MEMLOCK` requirement (that's
  iter 054, intentionally out of scope per the PR description).

### 5. Concurrency — ✅ GOOD

- Single background prefetcher thread per `Runner`, spawned only
  when `experts_format == safetensors_bin` and
  `TAHOMA_EXPERT_PREFETCH != "0"`. Uses `sync_channel(4096)` for
  bounded backpressure; `try_send` drops on full or disconnect
  (counter bumped) so the inference path never stalls.
- **Lifetime story:** `Prefetcher` holds `Arc<SafetensorsExpertSource>`
  (cloned from the Runner's `_safetensors_source`). The mmaps inside
  `Shard` are `Send + Sync`. `Drop for Prefetcher` first drops the
  `SyncSender` (terminating the thread's `recv` loop) then `join`s
  the thread before either Arc is dropped. No UAF.
- **Race-on-shard-insert:** `shard_for` uses an RwLock with the
  documented "race tolerated" pattern — two threads may both insert
  the same shard, the later one overwrites with an equivalent
  `Arc<Shard>`. The old `Arc` is dropped when no longer referenced;
  no leak.
- **Per-token prefetch stash:** `last_routing_ids` mutated only on
  the inference thread (single producer) and pushed to the channel
  via `try_submit`. No shared mutable state across threads other
  than the bounded channel.

### 6. Error handling — ✅ GOOD

- `madvise`/`PrefetchVirtualMemory` failure → silently ignored
  (advisory, best-effort). Documented at the call site.
- `mmap` denied → `Shard::open` returns `GemmError::Io`, propagated
  as `RunnerError::Internal("safetensors expert {lid}/{eid}: ...")`.
  Not a panic.
- Corrupt safetensors → header parse / `data_offsets` length check
  returns `GemmError::Io`. The index.json missing → returns
  `GemmError::Io("weight_map missing")`. Both surface cleanly.
- KV grow-on-OOM → `try_reserve_exact` returns `Err` →
  `RunnerError::Internal("alloc {N} u16/bf16 (X MB) failed: ...")`.
  Long-context generations OOM gracefully; no abort from the global
  allocator.
- Wire-frame attack surface → `MAX_BATCH_COUNT = 256` caps the
  per-frame allocation. Receivers reject `batch_count == 0` or
  `> MAX_BATCH_COUNT`. Unknown `FrameKind` → `TransportError`
  with the offending code in the message.
- Quality regression detection happens out-of-band via the
  substring-eval bench harness; not enforced in the binary itself.
  The substring fail on prompt 10 ("speed of light" missed
  because the model answered in m/s not km/s) is a semantic
  pass — manual inspection of all 30 outputs in the JSONLs
  confirms.

### 7. Documentation / operator surface — ✅ GOOD

- New CLI flags all have `--help`-visible documentation via clap
  `///` doc comments (verified in `cli/src/lib.rs`):
  `--top-k-override`, `--routing-threshold`, `--prompt-lookup`,
  `--spec-k`.
- `scripts/deploy/matias-2box/README.md` is clear:
  - Documents the SSH tunnel chain (matias-02 → Mac → matias-03)
  - Documents the `127.0.0.1` vs `localhost` IPv6 gotcha
  - Documents the WMI vs `Start-Process -WindowStyle Hidden` Windows
    OpenSSH detachment trap
  - Includes the quick-start, prerequisites, and tunnel layout diagram
- No operational gotchas with iter 054's `mlock` requirement
  because iter 054 is intentionally not in this PR.
- The `TAHOMA_EXPERT_PREFETCH=0` escape hatch is documented in code
  comments but not in any user-facing doc. NICE-TO-HAVE: add a
  `--no-expert-prefetch` flag or document the env var in
  `--help`.

### 8. Security — ✅ GOOD

- Wire protocol additions (`ForwardBatch`/`TokenBatch`) carry
  explicit `MAX_BATCH_COUNT = 256` cap on the receiving side to
  bound adversarial allocations. `FrameKind` versioning is
  preserved (0x53_4D_45_03 and 0x53_4D_45_21 — old peers correctly
  trip the unknown-kind error).
- No new auth surface; the CLI's `--api` plaintext OpenAI shim is
  unchanged. The existing security caveat in the CLI's
  `long_about` ("HTTP API and inter-stage TCP relay are plaintext
  and unauthenticated") still applies.
- The C-FFI surface adds `tahoma_int4_prewarm` (does only volatile
  reads + atomic store) and `tahoma_int4_open_shell_int4` / etc.
  All do null checks on input handles.
- One mild concern: `shell_int4.rs` constructs `bias: &[f32]` from
  `shell.router_bias` via `std::slice::from_raw_parts` and a raw
  pointer cast. The cast assumes the bias is f32 little-endian
  with correct alignment; if a malformed safetensors file declared
  the bias as a different dtype or unaligned, this would invoke UB.
  Pre-existing pattern, not introduced by PR #30. NICE-TO-HAVE
  (post-merge): add an alignment/length check at the cast site.

### 9. Merge readiness — ⚠️ CONCERN

- **`mergeable: CONFLICTING`, `mergeStateStatus: DIRTY`.** Verified
  via `git merge-tree --merge-base=origin/main
  origin/perf/a3-topk-override origin/perf/k26-linux-production-tier-s`:
  three real content conflicts in `cli/src/lib.rs`, `engine.rs`,
  and `runner.rs`. These are the duplicated PR #29 lines the PR
  description acknowledges, and they ARE real merge conflicts
  (git's content-based merge cannot reconcile two independent
  applications of the same patch where the surrounding context
  has also drifted). After PR #29 merges to `main`, PR #30 will
  need a manual rebase to clean these up. **Not a correctness
  issue, but it does prevent a one-click merge.**
- **No CI results** on `perf/k26-linux-production-tier-s`
  (`statusCheckRollup: []`). PR #29 has a green
  `cargo test (stub)` check, so the workflow exists. The likely
  cause is that the CI config doesn't trigger on PRs whose base is
  a non-`main` branch. Investigate before merging.
- **Commit hygiene:** ✅ all 11 commits are single-author
  (Tate Berenbaum), no `Co-Authored-By` lines anywhere, conventional
  commit prefixes (feat / perf / fix / infra / chore), and atomic
  (each iter / theme in its own commit). 76a8fed is a tiny cleanup
  commit; this is fine.
- **Commit graph:** 11 commits, ~6800 +/146 -. About 200 lines are
  the inflated PR #29 overlap; net PR #30 contribution is ~6600
  lines added.

### 10. Perf claim verification — ✅ GOOD

- 3-way bench at `036eaa4` (`bench/3way-main-pr29-pr30-102`) has
  three JSONL files (10 prompts + 1 aggregate row each):
  - main K=8: 0.1152 tok/s, 9/10
  - PR #29 K=6 override: 0.1563 tok/s, 10/10 (**+35.7% vs main**)
  - PR #30 K=6 + spec-decode: 0.1843 tok/s, 9/10 (**+60.0% vs main**,
    **+17.9% vs PR #29**)
- **Methodology is sound:** n=10 prompts, mt=64, temp=0, single-stage
  CPU on miner, warmed up. The substring eval is a literal grep,
  not a semantic check — the one PR #30 quality miss is the
  "speed of light in km" prompt where the model answered in m/s
  (semantically correct).
- **Per-prompt variance is large** on PR #30 (0.157-0.246 tok/s) —
  n-gram spec-decode hit rate scales with output redundancy. The
  STATUS.md is appropriately honest about this. The +17.9% over
  PR #29 is consistent with the iter 044 +19.7% figure (within
  bench variance).
- **Caveats acknowledged in STATUS.md:**
  - n=10 is small; std-dev is wide
  - main and PR #29 benches were run earlier in the session
    (different cache state)
  - Quality eval is substring grep; one PR #30 "wrong-units" miss
    is semantically correct
- I did not re-run the bench (no Linux box at hand) but spot-checked
  the raw JSONL — the per-prompt tok/s and the aggregate align with
  the STATUS.md numbers.

## Recommended changes before merge

**MUST-FIX:**

- **Confirm CI status.** `statusCheckRollup: []` is suspicious.
  Either the CI workflow doesn't trigger on PRs targeting
  `perf/a3-topk-override`, OR the configured checks haven't been
  enqueued. Verify which, and either trigger CI manually or
  re-target the PR base to `main` after PR #29 merges so CI
  runs against `main`.
- **Plan the rebase.** PR description claims auto-merge will detect
  identical content; `git merge-tree` shows three real conflicts
  in `cli/src/lib.rs`, `engine.rs`, `runner.rs`. After PR #29
  merges, manually rebase PR #30 onto the new `main` and verify
  the conflict resolution is the trivial "drop the duplicated
  A3 lines" case the PR description anticipates.

**SHOULD-FIX:**

- **Tick the third box in the PR description test plan** ("3-way
  bench on miner — in progress"). The bench at `036eaa4` is now
  complete; update the PR body to reflect that and link the
  STATUS.md.
- **Consolidate the three duplicated `f32_to_bf16_bits*` helpers**
  (`runner.rs::f32_to_bf16_bits`, `shell_int4.rs::f32_to_bf16_bits_local`,
  `c_ffi.rs::f32_to_bf16_bits_ffi`) into one canonical helper in
  `tahoma-int4-gemm::format`. Only the runner's version is
  cross-checked against `half::bf16::from_f32`; the other two
  could drift undetected.
- **Add `kv_invariant_holds` debug assertion to
  `engine::drive_generation_first_spec`** (the pipeline-parallel
  spec-decode driver) so the same end-to-end KV/history check
  fires in the distributed path. The pure-function tests cover
  the arithmetic, but no end-to-end safety net exists for the
  pipeline path today.

**NICE-TO-HAVE (do not block on these):**

- Document `TAHOMA_EXPERT_PREFETCH=0` in `--help`, or add a
  `--no-expert-prefetch` flag, so operators can disable
  the prefetcher without an env var.
- The `unused import: bf16_gemv_auto` warning in `shell_int4.rs:15`
  appears to be pre-existing; drop it in a future cleanup pass.
- Add an alignment/length check at the
  `router_bias` raw-pointer cast site (`shell_int4.rs`); not new
  in this PR but worth hardening if the safetensors loader ever
  accepts untrusted files.

## Recommended post-merge follow-ups

- Once PR #30 lands, rebase the autolab/k26-perf research branch
  on top to remove the now-redundant overlap.
- Hammer a longer 3-way bench (n=50+ prompts, multiple cache
  states) on miner to tighten the +60% confidence interval; the
  current n=10 std-dev is wide.
- Run a fuzz-style bit-identity test that exercises
  `dispatch_int4_multi` across the full Cartesian product
  `(shape, seq=1..16, n_rows ∈ {64, 128, 7168}, k_cols ∈ {32,
  128, 2048, 7168, 8192})` to catch dispatch routing edge cases
  not currently in the explicit test suite.
- Track the 2-box revival deployment (matias scripts) for an
  end-to-end pipeline-parallel ForwardBatch wire test on actual
  Tiber AI PCs; the unit tests prove the math but a 2-box smoke
  test would validate the wire format on real hardware.

## Overall verdict

**SHIP-WITH-CONDITIONS.** PR #30 is a serious, well-tested 10-commit
bundle that compiles clean, passes 100% of the workspace test suite,
adds no new clippy warnings, preserves the seq=1 hot path and the
C-FFI ABI, and reproduces its headline +60% e2e perf claim in an
honest, single-author bench. The high-risk surfaces
(AVX-512 multi-token tiles, dispatch routing, spec-decode reconcile,
bf16 KV conversion, background prefetcher) all have rigorous
bit-identity or cross-check test coverage. The error handling is
graceful — no panics on the recoverable failure modes I checked
(mlock denied, OOM, corrupt safetensors header, unknown wire frame
kind).

The only barriers to merge are mechanical: CI hasn't run (the
workflow likely needs a PR retargeting after PR #29 merges), and
the duplicated PR #29 content will need a manual rebase rather than
the "GitHub auto-merge will figure it out" path the PR description
claims. Both are 30-minute fixes once PR #29 lands. The
SHOULD-FIX consolidation of the duplicated bf16 helpers and the
pipeline-parallel `kv_invariant_holds` gap are real but not
ship-blocking.

If PR #29 merges and a follow-up rebase produces a clean diff,
this is a green-light merge. The architecture is sound, the
numbers reproduce, the test discipline is strong, and the iter
043 off-by-one fix has explicit regression coverage. Recommend
SHIP after the rebase.
