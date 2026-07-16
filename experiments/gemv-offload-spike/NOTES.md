# GEMV-offload spike: CascadiaInt4Gemv extension op (route 1)

**Question:** can the hybrid split's decode side execute its sym-INT4 weight
matmuls straight from the mmapped IR `.bin` — on OpenVINO's public extension
API, no OV changes — removing the decode model's plugin-resident weight copy?

**Answer: the mechanism works end-to-end and is token-parity-correct at ~70%
of stock decode throughput; the steady-state residency win on the CPU plugin
is SMALLER than theorized (see Honest surprises).**

## What was built

- `cpp/gemv_offload.cpp` (shim): a load-time `MatcherPass` rewrites every
  NNCF sym-INT4 chain `Const(i4 [N,G,g]) → Convert(f16) → Multiply(scales
  f16 [N,G,1]) → Reshape([N,K]) → MatMul(transpose_b)` into a
  `CascadiaInt4Gemv` op holding the weight/scale Constants as MEMBERS —
  invisible to the plugin compile, so no repack; the shared_ptrs keep the
  read_model mmap alive. Registered in-process via `ov::OpExtension`.
- Kernel: grouped sym-INT4 GEMV, f32 accumulation, `ov::parallel_for` rows;
  AVX2+FMA path (nibble sign via `(x^8)-8`, `unpacklo/hi_epi8(lo,hi)`
  restores low-nibble-first element order) with scalar fallback.
- `cascadia worker --gemv-offload` (ov-runtime, stateless static export,
  `--device CPU` only); decode compile drops CACHE_DIR (op members don't
  survive blob serialization). Bisect knobs `CASCADIA_GEMV_MAX/SKIP`.

## Measured (pawan-01, LNL 258V, Llama-3.2-1B INT4, all 113/113 MatMuls offloaded)

- **Parity:** token-identical to the stock kernel across tokenwise/chunked/
  hybrid legs, full generations (cross-kernel greedy match held despite
  differing accumulation order).
- **Throughput:** decode 12.4–13.9 tok/s vs 18.9 stock (~70%; scalar kernel
  was 5.4). Hybrid leg: TTFT 189.9 ms + 13.9 tok/s.
- **Memory (decode model only, hold probe, private bytes | working set):**
  stock steady 0.11 GB | 0.18 GB, stock PEAK-private 1.08 GB (compile);
  offload steady 0.34 GB | 0.89 GB, peak-private 0.34 GB.

## Honest surprises

1. **Stock CPU steady-state is already cheap for this IR.** Steady private
   of 0.11 GB says OV 2026.1's CPU executor does NOT hold a large private
   repacked copy of these decompress-subgraph weights at steady state
   (weights appear to stay constant/mmap-backed in this executor path) —
   contrary to the general prepareWeightsMemory repack expectation our
   research derived for the FC path. The offload's residency value on CPU is
   therefore: **~0.7 GB lower compile-time private peak** and **file-backed
   (evictable, cross-process shareable) instead of private pages** — not a
   large steady RSS delta. Re-verify per OV version/arch before leaning on
   either behavior.
2. Three integration traps cost most of the spike (each reproducible,
   worth knowing for ANY evaluate()-op extension):
   - **Precision pipeline:** the plugin retypes the graph (f16 IR → f32 CPU
     execution) but cannot retype a custom op — output type must FOLLOW the
     input, and evaluate() must handle f16 and f32 on both sides.
   - **Attribute-blind CSE:** with weights in op members, two ops with equal
     dims and the same input (a layer's k_proj/v_proj) compare EQUAL and get
     merged — v silently received k's output. Fix: a per-instance
     `weights_tag` attribute.
   - (Avoided by review, worth stating) `ov::parallel_for` blocks its caller
     and TBB work-steals other evaluates onto that thread — never share
     thread_local scratch across evaluate calls.
3. Debug flow that cracked it: numpy-vs-OV ground truth on real weights
   (kernel math exonerated in one run) + `CASCADIA_GEMV_MAX/SKIP` bisection
   (q,k ✓ / +v ✗ / v-alone ✓ ⇒ pairwise-merge, not math).

## Throughput push (second pass): 66-73% → 75-79% of stock, frontier mapped

Every step measured on hardware, parity re-asserted each time:

| Config | decode tok/s | kernel eff GB/s | note |
|---|---|---|---|
| scalar kernel (spike v1) | 5.4 | ~2.6 | |
| AVX2 single-acc | 12.4–13.9 | 19.1 | FMA-latency bound |
| + 4 accumulators, blocked parallel_for | **14.2 (default)** | 21.3 | |
| private TBB-independent pool | 11.8 | 15.4 | REJECTED: wake latency + oversubscription vs plugin threads |
| `INFERENCE_NUM_THREADS=8` (add E-cores) | 11.2 (stock drops too: 16.9) | 14.8 | REJECTED: E-cores gate every fork-join lane; OV's P-core-only LATENCY default is right |
| + AVX-VNNI dpbusd, dyn-quant int8 acts (`CASCADIA_GEMV_VNNI=1`) | **15.0 / 14.9 hybrid (79%)** | 22.1 | opt-in: dynamic quantization — parity HELD on the test prompt, but numerics differ by construction; keep off by default |

Key measurements behind the frontier:
- Single-core kernel (SEQ=1): 5.7 GB/s → the 4-P-core arena scales 3.7×
  (LATENCY hint = P-cores only on LNL, confirmed as the right choice).
- VNNI barely moved aggregate GB/s (21.3→22.1): per-core is no longer
  compute-bound — per-node fork-join + memory-level parallelism dominate.
- Floor analysis at 15 tok/s: kernel-wall 27.9 ms/token; non-kernel ≈ 39 ms
  (stock total: 52.9). Parity needs kernel ≈ 14 ms (≈ 44 GB/s) or fewer,
  larger nodes. **Identified next lever (follow-up, not this spike):
  graph-level fusion in the pass — one op for q/k/v and one for gate/up
  (shared inputs, concat + VariadicSplit) → 113 → ~66 nodes, halves
  fork-join count and doubles mean GEMV size.** Hybrid TTFT improved to
  186 ms along the way (best measured).
- The VNNI trick worth keeping: `w+8 == nibble ^ 8`, and dpbusd(u8,s8)
  accumulates exactly in i32 (no maddubs-style i16 saturation); bias removed
  per group via `− 8·Σq` with Σq computed once per token.

### Sibling fusion (third pass): built, measured NEUTRAL — hypothesis refuted

The op now supports multiple weight/scale SEGMENTS sharing one activation
and K (a layer's q/k/v → one op with N=3072; gate/up → N=16384), with a
post-rewrite surgery stage grouping siblings and splicing a VariadicSplit to
route slices back to the original consumers (`CASCADIA_GEMV_NOFUSE=1` A/B
knob). It works exactly as designed — calls/token 113 → 65, parity held —
and throughput DID NOT MOVE (fused 14.23 vs 14.16 tok/s; fused+VNNI
14.84/14.72 vs 15.0/14.9; kernel eff ~21-22 GB/s unchanged). Conclusion:
per-node fork-join was NOT the floor. Revised frontier:

1. The kernel wall (~29 ms/token) is a **~22 GB/s weight-streaming ceiling
   on the 4 P-cores** for this access pattern — compute-insensitive (VNNI
   flat) and node-count-insensitive (fusion flat). Stock's repacked blocked
   layout streams better (prefetch locality). Next levers here: software
   prefetch tuning, non-temporal loads, per-thread contiguous weight
   striping — or conceding the hot loops to a repacked cache (which
   reintroduces residency, defeating the point).
2. The remaining ~27 ms/token graph-side delta vs stock needs per-node
   attribution (OV `PERF_COUNT` profiling exposed through new shim FFI)
   before more guessing — candidates: broken eltwise fusions around custom
   ops, per-op output allocation, reference-node dispatch.

Kept fusion enabled by default anyway: fewer nodes, no cost, and it is the
foundation any future epilogue-fusion work builds on.

## PERF_COUNT attribution (fourth pass): the gap fully named

New instrumentation (kept): `Runtime::profiling()` FFI over OV
`get_profiling_info()` + `CASCADIA_PERF_DUMP=1` engine dump (compile with
`PERF_COUNT=YES` via `CASCADIA_OV_PROPS`). Per-node profile of one decode
infer, stock vs offloaded (times inflated by profiling; use ratios):

> **NPU blob-cache import warning (bitten once, cost a re-sweep):** an NPU
> compiled model IMPORTED from the blob cache defers ~300 ms of driver init
> to its FIRST inference (1B/3B prefill blobs, LNL driver 2026.1-era) — a
> cold in-process compile does not. First-request TTFT measured through a
> cache-imported prefill blob reads ~+300 ms high (206→~505 ms short,
> 687→~990 ms long on 1B — reproduced with both a PERF_COUNT-era and a clean
> cache, which is what exonerated PERF_COUNT "poisoning" as the first
> theory). Bench either with a cold compile or on request ≥2 of a warm
> process; production long-lived workers only pay it once at startup.

| node kind | stock | offloaded |
|---|---|---|
| weight matmuls | FullyConnectedCompressed ×113: **11.2–12.9 ms** (brgemm_avx2_f32, ≈51 GB/s) | Reference ×65 (our ops): **31.9 ms** (≈22 GB/s) |
| attention (SDPA subgraphs) | 13.5–16 ms | 13.9 ms (identical) |
| Broadcast (GQA expand) | 3.7–4.2 ms | 3.8 ms |
| Concat/RoPE/RMS/Reorder/Convert | ~1.8 ms | ~1.9 ms |

Two prior beliefs corrected by this data:
1. Stock matmuls are FAST (11–13 ms, not ~45): oneDNN's
   weights-decompression brgemm streams packed INT4 at ~51 GB/s on the 4
   P-cores.
2. The earlier "~27 ms unexplained graph overhead" DISSOLVES: ~25 ms/token
   of non-graph cost (host KV-ring set_input/present copies ≈ 67 MB/token +
   engine glue) is SHARED by both paths. There are no extra
   Converts/Reorders around the custom ops. The offload's entire deficit is
   kernel streaming: 31.9 vs 11.2 ms.

Kernel-side levers, all built and all measured ~flat (single-core stuck at
5.7→6.1 GB/s; aggregate ~22 GB/s):
- flat no-helper loops + __forceinline (kills hypothetical MSVC vector
  spills): +7% single-core only
- software prefetch (+128B ahead), 2-row register blocking (one act read
  feeding two weight streams), VNNI unroll-by-2 without the alternation
  branch: flat

Conclusion: our loop shape hits a genuine ~6 GB/s/core ceiling on MSVC/Lion
Cove that resists compute, ILP, MLP, and inlining fixes, while oneDNN's
brgemm does ~12.8 GB/s/core from an equivalent packed-INT4 stream. The two
credible endgames, in order:
1. **Embed oneDNN in the op**: build a dnnl matmul primitive with INT4
   weights-decompression args pointing at OUR mmapped weight memory —
   oneDNN-quality streaming, residency goals intact. A dependency/build
   project (the plugin's oneDNN isn't exported to extensions), not a kernel
   tweak. This is the recommended follow-up if the last ~4 tok/s matters.
2. The upstream RFC (docs/rfcs/): make OV's own executor do this
   contractually.

Also visible in the profile and relevant beyond this spike: attention
subgraphs (13–16 ms) + the shared ~25 ms/token ring/copy overhead are the
static path's real whole-pipeline costs — bigger prizes than the remaining
matmul delta, and they apply to STOCK too (separate workstream).

## oneDNN-embedding endgame (fifth pass): CLOSED BY DATA — the fast kernels are fork-only

Built the full embedding (kept, feature-gated `DNNL_DIR` at build /
`CASCADIA_GEMV_DNNL` at runtime, permanent fallback to built-in kernels on
any failure): dnnl 3.12 (conda-forge win-64, TBB runtime sharing the SDK's
tbb12) matmul primitives with INT4 weights-decompression created inside the
op — weights zero-copy over our mmapped `[N,K]` packed-i4 rows (dnnl plain
`ba` for dims `[K,N]`), scales transposed once to resident `[G,N]` f16
(~4 MB), `fpmath_mode(f16, apply_to_int)` + grouped weight scales
(mask 3, groups `{gsize,1}`).

Results:
- **Numerically correct end-to-end** (token-parity held through full
  generations — nibble order, scales, layout all right).
- **Mode 1 (our layout, zero-copy): `ref`-slow** — ~165 ms/op, 0.1 GB/s.
- **Mode 2 (`format_tag::any` + one resident reorder — dnnl's own layout
  choice): STILL `ref:any`.** Upstream oneDNN 3.12 has NO optimized kernel
  for s4 grouped-scale weights-decompression matmul on this shape at all.

Conclusion: OpenVINO's ~51 GB/s `FullyConnectedCompressed` kernels live in
the **openvinotoolkit/oneDNN fork** (and/or OV-internal JIT), not upstream
oneDNN. Embedding upstream dnnl cannot reach the plugin's throughput for
this workload, with any layout, at any residency cost. The remaining routes
to closing the last ~21%:
1. The upstream RFC (docs/rfcs/) — now with sharpened evidence: the
   capability demonstrably exists in Intel's fork and is absent upstream;
   the ask is to expose/contractualize it.
2. Building OV's oneDNN fork and calling its internal FC-compression
   primitives — unstable internal API, not sane to ship against.
3. Hand-writing a brgemm-class INT4 kernel (register-tiled multi-row,
   software-pipelined) — a multi-week kernel project.

The dnnl integration stays in-tree as the reproducible probe (zero cost
without DNNL_DIR).

## Verdict

Route 1 is **viable as engineering** (public API, parity-correct, one
optimization pass from stock-competitive) but its CPU residency payoff today
is compile-peak + page semantics, not steady RSS. Keep behind
`--gemv-offload` as an experiment; fold the executor-behavior finding into
the upstream RFC (docs/rfcs/openvino-inplace-int4-gemv.md) — the RFC's ask
narrows to making the mmap-backed behavior *contractual* and closing the
remaining ~30% kernel gap in-plugin. Next if pursued: per-node overhead
profile, weight-page prefetch, multi-row (chunk) support, GPU-device story.
