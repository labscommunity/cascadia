# PowerInfer port — technique map and attribution

This document tracks which ideas from the PowerInfer family of papers
and the PowerInfer / SmallThinker code base have been ported into
cascadia, which have not, and why.

## Source and licence

- Repository: <https://github.com/Tiiny-AI/PowerInfer>
- Licence: **MIT** (`/LICENSE` and `/smallthinker/LICENSE` in that
  repository). Copyright 2023 Georgi Gerganov, 2023 SJTU-IPADS,
  2023–2024 The ggml authors. MIT is compatible with cascadia's
  Apache-2.0 target.
- Papers cited:
  1. PowerInfer (SOSP'24) — arxiv 2312.12456. Yixin Song, Zeyu Mi,
     Haotong Xie, Haibo Chen.
  2. PowerInfer-2 — arxiv 2406.06282. Same group, mobile target.
  3. TurboSparse — arxiv 2406.05955. Same group, model-side
     sparsification surgery.

## Re-implementation, not copy

Every PowerInfer technique listed below is a clean-room Rust
re-implementation. We have not copied any source code from the
PowerInfer repository — the inspiration is at the algorithm and
design level, with comments in the relevant source files citing the
specific PowerInfer file or paper section that the design echoes.

Where a more general technique applies that originated outside
PowerInfer (e.g. magnitude thresholding for SwiGLU — CATS, CHESS) we
cite that source too.

## Per-technique status

| # | Technique | PowerInfer source | Cascadia status | Notes |
|---|---|---|---|---|
| 1 | Bounded LRU expert cache (`MAX_N_CACHED`) | SmallThinker `expert_cache.cpp` | **Ported** — `lru::LruCache` wrapping the three `ExpertCache` variants. `CASCADIA_MAX_EXPERTS_CACHED` env var + `--max-cached-experts` CLI flag. | Eviction stats logged per-step. Backwards-compatible at default (`0` = unbounded). |
| 2 | Two-phase Gate-first FFN sparsity | PowerInfer-2 §4.4, SmallThinker `fused_sparse_moe.cpp` | **Ported (SwiGLU adaptation)** — `expert_forward_sparse` + `dequant_gemv_int4_rows_subset_auto`. PI's ReLU "skip if Gate=0" replaced by CATS/CHESS-style magnitude threshold. `--ffn-sparsity-threshold τ`. | Bit-identical to dense at τ=0. Direct kernel speedup ≈ 1/active_frac (2.0× at 50%, 8.4× at 10% — see `bench_ffn_sparsity`). |
| 3 | Expert prefetch via lightweight predictor | SmallThinker `pipeline.cpp::prefetch` | **Partial — last-token predictor only.** `Runner.last_routing_ids` predicts next-token experts as "same as previous"; prefetcher thread calls `madvise(MADV_WILLNEED)`. Not gated by a separately-trained predictor MLP. | Already on main via autolab iter 029 (C1). |
| 4 | mmap'd packed expert weights | SmallThinker `expert_bundle.cpp` | **Equivalent already present** — `format::ExpertWeights` mmaps a per-(layer, expert) `.bin`; `safetensors_source` mmaps shared shards. The PI all-experts-in-one-file bundle would marginally improve FD pressure and 4 KB alignment for `O_DIRECT`, but the existing safetensors mmap path is already efficient. Tracked as a follow-up if `io_uring` work needs strict alignment guarantees. | See deferred §. |
| 5 | `io_uring`-driven async expert prefetch (5-stage neuron-cluster pipeline) | PowerInfer-2 §4.3 + SmallThinker `iou.cpp` | **In flight on a different branch** (`perf/io-uring-prefetch-074`; design at `docs/perf/io_uring_prefetch.md`). Not in this PR. | The current PR's bounded-LRU + sparsity changes are *complementary* to that work — they reduce the working set, which makes the prefetch budget easier. |
| 6 | Adaptive predictor MLP sized per-layer (sparsity + skewness) | PowerInfer §5.1 | **Not ported** — requires offline activation profiling + training a predictor MLP per FFN layer. The recipe is portable but the *recipe* is the deliverable; the actual predictor weights would need to be trained per model. Out of scope for an inference-only port. | See deferred §. |
| 7 | Hot/cold neuron ILP placement (`gpu_idx`, `gpu_bucket`) | PowerInfer §6.3 | **Not ported** — depends on (6); also assumes a discrete GPU asymmetry that doesn't exist on Intel UMA AI PCs (Lunar Lake). For Intel a 3-tier {iGPU, NPU, CPU} ILP would replace PI's 2-tier; design pending. | See deferred §. |
| 8 | Sparse `lm_head` with profiler MLP | PowerInfer SmallThinker `sparse_lmhead.cpp` | **Not ported** — also requires the trained profiler MLP. The K2.6 head is currently OV-compiled and quality-tuned; rewiring it to a profiler-gated row-sparse matmul without quality regression needs the trained predictor. | See deferred §. |
| 9 | Hardware-aware offline planner (per-device read-size, NPU graph cache) | PowerInfer-2 §5 | **Not ported.** Different framework — needed once we have multi-device placement (5). | See deferred §. |
| 10 | TurboSparse dReLU surgery + 150 B-token continued pretrain | TurboSparse §3.2, §5 | **Not applicable.** Training-side, not inference-side. The realistic path is to consume the published `huggingface.co/PowerInfer/TurboSparse-Mixtral-47B` checkpoint when we add Mixtral support; for K2.6 the training cost is prohibitive (paper figure is for 7B / 47B parameter models; K2.6 is ~1 T params). | See deferred §. |
| 11 | CUDA sparse-matmul kernels (`dequantize_mul_mat_axpy_sparse`, etc.) | PowerInfer `ggml-cuda.cu` | **Not applicable.** Wrong backend — replaced on the Intel side by OpenVINO + cascadia-int4-gemm. | — |

## What's in this PR

Two perf-relevant changes, both opt-in, both byte-identical to the
pre-PR path when disabled:

1. **Bounded LRU expert cache** — see commit
   `perf(sparse-moe): bounded LRU expert cache (CASCADIA_MAX_EXPERTS_CACHED)`.

   Caps the per-Runner expert pool by LRU. Required for memory-
   constrained Intel AI PCs (16 GiB Lunar Lake). Diagnostic counters
   for evictions + resident-count.

   - CLI: `--max-cached-experts N`
   - Env: `CASCADIA_MAX_EXPERTS_CACHED=N`
   - Default: `0` (unbounded; preserves pre-PR behaviour).

2. **Two-phase Gate-first FFN sparsity** — see commit
   `perf(int4-gemm): two-phase Gate-first FFN sparsity`.

   After the gate matmul, lanes whose `|silu(gate)| < τ · max_i
   |silu(gate_i)|` are dropped before the up + down phases. Direct
   kernel speedup ≈ 1/active_frac.

   - CLI: `--ffn-sparsity-threshold τ`
   - Env: `CASCADIA_FFN_SPARSITY_THRESHOLD=τ`
   - Default: `0.0` (dense; bit-identical to pre-PR).

Both changes ship behind explicit knobs because they trade quality
for throughput in some configurations. The defaults match the pre-PR
behaviour exactly.

## What's *not* in this PR (deferred)

Listed in a single place so the reader can see the full scope of
PowerInfer's work and what we've intentionally left out — most of it
is genuinely valuable, just out of scope for *this* PR.

- **Expert bundle (.cpack) file format.** The PI SmallThinker bundle
  packs all (layer, expert, matrix) into one 4-KB-aligned binary so a
  single `io_uring` SQE can load any matrix. cascadia already has
  efficient mmap'd expert access via the existing safetensors path;
  the bundle adds FD-pressure / alignment hygiene but not raw
  throughput. Will revisit if/when the in-flight `io_uring` work
  needs `O_DIRECT`.
- **Sparse-aware multi-token FFN kernel.** This PR sparsifies only
  the single-token (decode) path. Multi-token prefill has a different
  cache-and-tile structure (per-token masks would break the
  AVX-512 multi-token tile shape); deferred.
- **Trained predictor MLP for FFN sparsity.** The CATS / CHESS
  magnitude-threshold heuristic in this PR is a no-training stand-in.
  A trained per-layer predictor (PI §5.1) recovers ~5–10% additional
  sparsity at the same quality budget; needs offline calibration
  tooling.
- **Sparse lm_head with profiler MLP.** Same dependency on a trained
  predictor.
- **Hot/cold neuron ILP placement** across {iGPU, NPU, CPU}.
  Foundational for the future cascadia 3-tier model; needs the
  predictor work first.
- **TurboSparse dReLU surgery.** Training-side, not inference-side.
  Realistic path: consume their published TurboSparse-Mixtral-47B
  checkpoint when we add Mixtral support.
- **PowerInfer-2 5-stage neuron-cluster pipeline.** Maps to the
  in-flight `io_uring` prefetch branch (`perf/io-uring-prefetch-074`)
  rather than this PR.

## Reproducing the kernel bench

```text
cargo run --release -p cascadia-int4-gemm --bin bench_ffn_sparsity -- --iters 200
```

Local Apple Silicon (scalar fallback path), iters=50:

| active-frac | per-call    | speedup |
| ----------- | ----------- | ------- |
| 1.00 dense  |     1.08 ms | 1.00×   |
| 0.50        |   539.10 µs | 2.01×   |
| 0.30        |   344.35 µs | 3.15×   |
| 0.10        |   128.33 µs | 8.44×   |

On miner (Cascade Lake AVX-512 + VNNI, 16 cores) the AVX-512 path is
taken; per-call wall time is lower but the speedup curve is the same.

End-to-end (full K2.6 decode) gain depends on the model's activation
distribution at the chosen threshold. CATS / CHESS report 30–50%
sparsity at τ ≈ 0.10 with <1% perplexity hit on SwiGLU LLMs; K2.6
calibration is part of the bench task on the autolab loop, not this
PR.

## Acknowledgements

The PowerInfer team published an unusually well-engineered open
reference. Their work is the proximate source for techniques #1, #2,
and #3–9 above. We are also indebted to:

- The CATS authors (Lee et al. 2024, Apache-2.0) for the
  magnitude-threshold formulation of SwiGLU sparsity.
- The CHESS authors (Liu et al. 2024, MIT) for the threshold-by-channel
  variant we plan to evaluate as a follow-up.

If you build something on top of these changes that gets published,
please cite both the PowerInfer paper and (when applicable) CATS /
CHESS.
