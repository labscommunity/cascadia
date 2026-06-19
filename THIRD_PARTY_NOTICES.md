# Third-party notices

Cascadia is released under Apache-2.0 (target). This file lists
third-party code and ideas that have informed parts of the cascadia
implementation, together with their licences. Inclusion here does not
imply that any of those parties endorse cascadia.

## PowerInfer / SmallThinker (MIT)

**Repository:** <https://github.com/Tiiny-AI/PowerInfer>

**Authors:** Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen (SJTU-IPADS)
plus the upstream `ggml` / `llama.cpp` authors led by Georgi Gerganov.

**Licence:** MIT (Copyright 2023 Georgi Gerganov, 2023 SJTU-IPADS,
2023–2024 the ggml authors).

**What we borrowed (ideas, not code):**
- Bounded LRU expert cache pattern, ported as `cascadia-engine-sparse-moe`
  `ExpertCache` LRU bound (`MAX_N_CACHED` in their `expert_cache.cpp`).
- Two-phase Gate-first FFN sparsity pattern, ported as
  `cascadia-int4-gemm::expert_forward_sparse` (PowerInfer-2 §4.4 /
  SmallThinker `fused_sparse_moe.cpp`).

**Implementation:** clean-room Rust re-implementation. No PowerInfer
source files have been copied into the cascadia tree. See
rainier `docs/POWERINFER_PORT.md` for the full technique map
([github.com/labscommunity/rainier](https://github.com/labscommunity/rainier/blob/main/docs/POWERINFER_PORT.md)).

**Papers cited:**
- PowerInfer, arxiv:2312.12456
- PowerInfer-2, arxiv:2406.06282
- TurboSparse, arxiv:2406.05955

## CATS — Contextual Activation Sparsity (Apache-2.0)

**Paper:** Lee et al., 2024. arxiv:2404.08763.

**What we borrowed:** the relative-magnitude threshold formulation for
SwiGLU-family activations. Used in
`cascadia-int4-gemm::ffn_sparsity::build_active_mask`.

## CHESS — Channel-wise Sparsification (MIT)

**Paper:** Liu et al., 2024. arxiv:2409.01366.

**What we borrowed:** referenced as the per-channel-threshold variant
of CATS; planned for follow-up evaluation. Not yet ported.

## Rust `lru` crate (MIT)

**Crate:** <https://crates.io/crates/lru>

**Author:** Jerome Froelich.

**Licence:** MIT.

Used as a build dependency from `cascadia-engine-sparse-moe`. Source
not copied — pulled in via Cargo.
