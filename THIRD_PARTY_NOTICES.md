# Third-party notices

Cascadia is released under the Apache License 2.0 (see [LICENSE](LICENSE)). This file lists
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
source files have been copied into the cascadia tree.

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

## Cloudflare Clef (Apache-2.0)

**Repository:** <https://huggingface.co/Cloudflare/clef-flash> (also
<https://huggingface.co/Cloudflare/clef>), file `joint_schema_model.py`.

**Author:** Cloudflare, Inc.

**Licence:** Apache-2.0 (the model cards declare `license: apache-2.0`).
The upstream `joint_schema_model.py` carries no copyright header, and the
repository's `LICENSE` file is the Apache-2.0 text inherited from the
Qwen3.5 base model.

**What we borrowed (code):**
- `EvidenceRoutingLayer` and `JointSchemaHead` are copied verbatim into
  `tools/decision/clef_head.py`. `TensorHead` in the same file is the same
  maths over flat tensors, with `cosine_similarity` rewritten for the
  OpenVINO export. The export helpers and loaders are our additions.
- `encode_record` (prompt layout, option order, span positions) is ported
  to Rust in `crates/cascadia-api/src/decisions/clef.rs`.
- `tools/decision/codec_golden.py` imports the upstream module at test
  time to check the Rust codec against it; it is not vendored.

## Kev (Apache-2.0)

**Repository:** <https://github.com/jaredpalmer/kev> (model:
<https://huggingface.co/jaredpalmer/kev-9b>).

**Author:** Jared Palmer.

**Licence:** Apache-2.0 (Copyright 2026 Jared Palmer, per the repository's
`LICENSE`).

**What we borrowed (code):** the request-to-record mapping of `kev/api.py`
(`to_record`) and the prompt encoding of `kev/model.py` (`encode`, row
form), ported to Rust in `crates/cascadia-api/src/decisions/kev.rs`; the
confidence formula in `crates/cascadia-api/src/decisions/schema.rs` matches
the one `kev/api.py` mirrors from TypeSafe's system-one-adapter.

**Implementation:** Rust re-implementation. No Kev source files have been
copied into the cascadia tree.

## AutoTrust JEV (Apache-2.0)

**Repository:** <https://huggingface.co/autotrust/JEV-9B> (also
<https://huggingface.co/autotrust/JEV-27B>), file `vl/serve_decide.py`.

**Author:** AutoTrust.

**Licence:** Apache-2.0 (the model card declares `license: apache-2.0`;
the repository has no separate `LICENSE` file and `serve_decide.py` no
copyright header).

**What we borrowed (code):** the System 1 prompt layout of
`serve_decide.py` (`s1_pass`, strategy `single`, and `_parts` state
rendering), ported to Rust in `crates/cascadia-api/src/decisions/jev.rs`
and mirrored in `tools/decision/codec_golden.py` (`jev_text`).

**Implementation:** Rust and Python re-implementation. No JEV source files
have been copied into the cascadia tree.

## TypeSafe System One API and system-one-adapter (ideas, not code)

**Source:** TypeSafe AI's public System One API (`POST /v1/systemone`)
and its reference adapter `system-one-adapter` 0.2.1
(`src/system_one_adapter/_utils/confidence_metrics.py`), as cited by
`kev/api.py`.

**Author:** TypeSafe AI.

**Licence:** not stated in the material we used. Nothing was copied: we
read the formulas and the wire format through Kev's Apache-2.0
`kev/api.py`, which mirrors them, and through the public API shape.

**What we borrowed (ideas, not code):**
- the `/v1/systemone` request and response shape (state, typed `noul` /
  `choice` / `score` questions, answers keyed by question id), served by
  `crates/cascadia-api/src/decisions/handler.rs` and `schema.rs`;
- the confidence formulas: choice `(p_max - 1/K) / (1 - 1/K)` and score
  `max(0, 1 - E|level - mode| / D)`, in
  `crates/cascadia-api/src/decisions/schema.rs` and documented in
  `docs/decisions.md`.

**Implementation:** Rust re-implementation. No TypeSafe source files have
been copied into the cascadia tree.
