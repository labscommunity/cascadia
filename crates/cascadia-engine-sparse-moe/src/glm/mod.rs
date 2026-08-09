//! GLM-5.2 (`glm5`) Rust shell — DeepSeek-V3-style MLA + DSA sparse-MoE.
//! See `docs/architectures/glm5.md` for the full spec and the reuse map.
//!
//! Shares the pure numeric leaves (bf16 rounding, GEMV, RMSNorm, interleaved
//! rope) with the dsv4 shell via `crate::dsv4::{math, rope}` — additive, no
//! dsv4 edits. GLM differs above the leaves: sigmoid + `noaux_tc` routing (not
//! sqrtsoftplus), plain residual (no Hyper-Connections), raw-position DSA (not
//! block-compressed), interleaved rope with YaRN disabled (`original_seq_len=0`,
//! `base=8e6`).
//!
//! Every primitive is golden-tested 1:1 against the CPU reference in
//! `tools/glm5_ref/` (regenerate fixtures with
//! `tools/glm5_ref/gen_fixtures.py`).
//!
//! Build order (each lands with passing goldens):
//!   1. router gate                                  <- this milestone (M1)
//!   2. rope table + MLA attention + DSA indexer
//!   3. dense/MoE block + full model greedy parity
//!   4. engine/manifest wiring (`shell_backend = "rust_glm"`)

pub mod attn;
pub mod ffn;
pub mod gate;
pub mod grammar;
pub mod indexer;
pub mod kv_cache;
pub mod loader;
pub mod lookahead;
pub mod model;
pub mod moe;
pub mod mtp;
pub mod ov_attn;
pub mod ov_expert;
pub mod prof;
pub mod residency;
pub mod rope;
pub mod stage;
