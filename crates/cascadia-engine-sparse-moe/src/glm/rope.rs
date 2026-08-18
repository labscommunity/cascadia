//! GLM-5.2 rotary embeddings.
//!
//! GLM uses interleaved partial RoPE (`rope_interleave=true`) applied to the
//! `qk_rope_head_dim` (64) slice of q/k, with `rope_theta=8e6` and no YaRN
//! (`rope_type="default"`). The rotation math — adjacent (even, odd) pairs
//! rotated as one complex number — is identical to the dsv4 pair-interleaved
//! rotation, so we reuse `dsv4::rope` verbatim and only pin GLM's config here.
//! No inverse-rope (dsv4 applies that to attention outputs; GLM does not).
//!
//! Golden-tested against `tools/glm5_ref/kernels_ref.py`
//! (`precompute_freqs_cis` / `apply_rotary_emb`).

use crate::dsv4::rope::{precompute_freqs, Freqs};

/// GLM-5.2 rope dim (`qk_rope_head_dim`).
pub const ROPE_DIM: usize = 64;
/// GLM-5.2 `rope_theta`.
pub const ROPE_THETA: f32 = 8_000_000.0;

/// Build GLM's rope cos/sin table for positions `0..seqlen`. `original_seq_len
/// = 0` disables YaRN, so the factor/beta parameters are inert. Later the model
/// loader supplies `dim`/`theta` from the manifest; the constants here are the
/// known GLM-5.2 values used by the M1 primitive goldens.
pub fn glm_freqs(seqlen: usize) -> Freqs {
    precompute_freqs(ROPE_DIM, seqlen, 0, ROPE_THETA, 1.0, 32.0, 1.0)
}

/// Re-export: interleaved rotation of the last `rd` dims of `row` (use
/// `inverse = false` for GLM).
pub use crate::dsv4::rope::apply_rope_row;
