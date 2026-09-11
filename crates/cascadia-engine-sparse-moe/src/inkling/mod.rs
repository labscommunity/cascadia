//! Inkling (Thinking Machines, 975B / 41B-active MoE; family `inkling`) Rust
//! shell. Semantics follow transformers' `modeling_inkling.py` — the oracle the
//! fixture goldens are generated with — as fixed by the port contract
//! (`PORT_SPEC.md` §3). Per decoder layer:
//!
//! ```text
//!   h  = rmsnorm(x, attn_norm)
//!   a  = attention(h)          GQA; per-head q/k RMSNorm; causal short convs on
//!                              k and v; learned relative-position bias; 1/D
//!                              scale; sliding (512) or global keys; log scaling
//!   x  = x + sconv(attn_sconv, a)
//!   h2 = rmsnorm(x, mlp_norm)
//!   m  = dense SwiGLU · global_scale   |   Σ w_i E_i(h2) + Σ γ_s S_s(h2)
//!   x  = x + sconv(mlp_sconv, m)
//! ```
//!
//! with `x = rmsnorm(embed[t], embed_norm)` before layer 0 and
//! `logits = unembed · (rmsnorm(x, norm) / mup)` sliced to `unpadded_vocab`
//! after the last layer.
//!
//! Numeric contract (the GLM convention): bf16 weights for every linear, f32
//! accumulate, bf16 round at write-back; norms, router, convs, softmax and the
//! attention accumulate stay f32. Routed / shared experts reuse the glm
//! [`AnyExpert`](crate::glm::moe::AnyExpert) storage (bf16 goldens or mmap'd
//! int4 bins).
//!
//! Sequence state per layer = the KV cache plus four short-conv histories
//! (k, v, attn-out, mlp-out). All of it supports `reset`, `truncate`
//! (spec-decode rewind, bounded by the rewind slack — [`DEFAULT_REWIND`]) and
//! `snapshot` / `restore` (prefix cache), mirroring the glm KV API.
//!
//! Module map: [`conv`] (ShortConv), [`relpos`] (RelPos), [`gate`]
//! (`inkling_gate`), [`attn`] (AttentionLayer), [`ffn`] (SwiGLU re-exports +
//! w13 de-interleave), [`moe`] (MoeLayer / DenseMlp), [`model`] (Layer / Model).
//! `loader` / `stage` / the engine arm are built on top of these types.

/// `CASCADIA_INKLING_*` switch parsing — the glm helper (unset / empty / `0` /
/// `false` / `no` / `off` mean off; anything else means on).
pub use crate::glm::env_flag;

/// Default number of positions every sequence-state ring keeps *beyond* what
/// the forward needs, so [`truncate`](conv::ShortConv::truncate) can roll back
/// a rejected speculative draft. Sliding-window KV rings hold `window + rewind`
/// rows; conv histories hold `(K - 1) + rewind`. 4× the default n-gram draft
/// depth ([`crate::ngram_draft::DEFAULT_DRAFT_K`]); costs
/// `rewind · (2·Hkv·D + 2·Hkv·D + 2·H) · 4` bytes per layer (~2.5 MB at the
/// 975B dims). Rewinding further than the slack panics with a diagnostic.
pub const DEFAULT_REWIND: usize = 32;

/// RMSNorm in f32 with no output rounding: `y = w · x / sqrt(mean(x²) + eps)`,
/// row by row (`x` is `[rows, w.len()]`). `InklingRMSNorm` computes in f32 and
/// the contract keeps norms f32 (only the linears round to bf16), so this is
/// deliberately NOT `dsv4::math::rmsnorm`, which bf16-rounds its output.
pub fn rmsnorm_f32(x: &mut [f32], w: &[f32], eps: f32) {
    let dim = w.len();
    assert!(dim > 0, "rmsnorm_f32: empty weight");
    assert_eq!(x.len() % dim, 0, "rmsnorm_f32: len % dim != 0");
    for row in x.chunks_mut(dim) {
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let r = 1.0 / (ms + eps).sqrt();
        for (v, &wi) in row.iter_mut().zip(w) {
            *v = *v * r * wi;
        }
    }
}

pub mod attn;
pub mod conv;
pub mod ffn;
pub mod gate;
pub mod loader;
pub mod model;
pub mod moe;
pub mod relpos;
pub mod stage;

#[cfg(test)]
mod tests {
    use super::rmsnorm_f32;

    #[test]
    fn rmsnorm_f32_matches_definition_per_row() {
        // Two rows of dim 2: [3, 4] has mean-square 12.5; [0, 0] must not NaN.
        let mut x = vec![3.0f32, 4.0, 0.0, 0.0];
        let w = [2.0f32, 0.5];
        rmsnorm_f32(&mut x, &w, 1e-6);
        let r = 1.0 / (12.5f32 + 1e-6).sqrt();
        assert!((x[0] - 3.0 * r * 2.0).abs() < 1e-6);
        assert!((x[1] - 4.0 * r * 0.5).abs() < 1e-6);
        assert_eq!(&x[2..], &[0.0, 0.0]);
    }
}
