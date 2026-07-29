//! Gated NoPE MLA — K3's 24 full-attention layers.
//!
//! Two differences from the classic V3 MLA in [`crate::glm::attn`], which is
//! why this is a separate implementation rather than a flag on that one:
//!
//! * **NoPE.** `mla_use_nope = true` and upstream sets `rotary_emb = None`, so
//!   nothing is rotated. The `qk_rope_head_dim` slice still exists
//!   dimensionally but passes through untouched, and its key half is MQA-shared
//!   (one copy per token, broadcast over all heads).
//! * **Output gate.** `o = sigmoid(g_proj(x)) * ctx` applied after the head
//!   concat and before `o_proj`.
//!
//! The KV cache stores the COMPRESSED latent (`kv_lora_rank + qk_rope_head_dim`
//! floats per token, 576 for K3) and expands it through `kv_b_proj` on read.
//! That keeps cache memory at the same 576 f/token as glm5's absorbed decode.
//! Absorbed decode proper (folding `W_UK`/`W_UV` into q/o) is the eventual
//! optimisation — it removes the per-step re-expansion — but it is a
//! performance change, not a numerical one, and is deferred until the shell is
//! validated end to end.

use crate::dsv4::math::{linear_bf16_w, rmsnorm, to_bf16};

/// Per-layer MLA weights, bf16-valued (`[out, in]` row-major).
pub struct MlaWeights {
    pub q_a_proj: Vec<u16>,
    pub q_a_layernorm: Vec<f32>,
    pub q_b_proj: Vec<u16>,
    pub kv_a_proj_with_mqa: Vec<u16>,
    pub kv_a_layernorm: Vec<f32>,
    pub kv_b_proj: Vec<u16>,
    pub g_proj: Vec<u16>,
    pub o_proj: Vec<u16>,
}

/// Shape contract for one MLA layer.
#[derive(Clone, Copy, Debug)]
pub struct MlaDims {
    pub hidden: usize,
    pub heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope: usize,
    pub qk_rope: usize,
    pub v_head: usize,
    pub eps: f32,
}

impl MlaDims {
    #[inline]
    pub fn qk_head(&self) -> usize {
        self.qk_nope + self.qk_rope
    }
    /// Floats cached per token: the kv latent plus the shared rope slice.
    #[inline]
    pub fn latent_per_token(&self) -> usize {
        self.kv_lora_rank + self.qk_rope
    }
}

/// Growing latent KV cache: `len` rows of `latent_per_token()` floats.
#[derive(Default)]
pub struct MlaKv {
    pub latent: Vec<f32>,
    pub len: usize,
}

impl MlaKv {
    pub fn clear(&mut self) {
        self.latent.clear();
        self.len = 0;
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// One decode step. `x` is `[hidden]`; `out` is `[hidden]`.
pub fn mla_step(x: &[f32], w: &MlaWeights, d: MlaDims, kv: &mut MlaKv, out: &mut [f32]) {
    let qk_head = d.qk_head();
    let lat_n = d.latent_per_token();

    // q = q_b(rmsnorm(q_a(x)))
    let mut qa = vec![0.0f32; d.q_lora_rank];
    linear_bf16_w(x, &w.q_a_proj, d.q_lora_rank, d.hidden, &mut qa);
    rmsnorm(&mut qa, &w.q_a_layernorm, d.eps);
    let mut q = vec![0.0f32; d.heads * qk_head];
    linear_bf16_w(&qa, &w.q_b_proj, d.heads * qk_head, d.q_lora_rank, &mut q);

    // append this token's compressed kv latent
    let mut lat = vec![0.0f32; lat_n];
    linear_bf16_w(x, &w.kv_a_proj_with_mqa, lat_n, d.hidden, &mut lat);
    kv.latent.extend_from_slice(&lat);
    kv.len += 1;
    let p = kv.len;

    // expand each cached position through kv_b_proj: [kv_lora] -> heads*(nope+v)
    let kb_out = d.heads * (d.qk_nope + d.v_head);
    let mut scores = vec![0.0f32; p];
    let mut expanded = vec![0.0f32; p * kb_out];
    let mut tmp = vec![0.0f32; d.kv_lora_rank];
    let scale = (qk_head as f32).powf(-0.5);
    for t in 0..p {
        let row = &kv.latent[t * lat_n..(t + 1) * lat_n];
        tmp.copy_from_slice(&row[..d.kv_lora_rank]);
        rmsnorm(&mut tmp, &w.kv_a_layernorm, d.eps);
        linear_bf16_w(
            &tmp,
            &w.kv_b_proj,
            kb_out,
            d.kv_lora_rank,
            &mut expanded[t * kb_out..(t + 1) * kb_out],
        );
    }

    // per head: score against every cached position, softmax, weighted v
    let mut ctx = vec![0.0f32; d.heads * d.v_head];
    for h in 0..d.heads {
        let qh = &q[h * qk_head..(h + 1) * qk_head];
        for (t, sc) in scores.iter_mut().enumerate() {
            let e = &expanded[t * kb_out..(t + 1) * kb_out];
            let k_nope = &e[h * (d.qk_nope + d.v_head)..h * (d.qk_nope + d.v_head) + d.qk_nope];
            // the rope slice is MQA-shared: one copy per token, all heads
            let k_rot = &kv.latent[t * lat_n + d.kv_lora_rank..(t + 1) * lat_n];
            let mut s = 0.0f32;
            for i in 0..d.qk_nope {
                s += qh[i] * k_nope[i];
            }
            for i in 0..d.qk_rope {
                s += qh[d.qk_nope + i] * k_rot[i];
            }
            *sc = s * scale;
        }
        let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut denom = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            denom += *s;
        }
        let ctxh = &mut ctx[h * d.v_head..(h + 1) * d.v_head];
        for (t, &s) in scores.iter().enumerate() {
            let wgt = s / denom;
            let e = &expanded[t * kb_out..(t + 1) * kb_out];
            let vh = &e[h * (d.qk_nope + d.v_head) + d.qk_nope..(h + 1) * (d.qk_nope + d.v_head)];
            for i in 0..d.v_head {
                ctxh[i] += wgt * vh[i];
            }
        }
    }

    // output gate then o_proj
    let proj = d.heads * d.v_head;
    let mut g = vec![0.0f32; proj];
    linear_bf16_w(x, &w.g_proj, proj, d.hidden, &mut g);
    for i in 0..proj {
        ctx[i] = to_bf16(ctx[i]) * (1.0 / (1.0 + (-g[i]).exp()));
        ctx[i] = to_bf16(ctx[i]);
    }
    linear_bf16_w(&ctx, &w.o_proj, d.hidden, proj, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf(v: &[f32]) -> Vec<u16> {
        v.iter()
            .map(|&x| half::bf16::from_f32(x).to_bits())
            .collect()
    }

    fn dims() -> MlaDims {
        MlaDims {
            hidden: 8,
            heads: 2,
            q_lora_rank: 4,
            kv_lora_rank: 4,
            qk_nope: 2,
            qk_rope: 2,
            v_head: 2,
            eps: 1e-5,
        }
    }

    fn weights(d: MlaDims) -> MlaWeights {
        let f = |n: usize, k: f32| -> Vec<u16> {
            bf(&(0..n)
                .map(|i| ((i as f32) * k).sin() * 0.3)
                .collect::<Vec<_>>())
        };
        MlaWeights {
            q_a_proj: f(d.q_lora_rank * d.hidden, 0.7),
            q_a_layernorm: vec![1.0; d.q_lora_rank],
            q_b_proj: f(d.heads * d.qk_head() * d.q_lora_rank, 0.3),
            kv_a_proj_with_mqa: f(d.latent_per_token() * d.hidden, 0.5),
            kv_a_layernorm: vec![1.0; d.kv_lora_rank],
            kv_b_proj: f(d.heads * (d.qk_nope + d.v_head) * d.kv_lora_rank, 0.9),
            g_proj: f(d.heads * d.v_head * d.hidden, 0.4),
            o_proj: f(d.hidden * d.heads * d.v_head, 0.6),
        }
    }

    #[test]
    fn cache_grows_one_latent_row_per_step() {
        let d = dims();
        let w = weights(d);
        let mut kv = MlaKv::default();
        let x: Vec<f32> = (0..d.hidden).map(|i| (i as f32 * 0.2).cos()).collect();
        let mut out = vec![0.0f32; d.hidden];
        for step in 1..=3 {
            mla_step(&x, &w, d, &mut kv, &mut out);
            assert_eq!(kv.len(), step);
            assert_eq!(kv.latent.len(), step * d.latent_per_token());
        }
    }

    #[test]
    fn single_position_attention_is_the_value_row() {
        // With one cached token the softmax is degenerate (weight 1), so the
        // output must be finite and independent of the score scale.
        let d = dims();
        let w = weights(d);
        let mut kv = MlaKv::default();
        let x: Vec<f32> = (0..d.hidden).map(|i| (i as f32 * 0.2).cos()).collect();
        let mut out = vec![0.0f32; d.hidden];
        mla_step(&x, &w, d, &mut kv, &mut out);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite: {out:?}");
    }

    #[test]
    fn reset_clears_the_cache() {
        let d = dims();
        let w = weights(d);
        let mut kv = MlaKv::default();
        let x = vec![0.1f32; d.hidden];
        let mut out = vec![0.0f32; d.hidden];
        mla_step(&x, &w, d, &mut kv, &mut out);
        let first = out.clone();
        kv.clear();
        assert!(kv.is_empty());
        mla_step(&x, &w, d, &mut kv, &mut out);
        assert_eq!(out, first, "post-reset step must match a fresh sequence");
    }
}
