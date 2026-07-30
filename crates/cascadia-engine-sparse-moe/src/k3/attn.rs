//! Gated NoPE MLA — K3's 24 full-attention layers, absorbed-latent decode.
//!
//! Per token at `pos`:
//! ```text
//!   qr   = rmsnorm(q_a · x, q_a_ln)                    [q_lora]
//!   q    = q_b · qr                                    [h·qk_head]  ([nope|pe] per head)
//!   comp = kv_a_with_mqa · x                           [kv_lora + qk_rope]
//!   Lc   = rmsnorm(comp[..kv_lora], kv_a_ln)           [kv_lora]   cached latent
//!   Rc   = comp[kv_lora..]                             [qk_rope]   cached, shared, UNROTATED
//! ```
//! then per head `h` (`W_UK_h` = `kv_b` rows `[rbase..rbase+nope)`, `W_UV_h` =
//! rows `[rbase+nope..rbase+nope+v)`, `rbase = h·(nope+v)`):
//! ```text
//!   qabs     = W_UK_hᵀ · q_nope                        [kv_lora]   absorb the k up-proj
//!   score[t] = (qabs·Lc[t] + q_pe·Rc[t]) · scale       over cached t
//!   p        = softmax(score)
//!   clat     = Σ_t p[t]·Lc[t]                          [kv_lora]
//!   ctx_h    = W_UV_h · clat                           [v_head]    absorb the v up-proj
//!   out      = o_proj · (sigmoid(g_proj·x) ⊙ concat_h(ctx_h))
//! ```
//!
//! Two differences from the classic V3 MLA in [`crate::glm::attn`], which is why
//! this is a separate implementation rather than a flag on that one:
//!
//! * **NoPE.** `mla_use_nope = true` and upstream sets `rotary_emb = None`, so
//!   nothing is rotated. The `qk_rope_head_dim` slice still exists and still
//!   contributes to the score, just unrotated, and its key half is MQA-shared
//!   (one copy per token, read by every head).
//! * **Output gate.** `sigmoid(g_proj·x) ⊙ ctx` after the head concat, before
//!   `o_proj`.
//!
//! Absorbed, because materialising the cached latents back through `kv_b_proj`
//! costs a `[24576, 512]` projection per cached position per step — 18.7x slower
//! at ctx 2048 and growing linearly (see the ignored bench in this file).
//!
//! Numeric contract: bf16 after each linear / RMSNorm, absorb core in f32.
//! Equivalent to the materialised form but not bit-identical, so the reference
//! keeps the naive version and a test holds the two against each other.

use half::bf16;

use crate::dsv4::math::{dot_bf16w, linear_bf16_w, rmsnorm, to_bf16};

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
    /// Floats cached per token: the normalised kv latent plus the shared
    /// (unrotated) rope slice. 576 for K3.
    #[inline]
    pub fn latent_per_token(&self) -> usize {
        self.kv_lora_rank + self.qk_rope
    }
}

/// Growing latent KV cache: `len` rows of `[Lc | Rc]`.
///
/// `Lc` is stored ALREADY RMS-normalised, so the norm is paid once per token
/// instead of once per cached position per step.
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
    let (qk_head, lat_n) = (d.qk_head(), d.latent_per_token());
    let (nope, rope, vh, kvl) = (d.qk_nope, d.qk_rope, d.v_head, d.kv_lora_rank);

    // q = q_b(rmsnorm(q_a(x)))
    let mut qa = vec![0.0f32; d.q_lora_rank];
    linear_bf16_w(x, &w.q_a_proj, d.q_lora_rank, d.hidden, &mut qa);
    rmsnorm(&mut qa, &w.q_a_layernorm, d.eps);
    let mut q = vec![0.0f32; d.heads * qk_head];
    linear_bf16_w(&qa, &w.q_b_proj, d.heads * qk_head, d.q_lora_rank, &mut q);

    // append this token's [Lc | Rc]; Lc normalised now, once
    let mut comp = vec![0.0f32; lat_n];
    linear_bf16_w(x, &w.kv_a_proj_with_mqa, lat_n, d.hidden, &mut comp);
    rmsnorm(&mut comp[..kvl], &w.kv_a_layernorm, d.eps);
    kv.latent.extend_from_slice(&comp);
    kv.len += 1;
    let p = kv.len;

    let scale = (qk_head as f32).powf(-0.5);
    let row_stride = nope + vh; // kv_b rows per head
    let mut ctx = vec![0.0f32; d.heads * vh];
    let mut qabs = vec![0.0f32; kvl];
    let mut scores = vec![0.0f32; p];
    let mut clat = vec![0.0f32; kvl];

    for h in 0..d.heads {
        let qh = &q[h * qk_head..(h + 1) * qk_head];
        let (q_nope, q_pe) = qh.split_at(nope);
        let rbase = h * row_stride;

        // qabs = W_UK_h^T · q_nope — absorbs the k up-projection into the query,
        // so the cached latents are never expanded. Paid once per token.
        qabs.fill(0.0);
        for (i, &qi) in q_nope.iter().enumerate() {
            if qi == 0.0 {
                continue;
            }
            let wi = &w.kv_b_proj[(rbase + i) * kvl..(rbase + i + 1) * kvl];
            for (a, &wb) in qabs.iter_mut().zip(wi) {
                *a += bf16::from_bits(wb).to_f32() * qi;
            }
        }

        // score against the cached latents directly: 576 MACs per position
        for (t, sc) in scores.iter_mut().enumerate() {
            let row = &kv.latent[t * lat_n..(t + 1) * lat_n];
            let (lc, rc) = row.split_at(kvl);
            let mut s = 0.0f32;
            for (a, b) in qabs.iter().zip(lc) {
                s += a * b;
            }
            for i in 0..rope {
                s += q_pe[i] * rc[i];
            }
            *sc = s * scale;
        }

        let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut denom = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            denom += *s;
        }

        // clat = Σ p[t]·Lc[t], then expand ONCE through W_UV_h
        clat.fill(0.0);
        for (t, &s) in scores.iter().enumerate() {
            let wt = s / denom;
            let lc = &kv.latent[t * lat_n..t * lat_n + kvl];
            for (c, &l) in clat.iter_mut().zip(lc) {
                *c += wt * l;
            }
        }
        let ctxh = &mut ctx[h * vh..(h + 1) * vh];
        for (i, o) in ctxh.iter_mut().enumerate() {
            let r = rbase + nope + i;
            *o = dot_bf16w(&w.kv_b_proj[r * kvl..(r + 1) * kvl], &clat);
        }
    }

    // output gate, then o_proj
    let proj = d.heads * vh;
    let mut g = vec![0.0f32; proj];
    linear_bf16_w(x, &w.g_proj, proj, d.hidden, &mut g);
    for i in 0..proj {
        ctx[i] = to_bf16(to_bf16(ctx[i]) * (1.0 / (1.0 + (-g[i]).exp())));
    }
    linear_bf16_w(&ctx, &w.o_proj, d.hidden, proj, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf(v: &[f32]) -> Vec<u16> {
        v.iter().map(|&x| bf16::from_f32(x).to_bits()).collect()
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

    /// The materialised form absorbed decode replaces — expand every cached
    /// latent through `kv_b_proj`, then ordinary attention. Kept in the tests
    /// only, as the thing absorbed decode must agree with.
    fn mla_step_materialised(
        x: &[f32],
        w: &MlaWeights,
        d: MlaDims,
        kv: &mut MlaKv,
        out: &mut [f32],
    ) {
        let (qk_head, lat_n) = (d.qk_head(), d.latent_per_token());
        let (nope, rope, vh, kvl) = (d.qk_nope, d.qk_rope, d.v_head, d.kv_lora_rank);
        let mut qa = vec![0.0f32; d.q_lora_rank];
        linear_bf16_w(x, &w.q_a_proj, d.q_lora_rank, d.hidden, &mut qa);
        rmsnorm(&mut qa, &w.q_a_layernorm, d.eps);
        let mut q = vec![0.0f32; d.heads * qk_head];
        linear_bf16_w(&qa, &w.q_b_proj, d.heads * qk_head, d.q_lora_rank, &mut q);

        let mut comp = vec![0.0f32; lat_n];
        linear_bf16_w(x, &w.kv_a_proj_with_mqa, lat_n, d.hidden, &mut comp);
        rmsnorm(&mut comp[..kvl], &w.kv_a_layernorm, d.eps);
        kv.latent.extend_from_slice(&comp);
        kv.len += 1;
        let p = kv.len;

        let kb_out = d.heads * (nope + vh);
        let mut exp = vec![0.0f32; p * kb_out];
        for t in 0..p {
            let lc = &kv.latent[t * lat_n..t * lat_n + kvl];
            linear_bf16_w(
                lc,
                &w.kv_b_proj,
                kb_out,
                kvl,
                &mut exp[t * kb_out..(t + 1) * kb_out],
            );
        }
        let scale = (qk_head as f32).powf(-0.5);
        let mut ctx = vec![0.0f32; d.heads * vh];
        for h in 0..d.heads {
            let qh = &q[h * qk_head..(h + 1) * qk_head];
            let mut sc = vec![0.0f32; p];
            for (t, s) in sc.iter_mut().enumerate() {
                let e = &exp[t * kb_out..(t + 1) * kb_out];
                let kn = &e[h * (nope + vh)..h * (nope + vh) + nope];
                let rc = &kv.latent[t * lat_n + kvl..(t + 1) * lat_n];
                let mut acc = 0.0f32;
                for i in 0..nope {
                    acc += qh[i] * kn[i];
                }
                for i in 0..rope {
                    acc += qh[nope + i] * rc[i];
                }
                *s = acc * scale;
            }
            let m = sc.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut den = 0.0f32;
            for s in sc.iter_mut() {
                *s = (*s - m).exp();
                den += *s;
            }
            let ctxh = &mut ctx[h * vh..(h + 1) * vh];
            for (t, &s) in sc.iter().enumerate() {
                let wt = s / den;
                let e = &exp[t * kb_out..(t + 1) * kb_out];
                let v = &e[h * (nope + vh) + nope..(h + 1) * (nope + vh)];
                for i in 0..vh {
                    ctxh[i] += wt * v[i];
                }
            }
        }
        let proj = d.heads * vh;
        let mut g = vec![0.0f32; proj];
        linear_bf16_w(x, &w.g_proj, proj, d.hidden, &mut g);
        for i in 0..proj {
            ctx[i] = to_bf16(to_bf16(ctx[i]) * (1.0 / (1.0 + (-g[i]).exp())));
        }
        linear_bf16_w(&ctx, &w.o_proj, d.hidden, proj, out);
    }

    #[test]
    fn absorbed_agrees_with_the_materialised_form() {
        // The whole point: absorbing W_UK/W_UV into q/o must not change the
        // result, only the cost. Checked over a growing cache so the
        // per-position score path is exercised, not just the single-token case.
        let d = dims();
        let w = weights(d);
        let (mut a, mut b) = (MlaKv::default(), MlaKv::default());
        for step in 0..6 {
            let x: Vec<f32> = (0..d.hidden)
                .map(|i| ((i + step * 3) as f32 * 0.21).cos())
                .collect();
            let mut oa = vec![0.0f32; d.hidden];
            let mut ob = vec![0.0f32; d.hidden];
            mla_step(&x, &w, d, &mut a, &mut oa);
            mla_step_materialised(&x, &w, d, &mut b, &mut ob);
            for i in 0..d.hidden {
                assert!(
                    (oa[i] - ob[i]).abs() <= 2e-3 * ob[i].abs().max(1.0),
                    "step {step} elem {i}: absorbed {} vs materialised {}",
                    oa[i],
                    ob[i]
                );
            }
        }
    }

    /// On-demand cost comparison at K3's REAL dims. `#[ignore]`d because it is
    /// a measurement, not an assertion — timings make poor CI gates.
    ///
    /// The cache is filled DIRECTLY rather than by stepping, so setup is O(ctx)
    /// instead of O(ctx^2) and large contexts stay measurable.
    ///
    ///   cargo test -p cascadia-engine-sparse-moe --lib \
    ///       k3::attn::tests::absorbed_cost -- --ignored --nocapture
    #[test]
    #[ignore]
    fn absorbed_cost_vs_materialised_at_real_dims() {
        use std::time::Instant;
        let d = MlaDims {
            hidden: 7168,
            heads: 96,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope: 128,
            qk_rope: 64,
            v_head: 128,
            eps: 1e-5,
        };
        let w = weights(d);
        let x: Vec<f32> = (0..d.hidden).map(|i| (i as f32 * 0.017).cos()).collect();
        let mut out = vec![0.0f32; d.hidden];
        let lat_n = d.latent_per_token();

        let fill = |n: usize| -> MlaKv {
            MlaKv {
                latent: (0..n * lat_n).map(|i| ((i as f32) * 0.003).sin()).collect(),
                len: n,
            }
        };

        println!("one MLA layer, one decode step (K3 dims: 96 heads, kv_lora 512)");
        for ctx in [128usize, 512, 2048] {
            let mut a = fill(ctx);
            let t0 = Instant::now();
            mla_step(&x, &w, d, &mut a, &mut out);
            let abs_ms = t0.elapsed().as_secs_f64() * 1e3;

            let mut b = fill(ctx);
            let t1 = Instant::now();
            mla_step_materialised(&x, &w, d, &mut b, &mut out);
            let mat_ms = t1.elapsed().as_secs_f64() * 1e3;
            println!(
                "  ctx {ctx:>5}: absorbed {abs_ms:8.1} ms   materialised {mat_ms:9.1} ms   \
                 {:.1}x",
                mat_ms / abs_ms
            );
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
