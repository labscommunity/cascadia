//! GLM-5.2 MLA attention (classic DeepSeek-V3 MLA) — absorbed-latent decode.
//!
//! Per token at `pos`:
//!   qr   = rmsnorm(wq_a · x, q_a_ln)                       [q_lora]
//!   q    = wq_b · qr                                       [h·qk_head]  ([nope|pe] per head)
//!   rope(q_pe)                                             (last qk_rope of each head)
//!   comp = wkv_a · x                                       [kv_lora + qk_rope]
//!   Lc   = rmsnorm(comp[..kv_lora], kv_a_ln)               [kv_lora]   (cached latent)
//!   Rc   = rope(comp[kv_lora..])                           [qk_rope]   (cached k_pe, shared)
//! then, per head h (`W_UK_h` = kv_b rows `[rbase..rbase+nope]`,
//! `W_UV_h` = rows `[rbase+nope..rbase+nope+v]`, `rbase = h·(nope+v)`):
//!   qabs  = W_UK_hᵀ · q_nope                               [kv_lora]   (absorb the k up-proj)
//!   score[t] = (qabs·Lc[t] + q_pe·Rc[t]) · scale           over cached t
//!   p     = softmax(score)
//!   clat  = Σ_t p[t]·Lc[t]                                 [kv_lora]
//!   ctx_h = W_UV_h · clat                                  [v_head]    (absorb the v up-proj)
//!   out   = wo · concat_h(ctx_h)                           [hidden]
//!
//! `scale = qk_head^-0.5` (qk_head = qk_nope + qk_rope); no mscale (no YaRN),
//! no attention sink, no bias. Numeric contract (matches the CPU reference in
//! `tools/glm5_ref`): bf16 rounding after each linear / RMSNorm / rope; the
//! absorb core (qabs / score / softmax / clat / ctx) stays in f32.
//!
//! This is the decode path used for every position. Batched-naive prefill is a
//! later perf addition (M4); it is mathematically identical by linearity. DSA
//! sparsity (top-2048 gather) is layered on in a later slice — here attention
//! is over the full causal range `0..=pos`.

use super::rope::apply_rope_row;
use crate::dsv4::math::{dot, dot_bf16w, linear_bf16_w, rmsnorm};
use crate::dsv4::rope::Freqs;

/// Widen a bf16-bit weight to f32 (bf16 is the top 16 bits of an f32).
#[inline]
fn widen(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Attention projection weights. GEMV weights are stored as bf16 bits (`u16`) —
/// batch-1 projections are bandwidth-bound and the model is bf16-native, so
/// halving the streamed bytes ~halves the projection cost. RMSNorm weights stay
/// f32.
pub struct AttnWeights {
    pub wq_a: Vec<u16>,     // [q_lora, hidden]
    pub q_a_ln: Vec<f32>,   // [q_lora]
    pub wq_b: Vec<u16>,     // [h*qk_head, q_lora]
    pub wkv_a: Vec<u16>,    // [kv_lora + qk_rope, hidden]
    pub kv_a_ln: Vec<f32>,  // [kv_lora]
    pub wkv_b: Vec<u16>,    // [h*(qk_nope + v_head), kv_lora]
    pub wo: Vec<u16>,       // [hidden, h*v_head]
}

pub struct AttentionLayer {
    pub hidden: usize,
    pub h: usize,
    pub qk_nope: usize,
    pub qk_rope: usize,
    pub v_head: usize,
    pub kv_lora: usize,
    pub q_lora: usize,
    pub eps: f32,
    qk_head: usize, // qk_nope + qk_rope
    kv_out: usize,  // qk_nope + v_head (kv_b rows per head)
    scale: f32,
    w: AttnWeights,
    freqs: Freqs,
    lc: Vec<f32>, // [max_seq, kv_lora]
    rc: Vec<f32>, // [max_seq, qk_rope]
    len: usize,   // cached positions
}

impl AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden: usize,
        h: usize,
        qk_nope: usize,
        qk_rope: usize,
        v_head: usize,
        kv_lora: usize,
        q_lora: usize,
        max_seq: usize,
        eps: f32,
        w: AttnWeights,
        freqs: Freqs,
    ) -> Self {
        let qk_head = qk_nope + qk_rope;
        let kv_out = qk_nope + v_head;
        assert_eq!(w.wq_a.len(), q_lora * hidden);
        assert_eq!(w.q_a_ln.len(), q_lora);
        assert_eq!(w.wq_b.len(), h * qk_head * q_lora);
        assert_eq!(w.wkv_a.len(), (kv_lora + qk_rope) * hidden);
        assert_eq!(w.kv_a_ln.len(), kv_lora);
        assert_eq!(w.wkv_b.len(), h * kv_out * kv_lora);
        assert_eq!(w.wo.len(), hidden * h * v_head);
        Self {
            hidden,
            h,
            qk_nope,
            qk_rope,
            v_head,
            kv_lora,
            q_lora,
            eps,
            qk_head,
            kv_out,
            scale: (qk_head as f32).powf(-0.5),
            w,
            freqs,
            lc: vec![0.0; max_seq * kv_lora],
            rc: vec![0.0; max_seq * qk_rope],
            len: 0,
        }
    }

    /// Clear the KV cache.
    pub fn reset(&mut self) {
        self.lc.fill(0.0);
        self.rc.fill(0.0);
        self.len = 0;
    }

    /// Attend one token `x` (`[hidden]`) at absolute position `self.len`,
    /// appending its latent/k_pe to the cache. Returns `out` (`[hidden]`).
    /// Positions must be fed in order starting from 0.
    pub fn forward_token(&mut self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.hidden);
        let (h, nope, rope) = (self.h, self.qk_nope, self.qk_rope);
        let (vh, kvl, qk) = (self.v_head, self.kv_lora, self.qk_head);
        let pos = self.len;

        // q = wq_b · rmsnorm(wq_a · x), rope on each head's pe tail.
        let mut qr = vec![0.0f32; self.q_lora];
        linear_bf16_w(x, &self.w.wq_a, self.q_lora, self.hidden, &mut qr);
        rmsnorm(&mut qr, &self.w.q_a_ln, self.eps);
        let mut q = vec![0.0f32; h * qk];
        linear_bf16_w(&qr, &self.w.wq_b, h * qk, self.q_lora, &mut q);
        for hi in 0..h {
            apply_rope_row(&mut q[hi * qk..(hi + 1) * qk], &self.freqs, pos, rope, false);
        }

        // comp = wkv_a · x -> [latent | k_pe]; append normed latent + roped k_pe.
        let mut comp = vec![0.0f32; kvl + rope];
        linear_bf16_w(x, &self.w.wkv_a, kvl + rope, self.hidden, &mut comp);
        let mut lat = comp[..kvl].to_vec();
        rmsnorm(&mut lat, &self.w.kv_a_ln, self.eps);
        self.lc[pos * kvl..(pos + 1) * kvl].copy_from_slice(&lat);
        let mut kpe = comp[kvl..].to_vec();
        apply_rope_row(&mut kpe, &self.freqs, pos, rope, false);
        self.rc[pos * rope..(pos + 1) * rope].copy_from_slice(&kpe);
        self.len += 1;
        let n = self.len; // cached tokens, includes self (causal)

        // Absorbed attention, per head, over all cached tokens 0..n.
        let mut ctx = vec![0.0f32; h * vh];
        let mut qabs = vec![0.0f32; kvl];
        let mut score = vec![0.0f32; n];
        let mut clat = vec![0.0f32; kvl];
        for hi in 0..h {
            let qh = &q[hi * qk..(hi + 1) * qk];
            let qnope = &qh[..nope];
            let qpe = &qh[nope..];
            let rbase = hi * self.kv_out;

            // qabs = W_UK_hᵀ · q_nope  (Σ_d q_nope[d] · kv_b row (rbase+d)).
            qabs.iter_mut().for_each(|v| *v = 0.0);
            for (d, &qd) in qnope.iter().enumerate() {
                let row = &self.w.wkv_b[(rbase + d) * kvl..(rbase + d + 1) * kvl];
                for (a, &wb) in qabs.iter_mut().zip(row) {
                    *a += qd * widen(wb);
                }
            }

            // scores over cached latents; softmax (f32).
            let mut smax = f32::NEG_INFINITY;
            for t in 0..n {
                let sn = dot(&qabs, &self.lc[t * kvl..(t + 1) * kvl]);
                let sp = dot(qpe, &self.rc[t * rope..(t + 1) * rope]);
                let s = (sn + sp) * self.scale;
                score[t] = s;
                smax = smax.max(s);
            }
            let mut denom = 0.0f32;
            for s in score[..n].iter_mut() {
                *s = (*s - smax).exp();
                denom += *s;
            }

            // clat = Σ_t p[t]·Lc[t]; ctx_h = W_UV_h · clat.
            clat.iter_mut().for_each(|v| *v = 0.0);
            for t in 0..n {
                let p = score[t] / denom;
                for (c, &l) in clat.iter_mut().zip(&self.lc[t * kvl..(t + 1) * kvl]) {
                    *c += p * l;
                }
            }
            let ctx_h = &mut ctx[hi * vh..(hi + 1) * vh];
            for (j, o) in ctx_h.iter_mut().enumerate() {
                let row = &self.w.wkv_b[(rbase + nope + j) * kvl..(rbase + nope + j + 1) * kvl];
                *o = dot_bf16w(row, &clat);
            }
        }

        // out = wo · concat_h(ctx_h).
        let mut out = vec![0.0f32; self.hidden];
        linear_bf16_w(&ctx, &self.w.wo, self.hidden, h * vh, &mut out);
        out
    }
}
