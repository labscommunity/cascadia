//! GLM-5.2 DSA lightning indexer (IndexShare) — selects the top-`index_topk`
//! (2048) causal key positions per query, restricting which cached latents the
//! MLA attention scores/sums over (raw positions, no block compression).
//!
//! Per key at `pos`:  `kd = rope_first(layernorm(ix_wk·x, k_norm), qk_rope)`,
//! cached in `Ic[pos]` (`index_head_dim`, default 128; rope on the first
//! `qk_rope` dims only). Per query at `pos` (from the RMSNorm'd q_a residual
//! `qr`): `qi = rope_first(ix_wq·qr, qk_rope)` per index head, `w32 = ix_wp·x`.
//! Score:
//!   `isc[t] = (1/√nh)·Σ_h w32[h]·ReLU((1/√hd)·qi[h]·kd[t])`
//! Select `keep = min(pos+1, topk)`: threshold = keep-th largest score, emit
//! positions in ASCENDING order (`> thr` first, then `== thr`) — a
//! two-pass select. When `pos+1 ≤ topk` selection is a no-op (all keys); here we return
//! all positions, equivalent to the full causal range.
//!
//! Numeric contract (matches `tools/glm5_ref::indexer_ref`): bf16 rounding after
//! each linear / LayerNorm / rope; the score dots / ReLU / scale stay f32. The
//! rope reuses `apply_rope_row` — its interleaved output layout differs from
//! the half-split layout, but qi and kd share the convention so `qi·kd` is
//! unchanged.

use super::rope::apply_rope_row;
use crate::dsv4::math::{dot, linear_bf16_w, to_bf16};
use crate::dsv4::rope::Freqs;

/// Classic LayerNorm (mean + population variance, weight + bias), bf16-rounded
/// output. `x` is `[rows, dim]`; `w`/`b` are `[dim]`. eps 1e-6 for the indexer.
pub fn layernorm(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let dim = w.len();
    assert_eq!(b.len(), dim);
    assert_eq!(x.len() % dim, 0, "layernorm: len % dim != 0");
    for row in x.chunks_mut(dim) {
        let mu = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / dim as f32;
        let r = 1.0 / (var + eps).sqrt();
        for (v, (&wi, &bi)) in row.iter_mut().zip(w.iter().zip(b)) {
            *v = to_bf16((*v - mu) * r * wi + bi);
        }
    }
}

pub struct IndexerWeights {
    pub ix_wq: Vec<u16>,    // [nh*hd, q_lora]
    pub ix_wk: Vec<u16>,    // [hd, hidden]
    pub ix_wp: Vec<u16>,    // [nh, hidden]
    pub k_norm_w: Vec<f32>, // [hd]
    pub k_norm_b: Vec<f32>, // [hd]
}

pub struct Indexer {
    pub hidden: usize,
    pub q_lora: usize,
    pub nh: usize,       // index_n_heads
    pub hd: usize,       // index_head_dim
    pub rope_dim: usize, // qk_rope (rope applied to the first rope_dim of hd)
    pub eps: f32,
    scale_w: f32, // 1/sqrt(nh)
    scale_d: f32, // 1/sqrt(hd)
    w: IndexerWeights,
    freqs: Freqs,
    ic: Vec<f32>, // [max_seq, hd]  key cache
    len: usize,
}

impl Indexer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden: usize,
        q_lora: usize,
        nh: usize,
        hd: usize,
        rope_dim: usize,
        max_seq: usize,
        eps: f32,
        w: IndexerWeights,
        freqs: Freqs,
    ) -> Self {
        assert_eq!(w.ix_wq.len(), nh * hd * q_lora);
        assert_eq!(w.ix_wk.len(), hd * hidden);
        assert_eq!(w.ix_wp.len(), nh * hidden);
        assert_eq!(w.k_norm_w.len(), hd);
        assert_eq!(w.k_norm_b.len(), hd);
        assert!(rope_dim <= hd && rope_dim % 2 == 0);
        Self {
            hidden,
            q_lora,
            nh,
            hd,
            rope_dim,
            eps,
            scale_w: 1.0 / (nh as f32).sqrt(),
            scale_d: 1.0 / (hd as f32).sqrt(),
            w,
            freqs,
            ic: vec![0.0; max_seq * hd],
            len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.ic.fill(0.0);
        self.len = 0;
    }

    /// Number of cached keys.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Build and cache the index key for token `x` at position `self.len`.
    pub fn append_key(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.hidden);
        let pos = self.len;
        let mut kd = vec![0.0f32; self.hd];
        linear_bf16_w(x, &self.w.ix_wk, self.hd, self.hidden, &mut kd);
        layernorm(&mut kd, &self.w.k_norm_w, &self.w.k_norm_b, self.eps);
        apply_rope_row(&mut kd[..self.rope_dim], &self.freqs, pos, self.rope_dim, false);
        self.ic[pos * self.hd..(pos + 1) * self.hd].copy_from_slice(&kd);
        self.len += 1;
    }

    /// Per-key scores for a query at `query_pos` (0-based; keys `0..=query_pos`
    /// must already be cached). `qr` is the RMSNorm'd q_a residual, `x` the
    /// hidden state, both at the query position.
    pub fn scores(&self, qr: &[f32], x: &[f32], query_pos: usize) -> Vec<f32> {
        assert_eq!(qr.len(), self.q_lora);
        assert_eq!(x.len(), self.hidden);
        assert!(query_pos < self.len, "query_pos beyond cached keys");
        let (nh, hd) = (self.nh, self.hd);

        let mut qi = vec![0.0f32; nh * hd];
        linear_bf16_w(qr, &self.w.ix_wq, nh * hd, self.q_lora, &mut qi);
        for h in 0..nh {
            let base = h * hd;
            apply_rope_row(&mut qi[base..base + self.rope_dim], &self.freqs, query_pos, self.rope_dim, false);
        }
        let mut w32 = vec![0.0f32; nh];
        linear_bf16_w(x, &self.w.ix_wp, nh, self.hidden, &mut w32);

        let nk = query_pos + 1;
        let mut isc = vec![0.0f32; nk];
        for (t, s) in isc.iter_mut().enumerate() {
            let kt = &self.ic[t * hd..(t + 1) * hd];
            let mut a = 0.0f32;
            for h in 0..nh {
                let d0 = dot(&qi[h * hd..(h + 1) * hd], kt) * self.scale_d;
                if d0 > 0.0 {
                    a += w32[h] * d0;
                }
            }
            *s = a * self.scale_w;
        }
        isc
    }

    /// Select the top-`keep = min(query_pos+1, topk)` key positions for the
    /// query, in ascending position order (`> thr` then `== thr`
    /// two-pass). When `query_pos+1 ≤ topk` this returns every position (the
    /// no-op / full-causal case).
    pub fn select(&self, qr: &[f32], x: &[f32], query_pos: usize, topk: usize) -> Vec<u32> {
        let isc = self.scores(qr, x, query_pos);
        let nk = isc.len();
        let keep = nk.min(topk);
        if keep == 0 {
            return Vec::new();
        }
        let mut sorted = isc.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let thr = sorted[keep - 1];
        let mut sel = Vec::with_capacity(keep);
        for (t, &s) in isc.iter().enumerate() {
            if s > thr && sel.len() < keep {
                sel.push(t as u32);
            }
        }
        for (t, &s) in isc.iter().enumerate() {
            if s == thr && sel.len() < keep {
                sel.push(t as u32);
            }
        }
        sel
    }
}
