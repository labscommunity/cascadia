//! dsv4 full attention layer — port of `Attention` in
//! `tools/deepseek_v4_ref/model.py` (b = 1).
//!
//! Pipeline per token: q = wq_b(q_norm(wq_a(x))) split into heads, per-head
//! plain-RMS normalise, rope; kv = kv_norm(wkv(x)) (ONE shared latent — MQA),
//! rope + FP8 sim on non-rope channels; sparse attention over [sliding
//! window of `win` raw latents] + [compressed latents beyond it] with
//! per-head sinks; inverse-rope on the output; grouped low-rank O
//! (per-group wo_a einsum, then wo_b).
//!
//! KV cache layout (`[win + max_seq/ratio, hd]`): rows `[0, win)` are the
//! raw-latent ring (slot = pos % win), rows `[win, ..)` the compressed
//! rows (slot = win + block). Prefill attends over the *linear* kv list
//! (row t = position t, compressed appended at offset s) exactly like the
//! reference, then materialises the ring.

use super::compress::Compressor;
use super::indexer::Indexer;
use super::math::{act_quant_sim, linear_bf16, rms_normalize, rmsnorm, to_bf16};
use super::rope::{apply_rope_row, Freqs};

pub struct AttnWeights {
    pub wq_a: Vec<f32>,      // [q_lora, dim]
    pub q_norm_w: Vec<f32>,  // [q_lora]
    pub wq_b: Vec<f32>,      // [h*hd, q_lora]
    pub wkv: Vec<f32>,       // [hd, dim]
    pub kv_norm_w: Vec<f32>, // [hd]
    pub wo_a: Vec<f32>,      // [groups*o_lora, h*hd/groups]
    pub wo_b: Vec<f32>,      // [dim, groups*o_lora]
    pub sink: Vec<f32>,      // [h]
}

pub struct AttentionLayer {
    pub dim: usize,
    pub h: usize,
    pub hd: usize,
    pub rd: usize,
    pub q_lora: usize,
    pub o_lora: usize,
    pub groups: usize,
    pub win: usize,
    pub ratio: usize, // 0 = pure sliding window
    pub eps: f32,
    scale: f32,
    w: AttnWeights,
    pub comp: Option<Compressor>,
    pub idx: Option<Indexer>,
    pub freqs: Freqs,
    cache: Vec<f32>, // [win + max_seq/ratio, hd]
}

impl AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        h: usize,
        hd: usize,
        rd: usize,
        q_lora: usize,
        o_lora: usize,
        groups: usize,
        win: usize,
        ratio: usize,
        max_seq: usize,
        eps: f32,
        w: AttnWeights,
        comp: Option<Compressor>,
        idx: Option<Indexer>,
        freqs: Freqs,
    ) -> Self {
        assert_eq!(w.wq_a.len(), q_lora * dim);
        assert_eq!(w.wq_b.len(), h * hd * q_lora);
        assert_eq!(w.wkv.len(), hd * dim);
        assert_eq!(w.wo_a.len(), groups * o_lora * (h * hd / groups));
        assert_eq!(w.wo_b.len(), dim * groups * o_lora);
        assert_eq!(comp.is_some(), ratio > 0);
        assert_eq!(idx.is_some(), ratio == 4);
        let cache_rows = win + if ratio > 0 { max_seq / ratio } else { 0 };
        Self {
            dim,
            h,
            hd,
            rd,
            q_lora,
            o_lora,
            groups,
            win,
            ratio,
            eps,
            scale: (hd as f32).powf(-0.5),
            w,
            comp,
            idx,
            freqs,
            cache: vec![0.0; cache_rows * hd],
        }
    }

    /// Clear KV ring + compressed cache + compressor/indexer state.
    pub fn reset(&mut self) {
        self.cache.fill(0.0);
        if let Some(c) = self.comp.as_mut() {
            c.reset();
        }
        if let Some(i) = self.idx.as_mut() {
            i.reset();
        }
    }

    /// q_norm(wq_a(x)) — also feeds the indexer. [q_lora]
    fn qr_row(&self, x: &[f32]) -> Vec<f32> {
        let mut qr = vec![0.0f32; self.q_lora];
        linear_bf16(x, &self.w.wq_a, self.q_lora, self.dim, &mut qr);
        rmsnorm(&mut qr, &self.w.q_norm_w, self.eps);
        qr
    }

    /// full q for one token at `pos`: [h*hd], roped + per-head normalised.
    fn q_row(&self, qr: &[f32], pos: usize) -> Vec<f32> {
        let (h, hd) = (self.h, self.hd);
        let mut q = vec![0.0f32; h * hd];
        linear_bf16(qr, &self.w.wq_b, h * hd, self.q_lora, &mut q);
        rms_normalize(&mut q, hd, self.eps);
        for hi in 0..h {
            apply_rope_row(
                &mut q[hi * hd..(hi + 1) * hd],
                &self.freqs,
                pos,
                self.rd,
                false,
            );
        }
        q
    }

    /// kv latent for one token at `pos`: [hd], normed + roped + FP8 sim.
    fn kv_row(&self, x: &[f32], pos: usize) -> Vec<f32> {
        let mut kv = vec![0.0f32; self.hd];
        linear_bf16(x, &self.w.wkv, self.hd, self.dim, &mut kv);
        rmsnorm(&mut kv, &self.w.kv_norm_w, self.eps);
        apply_rope_row(&mut kv, &self.freqs, pos, self.rd, false);
        let non_rope = self.hd - self.rd;
        act_quant_sim(&mut kv[..non_rope], 64);
        kv
    }

    /// inverse rope + grouped low-rank O: [h*hd] -> [dim].
    fn o_proj(&self, mut o: Vec<f32>, pos: usize) -> Vec<f32> {
        let (h, hd, groups, o_lora) = (self.h, self.hd, self.groups, self.o_lora);
        for hi in 0..h {
            apply_rope_row(
                &mut o[hi * hd..(hi + 1) * hd],
                &self.freqs,
                pos,
                self.rd,
                true,
            );
        }
        let gd = h * hd / groups;
        let mut mid = vec![0.0f32; groups * o_lora];
        for g in 0..groups {
            let og = &o[g * gd..(g + 1) * gd];
            for r in 0..o_lora {
                let row = &self.w.wo_a[(g * o_lora + r) * gd..(g * o_lora + r + 1) * gd];
                let mut acc = 0.0f32;
                for k in 0..gd {
                    acc += og[k] * row[k];
                }
                mid[g * o_lora + r] = to_bf16(acc);
            }
        }
        let mut out = vec![0.0f32; self.dim];
        linear_bf16(&mid, &self.w.wo_b, self.dim, groups * o_lora, &mut out);
        out
    }

    /// window candidate rows for prefill position `i` (raw-kv linear space).
    fn window_idxs_prefill(&self, i: usize) -> Vec<i32> {
        let lo = i.saturating_sub(self.win - 1);
        let mut v: Vec<i32> = (lo..=i).map(|p| p as i32).collect();
        v.resize(self.win, -1);
        v
    }

    /// window candidate rows for decode at `pos` (ring-slot space).
    fn window_idxs_decode(&self, pos: usize) -> Vec<i32> {
        if pos >= self.win - 1 {
            (0..self.win).map(|s| s as i32).collect()
        } else {
            let mut v: Vec<i32> = (0..=pos).map(|s| s as i32).collect();
            v.resize(self.win, -1);
            v
        }
    }

    /// deterministic compressed-row indices for non-indexer layers.
    fn compress_idxs_static(&self, valid_rows: usize, offset: i32) -> Vec<i32> {
        (0..valid_rows).map(|r| r as i32 + offset).collect()
    }

    /// Prefill `s` tokens; returns attention output [s, dim] and fills the
    /// ring + compressed cache.
    pub fn prefill(&mut self, x: &[f32], s: usize) -> Vec<f32> {
        let (dim, hd, win) = (self.dim, self.hd, self.win);
        assert_eq!(x.len(), s * dim);
        // qr / q / raw kv per token
        let mut qr = vec![0.0f32; s * self.q_lora];
        for t in 0..s {
            let r = self.qr_row(&x[t * dim..(t + 1) * dim]);
            qr[t * self.q_lora..(t + 1) * self.q_lora].copy_from_slice(&r);
        }
        let mut kv_lin = vec![0.0f32; s * hd]; // linear kv list (row = position)
        for t in 0..s {
            let r = self.kv_row(&x[t * dim..(t + 1) * dim], t);
            kv_lin[t * hd..(t + 1) * hd].copy_from_slice(&r);
        }
        // compressed rows appended after the raw list (offset = s)
        let mut comp_rows: Vec<Vec<f32>> = Vec::new();
        let mut comp_idxs: Vec<Vec<i32>> = vec![Vec::new(); s];
        if self.ratio > 0 {
            let offset = s as i32;
            if let Some(idx) = self.idx.as_mut() {
                comp_idxs = idx.prefill(x, &qr, s, &self.freqs, offset);
            } else {
                for (t, ci) in comp_idxs.iter_mut().enumerate() {
                    *ci = self.compress_idxs_static((t + 1) / self.ratio, offset);
                }
            }
            comp_rows = self.comp.as_mut().unwrap().prefill(x, s, &self.freqs);
        }
        let mut kv_full = kv_lin.clone();
        for row in &comp_rows {
            kv_full.extend_from_slice(row);
        }
        // attend
        let mut out = vec![0.0f32; s * dim];
        for t in 0..s {
            let q = self.q_row(&qr[t * self.q_lora..(t + 1) * self.q_lora], t);
            let mut idxs = self.window_idxs_prefill(t);
            idxs.extend_from_slice(&comp_idxs[t]);
            let mut o = vec![0.0f32; self.h * hd];
            super::attn::sparse_attn_pos(
                &q,
                &kv_full,
                &self.w.sink,
                &idxs,
                self.h,
                hd,
                self.scale,
                &mut o,
            );
            let od = self.o_proj(o, t);
            out[t * dim..(t + 1) * dim].copy_from_slice(&od);
        }
        // materialise ring (slot = pos % win) + compressed cache rows
        for t in s.saturating_sub(win)..s {
            let slot = t % win;
            self.cache[slot * hd..(slot + 1) * hd].copy_from_slice(&kv_lin[t * hd..(t + 1) * hd]);
        }
        for (blk, row) in comp_rows.iter().enumerate() {
            let slot = win + blk;
            self.cache[slot * hd..(slot + 1) * hd].copy_from_slice(row);
        }
        out
    }

    /// One decode token at absolute `pos`; returns [dim].
    pub fn decode(&mut self, x: &[f32], pos: usize) -> Vec<f32> {
        let (dim, hd, win) = (self.dim, self.hd, self.win);
        assert_eq!(x.len(), dim);
        let qr = self.qr_row(x);
        let kv = self.kv_row(x, pos);
        let slot = pos % win;
        self.cache[slot * hd..(slot + 1) * hd].copy_from_slice(&kv);
        let mut idxs = self.window_idxs_decode(pos);
        if self.ratio > 0 {
            let offset = win as i32;
            if let Some(idx) = self.idx.as_mut() {
                idxs.extend(idx.decode(x, &qr, pos, &self.freqs, offset));
            } else {
                idxs.extend(self.compress_idxs_static((pos + 1) / self.ratio, offset));
            }
            if let Some(row) = self.comp.as_mut().unwrap().decode(x, pos, &self.freqs) {
                let cslot = win + pos / self.ratio;
                self.cache[cslot * hd..(cslot + 1) * hd].copy_from_slice(&row);
            }
        }
        let q = self.q_row(&qr, pos);
        let mut o = vec![0.0f32; self.h * hd];
        super::attn::sparse_attn_pos(
            &q,
            &self.cache,
            &self.w.sink,
            &idxs,
            self.h,
            hd,
            self.scale,
            &mut o,
        );
        self.o_proj(o, pos)
    }
}
