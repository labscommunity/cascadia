//! dsv4 DSA Indexer — selects the top-k compressed KV positions for sparse
//! attention. Port of `Indexer` in `tools/deepseek_v4_ref/model.py` (b = 1).
//!
//! Per token: q = wq_b(qr) split into index heads -> rope -> Hadamard ->
//! FP4 sim; its own rotate-variant Compressor maintains a compressed KV
//! cache of the SAME layer input x; head-weighted ReLU scores against that
//! cache pick `index_topk` compressed rows. Returned indices are offset by
//! the caller-provided base (raw-KV row count) and -1 marks invalid slots
//! (prefill causality).

use super::compress::Compressor;
use super::math::{fp4_act_quant_sim, hadamard, linear_bf16, to_bf16};
use super::rope::{apply_rope_row, Freqs};

pub struct IndexerWeights {
    pub wq_b: Vec<f32>,         // [h*d, q_lora]
    pub weights_proj: Vec<f32>, // [h, dim]
}

pub struct Indexer {
    pub h: usize,
    pub d: usize,
    pub rd: usize,
    pub topk: usize,
    pub ratio: usize,
    dim: usize,
    q_lora: usize,
    w: IndexerWeights,
    pub comp: Compressor,   // rotate variant, head_dim = d
    pub kv_cache: Vec<f32>, // [max_rows, d]
    softmax_scale: f32,
}

impl Indexer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        q_lora: usize,
        h: usize,
        d: usize,
        rd: usize,
        topk: usize,
        ratio: usize,
        max_seq: usize,
        w: IndexerWeights,
        comp: Compressor,
    ) -> Self {
        assert_eq!(w.wq_b.len(), h * d * q_lora);
        assert_eq!(w.weights_proj.len(), h * dim);
        assert_eq!(comp.d, d);
        assert!(comp.rotate);
        Self {
            h,
            d,
            rd,
            topk,
            ratio,
            dim,
            q_lora,
            w,
            comp,
            kv_cache: vec![0.0; (max_seq / ratio) * d],
            softmax_scale: (d as f32).powf(-0.5),
        }
    }

    /// Clear the compressed cache + compressor state for a new generation.
    pub fn reset(&mut self) {
        self.comp.reset();
        self.kv_cache.fill(0.0);
    }

    /// q for one token: wq_b -> per-head rope(pos) -> Hadamard -> FP4 sim.
    fn q_row(&self, qr: &[f32], pos: usize, freqs: &Freqs) -> Vec<f32> {
        let (h, d) = (self.h, self.d);
        let mut q = vec![0.0f32; h * d];
        linear_bf16(qr, &self.w.wq_b, h * d, self.q_lora, &mut q);
        for hi in 0..h {
            let head = &mut q[hi * d..(hi + 1) * d];
            apply_rope_row(head, freqs, pos, self.rd, false);
        }
        hadamard(&mut q, d, (d as f32).powf(-0.5));
        fp4_act_quant_sim(&mut q, 32);
        q
    }

    /// head-weight row for one token: bf16(weights_proj(x) * scale * h^-0.5).
    fn weights_row(&self, x: &[f32]) -> Vec<f32> {
        let mut w = vec![0.0f32; self.h];
        linear_bf16(x, &self.w.weights_proj, self.h, self.dim, &mut w);
        let s = self.softmax_scale * (self.h as f32).powf(-0.5);
        for v in w.iter_mut() {
            *v = to_bf16(*v * s);
        }
        w
    }

    /// scores of one token's q against compressed rows [0, rows):
    /// sum_h bf16(relu(bf16(q_h . kv_r)) * w_h), f32 accumulated, bf16 out.
    fn score_row(&self, q: &[f32], w: &[f32], rows: usize) -> Vec<f32> {
        let (h, d) = (self.h, self.d);
        let mut out = vec![0.0f32; rows];
        for (r, o) in out.iter_mut().enumerate() {
            let kvr = &self.kv_cache[r * d..(r + 1) * d];
            let mut acc = 0.0f32;
            for hi in 0..h {
                let qh = &q[hi * d..(hi + 1) * d];
                let mut dot = 0.0f32;
                for k in 0..d {
                    dot += qh[k] * kvr[k];
                }
                let s = to_bf16(dot).max(0.0); // einsum bf16 out, then relu
                acc += to_bf16(s * w[hi]);
            }
            *o = to_bf16(acc);
        }
        out
    }

    /// top-k row indices by score (desc), k = min(topk, rows).
    fn topk_rows(&self, scores: &[f32]) -> Vec<i32> {
        let k = self.topk.min(scores.len());
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order[..k].iter().map(|&r| r as i32).collect()
    }

    /// Prefill: returns per-position topk (each length min(topk, s/ratio)),
    /// entries offset by `offset`, -1 where causally invalid.
    pub fn prefill(
        &mut self,
        x: &[f32],
        qr: &[f32],
        s: usize,
        freqs: &Freqs,
        offset: i32,
    ) -> Vec<Vec<i32>> {
        let d = self.d;
        let rows = self.comp.prefill(x, s, freqs);
        for (blk, row) in rows.iter().enumerate() {
            self.kv_cache[blk * d..(blk + 1) * d].copy_from_slice(row);
        }
        let avail = s / self.ratio;
        let mut out = Vec::with_capacity(s);
        for t in 0..s {
            let q = self.q_row(&qr[t * self.q_lora..(t + 1) * self.q_lora], t, freqs);
            let w = self.weights_row(&x[t * self.dim..(t + 1) * self.dim]);
            let mut scores = self.score_row(&q, &w, avail);
            let valid = (t + 1) / self.ratio; // causal: row r usable iff r < (t+1)/ratio
            for (r, sc) in scores.iter_mut().enumerate() {
                if r >= valid {
                    *sc = f32::NEG_INFINITY;
                }
            }
            let idxs = self
                .topk_rows(&scores)
                .into_iter()
                .map(|r| if r as usize >= valid { -1 } else { r + offset })
                .collect();
            out.push(idxs);
        }
        out
    }

    /// Decode one token at absolute position `pos`; returns topk indices
    /// (length min(topk, (pos+1)/ratio)) offset by `offset`.
    pub fn decode(
        &mut self,
        x_row: &[f32],
        qr_row: &[f32],
        pos: usize,
        freqs: &Freqs,
        offset: i32,
    ) -> Vec<i32> {
        let d = self.d;
        if let Some(row) = self.comp.decode(x_row, pos, freqs) {
            let slot = pos / self.ratio;
            self.kv_cache[slot * d..(slot + 1) * d].copy_from_slice(&row);
        }
        let q = self.q_row(qr_row, pos, freqs);
        let w = self.weights_row(x_row);
        let rows = (pos + 1) / self.ratio;
        let scores = self.score_row(&q, &w, rows);
        self.topk_rows(&scores)
            .into_iter()
            .map(|r| r + offset)
            .collect()
    }
}
