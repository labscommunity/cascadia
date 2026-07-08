//! dsv4 KV Compressor — learned gated pooling of `ratio` consecutive tokens
//! into one cache row, with the overlap variant (ratio 4) and the stateful
//! incremental decode path. Port of `Compressor` in
//! `tools/deepseek_v4_ref/model.py` (b = 1).
//!
//! Semantics (per channel c of the pooled head_dim d — the softmax over
//! candidate slots is PER-CHANNEL):
//!
//! * non-overlap (`coff = 1`): candidates are the `ratio` rows of the block;
//!   out[c] = sum_i kv[i][c] * softmax_i(score[i][c] + ape[i][c]).
//! * overlap (`ratio == 4`, `coff = 2`): wkv/wgate emit 2*d channels; the
//!   FIRST d channels of a block serve as "overlap" candidates for the NEXT
//!   block, the SECOND d channels are the block's own candidates — 2*ratio
//!   slots total (block 0's overlap slots are -inf/zero).
//!
//! Decode keeps a `[coff*ratio, coff*d]` running state: slots `[0, ratio)`
//! hold the previous window, `[ratio, 2*ratio)` the in-progress window; on a
//! compress boundary the window shifts down. Every `ratio`-th token emits one
//! finished row: bf16 -> RMSNorm -> rope (position = first token of the
//! block) -> (indexer variant) Hadamard + FP4 sim, or (attention variant)
//! FP8 sim on the non-rope channels.

use super::math::{act_quant_sim, fp4_act_quant_sim, hadamard, linear_f32, rmsnorm, round_bf16};
use super::rope::{apply_rope_row, Freqs};

pub struct CompressorWeights {
    pub wkv: Vec<f32>,    // [coff*d, dim]
    pub wgate: Vec<f32>,  // [coff*d, dim]
    pub ape: Vec<f32>,    // [ratio, coff*d]
    pub norm_w: Vec<f32>, // [d]
}

pub struct Compressor {
    pub dim: usize,
    pub d: usize,
    pub rd: usize,
    pub ratio: usize,
    pub overlap: bool,
    pub rotate: bool,
    pub eps: f32,
    w: CompressorWeights,
    kv_state: Vec<f32>,    // [coff*ratio, coff*d]
    score_state: Vec<f32>, // [coff*ratio, coff*d], -inf initialised
}

impl Compressor {
    pub fn new(
        dim: usize,
        d: usize,
        rd: usize,
        ratio: usize,
        rotate: bool,
        eps: f32,
        w: CompressorWeights,
    ) -> Self {
        let overlap = ratio == 4;
        let coff = 1 + overlap as usize;
        let cd = coff * d;
        assert_eq!(w.wkv.len(), cd * dim);
        assert_eq!(w.wgate.len(), cd * dim);
        assert_eq!(w.ape.len(), ratio * cd);
        assert_eq!(w.norm_w.len(), d);
        Self {
            dim,
            d,
            rd,
            ratio,
            overlap,
            rotate,
            eps,
            w,
            kv_state: vec![0.0; coff * ratio * cd],
            score_state: vec![f32::NEG_INFINITY; coff * ratio * cd],
        }
    }

    fn coff(&self) -> usize {
        1 + self.overlap as usize
    }

    pub fn reset(&mut self) {
        self.kv_state.fill(0.0);
        self.score_state.fill(f32::NEG_INFINITY);
    }

    /// finish one pooled row: bf16 round -> rmsnorm -> rope at `pos` ->
    /// quant sim. `row` has width d.
    fn finish(&self, row: &mut [f32], pos: usize, freqs: &Freqs) {
        round_bf16(row);
        rmsnorm(row, &self.w.norm_w, self.eps);
        apply_rope_row(row, freqs, pos, self.rd, false);
        if self.rotate {
            hadamard(row, self.d, (self.d as f32).powf(-0.5));
            fp4_act_quant_sim(row, 32);
        } else {
            let non_rope = self.d - self.rd;
            act_quant_sim(&mut row[..non_rope], 64);
        }
    }

    /// Prefill over `s` tokens (`x` = [s, dim], f32 holding bf16 values).
    /// Returns the finished compressed rows (cache slots 0..s/ratio).
    pub fn prefill(&mut self, x: &[f32], s: usize, freqs: &Freqs) -> Vec<Vec<f32>> {
        let (dim, d, ratio, cd) = (self.dim, self.d, self.ratio, self.coff() * self.d);
        assert_eq!(x.len(), s * dim);
        // per-token kv/gate in f32 (the reference computes compression in fp32)
        let mut kv = vec![0.0f32; s * cd];
        let mut score = vec![0.0f32; s * cd];
        for t in 0..s {
            linear_f32(
                &x[t * dim..(t + 1) * dim],
                &self.w.wkv,
                cd,
                dim,
                &mut kv[t * cd..(t + 1) * cd],
            );
            linear_f32(
                &x[t * dim..(t + 1) * dim],
                &self.w.wgate,
                cd,
                dim,
                &mut score[t * cd..(t + 1) * cd],
            );
        }
        let remainder = s % ratio;
        let cutoff = s - remainder;
        let offset = if self.overlap { ratio } else { 0 };
        if self.overlap && cutoff >= ratio {
            for i in 0..ratio {
                let src = (cutoff - ratio + i) * cd;
                self.kv_state[i * cd..(i + 1) * cd].copy_from_slice(&kv[src..src + cd]);
                for c in 0..cd {
                    self.score_state[i * cd + c] = score[src + c] + self.w.ape[i * cd + c];
                }
            }
        }
        for r in 0..remainder {
            let src = (cutoff + r) * cd;
            let dst = (offset + r) * cd;
            self.kv_state[dst..dst + cd].copy_from_slice(&kv[src..src + cd]);
            for c in 0..cd {
                self.score_state[dst + c] = score[src + c] + self.w.ape[r * cd + c];
            }
        }
        let nblk = cutoff / ratio;
        let mut out = Vec::with_capacity(nblk);
        for blk in 0..nblk {
            // gather candidate (kv, score) slot lists of width d
            let mut cand_kv: Vec<&[f32]> = Vec::with_capacity(2 * ratio);
            let mut cand_sc: Vec<Vec<f32>> = Vec::with_capacity(2 * ratio);
            if self.overlap {
                for i in 0..ratio {
                    // slot i: previous block's row i, channels [..d]
                    if blk == 0 {
                        cand_kv.push(&[]); // zero row
                        cand_sc.push(vec![f32::NEG_INFINITY; d]);
                    } else {
                        let src = ((blk - 1) * ratio + i) * cd;
                        cand_kv.push(&kv[src..src + d]);
                        let mut sc = vec![0.0; d];
                        for c in 0..d {
                            sc[c] = score[src + c] + self.w.ape[i * cd + c];
                        }
                        cand_sc.push(sc);
                    }
                }
            }
            for i in 0..ratio {
                let src = (blk * ratio + i) * cd;
                let ch0 = if self.overlap { d } else { 0 }; // second-half channels when overlap
                cand_kv.push(&kv[src + ch0..src + ch0 + d]);
                let mut sc = vec![0.0; d];
                for c in 0..d {
                    sc[c] = score[src + ch0 + c] + self.w.ape[i * cd + ch0 + c];
                }
                cand_sc.push(sc);
            }
            let mut row = pool_per_channel(&cand_kv, &cand_sc, d);
            self.finish(&mut row, blk * ratio, freqs);
            out.push(row);
        }
        out
    }

    /// One decode token at absolute position `pos` (`x_row` = [dim]).
    /// Returns the finished row for cache slot pos/ratio when a block
    /// boundary completes.
    pub fn decode(&mut self, x_row: &[f32], pos: usize, freqs: &Freqs) -> Option<Vec<f32>> {
        let (dim, d, ratio, cd) = (self.dim, self.d, self.ratio, self.coff() * self.d);
        assert_eq!(x_row.len(), dim);
        let mut kv = vec![0.0f32; cd];
        let mut score = vec![0.0f32; cd];
        linear_f32(x_row, &self.w.wkv, cd, dim, &mut kv);
        linear_f32(x_row, &self.w.wgate, cd, dim, &mut score);
        let slot_in_win = pos % ratio;
        for c in 0..cd {
            score[c] += self.w.ape[slot_in_win * cd + c];
        }
        let should_compress = (pos + 1) % ratio == 0;
        let write = if self.overlap {
            ratio + slot_in_win
        } else {
            slot_in_win
        };
        self.kv_state[write * cd..(write + 1) * cd].copy_from_slice(&kv);
        self.score_state[write * cd..(write + 1) * cd].copy_from_slice(&score);
        if !should_compress {
            return None;
        }
        let mut cand_kv: Vec<&[f32]> = Vec::new();
        let mut cand_sc: Vec<Vec<f32>> = Vec::new();
        if self.overlap {
            for i in 0..ratio {
                // previous window, first-half channels
                let src = i * cd;
                cand_kv.push(&self.kv_state[src..src + d]);
                cand_sc.push(self.score_state[src..src + d].to_vec());
            }
            for i in 0..ratio {
                // current window, second-half channels
                let src = (ratio + i) * cd + d;
                cand_kv.push(&self.kv_state[src..src + d]);
                cand_sc.push(self.score_state[src..src + d].to_vec());
            }
        } else {
            for i in 0..ratio {
                let src = i * cd;
                cand_kv.push(&self.kv_state[src..src + d]);
                cand_sc.push(self.score_state[src..src + d].to_vec());
            }
        }
        let mut row = pool_per_channel(&cand_kv, &cand_sc, d);
        if self.overlap {
            // shift: previous window <- current window (full coff*d rows)
            for i in 0..ratio {
                let (dst, src) = (i * cd, (ratio + i) * cd);
                self.kv_state.copy_within(src..src + cd, dst);
                self.score_state.copy_within(src..src + cd, dst);
            }
        }
        self.finish(&mut row, pos + 1 - ratio, freqs);
        Some(row)
    }
}

/// out[c] = sum_slots kv[slot][c] * softmax_slots(score[slot][c]) — the
/// softmax runs per channel across slots; empty kv slices mean a zero row.
fn pool_per_channel(cand_kv: &[&[f32]], cand_sc: &[Vec<f32>], d: usize) -> Vec<f32> {
    let n = cand_kv.len();
    let mut out = vec![0.0f32; d];
    for c in 0..d {
        let mut m = f32::NEG_INFINITY;
        for sc in cand_sc.iter() {
            m = m.max(sc[c]);
        }
        let mut denom = 0.0f32;
        let mut acc = 0.0f32;
        for s in 0..n {
            let e = (cand_sc[s][c] - m).exp();
            denom += e;
            if !cand_kv[s].is_empty() {
                acc += e * cand_kv[s][c];
            }
        }
        out[c] = acc / denom;
    }
    out
}
