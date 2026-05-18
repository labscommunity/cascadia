//! Rust head forward — final RMSNorm + lm_head GEMV — with optional
//! per-rank vocab slicing for tensor parallelism.
//!
//! The K2.6 head is just two ops in a chain:
//!
//! ```text
//!     h_normed = rms_norm(h, model.norm.weight)        # [hidden]
//!     logits   = h_normed @ lm_head.weight.T           # [vocab]
//! ```
//!
//! Today the OV head IR bundles both. On the 2-box pipeline this runs
//! single-threaded on the last rank — per iter 003 instrumentation it
//! takes ~139 ms / token (~1.5% of decode time, smaller than the
//! shells but a clean architectural lever).
//!
//! This module is the Rust replacement and the substrate for **head
//! tensor parallelism**: each rank holds a vocab-row slice of
//! `lm_head.weight`, computes its partial logits in parallel, and the
//! sampling rank concatenates before sampling.
//!
//! Critical numerical contract: `rms_norm` is computed over the FULL
//! `[hidden]` vector and must be byte-identical across ranks. There is
//! no way to slice the normalization itself; what gets sliced is the
//! `[vocab, hidden]` lm_head matrix along its leading (vocab) dim. Both
//! the per-rank partial computation and the concatenation it feeds into
//! are exact (no overlap, no overlap-and-add, no all-reduce sum), so
//! TP introduces zero numerical error vs the single-rank head.

use crate::kernel_bf16::bf16_gemv_auto;

/// Match `shell::RMS_NORM_EPS`. Duplicated here so this module does not
/// pull a `pub use` from the (heavily-internal) shell module.
const RMS_NORM_EPS: f32 = 1.0e-6;

/// One rank's slice of the head: a vocab row range and the bf16 weights
/// for that range.
///
/// `weights_bf16` is `(vocab_end - vocab_start) * hidden * 2` bytes
/// row-major — exactly the byte range
/// [`crate::safetensors_source::SafetensorsExpertSource::lm_head_slice`]
/// returns.
///
/// `norm_bf16` is the full `[hidden]` final RMSNorm weight — every rank
/// loads the entire norm vector because normalization is not slice-able.
/// Its memory cost is tiny (`hidden * 2 = 14 KiB` for K2.6) so this
/// duplication has zero practical impact.
pub struct HeadSlice {
    pub vocab_start: usize,
    pub vocab_end: usize,
    pub hidden: usize,
    norm_bf16: &'static [u8],
    weights_bf16: &'static [u8],
}

impl HeadSlice {
    /// Construct a HeadSlice from raw safetensors byte slices.
    ///
    /// Panics on inconsistent input: `weights_bf16.len()` must equal
    /// `(vocab_end - vocab_start) * hidden * 2` and `norm_bf16.len()`
    /// must equal `hidden * 2`. The runtime constructor in
    /// `tahoma-engine-sparse-moe` validates these against the manifest
    /// before constructing.
    pub fn new(
        vocab_start: usize,
        vocab_end: usize,
        hidden: usize,
        norm_bf16: &'static [u8],
        weights_bf16: &'static [u8],
    ) -> Self {
        let n_rows = vocab_end
            .checked_sub(vocab_start)
            .expect("vocab_end >= vocab_start");
        assert_eq!(
            weights_bf16.len(),
            n_rows * hidden * 2,
            "HeadSlice: weights bytes {} != ({}-{}) * {} * 2",
            weights_bf16.len(),
            vocab_end,
            vocab_start,
            hidden
        );
        assert_eq!(
            norm_bf16.len(),
            hidden * 2,
            "HeadSlice: norm bytes {} != {} * 2",
            norm_bf16.len(),
            hidden
        );
        Self {
            vocab_start,
            vocab_end,
            hidden,
            norm_bf16,
            weights_bf16,
        }
    }

    /// Number of logits this slice produces.
    pub fn slice_len(&self) -> usize {
        self.vocab_end - self.vocab_start
    }

    /// Run RMSNorm + GEMV on `h_f32` (length = hidden) and return the
    /// partial logits for this slice (`vocab_start..vocab_end`).
    ///
    /// Layout matches the OV head's last-position output: a flat f32
    /// vector. The caller is responsible for concatenating partials
    /// across ranks before sampling.
    pub fn forward_partial(&self, h_f32: &[f32]) -> Vec<f32> {
        assert_eq!(h_f32.len(), self.hidden);
        let n_rows = self.slice_len();
        let h_normed = rms_norm_bf16(h_f32, self.norm_bf16, self.hidden);
        let mut out = vec![0.0f32; n_rows];
        bf16_gemv_auto(self.weights_bf16, &h_normed, n_rows, self.hidden, &mut out);
        out
    }

    /// Equivalent to `forward_partial(h_f32)` but takes a caller-owned
    /// `h_normed` (already through the final RMSNorm). Used by the
    /// last-rank engine when it has already computed `h_normed` once
    /// and wants to dispatch the GEMV against its own slice without
    /// re-running the (cheap) normalization.
    pub fn gemv_only(&self, h_normed: &[f32]) -> Vec<f32> {
        assert_eq!(h_normed.len(), self.hidden);
        let n_rows = self.slice_len();
        let mut out = vec![0.0f32; n_rows];
        bf16_gemv_auto(self.weights_bf16, h_normed, n_rows, self.hidden, &mut out);
        out
    }

    /// Run RMSNorm on `h_f32` (length = hidden) and return the
    /// normalized vector. This is the substrate shared between all
    /// ranks before any vocab slicing.
    pub fn norm_only(&self, h_f32: &[f32]) -> Vec<f32> {
        rms_norm_bf16(h_f32, self.norm_bf16, self.hidden)
    }
}

/// Final pre-head RMSNorm. Computed in f64 for the variance sum to keep
/// rounding parity with what the OV head produces (the OV head's
/// constant-folded RMSNorm computes the variance as a single fp32 sum;
/// for hidden=7168 the f32 sum accumulates noticeable rounding noise.
/// f64-accumulate matches the reference k25_generate.py path).
fn rms_norm_bf16(x: &[f32], weight_bf16: &[u8], dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), dim);
    assert_eq!(weight_bf16.len(), dim * 2);
    let mut var: f64 = 0.0;
    for v in x.iter() {
        var += (*v as f64) * (*v as f64);
    }
    let mean_sq = (var / dim as f64) as f32;
    let inv = (mean_sq + RMS_NORM_EPS).sqrt().recip();
    let mut out = vec![0.0f32; dim];
    for i in 0..dim {
        let lo = weight_bf16[i * 2];
        let hi = weight_bf16[i * 2 + 1];
        let bits = ((hi as u32) << 8) | (lo as u32);
        let w = f32::from_bits(bits << 16);
        out[i] = x[i] * inv * w;
    }
    out
}

/// Concatenate per-rank partial logits in vocab-row order into a single
/// `[vocab]` vector. Used by the sampling rank after gathering all
/// `FrameKind::HeadPartial` frames.
///
/// `partials` is an ordered list of `(vocab_start, partial_logits)`
/// pairs. The function sorts by `vocab_start` and copies into the
/// output. Gaps and overlaps are rejected — every vocab slot must be
/// covered by exactly one partial.
///
/// Returning a `Result` rather than panicking because the rank-0
/// caller's slicing is config-driven; a partition mismatch (e.g. the
/// total ranks count drifted between two boxes mid-deploy) should
/// surface as a recoverable error so the engine can fall back to the
/// single-rank head path on the sampling rank.
pub fn concat_partials(
    vocab_size: usize,
    partials: &[(usize, &[f32])],
) -> Result<Vec<f32>, String> {
    let mut sorted: Vec<(usize, &[f32])> = partials.iter().map(|&(s, p)| (s, p)).collect();
    sorted.sort_by_key(|&(s, _)| s);
    let mut out = vec![0.0f32; vocab_size];
    let mut expect_start = 0usize;
    for (start, partial) in &sorted {
        if *start != expect_start {
            return Err(format!(
                "concat_partials: gap or overlap at vocab {start} (expected {expect_start})"
            ));
        }
        let end = start + partial.len();
        if end > vocab_size {
            return Err(format!(
                "concat_partials: slice end {end} > vocab_size {vocab_size}"
            ));
        }
        out[*start..end].copy_from_slice(partial);
        expect_start = end;
    }
    if expect_start != vocab_size {
        return Err(format!(
            "concat_partials: total covered {expect_start} != vocab_size {vocab_size}"
        ));
    }
    Ok(out)
}

/// Compute an even row-split of `vocab_size` across `total` ranks.
/// Returns `(start, end)` for `rank` (0-based). Mirrors the layer-split
/// pattern in `engine::even_moe_split`. The remainder rows go to the
/// FIRST ranks so the last rank's slice is never larger than any
/// other — keeps the sampling rank's GEMV no slower than its peers.
///
/// `total == 1` returns `(0, vocab_size)`.
pub fn even_vocab_split(vocab_size: usize, rank: u32, total: u32) -> (usize, usize) {
    if total <= 1 {
        return (0, vocab_size);
    }
    let total_usize = total as usize;
    let rank_usize = rank.min(total - 1) as usize;
    let per = vocab_size / total_usize;
    let rem = vocab_size % total_usize;
    let extras_before = rank_usize.min(rem);
    let my_extra = if rank_usize < rem { 1 } else { 0 };
    let start = rank_usize * per + extras_before;
    let end = start + per + my_extra;
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    /// Convert an `[n_rows, n_cols]` f32 matrix into bf16 bytes,
    /// row-major. Used by the tests below to fabricate fake lm_head
    /// weights.
    fn f32_matrix_to_bf16_bytes(matrix: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(matrix.len() * 2);
        for &v in matrix {
            let bits = bf16::from_f32(v).to_bits();
            out.push((bits & 0xff) as u8);
            out.push((bits >> 8) as u8);
        }
        out
    }

    fn f32_vec_to_bf16_bytes(v: &[f32]) -> Vec<u8> {
        f32_matrix_to_bf16_bytes(v)
    }

    #[test]
    fn even_vocab_split_uniform() {
        // 8 vocab across 2 ranks -> (0,4), (4,8).
        assert_eq!(even_vocab_split(8, 0, 2), (0, 4));
        assert_eq!(even_vocab_split(8, 1, 2), (4, 8));
    }

    #[test]
    fn even_vocab_split_with_remainder_front_loaded() {
        // 163840 across 3 = 54613, 54613, 54614? No: 163840/3 = 54613r1
        // -> rank0=54614, rank1=54613, rank2=54613.
        let (s0, e0) = even_vocab_split(163840, 0, 3);
        let (s1, e1) = even_vocab_split(163840, 1, 3);
        let (s2, e2) = even_vocab_split(163840, 2, 3);
        assert_eq!(s0, 0);
        assert_eq!(e0, s1);
        assert_eq!(e1, s2);
        assert_eq!(e2, 163840);
        // First rank picks up the +1 row.
        assert_eq!(e0 - s0, 54614);
        assert_eq!(e1 - s1, 54613);
        assert_eq!(e2 - s2, 54613);
    }

    #[test]
    fn even_vocab_split_single_rank() {
        assert_eq!(even_vocab_split(163840, 0, 1), (0, 163840));
    }

    #[test]
    fn forward_partial_identity_recovers_input() {
        // hidden = 4, vocab = 6. Identity-ish lm_head (each row a
        // one-hot at column row%hidden, so row i extracts h[i%hidden]).
        // Norm weight = all-1s; we send in an already-normalized h so
        // the rms_norm should also be ~all-1s.
        let hidden = 4usize;
        let vocab = 6usize;
        // Build h with unit RMS: x = [1, 1, 1, 1] / 1 (norm of all-1s
        // over dim=4 is 1 because var=1 already).
        let h: Vec<f32> = vec![1.0; hidden];

        let norm: Vec<f32> = vec![1.0; hidden];
        let norm_bytes = f32_vec_to_bf16_bytes(&norm);
        // lm_head[row, col] = 1 iff col == row % hidden else 0.
        let mut lm_full: Vec<f32> = vec![0.0; vocab * hidden];
        for row in 0..vocab {
            lm_full[row * hidden + row % hidden] = 1.0;
        }
        let lm_bytes = f32_matrix_to_bf16_bytes(&lm_full);

        // Make a static-living buffer for the test by leaking. This
        // is OK in a unit test — the leaked allocation is freed when
        // the test process exits.
        let norm_static: &'static [u8] = Box::leak(norm_bytes.into_boxed_slice());
        let lm_static: &'static [u8] = Box::leak(lm_bytes.into_boxed_slice());

        let slice = HeadSlice::new(0, vocab, hidden, norm_static, lm_static);
        let logits = slice.forward_partial(&h);
        assert_eq!(logits.len(), vocab);
        // After RMSNorm with all-1 weights on all-1 input, h_normed ≈
        // [1, 1, 1, 1]. Then each logit row picks one h_normed
        // element, so logits ≈ [1, 1, 1, 1, 1, 1].
        for (i, &v) in logits.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-2,
                "row {i}: got {v}, expected ~1.0 (rmsnorm + identity-row lm_head)"
            );
        }
    }

    #[test]
    fn forward_partial_slice_subset_matches_full() {
        // hidden = 8, vocab = 16. Build a deterministic lm_head and
        // compare:
        //   (a) full HeadSlice 0..16 → 16 logits
        //   (b) split into HeadSlice 0..8 and 8..16 → concat to 16
        // The two outputs must be byte-identical (slice math is exact).
        let hidden = 8usize;
        let vocab = 16usize;
        // Deterministic, bounded inputs to keep bf16 round-trip noise
        // in the last bit only.
        let h: Vec<f32> = (0..hidden).map(|i| 0.1 + 0.05 * i as f32).collect();
        let norm: Vec<f32> = (0..hidden).map(|i| 0.5 + 0.01 * i as f32).collect();
        let norm_bytes = f32_vec_to_bf16_bytes(&norm);
        let lm_full: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i as f32 * 0.001).sin() * 0.1)
            .collect();
        let lm_bytes = f32_matrix_to_bf16_bytes(&lm_full);

        let norm_static: &'static [u8] = Box::leak(norm_bytes.clone().into_boxed_slice());
        let lm_static: &'static [u8] = Box::leak(lm_bytes.clone().into_boxed_slice());

        let full = HeadSlice::new(0, vocab, hidden, norm_static, lm_static);
        let full_logits = full.forward_partial(&h);

        // Build the lower / upper slices by directly slicing the bf16
        // bytes — exactly what `SafetensorsExpertSource::lm_head_slice`
        // does at runtime.
        let mid = vocab / 2;
        let row_bytes = hidden * 2;
        let lower_static: &'static [u8] =
            Box::leak(lm_bytes[..mid * row_bytes].to_vec().into_boxed_slice());
        let upper_static: &'static [u8] =
            Box::leak(lm_bytes[mid * row_bytes..].to_vec().into_boxed_slice());

        let lower = HeadSlice::new(0, mid, hidden, norm_static, lower_static);
        let upper = HeadSlice::new(mid, vocab, hidden, norm_static, upper_static);
        let lo = lower.forward_partial(&h);
        let hi = upper.forward_partial(&h);
        assert_eq!(lo.len(), mid);
        assert_eq!(hi.len(), vocab - mid);

        let combined = concat_partials(vocab, &[(0, &lo), (mid, &hi)]).expect("concat");
        assert_eq!(combined.len(), vocab);
        for (i, (&a, &b)) in full_logits.iter().zip(combined.iter()).enumerate() {
            // Both paths apply the same RMSNorm to the same h, then a
            // contiguous GEMV — equal bf16 weights, equal f32
            // accumulators. Equality should be exact in this scalar
            // single-threaded reference; allow 1e-6 to be safe against
            // platform fp ordering (avx512 parallel reduction sums in
            // pairs vs scalar left-to-right).
            assert!(
                (a - b).abs() < 1e-4,
                "vocab {i}: full={a} concat={b} delta={}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn concat_partials_detects_gap() {
        let vocab = 10;
        let a: Vec<f32> = vec![0.1; 3];
        let b: Vec<f32> = vec![0.2; 5]; // covers 3..8 — gap [8..10]
        let res = concat_partials(vocab, &[(0, &a), (3, &b)]);
        assert!(res.is_err(), "expected gap error, got {res:?}");
    }

    #[test]
    fn concat_partials_detects_overlap() {
        let vocab = 10;
        let a: Vec<f32> = vec![0.1; 6]; // covers 0..6
        let b: Vec<f32> = vec![0.2; 6]; // would cover 4..10 → overlap
        let res = concat_partials(vocab, &[(0, &a), (4, &b)]);
        assert!(res.is_err(), "expected overlap error, got {res:?}");
    }

    #[test]
    fn concat_partials_complete_coverage_round_trips() {
        let vocab = 10;
        let a: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let b: Vec<f32> = (6..10).map(|i| i as f32).collect();
        let out = concat_partials(vocab, &[(0, &a), (6, &b)]).expect("concat");
        let expected: Vec<f32> = (0..10).map(|i| i as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn gemv_only_matches_forward_partial() {
        // gemv_only should produce identical logits to forward_partial
        // when the caller pre-computes the same normed vector. Used by
        // the engine to avoid double-normalizing once we have h_normed.
        let hidden = 4usize;
        let vocab = 4usize;
        let h: Vec<f32> = vec![0.5, -0.3, 0.8, 0.1];
        let norm: Vec<f32> = vec![1.0; hidden];
        let norm_bytes = f32_vec_to_bf16_bytes(&norm);
        let lm_full: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32) * 0.01).collect();
        let lm_bytes = f32_matrix_to_bf16_bytes(&lm_full);

        let norm_static: &'static [u8] = Box::leak(norm_bytes.into_boxed_slice());
        let lm_static: &'static [u8] = Box::leak(lm_bytes.into_boxed_slice());
        let slice = HeadSlice::new(0, vocab, hidden, norm_static, lm_static);

        let normed = slice.norm_only(&h);
        let via_gemv = slice.gemv_only(&normed);
        let via_full = slice.forward_partial(&h);
        for (i, (&a, &b)) in via_full.iter().zip(via_gemv.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "logit {i}: full={a} gemv={b}");
        }
    }
}
