//! Row-blocked multi-token int4 GEMM kernel (iter 046).
//!
//! Follow-up to iter 042 ([`crate::kernel_avx512_multi`]) targeting the
//! oproj shape (N=7168, K=8192, 28 MB int4 ≈ 14 MB packed). Iter 042
//! flagged oproj as "DRAM-bandwidth-bound" and suggested AMX as the
//! next swing. **`perf stat` on miner disproved that hypothesis:**
//!
//! ```text
//! LLC-load-misses  =  2.27% of LLC loads      → NOT DRAM-bound
//! L1d-load-misses  = 30.44% of L1 loads       → L2/L3-latency-bound
//! IPC              = 1.10 insn/cycle          → pipeline-stall-bound
//! ```
//!
//! The 14 MB int4 weight matrix fits in the miner Xeon Gold 6252's
//! 35.8 MiB L3 cache, and almost all weight loads hit L3 (97.7% hit
//! rate). The bottleneck is not DRAM throughput — it's redundant
//! input reads.
//!
//! ## What this kernel changes
//!
//! Iter 042's inner loop reads `xs[t, k_tile]` once per (row, group, t):
//!
//! ```text
//! for r in row_chunk(64):      ← rayon-parallel
//!   for g in 0..n_groups:
//!     for t in 0..seq:
//!       x = load xs[t, g*32 .. g*32+32]    ← seq*n_rows*n_groups loads
//!       acc[t] += w_g * x
//! ```
//!
//! For oproj at seq=8 / K=8192, that's `8 * 7168 * 256 = 14.7M` xs
//! loads per call = 14.7M * 64 B = ~940 MB of xs traffic. The unique xs
//! data is only `seq * K * 4 = 256 KB`, so each xs byte is touched
//! ~3700 times per call. Even with perfect L1 residency, that's a lot
//! of L1 bandwidth.
//!
//! The blocked kernel reuses xs across a small `RB=2` row sub-block:
//!
//! ```text
//! for r_block in row_chunk by RB(2):
//!   for g in 0..n_groups:
//!     w0 = dequant(packed[r_block,   g])
//!     w1 = dequant(packed[r_block+1, g])
//!     for t in 0..seq:
//!       x = load xs[t, g*32 .. g*32+32]    ← halved: seq*n_rows/2 loads
//!       acc[0][t] += w0 * x
//!       acc[1][t] += w1 * x
//! ```
//!
//! Each xs slice is read for two rows instead of one — xs L1 traffic
//! halves. At RB=2 the register budget is tight: 2*seq accumulators +
//! 2*2 weight regs + 2 x regs = 2*8 + 4 + 2 = 22 ZMM at seq=8, fits
//! in 32 ZMM with headroom.
//!
//! ## Hardware tested
//!
//! - **Miner Xeon Gold 6252** (Cascade Lake, 24 cores, avx512f+bw+vl+vnni)
//!   — bench target. No AMX (AMX is 4th-gen Xeon+ only).
//! - **Matias-02/03** (Lunar Lake Core Ultra 7 258V) — no AMX.
//!
//! There is no AMX hardware in the tahoma fleet as of 2026-05-18. An
//! AMX path would require Sapphire Rapids / Granite Rapids Xeon or
//! Lunar Lake-X (none currently available).
//!
//! ## What iter 046 SKIPPED
//!
//! - **AMX intrinsic kernel (option A).** No hardware available; would
//!   ship dead code.
//! - **DRAM-bandwidth profiling (option C, partial).** `perf stat`
//!   already showed L3 hit rate is high; further deep-dive would
//!   require uncore PMU access which the test harness lacks.
//!
//! Iter 046 chose **option B (improved AVX-512 for oproj)** via
//! [`dequant_gemm_int4_multi_avx512_blocked`] — row-blocked xs reuse.
//!
//! An earlier draft also tried explicit `_mm_prefetch` hints. Those
//! lost vs iter 042's tile across all seq tested (0.63–0.87x), because
//! the HW prefetcher already handles the sequential packed stream
//! well and the extra prefetch instructions added front-end pressure.
//! See iter 046 commit message for the negative-result details.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
mod imp {
    use core::arch::x86_64::*;
    use rayon::prelude::*;

    use crate::format::bf16_bits_to_f32;
    use crate::GROUP_SIZE;

    /// Row-blocked multi-token int4 GEMM.
    ///
    /// Same outputs as [`crate::kernel_avx512_multi::dequant_gemm_int4_multi_avx512`]
    /// (numerically equivalent, but FMAs are reordered slightly — for
    /// each output `y[t, r]` the sum across `g` is identical, so per-cell
    /// results match bit-for-bit).
    ///
    /// Caller must check `is_x86_feature_detected!("avx512f,bw,vl")`.
    ///
    /// ## Inputs
    /// - `packed`: int4-packed weights, `[n_rows, k_cols/2]` bytes
    /// - `scale_bits`: bf16 LE, `[n_rows, k_cols/32]`
    /// - `xs`: f32 inputs, `[seq, k_cols]` flat
    /// - `ys`: f32 outputs, `[seq, n_rows]` flat
    ///
    /// `seq` must be 1..=`MAX_SEQ` (= 16). For seq > MAX_SEQ the caller
    /// should fall back to the iter 042 multi tile.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub unsafe fn dequant_gemm_int4_multi_avx512_blocked(
        packed: &[u8],
        scale_bits: &[u8],
        xs: &[f32],
        n_rows: usize,
        k_cols: usize,
        seq: usize,
        ys: &mut [f32],
    ) {
        assert_eq!(packed.len(), n_rows * (k_cols / 2));
        let n_groups = k_cols / GROUP_SIZE;
        assert_eq!(scale_bits.len(), n_rows * n_groups * 2);
        assert_eq!(xs.len(), seq * k_cols);
        assert_eq!(ys.len(), seq * n_rows);
        assert!(seq >= 1);

        let row_stride_packed = k_cols / 2;

        // Register budget at RB=2: 2*seq accs + 2*2 weights + 2 x = 22 ZMM
        // at seq=8 (fits in 32 ZMM with 10 spare). Going RB=4 needs 4*8+10
        // = 42 ZMM which spills; benchmarks confirm RB=4 loses at seq=8.
        const RB: usize = 2;
        // MAX_SEQ=16 keeps per-row-block accumulator state at RB*MAX_SEQ
        // = 32 ZMM exactly; seq > 16 spills and the row blocking benefit
        // is overwhelmed by spill traffic.
        const MAX_SEQ: usize = 16;
        assert!(
            seq <= MAX_SEQ,
            "blocked tile supports seq <= {MAX_SEQ}, got {seq}"
        );

        const ROW_CHUNK: usize = 64;
        let n_chunks = n_rows.div_ceil(ROW_CHUNK);
        let ys_ptr_addr = ys.as_ptr() as usize;

        (0..n_chunks).into_par_iter().for_each(|chunk_idx| {
            let r_start = chunk_idx * ROW_CHUNK;
            let r_end = ((chunk_idx + 1) * ROW_CHUNK).min(n_rows);
            let y_ptr = ys_ptr_addr as *mut f32;

            // Walk the chunk in RB-sized sub-blocks. If the chunk size
            // isn't divisible by RB (it is — 64 % 2 = 0), the tail row(s)
            // fall through to the per-row loop below.
            let mut rb = r_start;
            while rb + RB <= r_end {
                // Per-(row_off, t) accumulators: RB * seq __m512 each.
                let mut acc: [[__m512; MAX_SEQ]; RB] = [[_mm512_setzero_ps(); MAX_SEQ]; RB];
                for r_off in 0..RB {
                    for t in 0..seq {
                        acc[r_off][t] = _mm512_setzero_ps();
                    }
                }

                // Per-row weight pointers (computed once outside the
                // group loop).
                let row_packed: [&[u8]; RB] = [
                    &packed[(rb) * row_stride_packed..(rb + 1) * row_stride_packed],
                    &packed[(rb + 1) * row_stride_packed..(rb + 2) * row_stride_packed],
                ];
                let row_scales: [&[u8]; RB] = [
                    &scale_bits[(rb) * n_groups * 2..(rb + 1) * n_groups * 2],
                    &scale_bits[(rb + 1) * n_groups * 2..(rb + 2) * n_groups * 2],
                ];

                for g in 0..n_groups {
                    // Dequant both rows' weights for this group. Each
                    // row produces (w_lo, w_hi) = 2 × __m512 of f32.
                    let lo_mask = _mm_set1_epi8(0x0F);
                    let bias = _mm_set1_epi8(8);

                    // Row 0.
                    let s0_u16 =
                        u16::from_le_bytes([row_scales[0][g * 2], row_scales[0][g * 2 + 1]]);
                    let scale0_v = _mm512_set1_ps(bf16_bits_to_f32(s0_u16));
                    let p0_ptr = row_packed[0].as_ptr().add(g * (GROUP_SIZE / 2)) as *const __m128i;
                    let pk0 = _mm_loadu_si128(p0_ptr);
                    let ln0 = _mm_and_si128(pk0, lo_mask);
                    let hn0 = _mm_and_si128(_mm_srli_epi16::<4>(pk0), lo_mask);
                    let ls0 = _mm_sub_epi8(ln0, bias);
                    let hs0 = _mm_sub_epi8(hn0, bias);
                    let il0 = _mm_unpacklo_epi8(ls0, hs0);
                    let ih0 = _mm_unpackhi_epi8(ls0, hs0);
                    let w0_lo =
                        _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(il0)), scale0_v);
                    let w0_hi =
                        _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(ih0)), scale0_v);

                    // Row 1.
                    let s1_u16 =
                        u16::from_le_bytes([row_scales[1][g * 2], row_scales[1][g * 2 + 1]]);
                    let scale1_v = _mm512_set1_ps(bf16_bits_to_f32(s1_u16));
                    let p1_ptr = row_packed[1].as_ptr().add(g * (GROUP_SIZE / 2)) as *const __m128i;
                    let pk1 = _mm_loadu_si128(p1_ptr);
                    let ln1 = _mm_and_si128(pk1, lo_mask);
                    let hn1 = _mm_and_si128(_mm_srli_epi16::<4>(pk1), lo_mask);
                    let ls1 = _mm_sub_epi8(ln1, bias);
                    let hs1 = _mm_sub_epi8(hn1, bias);
                    let il1 = _mm_unpacklo_epi8(ls1, hs1);
                    let ih1 = _mm_unpackhi_epi8(ls1, hs1);
                    let w1_lo =
                        _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(il1)), scale1_v);
                    let w1_hi =
                        _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(ih1)), scale1_v);

                    // For each token: load xs ONCE, fmadd into both rows'
                    // accumulators. This is the key reuse.
                    for t in 0..seq {
                        let x_ptr = xs.as_ptr().add(t * k_cols + g * GROUP_SIZE);
                        let x_lo = _mm512_loadu_ps(x_ptr);
                        let x_hi = _mm512_loadu_ps(x_ptr.add(16));
                        acc[0][t] = _mm512_fmadd_ps(w0_lo, x_lo, acc[0][t]);
                        acc[0][t] = _mm512_fmadd_ps(w0_hi, x_hi, acc[0][t]);
                        acc[1][t] = _mm512_fmadd_ps(w1_lo, x_lo, acc[1][t]);
                        acc[1][t] = _mm512_fmadd_ps(w1_hi, x_hi, acc[1][t]);
                    }
                }

                // Horizontal-sum and scatter to y[t, r].
                for r_off in 0..RB {
                    let r = rb + r_off;
                    for t in 0..seq {
                        let v = _mm512_reduce_add_ps(acc[r_off][t]);
                        core::ptr::write(y_ptr.add(t * n_rows + r), v);
                    }
                }
                rb += RB;
            }

            // Tail row (if any). With RB=2 and ROW_CHUNK=64 this only
            // hits for the final chunk where r_end - r_start might be
            // odd (only at the very last chunk if n_rows is odd).
            // n_rows = 7168 for oproj is even, but support odd dims.
            while rb < r_end {
                let r = rb;
                let row_packed = &packed[r * row_stride_packed..(r + 1) * row_stride_packed];
                let row_scales = &scale_bits[r * n_groups * 2..(r + 1) * n_groups * 2];

                let mut acc: [__m512; MAX_SEQ] = [_mm512_setzero_ps(); MAX_SEQ];
                for t in 0..seq {
                    acc[t] = _mm512_setzero_ps();
                }
                for g in 0..n_groups {
                    let scale_u16 = u16::from_le_bytes([row_scales[g * 2], row_scales[g * 2 + 1]]);
                    let scale_v = _mm512_set1_ps(bf16_bits_to_f32(scale_u16));
                    let p_ptr = row_packed.as_ptr().add(g * (GROUP_SIZE / 2)) as *const __m128i;
                    let pk = _mm_loadu_si128(p_ptr);
                    let lo_mask = _mm_set1_epi8(0x0F);
                    let low_nibbles = _mm_and_si128(pk, lo_mask);
                    let high_nibbles = _mm_and_si128(_mm_srli_epi16::<4>(pk), lo_mask);
                    let bias = _mm_set1_epi8(8);
                    let ls = _mm_sub_epi8(low_nibbles, bias);
                    let hs = _mm_sub_epi8(high_nibbles, bias);
                    let il = _mm_unpacklo_epi8(ls, hs);
                    let ih = _mm_unpackhi_epi8(ls, hs);
                    let w_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(il)), scale_v);
                    let w_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(ih)), scale_v);
                    for t in 0..seq {
                        let x_ptr = xs.as_ptr().add(t * k_cols + g * GROUP_SIZE);
                        let x_lo = _mm512_loadu_ps(x_ptr);
                        let x_hi = _mm512_loadu_ps(x_ptr.add(16));
                        acc[t] = _mm512_fmadd_ps(w_lo, x_lo, acc[t]);
                        acc[t] = _mm512_fmadd_ps(w_hi, x_hi, acc[t]);
                    }
                }
                for t in 0..seq {
                    let v = _mm512_reduce_add_ps(acc[t]);
                    core::ptr::write(y_ptr.add(t * n_rows + r), v);
                }
                rb += 1;
            }
        });
    }
}

#[cfg(target_arch = "x86_64")]
pub use imp::dequant_gemm_int4_multi_avx512_blocked;

/// Auto-dispatch wrapper for the row-blocked multi-token int4 GEMM.
///
/// Routes to the row-blocked AVX-512 tile when CPU features are
/// available and `seq` is in the optimized range; falls back to the
/// iter 042 multi tile for seq > 16 (the blocked tile's MAX_SEQ).
///
/// Output layout: `ys[t * n_rows + r]` is the output for token `t`,
/// row `r` — identical layout to
/// [`crate::kernel_avx512_multi::dequant_gemm_int4_multi_auto`].
pub fn dequant_gemm_int4_multi_blocked_auto(
    packed: &[u8],
    scale_bits: &[u8],
    xs: &[f32],
    n_rows: usize,
    k_cols: usize,
    seq: usize,
    ys: &mut [f32],
) {
    assert_eq!(xs.len(), seq * k_cols);
    assert_eq!(ys.len(), seq * n_rows);
    #[cfg(target_arch = "x86_64")]
    {
        // Microbench on miner (iter 046, Xeon Gold 6252) showed:
        //   seq=2: blocked ~0.6-1.0x of iter 042 (regression, RB=2
        //          register pressure overcomes the halved xs reuse).
        //   seq=4: blocked ~0.7-2.0x of iter 042 (mixed).
        //   seq=8: blocked ~1.1-1.6x of iter 042 (consistent win).
        //   seq=16: blocked ~1.2-1.7x of iter 042 (consistent win).
        //
        // Threshold seq>=4 chosen because that's where the blocked
        // tile starts to consistently win. At seq=2-3 the iter 042
        // tile's lower register pressure dominates the xs-reuse gain.
        if seq >= 4
            && seq <= 16
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            unsafe {
                dequant_gemm_int4_multi_avx512_blocked(
                    packed, scale_bits, xs, n_rows, k_cols, seq, ys,
                );
            }
            return;
        }
    }
    // Fallback chain: iter 042 multi tile if available (seq>=2), else
    // per-token scalar loop.
    if seq >= 2 {
        crate::kernel_avx512_multi::dequant_gemm_int4_multi_auto(
            packed, scale_bits, xs, n_rows, k_cols, seq, ys,
        );
    } else {
        for t in 0..seq {
            let x_t = &xs[t * k_cols..(t + 1) * k_cols];
            let y_t = &mut ys[t * n_rows..(t + 1) * n_rows];
            crate::kernel_avx512::dequant_gemv_int4_auto(
                packed, scale_bits, x_t, n_rows, k_cols, y_t,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_avx512::dequant_gemv_int4_auto;

    fn make_test_data(n_rows: usize, k_cols: usize, seq: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
        assert!(k_cols.is_multiple_of(crate::GROUP_SIZE));
        let mut packed = vec![0u8; n_rows * k_cols / 2];
        for r in 0..n_rows {
            for c in 0..(k_cols / 2) {
                let v = ((r.wrapping_mul(31).wrapping_add(c)) & 0xFF) as u8;
                packed[r * (k_cols / 2) + c] = v;
            }
        }
        let n_groups = k_cols / crate::GROUP_SIZE;
        let mut scales = vec![0u8; n_rows * n_groups * 2];
        for r in 0..n_rows {
            for g in 0..n_groups {
                let s = 0.5f32 + (((r * 7 + g * 3) % 11) as f32) * 0.1;
                let bits = bf16_round(s);
                let off = (r * n_groups + g) * 2;
                scales[off] = (bits & 0xFF) as u8;
                scales[off + 1] = (bits >> 8) as u8;
            }
        }
        let mut xs = vec![0.0f32; seq * k_cols];
        for t in 0..seq {
            for c in 0..k_cols {
                xs[t * k_cols + c] = ((t * 17 + c * 5) as f32).sin() * 0.5;
            }
        }
        (packed, scales, xs)
    }

    fn bf16_round(x: f32) -> u16 {
        let bits = x.to_bits();
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        (rounded >> 16) as u16
    }

    /// At seq=1, dispatch falls through to the per-token loop, which
    /// is bit-identical to the single-token kernel.
    #[test]
    fn blocked_matches_per_token_loop_seq_1() {
        let n_rows = 64;
        let k_cols = 128;
        let seq = 1;
        let (packed, scales, xs) = make_test_data(n_rows, k_cols, seq);

        let mut y_single = vec![0.0f32; seq * n_rows];
        for t in 0..seq {
            dequant_gemv_int4_auto(
                &packed,
                &scales,
                &xs[t * k_cols..(t + 1) * k_cols],
                n_rows,
                k_cols,
                &mut y_single[t * n_rows..(t + 1) * n_rows],
            );
        }

        let mut y_blocked = vec![0.0f32; seq * n_rows];
        dequant_gemm_int4_multi_blocked_auto(
            &packed,
            &scales,
            &xs,
            n_rows,
            k_cols,
            seq,
            &mut y_blocked,
        );

        for i in 0..(seq * n_rows) {
            assert_eq!(
                y_single[i].to_bits(),
                y_blocked[i].to_bits(),
                "mismatch at i={i}: single={}, blocked={}",
                y_single[i],
                y_blocked[i]
            );
        }
    }

    /// seq=4 hits the blocked path. The blocked tile reorders FMAs
    /// (per-row then per-token vs per-token then per-row) but the sum
    /// across `g` for each output cell is identical, so results are
    /// bit-identical.
    #[test]
    fn blocked_matches_per_token_loop_seq_4() {
        let n_rows = 64;
        let k_cols = 128;
        let seq = 4;
        let (packed, scales, xs) = make_test_data(n_rows, k_cols, seq);

        let mut y_single = vec![0.0f32; seq * n_rows];
        for t in 0..seq {
            dequant_gemv_int4_auto(
                &packed,
                &scales,
                &xs[t * k_cols..(t + 1) * k_cols],
                n_rows,
                k_cols,
                &mut y_single[t * n_rows..(t + 1) * n_rows],
            );
        }

        let mut y_blocked = vec![0.0f32; seq * n_rows];
        dequant_gemm_int4_multi_blocked_auto(
            &packed,
            &scales,
            &xs,
            n_rows,
            k_cols,
            seq,
            &mut y_blocked,
        );

        for i in 0..(seq * n_rows) {
            assert_eq!(
                y_single[i].to_bits(),
                y_blocked[i].to_bits(),
                "mismatch at i={i}: single={}, blocked={}",
                y_single[i],
                y_blocked[i]
            );
        }
    }

    /// Odd n_rows exercises the RB=2 tail path.
    #[test]
    fn blocked_matches_per_token_loop_odd_rows() {
        let n_rows = 65;
        let k_cols = 256;
        let seq = 4;
        let (packed, scales, xs) = make_test_data(n_rows, k_cols, seq);

        let mut y_single = vec![0.0f32; seq * n_rows];
        for t in 0..seq {
            dequant_gemv_int4_auto(
                &packed,
                &scales,
                &xs[t * k_cols..(t + 1) * k_cols],
                n_rows,
                k_cols,
                &mut y_single[t * n_rows..(t + 1) * n_rows],
            );
        }

        let mut y_blocked = vec![0.0f32; seq * n_rows];
        dequant_gemm_int4_multi_blocked_auto(
            &packed,
            &scales,
            &xs,
            n_rows,
            k_cols,
            seq,
            &mut y_blocked,
        );

        for i in 0..(seq * n_rows) {
            assert_eq!(
                y_single[i].to_bits(),
                y_blocked[i].to_bits(),
                "mismatch at i={i} (odd rows): single={}, blocked={}",
                y_single[i],
                y_blocked[i]
            );
        }
    }

    /// K2.6 oproj-sized correctness: K=8192, seq=8.
    #[test]
    fn blocked_matches_per_token_loop_oproj_seq_8() {
        let n_rows = 128; // smaller than real (7168) for test speed
        let k_cols = 8192;
        let seq = 8;
        let (packed, scales, xs) = make_test_data(n_rows, k_cols, seq);

        let mut y_single = vec![0.0f32; seq * n_rows];
        for t in 0..seq {
            dequant_gemv_int4_auto(
                &packed,
                &scales,
                &xs[t * k_cols..(t + 1) * k_cols],
                n_rows,
                k_cols,
                &mut y_single[t * n_rows..(t + 1) * n_rows],
            );
        }

        let mut y_blocked = vec![0.0f32; seq * n_rows];
        dequant_gemm_int4_multi_blocked_auto(
            &packed,
            &scales,
            &xs,
            n_rows,
            k_cols,
            seq,
            &mut y_blocked,
        );

        for i in 0..(seq * n_rows) {
            assert_eq!(
                y_single[i].to_bits(),
                y_blocked[i].to_bits(),
                "mismatch at i={i} (oproj seq=8): single={}, blocked={}",
                y_single[i],
                y_blocked[i]
            );
        }
    }

    /// Direct comparison against iter 042's multi tile. The blocked
    /// tile reorders fmadds across rows but the per-output sum order
    /// (across `g`) is preserved, so outputs are bit-identical.
    #[test]
    fn blocked_matches_iter042_multi_seq_8() {
        use crate::kernel_avx512_multi::dequant_gemm_int4_multi_auto;
        let n_rows = 128;
        let k_cols = 8192;
        let seq = 8;
        let (packed, scales, xs) = make_test_data(n_rows, k_cols, seq);

        let mut y_iter042 = vec![0.0f32; seq * n_rows];
        dequant_gemm_int4_multi_auto(&packed, &scales, &xs, n_rows, k_cols, seq, &mut y_iter042);

        let mut y_blocked = vec![0.0f32; seq * n_rows];
        dequant_gemm_int4_multi_blocked_auto(
            &packed,
            &scales,
            &xs,
            n_rows,
            k_cols,
            seq,
            &mut y_blocked,
        );

        for i in 0..(seq * n_rows) {
            assert_eq!(
                y_iter042[i].to_bits(),
                y_blocked[i].to_bits(),
                "mismatch at i={i} (vs iter042): iter042={}, blocked={}",
                y_iter042[i],
                y_blocked[i]
            );
        }
    }
}
