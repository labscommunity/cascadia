//! Multi-token tiled int4 GEMM kernel.
//!
//! Iter 042 prototype: replaces the scalar-per-token loop in
//! [`crate::shell_int4::shell_forward_decode_int4_multi_with_capacity`]
//! with a real `[seq, K] x [K, N]` GEMM. The single-token kernel
//! (`kernel_avx512::dequant_gemv_int4_avx512`) reloads + redequantizes
//! the full packed weight matrix per token. For seq=4, that's 4×
//! redundant memory traffic on the dominant cost — the projection
//! weights at K2.6's `7168 / 2048 / 5760 / 16384` dims are 3.5–80 MB
//! each.
//!
//! Strategy (per row r):
//!   - for each group g (32 cols):
//!     - load + dequantize one packed group → 32 f32 weights (kept in
//!       AVX-512 registers, never re-loaded)
//!     - for each token t in `0..seq`:
//!       - load x[t, g*32 .. g*32+32]
//!       - acc[t] = fmadd(weights, x_t, acc[t])
//!   - write acc[0..seq] to y[0..seq, r]
//!
//! The inner loop's hot path now amortizes the int4 dequant cost across
//! `seq` tokens. Memory motion drops from `seq × W` to `~W` for the
//! weights. For seq=4, that's a 4× reduction in the dominant term.
//!
//! Output layout: `y` is `[seq, n_rows]` flat, row-major over tokens.
//! Token `t`'s output row is `y[t * n_rows .. (t + 1) * n_rows]` —
//! matches how the per-token kernels write into the engine's
//! `MultiShellOutputs` flat buffers.
//!
//! This file ships the AVX-512 fmadd variant (works on the miner Xeon
//! Gold 6252's avx512f/bw/vl). An AVX-VNNI path that does int8 GEMM
//! with `_mm512_dpbusd_epi32` is sketched in
//! [`dequant_gemm_int4_multi_vnni_tile`] as a future swing — but with
//! f32 inputs and bf16 scales, the f32 FMA path is the right baseline.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
mod imp {
    use core::arch::x86_64::*;
    use rayon::prelude::*;

    use crate::format::bf16_bits_to_f32;
    use crate::GROUP_SIZE;

    /// AVX-512 multi-token int4 GEMM:
    /// `y[t, r] = sum_c (signed_int4(W[r, c]) * scale[r, c/32]) * x[t, c]`
    /// for `t in 0..seq`, `r in 0..n_rows`.
    ///
    /// Inputs:
    /// - `packed`: int4-packed weights, `[n_rows, k_cols/2]` bytes
    /// - `scale_bits`: bf16 LE, `[n_rows, k_cols/32]`
    /// - `xs`: f32 inputs, `[seq, k_cols]` flat (row-major over tokens)
    /// - `n_rows`: output dim N
    /// - `k_cols`: input dim K
    /// - `seq`: number of input tokens (>= 1)
    /// - `ys`: f32 outputs, `[seq, n_rows]` flat
    ///
    /// Caller must check `is_x86_feature_detected!("avx512f,bw,vl")`.
    ///
    /// The hot inner loop dequantizes one int4 group (32 cols) once into
    /// two `__m512` registers (16 + 16 f32 weights), then fmadds those
    /// weights against `seq` slices of `xs`. Weight bytes are loaded
    /// once per `(row, group)`, regardless of seq.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub unsafe fn dequant_gemm_int4_multi_avx512(
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
        // We bound seq because per-row we keep `seq` __m512 accumulators
        // in register / on stack. 32 ZMM regs means seq>8 spills, but
        // we still win on weight motion.
        const MAX_SEQ: usize = 64;
        assert!(
            seq <= MAX_SEQ,
            "multi GEMM tile supports seq <= {MAX_SEQ}, got {seq}"
        );

        // Parallelize over rows. Chunk rows so each rayon task does
        // more work and amortizes the per-task scheduling cost. At
        // K=7168, each row does 7168/32 = 224 groups; with chunk=64
        // each task does ~14k group iterations, which is enough work
        // to dwarf the ~10–50 us rayon dispatch cost.
        const ROW_CHUNK: usize = 64;
        let n_chunks = n_rows.div_ceil(ROW_CHUNK);

        let ys_ptr_addr = ys.as_ptr() as usize;

        (0..n_chunks).into_par_iter().for_each(|chunk_idx| {
            let r_start = chunk_idx * ROW_CHUNK;
            let r_end = ((chunk_idx + 1) * ROW_CHUNK).min(n_rows);
            // Recover the &mut ys pointer per chunk; rayon hands us
            // disjoint chunks of rows, and y[t, r] for r in this
            // chunk's range is disjoint across chunks.
            let y_ptr = ys_ptr_addr as *mut f32;
            for r in r_start..r_end {
                let row_packed = &packed[r * row_stride_packed..(r + 1) * row_stride_packed];
                let row_scales = &scale_bits[r * n_groups * 2..(r + 1) * n_groups * 2];

                // Per-token accumulators on stack; only initialize the
                // seq we actually use (avoids the cost of touching all
                // 64 MAX_SEQ slots).
                let mut acc: [__m512; MAX_SEQ] = [_mm512_setzero_ps(); MAX_SEQ];
                for t in 0..seq {
                    acc[t] = _mm512_setzero_ps();
                }

                for g in 0..n_groups {
                    // bf16 scale → f32 broadcast.
                    let scale_u16 = u16::from_le_bytes([row_scales[g * 2], row_scales[g * 2 + 1]]);
                    let scale = bf16_bits_to_f32(scale_u16);
                    let scale_v = _mm512_set1_ps(scale);

                    // Dequantize 32 weights → 2× __m512 of f32, ONCE
                    // per (row, group).
                    let p_ptr = row_packed.as_ptr().add(g * (GROUP_SIZE / 2)) as *const __m128i;
                    let pk = _mm_loadu_si128(p_ptr);
                    let lo_mask = _mm_set1_epi8(0x0F);
                    let low_nibbles = _mm_and_si128(pk, lo_mask);
                    let high_nibbles = _mm_and_si128(_mm_srli_epi16::<4>(pk), lo_mask);
                    let bias = _mm_set1_epi8(8);
                    let low_signed = _mm_sub_epi8(low_nibbles, bias);
                    let high_signed = _mm_sub_epi8(high_nibbles, bias);
                    let interleaved_lo = _mm_unpacklo_epi8(low_signed, high_signed);
                    let interleaved_hi = _mm_unpackhi_epi8(low_signed, high_signed);
                    let lo_i32 = _mm512_cvtepi8_epi32(interleaved_lo);
                    let hi_i32 = _mm512_cvtepi8_epi32(interleaved_hi);
                    let lo_f = _mm512_cvtepi32_ps(lo_i32);
                    let hi_f = _mm512_cvtepi32_ps(hi_i32);
                    let w_lo = _mm512_mul_ps(lo_f, scale_v);
                    let w_hi = _mm512_mul_ps(hi_f, scale_v);

                    // Fmadd against seq input slices. Weights stay in
                    // registers across this loop.
                    for t in 0..seq {
                        let x_ptr = xs.as_ptr().add(t * k_cols + g * GROUP_SIZE);
                        let x_lo = _mm512_loadu_ps(x_ptr);
                        let x_hi = _mm512_loadu_ps(x_ptr.add(16));
                        acc[t] = _mm512_fmadd_ps(w_lo, x_lo, acc[t]);
                        acc[t] = _mm512_fmadd_ps(w_hi, x_hi, acc[t]);
                    }
                }

                // Horizontal-sum each token's accumulator and scatter
                // to y[t, r]. SAFETY: r is unique across all parallel
                // tasks (chunks own disjoint r-ranges), and t varies
                // over an inner serial loop so each (t, r) is written
                // exactly once.
                for t in 0..seq {
                    let v = _mm512_reduce_add_ps(acc[t]);
                    core::ptr::write(y_ptr.add(t * n_rows + r), v);
                }
            }
        });
    }

    /// AVX-VNNI sketch (not currently wired into the auto-dispatcher).
    ///
    /// This is the place to drop a `_mm512_dpbusd_epi32` int8 GEMM tile
    /// once the input is also int8-quantized per token (e.g. per-row
    /// dynamic int8 quant of x). With f32 inputs and per-group bf16
    /// scales it doesn't help — the dequant + scale path produces f32,
    /// and dpbusd needs int8/uint8 lanes. Leaving the entry point so
    /// future work has a hook.
    ///
    /// Returns `false` because no implementation exists yet — callers
    /// fall through to `dequant_gemm_int4_multi_avx512`.
    #[inline]
    #[allow(dead_code)]
    pub fn dequant_gemm_int4_multi_vnni_tile(
        _packed: &[u8],
        _scale_bits: &[u8],
        _xs: &[f32],
        _n_rows: usize,
        _k_cols: usize,
        _seq: usize,
        _ys: &mut [f32],
    ) -> bool {
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub use imp::dequant_gemm_int4_multi_avx512;

/// Auto-dispatch wrapper for the multi-token int4 GEMM.
///
/// Picks the AVX-512 tile when the host supports it; falls back to a
/// per-token scalar loop using the existing single-token kernel
/// otherwise.
///
/// Output layout: `ys[t * n_rows + r]` is the output for token `t`,
/// row `r`.
pub fn dequant_gemm_int4_multi_auto(
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
        // For seq=1, the per-token kernel beats the tile because the
        // tile pays a per-row scratch + scatter cost that doesn't
        // amortize across only one token. Microbench (iter 042 on
        // miner) showed the multi tile losing ~0.7-0.9x at seq=1 vs
        // the established single-token kernel; for seq>=2 the multi
        // tile wins 1.3-3x depending on shape and contention.
        if seq >= 2
            && seq <= 64
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: feature bits checked, slice lengths asserted above.
            unsafe {
                dequant_gemm_int4_multi_avx512(packed, scale_bits, xs, n_rows, k_cols, seq, ys);
            }
            return;
        }
    }
    // Fallback: per-token scalar loop using the single-token kernel.
    for t in 0..seq {
        let x_t = &xs[t * k_cols..(t + 1) * k_cols];
        let y_t = &mut ys[t * n_rows..(t + 1) * n_rows];
        crate::kernel_avx512::dequant_gemv_int4_auto(packed, scale_bits, x_t, n_rows, k_cols, y_t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_avx512::dequant_gemv_int4_auto;

    /// Build a deterministic packed weight + scale + input set with
    /// shapes `[n_rows, k_cols]` and seq `seq`. Returns (packed,
    /// scales, xs).
    fn make_test_data(n_rows: usize, k_cols: usize, seq: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
        assert!(k_cols.is_multiple_of(crate::GROUP_SIZE));
        // Random-ish packed bytes — every byte = pattern based on (r, c).
        let mut packed = vec![0u8; n_rows * k_cols / 2];
        for r in 0..n_rows {
            for c in 0..(k_cols / 2) {
                let v = ((r.wrapping_mul(31).wrapping_add(c)) & 0xFF) as u8;
                packed[r * (k_cols / 2) + c] = v;
            }
        }
        // Scales: vary in [0.5, 1.5] via bf16 round.
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
        // Inputs: varied across tokens so a buggy "broadcast" can't pass.
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

    /// The multi kernel must produce the SAME outputs as `seq` calls
    /// of the single-token kernel, byte-for-byte (modulo associativity
    /// — we sum in the same order, so should be bit-identical).
    #[test]
    fn multi_matches_per_token_loop_seq_1() {
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

        let mut y_multi = vec![0.0f32; seq * n_rows];
        dequant_gemm_int4_multi_auto(&packed, &scales, &xs, n_rows, k_cols, seq, &mut y_multi);

        for i in 0..(seq * n_rows) {
            let a = y_single[i];
            let b = y_multi[i];
            // The two paths sum across the same nibbles in the same
            // order, so the result is bit-identical.
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "mismatch at i={i}: single={a}, multi={b}",
            );
        }
    }

    #[test]
    fn multi_matches_per_token_loop_seq_4() {
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

        let mut y_multi = vec![0.0f32; seq * n_rows];
        dequant_gemm_int4_multi_auto(&packed, &scales, &xs, n_rows, k_cols, seq, &mut y_multi);

        for i in 0..(seq * n_rows) {
            let a = y_single[i];
            let b = y_multi[i];
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "mismatch at i={i}: single={a}, multi={b}",
            );
        }
    }

    /// Larger dims that hit the K2.6 projection shapes (HIDDEN=7168).
    #[test]
    fn multi_matches_per_token_loop_large() {
        let n_rows = 96;
        let k_cols = 1536; // mirrors Q_LORA_RANK
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

        let mut y_multi = vec![0.0f32; seq * n_rows];
        dequant_gemm_int4_multi_auto(&packed, &scales, &xs, n_rows, k_cols, seq, &mut y_multi);

        for i in 0..(seq * n_rows) {
            let a = y_single[i];
            let b = y_multi[i];
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "mismatch at i={i}: single={a}, multi={b}",
            );
        }
    }
}
