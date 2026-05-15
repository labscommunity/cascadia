//! AVX-512 SIMD version of the int4 GEMV kernel.
//!
//! Inner loop strategy:
//!   - Load 16 bytes from packed (= 32 nibbles)
//!   - Split into low/high nibbles using AND + SHR + AND
//!   - Subtract 8 to get signed [-8, 7]
//!   - Convert int8 → int32 → f32 (vpmovsxbd + vcvtdq2ps)
//!   - Multiply by scale + input, accumulate in f32 vector register
//!
//! We process 32 columns per group at once (one full group). Across 224
//! groups per row, the kernel does ~224 mul-add chains per row. Rows are
//! parallelized across rayon.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
mod imp {
    use core::arch::x86_64::*;
    use rayon::prelude::*;

    use crate::format::bf16_bits_to_f32;
    use crate::GROUP_SIZE;

    /// AVX-512 path. Caller must check `is_x86_feature_detected!("avx512f")`.
    /// Falls back to scalar (kernel.rs) if unavailable.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub unsafe fn dequant_gemv_int4_avx512(
        packed: &[u8],
        scale_bits: &[u8],
        x: &[f32],
        n_rows: usize,
        k_cols: usize,
        y: &mut [f32],
    ) {
        assert_eq!(packed.len(), n_rows * (k_cols / 2));
        let n_groups = k_cols / GROUP_SIZE;
        assert_eq!(scale_bits.len(), n_rows * n_groups * 2);
        assert_eq!(x.len(), k_cols);
        assert_eq!(y.len(), n_rows);
        let row_stride_packed = k_cols / 2;

        y.par_iter_mut().enumerate().for_each(|(r, yy)| {
            let row_packed = &packed[r * row_stride_packed..(r + 1) * row_stride_packed];
            let row_scales = &scale_bits[r * n_groups * 2..(r + 1) * n_groups * 2];
            // Use an f32 accumulator vector then horizontal-sum at the end.
            let mut acc = _mm512_setzero_ps();
            for g in 0..n_groups {
                let scale_u16 = u16::from_le_bytes([row_scales[g * 2], row_scales[g * 2 + 1]]);
                let scale = bf16_bits_to_f32(scale_u16);
                let scale_v = _mm512_set1_ps(scale);
                // 16 bytes of packed = 32 nibbles (one group).
                let p_ptr = row_packed.as_ptr().add(g * (GROUP_SIZE / 2)) as *const __m128i;
                let pk = _mm_loadu_si128(p_ptr);
                // pk: 16 i8 bytes. Low nibble holds even columns, high nibble odd.
                let lo_mask = _mm_set1_epi8(0x0F);
                let low_nibbles = _mm_and_si128(pk, lo_mask); // 16 even columns
                let high_nibbles =
                    _mm_and_si128(_mm_srli_epi16::<4>(pk), lo_mask); // 16 odd columns
                // sign: subtract 8
                let bias = _mm_set1_epi8(8);
                let low_signed = _mm_sub_epi8(low_nibbles, bias);
                let high_signed = _mm_sub_epi8(high_nibbles, bias);
                // Interleave low/high to match original column order (col 0, 1, 2, 3, ...).
                // We want lanes [low0, high0, low1, high1, ..., low15, high15].
                let interleaved_lo = _mm_unpacklo_epi8(low_signed, high_signed); // cols 0..15
                let interleaved_hi = _mm_unpackhi_epi8(low_signed, high_signed); // cols 16..31
                // Convert 16 i8 → 16 i32 via sign extend
                let lo_i32 = _mm512_cvtepi8_epi32(interleaved_lo);
                let hi_i32 = _mm512_cvtepi8_epi32(interleaved_hi);
                // Convert to f32
                let lo_f = _mm512_cvtepi32_ps(lo_i32);
                let hi_f = _mm512_cvtepi32_ps(hi_i32);
                // Multiply by scale
                let lo_w = _mm512_mul_ps(lo_f, scale_v);
                let hi_w = _mm512_mul_ps(hi_f, scale_v);
                // Multiply elementwise by x[col] and accumulate
                let x_ptr = x.as_ptr().add(g * GROUP_SIZE) as *const f32;
                let x_lo = _mm512_loadu_ps(x_ptr);
                let x_hi = _mm512_loadu_ps(x_ptr.add(16));
                acc = _mm512_fmadd_ps(lo_w, x_lo, acc);
                acc = _mm512_fmadd_ps(hi_w, x_hi, acc);
            }
            *yy = _mm512_reduce_add_ps(acc);
        });
    }
}

#[cfg(target_arch = "x86_64")]
pub use imp::dequant_gemv_int4_avx512;

/// Wrapper: pick AVX-512 if available, else fall back to the scalar
/// `kernel::dequant_gemv_int4`.
pub fn dequant_gemv_int4_auto(
    packed: &[u8],
    scale_bits: &[u8],
    x: &[f32],
    n_rows: usize,
    k_cols: usize,
    y: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: detected at runtime, kernel only reads via aligned-or-unaligned
            // intrinsics and writes into a properly-sized output slice.
            unsafe {
                dequant_gemv_int4_avx512(packed, scale_bits, x, n_rows, k_cols, y);
            }
            return;
        }
    }
    crate::kernel::dequant_gemv_int4(packed, scale_bits, x, n_rows, k_cols, y);
}
