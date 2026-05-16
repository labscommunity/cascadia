//! BF16 GEMV kernel (weights bf16, input f32, output f32). The Kimi K2.6
//! shells' big matmuls (q_a_proj, q_b_proj, kv_a_proj, kv_b_proj,
//! o_proj, shared_experts) are stored as plain bf16 in the safetensors
//! shards — int4 is only used for the experts and the router-extracted
//! gather IRs. We mmap the safetensors slice directly and run the GEMV
//! in AVX-512 by lifting bf16 to f32 via `<<16` and using vfmadd.

#![allow(unsafe_op_in_unsafe_fn)]

use rayon::prelude::*;

/// Scalar reference: y[r] = sum_c W_bf16[r, c] * x[c] for r in 0..n, c in 0..k.
pub fn bf16_gemv_scalar(
    weight_bf16: &[u8],
    x: &[f32],
    n_rows: usize,
    k_cols: usize,
    y: &mut [f32],
) {
    assert_eq!(weight_bf16.len(), n_rows * k_cols * 2);
    assert_eq!(x.len(), k_cols);
    assert_eq!(y.len(), n_rows);

    y.par_iter_mut().enumerate().for_each(|(r, yy)| {
        let row_start = r * k_cols * 2;
        let mut acc = 0.0f32;
        for c in 0..k_cols {
            let lo = weight_bf16[row_start + c * 2];
            let hi = weight_bf16[row_start + c * 2 + 1];
            let bits = ((hi as u32) << 8) | (lo as u32);
            let w_f32 = f32::from_bits(bits << 16);
            acc += w_f32 * x[c];
        }
        *yy = acc;
    });
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use core::arch::x86_64::*;
    use rayon::prelude::*;

    /// AVX-512 GEMV: bf16 weight × f32 input → f32 output.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub unsafe fn bf16_gemv_avx512(
        weight_bf16: &[u8],
        x: &[f32],
        n_rows: usize,
        k_cols: usize,
        y: &mut [f32],
    ) {
        assert_eq!(weight_bf16.len(), n_rows * k_cols * 2);
        assert_eq!(x.len(), k_cols);
        assert_eq!(y.len(), n_rows);
        let row_stride = k_cols * 2;
        // We process 16 elements at a time (one AVX-512 vector).
        let k_main = k_cols & !15; // multiple of 16

        y.par_iter_mut().enumerate().for_each(|(r, yy)| {
            let row_ptr = weight_bf16.as_ptr().add(r * row_stride);
            let mut acc = _mm512_setzero_ps();
            let mut c = 0;
            while c < k_main {
                // Load 16 u16 (= 16 bf16) from weight
                let w_u16 = _mm256_loadu_si256(row_ptr.add(c * 2) as *const __m256i);
                // Zero-extend to u32 then shift left 16 to place in upper half.
                let w_u32_lo = _mm512_cvtepu16_epi32(w_u16);
                let w_f32 = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(w_u32_lo));
                // Load 16 f32 from x
                let x_v = _mm512_loadu_ps(x.as_ptr().add(c));
                acc = _mm512_fmadd_ps(w_f32, x_v, acc);
                c += 16;
            }
            let mut sum = _mm512_reduce_add_ps(acc);
            // Tail
            while c < k_cols {
                let off = r * row_stride + c * 2;
                let lo = *row_ptr.add(c * 2 - r * row_stride);
                let hi = *row_ptr.add(c * 2 - r * row_stride + 1);
                let _ = (off, lo, hi);
                let lo = *weight_bf16.as_ptr().add(off);
                let hi = *weight_bf16.as_ptr().add(off + 1);
                let bits = ((hi as u32) << 8) | (lo as u32);
                let w_f32 = f32::from_bits(bits << 16);
                sum += w_f32 * x[c];
                c += 1;
            }
            *yy = sum;
        });
    }
}

#[cfg(target_arch = "x86_64")]
pub use avx512::bf16_gemv_avx512;

pub fn bf16_gemv_auto(weight_bf16: &[u8], x: &[f32], n_rows: usize, k_cols: usize, y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            unsafe {
                bf16_gemv_avx512(weight_bf16, x, n_rows, k_cols, y);
            }
            return;
        }
    }
    bf16_gemv_scalar(weight_bf16, x, n_rows, k_cols, y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn gemv_eye() {
        // Identity 4x4 bf16: y[r] = x[r].
        let n = 4;
        let k = 4;
        let mut w_bytes = vec![0u8; n * k * 2];
        for r in 0..n {
            for c in 0..k {
                let val = if r == c { 1.0 } else { 0.0 };
                let bits = bf16::from_f32(val).to_bits();
                w_bytes[(r * k + c) * 2] = (bits & 0xff) as u8;
                w_bytes[(r * k + c) * 2 + 1] = ((bits >> 8) & 0xff) as u8;
            }
        }
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y_scalar = vec![0.0f32; n];
        let mut y_auto = vec![0.0f32; n];
        bf16_gemv_scalar(&w_bytes, &x, n, k, &mut y_scalar);
        bf16_gemv_auto(&w_bytes, &x, n, k, &mut y_auto);
        for r in 0..n {
            assert!((y_scalar[r] - x[r]).abs() < 1e-3);
            assert!((y_auto[r] - x[r]).abs() < 1e-3);
        }
    }

    #[test]
    fn gemv_match_scalar_random() {
        let n = 32;
        let k = 64;
        let mut w = vec![0u8; n * k * 2];
        for (i, b) in w.iter_mut().enumerate() {
            *b = ((i * 7919 + 31) & 0xFF) as u8;
        }
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 0.1).collect();
        let mut y_scalar = vec![0.0f32; n];
        let mut y_auto = vec![0.0f32; n];
        bf16_gemv_scalar(&w, &x, n, k, &mut y_scalar);
        bf16_gemv_auto(&w, &x, n, k, &mut y_auto);
        for r in 0..n {
            assert!(
                (y_scalar[r] - y_auto[r]).abs() < 1e-4,
                "row {}: {} vs {}",
                r,
                y_scalar[r],
                y_auto[r]
            );
        }
    }
}
