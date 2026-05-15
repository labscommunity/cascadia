//! int4 GEMV kernel — y = weight @ x for one row at a time.
//!
//! Weight format: int4 with group_size=32, zero_point=8, symmetric. Each
//! group of 32 weights shares one bf16 scale. On disk the int4 values
//! sit in int32 columns of 8 nibbles each (LE), so reading the i'th
//! input column from row `r` is:
//!
//! ```text
//! byte = packed[r * stride + i / 2]
//! nibble = (byte >> (4 * (i & 1))) & 0x0F   // i even → low, odd → high
//! signed = (nibble as i32) - 8
//! weight_fp32 = signed as f32 * scale_fp32  // scale = scale[r, i/32]
//! ```
//!
//! `dequant_gemv_int4` runs that for one matmul on the full N output
//! rows in parallel via rayon.

use half::bf16;
use rayon::prelude::*;

use crate::format::bf16_bits_to_f32;
use crate::GROUP_SIZE;

/// Compute `y[r] = sum_c (signed_nibble(weight[r, c]) * scale[r, c/32]) * x[c]`
/// for r in 0..n_rows and c in 0..k_cols. n_groups = k_cols / GROUP_SIZE.
///
/// - `packed`:  bytes, len = n_rows * (k_cols / 2)
/// - `scale_bits`: bf16 raw bits, len = n_rows * n_groups * 2
/// - `x`:       f32 slice of length k_cols
/// - `y`:       f32 output slice of length n_rows (will be written)
pub fn dequant_gemv_int4(
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
        let row_scales_bits = &scale_bits[r * n_groups * 2..(r + 1) * n_groups * 2];
        let mut acc = 0.0f32;
        for g in 0..n_groups {
            let scale_bits_u16 =
                u16::from_le_bytes([row_scales_bits[g * 2], row_scales_bits[g * 2 + 1]]);
            let scale = bf16_bits_to_f32(scale_bits_u16);
            let group_packed = &row_packed[g * (GROUP_SIZE / 2)..(g + 1) * (GROUP_SIZE / 2)];
            // 16 bytes = 32 nibbles
            let mut group_dot = 0.0f32;
            for i in 0..(GROUP_SIZE / 2) {
                let byte = group_packed[i];
                let lo_nibble = (byte & 0x0F) as i32;
                let hi_nibble = ((byte >> 4) & 0x0F) as i32;
                let lo_signed = lo_nibble - 8;
                let hi_signed = hi_nibble - 8;
                let col = g * GROUP_SIZE + i * 2;
                group_dot += (lo_signed as f32) * x[col];
                group_dot += (hi_signed as f32) * x[col + 1];
            }
            acc += scale * group_dot;
        }
        *yy = acc;
    });
}

/// Run one expert's full FFN: y = down @ (silu(gate @ x) ⊙ (up @ x)).
/// Inputs and output are bf16 (input via &[bf16], output via &mut [bf16]).
pub fn expert_forward(
    x_bf16: &[bf16],
    gate_packed: &[u8],
    gate_scale: &[u8],
    up_packed: &[u8],
    up_scale: &[u8],
    down_packed: &[u8],
    down_scale: &[u8],
    out_bf16: &mut [bf16],
) {
    let hidden = x_bf16.len();
    let intermediate = gate_scale.len() / 2 / (hidden / GROUP_SIZE);

    // Convert input to f32 once.
    let mut x_f32 = vec![0.0f32; hidden];
    for (i, b) in x_bf16.iter().enumerate() {
        x_f32[i] = b.to_f32();
    }

    let mut gate_out = vec![0.0f32; intermediate];
    let mut up_out = vec![0.0f32; intermediate];
    crate::kernel_avx512::dequant_gemv_int4_auto(
        gate_packed, gate_scale, &x_f32, intermediate, hidden, &mut gate_out);
    crate::kernel_avx512::dequant_gemv_int4_auto(
        up_packed, up_scale, &x_f32, intermediate, hidden, &mut up_out);

    // intermediate = silu(gate_out) * up_out
    let mut inter = vec![0.0f32; intermediate];
    for i in 0..intermediate {
        let g = gate_out[i];
        // silu = g * sigmoid(g) = g / (1 + exp(-g))
        let silu = g / (1.0 + (-g).exp());
        inter[i] = silu * up_out[i];
    }

    // down @ inter -> out, but down is [hidden, intermediate]
    let mut out_f32 = vec![0.0f32; hidden];
    crate::kernel_avx512::dequant_gemv_int4_auto(
        down_packed,
        down_scale,
        &inter,
        hidden,
        intermediate,
        &mut out_f32,
    );

    for (i, v) in out_f32.iter().enumerate() {
        out_bf16[i] = bf16::from_f32(*v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tiny end-to-end test: zero weight → zero output regardless of input.
    #[test]
    fn zero_weight_zero_output() {
        // Weight zero-point is 8, so to get a true zero you need nibble=8.
        // 0x88 byte = both nibbles = 8 = signed 0.
        let n_rows = 4;
        let k_cols = 32; // one group
        let packed = vec![0x88u8; n_rows * k_cols / 2];
        // scale = 1.0 in bf16 (0x3f80)
        let scale_bits = vec![0x80, 0x3fu8].repeat(n_rows * (k_cols / GROUP_SIZE));
        let x: Vec<f32> = (0..k_cols).map(|i| i as f32 * 0.1).collect();
        let mut y = vec![999.0f32; n_rows];
        dequant_gemv_int4(&packed, &scale_bits, &x, n_rows, k_cols, &mut y);
        for &v in &y {
            assert!(v.abs() < 1e-6, "expected ~0, got {}", v);
        }
    }

    // One-hot weight test: signed=1 at col 0 of row 0, zero everywhere
    // else, scale=1.0 → y[0] = 1*x[0], y[1..]=0.
    #[test]
    fn unit_weight_picks_first_column() {
        let n_rows = 1;
        let k_cols = 32;
        // We need nibble=9 (signed +1) at col 0. That's the low nibble of
        // byte 0. So byte 0 = 0x89 (high=8, low=9).
        let mut packed = vec![0x88u8; n_rows * k_cols / 2];
        packed[0] = 0x89;
        // scale 1.0
        let scale_bits: Vec<u8> = vec![0x80, 0x3f]
            .into_iter()
            .cycle()
            .take(n_rows * (k_cols / GROUP_SIZE) * 2)
            .collect();
        let x: Vec<f32> = (0..k_cols).map(|i| i as f32 + 1.0).collect(); // 1, 2, 3, ...
        let mut y = vec![0.0f32; n_rows];
        dequant_gemv_int4(&packed, &scale_bits, &x, n_rows, k_cols, &mut y);
        // y[0] should be x[0] = 1.0
        assert!((y[0] - 1.0).abs() < 1e-5, "expected 1.0, got {}", y[0]);
    }
}
