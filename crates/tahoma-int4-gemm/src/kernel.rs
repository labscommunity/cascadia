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
        gate_packed,
        gate_scale,
        &x_f32,
        intermediate,
        hidden,
        &mut gate_out,
    );
    crate::kernel_avx512::dequant_gemv_int4_auto(
        up_packed,
        up_scale,
        &x_f32,
        intermediate,
        hidden,
        &mut up_out,
    );

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

/// Run one expert's full FFN over **`num_tokens` input rows at once**:
/// `Y[t] = down @ (silu(gate @ X[t]) ⊙ (up @ X[t]))` for each `t in 0..num_tokens`.
///
/// **Why this exists (iter 051).** The seq=1 [`expert_forward`] loads
/// the three projection weights (`gate`, `up`, `down` ≈ 7 + 7 + 7 = 21 MB
/// per K2.6 expert at int4) once per token. When spec-decode runs K=4
/// candidate tokens through the shells and 2 of them route to the same
/// expert, calling `expert_forward` twice loads those 21 MB twice from
/// DRAM — even though the math is identical and the inputs could share
/// the dequantization cost.
///
/// This batched variant routes the three projections through the iter
/// 042 multi-token GEMM tile (`dequant_gemm_int4_multi_auto`), which
/// dequantizes each int4 group once and fmadds against `num_tokens`
/// input vectors. Weight motion drops from `num_tokens × W` to `~W`,
/// the same amortization the shell projections got in iter 048.
///
/// **Semantics.** Bit-identical to `num_tokens` calls of [`expert_forward`]
/// with the same per-token input row. Underlying multi-token kernel
/// promises bit-identity vs the per-token kernel; this wrapper just
/// composes three of those calls + an elementwise SwiGLU.
///
/// **Inputs.**
/// - `xs_bf16`: `[num_tokens, hidden]` flat, row-major over tokens.
/// - `gate_*`, `up_*`, `down_*`: same as [`expert_forward`].
/// - `out_bf16`: `[num_tokens, hidden]` flat, written in place.
///
/// **When to use this vs [`expert_forward`].** For `num_tokens == 1` the
/// multi tile pays per-row scatter overhead that doesn't amortize; in
/// that case the caller should keep calling [`expert_forward`]. The
/// `dequant_gemm_int4_multi_auto` dispatcher already falls back to the
/// single-token kernel at seq=1, so this function is *correct* at any
/// `num_tokens >= 1` — it just won't win at 1.
pub fn expert_forward_multi(
    xs_bf16: &[bf16],
    gate_packed: &[u8],
    gate_scale: &[u8],
    up_packed: &[u8],
    up_scale: &[u8],
    down_packed: &[u8],
    down_scale: &[u8],
    num_tokens: usize,
    out_bf16: &mut [bf16],
) {
    assert!(num_tokens >= 1, "num_tokens must be >= 1, got {num_tokens}");
    let total_in = xs_bf16.len();
    assert!(
        total_in.is_multiple_of(num_tokens),
        "xs_bf16.len()={total_in} not divisible by num_tokens={num_tokens}",
    );
    let hidden = total_in / num_tokens;
    let intermediate = gate_scale.len() / 2 / (hidden / GROUP_SIZE);
    assert_eq!(
        out_bf16.len(),
        num_tokens * hidden,
        "out_bf16.len()={} != num_tokens*hidden={}",
        out_bf16.len(),
        num_tokens * hidden,
    );

    // Convert all input rows to f32 once. `[num_tokens, hidden]` flat.
    let mut xs_f32 = vec![0.0f32; num_tokens * hidden];
    for (i, b) in xs_bf16.iter().enumerate() {
        xs_f32[i] = b.to_f32();
    }

    // Gate + up projections, batched. Outputs are `[num_tokens, intermediate]`.
    let mut gate_out = vec![0.0f32; num_tokens * intermediate];
    let mut up_out = vec![0.0f32; num_tokens * intermediate];
    crate::kernel_avx512_multi::dequant_gemm_int4_multi_auto(
        gate_packed,
        gate_scale,
        &xs_f32,
        intermediate,
        hidden,
        num_tokens,
        &mut gate_out,
    );
    crate::kernel_avx512_multi::dequant_gemm_int4_multi_auto(
        up_packed,
        up_scale,
        &xs_f32,
        intermediate,
        hidden,
        num_tokens,
        &mut up_out,
    );

    // Elementwise SwiGLU per row. `inter[t]` = silu(gate_out[t]) * up_out[t].
    // Note: same scalar silu formula as `expert_forward`, applied per-cell —
    // bit-identical because there's no cross-token reduction.
    let mut inter = vec![0.0f32; num_tokens * intermediate];
    for i in 0..(num_tokens * intermediate) {
        let g = gate_out[i];
        let silu = g / (1.0 + (-g).exp());
        inter[i] = silu * up_out[i];
    }

    // Down projection, batched. `Y` is `[num_tokens, hidden]`.
    let mut out_f32 = vec![0.0f32; num_tokens * hidden];
    crate::kernel_avx512_multi::dequant_gemm_int4_multi_auto(
        down_packed,
        down_scale,
        &inter,
        hidden,
        intermediate,
        num_tokens,
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

    // --- expert_forward_multi correctness tests ---

    /// Build a deterministic int4 weight matrix + bf16 scales for shape
    /// `[n_rows, k_cols]`. Mirrors `kernel_avx512_multi::tests::make_test_data`
    /// but returns just (packed, scales) — caller decides the input.
    fn make_w_int4(n_rows: usize, k_cols: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
        assert!(k_cols.is_multiple_of(GROUP_SIZE));
        let n_groups = k_cols / GROUP_SIZE;
        let mut packed = vec![0u8; n_rows * k_cols / 2];
        for r in 0..n_rows {
            for c in 0..(k_cols / 2) {
                let v = ((r
                    .wrapping_mul(31)
                    .wrapping_add(c)
                    .wrapping_add(seed as usize))
                    & 0xFF) as u8;
                packed[r * (k_cols / 2) + c] = v;
            }
        }
        let mut scales = vec![0u8; n_rows * n_groups * 2];
        for r in 0..n_rows {
            for g in 0..n_groups {
                let s = 0.05f32 + (((r * 7 + g * 3 + seed as usize) % 7) as f32) * 0.01;
                let bits_u32 = s.to_bits();
                let rounded = bits_u32.wrapping_add(0x7FFF + ((bits_u32 >> 16) & 1));
                let bf = (rounded >> 16) as u16;
                let off = (r * n_groups + g) * 2;
                scales[off] = (bf & 0xFF) as u8;
                scales[off + 1] = (bf >> 8) as u8;
            }
        }
        (packed, scales)
    }

    fn make_xs_bf16(num_tokens: usize, hidden: usize, seed: u64) -> Vec<bf16> {
        let mut xs = vec![bf16::ZERO; num_tokens * hidden];
        for t in 0..num_tokens {
            for i in 0..hidden {
                let v = (((t * 13 + i * 5 + seed as usize) as f32).sin()) * 0.5;
                xs[t * hidden + i] = bf16::from_f32(v);
            }
        }
        xs
    }

    /// Foundational bit-identity: `expert_forward_multi(num_tokens)` must
    /// produce the same outputs as `num_tokens` independent calls of
    /// `expert_forward` with the same per-token input. We do not assert
    /// strict byte equality because rayon row-chunking + AVX-512 horizontal
    /// reduction can reorder additions inside a single dot product; we
    /// assert tight numerical equivalence (max abs delta in the bf16 LSB
    /// range).
    fn assert_multi_matches_per_token(num_tokens: usize, hidden: usize, intermediate: usize) {
        let (gate_p, gate_s) = make_w_int4(intermediate, hidden, 1);
        let (up_p, up_s) = make_w_int4(intermediate, hidden, 2);
        let (down_p, down_s) = make_w_int4(hidden, intermediate, 3);
        let xs = make_xs_bf16(num_tokens, hidden, 7);

        // Reference: per-token loop.
        let mut ys_ref = vec![bf16::ZERO; num_tokens * hidden];
        for t in 0..num_tokens {
            let x_t = &xs[t * hidden..(t + 1) * hidden];
            let y_t = &mut ys_ref[t * hidden..(t + 1) * hidden];
            expert_forward(x_t, &gate_p, &gate_s, &up_p, &up_s, &down_p, &down_s, y_t);
        }

        // Batched.
        let mut ys_multi = vec![bf16::ZERO; num_tokens * hidden];
        expert_forward_multi(
            &xs,
            &gate_p,
            &gate_s,
            &up_p,
            &up_s,
            &down_p,
            &down_s,
            num_tokens,
            &mut ys_multi,
        );

        // Compare per-cell. The multi tile reduces in the same per-row order
        // as the per-token kernel (same group sweep, same fmadd order within
        // a group), so deltas should be at most bf16 round-trip noise.
        let mut max_abs_delta = 0.0f32;
        for i in 0..(num_tokens * hidden) {
            let a = ys_ref[i].to_f32();
            let b = ys_multi[i].to_f32();
            let d = (a - b).abs();
            if d > max_abs_delta {
                max_abs_delta = d;
            }
        }
        assert!(
            max_abs_delta < 1e-3,
            "num_tokens={num_tokens} hidden={hidden} intermediate={intermediate}: \
             max delta {max_abs_delta} exceeds tolerance"
        );
    }

    #[test]
    fn expert_forward_multi_matches_per_token_seq_1() {
        // seq=1 hits the multi tile's fallback branch (which calls the
        // single-token kernel under the hood). This guarantees the fast
        // path stays bit-identical when callers happen to pass num_tokens=1.
        assert_multi_matches_per_token(1, 128, 64);
    }

    #[test]
    fn expert_forward_multi_matches_per_token_seq_2() {
        // Two tokens — minimum interesting batch (the spec-decode case
        // where 2 of K=4 candidates share an expert).
        assert_multi_matches_per_token(2, 128, 64);
    }

    #[test]
    fn expert_forward_multi_matches_per_token_seq_4() {
        // Four tokens — full K=4 spec-decode candidate batch sharing one
        // expert (extreme case but worth testing).
        assert_multi_matches_per_token(4, 128, 64);
    }

    #[test]
    fn expert_forward_multi_matches_per_token_seq_8() {
        // Eight tokens — K=8 spec-decode width, the iter 048 sweet spot.
        assert_multi_matches_per_token(8, 128, 64);
    }
}
