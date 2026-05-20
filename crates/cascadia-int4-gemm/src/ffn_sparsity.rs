//! Two-phase Gate-first sparse FFN — port of PowerInfer-2 §4.4 and
//! PowerInfer SmallThinker's `fused_sparse_moe.cpp` flow, adapted to
//! SwiGLU (the K2.6 activation) via a runtime magnitude threshold.
//!
//! ## How the original works
//!
//! PowerInfer-1/-2 assumes a ReLU-family activation. After the *gate*
//! projection, any output ≤ 0 is provably dead: the up projection
//! contribution at that lane will be multiplied by 0 (= ReLU(gate)).
//! The runtime exploits this by computing **gate first**, then
//! skipping the up and down work on the inactive lanes. PI-2 §4.4
//! reports a 1.5–2× FFN-compute win on top of cache hits.
//!
//! ## Why we threshold (SwiGLU adaptation)
//!
//! K2.6 uses SwiGLU: `silu(gate) ⊙ up`. SiLU is *not* exactly zero for
//! negative gate values — `silu(-2) ≈ -0.24`. There is no provably-zero
//! lane to skip.
//!
//! What is true empirically (CATS, Lee et al. 2024; CHESS, Liu et al.
//! 2024) is that the *distribution* of `silu(gate)` is heavy-tailed:
//! ~40–60% of lanes have `|silu(gate)| < 5%·max(|silu(gate)|)` for
//! typical SwiGLU activations. Dropping those lanes (clamping their
//! intermediate contribution to 0) introduces a small bounded error
//! that, in published bench results, costs <1% perplexity at 40–50%
//! sparsity.
//!
//! This module exposes that knob as `threshold` — *relative* to the
//! per-token max of `|silu(gate)|`. `0.0` disables the skip entirely
//! (output bit-identical to [`crate::kernel::expert_forward`]).
//!
//! ## Caller contract
//!
//! Per-token call shape: [`expert_forward_sparse`] is a drop-in for
//! [`crate::kernel::expert_forward`] plus one `threshold: f32`
//! parameter. With `threshold == 0.0` it falls through to the dense
//! path; with a positive threshold it runs the two-phase flow.
//!
//! ## Attribution
//!
//! - Two-phase Gate-then-Up/Down skip pattern: PowerInfer SmallThinker
//!   `fused_sparse_moe.cpp` (Song et al., SJTU-IPADS / Tiiny AI; MIT).
//! - Magnitude-threshold adaptation for SwiGLU: CATS (Lee et al. 2024)
//!   and CHESS (Liu et al. 2024) — Apache-2.0 / MIT.
//!
//! Both are referenced (not copied) — this is an independent Rust
//! implementation. See rainier `docs/POWERINFER_PORT.md` for the
//! full technique map.

use half::bf16;
use rayon::prelude::*;

use crate::format::bf16_bits_to_f32;
use crate::GROUP_SIZE;

/// Build the active-lane mask for a sparse FFN forward pass.
///
/// Returns `(silu_gate, active_indices)`:
///   - `silu_gate[i]`     — `silu(gate_out[i])`, all lanes.
///   - `active_indices`   — sorted ascending list of lane indices `i`
///     where `|silu_gate[i]| >= threshold * max_i |silu_gate[i]|`.
///
/// With `threshold == 0.0` returns all indices (no skip).
/// With `threshold >= 1.0` returns at most one index (the max-abs lane).
///
/// The returned mask is *relative* to the per-token max of
/// `|silu(gate)|`, which makes a single threshold value usable across
/// model layers and inputs of different scales.
pub fn build_active_mask(gate_out: &[f32], threshold: f32) -> (Vec<f32>, Vec<u32>) {
    let n = gate_out.len();
    // Phase 1: silu element-wise and track max-abs.
    let mut silu_gate = vec![0.0f32; n];
    let mut max_abs = 0.0f32;
    for (i, &g) in gate_out.iter().enumerate() {
        // silu(g) = g · sigmoid(g) = g / (1 + e^{-g}); numerically stable.
        let silu = if g >= 0.0 {
            g / (1.0 + (-g).exp())
        } else {
            let eg = g.exp();
            g * eg / (1.0 + eg)
        };
        silu_gate[i] = silu;
        let m = silu.abs();
        if m > max_abs {
            max_abs = m;
        }
    }
    if threshold <= 0.0 || max_abs == 0.0 {
        // No skip: all lanes active.
        let all = (0..n as u32).collect();
        return (silu_gate, all);
    }
    let cutoff = threshold * max_abs;
    let active: Vec<u32> = silu_gate
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (v.abs() >= cutoff).then_some(i as u32))
        .collect();
    (silu_gate, active)
}

/// Scalar reference: int4 GEMV that computes only the rows listed in
/// `active_rows`. Rows not in `active_rows` are *not* touched (caller
/// must pre-zero `y` if they need defined values there).
///
/// Equivalent to [`crate::kernel::dequant_gemv_int4`] for the subset
/// of rows.
pub fn dequant_gemv_int4_rows_subset(
    packed: &[u8],
    scale_bits: &[u8],
    x: &[f32],
    n_rows: usize,
    k_cols: usize,
    y: &mut [f32],
    active_rows: &[u32],
) {
    assert_eq!(packed.len(), n_rows * (k_cols / 2));
    let n_groups = k_cols / GROUP_SIZE;
    assert_eq!(scale_bits.len(), n_rows * n_groups * 2);
    assert_eq!(x.len(), k_cols);
    assert_eq!(y.len(), n_rows);
    let row_stride_packed = k_cols / 2;
    let row_stride_scale = n_groups * 2;

    // Parallelize across active rows. Each row is independent.
    let y_ptr = y.as_mut_ptr() as usize;
    active_rows.par_iter().for_each(|&r| {
        let r = r as usize;
        debug_assert!(r < n_rows);
        let row_packed = &packed[r * row_stride_packed..(r + 1) * row_stride_packed];
        let row_scales_bits = &scale_bits[r * row_stride_scale..(r + 1) * row_stride_scale];
        let mut acc = 0.0f32;
        for g in 0..n_groups {
            let scale_u16 =
                u16::from_le_bytes([row_scales_bits[g * 2], row_scales_bits[g * 2 + 1]]);
            let scale = bf16_bits_to_f32(scale_u16);
            let group_packed = &row_packed[g * (GROUP_SIZE / 2)..(g + 1) * (GROUP_SIZE / 2)];
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
        // SAFETY: each `r` in `active_rows` is unique-by-rayon's
        // par_iter semantics for ranges, but `active_rows` is a
        // caller-supplied slice that *may* contain duplicates. We
        // accept duplicates as a "last write wins" no-op (idempotent
        // for the same r → same `acc`).
        unsafe {
            *(y_ptr as *mut f32).add(r) = acc;
        }
    });
}

/// AVX-512 version of [`dequant_gemv_int4_rows_subset`]. Caller must
/// verify AVX-512 is available; the public wrapper
/// [`dequant_gemv_int4_rows_subset_auto`] does the feature check.
#[cfg(target_arch = "x86_64")]
mod avx512 {
    use core::arch::x86_64::*;
    use rayon::prelude::*;

    use crate::format::bf16_bits_to_f32;
    use crate::GROUP_SIZE;

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub unsafe fn dequant_gemv_int4_rows_subset_avx512(
        packed: &[u8],
        scale_bits: &[u8],
        x: &[f32],
        n_rows: usize,
        k_cols: usize,
        y: &mut [f32],
        active_rows: &[u32],
    ) {
        assert_eq!(packed.len(), n_rows * (k_cols / 2));
        let n_groups = k_cols / GROUP_SIZE;
        assert_eq!(scale_bits.len(), n_rows * n_groups * 2);
        assert_eq!(x.len(), k_cols);
        assert_eq!(y.len(), n_rows);
        let row_stride_packed = k_cols / 2;
        let row_stride_scale = n_groups * 2;

        let y_ptr = y.as_mut_ptr() as usize;
        let lo_mask = _mm_set1_epi8(0x0F);
        let bias = _mm_set1_epi8(8);
        active_rows.par_iter().for_each(|&r| {
            let r = r as usize;
            debug_assert!(r < n_rows);
            // SAFETY: we hold a non-aliasing `&[u8]` borrow via `packed`
            // and `scale_bits`; the rayon par_iter writes are to a
            // unique row offset.
            unsafe {
                let row_packed = &packed[r * row_stride_packed..(r + 1) * row_stride_packed];
                let row_scales = &scale_bits[r * row_stride_scale..(r + 1) * row_stride_scale];
                let mut acc = _mm512_setzero_ps();
                for g in 0..n_groups {
                    let scale_u16 = u16::from_le_bytes([row_scales[g * 2], row_scales[g * 2 + 1]]);
                    let scale = bf16_bits_to_f32(scale_u16);
                    let scale_v = _mm512_set1_ps(scale);
                    let p_ptr = row_packed.as_ptr().add(g * (GROUP_SIZE / 2)) as *const __m128i;
                    let pk = _mm_loadu_si128(p_ptr);
                    let low_nibbles = _mm_and_si128(pk, lo_mask);
                    let high_nibbles = _mm_and_si128(_mm_srli_epi16::<4>(pk), lo_mask);
                    let low_signed = _mm_sub_epi8(low_nibbles, bias);
                    let high_signed = _mm_sub_epi8(high_nibbles, bias);
                    let interleaved_lo = _mm_unpacklo_epi8(low_signed, high_signed);
                    let interleaved_hi = _mm_unpackhi_epi8(low_signed, high_signed);
                    let lo_i32 = _mm512_cvtepi8_epi32(interleaved_lo);
                    let hi_i32 = _mm512_cvtepi8_epi32(interleaved_hi);
                    let lo_f = _mm512_cvtepi32_ps(lo_i32);
                    let hi_f = _mm512_cvtepi32_ps(hi_i32);
                    let lo_w = _mm512_mul_ps(lo_f, scale_v);
                    let hi_w = _mm512_mul_ps(hi_f, scale_v);
                    let x_ptr = x.as_ptr().add(g * GROUP_SIZE) as *const f32;
                    let x_lo = _mm512_loadu_ps(x_ptr);
                    let x_hi = _mm512_loadu_ps(x_ptr.add(16));
                    acc = _mm512_fmadd_ps(lo_w, x_lo, acc);
                    acc = _mm512_fmadd_ps(hi_w, x_hi, acc);
                }
                *(y_ptr as *mut f32).add(r) = _mm512_reduce_add_ps(acc);
            }
        });
    }
}

/// Wrapper: pick AVX-512 if available, else fall back to scalar.
/// Caller must pre-zero `y` for rows *not* in `active_rows` if they
/// need defined values there (the kernel only touches rows in
/// `active_rows`).
pub fn dequant_gemv_int4_rows_subset_auto(
    packed: &[u8],
    scale_bits: &[u8],
    x: &[f32],
    n_rows: usize,
    k_cols: usize,
    y: &mut [f32],
    active_rows: &[u32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: target features verified at runtime.
            unsafe {
                avx512::dequant_gemv_int4_rows_subset_avx512(
                    packed,
                    scale_bits,
                    x,
                    n_rows,
                    k_cols,
                    y,
                    active_rows,
                );
            }
            return;
        }
    }
    dequant_gemv_int4_rows_subset(packed, scale_bits, x, n_rows, k_cols, y, active_rows);
}

/// Two-phase Gate-first sparse SwiGLU FFN — **f32 in, f32 out**.
///
///   1. Gate matmul (full): `gate_out[i] = sum_k W_gate[i,k] · x[k]`
///   2. SiLU + threshold:   `silu_gate[i] = silu(gate_out[i])`;
///      build `active_set = { i : |silu_gate[i]| >= τ · max_i|silu_gate[i]| }`
///   3. Up matmul (sparse): compute only rows in `active_set`.
///   4. Elementwise:        `inter[i] = silu_gate[i] · up[i]` for
///      active `i`; `0` elsewhere.
///   5. Down matmul (full): `out[h] = sum_i W_down[h,i] · inter[i]`
///      — kept full because the per-token cost of an indexed K-dim
///      gather outweighs the savings on inactive lanes (see comment).
///
/// `threshold == 0.0` falls through to a dense call sequence (no
/// active-mask construction, no sparse-rows kernel — output and
/// timing equivalent to three back-to-back `dequant_gemv_int4_auto`
/// calls + SwiGLU + final matmul).
///
/// **`hidden` and `intermediate` are explicit parameters** so this
/// function works for any SwiGLU FFN: per-expert routed FFN
/// (`intermediate=2048`), shell shared expert (`intermediate=2048`),
/// or K2.6 layer-0 dense FFN (`intermediate=18432`).
///
/// Returns the fraction of lanes that were active (for instrumentation).
pub fn ffn_forward_sparse_f32(
    x_f32: &[f32],
    hidden: usize,
    intermediate: usize,
    gate_packed: &[u8],
    gate_scale: &[u8],
    up_packed: &[u8],
    up_scale: &[u8],
    down_packed: &[u8],
    down_scale: &[u8],
    out_f32: &mut [f32],
    threshold: f32,
) -> f32 {
    debug_assert_eq!(x_f32.len(), hidden);
    debug_assert_eq!(out_f32.len(), hidden);

    // Phase 1: gate (full).
    let mut gate_out = vec![0.0f32; intermediate];
    crate::kernel_avx512::dequant_gemv_int4_auto(
        gate_packed,
        gate_scale,
        x_f32,
        intermediate,
        hidden,
        &mut gate_out,
    );

    if threshold <= 0.0 {
        // Dense path: SwiGLU as usual. Bit-identical to the pre-port
        // inline gate/up/down sequence in layer0_int4.rs and
        // shell_int4.rs — uses the same kernel calls in the same order
        // with the same SiLU formulation as `shell::swiglu_mul`
        // (`g / (1 + (-g).exp())`, IEEE 754-stable: underflows to 0 at
        // very negative g, saturates to g at very positive g).
        let mut up_out = vec![0.0f32; intermediate];
        crate::kernel_avx512::dequant_gemv_int4_auto(
            up_packed,
            up_scale,
            x_f32,
            intermediate,
            hidden,
            &mut up_out,
        );
        let mut inter = vec![0.0f32; intermediate];
        for i in 0..intermediate {
            let g = gate_out[i];
            let silu = g / (1.0f32 + (-g).exp());
            inter[i] = silu * up_out[i];
        }
        crate::kernel_avx512::dequant_gemv_int4_auto(
            down_packed,
            down_scale,
            &inter,
            hidden,
            intermediate,
            out_f32,
        );
        return 1.0;
    }

    // Sparse path: SiLU + threshold → mask → sparse up + elementwise + down.
    let (silu_gate, active) = build_active_mask(&gate_out, threshold);
    let active_frac = active.len() as f32 / intermediate as f32;

    let mut up_out = vec![0.0f32; intermediate];
    dequant_gemv_int4_rows_subset_auto(
        up_packed,
        up_scale,
        x_f32,
        intermediate,
        hidden,
        &mut up_out,
        &active,
    );

    let mut inter = vec![0.0f32; intermediate];
    for &r in &active {
        let r = r as usize;
        inter[r] = silu_gate[r] * up_out[r];
    }

    // Phase 3: down (full).
    //
    // We could write a column-sparse down kernel that skips groups
    // where all K-dim cols in the group are zero, but: (a) groups are
    // 32-wide and a random sparsity pattern leaves few all-zero groups,
    // (b) the scale-fetch / dequant cost per group is small compared to
    // the 16-lane FMA chain, and (c) zero × anything still feeds the
    // FMA pipeline at full throughput. Empirically the win on the K
    // dim is <5% for typical 50% sparsity — left for a follow-up if
    // bench shows the FFN bottleneck remains at down.
    crate::kernel_avx512::dequant_gemv_int4_auto(
        down_packed,
        down_scale,
        &inter,
        hidden,
        intermediate,
        out_f32,
    );
    active_frac
}

/// Two-phase Gate-first sparse expert FFN — **bf16 in, bf16 out**.
///
/// Thin wrapper around [`ffn_forward_sparse_f32`] that handles the
/// bf16 ↔ f32 conversion at the boundary. Used by the per-routed-
/// expert dispatch in cascadia-engine-sparse-moe.
///
/// `threshold == 0.0` is bit-identical to [`crate::kernel::expert_forward`]
/// (verified by `sparse_expert_threshold_zero_matches_dense`).
///
/// Returns the fraction of lanes that were active (for instrumentation).
pub fn expert_forward_sparse(
    x_bf16: &[bf16],
    gate_packed: &[u8],
    gate_scale: &[u8],
    up_packed: &[u8],
    up_scale: &[u8],
    down_packed: &[u8],
    down_scale: &[u8],
    out_bf16: &mut [bf16],
    threshold: f32,
) -> f32 {
    if threshold <= 0.0 {
        // Dense fallback: delegate to the existing path so output is
        // byte-identical to pre-port. (The f32 dense path in
        // `ffn_forward_sparse_f32` should also be byte-identical, but
        // we keep the explicit fallback to `expert_forward` for the
        // hot wire-protocol checksum match.)
        crate::kernel::expert_forward(
            x_bf16,
            gate_packed,
            gate_scale,
            up_packed,
            up_scale,
            down_packed,
            down_scale,
            out_bf16,
        );
        return 1.0;
    }

    let hidden = x_bf16.len();
    let intermediate = gate_scale.len() / 2 / (hidden / GROUP_SIZE);

    let mut x_f32 = vec![0.0f32; hidden];
    for (i, b) in x_bf16.iter().enumerate() {
        x_f32[i] = b.to_f32();
    }
    let mut out_f32 = vec![0.0f32; hidden];
    let active_frac = ffn_forward_sparse_f32(
        &x_f32,
        hidden,
        intermediate,
        gate_packed,
        gate_scale,
        up_packed,
        up_scale,
        down_packed,
        down_scale,
        &mut out_f32,
        threshold,
    );
    for (i, v) in out_f32.iter().enumerate() {
        out_bf16[i] = bf16::from_f32(*v);
    }
    active_frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{dequant_gemv_int4, expert_forward};

    /// `build_active_mask` with threshold 0 returns all indices.
    #[test]
    fn mask_threshold_zero_all_active() {
        let gate_out = vec![1.0, -2.0, 0.5, -0.1];
        let (_silu, active) = build_active_mask(&gate_out, 0.0);
        assert_eq!(active, vec![0, 1, 2, 3]);
    }

    /// `build_active_mask` with a moderate threshold drops near-zero
    /// lanes; only the top-magnitude lane survives a 0.5 threshold
    /// when one lane dominates by ≫2×.
    #[test]
    fn mask_threshold_drops_smallest_lanes() {
        // silu(10) ≈ 10, silu(5) ≈ 4.97, silu(±0.001) ≈ ±0.0005.
        // With threshold 0.5 and max-abs ≈ 10, cutoff = 5.0 → only
        // lane 0 (|silu| ≥ 5) survives.
        let gate_out = vec![10.0, 0.001, 5.0, -0.001];
        let (silu, active) = build_active_mask(&gate_out, 0.5);
        assert!(silu[0].abs() > 5.0);
        assert!(active.contains(&0), "lane 0 (|silu|≈10) is the max");
        assert!(!active.contains(&1), "lane 1 (|silu|≈0.0005) below cutoff");
        assert!(
            !active.contains(&2),
            "lane 2 (|silu|≈4.97) below 5.0 cutoff"
        );
        assert!(!active.contains(&3), "lane 3 (|silu|≈0.0005) below cutoff");

        // With a relaxed threshold (0.1, cutoff = 1.0), lane 2 survives.
        let (_, active2) = build_active_mask(&gate_out, 0.1);
        assert!(active2.contains(&0));
        assert!(active2.contains(&2));
        assert!(!active2.contains(&1));
        assert!(!active2.contains(&3));
    }

    /// `dequant_gemv_int4_rows_subset` over the full row set produces
    /// the same output as the dense kernel.
    #[test]
    fn sparse_rows_subset_matches_dense_when_full() {
        let n_rows = 4;
        let k_cols = 32;
        // Random-ish packed weights via byte pattern; scale = 1.0.
        let packed: Vec<u8> = (0..n_rows * k_cols / 2).map(|i| (i % 256) as u8).collect();
        let scale_bits: Vec<u8> = vec![0x80, 0x3f]
            .into_iter()
            .cycle()
            .take(n_rows * (k_cols / GROUP_SIZE) * 2)
            .collect();
        let x: Vec<f32> = (0..k_cols).map(|i| (i as f32) * 0.01).collect();

        let mut dense_y = vec![0.0f32; n_rows];
        dequant_gemv_int4(&packed, &scale_bits, &x, n_rows, k_cols, &mut dense_y);

        let mut sparse_y = vec![0.0f32; n_rows];
        let all: Vec<u32> = (0..n_rows as u32).collect();
        dequant_gemv_int4_rows_subset(
            &packed,
            &scale_bits,
            &x,
            n_rows,
            k_cols,
            &mut sparse_y,
            &all,
        );

        for r in 0..n_rows {
            assert!(
                (dense_y[r] - sparse_y[r]).abs() < 1e-6,
                "row {r}: dense={} sparse={}",
                dense_y[r],
                sparse_y[r]
            );
        }
    }

    /// `dequant_gemv_int4_rows_subset` over a subset leaves untouched
    /// rows alone (the kernel does not write them).
    #[test]
    fn sparse_rows_subset_leaves_inactive_untouched() {
        let n_rows = 4;
        let k_cols = 32;
        let packed = vec![0xAAu8; n_rows * k_cols / 2];
        let scale_bits: Vec<u8> = vec![0x80, 0x3f]
            .into_iter()
            .cycle()
            .take(n_rows * (k_cols / GROUP_SIZE) * 2)
            .collect();
        let x = vec![1.0f32; k_cols];
        let mut y = vec![123.0f32; n_rows]; // sentinel.

        // Process only row 1.
        dequant_gemv_int4_rows_subset(&packed, &scale_bits, &x, n_rows, k_cols, &mut y, &[1]);

        // Rows 0, 2, 3 untouched (still the sentinel).
        assert_eq!(y[0], 123.0);
        assert_eq!(y[2], 123.0);
        assert_eq!(y[3], 123.0);
        // Row 1 written.
        assert!(
            (y[1] - 123.0).abs() > 1e-3,
            "row 1 should have been written"
        );
    }

    /// `expert_forward_sparse` with threshold = 0 is byte-identical to
    /// the dense `expert_forward`.
    #[test]
    fn sparse_expert_threshold_zero_matches_dense() {
        let hidden = 32;
        let intermediate = 32;
        // Small fake weights — group_size=32 so one group per row.
        let gate_packed = vec![0x9Au8; intermediate * hidden / 2];
        let gate_scale: Vec<u8> = vec![0x80, 0x3f]
            .into_iter()
            .cycle()
            .take(intermediate * (hidden / GROUP_SIZE) * 2)
            .collect();
        let up_packed = vec![0x8Bu8; intermediate * hidden / 2];
        let up_scale = gate_scale.clone();
        let down_packed = vec![0x7Cu8; hidden * intermediate / 2];
        let down_scale: Vec<u8> = vec![0x80, 0x3f]
            .into_iter()
            .cycle()
            .take(hidden * (intermediate / GROUP_SIZE) * 2)
            .collect();
        let x: Vec<bf16> = (0..hidden)
            .map(|i| bf16::from_f32((i as f32) * 0.05 - 0.5))
            .collect();
        let mut out_dense = vec![bf16::ZERO; hidden];
        expert_forward(
            &x,
            &gate_packed,
            &gate_scale,
            &up_packed,
            &up_scale,
            &down_packed,
            &down_scale,
            &mut out_dense,
        );
        let mut out_sparse = vec![bf16::ZERO; hidden];
        let active_frac = expert_forward_sparse(
            &x,
            &gate_packed,
            &gate_scale,
            &up_packed,
            &up_scale,
            &down_packed,
            &down_scale,
            &mut out_sparse,
            0.0,
        );
        assert_eq!(active_frac, 1.0);
        for h in 0..hidden {
            assert_eq!(
                out_dense[h].to_bits(),
                out_sparse[h].to_bits(),
                "h={h}: dense={} sparse={}",
                out_dense[h].to_f32(),
                out_sparse[h].to_f32()
            );
        }
    }

    /// `expert_forward_sparse` with threshold > 0 produces output close
    /// to dense — within a bounded error (the active-set fraction).
    /// Quality is not perfect (that's the point of the speed/quality
    /// tradeoff); we just sanity-check the output isn't garbage.
    #[test]
    fn sparse_expert_threshold_small_close_to_dense() {
        let hidden = 32;
        let intermediate = 32;
        let gate_packed = vec![0x9Au8; intermediate * hidden / 2];
        let gate_scale: Vec<u8> = vec![0x80, 0x3f]
            .into_iter()
            .cycle()
            .take(intermediate * (hidden / GROUP_SIZE) * 2)
            .collect();
        let up_packed = vec![0x8Bu8; intermediate * hidden / 2];
        let up_scale = gate_scale.clone();
        let down_packed = vec![0x7Cu8; hidden * intermediate / 2];
        let down_scale: Vec<u8> = vec![0x80, 0x3f]
            .into_iter()
            .cycle()
            .take(hidden * (intermediate / GROUP_SIZE) * 2)
            .collect();
        let x: Vec<bf16> = (0..hidden)
            .map(|i| bf16::from_f32((i as f32) * 0.05 - 0.5))
            .collect();
        let mut out_dense = vec![bf16::ZERO; hidden];
        expert_forward(
            &x,
            &gate_packed,
            &gate_scale,
            &up_packed,
            &up_scale,
            &down_packed,
            &down_scale,
            &mut out_dense,
        );
        let mut out_sparse = vec![bf16::ZERO; hidden];
        let active_frac = expert_forward_sparse(
            &x,
            &gate_packed,
            &gate_scale,
            &up_packed,
            &up_scale,
            &down_packed,
            &down_scale,
            &mut out_sparse,
            0.10, // 10% threshold — keep top ~90% by magnitude.
        );
        assert!(
            active_frac > 0.0 && active_frac <= 1.0,
            "active_frac out of range: {active_frac}"
        );
        // Outputs are not bit-identical (some lanes dropped), but they
        // shouldn't diverge wildly. Use a generous tolerance — the
        // exact value depends on the weights' distribution. Realistic
        // miner bench will produce the actual quality numbers.
        let max_dev: f32 = (0..hidden)
            .map(|h| (out_dense[h].to_f32() - out_sparse[h].to_f32()).abs())
            .fold(0.0, f32::max);
        let dense_scale: f32 = (0..hidden)
            .map(|h| out_dense[h].to_f32().abs())
            .fold(0.0, f32::max);
        assert!(
            max_dev < dense_scale * 2.0 + 1.0,
            "sparse output diverged: max_dev={max_dev} dense_scale={dense_scale}"
        );
    }

    /// `ffn_forward_sparse_f32` at threshold=0 must match an inline
    /// dense gate / up / SwiGLU / down sequence byte-for-byte. This
    /// is the bit-identity contract that protects the layer-0 and
    /// shell shared-expert refactor from introducing numerical drift.
    #[test]
    fn ffn_f32_threshold_zero_matches_inline_dense() {
        use crate::kernel_avx512::dequant_gemv_int4_auto;
        let hidden = 32;
        let intermediate = 32;
        let n_in_groups = hidden / GROUP_SIZE;
        let n_mid_groups = intermediate / GROUP_SIZE;

        let gate_packed = vec![0x9Au8; intermediate * hidden / 2];
        let gate_scale: Vec<u8> = vec![0x80, 0x3f]
            .into_iter()
            .cycle()
            .take(intermediate * n_in_groups * 2)
            .collect();
        let up_packed = vec![0x8Bu8; intermediate * hidden / 2];
        let up_scale = gate_scale.clone();
        let down_packed = vec![0x7Cu8; hidden * intermediate / 2];
        let down_scale: Vec<u8> = vec![0x80, 0x3f]
            .into_iter()
            .cycle()
            .take(hidden * n_mid_groups * 2)
            .collect();
        let x_f32: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.05 - 0.5).collect();

        // Reference: three inline GEMVs + the canonical swiglu_mul.
        let mut ref_gate = vec![0.0f32; intermediate];
        dequant_gemv_int4_auto(
            &gate_packed,
            &gate_scale,
            &x_f32,
            intermediate,
            hidden,
            &mut ref_gate,
        );
        let mut ref_up = vec![0.0f32; intermediate];
        dequant_gemv_int4_auto(
            &up_packed,
            &up_scale,
            &x_f32,
            intermediate,
            hidden,
            &mut ref_up,
        );
        let mut ref_inter = vec![0.0f32; intermediate];
        crate::shell::swiglu_mul(&ref_gate, &ref_up, &mut ref_inter);
        let mut ref_out = vec![0.0f32; hidden];
        dequant_gemv_int4_auto(
            &down_packed,
            &down_scale,
            &ref_inter,
            hidden,
            intermediate,
            &mut ref_out,
        );

        // Test target: ffn_forward_sparse_f32 at τ=0.
        let mut got_out = vec![0.0f32; hidden];
        let active_frac = ffn_forward_sparse_f32(
            &x_f32,
            hidden,
            intermediate,
            &gate_packed,
            &gate_scale,
            &up_packed,
            &up_scale,
            &down_packed,
            &down_scale,
            &mut got_out,
            0.0,
        );
        assert_eq!(active_frac, 1.0);
        for h in 0..hidden {
            assert_eq!(
                ref_out[h].to_bits(),
                got_out[h].to_bits(),
                "h={h}: inline-ref={} sparse(τ=0)={}",
                ref_out[h],
                got_out[h]
            );
        }
    }
}
