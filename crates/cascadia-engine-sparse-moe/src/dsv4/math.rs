//! dsv4 numeric primitives: bf16 rounding, RMSNorm, Walsh-Hadamard, and the
//! QAT quant-dequant simulations (block FP8 e4m3 / FP4 e2m1 with power-of-2
//! scales) that DeepSeek-V4 applies at inference time.
//!
//! Bit-exactness contract (vs `tools/deepseek_v4_ref/kernels_ref.py`):
//! - `pow2_round_up` ports upstream `fast_round_scale`'s IEEE-754 bit trick
//!   verbatim, so scales are exact.
//! - `round_e4m3` / `round_e2m1` land exactly on the FP8/FP4 grids (RNE,
//!   ties-to-even-mantissa), so the quant sims are bit-exact.
//! - everything accumulates in f32 and rounds to bf16 at the same points the
//!   bf16 reference does.

use half::bf16;

/// Round an f32 to the nearest bf16 value, returned as f32 (RNE).
#[inline]
pub fn to_bf16(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

/// In-place bf16 rounding of a whole buffer.
pub fn round_bf16(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = to_bf16(*v);
    }
}

/// 2^ceil(log2(x)) for positive normal x — upstream `fast_round_scale` /
/// `fast_log2_ceil` bit tricks, ported exactly.
#[inline]
pub fn pow2_round_up(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x007f_ffff;
    let e = exp - 127 + if man != 0 { 1 } else { 0 };
    f32::from_bits((((e + 127) as u32) & 0xff) << 23)
}

/// Round-to-nearest-even onto the FP8 e4m3fn grid. Caller pre-clamps to
/// [-448, 448] (the quant sims always do).
#[inline]
pub fn round_e4m3(v: f32) -> f32 {
    if v == 0.0 {
        return 0.0;
    }
    let a = v.abs();
    let sign = if v < 0.0 { -1.0 } else { 1.0 };
    const MIN_NORMAL: f32 = 0.015625; // 2^-6
    if a < MIN_NORMAL {
        // subnormal grid: multiples of 2^-9
        let q = (a * 512.0).round_ties_even();
        return sign * q * (1.0 / 512.0);
    }
    // normal: RNE of the f32 mantissa down to 3 bits
    let bits = a.to_bits();
    const SHIFT: u32 = 23 - 3;
    let lsb = (bits >> SHIFT) & 1;
    let rounded = bits + ((1u32 << (SHIFT - 1)) - 1) + lsb;
    sign * f32::from_bits(rounded & !((1u32 << SHIFT) - 1))
}

/// e2m1 magnitude grid and RNE with ties-to-even-mantissa (grid entries with
/// even mantissa: 0, 1, 2, 4 at indices 0, 2, 4, 6).
const E2M1_GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const E2M1_MID: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
const E2M1_TIE_IDX: [usize; 7] = [0, 2, 2, 4, 4, 6, 6];

/// Round-to-nearest-even onto the FP4 e2m1 grid. Caller pre-clamps to ±6.
#[inline]
pub fn round_e2m1(v: f32) -> f32 {
    let a = v.abs();
    let sign = if v < 0.0 { -1.0 } else { 1.0 };
    let mut idx = E2M1_MID.len(); // a >= all midpoints -> top bucket
    for (k, &m) in E2M1_MID.iter().enumerate() {
        if a < m {
            idx = k;
            break;
        }
        if a == m {
            return sign * E2M1_GRID[E2M1_TIE_IDX[k]];
        }
    }
    sign * E2M1_GRID[idx]
}

/// Block-wise FP8 quant-dequant sim (upstream `act_quant(..., inplace=True)`
/// with power-of-2 scales): per `block`-sized group along the last dim,
/// s = 2^ceil(log2(max(amax, 1e-4) / 448)), x <- bf16(e4m3(clamp(x/s)) * s).
pub fn act_quant_sim(x: &mut [f32], block: usize) {
    assert_eq!(x.len() % block, 0, "act_quant_sim: len % block != 0");
    for grp in x.chunks_mut(block) {
        let amax = grp.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-4);
        let s = pow2_round_up(amax * (1.0 / 448.0));
        for v in grp.iter_mut() {
            let q = round_e4m3((*v / s).clamp(-448.0, 448.0));
            *v = to_bf16(q * s);
        }
    }
}

/// Block-wise FP4 quant-dequant sim (upstream `fp4_act_quant(...,
/// inplace=True)`): s = 2^ceil(log2(max(amax, 6*2^-126) / 6)),
/// x <- bf16(e2m1(clamp(x/s, ±6)) * s).
pub fn fp4_act_quant_sim(x: &mut [f32], block: usize) {
    assert_eq!(x.len() % block, 0, "fp4_act_quant_sim: len % block != 0");
    const AMAX_FLOOR: f32 = 6.0 * f32::MIN_POSITIVE * 0.5; // 6 * 2^-126 (2^-126 = MIN_POSITIVE)
    for grp in x.chunks_mut(block) {
        let amax = grp
            .iter()
            .fold(0.0f32, |m, &v| m.max(v.abs()))
            .max(6.0 * 2.0f32.powi(-126))
            .max(AMAX_FLOOR);
        let s = pow2_round_up(amax * (1.0 / 6.0));
        for v in grp.iter_mut() {
            let q = round_e2m1((*v / s).clamp(-6.0, 6.0));
            *v = to_bf16(q * s);
        }
    }
}

/// RMSNorm matching the reference: y = bf16(w * (x / sqrt(mean(x^2) + eps))),
/// all internal math f32, one bf16 rounding at the end. `x` is [rows, dim].
pub fn rmsnorm(x: &mut [f32], w: &[f32], eps: f32) {
    let dim = w.len();
    assert_eq!(x.len() % dim, 0, "rmsnorm: len % dim != 0");
    for row in x.chunks_mut(dim) {
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let r = 1.0 / (ms + eps).sqrt();
        for (v, &wi) in row.iter_mut().zip(w) {
            *v = to_bf16(*v * r * wi);
        }
    }
}

/// Per-head "plain" RMS normalisation used on q (`q *= rsqrt(mean(q^2)+eps)`)
/// — no weight, output bf16-rounded. `x` is [rows, dim].
pub fn rms_normalize(x: &mut [f32], dim: usize, eps: f32) {
    assert_eq!(x.len() % dim, 0);
    for row in x.chunks_mut(dim) {
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let r = 1.0 / (ms + eps).sqrt();
        for v in row.iter_mut() {
            *v = to_bf16(*v * r);
        }
    }
}

/// Walsh-Hadamard transform along the last dim (power of 2), scaled by
/// `scale`, f32 butterflies, output bf16-rounded (reference dtype behaviour).
pub fn hadamard(x: &mut [f32], dim: usize, scale: f32) {
    assert!(dim.is_power_of_two(), "hadamard: dim not a power of 2");
    assert_eq!(x.len() % dim, 0);
    for row in x.chunks_mut(dim) {
        let mut h = 1;
        while h < dim {
            let mut base = 0;
            while base < dim {
                for i in 0..h {
                    let a = row[base + i];
                    let b = row[base + h + i];
                    row[base + i] = a + b;
                    row[base + h + i] = a - b;
                }
                base += 2 * h;
            }
            h *= 2;
        }
        for v in row.iter_mut() {
            *v = to_bf16(*v * scale);
        }
    }
}

/// f32 GEMV: y[o] = sum_k w[o*k_dim + k] * x[k]; output bf16-rounded
/// (the reference's F.linear on bf16 rounds its output to bf16).
///
/// Output rows are independent, so we split them across cores with rayon.
/// The per-row accumulation order is unchanged, so the result is BIT-IDENTICAL
/// to the sequential version — this is pure throughput, no numerics change.
/// The MoE experts + every attention projection funnel through here, so this
/// is where the CPU time is; a single scalar thread was using 1/8 of the box.
pub fn linear_bf16(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize, y: &mut [f32]) {
    use rayon::prelude::*;
    assert_eq!(x.len(), in_dim);
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(y.len(), out_dim);
    y.par_iter_mut().enumerate().for_each(|(o, yy)| {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let mut acc = 0.0f32;
        for k in 0..in_dim {
            acc += row[k] * x[k];
        }
        *yy = to_bf16(acc);
    });
}

/// Same as [`linear_bf16`] but keeps the output in f32 (for the fp32 paths:
/// gate scores, compressor wkv/wgate). Bit-identical rayon parallelisation.
pub fn linear_f32(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize, y: &mut [f32]) {
    use rayon::prelude::*;
    assert_eq!(x.len(), in_dim);
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(y.len(), out_dim);
    y.par_iter_mut().enumerate().for_each(|(o, yy)| {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let mut acc = 0.0f32;
        for k in 0..in_dim {
            acc += row[k] * x[k];
        }
        *yy = acc;
    });
}
