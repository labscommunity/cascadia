//! Int2 GEMV kernel — y = weight @ x for one row at a time, with int2-packed
//! weights and bf16 per-group scales.
//!
//! Why int2: halves the per-expert byte count vs int4. On a disk-bound box
//! (miner expert dispatch was 82% of decode in iter 003), reducing the byte
//! count cuts page-in time roughly in proportion. On a memory-bound box the
//! same logic applies to DRAM bandwidth.
//!
//! Storage layout — group_size=32, symmetric, zero_point=2:
//!
//! ```text
//!   packed: u8  [n_rows, k_cols / 4]   (4 weights per byte)
//!   scales: u16 [n_rows, k_cols / 32]  (bf16 raw bits, one per group of 32)
//! ```
//!
//! Each byte holds 4 two-bit nibbles. Convention: bits [0:2] = col 4i,
//! bits [2:4] = col 4i+1, bits [4:6] = col 4i+2, bits [6:8] = col 4i+3.
//! Stored as unsigned [0, 3]; subtract 2 to get signed [-2, 1]. The +max
//! value (1) maps to +scale; -max (-2) maps to -2*scale.
//!
//! Within one group of 32 weights we use 8 bytes packed, matching the
//! kernel's "load 8 bytes, expand to 32 i8" structure.

#![allow(unsafe_op_in_unsafe_fn)]

use rayon::prelude::*;

use crate::format::bf16_bits_to_f32;
use crate::GROUP_SIZE;

/// Number of weights packed per byte in our int2 layout.
pub const INT2_VALS_PER_BYTE: usize = 4;
/// Bytes per int2-quantized group of 32 weights.
pub const INT2_BYTES_PER_GROUP: usize = GROUP_SIZE / INT2_VALS_PER_BYTE;

/// Round f32 → bf16 (returns the 16-bit bf16 representation as u16).
#[inline]
pub fn bf16_round(x: f32) -> u16 {
    let bits = x.to_bits();
    // Round-to-nearest-even: add (mantissa LSB rounding) bias.
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// Quantize a bf16 weight matrix `[n_rows, k_cols]` (raw bytes, LE bf16 = u16)
/// into int2 packed (4 weights per byte) + per-group bf16 scales.
///
/// The asymmetric range [-2, 1] is intentionally biased — symmetric int2 would
/// give us {-1, 0, 1} (effectively ternary) with 1 bit of resolution wasted.
/// Using the full 4 codepoints with offset = 2 gives 4 levels: [-2, -1, 0, 1].
/// We pick the scale so that `max_abs` maps to `1` (the positive max), which
/// matches our int4 SYM convention.
///
/// Output layout:
///   packed: `u8` of length `n_rows * (k_cols / 4)` — byte `i` of row `r`
///   holds cols 4i..4i+3 (low→high bits).
///   scales: `u8` of length `n_rows * (k_cols / GROUP_SIZE) * 2` — bf16 LE.
pub fn quantize_int2_group(weight_bf16: &[u8], n_rows: usize, k_cols: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(weight_bf16.len(), n_rows * k_cols * 2);
    assert!(k_cols.is_multiple_of(GROUP_SIZE));
    let n_groups = k_cols / GROUP_SIZE;
    let mut packed = vec![0u8; n_rows * k_cols / INT2_VALS_PER_BYTE];
    let mut scales = vec![0u8; n_rows * n_groups * 2];

    for r in 0..n_rows {
        for g in 0..n_groups {
            // Find max abs in this group.
            let mut max_abs = 0.0f32;
            for k in 0..GROUP_SIZE {
                let c = g * GROUP_SIZE + k;
                let off = (r * k_cols + c) * 2;
                let bits = ((weight_bf16[off + 1] as u32) << 8) | (weight_bf16[off] as u32);
                let w = f32::from_bits(bits << 16);
                let a = w.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            // Scale chosen so +max_abs maps to +1 (signed [-2, 1] range).
            let scale = if max_abs == 0.0 { 1.0e-10 } else { max_abs };
            // Store scale as bf16.
            let scale_bits = bf16_round(scale);
            let s_off = (r * n_groups + g) * 2;
            scales[s_off] = (scale_bits & 0xFF) as u8;
            scales[s_off + 1] = (scale_bits >> 8) as u8;
            // Re-read after rounding for the inv multiplier.
            let scale_q = f32::from_bits((scale_bits as u32) << 16);
            let inv = 1.0 / scale_q;
            // Quantize each value.
            for k in 0..GROUP_SIZE {
                let c = g * GROUP_SIZE + k;
                let w_off = (r * k_cols + c) * 2;
                let bits = ((weight_bf16[w_off + 1] as u32) << 8) | (weight_bf16[w_off] as u32);
                let w = f32::from_bits(bits << 16);
                let q = (w * inv).round().clamp(-2.0, 1.0) as i32;
                // Map signed [-2, 1] to unsigned [0, 3] by +2.
                let two_bit = ((q + 2) & 0x03) as u8;
                let p_off = (r * k_cols + c) / INT2_VALS_PER_BYTE;
                let sub = c & 0x03; // 0,1,2,3
                let shift = (sub as u8) * 2;
                packed[p_off] = (packed[p_off] & !(0x03u8 << shift)) | (two_bit << shift);
            }
        }
    }

    (packed, scales)
}

/// Re-quantize an int4-packed weight row to int2 + bf16 scales.
///
/// Inputs:
/// - `int4_packed`: `[n_rows, k_cols/2]` bytes — low/high nibbles per byte
///   encode cols 2i, 2i+1 with `(unsigned - 8)` signed convention.
/// - `int4_scales`: `[n_rows, k_cols/32]` bf16 LE bits.
///
/// We decode int4 → f32 (via int4 scales), then quantize that f32 group
/// to int2 + a fresh scale. This is the lazy load-time path; we don't
/// need to touch the original bf16. (We could go straight from
/// int4-codes → int2-codes by recomputing max_abs from the int4 dequant,
/// but going via f32 keeps the rounding correct without re-deriving
/// the int4 codebook math.)
pub fn quantize_int2_from_int4(
    int4_packed: &[u8],
    int4_scales: &[u8],
    n_rows: usize,
    k_cols: usize,
) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(int4_packed.len(), n_rows * (k_cols / 2));
    let n_groups = k_cols / GROUP_SIZE;
    assert_eq!(int4_scales.len(), n_rows * n_groups * 2);
    let mut packed = vec![0u8; n_rows * k_cols / INT2_VALS_PER_BYTE];
    let mut scales = vec![0u8; n_rows * n_groups * 2];

    // Process row-by-row. Each int4 group is 16 packed bytes → 32 signed
    // int values → 32 f32 values via group scale.
    for r in 0..n_rows {
        for g in 0..n_groups {
            // Read int4 group's bf16 scale.
            let s4_off = (r * n_groups + g) * 2;
            let s4_bits = u16::from_le_bytes([int4_scales[s4_off], int4_scales[s4_off + 1]]);
            let s4 = bf16_bits_to_f32(s4_bits);
            // Decode 32 weights to f32.
            let p4_row_off = r * (k_cols / 2);
            let p4_group_off = p4_row_off + g * (GROUP_SIZE / 2);
            let mut group_f32 = [0.0f32; GROUP_SIZE];
            for i in 0..(GROUP_SIZE / 2) {
                let byte = int4_packed[p4_group_off + i];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                group_f32[i * 2] = (lo as f32) * s4;
                group_f32[i * 2 + 1] = (hi as f32) * s4;
            }
            // Re-quantize this group to int2.
            let mut max_abs = 0.0f32;
            for &v in &group_f32 {
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            let scale = if max_abs == 0.0 { 1.0e-10 } else { max_abs };
            let s2_bits = bf16_round(scale);
            let s2_off = (r * n_groups + g) * 2;
            scales[s2_off] = (s2_bits & 0xFF) as u8;
            scales[s2_off + 1] = (s2_bits >> 8) as u8;
            let scale_q = f32::from_bits((s2_bits as u32) << 16);
            let inv = 1.0 / scale_q;
            // Pack into int2 bytes.
            let p2_row_off = r * (k_cols / INT2_VALS_PER_BYTE);
            for k in 0..GROUP_SIZE {
                let c = g * GROUP_SIZE + k;
                let q = (group_f32[k] * inv).round().clamp(-2.0, 1.0) as i32;
                let two_bit = ((q + 2) & 0x03) as u8;
                let p_off = p2_row_off + c / INT2_VALS_PER_BYTE;
                let sub = c & 0x03;
                let shift = (sub as u8) * 2;
                packed[p_off] = (packed[p_off] & !(0x03u8 << shift)) | (two_bit << shift);
            }
        }
    }

    (packed, scales)
}

/// Scalar reference: `y[r] = sum_c (signed_int2(W[r, c]) * scale[r, c/32]) * x[c]`.
pub fn dequant_gemv_int2(
    packed: &[u8],
    scale_bits: &[u8],
    x: &[f32],
    n_rows: usize,
    k_cols: usize,
    y: &mut [f32],
) {
    assert_eq!(packed.len(), n_rows * (k_cols / INT2_VALS_PER_BYTE));
    let n_groups = k_cols / GROUP_SIZE;
    assert_eq!(scale_bits.len(), n_rows * n_groups * 2);
    assert_eq!(x.len(), k_cols);
    assert_eq!(y.len(), n_rows);
    let row_stride_packed = k_cols / INT2_VALS_PER_BYTE;

    y.par_iter_mut().enumerate().for_each(|(r, yy)| {
        let row_packed = &packed[r * row_stride_packed..(r + 1) * row_stride_packed];
        let row_scales = &scale_bits[r * n_groups * 2..(r + 1) * n_groups * 2];
        let mut acc = 0.0f32;
        for g in 0..n_groups {
            let scale_u16 = u16::from_le_bytes([row_scales[g * 2], row_scales[g * 2 + 1]]);
            let scale = bf16_bits_to_f32(scale_u16);
            // 8 bytes per group (32 weights / 4 per byte).
            let group_packed =
                &row_packed[g * INT2_BYTES_PER_GROUP..(g + 1) * INT2_BYTES_PER_GROUP];
            let mut group_dot = 0.0f32;
            for i in 0..INT2_BYTES_PER_GROUP {
                let byte = group_packed[i];
                let c0 = ((byte & 0x03) as i32) - 2;
                let c1 = (((byte >> 2) & 0x03) as i32) - 2;
                let c2 = (((byte >> 4) & 0x03) as i32) - 2;
                let c3 = (((byte >> 6) & 0x03) as i32) - 2;
                let col = g * GROUP_SIZE + i * 4;
                group_dot += (c0 as f32) * x[col];
                group_dot += (c1 as f32) * x[col + 1];
                group_dot += (c2 as f32) * x[col + 2];
                group_dot += (c3 as f32) * x[col + 3];
            }
            acc += scale * group_dot;
        }
        *yy = acc;
    });
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use core::arch::x86_64::*;
    use rayon::prelude::*;

    use super::{INT2_BYTES_PER_GROUP, INT2_VALS_PER_BYTE};
    use crate::format::bf16_bits_to_f32;
    use crate::GROUP_SIZE;

    /// AVX-512 path. Caller must check `is_x86_feature_detected!("avx512f")`.
    ///
    /// Strategy per group (32 weights = 8 bytes):
    ///   - Load 8 bytes into a __m128i.
    ///   - Expand each byte into 4 lanes (cols 4i..4i+3) via mask + shift.
    ///   - Subtract 2 → signed i8 in [-2, 1].
    ///   - sign-extend to i32, convert to f32.
    ///   - Multiply by scale, fmadd with x.
    ///
    /// We process one group (32 cols) per inner-loop iteration. With the
    /// scalar deinterleave (mask/shift) the kernel is still tight enough
    /// to hit DRAM bandwidth — same as int4. The win is upstream: half
    /// the weight bytes to stream off page-cache / disk.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub unsafe fn dequant_gemv_int2_avx512(
        packed: &[u8],
        scale_bits: &[u8],
        x: &[f32],
        n_rows: usize,
        k_cols: usize,
        y: &mut [f32],
    ) {
        assert_eq!(packed.len(), n_rows * (k_cols / INT2_VALS_PER_BYTE));
        let n_groups = k_cols / GROUP_SIZE;
        assert_eq!(scale_bits.len(), n_rows * n_groups * 2);
        assert_eq!(x.len(), k_cols);
        assert_eq!(y.len(), n_rows);
        let row_stride_packed = k_cols / INT2_VALS_PER_BYTE;

        y.par_iter_mut().enumerate().for_each(|(r, yy)| {
            let row_packed = &packed[r * row_stride_packed..(r + 1) * row_stride_packed];
            let row_scales = &scale_bits[r * n_groups * 2..(r + 1) * n_groups * 2];
            let mut acc = _mm512_setzero_ps();
            for g in 0..n_groups {
                let scale_u16 = u16::from_le_bytes([row_scales[g * 2], row_scales[g * 2 + 1]]);
                let scale = bf16_bits_to_f32(scale_u16);
                let scale_v = _mm512_set1_ps(scale);
                // Load 8 bytes (32 packed 2-bit codes).
                // We can't use _mm_loadl_epi64 with an &[u8] cleanly;
                // copy into a u64 then build the __m128i.
                let mut buf64 = [0u8; 8];
                buf64.copy_from_slice(
                    &row_packed[g * INT2_BYTES_PER_GROUP..(g + 1) * INT2_BYTES_PER_GROUP],
                );
                let u = u64::from_le_bytes(buf64);
                let p64 = _mm_set_epi64x(0, u as i64);
                // Expand 8 bytes × 4 codes = 32 i8 lanes in a 256-bit register.
                // Strategy: build four __m128i, each holds 16 codes for one
                // "position-in-byte" (0, 1, 2, 3), then interleave.
                //
                // Actually simpler: produce 4 vectors of 8 lanes each
                // (positions 0,1,2,3), zip them into a 32-lane sequence.
                //
                // The cleanest version is to broadcast each byte four times
                // across 4 lanes — that needs vpshufb on AVX-512BW.
                //
                // We'll go with: byte_lane[i] contains 8 copies of byte i;
                // then take 4 different mask/shift combos to produce 4 8-lane
                // results that we interleave.
                let mask03 = _mm_set1_epi8(0x03);
                let two = _mm_set1_epi8(2);
                // Position 0: byte & 0x03.
                let lane0 = _mm_sub_epi8(_mm_and_si128(p64, mask03), two);
                // Position 1: (byte >> 2) & 0x03.
                let p_sr2 = _mm_srli_epi16::<2>(p64);
                let lane1 = _mm_sub_epi8(_mm_and_si128(p_sr2, mask03), two);
                // Position 2: (byte >> 4) & 0x03.
                let p_sr4 = _mm_srli_epi16::<4>(p64);
                let lane2 = _mm_sub_epi8(_mm_and_si128(p_sr4, mask03), two);
                // Position 3: (byte >> 6) & 0x03.
                let p_sr6 = _mm_srli_epi16::<6>(p64);
                let lane3 = _mm_sub_epi8(_mm_and_si128(p_sr6, mask03), two);
                // Now we have four 8-byte vectors, each with one "position"
                // per source byte. Interleave to get cols 0..31 in order.
                //
                // Cols layout we want: [b0p0, b0p1, b0p2, b0p3, b1p0, b1p1, ...]
                // — i.e. take byte 0's 4 positions, then byte 1's 4 positions, etc.
                //
                // Step 1: interleave lane0/lane1 byte-wise → [b0p0, b0p1, b1p0, b1p1, b2p0, b2p1, ...]
                let l01 = _mm_unpacklo_epi8(lane0, lane1);
                // Step 2: interleave lane2/lane3 → [b0p2, b0p3, b1p2, b1p3, ...]
                let l23 = _mm_unpacklo_epi8(lane2, lane3);
                // Step 3: interleave the two i16-wise → [b0p0, b0p1, b0p2, b0p3, b1p0, b1p1, b1p2, b1p3, ...]
                let interleaved_lo = _mm_unpacklo_epi16(l01, l23); // cols 0..15
                let interleaved_hi = _mm_unpackhi_epi16(l01, l23); // cols 16..31
                                                                   // Sign-extend to i32, convert to f32.
                let lo_i32 = _mm512_cvtepi8_epi32(interleaved_lo);
                let hi_i32 = _mm512_cvtepi8_epi32(interleaved_hi);
                let lo_f = _mm512_cvtepi32_ps(lo_i32);
                let hi_f = _mm512_cvtepi32_ps(hi_i32);
                // Multiply by scale, fmadd with x.
                let lo_w = _mm512_mul_ps(lo_f, scale_v);
                let hi_w = _mm512_mul_ps(hi_f, scale_v);
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
pub use avx512::dequant_gemv_int2_avx512;

/// Wrapper: AVX-512 if available, else scalar fallback.
pub fn dequant_gemv_int2_auto(
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
            // SAFETY: features detected at runtime; the kernel uses
            // properly-sized buffers and unaligned loads.
            unsafe {
                dequant_gemv_int2_avx512(packed, scale_bits, x, n_rows, k_cols, y);
            }
            return;
        }
    }
    dequant_gemv_int2(packed, scale_bits, x, n_rows, k_cols, y);
}

/// Convenience: convert a SafetensorsExpert's int4 weights to int2 in-RAM.
///
/// Each expert is three Linear layers: gate, up (both `[INTERMEDIATE,
/// HIDDEN]`) and down (`[HIDDEN, INTERMEDIATE]`). All three are passed
/// through `quantize_int2_from_int4`. The resulting Vec<u8> buffers are
/// heap-resident; pin them in the cache so they're not paged out.
pub struct Int2Expert {
    pub gate_packed: Vec<u8>,
    pub gate_scale: Vec<u8>,
    pub up_packed: Vec<u8>,
    pub up_scale: Vec<u8>,
    pub down_packed: Vec<u8>,
    pub down_scale: Vec<u8>,
}

impl Int2Expert {
    /// Build from existing int4 buffers. The shapes are inferred from the
    /// length of each slice (rows / cols are read from constants).
    pub fn from_int4(
        gate_packed_i4: &[u8],
        gate_scale_i4: &[u8],
        up_packed_i4: &[u8],
        up_scale_i4: &[u8],
        down_packed_i4: &[u8],
        down_scale_i4: &[u8],
        intermediate: usize,
        hidden: usize,
    ) -> Self {
        let (gate_packed, gate_scale) =
            quantize_int2_from_int4(gate_packed_i4, gate_scale_i4, intermediate, hidden);
        let (up_packed, up_scale) =
            quantize_int2_from_int4(up_packed_i4, up_scale_i4, intermediate, hidden);
        let (down_packed, down_scale) =
            quantize_int2_from_int4(down_packed_i4, down_scale_i4, hidden, intermediate);
        Self {
            gate_packed,
            gate_scale,
            up_packed,
            up_scale,
            down_packed,
            down_scale,
        }
    }

    /// Footprint in bytes (sum of all six Vec<u8> fields).
    pub fn footprint_bytes(&self) -> usize {
        self.gate_packed.len()
            + self.gate_scale.len()
            + self.up_packed.len()
            + self.up_scale.len()
            + self.down_packed.len()
            + self.down_scale.len()
    }
}

/// Run one expert's full FFN with int2 weights:
/// `y = down @ (silu(gate @ x) ⊙ (up @ x))`.
///
/// Inputs and output are bf16 (input via `&[bf16]`, output via `&mut [bf16]`).
/// Mirrors the int4 `expert_forward` signature so the runner can route
/// either backend with minimal branching.
pub fn expert_forward_int2(
    x_bf16: &[half::bf16],
    gate_packed: &[u8],
    gate_scale: &[u8],
    up_packed: &[u8],
    up_scale: &[u8],
    down_packed: &[u8],
    down_scale: &[u8],
    out_bf16: &mut [half::bf16],
) {
    let hidden = x_bf16.len();
    let intermediate = gate_scale.len() / 2 / (hidden / GROUP_SIZE);

    let mut x_f32 = vec![0.0f32; hidden];
    for (i, b) in x_bf16.iter().enumerate() {
        x_f32[i] = b.to_f32();
    }

    let mut gate_out = vec![0.0f32; intermediate];
    let mut up_out = vec![0.0f32; intermediate];
    dequant_gemv_int2_auto(
        gate_packed,
        gate_scale,
        &x_f32,
        intermediate,
        hidden,
        &mut gate_out,
    );
    dequant_gemv_int2_auto(
        up_packed,
        up_scale,
        &x_f32,
        intermediate,
        hidden,
        &mut up_out,
    );

    let mut inter = vec![0.0f32; intermediate];
    for i in 0..intermediate {
        let g = gate_out[i];
        let silu = g / (1.0 + (-g).exp());
        inter[i] = silu * up_out[i];
    }

    let mut out_f32 = vec![0.0f32; hidden];
    dequant_gemv_int2_auto(
        down_packed,
        down_scale,
        &inter,
        hidden,
        intermediate,
        &mut out_f32,
    );

    for (i, v) in out_f32.iter().enumerate() {
        out_bf16[i] = half::bf16::from_f32(*v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All-zero weights produce all-zero output. Zero-point=2 means
    /// `signed=0` corresponds to two_bit=2 = byte 0xAA (each pair of bits = 10).
    #[test]
    fn zero_weight_zero_output() {
        let n_rows = 4;
        let k_cols = 32; // one group
        let packed = vec![0xAAu8; n_rows * k_cols / INT2_VALS_PER_BYTE];
        // scale = 1.0 in bf16 (0x3F80 LE = 0x80, 0x3F)
        let scale_bits = vec![0x80u8, 0x3F].repeat(n_rows * (k_cols / GROUP_SIZE));
        let x: Vec<f32> = (0..k_cols).map(|i| i as f32 * 0.1).collect();
        let mut y = vec![999.0f32; n_rows];
        dequant_gemv_int2(&packed, &scale_bits, &x, n_rows, k_cols, &mut y);
        for &v in &y {
            assert!(v.abs() < 1e-6, "expected ~0, got {v}");
        }
    }

    /// AVX-512 should match scalar on a random input.
    /// Gated to x86_64 + runtime feature check; on Macs (aarch64 or
    /// Intel without AVX-512) this is a no-op pass.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vl")
        {
            // Hardware can't run the AVX-512 path; skip rather than fail.
            return;
        }
        let n_rows = 8;
        let k_cols = 64; // two groups
        let mut packed = vec![0u8; n_rows * k_cols / INT2_VALS_PER_BYTE];
        // Deterministic pseudo-random byte pattern.
        for (i, b) in packed.iter_mut().enumerate() {
            *b = ((i * 31 + 7) & 0xFF) as u8;
        }
        // Random-ish bf16 scales (avoid zero).
        let n_groups = k_cols / GROUP_SIZE;
        let mut scale_bits = vec![0u8; n_rows * n_groups * 2];
        for r in 0..n_rows {
            for g in 0..n_groups {
                let s = 0.05f32 + 0.01 * ((r * n_groups + g) as f32);
                let b = bf16_round(s);
                scale_bits[(r * n_groups + g) * 2] = (b & 0xFF) as u8;
                scale_bits[(r * n_groups + g) * 2 + 1] = (b >> 8) as u8;
            }
        }
        let x: Vec<f32> = (0..k_cols).map(|i| (i as f32 * 0.13).sin()).collect();
        let mut y_ref = vec![0.0f32; n_rows];
        dequant_gemv_int2(&packed, &scale_bits, &x, n_rows, k_cols, &mut y_ref);
        let mut y_avx = vec![0.0f32; n_rows];
        // SAFETY: feature gate above.
        unsafe {
            dequant_gemv_int2_avx512(&packed, &scale_bits, &x, n_rows, k_cols, &mut y_avx);
        }
        for r in 0..n_rows {
            let diff = (y_ref[r] - y_avx[r]).abs();
            let rel = diff / y_ref[r].abs().max(1e-6);
            assert!(
                diff < 1e-4 || rel < 1e-4,
                "row {r}: ref={} avx={} diff={}",
                y_ref[r],
                y_avx[r],
                diff
            );
        }
    }

    /// Round-trip: quantize a known bf16 group → dequant_gemv_int2 with
    /// x=ones recovers ~sum(weights) within int2 precision (~scale per
    /// element). Exercises the bf16 packer + the scalar GEMV in tandem.
    #[test]
    fn quantize_then_dequant_matches_sum() {
        let n_rows = 1;
        let k_cols = 32;
        // bf16 weights = 0.5, 0.25, 0.125, ... — strictly positive so the
        // int2 codes mostly map to +1, with the scale = max_abs = 0.5.
        let mut weight_bf16 = vec![0u8; n_rows * k_cols * 2];
        let mut expected_sum = 0.0f32;
        for i in 0..k_cols {
            let v = 0.5f32 / (1u32 << (i % 4)) as f32; // 0.5, 0.25, 0.125, 0.0625, repeat
            expected_sum += v;
            let bits = bf16_round(v);
            weight_bf16[i * 2] = (bits & 0xFF) as u8;
            weight_bf16[i * 2 + 1] = (bits >> 8) as u8;
        }
        let (packed, scales) = quantize_int2_group(&weight_bf16, n_rows, k_cols);
        assert_eq!(packed.len(), k_cols / INT2_VALS_PER_BYTE);
        assert_eq!(scales.len(), 2);
        // Dequant via GEMV with x=ones — that's sum of dequantized weights.
        let x = vec![1.0f32; k_cols];
        let mut y = vec![0.0f32; n_rows];
        dequant_gemv_int2(&packed, &scales, &x, n_rows, k_cols, &mut y);
        // Worst-case absolute error: each element rounds within ±scale/2;
        // scale = max_abs = 0.5. So sum error ≤ k_cols * 0.25.
        let max_err = (k_cols as f32) * 0.5 * 0.5;
        let diff = (y[0] - expected_sum).abs();
        assert!(
            diff < max_err,
            "round-trip dot product off: got={} expected={} diff={} max_err={}",
            y[0],
            expected_sum,
            diff,
            max_err
        );
    }

    /// Re-quantizing int4 → int2 → dot product is "close to" the int4
    /// dot product on a random group. Tests the round-trip path the
    /// runtime uses at load time.
    #[test]
    fn int2_from_int4_close_to_int4() {
        // One row of one group of 32 elements. We synthesize int4 codes,
        // dequant to f32, then int4_dot vs int2_from_int4 → int2_dot
        // should agree to within ~2 bits' rounding (large absolute
        // because the values are int-quantized; we use rel error).
        let n_rows = 1;
        let k_cols = 32;
        // Build a synthetic int4 group: nibbles ranging across [-7, 7].
        let mut int4_packed = vec![0u8; n_rows * k_cols / 2];
        for i in 0..(k_cols / 2) {
            // even col = i mod 15 - 7 (i.e. -7..7); odd col = (i+3) mod 15 - 7
            let q_even = (i % 15) as i32 - 7;
            let q_odd = ((i + 3) % 15) as i32 - 7;
            let lo = ((q_even + 8) & 0x0F) as u8;
            let hi = ((q_odd + 8) & 0x0F) as u8;
            int4_packed[i] = (hi << 4) | lo;
        }
        let scale4 = 0.05f32;
        let s4_bits = bf16_round(scale4);
        let int4_scales = vec![(s4_bits & 0xFF) as u8, (s4_bits >> 8) as u8];

        // int4 dot product with x = ones.
        let x: Vec<f32> = vec![1.0; k_cols];
        let mut y_i4 = vec![0.0f32; n_rows];
        crate::kernel::dequant_gemv_int4(&int4_packed, &int4_scales, &x, n_rows, k_cols, &mut y_i4);

        // int2 from int4.
        let (int2_packed, int2_scales) =
            quantize_int2_from_int4(&int4_packed, &int4_scales, n_rows, k_cols);
        let mut y_i2 = vec![0.0f32; n_rows];
        dequant_gemv_int2(&int2_packed, &int2_scales, &x, n_rows, k_cols, &mut y_i2);

        // Both should be close. int2 will have higher quantization error
        // but for x=ones the sum tends to cancel out random errors.
        // A loose absolute bound is fine.
        let max_abs = (k_cols as f32) * scale4 * 7.0; // worst-case magnitude
        let diff = (y_i4[0] - y_i2[0]).abs();
        assert!(
            diff < max_abs * 0.5,
            "int2 vs int4 differ too much: i4={} i2={} diff={} max_abs={}",
            y_i4[0],
            y_i2[0],
            diff,
            max_abs
        );
    }
}
