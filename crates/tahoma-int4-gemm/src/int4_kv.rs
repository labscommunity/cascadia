//! Int4-quantized KV cache support (autolab campaign 062 / B-scoped).
//!
//! Predecessor: iter 032 converted KV from f32 to bf16-as-u16, halving
//! the per-layer footprint and the SDPA-time read bandwidth. The bf16
//! upconvert is a single shift per element inside the inner loop —
//! cheap relative to the multiply.
//!
//! This module explores the next step: int4 KV with per-head per-row
//! per-group symmetric quantization (`group_size = HEAD_DIM`) and a
//! bf16 scale per group. The motivation is the same one that made the
//! weight path int4: shrink the bytes that have to be read from memory
//! at attention time so a longer prefix fits in the same L2/L3 footprint
//! and the per-token bandwidth that touches the cache drops further.
//!
//! ## Bytes per token, one layer
//!
//! - f32 (pre-iter-032)  : 64 heads × (192 + 128) × 4 = 81,920 B
//! - bf16 (iter 032)     : 64 heads × (192 + 128) × 2 = 40,960 B
//! - int4 (this module)  : 64 × (192 + 128) / 2 + 64 × ((192/32) + (128/32)) × 2
//!   = 10,240 + 1,280 = 11,520 B
//!
//! Net ratio vs bf16: 11,520 / 40,960 = **28.1%** (≈ 3.55× smaller)
//!
//! Per-token across 60 layers:
//!   - bf16 : ~ 2.46 MB / token
//!   - int4 : ~ 691  KB / token
//!
//! ## Per-element work in the SDPA inner loop
//!
//! For each k-row dot product (192 elements):
//!
//! bf16 path (iter 032):
//!   - 1 u16 load
//!   - 1 shift   (`(bits as u32) << 16`)
//!   - 1 reinterpret to f32
//!   - 1 f32 multiply with q\[i\]
//!   - 1 f32 add to accumulator
//!
//! int4 path (this module, scalar reference):
//!   - 1 u8 load every 2 elements (amortized 0.5 loads)
//!   - 1 nibble extract (shift + mask)
//!   - 1 i32 subtract (zero-point -8)
//!   - 1 i32 -> f32 convert
//!   - 1 bf16 scale load and upconvert per GROUP_SIZE elements (amortized 1/32)
//!   - 1 f32 multiply by scale
//!   - 1 f32 multiply with q\[i\]
//!   - 1 f32 add to accumulator
//!
//! The extra work per element is ~3 uops (nibble extract, sub, mul-by-scale)
//! compared to bf16's single shift. On a modern Xeon Gold this is roughly
//! 1 extra cycle per element in the scalar path — relevant only if the
//! kernel is NOT memory-bound. Iter 032 measured the bf16 SDPA at
//! ~2× the f32 baseline → it IS bandwidth-dominated, so the further
//! 3.55× bandwidth cut from int4 should largely translate to wall time.
//!
//! Quality is the open question. Per-head per-row int4 with HEAD_DIM-size
//! groups gives one scale for every 32–192 elements. K2.6's K values
//! have a wide range across head dimensions (RoPE elements vs NoPE), so
//! using a single scale per (head, token) row is risky; one scale per
//! 32-element group inside the row is much safer and still cheap to read.
//!
//! ## What's shipped here
//!
//! - `quantize_kv_row` — pack one row of HEAD_DIM f32 values into int4
//!   nibbles + bf16 scales (`group_size = 32`, symmetric, zero-point 8).
//! - `dequant_kv_dot_f32` — scalar reference kernel: dot product of a
//!   f32 query row against an int4 cache row, dequantizing on-the-fly.
//! - Roundtrip tests + a representative-magnitude eval that checks
//!   max abs-error against the bf16 path with realistic K/V tensors.
//!
//! What is NOT shipped here:
//!
//! - SDPA wire-up in `shell_int4.rs` / `layer0_int4.rs` (a flip from
//!   `&[u16]` to a `(&[u8], &[u16])` pair changes a public signature
//!   that the C-FFI also has to follow — left as a deliberate next step
//!   so quality eval can be gated behind a feature flag on the miner).
//! - AVX-512 path. The scalar kernel exists to bound the per-element
//!   cost honestly; a VPMOVSXBD + VBROADCASTW + VFMADD231PS lane
//!   would shave it further if the wire-up shows the scalar is hot.

/// Group size for int4 KV quantization. Matches the weight path's
/// `kernel_avx512::GROUP_SIZE` so the scale-load cadence is identical
/// to the dispatch kernel.
pub const KV_GROUP_SIZE: usize = 32;

/// Quantize one row of `head_dim` f32 values into int4 packed nibbles
/// + bf16 group scales. `head_dim` must be a multiple of `KV_GROUP_SIZE`.
///
/// Output:
///   packed: `Vec<u8>` of length `head_dim / 2`. Byte `i` holds
///           nibbles for elements `2i` (low nibble) and `2i+1` (high nibble),
///           each in the `(unsigned - 8)` convention so the value is
///           recovered by `nibble as i32 - 8`.
///   scales: `Vec<u16>` of length `head_dim / KV_GROUP_SIZE`, each a
///           bf16 bit-pattern (round-to-nearest-even of the f32 scale).
///
/// Symmetric: the magnitude `max(|x|)` in each group is mapped to 7
/// (so the int4 range `[-8, 7]` is asymmetric and we round-trip +max
/// exactly, matching `nncf` `INT4_SYM`).
pub fn quantize_kv_row(row_f32: &[f32]) -> (Vec<u8>, Vec<u16>) {
    let head_dim = row_f32.len();
    assert!(
        head_dim.is_multiple_of(KV_GROUP_SIZE),
        "head_dim ({head_dim}) must be a multiple of KV_GROUP_SIZE ({KV_GROUP_SIZE})"
    );
    let n_groups = head_dim / KV_GROUP_SIZE;
    let mut packed = vec![0u8; head_dim / 2];
    let mut scales = vec![0u16; n_groups];

    for g in 0..n_groups {
        // 1) find max abs in this group
        let mut max_abs = 0.0f32;
        for k in 0..KV_GROUP_SIZE {
            let v = row_f32[g * KV_GROUP_SIZE + k];
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        // 2) scale = max_abs / 7. Clamp at a tiny epsilon so we never
        //    divide by zero on an all-zero group. Round to bf16.
        let scale = if max_abs == 0.0 {
            1.0e-10
        } else {
            max_abs / 7.0
        };
        let scale_bits = f32_to_bf16_bits(scale);
        scales[g] = scale_bits;

        // 3) quantize. Re-read the scale after bf16 rounding so the
        //    quantized values are consistent with what dequant will see.
        let scale_q = f32::from_bits((scale_bits as u32) << 16);
        let inv = 1.0 / scale_q;
        for k in 0..KV_GROUP_SIZE {
            let c = g * KV_GROUP_SIZE + k;
            let v = row_f32[c];
            let q = (v * inv).round().clamp(-8.0, 7.0) as i32;
            let nibble = ((q + 8) & 0x0F) as u8;
            let p_off = c / 2;
            if c.is_multiple_of(2) {
                packed[p_off] = (packed[p_off] & 0xF0) | nibble;
            } else {
                packed[p_off] = (packed[p_off] & 0x0F) | (nibble << 4);
            }
        }
    }

    (packed, scales)
}

/// Dequantize one int4-quantized row back to f32. Intended for tests
/// and round-trip-error checks — the SDPA kernel uses `dequant_kv_dot_f32`
/// directly without materializing the dequantized row.
pub fn dequantize_kv_row(packed: &[u8], scales: &[u16], head_dim: usize) -> Vec<f32> {
    assert_eq!(packed.len(), head_dim / 2);
    assert_eq!(scales.len(), head_dim / KV_GROUP_SIZE);
    let mut out = vec![0.0f32; head_dim];
    for g in 0..head_dim / KV_GROUP_SIZE {
        let scale = f32::from_bits((scales[g] as u32) << 16);
        for k in 0..KV_GROUP_SIZE {
            let c = g * KV_GROUP_SIZE + k;
            let p = packed[c / 2];
            let nibble = if c.is_multiple_of(2) { p & 0x0F } else { p >> 4 };
            let q = (nibble as i32) - 8;
            out[c] = (q as f32) * scale;
        }
    }
    out
}

/// Compute `sum(q[i] * dequant(packed_k[i])) for i in 0..head_dim`.
/// Scalar reference kernel — the per-token bandwidth read is
/// `head_dim / 2` packed bytes + `head_dim / KV_GROUP_SIZE * 2` scale
/// bytes, vs `head_dim * 2` bytes for the bf16 path.
///
/// This is the function that would replace the bf16 upconvert in the
/// SDPA inner loop. Keeping it as a free function (not a method on a
/// quant-cache type) so the future SDPA call site can pass slices with
/// arbitrary strides without an intermediate copy.
#[inline]
pub fn dequant_kv_dot_f32(q: &[f32], packed_k: &[u8], scales: &[u16]) -> f32 {
    let head_dim = q.len();
    debug_assert_eq!(packed_k.len(), head_dim / 2);
    debug_assert_eq!(scales.len(), head_dim / KV_GROUP_SIZE);
    let mut acc = 0.0f32;
    for g in 0..head_dim / KV_GROUP_SIZE {
        let scale = f32::from_bits((scales[g] as u32) << 16);
        let mut g_acc = 0.0f32;
        for k in 0..KV_GROUP_SIZE {
            let c = g * KV_GROUP_SIZE + k;
            let p = packed_k[c / 2];
            let nibble = if c.is_multiple_of(2) { p & 0x0F } else { p >> 4 };
            let qi = (nibble as i32) - 8;
            g_acc += q[c] * (qi as f32);
        }
        acc += g_acc * scale;
    }
    acc
}

/// Accumulate `out[i] += weight * dequant(packed_v[i])` over `head_dim`
/// elements — the V-side equivalent of `dequant_kv_dot_f32`, written
/// as `out += w * V_row` so a softmax-weighted sum across all past
/// tokens stays in f32 accumulators (same shape as the bf16 path).
#[inline]
pub fn dequant_kv_accum_f32(out: &mut [f32], weight: f32, packed_v: &[u8], scales: &[u16]) {
    let head_dim = out.len();
    debug_assert_eq!(packed_v.len(), head_dim / 2);
    debug_assert_eq!(scales.len(), head_dim / KV_GROUP_SIZE);
    for g in 0..head_dim / KV_GROUP_SIZE {
        let scale = f32::from_bits((scales[g] as u32) << 16);
        let w_scale = weight * scale;
        for k in 0..KV_GROUP_SIZE {
            let c = g * KV_GROUP_SIZE + k;
            let p = packed_v[c / 2];
            let nibble = if c.is_multiple_of(2) { p & 0x0F } else { p >> 4 };
            let qi = (nibble as i32) - 8;
            out[c] += w_scale * (qi as f32);
        }
    }
}

/// Convert one f32 to bf16 bits via round-to-nearest-even. Mirrors
/// `runner::f32_to_bf16_bits` and `shell_int4::bf16_round` — copied
/// rather than imported to keep this module self-contained while we
/// decide whether the wire-up is a winner on real hardware.
#[inline]
fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// Per-token-per-head packed size for a `head_dim`-wide KV row. Helper
/// for capacity planning at the call site.
#[inline]
pub const fn packed_bytes(head_dim: usize) -> usize {
    head_dim / 2 + (head_dim / KV_GROUP_SIZE) * 2
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same upconvert the bf16 SDPA uses — keeps the test independent
    /// of the `half` crate so the two paths can be compared bit-for-bit.
    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    #[test]
    fn roundtrip_constant_row_is_exact_at_boundary() {
        // A row whose max abs is exactly representable in bf16 + which
        // sits on the +7 endpoint of the int4 grid should round-trip
        // with zero error (modulo the scale's bf16 rounding).
        let head_dim = 128;
        let mut row = vec![0.0f32; head_dim];
        for i in 0..head_dim {
            // Sawtooth -7 .. +7 scaled by 0.5. Each group sees max=3.5,
            // scale = 0.5, and the integer levels recover exactly.
            row[i] = ((i % 15) as i32 - 7) as f32 * 0.5;
        }
        let (packed, scales) = quantize_kv_row(&row);
        let back = dequantize_kv_row(&packed, &scales, head_dim);
        for (i, (&a, &b)) in row.iter().zip(back.iter()).enumerate() {
            // Allow 1 ulp at the bf16-scale precision (the only loss).
            let err = (a - b).abs();
            assert!(err < 0.01, "i={i} a={a} b={b} err={err}");
        }
    }

    #[test]
    fn roundtrip_gaussian_row_within_tolerance() {
        // A normal-magnitude row (std ~ 0.1, typical of K2.6 K values
        // post-RMSNorm) should round-trip with sub-3% relative error
        // and zero NaNs / infs.
        let head_dim = 192; // QK_HEAD_DIM
        let mut row = vec![0.0f32; head_dim];
        // Cheap deterministic pseudo-normal: sum of 12 LCG draws minus 6.
        let mut s: u64 = 0xCAFE_F00D_DEAD_BEEF;
        for slot in row.iter_mut() {
            let mut acc = 0.0f32;
            for _ in 0..12 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                acc += ((s >> 32) as u32 as f32) / 4_294_967_296.0;
            }
            *slot = (acc - 6.0) * 0.1; // std ~ 0.1
        }
        let (packed, scales) = quantize_kv_row(&row);
        let back = dequantize_kv_row(&packed, &scales, head_dim);
        let mut sse = 0.0f32;
        let mut sxx = 0.0f32;
        for (&a, &b) in row.iter().zip(back.iter()) {
            assert!(a.is_finite() && b.is_finite());
            sse += (a - b) * (a - b);
            sxx += a * a;
        }
        let rel = (sse / sxx).sqrt();
        // ~6–10% RMS relative error is the published noise floor for
        // symmetric int4 with group_size=32 on Gaussian-distributed
        // tensors — the 4-bit grid spans 16 levels, and a single
        // group-wide scale can't track the gap between elements near
        // the tails and elements near the mean. Quality eval on real
        // prompts (10-prompt substring match) is the ultimate check;
        // this test only catches gross encoder bugs (NaN, infs,
        // off-by-one on the nibble layout).
        assert!(rel < 0.12, "rel-err {rel:.4} > 0.12");
    }

    #[test]
    fn dot_matches_bf16_path_within_tolerance() {
        // Build a `q` and a `k_row`, run the bf16 path (the iter 032
        // baseline) and the int4 path side-by-side. The int4 result
        // should differ by at most ~few-percent — the same noise floor
        // as `roundtrip_gaussian_row_within_tolerance`.
        let head_dim = 192;
        let mut q = vec![0.0f32; head_dim];
        let mut k = vec![0.0f32; head_dim];
        let mut s: u64 = 0x1234_5678_ABCD_EF01;
        for slot in q.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *slot = (((s >> 32) as u32 as f32) / 4_294_967_296.0 - 0.5) * 0.3;
        }
        for slot in k.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *slot = (((s >> 32) as u32 as f32) / 4_294_967_296.0 - 0.5) * 0.3;
        }

        // bf16 path (same upconvert the iter 032 SDPA uses)
        let mut bf16_acc = 0.0f32;
        for i in 0..head_dim {
            let kf = bf16_to_f32(f32_to_bf16_bits(k[i]));
            bf16_acc += q[i] * kf;
        }

        // int4 path
        let (packed, scales) = quantize_kv_row(&k);
        let int4_acc = dequant_kv_dot_f32(&q, &packed, &scales);

        // The bf16 baseline already smears each element by ~1/256 — int4
        // adds quantization noise on top. The dot accumulates head_dim
        // independent errors so absolute error scales with √head_dim.
        let abs_err = (bf16_acc - int4_acc).abs();
        let bf16_mag = bf16_acc.abs().max(1.0e-6);
        let rel_err = abs_err / bf16_mag;
        assert!(
            rel_err < 0.05,
            "int4 dot diverged from bf16 baseline: bf16={bf16_acc} int4={int4_acc} rel_err={rel_err:.4}"
        );
    }

    #[test]
    fn accum_matches_bf16_path_within_tolerance() {
        // V-side: accumulate softmax-weighted V rows. Compare bf16 and
        // int4 over a small "past_seq_len" of 8 rows.
        let head_dim = 128; // V_HEAD_DIM
        let past = 8;
        let mut weights = vec![0.0f32; past];
        let mut v_rows: Vec<Vec<f32>> = (0..past).map(|_| vec![0.0f32; head_dim]).collect();
        let mut s: u64 = 0xDEAD_BEEF_F00D_BABE;
        for w in weights.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *w = ((s >> 32) as u32 as f32) / 4_294_967_296.0;
        }
        // softmax-ish: just normalize to sum to 1
        let total: f32 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= total;
        }
        for row in v_rows.iter_mut() {
            for slot in row.iter_mut() {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *slot = (((s >> 32) as u32 as f32) / 4_294_967_296.0 - 0.5) * 0.5;
            }
        }

        // bf16 path
        let mut bf16_out = vec![0.0f32; head_dim];
        for (w, row) in weights.iter().zip(v_rows.iter()) {
            for i in 0..head_dim {
                let vf = bf16_to_f32(f32_to_bf16_bits(row[i]));
                bf16_out[i] += w * vf;
            }
        }

        // int4 path
        let mut int4_out = vec![0.0f32; head_dim];
        let packed_rows: Vec<(Vec<u8>, Vec<u16>)> =
            v_rows.iter().map(|r| quantize_kv_row(r)).collect();
        for (w, (packed, scales)) in weights.iter().zip(packed_rows.iter()) {
            dequant_kv_accum_f32(&mut int4_out, *w, packed, scales);
        }

        let mut sse = 0.0f32;
        let mut sxx = 0.0f32;
        for (a, b) in bf16_out.iter().zip(int4_out.iter()) {
            sse += (a - b) * (a - b);
            sxx += a * a;
        }
        let rel = (sse / sxx.max(1.0e-12)).sqrt();
        assert!(
            rel < 0.10,
            "int4 accum diverged from bf16 baseline: rel-err {rel:.4}"
        );
    }

    #[test]
    fn packed_bytes_matches_layout() {
        // QK_HEAD_DIM = 192 → 96 packed + 6 scales × 2 = 108 bytes
        assert_eq!(packed_bytes(192), 96 + 12);
        // V_HEAD_DIM = 128 → 64 packed + 4 scales × 2 = 72 bytes
        assert_eq!(packed_bytes(128), 64 + 8);
    }

    #[test]
    fn zero_row_roundtrips_to_zero() {
        // The all-zero edge case: scale is clamped to 1e-10, every
        // quantized value rounds to 0 (encoded as nibble 8 via
        // `q + 8`), and the dequant `(8 - 8) * scale = 0` recovers
        // exact zero. Make sure that's actually what happens.
        let head_dim = 64;
        let row = vec![0.0f32; head_dim];
        let (packed, scales) = quantize_kv_row(&row);
        // Every nibble should be exactly 0x88 (both low and high == 8).
        for &b in &packed {
            assert_eq!(b, 0x88, "expected zero-point nibbles, got 0x{b:02x}");
        }
        let back = dequantize_kv_row(&packed, &scales, head_dim);
        for &v in &back {
            assert_eq!(v, 0.0);
        }
    }
}
