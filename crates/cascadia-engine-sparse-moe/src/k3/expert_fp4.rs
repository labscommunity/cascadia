//! fp4 (e2m1) expert weights with E8M0 group-32 scales.
//!
//! K3's routed experts ship as `mxfp4-pack-quantized`, which the dsv4
//! [`crate::dsv4::expert_mmap`] kernel cannot decode: that one assumes a LINEAR
//! symmetric grid (`(nibble - 8) * bf16_scale`), while e2m1 is a NONLINEAR grid
//! `{0, .5, 1, 1.5, 2, 3, 4, 6}` with a sign bit and a power-of-two shared
//! exponent per 32 weights.
//!
//! On-disk section layout, matching `tools/export_kimi_k3.py`:
//!
//! ```text
//! [out * in / 2]  packed nibbles   (low nibble = even column)
//! [out * in / 32] E8M0 scale bytes (value = 2^(byte - 127))
//! ```
//!
//! An expert is three such sections back to back: `w1` (gate) and `w3` (up),
//! both `[inter, dim]`, then `w2` (down) `[dim, inter]`.

/// e2m1 magnitude grid, indexed by the low 3 bits of a nibble.
pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Weights per E8M0 scale.
pub const GROUP: usize = 32;

/// Decode one nibble to its f32 value (sign in bit 3).
#[inline]
pub fn decode_nibble(n: u8) -> f32 {
    let mag = E2M1[(n & 0x07) as usize];
    if n & 0x08 != 0 {
        -mag
    } else {
        mag
    }
}

/// `2^(byte - 127)` — the E8M0 shared exponent.
///
/// Built from the exponent field rather than `powi`, which with a RUNTIME
/// exponent is repeated squaring. This runs once per group of 32 columns, so
/// ~344k times per section GEMV.
///
/// Bit-exact for all 256 inputs: `b` IS the biased exponent, so shifting it into
/// place is the identity (b=127 -> 1.0, b=128 -> 2.0). Only b=0 is special, its
/// value being subnormal, and it keeps the old result rather than flushing to 0.
///
/// MEASURE THIS WITH THE ACCUMULATOR SPLIT, NOT ALONE. On its own it is neutral
/// (3.340 -> 3.367 ms) because the FMA dependency chain in the AVX2 loop left
/// idle cycles that hid it. Remove that chain and this becomes the next
/// bottleneck; the two together are 3.340 -> 0.572 ms, and either alone is
/// nothing.
#[inline]
pub fn e8m0_to_f32(b: u8) -> f32 {
    if b == 0 {
        return 2.0f32.powi(-127);
    }
    f32::from_bits((b as u32) << 23)
}

/// Byte size of one packed `[out, in]` section (nibbles then scales).
#[inline]
pub fn section_bytes(out_dim: usize, in_dim: usize) -> usize {
    out_dim * in_dim / 2 + out_dim * (in_dim / GROUP)
}

/// Total bytes of one expert: `w1`, `w3` (`[inter, dim]`) then `w2` (`[dim, inter]`).
#[inline]
pub fn expert_bytes(dim: usize, inter: usize) -> usize {
    2 * section_bytes(inter, dim) + section_bytes(dim, inter)
}

/// `2 * decode_nibble(n)`, which is an exact small integer for every nibble.
///
/// The doubling is what makes the table representable in `i8`: the e2m1 grid
/// contains 0.5, so `decode_nibble` itself is not integral, but `2 * grid` is
/// `{0, 1, 2, 3, 4, 6, 8, 12}`. Recovering the true value is a multiply by 0.5,
/// a power of two, so `decode_nibble(n) == LUT2[n] as f32 * 0.5` EXACTLY — the
/// SIMD path folds that 0.5 into the group's shared scale and loses nothing.
///
/// A 16-entry byte table is precisely the shape `vpshufb` consumes, which is
/// why the nonlinear e2m1 grid can be decoded as cheaply as dsv4's linear one.
///
/// It is also what makes the VNNI path below possible: `vpdpbusd` multiplies
/// bytes, and these are the e2m1 grid AS bytes, exactly.
const LUT2: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// `CASCADIA_K3_SIMD=0` forces the scalar kernel, for A/B measurement.
#[cfg(target_arch = "x86_64")]
fn simd_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CASCADIA_K3_SIMD").as_deref() != Ok("0"))
}

/// `CASCADIA_K3_VNNI=1` enables the int8 VNNI kernel. OFF by default.
///
/// Off because it is the one path here that is not exact: it quantises the
/// ACTIVATIONS to int8 per group of 32. The weights are unaffected — they are
/// already the e2m1 grid and `LUT2` holds it exactly — but `x` is not, so
/// results differ from every other kernel in this file by more than rounding.
/// See `vnni_accuracy_vs_exact` for the measured size of that difference.
#[cfg(target_arch = "x86_64")]
fn vnni_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CASCADIA_K3_VNNI").as_deref() == Ok("1"))
}

/// Quantise activations to int8, one scale per [`GROUP`] columns.
///
/// `q` is `in_dim` bytes, `qd` is `in_dim / GROUP` floats. Symmetric, no zero
/// point: the reconstruction is `q[i] * qd[i / GROUP]`.
///
/// Clamped to ±127, not ±128. `vpdpbusd` takes one operand unsigned, and the
/// sign is carried by negating the activation where the weight is negative —
/// `-128` has no positive counterpart in i8, so it would wrap to itself and
/// silently flip that term's sign.
pub fn quantize_activations(x: &[f32], in_dim: usize, q: &mut [i8], qd: &mut [f32]) {
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(q.len(), in_dim);
    debug_assert_eq!(qd.len(), in_dim / GROUP);

    for ((d, xg), qg) in qd
        .iter_mut()
        .zip(x.chunks_exact(GROUP))
        .zip(q.chunks_exact_mut(GROUP))
    {
        let amax = xg.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        *d = amax / 127.0;
        // An all-zero group has no step; a reciprocal here would be infinite and
        // every value in the group would come out NaN.
        let inv = if *d > 0.0 { 1.0 / *d } else { 0.0 };
        for (o, &v) in qg.iter_mut().zip(xg) {
            *o = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

/// Scalar reference for the VNNI kernel: exact int8 arithmetic, no SIMD.
///
/// The per-group dot is computed in `i32` and is EXACT — `LUT2` values reach 12
/// and `q` reaches 127, so a group's 32 terms cannot exceed 48,768. All the
/// error in this path comes from [`quantize_activations`], none from here,
/// which is what lets the accuracy test attribute it.
pub fn dequant_row_dot_q8_scalar(
    packed_row: &[u8],
    scales_row: &[u8],
    q: &[i8],
    qd: &[f32],
    in_dim: usize,
) -> f32 {
    debug_assert_eq!(packed_row.len(), in_dim / 2);
    debug_assert_eq!(scales_row.len(), in_dim / GROUP);

    let mut acc = 0.0f32;
    for (gi, &sb) in scales_row.iter().enumerate() {
        let base = gi * GROUP;
        let mut sum = 0i32;
        for j in 0..GROUP / 2 {
            let b = packed_row[base / 2 + j];
            let c = base + 2 * j;
            sum += LUT2[(b & 0x0F) as usize] as i32 * q[c] as i32;
            sum += LUT2[(b >> 4) as usize] as i32 * q[c + 1] as i32;
        }
        // LUT2 is 2x the real grid, so the 0.5 rides along with the two scales.
        acc += sum as f32 * (e8m0_to_f32(sb) * 0.5 * qd[gi]);
    }
    acc
}

/// Fused dequant + dot for one output row:
/// `sum_k decode(nibble_k) * 2^(scale(g(k)) - 127) * x[k]`.
///
/// `packed_row` is `in_dim/2` bytes, `scales_row` is `in_dim/GROUP` bytes.
///
/// Dispatches to the AVX2 kernel when the CPU has it; the scalar body below
/// stays the reference and is what runs everywhere else.
pub fn dequant_row_dot(packed_row: &[u8], scales_row: &[u8], x: &[f32], in_dim: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // Lunar Lake and friends have AVX2 + AVX-VNNI but no AVX-512, so this
        // is the widest path that actually exists on the target AI-PCs.
        if simd_enabled() && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature-detected immediately above; slice lengths are
            // re-checked by the callee's debug asserts and by the loop bounds.
            return unsafe { avx2::dequant_row_dot_avx2(packed_row, scales_row, x, in_dim) };
        }
    }
    dequant_row_dot_scalar(packed_row, scales_row, x, in_dim)
}

/// Scalar reference for [`dequant_row_dot`]. Kept reachable so tests can A/B it.
pub fn dequant_row_dot_scalar(
    packed_row: &[u8],
    scales_row: &[u8],
    x: &[f32],
    in_dim: usize,
) -> f32 {
    debug_assert_eq!(packed_row.len(), in_dim / 2);
    debug_assert_eq!(scales_row.len(), in_dim / GROUP);
    debug_assert_eq!(x.len(), in_dim);

    let mut acc = 0.0f32;
    for (gi, &sb) in scales_row.iter().enumerate() {
        let s = e8m0_to_f32(sb);
        let base = gi * GROUP;
        let mut part = 0.0f32;
        // GROUP columns = GROUP/2 packed bytes
        for j in 0..GROUP / 2 {
            let b = packed_row[base / 2 + j];
            let c = base + 2 * j;
            part += decode_nibble(b & 0x0F) * x[c];
            part += decode_nibble(b >> 4) * x[c + 1];
        }
        acc += part * s;
    }
    acc
}

/// AVX2 (256-bit) fp4 kernel.
///
/// Why not reuse dsv4's int4 AVX-512 kernel: that one decodes a LINEAR grid, so
/// `nibble - 8` is the whole decode and a single `vpsubb` does it. e2m1 is
/// nonlinear, so the decode has to be a lookup — but the table is 16 bytes wide,
/// which is exactly one `vpshufb` per 128-bit lane. And the target hardware
/// (Lunar Lake AI-PCs) has no AVX-512 at all, so the 512-bit kernel would be
/// dead code there regardless.
///
/// The mask/shift/interleave/sign-extend skeleton is deliberately identical to
/// [`crate::dsv4::expert_mmap`]'s AVX2 kernel, so the column ordering is the one
/// that is already pinned by the dsv4 tests; only the decode step differs
/// (`vpshufb` through a table instead of `vpsubb` by 8).
///
/// One group (32 columns = 16 packed bytes) per iteration. The group's E8M0
/// scale is applied ONCE to the group's partial sum rather than to each weight,
/// which both saves multiplies and keeps the arithmetic close to the scalar
/// reference (which also scales the group total).
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{e8m0_to_f32, GROUP, LUT2};
    use core::arch::x86_64::*;

    /// Horizontal sum of 8 f32 lanes.
    #[target_feature(enable = "avx2")]
    unsafe fn hsum256(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps::<1>(v);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
        _mm_cvtss_f32(s)
    }

    /// Caller must have checked `avx2` and `fma`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dequant_row_dot_avx2(
        packed_row: &[u8],
        scales_row: &[u8],
        x: &[f32],
        in_dim: usize,
    ) -> f32 {
        // Hard asserts, not debug: the loads below are raw pointer arithmetic
        // driven by these lengths, so a mismatched caller would read OOB rather
        // than panic the way the scalar path's indexing does. Three branches per
        // row against ~112 group iterations is free.
        assert_eq!(packed_row.len(), in_dim / 2);
        assert_eq!(scales_row.len(), in_dim / GROUP);
        assert_eq!(x.len(), in_dim);

        let lut = _mm_loadu_si128(LUT2.as_ptr() as *const __m128i);
        let lo_mask = _mm_set1_epi8(0x0F);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for (gi, &sb) in scales_row.iter().enumerate() {
            let base = gi * GROUP;
            // 16 packed bytes = 32 nibbles = one full group.
            let pk = _mm_loadu_si128(packed_row.as_ptr().add(base / 2) as *const __m128i);
            // low nibble -> EVEN columns, high nibble -> ODD columns
            let lo_n = _mm_and_si128(pk, lo_mask);
            let hi_n = _mm_and_si128(_mm_srli_epi16::<4>(pk), lo_mask);
            // vpshufb: byte j of the result is lut[index_j & 0x0F]. Indices are
            // already masked to 0..15, so no lane is zeroed out.
            let lo_v = _mm_shuffle_epi8(lut, lo_n);
            let hi_v = _mm_shuffle_epi8(lut, hi_n);
            // Byte j of packed holds columns 2j (low) and 2j+1 (high), so
            // interleaving restores ascending column order.
            let cols_0_15 = _mm_unpacklo_epi8(lo_v, hi_v);
            let cols_16_31 = _mm_unpackhi_epi8(lo_v, hi_v);

            let xp = x.as_ptr().add(base);
            // The LUT is 2x the real magnitude; fold the 0.5 into the scale.
            // Both factors are powers of two, so the product is exact.
            let s_half = _mm256_set1_ps(e8m0_to_f32(sb) * 0.5);

            // FOUR accumulators, not one. Chaining these into a single register
            // makes each FMA wait on the previous one's ~4-cycle latency, so a
            // group costs ~16 cycles of latency where the hardware could retire
            // it in ~2 of throughput. Scaling x rather than w keeps the FMAs
            // independent without an extra multiply on the weight side.
            // i8 -> i32 -> f32, 8 columns at a time.
            let w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(cols_0_15));
            let w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(cols_0_15)));
            let w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(cols_16_31));
            let w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(cols_16_31)));

            let x0 = _mm256_mul_ps(_mm256_loadu_ps(xp), s_half);
            let x1 = _mm256_mul_ps(_mm256_loadu_ps(xp.add(8)), s_half);
            let x2 = _mm256_mul_ps(_mm256_loadu_ps(xp.add(16)), s_half);
            let x3 = _mm256_mul_ps(_mm256_loadu_ps(xp.add(24)), s_half);

            acc0 = _mm256_fmadd_ps(w0, x0, acc0);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);
            acc2 = _mm256_fmadd_ps(w2, x2, acc2);
            acc3 = _mm256_fmadd_ps(w3, x3, acc3);
        }
        // Pairwise, so the two adds are independent too.
        hsum256(_mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        ))
    }

    /// The int8 VNNI row dot, once per available instruction encoding.
    ///
    /// `vpdpbusd` exists twice on Intel: as AVX-VNNI (VEX, what Lunar Lake and
    /// every other AI-PC part has) and as AVX512-VNNI + VL (EVEX, what the older
    /// Xeons have). Same operation, different intrinsic, and no single target
    /// feature covers both — hence one body, two instantiations, picked at
    /// runtime. Verifying on an AVX512-VNNI box therefore does verify the
    /// arithmetic that will run on an AI-PC.
    ///
    /// Per group of 32 columns this is: one `vpshufb` decode, two `vpsignb` for
    /// the sign trick, one `vpdpbusd`, one convert and one FMA — against roughly
    /// 25 operations in the f32 kernel, which spends most of them widening
    /// `i8 -> i32 -> f32` purely so the FMA has something to eat.
    ///
    /// `vpdpbusd` is unsigned x signed. The weights carry the sign, so `|w|`
    /// becomes the unsigned operand and the sign moves onto the activation:
    /// `|w| * (q * sign(w)) == w * q`. Where `w == 0` both terms are zeroed,
    /// which is also correct.
    ///
    /// `vpdpbssd` (AVX-VNNI-INT8) is signed x signed and needs none of that, so
    /// the `signed` arm drops both `vpsignb`. Lunar Lake reports `avxvnniint8`,
    /// so that arm is the one AI-PCs actually take.
    macro_rules! vnni_row_dot {
        ($name:ident, $feat:literal, $dp:ident, unsigned) => {
            vnni_row_dot!(@body $name, $feat, $dp, |w, xq| (
                _mm256_sign_epi8(w, w),   // |w|
                _mm256_sign_epi8(xq, w),  // q * sign(w)
            ));
        };
        ($name:ident, $feat:literal, $dp:ident, signed) => {
            // Signed x signed: the operands go in as they are.
            vnni_row_dot!(@body $name, $feat, $dp, |w, xq| (w, xq));
        };
        (@body $name:ident, $feat:literal, $dp:ident, $ops:expr) => {
            /// Caller must have checked this kernel's features.
            #[target_feature(enable = $feat)]
            pub unsafe fn $name(
                packed_row: &[u8],
                scales_row: &[u8],
                q: &[i8],
                qd: &[f32],
                in_dim: usize,
            ) -> f32 {
                assert_eq!(packed_row.len(), in_dim / 2);
                assert_eq!(scales_row.len(), in_dim / GROUP);
                assert_eq!(q.len(), in_dim);
                assert_eq!(qd.len(), in_dim / GROUP);

                let lut = _mm_loadu_si128(LUT2.as_ptr() as *const __m128i);
                let lo_mask = _mm_set1_epi8(0x0F);
                // Four accumulators for the same reason as the f32 kernel: one
                // chained FMA per group would be latency-bound, not throughput.
                let mut acc = [_mm256_setzero_ps(); 4];

                for (gi, &sb) in scales_row.iter().enumerate() {
                    let base = gi * GROUP;
                    let pk = _mm_loadu_si128(packed_row.as_ptr().add(base / 2) as *const __m128i);
                    let lo_n = _mm_and_si128(pk, lo_mask);
                    let hi_n = _mm_and_si128(_mm_srli_epi16::<4>(pk), lo_mask);
                    let lo_v = _mm_shuffle_epi8(lut, lo_n);
                    let hi_v = _mm_shuffle_epi8(lut, hi_n);
                    // Interleave back to ascending column order, then join the
                    // two halves: 32 weights as bytes, ready to multiply.
                    let w = _mm256_set_m128i(
                        _mm_unpackhi_epi8(lo_v, hi_v),
                        _mm_unpacklo_epi8(lo_v, hi_v),
                    );
                    let xq = _mm256_loadu_si256(q.as_ptr().add(base) as *const __m256i);

                    let prep = $ops;
                    let (a, b) = prep(w, xq);
                    let dot = $dp(_mm256_setzero_si256(), a, b);

                    let s = _mm256_set1_ps(e8m0_to_f32(sb) * 0.5 * *qd.get_unchecked(gi));
                    let l = gi & 3;
                    acc[l] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), s, acc[l]);
                }
                hsum256(_mm256_add_ps(
                    _mm256_add_ps(acc[0], acc[1]),
                    _mm256_add_ps(acc[2], acc[3]),
                ))
            }
        };
    }

    vnni_row_dot!(
        dequant_row_dot_avxvnni,
        "avx2,fma,avxvnni",
        _mm256_dpbusd_avx_epi32,
        unsigned
    );
    vnni_row_dot!(
        dequant_row_dot_avx512vnni,
        "avx2,fma,avx512vnni,avx512vl",
        _mm256_dpbusd_epi32,
        unsigned
    );
    vnni_row_dot!(
        dequant_row_dot_avxvnniint8,
        "avx2,fma,avxvnniint8",
        _mm256_dpbssd_epi32,
        signed
    );

    /// Decode one packed weight row to `in_dim` f32, WITHOUT applying scales.
    ///
    /// This is the half of [`dequant_row_dot_avx2`] that does not depend on `x`:
    /// `vpshufb` through the table, the interleave, and the `i8 -> i32 -> f32`
    /// widening. Several activation rows against the same expert can share it.
    ///
    /// The scales are deliberately NOT folded in here. Applying them to `x` per
    /// row, exactly as the single-row kernel does, is what keeps the batched
    /// result bit-identical to running the rows one at a time — see
    /// [`dot_decoded_avx2`]. It costs 4 multiplies per group per row, which is
    /// nothing next to the decode this hoists.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn decode_row_avx2(packed_row: &[u8], in_dim: usize, w: &mut [f32]) {
        assert_eq!(packed_row.len(), in_dim / 2);
        assert!(w.len() >= in_dim);

        let lut = _mm_loadu_si128(LUT2.as_ptr() as *const __m128i);
        let lo_mask = _mm_set1_epi8(0x0F);
        let wp = w.as_mut_ptr();

        let mut base = 0usize;
        while base < in_dim {
            let pk = _mm_loadu_si128(packed_row.as_ptr().add(base / 2) as *const __m128i);
            let lo_n = _mm_and_si128(pk, lo_mask);
            let hi_n = _mm_and_si128(_mm_srli_epi16::<4>(pk), lo_mask);
            let lo_v = _mm_shuffle_epi8(lut, lo_n);
            let hi_v = _mm_shuffle_epi8(lut, hi_n);
            let cols_0_15 = _mm_unpacklo_epi8(lo_v, hi_v);
            let cols_16_31 = _mm_unpackhi_epi8(lo_v, hi_v);

            _mm256_storeu_ps(
                wp.add(base),
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(cols_0_15)),
            );
            _mm256_storeu_ps(
                wp.add(base + 8),
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(cols_0_15))),
            );
            _mm256_storeu_ps(
                wp.add(base + 16),
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(cols_16_31)),
            );
            _mm256_storeu_ps(
                wp.add(base + 24),
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(cols_16_31))),
            );
            base += GROUP;
        }
    }

    /// Dot an already-decoded weight row against one activation row.
    ///
    /// Bit-identical to [`dequant_row_dot_avx2`] by construction: same four
    /// accumulators, same lane-to-column assignment, same group order, same
    /// `x * s_half` scaling, same pairwise reduction. The only difference is
    /// that `w` arrives from a buffer instead of from `vpshufb`, and the decode
    /// produces exactly the integers `vpshufb` would have.
    ///
    /// That exactness is the point. Prefill batches rows and decode does not,
    /// and the prefix cache hands prefill's state to decode — so the two paths
    /// have to agree to the last bit, not merely to a tolerance.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_decoded_avx2(w: &[f32], scales_row: &[u8], x: &[f32], in_dim: usize) -> f32 {
        assert!(w.len() >= in_dim);
        assert_eq!(scales_row.len(), in_dim / GROUP);
        assert_eq!(x.len(), in_dim);

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for (gi, &sb) in scales_row.iter().enumerate() {
            let base = gi * GROUP;
            let wp = w.as_ptr().add(base);
            let xp = x.as_ptr().add(base);
            let s_half = _mm256_set1_ps(e8m0_to_f32(sb) * 0.5);

            let x0 = _mm256_mul_ps(_mm256_loadu_ps(xp), s_half);
            let x1 = _mm256_mul_ps(_mm256_loadu_ps(xp.add(8)), s_half);
            let x2 = _mm256_mul_ps(_mm256_loadu_ps(xp.add(16)), s_half);
            let x3 = _mm256_mul_ps(_mm256_loadu_ps(xp.add(24)), s_half);

            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(wp), x0, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(8)), x1, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(16)), x2, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(24)), x3, acc3);
        }
        hsum256(_mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        ))
    }
}

/// Which `vpdpbusd` encoding this CPU has, if any.
///
/// Checked once. AVX-VNNI is the AI-PC form and is preferred; AVX512-VNNI is
/// what the older server parts expose and exists here so the kernel can be
/// verified off-target.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq)]
enum VnniKind {
    None,
    /// `vpdpbssd`, signed x signed — no sign trick. Lunar Lake has this.
    AvxInt8,
    /// `vpdpbusd`, VEX encoding.
    Avx,
    /// `vpdpbusd`, EVEX. Older server parts; lets the kernel be verified
    /// off-target, since no AI-PC has it.
    Avx512,
}

#[cfg(target_arch = "x86_64")]
fn vnni_kind() -> VnniKind {
    use std::sync::OnceLock;
    static K: OnceLock<VnniKind> = OnceLock::new();
    *K.get_or_init(|| {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            VnniKind::None
        } else if is_x86_feature_detected!("avxvnniint8") {
            // Preferred where present: signed x signed removes two vpsignb per
            // group of 32 columns.
            VnniKind::AvxInt8
        } else if is_x86_feature_detected!("avxvnni") {
            VnniKind::Avx
        } else if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl") {
            VnniKind::Avx512
        } else {
            VnniKind::None
        }
    })
}

/// `y = W x` for a packed section. `data` is the section's bytes.
pub fn gemv(data: &[u8], out_dim: usize, in_dim: usize, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(data.len(), section_bytes(out_dim, in_dim));
    debug_assert_eq!(y.len(), out_dim);
    let nib_bytes = out_dim * in_dim / 2;
    let (nibs, scales) = data.split_at(nib_bytes);
    let row_nibs = in_dim / 2;
    let row_scales = in_dim / GROUP;

    #[cfg(target_arch = "x86_64")]
    {
        let kind = vnni_kind();
        if vnni_enabled() && simd_enabled() && kind != VnniKind::None && in_dim % GROUP == 0 {
            // Quantise once for the whole section: `x` is shared by every output
            // row, so this cost is amortised over `out_dim` dots (3072 of them).
            let mut q = vec![0i8; in_dim];
            let mut qd = vec![0.0f32; in_dim / GROUP];
            quantize_activations(x, in_dim, &mut q, &mut qd);
            for o in 0..out_dim {
                let pk = &nibs[o * row_nibs..(o + 1) * row_nibs];
                let sc = &scales[o * row_scales..(o + 1) * row_scales];
                // SAFETY: `vnni_kind` feature-detected each branch's encoding,
                // and the kernel re-checks every slice length with a hard assert.
                y[o] = unsafe {
                    match kind {
                        VnniKind::AvxInt8 => {
                            avx2::dequant_row_dot_avxvnniint8(pk, sc, &q, &qd, in_dim)
                        }
                        VnniKind::Avx => avx2::dequant_row_dot_avxvnni(pk, sc, &q, &qd, in_dim),
                        _ => avx2::dequant_row_dot_avx512vnni(pk, sc, &q, &qd, in_dim),
                    }
                };
            }
            return;
        }
    }

    for o in 0..out_dim {
        y[o] = dequant_row_dot(
            &nibs[o * row_nibs..(o + 1) * row_nibs],
            &scales[o * row_scales..(o + 1) * row_scales],
            x,
            in_dim,
        );
    }
}

/// `CASCADIA_K3_GEMM=0` forces the per-row `gemv` path, for A/B measurement.
fn gemm_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CASCADIA_K3_GEMM").as_deref() != Ok("0"))
}

/// `Y = X Wᵀ` for a packed section: `nrows` activation rows share one decode.
///
/// `xs` is `[nrows * in_dim]` and `ys` is `[nrows * out_dim]`, both row-major.
///
/// Prefill routes several tokens to the same expert, and calling [`gemv`] once
/// per token re-decodes the entire section every time. The decode does not
/// depend on `x`, so this hoists it: one pass over the weights serves up to
/// [`ROW_BLOCK`] rows.
///
/// Bit-identical to `nrows` separate [`gemv`] calls, not merely close: the
/// per-row arithmetic is the same sequence in the same order, and the hoisted
/// decode reproduces exactly the integers the fused kernel would have produced.
/// Prefill batches and decode does not, so the two must not drift.
pub fn gemm(data: &[u8], out_dim: usize, in_dim: usize, xs: &[f32], nrows: usize, ys: &mut [f32]) {
    debug_assert_eq!(data.len(), section_bytes(out_dim, in_dim));
    debug_assert_eq!(xs.len(), nrows * in_dim);
    debug_assert_eq!(ys.len(), nrows * out_dim);

    #[cfg(target_arch = "x86_64")]
    {
        if gemm_enabled()
            && nrows > 1
            && simd_enabled()
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: features detected immediately above; the callee re-checks
            // every slice length it indexes with a hard assert.
            unsafe { gemm_avx2(data, out_dim, in_dim, xs, nrows, ys) };
            return;
        }
    }
    let _ = gemm_enabled();
    gemm_rowwise(data, out_dim, in_dim, xs, nrows, ys)
}

/// Reference `gemm`: one [`gemv`] per row. Also the fallback off x86_64.
pub fn gemm_rowwise(
    data: &[u8],
    out_dim: usize,
    in_dim: usize,
    xs: &[f32],
    nrows: usize,
    ys: &mut [f32],
) {
    for r in 0..nrows {
        gemv(
            data,
            out_dim,
            in_dim,
            &xs[r * in_dim..(r + 1) * in_dim],
            &mut ys[r * out_dim..(r + 1) * out_dim],
        );
    }
}

/// Decode-once driver: one pass over the weights serves every row.
///
/// Output rows outer, activation rows inner. Each weight row is decoded a single
/// time into `wbuf` and then dotted against all `nrows` activations, so the
/// decode cost falls by a factor of `nrows` while the per-row arithmetic is
/// untouched. `wbuf` is `in_dim` f32 — 14 KB at K3's widest — so it stays in L1
/// across the inner loop, unlike the packed section it came from.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_avx2(
    data: &[u8],
    out_dim: usize,
    in_dim: usize,
    xs: &[f32],
    nrows: usize,
    ys: &mut [f32],
) {
    let nib_bytes = out_dim * in_dim / 2;
    let (nibs, scales) = data.split_at(nib_bytes);
    let row_nibs = in_dim / 2;
    let row_scales = in_dim / GROUP;
    let mut wbuf = vec![0.0f32; in_dim];

    for o in 0..out_dim {
        let pk = &nibs[o * row_nibs..(o + 1) * row_nibs];
        let sc = &scales[o * row_scales..(o + 1) * row_scales];
        avx2::decode_row_avx2(pk, in_dim, &mut wbuf);
        for r in 0..nrows {
            ys[r * out_dim + o] =
                avx2::dot_decoded_avx2(&wbuf, sc, &xs[r * in_dim..(r + 1) * in_dim], in_dim);
        }
    }
}

/// Dequantise a whole section to f32 `[out_dim * in_dim]` (tests / tooling).
pub fn dequant_section(data: &[u8], out_dim: usize, in_dim: usize, w: &mut [f32]) {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    let nib_bytes = out_dim * in_dim / 2;
    let (nibs, scales) = data.split_at(nib_bytes);
    for o in 0..out_dim {
        for c in 0..in_dim {
            let b = nibs[(o * in_dim + c) / 2];
            let n = if c % 2 == 0 { b & 0x0F } else { b >> 4 };
            let s = scales[o * (in_dim / GROUP) + c / GROUP];
            w[o * in_dim + c] = decode_nibble(n) * e8m0_to_f32(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nibble_grid_and_sign() {
        assert_eq!(decode_nibble(0x00), 0.0);
        assert_eq!(decode_nibble(0x07), 6.0);
        assert_eq!(decode_nibble(0x0F), -6.0);
        assert_eq!(decode_nibble(0x02), 1.0);
        assert_eq!(decode_nibble(0x0A), -1.0);
    }

    #[test]
    fn e8m0_unit_scale() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(128), 2.0);
        assert_eq!(e8m0_to_f32(126), 0.5);
    }

    #[test]
    fn section_sizes_match_the_exporter() {
        // 32x32 -> 512 nibble bytes + 32 scale bytes
        assert_eq!(section_bytes(32, 32), 544);
        // the tiny export writes 3 * 544 = 1632 bytes per expert
        assert_eq!(expert_bytes(32, 32), 1632);
    }

    /// Cheap deterministic byte stream — no dev-dependency on `rand`.
    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    fn random_row(in_dim: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
        let mut s = seed;
        let packed: Vec<u8> = (0..in_dim / 2).map(|_| lcg(&mut s) as u8).collect();
        // spread the exponent so a scale-indexing bug cannot hide
        let scales: Vec<u8> = (0..in_dim / GROUP)
            .map(|_| (124 + (lcg(&mut s) % 7)) as u8)
            .collect();
        let x: Vec<f32> = (0..in_dim)
            .map(|_| (lcg(&mut s) as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect();
        (packed, scales, x)
    }

    /// Magnitude the sum accumulated, so the tolerance tracks cancellation.
    fn accumulated_magnitude(packed: &[u8], scales: &[u8], x: &[f32], in_dim: usize) -> f32 {
        let mut m = 0.0f32;
        for c in 0..in_dim {
            let b = packed[c / 2];
            let n = if c % 2 == 0 { b & 0x0F } else { b >> 4 };
            m += (decode_nibble(n) * e8m0_to_f32(scales[c / GROUP]) * x[c]).abs();
        }
        m
    }

    #[test]
    fn simd_row_dot_matches_scalar() {
        // SIMD reassociates the sum, so f32 results differ in the last ulps;
        // the bar is relative to the accumulated magnitude, not equality.
        for (i, &in_dim) in [32usize, 64, 3072, 3584].iter().enumerate() {
            let (packed, scales, x) = random_row(in_dim, 0x5eed_0000 + i as u64);
            let got = dequant_row_dot(&packed, &scales, &x, in_dim);
            let want = dequant_row_dot_scalar(&packed, &scales, &x, in_dim);
            let mag = accumulated_magnitude(&packed, &scales, &x, in_dim);
            assert!(
                (got - want).abs() <= 1e-6 * mag,
                "in_dim {in_dim}: got {got} want {want} mag {mag}"
            );
        }
    }

    #[test]
    fn simd_covers_every_nibble_value() {
        // Two groups whose 64 columns walk 0x00..=0x0F four times, so the zero
        // entry, the whole positive half and the whole negative half all run.
        let in_dim = 64usize;
        let packed: Vec<u8> = (0..in_dim / 2)
            .map(|j| {
                let lo = (2 * j % 16) as u8;
                let hi = ((2 * j + 1) % 16) as u8;
                lo | (hi << 4)
            })
            .collect();
        // distinct scales per group
        let scales = vec![127u8, 130u8];
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.37).cos()).collect();

        let got = dequant_row_dot(&packed, &scales, &x, in_dim);
        let want = dequant_row_dot_scalar(&packed, &scales, &x, in_dim);
        let mag = accumulated_magnitude(&packed, &scales, &x, in_dim);
        assert!((got - want).abs() <= 1e-6 * mag, "got {got} want {want}");

        // and every nibble really did appear
        let mut seen = [false; 16];
        for c in 0..in_dim {
            let b = packed[c / 2];
            let n = if c % 2 == 0 { b & 0x0F } else { b >> 4 };
            seen[n as usize] = true;
        }
        assert!(seen.iter().all(|&v| v), "not all 16 nibbles exercised");
    }

    #[test]
    fn simd_honours_per_group_scales() {
        // Every group gets a different exponent; identical nibbles everywhere,
        // so the ONLY thing distinguishing groups is the scale index.
        let in_dim = 256usize;
        let packed = vec![0x27u8; in_dim / 2]; // low = 7 (+6.0), high = 2 (+1.0)
        let scales: Vec<u8> = (0..in_dim / GROUP).map(|g| (120 + g * 2) as u8).collect();
        let x = vec![1.0f32; in_dim];

        let got = dequant_row_dot(&packed, &scales, &x, in_dim);
        let want = dequant_row_dot_scalar(&packed, &scales, &x, in_dim);
        let mag = accumulated_magnitude(&packed, &scales, &x, in_dim);
        assert!((got - want).abs() <= 1e-6 * mag, "got {got} want {want}");

        // closed form: each group contributes 16*(6.0 + 1.0) * 2^(sb-127)
        let closed: f32 = scales.iter().map(|&sb| 112.0 * e8m0_to_f32(sb)).sum();
        assert!(
            (got - closed).abs() <= 1e-6 * mag,
            "got {got} closed {closed}"
        );
    }

    /// Scalar reference `gemv`, mirroring [`gemv`] but pinned to the scalar row dot.
    fn gemv_scalar(data: &[u8], out_dim: usize, in_dim: usize, x: &[f32], y: &mut [f32]) {
        let nib_bytes = out_dim * in_dim / 2;
        let (nibs, scales) = data.split_at(nib_bytes);
        let (row_nibs, row_scales) = (in_dim / 2, in_dim / GROUP);
        for o in 0..out_dim {
            y[o] = dequant_row_dot_scalar(
                &nibs[o * row_nibs..(o + 1) * row_nibs],
                &scales[o * row_scales..(o + 1) * row_scales],
                x,
                in_dim,
            );
        }
    }

    fn random_section(out_dim: usize, in_dim: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
        let mut s = seed;
        let nib = out_dim * in_dim / 2;
        let mut data = vec![0u8; section_bytes(out_dim, in_dim)];
        for b in data[..nib].iter_mut() {
            *b = lcg(&mut s) as u8;
        }
        for b in data[nib..].iter_mut() {
            *b = (124 + (lcg(&mut s) % 7)) as u8;
        }
        let x: Vec<f32> = (0..in_dim)
            .map(|_| (lcg(&mut s) as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect();
        (data, x)
    }

    #[test]
    fn simd_gemv_matches_scalar_gemv() {
        let (out_dim, in_dim) = (17usize, 3072usize); // odd row count on purpose
        let (data, x) = random_section(out_dim, in_dim, 0xabcd_1234);
        let mut got = vec![0.0f32; out_dim];
        let mut want = vec![0.0f32; out_dim];
        gemv(&data, out_dim, in_dim, &x, &mut got);
        gemv_scalar(&data, out_dim, in_dim, &x, &mut want);

        let nib = out_dim * in_dim / 2;
        let (nibs, scales) = data.split_at(nib);
        for o in 0..out_dim {
            let mag = accumulated_magnitude(
                &nibs[o * (in_dim / 2)..],
                &scales[o * (in_dim / GROUP)..],
                &x,
                in_dim,
            );
            assert!(
                (got[o] - want[o]).abs() <= 1e-6 * mag,
                "row {o}: got {} want {} mag {mag}",
                got[o],
                want[o]
            );
        }
    }

    /// `gemm` must be BIT-IDENTICAL to the same rows run one at a time.
    ///
    /// Not a tolerance: the hoisted decode is exact and the per-row arithmetic
    /// is unchanged, so any difference at all means the batched kernel is doing
    /// something the single-row one is not. Prefill batches, decode does not,
    /// and the prefix cache feeds one to the other — drift here is a real bug,
    /// so the test is written to catch it rather than to absorb it.
    ///
    /// Row counts 1..=9 include the boundaries a batched kernel gets wrong.
    #[test]
    fn gemm_is_bit_identical_to_rowwise_gemv() {
        let (out_dim, in_dim) = (13usize, 3072usize); // odd row count on purpose
        let (data, _) = random_section(out_dim, in_dim, 0x3333_7777);
        for nrows in 1..=9usize {
            let mut s = 0xfeed_0000 + nrows as u64;
            let xs: Vec<f32> = (0..nrows * in_dim)
                .map(|_| (lcg(&mut s) as f32 / u32::MAX as f32) * 2.0 - 1.0)
                .collect();
            let mut got = vec![0.0f32; nrows * out_dim];
            let mut want = vec![0.0f32; nrows * out_dim];
            gemm(&data, out_dim, in_dim, &xs, nrows, &mut got);
            gemm_rowwise(&data, out_dim, in_dim, &xs, nrows, &mut want);
            assert_eq!(got, want, "nrows {nrows}: batched gemm drifted from gemv");
        }
    }

    /// Each row must get ITS OWN activations.
    ///
    /// The equality check above would still pass if the kernel broadcast row 0
    /// to every output row and every row happened to hold the same activations.
    /// Feeding one nonzero row among zeros pins the mapping: every other output
    /// row must be exactly zero, and the live one must match `gemv`.
    #[test]
    fn gemm_does_not_cross_rows() {
        let (out_dim, in_dim) = (7usize, 1024usize);
        let (data, x1) = random_section(out_dim, in_dim, 0x9999_0001);
        let nrows = 6usize;
        for live in 0..nrows {
            let mut xs = vec![0.0f32; nrows * in_dim];
            xs[live * in_dim..(live + 1) * in_dim].copy_from_slice(&x1);
            let mut got = vec![0.0f32; nrows * out_dim];
            gemm(&data, out_dim, in_dim, &xs, nrows, &mut got);

            let mut want = vec![0.0f32; out_dim];
            gemv(&data, out_dim, in_dim, &x1, &mut want);
            for r in 0..nrows {
                for o in 0..out_dim {
                    let g = got[r * out_dim + o];
                    if r == live {
                        assert_eq!(g, want[o], "live row {live} col {o}");
                    } else {
                        assert_eq!(g, 0.0, "row {r} should be zero, live row is {live}");
                    }
                }
            }
        }
    }

    /// Quantise/dequantise must round-trip to within half a step.
    #[test]
    fn activation_quantization_round_trips() {
        let in_dim = 256usize;
        let mut s = 0x1234_5678u64;
        let x: Vec<f32> = (0..in_dim)
            .map(|_| (lcg(&mut s) as f32 / u32::MAX as f32) * 20.0 - 10.0)
            .collect();
        let mut q = vec![0i8; in_dim];
        let mut qd = vec![0.0f32; in_dim / GROUP];
        quantize_activations(&x, in_dim, &mut q, &mut qd);

        for ((gi, &d), (xg, qg)) in qd
            .iter()
            .enumerate()
            .zip(x.chunks_exact(GROUP).zip(q.chunks_exact(GROUP)))
        {
            for (j, (&xv, &qv)) in xg.iter().zip(qg).enumerate() {
                let back = qv as f32 * d;
                assert!(
                    (back - xv).abs() <= 0.5 * d + 1e-6,
                    "col {}: {xv} -> {back} (step {d})",
                    gi * GROUP + j
                );
            }
        }
    }

    /// An all-zero group must not produce NaN.
    ///
    /// `amax` is 0 there, so the step is 0 and the naive reciprocal is infinite;
    /// every quantised value would come out NaN and poison the whole dot.
    #[test]
    fn activation_quantization_survives_a_zero_group() {
        let in_dim = 64usize;
        let mut x = vec![0.0f32; in_dim];
        for (i, v) in x[32..].iter_mut().enumerate() {
            *v = (i as f32) * 0.25 - 4.0;
        }
        let mut q = vec![0i8; in_dim];
        let mut qd = vec![0.0f32; in_dim / GROUP];
        quantize_activations(&x, in_dim, &mut q, &mut qd);
        assert_eq!(qd[0], 0.0, "zero group should have a zero step");
        assert!(
            q[..32].iter().all(|&v| v == 0),
            "zero group must quantise to 0"
        );
        assert!(qd[1] > 0.0);
    }

    /// The VNNI kernel must agree with the exact int8 reference.
    ///
    /// Both do the same integer arithmetic, so any gap here is a SIMD bug — the
    /// sign trick, the column interleave, or the group-to-accumulator mapping —
    /// and NOT quantisation error, which both share. Off x86 this compares the
    /// scalar reference with itself and proves nothing; that is what the
    /// `vnni active` line in the bench is for.
    #[test]
    fn vnni_matches_int8_scalar_reference() {
        for &in_dim in &[64usize, 3072, 3584] {
            let (packed, scales, x) = random_row(in_dim, 0x7e57_0000 + in_dim as u64);
            let mut q = vec![0i8; in_dim];
            let mut qd = vec![0.0f32; in_dim / GROUP];
            quantize_activations(&x, in_dim, &mut q, &mut qd);
            let want = dequant_row_dot_q8_scalar(&packed, &scales, &q, &qd, in_dim);

            #[cfg(target_arch = "x86_64")]
            let got = match vnni_kind() {
                VnniKind::None => want,
                // SAFETY: detected immediately above.
                VnniKind::AvxInt8 => unsafe {
                    avx2::dequant_row_dot_avxvnniint8(&packed, &scales, &q, &qd, in_dim)
                },
                VnniKind::Avx => unsafe {
                    avx2::dequant_row_dot_avxvnni(&packed, &scales, &q, &qd, in_dim)
                },
                VnniKind::Avx512 => unsafe {
                    avx2::dequant_row_dot_avx512vnni(&packed, &scales, &q, &qd, in_dim)
                },
            };
            #[cfg(not(target_arch = "x86_64"))]
            let got = want;

            let mag = accumulated_magnitude(&packed, &scales, &x, in_dim);
            assert!(
                (got - want).abs() <= 1e-6 * mag,
                "in_dim {in_dim}: simd {got} vs scalar {want} (mag {mag})"
            );
        }
    }

    /// How much accuracy the int8 activation path actually costs.
    ///
    /// This is the number that decides whether `CASCADIA_K3_VNNI` is worth
    /// turning on, so it prints the measured error rather than only asserting a
    /// bound. The bound is deliberately loose — it exists to catch a broken
    /// kernel, not to certify the accuracy; read the printed figure for that.
    #[test]
    fn vnni_accuracy_vs_exact() {
        let mut worst = 0.0f64;
        for &in_dim in &[3072usize, 3584] {
            for seed in 0..8u64 {
                let (packed, scales, x) =
                    random_row(in_dim, 0xacc0_0000 + seed * 31 + in_dim as u64);
                let exact = dequant_row_dot_scalar(&packed, &scales, &x, in_dim);
                let mut q = vec![0i8; in_dim];
                let mut qd = vec![0.0f32; in_dim / GROUP];
                quantize_activations(&x, in_dim, &mut q, &mut qd);
                let q8 = dequant_row_dot_q8_scalar(&packed, &scales, &q, &qd, in_dim);

                let mag = accumulated_magnitude(&packed, &scales, &x, in_dim);
                let rel = ((q8 - exact).abs() / mag.max(1e-12)) as f64;
                worst = worst.max(rel);
            }
        }
        println!(
            "int8 activations: worst error {:.5}% of accumulated magnitude",
            worst * 100.0
        );
        assert!(
            worst < 0.02,
            "int8 activation error {worst} is far past quantisation noise — kernel bug, not rounding"
        );
    }

    /// f32 `gemv` vs the int8 VNNI `gemv`, at the real K3 expert dims.
    ///
    /// Needs a CPU with `vpdpbusd`; the printed `vnni` line says which encoding
    /// was used, or `None`, in which case both columns are the same f32 kernel
    /// and the ratio is meaningless. Run with
    /// `cargo test -p cascadia-engine-sparse-moe vnni_bench -- --nocapture`.
    #[test]
    fn vnni_bench() {
        use std::time::Instant;
        #[cfg(target_arch = "x86_64")]
        let kind = match vnni_kind() {
            VnniKind::AvxInt8 => "AVX-VNNI-INT8 (vpdpbssd)",
            VnniKind::Avx => "AVX-VNNI (vpdpbusd)",
            VnniKind::Avx512 => "AVX512-VNNI",
            VnniKind::None => "None",
        };
        #[cfg(not(target_arch = "x86_64"))]
        let kind = "None (not x86_64)";
        println!("vnni: {kind}");

        for &(out_dim, in_dim) in &[(3072usize, 3584usize), (3584usize, 3072usize)] {
            let (data, x) = random_section(out_dim, in_dim, 0x5a5a + out_dim as u64);
            let mut y = vec![0.0f32; out_dim];
            let reps = 5;

            // f32 path, reached directly so the env gate cannot confuse this.
            let nib = out_dim * in_dim / 2;
            let (nibs, scales) = data.split_at(nib);
            let f32_run = || {
                for o in 0..out_dim {
                    std::hint::black_box(dequant_row_dot(
                        &nibs[o * (in_dim / 2)..(o + 1) * (in_dim / 2)],
                        &scales[o * (in_dim / GROUP)..(o + 1) * (in_dim / GROUP)],
                        &x,
                        in_dim,
                    ));
                }
            };
            f32_run();

            let mut q = vec![0i8; in_dim];
            let mut qd = vec![0.0f32; in_dim / GROUP];
            #[cfg(target_arch = "x86_64")]
            let k = vnni_kind();
            let vnni_run = || {
                let mut q = q.clone();
                let mut qd = qd.clone();
                quantize_activations(&x, in_dim, &mut q, &mut qd);
                for o in 0..out_dim {
                    let pk = &nibs[o * (in_dim / 2)..(o + 1) * (in_dim / 2)];
                    let sc = &scales[o * (in_dim / GROUP)..(o + 1) * (in_dim / GROUP)];
                    #[cfg(target_arch = "x86_64")]
                    let v = unsafe {
                        match k {
                            VnniKind::AvxInt8 => {
                                avx2::dequant_row_dot_avxvnniint8(pk, sc, &q, &qd, in_dim)
                            }
                            VnniKind::Avx => avx2::dequant_row_dot_avxvnni(pk, sc, &q, &qd, in_dim),
                            VnniKind::Avx512 => {
                                avx2::dequant_row_dot_avx512vnni(pk, sc, &q, &qd, in_dim)
                            }
                            VnniKind::None => dequant_row_dot(pk, sc, &x, in_dim),
                        }
                    };
                    #[cfg(not(target_arch = "x86_64"))]
                    let v = dequant_row_dot(pk, sc, &x, in_dim);
                    std::hint::black_box(v);
                }
            };
            vnni_run();
            // Alternate and keep the best of each: timing one variant fully then
            // the other lets the clock ramp between them, and on a laptop part
            // that bias is larger than the effect being measured.
            let (mut base, mut vnni) = (f64::MAX, f64::MAX);
            for _ in 0..reps {
                let t = Instant::now();
                f32_run();
                base = base.min(t.elapsed().as_secs_f64());

                let t = Instant::now();
                vnni_run();
                vnni = vnni.min(t.elapsed().as_secs_f64());
            }

            quantize_activations(&x, in_dim, &mut q, &mut qd);
            let g = 2.0 * out_dim as f64 * in_dim as f64;
            println!(
                "{out_dim}x{in_dim}: f32 {:.3} ms ({:.1} GFLOP/s), vnni {:.3} ms ({:.1} GFLOP/s), {:.2}x",
                base * 1e3,
                g / base / 1e9,
                vnni * 1e3,
                g / vnni / 1e9,
                base / vnni
            );
            let _ = &mut y;
        }
    }

    /// What OUTLIERS do to the int8 activation path.
    ///
    /// `vnni_accuracy_vs_exact` uses uniform random activations and reports
    /// 0.027%. That number is not evidence about a real model. The quantiser
    /// takes an amax over each group of 32 and divides into 127 levels, so a
    /// single large value in a group costs every other value in it resolution —
    /// and LLM activations are known to carry exactly that kind of outlier.
    ///
    /// This sweeps one planted outlier per group from 1x to 1000x the rest and
    /// reports the resulting error, so the shape of the risk is on record before
    /// anyone argues about flipping `CASCADIA_K3_VNNI` on by default. It asserts
    /// only the mechanism (error grows with outlier ratio, stays finite); the
    /// verdict needs real activations, which needs the real model.
    #[test]
    fn vnni_accuracy_under_activation_outliers() {
        let in_dim = 3584usize;
        let (packed, scales, base_x) = random_row(in_dim, 0x0417_1ee7);

        println!("outlier x   worst-group err    dot err (% of |accumulated|)");
        let mut prev = 0.0f64;
        for &ratio in &[1.0f32, 10.0, 50.0, 100.0, 1000.0] {
            let mut x = base_x.clone();
            // One planted outlier per group of 32, at the same position each
            // time so the only variable is its magnitude.
            for g in x.chunks_mut(GROUP) {
                let m = g.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
                g[7] = m * ratio;
            }

            let exact = dequant_row_dot_scalar(&packed, &scales, &x, in_dim);
            let mut q = vec![0i8; in_dim];
            let mut qd = vec![0.0f32; in_dim / GROUP];
            quantize_activations(&x, in_dim, &mut q, &mut qd);
            let q8 = dequant_row_dot_q8_scalar(&packed, &scales, &q, &qd, in_dim);

            // Worst single-element reconstruction error, relative to the group's
            // step — this is what an outlier actually damages.
            let mut worst = 0.0f32;
            for (gi, (xg, qg)) in x.chunks_exact(GROUP).zip(q.chunks_exact(GROUP)).enumerate() {
                for (&xv, &qv) in xg.iter().zip(qg) {
                    worst = worst.max((qv as f32 * qd[gi] - xv).abs());
                }
            }
            let mag = accumulated_magnitude(&packed, &scales, &x, in_dim);
            let rel = ((q8 - exact).abs() / mag.max(1e-12)) as f64 * 100.0;
            println!("{ratio:>9.0}   {worst:>14.5}    {rel:>8.4}%");

            assert!(
                rel.is_finite(),
                "outlier {ratio}x produced a non-finite error"
            );
            prev = prev.max(rel);
        }
        assert!(
            prev < 5.0,
            "int8 activation error reached {prev}% even before real data — \
             the group-32 amax quantiser is not viable as written"
        );
    }

    /// Prints per-row `gemv` vs batched `gemm` at the real K3 expert dims.
    ///
    /// This is the measurement the batched path exists for; it needs an x86 host
    /// with AVX2, since everywhere else `gemm` is `gemm_rowwise` and both columns
    /// are the same code. Run with
    /// `cargo test -p cascadia-engine-sparse-moe gemm_bench -- --nocapture`.
    #[test]
    fn gemm_bench() {
        use std::time::Instant;
        #[cfg(target_arch = "x86_64")]
        let simd =
            simd_enabled() && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let simd = false;
        println!("avx2 gemm active: {simd} (false => both columns are gemm_rowwise)");

        let (out_dim, in_dim) = (3072usize, 3584usize);
        let (data, _) = random_section(out_dim, in_dim, 0x2468);
        let reps = 9;
        for &nrows in &[1usize, 2, 3, 4, 8, 16, 32, 64] {
            let mut s = 0xbead_0000 + nrows as u64;
            let xs: Vec<f32> = (0..nrows * in_dim)
                .map(|_| (lcg(&mut s) as f32 / u32::MAX as f32) * 2.0 - 1.0)
                .collect();
            let mut ys = vec![0.0f32; nrows * out_dim];

            // ALTERNATE the two variants and keep the best time for each.
            //
            // Timing A fully, then B, is wrong on a laptop-class part: the clock
            // ramps between them and whichever ran second wins. The `nrows = 1`
            // row is the control that exposed it — there both columns are the
            // SAME code (the batched path needs nrows > 1), and back-to-back
            // timing still reported the second one 5-14% faster, drifting run to
            // run. Alternating cancels the ramp; min-of-N rejects interference,
            // which one-sided noise otherwise adds to the mean.
            gemm_rowwise(&data, out_dim, in_dim, &xs, nrows, &mut ys);
            gemm(&data, out_dim, in_dim, &xs, nrows, &mut ys);
            let (mut rowwise, mut batched) = (f64::MAX, f64::MAX);
            for _ in 0..reps {
                let t = Instant::now();
                gemm_rowwise(&data, out_dim, in_dim, &xs, nrows, &mut ys);
                rowwise = rowwise.min(t.elapsed().as_secs_f64());

                let t = Instant::now();
                gemm(&data, out_dim, in_dim, &xs, nrows, &mut ys);
                batched = batched.min(t.elapsed().as_secs_f64());
            }

            let gflops = 2.0 * out_dim as f64 * in_dim as f64 * nrows as f64;
            println!(
                "nrows {nrows}: rowwise {:.3} ms ({:.1} GFLOP/s), batched {:.3} ms ({:.1} GFLOP/s), {:.2}x",
                rowwise * 1e3,
                gflops / rowwise / 1e9,
                batched * 1e3,
                gflops / batched / 1e9,
                rowwise / batched
            );
        }
    }

    /// Prints scalar vs dispatched `gemv` timing at the real K3 expert dims.
    /// Run with `cargo test -p cascadia-engine-sparse-moe simd_gemv_bench -- --nocapture`.
    /// On a non-x86 host both columns are the same scalar kernel.
    #[test]
    fn simd_gemv_bench() {
        use std::time::Instant;
        #[cfg(target_arch = "x86_64")]
        println!(
            "avx2 path active: {}",
            simd_enabled() && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        );
        #[cfg(not(target_arch = "x86_64"))]
        println!("avx2 path active: false (not x86_64)");
        for &(out_dim, in_dim) in &[(3072usize, 3584usize), (3584usize, 3072usize)] {
            let (data, x) = random_section(out_dim, in_dim, 0x1111 + out_dim as u64);
            let mut y = vec![0.0f32; out_dim];
            let reps = 3;

            gemv_scalar(&data, out_dim, in_dim, &x, &mut y); // warm the page cache
            let t0 = Instant::now();
            for _ in 0..reps {
                gemv_scalar(&data, out_dim, in_dim, &x, &mut y);
            }
            let scalar = t0.elapsed().as_secs_f64() / reps as f64;

            gemv(&data, out_dim, in_dim, &x, &mut y);
            let t1 = Instant::now();
            for _ in 0..reps {
                gemv(&data, out_dim, in_dim, &x, &mut y);
            }
            let simd = t1.elapsed().as_secs_f64() / reps as f64;

            println!(
                "gemv {out_dim}x{in_dim}: scalar {:.3} ms, dispatched {:.3} ms, {:.2}x",
                scalar * 1e3,
                simd * 1e3,
                scalar / simd
            );
        }
    }

    #[test]
    fn gemv_matches_explicit_dequant() {
        let (out_dim, in_dim) = (4usize, 64usize);
        let mut data = vec![0u8; section_bytes(out_dim, in_dim)];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i * 37 % 251) as u8;
        }
        // keep scales in a sane exponent range so the comparison is meaningful
        let nib = out_dim * in_dim / 2;
        for b in data[nib..].iter_mut() {
            *b = 127 - (*b % 3);
        }

        let mut w = vec![0.0f32; out_dim * in_dim];
        dequant_section(&data, out_dim, in_dim, &mut w);
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.13).sin()).collect();

        let mut y = vec![0.0f32; out_dim];
        gemv(&data, out_dim, in_dim, &x, &mut y);

        for o in 0..out_dim {
            let want: f32 = (0..in_dim).map(|c| w[o * in_dim + c] * x[c]).sum();
            assert!(
                (y[o] - want).abs() <= 1e-4 * want.abs().max(1.0),
                "row {o}: got {} want {want}",
                y[o]
            );
        }
    }
}
