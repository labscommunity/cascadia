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
#[inline]
pub fn e8m0_to_f32(b: u8) -> f32 {
    // exp2 via bit construction would overflow for b == 0; powi is fine here
    2.0f32.powi(b as i32 - 127)
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

/// Fused dequant + dot for one output row:
/// `sum_k decode(nibble_k) * 2^(scale(g(k)) - 127) * x[k]`.
///
/// `packed_row` is `in_dim/2` bytes, `scales_row` is `in_dim/GROUP` bytes.
pub fn dequant_row_dot(packed_row: &[u8], scales_row: &[u8], x: &[f32], in_dim: usize) -> f32 {
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

/// `y = W x` for a packed section. `data` is the section's bytes.
pub fn gemv(data: &[u8], out_dim: usize, in_dim: usize, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(data.len(), section_bytes(out_dim, in_dim));
    debug_assert_eq!(y.len(), out_dim);
    let nib_bytes = out_dim * in_dim / 2;
    let (nibs, scales) = data.split_at(nib_bytes);
    let row_nibs = in_dim / 2;
    let row_scales = in_dim / GROUP;
    for o in 0..out_dim {
        y[o] = dequant_row_dot(
            &nibs[o * row_nibs..(o + 1) * row_nibs],
            &scales[o * row_scales..(o + 1) * row_scales],
            x,
            in_dim,
        );
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
