//! Tensor helpers — bf16/f16 conversions, mask construction, accumulation.
//!
//! All tensors are stored as flat byte buffers (Vec<u8>) since the OV shim's
//! `Runtime::set_input` takes `&[u8]` and the dtype/shape separately. The
//! helpers below convert between numpy-style typed slices and those raw
//! buffers, doing the precision conversions OV won't do for us (notably
//! f32 -> bf16 because OV's CPU plugin demands bf16-typed bytes for the
//! shell's `x.1` and the experts' `x` inputs).

use half::{bf16, f16};

/// Build a causal mask of shape `[1, 1, seq, past + seq]`, f32, with 0
/// where the query can attend and -1e9 where it can't.
///
/// `-1e9` matches the convention the shells were exported with —
/// `-inf` triggered NaN paths in some attention kernels.
pub fn causal_mask_f32(seq: usize, past: usize) -> (Vec<f32>, [usize; 4]) {
    let full = past + seq;
    let mut m = vec![0.0f32; seq * full];
    for i in 0..seq {
        let j_max = past + i;
        if j_max + 1 < full {
            for j in (j_max + 1)..full {
                m[i * full + j] = -1.0e9;
            }
        }
    }
    (m, [1, 1, seq, full])
}

/// Convert f32 slice to bf16 raw bytes (little-endian, 2 bytes/element).
pub fn f32_to_bf16_bytes(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 2);
    for &v in src {
        let h = bf16::from_f32(v);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Convert bf16 raw bytes to a Vec<f32>.
pub fn bf16_bytes_to_f32(src: &[u8]) -> Vec<f32> {
    let n = src.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = src[i * 2];
        let hi = src[i * 2 + 1];
        let h = bf16::from_le_bytes([lo, hi]);
        out.push(h.to_f32());
    }
    out
}

/// Convert f16 raw bytes to a Vec<f32>.
pub fn f16_bytes_to_f32(src: &[u8]) -> Vec<f32> {
    let n = src.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = src[i * 2];
        let hi = src[i * 2 + 1];
        let h = f16::from_le_bytes([lo, hi]);
        out.push(h.to_f32());
    }
    out
}

/// Cast a &[f32] to its raw little-endian bytes. (On x86_64 + aarch64 this
/// is identical to `bytemuck::cast_slice` — we hand-roll to avoid the dep.)
pub fn f32_to_bytes(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 4);
    for &v in src {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Cast a &[i64] to its raw little-endian bytes.
pub fn i64_to_bytes(src: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 8);
    for &v in src {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Cast a raw bytes back to a Vec<i64>.
pub fn bytes_to_i64(src: &[u8]) -> Vec<i64> {
    let n = src.len() / 8;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&src[i * 8..i * 8 + 8]);
        out.push(i64::from_le_bytes(arr));
    }
    out
}

/// Concatenate two raw byte buffers along an interior axis. Used to grow
/// past_k / past_v along the sequence axis (axis=2 of [batch, heads, seq,
/// head_dim]). Returns the new bytes + new shape.
///
/// Both buffers must be the same dtype (passed via `bytes_per_elem`) and
/// must share all dims except `axis`.
pub fn concat_along_axis(
    a_bytes: &[u8],
    a_shape: &[usize],
    b_bytes: &[u8],
    b_shape: &[usize],
    axis: usize,
    bytes_per_elem: usize,
) -> (Vec<u8>, Vec<usize>) {
    assert_eq!(a_shape.len(), b_shape.len());
    assert!(axis < a_shape.len());
    for i in 0..a_shape.len() {
        if i != axis {
            assert_eq!(a_shape[i], b_shape[i], "dim {} mismatch", i);
        }
    }
    let mut new_shape = a_shape.to_vec();
    new_shape[axis] = a_shape[axis] + b_shape[axis];

    let outer: usize = a_shape[..axis].iter().product();
    let inner: usize = a_shape[axis + 1..].iter().product::<usize>() * bytes_per_elem;
    let a_slab = a_shape[axis] * inner;
    let b_slab = b_shape[axis] * inner;
    let total = outer * (a_slab + b_slab);
    let mut out = vec![0u8; total];
    for o in 0..outer {
        let src_a = &a_bytes[o * a_slab..(o + 1) * a_slab];
        let src_b = &b_bytes[o * b_slab..(o + 1) * b_slab];
        let dst_off = o * (a_slab + b_slab);
        out[dst_off..dst_off + a_slab].copy_from_slice(src_a);
        out[dst_off + a_slab..dst_off + a_slab + b_slab].copy_from_slice(src_b);
    }
    (out, new_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_seq1_past0() {
        let (m, shape) = causal_mask_f32(1, 0);
        assert_eq!(shape, [1, 1, 1, 1]);
        assert_eq!(m, vec![0.0]);
    }

    #[test]
    fn causal_mask_seq1_past5() {
        let (m, shape) = causal_mask_f32(1, 5);
        assert_eq!(shape, [1, 1, 1, 6]);
        assert_eq!(m, vec![0.0; 6]); // 1 query at position 5 attends to all 6 (0..5)
    }

    #[test]
    fn causal_mask_seq3_past0() {
        let (m, shape) = causal_mask_f32(3, 0);
        assert_eq!(shape, [1, 1, 3, 3]);
        // Row 0: [0, -1e9, -1e9]
        // Row 1: [0, 0, -1e9]
        // Row 2: [0, 0, 0]
        assert_eq!(m[0], 0.0);
        assert_eq!(m[1], -1.0e9);
        assert_eq!(m[2], -1.0e9);
        assert_eq!(m[3], 0.0);
        assert_eq!(m[4], 0.0);
        assert_eq!(m[5], -1.0e9);
        assert_eq!(m[6], 0.0);
        assert_eq!(m[7], 0.0);
        assert_eq!(m[8], 0.0);
    }

    #[test]
    fn bf16_roundtrip() {
        let xs = vec![0.0_f32, 1.0, -1.5, 3.125, -42.7];
        let bytes = f32_to_bf16_bytes(&xs);
        assert_eq!(bytes.len(), xs.len() * 2);
        let back = bf16_bytes_to_f32(&bytes);
        // bf16 has ~7-bit mantissa; allow generous tolerance.
        for (a, b) in xs.iter().zip(back.iter()) {
            assert!(
                (a - b).abs() <= a.abs() * 0.01 + 0.01,
                "roundtrip {} -> {}",
                a,
                b
            );
        }
    }

    #[test]
    fn concat_axis2() {
        // a: [1, 2, 2, 1] of i8 — values 0,1,2,3
        // b: [1, 2, 3, 1] of i8 — values 10,11,12,13,14,15
        let a = vec![0u8, 1, 2, 3];
        let b = vec![10u8, 11, 12, 13, 14, 15];
        let (out, shape) = concat_along_axis(&a, &[1, 2, 2, 1], &b, &[1, 2, 3, 1], 2, 1);
        // Expected layout: [batch=1][heads=2][seq=5][head=1]
        // head 0: a[0,1] + b[0,1,2] = 0,1,10,11,12
        // head 1: a[2,3] + b[3,4,5] = 2,3,13,14,15
        assert_eq!(shape, vec![1, 2, 5, 1]);
        assert_eq!(out, vec![0, 1, 10, 11, 12, 2, 3, 13, 14, 15]);
    }
}
