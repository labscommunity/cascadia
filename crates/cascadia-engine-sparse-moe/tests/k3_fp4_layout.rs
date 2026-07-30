//! Pin the fp4 expert byte layout against the format's own reference packer.
//!
//! The real export does NOT requantise: it copies upstream's `weight_packed` and
//! `weight_scale` bytes straight into the expert bin. That is only correct if our
//! reader agrees with upstream about nibble order, sign bit and scale encoding —
//! and every other fp4 test in the tree packs with OUR packer on both sides, so
//! they would all pass even if that agreement were wrong.
//!
//! The bytes below were produced by `compressed_tensors`' own
//! `pack_fp4_to_uint8` / `compress_mx_scale` (the `mxfp4-pack-quantized` writer),
//! and `WANT` by its `unpack_fp4_from_uint8` / `decompress_mx_scale`. A nibble
//! order flip changes ~92% of the decoded elements, so this is not a weak check.

use cascadia_engine_sparse_moe::k3::expert_fp4::{decode_nibble, e8m0_to_f32, gemv, section_bytes};

const OUT: usize = 8;
const IN: usize = 64;

/// `[OUT*IN/2]` packed nibbles then `[OUT*IN/32]` E8M0 scale bytes.
const SECTION: [u8; 272] = [
    0x4e, 0xf7, 0x3a, 0xfe, 0xd7, 0xdf, 0x76, 0x73, 0xcf, 0xc7, 0x62, 0xe7, 0xaa, 0xcf, 0x6f, 0xff,
    0x88, 0x71, 0xc5, 0xbc, 0x6e, 0xd6, 0x99, 0x13, 0x99, 0xa8, 0xdf, 0x5a, 0x06, 0xfa, 0xa5, 0x4a,
    0x11, 0x4b, 0xed, 0xbb, 0x5c, 0x93, 0xac, 0x3d, 0x2f, 0x23, 0xd9, 0x56, 0x9a, 0x54, 0x1c, 0x63,
    0x6d, 0xf7, 0x81, 0x5c, 0x74, 0x7f, 0xa5, 0xf5, 0xcf, 0x16, 0xdd, 0x56, 0x7e, 0x5f, 0x5c, 0x73,
    0x15, 0xe5, 0xeb, 0xef, 0xf7, 0x5c, 0xf1, 0x7f, 0xa3, 0xf7, 0xd7, 0x16, 0xd1, 0x1c, 0xf6, 0x1d,
    0xd4, 0x18, 0xa6, 0xc1, 0xcc, 0x1c, 0x9c, 0x15, 0xcb, 0x14, 0xc4, 0x62, 0x42, 0xc6, 0xef, 0x8d,
    0x1a, 0x4d, 0xc6, 0xae, 0x8f, 0x44, 0x34, 0x90, 0xd3, 0xcc, 0xfc, 0xd2, 0xfd, 0x5b, 0xca, 0x86,
    0x65, 0xb7, 0xd8, 0xd4, 0xaf, 0xe9, 0x4c, 0x7b, 0x6e, 0xdc, 0x7e, 0x16, 0xb6, 0x1c, 0xe6, 0xb1,
    0xe0, 0x3b, 0x74, 0x94, 0xde, 0x4a, 0xc1, 0x14, 0x69, 0xc1, 0xc3, 0x76, 0xdd, 0x02, 0x02, 0x14,
    0x6c, 0x49, 0xf4, 0x7d, 0x97, 0xe7, 0xca, 0xf7, 0xe6, 0x44, 0x47, 0x17, 0xd6, 0x9e, 0xb5, 0x2f,
    0x1e, 0x08, 0x34, 0xcc, 0x51, 0xae, 0x11, 0xc4, 0xd2, 0x82, 0x65, 0xc9, 0xa2, 0xbd, 0xd7, 0x27,
    0xf7, 0xe2, 0xfc, 0x57, 0xab, 0x82, 0x7f, 0xf1, 0x53, 0x67, 0xaf, 0x0f, 0xa6, 0xf9, 0x77, 0xff,
    0x83, 0xe5, 0xbe, 0x1d, 0x65, 0x2b, 0x64, 0xa1, 0x1d, 0x1e, 0x75, 0x59, 0xe8, 0x1d, 0xe5, 0x34,
    0x09, 0x77, 0x74, 0x67, 0x4b, 0xbb, 0xb5, 0x76, 0xd1, 0x47, 0x6f, 0x7d, 0x7f, 0x77, 0x9f, 0x69,
    0xfc, 0x76, 0x77, 0xf7, 0xe6, 0x86, 0xb7, 0xfd, 0xe4, 0xae, 0xf7, 0xb5, 0xf2, 0xf5, 0xfa, 0x47,
    0x7f, 0x2f, 0xef, 0x77, 0x36, 0x96, 0x11, 0xf5, 0xaf, 0x45, 0x46, 0x67, 0xcf, 0xfa, 0x31, 0x4e,
    0x7c, 0x7d, 0x7d, 0x7c, 0x7c, 0x7d, 0x7d, 0x7c, 0x7d, 0x7c, 0x7d, 0x7c, 0x7d, 0x7c, 0x7c, 0x7c,
];

const X: [f32; IN] = [
    3.81143272e-01,
    3.95260632e-01,
    -1.54234123e+00,
    -1.61337924e+00,
    -5.66966891e-01,
    -1.10870607e-01,
    -7.70224392e-01,
    -7.36351430e-01,
    1.76931822e+00,
    -5.27509511e-01,
    -1.50021279e+00,
    6.71278477e-01,
    1.22827068e-01,
    -4.10987586e-01,
    -2.56752014e-01,
    3.01810294e-01,
    -2.09595013e+00,
    1.72927105e+00,
    7.94396758e-01,
    -1.88319135e+00,
    8.33974481e-02,
    4.72546488e-01,
    1.06485903e-01,
    1.69139993e+00,
    -1.17084885e+00,
    8.59466314e-01,
    -9.49013352e-01,
    -4.27940190e-01,
    9.94468451e-01,
    1.83258846e-01,
    1.75590730e+00,
    -3.78912657e-01,
    -4.26953971e-01,
    -5.63495979e-02,
    -5.69425881e-01,
    1.51760614e+00,
    1.66358724e-01,
    -4.43566680e-01,
    1.77433896e+00,
    4.93567497e-01,
    1.60446644e+00,
    -8.07617962e-01,
    2.13949656e+00,
    -1.72252393e+00,
    1.65319943e+00,
    -1.00936711e+00,
    1.94200981e+00,
    8.57193530e-01,
    -5.50110042e-01,
    -1.55378747e+00,
    -1.36821598e-01,
    -5.61074674e-01,
    6.27740100e-02,
    2.90699899e-01,
    2.94897139e-01,
    -4.38140407e-02,
    1.08982337e+00,
    -1.44116127e+00,
    -4.79233444e-01,
    7.74489164e-01,
    1.62293661e+00,
    -9.86147702e-01,
    1.37075737e-01,
    -1.16683495e+00,
];

/// `W x` from the reference dequantisation, accumulated in f64.
const WANT: [f32; OUT] = [
    8.26859854e+00,
    -2.85726082e+00,
    2.86918947e+00,
    -3.59081435e+00,
    1.24872708e+01,
    1.54950227e+00,
    7.35099635e+00,
    -5.10949736e-01,
];

#[test]
fn fp4_section_layout_matches_reference_packer() {
    assert_eq!(SECTION.len(), section_bytes(OUT, IN), "section size");
    let mut got = [0.0f32; OUT];
    gemv(&SECTION, OUT, IN, &X, &mut got);
    for (o, (&g, &w)) in got.iter().zip(WANT.iter()).enumerate() {
        let tol = 1e-4 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "row {o}: got {g} want {w} (tol {tol})"
        );
    }
}

#[test]
fn nibble_order_flip_is_detected() {
    // Guards the test above: swap every byte's nibbles and the result must move.
    let mut swapped = SECTION;
    for b in swapped[..OUT * IN / 2].iter_mut() {
        *b = (*b >> 4) | ((*b & 0x0F) << 4);
    }
    let mut got = [0.0f32; OUT];
    gemv(&swapped, OUT, IN, &X, &mut got);
    let moved = got
        .iter()
        .zip(WANT.iter())
        .filter(|(g, w)| (**g - **w).abs() > 1e-3)
        .count();
    assert!(moved >= OUT - 1, "only {moved}/{OUT} rows moved");
}

#[test]
fn nibble_and_scale_encoding_match_reference() {
    // e2m1: magnitude in bits 0..2, sign in bit 3.
    const GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for (i, &v) in GRID.iter().enumerate() {
        assert_eq!(decode_nibble(i as u8), v);
        assert_eq!(decode_nibble(i as u8 | 0x08), -v);
    }
    // E8M0: value = 2^(byte - 127)
    assert_eq!(e8m0_to_f32(127), 1.0);
    assert_eq!(e8m0_to_f32(128), 2.0);
    assert_eq!(e8m0_to_f32(126), 0.5);
}
