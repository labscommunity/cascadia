//! Inkling causal short conv (`InklingShortConvolution`): hand-derivable unit
//! tests, decode/prefill/rewind/snapshot parity, and the HF golden
//! (`sconv_in` / `sconv_w` / `sconv_out` in `tests/fixtures/inkling/`).
//!
//! Regenerate fixtures:
//!   python tools/inkling_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/inkling

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::inkling::conv::ShortConv;

/// The HF-generated fixture file, or `None` (with a notice) when it has not
/// been generated yet — golden tests then skip instead of failing.
/// `INKLING_FIXTURES=<dir>` points at an out-of-tree `fixtures.safetensors`.
fn fixtures() -> Option<StFile> {
    let p = match std::env::var_os("INKLING_FIXTURES") {
        Some(dir) => PathBuf::from(dir).join("fixtures.safetensors"),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/inkling/fixtures.safetensors"),
    };
    if !p.exists() {
        eprintln!(
            "inkling fixtures missing at {}; skipping golden",
            p.display()
        );
        return None;
    }
    Some(StFile::open(&p).expect("open inkling fixtures"))
}

/// Mixed abs/rel closeness: |a-b| <= atol + rtol*|b|. Reports the worst offender.
fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut worst = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.is_finite() && w.is_finite(),
            "{name}: non-finite value at [{i}]: got {g} want {w}"
        );
        let d = (g - w).abs();
        if d > atol + rtol * w.abs() && d > worst.1 {
            worst = (i, d, g, w);
        }
    }
    assert!(
        worst.1 == 0.0,
        "{name}: worst diff {} at [{}]: got {} want {} (atol {atol} rtol {rtol})",
        worst.1,
        worst.0,
        worst.2,
        worst.3
    );
}

/// Deterministic pseudo-random inputs (no rand dependency).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// Reference: the spec formula, per channel, straight from the definition
/// `y[c,p] = Σ_j w[c,j]·u[c, p-(K-1)+j] + u[c,p]` with zero padding.
fn reference(u: &[f32], t: usize, w: &[f32], c: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; t * c];
    for p in 0..t {
        for ch in 0..c {
            let mut acc = 0.0f32;
            for j in 0..k {
                let pos = p as isize - (k as isize - 1) + j as isize;
                if pos >= 0 {
                    acc += w[ch * k + j] * u[pos as usize * c + ch];
                }
            }
            y[p * c + ch] = acc + u[p * c + ch];
        }
    }
    y
}

#[test]
fn impulse_response_is_reversed_kernel_plus_residual() {
    // One channel, K=4, unit impulse at position 0: the conv output at position
    // p is tap (K-1-p); position 0 also carries the residual `+1`.
    let w = vec![0.1f32, 0.2, 0.3, 0.4];
    let u = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
    let want = [0.4 + 1.0, 0.3, 0.2, 0.1, 0.0, 0.0];

    let mut conv = ShortConv::new(w.clone(), 1, 4);
    let got = conv.prefill(&u, u.len());
    assert_close("impulse prefill", &got, &want, 1e-7, 0.0);

    let mut conv = ShortConv::new(w, 1, 4);
    let got: Vec<f32> = u.iter().map(|&x| conv.decode(&[x])[0]).collect();
    assert_close("impulse decode", &got, &want, 1e-7, 0.0);
}

#[test]
fn zero_kernel_is_the_identity_through_the_residual() {
    let mut conv = ShortConv::new(vec![0.0; 3 * 4], 3, 4);
    let u = [1.5f32, -2.0, 0.25];
    assert_eq!(conv.decode(&u), u.to_vec());
    assert_eq!(conv.decode(&u), u.to_vec());
}

#[test]
fn multichannel_matches_the_spec_formula() {
    let (c, k, t) = (5usize, 4usize, 9usize);
    let mut rng = Lcg(11);
    let w = rng.vec(c * k);
    let u = rng.vec(t * c);
    let want = reference(&u, t, &w, c, k);
    let mut conv = ShortConv::new(w, c, k);
    let got = conv.prefill(&u, t);
    assert_close("multichannel", &got, &want, 1e-6, 1e-6);
}

#[test]
fn prefill_is_bit_identical_to_sequential_decode() {
    let (c, k, t) = (6usize, 4usize, 11usize);
    let mut rng = Lcg(23);
    let w = rng.vec(c * k);
    let u = rng.vec(t * c);

    let mut a = ShortConv::new(w.clone(), c, k);
    let pre = a.prefill(&u, t);

    let mut b = ShortConv::new(w, c, k);
    let mut seq = Vec::with_capacity(t * c);
    for row in u.chunks_exact(c) {
        seq.extend(b.decode(row));
    }
    assert_eq!(pre, seq, "prefill must equal sequential decode bit-for-bit");
    assert_eq!(a.len(), t);
    assert_eq!(b.len(), t);

    // A prefill that continues an existing history (chunked prefill).
    let mut c2 = ShortConv::new(vec![0.3; c * k], c, k);
    let mut d2 = ShortConv::new(vec![0.3; c * k], c, k);
    let whole = c2.prefill(&u, t);
    let mut chunked = d2.prefill(&u[..4 * c], 4);
    chunked.extend(d2.prefill(&u[4 * c..], t - 4));
    assert_eq!(
        whole, chunked,
        "chunked prefill must equal one-shot prefill"
    );
}

#[test]
fn truncate_then_redecode_matches_uninterrupted() {
    let (c, k, t) = (3usize, 4usize, 10usize);
    let mut rng = Lcg(5);
    let w = rng.vec(c * k);
    let u = rng.vec(t * c);
    let mut conv = ShortConv::new(w, c, k);
    let full = conv.prefill(&u, t);

    // Reject the last 4 positions (within the default rewind), then re-feed them.
    conv.truncate(6);
    assert_eq!(conv.len(), 6);
    let redo = conv.prefill(&u[6 * c..], 4);
    assert_eq!(redo, full[6 * c..].to_vec(), "re-decode after truncate");
}

#[test]
fn snapshot_restore_roundtrip_and_rewind_after_restore() {
    let (c, k, t) = (4usize, 4usize, 9usize);
    let mut rng = Lcg(77);
    let w = rng.vec(c * k);
    let u = rng.vec(t * c);
    let mut conv = ShortConv::new(w.clone(), c, k);
    let full = conv.prefill(&u, t);

    // Snapshot after 5 positions, continue, then reset + restore + continue.
    let mut conv = ShortConv::new(w.clone(), c, k);
    let _ = conv.prefill(&u[..5 * c], 5);
    let snap = conv.snapshot();
    assert_eq!(snap.len(), 5);
    conv.reset();
    assert_eq!(conv.len(), 0);
    conv.restore(&snap);
    assert_eq!(conv.len(), 5);
    let tail = conv.prefill(&u[5 * c..], t - 5);
    assert_eq!(tail, full[5 * c..].to_vec(), "decode after restore");

    // The restored history must also support a rewind: back to 3, re-feed.
    conv.truncate(3);
    let redo = conv.prefill(&u[3 * c..], t - 3);
    assert_eq!(redo, full[3 * c..].to_vec(), "rewind after restore");
}

#[test]
#[should_panic(expected = "exceeds the rewind slack")]
fn truncate_beyond_the_rewind_slack_panics() {
    let mut conv = ShortConv::with_rewind(vec![0.5; 2 * 4], 2, 4, 2);
    let _ = conv.prefill(&[0.0; 2 * 6], 6);
    conv.truncate(2); // rewinds 4 > slack 2
}

#[test]
fn rewind_slack_exactly_at_the_bound_is_allowed() {
    let (c, k, t) = (2usize, 4usize, 8usize);
    let mut rng = Lcg(9);
    let w = rng.vec(c * k);
    let u = rng.vec(t * c);
    let mut a = ShortConv::with_rewind(w.clone(), c, k, 3);
    let full = a.prefill(&u, t);
    a.truncate(t - 3);
    let redo = a.prefill(&u[(t - 3) * c..], 3);
    assert_eq!(redo, full[(t - 3) * c..].to_vec());
}

/// HF golden: `sconv_in` / `sconv_out` are stored channel-major `[C, T]`
/// (torch's conv1d layout); the Rust API is `[T, C]`. `sconv_out` is the
/// module output (conv + residual). f32 path: element-wise 1e-4.
#[test]
fn golden_sconv_matches_hf() {
    let Some(fx) = fixtures() else { return };
    let (ishape, x_ct) = fx.f32("sconv_in").expect("sconv_in");
    let (wshape, w) = fx.f32("sconv_w").expect("sconv_w");
    let (_, want_ct) = fx.f32("sconv_out").expect("sconv_out");
    let (c, t) = (ishape[0], ishape[1]);
    let k = wshape[wshape.len() - 1]; // [C, K] or [C, 1, K]
    assert_eq!(w.len(), c * k);

    let transpose = |ct: &[f32]| -> Vec<f32> {
        let mut tc = vec![0.0f32; t * c];
        for ch in 0..c {
            for p in 0..t {
                tc[p * c + ch] = ct[ch * t + p];
            }
        }
        tc
    };
    let x = transpose(&x_ct);
    let want = transpose(&want_ct);

    let mut conv = ShortConv::new(w, c, k);
    let got = conv.prefill(&x, t);
    assert_close("sconv_out", &got, &want, 1e-4, 1e-4);
}

// ---- rewind slack is measured from the write high-water mark -------------

/// The reviewer's repro: two truncates whose SUM exceeds the slack, each alone
/// within it. Measured against the already-lowered `len` the second passed
/// silently and the decode at 40 read ring rows positions 72..74 had
/// overwritten; measured against the high-water mark (100) it panics.
#[test]
#[should_panic(expected = "exceeds the rewind slack")]
fn consecutive_truncates_cannot_creep_past_the_slack() {
    let (c, k, rewind, t) = (3usize, 4usize, 32usize, 100usize);
    let mut rng = Lcg(31);
    let w = rng.vec(c * k);
    let u = rng.vec(t * c);
    let mut conv = ShortConv::with_rewind(w, c, k, rewind);
    let _ = conv.prefill(&u, t);
    conv.truncate(70); // 30 <= 32
    conv.truncate(40); // 60 positions below the high-water mark 100
}

#[test]
fn consecutive_truncates_within_the_slack_redecode_exactly() {
    let (c, k, rewind, t) = (3usize, 4usize, 32usize, 100usize);
    let mut rng = Lcg(32);
    let w = rng.vec(c * k);
    let u = rng.vec(t * c);
    let mut conv = ShortConv::with_rewind(w.clone(), c, k, rewind);
    let full = conv.prefill(&u, t);
    conv.truncate(90);
    let _ = conv.prefill(&u[90 * c..95 * c], 5); // partial re-decode: hwm stays 100
    conv.truncate(75); // 25 below 100
    conv.truncate(68); // 32 below 100: exactly the slack
    let redo = conv.prefill(&u[68 * c..], t - 68);
    assert_eq!(
        redo,
        full[68 * c..].to_vec(),
        "decode after two truncates must equal the uninterrupted sequence"
    );
    // and the uninterrupted sequence on a fresh conv is `full` (sanity).
    let mut fresh = ShortConv::with_rewind(w, c, k, rewind);
    assert_eq!(fresh.prefill(&u, t), full);
}

#[test]
fn truncate_snapshot_restore_truncate_within_the_slack_is_exact() {
    let (c, k, rewind, t) = (4usize, 4usize, 8usize, 40usize);
    let mut rng = Lcg(33);
    let w = rng.vec(c * k);
    let u = rng.vec(t * c);
    let mut conv = ShortConv::with_rewind(w.clone(), c, k, rewind);
    let full = conv.prefill(&u, t);
    conv.truncate(37); // 3 of the 8
    let snap = conv.snapshot();
    assert_eq!(snap.len(), 37);

    // Into a fresh conv (prefix-cache reuse): the remaining 5 of the slack,
    // measured from the ORIGINAL high-water mark 40, are still usable ...
    let mut fresh = ShortConv::with_rewind(w.clone(), c, k, rewind);
    fresh.restore(&snap);
    fresh.truncate(32);
    let redo = fresh.prefill(&u[32 * c..], t - 32);
    assert_eq!(redo, full[32 * c..].to_vec(), "fresh conv after restore");

    // ... and back into the same conv after a reset.
    conv.reset();
    conv.restore(&snap);
    conv.truncate(32);
    let redo = conv.prefill(&u[32 * c..], t - 32);
    assert_eq!(
        redo,
        full[32 * c..].to_vec(),
        "same conv after reset+restore"
    );
}

#[test]
#[should_panic(expected = "exceeds the rewind slack")]
fn truncate_snapshot_restore_truncate_beyond_the_slack_panics() {
    let (c, k, rewind, t) = (4usize, 4usize, 8usize, 40usize);
    let mut rng = Lcg(34);
    let w = rng.vec(c * k);
    let u = rng.vec(t * c);
    let mut conv = ShortConv::with_rewind(w.clone(), c, k, rewind);
    let _ = conv.prefill(&u, t);
    conv.truncate(37);
    let snap = conv.snapshot();
    let mut fresh = ShortConv::with_rewind(w, c, k, rewind);
    fresh.restore(&snap);
    fresh.truncate(31); // 9 below the original high-water mark 40 (slack 8)
}

/// Before the ring has wrapped every position is still in it, so a rewind to
/// 0 is legal even when it is longer than the slack.
#[test]
fn rewind_to_zero_before_the_ring_wraps_is_allowed() {
    let (c, k, rewind) = (2usize, 4usize, 2usize); // hist 5
    let mut rng = Lcg(36);
    let w = rng.vec(c * k);
    let u = rng.vec(5 * c);
    let mut conv = ShortConv::with_rewind(w, c, k, rewind);
    let full = conv.prefill(&u, 5);
    conv.truncate(0); // 5 > slack 2, but hwm 5 <= hist 5: nothing overwritten
    assert_eq!(conv.prefill(&u, 5), full);
}

// ---- O(1) reset ----------------------------------------------------------

/// `reset` no longer zeroes the ring: a decode only reads rows written since
/// the reset, so the previous sequence (here one that wrapped the ring many
/// times, then got truncated) must be invisible — bit-identical to a fresh conv.
#[test]
fn reset_then_decode_is_bit_identical_to_a_fresh_conv() {
    let (c, k, rewind) = (5usize, 4usize, 3usize); // hist 6
    let mut rng = Lcg(35);
    let w = rng.vec(c * k);
    let a = rng.vec(50 * c);
    let b = rng.vec(20 * c);
    let mut conv = ShortConv::with_rewind(w.clone(), c, k, rewind);
    let _ = conv.prefill(&a, 50);
    conv.truncate(48);
    conv.reset();
    assert_eq!(conv.len(), 0);
    let got = conv.prefill(&b, 20);
    let mut fresh = ShortConv::with_rewind(w, c, k, rewind);
    assert_eq!(
        got,
        fresh.prefill(&b, 20),
        "reset must hide the previous sequence entirely"
    );
    // A rewind right after the reset behaves like a fresh conv's too.
    conv.truncate(17);
    fresh.truncate(17);
    assert_eq!(
        conv.prefill(&b[17 * c..], 3),
        fresh.prefill(&b[17 * c..], 3)
    );
}
