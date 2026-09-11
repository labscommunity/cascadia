//! Inkling relative-position bias (`InklingRelativeLogits`): hand-derivable
//! unit tests plus the HF golden (`relpos_r` / `relpos_proj` / `relpos_bias`).
//!
//! Regenerate fixtures:
//!   python tools/inkling_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/inkling

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::inkling::relpos::RelPos;

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

/// d_rel 2, extent 3: proj rows [1,2,3] and [10,20,30]; r = [1, 0.5]
/// -> bias(d) = proj[0,d] + 0.5·proj[1,d] = 6, 12, 18.
fn small() -> (RelPos, Vec<f32>) {
    let proj = vec![1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
    (RelPos::new(proj, 2, 3), vec![1.0, 0.5])
}

#[test]
fn bias_matches_hand_dot() {
    let (rp, r) = small();
    assert_eq!(rp.bias(&r, 0), 6.0);
    assert_eq!(rp.bias(&r, 1), 12.0);
    assert_eq!(rp.bias(&r, 2), 18.0);
}

#[test]
fn bias_is_zero_at_and_beyond_extent() {
    let (rp, r) = small();
    assert_eq!(rp.bias(&r, 3), 0.0);
    assert_eq!(rp.bias(&r, 4), 0.0);
    assert_eq!(rp.bias(&r, 1000), 0.0);
}

#[test]
fn profile_is_the_whole_bias_row() {
    let (rp, r) = small();
    let prof = rp.profile(&r);
    assert_eq!(prof.len(), 3);
    for (d, &p) in prof.iter().enumerate() {
        assert_eq!(p, rp.bias(&r, d), "profile[{d}] vs bias");
    }
}

#[test]
fn row_zeroes_future_keys_and_far_past() {
    let (rp, r) = small();
    // q at 4, keys 0..=6: distances 4,3,2,1,0,-1,-2 -> only 2,1,0 are in range.
    let keys: Vec<usize> = (0..7).collect();
    let row = rp.row(&r, 4, &keys);
    assert_eq!(row, vec![0.0, 0.0, 18.0, 12.0, 6.0, 0.0, 0.0]);
}

#[test]
fn distinct_relative_states_give_distinct_profiles() {
    let (rp, _) = small();
    let a = rp.profile(&[1.0, 0.0]);
    let b = rp.profile(&[0.0, 1.0]);
    assert_eq!(a, vec![1.0, 2.0, 3.0]);
    assert_eq!(b, vec![10.0, 20.0, 30.0]);
}

/// HF golden: `relpos_r` `[Hq, d_rel]`, `relpos_proj` `[d_rel, extent]`,
/// `relpos_bias` `[Hq, kv]` for a query at position `kv - 1` over keys
/// `0..kv` (distances `kv-1 .. 0`; zero where `dist >= extent`). f32: 1e-4.
#[test]
fn golden_relpos_matches_hf() {
    let Some(fx) = fixtures() else { return };
    let (rshape, r) = fx.f32("relpos_r").expect("relpos_r");
    let (pshape, proj) = fx.f32("relpos_proj").expect("relpos_proj");
    let (bshape, want) = fx.f32("relpos_bias").expect("relpos_bias");
    let (hq, d_rel) = (rshape[0], rshape[1]);
    let extent = pshape[1];
    assert_eq!(pshape[0], d_rel);
    assert_eq!(bshape[0], hq);
    let kv = bshape[1];
    let q_pos = kv - 1;
    let keys: Vec<usize> = (0..kv).collect();

    let rp = RelPos::new(proj, d_rel, extent);
    let mut got = Vec::with_capacity(hq * kv);
    for h in 0..hq {
        got.extend(rp.row(&r[h * d_rel..(h + 1) * d_rel], q_pos, &keys));
    }
    assert_close("relpos_bias", &got, &want, 1e-4, 1e-4);
}
