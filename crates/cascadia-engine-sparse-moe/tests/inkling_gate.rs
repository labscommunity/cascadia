//! Inkling router (`InklingTopkRouter`): crafted-logit unit tests plus the HF
//! golden (`gate_logits` / `gate_bias` / `gate_idx` / `gate_w` / `gate_gamma`).
//!
//! Regenerate fixtures:
//!   python tools/inkling_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/inkling

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::inkling::gate::inkling_gate;

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
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.is_finite() && w.is_finite(),
            "{name}: non-finite value at [{i}]: got {g} want {w}"
        );
        let d = (g - w).abs();
        assert!(
            d <= atol + rtol * w.abs(),
            "{name}[{i}]: got {g} want {w} (diff {d}, atol {atol} rtol {rtol})"
        );
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn bias_steers_selection_and_weights_normalise_over_selected_plus_shared() {
    // 4 routed + 2 shared, all logits 0 -> every score 0.5. Bias picks
    // experts 2 then 0. den = 0.5·(2 selected + 2 shared) = 2.0, so every
    // weight and gamma is 0.25 · route_scale · global_scale.
    let logits = [0.0f32; 6];
    let bias = [0.1f32, 0.0, 0.2, 0.0];
    let out = inkling_gate(&logits, &bias, 2, 2, 8.0, 0.5);
    assert_eq!(out.idx, vec![2, 0]);
    assert_close("w", &out.w, &[1.0, 1.0], 1e-6, 0.0);
    assert_close("gammas", &out.gammas, &[1.0, 1.0], 1e-6, 0.0);
}

#[test]
fn ties_break_toward_the_lower_expert_id() {
    let logits = [0.0f32; 5]; // 3 routed + 2 shared, all tied
    let out = inkling_gate(&logits, &[0.0; 3], 2, 2, 1.0, 1.0);
    assert_eq!(out.idx, vec![0, 1]);
    // Selection-score order, not id order, when scores differ.
    let logits = [0.0f32, 1.0, 0.0, 0.0, 0.0];
    let out = inkling_gate(&logits, &[0.0; 3], 2, 2, 1.0, 1.0);
    assert_eq!(out.idx, vec![1, 0]);
}

#[test]
fn weights_use_raw_sigmoid_scores_not_the_biased_selection_score() {
    // Expert 1 has the lowest raw score but a huge bias: it is selected first,
    // yet its weight is its RAW sigmoid share.
    let logits = [2.0f32, -2.0, 0.0, 0.0, 0.5, -0.5];
    let bias = [0.0f32, 10.0, 0.0, 0.0];
    let out = inkling_gate(&logits, &bias, 2, 2, 1.0, 1.0);
    assert_eq!(out.idx, vec![1, 0]);
    let s: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let den = s[1] + s[0] + s[4] + s[5];
    assert_close("w", &out.w, &[s[1] / den, s[0] / den], 1e-6, 1e-6);
    assert_close("gammas", &out.gammas, &[s[4] / den, s[5] / den], 1e-6, 1e-6);
}

#[test]
fn weights_and_gammas_sum_to_route_scale_times_global_scale() {
    let logits = [0.3f32, -1.2, 2.5, 0.0, -0.7, 1.1, 0.4, 0.9, -0.2, 0.6];
    let bias = [0.05f32, -0.01, 0.0, 0.02, 0.0, 0.03, -0.04, 0.0];
    let (route_scale, global_scale) = (8.0f32, 0.75f32);
    let out = inkling_gate(&logits, &bias, 3, 2, route_scale, global_scale);
    assert_eq!(out.idx.len(), 3);
    assert_eq!(out.w.len(), 3);
    assert_eq!(out.gammas.len(), 2);
    let total: f32 = out.w.iter().sum::<f32>() + out.gammas.iter().sum::<f32>();
    assert!(
        (total - route_scale * global_scale).abs() < 1e-5,
        "total {total} != {}",
        route_scale * global_scale
    );
    // Every selected id is unique and in range.
    let mut ids = out.idx.clone();
    ids.dedup();
    assert_eq!(ids.len(), 3);
    assert!(ids.iter().all(|&i| i < 8));
}

#[test]
fn a_hot_shared_expert_sinks_the_routed_weights() {
    let cold = [1.0f32, 0.0, 0.0, 0.0, -20.0, -20.0]; // shared ~0 -> routed keep the mass
    let hot = [1.0f32, 0.0, 0.0, 0.0, 20.0, 20.0]; // shared ~1 each
    let a = inkling_gate(&cold, &[0.0; 4], 1, 2, 1.0, 1.0);
    let b = inkling_gate(&hot, &[0.0; 4], 1, 2, 1.0, 1.0);
    assert_eq!(a.idx, vec![0]);
    assert_eq!(b.idx, vec![0]);
    assert!(
        (a.w[0] - 1.0).abs() < 1e-6,
        "no shared mass -> routed weight 1"
    );
    let s0 = sigmoid(1.0);
    assert!((b.w[0] - s0 / (s0 + 2.0)).abs() < 1e-5, "shared sink share");
}

/// HF golden. `torch.topk(sorted=False)` leaves the selection order unspecified,
/// so ids/weights are compared as (id, weight) pairs sorted by id. f32: 1e-4.
#[test]
fn golden_gate_matches_hf() {
    let Some(fx) = fixtures() else { return };
    let (_, logits) = fx.f32("gate_logits").expect("gate_logits");
    let (bshape, bias) = fx.f32("gate_bias").expect("gate_bias");
    let (ishape, idx_want) = fx.i32("gate_idx").expect("gate_idx");
    let (_, w_want) = fx.f32("gate_w").expect("gate_w");
    let (gshape, gamma_want) = fx.f32("gate_gamma").expect("gate_gamma");
    let n_routed = bshape[0];
    let n_shared = gshape[gshape.len() - 1];
    let top_k = ishape[ishape.len() - 1];
    assert_eq!(
        logits.len(),
        n_routed + n_shared,
        "gate_logits must be routed + shared"
    );
    // Tiny config (PORT_SPEC §4): route_scale 8; global_scale from the fixture
    // if present (1.0 at init), else 1.0.
    let route_scale = 8.0f32;
    let global_scale = fx.f32("gate_global_scale").map(|t| t.1[0]).unwrap_or(1.0);

    let out = inkling_gate(&logits, &bias, top_k, n_shared, route_scale, global_scale);

    let mut got: Vec<(usize, f32)> = out.idx.iter().copied().zip(out.w.iter().copied()).collect();
    let mut want: Vec<(usize, f32)> = idx_want
        .iter()
        .map(|&i| i as usize)
        .zip(w_want.iter().copied())
        .collect();
    got.sort_by_key(|p| p.0);
    want.sort_by_key(|p| p.0);
    let got_ids: Vec<usize> = got.iter().map(|p| p.0).collect();
    let want_ids: Vec<usize> = want.iter().map(|p| p.0).collect();
    assert_eq!(got_ids, want_ids, "gate_idx");
    let got_w: Vec<f32> = got.iter().map(|p| p.1).collect();
    let want_w: Vec<f32> = want.iter().map(|p| p.1).collect();
    assert_close("gate_w", &got_w, &want_w, 1e-4, 1e-4);
    assert_close("gate_gamma", &out.gammas, &gamma_want, 1e-4, 1e-4);
}
