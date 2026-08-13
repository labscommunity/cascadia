//! Golden test for the GLM-5.2 MoE block (router + top-k experts + shared)
//! against the CPU reference (`tools/glm5_ref::moe_ref`).
//!
//! Regenerate fixtures:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::glm::moe::{AnyExpert, ExpertW, MoeLayer, MoeWeights};
use cascadia_engine_sparse_moe::glm::residency::UsageStats;

macro_rules! fixtures {
    () => {{
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/glm5/fixtures.safetensors");
        StFile::open(&p).expect("open fixtures")
    }};
}

fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut worst = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
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

fn bits(f: &[f32]) -> Vec<u16> {
    f.iter().map(|&v| (v.to_bits() >> 16) as u16).collect()
}

#[test]
fn moe_block_matches_reference() {
    let fx = fixtures!();
    let (hidden, n_experts, top_k, moe_inter, scale) = (32usize, 4usize, 2usize, 10usize, 2.5f32);

    let load_expert = |p: &str| ExpertW {
        wg: bits(&fx.f32(&format!("{p}.wg")).unwrap().1),
        wu: bits(&fx.f32(&format!("{p}.wu")).unwrap().1),
        wd: bits(&fx.f32(&format!("{p}.wd")).unwrap().1),
    };
    let experts: Vec<AnyExpert> = (0..n_experts)
        .map(|e| load_expert(&format!("moe.e{e}")).into())
        .collect();
    let w = MoeWeights {
        router_w: fx.f32("moe.router_w").unwrap().1,
        router_bias: fx.f32("moe.router_bias").unwrap().1,
        experts,
        shared: load_expert("moe.sh").into(),
    };
    // shared_inter == moe_inter * n_shared (n_shared = 1 here).
    let layer = MoeLayer::new(hidden, n_experts, top_k, moe_inter, moe_inter, scale, w);

    let (xshape, x) = fx.f32("moe.x").unwrap(); // [rows, hidden]
    let (_, want) = fx.f32("moe.out").unwrap();
    let rows = xshape[0];

    let mut got = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let y = layer.forward_token(&x[r * hidden..(r + 1) * hidden]);
        got[r * hidden..(r + 1) * hidden].copy_from_slice(&y);
    }
    assert_close("moe.out", &got, &want, 2e-3, 1e-2);
}

/// Batch-union MoE must be BIT-IDENTICAL to per-token routing: the dedup only
/// reorders which expert is loaded when, never the per-row accumulation. Uses
/// the same fixture as the golden above, driving several rows at once.
#[test]
fn moe_batch_union_bit_exact_vs_per_token() {
    let fx = fixtures!();
    let (hidden, n_experts, top_k, moe_inter, scale) = (32usize, 4usize, 2usize, 10usize, 2.5f32);

    let load_expert = |p: &str| ExpertW {
        wg: bits(&fx.f32(&format!("{p}.wg")).unwrap().1),
        wu: bits(&fx.f32(&format!("{p}.wu")).unwrap().1),
        wd: bits(&fx.f32(&format!("{p}.wd")).unwrap().1),
    };
    let experts: Vec<AnyExpert> = (0..n_experts)
        .map(|e| load_expert(&format!("moe.e{e}")).into())
        .collect();
    let w = MoeWeights {
        router_w: fx.f32("moe.router_w").unwrap().1,
        router_bias: fx.f32("moe.router_bias").unwrap().1,
        experts,
        shared: load_expert("moe.sh").into(),
    };
    let layer = MoeLayer::new(hidden, n_experts, top_k, moe_inter, moe_inter, scale, w);

    let (xshape, x) = fx.f32("moe.x").unwrap();
    let rows = xshape[0];

    // Per-token reference (the established golden path).
    let mut per_token = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let y = layer.forward_token(&x[r * hidden..(r + 1) * hidden]);
        per_token[r * hidden..(r + 1) * hidden].copy_from_slice(&y);
    }

    // Batch-union over all rows must match bit-for-bit.
    let batched = layer.forward_batch(&x, rows);
    assert_eq!(
        batched, per_token,
        "batch-union diverged from per-token routing"
    );
}

/// The learned-pin recorder: once a usage sink is attached, every routed
/// selection lands in the histogram keyed by this layer's global index. No
/// fixtures — zero weights are enough since we only assert routing bookkeeping.
#[test]
fn moe_records_routing_into_usage() {
    let (hidden, n_experts, top_k, inter) = (8usize, 4usize, 2usize, 4usize);
    let zero_expert = || {
        AnyExpert::from(ExpertW {
            wg: vec![0u16; inter * hidden],
            wu: vec![0u16; inter * hidden],
            wd: vec![0u16; hidden * inter],
        })
    };
    let w = MoeWeights {
        router_w: vec![0.0f32; n_experts * hidden], // equal logits -> deterministic top-k
        router_bias: vec![0.0f32; n_experts],
        experts: (0..n_experts).map(|_| zero_expert()).collect(),
        shared: zero_expert(),
    };
    let mut layer = MoeLayer::new(hidden, n_experts, top_k, inter, inter, 2.5, w);

    // Unattached: forward records nothing.
    let x = vec![0.1f32; hidden];
    let _ = layer.forward_token(&x);

    let usage = Arc::new(Mutex::new(UsageStats::new()));
    layer.attach_usage(7, Arc::clone(&usage));
    for _ in 0..3 {
        let _ = layer.forward_token(&x);
    }

    let u = usage.lock().unwrap();
    // Only the 3 post-attach tokens count: top_k selections each.
    assert_eq!(u.total, (top_k * 3) as u64);
    // Constant input -> the same top_k experts every token; all keyed to layer 7.
    let hot = u.hottest(n_experts);
    assert_eq!(hot.len(), top_k, "exactly top_k distinct experts fired");
    assert!(
        hot.iter().all(|&(l, _)| l == 7),
        "all recorded at global layer 7"
    );
}
