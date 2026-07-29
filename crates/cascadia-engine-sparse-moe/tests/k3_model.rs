//! End-to-end parity: the Rust k3 shell against the CPU reference's
//! `model.tokens -> model.logits` golden.
//!
//! This is the test that exercises everything at once — embed, the hybrid
//! KDA/MLA layer mix, the AttnRes block stack, LatentMoE with fp4 experts, the
//! model-level output mixture, and the head.
//!
//! Regenerate both inputs:
//!   python tools/kimi_k3_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/kimi_k3
//!   python tools/export_kimi_k3.py \
//!       --tiny crates/cascadia-engine-sparse-moe/tests/fixtures/kimi_k3_export

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::math::{linear_bf16_w, rmsnorm};
use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::k3::attn_res::apply_attn_res;
use cascadia_engine_sparse_moe::k3::loader::{load_embed, load_head, load_layers, K3Manifest};
use cascadia_engine_sparse_moe::k3::model::{forward_slice, max_blocks};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Both the golden tensors and the tiny export are gitignored, so a fresh
/// checkout skips rather than fails.
macro_rules! need {
    ($p:expr, $how:expr) => {{
        let p = $p;
        if !p.exists() {
            eprintln!("SKIP: {} absent ({})", p.display(), $how);
            return;
        }
        p
    }};
}

#[test]
fn tiny_model_matches_the_reference_logits() {
    let fx = need!(
        fixtures_dir().join("kimi_k3/fixtures.safetensors"),
        "run tools/kimi_k3_ref/gen_fixtures.py"
    );
    let dir = need!(
        fixtures_dir().join("kimi_k3_export"),
        "run tools/export_kimi_k3.py --tiny"
    );

    let f = StFile::open(&fx).expect("open fixtures");
    let (_, toks) = f.i32("model.tokens").expect("model.tokens");
    let (ls, want) = f.f32("model.logits").expect("model.logits");
    let (_, want_argmax) = f.i32("model.argmax").expect("model.argmax");

    let m = K3Manifest::load(&dir).expect("manifest");
    let d = m.dims();
    let embed = load_embed(&dir).expect("embed");
    let head = load_head(&dir).expect("head");
    let (mut layers, mut states) = load_layers(&dir, &m, 0, m.num_hidden_layers).expect("layers");

    let h = m.hidden_size;
    let maxb = max_blocks(m.num_hidden_layers, m.attn_res_block_size);
    let vocab = ls[1];
    assert_eq!(vocab, m.vocab_size, "fixture/manifest vocab mismatch");

    let mut got = vec![0.0f32; toks.len() * vocab];
    for (t, &tok) in toks.iter().enumerate() {
        // embed lookup (no rounding — the reference reads the table directly)
        let mut prefix = embed[tok as usize * h..(tok as usize + 1) * h].to_vec();
        // the block stack is per-token: rebuilt each forward, carried across layers
        let mut blocks = vec![0.0f32; maxb * h];

        let nb = forward_slice(&mut layers, &mut states, d, &mut prefix, &mut blocks, 0);
        assert_eq!(nb, maxb, "token {t}: expected {maxb} live blocks, got {nb}");

        // model-level output mixture, then the head
        let mut x = vec![0.0f32; h];
        apply_attn_res(
            &prefix,
            &blocks[..nb * h],
            &head.out_res_proj,
            &head.out_res_norm,
            m.rms_norm_eps,
            &mut x,
        );
        rmsnorm(&mut x, &head.norm, m.rms_norm_eps);
        linear_bf16_w(
            &x,
            &head.lm_head,
            vocab,
            h,
            &mut got[t * vocab..(t + 1) * vocab],
        );
    }

    // argmax must agree exactly — that is what generation actually depends on
    for t in 0..toks.len() {
        let row = &got[t * vocab..(t + 1) * vocab];
        let am = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap();
        assert_eq!(am, want_argmax[t], "token {t}: argmax differs");
    }

    let worst = got
        .iter()
        .zip(&want)
        .map(|(&g, &w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 5e-2, "logits drift too large: max|diff| = {worst}");
    eprintln!("k3 e2e: {} tokens, max|diff| = {worst:.3e}", toks.len());
}

#[test]
fn reset_reproduces_a_fresh_sequence() {
    let dir = need!(
        fixtures_dir().join("kimi_k3_export"),
        "run tools/export_kimi_k3.py --tiny"
    );
    let m = K3Manifest::load(&dir).expect("manifest");
    let d = m.dims();
    let embed = load_embed(&dir).expect("embed");
    let (mut layers, mut states) = load_layers(&dir, &m, 0, m.num_hidden_layers).expect("layers");

    let h = m.hidden_size;
    let maxb = max_blocks(m.num_hidden_layers, m.attn_res_block_size);

    let run = |layers: &mut _, states: &mut _| -> Vec<f32> {
        let mut prefix = embed[3 * h..4 * h].to_vec();
        let mut blocks = vec![0.0f32; maxb * h];
        forward_slice(layers, states, d, &mut prefix, &mut blocks, 0);
        prefix
    };

    let first = run(&mut layers, &mut states);
    // advance the state, then clear it — must return to the fresh result
    let _ = run(&mut layers, &mut states);
    for s in states.iter_mut() {
        s.clear();
    }
    let after_reset = run(&mut layers, &mut states);
    assert_eq!(
        first, after_reset,
        "clear() left KDA recurrence / conv / KV state behind"
    );
}
