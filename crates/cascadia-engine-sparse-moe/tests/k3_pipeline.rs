//! N-rank pipeline parity for the K3 shell.
//!
//! Chaining `total` [`K3Runner`] stages must produce bit-identical logits to a
//! single-process run. This is what actually exercises the widened AttnRes
//! wire: with `total > 1` the block-residual stack has to survive a rank
//! boundary, and a mid rank must resume with the right number of live slots.
//!
//! Requires the tiny export:
//!   python tools/export_kimi_k3.py \
//!       --tiny crates/cascadia-engine-sparse-moe/tests/fixtures/kimi_k3_export

use std::path::{Path, PathBuf};

use cascadia_engine_sparse_moe::k3::stage::K3Runner;
use cascadia_engine_sparse_moe::staged::StagedRunner;

fn export_dir() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kimi_k3_export");
    if p.exists() {
        Some(p)
    } else {
        eprintln!(
            "SKIP: {} absent (run tools/export_kimi_k3.py --tiny)",
            p.display()
        );
        None
    }
}

/// Drive `total` ranks over one prompt, relaying the wire rank to rank.
fn run_chain(dir: &Path, total: u32, toks: &[u32]) -> Vec<Vec<f32>> {
    let mut ranks: Vec<K3Runner> = (0..total)
        .map(|r| K3Runner::load(dir, r, total, 64).expect("load rank"))
        .collect();
    for r in ranks.iter_mut() {
        r.reset();
    }

    let mut out = Vec::with_capacity(toks.len());
    for (pos, &t) in toks.iter().enumerate() {
        let mut wire = ranks[0].embed_token(t);
        for r in ranks.iter_mut() {
            wire = r.forward_layers(wire, pos, Some(t));
        }
        out.push(ranks[total as usize - 1].head_logits(&wire));
    }
    out
}

#[test]
fn multi_rank_matches_single_process() {
    let Some(dir) = export_dir() else { return };
    let toks: Vec<u32> = vec![3, 17, 5, 28, 11, 2, 19];

    let single = run_chain(&dir, 1, &toks);
    // the tiny model has 6 layers, so 2/3/6 ranks all split cleanly, and 4
    // exercises an uneven split (2,2,1,1)
    for total in [2u32, 3, 4, 6] {
        let multi = run_chain(&dir, total, &toks);
        assert_eq!(multi.len(), single.len(), "{total} ranks: token count");
        for (t, (a, b)) in multi.iter().zip(&single).enumerate() {
            assert_eq!(
                a, b,
                "{total} ranks: logits differ at token {t} (expected bit-identical)"
            );
        }
        eprintln!("k3 pipeline: {total} ranks == single process");
    }
}

#[test]
fn wire_width_carries_the_block_stack() {
    let Some(dir) = export_dir() else { return };
    let r = K3Runner::load(&dir, 0, 1, 64).expect("load");
    // tiny: hidden 64, 6 layers, block size 2 -> 3 slots -> (1 + 3) * 64
    assert_eq!(r.hidden_size(), 4 * 64);
    assert_eq!(r.arch_name(), "k3");
    // an embedded token must be wire-width with a zeroed stack
    let w = r.embed_token(3);
    assert_eq!(w.len(), r.hidden_size());
    assert!(
        w[64..].iter().all(|&v| v == 0.0),
        "block slots must start zeroed"
    );
}

#[test]
fn reset_between_runs_is_clean() {
    let Some(dir) = export_dir() else { return };
    let toks: Vec<u32> = vec![7, 1, 22];
    let a = run_chain(&dir, 2, &toks);
    let b = run_chain(&dir, 2, &toks);
    assert_eq!(a, b, "two fresh chains must agree");
}
