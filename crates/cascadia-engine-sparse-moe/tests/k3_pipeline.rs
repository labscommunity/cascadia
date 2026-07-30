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

#[test]
fn batched_prefill_is_bit_exact_vs_per_token() {
    let Some(dir) = export_dir() else { return };
    let toks: Vec<u32> = vec![3, 17, 5, 28, 11, 2, 19];

    // per-token: the default StagedRunner path
    let mut a = K3Runner::load(&dir, 0, 1, 64).expect("load");
    a.reset();
    let w = a.hidden_size();
    let mut per_token = vec![0.0f32; toks.len() * w];
    for (pos, &t) in toks.iter().enumerate() {
        let h = a.embed_token(t);
        let o = a.forward_layers(h, pos, Some(t));
        per_token[pos * w..(pos + 1) * w].copy_from_slice(&o);
    }

    // batched: one call over all rows, batch-union MoE inside
    let mut b = K3Runner::load(&dir, 0, 1, 64).expect("load");
    b.reset();
    assert!(
        b.supports_batched_prefill(),
        "k3 must opt into batched prefill"
    );
    let mut batch = vec![0.0f32; toks.len() * w];
    for (r, &t) in toks.iter().enumerate() {
        batch[r * w..(r + 1) * w].copy_from_slice(&b.embed_token(t));
    }
    let batched = b.forward_layers_batch(batch, 0, toks.len());

    assert_eq!(
        batched, per_token,
        "batched prefill must be bit-identical to the per-token loop"
    );

    // and the head must agree on the final row
    let la = a.head_logits(&per_token[(toks.len() - 1) * w..]);
    let lb = b.head_logits(&batched[(toks.len() - 1) * w..]);
    assert_eq!(la, lb, "head logits differ after batched prefill");
}

#[test]
fn routing_is_recorded_and_autopin_stays_off_by_default() {
    use cascadia_engine_sparse_moe::k3::residency;
    let Some(dir) = export_dir() else { return };

    // decode a few tokens so the router actually fires
    let _ = run_chain(&dir, 1, &[3, 17, 5, 28]);

    // the histogram is process-global, so assert shape rather than exact counts
    let u = residency::snapshot();
    assert!(!u.is_empty(), "no routed selections recorded");
    assert!(u.total() > 0, "histogram total is zero");

    // a MoE layer must have a hottest set, and it must be within the expert count
    let hot = u.hottest_for(1, 4);
    assert!(!hot.is_empty(), "layer 1 recorded nothing");
    assert!(
        hot.iter().all(|&e| (e as usize) < 8),
        "expert id out of range: {hot:?}"
    );

    // autopin must not engage unless asked: an over-large pin set evicts the
    // cache serving the cold tail, so it is never on by default
    assert_eq!(
        residency::autopin_budget(17_547_264, 0),
        0,
        "autopin engaged without CASCADIA_K3_AUTOPIN"
    );
    let r = K3Runner::load(&dir, 0, 1, 64).expect("load");
    assert_eq!(r.pinned_experts(), 0, "experts pinned with autopin off");
}

#[test]
fn generation_matches_across_single_and_split_sources() {
    // The engine must not care whether the expert bins sit in one directory or
    // are symlinked across several filesystems. Covers the full generate path,
    // not just a single forward: state is carried across tokens.
    let Some(plain) = export_dir() else { return };
    let split =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kimi_k3_export_split");
    if !split.exists() {
        eprintln!(
            "SKIP: {} absent (run --tiny with --expert-roots)",
            split.display()
        );
        return;
    }
    // guard: the split fixture must really be symlinked, or this proves nothing
    let l = split.join("experts/layer_01.bin");
    assert!(
        std::fs::symlink_metadata(&l)
            .expect("layer_01.bin")
            .file_type()
            .is_symlink(),
        "split fixture is not symlinked"
    );

    let gen = |dir: &std::path::Path| {
        let mut r = K3Runner::load(dir, 0, 1, 64).expect("load");
        r.reset();
        r.generate_argmax(&[3, 17, 5, 28], 6)
    };
    assert_eq!(
        gen(&plain),
        gen(&split),
        "split export generated different tokens"
    );
}

/// Restoring a cached prefix and prefilling only the suffix must land on exactly
/// the same state as prefilling the whole prompt.
///
/// This is the assertion the feature lives or dies on. A restore swaps every
/// layer's state at once — the KDA recurrence and the MLA latent cache — and if
/// they disagree about position the run is silently wrong rather than broken, so
/// only an exact comparison catches it. The hit counter is checked too: without
/// it a green assertion could just mean the cache never engaged.
#[test]
fn prefix_restore_plus_suffix_matches_full_prefill() {
    let Some(dir) = export_dir() else { return };
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let split = 3; // cache covers [0,3), suffix is [3,5)

    // Asserted here rather than in its own test: the setting is an env var, and
    // cargo runs tests in the same process in parallel, so a sibling test that
    // sets it would race this one.
    std::env::remove_var("CASCADIA_K3_PREFIX_CACHE");
    assert!(
        !K3Runner::load(&dir, 0, 1, 64)
            .expect("load")
            .prefix_cache_enabled(),
        "cache must be off unless a byte budget is set"
    );

    std::env::set_var("CASCADIA_K3_PREFIX_CACHE", "268435456"); // 256 MiB
    let hs = {
        let r = K3Runner::load(&dir, 0, 1, 64).expect("load");
        r.hidden_size()
    };

    // reference: one uninterrupted prefill over the whole prompt
    let mut a = K3Runner::load(&dir, 0, 1, 64).expect("load");
    a.reset();
    let mut batch = vec![0.0f32; prompt.len() * hs];
    for (r, &t) in prompt.iter().enumerate() {
        batch[r * hs..(r + 1) * hs].copy_from_slice(&a.embed_token(t));
    }
    let full = a.forward_layers_batch(batch, 0, prompt.len());
    let want = a.head_logits(&full[(prompt.len() - 1) * hs..prompt.len() * hs]);

    // cached: prefill the head, snapshot, reset, restore, prefill only the suffix
    let mut b = K3Runner::load(&dir, 0, 1, 64).expect("load");
    assert!(b.prefix_cache_enabled(), "budget should enable the cache");
    b.reset();
    let mut head = vec![0.0f32; split * hs];
    for (r, &t) in prompt[..split].iter().enumerate() {
        head[r * hs..(r + 1) * hs].copy_from_slice(&b.embed_token(t));
    }
    b.forward_layers_batch(head, 0, split);
    b.cache_prefix(0xC0FFEE);

    let (hits_before, _) = cascadia_engine_sparse_moe::k3::prof::prefix_stats();
    b.reset();
    let reused = b.restore_prefix(0xC0FFEE).expect("cache should hit");
    assert_eq!(reused, split, "restore must report the covered length");
    let (hits_after, toks) = cascadia_engine_sparse_moe::k3::prof::prefix_stats();
    assert_eq!(hits_after, hits_before + 1, "a real hit must be recorded");
    assert!(toks >= split as u64);

    let n = prompt.len() - split;
    let mut tail = vec![0.0f32; n * hs];
    for (r, &t) in prompt[split..].iter().enumerate() {
        tail[r * hs..(r + 1) * hs].copy_from_slice(&b.embed_token(t));
    }
    let out = b.forward_layers_batch(tail, split, n);
    let got = b.head_logits(&out[(n - 1) * hs..n * hs]);
    std::env::remove_var("CASCADIA_K3_PREFIX_CACHE");

    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 * w.abs().max(1.0),
            "logit {i}: restored+suffix {g} vs full prefill {w}"
        );
    }
}
