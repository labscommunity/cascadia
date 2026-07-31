//! Prefix reuse against a real `K3Runner`, not a mock.
//!
//! The unit tests in `staged.rs` drive a stand-in runner, so they prove the
//! position arithmetic and nothing about K3's actual state. K3 carries state in
//! two forms — the KDA recurrence and the MLA latent cache — and a snapshot has
//! to restore BOTH to the same position or the run is silently wrong rather than
//! obviously broken. That is what this covers.
//!
//! The failure being guarded is not a crash. A restore that installs state for a
//! different number of positions than the caller skips produces plausible,
//! wrong tokens.
//!
//! Known limit: these compare sampled tokens, and at the tiny model's dimensions
//! an argmax is a coarse detector. Shifting the resume point by ONE position was
//! verified not to move the output here, so an off-by-one is not covered — the
//! unit tests in `staged.rs` assert the position sequence directly and are what
//! catch that. What these add is the part a mock cannot: real KDA and MLA state
//! surviving a snapshot and restore together.
//!
//! Requires the tiny export:
//!   python tools/export_kimi_k3.py \
//!       --tiny crates/cascadia-engine-sparse-moe/tests/fixtures/kimi_k3_export

use std::path::{Path, PathBuf};

use cascadia_engine_sparse_moe::k3::stage::K3Runner;
use cascadia_engine_sparse_moe::sampling::SamplingConfig;
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

/// `PrefixStore` reads its budget when the runner is built, so the env has to be
/// set before `load`. The variable is process-global and these tests run in
/// parallel, so set-then-load has to be atomic between them — without the lock a
/// test asking for a DISABLED cache can pick up another's budget and quietly
/// assert the opposite of what it meant to.
fn load_with_cache(dir: &Path, budget: &str) -> K3Runner {
    static ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: the lock makes this the only thread touching the variable, and it
    // is consumed by `load` before the guard drops.
    unsafe { std::env::set_var("CASCADIA_K3_PREFIX_CACHE", budget) };
    K3Runner::load(dir, 0, 1, 64).expect("load tiny k3")
}

#[test]
fn a_restored_prefix_generates_exactly_what_a_full_prefill_does() {
    let Some(dir) = export_dir() else { return };
    let cfg = SamplingConfig::default();
    let prompt: Vec<u32> = vec![3, 17, 5, 28, 11, 2, 19];
    let split = 4; // the earlier, shorter turn

    // cold reference over the whole prompt
    let mut cold = load_with_cache(&dir, "0");
    let want = cold.generate_reason(&prompt, 4, &cfg).0;

    // a shorter request first, whose state is then cached
    let mut warm = load_with_cache(&dir, "1000000000");
    assert!(
        warm.prefix_cache_enabled(),
        "budget not picked up; the rest of this test would prove nothing"
    );
    warm.generate_reason(&prompt[..split], 1, &cfg);
    let covered = warm.covered_positions().expect("k3 tracks pos");
    warm.cache_prefix(99);

    // now the full prompt, resuming that prefix
    let got = warm.generate_reason_from(&prompt, Some(99), 4, &cfg).0;

    assert_eq!(
        got, want,
        "reusing a prefix changed the output (covered={covered})"
    );
}

#[test]
fn a_restored_prefix_is_actually_used() {
    // Positive control, and the one this feature most needs. "Reuse matches a
    // full prefill" passes just as happily when the restore quietly does
    // nothing, which is exactly how this cache shipped broken: enabled,
    // reporting itself enabled, and inert.
    //
    // So restore a prefix built from a DIFFERENT prompt. If the state is really
    // installed the continuation must diverge; if it is ignored, the output is
    // the cold one and this fails.
    let Some(dir) = export_dir() else { return };
    let cfg = SamplingConfig::default();
    let prompt: Vec<u32> = vec![3, 17, 5, 28, 11, 2, 19];
    let other: Vec<u32> = vec![23, 8, 30, 14];

    let mut cold = load_with_cache(&dir, "0");
    let want_cold = cold.generate_reason(&prompt, 4, &cfg).0;

    let mut r = load_with_cache(&dir, "1000000000");
    r.generate_reason(&other, 1, &cfg);
    r.cache_prefix(77);
    let poisoned = r.generate_reason_from(&prompt, Some(77), 4, &cfg).0;

    assert_ne!(
        poisoned, want_cold,
        "a prefix from an unrelated prompt changed nothing — the restore is inert"
    );
}

#[test]
fn restoring_reports_the_positions_it_actually_installed() {
    // The engine skips exactly as many prompt tokens as this returns. If it
    // over-reports, later positions are written into slots that were never
    // filled; if it under-reports, they overwrite live state.
    let Some(dir) = export_dir() else { return };
    let cfg = SamplingConfig::default();
    let prompt: Vec<u32> = vec![7, 1, 4, 9, 3];

    let mut r = load_with_cache(&dir, "1000000000");
    r.generate_reason(&prompt[..3], 1, &cfg);
    let covered = r.covered_positions().expect("k3 tracks pos");
    r.cache_prefix(5);

    // a fresh runner must restore to the same coverage, not to its own idea
    let mut r2 = load_with_cache(&dir, "1000000000");
    r2.generate_reason(&prompt[..3], 1, &cfg);
    r2.cache_prefix(5);
    r2.reset();
    assert_eq!(
        r2.restore_prefix(5),
        Some(covered),
        "restore disagreed with what was cached"
    );

    // and a key nobody stored is a miss, never a silent zero-length restore
    assert_eq!(r2.restore_prefix(4242), None);
}

#[test]
fn a_disabled_cache_stores_nothing_to_restore() {
    // With no budget the store must stay empty, so a stray key cannot resurrect
    // state from an unrelated request.
    let Some(dir) = export_dir() else { return };
    let cfg = SamplingConfig::default();
    let mut r = load_with_cache(&dir, "0");
    assert!(!r.prefix_cache_enabled());
    r.generate_reason(&[3, 17, 5], 1, &cfg);
    r.cache_prefix(1);
    assert_eq!(
        r.restore_prefix(1),
        None,
        "disabled cache handed back state"
    );
}
