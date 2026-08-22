//! Hot/cold overlapped decode parity, probe mode: with CASCADIA_GLM5_HOTCOLD=1
//! the MoE residency-probes each routed expert and only background-reads the
//! cold ones (FreeToken-style split, arXiv:2608.16157). Whichever way each
//! expert is classified, accumulation stays in gate order, so greedy generation
//! must be **token-identical** to the plain mmap path (same reference the mmap
//! test checks). On a just-written fixture the bins are page-cache-resident, so
//! this run exercises the classify + all-hot fast path; the forced-cold binary
//! (glm5_hotcold_forced) covers the threaded read path deterministically.
//!
//! Separate test binary so the process-global CASCADIA_GLM5_HOTCOLD flag (read
//! once via OnceLock) is on for the whole run.
//!
//! Requires the export fixture (run tools/glm5_ref/gen_fixtures.py).

use std::path::PathBuf;

use cascadia_engine_sparse_moe::glm::stage::GlmRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;

#[test]
fn hotcold_probe_matches_mmap_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export");
    // Force mmap experts (so classification actually runs) + hot/cold probe on.
    std::env::set_var("CASCADIA_GLM5_EXPERTS", "mmap");
    std::env::set_var("CASCADIA_GLM5_HOTCOLD", "1");
    let got = GlmRunner::load_staged(&dir, 32, 0, 1, 0, 0, Default::default())
        .unwrap()
        .generate_argmax(&[1, 2, 3, 4], 4);
    std::env::remove_var("CASCADIA_GLM5_EXPERTS");
    std::env::remove_var("CASCADIA_GLM5_HOTCOLD");
    // Same reference as glm5_expert_mmap's mmap path → hot/cold is bit-identical.
    assert_eq!(
        got,
        vec![4u32, 10, 3, 15],
        "hot/cold probe path diverged from the mmap reference"
    );
}
