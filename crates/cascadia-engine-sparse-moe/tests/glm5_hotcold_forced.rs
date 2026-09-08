//! Hot/cold overlapped decode parity, forced-cold mode: CASCADIA_GLM5_HOTCOLD=cold
//! classifies every mmap routed expert as cold, so every slot takes the
//! background-read + swiglu_from path with the shared expert computed during
//! the overlap window (FreeToken-style split, arXiv:2608.16157). The read
//! buffers are byte-identical to the mmap and accumulation stays in gate
//! order, so greedy generation must be **token-identical** to the plain mmap
//! reference. This binary is the deterministic coverage for the threaded read
//! machinery, which probe mode skips whenever the fixture is page-cache-hot.
//!
//! Separate test binary so the process-global CASCADIA_GLM5_HOTCOLD flag (read
//! once via OnceLock) is on for the whole run.
//!
//! Requires the export fixture (run tools/glm5_ref/gen_fixtures.py).

use std::path::PathBuf;

use cascadia_engine_sparse_moe::glm::stage::GlmRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;

#[test]
fn hotcold_forced_matches_mmap_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export");
    // Force mmap experts (so the read path is actually exercised) + all-cold.
    std::env::set_var("CASCADIA_GLM5_EXPERTS", "mmap");
    std::env::set_var("CASCADIA_GLM5_HOTCOLD", "cold");
    let got = GlmRunner::load_staged(&dir, 32, 0, 1, 0, 0, Default::default())
        .unwrap()
        .generate_argmax(&[1, 2, 3, 4], 4);
    std::env::remove_var("CASCADIA_GLM5_EXPERTS");
    std::env::remove_var("CASCADIA_GLM5_HOTCOLD");
    // Same reference as glm5_expert_mmap's mmap path → forced-cold is bit-identical.
    assert_eq!(
        got,
        vec![4u32, 10, 3, 15],
        "forced-cold hot/cold path diverged from the mmap reference"
    );
}
