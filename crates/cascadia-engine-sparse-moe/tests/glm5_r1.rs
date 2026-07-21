//! Light-R1 read path parity: with CASCADIA_GLM5_R1READ enabled, the MoE reads
//! each routed expert's whole bin explicitly (concurrently, off the compute
//! path) instead of faulting mmap pages in mid-GEMV. The bytes are identical to
//! the mmap, so greedy generation must be **token-identical** to the plain mmap
//! path (same reference the mmap test checks).
//!
//! Separate test binary so the process-global CASCADIA_GLM5_R1READ flag (read
//! once via OnceLock) is on for the whole run.
//!
//! Requires the export fixture (run tools/glm5_ref/gen_fixtures.py).

use std::path::PathBuf;

use cascadia_engine_sparse_moe::glm::stage::GlmRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;

#[test]
fn r1_read_matches_mmap_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export");
    if !dir.join("manifest.json").exists() {
        eprintln!("SKIP: glm5_export absent (run tools/glm5_ref/gen_fixtures.py)");
        return;
    }
    // Force mmap experts (so the R1 read path is actually exercised) + R1 on.
    std::env::set_var("CASCADIA_GLM5_EXPERTS", "mmap");
    std::env::set_var("CASCADIA_GLM5_R1READ", "1");
    let got = GlmRunner::load_staged(&dir, 32, 0, 1, 0, 0).unwrap().generate_argmax(&[1, 2, 3, 4], 4);
    std::env::remove_var("CASCADIA_GLM5_EXPERTS");
    std::env::remove_var("CASCADIA_GLM5_R1READ");
    // Same reference as glm5_expert_mmap's mmap path → R1 is bit-identical.
    assert_eq!(got, vec![4u32, 10, 3, 15], "R1 read path diverged from the mmap reference");
}
