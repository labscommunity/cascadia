//! Prefill layer-streaming parity: with CASCADIA_GLM5_PREFILL_STREAM=all, the
//! batch-union prefill enqueues every routed expert of layer i+1 for the
//! lookahead worker to warm while layer i computes (FreeToken-style full-layer
//! double buffering, arXiv:2608.16157). Warming is read-only page-cache
//! traffic racing the compute thread's own mmap access, so greedy generation
//! must be **token-identical** to the plain mmap reference. `=all` (gate off)
//! is used so the enqueue + worker machinery runs deterministically even on a
//! page-cache-hot fixture.
//!
//! Separate test binary so the process-global CASCADIA_GLM5_PREFILL_STREAM
//! flag (read at load) is on for the whole run.
//!
//! Requires the export fixture (run tools/glm5_ref/gen_fixtures.py).

use std::path::PathBuf;

use cascadia_engine_sparse_moe::glm::stage::GlmRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;

#[test]
fn prefill_stream_matches_mmap_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export");
    // Force mmap experts (so the warm path is actually exercised) + stream-all.
    std::env::set_var("CASCADIA_GLM5_EXPERTS", "mmap");
    std::env::set_var("CASCADIA_GLM5_PREFILL_STREAM", "all");
    let got = GlmRunner::load_staged(&dir, 32, 0, 1, 0, 0, Default::default())
        .unwrap()
        .generate_argmax(&[1, 2, 3, 4], 4);
    std::env::remove_var("CASCADIA_GLM5_EXPERTS");
    std::env::remove_var("CASCADIA_GLM5_PREFILL_STREAM");
    // Same reference as glm5_expert_mmap's mmap path → streaming is inert on
    // output.
    assert_eq!(
        got,
        vec![4u32, 10, 3, 15],
        "prefill layer streaming diverged from the mmap reference"
    );
}
