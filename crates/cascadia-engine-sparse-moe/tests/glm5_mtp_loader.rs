//! MTP loader round-trip: `glm::loader::load_model` must read the MTP draft head
//! that `export_glm5.py --tiny` writes to `<dir>/mtp.safetensors` (bf16 block
//! experts) and wire it via `set_mtp`, so `generate_spec` runs and produces a
//! token-for-token IDENTICAL sequence to plain greedy `generate`. This validates
//! the exporter write, the loader read, and the speculative-decode loop end to
//! end on a real on-disk export.
//!
//! Regenerate the export:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::glm::loader::load_model;

#[test]
fn loader_mtp_spec_matches_greedy() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export");
    if !dir.join("mtp.safetensors").exists() {
        eprintln!("SKIP: {} absent (run tools/glm5_ref/gen_fixtures.py)", dir.display());
        return;
    }
    let prompt: Vec<u32> = vec![1, 2, 3, 4];
    let n_gen = 6usize;
    // Room for the transient verify KV peak (committed + g positions per round).
    let max_seq = prompt.len() + n_gen + 8;

    // Ground-truth greedy from the loaded model.
    let want = load_model(&dir, max_seq).expect("load export").generate(&prompt, n_gen);
    assert_eq!(want.len(), n_gen);

    for g in [1usize, 2, 3, 4] {
        let mut model = load_model(&dir, max_seq).expect("load export");
        assert!(model.has_mtp(), "export carries an MTP head");
        let out = model.generate_spec(&prompt, n_gen, g);
        assert_eq!(out.tokens, want, "loaded spec (g={g}) diverged from greedy");
        assert!(out.forwards <= n_gen + 1);
        assert!(out.accepted <= out.drafted);
        eprintln!("g={g}: forwards={} drafted={} accepted={}", out.forwards, out.drafted, out.accepted);
    }
}
