//! The context probe (`CASCADIA_INKLING_CONTEXT_BENCH`): fills a slot's caches
//! to N positions, decodes one row there, gives the pages back, and leaves the
//! runner exactly as usable as before (a stream opened afterwards decodes the
//! same tokens as without the probe). One binary, one test (process-wide env).

use std::path::PathBuf;

use cascadia_engine_sparse_moe::inkling::stage::InklingRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;

fn fixture() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export");
    if dir.join("manifest.json").exists() {
        Some(dir)
    } else {
        eprintln!("inkling_export fixture missing; skipping");
        None
    }
}

fn greedy(r: &mut InklingRunner, prompt: &[u32], n: usize) -> Vec<u32> {
    r.generate_argmax(prompt, n)
}

#[test]
fn probe_leaves_the_runner_usable_and_the_tokens_unchanged() {
    let Some(dir) = fixture() else { return };
    let prompt: Vec<u32> = vec![5, 33, 81, 53, 92, 85, 83, 43];
    // reference: no probe
    let mut plain =
        InklingRunner::load_staged(&dir, 4096, 0, 1, 0, 0, Some("eager".into()), None).unwrap();
    assert!(plain.configure_streams(2));
    let want = greedy(&mut plain, &prompt, 6);
    drop(plain);
    // with the probe: sizes below and above max_seq, one that the fixture's
    // memory certainly allows (the fixture is tiny) and one that must be skipped
    std::env::set_var("CASCADIA_INKLING_CONTEXT_BENCH", "64,1024,4096,100000");
    let mut probed =
        InklingRunner::load_staged(&dir, 4096, 0, 1, 0, 0, Some("eager".into()), None).unwrap();
    assert!(probed.configure_streams(2)); // runs the probe once
    assert!(probed.configure_streams(3)); // and not again
    std::env::remove_var("CASCADIA_INKLING_CONTEXT_BENCH");
    let got = greedy(&mut probed, &prompt, 6);
    assert_eq!(
        got, want,
        "the probe changed what the runner decodes afterwards"
    );
    // a stream opened after the probe starts at position 0 with an empty cache
    let slot = probed.open_stream().expect("a free slot");
    assert_eq!(probed.stream_pos(slot), 0);
    probed.close_stream(slot);
}
