//! End-to-end: load the tiny model from the EXPORTER's artifact layout
//! (manifest + shell safetensors + int4_bin experts) and greedy-decode.
//!
//! Regenerate the export fixture:
//!   python tools/export_deepseek_v4.py --tiny \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/dsv4_export
//!
//! Note: experts are int4-quantized by the exporter while reference.json was
//! produced by the bf16 reference — the greedy tokens must still match (the
//! tiny model's argmax margins dominate the ~7% int4 error). If a future
//! fixture regeneration flips this, the right fix is a requantized
//! reference, not loosening the assert.

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::loader::load_model;

#[test]
fn export_loads_and_greedy_decodes() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_export");
    let mut model = load_model(&dir, 64).expect("load tiny export");

    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
            .unwrap();
    let prompt: Vec<usize> = reference["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let want: Vec<usize> = reference["generated"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let logits = model.prefill(&prompt);
    let mut got = vec![argmax(&logits)];
    for step in 1..want.len() {
        let logits = model.decode(got[step - 1], prompt.len() + step - 1);
        got.push(argmax(&logits));
    }
    assert_eq!(got, want, "greedy tokens via exported artifacts diverge");
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

/// The pipeline wire drives every token — prompt included — through the
/// decode path one position at a time (like the M2 engine). That must
/// produce the same greedy tokens as batch prefill; and reset() must make
/// the model reusable for a second identical run.
#[test]
fn token_by_token_drive_matches_and_reset_reuses() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_export");
    let mut model = load_model(&dir, 64).expect("load tiny export");
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
            .unwrap();
    let prompt: Vec<usize> = reference["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let want: Vec<usize> = reference["generated"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let run = |model: &mut cascadia_engine_sparse_moe::dsv4::model::DsV4Model| {
        let mut next = 0usize;
        for (pos, &t) in prompt.iter().enumerate() {
            let copies = model.embed_ids(&[t]).pop().unwrap();
            let copies = model.forward_layers_decode(copies, pos, Some(t));
            next = argmax(&model.logits(&copies));
        }
        let mut got = vec![next];
        for step in 1..want.len() {
            let pos = prompt.len() + step - 1;
            let id = got[step - 1];
            let copies = model.embed_ids(&[id]).pop().unwrap();
            let copies = model.forward_layers_decode(copies, pos, Some(id));
            got.push(argmax(&model.logits(&copies)));
        }
        got
    };

    let got1 = run(&mut model);
    assert_eq!(got1, want, "token-by-token drive diverges from reference");
    model.reset();
    let got2 = run(&mut model);
    assert_eq!(got2, want, "second run after reset() diverges");
}

/// Dsv4Runner is what the sparse-moe Builder constructs when the manifest
/// says arch = "deepseek_v4". Single-stage generate_argmax must reproduce
/// the reference tokens end-to-end (tokenizer-free: raw ids).
#[test]
fn dsv4_runner_single_stage_matches_reference() {
    use cascadia_engine_sparse_moe::dsv4::stage::Dsv4Runner;
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_export");
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
            .unwrap();
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let want: Vec<u32> = reference["generated"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let mut r = Dsv4Runner::load_staged(&dir, 64, 0, 1, 0, 0).expect("load single stage");
    let got = r.generate_argmax(&prompt, want.len());
    assert_eq!(got, want, "runner single-stage greedy diverges");
}

/// Two Dsv4Runner ranks driven exactly like the wire protocol does (one
/// token per round; ids only on rank 0) must match the reference.
#[test]
fn dsv4_runner_two_rank_matches_reference() {
    use cascadia_engine_sparse_moe::dsv4::stage::Dsv4Runner;
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_export");
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
            .unwrap();
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let want: Vec<u32> = reference["generated"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let mut r0 = Dsv4Runner::load_staged(&dir, 64, 0, 2, 0, 0).expect("rank 0");
    let mut r1 = Dsv4Runner::load_staged(&dir, 64, 1, 2, 0, 0).expect("rank 1");
    r0.reset();
    r1.reset();

    let fwd = |r0: &mut Dsv4Runner, r1: &mut Dsv4Runner, tok: u32, pos: usize| -> usize {
        let h = r0.embed_token(tok);
        let h = r0.forward_layers(h, pos, Some(tok)); // ids stay on rank 0
        let h = r1.forward_layers(h, pos, None);
        argmax(&r1.head_logits(&h))
    };

    let mut pos = 0usize;
    let mut next = 0usize;
    for &t in &prompt {
        next = fwd(&mut r0, &mut r1, t, pos);
        pos += 1;
    }
    let mut got: Vec<u32> = vec![next as u32];
    for _ in 1..want.len() {
        next = fwd(&mut r0, &mut r1, next as u32, pos);
        pos += 1;
        got.push(next as u32);
    }
    assert_eq!(got, want, "runner two-rank greedy diverges");
}
