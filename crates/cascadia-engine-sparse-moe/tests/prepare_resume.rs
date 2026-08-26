use cascadia_engine_sparse_moe::prepare_resume;
use cascadia_types::task::GenerationTask;

fn test_tokenizer() -> tokenizers::Tokenizer {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tokenizer_flags/tokenizer.json"
    );
    tokenizers::Tokenizer::from_file(p).expect("checked-in fixture")
}

#[test]
fn prepare_resume_none_when_not_resuming() {
    let tok = test_tokenizer();
    let task = GenerationTask::new("t", "hi")
        .with_max_tokens(256)
        .with_temperature(0.7);
    let mut ids: Vec<i64> = vec![1, 2];
    let seed = prepare_resume(&task, &tok, &mut ids).unwrap();
    assert!(seed.is_none());
    assert_eq!(ids, vec![1, 2], "prompt ids untouched on a plain turn");
}

#[test]
fn prepare_resume_seeds_and_appends() {
    let tok = test_tokenizer();
    let mut task = GenerationTask::new("t", "hi")
        .with_max_tokens(256)
        .with_temperature(0.7);
    // Ids that DECODE in this fixture: 154841/154842 = "<think>"/"</think>",
    // both `special: false` so they survive skip_special_tokens=true. The
    // previous ids (5, 6) map to nothing in the WordLevel fixture, decoded to
    // "", and made the byte-length assertion below a vacuous 0 == 0.
    task.resume_token_ids = Some(vec![154841, 154842]);
    let mut ids: Vec<i64> = vec![1, 2];
    let seed = prepare_resume(&task, &tok, &mut ids).unwrap().unwrap();
    assert_eq!(
        ids,
        vec![1, 2, 154841, 154842],
        "resume ids appended after the prompt"
    );
    assert_eq!(seed.generated_u32, vec![154841u32, 154842]);
    assert_eq!(seed.seed_len, 2);
    // Literal, not the implementation's own expression. The WordLevel decoder
    // joins tokens with a space: "<think> </think>" = 16 bytes.
    assert_eq!(
        seed.emitted,
        "<think> </think>".len(),
        "emitted = prefix byte length"
    );
    assert_eq!(seed.emitted, 16);
}

#[test]
fn prepare_resume_rejects_out_of_vocab() {
    let tok = test_tokenizer();
    let mut task = GenerationTask::new("t", "hi")
        .with_max_tokens(256)
        .with_temperature(0.7);
    // One past the fixture's highest assigned id (154848, "</arg_key>") — the
    // bound is max_id + 1, NOT get_vocab_size(true) (the entry count, 7 here),
    // which would wrongly reject every real id in this gapped vocab.
    task.resume_token_ids = Some(vec![154849]);
    let mut ids: Vec<i64> = vec![1];
    assert!(prepare_resume(&task, &tok, &mut ids).is_err());
}
