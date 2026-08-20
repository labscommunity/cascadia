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
    task.resume_token_ids = Some(vec![5, 6]);
    let mut ids: Vec<i64> = vec![1, 2];
    let seed = prepare_resume(&task, &tok, &mut ids).unwrap().unwrap();
    assert_eq!(
        ids,
        vec![1, 2, 5, 6],
        "resume ids appended after the prompt"
    );
    assert_eq!(seed.generated_u32, vec![5u32, 6]);
    assert_eq!(seed.seed_len, 2);
    let prefix_text = tok.decode(&[5, 6], true).unwrap();
    assert_eq!(
        seed.emitted,
        prefix_text.len(),
        "emitted = prefix byte length"
    );
}

#[test]
fn prepare_resume_rejects_out_of_vocab() {
    let tok = test_tokenizer();
    let vocab = tok.get_vocab_size(true) as i32;
    let mut task = GenerationTask::new("t", "hi")
        .with_max_tokens(256)
        .with_temperature(0.7);
    task.resume_token_ids = Some(vec![vocab]); // one past the end
    let mut ids: Vec<i64> = vec![1];
    assert!(prepare_resume(&task, &tok, &mut ids).is_err());
}
