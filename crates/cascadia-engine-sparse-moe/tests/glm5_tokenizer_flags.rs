//! Pins the tokenizer property a deferred reasoning-split (finding literal
//! `</think>` in decoded text) will depend on: GLM-5's `</think>` is added-token
//! id 154842 with `special: false`, so `decode(.., skip_special_tokens = true)`
//! -- used at every decode site in `src/engine.rs` -- does not strip it. If a
//! future checkpoint ships `</think>` as `special: true`, the delimiter would
//! silently vanish from decoded text with no error and no other failing test.
//!
//! The fixture is a hand-authored, minimal `tokenizer.json` (not the real
//! 20MB GLM-5 file) carrying the real ids/`special` flags for the tokens that
//! matter, so this only pins the `tokenizers` crate's skip-special semantics
//! against that flag set -- it cannot catch a future checkpoint that flips
//! `special: true` on the real model, since nothing here reads the shipped
//! tokenizer. Catching that needs a load-time assertion node-side (follow-up,
//! not this test).

use std::path::PathBuf;

use tokenizers::Tokenizer;

fn load_fixture() -> Tokenizer {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tokenizer_flags/tokenizer.json");
    Tokenizer::from_file(&path).unwrap()
}

#[test]
fn think_close_tag_is_special_false_and_survives_skip_special_decode() {
    let tok = load_fixture();

    let ids = tok.encode("</think>", false).unwrap().get_ids().to_vec();
    assert_eq!(ids, vec![154842]);

    let text = tok.decode(&[154842], true).unwrap();
    assert!(text.contains("</think>"));
}

#[test]
fn tool_call_open_tag_survives_skip_special_decode() {
    let tok = load_fixture();

    let text = tok.decode(&[154843], true).unwrap();
    assert!(text.contains("<tool_call>"));
}

// Contrast case: a real `special: true` token (e.g. `<|endoftext|>`) IS
// stripped by the same skip_special_tokens=true decode. Without this, the two
// tests above could pass vacuously even if skip-special filtering did nothing.
#[test]
fn special_true_token_is_stripped_by_skip_special_decode() {
    let tok = load_fixture();

    let text = tok.decode(&[154820], true).unwrap();
    assert!(!text.contains("<|endoftext|>"));
}
