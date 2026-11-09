use super::*;
use llguidance::toktrie::ApproximateTokEnv;

/// The single-byte test vocab: token id `b` is byte `b`; 256..=261 are
/// special tokens, 261 (`<|end|>`) is its EOS.
const SINGLE_BYTE_EOS: usize = 261;

const WEATHER: &str = r#"{"type":"object","properties":{"city":{"type":"string"},"temperature_celsius":{"type":"number"},"conditions":{"type":"string","enum":["sunny","cloudy","rainy"]}},"required":["city","temperature_celsius","conditions"],"additionalProperties":false}"#;
const WEATHER_OK: &str = r#"{"city":"Paris","temperature_celsius":18,"conditions":"sunny"}"#;

const BROWSER_USE_SCHEMA: &str =
    include_str!("../../cascadia-engine-mock/tests/fixtures/browser_use_schema.json");
const PYDANTIC_OUTPUT_SCHEMA: &str =
    include_str!("../../cascadia-engine-mock/tests/fixtures/pydantic_output_schema.json");

fn single_byte() -> GrammarFactory {
    GrammarFactory::from_tok_env(
        ApproximateTokEnv::single_byte_env(),
        vec![SINGLE_BYTE_EOS as u32].into(),
    )
    .unwrap()
}

fn spec(body: &str) -> GrammarSpec {
    GrammarSpec {
        kind: GrammarKind::JsonSchema,
        body: body.to_string(),
    }
}

fn mask_for(gf: &GrammarFactory, body: &str) -> GrammarMask {
    gf.compile(&spec(body)).unwrap().matcher().unwrap()
}

/// Feeds `tokens` one at a time, each checked against the mask first.
/// `Err(i)` = token `i` was masked out.
fn walk(gf: &GrammarFactory, gm: &mut GrammarMask, tokens: &[u32]) -> Result<(), usize> {
    for (i, &t) in tokens.iter().enumerate() {
        let mut row = vec![0.0f32; gf.mask_width()];
        if gm.apply(&mut row).is_err() || !row[t as usize].is_finite() {
            return Err(i);
        }
        gm.accept(t).unwrap();
    }
    Ok(())
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

/// Whole document accepted and the grammar says nothing may follow.
fn accepts(gf: &GrammarFactory, body: &str, doc: &str) -> bool {
    let mut gm = mask_for(gf, body);
    walk(gf, &mut gm, &bytes(doc)).is_ok() && gm.finished()
}

fn masked_row(gf: &GrammarFactory, gm: &mut GrammarMask) -> Vec<f32> {
    let mut row = vec![0.0f32; gf.mask_width()];
    gm.apply(&mut row).unwrap();
    row
}

#[test]
fn conforming_json_is_accepted() {
    let gf = single_byte();
    assert!(accepts(&gf, WEATHER, WEATHER_OK));
}

#[test]
fn enum_value_outside_the_list_is_rejected() {
    let gf = single_byte();
    assert!(!accepts(
        &gf,
        WEATHER,
        r#"{"city":"Paris","temperature_celsius":18,"conditions":"stormy"}"#
    ));
}

#[test]
fn extra_property_is_rejected_when_additional_properties_is_false() {
    let gf = single_byte();
    assert!(!accepts(
        &gf,
        WEATHER,
        r#"{"city":"Paris","temperature_celsius":18,"conditions":"sunny","x":1}"#
    ));
}

#[test]
fn missing_required_property_is_rejected() {
    let gf = single_byte();
    assert!(!accepts(
        &gf,
        WEATHER,
        r#"{"city":"Paris","conditions":"sunny"}"#
    ));
}

#[test]
fn first_allowed_token_is_open_brace() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    let row = masked_row(&gf, &mut gm);
    let allowed: Vec<usize> = (0..row.len()).filter(|&i| row[i].is_finite()).collect();
    assert_eq!(allowed, vec![b'{' as usize]);
}

#[test]
fn whitespace_after_colon_is_masked() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    walk(&gf, &mut gm, &bytes(r#"{"city":"#)).unwrap();
    let row = masked_row(&gf, &mut gm);
    for ws in [b' ', b'\n', b'\t', b'\r'] {
        assert_eq!(row[ws as usize], f32::NEG_INFINITY, "byte {ws} allowed");
    }
    assert!(row[b'"' as usize].is_finite());
}

#[test]
fn stopped_only_after_the_closing_brace() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    let doc = bytes(WEATHER_OK);
    let (body, last) = doc.split_at(doc.len() - 1);
    walk(&gf, &mut gm, body).unwrap();
    assert!(!gm.finished());
    walk(&gf, &mut gm, last).unwrap();
    assert!(gm.finished());
}

#[test]
fn replay_of_a_valid_prefix_succeeds() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    gm.replay(&bytes(r#"{"city":"Pa"#)).unwrap();
    assert!(gm.allowed(u32::from(b'r')));
    walk(
        &gf,
        &mut gm,
        &bytes(r#"ris","temperature_celsius":18,"conditions":"sunny"}"#),
    )
    .unwrap();
    assert!(gm.finished());
}

#[test]
fn replay_of_an_invalid_prefix_is_a_backend_error() {
    // The server produced a resumed prefix, so a mismatch is not the client's fault.
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    let err = gm.replay(&bytes(r#"{"x"#)).unwrap_err();
    assert!(matches!(err, EngineError::Backend(_)), "{err:?}");
}

#[test]
fn eos_after_a_closed_object_keeps_the_finish() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    walk(&gf, &mut gm, &bytes(WEATHER_OK)).unwrap();
    assert!(gm.finished());
    let row = masked_row(&gf, &mut gm);
    assert!(row[SINGLE_BYTE_EOS].is_finite());
    gm.accept(SINGLE_BYTE_EOS as u32).unwrap();
    assert!(gm.finished());
}

#[test]
fn eos_ends_a_value_that_could_continue() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, r#"{"type":"integer"}"#);
    walk(&gf, &mut gm, &bytes("12")).unwrap();
    assert!(!gm.finished());
    gm.accept(SINGLE_BYTE_EOS as u32).unwrap();
    assert!(gm.finished());
}

#[test]
fn word_wise_mask_matches_bit_by_bit() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    for &t in &bytes(r#"{"city":"P"#) {
        let mut row: Vec<f32> = (0..gf.mask_width() + 37).map(|i| i as f32).collect();
        gm.apply(&mut row).unwrap();
        for (i, v) in row.iter().enumerate() {
            let want = i < gf.mask_width() && gm.allowed(i as u32);
            assert_eq!(v.is_finite(), want, "index {i}");
        }
        gm.accept(t).unwrap();
    }
}

#[test]
fn eos_is_allowed_only_once_the_json_is_complete() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    for &t in &bytes(WEATHER_OK) {
        let row = masked_row(&gf, &mut gm);
        assert_eq!(row[SINGLE_BYTE_EOS], f32::NEG_INFINITY);
        gm.accept(t).unwrap();
    }
    let row = masked_row(&gf, &mut gm);
    assert!(row[SINGLE_BYTE_EOS].is_finite());
    assert!(gm.allowed(SINGLE_BYTE_EOS as u32));
}

#[test]
fn eos_is_allowed_where_the_value_may_end_or_continue() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, r#"{"type":"integer"}"#);
    walk(&gf, &mut gm, &bytes("12")).unwrap();
    assert!(!gm.finished());
    let row = masked_row(&gf, &mut gm);
    assert!(row[SINGLE_BYTE_EOS].is_finite());
    assert!(row[b'3' as usize].is_finite());
}

#[test]
fn padded_row_is_masked_past_the_vocab() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    let mut row = vec![0.0f32; gf.mask_width() + 40];
    gm.apply(&mut row).unwrap();
    assert!(row[gf.mask_width()..]
        .iter()
        .all(|&v| v == f32::NEG_INFINITY));
    assert!(row[b'{' as usize].is_finite());
}

#[test]
fn apply_fails_when_no_index_of_the_row_is_allowed() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    // '{' (123) is the only allowed token and lies past this short row.
    let mut row = vec![0.0f32; 100];
    let err = gm.apply(&mut row).unwrap_err();
    assert!(matches!(err, EngineError::Backend(_)), "{err:?}");
}

#[test]
fn client_x_guidance_is_replaced() {
    let gf = single_byte();
    let body = r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false,"x-guidance":{"whitespace_flexible":true,"key_separator":": "}}"#;
    let mut gm = mask_for(&gf, body);
    walk(&gf, &mut gm, &bytes(r#"{"city""#)).unwrap();
    let row = masked_row(&gf, &mut gm);
    assert_eq!(row[b' ' as usize], f32::NEG_INFINITY);
    assert!(row[b':' as usize].is_finite());
    walk(&gf, &mut gm, &bytes(":")).unwrap();
    let row = masked_row(&gf, &mut gm);
    assert_eq!(row[b' ' as usize], f32::NEG_INFINITY);
}

#[test]
fn not_keyword_compiles_with_a_warning() {
    let gf = single_byte();
    let compiled = gf
        .compile(&spec(r#"{"type":"object","not":{"required":["a"]}}"#))
        .unwrap();
    assert!(
        compiled.warnings().iter().any(|w| w.contains("\"not\"")),
        "{:?}",
        compiled.warnings()
    );
}

#[test]
fn unique_items_and_unknown_format_compile_with_warnings() {
    let gf = single_byte();
    let body = r#"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"},"uniqueItems":true},"blob":{"type":"string","format":"binary"}},"required":["tags","blob"],"additionalProperties":false}"#;
    let compiled = gf.compile(&spec(body)).unwrap();
    let w = compiled.warnings();
    assert!(w.iter().any(|w| w.contains("uniqueItems")), "{w:?}");
    assert!(w.iter().any(|w| w.contains("binary")), "{w:?}");
    let mut gm = compiled.matcher().unwrap();
    walk(&gf, &mut gm, &bytes(r#"{"tags":["a","a"],"blob":"x"}"#)).unwrap();
    assert!(gm.finished());
}

#[test]
fn clean_schema_has_no_warnings() {
    let gf = single_byte();
    assert!(gf.compile(&spec(WEATHER)).unwrap().warnings().is_empty());
}

#[test]
fn pydantic_discriminated_union_compiles_and_accepts_each_branch() {
    let gf = single_byte();
    let body = r##"{"$defs":{"Cat":{"properties":{"pet_type":{"const":"cat","title":"Pet Type","type":"string"},"meows":{"title":"Meows","type":"integer"}},"required":["pet_type","meows"],"title":"Cat","type":"object"},"Dog":{"properties":{"pet_type":{"const":"dog","title":"Pet Type","type":"string"},"barks":{"title":"Barks","type":"number"}},"required":["pet_type","barks"],"title":"Dog","type":"object"}},"properties":{"pet":{"discriminator":{"mapping":{"cat":"#/$defs/Cat","dog":"#/$defs/Dog"},"propertyName":"pet_type"},"oneOf":[{"$ref":"#/$defs/Cat"},{"$ref":"#/$defs/Dog"}],"title":"Pet"}},"required":["pet"],"title":"Model","type":"object"}"##;
    assert!(accepts(
        &gf,
        body,
        r#"{"pet":{"pet_type":"cat","meows":3}}"#
    ));
    assert!(accepts(
        &gf,
        body,
        r#"{"pet":{"pet_type":"dog","barks":1.5}}"#
    ));
    assert!(!accepts(
        &gf,
        body,
        r#"{"pet":{"pet_type":"cow","meows":3}}"#
    ));
}

#[test]
fn broken_schemas_are_invalid_config() {
    let gf = single_byte();
    for body in [
        r#"{"type":42}"#,
        r##"{"type":"object","properties":{"a":{"$ref":"#/$defs/Missing"}},"required":["a"]}"##,
    ] {
        let err = gf.compile(&spec(body)).err();
        let Some(EngineError::InvalidConfig(msg)) = err else {
            panic!("{body}: {err:?}");
        };
        assert!(!msg.contains('\n'), "{msg}");
        assert!(msg.chars().count() <= MAX_ERROR_CHARS, "{msg}");
    }
}

#[test]
fn rejected_token_is_an_error_and_not_a_finish() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    assert!(gm.accept(u32::from(b'x')).is_err());
    assert!(!gm.finished());
    let mut row = vec![0.0f32; gf.mask_width()];
    assert!(gm.apply(&mut row).is_err());
    assert!(gm.accept(u32::from(b'{')).is_err());
}

#[test]
fn malformed_schema_text_is_invalid_config() {
    let gf = single_byte();
    for body in [r#"{"type":"#, "[1,2]"] {
        assert!(
            matches!(gf.compile(&spec(body)), Err(EngineError::InvalidConfig(_))),
            "{body}"
        );
    }
}

#[test]
fn error_messages_are_one_line_and_capped() {
    let long = format!("first line\n\n  second\tline {}", "x".repeat(1000));
    let msg = clean_message(&long);
    assert!(msg.starts_with("first line second line x"), "{msg}");
    assert_eq!(msg.chars().count(), MAX_ERROR_CHARS);
    assert_eq!(
        clean_message("é".repeat(400).as_str()).chars().count(),
        MAX_ERROR_CHARS
    );
}

#[test]
fn browser_use_schema_compiles_and_accepts_an_instance() {
    let gf = single_byte();
    let doc = r#"{"thinking":"t","evaluation_previous_goal":"e","memory":"m","next_goal":"n","action":[{"click_element_by_index":null,"input_text":null,"done":{"text":"ok","success":true}}]}"#;
    assert!(accepts(&gf, BROWSER_USE_SCHEMA.trim(), doc));
}

#[test]
fn pydantic_schema_compiles_and_accepts_an_instance() {
    let gf = single_byte();
    let doc = r#"{"name":"A","age":3,"email":null,"address":{"street":"s","city":"c","country":"NP"},"tags":[]}"#;
    assert!(accepts(&gf, PYDANTIC_OUTPUT_SCHEMA.trim(), doc));
}

#[test]
fn matchers_from_one_compiled_grammar_are_independent() {
    let gf = single_byte();
    let compiled = gf.compile(&spec(WEATHER)).unwrap();
    let mut a = compiled.matcher().unwrap();
    walk(&gf, &mut a, &bytes(r#"{"city""#)).unwrap();
    let mut b = compiled.matcher().unwrap();
    let row = masked_row(&gf, &mut b);
    assert!(row[b'{' as usize].is_finite());
    assert_eq!(row[b':' as usize], f32::NEG_INFINITY);
}

#[test]
fn grammar_types_can_move_across_threads() {
    fn send_sync<T: Send + Sync>() {}
    fn send<T: Send>() {}
    send_sync::<GrammarFactory>();
    send_sync::<CompiledGrammar>();
    send::<GrammarMask>();
}

// ---- a real Hugging Face tokenizer (tiny byte-level BPE built in-test) ----

/// GPT-2 byte-to-char table used by byte-level BPE vocabularies.
fn byte_char(b: u8) -> char {
    let self_mapped = |c: u32| {
        (0x21..=0x7E).contains(&c) || (0xA1..=0xAC).contains(&c) || (0xAE..=0xFF).contains(&c)
    };
    if self_mapped(u32::from(b)) {
        return char::from(b);
    }
    let shifted = (0..b).filter(|&x| !self_mapped(u32::from(x))).count() as u32;
    char::from_u32(0x100 + shifted).unwrap()
}

const OPEN_QUOTE: u32 = 256; // `{"`
const QUOTE_COLON: u32 = 257; // `":`
const COLON_SPACE: u32 = 258; // `: `
const ENDOFTEXT: u32 = 259;
const IM_END: u32 = 260;
const HF_VOCAB: usize = 261;

/// Byte `b` is token `b`, plus three merges and two special tokens.
fn tiny_bpe() -> tokenizers::Tokenizer {
    let mut vocab = serde_json::Map::new();
    for b in 0..=255u8 {
        vocab.insert(byte_char(b).to_string(), u32::from(b).into());
    }
    let merged = |a: u8, b: u8| format!("{}{}", byte_char(a), byte_char(b));
    vocab.insert(merged(b'{', b'"'), OPEN_QUOTE.into());
    vocab.insert(merged(b'"', b':'), QUOTE_COLON.into());
    vocab.insert(merged(b':', b' '), COLON_SPACE.into());
    let merges: Vec<String> = [(b'{', b'"'), (b'"', b':'), (b':', b' ')]
        .iter()
        .map(|&(a, b)| format!("{} {}", byte_char(a), byte_char(b)))
        .collect();
    let added = |id: u32, content: &str| {
        serde_json::json!({"id": id, "content": content, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true})
    };
    let byte_level = serde_json::json!({"type": "ByteLevel", "add_prefix_space": false,
        "trim_offsets": true, "use_regex": true});
    let json = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [added(ENDOFTEXT, "<|endoftext|>"), added(IM_END, "<|im_end|>")],
        "normalizer": null, "pre_tokenizer": byte_level, "post_processor": null,
        "decoder": byte_level,
        "model": {"type": "BPE", "dropout": null, "unk_token": null,
            "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false,
            "byte_fallback": false, "vocab": vocab, "merges": merges},
    });
    tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
}

#[test]
fn hf_factory_needs_an_eos_inside_the_vocab() {
    let tok = tiny_bpe();
    for eos in [&[][..], &[9_999], &[HF_VOCAB as u32]] {
        let err = GrammarFactory::from_tokenizer(&tok, eos).err();
        assert!(
            matches!(err, Some(EngineError::InvalidConfig(_))),
            "{eos:?}: {err:?}"
        );
    }
}

#[test]
fn hf_tokenizer_masks_with_merged_tokens_and_every_eos() {
    let tok = tiny_bpe();
    let gf = GrammarFactory::from_tokenizer(&tok, &[ENDOFTEXT, IM_END, 99_999]).unwrap();
    assert_eq!(gf.mask_width(), HF_VOCAB);
    let mut gm = mask_for(
        &gf,
        r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}"#,
    );

    // Forced text is held to its canonical tokenization: `{"` as the merged
    // token, not `{` then `"`.
    let row = masked_row(&gf, &mut gm);
    assert!(row[OPEN_QUOTE as usize].is_finite());
    assert_eq!(row[b'{' as usize], f32::NEG_INFINITY);
    assert_eq!(row[b' ' as usize], f32::NEG_INFINITY);

    let mut prefix = vec![OPEN_QUOTE];
    prefix.extend(bytes("city"));
    walk(&gf, &mut gm, &prefix).unwrap();
    let row = masked_row(&gf, &mut gm);
    assert!(row[QUOTE_COLON as usize].is_finite());
    assert_eq!(row[b'"' as usize], f32::NEG_INFINITY);
    gm.accept(QUOTE_COLON).unwrap();

    // compact JSON: no space after the colon, so the `: ` merge can't come either.
    let row = masked_row(&gf, &mut gm);
    assert_eq!(row[b' ' as usize], f32::NEG_INFINITY);
    assert_eq!(row[COLON_SPACE as usize], f32::NEG_INFINITY);

    let mut padded = vec![0.0f32; 300];
    gm.apply(&mut padded).unwrap();
    assert!(padded[HF_VOCAB..].iter().all(|&v| v == f32::NEG_INFINITY));

    walk(&gf, &mut gm, &bytes(r#""Paris""#)).unwrap();
    let row = masked_row(&gf, &mut gm);
    assert_eq!(row[ENDOFTEXT as usize], f32::NEG_INFINITY);
    assert_eq!(row[IM_END as usize], f32::NEG_INFINITY);
    walk(&gf, &mut gm, &bytes("}")).unwrap();
    assert!(gm.finished());
    let row = masked_row(&gf, &mut gm);
    assert!(row[ENDOFTEXT as usize].is_finite());
    assert!(row[IM_END as usize].is_finite());
}
