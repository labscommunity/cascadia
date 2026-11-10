use super::*;
use llguidance::toktrie::ApproximateTokEnv;

thread_local! {
    /// Schema builds (`create_parser` on a whole schema) on this thread.
    pub(super) static SCHEMA_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

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

/// The tokens of `s`, written compact in the tests, as the grammar wants
/// it: with `json.dumps` separators (a space after each `:` and `,` outside
/// strings). Prefixes that stop inside a string work too.
fn bytes(s: &str) -> Vec<u32> {
    raw_bytes(&dumps(s))
}

/// The tokens of `s` exactly as written.
fn raw_bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

/// Compact JSON text `s` with `json.dumps` separators.
fn dumps(s: &str) -> String {
    let (mut out, mut in_str, mut esc) = (String::new(), false, false);
    for c in s.chars() {
        out.push(c);
        if in_str {
            match (esc, c) {
                (true, _) => esc = false,
                (false, '\\') => esc = true,
                (false, '"') => in_str = false,
                _ => {}
            }
        } else if c == '"' {
            in_str = true;
        } else if c == ':' || c == ',' {
            out.push(' ');
        }
    }
    out
}

/// Whole document (written compact, see [`bytes`]) accepted and the
/// grammar says nothing may follow.
fn accepts(gf: &GrammarFactory, body: &str, doc: &str) -> bool {
    accepts_raw(gf, body, &dumps(doc))
}

/// [`accepts`] on `doc` exactly as written.
fn accepts_raw(gf: &GrammarFactory, body: &str, doc: &str) -> bool {
    let mut gm = mask_for(gf, body);
    walk(gf, &mut gm, &raw_bytes(doc)).is_ok() && gm.finished()
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
fn exactly_one_space_follows_a_colon_or_comma() {
    let gf = single_byte();
    let only_space = |row: &[f32]| {
        let allowed: Vec<usize> = (0..row.len()).filter(|&i| row[i].is_finite()).collect();
        assert_eq!(allowed, vec![b' ' as usize]);
    };
    let mut gm = mask_for(&gf, WEATHER);
    walk(&gf, &mut gm, &raw_bytes(r#"{"city":"#)).unwrap();
    only_space(&masked_row(&gf, &mut gm));
    walk(&gf, &mut gm, &raw_bytes(" ")).unwrap();
    let row = masked_row(&gf, &mut gm);
    for ws in [b' ', b'\n', b'\t', b'\r'] {
        assert_eq!(row[ws as usize], f32::NEG_INFINITY, "byte {ws} allowed");
    }
    assert!(row[b'"' as usize].is_finite());
    walk(&gf, &mut gm, &raw_bytes(r#""Paris","#)).unwrap();
    only_space(&masked_row(&gf, &mut gm));
    walk(&gf, &mut gm, &raw_bytes(" ")).unwrap();
    let row = masked_row(&gf, &mut gm);
    assert_eq!(row[b' ' as usize], f32::NEG_INFINITY);
    assert!(row[b'"' as usize].is_finite());
    // No whitespace anywhere else: not after `{`, not before `:`.
    let mut gm = mask_for(&gf, WEATHER);
    assert_eq!(masked_row(&gf, &mut gm)[b' ' as usize], f32::NEG_INFINITY);
    walk(&gf, &mut gm, &raw_bytes(r#"{"#)).unwrap();
    assert_eq!(masked_row(&gf, &mut gm)[b' ' as usize], f32::NEG_INFINITY);
    walk(&gf, &mut gm, &raw_bytes(r#""city""#)).unwrap();
    assert_eq!(masked_row(&gf, &mut gm)[b' ' as usize], f32::NEG_INFINITY);
    // The compact form is refused.
    assert!(!accepts_raw(&gf, WEATHER, WEATHER_OK));
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
        // Starts inside a string, so written out as the grammar wants it.
        &raw_bytes(r#"ris", "temperature_celsius": 18, "conditions": "sunny"}"#),
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

    // json.dumps separators: the space after the colon comes next (the `: `
    // merge can't, the colon is already in `":`), then no second one.
    let row = masked_row(&gf, &mut gm);
    assert!(row[b' ' as usize].is_finite());
    assert_eq!(row[COLON_SPACE as usize], f32::NEG_INFINITY);
    assert_eq!(row[b'"' as usize], f32::NEG_INFINITY);
    gm.accept(u32::from(b' ')).unwrap();
    assert_eq!(masked_row(&gf, &mut gm)[b' ' as usize], f32::NEG_INFINITY);

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

/// The allowed set read back through `apply` on a zero row: what callers did
/// before `mask_words`.
fn words_via_apply(gm: &mut GrammarMask, width: usize) -> EngineResult<Vec<u32>> {
    let mut row = vec![0.0f32; width];
    gm.apply(&mut row)?;
    let mut words = vec![0u32; width.div_ceil(32)];
    for (i, v) in row.iter().enumerate() {
        if v.is_finite() {
            words[i / 32] |= 1 << (i % 32);
        }
    }
    Ok(words)
}

#[test]
fn mask_words_are_the_allowed_set_bit_for_bit() {
    let gf = single_byte();
    let width = gf.mask_width();
    let mut gm = mask_for(&gf, WEATHER);
    // Start, inside a key, inside a string, after a number, finished.
    let mut checked = 0;
    for (i, &t) in bytes(WEATHER_OK).iter().enumerate() {
        if [0, 3, 10, 40].contains(&i) {
            let words = gm.mask_words().unwrap();
            assert_eq!(words.len(), width.div_ceil(32));
            for tok in 0..words.len() * 32 {
                let bit = words[tok / 32] >> (tok % 32) & 1 == 1;
                assert_eq!(bit, tok < width && gm.allowed(tok as u32), "token {tok}");
            }
            assert_eq!(words, words_via_apply(&mut gm, width).unwrap());
            checked += 1;
        }
        gm.accept(t).unwrap();
    }
    assert_eq!(checked, 4);
    assert!(gm.finished());
    let done = gm.mask_words().unwrap();
    assert_eq!(done, words_via_apply(&mut gm, width).unwrap());
    let eos = SINGLE_BYTE_EOS;
    assert_eq!(done[eos / 32] >> (eos % 32) & 1, 1);
}

#[test]
fn mask_words_fail_in_the_error_state() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    assert!(gm.accept(u32::from(b'x')).is_err());
    let err = gm.mask_words().unwrap_err();
    assert!(matches!(err, EngineError::Backend(_)), "{err:?}");
}

#[test]
fn greedy_pick_takes_the_best_allowed_token() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, r#"{"type":"integer"}"#);
    walk(&gf, &mut gm, &bytes("12")).unwrap();
    // Digits and EOS are allowed; the best raw logit ('a') is not.
    let mut row = vec![0.0f32; gf.mask_width()];
    row[b'a' as usize] = 9.0;
    row[b'7' as usize] = 3.0;
    row[b'4' as usize] = 3.0;
    assert_eq!(
        gm.pick_greedy(&row).unwrap(),
        b'4' as usize,
        "first on ties"
    );
    let words = gm.mask_words().unwrap();
    assert_eq!(argmax_allowed(&row, &words), Some(b'4' as usize));
}

#[test]
fn an_all_non_finite_row_picks_the_lowest_allowed_token_never_a_disallowed_eos() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    // Only '{' may come first; EOS (the last index) is disallowed.
    for fill in [f32::NEG_INFINITY, f32::NAN] {
        let row = vec![fill; gf.mask_width()];
        assert_eq!(gm.pick_greedy(&row).unwrap(), b'{' as usize);
        let words = gm.mask_words().unwrap();
        assert_eq!(argmax_allowed(&row, &words), Some(b'{' as usize));
    }
    // NaN at an allowed index never beats a finite allowed logit.
    let mut gm = mask_for(&gf, r#"{"type":"integer"}"#);
    walk(&gf, &mut gm, &bytes("1")).unwrap();
    let mut row = vec![f32::NEG_INFINITY; gf.mask_width()];
    row[b'0' as usize] = f32::NAN;
    row[b'5' as usize] = -1.0;
    assert_eq!(gm.pick_greedy(&row).unwrap(), b'5' as usize);
}

#[test]
fn a_pick_with_nothing_allowed_in_the_row_is_an_error() {
    let gf = single_byte();
    let mut gm = mask_for(&gf, WEATHER);
    // '{' (123) lies past this short row.
    assert!(gm.pick_greedy(&[0.0; 100]).is_err());
    let words = gm.mask_words().unwrap();
    assert_eq!(argmax_allowed(&[0.0; 100], &words), None);
    assert_eq!(argmax_allowed(&[0.0; 4], &[]), None);
}

/// zod 4 before 4.6 emits this look-ahead pattern for `z.email()`; the
/// regex engine has no look-around.
const ZOD_EMAIL: &str = r#"{"type":"object","properties":{"email":{"type":"string","format":"email","pattern":"^(?!\\.)(?!.*\\.\\.)([A-Za-z0-9_'+\\-\\.]*)[A-Za-z0-9_+-]@([A-Za-z0-9][A-Za-z0-9\\-]*\\.)+[A-Za-z]{2,}$"}},"required":["email"],"additionalProperties":false}"#;

#[test]
fn unparseable_pattern_is_dropped_with_a_warning_and_format_still_applies() {
    let gf = single_byte();
    let compiled = gf.compile(&spec(ZOD_EMAIL)).unwrap();
    let w = compiled.warnings();
    assert!(
        w.iter().any(|w| w.contains("pattern") && w.contains("(?!")),
        "{w:?}"
    );
    assert!(accepts(&gf, ZOD_EMAIL, r#"{"email":"ada@example.com"}"#));
    // `format: email` is still enforced.
    assert!(!accepts(&gf, ZOD_EMAIL, r#"{"email":"not an email"}"#));
}

#[test]
fn unparseable_pattern_property_key_is_dropped_with_a_warning() {
    let gf = single_byte();
    let body = r#"{"type":"object","patternProperties":{"^(?!x)[a-z]+$":{"type":"integer"}},"properties":{"id":{"type":"string","pattern":"^[0-9]+$"}}}"#;
    let compiled = gf.compile(&spec(body)).unwrap();
    assert!(
        compiled.warnings().iter().any(|w| w.contains("(?!x)")),
        "{:?}",
        compiled.warnings()
    );
    // The parseable pattern beside it is still enforced.
    assert!(accepts(&gf, body, r#"{"id":"42"}"#));
    assert!(!accepts(&gf, body, r#"{"id":"4a"}"#));
}

#[test]
fn a_valid_pattern_is_still_enforced() {
    let gf = single_byte();
    let body = r#"{"type":"object","properties":{"code":{"type":"string","pattern":"^[A-Z]{3}$"}},"required":["code"],"additionalProperties":false}"#;
    let compiled = gf.compile(&spec(body)).unwrap();
    assert!(compiled.warnings().is_empty(), "{:?}", compiled.warnings());
    assert!(accepts(&gf, body, r#"{"code":"ABC"}"#));
    assert!(!accepts(&gf, body, r#"{"code":"abc"}"#));
}

#[test]
fn byte_level_factory_is_shared_and_compiles() {
    let a = GrammarFactory::byte_level();
    let b = GrammarFactory::byte_level();
    assert!(Arc::ptr_eq(&a.factory, &b.factory));
    assert_eq!(a.mask_width(), SINGLE_BYTE_EOS + 1);
    assert!(accepts(&a, WEATHER, WEATHER_OK));
}

#[test]
fn many_unparseable_patterns_cost_two_builds() {
    let gf = single_byte();
    let props: serde_json::Map<String, serde_json::Value> = (0..200)
        .map(|i| {
            (
                format!("p{i}"),
                serde_json::json!({"type": "string", "pattern": format!("^(?!x{i})[a-z]+$")}),
            )
        })
        .collect();
    let body = serde_json::json!({"type": "object", "properties": props}).to_string();
    let before = SCHEMA_BUILDS.with(|c| c.get());
    let compiled = gf.compile(&spec(&body)).unwrap();
    assert_eq!(SCHEMA_BUILDS.with(|c| c.get()) - before, 2);
    assert_eq!(compiled.warnings().len(), 200);
}

#[test]
fn backreference_and_look_behind_patterns_are_dropped() {
    let gf = single_byte();
    for p in [r"^(a)\\1$", r"(?<=a)b", r"(?<!a)b", r"(?=a)a"] {
        let body = format!(r#"{{"type":"string","pattern":"{p}"}}"#);
        let compiled = gf.compile(&spec(&body)).unwrap();
        assert_eq!(
            compiled.warnings().len(),
            1,
            "{p}: {:?}",
            compiled.warnings()
        );
    }
}

#[test]
fn runtime_grammar_errors_hide_the_parser_state() {
    let gf = single_byte();
    let generic = |e: EngineError| {
        assert!(
            matches!(&e, EngineError::Backend(m) if m == "structured output generation failed"),
            "{e:?}"
        )
    };
    let mut gm = mask_for(&gf, WEATHER);
    generic(gm.accept(u32::from(b'x')).unwrap_err());
    let mut row = vec![0.0f32; gf.mask_width()];
    generic(gm.apply(&mut row).unwrap_err());
    let mut gm = mask_for(&gf, WEATHER);
    generic(gm.replay(&bytes(r#"{"x"#)).unwrap_err());
}

/// The bit-by-bit pick `argmax_allowed` must match.
fn argmax_allowed_bitwise(row: &[f32], words: &[u32]) -> Option<usize> {
    let mut lowest = None;
    let mut best: Option<(usize, f32)> = None;
    'words: for (w, &word) in words.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let t = w * 32 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let Some(&v) = row.get(t) else {
                break 'words;
            };
            lowest.get_or_insert(t);
            if v > f32::NEG_INFINITY && best.is_none_or(|(_, b)| v > b) {
                best = Some((t, v));
            }
        }
    }
    best.map(|(t, _)| t).or(lowest)
}

#[test]
fn word_wise_argmax_matches_bit_by_bit() {
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    for case in 0..3000 {
        let len = (next() % 300) as usize;
        let n_words = (next() % 12) as usize;
        let words: Vec<u32> = (0..n_words)
            .map(|_| match next() % 4 {
                0 => 0,
                1 => u32::MAX,
                2 => next() & next(),
                _ => next(),
            })
            .collect();
        // Few distinct values, so ties are common.
        let row: Vec<f32> = (0..len)
            .map(|_| match next() % 16 {
                0 => f32::NAN,
                1 => f32::NEG_INFINITY,
                2 => f32::INFINITY,
                _ => (next() % 5) as f32 - 2.0,
            })
            .collect();
        assert_eq!(
            argmax_allowed(&row, &words),
            argmax_allowed_bitwise(&row, &words),
            "case {case}: row {row:?} words {words:x?}"
        );
    }
}

// ---- a word-level tokenizer: one token per character, decoded by concatenation ----

/// Token `b` is the character U+00`b` (0..=255); `</s>` is the added EOS.
const CHAR_EOS: u32 = 256;

/// A word-level character vocab split one character per token, with a
/// `decoder` of `decoder` (`Fuse`: decoding concatenates the token strings).
fn char_word_level(decoder: serde_json::Value) -> tokenizers::Tokenizer {
    let mut vocab = serde_json::Map::new();
    for b in 0..=255u8 {
        vocab.insert(char::from(b).to_string(), u32::from(b).into());
    }
    let json = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [{"id": CHAR_EOS, "content": "</s>", "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
        "normalizer": null,
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "."},
            "behavior": "Isolated", "invert": false},
        "post_processor": null,
        "decoder": decoder,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "\u{0}"},
    });
    tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
}

#[test]
fn word_level_char_tokenizer_with_a_fuse_decoder_builds_and_masks() {
    let tok = char_word_level(serde_json::json!({"type": "Fuse"}));
    let gf = GrammarFactory::from_tokenizer(&tok, &[CHAR_EOS]).unwrap();
    assert_eq!(gf.mask_width(), CHAR_EOS as usize + 1);
    assert!(accepts(&gf, WEATHER, WEATHER_OK));
    // Without json.dumps separators, off the enum, or an extra key: rejected.
    assert!(!accepts_raw(&gf, WEATHER, WEATHER_OK));
    assert!(!accepts(
        &gf,
        WEATHER,
        r#"{"city":"Paris","temperature_celsius":18,"conditions":"snowy"}"#
    ));
    assert!(!accepts(
        &gf,
        WEATHER,
        r#"{"city":"Paris","temperature_celsius":"warm","conditions":"sunny"}"#
    ));

    // EOS only once the object is closed; the special token never inside it.
    let mut gm = mask_for(&gf, WEATHER);
    let doc = bytes(WEATHER_OK);
    walk(&gf, &mut gm, &doc[..doc.len() - 1]).unwrap();
    assert_eq!(
        masked_row(&gf, &mut gm)[CHAR_EOS as usize],
        f32::NEG_INFINITY
    );
    walk(&gf, &mut gm, &doc[doc.len() - 1..]).unwrap();
    assert!(gm.finished());
    assert!(masked_row(&gf, &mut gm)[CHAR_EOS as usize].is_finite());
}

#[test]
fn word_level_without_a_decoder_is_unsupported() {
    // `tokenizers` joins tokens with spaces when there is no decoder, so the
    // engine's text would not be the concatenation the grammar masks.
    let tok = char_word_level(serde_json::Value::Null);
    assert_eq!(
        tok.decode(&[u32::from(b'{'), u32::from(b'}')], true)
            .unwrap(),
        "{ }"
    );
    let err = GrammarFactory::from_tokenizer(&tok, &[CHAR_EOS]).err();
    assert!(
        matches!(&err, Some(EngineError::InvalidConfig(m)) if m.contains("unsupported")),
        "{err:?}"
    );
}

#[test]
fn word_level_with_a_decoder_that_rewrites_tokens_is_still_unsupported() {
    let tok = char_word_level(
        serde_json::json!({"type": "Metaspace", "replacement": "\u{2581}",
        "prepend_scheme": "always", "split": true}),
    );
    let err = GrammarFactory::from_tokenizer(&tok, &[CHAR_EOS]).err();
    assert!(
        matches!(&err, Some(EngineError::InvalidConfig(m)) if m.contains("unsupported")),
        "{err:?}"
    );
}

/// What llguidance does with a schema whose bounds no value meets: it
/// refuses each one at build time, naming the pair, rather than building a
/// grammar no output can finish. `contains` is not enforced, so
/// `minContains > maxContains` builds with a warning. An integer range
/// with no integer in it is refused too; one holding an integer builds. The API refuses all
/// of them but the last before any engine sees them: llama-server builds
/// a grammar for such a schema and then decodes until killed.
#[test]
fn unsatisfiable_bounds_are_refused_at_build() {
    let gf = single_byte();
    for (body, says) in [
        (
            r#"{"type":"string","minLength":5,"maxLength":2}"#,
            "minLength (5) is greater than maxLength (2)",
        ),
        (
            r#"{"type":"array","minItems":5,"maxItems":2}"#,
            "minItems (5) is greater than maxItems (2)",
        ),
        (
            r#"{"type":"object","minProperties":5,"maxProperties":2}"#,
            "minProperties > maxProperties",
        ),
        (
            r#"{"type":"number","minimum":5,"maximum":2}"#,
            "minimum (5) is greater than maximum (2)",
        ),
        (
            r#"{"type":"number","exclusiveMinimum":2,"exclusiveMaximum":2}"#,
            "exclusiveMinimum (2) is equal to exclusiveMaximum (2)",
        ),
        (
            r#"{"type":"number","minimum":2,"exclusiveMaximum":2}"#,
            "minimum (2) is equal to exclusiveMaximum (2)",
        ),
        (
            r#"{"type":"number","exclusiveMinimum":2,"maximum":2}"#,
            "exclusiveMinimum (2) is equal to maximum (2)",
        ),
        (
            r#"{"type":"object","properties":{"a":{"type":"string","minLength":5,"maxLength":2}},"required":["a"]}"#,
            "required property 'a' is unsatisfiable",
        ),
        // Integer ranges holding numbers but no integer.
        (
            r#"{"type":"integer","exclusiveMinimum":2,"exclusiveMaximum":3}"#,
            "integer interval (2, 3) contains no integer values",
        ),
        (
            r#"{"type":"integer","minimum":2.5,"maximum":2.9}"#,
            "integer interval [2.5, 2.9] contains no integer values",
        ),
        (
            r#"{"type":["integer"],"minimum":2.1,"exclusiveMaximum":3}"#,
            "integer interval [2.1, 3) contains no integer values",
        ),
    ] {
        let err = gf
            .compile(&spec(body))
            .err()
            .unwrap_or_else(|| panic!("{body} built"));
        let msg = err.to_string();
        assert!(msg.contains("Unsatisfiable schema"), "{body}: {msg}");
        assert!(msg.contains(says), "{body}: {msg}");
    }
    for body in [
        r#"{"type":"integer","exclusiveMinimum":2,"exclusiveMaximum":4}"#,
        r#"{"type":["integer","number"],"exclusiveMinimum":2,"exclusiveMaximum":3}"#,
    ] {
        gf.compile(&spec(body))
            .unwrap_or_else(|e| panic!("{body}: {e}"));
    }
    let contains = gf
        .compile(&spec(
            r#"{"type":"array","contains":{"type":"string"},"minContains":5,"maxContains":2}"#,
        ))
        .unwrap();
    assert!(
        contains
            .warnings()
            .iter()
            .any(|w| w.contains("minContains")),
        "{:?}",
        contains.warnings()
    );
}
