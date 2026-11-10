//! Decoding that keeps a model's tool-call marker tokens.
//!
//! The engines decode with special tokens skipped, which is right for
//! turn and control markers but erases a tool-call dialect built out of
//! special tokens. Gemma 4 writes a call as
//! `<|tool_call>call:NAME{key:<|"|>text<|"|>}<tool_call|>`, and all three
//! markers are special tokens, so a plain decode hands the API
//! `call:NAME{key:text}`, which it can no longer read as a call.
//!
//! [`tool_marker_ids`] finds those markers in a vocabulary once, at load;
//! [`decode_keeping`] decodes like `decode(ids, true)` but writes the listed
//! ids out as their literal text. A vocabulary without the markers gets an
//! empty list, and an empty list decodes exactly like before.

use tokenizers::Tokenizer;

/// Gemma 4's tool-call markers, in the order they are looked up.
const TOOL_MARKERS: [&str; 3] = ["<|tool_call>", "<tool_call|>", "<|\"|>"];

/// Gemma 4's tool-call markers in this vocabulary (`<|tool_call>`,
/// `<tool_call|>`, `<|"|>`), as `(id, text)`; empty when absent.
///
/// Only special added tokens count: a plain decode already writes out
/// anything that isn't special, so listing it would change nothing but
/// could change spacing.
pub fn tool_marker_ids(tok: &Tokenizer) -> Vec<(u32, String)> {
    let added = tok.get_added_tokens_decoder();
    TOOL_MARKERS
        .iter()
        .filter_map(|&text| {
            let id = tok.token_to_id(text)?;
            let token = added.get(&id)?;
            (token.special && token.content == text).then(|| (id, text.to_string()))
        })
        .collect()
}

/// `tok.decode(ids, true)`, except the listed ids are kept as their literal
/// text.
///
/// The ids are split at the kept ids; each run between them is decoded with
/// special tokens skipped and the kept ids' text is inserted between the
/// runs. With an empty `keep` this is exactly `tok.decode(ids, true)`.
pub fn decode_keeping(
    tok: &Tokenizer,
    ids: &[u32],
    keep: &[(u32, String)],
) -> tokenizers::Result<String> {
    if keep.is_empty() {
        return tok.decode(ids, true);
    }
    let mut text = String::new();
    let mut run_start = 0;
    for (i, id) in ids.iter().enumerate() {
        if let Some((_, literal)) = keep.iter().find(|(k, _)| k == id) {
            if run_start < i {
                text.push_str(&tok.decode(&ids[run_start..i], true)?);
            }
            text.push_str(literal);
            run_start = i + 1;
        }
    }
    if run_start < ids.len() {
        text.push_str(&tok.decode(&ids[run_start..], true)?);
    }
    Ok(text)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A WordLevel vocabulary over a few pieces of a Gemma 4 call, with the
    /// three tool-call markers and an end-of-turn marker as special added
    /// tokens. The `Fuse` decoder joins pieces without spaces, as Gemma's does.
    pub(crate) fn marker_tokenizer() -> Tokenizer {
        let json = r#"{"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[
              {"id":4,"content":"<|tool_call>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
              {"id":5,"content":"<tool_call|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
              {"id":6,"content":"<|\"|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
              {"id":7,"content":"<turn|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
            "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
            "post_processor":null,"decoder":{"type":"Fuse"},
            "model":{"type":"WordLevel",
                "vocab":{"a":0,"b":1,"call:f{x:":2,"}":3,"<|tool_call>":4,"<tool_call|>":5,"<|\"|>":6,"<turn|>":7},
                "unk_token":"a"}}"#;
        Tokenizer::from_bytes(json.as_bytes()).expect("build marker tokenizer")
    }

    /// The same vocabulary without any tool-call marker.
    fn plain_tokenizer() -> Tokenizer {
        let json = r#"{"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[
              {"id":2,"content":"<turn|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
            "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
            "post_processor":null,"decoder":{"type":"Fuse"},
            "model":{"type":"WordLevel","vocab":{"a":0,"b":1,"<turn|>":2},"unk_token":"a"}}"#;
        Tokenizer::from_bytes(json.as_bytes()).expect("build plain tokenizer")
    }

    /// `<|tool_call>call:f{x:<|"|>a<|"|>}<tool_call|><turn|>` as ids.
    pub(crate) const CALL_IDS: [u32; 8] = [4, 2, 6, 0, 6, 3, 5, 7];
    pub(crate) const CALL_TEXT: &str = "<|tool_call>call:f{x:<|\"|>a<|\"|>}<tool_call|>";

    #[test]
    fn finds_the_three_markers() {
        let tok = marker_tokenizer();
        assert_eq!(
            tool_marker_ids(&tok),
            vec![
                (4, "<|tool_call>".to_string()),
                (5, "<tool_call|>".to_string()),
                (6, "<|\"|>".to_string()),
            ]
        );
    }

    #[test]
    fn keeps_only_the_markers() {
        let tok = marker_tokenizer();
        let keep = tool_marker_ids(&tok);
        let text = decode_keeping(&tok, &CALL_IDS, &keep).unwrap();
        assert_eq!(text, CALL_TEXT, "markers kept, `<turn|>` dropped");
    }

    #[test]
    fn empty_keep_list_equals_plain_decode() {
        let tok = marker_tokenizer();
        for ids in [&CALL_IDS[..], &[0, 1, 7], &[], &[4], &[7, 7]] {
            assert_eq!(
                decode_keeping(&tok, ids, &[]).unwrap(),
                tok.decode(ids, true).unwrap()
            );
        }
        assert_eq!(decode_keeping(&tok, &CALL_IDS, &[]).unwrap(), "call:f{x:a}");
    }

    #[test]
    fn no_markers_in_vocab_gives_empty_list() {
        let tok = plain_tokenizer();
        let keep = tool_marker_ids(&tok);
        assert!(keep.is_empty());
        let ids = [0, 2, 1];
        assert_eq!(
            decode_keeping(&tok, &ids, &keep).unwrap(),
            tok.decode(&ids, true).unwrap()
        );
    }

    #[test]
    fn incremental_diffs_are_consistent() {
        let tok = marker_tokenizer();
        let keep = tool_marker_ids(&tok);
        let mut prev = String::new();
        let mut client = String::new();
        for n in 1..=CALL_IDS.len() {
            let full = decode_keeping(&tok, &CALL_IDS[..n], &keep).unwrap();
            let delta = full
                .strip_prefix(prev.as_str())
                .unwrap_or_else(|| panic!("decode of {n} ids is not an extension: {full:?}"));
            client.push_str(delta);
            prev = full;
        }
        assert_eq!(client, CALL_TEXT);
    }
}
