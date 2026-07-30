//! Does K3's tiktoken pre-tokenizer regex work in the engine's regex flavour?
//!
//! K3 ships a tiktoken BPE whose `pat_str` (in `tokenization_kimi.py`) uses
//! Java/ICU character-class INTERSECTION — `[\p{Lu}...&&[^\p{Han}]]`, i.e.
//! "letters and marks except Han". If the engine's engine cannot parse that,
//! converting the tokenizer means rewriting the pattern, and a mis-rewrite
//! mis-splits text in a way that looks like a model quality bug.
//!
//! This crate builds `tokenizers` with the `onig` feature (Oniguruma), which
//! documents `&&` support. This test is the check.

/// Verbatim from `tokenization_kimi.py::TikTokenTokenizer.pat_str`.
const K3_PAT: &str = concat!(
    r"[\p{Han}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{PreTokenizedString, PreTokenizer, SplitDelimiterBehavior};

fn split_with_k3_pattern(s: &str) -> Result<Vec<String>, String> {
    let sp = Split::new(
        SplitPattern::Regex(K3_PAT.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| format!("{e}"))?;
    let mut pre = PreTokenizedString::from(s);
    sp.pre_tokenize(&mut pre).map_err(|e| format!("{e}"))?;
    Ok(pre
        .get_splits(
            tokenizers::OffsetReferential::Original,
            tokenizers::OffsetType::Byte,
        )
        .into_iter()
        .map(|(t, _, _)| t.to_string())
        .collect())
}

#[test]
fn the_engines_regex_accepts_the_intersection_pattern() {
    match split_with_k3_pattern("hello") {
        Ok(v) => eprintln!("accepted; split -> {v:?}"),
        Err(e) => panic!("engine rejected K3's pat_str: {e}"),
    }
}

#[test]
fn han_is_split_from_latin_as_the_intersection_intends() {
    // The whole point of `&&[^\p{Han}]`: a Han run is its own token and must NOT
    // be absorbed into an adjacent Latin word. Drop the intersection and
    // alternatives 1/2 swallow the Han characters — a silent mis-split.
    let pieces = split_with_k3_pattern("hello世界world 123").expect("split");
    eprintln!("pieces: {pieces:?}");
    assert!(
        pieces.iter().any(|p| p == "世界"),
        "Han run not isolated: {pieces:?}"
    );
    assert!(
        !pieces.iter().any(|p| p.contains('世') && p.contains('h')),
        "Han merged into a latin piece: {pieces:?}"
    );
}
