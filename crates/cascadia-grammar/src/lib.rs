//! Token masks that keep a model's output inside a JSON Schema.
//!
//! Built on llguidance. Pure CPU: no OpenVINO, so it unit-tests on every
//! platform. One [`GrammarFactory`] per engine (built from the tokenizer the
//! engine already loads and every EOS id it stops on), one
//! [`CompiledGrammar`] per request schema, one [`GrammarMask`] per sequence.
//!
//! The grammar is compact JSON: no whitespace between tokens, so a model
//! can't loop on spaces or newlines. Schema keywords llguidance accepts but
//! does not enforce (`not`, `uniqueItems`, unknown `format`s, ...) compile and
//! are listed by [`CompiledGrammar::warnings`]; a schema that can't be built
//! at all (bad type, unresolvable `$ref`) is an error at
//! [`GrammarFactory::compile`].

use std::sync::Arc;

use cascadia_engine::{EngineError, EngineResult};
use cascadia_types::{GrammarKind, GrammarSpec};
use llguidance::api::{StopReason, TopLevelGrammar};
use llguidance::toktrie::{SimpleVob, TokEnv};
use llguidance::{Matcher, ParserFactory};
use toktrie_hf_tokenizers::ByteTokenizer;

/// Longest error message handed back to the client.
const MAX_ERROR_CHARS: usize = 300;

/// One line, capped at [`MAX_ERROR_CHARS`] characters.
fn clean_message(raw: &str) -> String {
    let one_line = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= MAX_ERROR_CHARS {
        return one_line;
    }
    one_line.chars().take(MAX_ERROR_CHARS).collect()
}

fn invalid_schema(e: impl std::fmt::Display) -> EngineError {
    EngineError::InvalidConfig(clean_message(&format!("response_format schema: {e}")))
}

fn backend(e: impl std::fmt::Display) -> EngineError {
    EngineError::Backend(clean_message(&format!("grammar: {e}")))
}

/// A grammar failure while generating. llguidance's text describes its
/// parser state, which is for the log, not the client.
fn generation_failed(e: impl std::fmt::Display) -> EngineError {
    tracing::warn!(error = %clean_message(&e.to_string()), "structured output generation failed");
    EngineError::Backend("structured output generation failed".into())
}

/// Builds grammars for one tokenizer. Cheap to share: clone it or keep it in an `Arc`.
#[derive(Clone)]
pub struct GrammarFactory {
    factory: Arc<ParserFactory>,
    mask_width: usize,
    eos: Arc<[u32]>,
}

impl GrammarFactory {
    /// `eos` is every token id the engine stops on. Ids outside the tokenizer
    /// vocab are dropped; if none remains this fails, because a grammar that
    /// can't emit EOS can't end.
    pub fn from_tokenizer(tokenizer: &tokenizers::Tokenizer, eos: &[u32]) -> EngineResult<Self> {
        let mut bt = ByteTokenizer::from_tokenizer(tokenizer.clone()).map_err(|e| {
            EngineError::InvalidConfig(clean_message(&format!(
                "tokenizer unsupported for structured output: {e}"
            )))
        })?;
        let vocab = bt.tokrx_info().vocab_size;
        let eos: Vec<u32> = eos.iter().copied().filter(|&t| t < vocab).collect();
        if eos.is_empty() {
            return Err(EngineError::InvalidConfig(
                "structured output needs an end-of-sequence token inside the tokenizer vocab"
                    .into(),
            ));
        }
        bt.set_eos_tokens(&eos);
        let tok_env = bt.into_tok_env(None).map_err(backend)?;
        Self::from_tok_env(tok_env, eos.into())
    }

    fn from_tok_env(tok_env: TokEnv, eos: Arc<[u32]>) -> EngineResult<Self> {
        let mask_width = tok_env.tok_trie().vocab_size();
        let mut factory = ParserFactory::new_simple(&tok_env).map_err(backend)?;
        // A server: errors and warnings come back as values, nothing goes to stderr.
        factory.quiet().set_stderr_log_level(0);
        Ok(Self {
            factory: Arc::new(factory),
            mask_width,
            eos,
        })
    }

    /// A factory over a tokenizer-free vocab where token id `b` is byte `b`
    /// (0..=255) and 256..=261 are special tokens; 261 is its EOS. Built once
    /// and shared. Building a schema's grammar costs about the same here as
    /// with a real tokenizer, so a server can check that a schema compiles in
    /// time before any engine sees it.
    pub fn byte_level() -> Self {
        static BYTE_LEVEL: std::sync::OnceLock<GrammarFactory> = std::sync::OnceLock::new();
        BYTE_LEVEL
            .get_or_init(|| {
                Self::from_tok_env(
                    llguidance::toktrie::ApproximateTokEnv::single_byte_env(),
                    vec![261].into(),
                )
                .expect("byte-level vocab builds")
            })
            .clone()
    }

    /// [`Self::byte_level`], for engine tests that drive a mask over
    /// hand-built logits rows.
    #[cfg(feature = "test-util")]
    pub fn single_byte_for_tests() -> Self {
        Self::byte_level()
    }

    /// Number of real tokens the mask covers (the tokenizer vocab). A model's
    /// logits row may be longer; [`GrammarMask::apply`] masks the excess.
    pub fn mask_width(&self) -> usize {
        self.mask_width
    }

    /// Builds the grammar for `spec`, failing with `InvalidConfig` when the
    /// schema can't be built (bad JSON, bad type, unresolvable `$ref`, ...).
    /// A `pattern` (or `patternProperties` key) the regex engine can't parse,
    /// such as a look-ahead, is dropped and listed in the warnings, like
    /// the other keywords that compile but aren't enforced.
    pub fn compile(&self, spec: &GrammarSpec) -> EngineResult<CompiledGrammar> {
        match spec.kind {
            GrammarKind::JsonSchema => {
                // llguidance's serde_json keeps object key order, so properties
                // are generated in the order the client listed them.
                let mut schema: serde_json::Value =
                    serde_json::from_str(&spec.body).map_err(invalid_schema)?;
                let obj = schema
                    .as_object_mut()
                    .ok_or_else(|| invalid_schema("must be a JSON object"))?;
                // Ours replaces any the client sent. Compact JSON; oneOf is
                // treated as anyOf (pydantic's discriminated unions); keywords
                // llguidance can't enforce become warnings, not errors.
                obj.insert("x-guidance".into(), guidance_options());
                match self.compile_schema(schema.clone(), Vec::new()) {
                    Err(e) if is_regex_error(&e) => {
                        // Only on a regex error: drop every pattern the regex
                        // front end can't parse (a parse, no grammar build),
                        // then build once more.
                        let mut dropped = Vec::new();
                        visit_patterns(&mut schema, &mut |p| {
                            let ok = pattern_parses(p);
                            if !ok && !dropped.iter().any(|d| d == p) {
                                dropped.push(p.to_owned());
                            }
                            ok
                        });
                        if dropped.is_empty() {
                            return Err(e);
                        }
                        let warnings = dropped
                            .iter()
                            .map(|p| {
                                clean_message(&format!(
                                    "pattern not enforced (the regex engine can't parse it): {p}"
                                ))
                            })
                            .collect();
                        self.compile_schema(schema, warnings)
                    }
                    other => other,
                }
            }
        }
    }

    fn compile_schema(
        &self,
        schema: serde_json::Value,
        mut warnings: Vec<String>,
    ) -> EngineResult<CompiledGrammar> {
        #[cfg(test)]
        tests::SCHEMA_BUILDS.with(|c| c.set(c.get() + 1));
        let grammar = TopLevelGrammar::from_json_schema(schema);
        // Building the parser compiles the schema, so every error surfaces here.
        let parser = self
            .factory
            .create_parser(grammar)
            .map_err(invalid_schema)?;
        let mut matcher = Matcher::new(Ok(parser));
        if let Some(e) = matcher.get_error() {
            return Err(invalid_schema(e));
        }
        warnings.extend(matcher.grammar_warnings());
        Ok(CompiledGrammar {
            pristine: Arc::new(matcher),
            warnings: warnings.into(),
            mask_width: self.mask_width,
            eos: self.eos.clone(),
        })
    }
}

/// Whether llguidance's regex front end parses `pattern`: the rewrite and the
/// parser it applies to a JSON Schema `pattern`, without building a grammar.
fn pattern_parses(pattern: &str) -> bool {
    llguidance::derivre::RegexBuilder::new()
        .mk_regex_for_serach(&llguidance::regex_to_lark(pattern, "dw"))
        .is_ok()
}

/// Schema keywords whose values are data, not schemas: never searched for patterns.
const DATA_KEYWORDS: &[&str] = &["const", "enum", "default", "examples", "x-guidance"];

fn guidance_options() -> serde_json::Value {
    serde_json::json!({
        "whitespace_flexible": false,
        "coerce_one_of": true,
        "lenient": true,
    })
}

fn is_regex_error(e: &EngineError) -> bool {
    matches!(e, EngineError::InvalidConfig(m) if m.contains("regex"))
}

/// Walks every schema object in `v`, removing a string `pattern` and each
/// `patternProperties` key for which `keep` is false.
fn visit_patterns(v: &mut serde_json::Value, keep: &mut impl FnMut(&str) -> bool) {
    match v {
        serde_json::Value::Object(obj) => {
            if let Some(serde_json::Value::String(p)) = obj.get("pattern") {
                if !keep(p) {
                    obj.remove("pattern");
                }
            }
            if let Some(serde_json::Value::Object(props)) = obj.get_mut("patternProperties") {
                props.retain(|k, _| keep(k));
            }
            for (k, child) in obj.iter_mut() {
                if !DATA_KEYWORDS.contains(&k.as_str()) {
                    visit_patterns(child, keep);
                }
            }
        }
        serde_json::Value::Array(items) => {
            for child in items {
                visit_patterns(child, keep);
            }
        }
        _ => {}
    }
}

/// A schema built for one tokenizer. [`CompiledGrammar::matcher`] hands out a
/// fresh per-sequence mask without rebuilding the grammar.
#[derive(Clone)]
pub struct CompiledGrammar {
    pristine: Arc<Matcher>,
    warnings: Arc<[String]>,
    mask_width: usize,
    eos: Arc<[u32]>,
}

impl CompiledGrammar {
    /// What llguidance accepted but does not enforce, for the engine's log.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// A fresh mask at the start of the output.
    /// Clones the compiled parser (lexer caches stay shared), no recompile.
    pub fn matcher(&self) -> EngineResult<GrammarMask> {
        let matcher = (*self.pristine).clone();
        if let Some(e) = matcher.get_error() {
            return Err(generation_failed(e));
        }
        Ok(GrammarMask {
            matcher,
            mask_width: self.mask_width,
            eos: self.eos.clone(),
            current: None,
        })
    }
}

/// Per-sequence grammar state: which tokens may come next.
pub struct GrammarMask {
    matcher: Matcher,
    mask_width: usize,
    eos: Arc<[u32]>,
    /// Mask for the current position, computed once per step.
    current: Option<SimpleVob>,
}

impl GrammarMask {
    /// Accepts tokens already in the output (a resumed sequence). The server
    /// produced those tokens, so one the grammar rejects is a server-side
    /// failure (`Backend`), not the client's.
    pub fn replay(&mut self, tokens: &[u32]) -> EngineResult<()> {
        self.current = None;
        self.matcher.consume_tokens(tokens).map_err(|e| {
            generation_failed(format!("resumed output does not match the schema: {e}"))
        })
    }

    fn current_mask(&mut self) -> EngineResult<&SimpleVob> {
        if self.current.is_none() {
            // Once stopped this is the EOS set: every EOS id the engine stops on.
            let mask = self
                .matcher
                .compute_mask_or_eos()
                .map_err(generation_failed)?;
            self.current = Some(mask);
        }
        Ok(self.current.as_ref().expect("mask computed above"))
    }

    /// Sets every disallowed index of `row` to `-inf`, including indices past
    /// the tokenizer vocab (a padded model head). `Err` when nothing in `row`
    /// is allowed.
    pub fn apply(&mut self, row: &mut [f32]) -> EngineResult<()> {
        let width = self.mask_width.min(row.len());
        let mask = self.current_mask()?;
        let words = mask.as_slice();
        let mut any = false;
        // 32 tokens per mask word: whole words are filled or skipped, only
        // mixed words are tested bit by bit.
        for (w, chunk) in row[..width].chunks_mut(32).enumerate() {
            let mut bits = words.get(w).copied().unwrap_or(0);
            if chunk.len() < 32 {
                // The last, partial word: only the bits that index into the row.
                bits &= (1u32 << chunk.len()) - 1;
            }
            if bits == 0 {
                chunk.fill(f32::NEG_INFINITY);
                continue;
            }
            any = true;
            if bits == u32::MAX && chunk.len() == 32 {
                continue;
            }
            for (b, l) in chunk.iter_mut().enumerate() {
                if bits & (1u32 << b) == 0 {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        row[width..].fill(f32::NEG_INFINITY);
        if any {
            Ok(())
        } else {
            Err(EngineError::Backend(
                "grammar: no token of the logits row is allowed".into(),
            ))
        }
    }

    /// The current allowed set as `ceil(mask_width / 32)` words: bit `t % 32`
    /// of word `t / 32` is set when token `t` may come next; no bit past the
    /// tokenizer vocab is set. Read from the cached mask, no logits row. The
    /// same errors as [`Self::apply`]: the error state, or nothing allowed.
    pub fn mask_words(&mut self) -> EngineResult<Vec<u32>> {
        let width = self.mask_width;
        let mask = self.current_mask()?.as_slice();
        let mut words: Vec<u32> = (0..width.div_ceil(32))
            .map(|w| mask.get(w).copied().unwrap_or(0))
            .collect();
        if let (Some(last), tail @ 1..) = (words.last_mut(), width % 32) {
            *last &= (1u32 << tail) - 1;
        }
        if words.iter().all(|&w| w == 0) {
            return Err(EngineError::Backend(
                "grammar: no token of the vocabulary is allowed".into(),
            ));
        }
        Ok(words)
    }

    /// Greedy pick from `row` among the tokens that may come next (see
    /// [`argmax_allowed`]); indices past the tokenizer vocab are never picked.
    /// The matcher is not advanced. `Err` in the error state or when nothing
    /// in `row` is allowed, as [`Self::apply`].
    pub fn pick_greedy(&mut self, row: &[f32]) -> EngineResult<usize> {
        let width = self.mask_width.min(row.len());
        let mask = self.current_mask()?;
        argmax_allowed(&row[..width], mask.as_slice()).ok_or_else(|| {
            EngineError::Backend("grammar: no token of the logits row is allowed".into())
        })
    }

    /// Whether `token` may come next.
    pub fn allowed(&mut self, token: u32) -> bool {
        let width = self.mask_width;
        (token as usize) < width && self.current_mask().is_ok_and(|mask| mask.is_allowed(token))
    }

    /// Advances past the token the engine picked. A token the grammar rejects
    /// is an error, and the mask stays in that error state: [`Self::apply`]
    /// and later accepts fail, [`Self::finished`] stays false. An EOS token
    /// after a clean finish is accepted and changes nothing (the engine may
    /// commit the EOS it picked from the finished mask).
    pub fn accept(&mut self, token: u32) -> EngineResult<()> {
        if self.finished() && self.eos.contains(&token) {
            return Ok(());
        }
        self.current = None;
        self.matcher.consume_token(token).map_err(generation_failed)
    }

    /// A clean finish: the JSON is complete and nothing else may follow, or
    /// EOS was accepted. The engine ends the sequence with
    /// `finish_reason: "stop"`. False in the error state (llguidance's own
    /// `is_stopped` is true there too).
    pub fn finished(&self) -> bool {
        !self.matcher.is_error()
            && matches!(
                self.matcher.stop_reason(),
                StopReason::NoExtension | StopReason::EndOfSentence
            )
    }
}

/// Greedy pick among the indices `words` allows (bit `t % 32` of word
/// `t / 32` set: index `t` allowed; indices past the words are not): the
/// allowed index with the largest logit, the first on ties. NaN and `-inf`
/// logits never win; when every allowed logit is one of them, the lowest
/// allowed index. Never a disallowed index; `None` when no index of `row` is
/// allowed.
pub fn argmax_allowed(row: &[f32], words: &[u32]) -> Option<usize> {
    let mut lowest = None;
    let mut best: Option<(usize, f32)> = None;
    // 32 indices per word: empty words are skipped, full words take a plain
    // max, only mixed words are walked bit by bit.
    for (w, (&word, chunk)) in words.iter().zip(row.chunks(32)).enumerate() {
        let base = w * 32;
        let bits = if chunk.len() < 32 {
            word & ((1u32 << chunk.len()) - 1)
        } else {
            word
        };
        if bits == 0 {
            continue;
        }
        lowest.get_or_insert(base + bits.trailing_zeros() as usize);
        if bits == u32::MAX {
            // NaN never compares greater, so it never becomes the max.
            let m = chunk
                .iter()
                .fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m });
            if m > f32::NEG_INFINITY && best.is_none_or(|(_, b)| m > b) {
                let i = chunk
                    .iter()
                    .position(|&v| v == m)
                    .expect("max is in the chunk");
                best = Some((base + i, m));
            }
            continue;
        }
        let mut bits = bits;
        while bits != 0 {
            let t = base + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let v = row[t];
            if v > f32::NEG_INFINITY && best.is_none_or(|(_, b)| v > b) {
                best = Some((t, v));
            }
        }
    }
    best.map(|(t, _)| t).or(lowest)
}

#[cfg(test)]
mod tests;
