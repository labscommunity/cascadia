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

    /// A factory over a test vocab where token id `b` is byte `b` (0..=255)
    /// and 256..=261 are special tokens; 261 is its EOS. For engine tests that
    /// drive a mask over hand-built logits rows.
    #[cfg(feature = "test-util")]
    pub fn single_byte_for_tests() -> Self {
        Self::from_tok_env(
            llguidance::toktrie::ApproximateTokEnv::single_byte_env(),
            vec![261].into(),
        )
        .expect("single-byte test vocab builds")
    }

    /// Number of real tokens the mask covers (the tokenizer vocab). A model's
    /// logits row may be longer; [`GrammarMask::apply`] masks the excess.
    pub fn mask_width(&self) -> usize {
        self.mask_width
    }

    /// Builds the grammar for `spec`, failing with `InvalidConfig` when the
    /// schema can't be built (bad JSON, bad type, unresolvable `$ref`, ...).
    pub fn compile(&self, spec: &GrammarSpec) -> EngineResult<CompiledGrammar> {
        let grammar = match spec.kind {
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
                obj.insert(
                    "x-guidance".into(),
                    serde_json::json!({
                        "whitespace_flexible": false,
                        "coerce_one_of": true,
                        "lenient": true,
                    }),
                );
                TopLevelGrammar::from_json_schema(schema)
            }
        };
        // Building the parser compiles the schema, so every error surfaces here.
        let parser = self
            .factory
            .create_parser(grammar)
            .map_err(invalid_schema)?;
        let mut matcher = Matcher::new(Ok(parser));
        if let Some(e) = matcher.get_error() {
            return Err(invalid_schema(e));
        }
        let warnings = matcher.grammar_warnings().into();
        Ok(CompiledGrammar {
            pristine: Arc::new(matcher),
            warnings,
            mask_width: self.mask_width,
            eos: self.eos.clone(),
        })
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
            return Err(backend(e));
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
        self.matcher
            .consume_tokens(tokens)
            .map_err(|e| backend(format!("resumed output does not match the schema: {e}")))
    }

    fn current_mask(&mut self) -> EngineResult<&SimpleVob> {
        if self.current.is_none() {
            // Once stopped this is the EOS set: every EOS id the engine stops on.
            let mask = self.matcher.compute_mask_or_eos().map_err(backend)?;
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
        self.matcher.consume_token(token).map_err(backend)
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

#[cfg(test)]
mod tests;
