//! Token masks that keep a model's output inside a JSON Schema.
//!
//! Built on llguidance. Pure CPU: no OpenVINO, so it unit-tests on every
//! platform. One [`GrammarFactory`] per engine (built from the tokenizer the
//! engine already loads and every EOS id it stops on), one
//! [`CompiledGrammar`] per request schema, one [`GrammarMask`] per sequence.
//!
//! The grammar is compact JSON: no whitespace between tokens, so a model
//! can't loop on spaces or newlines. Schema keywords llguidance does not
//! implement are an error at [`GrammarFactory::compile`], not silently
//! ignored.

use std::sync::Arc;

use cascadia_engine::{EngineError, EngineResult};
use cascadia_types::{GrammarKind, GrammarSpec};
use llguidance::api::TopLevelGrammar;
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
        Self::from_tok_env(tok_env)
    }

    fn from_tok_env(tok_env: TokEnv) -> EngineResult<Self> {
        let mask_width = tok_env.tok_trie().vocab_size();
        let mut factory = ParserFactory::new_simple(&tok_env).map_err(backend)?;
        // A server: errors come back as values, nothing goes to stderr.
        factory.quiet();
        Ok(Self {
            factory: Arc::new(factory),
            mask_width,
        })
    }

    /// Number of real tokens the mask covers (the tokenizer vocab). A model's
    /// logits row may be longer; [`GrammarMask::apply`] masks the excess.
    pub fn mask_width(&self) -> usize {
        self.mask_width
    }

    /// Builds the grammar for `spec`, failing with `InvalidConfig` when the
    /// schema can't be enforced (bad JSON, unsupported keyword, ...).
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
                obj.insert(
                    "x-guidance".into(),
                    serde_json::json!({"whitespace_flexible": false, "lenient": false}),
                );
                TopLevelGrammar::from_json_schema(schema)
            }
        };
        // Building the parser compiles the schema, so every error surfaces here.
        let parser = self
            .factory
            .create_parser(grammar)
            .map_err(invalid_schema)?;
        let matcher = Matcher::new(Ok(parser));
        if let Some(e) = matcher.get_error() {
            return Err(invalid_schema(e));
        }
        Ok(CompiledGrammar {
            pristine: Arc::new(matcher),
            mask_width: self.mask_width,
        })
    }
}

/// A schema built for one tokenizer. [`CompiledGrammar::matcher`] hands out a
/// fresh per-sequence mask without rebuilding the grammar.
#[derive(Clone)]
pub struct CompiledGrammar {
    pristine: Arc<Matcher>,
    mask_width: usize,
}

impl CompiledGrammar {
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
            current: None,
        })
    }
}

/// Per-sequence grammar state: which tokens may come next.
pub struct GrammarMask {
    matcher: Matcher,
    mask_width: usize,
    /// Mask for the current position, computed once per step.
    current: Option<SimpleVob>,
}

impl GrammarMask {
    /// Accepts tokens already in the output (a resumed sequence). A token the
    /// grammar rejects means the continuation can never be valid: `InvalidConfig`.
    pub fn replay(&mut self, tokens: &[u32]) -> EngineResult<()> {
        self.current = None;
        self.matcher.consume_tokens(tokens).map_err(|e| {
            EngineError::InvalidConfig(clean_message(&format!(
                "resumed output does not match the response_format schema: {e}"
            )))
        })
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
        let width = self.mask_width;
        let mask = self.current_mask()?;
        let mut any = false;
        for (i, l) in row.iter_mut().enumerate() {
            if i < width && mask.is_allowed(i as u32) {
                any = true;
            } else {
                *l = f32::NEG_INFINITY;
            }
        }
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

    /// Advances past the token the engine picked.
    pub fn accept(&mut self, token: u32) -> EngineResult<()> {
        self.current = None;
        self.matcher.consume_token(token).map_err(backend)
    }

    /// The JSON is complete and nothing else may follow: the engine finishes
    /// the sequence (`finish_reason: "stop"`) without generating EOS.
    pub fn is_stopped(&self) -> bool {
        // An error also stops llguidance's matcher; that is not a finished JSON.
        !self.matcher.is_error() && self.matcher.is_stopped()
    }
}

#[cfg(test)]
mod tests;
