//! Token masks that keep a model's output inside a JSON Schema.
//!
//! Built on llguidance. Pure CPU: no OpenVINO, so it unit-tests on every
//! platform. One [`GrammarFactory`] per engine (built from the tokenizer the
//! engine already loads and every EOS id it stops on), one
//! [`CompiledGrammar`] per request schema, one [`GrammarMask`] per sequence.
//!
//! The grammar is JSON with `json.dumps` separators (`": "` and `", "`) and
//! no other whitespace between tokens, so a model can't loop on spaces or
//! newlines. Schema keywords llguidance accepts but
//! does not enforce (`not`, `uniqueItems`, unknown `format`s, ...) compile and
//! are listed by [`CompiledGrammar::warnings`]; a schema that can't be built
//! at all (bad type, unresolvable `$ref`) is an error at
//! [`GrammarFactory::compile`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use cascadia_engine::{EngineError, EngineResult};
use cascadia_types::{GrammarKind, GrammarSpec};
use llguidance::api::{StopReason, TopLevelGrammar};
use llguidance::toktrie::{SimpleVob, TokEnv, TokRxInfo, TokTrie, TokenizerEnv};
use llguidance::{Matcher, ParserFactory};
use tokenizers::{DecoderWrapper, ModelWrapper};
use toktrie_hf_tokenizers::ByteTokenizer;

/// What a structured-output request fails with when a later pipeline stage
/// runs a build that can't apply a grammar mask (or didn't acknowledge one).
/// Engines report it verbatim, so an embedder can recognise it and route the
/// request elsewhere.
pub const STAGE_UPGRADE_MESSAGE: &str =
    "a later pipeline stage runs a build without structured-output support; upgrade every stage";

/// Why a schema with an integer llguidance can't build is refused. The
/// value is compared as a 64-bit float, so a value the client wrote just
/// above -2^63 (-9223372036854775807) is refused too.
const LOW_INTEGER_MESSAGE: &str = "an integer `const`, `enum` value or upper bound at or below \
     about -9.2e18 (-2^63, compared as a 64-bit float) is not supported";

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
    ///
    /// Byte-level and byte-fallback BPE tokenizers are supported, and a
    /// word-level vocab with a `Fuse` decoder, which only joins the token
    /// strings; any other tokenizer (a word-level vocab without a decoder
    /// joins tokens with spaces) is `InvalidConfig`.
    pub fn from_tokenizer(tokenizer: &tokenizers::Tokenizer, eos: &[u32]) -> EngineResult<Self> {
        let mut bt = match ByteTokenizer::from_tokenizer(tokenizer.clone()) {
            Ok(bt) => bt,
            Err(_) if decodes_by_concatenation(tokenizer) => {
                return Self::from_concatenating_tokenizer(tokenizer, eos);
            }
            Err(e) => {
                return Err(EngineError::InvalidConfig(clean_message(&format!(
                    "tokenizer unsupported for structured output: {e}"
                ))));
            }
        };
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

    /// A tokenizer whose decoded text is its token strings joined: token
    /// `t`'s bytes are the UTF-8 of its vocab string. Special added tokens
    /// are marked special, so the grammar never puts them inside the JSON.
    fn from_concatenating_tokenizer(
        tokenizer: &tokenizers::Tokenizer,
        eos: &[u32],
    ) -> EngineResult<Self> {
        let vocab = tokenizer.get_vocab_size(true) as u32;
        let eos: Vec<u32> = eos.iter().copied().filter(|&t| t < vocab).collect();
        let Some(&first_eos) = eos.first() else {
            return Err(EngineError::InvalidConfig(
                "structured output needs an end-of-sequence token inside the tokenizer vocab"
                    .into(),
            ));
        };
        let specials: std::collections::HashSet<u32> = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, t)| t.special)
            .map(|(id, _)| id)
            .collect();
        let words: Vec<Vec<u8>> = (0..vocab)
            .map(|id| match tokenizer.id_to_token(id) {
                Some(s) if specials.contains(&id) => {
                    let mut b = vec![TokTrie::SPECIAL_TOKEN_MARKER];
                    b.extend_from_slice(s.as_bytes());
                    b
                }
                Some(s) => s.into_bytes(),
                None => Vec::new(),
            })
            .collect();
        let info = TokRxInfo::new(vocab, first_eos);
        let mut trie = TokTrie::from(&info, &words);
        if eos.len() > 1 {
            trie = trie.with_eos_tokens(&eos);
        }
        let tok_env: TokEnv = Arc::new(ConcatenatingTokEnv {
            tokenizer: tokenizer.clone(),
            trie,
        });
        Self::from_tok_env(tok_env, eos.into())
    }

    fn from_tok_env(tok_env: TokEnv, eos: Arc<[u32]>) -> EngineResult<Self> {
        let mask_width = tok_env.tok_trie().vocab_size();
        let mut factory = ParserFactory::new_simple(&tok_env).map_err(backend)?;
        // A server: errors and warnings come back as values, nothing goes to stderr.
        factory.quiet().set_stderr_log_level(0);
        let limits = factory.limits_mut();
        limits.max_items_in_row = MAX_ITEMS_IN_ROW;
        limits.step_max_items = STEP_MAX_ITEMS;
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

    /// [`Self::byte_level`] with llguidance's row cap lowered to
    /// `max_items_in_row`, so a test can build a schema that fails on its
    /// first token without one past the API's size cap.
    #[cfg(any(test, feature = "test-util"))]
    pub fn byte_level_with_row_cap_for_tests(max_items_in_row: usize) -> Self {
        let mut f = Self::from_tok_env(
            llguidance::toktrie::ApproximateTokEnv::single_byte_env(),
            vec![261].into(),
        )
        .expect("byte-level vocab builds");
        Arc::get_mut(&mut f.factory)
            .expect("a new factory is not shared")
            .limits_mut()
            .max_items_in_row = max_items_in_row;
        f
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
                // llguidance never fetches: a ref to a document the schema
                // doesn't hold is refused with a clear 400.
                if let Some(why) = refs::ref_outside_the_schema(&schema) {
                    return Err(invalid_schema(why));
                }
                if int_bounds::has_integer_llguidance_cannot_build(&schema) {
                    return Err(invalid_schema(LOW_INTEGER_MESSAGE));
                }
                let obj = schema
                    .as_object_mut()
                    .ok_or_else(|| invalid_schema("must be a JSON object"))?;
                // Ours replaces any the client sent. json.dumps separators; oneOf is
                // treated as anyOf (pydantic's discriminated unions); keywords
                // llguidance can't enforce become warnings, not errors.
                obj.insert("x-guidance".into(), guidance_options());
                int_bounds::clamp_for_llguidance(&mut schema);
                let mut bounded = schema.clone();
                if bound_integers(&mut bounded) == 0 {
                    return self.compile_with_patterns_dropped(schema, Vec::new());
                }
                // The added minimum is below what llguidance builds.
                int_bounds::clamp_for_llguidance(&mut bounded);
                match self.compile_with_patterns_dropped(bounded, Vec::new()) {
                    // Each bounded integer is a range regex of its own: a
                    // schema with hundreds of them can pass llguidance's
                    // lexer budget. It is built without the added bounds
                    // then, as the client wrote it.
                    Err(e) if is_lexer_too_big(&e) => self.compile_with_patterns_dropped(
                        schema,
                        vec!["integer length not bounded (the schema has too many \
                              integers for the bounded grammar)"
                            .into()],
                    ),
                    // The added bounds can leave no value where the schema
                    // narrows an integer through a keyword the bounding
                    // doesn't read (an `allOf` with a `const` past them):
                    // built as the client wrote it.
                    Err(e) if is_unsatisfiable(&e) => self.compile_with_patterns_dropped(
                        schema,
                        vec!["integer length not bounded (the bounded schema \
                              admits no value)"
                            .into()],
                    ),
                    other => other,
                }
            }
        }
    }

    /// [`Self::compile_schema`], and once more without every pattern the
    /// regex front end can't parse when the build fails on a regex error.
    fn compile_with_patterns_dropped(
        &self,
        mut schema: serde_json::Value,
        warnings: Vec<String>,
    ) -> EngineResult<CompiledGrammar> {
        match self.compile_schema(schema.clone(), warnings.clone()) {
            Err(e) if is_regex_error(&e) => {
                // Only on a regex error: drop every pattern the regex
                // front end can't parse (a parse, no grammar build),
                // then build once more.
                let dropped = drop_unparseable_patterns(&mut schema);
                if dropped.is_empty() {
                    return Err(e);
                }
                let mut warnings = warnings;
                warnings.extend(dropped.iter().map(|p| {
                    clean_message(&format!(
                        "pattern not enforced (the regex engine can't parse it): {p}"
                    ))
                }));
                self.compile_schema(schema, warnings)
            }
            other => other,
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
        // Building the parser compiles the schema, so every error surfaces
        // here. llguidance's build recurses once per optional property, deep
        // enough for a schema inside the API's limits to overflow a 2 MiB
        // thread (which aborts the process): build on a thread of its own
        // with a stack that can't run out, whoever the caller is.
        let factory = &self.factory;
        let built = std::thread::scope(|scope| {
            let build = std::thread::Builder::new()
                .name("grammar-build".into())
                .stack_size(BUILD_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    let parser = factory.create_parser(grammar).map_err(invalid_schema)?;
                    let mut matcher = Matcher::new(Ok(parser));
                    if let Some(e) = matcher.get_error() {
                        return Err(invalid_schema(e));
                    }
                    let warnings = matcher.grammar_warnings();
                    Ok((matcher, warnings))
                })
                .map_err(|e| EngineError::Backend(format!("grammar build thread: {e}")))?;
            // A panic in the build is the caller's, as if it had built inline.
            build
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        });
        let (matcher, build_warnings) = built?;
        warnings.extend(build_warnings);
        Ok(CompiledGrammar {
            pristine: Arc::new(matcher),
            warnings: warnings.into(),
            mask_width: self.mask_width,
            eos: self.eos.clone(),
        })
    }
}

/// llguidance's cap on the Earley items in one parser row. After `{`, each
/// optional property of an object puts about four items in one row; the
/// widest object the API admits (7,719 properties of any type within the
/// 64 KiB cap) needs 30,883, alone or as an array's items. Half again as
/// many, rounded up. llguidance's default (2000, about 500 properties) let
/// such a schema build and give its first mask, then fail on the first
/// accepted token.
pub(crate) const MAX_ITEMS_IN_ROW: usize = 46_500;

/// llguidance's cap on the Earley items one mask may create. A mask walks
/// the tokenizer's vocab through the parser: with thousands of keys open,
/// many tokens start one. The widest admitted object took at most 216,164
/// items for one mask over Qwen2.5's and Llama 3.1's vocabs (any-typed,
/// typed and mixed properties, at the top level or as an array's items);
/// the byte-level vocab never passes 62,000. Half again as many, rounded
/// up. llguidance's default is 50,000.
pub(crate) const STEP_MAX_ITEMS: usize = 325_000;

/// Most first tokens [`CompiledGrammar::check_first_step`] tries: `{`, `[`
/// and a couple of others (`"`, a digit, ...), whose rows don't grow with
/// the schema's width.
pub const FIRST_STEP_TOKENS: usize = 4;

/// Stack for the thread that builds a grammar (see `compile_schema`). Only
/// reserved address space: pages are committed as the build touches them.
/// llguidance recurses once per optional property (about 350 bytes a level
/// in an optimized macOS build, more on Windows); a schema at the API's
/// size cap has fewer than 10,000 of them.
const BUILD_STACK_BYTES: usize = 256 << 20;

/// Whether decoding `tokenizer` only joins its token strings: a word-level
/// vocab with a `Fuse` decoder. Without a decoder `tokenizers` joins tokens
/// with spaces, so that case is not concatenation.
fn decodes_by_concatenation(tokenizer: &tokenizers::Tokenizer) -> bool {
    matches!(tokenizer.get_model(), ModelWrapper::WordLevel(_))
        && matches!(tokenizer.get_decoder(), Some(DecoderWrapper::Fuse(_)))
}

/// Token environment for a tokenizer that decodes by concatenation.
struct ConcatenatingTokEnv {
    tokenizer: tokenizers::Tokenizer,
    trie: TokTrie,
}

impl ConcatenatingTokEnv {
    fn encode(&self, s: &str) -> Vec<u32> {
        match self.tokenizer.encode(s, false) {
            Ok(enc) => enc.get_ids().to_vec(),
            // Text the tokenizer can't encode: the trie's greedy split.
            Err(_) => self.trie.greedy_tokenize(s.as_bytes()),
        }
    }
}

impl TokenizerEnv for ConcatenatingTokEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<u32> {
        self.trie
            .tokenize_with_greedy_fallback(s, |s| self.encode(s))
    }

    fn tokenize_bytes_special(&self, s: &[u8]) -> Vec<u32> {
        self.trie.tokenize_with_greedy_fallback(s, |s| {
            self.trie.tokenize_with_special(s, |s| self.encode(s))
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

/// Removes from `schema` every `pattern`, and every `patternProperties` key,
/// that llguidance's regex front end can't parse (look-around, a
/// backreference, ...), and returns the removed regexes, each once, in the
/// order found. Everything else, key order included, is left as it was.
pub fn drop_unparseable_patterns(schema: &mut serde_json::Value) -> Vec<String> {
    let mut dropped: Vec<String> = Vec::new();
    visit_patterns(schema, &mut |p| {
        let ok = pattern_parses(p);
        if !ok && !dropped.iter().any(|d| d == p) {
            dropped.push(p.to_owned());
        }
        ok
    });
    dropped
}

/// Keywords whose value maps names (property names, definition names, ...)
/// to schemas.
pub(crate) const NAME_MAPS: &[&str] = &[
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
];

/// Schema keywords whose values are data, not schemas: never searched for patterns.
const DATA_KEYWORDS: &[&str] = &["const", "enum", "default", "examples", "x-guidance"];

fn guidance_options() -> serde_json::Value {
    serde_json::json!({
        "whitespace_flexible": false,
        // `json.dumps` separators: models write JSON this way, and forbidding
        // the space after `:` and `,` pushes them off their distribution. No
        // other whitespace, so a model can't loop on spaces or newlines.
        "key_separator": ": ",
        "item_separator": ", ",
        "coerce_one_of": true,
        "lenient": true,
    })
}

fn is_lexer_too_big(e: &EngineError) -> bool {
    matches!(e, EngineError::InvalidConfig(m) if m.contains("initial lexer configuration (grammar) too big"))
}

fn is_unsatisfiable(e: &EngineError) -> bool {
    matches!(e, EngineError::InvalidConfig(m) if m.contains("Unsatisfiable schema"))
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
                let k = k.as_str();
                if DATA_KEYWORDS.contains(&k) {
                    continue;
                }
                match child {
                    // A name here (a property called `default`) is not a
                    // keyword: walk the map's schemas.
                    serde_json::Value::Object(map) if NAME_MAPS.contains(&k) => {
                        for schema in map.values_mut() {
                            visit_patterns(schema, keep);
                        }
                    }
                    _ => visit_patterns(child, keep),
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

    /// Takes the first step of an answer: the first mask, then up to
    /// [`FIRST_STEP_TOKENS`] of the tokens it allows (`{` and `[` first, then
    /// the lowest ids; never EOS), each on a copy of the matcher with the
    /// mask after it, and after `[` the `{` of an object item too. A grammar
    /// can build and give its first mask, then fail on its first token: the
    /// parser row after an object's `{` holds about four items per optional
    /// property, and llguidance refuses a row past its limit. That row comes
    /// right after the `{`, of the top-level object or of an array's first
    /// item, and no later row was larger at the API's size cap (30,883
    /// items, see [`MAX_ITEMS_IN_ROW`]); this finds it before a request is
    /// queued. At most `2 * FIRST_STEP_TOKENS` masks. `cancel` is read
    /// between them: once set, this stops with an error.
    pub fn check_first_step(&self, cancel: &AtomicBool) -> EngineResult<()> {
        let cancelled = || {
            if cancel.load(Ordering::Relaxed) {
                Err(EngineError::Backend(
                    "grammar: first-step check cancelled".into(),
                ))
            } else {
                Ok(())
            }
        };
        cancelled()?;
        let mut first = self.matcher()?;
        let words = first.mask_words()?;
        let trie = first.matcher.tok_env().map_err(generation_failed)?;
        let id_of = |bytes: &[u8]| trie.tok_trie().token_id(bytes);
        let (open_object, open_array) = (id_of(b"{"), id_of(b"["));
        let allowed = |words: &[u32], t: u32| {
            words
                .get(t as usize / 32)
                .is_some_and(|w| w & (1 << (t % 32)) != 0)
        };
        let mut tokens: Vec<u32> = [open_object, open_array]
            .into_iter()
            .flatten()
            .filter(|&t| allowed(&words, t))
            .collect();
        let rest: Vec<u32> = (0..words.len() as u32 * 32)
            .filter(|&t| allowed(&words, t) && !tokens.contains(&t) && !self.eos.contains(&t))
            .take(FIRST_STEP_TOKENS.saturating_sub(tokens.len()))
            .collect();
        tokens.extend(rest);
        for token in tokens {
            cancelled()?;
            let mut next = first.fork();
            next.accept(token)?;
            if next.finished() {
                continue;
            }
            let after = next.mask_words()?;
            if let Some(item) =
                open_object.filter(|&o| Some(token) == open_array && allowed(&after, o))
            {
                cancelled()?;
                let mut inner = next.fork();
                inner.accept(item)?;
                if !inner.finished() {
                    inner.mask_words()?;
                }
            }
        }
        Ok(())
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
    /// A copy at the same point, without the cached mask.
    fn fork(&self) -> Self {
        Self {
            matcher: self.matcher.clone(),
            mask_width: self.mask_width,
            eos: self.eos.clone(),
            current: None,
        }
    }

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

mod llama_cost;
mod llama_inline;
pub use llama_cost::{
    llama_grammar_cost, llama_grammar_cost_error, LLAMA_VISIT_COST, MAX_LLAMA_GRAMMAR_COST,
};
pub use llama_inline::{inline_refs, InlineError, Inlined, MAX_INLINED_BYTES};

mod xgrammar_cost;
pub use xgrammar_cost::{xgrammar_cost, xgrammar_cost_error, MAX_XGRAMMAR_COST};

mod int_bounds;
mod refs;
pub use int_bounds::{bound_integers, ADDED_INTEGER_MAXIMUM, ADDED_INTEGER_MINIMUM};
pub use refs::ref_outside_the_schema;

mod stage_mask;
pub use stage_mask::{
    MaskMiss, MaskRecvCache, MaskSendCache, WireMask, MASK_MISS_MESSAGE, MASK_MISS_TOKEN,
    MASK_PROBE, MASK_PROBE_OK_TOKEN, MASK_PROBE_REPROBE_AFTER, MASK_PROBE_TIMEOUT, REPEAT,
};
