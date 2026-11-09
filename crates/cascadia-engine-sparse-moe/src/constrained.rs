//! Structured output (`response_format`): the per-engine glue around
//! [`cascadia_grammar`]. One [`GrammarSlot`] per engine (built where rank 0
//! loads its tokenizer), one [`CompiledGrammar`] per queued task (compiled at
//! `submit`, so a schema that can't be built is a 400 before the task queues),
//! one [`GrammarMask`] per running sequence (created when the task starts).
//!
//! Tasks without a grammar never touch this module: no mask, no matcher.

use cascadia_engine::{EngineError, EngineResult};
use cascadia_grammar::{CompiledGrammar, GrammarFactory, GrammarMask};

use crate::dist::{MaskCols, MaskLatch, MaskedReply};
use cascadia_types::GenerationTask;
use tokenizers::Tokenizer;
use tracing::{info, warn};

/// The engine's grammar factory, or why it has none.
#[derive(Clone)]
pub(crate) struct GrammarSlot(Result<GrammarFactory, String>);

impl Default for GrammarSlot {
    fn default() -> Self {
        Self(Err("the model has no tokenizer on this stage".into()))
    }
}

impl GrammarSlot {
    /// Built once at load from the tokenizer and every EOS id the engine
    /// stops on. A failure is logged here, once; json tasks then get a 400.
    pub(crate) fn build(tokenizer: &Tokenizer, eos: &[u32]) -> Self {
        match GrammarFactory::from_tokenizer(tokenizer, eos) {
            Ok(f) => Self(Ok(f)),
            Err(e) => {
                let reason = match e {
                    EngineError::InvalidConfig(m) | EngineError::Backend(m) => m,
                    other => other.to_string(),
                };
                warn!(error = %reason, "structured output unavailable for this model");
                Self(Err(reason))
            }
        }
    }

    /// Tokens a mask covers (the tokenizer vocab); `None` without a factory.
    pub(crate) fn mask_width(&self) -> Option<usize> {
        self.0.as_ref().ok().map(GrammarFactory::mask_width)
    }

    #[cfg(test)]
    pub(crate) fn from_factory(f: GrammarFactory) -> Self {
        Self(Ok(f))
    }

    /// Compiles the task's grammar at submit. `Ok(None)` for a task without
    /// one. Schema errors are `InvalidConfig` (400).
    pub(crate) fn compile(&self, task: &GenerationTask) -> EngineResult<Option<CompiledGrammar>> {
        let Some(spec) = task.grammar.as_ref() else {
            return Ok(None);
        };
        let factory = self.0.as_ref().map_err(|reason| {
            EngineError::InvalidConfig(format!(
                "structured output is not available for this model: {reason}"
            ))
        })?;
        let compiled = factory.compile(spec)?;
        if !compiled.warnings().is_empty() {
            info!(
                task = %task.task_id,
                warnings = ?compiled.warnings(),
                "response_format schema uses keywords that are accepted but not enforced"
            );
        }
        Ok(Some(compiled))
    }
}

/// What a json task is told while its pipeline can't apply masks.
pub(crate) const UPGRADE_EVERY_STAGE: &str =
    "a later pipeline stage runs a build without structured-output support; upgrade every stage";

/// A masked step that came back without an acknowledgement.
pub(crate) fn unconfirmed_step(reason: &str) -> String {
    format!(
        "structured output: the pipeline did not acknowledge the masked step ({reason}); \
         likely cause: {UPGRADE_EVERY_STAGE}"
    )
}

/// A masked step a later stage acknowledged but could not serve.
pub(crate) const REFUSED_STEP: &str =
    "structured output: a later pipeline stage could not apply the mask for this step";

/// The mask for the task's next token as wire words over `width` tokens
/// (bit `i` allows token `i`). `Err` at a grammar dead end.
pub(crate) fn mask_words(gm: &mut GrammarMask, width: usize) -> EngineResult<Vec<u32>> {
    let mut row = vec![0.0f32; width];
    gm.apply(&mut row)?;
    let mut words = vec![0u32; width.div_ceil(32)];
    for (i, v) in row.iter().enumerate() {
        if *v != f32::NEG_INFINITY {
            words[i / 32] |= 1 << (i % 32);
        }
    }
    Ok(words)
}

/// Rank 0 of a pipeline: tags for masked steps, and the latch that remembers
/// the pipeline could not apply one.
#[derive(Debug, Default)]
pub(crate) struct StageMasks {
    pub(crate) latch: MaskLatch,
    tag: u32,
}

impl StageMasks {
    /// A fresh non-zero tag.
    pub(crate) fn next_tag(&mut self) -> u32 {
        self.tag = self.tag.wrapping_add(1).max(1);
        self.tag
    }

    /// The error for a json task while the latch refuses, else `None`.
    pub(crate) fn refusal(&self) -> Option<String> {
        self.latch
            .refuses()
            .then(|| UPGRADE_EVERY_STAGE.to_string())
    }

    /// Mask columns for a `rows`-row frame whose last row picks the next
    /// token of `gm`'s task, with that row's tag. `Err` at a dead end.
    pub(crate) fn last_row(
        &mut self,
        gm: &mut GrammarMask,
        width: usize,
        rows: usize,
    ) -> EngineResult<(MaskCols, u32)> {
        let words = mask_words(gm, width)?;
        let tag = self.next_tag();
        let mut cols = MaskCols::new(rows, words.len());
        cols.set(rows - 1, tag, &words);
        Ok((cols, tag))
    }

    /// The token of a masked step's reply, or the task's error; latches on
    /// a reply without acknowledgement.
    pub(crate) fn outcome(&self, reply: MaskedReply) -> Result<i64, String> {
        match reply {
            MaskedReply::Token(t) => Ok(t),
            MaskedReply::Refused => Err(REFUSED_STEP.to_string()),
            MaskedReply::Unconfirmed(reason) => {
                self.latch.set();
                warn!(%reason, "masked step not acknowledged; json requests refused until a stage acknowledges or the re-probe interval passes");
                Err(unconfirmed_step(&reason))
            }
        }
    }
}

/// A fresh matcher for a starting task, advanced over the resume seed (tokens
/// already in the output). A seed the grammar rejects is a server error.
pub(crate) fn activate(compiled: &CompiledGrammar, seed: &[u32]) -> EngineResult<GrammarMask> {
    let mut gm = compiled.matcher()?;
    if !seed.is_empty() {
        gm.replay(seed).map_err(|e| match e {
            EngineError::InvalidConfig(m) => EngineError::Backend(m),
            other => other,
        })?;
    }
    Ok(gm)
}

/// Advances the matcher past a committed token. `Ok(true)` when the JSON is
/// complete (the task ends with `finish_reason: "stop"`).
pub(crate) fn commit(gm: Option<&mut GrammarMask>, token: i64) -> EngineResult<bool> {
    let Some(gm) = gm else {
        return Ok(false);
    };
    let token = u32::try_from(token)
        .map_err(|_| EngineError::Backend(format!("grammar: token id {token} out of range")))?;
    gm.accept(token)?;
    Ok(gm.finished())
}

/// Test tokenizer: token `b` is byte `b` (byte-level BPE, no merges), plus
/// [`test_env::EOS`] as a special end token.
#[cfg(test)]
pub(crate) mod test_env {
    use super::*;

    pub(crate) const EOS: u32 = 256;
    pub(crate) const VOCAB: usize = 257;

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

    pub(crate) fn tokenizer() -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        for b in 0..=255u8 {
            vocab.insert(byte_char(b).to_string(), u32::from(b).into());
        }
        let byte_level = serde_json::json!({"type": "ByteLevel", "add_prefix_space": false,
            "trim_offsets": true, "use_regex": true});
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [{"id": EOS, "content": "<|end|>", "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
            "normalizer": null, "pre_tokenizer": byte_level, "post_processor": null,
            "decoder": byte_level,
            "model": {"type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false,
                "byte_fallback": false, "vocab": vocab, "merges": []},
        });
        Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
    }

    pub(crate) fn factory() -> GrammarFactory {
        GrammarFactory::from_tokenizer(&tokenizer(), &[EOS]).unwrap()
    }

    pub(crate) const WEATHER: &str = r#"{"type":"object","properties":{"city":{"type":"string"},"conditions":{"type":"string","enum":["sunny","cloudy","rainy"]}},"required":["city","conditions"],"additionalProperties":false}"#;

    pub(crate) fn spec(body: &str) -> cascadia_types::GrammarSpec {
        cascadia_types::GrammarSpec {
            kind: cascadia_types::GrammarKind::JsonSchema,
            body: body.to_string(),
        }
    }

    pub(crate) fn mask(body: &str) -> GrammarMask {
        factory().compile(&spec(body)).unwrap().matcher().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::test_env::*;
    use super::*;

    fn task_with(body: Option<&str>) -> GenerationTask {
        let mut t = GenerationTask::new("t", "hi");
        t.grammar = body.map(spec);
        t
    }

    #[test]
    fn task_without_grammar_compiles_to_nothing_even_without_a_factory() {
        assert!(GrammarSlot::default()
            .compile(&task_with(None))
            .unwrap()
            .is_none());
    }

    #[test]
    fn missing_factory_is_a_400_naming_the_reason() {
        let err = GrammarSlot::default()
            .compile(&task_with(Some(WEATHER)))
            .err();
        match err {
            Some(EngineError::InvalidConfig(m)) => assert!(
                m.starts_with("structured output is not available for this model: "),
                "{m}"
            ),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn factory_without_an_eos_in_the_vocab_reports_why() {
        let slot = GrammarSlot::build(&tokenizer(), &[9_999]);
        let err = slot.compile(&task_with(Some(WEATHER))).err();
        assert!(
            matches!(err, Some(EngineError::InvalidConfig(_))),
            "{err:?}"
        );
    }

    #[test]
    fn unbuildable_schema_is_invalid_config() {
        let slot = GrammarSlot::from_factory(factory());
        let err = slot.compile(&task_with(Some(r#"{"type":42}"#))).err();
        assert!(
            matches!(err, Some(EngineError::InvalidConfig(_))),
            "{err:?}"
        );
    }

    #[test]
    fn activation_replays_a_valid_seed_and_rejects_a_bad_one_as_backend() {
        let compiled = factory().compile(&spec(WEATHER)).unwrap();
        let mut gm = activate(&compiled, b"{\"city\":\"P".map(u32::from).as_slice()).unwrap();
        assert!(gm.allowed(u32::from(b'a')));
        let err = activate(&compiled, b"[1]".map(u32::from).as_slice()).err();
        assert!(matches!(err, Some(EngineError::Backend(_))), "{err:?}");
    }

    #[test]
    fn commit_reports_the_finish_and_rejects_a_disallowed_token() {
        let mut gm = mask(WEATHER);
        for &b in br#"{"city":"P","conditions":"sunny""# {
            assert!(!commit(Some(&mut gm), i64::from(b)).unwrap());
        }
        assert!(commit(Some(&mut gm), i64::from(b'}')).unwrap());
        let mut gm = mask(WEATHER);
        assert!(commit(Some(&mut gm), i64::from(b'x')).is_err());
        assert!(!commit(None, 5).unwrap());
    }
}
