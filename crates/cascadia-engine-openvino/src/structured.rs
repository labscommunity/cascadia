//! Structured output (`response_format`) plumbing shared by the step-wise
//! engines: the per-model grammar factory, the submit-time grammar build and
//! the activation of a task's matcher. Each engine keeps its own topology
//! checks and its own multi-stage refusal.

use cascadia_engine::{EngineError, EngineResult};
use cascadia_grammar::{CompiledGrammar, GrammarFactory, GrammarMask};
use cascadia_types::{GenerationTask, TaskId};
use tokenizers::Tokenizer;
use tracing::{info, warn};

/// The grammar factory for `tokenizer`, built from every EOS id generation
/// stops on, or why structured output is unavailable. A failure is logged once
/// here, prefixed with `engine`; json tasks then get the reason at submit.
pub(crate) fn build_grammar_factory(
    engine: &str,
    tokenizer: &Tokenizer,
    eos: &[u32],
) -> Result<GrammarFactory, String> {
    GrammarFactory::from_tokenizer(tokenizer, eos)
        .map_err(|e| match e {
            EngineError::InvalidConfig(m) | EngineError::Backend(m) => m,
            other => other.to_string(),
        })
        .inspect_err(|reason| unavailable(engine, reason))
}

/// Logs why structured output is unavailable on `engine`.
pub(crate) fn unavailable(engine: &str, reason: &str) {
    warn!(%reason, "{engine}: structured output unavailable for this model");
}

/// Submit-time grammar build: `None` for a task without a grammar; an
/// `InvalidConfig` (400) when the model can't do structured output or the
/// schema doesn't compile.
pub(crate) fn compile_task_grammar(
    factory: &Result<GrammarFactory, String>,
    task: &GenerationTask,
) -> EngineResult<Option<CompiledGrammar>> {
    let Some(spec) = task.grammar.as_ref() else {
        return Ok(None);
    };
    let factory = factory.as_ref().map_err(|reason| {
        EngineError::InvalidConfig(format!(
            "structured output is not available for this model: {reason}"
        ))
    })?;
    factory.compile(spec).map(Some)
}

/// One `info!` per task whose schema uses keywords the grammar accepts but
/// does not enforce.
pub(crate) fn log_grammar_warnings(
    engine: &str,
    task_id: &TaskId,
    grammar: Option<&CompiledGrammar>,
) {
    if let Some(g) = grammar.filter(|g| !g.warnings().is_empty()) {
        info!(
            task = %task_id,
            warnings = ?g.warnings(),
            "{engine}: response_format keywords accepted but not enforced"
        );
    }
}

/// Activation: a fresh matcher advanced over the resume seed. A seed the
/// grammar rejects is a backend error.
pub(crate) fn start_grammar(compiled: &CompiledGrammar, seed: &[i64]) -> EngineResult<GrammarMask> {
    let mut mask = compiled.matcher()?;
    if !seed.is_empty() {
        let seed: Vec<u32> = seed.iter().map(|&t| t as u32).collect();
        mask.replay(&seed)?;
    }
    Ok(mask)
}

/// The allowed set of a grammar task's next pick as u32 words over `width`
/// tokens (bit `t % 32` of word `t / 32`: token `t` allowed; no bit at or past
/// `width`), read from the matcher's cached mask without a logits row. `Err`
/// in the error state and when nothing below `width` is allowed: the head
/// fails the task rather than send a mask that allows nothing.
pub(crate) fn mask_words_over(mask: &mut GrammarMask, width: usize) -> EngineResult<Vec<u32>> {
    let mut words = mask.mask_words()?;
    words.resize(width.div_ceil(32), 0);
    if let (Some(last), tail @ 1..) = (words.last_mut(), width % 32) {
        *last &= (1u32 << tail) - 1;
    }
    if words.iter().all(|&w| w == 0) {
        return Err(EngineError::Backend(
            "grammar: no token of the logits row is allowed".into(),
        ));
    }
    Ok(words)
}

/// What [`mask_words_over`] replaced: the allowed set read back through
/// `apply` on a `width`-wide zero row. Tests compare the two.
#[cfg(test)]
pub(crate) fn words_via_apply(mask: &mut GrammarMask, width: usize) -> EngineResult<Vec<u32>> {
    let mut row = vec![0f32; width];
    mask.apply(&mut row)?;
    let mut words = vec![0u32; width.div_ceil(32)];
    for (i, v) in row.iter().enumerate() {
        if v.is_finite() {
            words[i / 32] |= 1 << (i % 32);
        }
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The EOS of [`GrammarFactory::single_byte_for_tests`].
    const TEST_EOS: u32 = 261;
    const OK_ENUM: &str = r#"{"type":"string","enum":["ok"]}"#;
    const WEATHER: &str = r#"{"type":"object","properties":{"city":{"type":"string"},"temperature_celsius":{"type":"number"},"conditions":{"type":"string","enum":["sunny","cloudy","rainy"]}},"required":["city","temperature_celsius","conditions"],"additionalProperties":false}"#;

    fn json_spec(body: &str) -> cascadia_types::GrammarSpec {
        cascadia_types::GrammarSpec {
            kind: cascadia_types::GrammarKind::JsonSchema,
            body: body.to_string(),
        }
    }

    fn compiled(body: &str) -> CompiledGrammar {
        GrammarFactory::single_byte_for_tests()
            .compile(&json_spec(body))
            .unwrap()
    }

    fn ids(s: &str) -> Vec<i64> {
        s.bytes().map(i64::from).collect()
    }

    #[test]
    fn submit_compiles_the_grammar_or_refuses_with_invalid_config() {
        let gf: Result<GrammarFactory, String> = Ok(GrammarFactory::single_byte_for_tests());
        let plain = GenerationTask::new("t", "p");
        assert!(compile_task_grammar(&gf, &plain).unwrap().is_none());

        let good = GenerationTask::new("t", "p").with_grammar(json_spec(WEATHER));
        assert!(compile_task_grammar(&gf, &good).unwrap().is_some());

        let bad = GenerationTask::new("t", "p").with_grammar(json_spec(r#"{"type":42}"#));
        assert!(matches!(
            compile_task_grammar(&gf, &bad),
            Err(EngineError::InvalidConfig(_))
        ));

        let none: Result<GrammarFactory, String> = Err("no EOS in vocab".into());
        assert!(compile_task_grammar(&none, &plain).unwrap().is_none());
        match compile_task_grammar(&none, &good) {
            Err(EngineError::InvalidConfig(m)) => assert_eq!(
                m,
                "structured output is not available for this model: no EOS in vocab"
            ),
            other => panic!(
                "expected InvalidConfig, got {:?}",
                other.map(|g| g.is_some())
            ),
        }
    }

    #[test]
    fn the_resume_seed_is_replayed_and_a_rejected_seed_is_a_backend_error() {
        let cg = compiled(OK_ENUM);
        let mut mask = start_grammar(&cg, &ids("\"o")).unwrap();
        assert!(mask.allowed(b'k' as u32));
        assert!(!mask.allowed(b'o' as u32));
        let err = start_grammar(&cg, &ids("x")).err().unwrap();
        assert!(matches!(err, EngineError::Backend(_)), "{err:?}");
        assert!(matches!(
            start_grammar(&cg, &ids("\"x")),
            Err(EngineError::Backend(_))
        ));
        // The EOS of a finished grammar is still allowed.
        let mut done = start_grammar(&cg, &ids("\"ok\"")).unwrap();
        assert!(done.allowed(TEST_EOS));
    }

    #[test]
    fn a_fresh_matcher_starts_at_the_grammar_start() {
        let mut mask = start_grammar(&compiled(OK_ENUM), &[]).unwrap();
        assert!(mask.allowed(b'"' as u32));
        assert!(!mask.allowed(b'o' as u32));
    }

    #[test]
    fn mask_words_over_equal_the_apply_readback_at_any_width() {
        let full = GrammarFactory::single_byte_for_tests().mask_width();
        let cg = compiled(WEATHER);
        for prefix in [
            "",
            "{\"",
            "{\"city\":\"Pa",
            "{\"city\":\"P\",\"temperature_celsius\":1",
        ] {
            let mut mask = start_grammar(&cg, &ids(prefix)).unwrap();
            for width in [1, 100, 124, 128, full - 1, full, full + 40] {
                let want = words_via_apply(&mut mask, width);
                let got = mask_words_over(&mut mask, width);
                match (want, got) {
                    (Ok(w), Ok(g)) => assert_eq!(g, w, "{prefix:?} at {width}"),
                    (Err(_), Err(_)) => {}
                    (w, g) => panic!("{prefix:?} at {width}: {w:?} vs {g:?}"),
                }
            }
        }
        let mut done = start_grammar(&compiled(OK_ENUM), &ids("\"ok\"")).unwrap();
        assert_eq!(
            mask_words_over(&mut done, full).unwrap(),
            words_via_apply(&mut done, full).unwrap()
        );
        // EOS (261) lies past 256 tokens: nothing allowed there.
        assert!(mask_words_over(&mut done, 256).is_err());
    }
}
