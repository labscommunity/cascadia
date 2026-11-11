//! Structured output (`response_format`): the per-engine glue around
//! [`cascadia_grammar`]. One [`GrammarSlot`] per engine (set up where rank 0
//! loads its tokenizer, built on the first json task), one [`CompiledGrammar`] per queued task (compiled at
//! `submit`, so a schema that can't be built is a 400 before the task queues),
//! one [`GrammarMask`] per running sequence (created when the task starts).
//!
//! Tasks without a grammar never touch this module: no mask, no matcher.

use cascadia_engine::{EngineError, EngineResult};
use cascadia_grammar::{CompiledGrammar, GrammarFactory, GrammarMask};
use cascadia_metrics::{count_constraint_unenforced, unenforced_warns, UnenforcedReason};

use crate::dist::{MaskCols, MaskLatch, MaskedReply};
use cascadia_types::{GenerationTask, GrammarDropped, TaskId};
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

/// Builds the factory. Run again only after a [`BuildFailure::transient`]
/// failure.
type BuildFactory = Box<dyn FnMut() -> Result<GrammarFactory, BuildFailure> + Send>;

/// Why a factory build failed, and whether a later build may succeed.
struct BuildFailure {
    reason: String,
    /// The tokenizer file could not be read (an I/O error): the next json
    /// request at least [`FACTORY_RETRY`] later builds again. Otherwise the
    /// failure comes from the tokenizer's content and is kept.
    transient: bool,
}

impl BuildFailure {
    fn transient(reason: String) -> Self {
        Self {
            reason,
            transient: true,
        }
    }
}

impl From<EngineError> for BuildFailure {
    fn from(e: EngineError) -> Self {
        let reason = match e {
            EngineError::InvalidConfig(m) | EngineError::Backend(m) => m,
            other => other.to_string(),
        };
        Self {
            reason,
            transient: false,
        }
    }
}

/// The least time between two builds after a transient failure.
const FACTORY_RETRY: std::time::Duration = std::time::Duration::from_secs(60);

/// The engine's grammar factory, built on the first json task (building it
/// costs about half a second and over 100 MiB for a large vocabulary, which
/// requests without a grammar never need), or why it has none. Copies share
/// the build and its result.
#[derive(Clone, Default)]
pub(crate) struct GrammarSlot(Option<std::sync::Arc<LazyFactory>>);

struct LazyFactory {
    /// The build, until it succeeds or fails for good; and the last
    /// transient failure with its time.
    pending: std::sync::Mutex<Pending>,
    built: std::sync::OnceLock<Result<GrammarFactory, String>>,
}

struct Pending {
    build: Option<BuildFactory>,
    retry: Option<(std::time::Instant, String)>,
}

/// Why a slot without a build has no factory.
const NO_TOKENIZER: &str = "the model has no tokenizer on this stage";

/// The message carried by a caught panic.
fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "no message".to_string())
}

impl GrammarSlot {
    /// A slot whose factory `build` makes on first use. A failure (or a
    /// panic) is logged then, once, and kept: json tasks get a 400 naming
    /// it.
    #[cfg(test)]
    pub(crate) fn lazy(
        build: impl FnOnce() -> EngineResult<GrammarFactory> + Send + 'static,
    ) -> Self {
        let mut build = Some(build);
        Self::retrying(move || match build.take() {
            Some(b) => b().map_err(BuildFailure::from),
            None => Err(BuildFailure::from(EngineError::Backend(
                "the grammar factory build ran twice".into(),
            ))),
        })
    }

    fn retrying(
        build: impl FnMut() -> Result<GrammarFactory, BuildFailure> + Send + 'static,
    ) -> Self {
        Self(Some(std::sync::Arc::new(LazyFactory {
            pending: std::sync::Mutex::new(Pending {
                build: Some(Box::new(build)),
                retry: None,
            }),
            built: std::sync::OnceLock::new(),
        })))
    }

    /// The factory for the tokenizer at `path` and every EOS id the engine
    /// stops on; the file is read again on first use, so the engine keeps
    /// no second copy of the tokenizer meanwhile. A read error is retried
    /// on a later json request (see [`FACTORY_RETRY`]); a tokenizer that
    /// can't be parsed or has no usable EOS is not.
    pub(crate) fn from_tokenizer_file(path: std::path::PathBuf, eos: Vec<u32>) -> Self {
        Self::retrying(move || {
            let bytes = std::fs::read(&path)
                .map_err(|e| BuildFailure::transient(format!("read {}: {e}", path.display())))?;
            let tok = Tokenizer::from_bytes(&bytes).map_err(|e| {
                BuildFailure::from(EngineError::Backend(format!(
                    "load {}: {e}",
                    path.display()
                )))
            })?;
            Ok(GrammarFactory::from_tokenizer(&tok, &eos)?)
        })
    }

    /// [`Self::lazy`] over a tokenizer in memory.
    #[cfg(test)]
    pub(crate) fn build(tokenizer: &Tokenizer, eos: &[u32]) -> Self {
        let (tok, eos) = (tokenizer.clone(), eos.to_vec());
        Self::lazy(move || GrammarFactory::from_tokenizer(&tok, &eos))
    }

    #[cfg(test)]
    pub(crate) fn from_factory(f: GrammarFactory) -> Self {
        Self::lazy(move || Ok(f))
    }

    /// The factory, built now if this is its first use.
    fn factory(&self) -> Result<&GrammarFactory, String> {
        self.factory_at(std::time::Instant::now())
    }

    /// [`Self::factory`] at `now`.
    fn factory_at(&self, now: std::time::Instant) -> Result<&GrammarFactory, String> {
        let Some(lazy) = self.0.as_ref() else {
            return Err(NO_TOKENIZER.to_string());
        };
        if let Some(done) = lazy.built.get() {
            return done.as_ref().map_err(Clone::clone);
        }
        let mut pending = lazy.pending.lock().unwrap_or_else(|e| e.into_inner());
        // Another caller may have finished the build while this one waited.
        if let Some(done) = lazy.built.get() {
            return done.as_ref().map_err(Clone::clone);
        }
        if let Some((at, reason)) = &pending.retry {
            if now.saturating_duration_since(*at) < FACTORY_RETRY {
                return Err(reason.clone());
            }
        }
        let Some(build) = pending.build.as_mut() else {
            return Err("the grammar factory build did not finish".into());
        };
        let started = std::time::Instant::now();
        let outcome = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(build)) {
            Ok(Ok(f)) => {
                info!(
                    mask_width = f.mask_width(),
                    build_ms = started.elapsed().as_millis() as u64,
                    "structured output: grammar factory built"
                );
                Ok(f)
            }
            Ok(Err(BuildFailure {
                reason,
                transient: true,
            })) => {
                warn!(
                    error = %reason,
                    retry_after_s = FACTORY_RETRY.as_secs(),
                    "structured output unavailable for now: reading the tokenizer failed"
                );
                pending.retry = Some((now, reason.clone()));
                return Err(reason);
            }
            Ok(Err(BuildFailure { reason, .. })) => {
                warn!(error = %reason, "structured output unavailable for this model");
                Err(reason)
            }
            Err(payload) => {
                let reason = format!(
                    "the grammar factory build panicked: {}",
                    panic_reason(payload)
                );
                warn!(error = %reason, "structured output unavailable for this model");
                Err(reason)
            }
        };
        *pending = Pending {
            build: None,
            retry: None,
        };
        let done = lazy.built.get_or_init(|| outcome);
        drop(pending);
        done.as_ref().map_err(Clone::clone)
    }

    /// Tokens a mask covers (the tokenizer vocab); `None` without a factory.
    /// Builds the factory on first use.
    pub(crate) fn mask_width(&self) -> Option<usize> {
        self.factory().ok().map(GrammarFactory::mask_width)
    }

    /// Compiles the task's grammar at submit. `Ok(None)` for a task without
    /// one. Schema errors are `InvalidConfig` (400).
    pub(crate) fn compile(&self, task: &GenerationTask) -> EngineResult<Option<CompiledGrammar>> {
        let Some(spec) = task.grammar.as_ref() else {
            return Ok(None);
        };
        let factory = self.factory().map_err(|reason| {
            EngineError::InvalidConfig(format!(
                "structured output is not available for this model: {reason}"
            ))
        })?;
        let compiled = factory.compile(spec)?;
        if !compiled.warnings().is_empty() {
            info!(
                task = %task.task_id,
                warnings = ?compiled.warnings(),
                "sparse-moe: response_format keywords accepted but not enforced"
            );
        }
        Ok(Some(compiled))
    }
}

impl GrammarSlot {
    /// [`Self::compile`] at submit: a best-effort grammar that has no factory
    /// or doesn't compile is dropped from the task (see
    /// [`drop_best_effort_grammar`]) and the task runs unconstrained; a
    /// strict one is refused as before.
    pub(crate) fn admit(&self, task: &mut GenerationTask) -> EngineResult<Option<CompiledGrammar>> {
        match self.compile(task) {
            Err(e) if is_best_effort(task) => {
                let reason = if self.factory().is_err() {
                    UnenforcedReason::NoTokenizer
                } else {
                    UnenforcedReason::SchemaRejected
                };
                info!(task = %task.task_id, error = %e, "a best-effort grammar can't be built");
                drop_best_effort_grammar(task, reason);
                Ok(None)
            }
            other => other,
        }
    }
}

/// Whether `task` carries a best-effort grammar.
pub(crate) fn is_best_effort(task: &GenerationTask) -> bool {
    task.grammar.as_ref().is_some_and(|g| g.best_effort)
}

/// Removes a best-effort grammar the pipeline can't apply from `task`,
/// logging and counting why, so the task runs unconstrained. `false` (the
/// task is untouched) for a task without a grammar or with a strict one: the
/// caller then refuses it as before.
pub(crate) fn drop_best_effort_grammar(
    task: &mut GenerationTask,
    reason: UnenforcedReason,
) -> bool {
    // Recorded on the task: its final chunk then says the answer was
    // generated unconstrained.
    if !task.drop_best_effort_grammar() {
        return false;
    }
    note_dropped(&task.task_id, reason);
    true
}

/// Logs and counts one best-effort grammar dropped for `reason`.
fn note_dropped(task_id: &TaskId, reason: UnenforcedReason) {
    // A per-server reason would warn on every request: once, then debug.
    if unenforced_warns(reason) {
        warn!(
            task = %task_id,
            reason = reason.as_str(),
            "output constraint not enforced; answering unconstrained"
        );
    } else {
        debug!(
            task = %task_id,
            reason = reason.as_str(),
            "output constraint not enforced; answering unconstrained"
        );
    }
    count_constraint_unenforced(reason);
}

/// A running task's handle for dropping its grammar mid-generation: the
/// task's id and drop flag while its grammar is best-effort, empty for a
/// strict grammar (or none). Taken by the first drop.
#[derive(Clone, Debug, Default)]
pub(crate) struct BestEffort(Option<(TaskId, GrammarDropped)>);

impl BestEffort {
    /// The handle of `task`, empty unless its grammar is best-effort.
    pub(crate) fn of(task: &GenerationTask) -> Self {
        Self(is_best_effort(task).then(|| (task.task_id.clone(), task.grammar_dropped.clone())))
    }

    /// The task's matcher failed while generating (a token commit or a mask
    /// ran past a parser limit, or the grammar allows no token): a
    /// best-effort grammar is dropped, `grammar` cleared and the drop
    /// recorded on the task, logged and counted as `too_complex`, and the
    /// task carries on unconstrained from the output it has. `false`,
    /// nothing changed, for a strict grammar: the caller fails the task as
    /// before.
    pub(crate) fn drop_failed<G>(
        &mut self,
        grammar: &mut Option<G>,
        error: &dyn std::fmt::Display,
    ) -> bool {
        if grammar.is_none() {
            return false;
        }
        let Some((task_id, dropped)) = self.0.take() else {
            return false;
        };
        dropped.set();
        *grammar = None;
        info!(task = %task_id, %error, "a best-effort grammar failed while generating");
        note_dropped(&task_id, UnenforcedReason::TooComplex);
        true
    }

    /// Before a step that would send or apply `grammar`'s mask: a
    /// best-effort grammar whose mask can't be computed is dropped
    /// ([`Self::drop_failed`]), so the step goes unmasked. The matcher
    /// caches the mask, so the step's own read of it costs nothing. A
    /// strict grammar is not touched here: its step fails as before.
    pub(crate) fn drop_unmaskable<G: std::borrow::BorrowMut<GrammarMask>>(
        &mut self,
        grammar: &mut Option<G>,
    ) -> bool {
        if self.0.is_none() {
            return false;
        }
        match grammar.as_mut().map(|g| g.borrow_mut().mask_words()) {
            Some(Err(e)) => self.drop_failed(grammar, &e),
            _ => false,
        }
    }

    /// A masked pick `pick(mask)`, made again unmasked after dropping a
    /// best-effort grammar whose mask failed ([`Self::drop_failed`]).
    pub(crate) fn pick<G: std::borrow::BorrowMut<GrammarMask>>(
        &mut self,
        grammar: &mut Option<G>,
        mut pick: impl FnMut(Option<&mut GrammarMask>) -> EngineResult<i64>,
    ) -> EngineResult<i64> {
        match pick(grammar.as_mut().map(|g| g.borrow_mut())) {
            Err(e) if self.drop_failed(grammar, &e) => pick(None),
            other => other,
        }
    }

    /// [`commit`], dropping a best-effort grammar the token can't be
    /// committed to ([`Self::drop_failed`]): `Ok(false)` then, the task
    /// carries on unconstrained.
    pub(crate) fn commit<G: std::borrow::BorrowMut<GrammarMask>>(
        &mut self,
        grammar: &mut Option<G>,
        token: i64,
    ) -> EngineResult<bool> {
        match commit(grammar.as_mut().map(|g| g.borrow_mut()), token) {
            Err(e) if self.drop_failed(grammar, &e) => Ok(false),
            other => other,
        }
    }
}

/// Rank 0's admission of a json task while its pipeline's latch refuses
/// (set by an unacknowledged step or an unanswered probe, this one's
/// included): a best-effort grammar is dropped, with its compiled form, and
/// the task runs unmasked. An expired latch drops nothing: the task
/// re-probes like a strict one. `true` when dropped.
pub(crate) fn drop_on_latched_link(
    task: &mut GenerationTask,
    compiled: &mut Option<CompiledGrammar>,
    masks: &StageMasks,
) -> bool {
    let dropped = compiled.is_some()
        && masks.latch.refuses()
        && drop_best_effort_grammar(task, UnenforcedReason::StageUnsupported);
    if dropped {
        *compiled = None;
    }
    dropped
}

/// What a json task is told while its pipeline can't apply masks.
pub(crate) const UPGRADE_EVERY_STAGE: &str = cascadia_grammar::STAGE_UPGRADE_MESSAGE;

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

/// The mask for the task's next token as wire words over the tokenizer
/// vocab (bit `i` allows token `i`). `Err` at a grammar dead end.
pub(crate) fn mask_words(gm: &mut GrammarMask) -> EngineResult<Vec<u32>> {
    gm.mask_words()
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
        rows: usize,
    ) -> EngineResult<(MaskCols, u32)> {
        let words = mask_words(gm)?;
        let tag = self.next_tag();
        let mut cols = MaskCols::new(rows, words.len());
        cols.set(rows - 1, tag, &words);
        Ok((cols, tag))
    }

    /// A masked stream frame whose reply failed (a timeout, a closed link, a
    /// frame out of step): the streams' error. On a link without evidence
    /// that every stage masks, a stage on an older build that never answers
    /// is the likely cause, so json requests are refused for a while. On a
    /// link whose stages acknowledged a mask, a slow or dead stage is: the
    /// plain error, and no latch.
    pub(crate) fn stream_failure(&self, reason: String, evidence: bool) -> String {
        if evidence {
            return reason;
        }
        self.latch.set();
        unconfirmed_step(&reason)
    }

    /// The token of a masked step's reply, or the task's error; latches on
    /// a reply without acknowledgement.
    pub(crate) fn outcome(&self, reply: MaskedReply) -> Result<i64, String> {
        match reply {
            MaskedReply::Token(t) => Ok(t),
            MaskedReply::Refused => Err(REFUSED_STEP.to_string()),
            // A stage lost its mask state: fail the task (the client may
            // retry); nothing is latched and no evidence is recorded.
            MaskedReply::MaskMiss => Err(cascadia_grammar::MASK_MISS_MESSAGE.to_string()),
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
        cascadia_types::GrammarSpec::json_schema(body)
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

    /// A slot whose factory build counts its runs and ends in `result`.
    fn counting_slot(ok: bool) -> (GrammarSlot, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let runs = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let r = runs.clone();
        let slot = GrammarSlot::lazy(move || {
            r.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if ok {
                Ok(factory())
            } else {
                Err(EngineError::InvalidConfig("no EOS in the vocab".into()))
            }
        });
        (slot, runs)
    }

    #[test]
    fn the_factory_is_built_on_the_first_json_task_only() {
        use std::sync::atomic::Ordering::SeqCst;
        let (slot, runs) = counting_slot(true);
        let copy = slot.clone();
        assert!(slot.compile(&task_with(None)).unwrap().is_none());
        assert_eq!(runs.load(SeqCst), 0, "built before a json task");
        assert!(slot.compile(&task_with(Some(WEATHER))).unwrap().is_some());
        assert!(copy.compile(&task_with(Some(WEATHER))).unwrap().is_some());
        assert_eq!(slot.mask_width(), Some(VOCAB));
        assert_eq!(runs.load(SeqCst), 1, "built once, shared by copies");
    }

    #[test]
    fn a_failed_factory_build_is_cached_and_every_json_task_gets_a_400() {
        use std::sync::atomic::Ordering::SeqCst;
        let (slot, runs) = counting_slot(false);
        for _ in 0..2 {
            match slot.compile(&task_with(Some(WEATHER))).err() {
                Some(EngineError::InvalidConfig(m)) => assert_eq!(
                    m,
                    "structured output is not available for this model: no EOS in the vocab"
                ),
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(runs.load(SeqCst), 1);
        assert_eq!(slot.mask_width(), None);
    }

    #[test]
    fn a_slot_from_a_tokenizer_file_reads_it_on_the_first_json_task() {
        let dir = std::env::temp_dir().join(format!("grammar-slot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        let slot = GrammarSlot::from_tokenizer_file(path.clone(), vec![EOS]);
        // Written after the slot exists: construction did not read it.
        tokenizer().save(&path, false).unwrap();
        assert!(slot.compile(&task_with(Some(WEATHER))).unwrap().is_some());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Warn-level events (their `message` and `error` fields) logged while
    /// `f` runs on this thread.
    fn warnings_during<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
        use std::sync::{Arc, Mutex};
        use tracing::field::{Field, Visit};
        #[derive(Clone, Default)]
        struct Collect(Arc<Mutex<Vec<String>>>);
        struct Fields(String);
        impl Visit for Fields {
            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                self.0.push_str(&format!("{}={value:?} ", field.name()));
            }
        }
        impl tracing::Subscriber for Collect {
            fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                tracing::span::Id::from_u64(1)
            }
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            fn event(&self, event: &tracing::Event<'_>) {
                if *event.metadata().level() == tracing::Level::WARN {
                    let mut v = Fields(String::new());
                    event.record(&mut v);
                    self.0.lock().unwrap().push(v.0);
                }
            }
            fn enter(&self, _: &tracing::span::Id) {}
            fn exit(&self, _: &tracing::span::Id) {}
        }
        let c = Collect::default();
        let out = tracing::subscriber::with_default(c.clone(), f);
        let logged = c.0.lock().unwrap().clone();
        (out, logged)
    }

    fn unavailable(slot: &GrammarSlot) -> String {
        match slot.compile(&task_with(Some(WEATHER))).err() {
            Some(EngineError::InvalidConfig(m)) => m,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_panicking_factory_build_is_logged_once_and_kept() {
        use std::sync::atomic::Ordering::SeqCst;
        let runs = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let r = runs.clone();
        let slot = GrammarSlot::lazy(move || {
            r.fetch_add(1, SeqCst);
            panic!("vocab table out of range")
        });
        let ((first, second), logged) =
            warnings_during(|| (unavailable(&slot), unavailable(&slot)));
        assert!(
            first.contains("panicked: vocab table out of range"),
            "{first}"
        );
        assert_eq!(first, second);
        assert_eq!(runs.load(SeqCst), 1);
        assert_eq!(logged.len(), 1, "{logged:?}");
        assert!(logged[0].contains("vocab table out of range"), "{logged:?}");
    }

    #[test]
    fn a_tokenizer_read_error_is_retried_at_most_once_a_minute() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        let slot = GrammarSlot::from_tokenizer_file(path.clone(), vec![EOS]);
        let (first, logged) = warnings_during(|| unavailable(&slot));
        assert!(first.contains("tokenizer.json"), "{first}");
        assert_eq!(logged.len(), 1, "{logged:?}");
        // The file turns up; within the minute the failure stands.
        tokenizer().save(&path, false).unwrap();
        let (again, logged) = warnings_during(|| unavailable(&slot));
        assert_eq!(again, first);
        assert!(logged.is_empty(), "{logged:?}");
        // A minute later the next json request reads it again.
        let later = std::time::Instant::now() + FACTORY_RETRY;
        assert_eq!(
            slot.factory_at(later).map(GrammarFactory::mask_width),
            Ok(VOCAB)
        );
        assert!(slot.compile(&task_with(Some(WEATHER))).unwrap().is_some());
    }

    #[test]
    fn a_tokenizer_whose_content_fails_is_kept_failed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, b"{ not a tokenizer").unwrap();
        let slot = GrammarSlot::from_tokenizer_file(path.clone(), vec![EOS]);
        let (first, logged) = warnings_during(|| unavailable(&slot));
        assert_eq!(logged.len(), 1, "{logged:?}");
        tokenizer().save(&path, false).unwrap();
        let later = std::time::Instant::now() + FACTORY_RETRY;
        let kept = slot.factory_at(later).err().expect("still failed");
        assert!(first.ends_with(&kept), "{first} / {kept}");
    }

    fn unenforced(reason: cascadia_metrics::UnenforcedReason) -> u64 {
        cascadia_metrics::CONSTRAINT_UNENFORCED_TOTAL
            .with_label_values(&[reason.as_str()])
            .get()
    }

    fn soft_task(body: &str) -> GenerationTask {
        GenerationTask::new("t", "hi")
            .with_grammar(cascadia_types::GrammarSpec::best_effort_json_schema(body))
    }

    #[test]
    fn admit_drops_a_best_effort_grammar_that_does_not_compile() {
        use cascadia_metrics::UnenforcedReason;
        let slot = GrammarSlot::from_factory(factory());
        let before = unenforced(UnenforcedReason::SchemaRejected);
        let mut t = soft_task(r#"{"type":42}"#);
        assert!(slot.admit(&mut t).unwrap().is_none());
        assert_eq!(t.grammar, None);
        // Recorded on the task, for its final chunk.
        assert!(t.grammar_dropped.get());
        assert!(unenforced(UnenforcedReason::SchemaRejected) > before);
        // One that compiles is kept.
        let mut t = soft_task(WEATHER);
        assert!(slot.admit(&mut t).unwrap().is_some());
        assert!(t.grammar.is_some());
        assert!(!t.grammar_dropped.get());
        // A strict one is refused as before.
        let mut t = task_with(Some(r#"{"type":42}"#));
        assert!(matches!(
            slot.admit(&mut t),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(t.grammar.is_some());
    }

    #[test]
    fn admit_drops_a_best_effort_grammar_without_a_factory() {
        use cascadia_metrics::UnenforcedReason;
        let before = unenforced(UnenforcedReason::NoTokenizer);
        let mut t = soft_task(WEATHER);
        assert!(GrammarSlot::default().admit(&mut t).unwrap().is_none());
        assert_eq!(t.grammar, None);
        assert!(unenforced(UnenforcedReason::NoTokenizer) > before);
        let mut t = task_with(Some(WEATHER));
        assert!(matches!(
            GrammarSlot::default().admit(&mut t),
            Err(EngineError::InvalidConfig(_))
        ));
        let mut plain = task_with(None);
        assert!(GrammarSlot::default().admit(&mut plain).unwrap().is_none());
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
        let mut gm = activate(&compiled, b"{\"city\": \"P".map(u32::from).as_slice()).unwrap();
        assert!(gm.allowed(u32::from(b'a')));
        let err = activate(&compiled, b"[1]".map(u32::from).as_slice()).err();
        assert!(matches!(err, Some(EngineError::Backend(_))), "{err:?}");
    }

    #[test]
    fn mask_words_are_the_matchers_allowed_set() {
        let mut gm = mask(WEATHER);
        gm.replay(&br#"{"city": "P"#.map(u32::from)).unwrap();
        let words = mask_words(&mut gm).unwrap();
        assert_eq!(words.len(), VOCAB.div_ceil(32));
        for t in 0..VOCAB as u32 {
            let bit = words[t as usize / 32] >> (t % 32) & 1 == 1;
            assert_eq!(bit, gm.allowed(t), "token {t}");
        }
        // A rejected token leaves the matcher in its error state: an error.
        let _ = gm.accept(0x01);
        assert!(mask_words(&mut gm).is_err());
    }

    #[test]
    fn a_failed_masked_stream_reply_latches_only_on_a_link_without_evidence() {
        let reason = "stream reply timeout after 1s".to_string();
        // No evidence yet: an older stage that never answers is the likely cause.
        let masks = StageMasks::default();
        let e = masks.stream_failure(reason.clone(), false);
        assert!(masks.latch.is_set());
        assert!(
            e.contains(&reason) && e.ends_with(UPGRADE_EVERY_STAGE),
            "{e}"
        );
        // Every stage on the link acknowledged a mask: a slow or dead stage,
        // the plain stream error, json stays on.
        let masks = StageMasks::default();
        let e = masks.stream_failure(reason.clone(), true);
        assert!(!masks.latch.is_set());
        assert_eq!(e, reason);
    }

    #[test]
    fn commit_reports_the_finish_and_rejects_a_disallowed_token() {
        let mut gm = mask(WEATHER);
        for &b in br#"{"city": "P", "conditions": "sunny""# {
            assert!(!commit(Some(&mut gm), i64::from(b)).unwrap());
        }
        assert!(commit(Some(&mut gm), i64::from(b'}')).unwrap());
        let mut gm = mask(WEATHER);
        assert!(commit(Some(&mut gm), i64::from(b'x')).is_err());
        assert!(!commit(None, 5).unwrap());
    }

    /// A matcher in llguidance's error state, as a parser limit hit
    /// mid-generation leaves it: every later mask and commit fails.
    fn failed_mask(body: &str) -> GrammarMask {
        let mut gm = mask(body);
        assert!(gm.accept(0x01).is_err());
        gm
    }

    #[test]
    fn a_best_effort_grammar_that_fails_mid_generation_is_dropped_and_counted() {
        use cascadia_metrics::UnenforcedReason;
        let before = unenforced(UnenforcedReason::TooComplex);
        let task = soft_task(WEATHER);
        let mut handle = BestEffort::of(&task);
        let mut gm = Some(failed_mask(WEATHER));
        // A failed commit: dropped, the task carries on unconstrained.
        assert!(!handle.commit(&mut gm, i64::from(b'{')).unwrap());
        assert!(gm.is_none());
        assert!(task.grammar_dropped.get());
        assert!(unenforced(UnenforcedReason::TooComplex) > before);
        // The handle is spent: nothing left to drop.
        let mut again = Some(failed_mask(WEATHER));
        assert!(!handle.drop_failed(&mut again, &"x"));
        assert!(again.is_some());

        // A failed mask, before the step: dropped, the step goes unmasked.
        let task = soft_task(WEATHER);
        let mut gm = Some(failed_mask(WEATHER));
        assert!(BestEffort::of(&task).drop_unmaskable(&mut gm));
        assert!(gm.is_none() && task.grammar_dropped.get());
        // A healthy one is kept, its mask computed once.
        let task = soft_task(WEATHER);
        let mut gm = Some(mask(WEATHER));
        assert!(!BestEffort::of(&task).drop_unmaskable(&mut gm));
        assert!(gm.is_some() && !task.grammar_dropped.get());
    }

    #[test]
    fn a_strict_grammar_that_fails_mid_generation_still_errors() {
        let task = task_with(Some(WEATHER));
        let mut handle = BestEffort::of(&task);
        let mut gm = Some(failed_mask(WEATHER));
        assert!(!handle.drop_unmaskable(&mut gm));
        assert!(handle.commit(&mut gm, i64::from(b'{')).is_err());
        assert!(gm.is_some());
        assert!(!task.grammar_dropped.get());
    }
}
