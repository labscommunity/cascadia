//! Mock engine for tests.
//!
//! Deterministically echoes the prompt back, one token per `step()`,
//! capped at `max_tokens`. Useful for testing the runner, API, and CLI
//! without real model weights.
//!
//! A task carrying a JSON Schema grammar is answered instead with a
//! deterministic minimal instance of that schema (see [`minimal_instance`]),
//! streamed in short pieces whose concatenation is the compact JSON.

mod instance;

pub use instance::minimal_instance;

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::stream;

/// Longest piece (in chars) a JSON-schema answer is streamed in.
const PIECE_CHARS: usize = 8;

/// A queued task and how far its output has been emitted.
struct Pending {
    task: GenerationTask,
    emitted: usize,
    /// The precomputed answer for a JSON-schema task, in streaming order.
    /// `None` echoes the prompt's words.
    pieces: Option<Vec<String>>,
}

#[derive(Default)]
pub struct MockEngine {
    pending: Vec<Pending>,
    fail_next_step: Option<String>,
}

/// The minimal instance of the task's JSON Schema, compact, split into
/// pieces of at most `PIECE_CHARS` chars. `None` when the task has no grammar.
fn schema_pieces(task: &GenerationTask) -> EngineResult<Option<Vec<String>>> {
    let Some(grammar) = &task.grammar else {
        return Ok(None);
    };
    match grammar.kind {
        cascadia_types::GrammarKind::JsonSchema => {
            let schema: serde_json::Value = serde_json::from_str(&grammar.body).map_err(|e| {
                EngineError::InvalidConfig(format!("response_format json_schema: {e}"))
            })?;
            let text = minimal_instance(&schema).to_string();
            let chars: Vec<char> = text.chars().collect();
            Ok(Some(
                chars
                    .chunks(PIECE_CHARS)
                    .map(|c| c.iter().collect())
                    .collect(),
            ))
        }
    }
}

impl MockEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Test hook: make the next `step()` return `EngineError::Backend(msg)`.
    pub fn fail_next_step(&mut self, msg: impl Into<String>) {
        self.fail_next_step = Some(msg.into());
    }
}

impl Engine for MockEngine {
    fn warmup(&mut self) {}

    fn cancel(&mut self, task_id: &TaskId) {
        // Step-wise engine: drop the task from the queue so step() stops
        // emitting its words. cancel() and step() never overlap (both
        // &mut self behind the runner mutex), so this is race-free.
        self.pending.retain(|p| &p.task.task_id != task_id);
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if self.pending.iter().any(|p| p.task.task_id == task.task_id) {
            return Ok(());
        }
        // Option B resume: the seed ids are already-emitted, so start the
        // echo cursor past them. They are never re-emitted by step().
        //
        // Id-FIDELITY, not just length: the mock's own emitted id for position
        // i is exactly `i` (see step()), so a resume seed claiming this stream
        // must be [0..K). Length-only checking let a splicer that shuffled or
        // corrupted ids pass every mock-based integration test (review).
        let seed = task.resume_ids().map(<[i32]>::to_vec).unwrap_or_default();
        if let Some((i, &bad)) = seed.iter().enumerate().find(|(i, &id)| id != *i as i32) {
            return Err(EngineError::Backend(format!(
                "mock resume seed id mismatch at {i}: got {bad}, this stream emitted {i}"
            )));
        }
        let pieces = schema_pieces(&task)?;
        self.pending.push(Pending {
            task,
            emitted: seed.len(),
            pieces,
        });
        Ok(())
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if let Some(msg) = self.fail_next_step.take() {
            return Err(EngineError::Backend(msg));
        }
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let Pending {
            task,
            emitted,
            pieces,
        } = self.pending.remove(0);
        // Test sentinel: a prompt containing `__engine_error__` fails the
        // task loud (emits an error chunk) so consumers' failure paths —
        // e.g. the API's 5xx mapping — can be exercised deterministically.
        if task.prompt.contains("__engine_error__") {
            return Ok(vec![(
                task.task_id.clone(),
                Chunk::error(task.task_id, "mock injected engine failure"),
            )]);
        }
        let max = task.max_tokens.max(1) as usize;
        let words: Vec<&str> = match &pieces {
            Some(p) => p.iter().map(String::as_str).collect(),
            None => task.prompt.split_whitespace().collect(),
        };
        if emitted >= max || emitted >= words.len() {
            // Distinguish the stop reason like a real engine: exhausting the
            // prompt's words is a natural `stop`; hitting the token cap first
            // is `length`. Lets the API's finish_reason path be tested without
            // a real model.
            let reason = if emitted >= words.len() {
                cascadia_types::FinishReason::Stop
            } else {
                cascadia_types::FinishReason::Length
            };
            return Ok(vec![(
                task.task_id.clone(),
                Chunk::final_marker(task.task_id, "").with_finish_reason(reason),
            )]);
        }
        // Prompt words are echoed with a trailing space; schema pieces are
        // emitted exactly so their concatenation is the compact JSON.
        let text = match pieces {
            Some(_) => words[emitted].to_string(),
            None => words[emitted].to_string() + " ",
        };
        let chunk =
            Chunk::token(&task.task_id, emitted as i64, text).with_token_ids(vec![emitted as i64]);
        let task_id = task.task_id.clone();
        self.pending.push(Pending {
            task,
            emitted: emitted + 1,
            pieces,
        });
        Ok(vec![(task_id, chunk)])
    }
}

#[derive(Default)]
pub struct MockBuilder {
    connected: bool,
    loaded: bool,
}

impl MockBuilder {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Builder for MockBuilder {
    async fn connect(&mut self, _peers: PeerLayout) -> EngineResult<()> {
        self.connected = true;
        Ok(())
    }

    async fn load(&mut self, _shard: ShardSpec) -> EngineResult<LoadStream> {
        self.loaded = true;
        let progress = vec![
            LoadProgress::message("mock load: starting"),
            LoadProgress::ready(),
        ];
        Ok(Box::pin(stream::iter(progress)))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        if !self.loaded {
            return Err(EngineError::NotLoaded);
        }
        Ok(Box::new(MockEngine::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_types::{PeerLayout, ShardSpec};

    #[tokio::test]
    async fn build_after_connect_and_load() {
        let mut builder = MockBuilder::new();
        builder.connect(PeerLayout::single_stage()).await.unwrap();
        let _stream = builder
            .load(ShardSpec::single_stage("m", "CPU"))
            .await
            .unwrap();
        let engine = Box::new(builder).build().expect("build should succeed");
        let _ = engine; // moved
    }

    #[tokio::test]
    async fn build_before_load_errors() {
        let mut builder = MockBuilder::new();
        builder.connect(PeerLayout::single_stage()).await.unwrap();
        let res = Box::new(builder).build();
        assert!(matches!(res, Err(EngineError::NotLoaded)));
    }

    #[test]
    fn submit_then_step_emits_words() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "the quick brown fox").with_max_tokens(2))
            .unwrap();
        let mut emitted = Vec::new();
        for _ in 0..6 {
            for (_, chunk) in e.step().unwrap() {
                emitted.push(chunk);
            }
        }
        // Two tokens then a final marker.
        assert_eq!(emitted.len(), 3);
        assert!(emitted[0].text.starts_with("the"));
        assert!(emitted[1].text.starts_with("quick"));
        assert!(emitted[2].is_final);
    }

    #[test]
    fn cancel_drops_pending_task_and_stops_emission() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "the quick brown fox jumps").with_max_tokens(64))
            .unwrap();
        // One token out, then cancel mid-generation.
        let first = e.step().unwrap();
        assert_eq!(first.len(), 1);
        assert!(first[0].1.text.starts_with("the"));
        e.cancel(&"t1".to_string());
        // Task is gone — no further compute / emission.
        assert!(e.step().unwrap().is_empty());
        // Cancelling an unknown id is a harmless no-op.
        e.cancel(&"nope".to_string());
    }

    #[test]
    fn duplicate_submit_is_noop() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "hi")).unwrap();
        e.submit(GenerationTask::new("t1", "hi")).unwrap();
        // We should still have exactly one task pending.
        let chunks = e.step().unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn cancel_pending_task_removes_it() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "alpha bravo")).unwrap();
        e.submit(GenerationTask::new("t2", "charlie delta"))
            .unwrap();
        e.cancel(&"t1".to_string());
        // Only t2 ever emits; t1 is gone.
        for _ in 0..8 {
            for (tid, _) in e.step().unwrap() {
                assert_eq!(tid, "t2");
            }
        }
    }

    #[test]
    fn cancel_active_task_lets_next_task_activate_immediately() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "alpha bravo charlie").with_max_tokens(8))
            .unwrap();
        // t1 is mid-generation (one token emitted).
        assert_eq!(e.step().unwrap()[0].0, "t1");
        e.cancel(&"t1".to_string());
        e.submit(GenerationTask::new("t2", "delta echo").with_max_tokens(8))
            .unwrap();
        // The very next step serves t2 — no draining of the abandoned t1.
        assert_eq!(e.step().unwrap()[0].0, "t2");
    }

    #[test]
    fn resume_emits_only_tail_tokens() {
        let mut e = MockEngine::new();
        // Prompt echoes to [a b c d]; resume_token_ids covers [a b] (the
        // mock's own ids for the first two positions), so only c/d are new.
        let mut task = GenerationTask::new("t1", "a b c d").with_max_tokens(64);
        task.resume_token_ids = Some(vec![0, 1]);
        e.submit(task).unwrap();
        let mut emitted = Vec::new();
        for _ in 0..8 {
            for (_, chunk) in e.step().unwrap() {
                emitted.push(chunk);
            }
        }
        // Exactly 2 new tokens (c, d) then a final marker — the prefix a/b
        // is never re-emitted.
        assert_eq!(emitted.len(), 3);
        assert!(emitted[0].text.starts_with('c'));
        assert!(emitted[1].text.starts_with('d'));
        assert!(emitted[2].is_final);
    }

    #[test]
    fn resume_with_wrong_ids_is_rejected() {
        let mut e = MockEngine::new();
        let mut task = GenerationTask::new("t1", "a b c d").with_max_tokens(64);
        // Shuffled ids: right length, wrong content — a corrupted accumulator.
        task.resume_token_ids = Some(vec![1, 0]);
        assert!(
            e.submit(task).is_err(),
            "length-only consumption would have accepted a corrupt prefix"
        );
    }

    #[test]
    fn resume_at_full_budget_emits_zero_new_tokens() {
        let mut e = MockEngine::new();
        // The seed alone exhausts max_tokens, so the task finals immediately
        // with no new tokens. The prompt has MORE words than the budget so the
        // finish reason provably comes from the BUDGET branch (Length), not
        // word exhaustion (Stop) — with a 4-word prompt both conditions were
        // true and the tie-break masked which branch fired.
        let mut task = GenerationTask::new("t1", "a b c d e f").with_max_tokens(4);
        task.resume_token_ids = Some(vec![0, 1, 2, 3]);
        e.submit(task).unwrap();
        let emitted = e.step().unwrap();
        assert_eq!(emitted.len(), 1);
        assert!(emitted[0].1.is_final);
        assert_eq!(
            emitted[0].1.finish_reason,
            Some(cascadia_types::FinishReason::Length),
            "an exhausted resume budget is a Length stop, like every real engine"
        );
    }

    #[test]
    fn injected_step_error_surfaces_as_err() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "hello world")).unwrap();
        e.fail_next_step("boom");
        let res = e.step();
        assert!(matches!(res, Err(EngineError::Backend(ref m)) if m == "boom"));
        // Error is one-shot; the engine resumes on the next step.
        assert!(!e.step().unwrap().is_empty());
    }

    fn json_schema_task(id: &str, body: &str) -> GenerationTask {
        GenerationTask::new(id, "ignored prompt words")
            .with_grammar(cascadia_types::GrammarSpec::json_schema(body))
    }

    fn drain(e: &mut MockEngine) -> Vec<Chunk> {
        let mut out = Vec::new();
        for _ in 0..64 {
            out.extend(e.step().unwrap().into_iter().map(|(_, c)| c));
        }
        out
    }

    const OBJECT_WITH_INT: &str =
        r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#;

    #[test]
    fn json_schema_task_streams_the_minimal_instance() {
        let mut e = MockEngine::new();
        e.submit(json_schema_task("t1", OBJECT_WITH_INT)).unwrap();
        let chunks = drain(&mut e);
        let (last, tokens) = chunks.split_last().unwrap();
        let text: String = tokens.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(text, r#"{"a":0}"#);
        assert!(last.is_final);
        assert_eq!(last.finish_reason, Some(cascadia_types::FinishReason::Stop));
    }

    #[test]
    fn json_schema_pieces_are_short_and_rejoin_exactly() {
        let mut e = MockEngine::new();
        let body = r#"{"type":"object","properties":{"name":{"type":"string","minLength":20}},"required":["name"]}"#;
        e.submit(json_schema_task("t1", body).with_max_tokens(64))
            .unwrap();
        let chunks = drain(&mut e);
        let (last, tokens) = chunks.split_last().unwrap();
        assert!(
            tokens.len() > 1,
            "the instance is split into several pieces"
        );
        for (i, c) in tokens.iter().enumerate() {
            assert!(c.text.chars().count() <= 8, "piece too long: {:?}", c.text);
            assert_eq!(c.token_ids, vec![i as i64]);
        }
        let text: String = tokens.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(text, format!(r#"{{"name":"{}"}}"#, "a".repeat(20)));
        assert_eq!(last.finish_reason, Some(cascadia_types::FinishReason::Stop));
    }

    #[test]
    fn json_schema_task_respects_max_tokens() {
        let mut e = MockEngine::new();
        let body = r#"{"type":"object","properties":{"name":{"type":"string","minLength":20}},"required":["name"]}"#;
        e.submit(json_schema_task("t1", body).with_max_tokens(1))
            .unwrap();
        let chunks = drain(&mut e);
        assert_eq!(chunks.len(), 2, "one piece then the final marker");
        assert!(!chunks[0].is_final);
        assert!(chunks[1].is_final);
        assert_eq!(
            chunks[1].finish_reason,
            Some(cascadia_types::FinishReason::Length)
        );
    }

    #[test]
    fn json_schema_task_with_invalid_body_is_rejected() {
        let mut e = MockEngine::new();
        let res = e.submit(json_schema_task("t1", "{not json"));
        match res {
            Err(EngineError::InvalidConfig(m)) => {
                assert!(m.starts_with("response_format json_schema: "), "{m}")
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
        assert!(e.step().unwrap().is_empty(), "nothing was queued");
    }

    #[test]
    fn task_without_grammar_still_echoes_words_with_trailing_spaces() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "alpha bravo")).unwrap();
        let chunks = drain(&mut e);
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["alpha ", "bravo ", ""]);
        assert_eq!(
            chunks[2].finish_reason,
            Some(cascadia_types::FinishReason::Stop)
        );
    }
}
