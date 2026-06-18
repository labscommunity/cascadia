//! Mock engine for tests.
//!
//! Deterministically echoes the prompt back, one token per `step()`,
//! capped at `max_tokens`. Useful for testing the runner, API, and CLI
//! without real model weights.

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::stream;

#[derive(Default)]
pub struct MockEngine {
    pending: Vec<(GenerationTask, usize)>,
    fail_next_step: Option<String>,
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
        self.pending.retain(|(t, _)| &t.task_id != task_id);
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if self.pending.iter().any(|(t, _)| t.task_id == task.task_id) {
            return Ok(());
        }
        self.pending.push((task, 0));
        Ok(())
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if let Some(msg) = self.fail_next_step.take() {
            return Err(EngineError::Backend(msg));
        }
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let (task, emitted) = self.pending.remove(0);
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
        let words: Vec<&str> = task.prompt.split_whitespace().collect();
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
        let token = words[emitted].to_string();
        let chunk = Chunk::token(&task.task_id, emitted as i64, token + " ");
        let task_id = task.task_id.clone();
        self.pending.push((task, emitted + 1));
        Ok(vec![(task_id, chunk)])
    }

    fn cancel(&mut self, task_id: &TaskId) {
        self.pending.retain(|(t, _)| t.task_id != *task_id);
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
        let first = e.step();
        assert_eq!(first.len(), 1);
        assert!(first[0].1.text.starts_with("the"));
        e.cancel(&"t1".to_string());
        // Task is gone — no further compute / emission.
        assert!(e.step().is_empty());
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
    fn injected_step_error_surfaces_as_err() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "hello world")).unwrap();
        e.fail_next_step("boom");
        let res = e.step();
        assert!(matches!(res, Err(EngineError::Backend(ref m)) if m == "boom"));
        // Error is one-shot; the engine resumes on the next step.
        assert!(!e.step().unwrap().is_empty());
    }
}
