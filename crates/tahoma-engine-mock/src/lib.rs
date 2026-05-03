//! Mock engine for tests.
//!
//! Deterministically echoes the prompt back, one token per `step()`,
//! capped at `max_tokens`. Useful for testing the runner, API, and CLI
//! without real model weights.

use async_trait::async_trait;
use futures::stream;
use tahoma_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use tahoma_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};

#[derive(Default)]
pub struct MockEngine {
    pending: Vec<(GenerationTask, usize)>,
}

impl MockEngine {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Engine for MockEngine {
    fn warmup(&mut self) {}

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if self.pending.iter().any(|(t, _)| t.task_id == task.task_id) {
            return Ok(());
        }
        self.pending.push((task, 0));
        Ok(())
    }

    fn step(&mut self) -> Vec<(TaskId, Chunk)> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let (task, emitted) = self.pending.remove(0);
        let max = task.max_tokens.max(1) as usize;
        let words: Vec<&str> = task.prompt.split_whitespace().collect();
        if emitted >= max || emitted >= words.len() {
            return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
        }
        let token = words[emitted].to_string();
        let chunk = Chunk::token(&task.task_id, emitted as i64, token + " ");
        let task_id = task.task_id.clone();
        self.pending.push((task, emitted + 1));
        vec![(task_id, chunk)]
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
    use tahoma_types::{PeerLayout, ShardSpec};

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
            for (_, chunk) in e.step() {
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
    fn duplicate_submit_is_noop() {
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "hi")).unwrap();
        e.submit(GenerationTask::new("t1", "hi")).unwrap();
        // We should still have exactly one task pending.
        let chunks = e.step();
        assert!(!chunks.is_empty());
    }
}
