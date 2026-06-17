//! Mock engine for tests.
//!
//! Deterministically echoes the prompt back, one token per `step()`,
//! capped at `max_tokens`. Useful for testing the runner, API, and CLI
//! without real model weights.

use std::sync::Arc;

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_transport::{ActivationClient, ActivationServer, ByteStream};
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::stream;
use tokio::sync::Mutex;

#[derive(Default)]
pub struct MockEngine {
    pending: Vec<(GenerationTask, usize)>,
    upstream: Option<Arc<Mutex<ActivationServer>>>,
    downstream: Option<Arc<Mutex<ActivationClient>>>,
}

impl MockEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn has_upstream(&self) -> bool {
        self.upstream.is_some()
    }

    pub fn has_downstream(&self) -> bool {
        self.downstream.is_some()
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Block on one upstream recv, holding the transport guard for the whole
    /// call — mirrors the real engines' sync-`step`-over-async-transport shape
    /// so tests can prove peer death surfaces a recv error (releasing the
    /// guard). Must be called from a blocking context (not inside the runtime).
    pub fn recv_once_upstream(&mut self, handle: &tokio::runtime::Handle) -> EngineResult<usize> {
        let up = self.upstream.clone().ok_or(EngineError::NotConnected)?;
        handle.block_on(async move {
            let mut g = up.lock().await;
            let (t, _) = g.recv().await.map_err(|e| EngineError::Backend(e.to_string()))?;
            Ok(t.data.len())
        })
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

    fn reattach_streams(&mut self, up: Option<ByteStream>, down: Option<ByteStream>) -> EngineResult<()> {
        if let Some(s) = up {
            self.upstream = Some(Arc::new(Mutex::new(ActivationServer::from_stream(s, "mock-up"))));
        }
        if let Some(s) = down {
            self.downstream = Some(Arc::new(Mutex::new(ActivationClient::from_stream(s, "mock-down"))));
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct MockBuilder {
    connected: bool,
    loaded: bool,
    upstream: Option<Arc<Mutex<ActivationServer>>>,
    downstream: Option<Arc<Mutex<ActivationClient>>>,
}

impl MockBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    fn into_engine(self) -> MockEngine {
        MockEngine {
            pending: Vec::new(),
            upstream: self.upstream,
            downstream: self.downstream,
        }
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

    async fn connect_streams(&mut self, up: Option<ByteStream>, down: Option<ByteStream>) -> EngineResult<()> {
        if let Some(s) = up {
            self.upstream = Some(Arc::new(Mutex::new(ActivationServer::from_stream(s, "mock-up"))));
        }
        if let Some(s) = down {
            self.downstream = Some(Arc::new(Mutex::new(ActivationClient::from_stream(s, "mock-down"))));
        }
        self.connected = true;
        Ok(())
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        if !self.loaded {
            return Err(EngineError::NotLoaded);
        }
        Ok(Box::new(self.into_engine()))
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reattach_upstream_swaps_to_live_stream_and_state_survives() {
        use cascadia_transport::{ActivationClient, ByteStream, DType, Tensor};
        let mut e = MockEngine::new();
        e.submit(GenerationTask::new("t1", "hello world")).unwrap(); // state that must survive
        let (peer1, es1) = tokio::io::duplex(64 * 1024);
        e.reattach_streams(Some(Box::new(es1) as ByteStream), None).unwrap();
        drop(peer1); // kill stream #1
        let (peer2, es2) = tokio::io::duplex(64 * 1024);
        e.reattach_streams(Some(Box::new(es2) as ByteStream), None).unwrap();
        assert!(e.has_upstream() && !e.has_downstream(), "per-direction: downstream untouched");
        assert_eq!(e.pending_len(), 1, "engine state survives the re-pair");
        let mut peer2_client = ActivationClient::from_stream(Box::new(peer2), "test-peer");
        let send = tokio::spawn(async move {
            peer2_client.send(&Tensor::from_2d(DType::F32, 1, 1, vec![0, 0, 128, 63])).await.unwrap();
        });
        let handle = tokio::runtime::Handle::current();
        let n = tokio::task::spawn_blocking(move || e.recv_once_upstream(&handle))
            .await.unwrap().expect("recv on the re-attached upstream must succeed");
        send.await.unwrap();
        assert_eq!(n, 4, "received the 4-byte f32 payload over the fresh stream");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dead_upstream_recv_errors_releasing_guard() {
        use cascadia_transport::ByteStream;
        let (peer, engine_side) = tokio::io::duplex(64 * 1024);
        let mut e = MockEngine::new();
        e.reattach_streams(Some(Box::new(engine_side) as ByteStream), None).unwrap();
        assert!(e.has_upstream(), "precondition: upstream actually attached");
        drop(peer); // peer dies -> EOF on the engine read half
        let handle = tokio::runtime::Handle::current();
        let res = tokio::task::spawn_blocking(move || e.recv_once_upstream(&handle))
            .await.unwrap();
        assert!(matches!(res, Err(EngineError::Backend(_))),
            "dead upstream must surface a recv error, got {res:?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn connect_streams_carries_live_upstream_into_built_engine() {
        use cascadia_transport::{ActivationClient, ByteStream, DType, Tensor};
        let (peer, engine_side) = tokio::io::duplex(64 * 1024);
        let mut b = MockBuilder::new();
        b.connect_streams(Some(Box::new(engine_side) as ByteStream), None).await.unwrap();
        let mut e = b.into_engine(); // the exact move build() performs
        assert!(e.has_upstream() && !e.has_downstream(), "connect_streams installed only upstream");
        let mut peer_client = ActivationClient::from_stream(Box::new(peer), "test-peer");
        let send = tokio::spawn(async move {
            peer_client.send(&Tensor::from_2d(DType::F32, 1, 1, vec![0, 0, 128, 63])).await.unwrap();
        });
        let handle = tokio::runtime::Handle::current();
        let n = tokio::task::spawn_blocking(move || e.recv_once_upstream(&handle))
            .await.unwrap().expect("engine built via connect_streams must recv over the injected stream");
        send.await.unwrap();
        assert_eq!(n, 4);
    }
}
