//! Engine + Builder wiring for the sparse-MoE runner.
//!
//! Single-stage engine — no upstream/downstream peers. This is the
//! "killer demo" path for Kimi K2.6 today; pipeline parallelism over
//! multiple miners is future work that will need to split the layer
//! list across stages and reuse the transport.

use std::path::PathBuf;
use std::thread::JoinHandle;

use async_trait::async_trait;
use futures::stream;
use tahoma_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use tahoma_ov_genai_shim::PluginConfig;
use tahoma_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use tokenizers::Tokenizer;
use tracing::{info, warn};

use crate::runner::{Runner, RunnerError};

#[derive(Default, Debug, Clone)]
pub struct SparseMoEBuilderConfig {
    pub model_dir: PathBuf,
    pub device: String,
    pub cache_dir: Option<String>,
    pub max_cached_experts: u32,
}

impl SparseMoEBuilderConfig {
    pub fn new(model_dir: impl Into<PathBuf>, device: impl Into<String>) -> Self {
        Self {
            model_dir: model_dir.into(),
            device: device.into(),
            cache_dir: None,
            max_cached_experts: 200,
        }
    }
}

pub struct SparseMoEBuilder {
    pub config: SparseMoEBuilderConfig,
    runner: Option<Runner>,
    tokenizer: Option<Tokenizer>,
}

impl SparseMoEBuilder {
    pub fn new(config: SparseMoEBuilderConfig) -> Self {
        Self {
            config,
            runner: None,
            tokenizer: None,
        }
    }
}

#[async_trait]
impl Builder for SparseMoEBuilder {
    async fn connect(&mut self, peers: PeerLayout) -> EngineResult<()> {
        // Single-stage engine — reject any peers.
        if peers.upstream.is_some() || peers.downstream.is_some() {
            return Err(EngineError::PeerRejected(
                "sparse-moe engine is single-stage; peers must be empty".into(),
            ));
        }
        Ok(())
    }

    async fn load(&mut self, _shard: ShardSpec) -> EngineResult<LoadStream> {
        // Build a plugin config from our cache_dir option.
        let mut plugin = PluginConfig::new();
        if let Some(d) = &self.config.cache_dir {
            plugin = plugin.with("CACHE_DIR", d.clone());
        }

        // Compile everything on a worker thread so the runner stays
        // outside the tokio runtime (OV's TBB pool conflicts with tokio
        // thread parking; we keep the runner Send-only and call into it
        // from a dedicated worker).
        let cfg = self.config.clone();
        let plugin_for_worker = plugin.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let join: JoinHandle<Result<Runner, RunnerError>> = std::thread::spawn(move || {
            tx.send(LoadProgress::message("loading sparse-MoE model"))
                .ok();
            Runner::load(cfg.model_dir.clone(), &cfg.device, plugin_for_worker)
        });

        // Pull events from the worker as it runs, then await the result.
        let runner = match join.join() {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                return Err(EngineError::Backend(format!("runner load: {e}")));
            }
            Err(_) => {
                return Err(EngineError::Backend("runner load worker panicked".into()));
            }
        };

        // Tokenizer — must be HF tokenizer.json next to the manifest. (tiktoken
        // BPEs would need a converter; out of scope here.)
        let tok_path = self.config.model_dir.join("tokenizer.json");
        if tok_path.exists() {
            let t = Tokenizer::from_file(&tok_path)
                .map_err(|e| EngineError::Backend(format!("load tokenizer.json: {e}")))?;
            self.tokenizer = Some(t);
        } else {
            warn!(
                "no tokenizer.json at {} — engine will only accept pre-tokenized inputs",
                tok_path.display()
            );
        }

        self.runner = Some(runner);

        // Stream the (already-emitted) progress events back to the caller.
        let drained: Vec<LoadProgress> = rx.try_iter().collect();
        Ok(Box::pin(stream::iter(drained)))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        let runner = self.runner.ok_or(EngineError::NotLoaded)?;
        let tokenizer = self
            .tokenizer
            .ok_or_else(|| EngineError::Backend("tokenizer.json missing".into()))?;
        Ok(Box::new(SparseMoEEngine {
            runner,
            tokenizer,
            pending: Vec::new(),
        }))
    }
}

pub struct SparseMoEEngine {
    runner: Runner,
    tokenizer: Tokenizer,
    pending: Vec<GenerationTask>,
}

impl Engine for SparseMoEEngine {
    fn warmup(&mut self) {
        // One short greedy generation to warm the JIT caches + populate
        // OV's compile cache on disk.
        info!("warmup: generating 1 token to warm caches");
        let prompt_ids = self
            .tokenizer
            .encode("Hello", false)
            .map(|e| e.get_ids().iter().map(|&u| u as i64).collect::<Vec<_>>())
            .unwrap_or_else(|_| vec![1i64]);
        let _ = self.runner.generate_argmax(&prompt_ids, 1);
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        self.pending.push(task);
        Ok(())
    }

    fn step(&mut self) -> Vec<(TaskId, Chunk)> {
        let task = match self.pending.pop() {
            Some(t) => t,
            None => return Vec::new(),
        };
        let prompt_ids: Vec<i64> = match self.tokenizer.encode(task.prompt.as_str(), true) {
            Ok(enc) => enc.get_ids().iter().map(|&u| u as i64).collect(),
            Err(e) => {
                warn!(task = %task.task_id, "tokenizer encode failed: {e}");
                return Vec::new();
            }
        };
        let max_new = task.max_tokens.max(1) as usize;
        let generated = match self.runner.generate_argmax(&prompt_ids, max_new) {
            Ok(g) => g,
            Err(e) => {
                warn!(task = %task.task_id, "runner failed: {e}");
                return Vec::new();
            }
        };
        let ids_u32: Vec<u32> = generated.iter().map(|&i| i as u32).collect();
        let text = self
            .tokenizer
            .decode(&ids_u32, true)
            .unwrap_or_else(|_| String::new());
        let chunk = Chunk::final_marker(task.task_id.clone(), text);
        vec![(task.task_id.clone(), chunk)]
    }
}
