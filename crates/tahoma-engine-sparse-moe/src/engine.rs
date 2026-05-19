//! Engine + Builder wiring for the sparse-MoE runner.
//!
//! Two roles:
//! - **Single-stage** (`total == 1`): one Engine holds the full model
//!   and runs `Runner::generate` directly. The path that hit 100% on
//!   the K2.6 quality eval in PR #7.
//! - **Pipeline-parallel** (`total >= 2`): N Engines hold contiguous
//!   layer slices and exchange F32 hidden states over `tahoma-transport`.
//!   Rank 0 owns the API, layer 0, and the prefill+decode driver loop.
//!   Last rank owns the head and sampler. Middle ranks just relay.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream;
use tahoma_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use tahoma_ov_genai_shim::PluginConfig;
use tahoma_transport::{ActivationClient, ActivationServer};
use tahoma_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use tokenizers::Tokenizer;
use tokio::sync::Mutex as TokioMutex;
use tracing::{info, warn};

use crate::dist::{
    forward_reset, recv_forward_body_server, recv_kind_client, recv_kind_server,
    recv_token_body_client, send_forward, send_reset, send_token_upstream, FrameKind,
    StageTransport,
};
use crate::runner::{LayerRange, Runner, RunnerError};

#[derive(Default, Debug, Clone)]
pub struct SparseMoEBuilderConfig {
    pub model_dir: PathBuf,
    pub device: String,
    pub cache_dir: Option<String>,
    pub max_cached_experts: u32,
    /// Pipeline stage index (0-based).
    pub rank: u32,
    /// Number of pipeline stages.
    pub total: u32,
}

impl SparseMoEBuilderConfig {
    pub fn new(model_dir: impl Into<PathBuf>, device: impl Into<String>) -> Self {
        Self {
            model_dir: model_dir.into(),
            device: device.into(),
            cache_dir: None,
            max_cached_experts: 200,
            rank: 0,
            total: 1,
        }
    }

    pub fn with_rank(mut self, rank: u32, total: u32) -> Self {
        self.rank = rank;
        self.total = total;
        self
    }
}

pub struct SparseMoEBuilder {
    pub config: SparseMoEBuilderConfig,
    runner: Option<Runner>,
    tokenizer: Option<Tokenizer>,
    listen_host: String,
    listen_port: Option<u16>,
    transport: StageTransport,
}

impl SparseMoEBuilder {
    pub fn new(config: SparseMoEBuilderConfig) -> Self {
        Self {
            config,
            runner: None,
            tokenizer: None,
            listen_host: "0.0.0.0".into(),
            listen_port: None,
            transport: StageTransport::default(),
        }
    }
}

#[async_trait]
impl Builder for SparseMoEBuilder {
    fn configure_listen(&mut self, host: &str, port: u16) {
        self.listen_host = host.to_string();
        self.listen_port = Some(port);
    }

    async fn connect(&mut self, peers: PeerLayout) -> EngineResult<()> {
        let single = self.config.total <= 1;
        let has_upstream = peers.upstream.is_some();
        let has_downstream = peers.downstream.is_some();
        if single {
            if has_upstream || has_downstream {
                return Err(EngineError::PeerRejected(
                    "sparse-moe with total=1 cannot have peers".into(),
                ));
            }
            return Ok(());
        }

        // Multi-stage: bind upstream listener first so downstream peer
        // can connect to us, then connect outbound to our downstream,
        // then accept the upstream connection. This matches the order
        // ov-runtime / ov-dist-spec use to avoid the bind-vs-connect
        // race at startup.
        if has_upstream {
            let port = self.listen_port.ok_or_else(|| {
                EngineError::InvalidConfig(
                    "non-first rank requires configure_listen() before connect()".into(),
                )
            })?;
            let mut server = ActivationServer::new(self.listen_host.clone(), port);
            server.start().await.map_err(|e| {
                EngineError::Backend(format!("listen {}:{}: {}", self.listen_host, port, e))
            })?;
            self.transport.upstream = Some(Arc::new(TokioMutex::new(server)));
            info!(
                host = %self.listen_host,
                port,
                "sparse-moe upstream bound"
            );
        }

        if let Some(down) = peers.downstream.as_ref() {
            let mut client = ActivationClient::new(down.host.clone(), down.port);
            client
                .connect_with_timeout(std::time::Duration::from_secs(60))
                .await
                .map_err(|e| {
                    EngineError::Backend(format!("connect to {}:{}: {}", down.host, down.port, e))
                })?;
            self.transport.downstream = Some(Arc::new(TokioMutex::new(client)));
            info!(host = %down.host, port = down.port, "sparse-moe downstream connected");
        }

        if let Some(srv) = self.transport.upstream.as_ref() {
            let mut guard = srv.lock().await;
            guard
                .accept()
                .await
                .map_err(|e| EngineError::Backend(format!("accept upstream: {e}")))?;
            info!("sparse-moe upstream peer accepted");
        }

        Ok(())
    }

    async fn load(&mut self, shard: ShardSpec) -> EngineResult<LoadStream> {
        let mut plugin = PluginConfig::new();
        if let Some(d) = &self.config.cache_dir {
            plugin = plugin.with("CACHE_DIR", d.clone());
        }

        // Build a LayerRange from the ShardSpec + config. For total==1
        // we keep the historical behavior (load everything). For
        // multi-stage we honor layer_start/layer_end. If they're zero
        // (e.g. the CLI hasn't computed them), derive an even split
        // from the manifest's num_layers and our rank/total.
        let total = self.config.total.max(1);
        let rank = self.config.rank.min(total - 1);
        let is_first = shard.is_first_stage || rank == 0;
        let is_last = shard.is_last_stage || rank == total - 1;
        let mut layer_start = shard.layer_start;
        let mut layer_end = shard.layer_end;
        if total > 1 && layer_start == 0 && layer_end == 0 {
            let total_moe = read_manifest_moe_count(&self.config.model_dir)?;
            // Reject splits that would leave a rank with zero MoE layers
            // — that rank is dead weight (still on the wire, still pays
            // every round-trip's latency) and almost certainly a config
            // mistake. K2.6 has 60 MoE layers, so total > 60 is silly.
            if total > total_moe {
                return Err(EngineError::InvalidConfig(format!(
                    "total={total} > num_moe_layers={total_moe}; some rank would hold zero layers"
                )));
            }
            let (s, e) = even_moe_split(total_moe, rank, total);
            layer_start = s;
            layer_end = e;
            info!(
                rank,
                total,
                total_moe_layers = total_moe,
                layer_start,
                layer_end,
                "computed even MoE-layer split"
            );
        } else if total == 1 {
            // Single-stage: load every MoE layer, regardless of what
            // ShardSpec says.
            layer_start = 0;
            layer_end = u32::MAX;
        }
        let range = LayerRange {
            layer_start,
            layer_end,
            is_first,
            is_last,
        };

        let cfg = self.config.clone();
        let plugin_for_worker = plugin.clone();
        let range_for_worker = range.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let join: JoinHandle<Result<Runner, RunnerError>> = std::thread::spawn(move || {
            tx.send(LoadProgress::message("loading sparse-MoE model"))
                .ok();
            Runner::load(
                cfg.model_dir.clone(),
                &cfg.device,
                plugin_for_worker,
                range_for_worker,
            )
        });

        let runner = match join.join() {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                return Err(EngineError::Backend(format!("runner load: {e}")));
            }
            Err(_) => {
                return Err(EngineError::Backend("runner load worker panicked".into()));
            }
        };

        // Tokenizer is only needed on rank 0 (the API rank).
        if rank == 0 {
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
        }

        self.runner = Some(runner);

        let drained: Vec<LoadProgress> = rx.try_iter().collect();
        Ok(Box::pin(stream::iter(drained)))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        let runner = self.runner.ok_or(EngineError::NotLoaded)?;
        let total = self.config.total.max(1);
        let rank = self.config.rank.min(total - 1);
        let runtime_handle = tokio::runtime::Handle::try_current()
            .map_err(|_| EngineError::Backend("Builder::build outside tokio context".into()))?;
        if rank == 0 && self.tokenizer.is_none() {
            return Err(EngineError::Backend("tokenizer.json missing".into()));
        }
        Ok(Box::new(SparseMoEEngine {
            runner,
            tokenizer: self.tokenizer,
            pending: VecDeque::new(),
            peer_disconnected: false,
            transport: self.transport,
            runtime_handle,
            rank,
            total,
            last_rank_history: Vec::new(),
            last_rank_rng: 0,
            last_rank_rng_seeded: false,
        }))
    }
}

/// Build a SamplingConfig for one task. Currently the only knob the
/// public `GenerationTask` surface exposes is `temperature`; the rest
/// are hard-coded to defaults that have worked well in K2.6 evals.
/// Lifted out so the single-stage and multi-stage entry paths can't
/// drift.
fn sampling_from_task(task: &GenerationTask) -> crate::sampling::SamplingConfig {
    crate::sampling::SamplingConfig {
        temperature: task.temperature.max(0.0),
        top_p: 1.0,
        repetition_penalty: 1.05,
        repetition_window: 64,
        seed: None,
    }
}

/// Translate the public adaptive-stop fields on `GenerationTask` into
/// the engine-internal [`StopConditions`]. Lifted out so both the
/// single-stage path and the rank-0 driver compute the same conditions
/// from the same task.
fn stop_conditions_from_task(task: &GenerationTask) -> crate::sampling::StopConditions {
    crate::sampling::StopConditions {
        stop: task.stop.clone().unwrap_or_default(),
        stop_on_repetition: task.stop_on_repetition,
    }
}

/// Distributed-mode counterpart to the closure-flavored
/// `check_adaptive_stop` inside the runner. Borrows the tokenizer
/// directly (rank 0 owns it) instead of via a closure to keep the
/// driver loop's borrow checker happy.
///
/// Returns true if any configured condition matched; the caller breaks
/// the decode loop. When `tokenizer` is None, stop-sequence checks are
/// skipped (the driver still does the repetition check on raw tokens).
fn check_adaptive_stop_dist(
    generated: &[i64],
    stop: &crate::sampling::StopConditions,
    tokenizer: Option<&Tokenizer>,
    _task: &GenerationTask,
) -> bool {
    if !stop.stop.is_empty() {
        if let Some(tok) = tokenizer {
            // Same 64-token tail budget the single-stage path uses;
            // see comment there for the rationale.
            const DECODE_TAIL_BUDGET: usize = 64;
            let n = generated.len().min(DECODE_TAIL_BUDGET);
            let start = generated.len() - n;
            let ids_u32: Vec<u32> = generated[start..].iter().map(|&i| i as u32).collect();
            let tail = tok.decode(&ids_u32, true).unwrap_or_default();
            if crate::sampling::text_ends_with_any(&tail, &stop.stop) {
                return true;
            }
        }
    }
    if stop.stop_on_repetition && crate::sampling::is_repetition_loop(generated) {
        return true;
    }
    false
}

fn read_manifest_moe_count(model_dir: &std::path::Path) -> EngineResult<u32> {
    let m = crate::manifest::Manifest::load(model_dir)
        .map_err(|e| EngineError::Backend(format!("read manifest: {e}")))?;
    Ok(m.moe_layer_ids().len() as u32)
}

/// Even split of the MoE layer indices across `total` ranks.
/// Returns `(layer_start_inclusive, layer_end_exclusive)` in *manifest*
/// coordinates — i.e. MoE layer ids are 1..=num_moe (dense layer 0
/// excluded). The first rank's range starts at 1; the last rank's
/// range ends at `num_moe + 1` so its `layer_end` exclusive covers the
/// final MoE layer.
fn even_moe_split(total_moe: u32, rank: u32, total: u32) -> (u32, u32) {
    if total <= 1 {
        return (0, u32::MAX);
    }
    let per = total_moe / total;
    let rem = total_moe % total;
    // The first `rem` ranks get one extra layer.
    let extras_before = rank.min(rem);
    let base_count = per;
    let my_extra = if rank < rem { 1 } else { 0 };
    let my_start_idx = rank * base_count + extras_before;
    let my_count = base_count + my_extra;
    // MoE layer ids are 1-based per the manifest (layer 0 is dense).
    let start = my_start_idx + 1;
    let end = start + my_count;
    (start, end)
}

/// Hard cap on the pending-task queue. step() processes one task end-to-end
/// per call, so the OS-level backpressure of returning QueueFull is
/// preferable to silently accreting tasks the engine will not reach for
/// minutes. 8 is high enough that a small burst (e.g. a benchmark looping
/// over a few prompts) is fine without immediately rejecting.
const MAX_PENDING_TASKS: usize = 8;

/// Cool-off the worker loop hits on the run_relay_loop tight cycle when
/// either (a) the upstream peer closed and we're waiting for the runner to
/// notice we're done, or (b) we just rejected a frame and need to avoid
/// hot-spinning while a misbehaving peer keeps trying. Matches the cadence
/// dist_spec uses for the same situation.
const WORKER_BACKOFF: Duration = Duration::from_millis(200);

pub struct SparseMoEEngine {
    runner: Runner,
    tokenizer: Option<Tokenizer>,
    pending: VecDeque<GenerationTask>,
    transport: StageTransport,
    runtime_handle: tokio::runtime::Handle,
    rank: u32,
    total: u32,
    /// Set on a worker rank when the upstream socket closes cleanly. Keeps
    /// step_worker from hot-spinning on `recv_kind_server` returning
    /// `Ok(None)` over and over.
    peer_disconnected: bool,
    /// Last-rank only: tokens this rank has sampled since the last
    /// `Reset`. Used as the `history` argument to `sampling::sample` so
    /// the repetition penalty has the recent local emit-stream to
    /// reference. Prompt tokens are NOT mirrored here — the prompt
    /// flows only as hidden states through the pipeline — so the
    /// rep-penalty window covers only generated tokens. Acceptable
    /// limitation for v1; documented in dist.rs.
    last_rank_history: Vec<i64>,
    /// xorshift64* state for the last rank's sampler. Seeded lazily
    /// from the first Forward frame's `SamplingConfig.seed` after a
    /// `Reset` so deterministic-seed mode reproduces across runs.
    last_rank_rng: u64,
    last_rank_rng_seeded: bool,
}

impl SparseMoEEngine {
    /// Bridge sync `Engine::step` code to an async transport future.
    /// Delegates to `tahoma_runner::run_async`, which consults the
    /// thread-local `BlockingContextGuard` flag — set by
    /// `Runner::run_relay_loop` for worker ranks — to pick the
    /// cheapest safe `block_on` variant. On a worker thread that's
    /// ~250x cheaper than wrapping in `block_in_place`.
    fn block_on<F: std::future::Future>(&self, fut: F) -> F::Output {
        tahoma_runner::run_async(&self.runtime_handle, fut)
    }

    fn is_last(&self) -> bool {
        self.transport.is_last()
    }
}

impl Engine for SparseMoEEngine {
    fn warmup(&mut self) {
        // Only rank 0 has a tokenizer and the driver loop; the workers
        // are warmed by the first real generation's prefill. Doing a
        // dedicated warmup on workers would require us to drive the
        // whole pipeline from rank 0 with a dummy task; not worth the
        // complexity for one fewer cold step per rank.
        if self.total > 1 && self.rank != 0 {
            info!("warmup: skipping on rank {}/{}", self.rank, self.total);
            return;
        }
        if let Some(tok) = self.tokenizer.as_ref() {
            info!("warmup: generating 1 token to warm caches");
            let prompt_ids = tok
                .encode("Hello", false)
                .map(|e| e.get_ids().iter().map(|&u| u as i64).collect::<Vec<_>>())
                .unwrap_or_else(|_| vec![1i64]);
            if self.total == 1 {
                let _ = self.runner.generate_argmax(&prompt_ids, 1);
            }
            // For multi-stage we skip warmup too — same reasoning as
            // workers above. A real prompt will trigger the JIT on
            // every rank in lockstep.
        }
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if self.rank != 0 {
            return Err(EngineError::InvalidConfig(
                "only rank 0 accepts tasks; worker ranks drive themselves from upstream frames"
                    .into(),
            ));
        }
        if self.pending.len() >= MAX_PENDING_TASKS {
            return Err(EngineError::QueueFull {
                queued: self.pending.len(),
                cap: MAX_PENDING_TASKS,
            });
        }
        self.pending.push_back(task);
        Ok(())
    }

    fn step(&mut self) -> Vec<(TaskId, Chunk)> {
        if self.total == 1 {
            return self.step_single_stage();
        }
        if self.rank == 0 {
            self.step_first()
        } else {
            self.step_worker()
        }
    }

    fn close(&mut self) {
        // Best-effort transport teardown. We block_on each socket's close()
        // sequentially because the engine is being torn down; lock
        // contention isn't a worry. Errors are logged but swallowed —
        // close is idempotent and we want every peer to get a clean FIN.
        if let Some(srv) = self.transport.upstream.take() {
            let _ = self.block_on(async move {
                srv.lock().await.close().await;
            });
        }
        if let Some(cli) = self.transport.downstream.take() {
            let _ = self.block_on(async move {
                cli.lock().await.close().await;
            });
        }
        self.peer_disconnected = true;
    }
}

impl SparseMoEEngine {
    fn step_single_stage(&mut self) -> Vec<(TaskId, Chunk)> {
        let task = match self.pending.pop_front() {
            Some(t) => t,
            None => return Vec::new(),
        };
        let tokenizer = match self.tokenizer.as_ref() {
            Some(t) => t,
            None => {
                warn!(task = %task.task_id, "single-stage engine has no tokenizer");
                let final_chunk = Chunk::final_marker(task.task_id.clone(), "");
                return vec![(task.task_id, final_chunk)];
            }
        };
        let started = std::time::Instant::now();
        let prompt_ids: Vec<i64> = match tokenizer.encode(task.prompt.as_str(), true) {
            Ok(enc) => enc.get_ids().iter().map(|&u| u as i64).collect(),
            Err(e) => {
                warn!(task = %task.task_id, "tokenizer encode failed: {e}");
                let final_chunk = Chunk::final_marker(task.task_id.clone(), "");
                return vec![(task.task_id, final_chunk)];
            }
        };
        let max_new = task.max_tokens.max(1) as usize;
        let sampling_cfg = sampling_from_task(&task);
        let stop_cond = stop_conditions_from_task(&task);
        // Cap the per-step decode cost: detokenize only the tail of the
        // running stream, not every emitted token. The longest stop
        // sequence sets the lower bound; 64 tokens is a comfortable
        // upper bound for any plausible stop string (the K2.6 BPE rarely
        // splits a short marker like "Human:" or "\n\n" into more than
        // a couple of pieces).
        const DECODE_TAIL_BUDGET: usize = 64;
        let generated = if stop_cond.any() {
            let tokenizer_for_tail = tokenizer.clone();
            let decode_tail = move |gen: &[i64]| -> String {
                let n = gen.len().min(DECODE_TAIL_BUDGET);
                let start = gen.len() - n;
                let ids_u32: Vec<u32> = gen[start..].iter().map(|&i| i as u32).collect();
                tokenizer_for_tail
                    .decode(&ids_u32, true)
                    .unwrap_or_default()
            };
            match self.runner.generate_with_stop(
                &prompt_ids,
                max_new,
                &sampling_cfg,
                &stop_cond,
                Some(decode_tail),
            ) {
                Ok((g, reason)) => {
                    info!(task = %task.task_id, ?reason, "adaptive stop");
                    g
                }
                Err(e) => {
                    warn!(task = %task.task_id, "runner failed: {e}");
                    let final_chunk = Chunk::final_marker(task.task_id.clone(), "");
                    return vec![(task.task_id, final_chunk)];
                }
            }
        } else {
            match self.runner.generate(&prompt_ids, max_new, &sampling_cfg) {
                Ok(g) => g,
                Err(e) => {
                    warn!(task = %task.task_id, "runner failed: {e}");
                    let final_chunk = Chunk::final_marker(task.task_id.clone(), "");
                    return vec![(task.task_id, final_chunk)];
                }
            }
        };
        let n_tokens = generated.len() as u32;
        let ids_u32: Vec<u32> = generated.iter().map(|&i| i as u32).collect();
        let mut text = tokenizer
            .decode(&ids_u32, true)
            .unwrap_or_else(|_| String::new());
        if let Some(stripped) = text.strip_prefix(&task.prompt) {
            text = stripped.trim_start().to_string();
        }
        // If the user supplied a stop sequence and the tail of `text`
        // ends with it, strip the marker so the visible completion
        // matches OpenAI-style behavior (the stop string is consumed,
        // not echoed). This is the same convention vLLM uses.
        if let Some(stops) = task.stop.as_ref() {
            for s in stops {
                if !s.is_empty() && text.ends_with(s.as_str()) {
                    text.truncate(text.len() - s.len());
                    break;
                }
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        info!(
            task = %task.task_id,
            tokens = n_tokens,
            elapsed_s = elapsed,
            tok_s = if elapsed > 0.0 { n_tokens as f64 / elapsed } else { 0.0 },
            "task done (single-stage)"
        );
        let mut chunk = Chunk::final_marker(task.task_id.clone(), text);
        chunk.n_tokens = Some(n_tokens);
        vec![(task.task_id.clone(), chunk)]
    }

    /// Rank 0 driver: tokenize, drive prefill + decode through the
    /// transport pipeline, return one final chunk for the task.
    fn step_first(&mut self) -> Vec<(TaskId, Chunk)> {
        let task = match self.pending.pop_front() {
            Some(t) => t,
            None => return Vec::new(),
        };
        let started = std::time::Instant::now();
        let prompt_ids: Vec<i64> = {
            let Some(tok) = self.tokenizer.as_ref() else {
                warn!(task = %task.task_id, "rank-0 engine has no tokenizer");
                return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
            };
            match tok.encode(task.prompt.as_str(), true) {
                Ok(enc) => enc.get_ids().iter().map(|&u| u as i64).collect(),
                Err(e) => {
                    warn!(task = %task.task_id, "tokenizer encode failed: {e}");
                    return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
                }
            }
        };

        let max_new = task.max_tokens.max(1) as usize;
        let sampling_cfg = sampling_from_task(&task);
        let stop_cond = stop_conditions_from_task(&task);
        let downstream = match self.transport.downstream.clone() {
            Some(d) => d,
            None => {
                warn!("rank 0 has no downstream peer in multi-stage mode");
                return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
            }
        };

        // RESET downstream so workers clear their KV caches.
        self.runner.reset_kv();
        if let Err(e) = self.block_on(send_reset(&downstream)) {
            warn!(task = %task.task_id, "send_reset: {e}");
            return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
        }

        // Drive the full generation. result_tokens collects new tokens
        // generated AFTER the prompt (we discard the prefill responses).
        let result_tokens = match self.drive_generation_first(
            &prompt_ids,
            max_new,
            &sampling_cfg,
            &stop_cond,
            &task,
            &downstream,
        ) {
            Ok(g) => g,
            Err(e) => {
                warn!(task = %task.task_id, "rank-0 driver failed: {e}");
                return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
            }
        };

        let n_tokens = result_tokens.len() as u32;
        let ids_u32: Vec<u32> = result_tokens.iter().map(|&i| i as u32).collect();
        let Some(tokenizer) = self.tokenizer.as_ref() else {
            warn!("tokenizer disappeared mid-task");
            return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
        };
        let mut text = tokenizer
            .decode(&ids_u32, true)
            .unwrap_or_else(|_| String::new());
        if let Some(stripped) = text.strip_prefix(&task.prompt) {
            text = stripped.trim_start().to_string();
        }
        // Same stop-sequence trim as the single-stage path so multi-
        // stage output matches the OpenAI convention (stop string is
        // consumed, not echoed).
        if let Some(stops) = task.stop.as_ref() {
            for s in stops {
                if !s.is_empty() && text.ends_with(s.as_str()) {
                    text.truncate(text.len() - s.len());
                    break;
                }
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        info!(
            task = %task.task_id,
            tokens = n_tokens,
            elapsed_s = elapsed,
            tok_s = if elapsed > 0.0 { n_tokens as f64 / elapsed } else { 0.0 },
            rank = self.rank,
            total = self.total,
            "task done (rank-0 driver)"
        );
        let mut chunk = Chunk::final_marker(task.task_id.clone(), text);
        chunk.n_tokens = Some(n_tokens);
        vec![(task.task_id.clone(), chunk)]
    }

    /// Rank 0 generation loop. For each prompt token + each decode
    /// step: embed → forward through my shells → send hidden
    /// downstream → recv sampled token back. Discards prefill samples
    /// except the last (which becomes the first generated token).
    ///
    /// `stop` and `task` carry the adaptive-stop signals (user stop
    /// strings, repetition flag). When neither is active the loop is
    /// byte-for-byte identical to the pre-PR path: EOS or `max_new`.
    fn drive_generation_first(
        &mut self,
        prompt_ids: &[i64],
        max_new: usize,
        cfg: &crate::sampling::SamplingConfig,
        stop: &crate::sampling::StopConditions,
        task: &GenerationTask,
        downstream: &Arc<TokioMutex<ActivationClient>>,
    ) -> Result<Vec<i64>, String> {
        let hidden = self.runner.manifest.hidden_size as usize;
        let eos: Vec<i64> = self
            .runner
            .manifest
            .eos_token_ids
            .iter()
            .map(|&x| x as i64)
            .collect();
        let mut history: Vec<i64> = Vec::with_capacity(prompt_ids.len() + max_new);
        let mut generated: Vec<i64> = Vec::with_capacity(max_new);
        let stop_active = stop.any();
        // Tokenizer is only present on rank 0; we keep a clone for the
        // detokenizer-tail closure so the borrow checker is happy while
        // we still mutate `self` for the forward call. The clone is
        // cheap (tokenizers's `Clone` is Arc-backed for the model
        // table).
        let tokenizer_for_tail = if stop_active && !stop.stop.is_empty() {
            self.tokenizer.clone()
        } else {
            None
        };

        // Prefill: feed each prompt token; the very last response from
        // last-rank becomes the first generated token.
        info!(
            prompt_len = prompt_ids.len(),
            "prefill (token-by-token, distributed)"
        );
        for (i, &t) in prompt_ids.iter().enumerate() {
            history.push(t);
            let token_back = self
                .forward_one_token_first(&history, cfg, downstream)
                .map_err(|e| format!("prefill step {i}: {e}"))?;
            if i + 1 == prompt_ids.len() {
                // Last prefill step: this is the first generated token.
                if eos.contains(&token_back) {
                    return Ok(generated);
                }
                generated.push(token_back);
                history.push(token_back);
                if stop_active
                    && check_adaptive_stop_dist(&generated, stop, tokenizer_for_tail.as_ref(), task)
                {
                    return Ok(generated);
                }
            }
            // Otherwise discard (intermediate prefill samples are stale).
            let _ = token_back;
        }

        // Decode loop.
        for step_i in 1..max_new {
            let token_back = self
                .forward_one_token_first(&history, cfg, downstream)
                .map_err(|e| format!("decode step {step_i}: {e}"))?;
            if eos.contains(&token_back) {
                break;
            }
            generated.push(token_back);
            history.push(token_back);
            if stop_active
                && check_adaptive_stop_dist(&generated, stop, tokenizer_for_tail.as_ref(), task)
            {
                break;
            }
        }
        let _ = hidden;
        Ok(generated)
    }

    /// Run one (prefill or decode) step on rank 0: embed via layer 0,
    /// run my shells, send hidden state downstream, receive the
    /// sampled token back along the same socket.
    fn forward_one_token_first(
        &mut self,
        history: &[i64],
        cfg: &crate::sampling::SamplingConfig,
        downstream: &Arc<TokioMutex<ActivationClient>>,
    ) -> Result<i64, String> {
        let hidden = self.runner.manifest.hidden_size as usize;
        let past_seq_len: u32 = history
            .len()
            .checked_sub(1)
            .ok_or_else(|| "forward_one_token_first: empty history".to_string())?
            as u32;

        // Layer 0 (stateful) on the most recently appended token.
        // The history grows by one each prefill/decode step, so each
        // step advances the layer-0 KV cache by exactly one slot.
        let last_id = *history
            .last()
            .ok_or_else(|| "forward_one_token_first: empty history".to_string())?;
        let h_tail = self
            .runner
            .forward_layer0_step(last_id)
            .map_err(|e| format!("layer0_step: {e}"))?;
        let h_after_shells = self
            .runner
            .forward_shells(&h_tail, &[1, 1, hidden], past_seq_len as usize)
            .map_err(|e| format!("forward_shells: {e}"))?;

        // Send hidden downstream and wait for token to come back.
        self.block_on(async {
            send_forward(
                downstream,
                past_seq_len,
                cfg,
                &h_after_shells,
                [1, 1, hidden as u32],
            )
            .await
            .map_err(|e| format!("send_forward: {e}"))?;
            match recv_kind_client(downstream).await {
                Ok(Some(FrameKind::Token)) => {
                    let token = recv_token_body_client(downstream)
                        .await
                        .map_err(|e| format!("recv_token: {e}"))?;
                    Ok(token)
                }
                Ok(Some(other)) => Err(format!("unexpected frame after forward: {other:?}")),
                Ok(None) => Err("downstream closed during recv_kind".into()),
                Err(e) => Err(format!("recv_kind: {e}")),
            }
        })
    }

    /// Worker step: process exactly one frame from upstream and emit
    /// its response (either FORWARD downstream + TOKEN back upstream
    /// for middle ranks, or TOKEN back upstream for the last rank).
    ///
    /// Returning an empty Vec is the worker's normal idle/done signal —
    /// the runner's `run_relay_loop` calls `step()` in a tight loop, so
    /// we sleep briefly on disconnect / error to avoid pegging a core
    /// while the runner is being torn down by the operator.
    fn step_worker(&mut self) -> Vec<(TaskId, Chunk)> {
        if self.peer_disconnected {
            std::thread::sleep(WORKER_BACKOFF);
            return Vec::new();
        }
        let upstream = match self.transport.upstream.clone() {
            Some(u) => u,
            None => {
                warn!("worker rank has no upstream peer");
                self.peer_disconnected = true;
                std::thread::sleep(WORKER_BACKOFF);
                return Vec::new();
            }
        };
        match self.handle_one_frame(&upstream) {
            Ok(_) => Vec::new(),
            Err(e) => {
                warn!(rank = self.rank, "worker frame failed: {e}");
                // Don't hot-spin on a misbehaving peer. dist_spec uses
                // the same 200 ms cool-off; same logic applies here.
                std::thread::sleep(WORKER_BACKOFF);
                Vec::new()
            }
        }
    }

    fn handle_one_frame(
        &mut self,
        upstream: &Arc<TokioMutex<ActivationServer>>,
    ) -> Result<(), String> {
        let downstream = self.transport.downstream.clone();
        let kind = self
            .block_on(recv_kind_server(upstream))
            .map_err(|e| format!("recv_kind: {e}"))?;
        let Some(kind) = kind else {
            // Clean upstream close — the driver finished its session.
            // Latch the flag so subsequent step()s back off without
            // re-entering the (now-doomed) recv loop.
            info!(rank = self.rank, "upstream closed cleanly; worker idling");
            self.peer_disconnected = true;
            return Ok(());
        };
        match kind {
            FrameKind::Reset => {
                self.runner.reset_kv();
                // Last-rank sampling state belongs to this session;
                // also wipe it so the next prompt starts with empty
                // rep-penalty history + an unseeded RNG.
                self.last_rank_history.clear();
                self.last_rank_rng_seeded = false;
                if let Some(down) = downstream.as_ref() {
                    self.block_on(forward_reset(down))
                        .map_err(|e| format!("forward_reset: {e}"))?;
                }
                Ok(())
            }
            FrameKind::Forward => {
                let (past_seq_len, sampling_cfg, hidden_f32, in_shape) = self
                    .block_on(recv_forward_body_server(upstream))
                    .map_err(|e| format!("recv_forward: {e}"))?;
                let hidden = self.runner.manifest.hidden_size as usize;
                if in_shape[0] != 1 || in_shape[1] != 1 || in_shape[2] as usize != hidden {
                    return Err(format!(
                        "forward shape unexpected {:?} vs hidden {}",
                        in_shape, hidden
                    ));
                }
                let h_after = self
                    .runner
                    .forward_shells(&hidden_f32, &[1, 1, hidden], past_seq_len as usize)
                    .map_err(|e| format!("forward_shells: {e}"))?;

                if self.is_last() {
                    // Run head, sample with the caller's config, send
                    // TOKEN upstream. The first Forward of a session
                    // seeds our RNG so deterministic seeds reproduce
                    // across runs.
                    let logits = self
                        .runner
                        .forward_head_last(&h_after, 1)
                        .map_err(|e| format!("forward_head: {e}"))?;
                    if !self.last_rank_rng_seeded {
                        self.last_rank_rng = crate::sampling::init_rng(sampling_cfg.seed);
                        self.last_rank_rng_seeded = true;
                    }
                    let token = crate::sampling::sample(
                        &logits,
                        &self.last_rank_history,
                        &sampling_cfg,
                        &mut self.last_rank_rng,
                    );
                    self.last_rank_history.push(token);
                    self.block_on(send_token_upstream(upstream, token))
                        .map_err(|e| format!("send_token: {e}"))?;
                    Ok(())
                } else {
                    let down =
                        downstream.ok_or_else(|| "mid rank missing downstream".to_string())?;
                    self.block_on(async {
                        send_forward(
                            &down,
                            past_seq_len,
                            &sampling_cfg,
                            &h_after,
                            [1, 1, hidden as u32],
                        )
                        .await
                        .map_err(|e| format!("send_forward: {e}"))?;
                        let token = match recv_kind_client(&down).await {
                            Ok(Some(FrameKind::Token)) => recv_token_body_client(&down)
                                .await
                                .map_err(|e| format!("recv_token: {e}"))?,
                            Ok(Some(other)) => {
                                return Err(format!("unexpected mid-rank frame: {other:?}"));
                            }
                            Ok(None) => return Err("downstream closed mid-frame".into()),
                            Err(e) => return Err(format!("recv_kind: {e}")),
                        };
                        send_token_upstream(upstream, token)
                            .await
                            .map_err(|e| format!("send_token: {e}"))
                    })?;
                    Ok(())
                }
            }
            FrameKind::Token => Err(format!(
                "rank {} received unexpected TOKEN from upstream",
                self.rank
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_moe_split_uniform() {
        assert_eq!(even_moe_split(60, 0, 2), (1, 31));
        assert_eq!(even_moe_split(60, 1, 2), (31, 61));
    }

    #[test]
    fn even_moe_split_with_remainder() {
        // 60 across 7 ranks: 60/7 = 8 remainder 4. Ranks 0..3 get 9
        // each; ranks 4..6 get 8 each.
        let mut prev_end = 1u32;
        let mut total = 0u32;
        for r in 0..7 {
            let (s, e) = even_moe_split(60, r, 7);
            assert_eq!(s, prev_end, "rank {r}: start={s} prev_end={prev_end}");
            assert!(e > s);
            total += e - s;
            prev_end = e;
        }
        assert_eq!(total, 60);
        assert_eq!(prev_end, 61);
    }

    #[test]
    fn single_stage_yields_full_range() {
        let (s, e) = even_moe_split(60, 0, 1);
        assert_eq!((s, e), (0, u32::MAX));
    }

    #[test]
    fn stop_conditions_from_task_propagates_fields() {
        // A blank task → no conditions.
        let t = GenerationTask::new("t", "hi");
        let sc = stop_conditions_from_task(&t);
        assert!(sc.stop.is_empty());
        assert!(!sc.stop_on_repetition);
        assert!(!sc.any());

        // Populated: both stop list and repetition flag carry over.
        let t = GenerationTask::new("t", "hi")
            .with_stop(vec!["\n\n".into(), "Human:".into()])
            .with_stop_on_repetition(true);
        let sc = stop_conditions_from_task(&t);
        assert_eq!(sc.stop, vec!["\n\n".to_string(), "Human:".to_string()]);
        assert!(sc.stop_on_repetition);
        assert!(sc.any());
    }

    #[test]
    fn check_adaptive_stop_dist_no_tokenizer_skips_stop_seq_but_keeps_repetition() {
        // Stop sequence is set, but no tokenizer available → that branch
        // does nothing. Repetition flag still fires on a token-level loop.
        let task = GenerationTask::new("t", "");
        let stop = crate::sampling::StopConditions {
            stop: vec!["\n\n".into()],
            stop_on_repetition: true,
        };
        // 1 2 (very × 12) — well above the repetition floor.
        let very = 7i64;
        let mut gen: Vec<i64> = vec![1, 2];
        for _ in 0..12 {
            gen.push(very);
        }
        // No tokenizer → stop-seq skipped, but repetition trips.
        assert!(check_adaptive_stop_dist(&gen, &stop, None, &task));

        // Same generation, repetition disabled, no tokenizer → no stop.
        let stop2 = crate::sampling::StopConditions {
            stop: vec!["\n\n".into()],
            stop_on_repetition: false,
        };
        assert!(!check_adaptive_stop_dist(&gen, &stop2, None, &task));
    }
}
