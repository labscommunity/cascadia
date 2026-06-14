//! Engine + Builder wiring for the sparse-MoE runner.
//!
//! Two roles:
//! - **Single-stage** (`total == 1`): one Engine holds the full model
//!   and runs `Runner::generate` directly. The path that hit 100% on
//!   the K2.6 quality eval in PR #7.
//! - **Pipeline-parallel** (`total >= 2`): N Engines hold contiguous
//!   layer slices and exchange F32 hidden states over `cascadia-transport`.
//!   Rank 0 owns the API, layer 0, and the prefill+decode driver loop.
//!   Last rank owns the head and sampler. Middle ranks just relay.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_ov_genai_shim::PluginConfig;
use cascadia_transport::{ActivationClient, ActivationServer};
use cascadia_types::{
    Chunk, FinishReason, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId,
};
use futures::stream;
use tokenizers::Tokenizer;
use tokio::sync::Mutex as TokioMutex;
use tracing::{info, warn};

use crate::dist::{
    forward_reset, recv_forward_batch_body_server, recv_forward_body_server, recv_kind_client,
    recv_kind_server, recv_token_batch_body_client, recv_token_body_client, send_forward,
    send_forward_batch, send_reset, send_token_batch_upstream, send_token_upstream, FrameKind,
    StageTransport,
};
use crate::kv_prefix_cache::KvPrefixCache;
use crate::manifest::Manifest;
use crate::ov_moe::OvMoeRunner;
use std::num::NonZeroUsize;

use crate::runner::{LayerRange, Runner, RunnerError, RunnerOptions};

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
    /// If `Some(k)` and `k < manifest.top_k`, only the first k experts per
    /// token are dispatched per shell layer. See `docs/A3_TOPK_REDUCTION.md`.
    pub top_k_override: Option<u32>,
    /// Skip experts whose router weight falls below this threshold.
    /// 0.0 / None = disabled. Applied AFTER `top_k_override`.
    pub routing_threshold: Option<f32>,
    /// If `Some(k > 0)`, the engine runs n-gram (Prompt-Lookup,
    /// Yang et al. 2025) speculative decode instead of the plain greedy
    /// generate path. K is the per-round draft proposal length;
    /// see [`crate::ngram_draft::DEFAULT_DRAFT_K`] (currently 8) for the
    /// recommended default.
    ///
    /// - **Single-stage** (`total == 1`): uses
    ///   [`crate::runner::Runner::generate_speculative`] directly.
    /// - **Pipeline-parallel** (`total > 1`): rank 0 batches the K draft
    ///   verifies into one [`crate::dist::FrameKind::ForwardBatch`]
    ///   frame per round (one wire round-trip per round vs K round-trips
    ///   the per-token Forward path would pay). Worker ranks
    ///   transparently service ForwardBatch alongside the per-token
    ///   path — no per-rank flag needed.
    pub spec_decode_k: Option<u32>,
    /// Max entries in the static-prompt KV-prefix cache. `0` disables
    /// the cache (default). Each entry holds the populated K/V buffers
    /// for the cached prompt prefix — at K2.6 dimensions a 512-token
    /// snapshot is ~150 MiB, so practical caps for a chat workload
    /// are 1..8. See [`crate::kv_prefix_cache`] for the full design.
    ///
    /// Single-stage only on this PR — wiring the cache across pipeline
    /// stages requires a new transport frame for snapshot exchange,
    /// deferred to a follow-up. With `total > 1` this field is a no-op.
    pub kv_prefix_cache_size: u32,
    /// Magnitude threshold for the two-phase Gate-first FFN sparsity
    /// skip — see [`crate::runner::Runner::set_ffn_sparsity_threshold`].
    /// `0.0` = disabled (dense fallback; output bit-identical).
    /// Useful range for SwiGLU: 0.05–0.15. Higher = more skip + lower
    /// quality. See rainier `docs/POWERINFER_PORT.md`.
    pub ffn_sparsity_threshold: f32,
    /// Issue #35 — use the AXPY-form down kernel from
    /// [`cascadia_int4_gemm::ffn_axpy`] in the sparse FFN path. Only
    /// active when [`Self::ffn_sparsity_threshold`] > 0. Per-expert
    /// transposed-and-requantized down weights are persisted to
    /// [`Self::ffn_axpy_cache_dir`] (default
    /// `<model_dir>/.cascadia_transposed_down_v1/`) and mmap'd at
    /// runtime — this is the on-disk-cache fix that addresses PR #43's
    /// initial mmap-page-cache-eviction regression (see rainier
    /// `docs/AXPY_REGRESSION_ANALYSIS.md`).
    pub ffn_axpy_down: bool,
    /// Override location for the AXPY-form transposed-down cache.
    /// `None` ⇒ `<model_dir>/.cascadia_transposed_down_v1/`. Set
    /// to a fast local NVMe path if the model directory is on a
    /// slow / read-only mount.
    pub ffn_axpy_cache_dir: Option<PathBuf>,
    /// When `true`, eagerly pre-build the AXPY transposed-down
    /// cache for every `(layer, expert)` this rank may dispatch,
    /// at `Builder::build` time. Avoids the in-line build cost on
    /// the first prompt that touches each expert (recommended for
    /// diverse-prompt production workloads). At K2.6 dimensions
    /// the prebuild cost is ~7 min single-threaded or ~20 s with
    /// rayon; disk cost ~190 GiB on top of the model.
    pub ffn_axpy_prebuild: bool,
    /// Issue #38 (CHESS) — load per-channel FFN sparsity thresholds
    /// from this file. The file is produced by the
    /// `calibrate_ffn_thresholds` bin from a capture run. `None`
    /// (default) keeps the scalar [`Self::ffn_sparsity_threshold`]
    /// behaviour. When set, the per-channel τ vector takes precedence
    /// over the scalar τ for any layer covered by the file.
    pub ffn_sparsity_thresholds_file: Option<PathBuf>,
    /// Issue #38 (CHESS) — when set, record per-(layer, channel)
    /// `|silu(gate[c])| / max_j |silu(gate[j])|` histograms during
    /// every routed expert call. The histograms are dumped to this
    /// directory on engine close. Used as input to
    /// `calibrate_ffn_thresholds`. `None` (default) disables capture.
    /// Only meaningful while the AXPY-form path is active — the
    /// non-AXPY (bf16 boundary) path doesn't surface `silu(gate)`.
    pub ffn_sparsity_capture_dir: Option<PathBuf>,
}

impl SparseMoEBuilderConfig {
    pub fn new(model_dir: impl Into<PathBuf>, device: impl Into<String>) -> Self {
        Self {
            model_dir: model_dir.into(),
            device: device.into(),
            cache_dir: None,
            // 0 = unbounded (default); positive = LRU cap. The env var
            // `CASCADIA_MAX_EXPERTS_CACHED` overrides this if set.
            // See PowerInfer SmallThinker `MAX_N_CACHED` (MIT) —
            // rainier `docs/POWERINFER_PORT.md`.
            max_cached_experts: 0,
            rank: 0,
            total: 1,
            top_k_override: None,
            routing_threshold: None,
            spec_decode_k: None,
            kv_prefix_cache_size: 0,
            ffn_sparsity_threshold: 0.0,
            ffn_axpy_down: false,
            ffn_axpy_cache_dir: None,
            ffn_axpy_prebuild: false,
            ffn_sparsity_thresholds_file: None,
            ffn_sparsity_capture_dir: None,
        }
    }

    pub fn with_rank(mut self, rank: u32, total: u32) -> Self {
        self.rank = rank;
        self.total = total;
        self
    }

    /// Enable speculative decoding with the given draft K. Single-stage
    /// only; ignored on multi-stage configs.
    pub fn with_spec_decode_k(mut self, k: u32) -> Self {
        self.spec_decode_k = if k == 0 { None } else { Some(k) };
        self
    }

    /// Set the KV-prefix cache capacity (number of entries). `0`
    /// disables the cache.
    pub fn with_kv_prefix_cache_size(mut self, n: u32) -> Self {
        self.kv_prefix_cache_size = n;
        self
    }

    /// Set the LRU bound on the expert cache (number of entries). `0`
    /// = unbounded (default; preserves pre-LRU behaviour). Positive =
    /// cap on resident experts; LRU eviction when the cache is full.
    ///
    /// Inspired by PowerInfer SmallThinker's `MAX_N_CACHED` env var.
    /// At K2.6 dimensions each cached expert is ≈25 MiB so a cap of 256
    /// roughly bounds the expert pool at 6.4 GiB. See
    /// rainier `docs/POWERINFER_PORT.md` for guidance.
    pub fn with_max_cached_experts(mut self, n: u32) -> Self {
        self.max_cached_experts = n;
        self
    }

    /// Set the magnitude threshold for two-phase Gate-first FFN
    /// sparsity. `0.0` = dense (default; output bit-identical to the
    /// pre-port path). `0.05`–`0.15` is the useful range for SwiGLU.
    /// See [`crate::runner::Runner::set_ffn_sparsity_threshold`] and
    /// rainier `docs/POWERINFER_PORT.md`.
    pub fn with_ffn_sparsity_threshold(mut self, t: f32) -> Self {
        self.ffn_sparsity_threshold = if t > 0.0 { t } else { 0.0 };
        self
    }

    /// Issue #35 — enable the AXPY-form down kernel. Only has an
    /// effect when `ffn_sparsity_threshold > 0`. Lazily builds + caches
    /// transposed-and-requantized down weights per expert (~8.26 MiB
    /// extra heap per cached expert at K2.6 dimensions).
    pub fn with_ffn_axpy_down(mut self, on: bool) -> Self {
        self.ffn_axpy_down = on;
        self
    }
}

/// Build the [`RunnerOptions`] from the engine config + environment.
///
/// Env-var override order (highest precedence first):
///   1. `CASCADIA_MAX_EXPERTS_CACHED` (decimal integer; `0` = unbounded)
///   2. `config.max_cached_experts`   (`0` = unbounded)
///
/// Bad env-var values (non-integer, negative) are logged and fall
/// through to the config value.
pub(crate) fn resolve_runner_options(cfg: &SparseMoEBuilderConfig) -> RunnerOptions {
    let from_env = std::env::var("CASCADIA_MAX_EXPERTS_CACHED")
        .ok()
        .and_then(|s| {
            s.trim()
                .parse::<u32>()
                .map_err(|e| {
                    warn!(env = %s, err = %e, "ignoring invalid CASCADIA_MAX_EXPERTS_CACHED");
                })
                .ok()
        });
    let raw = from_env.unwrap_or(cfg.max_cached_experts);
    let max_cached_experts = if raw == 0 {
        None
    } else {
        // `raw > 0` ⇒ safe cast.
        Some(NonZeroUsize::new(raw as usize).expect("raw > 0"))
    };
    info!(
        from_env = ?from_env,
        from_cfg = cfg.max_cached_experts,
        resolved = ?max_cached_experts.map(NonZeroUsize::get),
        "resolved expert-cache LRU cap (None = unbounded)"
    );
    // `ffn_axpy_prebuild` is consumed in `Builder::build` after the
    // Runner is constructed, not via RunnerOptions — keeps the
    // Runner constructor side-effect-free.
    RunnerOptions {
        max_cached_experts,
        ffn_sparsity_threshold: cfg.ffn_sparsity_threshold,
        ffn_axpy_down: cfg.ffn_axpy_down,
        ffn_axpy_cache_dir: cfg.ffn_axpy_cache_dir.clone(),
        ffn_sparsity_thresholds_file: cfg.ffn_sparsity_thresholds_file.clone(),
        ffn_sparsity_capture_dir: cfg.ffn_sparsity_capture_dir.clone(),
    }
}

pub struct SparseMoEBuilder {
    pub config: SparseMoEBuilderConfig,
    runner: Option<Runner>,
    /// Set instead of `runner` when the manifest selects the OV-IR shell
    /// backend (MiniMax-M2). Single-stage only.
    ov_runner: Option<OvMoeRunner>,
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
            ov_runner: None,
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

        // OV-IR shell backend (MiniMax-M2): the whole architecture lives in
        // the traced graphs, so we run a dedicated runner instead of the
        // K2.6 MLA kernel. Single-stage loads the whole model; multi-stage
        // (total > 1) loads a contiguous layer slice per rank and exchanges
        // F32 hidden states over the same engine-agnostic dist wire protocol
        // the K2.6 path uses (rank 0 embeds + drives, the last rank runs the
        // head + sampler).
        let is_ov = Manifest::load(&self.config.model_dir)
            .map(|m| m.is_ov_shell())
            .unwrap_or(false);
        if is_ov {
            let cfg = self.config.clone();
            let total = cfg.total.max(1);
            let rank = cfg.rank.min(total - 1);
            // Explicit per-rank layer range from the ShardSpec (CLI
            // --layer-start/--layer-end). `layer_end == 0` means "unset" →
            // load_staged falls back to the even split.
            let (layer_start, layer_end) = (shard.layer_start, shard.layer_end);
            let opts = resolve_runner_options(&cfg);
            let cap = opts.max_cached_experts;
            let plugin_for_worker = plugin.clone();
            let (tx, rx) = std::sync::mpsc::channel();
            let join: JoinHandle<Result<OvMoeRunner, crate::ov_moe::OvMoeError>> =
                std::thread::spawn(move || {
                    tx.send(LoadProgress::message(
                        "loading MiniMax-M2 (OV-IR sparse-MoE)",
                    ))
                    .ok();
                    OvMoeRunner::load_staged(
                        cfg.model_dir.clone(),
                        &cfg.device,
                        plugin_for_worker,
                        cap,
                        rank,
                        total,
                        layer_start,
                        layer_end,
                        false, // force_split: production splits by device, not forced
                    )
                });
            let ov_runner = match join.join() {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => return Err(EngineError::Backend(format!("ov-moe load: {e}"))),
                Err(_) => return Err(EngineError::Backend("ov-moe load worker panicked".into())),
            };
            // The tokenizer is only needed on rank 0 (the API rank); worker
            // ranks drive themselves from the hidden states on the wire.
            if rank == 0 {
                let tok_path = self.config.model_dir.join("tokenizer.json");
                if tok_path.exists() {
                    self.tokenizer =
                        Some(Tokenizer::from_file(&tok_path).map_err(|e| {
                            EngineError::Backend(format!("load tokenizer.json: {e}"))
                        })?);
                } else {
                    warn!(
                        "no tokenizer.json at {} — engine will only accept pre-tokenized inputs",
                        tok_path.display()
                    );
                }
            }
            self.ov_runner = Some(ov_runner);
            let drained: Vec<LoadProgress> = rx.try_iter().collect();
            return Ok(Box::pin(stream::iter(drained)));
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
        let runner_opts = resolve_runner_options(&cfg);
        let (tx, rx) = std::sync::mpsc::channel();
        let join: JoinHandle<Result<Runner, RunnerError>> = std::thread::spawn(move || {
            tx.send(LoadProgress::message("loading sparse-MoE model"))
                .ok();
            Runner::load_with_options(
                cfg.model_dir.clone(),
                &cfg.device,
                plugin_for_worker,
                range_for_worker,
                runner_opts,
            )
        });

        let mut runner = match join.join() {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                return Err(EngineError::Backend(format!("runner load: {e}")));
            }
            Err(_) => {
                return Err(EngineError::Backend("runner load worker panicked".into()));
            }
        };
        // Plumb per-token expert-dispatch overrides into the runner.
        runner.set_top_k_override(self.config.top_k_override);
        runner.set_routing_threshold(self.config.routing_threshold);
        runner.set_ffn_sparsity_threshold(self.config.ffn_sparsity_threshold);
        runner.set_ffn_axpy_down(self.config.ffn_axpy_down);
        if self.config.ffn_axpy_prebuild {
            runner
                .prebuild_axpy_cache()
                .map_err(|e| EngineError::Backend(format!("AXPY prebuild: {e}")))?;
        }

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
        if let Some(ov) = self.ov_runner {
            let total = self.config.total.max(1);
            let rank = self.config.rank.min(total - 1);
            // Only rank 0 (the API rank) needs a tokenizer; worker ranks
            // drive themselves from the hidden states on the wire.
            if rank == 0 && self.tokenizer.is_none() {
                return Err(EngineError::Backend(
                    "tokenizer.json missing (required for the MiniMax-M2 API rank)".into(),
                ));
            }
            let runtime_handle = tokio::runtime::Handle::try_current()
                .map_err(|_| EngineError::Backend("Builder::build outside tokio context".into()))?;
            if total > 1 {
                info!(
                    rank,
                    total, "built MiniMax-M2 OV-IR engine (pipeline-parallel)"
                );
            } else {
                info!("built MiniMax-M2 OV-IR engine (single-stage)");
            }
            return Ok(Box::new(OvMoeEngine::new(
                ov,
                self.tokenizer,
                self.transport,
                runtime_handle,
                rank,
                total,
            )));
        }
        let runner = self.runner.ok_or(EngineError::NotLoaded)?;
        let total = self.config.total.max(1);
        let rank = self.config.rank.min(total - 1);
        let runtime_handle = tokio::runtime::Handle::try_current()
            .map_err(|_| EngineError::Backend("Builder::build outside tokio context".into()))?;
        if rank == 0 && self.tokenizer.is_none() {
            return Err(EngineError::Backend("tokenizer.json missing".into()));
        }
        // Spec-decode is honored on both single-stage and pipeline-
        // parallel paths now that FrameKind::ForwardBatch lets rank 0
        // verify K positions per wire round-trip. The single-stage path
        // uses `Runner::generate_speculative` directly; the
        // pipeline-parallel path uses `drive_generation_first_spec`
        // (added in PR #11), which batches the verify forwards over
        // ForwardBatch frames. Worker ranks transparently service either
        // path — they handle ForwardBatch frames as they arrive.
        let spec_decode_k = self.config.spec_decode_k;
        // KV-prefix cache is single-stage only on this PR. Warn (don't
        // error) on multi-stage configs so the same CLI flag works in
        // both topologies — the multi-stage path silently ignores it.
        let kv_cache_size = if total > 1 {
            if self.config.kv_prefix_cache_size > 0 {
                warn!(
                    requested = self.config.kv_prefix_cache_size,
                    total,
                    "kv-prefix-cache disabled: multi-stage cache requires per-stage snapshot exchange (not implemented yet)"
                );
            }
            0
        } else {
            self.config.kv_prefix_cache_size
        };
        let kv_prefix_cache = KvPrefixCache::new(kv_cache_size as usize);
        if kv_prefix_cache.enabled() {
            // Estimate footprint from the loaded runner's per-rank
            // KV layer count + manifest head dims. Assume a 512-token
            // representative prompt; the actual cost scales linearly.
            // At K2.6 single-stage (61 KV layers, 64 heads, 320 dim,
            // bf16) one 512-token snapshot is ~1.25 GiB, so even a
            // cap of 4 puts ~5 GiB on the resident set. Print a hard
            // warning above 16 entries; INFO otherwise.
            const REPRESENTATIVE_PROMPT_TOKENS: usize = 512;
            const HEAVY_ENTRY_THRESHOLD: usize = 16;
            let bytes_per_token = runner.estimated_snapshot_bytes_per_token();
            let est_per_snapshot = bytes_per_token * REPRESENTATIVE_PROMPT_TOKENS;
            let est_total = est_per_snapshot.saturating_mul(kv_prefix_cache.capacity());
            info!(
                capacity = kv_prefix_cache.capacity(),
                est_bytes_per_snapshot_512tok = est_per_snapshot,
                est_total_bytes_512tok = est_total,
                "kv-prefix-cache enabled (single-stage)"
            );
            if kv_prefix_cache.capacity() > HEAVY_ENTRY_THRESHOLD {
                warn!(
                    capacity = kv_prefix_cache.capacity(),
                    est_total_gib_512tok = est_total as f64 / (1024.0 * 1024.0 * 1024.0),
                    "kv-prefix-cache size is large; at K2.6 single-stage dims one snapshot is ~{:.0} MiB so resident-set growth is unbounded by capacity * snapshot bytes. Consider a smaller cap.",
                    est_per_snapshot as f64 / (1024.0 * 1024.0),
                );
            }
            // Spec-decode and the cache are orthogonal optimisations
            // today: the spec-decode generate path doesn't consult or
            // populate the cache (see `step_single_stage`). For greedy
            // requests with spec-decode on, the cache is silently
            // bypassed; for temp > 0 requests spec-decode falls back to
            // plain generate which does use the cache. Surface the
            // partial-coverage caveat at startup so the user isn't
            // surprised by missing hits in greedy mode.
            if let Some(k) = spec_decode_k {
                warn!(
                    spec_decode_k = k,
                    "kv-prefix-cache is bypassed on the spec-decode (greedy) generate path; only temperature>0 requests will populate / hit the cache while --spec-decode-k is active"
                );
            }
        }
        Ok(Box::new(SparseMoEEngine {
            runner,
            tokenizer: self.tokenizer,
            pending: VecDeque::new(),
            peer_disconnected: false,
            disconnect_reported: false,
            transport: self.transport,
            runtime_handle,
            rank,
            total,
            last_rank_history: Vec::new(),
            last_rank_rng: 0,
            last_rank_rng_seeded: false,
            spec_decode_k,
            kv_prefix_cache,
        }))
    }
}

/// Build a SamplingConfig for one task from the request's `temperature` +
/// `SamplingParams` (top_p / top_k / seed / frequency & presence penalty).
/// `repetition_penalty` / `repetition_window` are K2.6-tuned defaults not
/// exposed on the OpenAI surface. Lifted out so the single-stage and
/// multi-stage entry paths can't drift.
/// Infer the OpenAI `finish_reason` for a completed decode: hitting the token
/// cap is `length`; stopping short of it (EOS / stop sequence) is `stop`. The
/// runner returns the generated ids excluding EOS, so `n >= max_new` means the
/// cap was the limiter. An EOS landing exactly at the cap reports `length`.
fn finish_reason_for(n_tokens: usize, max_new: usize) -> FinishReason {
    if n_tokens >= max_new {
        FinishReason::Length
    } else {
        FinishReason::Stop
    }
}

fn sampling_from_task(task: &GenerationTask) -> crate::sampling::SamplingConfig {
    let s = &task.sampling;
    crate::sampling::SamplingConfig {
        temperature: task.temperature.max(0.0),
        top_p: if s.top_p > 0.0 { s.top_p.min(1.0) } else { 1.0 },
        top_k: s.top_k,
        frequency_penalty: s.frequency_penalty,
        presence_penalty: s.presence_penalty,
        repetition_penalty: 1.05,
        repetition_window: 64,
        seed: s.seed,
    }
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

/// Whether a worker-rank `step()` should surface its latched upstream
/// disconnect as a connection-fatal `Err` this call: yes exactly once, on the
/// first step after the link drops. After that the one-shot is spent so a
/// re-poll (the relay loop has already exited on the first one) doesn't flood.
/// Pure, for testing.
fn worker_should_report_disconnect(peer_disconnected: bool, already_reported: bool) -> bool {
    peer_disconnected && !already_reported
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
    /// Worker rank one-shot: true after `step()` has surfaced the latched
    /// disconnect as a connection-fatal `Err` to the relay loop. Stops the
    /// fatal Err from being re-emitted if `step()` is somehow polled again
    /// before the stage is rebuilt (the relay loop exits on the first one).
    disconnect_reported: bool,
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
    /// If `Some(k > 0)`, single-stage path runs n-gram speculative
    /// decode with draft K. Only honored when `total == 1` — checked at
    /// engine construction. None = plain greedy / sampled generate.
    spec_decode_k: Option<u32>,
    /// Single-stage static-prompt KV-prefix cache. Empty + capacity=0
    /// when the user didn't pass `--kv-prefix-cache-size` (default).
    /// Holds at most `capacity` packed snapshots; on lookup we restore
    /// the longest matching prefix's snapshot into the runner so the
    /// generate path skips that portion of prefill. See
    /// [`crate::kv_prefix_cache`] for the cache semantics.
    kv_prefix_cache: KvPrefixCache,
}

impl SparseMoEEngine {
    /// Bridge sync `Engine::step` code to an async transport future.
    /// Delegates to `cascadia_runner::run_async`, which consults the
    /// thread-local `BlockingContextGuard` flag — set by
    /// `Runner::run_relay_loop` for worker ranks — to pick the
    /// cheapest safe `block_on` variant. On a worker thread that's
    /// ~250x cheaper than wrapping in `block_in_place`.
    fn block_on<F: std::future::Future>(&self, fut: F) -> F::Output {
        cascadia_runner::run_async(&self.runtime_handle, fut)
    }

    fn is_last(&self) -> bool {
        self.transport.is_last()
    }

    /// Worker-side helper: if the runner's KV is AHEAD of the driver's
    /// requested `past_seq_len`, shrink it back. Used to absorb
    /// spec-decode rejections without a dedicated wire frame — the
    /// driver is the source of truth for the accepted prefix length and
    /// the worker trusts the past_seq_len carried on every Forward /
    /// ForwardBatch frame. No-op when the worker is already at or
    /// behind the target (the non-spec path).
    ///
    /// Also handles the last-rank sampling-history book-keeping: we
    /// strip the most-recently-pushed tokens off `last_rank_history`
    /// so the repetition-penalty window doesn't double-count rejected
    /// drafts. Mid-ranks have no sampling state, so this is a layers-
    /// only rewind there.
    fn maybe_rewind_to(&mut self, target_past_seq_len: usize) {
        let lens = self.runner.kv_past_seq_lens();
        let Some(&current) = lens.iter().max() else {
            return;
        };
        if target_past_seq_len < current {
            let n = current - target_past_seq_len;
            self.runner.rewind_kv(n);
            if self.is_last() {
                let pop = n.min(self.last_rank_history.len());
                let new_len = self.last_rank_history.len() - pop;
                self.last_rank_history.truncate(new_len);
            }
        }
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

    fn cancel(&mut self, task_id: &TaskId) {
        // Drop the queued task. step() runs one task end-to-end and holds the
        // engine mutex for that whole generation, so cancel() (also &mut self)
        // runs only between tasks — it prevents a queued task from starting but
        // cannot interrupt one already decoding. Mid-generation interruption
        // would need an out-of-mutex cancel token threaded into the decode
        // loop (architectural follow-up).
        self.pending.retain(|t| &t.task_id != task_id);
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if self.total == 1 {
            return Ok(self.step_single_stage());
        }
        // step_first handles its own errors terminally (final-marker chunk on
        // the driver), so rank 0 never surfaces an Err here.
        if self.rank == 0 {
            return Ok(self.step_first());
        }
        // Worker rank. step_worker returns empty on a latched upstream
        // disconnect. The worker's upstream socket can only be re-accepted by
        // a rebuild, so once disconnected, surface a connection-fatal Err to
        // run_relay_loop (its ONLY driver — rank-0/single-stage go through
        // generate()) so it exits and systemd rebuilds the stage, instead of
        // backing off Ok(empty) forever. Emit it exactly once; the loop bails
        // on the first fatal Err.
        let produced = self.step_worker();
        if worker_should_report_disconnect(self.peer_disconnected, self.disconnect_reported) {
            self.disconnect_reported = true;
            return Err(EngineError::NotConnected);
        }
        Ok(produced)
    }

    fn close(&mut self) {
        // Issue #38: dump the gate-capture histograms before tearing
        // down the runner. A clean shutdown (SIGTERM via
        // cascadia-cli's graceful-shutdown wiring) thus persists
        // calibration data without further user action.
        match self.runner.dump_gate_capture() {
            Ok((0, _)) => {}
            Ok((n_layers, total_samples)) => {
                info!(
                    n_layers,
                    total_samples, "dumped FFN gate-capture histograms (CHESS calibration)"
                );
            }
            Err(e) => {
                warn!("failed to dump gate-capture histograms on close: {e}");
            }
        }
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
        // Choose generate path: spec-decode if configured (and the
        // sampling config is greedy — the spec-decode helper falls
        // back to plain generate on temp>0 anyway, but this keeps the
        // log message accurate). KV-prefix cache is only consulted on
        // the plain generate path: the spec-decode helper consumes
        // KV-cache state internally, so wiring prefix-cache replay into
        // it would need extra plumbing — deferred.
        let use_spec = self.spec_decode_k.is_some() && sampling_cfg.temperature <= 0.0;
        let generated = if use_spec {
            let k = self.spec_decode_k.unwrap();
            let mut draft = crate::ngram_draft::Draft::new().with_draft_k(k as usize);
            info!(
                task = %task.task_id,
                draft_k = k,
                "using n-gram speculative decode"
            );
            match self
                .runner
                .generate_speculative(&prompt_ids, max_new, &sampling_cfg, &mut draft)
            {
                Ok(g) => g,
                Err(e) => {
                    warn!(task = %task.task_id, "runner spec-decode failed: {e}");
                    let final_chunk = Chunk::final_marker(task.task_id.clone(), "");
                    return vec![(task.task_id, final_chunk)];
                }
            }
        } else {
            let cache_opt: Option<&mut KvPrefixCache> = if self.kv_prefix_cache.enabled() {
                Some(&mut self.kv_prefix_cache)
            } else {
                None
            };
            match self
                .runner
                .generate_with_cache(&prompt_ids, max_new, &sampling_cfg, cache_opt)
            {
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
        chunk.finish_reason = Some(finish_reason_for(n_tokens as usize, max_new));
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
        // If spec-decode is enabled AND we're greedy (temp==0), use the
        // batched ForwardBatch path; otherwise fall back to the
        // per-token Forward driver.
        let use_spec =
            self.spec_decode_k.map(|k| k > 0).unwrap_or(false) && sampling_cfg.temperature <= 0.0;
        let result_tokens = if use_spec {
            let k = self.spec_decode_k.unwrap();
            info!(
                task = %task.task_id,
                draft_k = k,
                "using n-gram speculative decode (pipeline-parallel)"
            );
            match self.drive_generation_first_spec(
                &prompt_ids,
                max_new,
                k as usize,
                &sampling_cfg,
                &downstream,
            ) {
                Ok(g) => g,
                Err(e) => {
                    warn!(task = %task.task_id, "rank-0 spec-decode driver failed: {e}");
                    return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
                }
            }
        } else {
            match self.drive_generation_first(&prompt_ids, max_new, &sampling_cfg, &downstream) {
                Ok(g) => g,
                Err(e) => {
                    warn!(task = %task.task_id, "rank-0 driver failed: {e}");
                    return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
                }
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
        chunk.finish_reason = Some(finish_reason_for(n_tokens as usize, max_new));
        vec![(task.task_id.clone(), chunk)]
    }

    /// Rank 0 generation loop. For each prompt token + each decode
    /// step: embed → forward through my shells → send hidden
    /// downstream → recv sampled token back. Discards prefill samples
    /// except the last (which becomes the first generated token).
    fn drive_generation_first(
        &mut self,
        prompt_ids: &[i64],
        max_new: usize,
        cfg: &crate::sampling::SamplingConfig,
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
        // autolab/k26-perf q1 instrumentation: split timing of send vs round-trip.
        let wire_t0 = Instant::now();
        let result = self.block_on(async {
            send_forward(
                downstream,
                past_seq_len,
                cfg,
                &h_after_shells,
                [1, 1, hidden as u32],
            )
            .await
            .map_err(|e| format!("send_forward: {e}"))?;
            let send_done_us = wire_t0.elapsed().as_micros() as u64;
            match recv_kind_client(downstream).await {
                Ok(Some(FrameKind::Token)) => {
                    let token = recv_token_body_client(downstream)
                        .await
                        .map_err(|e| format!("recv_token: {e}"))?;
                    Ok((token, send_done_us))
                }
                Ok(Some(other)) => Err(format!("unexpected frame after forward: {other:?}")),
                Ok(None) => Err("downstream closed during recv_kind".into()),
                Err(e) => Err(format!("recv_kind: {e}")),
            }
        });
        match result {
            Ok((token, send_done_us)) => {
                let total_wire_us = wire_t0.elapsed().as_micros() as u64;
                info!(
                    stage = "rank0_wire",
                    send_done_us,
                    total_wire_us,
                    downstream_compute_us = total_wire_us.saturating_sub(send_done_us),
                    "stage_timing"
                );
                Ok(token)
            }
            Err(e) => Err(e),
        }
    }

    /// Pipeline-parallel speculative-decode driver on rank 0. Mirrors
    /// [`crate::runner::Runner::generate_speculative`] but issues
    /// `FrameKind::ForwardBatch` for each verify round so K positions
    /// fit in one wire round-trip instead of K round-trips.
    ///
    /// Greedy-only by construction — temperature > 0 would change the
    /// distribution under greedy acceptance; callers must dispatch the
    /// non-greedy path to `drive_generation_first` instead.
    fn drive_generation_first_spec(
        &mut self,
        prompt_ids: &[i64],
        max_new: usize,
        draft_k: usize,
        cfg: &crate::sampling::SamplingConfig,
        downstream: &Arc<TokioMutex<ActivationClient>>,
    ) -> Result<Vec<i64>, String> {
        let mut draft = crate::ngram_draft::Draft::new().with_draft_k(draft_k);
        // Reset state on both ends. Rank-0's runner.reset_kv was
        // already called by the caller (step_first sends Reset
        // downstream and clears its own KV); we just need draft reset.
        draft.reset();

        let eos: Vec<i64> = self
            .runner
            .manifest
            .eos_token_ids
            .iter()
            .map(|&x| x as i64)
            .collect();
        let mut history: Vec<i64> = Vec::with_capacity(prompt_ids.len() + max_new);
        let mut generated: Vec<i64> = Vec::with_capacity(max_new);

        // Prefill: same shape as the non-spec driver — one Forward per
        // prompt token. Keeps the wire payload narrow during prefill
        // (no draft model is consulted until decode begins).
        info!(
            prompt_len = prompt_ids.len(),
            "spec: prefill (token-by-token, distributed)"
        );
        for (i, &t) in prompt_ids.iter().enumerate() {
            history.push(t);
            let token_back = self
                .forward_one_token_first(&history, cfg, downstream)
                .map_err(|e| format!("spec prefill step {i}: {e}"))?;
            if i + 1 == prompt_ids.len() {
                if eos.contains(&token_back) {
                    return Ok(generated);
                }
                generated.push(token_back);
                history.push(token_back);
                draft.warm_with_prompt(prompt_ids);
                draft.append(token_back);
            }
            let _ = token_back;
        }

        let mut n_rounds: u32 = 0;
        let mut n_drafts_total: u32 = 0;
        let mut n_accepted_total: u32 = 0;

        while generated.len() < max_new {
            let budget = max_new - generated.len();
            let mut drafts = draft.propose();
            if drafts.len() > budget {
                drafts.truncate(budget);
            }

            // No proposal → fall back to one standard forward step.
            if drafts.is_empty() {
                let token_back = self
                    .forward_one_token_first(&history, cfg, downstream)
                    .map_err(|e| format!("spec fallback step: {e}"))?;
                if eos.contains(&token_back) {
                    break;
                }
                history.push(token_back);
                generated.push(token_back);
                draft.append(token_back);
                continue;
            }

            n_rounds += 1;
            n_drafts_total += drafts.len() as u32;

            // Run K verify forwards as ONE wire round-trip.
            // Each forward conditions on history+drafts[0..i] and
            // produces the target's prediction for position i.
            // past_seq_len_start = history.len() - 1 BEFORE appending
            // any draft (matches the single-token path's convention:
            // forward_one_token_first uses past_seq_len = history.len()-1
            // because it embeds the LAST history token).
            let target_samples = self
                .forward_batch_first(&drafts, &mut history, cfg, downstream)
                .map_err(|e| format!("spec batch verify (round {n_rounds}): {e}"))?;
            // Acceptance: longest matching prefix.
            let accepted = crate::spec_decode::count_accepted(&drafts, &target_samples);
            n_accepted_total += accepted as u32;

            let bonus_forward_ran = accepted == drafts.len();
            let bonus: i64 = if !bonus_forward_ran {
                target_samples[accepted]
            } else {
                // All accepted: run one extra forward (the bonus) to
                // get the next round's prev_correction. Same as the
                // single-stage path; kept as a single-token Forward so
                // we don't introduce a 1-token ForwardBatch frame.
                self.forward_one_token_first(&history, cfg, downstream)
                    .map_err(|e| format!("spec bonus forward (round {n_rounds}): {e}"))?
            };

            // Reconciliation: pop rejected drafts from history, then
            // rewind KV so the next round's past_seq_len_start matches
            // the rank-0 KV state.
            //
            // We defer to `spec_decode::reconcile_after_round` with
            // `pending_token_in_history=true`, the runner / pipeline-
            // parallel convention: history pre-pushes `first_gen`
            // before the K-loop and we append `bonus` AFTER the K-loop,
            // so the post-round trail between history.len() and KV
            // must be 1 (the bonus's pending slot, which the next
            // round's first forward will fill). The helper's contract
            // documents the same K-A-1 (partial) / 0 (all-accepted)
            // arithmetic this driver historically computed inline; see
            // fix/spec-decode-reconcile-off-by-one-043 for the
            // regression tests covering this path.
            let r = crate::spec_decode::reconcile_after_round(
                drafts.len(),
                accepted,
                bonus_forward_ran,
                true,
            );
            if r.history_pop > 0 {
                history.truncate(history.len() - r.history_pop);
            }
            if r.kv_rewind > 0 {
                self.runner.rewind_kv(r.kv_rewind);
            }

            let mut hit_eos = false;
            let mut bonus_pushed_to_history = false;
            for &t in drafts.iter().take(accepted) {
                if eos.contains(&t) {
                    hit_eos = true;
                    break;
                }
                generated.push(t);
                draft.append(t);
                if generated.len() >= max_new {
                    break;
                }
            }
            if !hit_eos && generated.len() < max_new {
                if eos.contains(&bonus) {
                    hit_eos = true;
                } else {
                    history.push(bonus);
                    draft.append(bonus);
                    generated.push(bonus);
                    bonus_pushed_to_history = true;
                }
            }

            // Debug invariant — same shape as the single-stage runner's
            // generate_speculative path. After this round's reconcile,
            // every layer's past_seq_len must trail history.len() by
            // exactly 1 when the bonus rode through (the next round's
            // first verify forward will fill its KV slot), and by 0
            // when we cut the round short (EOS hit or max_new saturated
            // before the bonus push). Strip from prod paths.
            //
            // The pipeline-parallel driver uses
            // `pending_token_in_history=true` in reconcile_after_round
            // for the same reason: history pre-pushes first_gen and
            // appends each round's bonus, both riding ahead of KV by
            // one slot. See fix/spec-decode-reconcile-off-by-one-043
            // for the regression tests that pin this convention.
            let expected_drift = if bonus_pushed_to_history { 1 } else { 0 };
            debug_assert!(
                self.runner.kv_invariant_holds(&history, expected_drift),
                "KV invariant broken in distributed spec-decode (expected drift {expected_drift})"
            );

            if hit_eos {
                break;
            }
        }

        info!(
            tokens = generated.len(),
            n_rounds,
            total_drafts = n_drafts_total,
            total_accepted = n_accepted_total,
            accept_rate = if n_drafts_total > 0 {
                n_accepted_total as f32 / n_drafts_total as f32
            } else {
                0.0
            },
            "spec_decode_pipeline done"
        );
        Ok(generated)
    }

    /// Run K verify forwards through the pipeline in one batched wire
    /// round-trip. Pushes the K drafted tokens into `history` as it
    /// runs each layer-0 + shell pass (so the next forward conditions
    /// on the drafted prefix). Returns the K target-sampled tokens (one
    /// per draft position) so the caller can run `count_accepted`.
    ///
    /// Note on KV: this method ADVANCES rank-0's KV by exactly K slots
    /// (one per layer-0 + shell step). The caller is responsible for
    /// `rewind_kv(rejected_count)` after running `count_accepted`.
    fn forward_batch_first(
        &mut self,
        drafts: &[i64],
        history: &mut Vec<i64>,
        cfg: &crate::sampling::SamplingConfig,
        downstream: &Arc<TokioMutex<ActivationClient>>,
    ) -> Result<Vec<i64>, String> {
        let hidden = self.runner.manifest.hidden_size as usize;
        let k = drafts.len();
        if k == 0 {
            return Err("forward_batch_first called with empty drafts".into());
        }
        let past_seq_len_start = history
            .len()
            .checked_sub(1)
            .ok_or_else(|| "forward_batch_first: empty history".to_string())?
            as u32;

        // Run K rank-0 (layer 0 + shells) forwards locally, gathering K
        // hidden rows. After this loop:
        // - history grew by K (drafted tokens pushed in order)
        // - runner's KV grew by K slots (layer 0 + every shell)
        let mut h_batch: Vec<f32> = Vec::with_capacity(k * hidden);
        for (i, &draft_tok) in drafts.iter().enumerate() {
            // last token currently in history is what layer 0 should
            // embed for THIS step. For step 0 it's the prior bonus /
            // first generated token; for step i>0 it's drafts[i-1]
            // because we pushed it on the previous iteration.
            let last_id = *history
                .last()
                .ok_or_else(|| "forward_batch_first: empty history mid-step".to_string())?;
            let past_seq_len = past_seq_len_start as usize + i;
            let h_tail = self
                .runner
                .forward_layer0_step(last_id)
                .map_err(|e| format!("layer0_step (batch {i}): {e}"))?;
            let h_after = self
                .runner
                .forward_shells(&h_tail, &[1, 1, hidden], past_seq_len)
                .map_err(|e| format!("forward_shells (batch {i}): {e}"))?;
            h_batch.extend_from_slice(&h_after);
            // Push the drafted token AFTER computing this step's
            // hidden (which conditions on the prior history). The next
            // iteration's layer 0 will embed `draft_tok`.
            history.push(draft_tok);
        }

        let wire_t0 = Instant::now();
        let result = self.block_on(async {
            send_forward_batch(
                downstream,
                past_seq_len_start,
                k as u32,
                cfg,
                &h_batch,
                [1, k as u32, hidden as u32],
            )
            .await
            .map_err(|e| format!("send_forward_batch: {e}"))?;
            let send_done_us = wire_t0.elapsed().as_micros() as u64;
            match recv_kind_client(downstream).await {
                Ok(Some(FrameKind::TokenBatch)) => {
                    let tokens = recv_token_batch_body_client(downstream)
                        .await
                        .map_err(|e| format!("recv_token_batch: {e}"))?;
                    Ok((tokens, send_done_us))
                }
                Ok(Some(other)) => Err(format!("unexpected frame after forward_batch: {other:?}")),
                Ok(None) => Err("downstream closed during recv_kind (batch)".into()),
                Err(e) => Err(format!("recv_kind (batch): {e}")),
            }
        });
        match result {
            Ok((tokens, send_done_us)) => {
                let total_wire_us = wire_t0.elapsed().as_micros() as u64;
                info!(
                    stage = "rank0_wire_batch",
                    k,
                    send_done_us,
                    total_wire_us,
                    downstream_compute_us = total_wire_us.saturating_sub(send_done_us),
                    "stage_timing"
                );
                if tokens.len() != k {
                    return Err(format!(
                        "forward_batch_first: expected {k} tokens back, got {}",
                        tokens.len()
                    ));
                }
                Ok(tokens)
            }
            Err(e) => Err(e),
        }
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
                // Self-rewind if the driver's past_seq_len is behind our
                // local KV (spec-decode rejection in the prior round —
                // see drive_generation_first_spec). The driver is the
                // source of truth for the accepted prefix length; the
                // worker trusts it and shrinks its KV before this
                // forward writes its slot. No-op when past_seq_len is
                // current (the normal non-spec path).
                self.maybe_rewind_to(past_seq_len as usize);
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
            FrameKind::ForwardBatch => self.handle_forward_batch_frame(upstream, downstream),
            FrameKind::Token => Err(format!(
                "rank {} received unexpected TOKEN from upstream",
                self.rank
            )),
            FrameKind::TokenBatch => Err(format!(
                "rank {} received unexpected TOKEN_BATCH from upstream",
                self.rank
            )),
        }
    }

    /// Worker-side ForwardBatch handler. Drives K sequential shell
    /// forwards over the K hidden rows received in one frame, then
    /// either runs head+sample × K on the last rank (TokenBatch back
    /// upstream) or relays the K hiddens downstream and waits for a
    /// TokenBatch to forward upstream on a mid rank.
    ///
    /// The wire savings vs K separate Forward frames is one round-trip
    /// of latency per K verifies. At the cascadia-enterprise fleet's 22 ms cross-host RT,
    /// K=8 saves ~7 round-trips per spec-decode round (~154 ms). The
    /// per-token shell compute is unchanged — `shell_forward_decode_int4`
    /// still only accepts seq=1, so this is a wire-batching unlock, not
    /// a kernel-batching one.
    fn handle_forward_batch_frame(
        &mut self,
        upstream: &Arc<TokioMutex<ActivationServer>>,
        downstream: Option<Arc<TokioMutex<ActivationClient>>>,
    ) -> Result<(), String> {
        let (past_seq_len_start, batch_count, sampling_cfg, hidden_f32, in_shape) = self
            .block_on(recv_forward_batch_body_server(upstream))
            .map_err(|e| format!("recv_forward_batch: {e}"))?;
        let hidden = self.runner.manifest.hidden_size as usize;
        if in_shape[0] != 1 || in_shape[1] != batch_count || in_shape[2] as usize != hidden {
            return Err(format!(
                "forward_batch shape unexpected {:?} (expected [1, {batch_count}, {hidden}])",
                in_shape
            ));
        }
        let k = batch_count as usize;

        // Self-rewind if the driver's past_seq_len_start is behind our
        // local KV (spec-decode rejection in the prior round). Trusting
        // the driver here lets us skip a dedicated RewindBatch frame.
        self.maybe_rewind_to(past_seq_len_start as usize);

        // Run K sequential shell forwards. Each forward consumes one
        // hidden row and advances the per-layer KV cache by 1 slot.
        // We accumulate the per-position post-shell hidden states so
        // mid-ranks can relay them downstream as a single ForwardBatch.
        let mut h_post_batch: Vec<f32> = if !self.is_last() {
            Vec::with_capacity(k * hidden)
        } else {
            Vec::new()
        };
        let mut tokens_out: Vec<i64> = Vec::with_capacity(k);

        // Last-rank-only: lazily seed RNG from the FIRST forward batch's
        // sampling config.
        if self.is_last() && !self.last_rank_rng_seeded {
            self.last_rank_rng = crate::sampling::init_rng(sampling_cfg.seed);
            self.last_rank_rng_seeded = true;
        }

        for i in 0..k {
            let h_row = &hidden_f32[i * hidden..(i + 1) * hidden];
            let past_seq_len = past_seq_len_start as usize + i;
            let h_after = self
                .runner
                .forward_shells(h_row, &[1, 1, hidden], past_seq_len)
                .map_err(|e| format!("forward_shells (batch step {i}): {e}"))?;
            if self.is_last() {
                let logits = self
                    .runner
                    .forward_head_last(&h_after, 1)
                    .map_err(|e| format!("forward_head (batch step {i}): {e}"))?;
                let token = crate::sampling::sample(
                    &logits,
                    &self.last_rank_history,
                    &sampling_cfg,
                    &mut self.last_rank_rng,
                );
                // Mirror the existing per-token Forward path: push into
                // last_rank_history so the rep-penalty window includes
                // this token in subsequent forwards. The caller is
                // responsible for any rewind on rejection — the worker
                // does not know which drafts were accepted.
                self.last_rank_history.push(token);
                tokens_out.push(token);
            } else {
                h_post_batch.extend_from_slice(&h_after);
            }
        }

        if self.is_last() {
            self.block_on(send_token_batch_upstream(upstream, &tokens_out))
                .map_err(|e| format!("send_token_batch: {e}"))?;
            Ok(())
        } else {
            let down = downstream.ok_or_else(|| "mid rank missing downstream".to_string())?;
            self.block_on(async {
                send_forward_batch(
                    &down,
                    past_seq_len_start,
                    batch_count,
                    &sampling_cfg,
                    &h_post_batch,
                    [1, batch_count, hidden as u32],
                )
                .await
                .map_err(|e| format!("send_forward_batch: {e}"))?;
                let tokens = match recv_kind_client(&down).await {
                    Ok(Some(FrameKind::TokenBatch)) => recv_token_batch_body_client(&down)
                        .await
                        .map_err(|e| format!("recv_token_batch: {e}"))?,
                    Ok(Some(other)) => {
                        return Err(format!(
                            "unexpected mid-rank frame after ForwardBatch: {other:?}"
                        ));
                    }
                    Ok(None) => return Err("downstream closed mid-batch".into()),
                    Err(e) => return Err(format!("recv_kind (batch): {e}")),
                };
                send_token_batch_upstream(upstream, &tokens)
                    .await
                    .map_err(|e| format!("send_token_batch (relay): {e}"))
            })?;
            Ok(())
        }
    }
}

const OV_MAX_PENDING: usize = 64;

/// Engine for the OV-IR shell backend (MiniMax-M2).
///
/// - **Single-stage** (`total == 1`): one engine holds the whole model and
///   drives [`OvMoeRunner::generate`] end-to-end (the path the tiny
///   correctness test and the 230B smoke exercise).
/// - **Pipeline-parallel** (`total > 1`): each rank holds a contiguous
///   layer slice. Rank 0 embeds the token, runs its shells, and ships the
///   F32 hidden state downstream over `cascadia-transport`; middle ranks
///   relay; the last rank runs the head + sampler and returns the token
///   upstream. The wire protocol is the engine-agnostic [`crate::dist`]
///   one the K2.6 path uses (Forward / Reset / Token frames). No
///   spec-decode (M2 sends only per-token Forward frames).
pub struct OvMoeEngine {
    runner: OvMoeRunner,
    tokenizer: Option<Tokenizer>,
    pending: VecDeque<GenerationTask>,
    transport: StageTransport,
    runtime_handle: tokio::runtime::Handle,
    rank: u32,
    total: u32,
    /// Set on a worker rank when the upstream socket closes cleanly, so
    /// `step_worker` doesn't hot-spin on `recv_kind_server` → `Ok(None)`.
    peer_disconnected: bool,
    /// Last-rank only: tokens this rank has sampled since the last `Reset`,
    /// used as the repetition-penalty `history`. Like the K2.6 path, prompt
    /// tokens are not mirrored here (they flow only as hidden states), so
    /// the rep-penalty window covers generated tokens only. Greedy
    /// (`repetition_penalty == 1.0`) is unaffected, so a greedy pipeline
    /// run matches the single-stage greedy output exactly.
    last_rank_history: Vec<i64>,
    last_rank_rng: u64,
    last_rank_rng_seeded: bool,
}

impl OvMoeEngine {
    fn new(
        runner: OvMoeRunner,
        tokenizer: Option<Tokenizer>,
        transport: StageTransport,
        runtime_handle: tokio::runtime::Handle,
        rank: u32,
        total: u32,
    ) -> Self {
        Self {
            runner,
            tokenizer,
            pending: VecDeque::new(),
            transport,
            runtime_handle,
            rank,
            total,
            peer_disconnected: false,
            last_rank_history: Vec::new(),
            last_rank_rng: 0,
            last_rank_rng_seeded: false,
        }
    }

    /// Bridge sync `Engine::step` code to an async transport future (same
    /// thread-local fast path the K2.6 engine uses on worker ranks).
    fn block_on<F: std::future::Future>(&self, fut: F) -> F::Output {
        cascadia_runner::run_async(&self.runtime_handle, fut)
    }

    fn is_last(&self) -> bool {
        self.transport.is_last()
    }

    /// Single-stage path: tokenize, run the whole model, decode.
    fn step_single_stage(&mut self) -> Vec<(TaskId, Chunk)> {
        let task = match self.pending.pop_front() {
            Some(t) => t,
            None => return Vec::new(),
        };
        let Some(tok) = self.tokenizer.as_ref() else {
            warn!(task = %task.task_id, "MiniMax-M2 engine has no tokenizer");
            return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
        };
        let started = Instant::now();
        let prompt_ids: Vec<u32> = match tok.encode(task.prompt.as_str(), true) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                warn!(task = %task.task_id, "tokenizer encode failed: {e}");
                return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
            }
        };
        let max_new = task.max_tokens.max(1) as usize;
        let sampling_cfg = sampling_from_task(&task);
        let generated = match self.runner.generate(&prompt_ids, max_new, &sampling_cfg) {
            Ok(g) => g,
            Err(e) => {
                warn!(task = %task.task_id, "MiniMax-M2 generate failed: {e}");
                return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
            }
        };
        let n_tokens = generated.len() as u32;
        let text = tok.decode(&generated, true).unwrap_or_default();
        let elapsed = started.elapsed().as_secs_f64();
        info!(
            task = %task.task_id,
            tokens = n_tokens,
            elapsed_s = elapsed,
            tok_s = if elapsed > 0.0 { n_tokens as f64 / elapsed } else { 0.0 },
            "task done (MiniMax-M2 single-stage)"
        );
        let mut chunk = Chunk::final_marker(task.task_id.clone(), text);
        chunk.n_tokens = Some(n_tokens);
        chunk.finish_reason = Some(finish_reason_for(n_tokens as usize, max_new));
        vec![(task.task_id.clone(), chunk)]
    }

    /// Rank-0 driver: embed + run my shells, ship hidden downstream, await
    /// the sampled token back. Returns the token the last rank sampled.
    fn forward_one_token_first(
        &mut self,
        token: u32,
        pos: usize,
        cfg: &crate::sampling::SamplingConfig,
        downstream: &Arc<TokioMutex<ActivationClient>>,
    ) -> Result<i64, String> {
        let hidden = self
            .runner
            .embed_token(token)
            .map_err(|e| format!("embed: {e}"))?;
        let hidden = self
            .runner
            .forward_layers(hidden, pos)
            .map_err(|e| format!("forward: {e}"))?;
        let h = self.runner.hidden_size() as u32;
        self.block_on(async {
            send_forward(downstream, pos as u32, cfg, &hidden, [1, 1, h])
                .await
                .map_err(|e| format!("send_forward: {e}"))?;
            match recv_kind_client(downstream).await {
                Ok(Some(FrameKind::Token)) => recv_token_body_client(downstream)
                    .await
                    .map_err(|e| format!("recv_token: {e}")),
                Ok(Some(other)) => Err(format!("rank 0 expected Token, got {other:?}")),
                Ok(None) => Err("downstream closed before Token".into()),
                Err(e) => Err(format!("recv_kind: {e}")),
            }
        })
    }

    /// Rank-0 pipeline driver: tokenize, reset the ring, drive prefill +
    /// decode one token per wire round-trip, decode, emit one final chunk.
    fn step_first(&mut self) -> Vec<(TaskId, Chunk)> {
        let task = match self.pending.pop_front() {
            Some(t) => t,
            None => return Vec::new(),
        };
        let id = task.task_id.clone();
        let Some(downstream) = self.transport.downstream.clone() else {
            warn!(task = %id, "rank 0 has no downstream; cannot drive pipeline");
            return vec![(id.clone(), Chunk::error(id, "rank 0 missing downstream"))];
        };
        if self.tokenizer.is_none() {
            warn!(task = %id, "MiniMax-M2 rank 0 has no tokenizer");
            return vec![(id.clone(), Chunk::final_marker(id, ""))];
        }
        let started = Instant::now();
        let prompt_ids: Vec<u32> = match self
            .tokenizer
            .as_ref()
            .unwrap()
            .encode(task.prompt.as_str(), true)
        {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                warn!(task = %id, "tokenizer encode failed: {e}");
                return vec![(id.clone(), Chunk::final_marker(id, ""))];
            }
        };
        if prompt_ids.is_empty() {
            return vec![(id.clone(), Chunk::final_marker(id, ""))];
        }
        let max_new = task.max_tokens.max(1) as usize;
        let cfg = sampling_from_task(&task);
        let eos = self.runner.eos_token_ids().to_vec();

        // Reset KV across the whole pipeline before the new generation.
        self.runner.reset();
        if let Err(e) = self.block_on(send_reset(&downstream)) {
            warn!(task = %id, "send_reset failed: {e}");
            return vec![(id.clone(), Chunk::error(id, format!("reset: {e}")))];
        }

        // Prefill: feed every prompt token. The sample produced after the
        // LAST prompt token is the first generated token — matching
        // `OvMoeRunner::generate_timed`.
        let mut pos = 0usize;
        let mut next: i64 = -1;
        for &t in &prompt_ids {
            match self.forward_one_token_first(t, pos, &cfg, &downstream) {
                Ok(tok_back) => next = tok_back,
                Err(e) => {
                    warn!(task = %id, "prefill forward failed: {e}");
                    return vec![(id.clone(), Chunk::error(id, e))];
                }
            }
            pos += 1;
        }

        // Decode: like generate_timed, push the token then stop on max_new
        // or EOS (the EOS token is included in the output).
        let mut generated: Vec<u32> = Vec::with_capacity(max_new);
        loop {
            let next_u = next as u32;
            generated.push(next_u);
            if generated.len() >= max_new || eos.contains(&next_u) {
                break;
            }
            match self.forward_one_token_first(next_u, pos, &cfg, &downstream) {
                Ok(tok_back) => next = tok_back,
                Err(e) => {
                    warn!(task = %id, "decode forward failed; emitting partial output: {e}");
                    break;
                }
            }
            pos += 1;
        }

        let n_tokens = generated.len() as u32;
        let text = self
            .tokenizer
            .as_ref()
            .unwrap()
            .decode(&generated, true)
            .unwrap_or_default();
        let elapsed = started.elapsed().as_secs_f64();
        info!(
            task = %id,
            tokens = n_tokens,
            total = self.total,
            elapsed_s = elapsed,
            tok_s = if elapsed > 0.0 { n_tokens as f64 / elapsed } else { 0.0 },
            "task done (MiniMax-M2 pipeline-parallel)"
        );
        let mut chunk = Chunk::final_marker(id.clone(), text);
        chunk.n_tokens = Some(n_tokens);
        chunk.finish_reason = Some(finish_reason_for(n_tokens as usize, max_new));
        vec![(id, chunk)]
    }

    /// Worker rank (rank > 0): service one frame from upstream per call.
    fn step_worker(&mut self) -> Vec<(TaskId, Chunk)> {
        if self.peer_disconnected {
            std::thread::sleep(WORKER_BACKOFF);
            return Vec::new();
        }
        let Some(upstream) = self.transport.upstream.clone() else {
            warn!("worker rank has no upstream socket");
            std::thread::sleep(WORKER_BACKOFF);
            return Vec::new();
        };
        let downstream = self.transport.downstream.clone();
        let kind = match self.block_on(recv_kind_server(&upstream)) {
            Ok(Some(k)) => k,
            Ok(None) => {
                self.peer_disconnected = true;
                return Vec::new();
            }
            Err(e) => {
                warn!("worker recv_kind failed: {e}");
                std::thread::sleep(WORKER_BACKOFF);
                return Vec::new();
            }
        };
        let res = match kind {
            FrameKind::Reset => {
                self.runner.reset();
                self.last_rank_history.clear();
                self.last_rank_rng_seeded = false;
                match downstream.as_ref() {
                    Some(down) => self
                        .block_on(forward_reset(down))
                        .map_err(|e| format!("forward_reset: {e}")),
                    None => Ok(()),
                }
            }
            FrameKind::Forward => self.handle_forward(&upstream, downstream.as_ref()),
            other => Err(format!(
                "worker received unsupported frame {other:?} (MiniMax-M2 has no spec-decode batching)"
            )),
        };
        if let Err(e) = res {
            warn!("worker frame failed: {e}");
            std::thread::sleep(WORKER_BACKOFF);
        }
        Vec::new()
    }

    /// Handle one Forward frame on a worker rank: run my shells, then
    /// either (last rank) head + sample + Token upstream, or (middle rank)
    /// relay the hidden downstream and pass the returned Token back up.
    fn handle_forward(
        &mut self,
        upstream: &Arc<TokioMutex<ActivationServer>>,
        downstream: Option<&Arc<TokioMutex<ActivationClient>>>,
    ) -> Result<(), String> {
        let (past_seq_len, sampling_cfg, hidden_f32, _shape) = self
            .block_on(recv_forward_body_server(upstream))
            .map_err(|e| format!("recv_forward_body: {e}"))?;
        let pos = past_seq_len as usize;
        let hidden = self
            .runner
            .forward_layers(hidden_f32, pos)
            .map_err(|e| format!("forward: {e}"))?;
        if self.is_last() {
            let logits = self
                .runner
                .head_logits(&hidden)
                .map_err(|e| format!("head: {e}"))?;
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
            let down = downstream.ok_or("mid rank missing downstream")?;
            let h = self.runner.hidden_size() as u32;
            self.block_on(async {
                send_forward(down, past_seq_len, &sampling_cfg, &hidden, [1, 1, h])
                    .await
                    .map_err(|e| format!("send_forward: {e}"))?;
                match recv_kind_client(down).await {
                    Ok(Some(FrameKind::Token)) => {
                        let t = recv_token_body_client(down)
                            .await
                            .map_err(|e| format!("recv_token: {e}"))?;
                        send_token_upstream(upstream, t)
                            .await
                            .map_err(|e| format!("relay token: {e}"))?;
                        Ok(())
                    }
                    Ok(Some(other)) => Err(format!("mid rank expected Token, got {other:?}")),
                    Ok(None) => Err("downstream closed before Token".into()),
                    Err(e) => Err(format!("recv_kind: {e}")),
                }
            })
        }
    }
}

impl Engine for OvMoeEngine {
    fn warmup(&mut self) {
        // Only rank 0 (single-stage or the API rank) self-warms; worker
        // ranks are warmed by the first real generation's prefill, which
        // drives the whole ring in lockstep.
        if self.total == 1 {
            if let Some(tok) = self.tokenizer.as_ref() {
                let ids: Vec<u32> = tok
                    .encode("Hello", false)
                    .map(|e| e.get_ids().to_vec())
                    .unwrap_or_else(|_| vec![1]);
                let _ = self.runner.generate_argmax(&ids, 1);
                info!("warmup: generated 1 token (MiniMax-M2)");
            }
        }
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if self.rank != 0 {
            return Err(EngineError::InvalidConfig(
                "only rank 0 accepts tasks; worker ranks drive themselves from upstream frames"
                    .into(),
            ));
        }
        if self.pending.len() >= OV_MAX_PENDING {
            return Err(EngineError::QueueFull {
                queued: self.pending.len(),
                cap: OV_MAX_PENDING,
            });
        }
        self.pending.push_back(task);
        Ok(())
    }

    fn cancel(&mut self, task_id: &TaskId) {
        // Same as the dense sparse-MoE engine: drop queued tasks; an in-flight
        // monolithic generation runs to its step boundary.
        self.pending.retain(|t| &t.task_id != task_id);
    }

    fn step(&mut self) -> Vec<(TaskId, Chunk)> {
        if self.total <= 1 {
            return self.step_single_stage();
        }
        if self.rank == 0 {
            self.step_first()
        } else {
            self.step_worker()
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

    /// A latched worker disconnect must surface a connection-fatal Err to the
    /// relay loop — exactly once — so run_relay_loop exits and systemd
    /// rebuilds the stage instead of backing off Ok(empty) forever. The Err
    /// step() emits (EngineError::NotConnected) must be recognized as fatal.
    #[test]
    fn worker_disconnect_reports_fatal_once() {
        // Connected: nothing to report.
        assert!(!worker_should_report_disconnect(false, false));
        // First step after the link drops: report.
        assert!(worker_should_report_disconnect(true, false));
        // Already reported: suppressed (don't flood if re-polled).
        assert!(!worker_should_report_disconnect(true, true));
        // The Err step() returns on that one report is connection-fatal, so
        // run_relay_loop exits ConnectionFatal.
        assert!(EngineError::NotConnected.is_connection_fatal());
    }
}
