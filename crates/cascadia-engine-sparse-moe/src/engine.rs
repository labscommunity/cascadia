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
    forward_reset, recv_forward_batch_body_server, recv_forward_body_server, recv_key_body_server,
    recv_kind_client, recv_kind_server, recv_token_batch_body_client, recv_token_body_client,
    send_cache_prefix, send_forward, send_forward_batch, send_forward_batch_prefill,
    send_forward_batch_prefill_nosample, send_forward_nosample, send_forward_prefill, send_reset,
    send_restore_prefix, send_token_batch_upstream, send_token_upstream, FrameKind, StageTransport,
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
    // ---- glm5 tunables, config-threaded ----------------------------------
    // Each of these is otherwise read from the process environment deep inside
    // the engine. A host that loads the engine in-process cannot set those
    // (`set_var` is `unsafe` under edition 2024, and process-global), so they
    // travel as config instead. `None` means "use the env value", preserving
    // every existing env-driven deployment byte-for-byte.
    //
    // Deliberately absent: `profile`. `CASCADIA_GLM5_PROFILE` is read from a
    // `OnceLock` process global in the decode loop, which no config can reach;
    // it stays env-only rather than becoming a field that silently does nothing.
    /// Context budget. `None` → `CASCADIA_GLM5_MAX_SEQ`, else [`crate::glm::stage::GLM5_DEFAULT_MAX_SEQ`].
    pub max_seq: Option<usize>,
    /// KV-prefix-cache depth. `None` → `CASCADIA_GLM5_PREFIX_CACHE` (0 = off).
    pub prefix_cache_depth: Option<u32>,
    /// bf16 MLA latent / k_pe KV cache. `None` → `CASCADIA_GLM5_BF16_KV` (off).
    pub bf16_kv: Option<bool>,
    /// Force exact-f32 embed + lm_head. `None` → `CASCADIA_GLM5_F32_HEAD`.
    /// Modelled as the opt-OUT: bf16 head is the default-ON production path.
    pub f32_head: Option<bool>,
    /// Async lookahead expert prefetch. `None` → `CASCADIA_GLM5_LOOKAHEAD` (off).
    pub lookahead: Option<bool>,
    /// Expert storage mode (`"eager"` | `"mmap"`). `None` → `CASCADIA_GLM5_EXPERTS`.
    pub experts_mode: Option<String>,
    /// OV expert cache count backstop. `None` → `CASCADIA_GLM5_OV_CACHE` (64).
    pub ov_cache_entries: Option<u32>,
    /// OV expert cache byte budget in MiB (primary bound). `None` →
    /// `CASCADIA_GLM5_OV_CACHE_MB` (2048).
    pub ov_cache_mb: Option<u64>,
    /// Extra `(key, value)` OV plugin properties plumbed verbatim from the CLI.
    /// Applied via the shared `PluginConfig` to every OV-compiled IR on this
    /// rank: embedding, transformer shells, head, and — for `ov_ir`-format
    /// experts — each per-expert IR. Exempt: the per-layer router subgraphs
    /// (compiled with a fresh empty plugin) and `int4_bin` experts (no OV
    /// compile at all).
    pub ov_properties: Vec<(String, String)>,
    pub max_cached_experts: u32,
    /// Pipeline stage index (0-based).
    pub rank: u32,
    /// Number of pipeline stages.
    pub total: u32,
    /// If `Some(k)` and `k < manifest.top_k`, only the first k experts per
    /// token are dispatched per shell layer. See `docs/perf/A3_TOPK_REDUCTION.md`.
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
    /// quality.
    pub ffn_sparsity_threshold: f32,
    /// Issue #35 — use the AXPY-form down kernel from
    /// [`cascadia_int4_gemm::ffn_axpy`] in the sparse FFN path. Only
    /// active when [`Self::ffn_sparsity_threshold`] > 0. Per-expert
    /// transposed-and-requantized down weights are persisted to
    /// [`Self::ffn_axpy_cache_dir`] (default
    /// `<model_dir>/.cascadia_transposed_down_v1/`) and mmap'd at
    /// runtime — this is the on-disk-cache fix that addresses PR #43's
    /// initial mmap-page-cache-eviction regression.
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
            // All `None` = "defer to env", so constructing a config changes
            // nothing for existing env-driven deployments.
            max_seq: None,
            prefix_cache_depth: None,
            bf16_kv: None,
            f32_head: None,
            lookahead: None,
            experts_mode: None,
            ov_cache_entries: None,
            ov_cache_mb: None,
            ov_properties: Vec::new(),
            // 0 = unbounded (default); positive = LRU cap. The env var
            // `CASCADIA_MAX_EXPERTS_CACHED` overrides this if set.
            // See PowerInfer SmallThinker `MAX_N_CACHED` (MIT).
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
    /// roughly bounds the expert pool at 6.4 GiB.
    pub fn with_max_cached_experts(mut self, n: u32) -> Self {
        self.max_cached_experts = n;
        self
    }

    /// Set the magnitude threshold for two-phase Gate-first FFN
    /// sparsity. `0.0` = dense (default; output bit-identical to the
    /// pre-port path). `0.05`–`0.15` is the useful range for SwiGLU.
    /// See [`crate::runner::Runner::set_ffn_sparsity_threshold`].
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

/// dsv4 context-budget precedence: config → `env` (the caller's read of
/// `CASCADIA_DSV4_MAX_SEQ`) → [`crate::dsv4::stage::DSV4_DEFAULT_MAX_SEQ`].
/// `Some(0)`/absent config and a missing, unparseable, or zero env value all
/// fall through — `0` keeps meaning "fall through to the default", not
/// "zero context", for both sources alike.
///
/// Takes `env` as a parameter (rather than reading `std::env` itself) so
/// it's a pure function tests can exercise without mutating process-global
/// state.
pub fn resolve_dsv4_max_seq(config: Option<usize>, env: Option<&str>) -> usize {
    config
        .filter(|&n| n > 0)
        .or_else(|| {
            env.and_then(|s| s.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
        })
        .unwrap_or(crate::dsv4::stage::DSV4_DEFAULT_MAX_SEQ)
}

pub struct SparseMoEBuilder {
    pub config: SparseMoEBuilderConfig,
    runner: Option<Runner>,
    /// Set instead of `runner` when the manifest selects the OV-IR shell
    /// backend (MiniMax-M2). Single-stage only.
    ov_runner: Option<OvMoeRunner>,
    /// Set when the manifest's arch is "deepseek_v4" (Rust dsv4 shell +
    /// int4 experts; its manifest schema differs from [`Manifest`]).
    dsv4_runner: Option<crate::dsv4::stage::Dsv4Runner>,
    /// Set when the manifest's arch is "glm5" (Rust glm5 shell + int4 experts).
    glm_runner: Option<crate::glm::stage::GlmRunner>,
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
            dsv4_runner: None,
            glm_runner: None,
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
            // Cold pipeline start: a downstream rank only begins listening after
            // it has loaded its multi-GB slice, so this connect must out-wait the
            // slowest cold load. 60s was too short for the dsv4 ~30GB/rank load
            // off a cold page cache — the upstream gave up and the chain never
            // formed. Default 300s; override with CASCADIA_CONNECT_TIMEOUT_SECS.
            let connect_secs = std::env::var("CASCADIA_CONNECT_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(300);
            client
                .connect_with_timeout(std::time::Duration::from_secs(connect_secs))
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
        for (key, val) in &self.config.ov_properties {
            plugin = plugin.with(key, val);
        }

        // DeepSeek-V4 (dsv4) shell backend: its manifest is a different
        // schema, so peek the arch before the strict Manifest parse. Rust
        // shell + int4 experts; token-by-token pipeline like the M2 path
        // (ids never cross the wire — hash layers live on rank 0).
        let is_dsv4 = std::fs::read_to_string(self.config.model_dir.join("manifest.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .map(|v| v.get("arch").and_then(|a| a.as_str()) == Some("deepseek_v4"))
            .unwrap_or(false);
        if is_dsv4 {
            let total = self.config.total.max(1);
            let rank = self.config.rank.min(total - 1);
            // Context budget sizes the rope table + KV/compressed/indexer
            // caches (memory scales with it). Config-first, then
            // CASCADIA_DSV4_MAX_SEQ; fall back to the default on a missing,
            // unparseable, or zero value.
            let max_seq = resolve_dsv4_max_seq(
                self.config.max_seq,
                std::env::var("CASCADIA_DSV4_MAX_SEQ").ok().as_deref(),
            );
            if max_seq > 131072 {
                warn!(
                    max_seq,
                    "dsv4 max_seq is large; per-layer caches scale with it"
                );
            }
            let runner = crate::dsv4::stage::Dsv4Runner::load_staged_with_experts(
                &self.config.model_dir,
                max_seq,
                rank,
                total,
                shard.layer_start,
                shard.layer_end,
                self.config.experts_mode.as_deref(),
            )
            .map_err(|e| EngineError::Backend(format!("dsv4 load: {e}")))?;
            if rank == 0 {
                let tok_path = self.config.model_dir.join("tokenizer.json");
                if tok_path.exists() {
                    self.tokenizer =
                        Some(Tokenizer::from_file(&tok_path).map_err(|e| {
                            EngineError::Backend(format!("load tokenizer.json: {e}"))
                        })?);
                } else {
                    warn!(
                        "no tokenizer.json at {} — dsv4 engine will only accept pre-tokenized inputs",
                        tok_path.display()
                    );
                }
            }
            self.dsv4_runner = Some(runner);
            return Ok(Box::pin(stream::iter(vec![LoadProgress::message(
                "loaded DeepSeek-V4 stage (dsv4 Rust shell)",
            )])));
        }

        // glm5 shell backend: like dsv4, its manifest schema differs from the
        // strict Manifest, so peek the arch first. Rust MLA+DSA shell + int4
        // experts; 1x-width hidden on the same dist wire (no HC copies, no
        // input_ids past rank 0).
        let is_glm = std::fs::read_to_string(self.config.model_dir.join("manifest.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .map(|v| v.get("arch").and_then(|a| a.as_str()) == Some("glm5"))
            .unwrap_or(false);
        if is_glm {
            let total = self.config.total.max(1);
            let rank = self.config.rank.min(total - 1);
            let max_seq = self
                .config
                .max_seq
                .or_else(|| {
                    std::env::var("CASCADIA_GLM5_MAX_SEQ")
                        .ok()
                        .and_then(|s| s.trim().parse::<usize>().ok())
                        // `=0` keeps meaning "fall through to the default",
                        // not "zero context".
                        .filter(|&n| n > 0)
                })
                .unwrap_or(crate::glm::stage::GLM5_DEFAULT_MAX_SEQ);
            let runner = crate::glm::stage::GlmRunner::load_staged(
                &self.config.model_dir,
                max_seq,
                rank,
                total,
                shard.layer_start,
                shard.layer_end,
                crate::glm::stage::StageOpts::resolve(
                    self.config.f32_head,
                    self.config.bf16_kv,
                    self.config.experts_mode.clone(),
                    self.config.lookahead,
                    self.config.prefix_cache_depth,
                    self.config.ov_cache_entries,
                    self.config.ov_cache_mb,
                ),
            )
            .map_err(|e| EngineError::Backend(format!("glm5 load: {e}")))?;
            if rank == 0 {
                let tok_path = self.config.model_dir.join("tokenizer.json");
                if tok_path.exists() {
                    self.tokenizer =
                        Some(Tokenizer::from_file(&tok_path).map_err(|e| {
                            EngineError::Backend(format!("load tokenizer.json: {e}"))
                        })?);
                } else {
                    warn!(
                        "no tokenizer.json at {} — glm5 engine will only accept pre-tokenized inputs",
                        tok_path.display()
                    );
                }
            }
            self.glm_runner = Some(runner);
            return Ok(Box::pin(stream::iter(vec![LoadProgress::message(
                "loaded glm5 stage (glm5 Rust shell)",
            )])));
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
        if let Some(runner) = self.dsv4_runner {
            let total = self.config.total.max(1);
            let rank = self.config.rank.min(total - 1);
            if rank == 0 && self.tokenizer.is_none() {
                return Err(EngineError::Backend(
                    "tokenizer.json missing (required for the DeepSeek-V4 API rank)".into(),
                ));
            }
            let runtime_handle = tokio::runtime::Handle::try_current()
                .map_err(|_| EngineError::Backend("Builder::build outside tokio context".into()))?;
            info!(rank, total, "built DeepSeek-V4 dsv4 engine");
            return Ok(Box::new(Dsv4Engine::new(
                runner,
                self.tokenizer,
                self.transport,
                runtime_handle,
                rank,
                total,
                // dsv4 has no config-threaded prefix cache; `None` keeps its
                // existing env-only behaviour byte-for-byte.
                None,
            )));
        }
        if let Some(runner) = self.glm_runner {
            let total = self.config.total.max(1);
            let rank = self.config.rank.min(total - 1);
            if rank == 0 && self.tokenizer.is_none() {
                return Err(EngineError::Backend(
                    "tokenizer.json missing (required for the glm5 API rank)".into(),
                ));
            }
            let runtime_handle = tokio::runtime::Handle::try_current()
                .map_err(|_| EngineError::Backend("Builder::build outside tokio context".into()))?;
            info!(rank, total, "built glm5 engine");
            return Ok(Box::new(PipelineEngine::new(
                runner,
                self.tokenizer,
                self.transport,
                runtime_handle,
                rank,
                total,
                // Same value the per-rank SliceKvCache got, so the rank-0 index
                // and the per-rank caches stay in lockstep.
                self.config.prefix_cache_depth.map(|d| d as usize),
            )));
        }
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

/// Incremental UTF-8-safe streaming delta. `full` is the whole output decoded so
/// far (`decode(generated)`), `emitted` the byte length already streamed; return
/// the new bytes to stream and advance `emitted`. A trailing U+FFFD means a
/// multi-byte char is split across tokens (the tokenizer decoded a partial code
/// unit) — return nothing and hold `emitted` until the next token completes the
/// char, so no delta ever carries a lone replacement byte. Mirrors the OV
/// pipeline engines (qwen36) and assumes `decode` is prefix-stable across a
/// growing token list (true for these BPE tokenizers).
fn utf8_safe_delta(full: &str, emitted: &mut usize) -> String {
    if full.ends_with('\u{FFFD}') {
        return String::new();
    }
    let d = full.get(*emitted..).unwrap_or("").to_string();
    *emitted = full.len();
    d
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

/// Per-token reply budget: no widening. Mirrors `reply_deadline`'s body
/// (split out, same as `prefill_reply_budget` below, purely so a test can pin
/// "prefill exceeds per-token" against the real function instead of a
/// disconnected literal). Pure, for testing.
fn per_token_reply_budget(recv_timeout: std::time::Duration) -> std::time::Duration {
    recv_timeout
}

/// Safety margin the clamped prefill budget must stay under the frame-idle
/// ceiling by. `recv_token_reply`'s outer `tokio::time::timeout` and the
/// transport layer's OWN idle-ceiling timeout (inside
/// `recv_exact_frame_start`, guarding `recv_kind_client`) race the same
/// silent-peer wait — whichever fires first decides the outcome. If our
/// budget reaches or exceeds the ceiling, the transport-layer timeout can win
/// that race: it surfaces `TransportError::FrameIdleCeiling`, which
/// `ActivationClient::drop_connection_if_recv_fatal` treats as
/// connection-fatal and permanently drops the client socket (only
/// `Builder::connect` redials it) — the exact defect D failure mode, just
/// reached via a wide-open `PREFILL_REPLY_TIMEOUT_FACTOR` instead of an
/// unbounded recv. 30s is generous next to scheduling jitter and small next
/// to the ~900s default ceiling.
const PREFILL_BUDGET_CEILING_MARGIN: std::time::Duration = std::time::Duration::from_secs(30);

/// Widened deadline for a BATCHED prefill window's owed Token reply — see
/// `PipelineEngine::reply_deadline_prefill`. Split out as a free function
/// (rather than inline in the generic `impl<R: StagedRunner>` block) so the
/// arithmetic is unit testable without standing up a `StagedRunner`.
///
/// Clamped to stay `PREFILL_BUDGET_CEILING_MARGIN` under `frame_idle_ceiling`
/// when one is configured (`None` = disabled, nothing to stay under) — an
/// operator raising `CASCADIA_ACTIVATION_TIMEOUT_SECS` (a natural response to
/// prefills timing out on a slow fleet) widens `recv_timeout` and, with it,
/// this budget; left unclamped, a high enough setting reaches the frame-idle
/// ceiling and reinstates defect D (see `PREFILL_BUDGET_CEILING_MARGIN`). Pure,
/// for testing.
fn prefill_reply_budget(
    recv_timeout: std::time::Duration,
    frame_idle_ceiling: Option<std::time::Duration>,
) -> std::time::Duration {
    let widened = recv_timeout.saturating_mul(cascadia_transport::PREFILL_REPLY_TIMEOUT_FACTOR);
    match frame_idle_ceiling {
        Some(ceiling) => widened.min(ceiling.saturating_sub(PREFILL_BUDGET_CEILING_MARGIN)),
        None => widened,
    }
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
                let final_chunk =
                    Chunk::error(task.task_id.clone(), "engine has no tokenizer".to_string());
                return vec![(task.task_id, final_chunk)];
            }
        };
        let started = std::time::Instant::now();
        let prompt_ids: Vec<i64> = match tokenizer.encode(task.prompt.as_str(), true) {
            Ok(enc) => enc.get_ids().iter().map(|&u| u as i64).collect(),
            Err(e) => {
                warn!(task = %task.task_id, "tokenizer encode failed: {e}");
                let final_chunk = Chunk::error(
                    task.task_id.clone(),
                    format!("tokenizer encode failed: {e}"),
                );
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
                    let final_chunk =
                        Chunk::error(task.task_id.clone(), format!("spec-decode failed: {e}"));
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
                    let final_chunk =
                        Chunk::error(task.task_id.clone(), format!("runner failed: {e}"));
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
                let e = Chunk::error(task.task_id.clone(), "engine has no tokenizer".to_string());
                return vec![(task.task_id, e)];
            };
            match tok.encode(task.prompt.as_str(), true) {
                Ok(enc) => enc.get_ids().iter().map(|&u| u as i64).collect(),
                Err(e) => {
                    warn!(task = %task.task_id, "tokenizer encode failed: {e}");
                    return vec![(
                        task.task_id.clone(),
                        Chunk::error(task.task_id, format!("tokenizer encode failed: {e}")),
                    )];
                }
            }
        };

        let max_new = task.max_tokens.max(1) as usize;
        let sampling_cfg = sampling_from_task(&task);
        let downstream = match self.transport.downstream.clone() {
            Some(d) => d,
            None => {
                warn!("rank 0 has no downstream peer in multi-stage mode");
                return vec![(
                    task.task_id.clone(),
                    Chunk::error(
                        task.task_id,
                        "rank 0 has no downstream peer in multi-stage mode".to_string(),
                    ),
                )];
            }
        };

        // RESET downstream so workers clear their KV caches.
        self.runner.reset_kv();
        if let Err(e) = self.block_on(send_reset(&downstream)) {
            warn!(task = %task.task_id, "send_reset: {e}");
            return vec![(
                task.task_id.clone(),
                Chunk::error(task.task_id, format!("send_reset: {e}")),
            )];
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
                    return vec![(
                        task.task_id.clone(),
                        Chunk::error(
                            task.task_id,
                            format!("rank-0 spec-decode driver failed: {e}"),
                        ),
                    )];
                }
            }
        } else {
            match self.drive_generation_first(&prompt_ids, max_new, &sampling_cfg, &downstream) {
                Ok(g) => g,
                Err(e) => {
                    warn!(task = %task.task_id, "rank-0 driver failed: {e}");
                    return vec![(
                        task.task_id.clone(),
                        Chunk::error(task.task_id, format!("rank-0 driver failed: {e}")),
                    )];
                }
            }
        };

        let n_tokens = result_tokens.len() as u32;
        let ids_u32: Vec<u32> = result_tokens.iter().map(|&i| i as u32).collect();
        let Some(tokenizer) = self.tokenizer.as_ref() else {
            warn!("tokenizer disappeared mid-task");
            return vec![(
                task.task_id.clone(),
                Chunk::error(task.task_id, "tokenizer disappeared mid-task".to_string()),
            )];
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
        // Streamed-prefill sync window: every N intermediate tokens go as a
        // blocking ForwardNoSample (which acks) instead of a one-way
        // ForwardPrefill. This bounds the number of un-acked frames in flight
        // so rank 0's single final recv can't exceed the transport frame-idle
        // ceiling on a very long prompt, and it surfaces a mid-prefill failure
        // within one window rather than only at the very end.
        const PREFILL_SYNC_EVERY: usize = 256;
        for (i, &t) in prompt_ids.iter().enumerate() {
            history.push(t);
            if i + 1 == prompt_ids.len() {
                // Last prompt token: a sampling Forward; the token it returns
                // is the first generated token.
                let token_back = self
                    .forward_one_token_first(&history, cfg, downstream, true)
                    .map_err(|e| format!("prefill final step {i}: {e}"))?;
                if eos.contains(&token_back) {
                    return Ok(generated);
                }
                generated.push(token_back);
                history.push(token_back);
            } else if (i + 1) % PREFILL_SYNC_EVERY == 0 {
                // Periodic blocking sync (discards the dummy Token(-1) ack).
                self.forward_one_token_first(&history, cfg, downstream, false)
                    .map_err(|e| format!("prefill sync step {i}: {e}"))?;
            } else {
                // Intermediate prompt tokens stream one-way (no ack) so they
                // pipeline through the ranks — the compute overlaps instead of
                // one blocking round-trip each. See FrameKind::ForwardPrefill.
                self.forward_prefill_step_first(&history, cfg, downstream)
                    .map_err(|e| format!("prefill stream step {i}: {e}"))?;
            }
        }

        // Decode loop.
        for step_i in 1..max_new {
            let token_back = self
                .forward_one_token_first(&history, cfg, downstream, true)
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

    /// Rank-0 streamed-prefill step: embed + run my shells on the most
    /// recently appended history token, then fire the hidden downstream as a
    /// one-way `ForwardPrefill` and return WITHOUT waiting for a reply. Used
    /// for every prompt token except the last, so the prompt pipelines through
    /// the ranks instead of one blocking round-trip per token.
    fn forward_prefill_step_first(
        &mut self,
        history: &[i64],
        cfg: &crate::sampling::SamplingConfig,
        downstream: &Arc<TokioMutex<ActivationClient>>,
    ) -> Result<(), String> {
        let hidden = self.runner.manifest.hidden_size as usize;
        let past_seq_len: u32 = history
            .len()
            .checked_sub(1)
            .ok_or_else(|| "forward_prefill_step_first: empty history".to_string())?
            as u32;
        let last_id = *history
            .last()
            .ok_or_else(|| "forward_prefill_step_first: empty history".to_string())?;
        let h_tail = self
            .runner
            .forward_layer0_step(last_id)
            .map_err(|e| format!("layer0_step: {e}"))?;
        let h_after_shells = self
            .runner
            .forward_shells(&h_tail, &[1, 1, hidden], past_seq_len as usize)
            .map_err(|e| format!("forward_shells: {e}"))?;
        self.block_on(send_forward_prefill(
            downstream,
            past_seq_len,
            cfg,
            &h_after_shells,
            [1, 1, hidden as u32],
        ))
        .map_err(|e| format!("send_forward_prefill: {e}"))
    }

    /// Body of the `ForwardPrefill` frame handler (extracted so the caller can
    /// latch a fatal disconnect on any error — a one-way frame has no ack path,
    /// so a recoverable failure would silently skip a KV slot). Advances KV for
    /// the streamed prompt token; mid ranks relay it downstream one-way, the
    /// last rank stops here (no head / sample / record).
    fn handle_forward_prefill(
        &mut self,
        upstream: &Arc<TokioMutex<ActivationServer>>,
        downstream: Option<Arc<TokioMutex<ActivationClient>>>,
    ) -> Result<(), String> {
        let (past_seq_len, sampling_cfg, hidden_f32, in_shape) = self
            .block_on(recv_forward_body_server(upstream))
            .map_err(|e| format!("recv_forward(prefill): {e}"))?;
        let hidden = self.runner.manifest.hidden_size as usize;
        if in_shape[0] != 1 || in_shape[1] != 1 || in_shape[2] as usize != hidden {
            return Err(format!(
                "prefill shape unexpected {:?} vs hidden {}",
                in_shape, hidden
            ));
        }
        self.maybe_rewind_to(past_seq_len as usize);
        let h_after = self
            .runner
            .forward_shells(&hidden_f32, &[1, 1, hidden], past_seq_len as usize)
            .map_err(|e| format!("forward_shells(prefill): {e}"))?;
        if !self.is_last() {
            let down = downstream.ok_or_else(|| "mid rank missing downstream".to_string())?;
            self.block_on(send_forward_prefill(
                &down,
                past_seq_len,
                &sampling_cfg,
                &h_after,
                [1, 1, hidden as u32],
            ))
            .map_err(|e| format!("relay forward_prefill: {e}"))?;
        }
        Ok(())
    }

    /// Run one (prefill or decode) step on rank 0: embed via layer 0,
    /// run my shells, send hidden state downstream, receive the
    /// sampled token back along the same socket. `sample_back = false`
    /// sends ForwardNoSample (prefill-intermediate): downstream state
    /// still advances, but the last rank skips head/sample/record and
    /// acks with a dummy Token(-1).
    fn forward_one_token_first(
        &mut self,
        history: &[i64],
        cfg: &crate::sampling::SamplingConfig,
        downstream: &Arc<TokioMutex<ActivationClient>>,
        sample_back: bool,
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
        let wire_t0 = Instant::now();
        let result = self.block_on(async {
            if sample_back {
                send_forward(
                    downstream,
                    past_seq_len,
                    cfg,
                    &h_after_shells,
                    [1, 1, hidden as u32],
                )
                .await
                .map_err(|e| format!("send_forward: {e}"))?;
            } else {
                send_forward_nosample(
                    downstream,
                    past_seq_len,
                    cfg,
                    &h_after_shells,
                    [1, 1, hidden as u32],
                )
                .await
                .map_err(|e| format!("send_forward_nosample: {e}"))?;
            }
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
            // Same as the non-spec driver: only the LAST prompt step
            // samples; intermediates go as ForwardNoSample so their
            // discarded tokens never pollute the last rank's history.
            let token_back = self
                .forward_one_token_first(&history, cfg, downstream, i + 1 == prompt_ids.len())
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
                    .forward_one_token_first(&history, cfg, downstream, true)
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
                self.forward_one_token_first(&history, cfg, downstream, true)
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
            FrameKind::ForwardPrefill => {
                // A streamed prefill frame is ONE-WAY (no ack), so a failure
                // here has no return path: left recoverable it would be
                // silently skipped, leaving a KV hole (wrong output for the
                // whole generation) or hanging rank 0's final recv. Make any
                // error fatal — latch peer_disconnected so the stage tears down
                // and rebuilds, and rank 0's final Forward surfaces the failure
                // instead of the run completing with corrupt state.
                let res = self.handle_forward_prefill(upstream, downstream);
                if res.is_err() {
                    self.peer_disconnected = true;
                }
                res
            }
            k @ (FrameKind::Forward | FrameKind::ForwardNoSample) => {
                // ForwardNoSample (prefill-intermediate): identical body and
                // state advance, but the last rank skips head+sample+history
                // and acks with a dummy Token(-1) — keeps phantom prefill
                // samples out of the penalty history.
                let sample = matches!(k, FrameKind::Forward);
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
                    if !sample {
                        return self
                            .block_on(send_token_upstream(upstream, -1))
                            .map_err(|e| format!("send_token (nosample ack): {e}"));
                    }
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
                        if sample {
                            send_forward(
                                &down,
                                past_seq_len,
                                &sampling_cfg,
                                &h_after,
                                [1, 1, hidden as u32],
                            )
                            .await
                            .map_err(|e| format!("send_forward: {e}"))?;
                        } else {
                            send_forward_nosample(
                                &down,
                                past_seq_len,
                                &sampling_cfg,
                                &h_after,
                                [1, 1, hidden as u32],
                            )
                            .await
                            .map_err(|e| format!("send_forward_nosample: {e}"))?;
                        }
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
            FrameKind::ForwardBatchPrefill | FrameKind::ForwardBatchPrefillNoSample => {
                Err(format!(
                    "rank {} received ForwardBatchPrefill (Rust-shell prefill batching only)",
                    self.rank
                ))
            }
            FrameKind::RestorePrefix | FrameKind::CachePrefix => Err(format!(
                "rank {} received a KV-prefix-cache frame (glm5 pipeline engine only)",
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
    /// of latency per K verifies. At a measured 22 ms cross-host RTT,
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
            let e = Chunk::error(task.task_id.clone(), "engine has no tokenizer".to_string());
            return vec![(task.task_id, e)];
        };
        let started = Instant::now();
        let prompt_ids: Vec<u32> = match tok.encode(task.prompt.as_str(), true) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                warn!(task = %task.task_id, "tokenizer encode failed: {e}");
                return vec![(
                    task.task_id.clone(),
                    Chunk::error(task.task_id, format!("tokenizer encode failed: {e}")),
                )];
            }
        };
        let max_new = task.max_tokens.max(1) as usize;
        let sampling_cfg = sampling_from_task(&task);
        let generated = match self.runner.generate(&prompt_ids, max_new, &sampling_cfg) {
            Ok(g) => g,
            Err(e) => {
                warn!(task = %task.task_id, "MiniMax-M2 generate failed: {e}");
                return vec![(
                    task.task_id.clone(),
                    Chunk::error(task.task_id, format!("MiniMax-M2 generate failed: {e}")),
                )];
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
    /// `sample_back = false` sends ForwardNoSample (prefill-intermediate):
    /// the last rank advances state but skips head/sample/record so
    /// discarded prefill tokens never pollute its penalty history.
    fn forward_one_token_first(
        &mut self,
        token: u32,
        pos: usize,
        cfg: &crate::sampling::SamplingConfig,
        downstream: &Arc<TokioMutex<ActivationClient>>,
        sample_back: bool,
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
            if sample_back {
                send_forward(downstream, pos as u32, cfg, &hidden, [1, 1, h])
                    .await
                    .map_err(|e| format!("send_forward: {e}"))?;
            } else {
                send_forward_nosample(downstream, pos as u32, cfg, &hidden, [1, 1, h])
                    .await
                    .map_err(|e| format!("send_forward_nosample: {e}"))?;
            }
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
            let e = Chunk::error(id.clone(), "engine has no tokenizer".to_string());
            return vec![(id, e)];
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
                return vec![(
                    id.clone(),
                    Chunk::error(id, format!("tokenizer encode failed: {e}")),
                )];
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
        // `OvMoeRunner::generate_timed`. Intermediate steps go as
        // ForwardNoSample so their discarded tokens never enter the last
        // rank's penalty history (see FrameKind::ForwardNoSample).
        let mut pos = 0usize;
        let mut next: i64 = -1;
        let n_prompt = prompt_ids.len();
        for (i, &t) in prompt_ids.iter().enumerate() {
            let sample_back = i + 1 == n_prompt;
            match self.forward_one_token_first(t, pos, &cfg, &downstream, sample_back) {
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
            match self.forward_one_token_first(next_u, pos, &cfg, &downstream, true) {
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
            FrameKind::Forward => self.handle_forward(&upstream, downstream.as_ref(), true),
            FrameKind::ForwardNoSample => {
                self.handle_forward(&upstream, downstream.as_ref(), false)
            }
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
    /// `sample = false` (ForwardNoSample, prefill-intermediate): still run
    /// my shells (state must advance), but the last rank skips
    /// head+sample+history and returns a dummy Token(-1).
    fn handle_forward(
        &mut self,
        upstream: &Arc<TokioMutex<ActivationServer>>,
        downstream: Option<&Arc<TokioMutex<ActivationClient>>>,
        sample: bool,
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
            if !sample {
                return self
                    .block_on(send_token_upstream(upstream, -1))
                    .map_err(|e| format!("send_token (nosample ack): {e}"));
            }
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
                if sample {
                    send_forward(down, past_seq_len, &sampling_cfg, &hidden, [1, 1, h])
                        .await
                        .map_err(|e| format!("send_forward: {e}"))?;
                } else {
                    send_forward_nosample(down, past_seq_len, &sampling_cfg, &hidden, [1, 1, h])
                        .await
                        .map_err(|e| format!("send_forward_nosample: {e}"))?;
                }
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

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        // Behavior-preserving migration to EngineResult: step_single_stage /
        // step_first / step_worker handle their own errors terminally (the
        // driver emits a final-marker chunk; the worker latches), so there is
        // nothing to surface as Err here.
        if self.total <= 1 {
            return Ok(self.step_single_stage());
        }
        Ok(if self.rank == 0 {
            self.step_first()
        } else {
            self.step_worker()
        })
    }
}

// ---------------------------------------------------------------------
// DeepSeek-V4 (dsv4) engine — same pipeline shape as the MiniMax-M2
// engine above: rank 0 tokenizes + drives one wire round-trip per token
// (prompt included; validated equivalent to batch prefill), workers run
// their layer slice, the last rank samples and returns the token. The
// hidden on the wire is the flattened HC copies (hc * dim f32); token
// ids never leave rank 0 (hash-gate layers are always in stage 0).
// ---------------------------------------------------------------------

use crate::staged::StagedRunner;

/// dsv4's concrete engine (historical name kept for its tests + build()).
pub type Dsv4Engine = PipelineEngine<crate::dsv4::stage::Dsv4Runner>;

/// Arch-agnostic pipeline engine over a [`StagedRunner`]: rank 0 tokenizes +
/// drives one wire round-trip per token (prompt included), workers run their
/// layer slice, the last rank samples. Shared by the dsv4 and glm5 Rust shells
/// (minimax keeps its own engine — different disconnect semantics).
pub struct PipelineEngine<R: StagedRunner> {
    runner: R,
    tokenizer: Option<Tokenizer>,
    pending: VecDeque<GenerationTask>,
    transport: StageTransport,
    runtime_handle: tokio::runtime::Handle,
    rank: u32,
    total: u32,
    peer_disconnected: bool,
    disconnect_reported: bool,
    last_rank_history: Vec<i64>,
    last_rank_rng: u64,
    last_rank_rng_seeded: bool,
    /// Rank-0 KV-prefix index: `(cached prompt prefix tokens, its key)`.
    /// Insertion-ordered LRU capped at `prefix_cap`, in lockstep with every
    /// rank's `SliceKvCache`, so a rank-0 index hit is guaranteed still cached on
    /// all ranks. Unused unless the runner enables prefix caching.
    prefix_index: Vec<(Vec<u32>, u64)>,
    prefix_next_key: u64,
    prefix_cap: usize,
    /// In-flight streamed generation for rank 0. `None` between tasks; `Some`
    /// while a task is decoding token-by-token (one `Chunk::token` per `step`).
    /// Rank 0 serves ONE task at a time (as the monolithic driver did): a new
    /// task is popped from `pending` only when this is `None`.
    active: Option<PipeActive>,
}

/// Rank-0 per-token streaming state: everything the decode loop threaded as
/// locals in the old monolithic `step_first`, persisted across `step()` calls
/// so each call emits exactly one token. `next` is the token to emit on the
/// NEXT `step` (the prefill sample seeds it); `pos` is the KV position the last
/// forwarded token occupied; `emitted` is the byte length of the decoded output
/// already streamed (for UTF-8-safe delta slicing).
struct PipeActive {
    id: TaskId,
    downstream: Arc<TokioMutex<ActivationClient>>,
    cfg: crate::sampling::SamplingConfig,
    eos: Vec<u32>,
    max_new: usize,
    max_seq: usize,
    started: Instant,
    /// The prompt tokens actually prefilled (`prompt_ids[..n_prefill]`), kept
    /// for the post-decode prefix-cache key.
    prompt_ids: Vec<u32>,
    n_prefill: usize,
    next: i64,
    pos: usize,
    generated: Vec<u32>,
    emitted: usize,
    hit_context_cap: bool,
}

impl<R: StagedRunner> PipelineEngine<R> {
    /// `prefix_cap`: rank-0 KV-prefix-cache depth. `None` defers to
    /// `CASCADIA_GLM5_PREFIX_CACHE`, preserving the env-only behaviour. This
    /// index must stay in lockstep with every rank's `SliceKvCache`, so the
    /// value threaded here is the same one handed to `StageOpts`.
    fn new(
        runner: R,
        tokenizer: Option<Tokenizer>,
        transport: StageTransport,
        runtime_handle: tokio::runtime::Handle,
        rank: u32,
        total: u32,
        prefix_cap: Option<usize>,
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
            disconnect_reported: false,
            last_rank_history: Vec::new(),
            last_rank_rng: 0,
            last_rank_rng_seeded: false,
            prefix_index: Vec::new(),
            prefix_next_key: 0,
            prefix_cap: prefix_cap
                .or_else(|| {
                    std::env::var("CASCADIA_GLM5_PREFIX_CACHE")
                        .ok()
                        .and_then(|s| s.trim().parse::<usize>().ok())
                })
                .unwrap_or(0),
            active: None,
        }
    }

    /// Longest cached prompt prefix that is a STRICT prefix of `prompt` (length
    /// `< prompt.len()`, so at least the last token is always prefilled to
    /// produce logits). Returns `(key, len)`.
    fn prefix_reuse(&self, prompt: &[u32]) -> Option<(u64, usize)> {
        let mut best: Option<(u64, usize)> = None;
        for (toks, key) in &self.prefix_index {
            let k = toks.len();
            if k < prompt.len() && prompt[..k] == toks[..] && best.map_or(true, |(_, b)| k > b) {
                best = Some((*key, k));
            }
        }
        best
    }

    /// Record `prompt`'s KV (LRU-evicting in lockstep with the ranks'
    /// `SliceKvCache`). Returns the key to cache under — the existing one when
    /// `prompt` is already indexed, a fresh one otherwise.
    ///
    /// Reusing the key on a repeat is what keeps the lockstep claim true. This
    /// index dedups by token sequence; every rank's `SliceKvCache` dedups by
    /// key. Minting a fresh key for an already-indexed sequence dropped the old
    /// index entry without evicting anything, while the ranks saw an unfamiliar
    /// key, appended it, and evicted their oldest — so the two sides drifted
    /// apart and `prefix_reuse` could hand out a key no rank still held. With
    /// the key reused, `cache_prefix` hits the key-dedup branch in
    /// `SliceKvCache::insert` and both sides refresh the same entry to MRU.
    fn prefix_remember(&mut self, prompt: &[u32]) -> u64 {
        if let Some(i) = self.prefix_index.iter().position(|(t, _)| t == prompt) {
            // Refresh to MRU without changing the set; no trim can be due.
            let entry = self.prefix_index.remove(i);
            let key = entry.1;
            self.prefix_index.push(entry);
            return key;
        }
        let key = self.prefix_next_key;
        self.prefix_next_key += 1;
        self.prefix_index.push((prompt.to_vec(), key));
        while self.prefix_index.len() > self.prefix_cap {
            self.prefix_index.remove(0);
        }
        key
    }

    fn block_on<F: std::future::Future>(&self, fut: F) -> F::Output {
        cascadia_runner::run_async(&self.runtime_handle, fut)
    }

    fn is_last(&self) -> bool {
        self.transport.is_last()
    }

    fn step_single_stage(&mut self) -> Vec<(TaskId, Chunk)> {
        let task = match self.pending.pop_front() {
            Some(t) => t,
            None => return Vec::new(),
        };
        let Some(tok) = self.tokenizer.as_ref() else {
            warn!(task = %task.task_id, "dsv4 engine has no tokenizer");
            let e = Chunk::error(task.task_id.clone(), "engine has no tokenizer".to_string());
            return vec![(task.task_id, e)];
        };
        let started = Instant::now();
        let prompt_ids: Vec<u32> = match tok.encode(task.prompt.as_str(), true) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                warn!(task = %task.task_id, "tokenizer encode failed: {e}");
                return vec![(
                    task.task_id.clone(),
                    Chunk::error(task.task_id, format!("tokenizer encode failed: {e}")),
                )];
            }
        };
        if prompt_ids.is_empty() {
            return vec![(task.task_id.clone(), Chunk::final_marker(task.task_id, ""))];
        }
        let max_new = task.max_tokens.max(1) as usize;
        let sampling_cfg = sampling_from_task(&task);
        // Mirror step_first: an over-budget prompt is truncated (head kept, tail
        // dropped) inside generate — never silently, so log it loudly.
        let max_seq = self.runner.max_seq();
        if prompt_ids.len() > max_seq {
            warn!(
                task = %task.task_id,
                prompt_tokens = prompt_ids.len(),
                context_budget = max_seq,
                dropped = prompt_ids.len() - max_seq,
                "prompt exceeds context budget; dropping the tail (newest tokens) — \
                 the response answers only the first {max_seq} tokens"
            );
        }
        let (generated, hit_context_cap) =
            self.runner
                .generate_reason(&prompt_ids, max_new, &sampling_cfg);
        let n_tokens = generated.len() as u32;
        let text = tok.decode(&generated, true).unwrap_or_default();
        let elapsed = started.elapsed().as_secs_f64();
        info!(
            task = %task.task_id,
            tokens = n_tokens,
            elapsed_s = elapsed,
            tok_s = if elapsed > 0.0 { n_tokens as f64 / elapsed } else { 0.0 },
            "task done (dsv4 single-stage)"
        );
        let mut chunk = Chunk::final_marker(task.task_id.clone(), text);
        chunk.n_tokens = Some(n_tokens);
        // Same omission as the pipeline path: the API reads this and reported 0.
        // `prompt_ids` is already truncated to the context budget above, so its
        // length is the count actually processed.
        chunk.prompt_tokens = Some(prompt_ids.len() as u32);
        chunk.finish_reason = Some(if hit_context_cap {
            FinishReason::Length
        } else {
            finish_reason_for(n_tokens as usize, max_new)
        });
        vec![(task.task_id.clone(), chunk)]
    }

    /// Rank-0 driver: embed + my layers (with the token id for the hash
    /// gates), ship hidden downstream, await the sampled token back.
    /// `sample_back = false` sends ForwardNoSample (prefill-intermediate):
    /// the last rank advances state but must NOT head/sample/record, so
    /// discarded prefill tokens never pollute its penalty history.
    fn forward_one_token_first(
        &mut self,
        token: u32,
        pos: usize,
        cfg: &crate::sampling::SamplingConfig,
        downstream: &Arc<TokioMutex<ActivationClient>>,
        sample_back: bool,
        reply_deadline: std::time::Duration,
    ) -> Result<i64, String> {
        let hidden = self.runner.embed_token(token);
        let hidden = self.runner.forward_layers(hidden, pos, Some(token));
        let h = self.runner.hidden_size() as u32;
        self.block_on(async {
            if sample_back {
                send_forward(downstream, pos as u32, cfg, &hidden, [1, 1, h])
                    .await
                    .map_err(|e| format!("send_forward: {e}"))?;
            } else {
                send_forward_nosample(downstream, pos as u32, cfg, &hidden, [1, 1, h])
                    .await
                    .map_err(|e| format!("send_forward_nosample: {e}"))?;
            }
            // The Token reply is owed to a frame we just sent — bound the wait on
            // a strict deadline (never the idle ceiling) so a downstream that dies
            // mid-request surfaces as a fast error instead of wedging the task and
            // every upstream rank; see [`crate::dist::recv_token_reply`].
            crate::dist::recv_token_reply(downstream, reply_deadline).await
        })
    }

    /// Strict deadline for the Token reply owed to one forwarded token.
    ///
    /// dsv4 forwards the prompt one token at a time (no batched-prefill frame),
    /// so a prefill reply and a decode reply carry the same single-token compute
    /// through the chain — there is no whole-prompt tail to wait out, and hence
    /// no reason to apply the batched `PREFILL_REPLY_TIMEOUT_FACTOR` widening
    /// here. A healthy per-token reply lands well inside the observed TTFT
    /// (~seconds); `recv_timeout` (default 60 s, env/config tunable) is a wide
    /// margin over that yet short enough that a black-holed peer — one that dies
    /// without a clean FIN/RST, the failure mode seen on this fleet — surfaces
    /// as a fast error inside a client's request window instead of a wedge.
    fn reply_deadline() -> std::time::Duration {
        per_token_reply_budget(cascadia_transport::recv_timeout())
    }

    /// Strict deadline for the Token reply owed to a BATCHED prefill window
    /// (`forward_prompt_batch_first` / the mid-rank relay in
    /// `handle_forward_batch_prefill`): unlike dsv4's per-token forwarding,
    /// every remaining downstream stage must run the WHOLE window (up to
    /// `MAX_BATCH_COUNT` rows) through compute before replying, so the
    /// per-token `reply_deadline` above is far too tight — the same widening
    /// `recv_token_from_downstream` in cascadia-engine-openvino applies for
    /// its chunked-prefill path, so the two crates agree on what "a prefill
    /// budget" means. Clamped under the frame-idle ceiling — see
    /// `prefill_reply_budget`.
    fn reply_deadline_prefill() -> std::time::Duration {
        prefill_reply_budget(
            cascadia_transport::recv_timeout(),
            cascadia_transport::frame_idle_ceiling(),
        )
    }

    fn step_first(&mut self) -> Vec<(TaskId, Chunk)> {
        // Token-streaming state machine. When idle, `begin_generation` pops the
        // next task, runs prefill (batched or per-token), and stashes the
        // in-flight state in `self.active`; it returns `Some(chunks)` only on a
        // terminal outcome (no pending task -> empty, or an encode / reset /
        // prefill failure -> a final / error chunk) and leaves `active` unset.
        // On success it returns `None` and we fall through to `decode_step`,
        // which emits the first token this same call. Every subsequent `step`
        // decodes and emits exactly one token, finalizing (post-decode prefix
        // cache + final marker) when the task ends. This mirrors the OV
        // pipeline engines (qwen36) so glm5 + dsv4 stream per-token instead of
        // returning the whole completion on one final marker — the client sees
        // the first token after TTFT, not after the full (slow, I/O-bound)
        // decode.
        if self.active.is_none() {
            if let Some(terminal) = self.begin_generation() {
                return terminal;
            }
        }
        self.decode_step()
    }

    /// Pop + prefill the next task, seeding `self.active` with the first token
    /// to emit. Returns `Some(chunks)` on a terminal outcome (no pending task,
    /// or a setup failure) with `active` left `None`; `None` once streaming has
    /// begun (`active` is `Some` and `active.next` is the first generated
    /// token). This is the old monolithic `step_first` up to — but not
    /// including — the decode loop, which is now `decode_step`.
    fn begin_generation(&mut self) -> Option<Vec<(TaskId, Chunk)>> {
        let task = match self.pending.pop_front() {
            Some(t) => t,
            None => return Some(Vec::new()),
        };
        let id = task.task_id.clone();
        let Some(downstream) = self.transport.downstream.clone() else {
            warn!(task = %id, "rank 0 has no downstream; cannot drive pipeline");
            return Some(vec![(
                id.clone(),
                Chunk::error(id, "rank 0 missing downstream"),
            )]);
        };
        if self.tokenizer.is_none() {
            warn!(task = %id, "dsv4 rank 0 has no tokenizer");
            return Some(vec![(
                id.clone(),
                Chunk::error(id, "dsv4 rank 0 has no tokenizer".to_string()),
            )]);
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
                return Some(vec![(
                    id.clone(),
                    Chunk::error(id, format!("tokenizer encode failed: {e}")),
                )]);
            }
        };
        if prompt_ids.is_empty() {
            return Some(vec![(id.clone(), Chunk::final_marker(id, ""))]);
        }
        let max_new = task.max_tokens.max(1) as usize;
        let cfg = sampling_from_task(&task);
        let eos = self.runner.eos_token_ids().to_vec();

        self.runner.reset();
        if let Err(e) = self.block_on(send_reset(&downstream)) {
            warn!(task = %id, "send_reset failed: {e}");
            return Some(vec![(id.clone(), Chunk::error(id, format!("reset: {e}")))]);
        }

        // Prefill: only the LAST prompt step asks the last rank to sample
        // (that sample is the first generated token). Intermediate steps go
        // as ForwardNoSample so their discarded tokens never enter the last
        // rank's penalty history — recording them poisoned the first real
        // token (rep-penalty demoted the true continuation) and was the
        // 4-node "deterministic garbage while single-stage is coherent" bug.
        let max_seq = self.runner.max_seq();
        let mut pos = 0usize;
        let mut next: i64 = -1;
        let n_prompt = prompt_ids.len();
        // Truncate an over-long prompt to the context budget; the last forwarded
        // token is the one that asks the last rank to sample (its sampled result
        // is the first generated token). take() keeps the HEAD, so truncation
        // drops the tail — i.e. the newest turn. Never do this silently: a
        // caller whose prompt overflowed the budget is served an answer to a
        // prompt it did not send, so log loudly with the numbers. (A future
        // improvement is a budget-aware trim or an explicit 4xx over budget.)
        let n_prefill = n_prompt.min(max_seq);
        if n_prompt > max_seq {
            warn!(
                task = %id,
                prompt_tokens = n_prompt,
                context_budget = max_seq,
                dropped = n_prompt - max_seq,
                "prompt exceeds context budget; dropping the tail (newest tokens) — \
                 the response answers only the first {max_seq} tokens"
            );
        }
        if self.runner.supports_batched_prefill() && n_prefill > 0 {
            // KV-prefix cache (opt-in): restore the longest cached prefix on every
            // rank, then batch-prefill only the suffix (base = reused length).
            // Skipping the shared prefix skips re-streaming its experts — ~78% of
            // decode time on this NVMe-bound workload. `reuse == 0` (disabled or a
            // miss) is the original full-prefill behaviour.
            let reuse = if self.runner.prefix_cache_enabled() {
                match self.prefix_reuse(&prompt_ids[..n_prefill]) {
                    // Only reuse when this rank actually restored, and to the
                    // length the index promised. A miss means the index and the
                    // slice caches disagree; a different length means the cached
                    // snapshot does not describe `k` tokens. Either way the peers
                    // have not been told to restore yet, so falling back to a full
                    // prefill is safe — but `restore_prefix` sets `pos` on a hit,
                    // so undo it before prefilling from base 0.
                    Some((key, k)) if self.runner.restore_prefix(key) == Some(k) => {
                        if let Err(e) = self.block_on(send_restore_prefix(&downstream, key)) {
                            warn!(task = %id, "send_restore_prefix failed: {e}");
                            return Some(vec![(
                                id.clone(),
                                Chunk::error(id, format!("restore_prefix: {e}")),
                            )]);
                        }
                        info!(
                            task = %id,
                            reused = k,
                            prompt = n_prefill,
                            suffix = n_prefill - k,
                            "glm5 kv-prefix cache HIT: prefilling suffix only"
                        );
                        k
                    }
                    Some((key, k)) => {
                        warn!(
                            task = %id,
                            expected = k,
                            "glm5 kv-prefix cache index/slice disagreement; full prefill"
                        );
                        // Drop the stale entry so the next turn of this
                        // conversation does not re-miss on the same key.
                        self.prefix_index.retain(|(_, k2)| *k2 != key);
                        self.runner.reset();
                        0
                    }
                    None => 0,
                }
            } else {
                0
            };
            // Prefill the suffix in windows of at most MAX_BATCH_COUNT rows (the
            // wire caps one ForwardBatch at MAX_BATCH_COUNT, since it was built for
            // spec-decode K). KV accumulates across windows exactly as it would
            // token-by-token; every window but the last is NoSample, and only the
            // final ForwardBatchPrefill samples the first generated token — so no
            // mid-prompt row is recorded into the last rank's rep-penalty history.
            let suffix = &prompt_ids[reuse..n_prefill];
            let win = crate::dist::MAX_BATCH_COUNT as usize;
            let n_windows = suffix.len().div_ceil(win);
            let mut wbase = reuse;
            for (wi, chunk) in suffix.chunks(win).enumerate() {
                let is_last = wi + 1 == n_windows;
                match self.forward_prompt_batch_first(chunk, wbase, &cfg, &downstream, is_last) {
                    Ok(tok_back) => {
                        if is_last {
                            next = tok_back;
                        }
                    }
                    Err(e) => {
                        warn!(task = %id, "batched prefill window {wi} failed: {e}");
                        return Some(vec![(id.clone(), Chunk::error(id, e))]);
                    }
                }
                wbase += chunk.len();
            }
            pos = n_prefill;
        } else {
            for (i, &t) in prompt_ids.iter().take(n_prefill).enumerate() {
                let sample_back = i + 1 == n_prefill;
                match self.forward_one_token_first(
                    t,
                    pos,
                    &cfg,
                    &downstream,
                    sample_back,
                    Self::reply_deadline(),
                ) {
                    Ok(tok_back) => next = tok_back,
                    Err(e) => {
                        warn!(task = %id, "prefill forward failed: {e}");
                        return Some(vec![(id.clone(), Chunk::error(id, e))]);
                    }
                }
                pos += 1;
            }
        }

        // Prefill produced the first token (`next`) at KV position `pos`. Stash
        // the in-flight state; `decode_step` emits `next` on the next (this)
        // call and drives one token per `step` from here.
        self.active = Some(PipeActive {
            id,
            downstream,
            cfg,
            eos,
            max_new,
            max_seq,
            started,
            prompt_ids: prompt_ids[..n_prefill].to_vec(),
            n_prefill,
            next,
            pos,
            generated: Vec::with_capacity(max_new),
            emitted: 0,
            hit_context_cap: false,
        });
        None
    }

    /// Emit one token from the in-flight task, or finalize when it ends. This is
    /// one iteration of the old monolithic decode loop: push `next` to
    /// `generated`, stream its UTF-8-safe delta, then — in the SAME post-push
    /// order the loop used — stop on max_new / EOS / context-cap, or forward the
    /// token to obtain the next one. Preserving that order keeps the emitted
    /// token set, text, count, and finish_reason bit-identical to the batch
    /// driver. A finalizing step returns `[token, final_marker]`; the runner
    /// buffers the marker and drains it on the next poll (see
    /// `cascadia_runner::ChunkStream::poll_next`).
    fn decode_step(&mut self) -> Vec<(TaskId, Chunk)> {
        let id = self.active.as_ref().unwrap().id.clone();
        let next = self.active.as_ref().unwrap().next;
        let next_u = next as u32;

        // Push first (mirrors the old loop: the token sampled from the last
        // in-range position is always emitted, even when it triggers the stop).
        self.active.as_mut().unwrap().generated.push(next_u);

        // UTF-8-safe delta: decode the whole output so far and stream the bytes
        // past what we've already emitted. A trailing U+FFFD means a multi-byte
        // char is split across tokens — hold the delta until the next token
        // completes it (matches qwen36). The token is still counted (n_tokens=1)
        // so the completion total equals `generated.len()` as the batch driver
        // reported it.
        let full = self
            .tokenizer
            .as_ref()
            .unwrap()
            .decode(&self.active.as_ref().unwrap().generated, true)
            .unwrap_or_default();
        let delta = {
            let a = self.active.as_mut().unwrap();
            utf8_safe_delta(&full, &mut a.emitted)
        };
        let mut token_chunk = Chunk::token(id.clone(), next, delta);
        token_chunk.n_tokens = Some(1);

        let (gen_len, max_new, is_eos, pos, max_seq) = {
            let a = self.active.as_ref().unwrap();
            (
                a.generated.len(),
                a.max_new,
                a.eos.contains(&next_u),
                a.pos,
                a.max_seq,
            )
        };
        // Stop on the token cap or an EOS (checked after the push, as the loop
        // did): the just-emitted token was the last one.
        if gen_len >= max_new || is_eos {
            let mut out = vec![(id, token_chunk)];
            out.extend(self.finalize_pipeline(true));
            return out;
        }
        // Stop before forwarding at pos == max_seq (the caches can't hold that
        // absolute position); the token sampled from the last in-range position
        // was already emitted above. `length`, not `stop`.
        if pos >= max_seq {
            self.active.as_mut().unwrap().hit_context_cap = true;
            let mut out = vec![(id, token_chunk)];
            out.extend(self.finalize_pipeline(true));
            return out;
        }

        // Forward the emitted token to sample the next one.
        let downstream = self.active.as_ref().unwrap().downstream.clone();
        let cfg = self.active.as_ref().unwrap().cfg.clone();
        match self.forward_one_token_first(
            next_u,
            pos,
            &cfg,
            &downstream,
            true,
            Self::reply_deadline(),
        ) {
            Ok(tok_back) => {
                let a = self.active.as_mut().unwrap();
                a.next = tok_back;
                a.pos += 1;
                vec![(id, token_chunk)]
            }
            Err(e) => {
                // A mid-decode wire failure emits the partial output as a clean
                // completion (matches the old loop's `break`): the streamed
                // tokens stand, finish_reason falls out of the token count.
                warn!(task = %id, "decode forward failed; emitting partial output: {e}");
                let mut out = vec![(id, token_chunk)];
                out.extend(self.finalize_pipeline(false));
                out
            }
        }
    }

    /// Finish the in-flight task: log throughput, cache the full
    /// [prompt + response] KV prefix (opt-in, and only when `cache`), and emit
    /// the final marker with the OpenAI finish_reason. Consumes `self.active`.
    /// The final marker carries no text — every visible byte already streamed
    /// as a token delta.
    ///
    /// `cache` is false when finalizing from a failed forward. `a.pos` only
    /// advances on the `Ok` arm, but `forward_one_token_first` mutates the
    /// runner's KV before it touches the wire — so after a failure the runner
    /// holds one position more than `a.pos` describes. Caching there would
    /// store a snapshot one longer than the key naming it, and the next turn
    /// restoring that key would land every rank at a position rank 0 does not
    /// prefill from. Skipping it also avoids inserting into the index while a
    /// `send_cache_prefix` down the just-failed wire drops the peers' copy.
    fn finalize_pipeline(&mut self, cache: bool) -> Vec<(TaskId, Chunk)> {
        let Some(a) = self.active.take() else {
            return Vec::new();
        };
        let n_tokens = a.generated.len() as u32;
        let elapsed = a.started.elapsed().as_secs_f64();
        info!(
            task = %a.id,
            tokens = n_tokens,
            total = self.total,
            elapsed_s = elapsed,
            tok_s = if elapsed > 0.0 { n_tokens as f64 / elapsed } else { 0.0 },
            "task done (dsv4 pipeline-parallel)"
        );
        // Cache the full [prompt + generated] KV (positions [0, pos)) on every
        // rank, so the NEXT turn — which resends this response as an assistant
        // message — reuses the whole conversation (prompt AND response), not just
        // the prompt. The key is exactly the tokens whose KV is now cached (pos
        // covers the prompt plus every forwarded generated token).
        if cache && self.runner.prefix_cache_enabled() {
            let gen_cached = a.pos.saturating_sub(a.n_prefill).min(a.generated.len());
            let mut key_tokens = a.prompt_ids.clone();
            key_tokens.extend_from_slice(&a.generated[..gen_cached]);
            let key = self.prefix_remember(&key_tokens);
            self.runner.cache_prefix(key);
            if let Err(e) = self.block_on(send_cache_prefix(&a.downstream, key)) {
                warn!(task = %a.id, "post-decode send_cache_prefix failed: {e}");
            }
        }
        let mut chunk = Chunk::final_marker(a.id.clone(), "");
        // The completion's tokens all rode the streamed token chunks (each
        // n_tokens=1); the final marker carries none. Set it explicitly to 0 so
        // an orchestrator summing the per-frame SSE `n_tokens` field (which
        // renders `n_tokens.unwrap_or(1)`) doesn't count a phantom token here.
        chunk.n_tokens = Some(0);
        // Without this the API reports `prompt_tokens: 0` on every response —
        // it reads `chunk.prompt_tokens`, and this engine never set it (the OV
        // engines do). `n_prefill`, not `prompt_ids.len()`, because an
        // over-budget prompt is truncated and the count should describe the
        // tokens actually processed.
        chunk.prompt_tokens = Some(a.n_prefill as u32);
        chunk.finish_reason = Some(if a.hit_context_cap {
            FinishReason::Length
        } else {
            finish_reason_for(n_tokens as usize, a.max_new)
        });
        vec![(a.id, chunk)]
    }

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
            FrameKind::Forward => self.handle_forward(&upstream, downstream.as_ref(), true),
            FrameKind::ForwardNoSample => {
                self.handle_forward(&upstream, downstream.as_ref(), false)
            }
            FrameKind::ForwardBatchPrefill => {
                // Batched prompt prefill (Rust shell). Fatal on error, like the
                // other prefill frames: a KV hole would corrupt the whole run.
                let res = self.handle_forward_batch_prefill(&upstream, downstream.as_ref(), true);
                if res.is_err() {
                    self.peer_disconnected = true;
                }
                res
            }
            FrameKind::ForwardBatchPrefillNoSample => {
                // An intermediate window of a prompt longer than MAX_BATCH_COUNT:
                // advance KV over the rows, sample nothing. Same fatal-on-error.
                let res = self.handle_forward_batch_prefill(&upstream, downstream.as_ref(), false);
                if res.is_err() {
                    self.peer_disconnected = true;
                }
                res
            }
            FrameKind::RestorePrefix => match self.block_on(recv_key_body_server(&upstream)) {
                Ok(key) => {
                    // Rank 0 only sends this after restoring the key itself, so a
                    // miss here means this rank's slice cache has diverged from the
                    // driver's index — every rank is meant to hold the same keys.
                    // Continuing would leave this rank at position 0 while rank 0
                    // prefills from the reused length, which trips the batch
                    // position assert one frame later with a generic message.
                    // Latch instead, the same way a malformed frame does, so the
                    // stage tears down at the point of divergence. Don't relay: the
                    // ranks below us must not restore either.
                    if self.runner.restore_prefix(key).is_none() {
                        self.peer_disconnected = true;
                        Err(format!(
                            "restore_prefix: key {key} absent from this rank's slice cache \
                             (kv-prefix cache diverged from rank 0)"
                        ))
                    } else {
                        match downstream.as_ref() {
                            Some(down) => self
                                .block_on(send_restore_prefix(down, key))
                                .map_err(|e| format!("relay restore_prefix: {e}")),
                            None => Ok(()),
                        }
                    }
                }
                Err(e) => {
                    self.peer_disconnected = true;
                    Err(format!("recv restore_prefix key: {e}"))
                }
            },
            FrameKind::CachePrefix => match self.block_on(recv_key_body_server(&upstream)) {
                Ok(key) => {
                    self.runner.cache_prefix(key);
                    match downstream.as_ref() {
                        Some(down) => self
                            .block_on(send_cache_prefix(down, key))
                            .map_err(|e| format!("relay cache_prefix: {e}")),
                        None => Ok(()),
                    }
                }
                Err(e) => {
                    self.peer_disconnected = true;
                    Err(format!("recv cache_prefix key: {e}"))
                }
            },
            other => Err(format!(
                "worker received unsupported frame {other:?} (dsv4 has no spec-decode batching)"
            )),
        };
        if let Err(e) = res {
            warn!("worker frame failed: {e}");
            std::thread::sleep(WORKER_BACKOFF);
        }
        Vec::new()
    }

    /// `sample = false` (ForwardNoSample, prefill-intermediate): still run
    /// my layers (KV/compressor state must advance), but the last rank
    /// skips head+sample+history and returns a dummy Token(-1) — rank 0
    /// discards it. Keeps phantom prefill samples out of the penalty
    /// history and skips the vocab-width head GEMV per prompt token.
    fn handle_forward(
        &mut self,
        upstream: &Arc<TokioMutex<ActivationServer>>,
        downstream: Option<&Arc<TokioMutex<ActivationClient>>>,
        sample: bool,
    ) -> Result<(), String> {
        let (past_seq_len, sampling_cfg, hidden_f32, _shape) = self
            .block_on(recv_forward_body_server(upstream))
            .map_err(|e| format!("recv_forward_body: {e}"))?;
        let pos = past_seq_len as usize;
        // A malformed or over-range frame is a protocol violation we cannot
        // serve: forwarding a wrong-width hidden, or one at a position the
        // caches can't hold, would panic (OOB) mid-pipeline. Latch the peer as
        // disconnected and return Err so step() surfaces a connection-fatal
        // error — the relay loop tears the stage down and rebuilds, and rank 0
        // unblocks on the closed socket. Do NOT reply a token here: an ack of
        // -1 would OOB-panic rank 0's embed, and staying silent would hang it.
        let expect_w = self.runner.hidden_size();
        if hidden_f32.len() != expect_w {
            self.peer_disconnected = true;
            return Err(format!(
                "forward frame hidden width {} != expected {expect_w}",
                hidden_f32.len()
            ));
        }
        if pos >= self.runner.max_seq() {
            self.peer_disconnected = true;
            return Err(format!(
                "forward position {pos} exceeds context budget {}",
                self.runner.max_seq()
            ));
        }
        let hidden = self.runner.forward_layers(hidden_f32, pos, None);
        if self.is_last() {
            if !sample {
                return self
                    .block_on(send_token_upstream(upstream, -1))
                    .map_err(|e| format!("send_token (nosample ack): {e}"));
            }
            let logits = self.runner.head_logits(&hidden);
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
            // Same design rule as rank 0's head: the Token reply is owed to a frame
            // we just forwarded, so bound the wait on the strict deadline. If our
            // own downstream is dead, this relay must surface an error that tears
            // the stage down — an unbounded recv here would wedge this rank AND
            // every upstream rank blocked waiting on our reply.
            let reply_deadline = Self::reply_deadline();
            self.block_on(async {
                if sample {
                    send_forward(down, past_seq_len, &sampling_cfg, &hidden, [1, 1, h])
                        .await
                        .map_err(|e| format!("send_forward: {e}"))?;
                } else {
                    send_forward_nosample(down, past_seq_len, &sampling_cfg, &hidden, [1, 1, h])
                        .await
                        .map_err(|e| format!("send_forward_nosample: {e}"))?;
                }
                let t = crate::dist::recv_token_reply(down, reply_deadline).await?;
                send_token_upstream(upstream, t)
                    .await
                    .map_err(|e| format!("relay token: {e}"))
            })
        }
    }

    /// Worker-side ForwardBatchPrefill handler: run ALL `rows` prompt positions
    /// through my layers as one batch (batch-union MoE dedups their experts),
    /// then — on the last rank — head+sample ONLY the final row (the first
    /// generated token; earlier prompt rows are neither sampled nor recorded,
    /// matching ForwardNoSample) and reply with a single Token. Mid ranks relay
    /// the batch downstream and pass the single Token back upstream. Only the
    /// Rust shell (which needs no per-position token id) drives this path.
    fn handle_forward_batch_prefill(
        &mut self,
        upstream: &Arc<TokioMutex<ActivationServer>>,
        downstream: Option<&Arc<TokioMutex<ActivationClient>>>,
        sample: bool,
    ) -> Result<(), String> {
        let (base, batch_count, sampling_cfg, hidden_f32, in_shape) = self
            .block_on(recv_forward_batch_body_server(upstream))
            .map_err(|e| format!("recv_forward_batch(prefill): {e}"))?;
        let hidden = self.runner.hidden_size();
        let rows = batch_count as usize;
        let base = base as usize;
        if in_shape[0] != 1 || in_shape[1] != batch_count || in_shape[2] as usize != hidden {
            self.peer_disconnected = true;
            return Err(format!(
                "forward_batch_prefill shape {in_shape:?} (expected [1, {batch_count}, {hidden}])"
            ));
        }
        if hidden_f32.len() != rows * hidden {
            self.peer_disconnected = true;
            return Err(format!(
                "forward_batch_prefill hidden len {} != {rows}*{hidden}",
                hidden_f32.len()
            ));
        }
        if base + rows > self.runner.max_seq() {
            self.peer_disconnected = true;
            return Err(format!(
                "forward_batch_prefill positions {base}..{} exceed context budget {}",
                base + rows,
                self.runner.max_seq()
            ));
        }
        let h = self.runner.forward_layers_batch(hidden_f32, base, rows);
        if self.is_last() {
            if !sample {
                // Intermediate window: KV is advanced above; sample nothing and
                // record nothing (recording mid-prompt rows would poison the
                // rep-penalty history). Reply with a dummy Token(-1) ack so the
                // driver can pipeline the next window.
                return self
                    .block_on(send_token_upstream(upstream, -1))
                    .map_err(|e| format!("send_token (batch prefill nosample): {e}"));
            }
            let last = (rows - 1) * hidden;
            let logits = self.runner.head_logits(&h[last..last + hidden]);
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
                .map_err(|e| format!("send_token (batch prefill): {e}"))
        } else {
            let down = downstream.ok_or("mid rank missing downstream")?;
            self.block_on(async {
                // Relay with the matching kind so the last rank knows whether to
                // sample this window.
                if sample {
                    send_forward_batch_prefill(
                        down,
                        base as u32,
                        batch_count,
                        &sampling_cfg,
                        &h,
                        [1, batch_count, hidden as u32],
                    )
                    .await
                    .map_err(|e| format!("send_forward_batch_prefill: {e}"))?;
                } else {
                    send_forward_batch_prefill_nosample(
                        down,
                        base as u32,
                        batch_count,
                        &sampling_cfg,
                        &h,
                        [1, batch_count, hidden as u32],
                    )
                    .await
                    .map_err(|e| format!("send_forward_batch_prefill_nosample: {e}"))?;
                }
                // This window's Token reply is owed only after every downstream
                // stage finishes the WHOLE window, not one token — the per-token
                // `reply_deadline` above would read a merely-busy peer as dead.
                // An unbounded recv here (the prior bug) falls through to the
                // 900s frame-idle ceiling, which is connection-fatal and drops
                // `ActivationClient::sock` for good (only `Builder::connect`
                // redials it) — killing this link, and every rank upstream of
                // it, for the rest of the process.
                let t = crate::dist::recv_token_reply(down, Self::reply_deadline_prefill()).await?;
                send_token_upstream(upstream, t)
                    .await
                    .map_err(|e| format!("relay token: {e}"))
            })
        }
    }

    /// Rank-0 batched prefill: embed the whole prompt, run my layer slice as one
    /// batch, ship it downstream as a single ForwardBatchPrefill, and get back
    /// the first generated token. Advances `self.runner` KV by `rows`. Used only
    /// when the runner supports batched prefill (no per-position token id needed).
    fn forward_prompt_batch_first(
        &mut self,
        prompt: &[u32],
        base: usize,
        cfg: &crate::sampling::SamplingConfig,
        downstream: &Arc<TokioMutex<ActivationClient>>,
        sample: bool,
    ) -> Result<i64, String> {
        let hidden = self.runner.hidden_size();
        let rows = prompt.len();
        let mut batch = vec![0.0f32; rows * hidden];
        for (r, &t) in prompt.iter().enumerate() {
            batch[r * hidden..(r + 1) * hidden].copy_from_slice(&self.runner.embed_token(t));
        }
        // `base` is the KV position this window starts at (0 = full prefill; > 0
        // for a restored prefix or a later window of a >MAX_BATCH_COUNT prompt).
        // Each rank appends rows at [base, ..). `sample=false` windows advance KV
        // without sampling and reply with a dummy Token(-1).
        let h = self.runner.forward_layers_batch(batch, base, rows);
        let cfg = cfg.clone();
        self.block_on(async {
            if sample {
                send_forward_batch_prefill(
                    downstream,
                    base as u32,
                    rows as u32,
                    &cfg,
                    &h,
                    [1, rows as u32, hidden as u32],
                )
                .await
                .map_err(|e| format!("send_forward_batch_prefill: {e}"))?;
            } else {
                send_forward_batch_prefill_nosample(
                    downstream,
                    base as u32,
                    rows as u32,
                    &cfg,
                    &h,
                    [1, rows as u32, hidden as u32],
                )
                .await
                .map_err(|e| format!("send_forward_batch_prefill_nosample: {e}"))?;
            }
            // Same rule as the mid-rank relay: this window's Token reply is
            // owed only after every downstream stage finishes the whole
            // window, so it needs the widened prefill deadline, not the
            // per-token one — and never the raw, unbounded recv (see the
            // comment on the mid-rank relay above `handle_forward_batch_prefill`
            // for why that's connection-fatal, not just slow).
            crate::dist::recv_token_reply(downstream, Self::reply_deadline_prefill()).await
        })
    }
}

impl<R: StagedRunner> Engine for PipelineEngine<R> {
    fn warmup(&mut self) {
        if self.total == 1 {
            if let Some(tok) = self.tokenizer.as_ref() {
                let ids: Vec<u32> = tok
                    .encode("Hello", false)
                    .map(|e| e.get_ids().to_vec())
                    .unwrap_or_else(|_| vec![1]);
                let _ = self.runner.generate_argmax(&ids, 1);
                info!("warmup: generated 1 token (dsv4)");
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
        // Drop it from the queue if it hasn't started. If it's the in-flight
        // streamed task (rank 0, pipeline mode), drop `active` too so decoding
        // stops at the current token boundary instead of grinding out the rest
        // of a completion no one will read — an SSE client that disconnects
        // mid-stream triggers this via `ChunkStream::drop`. The next task's
        // `begin_generation` issues a fresh reset + send_reset, so abandoning
        // mid-sequence leaves no stale KV state to clean up here.
        self.pending.retain(|t| &t.task_id != task_id);
        if self.active.as_ref().is_some_and(|a| &a.id == task_id) {
            self.active = None;
        }
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if self.total <= 1 {
            return Ok(self.step_single_stage());
        }
        // step_first handles its own errors terminally (final-marker/error
        // chunk on the driver), so rank 0 never surfaces an Err here.
        if self.rank == 0 {
            return Ok(self.step_first());
        }
        // Worker rank. step_worker returns empty once an upstream disconnect (or
        // a protocol violation escalated to one in handle_forward) latches
        // peer_disconnected. The upstream socket can only be re-accepted by a
        // rebuild, so surface a connection-fatal Err to run_relay_loop (its only
        // driver) exactly once — mirroring SparseMoEEngine::step — so the stage
        // tears down and rebuilds instead of backing off Ok(empty) forever.
        let produced = self.step_worker();
        if worker_should_report_disconnect(self.peer_disconnected, self.disconnect_reported) {
            self.disconnect_reported = true;
            return Err(EngineError::NotConnected);
        }
        Ok(produced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-token streaming deltas (PipelineEngine::decode_step) must
    /// reconstruct the full text exactly, never emit a lone U+FFFD, and hold a
    /// multi-byte char that a tokenizer decodes across two tokens (the partial
    /// step surfaces as a trailing replacement char) until it completes. `full`
    /// is `decode(generated)` growing one token per step.
    #[test]
    fn utf8_safe_delta_streams_full_text_across_multibyte_split() {
        // Steps: "Hi" -> partial emoji (trailing U+FFFD) -> completed "Hi 🚀"
        // -> "Hi 🚀!". The rocket is 4 bytes; its first token decodes to a
        // partial code unit (U+FFFD), the second completes it.
        let steps = ["Hi", "Hi \u{FFFD}", "Hi 🚀", "Hi 🚀!"];
        let mut emitted = 0usize;
        let mut streamed = String::new();
        for full in steps {
            let d = utf8_safe_delta(full, &mut emitted);
            assert!(
                !d.contains('\u{FFFD}'),
                "a streamed delta must never carry a replacement char: {d:?}"
            );
            streamed.push_str(&d);
        }
        assert_eq!(
            streamed, "Hi 🚀!",
            "concatenated deltas must reconstruct the final decoded text"
        );
        // The held step emitted nothing; the completing step emitted the whole
        // rocket at once.
        assert_eq!(emitted, "Hi 🚀!".len());
    }

    /// Steady-state (no multi-byte split): each step emits exactly the newly
    /// decoded suffix, and an EOS/empty final decode adds no phantom bytes.
    #[test]
    fn utf8_safe_delta_ascii_is_one_suffix_per_step() {
        let steps = ["The", "The quick", "The quick brown"];
        let mut emitted = 0usize;
        let mut out = String::new();
        for full in steps {
            out.push_str(&utf8_safe_delta(full, &mut emitted));
        }
        assert_eq!(out, "The quick brown");
        // A final decode identical to the last (e.g. an EOS skipped by
        // skip_special_tokens) yields an empty delta.
        assert_eq!(utf8_safe_delta("The quick brown", &mut emitted), "");
    }

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

    // -------- batched-prefill reply deadline (regression for the unbounded
    // recv_kind_client/recv_token_body_client pair — see `prefill_reply_budget`
    // and its two call sites in `handle_forward_batch_prefill` /
    // `forward_prompt_batch_first` — AND for the follow-up bug where an
    // unclamped widened budget could reach the frame-idle ceiling and
    // reinstate the same failure mode via a different path) --------

    #[test]
    fn prefill_reply_budget_widens_by_the_documented_factor() {
        // Pins the concrete factor (10) and a concrete expected duration —
        // NOT the implementation's own multiplication echoed back at itself,
        // which would pass even if the multiplication were wrong.
        assert_eq!(cascadia_transport::PREFILL_REPLY_TIMEOUT_FACTOR, 10);
        let recv_timeout = std::time::Duration::from_secs(60);
        assert_eq!(
            prefill_reply_budget(recv_timeout, None),
            std::time::Duration::from_secs(600)
        );
    }

    #[test]
    fn prefill_reply_budget_exceeds_the_per_token_reply_deadline() {
        // The defect under test: a batched prefill window waits on every
        // remaining stage's WHOLE window of compute, not one token, so its
        // budget must be strictly wider than the per-token reply deadline.
        // Compares against `per_token_reply_budget` (the real function
        // `reply_deadline` delegates to) rather than a disconnected literal,
        // so it pins the actual relationship between the two, not a copy of it.
        let recv_timeout = std::time::Duration::from_secs(60);
        assert!(prefill_reply_budget(recv_timeout, None) > per_token_reply_budget(recv_timeout));
    }

    #[test]
    fn prefill_reply_budget_saturates_instead_of_panicking() {
        // saturating_mul must not panic on an absurd configured recv_timeout;
        // Duration's `*` would.
        let huge = std::time::Duration::MAX;
        assert_eq!(prefill_reply_budget(huge, None), std::time::Duration::MAX);
    }

    #[test]
    fn prefill_reply_budget_unclamped_when_no_ceiling_is_configured() {
        // `None` == the operator disabled the ceiling entirely
        // (`set_frame_idle_ceiling_secs(0)`) — nothing to stay under.
        let recv_timeout = std::time::Duration::from_secs(60);
        assert_eq!(
            prefill_reply_budget(recv_timeout, None),
            recv_timeout.saturating_mul(cascadia_transport::PREFILL_REPLY_TIMEOUT_FACTOR)
        );
    }

    #[test]
    fn prefill_reply_budget_clamps_below_the_frame_idle_ceiling() {
        // The exact review scenario: CASCADIA_ACTIVATION_TIMEOUT_SECS=90 (a
        // natural operator response to prefills timing out on a slow fleet)
        // widens the naive budget to 900s == DEFAULT_FRAME_IDLE_CEILING. Left
        // unclamped, the transport layer's own idle-ceiling timeout can then
        // win the race inside `recv_token_reply` and drop the socket for
        // good — reinstating defect D via a different path. The clamped
        // budget must land strictly under the ceiling.
        let recv_timeout = std::time::Duration::from_secs(90);
        let ceiling = std::time::Duration::from_secs(900);
        let budget = prefill_reply_budget(recv_timeout, Some(ceiling));
        assert!(
            budget < ceiling,
            "budget {budget:?} must stay strictly under the frame-idle ceiling {ceiling:?}"
        );
        assert_eq!(budget, ceiling - PREFILL_BUDGET_CEILING_MARGIN);
    }

    #[test]
    fn prefill_reply_budget_ceiling_clamp_never_underflows() {
        // A ceiling configured at or below the margin must clamp to zero
        // (fails prefill windows fast) rather than panicking on subtraction
        // underflow — an operator can set an unusually small idle ceiling.
        let recv_timeout = std::time::Duration::from_secs(60);
        let tiny_ceiling = std::time::Duration::from_secs(5);
        assert_eq!(
            prefill_reply_budget(recv_timeout, Some(tiny_ceiling)),
            std::time::Duration::ZERO
        );
    }

    #[test]
    fn prefill_reply_budget_leaves_the_default_config_unclamped() {
        // With the fleet's actual defaults (60s recv_timeout, 900s ceiling)
        // the widened budget (600s) must NOT be affected by the clamp — the
        // clamp is a safety net for high operator-configured recv_timeout,
        // not a silent shrink of the common case.
        let recv_timeout = std::time::Duration::from_secs(60);
        let ceiling = cascadia_transport::DEFAULT_FRAME_IDLE_CEILING;
        assert_eq!(
            prefill_reply_budget(recv_timeout, Some(ceiling)),
            std::time::Duration::from_secs(600)
        );
    }
}
