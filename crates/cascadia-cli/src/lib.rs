//! cascadia CLI.
//!
//! Mirrors `python -m cascadia worker` from `cascadia/cli.py`. Only the
//! subset needed for this session's MVP — single-node inference + API
//! server. Multi-stage / discovery flags are accepted but enforced
//! against engine support.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use cascadia_engine::Builder;
use cascadia_engine_mock::MockBuilder;
use cascadia_engine_openvino::{
    Gemma4Builder, OvDistSpecBuilder, OvDistSpecWorkerBuilder, OvGenaiBuilder, OvRuntimeBuilder,
    Qwen36Builder,
};
use cascadia_engine_sparse_moe::{SparseMoEBuilder, SparseMoEBuilderConfig};
use cascadia_runner::Runner;
use cascadia_types::{GenerationTask, PeerEndpoint, PeerLayout, ShardSpec};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use futures::StreamExt;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

pub mod discover;
pub mod doctor;
pub mod placement;
pub mod profile;
pub mod profile_stage;
pub mod run_placement;
use discover::{cmd_discover, DiscoverArgs};
use doctor::{cmd_doctor, DoctorArgs};
use placement::{cmd_place, PlaceArgs};
use profile::{cmd_profile_devices, ProfileDevicesArgs};
use profile_stage::{cmd_profile_per_stage, PerStageArgs};
use run_placement::{cmd_run_placement, RunPlacementArgs};

/// String form of an engine kind, used in NodeInfo.engines for discovery
/// and in the dashboard's "Engines" pill list. Stable wire format —
/// matches the strings `cascadia engines` already prints.
fn engine_name(kind: EngineKind) -> &'static str {
    match kind {
        EngineKind::Mock => "mock",
        EngineKind::OvGenai => "ov-genai",
        EngineKind::OvRuntime => "ov-runtime",
        EngineKind::OvDistSpec => "ov-dist-spec",
        EngineKind::Gemma4 => "gemma4",
        EngineKind::SparseMoe => "sparse-moe",
        EngineKind::Qwen36Moe => "qwen36-moe",
    }
}

/// Measure round-trip time to a peer via a 1 s-timeout TCP connect.
/// Returns `None` if the peer is unreachable or the connect timed out.
/// The connect handshake gives the best low-overhead RTT signal we can
/// take without speaking the activation-relay protocol.
async fn probe_peer(host: &str, port: u16) -> Option<f64> {
    let start = std::time::Instant::now();
    let addr = format!("{host}:{port}");
    let connect = tokio::net::TcpStream::connect(&addr);
    match tokio::time::timeout(std::time::Duration::from_secs(1), connect).await {
        Ok(Ok(_stream)) => Some(start.elapsed().as_secs_f64() * 1000.0),
        _ => None,
    }
}

/// Local node identity + hardware specs for the dashboard node card.
/// Gathered once at worker startup. Read off the main async path via
/// `spawn_blocking` (see `cmd_worker`) because `sysinfo` does blocking
/// syscalls. Falls back to empty/0/"node" for anything undeterminable.
struct NodeSpecs {
    hostname: String,
    memory_mb: u64,
    cpu_model: String,
    cpu_cores: u32,
    os: String,
}

impl Default for NodeSpecs {
    fn default() -> Self {
        Self {
            hostname: "node".to_owned(),
            memory_mb: 0,
            cpu_model: String::new(),
            cpu_cores: 0,
            os: String::new(),
        }
    }
}

/// Blocking: probe RAM + CPU + OS + hostname via `sysinfo` in one pass.
/// Uses a narrow `RefreshKind` (RAM + CPU only) rather than `new_all()`,
/// which would also enumerate every process/disk/network just to read
/// three fields. `hostname` comes from `sysinfo` too (no `hostname`
/// subprocess that would yield "node" — and a node_id collision — when
/// the binary isn't on PATH).
fn gather_node_specs() -> NodeSpecs {
    use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
    let sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_memory(MemoryRefreshKind::nothing().with_ram())
            .with_cpu(CpuRefreshKind::nothing()),
    );
    // sysinfo reports total_memory() in bytes (0.30+).
    let memory_mb = sys.total_memory() / 1024 / 1024;
    let cpu_model = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_default();
    let cpu_cores = sys.cpus().len() as u32;
    let os = match (System::name(), System::os_version()) {
        (Some(n), Some(v)) if !v.is_empty() => format!("{n} {v}"),
        (Some(n), _) => n,
        _ => System::long_os_version().unwrap_or_default(),
    };
    let hostname = System::host_name()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "node".to_owned());
    NodeSpecs {
        hostname,
        memory_mb,
        cpu_model,
        cpu_cores,
        os,
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "cascadia",
    version,
    about = "Distributed LLM inference for Intel hardware",
    long_about = "Distributed LLM inference for Intel hardware.\n\n\
        SECURITY: cascadia's HTTP API and inter-stage TCP relay are \
        plaintext and unauthenticated. Bind only to trusted networks \
        (LAN, loopback) or terminate TLS + auth at a reverse proxy in \
        front of `--api`. See SECURITY.md in the repository root \
        for the full threat model."
)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Command,

    /// Logging level (overrides RUST_LOG).
    #[arg(long, default_value = "info", global = true)]
    pub log_level: String,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Check the environment + hardware and report what's ready. Run
    /// this FIRST — it catches the silent OpenVINO CPU-only fallback
    /// that otherwise shows up as mysterious slowness. See `cascadia
    /// doctor --help`.
    Doctor(DoctorArgs),
    /// Run a model on a single machine (sugar for a one-stage worker with an
    /// OpenAI API). Takes a local model directory — a whole-model OpenVINO IR,
    /// or a `cascadia shard` tree. See `cascadia run --help`.
    Run(RunArgs),
    /// Run a pipeline-stage worker.
    Worker(WorkerArgs),
    /// List Cascadia peers advertising on the local network. See
    /// `cascadia discover --help`.
    Discover(DiscoverArgs),
    /// List registered inference engines.
    Engines,
    /// Shard a HuggingFace causal-LM model into per-stage OpenVINO IRs
    /// for distributed inference. See `cascadia shard --help`.
    ///
    /// Model-specific dispatch happens inside the embedded exporter: a
    /// gemma-4 OpenVINO-IR dir (`openvino_language_model.xml` present) routes
    /// to the text-surgery path (`tools/gemma4_surgery/export_gemma4_text.py`),
    /// while safetensors gemma-4 still uses the torch `export_gemma4.py`.
    Shard(ShardArgs),
    /// Profile available OV devices on this host against a single
    /// model. Writes `device_profile.json`. See `cascadia
    /// profile-devices --help`. Step 1 of issue #41 (three-tier
    /// {iGPU, NPU, CPU} ILP placement).
    ProfileDevices(ProfileDevicesArgs),
    /// Profile each stage of a multi-stage shard on each device (latency +
    /// memory + op-support) and write `placement_profile.json` — the cost
    /// table for `cascadia place`. Step 1.5 of issue #41.
    ProfileStages(PerStageArgs),
    /// Solve three-tier {iGPU, NPU, CPU} placement from a per-stage cost
    /// profile and write `placement.json`. Step 2 of issue #41 — the ILP
    /// over `profile-stages` output.
    Place(PlaceArgs),
    /// Launch a heterogeneous pipeline from a `placement.json`: one worker
    /// per stage, each pinned to its assigned device. Step 3 of issue #41.
    RunPlacement(RunPlacementArgs),
    /// Generate a shell completion script (bash, zsh, fish, …). See
    /// `cascadia completions --help`.
    Completions(CompletionsArgs),
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineKind {
    Mock,
    OvGenai,
    OvRuntime,
    OvDistSpec,
    /// Gemma 4 multi-stage engine. Drives `gemma4_cached_v1` shards
    /// (per-layer-type asymmetric attention, KV-sharing, per-layer
    /// embeddings, baked softcap) produced by `tools/export_gemma4.py`.
    Gemma4,
    /// Kimi K2.6-style sparse-MoE engine. Routes only the top-k experts
    /// per token (not all 384) and runs the expert matmuls through the
    /// hand-rolled AVX-512 int4 GEMM kernel. Single-stage, CPU-targeted.
    SparseMoe,
    /// Qwen3.6-35B-A3B staged engine. Runs the IR-surgery shard chain
    /// (`tools/qwen36_surgery/export_qwen36_moe.py` output dir with
    /// manifest.json) in-process; greedy-only, batch=1. CPU-targeted
    /// for decode (see docs/architectures/qwen36-moe-support.md).
    Qwen36Moe,
}

#[derive(Parser, Debug, Clone)]
pub struct WorkerArgs {
    /// 0-based stage index.
    #[arg(long)]
    pub rank: u32,

    /// Total number of stages.
    #[arg(long)]
    pub total: u32,

    /// First transformer layer this stage holds (global, 0-based, inclusive).
    /// Together with --layer-end, pins an explicit asymmetric layer split
    /// instead of the default even split across ranks — e.g. a high-RAM CPU
    /// node holding most layers while small iGPU nodes hold a few each.
    /// `0/0` (both default) = even split. MiniMax-M2 sparse-moe only.
    #[arg(long, default_value_t = 0)]
    pub layer_start: u32,

    /// One-past-the-last transformer layer this stage holds (global,
    /// exclusive). `0` = unset (even split). See --layer-start.
    #[arg(long, default_value_t = 0)]
    pub layer_end: u32,

    /// Local model directory: a whole-model OpenVINO IR for ov-genai, or a
    /// `cascadia shard` tree for the staged engines. NOT an HF repo id — the
    /// worker never downloads or converts models; only `cascadia shard` does.
    #[arg(long)]
    pub model: String,

    /// Name reported by `/v1/models` and accepted as the `model` field in
    /// requests. Defaults to the basename of `--model` (so a local path like
    /// `C:\models\dsv4-4stage` is served as `dsv4-4stage`, not the raw path,
    /// which is ugly and breaks clients that build request JSON naively).
    #[arg(long)]
    pub served_model_name: Option<String>,

    /// Bind address for the upstream-receiving socket (default :9100).
    #[arg(long, default_value = ":9100")]
    pub listen: String,

    /// Downstream peer (host:port) — required for non-last stages.
    #[arg(long)]
    pub next: Option<String>,

    /// API bind address (e.g. :8000) — only valid for rank 0.
    #[arg(long)]
    pub api: Option<String>,

    /// OpenVINO device target. Forwarded verbatim to ov::Core::compile_model.
    ///
    /// Valid forms:
    ///   CPU                       host CPU
    ///   GPU, GPU.0, GPU.1, ...    a specific GPU (iGPU is .0 by convention)
    ///   NPU, NPU.0, ...           Neural Processing Unit (Lunar Lake / later)
    ///   AUTO                      let OpenVINO pick
    ///   MULTI:GPU.1,GPU.0,CPU     round-robin across devices
    ///   HETERO:GPU.1,CPU          split the graph by op affinity
    ///   BATCH:GPU                 auto-batch (throughput-favored)
    ///
    /// Run `cascadia doctor` to see which devices OpenVINO can see on this host.
    #[arg(long, default_value = "CPU", verbatim_doc_comment)]
    pub device: String,

    /// Inference engine.
    #[arg(long, value_enum, default_value_t = EngineKind::Mock)]
    pub engine: EngineKind,

    /// OpenVINO compiled-blob cache dir (sets plugin CACHE_DIR).
    /// Used by ov-genai (and ov-runtime / ov-dist-spec when ported).
    ///
    /// When unset (the CLI default), the engine builders default to
    /// `<cache_root>/cascadia/ov-cache` where `<cache_root>` is the
    /// platform user-cache dir (`~/.cache` on Linux, `~/Library/Caches`
    /// on macOS, `%LOCALAPPDATA%` on Windows). This converts the
    /// ~20-second cold-start GPU compile of a fresh Qwen3-1.7B IR into
    /// a ~1-second warm load on every subsequent invocation — the
    /// single biggest operator-facing latency win for the ov-genai
    /// path. Pass `--ov-cache-dir ""` (empty string) to disable.
    #[arg(long)]
    pub ov_cache_dir: Option<String>,

    /// OV GPU KV-cache precision (u8 / f16). Defaults already optimal.
    #[arg(long)]
    pub ov_kv_precision: Option<String>,

    /// OV GPU dynamic-quantization group size.
    #[arg(long)]
    pub ov_dyn_quant_group: Option<String>,

    /// OV performance hint (PERFORMANCE_HINT). LATENCY suits single-user
    /// decode; THROUGHPUT enables NUM_STREAMS auto-tuning. See OpenVINO
    /// high-level-performance-hints docs (2026).
    #[arg(long, value_enum, value_name = "MODE")]
    pub ov_performance_mode: Option<OvPerformanceMode>,

    /// OV inference precision hint: f16, bf16, or f32. Strongly recommended
    /// on Xe2/Battlemage GPUs, where f16/bf16 share XMX throughput but the
    /// default can silently fall back to f32. See OpenVINO gpu-device docs.
    #[arg(long, value_name = "PREC")]
    pub ov_inference_precision: Option<String>,

    /// OV number of parallel inference streams (NUM_STREAMS).
    /// See OpenVINO optimizing-throughput (streams-and-threads) docs (2026).
    #[arg(long, value_name = "N")]
    pub ov_num_streams: Option<u32>,

    /// OV host CPU inference thread cap (INFERENCE_NUM_THREADS). CPU plugin
    /// only. See OpenVINO CPU-device docs (2026).
    #[arg(long, value_name = "N")]
    pub ov_num_threads: Option<u32>,

    /// OV: allow internal auto-batching on the GPU plugin (ALLOW_AUTO_BATCHING).
    /// See OpenVINO automatic-batching docs (2026).
    #[arg(long)]
    pub ov_allow_auto_batching: bool,

    /// OV execution mode (EXECUTION_MODE_HINT). PERFORMANCE trades a little
    /// accuracy for throughput. See OpenVINO precision-control
    /// (execution-mode) docs (2026).
    #[arg(long, value_enum, value_name = "MODE")]
    pub ov_execution_mode: Option<OvExecutionMode>,

    /// NPU LLM prefill chunk size (NPUW_LLM_PREFILL_CHUNK_SIZE, OV 2025.3+).
    /// Applied only with --engine ov-genai on an NPU device; dropped with a
    /// warning otherwise (only ov-genai routes it through an ov::genai
    /// LLMPipeline).
    #[arg(long, value_name = "TOKS")]
    pub npu_prefill_chunk_size: Option<u32>,

    /// NPU max prompt length (MAX_PROMPT_LEN, static-shape constraint).
    /// Applied only with --engine ov-genai on an NPU device; dropped with a
    /// warning otherwise (only ov-genai routes it through an ov::genai
    /// LLMPipeline).
    #[arg(long, value_name = "TOKS")]
    pub npu_max_prompt_len: Option<u32>,

    /// NPU min response length (MIN_RESPONSE_LEN, static-shape constraint).
    /// Applied only with --engine ov-genai on an NPU device; dropped with a
    /// warning otherwise (only ov-genai routes it through an ov::genai
    /// LLMPipeline).
    #[arg(long, value_name = "TOKS")]
    pub npu_min_response_len: Option<u32>,

    /// Device for the chunked-prefill IR variant (default: same as
    /// --device). ov-runtime engine on a stateless static export
    /// (tools/export_shards.py --target npu --static-prefill-seq N) only.
    /// The phase split: e.g. `--device CPU --prefill-device NPU` runs the
    /// compute-bound wide prefill on the NPU and the bandwidth-bound seq=1
    /// decode on the CPU, sharing one host-side KV ring. Accepts the same
    /// OpenVINO device forms as --device. Note both compilations hold their
    /// own copy of the stage weights (~2x weight RSS when the devices
    /// differ); KV is not duplicated.
    #[arg(long)]
    pub prefill_device: Option<String>,

    /// Ignore the export's chunked-prefill IR variant and prefill one token
    /// per step (ov-runtime static path only; conflicts with
    /// --prefill-device). Escape hatch / A-B baseline knob.
    #[arg(long, default_value_t = false)]
    pub no_chunked_prefill: bool,

    /// Park the chunked-prefill model between prefills (ov-runtime static
    /// path only): after each task's prefill phase the prefill CompiledModel
    /// is dropped — freeing its resident stage-weight copy, the structural
    /// memory cost of the two-model split — and re-created from the compile
    /// cache at the next prefill. Trades a per-task reload (logged as
    /// reload_ms) for ~1x steady-state weight residency; for memory-tight
    /// stages. Composes with --prefill-device.
    #[arg(long, default_value_t = false)]
    pub park_prefill: bool,

    /// SPIKE: execute the decode model's sym-INT4 weight matmuls through the
    /// CascadiaInt4Gemv extension op straight from the mmapped IR .bin — the
    /// CPU plugin never makes its own resident repacked weight copy, so a
    /// hybrid stage's decode side costs ~0 extra weight RAM. ov-runtime,
    /// stateless static exports, --device CPU only. Numeric note: the op's
    /// accumulation order differs from oneDNN's, so output parity vs the
    /// stock kernel is validated empirically, not guaranteed bit-for-bit.
    #[arg(long, default_value_t = false)]
    pub gemv_offload: bool,

    /// Speculative-decode draft model path (FastDraft companion).
    #[arg(long)]
    pub draft_model: Option<String>,

    /// Device for the draft model (default: same as --device).
    /// Accepts the same OpenVINO device forms as --device.
    #[arg(long)]
    pub draft_device: Option<String>,

    /// Speculative-decode draft length per round.
    #[arg(long, default_value_t = 5)]
    pub spec_k: u32,

    /// Enable Prompt Lookup decoding with n-gram size N. Mutually
    /// exclusive with --draft-model.
    #[arg(long, default_value_t = 0)]
    pub prompt_lookup: u32,

    /// Packed multi-slot continuous batching (NPU, ov-runtime static exports):
    /// serve N concurrent requests in ONE inference by packing them into the
    /// sequence dimension with a per-row mask. Requires a packed variant beside
    /// the decode IR (`tools/packed_variant.py --slots N`). 0 = off.
    #[arg(long, default_value_t = 0)]
    pub packed_slots: u32,

    /// Reserve N KV slots as a read-only SHARED prefix that every packed slot
    /// may attend to — prefix caching without paged attention. The first
    /// admitted request populates it; later requests sharing that prompt prefix
    /// skip re-prefilling those tokens. Taken from the same window, so it costs
    /// per-slot context. Requires --packed-slots. 0 = off.
    #[arg(long, default_value_t = 0)]
    pub packed_prefix: u32,

    /// Continuous batching (#20, ov-genai only): serve concurrent requests
    /// through one ContinuousBatchingPipeline (paged attention; CPU/GPU
    /// plugins) instead of one generation at a time. Incompatible with
    /// --draft-model / --prompt-lookup.
    #[arg(long)]
    pub cb: bool,

    /// KV-cache size in GB for --cb (0 = ov-genai dynamic allocation).
    #[arg(long, default_value_t = 0)]
    pub cb_cache_size: u64,

    /// Max sequences batched per iteration for --cb (0 = ov-genai default,
    /// 256).
    #[arg(long, default_value_t = 0)]
    pub cb_max_num_seqs: u64,

    /// Max tokens batched per iteration for --cb (0 = ov-genai default,
    /// 256).
    #[arg(long, default_value_t = 0)]
    pub cb_max_batched_tokens: u64,

    /// Override the dynamic-split-fuse scheduler toggle for --cb
    /// (unset = ov-genai default, on).
    #[arg(long)]
    pub cb_dynamic_split_fuse: Option<bool>,

    /// Enable KV-block prefix caching across requests for --cb
    /// (unset = ov-genai default, off).
    #[arg(long)]
    pub cb_prefix_caching: Option<bool>,

    /// Max new tokens for stdin mode.
    #[arg(long, default_value_t = 64)]
    pub max_tokens: u32,

    /// Override the engines list advertised in the mDNS NodeInfo (comma-
    /// separated, e.g. "ov-genai,ov-runtime"). Decouples what shows in the
    /// dashboard from the actually-loaded engine — useful when running a
    /// mock worker for discovery testing but you want the card to look
    /// real. Default: derived from --engine.
    #[arg(long, value_delimiter = ',')]
    pub advertise_engines: Vec<String>,

    /// Override the device label advertised in mDNS NodeInfo. Distinct
    /// from --device, which selects the OpenVINO target device; this is
    /// purely a dashboard cosmetic. Default: copies --device.
    #[arg(long)]
    pub advertise_device: Option<String>,

    /// Override the MoE top-K dispatch (sparse-moe engine only).
    /// If set and < manifest top_k, dispatch only the first K' experts
    /// per token. K2.6 manifest default is 8; K=4 gives +146% tok/s on
    /// 10-prompt eval with matched quality. See docs/perf/A3_TOPK_REDUCTION.md.
    #[arg(long)]
    pub top_k_override: Option<u32>,

    /// Skip experts whose router weight falls below this threshold
    /// (sparse-moe engine only). 0.0 = disabled. Applied after
    /// top_k_override, so the effective set is the routed top-K whose
    /// routing weight >= threshold.
    #[arg(long)]
    pub routing_threshold: Option<f32>,

    /// KV-prefix cache size (sparse-moe engine only). `0` = disabled
    /// (default). When > 0, the engine caches the post-prefill KV
    /// snapshot for each unique prompt-token sequence; subsequent
    /// requests whose prompt starts with a cached prefix skip the
    /// matched portion of prefill. Best fit: chat workloads where the
    /// system prompt is shared across requests.
    ///
    /// Single-stage only on this PR; ignored on `--total > 1` configs
    /// (would require a transport frame to exchange snapshots across
    /// stages). One snapshot at K2.6 dimensions for a 512-token prompt
    /// is ~150 MiB, so practical caps are 1..8.
    #[arg(long, default_value_t = 0)]
    pub kv_prefix_cache_size: u32,

    /// LRU bound on the expert cache (sparse-moe engine only). `0` =
    /// unbounded (default; preserves pre-LRU behaviour). Positive =
    /// max number of resident experts; least-recently-used is dropped
    /// to make room on miss. At K2.6 dimensions ≈25 MiB per expert
    /// (int4_bin path) or ≈75 MiB (ov_ir path), so a cap of 256 caps
    /// resident expert RAM at ~6–20 GiB depending on backend.
    ///
    /// Inspired by PowerInfer SmallThinker's `MAX_N_CACHED` env var
    /// (MIT-licensed). The env var `CASCADIA_MAX_EXPERTS_CACHED`
    /// overrides this flag at runtime.
    #[arg(long, default_value_t = 0, env = "CASCADIA_MAX_EXPERTS_CACHED")]
    pub max_cached_experts: u32,

    /// Two-phase Gate-first FFN sparsity threshold (sparse-moe engine
    /// only). `0.0` = dense fallback (default; output bit-identical to
    /// pre-port path). Positive = skip up/down lanes where
    /// `|silu(gate)|` falls below `threshold · max_i|silu(gate_i)|`
    /// per token.
    ///
    /// Useful range for SwiGLU (K2.6): `0.05`–`0.15`. Higher = more
    /// skip + lower quality. Validate quality regressions against
    /// dense before deploying.
    ///
    /// Inspired by PowerInfer-2 §4.4 (skip Up/Down when Gate=0; ReLU)
    /// adapted to SwiGLU via the CATS / CHESS magnitude-threshold
    /// approach.
    #[arg(long, default_value_t = 0.0, env = "CASCADIA_FFN_SPARSITY_THRESHOLD")]
    pub ffn_sparsity_threshold: f32,

    /// Issue #35 — use the AXPY-form down kernel for the sparse FFN
    /// path. Only meaningful when `--ffn-sparsity-threshold > 0`.
    ///
    /// In the AXPY form, the down projection becomes
    /// `y += silu(gate)[r] * up[r] * down_t[r]` accumulated over only
    /// the active intermediate lanes `r`. Inactive lanes are skipped
    /// entirely — no FMA, no weight load. Kernel speedup ceiling
    /// `1 / active_frac` vs the dense down projection.
    ///
    /// Cost: each cached expert builds a transposed-and-requantized
    /// down weight on first use (~5 ms CPU + ~8.26 MiB extra heap
    /// per cached expert at K2.6 dimensions). The transposed cache
    /// shares `--max-cached-experts` as its capacity.
    ///
    /// Ports PowerInfer SmallThinker `fused_sparse_moe.cpp:174-186`
    /// (MIT).
    #[arg(long, default_value_t = false, env = "CASCADIA_FFN_AXPY_DOWN")]
    pub ffn_axpy_down: bool,

    /// Pre-build the AXPY transposed-down cache for every
    /// `(layer, expert)` at Runner construction (slow one-time
    /// cost; avoids per-prompt build latency thereafter). Only
    /// meaningful with `--ffn-axpy-down` enabled. K2.6 prebuild
    /// is ~20 s with rayon; disk: ~190 GiB on top of the model.
    /// Recommended for production deployments with diverse prompts.
    #[arg(long, default_value_t = false, env = "CASCADIA_FFN_AXPY_PREBUILD")]
    pub ffn_axpy_prebuild: bool,

    /// Issue #38 (CHESS) — load per-channel FFN sparsity thresholds
    /// from this JSON file (produced by the `calibrate_ffn_thresholds`
    /// tool). Takes precedence over `--ffn-sparsity-threshold` for
    /// any layer covered by the file; uncovered layers fall back to
    /// the scalar value.
    ///
    /// Per-channel τ typically achieves 2× the sparsity-at-quality of
    /// a single global τ — some channels reliably dominate the
    /// magnitude distribution, others are reliably negligible. See
    /// `docs/perf/CHESS_PER_CHANNEL.md` for the calibration workflow.
    #[arg(long, env = "CASCADIA_FFN_SPARSITY_THRESHOLDS_FILE")]
    pub ffn_sparsity_thresholds_file: Option<std::path::PathBuf>,

    /// Issue #38 (CHESS) — capture per-(layer, channel) `silu(gate)`
    /// histograms during routed expert calls and dump them to this
    /// directory on shutdown. Feed the dump into
    /// `calibrate_ffn_thresholds` to produce a per-channel threshold
    /// file. Only effective when `--ffn-axpy-down` is also on (the
    /// non-AXPY path doesn't surface `silu(gate)` to the runner).
    ///
    /// Memory cost: 60 layers × 2048 channels × 128 bins × 4 B
    /// ≈ 60 MiB resident on K2.6 single-stage; bounded.
    #[arg(long, env = "CASCADIA_FFN_SPARSITY_CAPTURE_DIR")]
    pub ffn_sparsity_capture_dir: Option<std::path::PathBuf>,
}

/// `cascadia run` — single-machine inference with sane defaults.
///
/// This is the "Ollama-style" first-run path: point it at a model and
/// it serves an OpenAI-compatible API, no rank/total/listen bookkeeping.
/// It maps onto a one-stage [`WorkerArgs`]. For multi-machine pipeline
/// parallelism, use `cascadia worker` directly.
#[derive(Parser, Debug, Clone)]
pub struct RunArgs {
    /// Local model directory: a whole-model OpenVINO IR (`optimum-cli export
    /// openvino`, or a pre-exported `*-int4-ov` download) or a `cascadia shard`
    /// tree. Not an HF repo id — `run` does not download or convert models.
    pub model: String,

    /// Device hint: GPU / CPU / NPU. Defaults to GPU — run `cascadia
    /// doctor` first to confirm the iGPU is visible to OpenVINO.
    /// Also accepts indexed/compound OpenVINO forms (GPU.N, AUTO,
    /// MULTI:, HETERO:, BATCH:) — see `cascadia worker --help`.
    #[arg(long, default_value = "GPU")]
    pub device: String,

    /// Inference engine. Defaults to `ov-genai` (single-stage, whole-model
    /// OpenVINO IR). Use `--engine ov-runtime` for a `cascadia shard` tree.
    #[arg(long, value_enum, default_value_t = EngineKind::OvGenai)]
    pub engine: EngineKind,

    /// API bind address. Defaults to `:8000` (all interfaces, port
    /// 8000). Pass e.g. `127.0.0.1:8000` to bind loopback only.
    #[arg(long, default_value = ":8000")]
    pub api: String,
}

#[derive(Parser, Debug, Clone)]
pub struct CompletionsArgs {
    /// Shell to generate a completion script for.
    #[arg(value_enum)]
    pub shell: Shell,
}

impl WorkerArgs {
    /// Build a single-stage (`--rank 0 --total 1`) worker config from the
    /// reduced `cascadia run` surface, leaving every advanced knob at its
    /// `worker` default. Keeping this here (rather than spreading defaults
    /// into `cmd_run`) means `run` and `worker` can't silently drift.
    fn single_node(model: String, device: String, engine: EngineKind, api: String) -> Self {
        WorkerArgs {
            rank: 0,
            total: 1,
            layer_start: 0,
            layer_end: 0,
            model,
            served_model_name: None,
            listen: ":9100".into(),
            next: None,
            api: Some(api),
            device,
            engine,
            ov_cache_dir: None,
            ov_kv_precision: None,
            ov_dyn_quant_group: None,
            ov_performance_mode: None,
            ov_inference_precision: None,
            ov_num_streams: None,
            ov_num_threads: None,
            ov_allow_auto_batching: false,
            ov_execution_mode: None,
            npu_prefill_chunk_size: None,
            npu_max_prompt_len: None,
            npu_min_response_len: None,
            prefill_device: None,
            no_chunked_prefill: false,
            park_prefill: false,
            gemv_offload: false,
            draft_model: None,
            draft_device: None,
            spec_k: 5,
            prompt_lookup: 0,
            packed_slots: 0,
            packed_prefix: 0,
            cb: false,
            cb_cache_size: 0,
            cb_max_num_seqs: 0,
            cb_max_batched_tokens: 0,
            cb_dynamic_split_fuse: None,
            cb_prefix_caching: None,
            max_tokens: 64,
            advertise_engines: Vec::new(),
            advertise_device: None,
            top_k_override: None,
            routing_threshold: None,
            kv_prefix_cache_size: 0,
            max_cached_experts: 0,
            ffn_sparsity_threshold: 0.0,
            ffn_axpy_down: false,
            ffn_axpy_prebuild: false,
            ffn_sparsity_thresholds_file: None,
            ffn_sparsity_capture_dir: None,
        }
    }
}

/// Default fixed total context length for NPU static-shape shards. Shared by
/// the clap default and the cpu-gpu "ignored" guard so they can't drift.
const DEFAULT_STATIC_CONTEXT: u32 = 1024;

#[derive(Parser, Debug, Clone)]
pub struct ShardArgs {
    /// HuggingFace repo id (e.g. unsloth/Meta-Llama-3.1-8B-Instruct), a local
    /// directory with safetensors + config.json, or — for the Gemma-4 / Qwen3.6
    /// surgery paths — an already-exported OpenVINO IR directory.
    #[arg(long)]
    pub model: String,

    /// Output directory for the shard tree (will be created).
    #[arg(long, short = 'o')]
    pub output_dir: String,

    /// Number of pipeline stages to split the model into.
    #[arg(long)]
    pub num_stages: u32,

    /// Weight quantization. INT4 is the typical choice for Intel
    /// hardware; FP16 if NNCF is unavailable or you want max quality.
    #[arg(long, value_enum, default_value_t = ShardQuant::Int4)]
    pub quantization: ShardQuant,

    /// Deployment target. `npu` emits a stateless static-shape shard
    /// (no make_stateful, fixed seq/KV) that the NPU compiler accepts;
    /// `cpu-gpu` (default) emits the stateful dynamic-shape shard.
    #[arg(long, value_enum, default_value_t = ShardTarget::CpuGpu)]
    pub target: ShardTarget,

    /// NPU only: fixed query-window length. Must be 1 for `--target npu`
    /// (the runtime decodes one token per step); ignored otherwise.
    #[arg(long, default_value_t = 1)]
    pub static_seq: u32,

    /// NPU only: fixed total context length (past-KV length =
    /// static_context - static_seq); ignored without `--target npu`.
    #[arg(long, default_value_t = DEFAULT_STATIC_CONTEXT)]
    pub static_context: u32,

    /// NPU only: also emit a chunked-prefill IR variant with a fixed seq=N
    /// query window (openvino_prefill_model.xml). Enables `cascadia worker
    /// --prefill-device` phase-split execution (e.g. prefill on NPU, decode
    /// on CPU) and ~N× fewer prefill forwards even single-device. 0 = off.
    #[arg(long, default_value_t = 0)]
    pub static_prefill_seq: u32,

    /// torch default dtype during export. FP16 reduces memory and is
    /// required for `--target npu` (the runtime feeds KV as f16).
    #[arg(long, value_enum, default_value_t = ShardDtype::Fp16)]
    pub default_dtype: ShardDtype,

    /// Explicit per-stage layer boundaries, comma-separated. With
    /// `--num-stages 3 --layer-split 16,24` on a 32-layer model:
    /// stage 0 = [0,16), stage 1 = [16,24), stage 2 = [24,32).
    /// If omitted, layers are split uniformly.
    #[arg(long)]
    pub layer_split: Option<String>,

    /// Export only this stage index (debug). Useful for re-exporting
    /// one stage after a config tweak without re-running the others.
    #[arg(long)]
    pub stage: Option<u32>,

    /// Override the python interpreter used to run the bundled exporter.
    /// Defaults to `python3` then `python` on PATH.
    #[arg(long)]
    pub python: Option<String>,

    /// Skip the Python interpreter detection check (assumes the chosen
    /// interpreter has nncf, openvino, transformers, torch, safetensors
    /// installed). Use this if you know your env is good and want a
    /// faster start.
    #[arg(long)]
    pub skip_check: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum ShardQuant {
    Fp16,
    Int4,
    Int4Asym,
    Int8,
}

impl ShardQuant {
    fn as_arg(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Int4 => "int4",
            Self::Int4Asym => "int4_asym",
            Self::Int8 => "int8",
        }
    }
}

/// Deployment target for a shard. `npu` emits a stateless static-shape
/// shard the NPU compiler accepts; `cpu-gpu` emits the stateful shard.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardTarget {
    #[value(name = "cpu-gpu")]
    CpuGpu,
    Npu,
}

impl ShardTarget {
    fn as_arg(self) -> &'static str {
        match self {
            Self::CpuGpu => "cpu-gpu",
            Self::Npu => "npu",
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardDtype {
    Fp16,
    Fp32,
}

impl ShardDtype {
    fn as_arg(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
        }
    }
}

/// OpenVINO performance hint (`PERFORMANCE_HINT`). These map to
/// `ov::hint::PerformanceMode` and are device-independent, so — unlike
/// `--ov-inference-precision`, whose valid set is device-specific (`bf16` /
/// `dynamic` vary per device) and is therefore left a free string for OV to
/// validate — we enumerate them and reject typos at parse time.
#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum OvPerformanceMode {
    #[value(name = "LATENCY")]
    Latency,
    #[value(name = "THROUGHPUT")]
    Throughput,
    #[value(name = "CUMULATIVE_THROUGHPUT")]
    CumulativeThroughput,
}

impl OvPerformanceMode {
    fn as_ov(self) -> &'static str {
        match self {
            Self::Latency => "LATENCY",
            Self::Throughput => "THROUGHPUT",
            Self::CumulativeThroughput => "CUMULATIVE_THROUGHPUT",
        }
    }
}

/// OpenVINO execution mode (`EXECUTION_MODE_HINT`). Maps to
/// `ov::hint::ExecutionMode`; device-independent, so enumerated like
/// [`OvPerformanceMode`].
#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum OvExecutionMode {
    #[value(name = "ACCURACY")]
    Accuracy,
    #[value(name = "PERFORMANCE")]
    Performance,
}

impl OvExecutionMode {
    fn as_ov(self) -> &'static str {
        match self {
            Self::Accuracy => "ACCURACY",
            Self::Performance => "PERFORMANCE",
        }
    }
}

pub async fn run(cli: Cli) -> Result<()> {
    init_tracing(&cli.log_level);
    match cli.cmd {
        Command::Doctor(args) => cmd_doctor(args),
        Command::Run(args) => cmd_run(args).await,
        Command::Engines => cmd_engines(),
        Command::Worker(args) => cmd_worker(args).await,
        Command::Discover(args) => cmd_discover(args).await,
        Command::Shard(args) => cmd_shard(args).await,
        Command::ProfileDevices(args) => cmd_profile_devices(args),
        Command::ProfileStages(args) => cmd_profile_per_stage(args),
        Command::Place(args) => cmd_place(args),
        Command::RunPlacement(args) => cmd_run_placement(args).await,
        Command::Completions(args) => cmd_completions(args),
    }
}

async fn cmd_run(args: RunArgs) -> Result<()> {
    info!(model = %args.model, device = %args.device, engine = ?args.engine, "cascadia run (single machine)");
    let worker = WorkerArgs::single_node(args.model, args.device, args.engine, args.api);
    cmd_worker(worker).await
}

fn cmd_completions(args: CompletionsArgs) -> Result<()> {
    let mut cmd = Cli::command();
    let bin = cmd.get_name().to_string();
    clap_complete::generate(args.shell, &mut cmd, bin, &mut std::io::stdout());
    Ok(())
}

fn init_tracing(level: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();
}

fn cmd_engines() -> Result<()> {
    println!("  mock           deterministic word-echo engine for tests");
    println!("  ov-genai       single-stage openvino_genai.LLMPipeline; FastDraft + Prompt Lookup");
    println!("  ov-runtime     multi-stage stateful KV cache; pre-exported per-stage v3+ shards");
    println!("  ov-dist-spec   multi-stage spec decode (mask-based KV rewind); v5 shards");
    println!("  gemma4         Gemma 4 multi-stage (per-layer-type attn, KV-sharing, PLI); gemma4_cached_v1 shards");
    println!("  sparse-moe     Kimi K2.6 (AVX-512 int4 GEMM + Rust MLA shells) or MiniMax-M2 (OV-IR shells); single-stage top-k expert dispatch");
    println!("  qwen36-moe     Qwen3.6-35B-A3B staged chain (GatedDeltaNet + MoE); qwen3_5_moe IR-surgery shards");
    Ok(())
}

fn parse_addr(s: &str, default_host: &str) -> Result<(String, u16)> {
    if let Some(port) = s.strip_prefix(':') {
        return Ok((default_host.to_string(), port.parse().context("port")?));
    }
    let (h, p) = s
        .rsplit_once(':')
        .ok_or_else(|| anyhow!("address must be host:port (got {s:?})"))?;
    Ok((h.to_string(), p.parse().context("port")?))
}

/// Pick the OV plugin's CACHE_DIR for this worker.
///
/// Resolution order:
/// 1. If `arg` is `Some("")`, return `None` (explicit disable).
/// 2. If `arg` is `Some(path)`, return `Some(path)` (operator override).
/// 3. If `arg` is `None`, return the platform user-cache dir +
///    `cascadia/ov-cache` (Linux: `~/.cache/cascadia/ov-cache`,
///    macOS: `~/Library/Caches/cascadia/ov-cache`, Windows:
///    `%LOCALAPPDATA%/cascadia/ov-cache`).
/// 4. If even the platform default isn't resolvable, return `None`
///    (the OV plugin then runs without a cache — pre-PR behaviour).
///
/// The implicit default exists because a cold ov-genai LLMPipeline
/// compile on Intel iGPU takes ~20 s for a 1.7 B model (measured on
/// Lunar Lake), versus ~1 s when the cache is warm. PowerInfer's
/// SmallThinker fork ships an equivalent default in `llama-cli`. We
/// match that operator UX.
fn resolve_ov_cache_dir(arg: Option<&str>) -> Option<String> {
    match arg {
        Some("") => None,
        Some(p) => Some(p.to_string()),
        None => dirs::cache_dir().map(|p| {
            p.join("cascadia")
                .join("ov-cache")
                .to_string_lossy()
                .into_owned()
        }),
    }
}

/// True when `device` names an OpenVINO NPU plugin (e.g. "NPU", "NPU.0").
/// NPU-only plugin properties error if passed to a non-NPU plugin, so the
/// gate below strips them unless this returns true. Matching the issue #13
/// contract, the check is a case-insensitive `starts_with("NPU")`; compound
/// AUTO/HETERO device strings that merely mention NPU are deliberately not
/// treated as NPU targets.
fn device_is_npu(device: &str) -> bool {
    device.trim().to_ascii_uppercase().starts_with("NPU")
}

/// Translate the `--ov-*` / `--npu-*` performance flags on `args` into the
/// `(key, value)` OpenVINO plugin properties fed to the OV engine builders.
/// General hints apply to any engine; the NPU-only knobs are emitted only for
/// an NPU device on the `ov-genai` engine (the sole engine that routes them
/// through an `ov::genai` LLMPipeline — see the NPU block below). Returns
/// entries in a stable order (general hints first, NPU knobs last) so callers
/// and tests see a deterministic list.
fn ov_perf_properties(args: &WorkerArgs) -> Vec<(String, String)> {
    let mut props: Vec<(String, String)> = Vec::new();

    // General performance hints. PERFORMANCE_HINT / INFERENCE_PRECISION_HINT /
    // EXECUTION_MODE_HINT are plugin-agnostic ov::hint properties. NUM_STREAMS,
    // INFERENCE_NUM_THREADS (CPU-oriented) and ALLOW_AUTO_BATCHING (GPU/AUTO)
    // are only effective on the plugins that own them; they are opt-in, so we
    // forward the user's request verbatim and let OV accept or reject it rather
    // than second-guessing the target device here.
    if let Some(mode) = args.ov_performance_mode {
        props.push(("PERFORMANCE_HINT".into(), mode.as_ov().into()));
    }
    if let Some(prec) = &args.ov_inference_precision {
        props.push(("INFERENCE_PRECISION_HINT".into(), prec.clone()));
    }
    if let Some(n) = args.ov_num_streams {
        props.push(("NUM_STREAMS".into(), n.to_string()));
    }
    if let Some(n) = args.ov_num_threads {
        props.push(("INFERENCE_NUM_THREADS".into(), n.to_string()));
    }
    if args.ov_allow_auto_batching {
        props.push(("ALLOW_AUTO_BATCHING".into(), "true".into()));
    }
    if let Some(mode) = args.ov_execution_mode {
        props.push(("EXECUTION_MODE_HINT".into(), mode.as_ov().into()));
    }

    // NPU-only knobs. These are ov::genai LLMPipeline convenience keys — the
    // GenAI NPU path consumes MAX_PROMPT_LEN / MIN_RESPONSE_LEN and drives the
    // NPUW LLM pipeline (NPUW_LLM_PREFILL_CHUNK_SIZE). Only the ov-genai engine
    // routes properties through an LLMPipeline; every other OV engine compiles
    // via raw ov::Core::compile_model, which does not understand these keys (it
    // rejects or ignores them). So gate on BOTH the device (NPU) and the engine
    // (ov-genai). Keeping the gate here (single source of truth) means the
    // builders and the C++ shim never see an NPU key on a run that can't use it.
    if device_is_npu(&args.device) && matches!(args.engine, EngineKind::OvGenai) {
        if let Some(n) = args.npu_prefill_chunk_size {
            props.push(("NPUW_LLM_PREFILL_CHUNK_SIZE".into(), n.to_string()));
        }
        if let Some(n) = args.npu_max_prompt_len {
            props.push(("MAX_PROMPT_LEN".into(), n.to_string()));
        }
        if let Some(n) = args.npu_min_response_len {
            props.push(("MIN_RESPONSE_LEN".into(), n.to_string()));
        }
    }

    props
}

/// Load a whole-model chat template for ov-genai, tolerating both layouts:
/// `tokenizer/tokenizer_config.json` (HF subdir) and a root-level
/// `tokenizer_config.json` / `chat_template.jinja` (OV int4 exports).
fn ovgenai_chat_template(model: &str) -> cascadia_api::ChatTemplateConfig {
    let p = std::path::Path::new(model);
    let sub = cascadia_api::load_chat_template_config(p);
    if sub.template.is_some() {
        sub
    } else {
        cascadia_api::load_chat_template_config_at(p)
    }
}

/// Warn when a performance flag the user explicitly set will be silently
/// ignored for the chosen engine/device, so an ineffective flag is visible at
/// runtime instead of vanishing (mirrors the `--ffn-sparsity-capture-dir`
/// warning in `build_builder`'s sparse-moe arm).
/// `--cb` targeting the CPU plugin specifically.
///
/// Deliberately an exact match rather than a prefix: `AUTO`/`HETERO` strings
/// that may resolve to CPU are not flagged, because we cannot tell at
/// parse time what the plugin will pick.
fn cb_on_plain_cpu(args: &WorkerArgs) -> bool {
    args.cb && args.device.trim().eq_ignore_ascii_case("CPU")
}

fn warn_ignored_ov_perf_flags(args: &WorkerArgs) {
    // NPU LLM knobs apply only to ov-genai on an NPU device (see
    // `ov_perf_properties`). If the user set one but that gate won't fire, the
    // value is dropped — say so, or they chase a shape/truncation bug with no
    // signal pointing at the ignored flag.
    let npu_flag_set = args.npu_prefill_chunk_size.is_some()
        || args.npu_max_prompt_len.is_some()
        || args.npu_min_response_len.is_some();
    let npu_gate_open = device_is_npu(&args.device) && matches!(args.engine, EngineKind::OvGenai);
    if npu_flag_set && !npu_gate_open {
        tracing::warn!(
            engine = ?args.engine,
            device = %args.device,
            "ignoring --npu-* flags: NPU LLM knobs apply only with \
             --engine ov-genai on an NPU device"
        );
    }

    // qwen36-moe compiles with a fixed plugin config and receives no OV perf
    // properties (some hints break its IRs — see qwen36.rs). If the user set
    // general hints, warn they won't take effect on this engine.
    if matches!(args.engine, EngineKind::Qwen36Moe) && !ov_perf_properties(args).is_empty() {
        tracing::warn!(
            "ignoring --ov-* performance flags: the qwen36-moe engine compiles \
             with a fixed plugin config and does not apply them"
        );
    }

    // --cb on CPU is a narrow win and has a severe failure mode. Measured on
    // Lunar Lake across Phi-3.5-mini and Qwen3-8B: short prompts at concurrency
    // gain 1.6-2.1x, but a ~1200-token prompt collapses to ~0.2x — a five-fold
    // loss, on both models, and NOT recoverable by raising
    // --cb-max-batched-tokens (which does help on GPU). Nothing in the output
    // says why, so an operator serving RAG-style traffic from a CPU worker
    // would just see it get slower. Warn rather than reject: CPU + short
    // prompts is a legitimate configuration.
    if cb_on_plain_cpu(args) {
        tracing::warn!(
            device = %args.device,
            "--cb on CPU only pays off for short prompts under concurrency; \
             long-context workloads measure ~5x SLOWER than without --cb, and \
             --cb-max-batched-tokens does not recover it. Benchmark your own \
             prompt shape — see docs/engines/ov-genai.md"
        );
    }
}

/// Last path/repo segment of a model id, for the example commands below.
fn model_stem(model: &str) -> String {
    model
        .rsplit(['/', '\\'])
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("model")
        .to_string()
}

/// How to produce a model this engine can serve.
fn export_hint(model: &str, engine: EngineKind) -> String {
    let stem = model_stem(model);
    match engine {
        // Whole-model IR from Intel's exporter; `cascadia shard` can't make one.
        EngineKind::OvGenai => "  download a pre-exported *-int4-ov directory, or export one with\n  \
             `optimum-cli export openvino` (see docs/engines/ov-genai.md), then:\n    \
             cascadia run <ir-dir>"
            .to_string(),
        // Staged engines read a `cascadia shard` tree; the exporter picks the
        // per-family path, so name the engine that matches this tree.
        EngineKind::Gemma4 => format!(
            "  cascadia shard --model {model} --output-dir ./{stem}-2stage --num-stages 2\n    \
             cascadia worker --engine gemma4 --model ./{stem}-2stage ..."
        ),
        EngineKind::Qwen36Moe => format!(
            "  cascadia shard --model <qwen3.6 int4-ov dir> --output-dir ./{stem}-2stage --num-stages 2\n    \
             cascadia run ./{stem}-2stage --engine qwen36-moe"
        ),
        // sparse-moe consumes a manifest.json expert tree, not a shard tree.
        EngineKind::SparseMoe => {
            "  sparse-moe needs a manifest.json + per-expert tree — see\n  \
             docs/architectures/moe.md (e.g. tools/export_minimax_m2.py)."
                .to_string()
        }
        EngineKind::OvDistSpec => format!(
            "  cascadia shard --model {model} --output-dir ./{stem}-2stage --num-stages 2\n    \
             cascadia worker --engine ov-dist-spec --model ./{stem}-2stage --draft-model <ir-dir> ..."
        ),
        _ => format!(
            "  cascadia shard --model {model} --output-dir ./{stem}-1stage --num-stages 1\n    \
             cascadia run ./{stem}-1stage --engine ov-runtime"
        ),
    }
}

/// Why a `--model` was rejected, and what to do instead. Engines load a
/// pre-exported model from a directory; only `cascadia shard` downloads.
///
/// `org/name` is ambiguous — it is a valid HF repo id *and* a valid relative
/// path — so don't guess: say both, and give the export command either way.
fn missing_model_error(model: &str, engine: EngineKind) -> anyhow::Error {
    let hint = export_hint(model, engine);
    anyhow!(
        "no model directory at {model}\n\
         \n\
         If that is a path, it does not exist. If it is a HuggingFace repo id:\n\
         engines load a pre-exported model from disk and never download or convert\n\
         one. Export first:\n\
         \n\
         {hint}"
    )
}

/// Reject a `--model` (or, where the engine uses it, a `--draft-model`) that
/// names nothing on disk, with the command to run instead. Without this an HF
/// repo id — which the old docs told people to pass — surfaces as a bare "model
/// not found", which reads like a missing file rather than a missing export.
fn preflight_model_path(args: &WorkerArgs) -> Result<()> {
    if matches!(args.engine, EngineKind::Mock) {
        return Ok(());
    }
    if !std::path::Path::new(&args.model).exists() {
        return Err(missing_model_error(&args.model, args.engine));
    }
    // Only these read --draft-model, and ov-dist-spec only on the driver (rank 0)
    // — a relay rank launched from the same argv must not be rejected for a draft
    // IR it never opens.
    let uses_draft = match args.engine {
        EngineKind::OvGenai => true,
        EngineKind::OvDistSpec => args.rank == 0,
        _ => false,
    };
    if uses_draft {
        if let Some(draft) = &args.draft_model {
            if !std::path::Path::new(draft).exists() {
                return Err(anyhow!(
                    "no draft-model directory at {draft}\n\
                     --draft-model takes a local OpenVINO IR directory (e.g. a FastDraft\n\
                     *-int8-ov download), not an HF repo id."
                ));
            }
        }
    }
    Ok(())
}

fn build_builder(args: &WorkerArgs) -> Result<Box<dyn Builder>> {
    warn_ignored_ov_perf_flags(args);
    match args.engine {
        EngineKind::Mock => Ok(Box::new(MockBuilder::new())),
        EngineKind::OvGenai => {
            if args.total != 1 {
                return Err(anyhow!("ov-genai is single-stage only; use --total 1"));
            }
            if args.draft_model.is_some() && args.prompt_lookup > 0 {
                return Err(anyhow!(
                    "--draft-model and --prompt-lookup are mutually exclusive"
                ));
            }
            if args.cb && (args.draft_model.is_some() || args.prompt_lookup > 0) {
                return Err(anyhow!(
                    "--cb is incompatible with --draft-model / --prompt-lookup"
                ));
            }
            let mut b = OvGenaiBuilder::new(&args.model, &args.device);
            if args.cb {
                b = b.with_continuous_batching(cascadia_ov_genai_shim::CbSchedulerConfig {
                    cache_size_gb: args.cb_cache_size,
                    max_num_seqs: args.cb_max_num_seqs,
                    max_num_batched_tokens: args.cb_max_batched_tokens,
                    dynamic_split_fuse: args.cb_dynamic_split_fuse,
                    enable_prefix_caching: args.cb_prefix_caching,
                });
            }
            if let Some(dir) = resolve_ov_cache_dir(args.ov_cache_dir.as_deref()) {
                b = b.with_cache_dir(&dir);
            }
            if let Some(prec) = &args.ov_kv_precision {
                b = b.with_kv_cache_precision(prec);
            }
            if let Some(group) = &args.ov_dyn_quant_group {
                b = b.with_dyn_quant_group(group);
            }
            b = b.with_ov_properties(ov_perf_properties(args));
            if let Some(draft) = &args.draft_model {
                let device = args
                    .draft_device
                    .clone()
                    .unwrap_or_else(|| args.device.clone());
                b = b.with_draft(draft, &device, args.spec_k);
            } else if args.prompt_lookup > 0 {
                b = b.with_prompt_lookup(args.prompt_lookup);
            }
            // If the model ships a chat template, the API renders it (honoring
            // enable_thinking) and the engine tells ov-genai to skip its own
            // internal apply. No template → leave ov-genai's apply on.
            b = b.with_prompt_pretemplated(ovgenai_chat_template(&args.model).template.is_some());
            Ok(Box::new(b))
        }
        EngineKind::OvRuntime => {
            let mut b = OvRuntimeBuilder::new(&args.model, args.rank, args.total, &args.device);
            if let Some(dev) = &args.prefill_device {
                b = b.with_prefill_device(dev);
            }
            b = b.with_chunked_prefill_disabled(args.no_chunked_prefill);
            b.packed_slots = args.packed_slots;
            b.packed_prefix = args.packed_prefix;
            b = b.with_prefill_parking(args.park_prefill);
            b = b.with_gemv_offload(args.gemv_offload);
            if let Some(dir) = resolve_ov_cache_dir(args.ov_cache_dir.as_deref()) {
                b = b.with_cache_dir(&dir);
            }
            if let Some(prec) = &args.ov_kv_precision {
                b = b.with_kv_cache_precision(prec);
            }
            if let Some(group) = &args.ov_dyn_quant_group {
                b = b.with_dyn_quant_group(group);
            }
            b = b.with_ov_properties(ov_perf_properties(args));
            Ok(Box::new(b))
        }
        EngineKind::Gemma4 => {
            let mut b = Gemma4Builder::new(&args.model, args.rank, args.total, &args.device);
            if let Some(dir) = resolve_ov_cache_dir(args.ov_cache_dir.as_deref()) {
                b = b.with_cache_dir(&dir);
            }
            if let Some(prec) = &args.ov_kv_precision {
                b = b.with_kv_cache_precision(prec);
            }
            if let Some(group) = &args.ov_dyn_quant_group {
                b = b.with_dyn_quant_group(group);
            }
            b = b.with_ov_properties(ov_perf_properties(args));
            Ok(Box::new(b))
        }
        EngineKind::OvDistSpec => {
            // Driver = rank 0 (needs --draft-model). Worker = rank > 0.
            if args.rank == 0 {
                let draft = args
                    .draft_model
                    .as_deref()
                    .ok_or_else(|| anyhow!("ov-dist-spec rank 0 requires --draft-model"))?;
                let mut b = OvDistSpecBuilder::new(&args.model, draft, &args.device, args.spec_k);
                if let Some(d) = &args.draft_device {
                    b = b.with_draft_device(d);
                }
                if let Some(dir) = &args.ov_cache_dir {
                    b = b.with_cache_dir(dir);
                }
                if let Some(prec) = &args.ov_kv_precision {
                    b = b.with_kv_cache_precision(prec);
                }
                if let Some(group) = &args.ov_dyn_quant_group {
                    b = b.with_dyn_quant_group(group);
                }
                b = b.with_ov_properties(ov_perf_properties(args));
                Ok(Box::new(b))
            } else {
                let mut b =
                    OvDistSpecWorkerBuilder::new(&args.model, args.rank, args.total, &args.device);
                if let Some(dir) = &args.ov_cache_dir {
                    b = b.with_cache_dir(dir);
                }
                if let Some(prec) = &args.ov_kv_precision {
                    b = b.with_kv_cache_precision(prec);
                }
                if let Some(group) = &args.ov_dyn_quant_group {
                    b = b.with_dyn_quant_group(group);
                }
                b = b.with_ov_properties(ov_perf_properties(args));
                Ok(Box::new(b))
            }
        }
        EngineKind::SparseMoe => {
            let mut cfg = SparseMoEBuilderConfig::new(&args.model, &args.device)
                .with_rank(args.rank, args.total)
                .with_kv_prefix_cache_size(args.kv_prefix_cache_size);
            if let Some(dir) = resolve_ov_cache_dir(args.ov_cache_dir.as_deref()) {
                cfg.cache_dir = Some(dir);
            }
            cfg.ov_properties = ov_perf_properties(args);
            cfg.top_k_override = args.top_k_override;
            cfg.routing_threshold = args.routing_threshold;
            cfg.max_cached_experts = args.max_cached_experts;
            cfg.ffn_sparsity_threshold = args.ffn_sparsity_threshold;
            cfg.ffn_axpy_down = args.ffn_axpy_down;
            cfg.ffn_axpy_prebuild = args.ffn_axpy_prebuild;
            cfg.ffn_sparsity_thresholds_file = args.ffn_sparsity_thresholds_file.clone();
            cfg.ffn_sparsity_capture_dir = args.ffn_sparsity_capture_dir.clone();
            // Issue #38: capture surfaces silu(gate) via the AXPY
            // scratch — if the user asked to capture but didn't ask
            // for AXPY, warn (the capture will silently be empty
            // because the non-AXPY path doesn't expose silu(gate)).
            if args.ffn_sparsity_capture_dir.is_some() && !args.ffn_axpy_down {
                tracing::warn!(
                    "--ffn-sparsity-capture-dir requires --ffn-axpy-down to surface silu(gate); \
                     capture will be empty without it"
                );
            }
            // N-gram speculative decode (sparse-moe single-stage).
            // The `--prompt-lookup` flag is the canonical opt-in (it
            // already drives the ov-genai prompt-lookup path); when
            // set on sparse-moe single-stage, we enable n-gram spec
            // decode with K = max(prompt_lookup, spec_k). On
            // multi-stage configs this is ignored — see
            // `SparseMoEBuilder::build` for the warning.
            let spec_k = if args.prompt_lookup > 0 {
                Some(args.spec_k.max(args.prompt_lookup))
            } else {
                None
            };
            if let Some(k) = spec_k {
                cfg = cfg.with_spec_decode_k(k);
            }
            Ok(Box::new(SparseMoEBuilder::new(cfg)))
        }
        EngineKind::Qwen36Moe => Ok(Box::new(
            Qwen36Builder::new(&args.model, &args.device).with_rank(args.rank, args.total),
        )),
    }
}

/// Per-connection accept loop that calls `set_nodelay(true)` on every
/// accepted stream and serves it via hyper directly. axum 0.7's
/// `axum::serve(TcpListener, app)` doesn't expose any TCP_NODELAY
/// hook (no `Listener` trait until axum 0.8), so we drive hyper
/// ourselves. Per-token SSE chunks then arrive on the wire as
/// individual ~220 B packets at the engine's natural ~12 tok/s
/// cadence instead of being Nagle-aggregated into ~3500 B bursts
/// every ~1 s — the original demo's bursty token feel was driven
/// almost entirely by this default.
async fn serve_with_nodelay(listener: tokio::net::TcpListener, app: axum::Router) -> Result<()> {
    use hyper::server::conn::http1;
    use hyper_util::rt::TokioIo;
    use std::convert::Infallible;
    use tower::Service;

    loop {
        let (stream, _addr) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "accept failed; retrying after 50ms");
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                continue;
            }
        };
        if let Err(e) = stream.set_nodelay(true) {
            tracing::warn!(error = %e, "set_nodelay failed on accepted stream");
        }
        let io = TokioIo::new(stream);
        let app_for_conn = app.clone();
        // Adapt axum::Router (a tower::Service<Request>) to a hyper
        // service. We clone the router PER REQUEST inside the Fn
        // closure (hyper's service_fn requires Fn, not FnMut, but
        // tower::Service::call needs &mut self — clone is the cheap
        // workaround since axum::Router is Arc-backed).
        let svc = hyper::service::service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
            let mut router = app_for_conn.clone();
            let req = req.map(axum::body::Body::new);
            let fut = router.call(req);
            async move {
                let res = fut.await;
                Ok::<_, Infallible>(res.unwrap_or_else(|e: Infallible| match e {}))
            }
        });
        tokio::spawn(async move {
            if let Err(e) = http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, svc)
                .await
            {
                tracing::debug!(error = %e, "connection serve ended");
            }
        });
    }
}

/// Which signal triggered shutdown. Logged for postmortem context so an
/// operator looking at worker logs can distinguish an orchestrator
/// `SIGTERM` (typical) from an interactive `Ctrl-C` (rare in prod).
#[derive(Debug, Clone, Copy)]
enum ShutdownSignal {
    Sigterm,
    Sigint,
}

impl std::fmt::Display for ShutdownSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sigterm => f.write_str("SIGTERM"),
            Self::Sigint => f.write_str("SIGINT"),
        }
    }
}

/// Pre-install signal handlers (SIGTERM via `tokio::signal::unix` +
/// SIGINT via `tokio::signal::ctrl_c`) and return a future that resolves
/// to whichever signal fires first.
///
/// **Why pre-install rather than install-on-poll inside `select!`**:
/// `signal(SignalKind::terminate())` registers the handler with the
/// async signal driver — until that call completes, the process uses
/// the default disposition (terminate). If we installed it on the first
/// poll of the select! arm, a signal arriving in the narrow window
/// between `listener.bind()` and the first `select!` poll would kill
/// the worker before any graceful path runs. Pre-installing closes
/// that gap.
///
/// On unix: catches both SIGTERM (operator `kill <pid>` / systemd
/// stop / k8s pod termination) and SIGINT (Ctrl-C).
///
/// On non-unix (Windows): only Ctrl-C is supported by tokio
/// (`SignalKind` is unix-only). A Windows `taskkill /F` will NOT
/// trigger graceful close — it sends SIGKILL-equivalent and the
/// process exits immediately. That's a Windows-platform limitation,
/// not a bug in this code; document accordingly so on-call doesn't
/// chase it as a leak.
#[cfg(unix)]
fn install_shutdown_signal_handler(
) -> std::io::Result<impl std::future::Future<Output = ShutdownSignal>> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate())?;
    Ok(async move {
        tokio::select! {
            _ = sigterm.recv() => ShutdownSignal::Sigterm,
            _ = tokio::signal::ctrl_c() => ShutdownSignal::Sigint,
        }
    })
}

#[cfg(not(unix))]
fn install_shutdown_signal_handler(
) -> std::io::Result<impl std::future::Future<Output = ShutdownSignal>> {
    Ok(async {
        let _ = tokio::signal::ctrl_c().await;
        ShutdownSignal::Sigint
    })
}

/// Reject phase-split flag combinations before a worker binds any socket. Pure
/// function of the args so it is unit-testable (the guards were inline in the
/// async `cmd_worker`, reachable only by running a full worker).
fn validate_worker_runtime_flags(args: &WorkerArgs) -> Result<()> {
    // Fail loud, not silent: the phase-split flags are load-bearing when
    // given, and only the ov-runtime engine implements them.
    if (args.prefill_device.is_some()
        || args.no_chunked_prefill
        || args.park_prefill
        || args.gemv_offload)
        && args.engine != EngineKind::OvRuntime
    {
        return Err(anyhow!(
            "--prefill-device / --no-chunked-prefill / --park-prefill / --gemv-offload \
             require --engine ov-runtime (the chunked-prefill phase split lives in the \
             static-KV runtime path)"
        ));
    }
    if (args.prefill_device.is_some() || args.park_prefill) && args.no_chunked_prefill {
        return Err(anyhow!(
            "--prefill-device / --park-prefill conflict with --no-chunked-prefill"
        ));
    }
    if args.packed_prefix > 0 && args.packed_slots == 0 {
        return Err(anyhow!(
            "--packed-prefix requires --packed-slots (the shared prefix lives in the packed \
             KV window)"
        ));
    }
    // Prefix reuse is first-stage-only state. Only rank 0 admits requests, and
    // admission is what populates the shared prefix region; a relay stage never
    // admits, so its `prefix_valid` stays 0 and it clamps every plan row's reuse
    // length to 0. Rank 0 would then skip prefilling the reused tokens while the
    // downstream stages hold no KV for them — wrong tokens, no error, on any rank.
    if args.packed_prefix > 0 && args.total != 1 {
        return Err(anyhow!(
            "--packed-prefix is single-stage only (--total 1); a relay stage cannot populate the \
             shared prefix, so it would open zero shared columns for the prompt tokens rank 0 \
             skipped prefilling and silently corrupt multi-stage output. Drop --packed-prefix to \
             keep packed multi-stage decode, or run the worker as a single stage"
        ));
    }
    // Packed multi-slot decode lives in the ov-runtime static (NPU-target) path.
    if args.packed_slots > 0 {
        if args.engine != EngineKind::OvRuntime {
            return Err(anyhow!(
                "--packed-slots requires --engine ov-runtime (packed multi-slot decode is a \
                 static-KV path feature)"
            ));
        }
        if args.packed_slots < 2 {
            return Err(anyhow!("--packed-slots must be 0 (off) or >= 2"));
        }
        // Multi-stage packed is allowed again: the #122 wedge was driver
        // starvation on rank 0 (sync engine-mutex blocking pinned every
        // tokio worker, so the token-frame reply could neither be read nor
        // timed out), fixed in cascadia-runner (tokio workers never block on
        // the engine mutex: polls park on a failed try_lock, cancels/drops
        // defer, and submits go off-worker via spawn_blocking) plus
        // deadlined+poisoning reply recvs and an on-wire NACK in the packed
        // exchange. Every stage must run the same --packed-slots value
        // (baked into the packed IR shape).
    }
    // Continuous batching (#20) lives in the ov-genai CBP path only. It is a
    // different mechanism to --packed-slots above: OV's paged attention on the
    // CPU/GPU plugins, versus our sequence-packing on the NPU static path.
    if (args.cb
        || args.cb_cache_size > 0
        || args.cb_max_num_seqs > 0
        || args.cb_max_batched_tokens > 0
        || args.cb_dynamic_split_fuse.is_some()
        || args.cb_prefix_caching.is_some())
        && args.engine != EngineKind::OvGenai
    {
        return Err(anyhow!(
            "--cb / --cb-* flags require --engine ov-genai (continuous batching is \
             served by ov-genai's ContinuousBatchingPipeline)"
        ));
    }
    if !args.cb
        && (args.cb_cache_size > 0
            || args.cb_max_num_seqs > 0
            || args.cb_max_batched_tokens > 0
            || args.cb_dynamic_split_fuse.is_some()
            || args.cb_prefix_caching.is_some())
    {
        return Err(anyhow!("--cb-* tuning flags require --cb"));
    }
    // Paged attention is a CPU/GPU-plugin capability; on NPU ov-genai serves
    // the static NPUW pipeline. Without this gate the operator waits out a
    // full model compile only to get a raw OpenVINO exception. (NPU operators
    // wanting concurrency use --engine ov-runtime --packed-slots instead.)
    if args.cb && device_is_npu(&args.device) {
        return Err(anyhow!(
            "--cb requires a CPU or GPU device; NPU serves ov-genai's static NPUW \
             pipeline and cannot continuous-batch — use --engine ov-runtime with \
             --packed-slots for NPU concurrency"
        ));
    }
    Ok(())
}

async fn cmd_worker(args: WorkerArgs) -> Result<()> {
    if args.rank >= args.total {
        return Err(anyhow!(
            "--rank must be in [0, {}); got {}",
            args.total,
            args.rank
        ));
    }
    validate_worker_runtime_flags(&args)?;
    let is_first = args.rank == 0;
    let is_last = args.rank == args.total - 1;

    // Only rank 0 reaches the API bind; every other rank returns from the
    // relay loop first. Say so rather than dropping the flag silently.
    if args.api.is_some() && !is_first {
        warn!(rank = args.rank, "--api is ignored on ranks other than 0");
    }

    info!(
        engine = ?args.engine,
        rank = args.rank,
        total = args.total,
        device = %args.device,
        model = %args.model,
        "cascadia worker starting"
    );

    let (listen_host, listen_port) = parse_addr(&args.listen, "0.0.0.0")?;

    let upstream = if is_first {
        None
    } else {
        Some(PeerEndpoint::new(listen_host.clone(), listen_port))
    };
    let downstream = if is_last {
        None
    } else {
        let next = args
            .next
            .as_deref()
            .ok_or_else(|| anyhow!("--next is required for non-last stages"))?;
        let (h, p) = parse_addr(next, "127.0.0.1")?;
        Some(PeerEndpoint::new(h, p))
    };
    let peers = PeerLayout {
        upstream,
        downstream,
    };

    let shard = ShardSpec {
        model_id: args.model.clone(),
        layer_start: args.layer_start,
        layer_end: args.layer_end,
        total_layers: 0,
        device: args.device.clone(),
        is_first_stage: is_first,
        is_last_stage: is_last,
        tp_size: 1,
        tp_rank: 0,
    };

    preflight_model_path(&args)?;
    let builder = build_builder(&args)?;
    let runner = Arc::new(Runner::new(builder));
    let listen = if !is_first {
        Some((listen_host.as_str(), listen_port))
    } else {
        None
    };
    runner.start_with_listen(peers, shard, listen).await?;

    // Probe listener: bind a TCP socket on listen_port so the
    // coordinator's latency probe loop has something to handshake
    // against. Real engines bind this port themselves (for activation
    // relay) via Engine::configure_listen, but the mock engine's impl
    // is a no-op — without this, latency probes against mock-engine
    // workers find a closed port and the matrix stays empty in the
    // dashboard demo. If the engine already bound the port, `bind`
    // fails with AddrInUse and we silently step aside (the engine's
    // listener handles probes identically at the TCP layer).
    let probe_addr = format!("0.0.0.0:{listen_port}");
    match tokio::net::TcpListener::bind(&probe_addr).await {
        Ok(listener) => {
            info!(addr = %probe_addr, "probe listener bound");
            tokio::spawn(async move {
                loop {
                    match listener.accept().await {
                        Ok(_) => {
                            // Drop the connection immediately — the
                            // probe only needs the connect handshake.
                        }
                        Err(e) => {
                            tracing::debug!(error = %e, "probe accept failed");
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                    }
                }
            });
        }
        Err(e) => {
            tracing::debug!(
                error = %e,
                addr = %probe_addr,
                "probe listener could not bind; engine likely owns the port"
            );
        }
    }

    // Every worker advertises itself via mDNS — not just rank 0 — so the
    // coordinator's dashboard /api/topology can render the full cluster
    // and not just self. Best-effort: a host without a working multicast
    // path (CI sandbox, restricted LAN) still serves; the dashboard just
    // shows fewer nodes. We bind `_discovery` for the rest of this
    // function so its Drop unregisters the mDNS record cleanly on
    // shutdown (relay loop or `serve_with_nodelay`).
    let topology = cascadia_topology::Topology::new();
    let engines = if !args.advertise_engines.is_empty() {
        args.advertise_engines.clone()
    } else {
        vec![engine_name(args.engine).to_owned()]
    };
    let device = args
        .advertise_device
        .clone()
        .unwrap_or_else(|| args.device.clone());
    // sysinfo + hostname do blocking syscalls; gather them off the async
    // runtime thread so they don't stall worker startup.
    let specs = tokio::task::spawn_blocking(gather_node_specs)
        .await
        .unwrap_or_default();
    // Advertise the API/dashboard port separately from the relay port so
    // the dashboard can show a node's reachable address (the relay `port`
    // isn't an HTTP endpoint). None for relay-only stages.
    // Only rank 0 serves an API; relay ranks return before the API block, so a
    // --api passed to them (one systemd template covers every rank) must not be
    // advertised as an endpoint the dashboard can't reach.
    let api_port = is_first
        .then_some(args.api.as_deref())
        .flatten()
        .and_then(|a| parse_addr(a, "0.0.0.0").ok())
        .map(|(_, p)| p);
    let self_node = cascadia_topology::NodeInfo {
        node_id: format!("{}-r{}", specs.hostname, args.rank),
        host: cascadia_discovery::local_ip().to_string(),
        port: listen_port,
        api_port,
        namespace: "default".to_owned(),
        device,
        memory_mb: specs.memory_mb,
        cpu_model: specs.cpu_model,
        cpu_cores: specs.cpu_cores,
        os: specs.os,
        engines,
        last_seen: 0.0,
    };
    topology.add_node(self_node.clone());
    let mut discovery = cascadia_discovery::DiscoveryService::new(topology.clone(), "default");
    if let Err(e) = discovery.start(self_node.clone()) {
        tracing::warn!(error = %e, "mDNS discovery failed to start; cluster topology may be incomplete");
    }
    let _discovery = discovery;

    // Self-heartbeat: re-insert our own NodeInfo every 2 s so last_seen
    // stays current in the local topology even when no mDNS event has
    // fired. Without this the dashboard's "live" indicator goes cold
    // after FRESH_THRESHOLD_S because add_node only sets last_seen once.
    // mDNS-discovered peers refresh on each ServiceResolved event from
    // the daemon (TTL-driven, slower); a generous freshness window on
    // the frontend covers the gap.
    let topology_for_heartbeat = topology.clone();
    let self_id_for_heartbeat = self_node.node_id.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(2));
        tick.tick().await; // skip the immediate first tick — we already added the node above.
        loop {
            tick.tick().await;
            // Only the timestamp changes — touch() updates last_seen in
            // place instead of re-cloning + re-inserting the whole NodeInfo.
            topology_for_heartbeat.touch(&self_id_for_heartbeat);
        }
    });

    // Latency probe loop: every 5 s open a TCP connect to each peer's
    // advertised port and measure round-trip time. Populates the
    // dashboard's edge matrix with the cluster's actual measured
    // latencies (which is the whole topology pitch — we store edges
    // exo doesn't even have).
    //
    // Only outgoing edges from self are populated here, so the matrix
    // has one row filled per coordinator. A symmetric N×N view would
    // require cross-host measurement sharing (gossip or query-by-id);
    // out of scope for the MVP.
    let topology_for_probe = topology.clone();
    let self_id_for_probe = self_node.node_id.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            tick.tick().await;
            // Snapshot just (id, host, port) — no full NodeInfo clone — and
            // probe all peers concurrently with join_all rather than an
            // unbounded tokio::spawn per peer per tick.
            let peers: Vec<(String, String, u16)> = topology_for_probe
                .nodes()
                .into_iter()
                .filter(|n| n.node_id != self_id_for_probe)
                .map(|n| (n.node_id, n.host, n.port))
                .collect();
            let results = futures::future::join_all(
                peers
                    .into_iter()
                    .map(|(id, host, port)| async move { (id, probe_peer(&host, port).await) }),
            )
            .await;
            for (dst_id, latency) in results {
                if let Some(latency_ms) = latency {
                    // record_latency, not measure(.., 0.0), so a future
                    // bandwidth probe on the same edge isn't clobbered.
                    topology_for_probe.record_latency(
                        self_id_for_probe.clone(),
                        dst_id,
                        latency_ms,
                    );
                }
            }
        }
    });

    if !is_first {
        info!("entering relay loop");
        let r = runner.clone();
        let exit = tokio::task::spawn_blocking(move || r.run_relay_loop()).await;
        // A connection-fatal exit means the peer link is dead and can't be
        // re-accepted without a rebuild. Return Err so the process exits
        // non-zero and systemd's `Restart=on-failure` rebuilds the stage,
        // rather than exit 0 (a "success" systemd won't restart) and leave
        // the pipeline a stage short. A clean SlotEmpty exit returns Ok.
        match exit {
            Ok(cascadia_runner::RelayExit::ConnectionFatal) => {
                return Err(anyhow!("relay loop exited: peer link dead; rebuild needed"));
            }
            Ok(cascadia_runner::RelayExit::SlotEmpty) => return Ok(()),
            Err(join_err) => return Err(anyhow!("relay loop task panicked: {join_err}")),
        }
    }

    if let Some(api_addr) = &args.api {
        let (api_host, api_port) = parse_addr(api_addr, "0.0.0.0")?;
        // Read the model's HF chat_template + bos/eos tokens at startup
        // so /v1/chat/completions can render multi-turn prompts in the
        // exact format the model was trained on. Falls back gracefully
        // to the legacy "role: content" join if the file or fields are
        // missing.
        let chat_template = match args.engine {
            // ov-genai: render the template API-side so enable_thinking is
            // honored; the engine sets apply_chat_template=false so ov-genai
            // doesn't double-wrap (build_builder mirrors via with_prompt_pretemplated).
            EngineKind::OvGenai => ovgenai_chat_template(&args.model),
            // qwen36 surgery trees keep chat_template.jinja at the model
            // root with no tokenizer_config.json.
            EngineKind::Qwen36Moe => {
                cascadia_api::load_chat_template_config_at(std::path::Path::new(&args.model))
            }
            // sparse-moe exports (dsv4) keep the tokenizer files and
            // chat_template.jinja at the model root, not in a tokenizer/
            // subdir. Read the root first; fall back to the subdir layout so
            // other sparse-moe models (MiniMax-M2) that ship a tokenizer/
            // subdir still resolve their template.
            EngineKind::SparseMoe => {
                let root =
                    cascadia_api::load_chat_template_config_at(std::path::Path::new(&args.model));
                if root.template.is_some() {
                    root
                } else {
                    cascadia_api::load_chat_template_config(std::path::Path::new(&args.model))
                }
            }
            _ => cascadia_api::load_chat_template_config(std::path::Path::new(&args.model)),
        };
        if chat_template.template.is_some() {
            info!(
                model = %args.model,
                bos = chat_template.bos_token.is_some(),
                eos = chat_template.eos_token.is_some(),
                "loaded chat_template from tokenizer_config.json"
            );
        } else {
            info!(
                model = %args.model,
                "no chat_template found; /v1/chat/completions will use legacy formatting"
            );
        }
        let mut cfg = cascadia_api::Config::default();
        cfg.chat_template = chat_template;
        // ov-genai owns native templating: render the template API-side only for
        // the thinking-OFF path (engine sets apply_chat_template=false then);
        // thinking-ON stays on ov-genai's native template, untouched.
        cfg.defer_template_on_thinking = matches!(args.engine, EngineKind::OvGenai);
        let max_concurrent = cfg.max_concurrent_requests as u64;
        // Shared live counters: the API bumps them on the chat hot path,
        // the dashboard's /api/stats reads them — same Arc, so the cluster
        // view updates as prompts run.
        let api_stats = Arc::new(cascadia_api::ApiStats::default());
        // Served model name: explicit override, else the basename of --model
        // (split on both separators so a Windows path works on any build).
        let served_model = args.served_model_name.clone().unwrap_or_else(|| {
            args.model
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or(args.model.as_str())
                .to_string()
        });
        let api_router = cascadia_api::make_router_with_stats(
            runner.clone(),
            served_model,
            cfg,
            api_stats.clone(),
        );

        // `topology` was populated above (every worker advertises +
        // browses) so by the time the dashboard binds, mDNS may already
        // have discovered other ranks on the LAN.
        let dash_state = cascadia_dashboard::DashboardState {
            topology,
            stats: api_stats,
            max_concurrent,
        };
        // Compose: OpenAI-compat routes (/v1/*, /health) stay at root for
        // backward compatibility with existing clients; dashboard-internal
        // routes live at /api/* plus the SPA (when the `dashboard-embed`
        // feature is on) at /. The dashboard router carries the SPA
        // fallback, so it must be merged second.
        let app = api_router.merge(cascadia_dashboard::make_router(dash_state));
        let listener = tokio::net::TcpListener::bind((api_host.as_str(), api_port)).await?;
        info!(host = %api_host, port = api_port, "API + dashboard serving");
        // NODELAY-on-accept wrapper. tokio's TcpStream defaults to
        // NODELAY=false (Nagle on); for SSE streaming small per-token
        // chunks, Nagle aggregates them into ~3500 B bursts every
        // ~1 s — the demo experiences this as bursty token arrival
        // in the browser. Wrap the listener so every accepted stream
        // gets set_nodelay(true). Per-event flush on the body side
        // is in cascadia_api::stream_completion (Body::from_stream of
        // raw bytes, not the axum::Sse wrapper which has its own
        // KeepAlive batching).
        // Pre-install the SIGTERM/SIGINT handler BEFORE the select! so
        // there's no window between bind() and the first poll where a
        // signal would default-terminate the worker. If install fails
        // we surface it instead of silently falling back to SIGINT-only:
        // a worker that quietly ignores SIGTERM is a graveyard for
        // orphaned processes under systemd / k8s.
        let shutdown_signal =
            install_shutdown_signal_handler().context("install_shutdown_signal_handler")?;
        // Race the axum serve against SIGTERM/SIGINT. Without this,
        // `kill <pid>` exits the tokio runtime before `Runner::close()`
        // runs, which silently drops any teardown work close() does
        // (flushing caches, draining transports, etc.). On signal we
        // fall through to runner.close() below.
        tokio::select! {
            res = serve_with_nodelay(listener, app) => {
                res.context("serve_with_nodelay")?;
            }
            sig = shutdown_signal => {
                info!(
                    signal = %sig,
                    pid = std::process::id(),
                    "shutdown signal received; running graceful close"
                );
            }
        }
        // Bounded close so a stuck peer / transport doesn't force the
        // operator to escalate `kill` → `kill -9`. `Runner::close()` is
        // sync (parking_lot::Mutex) and may block_on async transport
        // teardown internally; dispatch via spawn_blocking so the timer
        // can actually fire, then await with a deadline. On timeout we
        // log loudly and return — the process exit will SIGKILL any
        // straggler threads, which is the right outcome at that point.
        const SHUTDOWN_CLOSE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
        let r = runner.clone();
        let close_task = tokio::task::spawn_blocking(move || r.close());
        match tokio::time::timeout(SHUTDOWN_CLOSE_TIMEOUT, close_task).await {
            Ok(Ok(())) => info!("runner.close() complete"),
            Ok(Err(join_err)) => tracing::warn!(
                error = %join_err,
                "runner.close() task panicked"
            ),
            Err(_) => tracing::warn!(
                timeout_s = SHUTDOWN_CLOSE_TIMEOUT.as_secs(),
                "runner.close() exceeded timeout; abandoning teardown"
            ),
        }
        return Ok(());
    }

    // No API — read prompts from stdin.
    info!("stdin mode: type a prompt and press enter");
    use tokio::io::{AsyncBufReadExt, BufReader};
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let mut counter = 0usize;
    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        counter += 1;
        let task = GenerationTask {
            task_id: format!("stdin-{counter}"),
            prompt: line,
            max_tokens: args.max_tokens,
            temperature: 0.0,
            logprobs: 0,
            sampling: cascadia_types::SamplingParams::default(),
            enable_thinking: false,
            trust_remote_code: false,
        };
        let mut stream = runner.generate(task)?;
        while let Some(chunk) = stream.next().await {
            print!("{}", chunk.text);
            if chunk.is_final {
                println!();
            }
        }
    }
    Ok(())
}

/// The bundled Python exporter, included into the binary at build time.
/// Written to a temp file at runtime and invoked as a subprocess.
/// `CARGO_MANIFEST_DIR` is `<workspace>/crates/cascadia-cli`; jumping two
/// levels up reaches the workspace root reliably on both Unix and Windows
/// (raw `../../../tools/...` mixes path separators on Windows and breaks
/// `include_str!`).
const EXPORT_SCRIPT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/export_shards.py"
));

/// The exporter's pinned deps, compiled in beside the exporter itself so the
/// pins we print always match the exporter in *this* binary — a release bundle
/// carries no requirements.txt to point at.
const EXPORT_REQUIREMENTS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/requirements.txt"
));

/// `pip install` line for the exporter deps, rendered from EXPORT_REQUIREMENTS
/// and aimed at `python`'s own pip — installing into whatever `pip` happens to
/// be on PATH is how you end up with the deps in the wrong interpreter.
///
/// Every entry is double-quoted: `>=`/`<`/`[extras]` need quoting, and double
/// quotes are the only form that survives sh, zsh, PowerShell and cmd.exe alike.
pub(crate) fn export_pip_install_line(python: &str) -> String {
    let pkgs: Vec<String> = EXPORT_REQUIREMENTS
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim()) // drop comments
        .filter(|l| !l.is_empty())
        .map(|p| format!("\"{p}\""))
        .collect();
    // Quote the interpreter only when it needs it. No PowerShell `&` call
    // operator: `cfg!(windows)` is the build target, not the shell the user is
    // in, and `&` is a syntax error in cmd.exe and Git Bash. PowerShell users
    // with a spaced path add their own `&`; every other shell takes this as-is.
    let py = if python.contains(char::is_whitespace) {
        format!("\"{python}\"")
    } else {
        python.to_string()
    };
    format!("{py} -m pip install {}", pkgs.join(" "))
}

/// Sibling modules the exporter imports at runtime: the dedicated Gemma-4
/// exporter (dispatched for gemma4 models) and the short-alias registry.
/// They must be written next to export_shards.py in the temp dir so its
/// `from export_gemma4 import ...` / `from model_aliases import ...` resolve.
const GEMMA4_SCRIPT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/export_gemma4.py"
));
const ALIASES_SCRIPT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/model_aliases.py"
));
/// Qwen3.5/3.6 hybrid-MoE exporter (IR surgery on the official int4 IR),
/// dispatched by export_shards.py for model_type qwen3_5_moe.
const QWEN36_SCRIPT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/qwen36_surgery/export_qwen36_moe.py"
));
/// Gemma-4 VLM-IR -> text-only surgery exporter (grafts a text front-end and
/// slices at decoder boundaries on the OpenVINO IR, no torch), dispatched by
/// export_shards.py when --model is a gemma-4 OpenVINO-IR dir.
const GEMMA4_TEXT_SCRIPT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/gemma4_surgery/export_gemma4_text.py"
));

/// Assemble the flag list passed to the embedded Python exporter (the args
/// after `python -u <script>`). Pure + side-effect-free so it's unit-testable
/// without spawning Python. Emits `--model --output-dir --num-stages
/// --quantization --target --default-dtype`, then `--static-seq` /
/// `--static-context` (NPU path only — on cpu-gpu they are simply not
/// forwarded; `cmd_shard` warns about the ignore), then `--layer-split` /
/// `--stage` when set.
///
/// Fails fast on the stable wire-format rule (NPU feeds KV as f16, so the
/// default dtype must be fp16); the exporter owns the static-shape rules
/// (static-seq == 1, context > seq) since it may relax them (e.g. chunked
/// prefill) without a CLI change.
fn shard_exporter_flags(args: &ShardArgs) -> Result<Vec<String>> {
    if args.target == ShardTarget::Npu && args.default_dtype != ShardDtype::Fp16 {
        return Err(anyhow!("--default-dtype must be fp16 for --target npu"));
    }
    if args.static_prefill_seq != 0 {
        if args.target != ShardTarget::Npu {
            return Err(anyhow!("--static-prefill-seq requires --target npu"));
        }
        if args.static_prefill_seq == 1 {
            return Err(anyhow!(
                "--static-prefill-seq must be 0 (off) or >= 2 (the chunked-prefill window)"
            ));
        }
        if args.static_prefill_seq > args.static_context.saturating_sub(1) {
            return Err(anyhow!(
                "--static-prefill-seq ({}) must be <= --static-context - 1 ({}) — a chunk \
                 wider than the KV window would evict its own tokens mid-chunk",
                args.static_prefill_seq,
                args.static_context.saturating_sub(1)
            ));
        }
    }
    let mut flags = vec![
        "--model".into(),
        args.model.clone(),
        "--output-dir".into(),
        args.output_dir.clone(),
        "--num-stages".into(),
        args.num_stages.to_string(),
        "--quantization".into(),
        args.quantization.as_arg().into(),
        "--target".into(),
        args.target.as_arg().into(),
        "--default-dtype".into(),
        args.default_dtype.as_arg().into(),
    ];
    if args.target == ShardTarget::Npu {
        flags.push("--static-seq".into());
        flags.push(args.static_seq.to_string());
        flags.push("--static-context".into());
        flags.push(args.static_context.to_string());
        if args.static_prefill_seq != 0 {
            flags.push("--static-prefill-seq".into());
            flags.push(args.static_prefill_seq.to_string());
        }
    }
    if let Some(s) = &args.layer_split {
        flags.push("--layer-split".into());
        flags.push(s.clone());
    }
    if let Some(s) = args.stage {
        flags.push("--stage".into());
        flags.push(s.to_string());
    }
    Ok(flags)
}

/// The imports the bundled exporter needs. Shared with `cascadia doctor` so the
/// two can't disagree about what "export packages present" means.
pub(crate) const EXPORT_IMPORTS: &str =
    "import torch, openvino, transformers, safetensors, huggingface_hub";

/// A Python interpreter and what it can do.
#[derive(Debug)]
pub(crate) struct PythonEnv {
    pub path: String,
    /// Version summary from the dependency probe; `None` if the imports failed
    /// (or `check_deps` was false), so callers can skip probing a second time.
    pub deps: Option<String>,
}

/// True only if the interpreter runs AND exits 0. Spawning is not enough: the
/// Windows Store `python3.exe` alias spawns, prints "Python was not found" and
/// exits 9009 — treating that as an interpreter is the very bug we're fixing.
fn python_runs(p: &str) -> bool {
    std::process::Command::new(p)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Import the exporter's deps and report their versions. `None` if the imports
/// fail — that interpreter can't run the exporter.
fn probe_export_deps(p: &str) -> Option<String> {
    let code = format!(
        "{EXPORT_IMPORTS}; import sys; \
         print('python', sys.version.split()[0]); \
         print('torch', torch.__version__); \
         print('openvino', openvino.__version__); \
         print('transformers', transformers.__version__)"
    );
    let out = std::process::Command::new(p)
        .args(["-c", &code])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Resolve the interpreter to run the embedded exporter with: prefer one that
/// can import the deps, since on Windows `python3` is often a stub with no
/// packages while `python` is the real install. Falls back to any interpreter
/// that runs, so the caller can still surface the real `ModuleNotFoundError`.
pub(crate) fn resolve_python(explicit: Option<&str>, check_deps: bool) -> Result<PythonEnv> {
    let candidates: Vec<String> = match explicit {
        Some(p) => vec![p.to_string()],
        None => vec!["python3".to_string(), "python".to_string()],
    };
    let mut runnable: Option<String> = None;
    for p in &candidates {
        if check_deps {
            if let Some(deps) = probe_export_deps(p) {
                return Ok(PythonEnv {
                    path: p.clone(),
                    deps: Some(deps),
                });
            }
        }
        if runnable.is_none() && python_runs(p) {
            runnable = Some(p.clone());
        }
    }
    match (runnable, explicit) {
        (Some(path), _) => Ok(PythonEnv { path, deps: None }),
        (None, Some(p)) => Err(anyhow!(
            "cannot run the interpreter passed to --python: {p:?}"
        )),
        (None, None) => Err(anyhow!(
            "no Python interpreter found on PATH. Install Python 3.10+ or pass --python <path>."
        )),
    }
}

async fn cmd_shard(args: ShardArgs) -> Result<()> {
    use std::process::{Command, Stdio};

    // Build (and validate) the exporter flag list up front so the NPU
    // fail-fast guard fires before we probe Python or spawn anything.
    let flags = shard_exporter_flags(&args)?;

    // static-seq/static-context are NPU-only; the exporter ignores them on the
    // cpu-gpu path. Warn if the user set them there so it isn't silent.
    if args.target != ShardTarget::Npu
        && (args.static_seq != 1 || args.static_context != DEFAULT_STATIC_CONTEXT)
    {
        eprintln!("warning: --static-seq/--static-context are ignored without --target npu");
    }

    // `--skip-check` means exactly that: don't import the deps just to pick an
    // interpreter (importing torch costs seconds).
    let env = resolve_python(args.python.as_deref(), !args.skip_check)?;
    let python = env.path.clone();

    if !args.skip_check {
        eprintln!("Checking Python environment for required packages...");
        match env.deps {
            Some(versions) => {
                for line in versions.lines() {
                    eprintln!("  {line}");
                }
                // NNCF is optional but warn if missing for INT4 quant.
                let nncf = Command::new(&python)
                    .args(["-c", "import nncf; print('nncf', nncf.__version__)"])
                    .output();
                match nncf {
                    Ok(o) if o.status.success() => {
                        eprintln!("  {}", String::from_utf8_lossy(&o.stdout).trim());
                    }
                    _ => {
                        if matches!(
                            args.quantization,
                            ShardQuant::Int4 | ShardQuant::Int4Asym | ShardQuant::Int8
                        ) {
                            eprintln!("  WARNING: nncf not installed; falls back to FP16 weights");
                        }
                    }
                }
            }
            // The chosen interpreter runs but can't import the deps: re-run the
            // import so the user sees the real ModuleNotFoundError.
            None => {
                let out = Command::new(&python)
                    .args(["-c", EXPORT_IMPORTS])
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .output();
                let stderr = out
                    .map(|o| String::from_utf8_lossy(&o.stderr).into_owned())
                    .unwrap_or_default();
                return Err(anyhow!(
                    "python environment check failed for interpreter {python:?}:\n{}\n\
                     Install: {}\n\
                     (or point at another interpreter with --python <path>)",
                    stderr.trim(),
                    export_pip_install_line(&python)
                ));
            }
        }
    }

    // Write the embedded exporter + its sibling modules into a temp dir so
    // export_shards.py's `from export_gemma4 import ...` / `from model_aliases
    // import ...` (run from that dir) resolve. `_tmpdir` is held until the end
    // of this function so the dir survives until the exporter exits.
    let _tmpdir = tempfile::Builder::new()
        .prefix("cascadia-export-")
        .tempdir()
        .context("creating temp dir for embedded exporter")?;
    let script_path = _tmpdir.path().join("export_shards.py");
    for (name, body) in [
        ("export_shards.py", EXPORT_SCRIPT),
        ("export_gemma4.py", GEMMA4_SCRIPT),
        ("model_aliases.py", ALIASES_SCRIPT),
        ("export_qwen36_moe.py", QWEN36_SCRIPT),
        ("export_gemma4_text.py", GEMMA4_TEXT_SCRIPT),
    ] {
        std::fs::write(_tmpdir.path().join(name), body)
            .with_context(|| format!("writing embedded {name} to temp dir"))?;
    }

    // Build python argv: `python -u <script> <flags…>` (flags already
    // validated + assembled by `shard_exporter_flags`).
    let mut cmd = Command::new(&python);
    cmd.arg("-u").arg(&script_path).args(&flags);
    eprintln!(
        "Running exporter: {} -u <embedded> {}",
        python,
        flags.join(" ")
    );
    let status = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("spawning python exporter")?;
    if !status.success() {
        return Err(anyhow!(
            "exporter exited with {} — see output above",
            status
        ));
    }
    // qwen3_5_moe shards run the in-process stage chain, not the
    // per-stage worker mesh; give the right invocation per manifest arch.
    let arch =
        std::fs::read_to_string(std::path::Path::new(&args.output_dir).join("manifest.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v["arch"].as_str().map(String::from))
            .unwrap_or_default();
    if arch == "qwen3_5_moe" {
        eprintln!(
            "\nShard tree written to {}. Run with:\n  cascadia run {} \
             --engine qwen36-moe --device CPU --api :8000",
            args.output_dir, args.output_dir
        );
    } else if args.num_stages == 1 {
        // Single stage: rank 0 is also the last stage, so there is no --next.
        eprintln!(
            "\nShard tree written to {}. Run with:\n  cascadia run {} \
             --engine ov-runtime --device GPU",
            args.output_dir, args.output_dir
        );
    } else {
        eprintln!(
            "\nShard tree written to {}. Run with:\n  cascadia worker --rank 0 --total {} \
             --engine ov-runtime --device GPU --model {} \
             --next <next-host>:9100 --api :8000",
            args.output_dir, args.num_stages, args.output_dir
        );
    }
    Ok(())
}

#[cfg(test)]
mod python_tests {
    use super::*;

    fn worker(model: &str, engine: EngineKind) -> WorkerArgs {
        let mut a = WorkerArgs::single_node(model.into(), "GPU".into(), engine, ":8000".into());
        a.engine = engine;
        a
    }

    /// The phase-split flags only exist in the ov-runtime static path; using
    /// one on another engine is rejected loudly (here: --gemv-offload on
    /// ov-genai), not silently ignored.
    #[test]
    fn worker_flags_reject_phase_split_without_ov_runtime() {
        let mut a = worker("m", EngineKind::OvGenai);
        a.gemv_offload = true;
        let err = validate_worker_runtime_flags(&a).unwrap_err().to_string();
        assert!(err.contains("ov-runtime"), "{err}");
    }

    /// A prefill device (or parking) is meaningless with chunked prefill
    /// disabled — the two are mutually exclusive.
    #[test]
    fn worker_flags_reject_prefill_device_conflict_with_no_chunked() {
        let mut a = worker("m", EngineKind::OvRuntime);
        a.prefill_device = Some("NPU".into());
        a.no_chunked_prefill = true;
        let err = validate_worker_runtime_flags(&a).unwrap_err().to_string();
        assert!(err.contains("conflict"), "{err}");
    }

    /// Packed multi-slot decode is an ov-runtime static-path feature; using it
    /// on another engine or at N<2 is rejected loudly. Multi-stage packed is
    /// ACCEPTED again — the #122 wedge (worker-thread starvation on rank 0)
    /// is fixed, so the old `--total 1` gate would only block a working
    /// configuration.
    #[test]
    fn worker_flags_gate_packed_slots() {
        let mut a = worker("m", EngineKind::OvGenai);
        a.packed_slots = 8;
        let err = validate_worker_runtime_flags(&a).unwrap_err().to_string();
        assert!(err.contains("ov-runtime"), "{err}");

        let mut a = worker("m", EngineKind::OvRuntime);
        a.packed_slots = 1;
        let err = validate_worker_runtime_flags(&a).unwrap_err().to_string();
        assert!(err.contains(">= 2"), "{err}");

        let mut a = worker("m", EngineKind::OvRuntime);
        a.packed_slots = 8;
        a.total = 2;
        assert!(validate_worker_runtime_flags(&a).is_ok());

        let mut a = worker("m", EngineKind::OvRuntime);
        a.packed_slots = 8;
        assert!(validate_worker_runtime_flags(&a).is_ok());

        // the shared prefix lives inside the packed window, so it needs slots
        let mut a = worker("m", EngineKind::OvRuntime);
        a.packed_prefix = 128;
        let err = validate_worker_runtime_flags(&a).unwrap_err().to_string();
        assert!(err.contains("--packed-slots"), "{err}");

        let mut a = worker("m", EngineKind::OvRuntime);
        a.packed_slots = 4;
        a.packed_prefix = 128;
        assert!(validate_worker_runtime_flags(&a).is_ok());
    }

    /// Prefix reuse is first-stage-only state: only rank 0 admits requests, and
    /// admission is what fills the shared prefix region, so a relay stage clamps
    /// every plan row's reuse to 0 and holds no KV for the tokens rank 0 skipped
    /// prefilling — wrong output with no error anywhere. Rejected at the CLI
    /// until relay-side prefix population exists. Multi-stage packed WITHOUT
    /// --packed-prefix must stay accepted (it was un-gated deliberately).
    #[test]
    fn worker_flags_gate_packed_prefix_multi_stage() {
        let mut a = worker("m", EngineKind::OvRuntime);
        a.packed_slots = 4;
        a.packed_prefix = 128;
        a.total = 2;
        let err = validate_worker_runtime_flags(&a).unwrap_err().to_string();
        assert!(
            err.contains("--packed-prefix is single-stage only"),
            "{err}"
        );
        assert!(err.contains("--total 1"), "{err}");

        // Single stage is the supported configuration.
        let mut a = worker("m", EngineKind::OvRuntime);
        a.packed_slots = 4;
        a.packed_prefix = 128;
        a.total = 1;
        assert!(validate_worker_runtime_flags(&a).is_ok());

        // Packed multi-stage decode without prefix reuse is untouched.
        let mut a = worker("m", EngineKind::OvRuntime);
        a.packed_slots = 4;
        a.total = 2;
        assert!(validate_worker_runtime_flags(&a).is_ok());
    }

    /// The two continuous-batching mechanisms target different engines, so
    /// asking for both at once is always rejected — whichever engine is named,
    /// the other flag's gate fires. Pins that the ov-genai (#116) and
    /// ov-runtime packed paths stay mutually exclusive.
    #[test]
    fn worker_flags_reject_cb_and_packed_slots_together() {
        let mut a = worker("m", EngineKind::OvGenai);
        a.cb = true;
        a.packed_slots = 8;
        let err = validate_worker_runtime_flags(&a).unwrap_err().to_string();
        assert!(err.contains("ov-runtime"), "{err}");

        let mut a = worker("m", EngineKind::OvRuntime);
        a.cb = true;
        a.packed_slots = 8;
        let err = validate_worker_runtime_flags(&a).unwrap_err().to_string();
        assert!(err.contains("ov-genai"), "{err}");
    }

    /// A valid phase split (ov-runtime + prefill device, chunked enabled) passes.
    #[test]
    fn worker_flags_accept_valid_phase_split() {
        let mut a = worker("m", EngineKind::OvRuntime);
        a.prefill_device = Some("NPU".into());
        assert!(validate_worker_runtime_flags(&a).is_ok());
    }

    /// Continuous batching lives in ov-genai's CBP path; --cb on any other
    /// engine is rejected loudly, not silently ignored.
    #[test]
    fn worker_flags_reject_cb_without_ov_genai() {
        let mut a = worker("m", EngineKind::OvRuntime);
        a.cb = true;
        let err = validate_worker_runtime_flags(&a).unwrap_err().to_string();
        assert!(err.contains("ov-genai"), "{err}");

        // Tuning flags alone (without --cb) trip the same engine gate.
        let mut a = worker("m", EngineKind::Mock);
        a.cb_max_num_seqs = 32;
        let err = validate_worker_runtime_flags(&a).unwrap_err().to_string();
        assert!(err.contains("ov-genai"), "{err}");
    }

    /// --cb-* tuning knobs without --cb are a misconfiguration, even on
    /// ov-genai — the operator believes batching is on when it is not.
    #[test]
    fn worker_flags_reject_cb_tuning_without_cb() {
        let mut a = worker("m", EngineKind::OvGenai);
        a.cb_cache_size = 4;
        let err = validate_worker_runtime_flags(&a).unwrap_err().to_string();
        assert!(err.contains("--cb"), "{err}");
    }

    /// The CPU long-prompt collapse is a documented hazard with no runtime
    /// signal of its own, so the worker says something at startup. Warn, not
    /// reject — CPU with short prompts is a real 1.6-2.1x win.
    #[test]
    fn cb_on_cpu_is_flagged_but_gpu_and_npu_are_not() {
        let mut a = worker("m", EngineKind::OvGenai);
        a.cb = true;
        for (device, want) in [
            ("CPU", true),
            ("cpu", true),
            (" CPU ", true),
            ("GPU", false),
            ("GPU.0", false),
            ("NPU", false),
            // Compound strings may resolve to CPU at runtime; we cannot know.
            ("AUTO", false),
            ("HETERO:CPU,GPU", false),
        ] {
            a.device = device.to_string();
            assert_eq!(cb_on_plain_cpu(&a), want, "device={device}");
        }
        // Without --cb there is nothing to warn about.
        a.cb = false;
        a.device = "CPU".to_string();
        assert!(!cb_on_plain_cpu(&a));
    }

    /// `--cb` on an NPU device is rejected up front. The docs say paged
    /// attention cannot work there; without the gate the operator pays a full
    /// model compile before OpenVINO throws.
    #[test]
    fn worker_flags_reject_cb_on_npu() {
        let mut a = worker("m", EngineKind::OvGenai);
        a.cb = true;
        a.device = "NPU".to_string();
        let err = validate_worker_runtime_flags(&a).unwrap_err().to_string();
        assert!(err.contains("NPU"), "{err}");

        // A compound AUTO/HETERO string that merely mentions NPU is not an
        // NPU target (matches device_is_npu's documented contract).
        let mut a = worker("m", EngineKind::OvGenai);
        a.cb = true;
        a.device = "HETERO:NPU,CPU".to_string();
        assert!(validate_worker_runtime_flags(&a).is_ok());
    }

    /// The full CB flag set on ov-genai passes validation.
    #[test]
    fn worker_flags_accept_cb_on_ov_genai() {
        let mut a = worker("m", EngineKind::OvGenai);
        a.cb = true;
        a.cb_cache_size = 4;
        a.cb_max_num_seqs = 32;
        a.cb_max_batched_tokens = 2048;
        a.cb_dynamic_split_fuse = Some(true);
        a.cb_prefix_caching = Some(true);
        assert!(validate_worker_runtime_flags(&a).is_ok());
    }

    #[test]
    fn explicit_interpreter_that_cannot_run_names_the_path() {
        // Must not tell the user to "pass --python" — they just did.
        let err = resolve_python(Some("cascadia-no-such-python"), false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("cascadia-no-such-python"), "{err}");
        assert!(!err.contains("on PATH"), "{err}");
    }

    #[test]
    fn pip_line_quotes_every_specifier_and_targets_the_interpreter() {
        let line = export_pip_install_line("python3");
        assert!(line.starts_with("python3 -m pip install "), "{line}");
        // Double quotes: the only form valid in sh, zsh, PowerShell and cmd.exe.
        assert!(line.contains("\"transformers>=5.2,<5.5\""), "{line}");
        assert!(line.contains("\"safetensors>=0.4\""), "{line}");
        // Comments and blank lines from requirements.txt never leak through.
        assert!(!line.contains('#'), "{line}");
        // An interpreter path with spaces stays one argument, and the line starts
        // with the path itself: a leading `&` would be PowerShell-only and a
        // syntax error in cmd.exe and Git Bash, which are the same target.
        let q = export_pip_install_line("C:\\Program Files\\Python\\python.exe");
        assert!(
            q.starts_with("\"C:\\Program Files\\Python\\python.exe\" -m pip"),
            "{q}"
        );
        assert!(!q.starts_with('&'), "{q}");
    }

    #[test]
    fn model_stem_handles_paths_and_repo_ids() {
        assert_eq!(model_stem("unsloth/Meta-Llama-3.1-8B"), "Meta-Llama-3.1-8B");
        assert_eq!(model_stem("C:\\models\\foo"), "foo");
        assert_eq!(model_stem("foo/"), "model");
        assert_eq!(model_stem(""), "model");
    }

    #[test]
    fn preflight_rejects_an_hf_id_with_the_command_to_run() {
        let err = preflight_model_path(&worker(
            "unsloth/Meta-Llama-3.1-8B-Instruct",
            EngineKind::OvRuntime,
        ))
        .unwrap_err()
        .to_string();
        assert!(err.contains("cascadia shard"), "{err}");
        // ov-genai cannot be fed a shard tree, so it gets different advice.
        let genai = preflight_model_path(&worker("org/name", EngineKind::OvGenai))
            .unwrap_err()
            .to_string();
        assert!(genai.contains("optimum-cli"), "{genai}");
    }

    #[test]
    fn preflight_covers_both_a_missing_path_and_an_hf_id() {
        // `out/llama-1stage` is a valid relative path AND shaped like a repo id,
        // so the message must not guess — it says both.
        let err = preflight_model_path(&worker("out/llama-1stage", EngineKind::OvRuntime))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("If that is a path, it does not exist"),
            "{err}"
        );
        assert!(err.contains("HuggingFace repo id"), "{err}");
    }

    #[test]
    fn preflight_advice_matches_the_engine() {
        // Each engine reads a different tree; don't send them all to ov-runtime.
        for (engine, needle) in [
            (EngineKind::Gemma4, "--engine gemma4"),
            (EngineKind::Qwen36Moe, "--engine qwen36-moe"),
            (EngineKind::SparseMoe, "manifest.json"),
            (EngineKind::OvGenai, "optimum-cli"),
            (EngineKind::OvRuntime, "--engine ov-runtime"),
        ] {
            let err = preflight_model_path(&worker("org/name", engine))
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{engine:?}: {err}");
        }
    }

    #[test]
    fn preflight_ignores_a_draft_model_the_engine_never_reads() {
        // A relay rank launched from the same argv must not be rejected for a
        // draft IR it never opens.
        let dir = tempfile::tempdir().unwrap();
        let mut args = worker(&dir.path().to_string_lossy(), EngineKind::OvRuntime);
        args.draft_model = Some("unsloth/Llama-3.2-1B-Instruct".into());
        assert!(preflight_model_path(&args).is_ok());

        let mut relay = worker(&dir.path().to_string_lossy(), EngineKind::OvDistSpec);
        relay.draft_model = Some("unsloth/Llama-3.2-1B-Instruct".into());
        relay.rank = 1;
        relay.total = 2;
        assert!(preflight_model_path(&relay).is_ok());
        // …but the driver (rank 0) does read it.
        relay.rank = 0;
        assert!(preflight_model_path(&relay).is_err());
    }

    #[test]
    fn preflight_accepts_an_existing_dir_and_the_mock_engine() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().into_owned();
        assert!(preflight_model_path(&worker(&path, EngineKind::OvRuntime)).is_ok());
        assert!(preflight_model_path(&worker("mock-model", EngineKind::Mock)).is_ok());
    }

    #[test]
    fn preflight_rejects_a_draft_model_that_is_not_a_path() {
        let dir = tempfile::tempdir().unwrap();
        let mut args = worker(&dir.path().to_string_lossy(), EngineKind::OvGenai);
        args.draft_model = Some("unsloth/Llama-3.2-1B-Instruct".into());
        let err = preflight_model_path(&args).unwrap_err().to_string();
        assert!(err.contains("draft-model"), "{err}");
    }
}

#[cfg(test)]
mod ov_property_tests {
    use super::*;

    fn args_for(device: &str) -> WorkerArgs {
        args_for_engine(device, EngineKind::OvGenai)
    }

    fn args_for_engine(device: &str, engine: EngineKind) -> WorkerArgs {
        WorkerArgs::single_node(
            "model".into(),
            device.into(),
            engine,
            "127.0.0.1:8080".into(),
        )
    }

    fn prop<'a>(props: &'a [(String, String)], key: &str) -> Option<&'a str> {
        props
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    #[test]
    fn device_is_npu_matches_npu_plugins_only() {
        assert!(device_is_npu("NPU"));
        assert!(device_is_npu("NPU.0"));
        assert!(device_is_npu("npu"));
        assert!(device_is_npu("  NPU  "));
        assert!(!device_is_npu("GPU"));
        assert!(!device_is_npu("CPU"));
        // Compound AUTO/HETERO strings that merely mention NPU are not NPU targets.
        assert!(!device_is_npu("AUTO:NPU,CPU"));
    }

    #[test]
    fn unset_flags_produce_no_properties() {
        let args = args_for("GPU");
        assert!(ov_perf_properties(&args).is_empty());
    }

    #[test]
    fn general_hints_map_to_ov_property_keys() {
        let mut args = args_for("GPU");
        args.ov_performance_mode = Some(OvPerformanceMode::Latency);
        args.ov_inference_precision = Some("f16".into());
        args.ov_num_streams = Some(2);
        args.ov_num_threads = Some(8);
        args.ov_allow_auto_batching = true;
        args.ov_execution_mode = Some(OvExecutionMode::Performance);

        let props = ov_perf_properties(&args);
        assert_eq!(prop(&props, "PERFORMANCE_HINT"), Some("LATENCY"));
        assert_eq!(prop(&props, "INFERENCE_PRECISION_HINT"), Some("f16"));
        assert_eq!(prop(&props, "NUM_STREAMS"), Some("2"));
        assert_eq!(prop(&props, "INFERENCE_NUM_THREADS"), Some("8"));
        assert_eq!(prop(&props, "ALLOW_AUTO_BATCHING"), Some("true"));
        assert_eq!(prop(&props, "EXECUTION_MODE_HINT"), Some("PERFORMANCE"));
    }

    #[test]
    fn auto_batching_absent_when_flag_unset() {
        let args = args_for("GPU");
        assert_eq!(
            prop(&ov_perf_properties(&args), "ALLOW_AUTO_BATCHING"),
            None
        );
    }

    #[test]
    fn npu_props_dropped_on_non_npu_device() {
        let mut args = args_for("GPU");
        args.npu_prefill_chunk_size = Some(512);
        args.npu_max_prompt_len = Some(1024);
        args.npu_min_response_len = Some(128);

        let props = ov_perf_properties(&args);
        assert_eq!(prop(&props, "NPUW_LLM_PREFILL_CHUNK_SIZE"), None);
        assert_eq!(prop(&props, "MAX_PROMPT_LEN"), None);
        assert_eq!(prop(&props, "MIN_RESPONSE_LEN"), None);
    }

    #[test]
    fn npu_props_included_on_npu_device() {
        let mut args = args_for("NPU.0");
        args.npu_prefill_chunk_size = Some(512);
        args.npu_max_prompt_len = Some(1024);
        args.npu_min_response_len = Some(128);

        let props = ov_perf_properties(&args);
        assert_eq!(prop(&props, "NPUW_LLM_PREFILL_CHUNK_SIZE"), Some("512"));
        assert_eq!(prop(&props, "MAX_PROMPT_LEN"), Some("1024"));
        assert_eq!(prop(&props, "MIN_RESPONSE_LEN"), Some("128"));
    }

    #[test]
    fn npu_props_dropped_on_non_genai_engine_even_on_npu_device() {
        // MAX_PROMPT_LEN / MIN_RESPONSE_LEN / NPUW_LLM_PREFILL_CHUNK_SIZE are
        // ov::genai LLMPipeline convenience keys. Only ov-genai routes props
        // through the GenAI pipeline; the other OV engines hit raw
        // compile_model, which cannot consume them. So they must be gated on
        // engine == ov-genai, not device alone.
        for engine in [
            EngineKind::OvRuntime,
            EngineKind::Gemma4,
            EngineKind::OvDistSpec,
            EngineKind::SparseMoe,
        ] {
            let mut args = args_for_engine("NPU.0", engine);
            args.npu_prefill_chunk_size = Some(512);
            args.npu_max_prompt_len = Some(1024);
            args.npu_min_response_len = Some(128);

            let props = ov_perf_properties(&args);
            assert_eq!(
                prop(&props, "NPUW_LLM_PREFILL_CHUNK_SIZE"),
                None,
                "{engine:?}"
            );
            assert_eq!(prop(&props, "MAX_PROMPT_LEN"), None, "{engine:?}");
            assert_eq!(prop(&props, "MIN_RESPONSE_LEN"), None, "{engine:?}");
        }
    }

    #[test]
    fn general_hints_still_apply_on_non_genai_engine() {
        // The engine gate is NPU-knob-specific; general hints stay engine-agnostic.
        let mut args = args_for_engine("GPU", EngineKind::OvRuntime);
        args.ov_performance_mode = Some(OvPerformanceMode::Latency);
        let props = ov_perf_properties(&args);
        assert_eq!(prop(&props, "PERFORMANCE_HINT"), Some("LATENCY"));
    }
}

#[cfg(test)]
mod cli_version_tests {
    use super::*;

    #[test]
    fn version_flag_reports_the_workspace_version() {
        let err = Cli::try_parse_from(["cascadia", "--version"])
            .expect_err("--version stops parsing with a DisplayVersion error");
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
        assert!(err.to_string().contains(env!("CARGO_PKG_VERSION")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a `cascadia shard …` argv into its `ShardArgs` (panics on a
    /// non-shard parse) so the flag-builder tests exercise the real clap path.
    fn parse_shard(argv: &[&str]) -> ShardArgs {
        let cli = Cli::try_parse_from(argv).expect("parse shard argv");
        match cli.cmd {
            Command::Shard(args) => args,
            _ => panic!("expected shard subcommand"),
        }
    }

    /// True if `flag` appears immediately followed by `value` in the argv.
    fn has_pair(flags: &[String], flag: &str, value: &str) -> bool {
        flags.windows(2).any(|w| w[0] == flag && w[1] == value)
    }

    /// NPU with a non-fp16 default dtype fails fast in the pure builder.
    #[test]
    fn shard_flags_npu_fp32_is_err() {
        let args = parse_shard(&[
            "cascadia",
            "shard",
            "--model",
            "m",
            "--output-dir",
            "o",
            "--num-stages",
            "2",
            "--target",
            "npu",
            "--default-dtype",
            "fp32",
        ]);
        assert!(shard_exporter_flags(&args).is_err());
    }

    /// NPU (fp16) forwards target, dtype, and the static-shape flags.
    #[test]
    fn shard_flags_npu_forwards_static() {
        let args = parse_shard(&[
            "cascadia",
            "shard",
            "--model",
            "m",
            "--output-dir",
            "o",
            "--num-stages",
            "2",
            "--target",
            "npu",
        ]);
        let flags = shard_exporter_flags(&args).expect("npu fp16 flags");
        assert!(has_pair(&flags, "--target", "npu"));
        assert!(has_pair(&flags, "--default-dtype", "fp16"));
        assert!(flags.iter().any(|f| f == "--static-seq"));
        assert!(flags.iter().any(|f| f == "--static-context"));
    }

    /// Default (cpu-gpu) forwards target + dtype but omits the static flags.
    #[test]
    fn shard_flags_cpu_gpu_omits_static() {
        let args = parse_shard(&[
            "cascadia",
            "shard",
            "--model",
            "m",
            "--output-dir",
            "o",
            "--num-stages",
            "2",
        ]);
        let flags = shard_exporter_flags(&args).expect("cpu-gpu flags");
        assert!(has_pair(&flags, "--target", "cpu-gpu"));
        assert!(has_pair(&flags, "--default-dtype", "fp16"));
        assert!(!flags.iter().any(|f| f == "--static-seq"));
        assert!(!flags.iter().any(|f| f == "--static-context"));
    }

    /// `--static-prefill-seq` needs `--target npu` (chunked prefill is a
    /// static-KV-path feature); the default cpu-gpu target rejects it.
    #[test]
    fn shard_flags_static_prefill_seq_requires_npu() {
        let args = parse_shard(&[
            "cascadia",
            "shard",
            "--model",
            "m",
            "--output-dir",
            "o",
            "--num-stages",
            "1",
            "--static-prefill-seq",
            "4",
        ]);
        let err = shard_exporter_flags(&args).unwrap_err().to_string();
        assert!(err.contains("--target npu"), "{err}");
    }

    /// `--static-prefill-seq 1` is nonsense (a 1-wide chunk is the seq=1 decode
    /// path); only 0 (off) or >= 2 is valid.
    #[test]
    fn shard_flags_static_prefill_seq_one_rejected() {
        let args = parse_shard(&[
            "cascadia",
            "shard",
            "--model",
            "m",
            "--output-dir",
            "o",
            "--num-stages",
            "1",
            "--target",
            "npu",
            "--static-prefill-seq",
            "1",
        ]);
        let err = shard_exporter_flags(&args).unwrap_err().to_string();
        assert!(err.contains(">= 2"), "{err}");
    }

    /// A chunk wider than the KV window would evict its own tokens mid-chunk:
    /// `--static-prefill-seq` must be <= `--static-context - 1`.
    #[test]
    fn shard_flags_static_prefill_seq_exceeds_context_rejected() {
        let args = parse_shard(&[
            "cascadia",
            "shard",
            "--model",
            "m",
            "--output-dir",
            "o",
            "--num-stages",
            "1",
            "--target",
            "npu",
            "--static-context",
            "8",
            "--static-prefill-seq",
            "8",
        ]);
        let err = shard_exporter_flags(&args).unwrap_err().to_string();
        assert!(err.contains("evict"), "{err}");
    }

    /// A valid `--static-prefill-seq` is forwarded to the exporter verbatim
    /// (a dropped/defaulted value would silently export a wrong-shape variant).
    #[test]
    fn shard_flags_static_prefill_seq_forwarded() {
        let args = parse_shard(&[
            "cascadia",
            "shard",
            "--model",
            "m",
            "--output-dir",
            "o",
            "--num-stages",
            "1",
            "--target",
            "npu",
            "--static-context",
            "1024",
            "--static-prefill-seq",
            "64",
        ]);
        let flags = shard_exporter_flags(&args).expect("valid prefill-seq flags");
        assert!(has_pair(&flags, "--static-prefill-seq", "64"), "{flags:?}");
    }

    /// Golden vector: pins VALUES (not just flag presence), `--quantization`
    /// forwarding, and flag/value adjacency for the whole NPU argv — a bug
    /// that pushes a default instead of the user's value passes every
    /// presence-only assertion and exports a wrong-shape shard silently.
    #[test]
    fn shard_flags_npu_golden_vector() {
        let args = parse_shard(&[
            "cascadia",
            "shard",
            "--model",
            "m",
            "--output-dir",
            "o",
            "--num-stages",
            "2",
            "--quantization",
            "int8",
            "--target",
            "npu",
            "--static-context",
            "2048",
        ]);
        let expected: Vec<String> = [
            "--model",
            "m",
            "--output-dir",
            "o",
            "--num-stages",
            "2",
            "--quantization",
            "int8",
            "--target",
            "npu",
            "--default-dtype",
            "fp16",
            "--static-seq",
            "1",
            "--static-context",
            "2048",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(shard_exporter_flags(&args).expect("npu golden"), expected);
    }

    /// `--layer-split`/`--stage` are forwarded when present.
    #[test]
    fn shard_flags_forwards_layer_split_and_stage() {
        let args = parse_shard(&[
            "cascadia",
            "shard",
            "--model",
            "m",
            "--output-dir",
            "o",
            "--num-stages",
            "2",
            "--layer-split",
            "8",
            "--stage",
            "1",
        ]);
        let flags = shard_exporter_flags(&args).expect("layer-split/stage flags");
        assert!(has_pair(&flags, "--layer-split", "8"));
        assert!(has_pair(&flags, "--stage", "1"));
    }

    /// `--target npu` parses into the NPU variant.
    #[test]
    fn shard_target_npu_parses() {
        let cli = Cli::try_parse_from([
            "cascadia",
            "shard",
            "--model",
            "m",
            "--output-dir",
            "o",
            "--num-stages",
            "2",
            "--target",
            "npu",
        ])
        .expect("parse shard --target npu");
        let Command::Shard(args) = cli.cmd else {
            panic!("expected shard subcommand");
        };
        assert_eq!(args.target, ShardTarget::Npu);
    }

    /// Omitting `--target` defaults to the cpu-gpu path (backward compatible).
    #[test]
    fn shard_target_defaults_to_cpu_gpu() {
        let cli = Cli::try_parse_from([
            "cascadia",
            "shard",
            "--model",
            "m",
            "--output-dir",
            "o",
            "--num-stages",
            "2",
        ])
        .expect("parse shard without --target");
        let Command::Shard(args) = cli.cmd else {
            panic!("expected shard subcommand");
        };
        assert_eq!(args.target, ShardTarget::CpuGpu);
    }
}
