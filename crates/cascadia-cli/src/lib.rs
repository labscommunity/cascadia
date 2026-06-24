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
    Qwen36Builder, TpRuntimeBuilder,
};
use cascadia_engine_sparse_moe::{SparseMoEBuilder, SparseMoEBuilderConfig};
use cascadia_runner::Runner;
use cascadia_types::{GenerationTask, PeerEndpoint, PeerLayout, ShardSpec};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use futures::StreamExt;
use tracing::info;
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
        EngineKind::TpRuntime => "tp-runtime",
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
    about = "Distributed LLM inference for Intel hardware",
    long_about = "Distributed LLM inference for Intel hardware.\n\n\
        SECURITY: cascadia's HTTP API and inter-stage TCP relay are \
        plaintext and unauthenticated. Bind only to trusted networks \
        (LAN, loopback) or terminate TLS + auth at a reverse proxy in \
        front of `--api`. See rust/docs/STATUS.md \"Security model\" \
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
    /// Run a model on a single machine (sugar for a one-stage worker
    /// with an OpenAI API). See `cascadia run --help`.
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

#[derive(ValueEnum, Clone, Copy, Debug)]
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
    /// Megatron tensor-parallel engine. Each rank holds a 1/N slice of every
    /// layer (segmented IR from `tools/tp_export.py`) and all-reduces partials
    /// after attention + MLP over the transport. 2-node bidirectional peer link
    /// (--rank = tp_rank, --total = tp_size, --next = peer, --listen = own).
    TpRuntime,
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

    /// HF model id or local model directory.
    #[arg(long)]
    pub model: String,

    /// Bind address for the upstream-receiving socket (default :9100).
    #[arg(long, default_value = ":9100")]
    pub listen: String,

    /// Downstream peer (host:port) — required for non-last stages.
    #[arg(long)]
    pub next: Option<String>,

    /// API bind address (e.g. :8000) — only valid for rank 0.
    #[arg(long)]
    pub api: Option<String>,

    /// Device hint: CPU / GPU / NPU.
    #[arg(long, default_value = "CPU")]
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

    /// Speculative-decode draft model path (FastDraft companion).
    #[arg(long)]
    pub draft_model: Option<String>,

    /// Device for the draft model (default: same as --device).
    #[arg(long)]
    pub draft_device: Option<String>,

    /// Speculative-decode draft length per round.
    #[arg(long, default_value_t = 5)]
    pub spec_k: u32,

    /// Enable Prompt Lookup decoding with n-gram size N. Mutually
    /// exclusive with --draft-model.
    #[arg(long, default_value_t = 0)]
    pub prompt_lookup: u32,

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
    /// 10-prompt eval with matched quality. See docs/A3_TOPK_REDUCTION.md.
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
    /// (MIT-licensed; see rainier `docs/POWERINFER_PORT.md`). The env
    /// var `CASCADIA_MAX_EXPERTS_CACHED` overrides this flag at runtime.
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
    /// approach. See rainier `docs/POWERINFER_PORT.md`.
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
    /// (MIT). See rainier `docs/POWERINFER_PORT.md`.
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
    /// HF model id (e.g. `unsloth/Meta-Llama-3.1-8B-Instruct`) or a
    /// local model / shard directory.
    pub model: String,

    /// Device hint: GPU / CPU / NPU. Defaults to GPU — run `cascadia
    /// doctor` first to confirm the iGPU is visible to OpenVINO.
    #[arg(long, default_value = "GPU")]
    pub device: String,

    /// Inference engine. Defaults to `ov-genai` (single-stage, auto-
    /// exports from an HF id).
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
            listen: ":9100".into(),
            next: None,
            api: Some(api),
            device,
            engine,
            ov_cache_dir: None,
            ov_kv_precision: None,
            ov_dyn_quant_group: None,
            draft_model: None,
            draft_device: None,
            spec_k: 5,
            prompt_lookup: 0,
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

#[derive(Parser, Debug, Clone)]
pub struct ShardArgs {
    /// HuggingFace repo id (e.g. unsloth/Meta-Llama-3.1-8B-Instruct)
    /// or path to a local directory containing safetensors + config.json.
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

fn build_builder(args: &WorkerArgs) -> Result<Box<dyn Builder>> {
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
            let mut b = OvGenaiBuilder::new(&args.model, &args.device);
            if let Some(dir) = resolve_ov_cache_dir(args.ov_cache_dir.as_deref()) {
                b = b.with_cache_dir(&dir);
            }
            if let Some(prec) = &args.ov_kv_precision {
                b = b.with_kv_cache_precision(prec);
            }
            if let Some(group) = &args.ov_dyn_quant_group {
                b = b.with_dyn_quant_group(group);
            }
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
            if let Some(dir) = resolve_ov_cache_dir(args.ov_cache_dir.as_deref()) {
                b = b.with_cache_dir(&dir);
            }
            if let Some(prec) = &args.ov_kv_precision {
                b = b.with_kv_cache_precision(prec);
            }
            if let Some(group) = &args.ov_dyn_quant_group {
                b = b.with_dyn_quant_group(group);
            }
            Ok(Box::new(b))
        }
        EngineKind::TpRuntime => {
            let mut b = TpRuntimeBuilder::new(&args.model, args.rank, args.total, &args.device);
            if let Some(dir) = resolve_ov_cache_dir(args.ov_cache_dir.as_deref()) {
                b = b.with_cache_dir(&dir);
            }
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

async fn cmd_worker(args: WorkerArgs) -> Result<()> {
    if args.rank >= args.total {
        return Err(anyhow!(
            "--rank must be in [0, {}); got {}",
            args.total,
            args.rank
        ));
    }
    let is_first = args.rank == 0;
    let is_last = args.rank == args.total - 1;
    // Tensor parallelism is a 2-node bidirectional peer link, not a daisy chain:
    // EVERY rank both listens (server) and dials the peer (client), regardless of
    // is_first/is_last. rank 0 still drives generation (is_first), rank 1 relays.
    let is_tp = matches!(args.engine, EngineKind::TpRuntime);

    info!(
        engine = ?args.engine,
        rank = args.rank,
        total = args.total,
        device = %args.device,
        model = %args.model,
        "cascadia worker starting"
    );

    let (listen_host, listen_port) = parse_addr(&args.listen, "0.0.0.0")?;

    let upstream = if is_first && !is_tp {
        None
    } else {
        Some(PeerEndpoint::new(listen_host.clone(), listen_port))
    };
    let downstream = if is_last && !is_tp {
        None
    } else {
        let next = args
            .next
            .as_deref()
            .ok_or_else(|| anyhow!("--next is required for non-last stages (and for tp-runtime peers)"))?;
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
        tp_size: if is_tp { args.total } else { 1 },
        tp_rank: if is_tp { args.rank } else { 0 },
    };

    let builder = build_builder(&args)?;
    let runner = Arc::new(Runner::new(builder));
    let listen = if !is_first || is_tp {
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
    let api_port = args
        .api
        .as_deref()
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
        tokio::task::spawn_blocking(move || r.run_relay_loop())
            .await
            .ok();
        return Ok(());
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
        let api_router = cascadia_api::make_router_with_stats(
            runner.clone(),
            args.model.clone(),
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

async fn cmd_shard(args: ShardArgs) -> Result<()> {
    use std::process::{Command, Stdio};

    let python = if let Some(p) = args.python.as_deref() {
        p.to_string()
    } else {
        // Try `python3` then `python`. clap doesn't run them — we just pick one.
        if Command::new("python3").arg("--version").output().is_ok() {
            "python3".into()
        } else if Command::new("python").arg("--version").output().is_ok() {
            "python".into()
        } else {
            return Err(anyhow!(
                "no Python interpreter found on PATH. Install Python 3.10+ or pass --python <path>."
            ));
        }
    };

    if !args.skip_check {
        eprintln!("Checking Python environment for required packages...");
        let probe = Command::new(&python)
            .args([
                "-c",
                "import torch, openvino, transformers, safetensors, huggingface_hub; \
                 print('python', __import__('sys').version.split()[0]); \
                 print('torch', torch.__version__); \
                 print('openvino', openvino.__version__); \
                 print('transformers', transformers.__version__);",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();
        match probe {
            Ok(out) if out.status.success() => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                for line in stdout.lines() {
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
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                return Err(anyhow!(
                    "python environment check failed:\n{}\n\
                     Install: pip install torch openvino transformers safetensors \
                     huggingface_hub nncf",
                    stderr.trim()
                ));
            }
            Err(e) => {
                return Err(anyhow!("couldn't run python interpreter {python:?}: {e}"));
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
    ] {
        std::fs::write(_tmpdir.path().join(name), body)
            .with_context(|| format!("writing embedded {name} to temp dir"))?;
    }

    // Build python argv.
    let mut cmd = Command::new(&python);
    cmd.arg("-u")
        .arg(&script_path)
        .arg("--model")
        .arg(&args.model)
        .arg("--output-dir")
        .arg(&args.output_dir)
        .arg("--num-stages")
        .arg(args.num_stages.to_string())
        .arg("--quantization")
        .arg(args.quantization.as_arg());
    if let Some(s) = &args.layer_split {
        cmd.arg("--layer-split").arg(s);
    }
    if let Some(s) = args.stage {
        cmd.arg("--stage").arg(s.to_string());
    }
    eprintln!(
        "Running exporter: {} -u <embedded> --model {} --output-dir {} --num-stages {}",
        python, args.model, args.output_dir, args.num_stages
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
