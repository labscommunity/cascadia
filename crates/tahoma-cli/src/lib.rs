//! tahoma CLI.
//!
//! Mirrors `python -m tahoma worker` from `tahoma/cli.py`. Only the
//! subset needed for this session's MVP — single-node inference + API
//! server. Multi-stage / discovery flags are accepted but enforced
//! against engine support.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use futures::StreamExt;
use tahoma_engine::Builder;
use tahoma_engine_mock::MockBuilder;
use tahoma_engine_openvino::{
    OvDistSpecBuilder, OvDistSpecWorkerBuilder, OvGenaiBuilder, OvRuntimeBuilder,
};
use tahoma_engine_sparse_moe::{LayerRangeStrategy, SparseMoEBuilder, SparseMoEBuilderConfig};
use tahoma_runner::Runner;
use tahoma_types::{GenerationTask, PeerEndpoint, PeerLayout, ShardSpec};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "tahoma",
    about = "Distributed LLM inference for Intel hardware",
    long_about = "Distributed LLM inference for Intel hardware.\n\n\
        SECURITY: tahoma's HTTP API and inter-stage TCP relay are \
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
    /// Run a pipeline-stage worker.
    Worker(WorkerArgs),
    /// List registered inference engines.
    Engines,
    /// Shard a HuggingFace causal-LM model into per-stage OpenVINO IRs
    /// for distributed inference. See `tahoma shard --help`.
    Shard(ShardArgs),
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum EngineKind {
    Mock,
    OvGenai,
    OvRuntime,
    OvDistSpec,
    /// Kimi K2.6-style sparse-MoE engine. Routes only the top-k experts
    /// per token (not all 384) and runs the expert matmuls through the
    /// hand-rolled AVX-512 int4 GEMM kernel. Single-stage, CPU-targeted.
    SparseMoe,
}

#[derive(Parser, Debug, Clone)]
pub struct WorkerArgs {
    /// 0-based stage index.
    #[arg(long)]
    pub rank: u32,

    /// Total number of stages.
    #[arg(long)]
    pub total: u32,

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

    /// Explicit MoE-layer slice for this rank, in `start..end` form
    /// (half-open, 1-based — dense layer 0 is implicit). Overrides the
    /// default even split. K2.6 has 60 MoE layers; e.g. on a 2-box
    /// pipeline rank 0 might pass `--layer-range 1..29` and rank 1
    /// `--layer-range 29..61` for a 28/32 split (moves 2 shells off
    /// the bottleneck rank — see iter-003 / iter-081 instrumentation).
    /// Currently only the `sparse-moe` engine honors this flag.
    #[arg(long)]
    pub layer_range: Option<String>,

    /// Auto-rebalance strategy. `auto` reads per-stage timing
    /// instrumentation and computes a balanced split; today it falls
    /// back to the even split with a warning (iter-082 wires the
    /// timing-driven logic). Mutually exclusive with `--layer-range`.
    #[arg(long, value_enum)]
    pub rank_balance: Option<RankBalance>,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankBalance {
    /// Read instrumentation, rebalance automatically. **Skeleton only**
    /// — falls back to even split with a warning until iter-082 wires
    /// the timing-driven logic.
    Auto,
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
        Command::Engines => cmd_engines(),
        Command::Worker(args) => cmd_worker(args).await,
        Command::Shard(args) => cmd_shard(args).await,
    }
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
    println!("  sparse-moe     Kimi K2.6 sparse top-8 dispatch; AVX-512 int4 GEMM + Rust shells");
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

/// Parse the `--layer-range` argument's `start..end` literal into a
/// half-open pair. Rejects swapped / missing components early so the
/// caller gets a clean clap-style error, not an `EngineError` 30
/// seconds into model load. Range semantics (`start < end`, `start >=
/// 1`, etc.) are validated by the engine against the actual manifest
/// — we just need to get two parseable integers out of the literal
/// here.
fn parse_layer_range(s: &str) -> Result<(u32, u32)> {
    let (start_s, end_s) = s.split_once("..").ok_or_else(|| {
        anyhow!(
            "--layer-range must be in `start..end` form (got {s:?}); \
             e.g. `--layer-range 1..29` for the first 28 MoE layers"
        )
    })?;
    let start: u32 = start_s
        .trim()
        .parse()
        .with_context(|| format!("--layer-range start `{start_s}`"))?;
    let end: u32 = end_s
        .trim()
        .parse()
        .with_context(|| format!("--layer-range end `{end_s}`"))?;
    if start >= end {
        return Err(anyhow!(
            "--layer-range start ({start}) must be < end ({end})"
        ));
    }
    Ok((start, end))
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
            if let Some(dir) = &args.ov_cache_dir {
                b = b.with_cache_dir(dir);
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
            Ok(Box::new(b))
        }
        EngineKind::OvRuntime => {
            let mut b = OvRuntimeBuilder::new(&args.model, args.rank, args.total, &args.device);
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
            if args.layer_range.is_some() && args.rank_balance.is_some() {
                return Err(anyhow!(
                    "--layer-range and --rank-balance are mutually exclusive; \
                     pass one or the other"
                ));
            }
            let mut cfg = SparseMoEBuilderConfig::new(&args.model, &args.device)
                .with_rank(args.rank, args.total);
            if let Some(dir) = &args.ov_cache_dir {
                cfg.cache_dir = Some(dir.clone());
            }
            if let Some(lr) = args.layer_range.as_deref() {
                let (start, end) = parse_layer_range(lr)?;
                cfg = cfg.with_layer_range_strategy(LayerRangeStrategy::Explicit { start, end });
            } else if matches!(args.rank_balance, Some(RankBalance::Auto)) {
                cfg = cfg.with_layer_range_strategy(LayerRangeStrategy::Auto);
            }
            Ok(Box::new(SparseMoEBuilder::new(cfg)))
        }
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

    info!(
        engine = ?args.engine,
        rank = args.rank,
        total = args.total,
        device = %args.device,
        model = %args.model,
        "tahoma worker starting"
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
        layer_start: 0,
        layer_end: 0,
        total_layers: 0,
        device: args.device.clone(),
        is_first_stage: is_first,
        is_last_stage: is_last,
        tp_size: 1,
        tp_rank: 0,
    };

    let builder = build_builder(&args)?;
    let runner = Arc::new(Runner::new(builder));
    let listen = if !is_first {
        Some((listen_host.as_str(), listen_port))
    } else {
        None
    };
    runner.start_with_listen(peers, shard, listen).await?;

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
        let chat_template =
            tahoma_api::load_chat_template_config(std::path::Path::new(&args.model));
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
        let mut cfg = tahoma_api::Config::default();
        cfg.chat_template = chat_template;
        let app = tahoma_api::make_router_with_config(runner.clone(), args.model.clone(), cfg);
        let listener = tokio::net::TcpListener::bind((api_host.as_str(), api_port)).await?;
        info!(host = %api_host, port = api_port, "API serving");
        // NODELAY-on-accept wrapper. tokio's TcpStream defaults to
        // NODELAY=false (Nagle on); for SSE streaming small per-token
        // chunks, Nagle aggregates them into ~3500 B bursts every
        // ~1 s — the demo experiences this as bursty token arrival
        // in the browser. Wrap the listener so every accepted stream
        // gets set_nodelay(true). Per-event flush on the body side
        // is in tahoma_api::stream_completion (Body::from_stream of
        // raw bytes, not the axum::Sse wrapper which has its own
        // KeepAlive batching).
        serve_with_nodelay(listener, app)
            .await
            .context("serve_with_nodelay")?;
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
/// `CARGO_MANIFEST_DIR` is `<workspace>/crates/tahoma-cli`; jumping two
/// levels up reaches the workspace root reliably on both Unix and Windows
/// (raw `../../../tools/...` mixes path separators on Windows and breaks
/// `include_str!`).
const EXPORT_SCRIPT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/export_shards.py"
));

async fn cmd_shard(args: ShardArgs) -> Result<()> {
    use std::io::Write;
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

    // Write the embedded script to a temp file so we have a real path.
    let mut tmp = tempfile::Builder::new()
        .prefix("tahoma-export-")
        .suffix(".py")
        .tempfile()
        .context("creating temp file for embedded exporter")?;
    tmp.write_all(EXPORT_SCRIPT.as_bytes())
        .context("writing embedded exporter to temp file")?;
    tmp.flush().ok();
    let script_path = tmp.path().to_owned();

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
    eprintln!(
        "\nShard tree written to {}. Run with:\n  tahoma worker --rank 0 --total {} \
         --engine ov-runtime --device GPU --model {} \
         --next <next-host>:9100 --api :8000",
        args.output_dir, args.num_stages, args.output_dir
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parse_layer_range_accepts_28_32_split_for_k26() {
        // The headline iter-081 case: K2.6's 60 MoE layers split 28/32
        // (rank 0 holds shells 1..29 = 28 layers, rank 1 holds 29..61
        // = 32 layers + the head).
        assert_eq!(parse_layer_range("1..29").unwrap(), (1, 29));
        assert_eq!(parse_layer_range("29..61").unwrap(), (29, 61));
    }

    #[test]
    fn parse_layer_range_tolerates_whitespace() {
        // clap strips outer whitespace, but a defensive trim around
        // the `..` lets `--layer-range "1 .. 29"` work too.
        assert_eq!(parse_layer_range("1 .. 29").unwrap(), (1, 29));
    }

    #[test]
    fn parse_layer_range_rejects_missing_separator() {
        let err = parse_layer_range("1-29").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("start..end"), "got {msg}");
    }

    #[test]
    fn parse_layer_range_rejects_non_numeric() {
        assert!(parse_layer_range("foo..29").is_err());
        assert!(parse_layer_range("1..bar").is_err());
    }

    #[test]
    fn parse_layer_range_rejects_inverted_or_empty() {
        // start >= end is a config error caught at the CLI boundary
        // — engine-side validation catches it again as defense in
        // depth, but the user-facing message is friendlier here.
        assert!(parse_layer_range("29..1").is_err());
        assert!(parse_layer_range("5..5").is_err());
    }

    #[test]
    fn worker_args_parses_layer_range_flag() {
        // Smoke-test the clap derive: --layer-range should be
        // optional, and when passed it should round-trip through to
        // WorkerArgs.layer_range as Some(string).
        let cli = Cli::try_parse_from([
            "tahoma",
            "worker",
            "--rank",
            "0",
            "--total",
            "2",
            "--model",
            "/tmp/k26",
            "--engine",
            "sparse-moe",
            "--layer-range",
            "1..29",
        ])
        .expect("clap parse");
        match cli.cmd {
            Command::Worker(args) => {
                assert_eq!(args.layer_range.as_deref(), Some("1..29"));
                assert_eq!(args.rank_balance, None);
            }
            _ => panic!("expected Worker subcommand"),
        }
    }

    #[test]
    fn worker_args_parses_rank_balance_auto_flag() {
        let cli = Cli::try_parse_from([
            "tahoma",
            "worker",
            "--rank",
            "0",
            "--total",
            "2",
            "--model",
            "/tmp/k26",
            "--engine",
            "sparse-moe",
            "--rank-balance",
            "auto",
        ])
        .expect("clap parse");
        match cli.cmd {
            Command::Worker(args) => {
                assert_eq!(args.rank_balance, Some(RankBalance::Auto));
                assert_eq!(args.layer_range, None);
            }
            _ => panic!("expected Worker subcommand"),
        }
    }

    #[test]
    fn worker_args_default_omits_balance_flags() {
        // Critical: existing CLI invocations without the new flags
        // must still parse, with both options coming out as None so
        // the engine takes the historical even-split path.
        let cli = Cli::try_parse_from([
            "tahoma",
            "worker",
            "--rank",
            "0",
            "--total",
            "2",
            "--model",
            "/tmp/k26",
            "--engine",
            "sparse-moe",
        ])
        .expect("clap parse");
        match cli.cmd {
            Command::Worker(args) => {
                assert!(args.layer_range.is_none(), "default must be None");
                assert!(args.rank_balance.is_none(), "default must be None");
            }
            _ => panic!("expected Worker subcommand"),
        }
    }
}
