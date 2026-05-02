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
use tahoma_runner::Runner;
use tahoma_types::{GenerationTask, PeerEndpoint, PeerLayout, ShardSpec};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "tahoma", about = "Distributed LLM inference for Intel hardware")]
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
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum EngineKind {
    Mock,
    OvGenai,
    OvRuntime,
    OvDistSpec,
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
}

pub async fn run(cli: Cli) -> Result<()> {
    init_tracing(&cli.log_level);
    match cli.cmd {
        Command::Engines => cmd_engines(),
        Command::Worker(args) => cmd_worker(args).await,
    }
}

fn init_tracing(level: &str) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level));
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
                let mut b =
                    OvDistSpecBuilder::new(&args.model, draft, &args.device, args.spec_k);
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
                let mut b = OvDistSpecWorkerBuilder::new(
                    &args.model,
                    args.rank,
                    args.total,
                    &args.device,
                );
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
        let app = tahoma_api::make_router(runner.clone(), args.model.clone());
        let listener = tokio::net::TcpListener::bind((api_host.as_str(), api_port)).await?;
        info!(host = %api_host, port = api_port, "API serving");
        axum::serve(listener, app).await.context("axum serve")?;
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
