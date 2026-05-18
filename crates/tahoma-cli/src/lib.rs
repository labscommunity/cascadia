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
use tahoma_engine_sparse_moe::{SparseMoEBuilder, SparseMoEBuilderConfig};
use tahoma_runner::Runner;
use tahoma_types::{GenerationTask, PeerEndpoint, PeerLayout, ShardSpec};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

/// String form of an engine kind, used in NodeInfo.engines for discovery
/// and in the dashboard's "Engines" pill list. Stable wire format —
/// matches the strings `tahoma engines` already prints.
fn engine_name(kind: EngineKind) -> &'static str {
    match kind {
        EngineKind::Mock => "mock",
        EngineKind::OvGenai => "ov-genai",
        EngineKind::OvRuntime => "ov-runtime",
        EngineKind::OvDistSpec => "ov-dist-spec",
        EngineKind::SparseMoe => "sparse-moe",
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

/// Best-effort hostname for the local node. Used as the human-readable
/// prefix of `node_id`. Falls back to "node" if the `hostname` command is
/// unavailable or returns something empty.
fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "node".to_owned())
}

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
            let mut cfg = SparseMoEBuilderConfig::new(&args.model, &args.device)
                .with_rank(args.rank, args.total);
            if let Some(dir) = &args.ov_cache_dir {
                cfg.cache_dir = Some(dir.clone());
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

    // Every worker advertises itself via mDNS — not just rank 0 — so the
    // coordinator's dashboard /api/topology can render the full cluster
    // and not just self. Best-effort: a host without a working multicast
    // path (CI sandbox, restricted LAN) still serves; the dashboard just
    // shows fewer nodes. We bind `_discovery` for the rest of this
    // function so its Drop unregisters the mDNS record cleanly on
    // shutdown (relay loop or `serve_with_nodelay`).
    let topology = tahoma_topology::Topology::new();
    let engines = if !args.advertise_engines.is_empty() {
        args.advertise_engines.clone()
    } else {
        vec![engine_name(args.engine).to_owned()]
    };
    let device = args
        .advertise_device
        .clone()
        .unwrap_or_else(|| args.device.clone());
    let self_node = tahoma_topology::NodeInfo {
        node_id: format!("{}-r{}", hostname(), args.rank),
        host: tahoma_discovery::local_ip().to_string(),
        port: listen_port,
        namespace: "default".to_owned(),
        device,
        memory_mb: 0,
        engines,
        last_seen: 0.0,
    };
    topology.add_node(self_node.clone());
    let mut discovery = tahoma_discovery::DiscoveryService::new(topology.clone(), "default");
    if let Err(e) = discovery.start(self_node.clone()) {
        warn!(error = %e, "mDNS discovery failed to start; cluster topology may be incomplete");
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
    let self_node_for_heartbeat = self_node.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(2));
        tick.tick().await; // skip the immediate first tick — we already added the node above.
        loop {
            tick.tick().await;
            topology_for_heartbeat.add_node(self_node_for_heartbeat.clone());
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
            let nodes = topology_for_probe.nodes();
            let probes: Vec<_> = nodes
                .into_iter()
                .filter(|n| n.node_id != self_id_for_probe)
                .map(|n| {
                    let host = n.host.clone();
                    let port = n.port;
                    let id = n.node_id.clone();
                    tokio::spawn(async move { (id, probe_peer(&host, port).await) })
                })
                .collect();
            for jh in probes {
                if let Ok((dst_id, Some(latency_ms))) = jh.await {
                    topology_for_probe.measure(
                        self_id_for_probe.clone(),
                        dst_id,
                        latency_ms,
                        0.0,
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
        let cfg = tahoma_api::Config {
            chat_template,
            ..Default::default()
        };
        let max_concurrent = cfg.max_concurrent_requests as u64;
        let api_router =
            tahoma_api::make_router_with_config(runner.clone(), args.model.clone(), cfg);

        // `topology` was populated above (every worker advertises +
        // browses) so by the time the dashboard binds, mDNS may already
        // have discovered other ranks on the LAN.
        let dash_state = tahoma_dashboard::DashboardState {
            topology,
            stats: tahoma_dashboard::DashboardStats::new(max_concurrent),
        };
        // Compose: OpenAI-compat routes (/v1/*, /health) stay at root for
        // backward compatibility with existing clients; dashboard-internal
        // routes live at /api/* plus the SPA (when the `dashboard-embed`
        // feature is on) at /. The dashboard router carries the SPA
        // fallback, so it must be merged second.
        let app = api_router.merge(tahoma_dashboard::make_router(dash_state));

        let listener = tokio::net::TcpListener::bind((api_host.as_str(), api_port)).await?;
        info!(host = %api_host, port = api_port, "API + dashboard serving");
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
