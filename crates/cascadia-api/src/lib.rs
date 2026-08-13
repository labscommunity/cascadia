//! OpenAI-compatible HTTP server.
//!
//! Mirrors the subset of `cascadia/api/server.py` needed for parity with PR
//! #2's e2e bench: `/health`, `/v1/models`, `/v1/chat/completions`
//! (non-streaming + SSE streaming). Tools, logprobs, /events, /state,
//! Ollama dialect deferred — see Phase 5 follow-up.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::SystemTime;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use cascadia_runner::Runner;
use cascadia_types::GenerationTask;
use chrono::Utc;
use futures::stream::{self, Stream};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tracing::{info, warn};
use uuid::Uuid;

/// Maximum HTTP request body the chat-completions endpoint accepts. 64
/// KiB is plenty for any legitimate chat completion (prompt + system +
/// few past turns); 1 MiB-class adversarial bodies are rejected with
/// 413 Payload Too Large. Increase only with awareness of the
/// downstream KV-cache cost (attention is O(seq²)).
pub const DEFAULT_MAX_BODY_BYTES: usize = 64 * 1024;

/// Default cap on concurrent chat-completion requests. The single
/// engine processes one task at a time anyway, so a small queue is
/// healthier than unbounded admission. Adjustable via
/// [`Config::max_concurrent_requests`].
pub const DEFAULT_MAX_CONCURRENT: usize = 16;

/// Default cap on tokenized prompt length passed to the engine.
/// Bounded so a 64 KiB JSON body of "a"*N characters that compresses
/// to a multi-million-token prompt cannot trigger O(seq²) GPU work.
/// Mirrors the Llama-3.1 default 128 K context window minus headroom.
pub const DEFAULT_MAX_PROMPT_BYTES: usize = 32 * 1024;

/// Live request/token counters bumped on the chat hot path. Shared (via
/// `Arc`) with the dashboard's `/api/stats`. Defined in `cascadia-types`
/// (re-exported here) so the dashboard can read it without depending on
/// this whole HTTP-server crate.
pub use cascadia_types::ApiStats;

/// Tokens a chunk contributes to the cumulative counter. The convention
/// (engine `n_tokens` authoritative, else one per non-empty chunk — #55)
/// lives on [`cascadia_types::Chunk::token_count`] so the API's `usage`,
/// the dashboard counters, and the runner's Prometheus metrics can never
/// drift apart; this alias keeps the historical call sites and tests.
fn chunk_token_count(chunk: &cascadia_types::Chunk) -> u32 {
    chunk.token_count()
}

/// RAII guard: bumps `requests_in_flight` on construction and decrements it
/// on drop, so the gauge is correct even on early return, client disconnect,
/// or a mid-stream engine error. Mirrors into the Prometheus
/// `cascadia_inflight_tasks` gauge (#16) so both the dashboard and
/// `/metrics` see the same lifecycle.
struct InFlightGuard(Arc<ApiStats>);

impl InFlightGuard {
    fn new(stats: Arc<ApiStats>) -> Self {
        stats.requests_in_flight.fetch_add(1, Ordering::Relaxed);
        cascadia_metrics::INFLIGHT_TASKS.inc();
        InFlightGuard(stats)
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.requests_in_flight.fetch_sub(1, Ordering::Relaxed);
        cascadia_metrics::INFLIGHT_TASKS.dec();
    }
}

/// Count a pre-engine rejection in `cascadia_api_rejected_total`. Reasons
/// are a closed set — see the metric's docs in `cascadia-metrics`.
fn count_rejected(reason: &str) {
    cascadia_metrics::API_REJECTED_TOTAL
        .with_label_values(&[reason])
        .inc();
}

#[derive(Clone)]
pub struct AppState {
    pub runner: Arc<Runner>,
    pub model_id: String,
    pub permits: Arc<Semaphore>,
    /// Live counters, shared with the dashboard. See [`ApiStats`].
    pub stats: Arc<ApiStats>,
    pub max_prompt_bytes: usize,
    /// Pre-built minijinja environment for the model's HF chat template (from
    /// tokenizer_config.json's `chat_template` field or a sibling
    /// `chat_template.jinja`), parsed ONCE at router construction. When set,
    /// /v1/chat/completions renders messages through it; when None (no template
    /// or a parse error), falls back to the legacy "role: content\n…" join.
    pub chat_env: Option<Arc<minijinja::Environment<'static>>>,
    pub bos_token: Arc<str>,
    pub eos_token: Arc<str>,
    /// Engines that own their native chat templating (ov-genai): defer to the
    /// engine for the thinking-ON path and only render here when thinking is
    /// OFF (to inject the empty `<think></think>`). Keeps the working
    /// thinking-on path byte-identical to the engine's native render.
    pub defer_template_on_thinking: bool,
    /// Pipeline readiness for `/health`. Starts `true`; a completion that fails
    /// with a 5xx (e.g. the distributed pipeline dropped a peer link) flips it
    /// `false`, a success flips it back `true` — so `/health` reflects the real
    /// state of the chain instead of always returning 200 while requests 503.
    pub ready: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Clone, Debug, Default)]
pub struct ChatTemplateConfig {
    pub template: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub max_body_bytes: usize,
    pub max_concurrent_requests: usize,
    pub max_prompt_bytes: usize,
    pub chat_template: ChatTemplateConfig,
    /// See [`AppState::defer_template_on_thinking`]. Set by the CLI for ov-genai.
    pub defer_template_on_thinking: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT,
            max_prompt_bytes: DEFAULT_MAX_PROMPT_BYTES,
            chat_template: ChatTemplateConfig::default(),
            defer_template_on_thinking: false,
        }
    }
}

/// Read tokenizer_config.json from the shard's `tokenizer/` subdir and
/// extract the chat_template + bos/eos token strings. Returns Default
/// (all None) if any step fails — the caller falls back to the legacy
/// "role: content" formatting in that case. Some HF tokenizer configs
/// store special tokens as objects ({"content": "...", ...}) rather than
/// strings; both shapes are handled.
///
/// Newer HF exports (Gemma 3/4, recent Llama) drop the inline
/// `chat_template` JSON field and ship the template as a sibling
/// `chat_template.jinja` file instead — too large/complex to embed in
/// JSON. When the inline field is absent we fall back to that file, so
/// instruct models render through their real template rather than the
/// legacy formatter (which degenerates instruct models).
pub fn load_chat_template_config(model_dir: &std::path::Path) -> ChatTemplateConfig {
    // Strict, unchanged semantics: a present AND parsable
    // tokenizer_config.json is required before anything (including the
    // sibling jinja file) is considered.
    let tok_dir = model_dir.join("tokenizer");
    let Ok(bytes) = std::fs::read(tok_dir.join("tokenizer_config.json")) else {
        return ChatTemplateConfig::default();
    };
    if serde_json::from_slice::<serde_json::Value>(&bytes).is_err() {
        return ChatTemplateConfig::default();
    }
    load_chat_template_config_at(&tok_dir)
}

/// Like [`load_chat_template_config`] but reads `dir` itself instead of a
/// `tokenizer/` subdir, and accepts a `chat_template.jinja` without any
/// `tokenizer_config.json`. For model layouts that keep tokenizer files at
/// the model root (the qwen36 surgery shard tree, which ships the jinja
/// file but no tokenizer_config).
pub fn load_chat_template_config_at(tok_dir: &std::path::Path) -> ChatTemplateConfig {
    let v = std::fs::read(tok_dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .unwrap_or_default();
    let template = v
        .get("chat_template")
        .and_then(|t| t.as_str())
        .map(|s| s.to_owned())
        .or_else(|| std::fs::read_to_string(tok_dir.join("chat_template.jinja")).ok())
        // An empty / whitespace-only template (a zero-byte placeholder, or a
        // download truncated to empty) must NOT be treated as usable: rendering
        // through it produces an empty prompt, which is worse than the legacy
        // "role: content" formatter. Drop to None so the caller falls back.
        .filter(|s| !s.trim().is_empty());
    let extract_token = |key: &str| -> Option<String> {
        let val = v.get(key)?;
        if let Some(s) = val.as_str() {
            return Some(s.to_owned());
        }
        val.get("content")
            .and_then(|c| c.as_str())
            .map(|s| s.to_owned())
    };
    ChatTemplateConfig {
        template,
        bos_token: extract_token("bos_token"),
        eos_token: extract_token("eos_token"),
    }
}

/// Build the router with default request-size + concurrency limits.
/// For custom limits use [`make_router_with_config`].
pub fn make_router(runner: Arc<Runner>, model_id: impl Into<String>) -> Router {
    make_router_with_config(runner, model_id, Config::default())
}

pub fn make_router_with_config(
    runner: Arc<Runner>,
    model_id: impl Into<String>,
    cfg: Config,
) -> Router {
    make_router_with_stats(runner, model_id, cfg, Arc::new(ApiStats::default()))
}

/// Like [`make_router_with_config`] but takes a caller-owned [`ApiStats`] so
/// the same counter set can be shared with the dashboard's `/api/stats`.
pub fn make_router_with_stats(
    runner: Arc<Runner>,
    model_id: impl Into<String>,
    cfg: Config,
    stats: Arc<ApiStats>,
) -> Router {
    let state = AppState {
        runner,
        model_id: model_id.into(),
        permits: Arc::new(Semaphore::new(cfg.max_concurrent_requests)),
        stats,
        max_prompt_bytes: cfg.max_prompt_bytes,
        // Parse the chat template once here (not per request). A parse error
        // downgrades to the legacy formatter rather than failing every request.
        chat_env: cfg
            .chat_template
            .template
            .as_deref()
            .and_then(|src| match build_chat_env(src) {
                Ok(env) => {
                    report_chat_template_smoke(&env);
                    Some(Arc::new(env))
                }
                Err(e) => {
                    warn!(error = %e, "chat_template failed to parse at startup; using legacy formatter");
                    None
                }
            }),
        bos_token: Arc::from(cfg.chat_template.bos_token.unwrap_or_default()),
        eos_token: Arc::from(cfg.chat_template.eos_token.unwrap_or_default()),
        defer_template_on_thinking: cfg.defer_template_on_thinking,
        ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
    };
    // Register the label-less metric families up-front so a scrape before
    // any traffic still lists them (#16). Idempotent.
    cascadia_metrics::init();
    let mw_state = state.clone();
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/cancel/:task_id", post(cancel))
        .with_state(state)
        // Flip `ready` from each completion's final status so /health tracks the
        // pipeline (a peer-link drop fails generation with a 5xx before streaming
        // starts, so the response status is authoritative for that case).
        .layer(axum::middleware::from_fn_with_state(
            mw_state,
            track_pipeline_health,
        ))
        // Cap the JSON body so an attacker can't OOM the server with
        // a multi-GB request. Apply at router level so it applies to
        // every route, not just chat_completions.
        .layer(DefaultBodyLimit::max(cfg.max_body_bytes))
        // Outermost: request count + latency per matched route (#16), so it
        // observes rejections from the inner layers too.
        .layer(axum::middleware::from_fn(track_http_metrics))
}

/// Request middleware: per-route request counter + latency histogram. Uses
/// the MATCHED route template (bounded cardinality), never the raw URI; a
/// request that matched no route is bucketed as `other`. For streaming
/// (SSE) responses the duration is time-to-response-head — full generation
/// time is `cascadia_generation_duration_seconds` (see `cascadia-metrics`).
async fn track_http_metrics(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let endpoint = req
        .extensions()
        .get::<axum::extract::MatchedPath>()
        .map(|p| p.as_str().to_owned())
        .unwrap_or_else(|| "other".to_owned());
    let start = std::time::Instant::now();
    let resp = next.run(req).await;
    cascadia_metrics::HTTP_REQUEST_DURATION_SECONDS
        .with_label_values(&[&endpoint])
        .observe(start.elapsed().as_secs_f64());
    cascadia_metrics::HTTP_REQUESTS_TOTAL
        .with_label_values(&[&endpoint, resp.status().as_str()])
        .inc();
    resp
}

/// `GET /metrics` — Prometheus text exposition format (#16). Serves the
/// process-global registry: request, generation, engine, and transport
/// metrics. Only stages started with `--api` expose it (same as every
/// other HTTP route); relay-only stages have no HTTP listener.
async fn metrics() -> impl IntoResponse {
    let (content_type, body) = cascadia_metrics::encode_text();
    ([(header::CONTENT_TYPE, content_type)], body)
}

/// Response middleware: for completion routes, flip `AppState.ready` from the
/// response status (5xx → not ready, 2xx → ready) so `/health` reflects whether
/// the pipeline is actually serving. Non-completion routes don't touch it.
async fn track_pipeline_health(
    axum::extract::State(state): axum::extract::State<AppState>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use std::sync::atomic::Ordering;
    let is_completion = matches!(req.uri().path(), "/v1/chat/completions" | "/v1/completions");
    let resp = next.run(req).await;
    if is_completion {
        let s = resp.status();
        if s.is_server_error() {
            state.ready.store(false, Ordering::Relaxed);
        } else if s.is_success() {
            state.ready.store(true, Ordering::Relaxed);
        }
    }
    resp
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn health(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> (StatusCode, Json<HealthResponse>) {
    use std::sync::atomic::Ordering;
    if state.ready.load(Ordering::Relaxed) {
        (StatusCode::OK, Json(HealthResponse { status: "ok" }))
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(HealthResponse { status: "degraded" }),
        )
    }
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: &'static str,
    created: i64,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelEntry>,
}

async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelEntry {
            id: state.model_id.clone(),
            object: "model",
            created: 0,
            owned_by: "cascadia",
        }],
    })
}

#[derive(Deserialize, Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, deserialize_with = "de_null_content")]
    pub content: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

/// OpenAI tool spec (request input). `function` is opaque JSON (name,
/// description, parameters schema) forwarded verbatim to the chat template.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Tool {
    pub r#type: String,
    pub function: serde_json::Value,
}

/// A structured tool call (response output + round-trip input).
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded argument object (OpenAI ships this as a string).
    pub arguments: String,
}

/// null/missing content -> "" (assistant tool-call turns send content: null).
fn de_null_content<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_default())
}

/// OpenAI `stop` is either a single string or an array of strings.
#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum StopSpec {
    Single(String),
    Multiple(Vec<String>),
}

/// OpenAI `stream_options`. Only `include_usage` is honored today.
#[derive(Deserialize, Debug, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopSpec>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// OpenAI `logprobs` (bool). When true, per-token logprobs are returned
    /// (engine-permitting).
    #[serde(default)]
    pub logprobs: bool,
    /// OpenAI `top_logprobs` (0..=20): how many alternatives per position.
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Hybrid-reasoning switch (Qwen3+ convention). Omitted = model-default
    /// (thinking on, subject to the `reasoning_effort: "none"` override below);
    /// an explicit `true`/`false` always wins. `Option` rather than `bool` so
    /// "explicitly set" is distinguishable from "defaulted" — see
    /// [`ChatCompletionRequest::effective_enable_thinking`].
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// Reasoning depth, mapped onto GLM-5's two template levels (High, Max) by
    /// [`ChatCompletionRequest::effective_reasoning_effort`] before it reaches
    /// the template — see that method for the mapping table.
    ///
    /// `"none"` additionally disables thinking (sets the effective
    /// `enable_thinking` to false), unless the request explicitly set
    /// `enable_thinking` itself, which always wins. This exists so
    /// OpenAI-shaped clients, which have no thinking toggle, still have an
    /// off-switch.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// vLLM/SGLang's convention for reaching the underlying chat template:
    /// `{"enable_thinking": false}`. Outranks the legacy top-level
    /// `enable_thinking` and the `reasoning_effort: "none"` off-switch — see
    /// [`ChatCompletionRequest::effective_enable_thinking`] for the full
    /// precedence chain. The map is free-form in the ecosystem; members other
    /// than `enable_thinking` are accepted and ignored, never forwarded to
    /// the template.
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
}

/// `chat_template_kwargs` as sent by vLLM/SGLang-shaped clients. Only
/// `enable_thinking` is consumed; unknown members deserialize to nothing and
/// are silently dropped rather than erroring.
#[derive(Deserialize, Debug, Default)]
pub struct ChatTemplateKwargs {
    #[serde(default)]
    pub enable_thinking: Option<bool>,
}

impl ChatCompletionRequest {
    /// Build the engine-facing `SamplingParams`, applying OpenAI defaults
    /// (top_p=1.0, penalties=0.0) when a field is omitted.
    fn sampling_params(&self) -> cascadia_types::SamplingParams {
        cascadia_types::SamplingParams {
            top_p: self.top_p.unwrap_or(1.0),
            top_k: self.top_k.unwrap_or(0),
            seed: self.seed,
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            stop: self
                .stop
                .as_ref()
                .map(|s| match s {
                    StopSpec::Single(x) => vec![x.clone()],
                    StopSpec::Multiple(v) => v.clone(),
                })
                .unwrap_or_default(),
        }
    }

    /// Top-logprobs count to request from the engine (0 = disabled).
    fn logprobs_count(&self) -> u32 {
        if self.logprobs {
            self.top_logprobs.unwrap_or(1).clamp(1, 20)
        } else {
            0
        }
    }

    /// Resolves `enable_thinking`, highest precedence first:
    /// `chat_template_kwargs.enable_thinking` (the vLLM/SGLang convention),
    /// then the legacy top-level `enable_thinking`, then `reasoning_effort:
    /// "none"` turning thinking off; omission of all three means on. Both
    /// explicit toggles must beat `"none"` for the same reason: `"none"`
    /// exists to give clients *without* a toggle an off-switch, and must
    /// never override a client that used one.
    fn effective_enable_thinking(&self) -> bool {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|k| k.enable_thinking)
            .or(self.enable_thinking)
            .unwrap_or_else(|| self.reasoning_effort.as_deref() != Some("none"))
    }

    /// Maps the OpenAI effort vocabulary onto what GLM-5's template can
    /// express — exactly two levels, selected by the literal string `"high"`
    /// vs. anything else (including undefined), which resolves to Max:
    ///
    /// | client sends                                       | template sees   |
    /// | --------------------------------------------------- | --------------- |
    /// | omitted, `minimal`, `low`, `medium`, `high`, `none`, unrecognised | `"high"` |
    /// | `xhigh`, `max`                                       | `"max"` (also resolves to Max) |
    ///
    /// Omission must NOT reach the template as undefined — that is the Max-by
    /// -omission defect this mapping exists to close. `"none"` sits in the
    /// conservative `"high"` bucket rather than with `xhigh`/`max`: it signals
    /// *less* than minimal effort, not more, and its value is moot for the
    /// common case anyway since `effective_enable_thinking` gates the whole
    /// effort line off. It only matters when an explicit `enable_thinking:
    /// true` keeps thinking on despite `"none"` — bucketing it with Max there
    /// would silently escalate, which is the exact failure this mapping exists
    /// to prevent.
    ///
    /// Returns a literal `"max"` rather than `None`/undefined for `xhigh`/
    /// `max`: an explicit choice and an omitted one both currently render the
    /// same template output (`defined and == 'high'` is false either way),
    /// but they are not the same fact, and collapsing them to `None` would
    /// make "caller chose max" indistinguishable from "caller said nothing"
    /// at the Rust type level.
    fn effective_reasoning_effort(&self) -> &'static str {
        match self.reasoning_effort.as_deref() {
            Some("xhigh") | Some("max") => "max",
            _ => "high",
        }
    }
}

fn default_max_tokens() -> u32 {
    256
}

#[derive(Serialize)]
struct ChatChoiceMessage {
    role: &'static str,
    /// `Option` so a tool-call turn emits `content: null` (OpenAI shape).
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

/// OpenAI chat `logprobs` object: one entry per generated token under
/// `content`. Present only when the request set `logprobs: true` AND the
/// engine emitted per-token logprobs.
#[derive(Serialize)]
struct ChatLogprobs {
    content: Vec<cascadia_types::TokenLogprobs>,
}

#[derive(Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatChoiceMessage,
    finish_reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<ChatLogprobs>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

// --- Legacy /v1/completions (OpenAI-compatible) -------------------------

/// OpenAI `prompt` is a single string or an array of strings.
#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum PromptSpec {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Deserialize, Debug)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: PromptSpec,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopSpec>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub logprobs: Option<u32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Prepend the prompt to the returned text (OpenAI `echo`).
    #[serde(default)]
    pub echo: bool,
}

impl CompletionRequest {
    fn sampling_params(&self) -> cascadia_types::SamplingParams {
        cascadia_types::SamplingParams {
            top_p: self.top_p.unwrap_or(1.0),
            top_k: self.top_k.unwrap_or(0),
            seed: self.seed,
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            stop: self
                .stop
                .as_ref()
                .map(|s| match s {
                    StopSpec::Single(x) => vec![x.clone()],
                    StopSpec::Multiple(v) => v.clone(),
                })
                .unwrap_or_default(),
        }
    }

    /// Legacy `logprobs` is an integer count (number of top logprobs), unlike
    /// the chat endpoint's bool. 0 / None disables.
    fn logprobs_count(&self) -> u32 {
        self.logprobs.unwrap_or(0).min(20)
    }
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: u32,
    finish_reason: &'static str,
    // NOTE: the real OpenAI *legacy* completions `logprobs` object has a
    // different shape than chat's (`{tokens, token_logprobs, top_logprobs,
    // text_offset}` vs chat's `{content: [...]}`). We reuse the chat-style
    // `ChatLogprobs` here for now; it's moot until an engine actually emits
    // per-token logprobs (none do yet — same deferral as #14), at which point
    // converting to the legacy schema is the follow-up. The field is omitted
    // entirely when empty, so a legacy client sees no malformed object today.
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<ChatLogprobs>,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

fn now_unix() -> i64 {
    Utc::now().timestamp()
}

/// Last-resort formatting when no chat template is available.
///
/// The previous behaviour joined every turn as `role: content\n…`,
/// which works on permissive base models but produces visibly broken
/// transcripts on instruct models and on the mock engine — the engine
/// sees the assistant's prior turn re-inserted as input, and on the
/// next request the response includes that prior content verbatim.
///
/// Without a real chat template we can't reconstruct the model's
/// multi-turn format honestly, so render only the latest user message.
/// That gives one-shot prompting (no multi-turn memory) but doesn't
/// produce confusing duplicated output. Multi-turn coherence requires
/// a real tokenizer_config.json with `chat_template` populated.
///
/// When there's no `user` turn at all (e.g. a system-only request), fall
/// back to the last message of any role rather than rendering empty — an
/// empty render trips the 400 "no prompt content" guard and would reject
/// a request the old all-turns formatter admitted.
fn render_prompt_legacy(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .or_else(|| messages.last())
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

/// Build the minijinja environment for a model's HF chat template ONCE, with
/// the template pre-parsed and the HF-compat shims installed. The template is
/// fixed at model load, so this is built at router construction and reused
/// across every request — the ~17 KB Gemma 3/4 macro template is parsed once,
/// not on each `/v1/chat/completions` call. Returns Err on parse failure so the
/// caller can fall back to the legacy formatter.
/// serde_json formatter matching Python's default `json.dumps` separators
/// (`", "` and `": "`). serde_json's compact formatter omits both spaces.
struct PyDumpsCompact;

impl serde_json::ser::Formatter for PyDumpsCompact {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

/// Escape every non-ASCII scalar as `\uXXXX` (surrogate pairs above the BMP),
/// the way `json.dumps(ensure_ascii=True)` does. Safe to run over serialized
/// JSON: structural bytes are all ASCII, so only string contents are touched.
fn escape_non_ascii(s: &str) -> String {
    if s.is_ascii() {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut buf = [0u16; 2];
            for unit in c.encode_utf16(&mut buf) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

/// `tojson` with Python `json.dumps` semantics, shadowing minijinja's builtin.
///
/// transformers replaces Jinja2's `tojson` with a bare `json.dumps(...,
/// ensure_ascii=False)`, so every HF chat template was authored against that
/// output. minijinja's builtin differs twice over: it escapes `< > & '` to
/// `\uXXXX` for HTML safety, and it uses serde_json's space-free separators.
/// Rendering through it puts tool schemas on the wire in bytes the model never
/// saw during training. The builtin also rejects any kwarg but `indent`, and a
/// rejected kwarg fails the whole render — which is how `ensure_ascii=False`
/// silently cost GLM-5 its tools.
fn py_tojson(
    value: minijinja::value::Value,
    indent: Option<minijinja::value::Value>,
    args: minijinja::value::Kwargs,
) -> Result<minijinja::value::Value, minijinja::Error> {
    use minijinja::value::Value;
    use minijinja::{Error, ErrorKind};

    let indent = match indent {
        Some(v) => Some(v),
        None => args.get::<Option<Value>>("indent")?,
    };
    let indent = match indent {
        None => None,
        Some(v) => match bool::try_from(v.clone()).ok() {
            Some(true) => Some(2),
            Some(false) => None,
            None => Some(usize::try_from(v)?),
        },
    };
    let ensure_ascii = args.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    // transformers' shim also accepts sort_keys and separators. Consume them so
    // an unknown-kwarg error can't reroute the render into the legacy
    // formatter, but refuse non-default values instead of ignoring them — a
    // silently wrong prompt is the failure mode this whole filter exists to
    // remove.
    if args.get::<Option<bool>>("sort_keys")?.unwrap_or(false) {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "tojson: sort_keys=True is not supported",
        ));
    }
    if args.get::<Option<Value>>("separators")?.is_some() {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "tojson: separators= is not supported",
        ));
    }
    args.assert_all_used()?;

    let mut out = Vec::<u8>::new();
    let res = match indent {
        // Python switches to `(',', ': ')` under indent, which is exactly what
        // serde_json's pretty formatter already emits.
        Some(n) => {
            let pad = " ".repeat(n);
            let fmt = serde_json::ser::PrettyFormatter::with_indent(pad.as_bytes());
            let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
            serde::Serialize::serialize(&value, &mut ser)
        }
        None => {
            let mut ser = serde_json::Serializer::with_formatter(&mut out, PyDumpsCompact);
            serde::Serialize::serialize(&value, &mut ser)
        }
    };
    res.map_err(|err| {
        Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(err)
    })?;

    let json = String::from_utf8(out).map_err(|err| {
        Error::new(ErrorKind::InvalidOperation, "JSON output was not UTF-8").with_source(err)
    })?;
    let json = if ensure_ascii {
        escape_non_ascii(&json)
    } else {
        json
    };
    // Safe-string like the builtin: the value is already JSON-encoded and must
    // not be HTML-escaped again if a caller enables autoescape.
    Ok(Value::from_safe_string(json))
}

/// Render a spread of message shapes through `env` and report the ones that
/// fail for engine-compatibility reasons.
///
/// A template that parses is not a template that renders. Every compatibility
/// gap found so far — a `tojson` kwarg, a Python string method, `strftime_now`
/// — lives inside a branch that only executes for a particular message shape,
/// so a tools-only failure stays invisible until the first tools request.
/// Running the shapes at load turns that into a startup signal.
///
/// `Rejected` outcomes are omitted: a template refusing a system role is it
/// doing its job, not a defect.
/// Log the smoke battery's verdict once, at load.
fn report_chat_template_smoke(env: &minijinja::Environment<'static>) {
    for (case, err) in chat_template_smoke_failures(env) {
        tracing::error!(
            case,
            error = %err,
            "chat_template does not render this message shape; requests of this shape will fail"
        );
    }
}

fn chat_template_smoke_failures(
    env: &minijinja::Environment<'static>,
) -> Vec<(&'static str, String)> {
    fn m(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
    let tools = [Tool {
        r#type: "function".into(),
        function: serde_json::json!({
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }),
    }];
    let call = m("assistant", "");
    let call = ChatMessage {
        tool_calls: Some(vec![ToolCall {
            id: "call_0".into(),
            r#type: "function".into(),
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: r#"{"city":"Paris"}"#.into(),
            },
        }]),
        ..call
    };
    let tool_reply = ChatMessage {
        tool_call_id: Some("call_0".into()),
        ..m("tool", "18C and clear")
    };

    let plain = [m("user", "hi")];
    let with_system = [m("system", "be brief"), m("user", "hi")];
    let multi_turn = [m("user", "hi"), m("assistant", "hello"), m("user", "more")];
    let round_trip = [m("user", "weather in Paris?"), call, tool_reply];

    let cases: [(&'static str, &[ChatMessage], Option<&[Tool]>, bool); 7] = [
        ("plain", &plain, None, true),
        ("system", &with_system, None, true),
        ("multi-turn", &multi_turn, None, true),
        ("tools", &plain, Some(&tools), true),
        ("tools+thinking-off", &plain, Some(&tools), false),
        ("tool-call round trip", &round_trip, Some(&tools), true),
        ("thinking-off", &plain, None, false),
    ];

    cases
        .into_iter()
        .filter_map(|(name, msgs, tools, thinking)| {
            match render_with_chat_env(env, msgs, "", "", thinking, None, tools) {
                Err(PromptRenderError::Failed(e)) => Some((name, e)),
                _ => None,
            }
        })
        .collect()
}

fn build_chat_env(template_src: &str) -> Result<minijinja::Environment<'static>, String> {
    use minijinja::value::Value;
    use minijinja::{Environment, Error, ErrorKind};

    let mut env = Environment::new();
    // HF chat templates were authored against transformers' Jinja2 env,
    // which calls Python string methods (.startswith, .endswith, .strip,
    // .split, .find, .replace, …). Minijinja doesn't ship those by
    // default, so we install the pycompat unknown-method callback —
    // Qwen3's template, for example, calls messages[0].role.startswith
    // and would otherwise fail.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    // HF templates throw via raise_exception("...") on malformed input
    // (e.g. a system message after the first user message). We surface
    // the message as a render error so the caller can decide whether to
    // fall back; either way the request shouldn't 500.
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, msg.clone())
            .with_source(TemplateRejection(msg)))
    });
    // Some templates (Llama 3) call strftime_now() to embed the current date in
    // the system prompt. UTC keeps the render reproducible across a fleet whose
    // nodes need not share a timezone. An unsupported format is an error, not
    // an empty string: a blank date reads as a model quirk, not a bug.
    env.add_function("strftime_now", |fmt: String| -> Result<Value, Error> {
        use std::fmt::Write as _;
        let mut out = String::new();
        write!(out, "{}", chrono::Utc::now().format(&fmt)).map_err(|_| {
            Error::new(
                ErrorKind::InvalidOperation,
                format!("strftime_now: unsupported format string {fmt:?}"),
            )
        })?;
        Ok(Value::from(out))
    });
    // Shadows minijinja's HTML-safe builtin; see `py_tojson`.
    env.add_filter("tojson", py_tojson);

    // HF templates authored for Jinja2 use the Python `.N` subscript
    // (`m.content.0.type`) which minijinja's parser rejects as a float
    // ("unexpected float, expected identifier or integer"). Rewrite it to the
    // bracket form `m.content[0]` that minijinja accepts. GLM-5's tool handling
    // hits this; without the rewrite the whole template fails to parse and the
    // API silently drops to the legacy formatter.
    let src = sanitize_numeric_dot_index(template_src);

    // add_template_owned (not add_template) so the Environment owns the source
    // and is 'static — it can then live in AppState behind an Arc.
    env.add_template_owned("chat", src.into_owned())
        .map_err(|e| format!("template parse: {e}"))?;
    Ok(env)
}

/// Rewrite Jinja2 numeric dot-index (`foo.0`, `bar.12`) to bracket-index
/// (`foo[0]`, `bar[12]`). Only a `.` that follows an identifier char / `]` / `)`
/// and precedes a digit run is converted, so real float literals (`1.5`, where
/// the `.` follows a digit) are left untouched. Byte-slice copies keep it
/// UTF-8-safe (the `.` and digits are ASCII, always char boundaries).
fn sanitize_numeric_dot_index(src: &str) -> std::borrow::Cow<'_, str> {
    let b = src.as_bytes();
    let mut out = String::new();
    let mut last = 0usize;
    let mut i = 0usize;
    let mut changed = false;
    while i < b.len() {
        if b[i] == b'.' && i > 0 {
            let prev = b[i - 1];
            let prev_base =
                prev.is_ascii_alphabetic() || prev == b'_' || prev == b']' || prev == b')';
            if prev_base && b.get(i + 1).is_some_and(|a| a.is_ascii_digit()) {
                let mut j = i + 1;
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
                out.push_str(&src[last..i]);
                out.push('[');
                out.push_str(&src[i + 1..j]);
                out.push(']');
                last = j;
                i = j;
                changed = true;
                continue;
            }
        }
        i += 1;
    }
    if !changed {
        return std::borrow::Cow::Borrowed(src);
    }
    out.push_str(&src[last..]);
    std::borrow::Cow::Owned(out)
}

/// Render messages through a pre-built chat environment (see [`build_chat_env`]).
/// Returns Err on any render error so the caller can fall back to
/// [`render_prompt_legacy`] rather than fail the request entirely.
fn render_with_chat_env(
    env: &minijinja::Environment<'static>,
    messages: &[ChatMessage],
    bos_token: &str,
    eos_token: &str,
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
    tools: Option<&[Tool]>,
) -> Result<String, PromptRenderError> {
    use minijinja::context;
    use minijinja::value::Value;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| PromptRenderError::Failed(format!("template lookup: {e}")))?;
    let messages_value: Vec<Value> = messages
        .iter()
        .map(|m| {
            let mut obj = serde_json::json!({ "role": m.role, "content": m.content });
            if let Some(tc) = &m.tool_calls {
                // HF chat templates (Qwen3 etc.) iterate `tool_call.arguments | items`,
                // expecting a JSON object; OpenAI ships `arguments` as a string. Parse it for
                // the template context (the wire format stays a string) so the render doesn't
                // error ("cannot convert value into pairs") and silently fall back to legacy.
                let mut tcv = serde_json::to_value(tc).unwrap_or_default();
                if let Some(arr) = tcv.as_array_mut() {
                    for call in arr {
                        if let Some(s) = call
                            .pointer("/function/arguments")
                            .and_then(|v| v.as_str())
                            .map(str::to_owned)
                        {
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&s) {
                                call["function"]["arguments"] = parsed;
                            }
                        }
                    }
                }
                obj["tool_calls"] = tcv;
            }
            if let Some(id) = &m.tool_call_id {
                obj["tool_call_id"] = serde_json::json!(id);
            }
            if let Some(name) = &m.name {
                obj["name"] = serde_json::json!(name);
            }
            Value::from_serialize(obj)
        })
        .collect();
    let tools_value: Option<Vec<Value>> =
        tools.map(|ts| ts.iter().map(Value::from_serialize).collect());
    let ctx = context! {
        messages => messages_value,
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
        // Left undefined when None so the template's own default applies —
        // GLM-5 resolves undefined to 'max'. Passing an empty string instead
        // would define it and still resolve to 'max', but would misrepresent
        // "caller said nothing" as "caller chose".
        reasoning_effort => reasoning_effort,
        bos_token => bos_token,
        eos_token => eos_token,
        tools => tools_value,
    };
    tmpl.render(ctx).map_err(|e| classify_render_error(&e))
}

/// Marker attached to `raise_exception(...)` errors so a template's own
/// validation stays distinguishable from an engine compatibility gap after
/// minijinja has wrapped it in template context.
#[derive(Debug)]
struct TemplateRejection(String);

impl std::fmt::Display for TemplateRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TemplateRejection {}

/// Why a chat-prompt render produced no prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptRenderError {
    /// The request cannot be served as asked: the template rejected it via
    /// `raise_exception(...)` (role ordering, an unsupported role), or it needs
    /// a capability this model lacks. A client error — resending unchanged
    /// cannot help.
    Rejected(String),
    /// The template could not be rendered, or none is loaded and the request
    /// needs one. A server error — the request was well-formed.
    Failed(String),
}

impl std::fmt::Display for PromptRenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(m) => write!(f, "chat template rejected the request: {m}"),
            Self::Failed(m) => write!(f, "chat template render failed: {m}"),
        }
    }
}

impl std::error::Error for PromptRenderError {}

fn classify_render_error(err: &minijinja::Error) -> PromptRenderError {
    let mut cur: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(e) = cur {
        if let Some(reject) = e.downcast_ref::<TemplateRejection>() {
            return PromptRenderError::Rejected(reject.0.clone());
        }
        cur = e.source();
    }
    PromptRenderError::Failed(format!("template render: {err}"))
}

/// The one place that decides what a failed render means.
///
/// A template render is the only thing that can put tool definitions into a
/// prompt — [`render_prompt_legacy`] has no way to express them, and emits none
/// of the model's role markers. Falling back on a tools request therefore
/// returns a confident answer to a question the model was never asked, so tools
/// requests fail loudly instead. Tool-less requests keep the fallback, since a
/// degraded prompt still beats a dead endpoint.
fn render_or_fallback(
    env: Option<&minijinja::Environment<'static>>,
    messages: &[ChatMessage],
    bos_token: &str,
    eos_token: &str,
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
    tools: Option<&[Tool]>,
) -> Result<String, PromptRenderError> {
    let has_tools = tools.is_some_and(|t| !t.is_empty());
    let Some(env) = env else {
        if has_tools {
            // vLLM likewise refuses tool use without a chat template rather
            // than answering as if no tools had been sent.
            return Err(PromptRenderError::Rejected(
                "this model has no chat template, so tool definitions cannot be rendered".into(),
            ));
        }
        return Ok(render_prompt_legacy(messages));
    };
    match render_with_chat_env(
        env,
        messages,
        bos_token,
        eos_token,
        enable_thinking,
        reasoning_effort,
        tools,
    ) {
        Ok(s) => Ok(s),
        // A raise_exception is the template validating its own input; other
        // engines surface it rather than papering over it.
        Err(e @ PromptRenderError::Rejected(_)) => Err(e),
        Err(e @ PromptRenderError::Failed(_)) if has_tools => Err(e),
        Err(e) => {
            tracing::error!(
                error = %e,
                "chat_template render failed; falling back to legacy formatter — prompt will lack role markers"
            );
            Ok(render_prompt_legacy(messages))
        }
    }
}

fn render_prompt(
    state: &AppState,
    messages: &[ChatMessage],
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
    tools: Option<&[Tool]>,
) -> Result<String, PromptRenderError> {
    // ov-genai applies the model's own template, so the engine gets raw text.
    // Tools would be dropped on the floor here, so refuse rather than pretend.
    // `reasoning_effort` (the caller's parameter above) is also dropped on
    // this branch, intentionally: ov-genai applies its own template with the
    // effort undefined, so there is nowhere for the mapped value to go.
    if state.defer_template_on_thinking && enable_thinking {
        if tools.is_some_and(|t| !t.is_empty()) {
            return Err(PromptRenderError::Failed(
                "engine-side templating is active for thinking requests; tools cannot be forwarded"
                    .into(),
            ));
        }
        return Ok(render_prompt_legacy(messages));
    }
    render_or_fallback(
        state.chat_env.as_deref(),
        messages,
        &state.bos_token,
        &state.eos_token,
        enable_thinking,
        reasoning_effort,
        tools,
    )
}

/// Decide the response message + finish_reason from accumulated text.
/// `tool_choice == Some("none")` skips parsing (always "stop").
fn build_choice(
    buf: String,
    tool_choice: &Option<serde_json::Value>,
    tools_present: bool,
    enable_thinking: bool,
) -> (ChatChoiceMessage, &'static str) {
    let parse_enabled =
        tools_present && tool_choice.as_ref().and_then(|v| v.as_str()) != Some("none");
    if parse_enabled {
        if let Some(calls) = parse_tool_calls(tool_call_scan_text(&buf, enable_thinking)) {
            return (
                ChatChoiceMessage {
                    role: "assistant",
                    content: None,
                    tool_calls: Some(calls),
                },
                "tool_calls",
            );
        }
    }
    (
        ChatChoiceMessage {
            role: "assistant",
            content: Some(buf),
            tool_calls: None,
        },
        "stop",
    )
}

/// Closing delimiter every hybrid-reasoning template this stack serves uses
/// (GLM-4.5+, Qwen3, R1-distill). Deliberately NOT a dialect registry: only
/// one delimiter is in use today. Add a second the day a second one ships on
/// real hardware, not before.
const THINK_CLOSE: &str = "</think>";

/// Restrict tool-call scanning to text after the model's private scratchpad,
/// so a `<tool_call>` block drafted while reasoning isn't executed as a real
/// call. Splits on the FIRST `</think>` — the answer itself may go on to
/// mention the literal string again, and that must not re-open the scratchpad.
///
/// Armed by `enable_thinking` from the REQUEST, not by sniffing `buf`:
/// tool-call markers are self-announcing, a reasoning delimiter is not, so
/// arming is decided by the REQUEST. Sniffing the first `</think>` regardless
/// of arming state is wrong in both directions — thinking-off has no
/// scratchpad, so a real call whose own arguments happen to contain the
/// literal `</think>` gets truncated and silently dropped (a regression this
/// arming closes); thinking-on still cannot be fully unambiguous since the
/// scratchpad and the delimiter both come from the model, but that is the
/// pre-existing, narrower risk this function was written to bound.
///
/// A request truncated mid-reasoning has no `</think>` at all, so the whole
/// buffer — scratchpad included — is still scanned and a drafted call can
/// still fire. Closing that gap needs a request-armed reasoning/answer split,
/// deferred to a follow-up along with the wiring that would use it.
fn tool_call_scan_text(buf: &str, enable_thinking: bool) -> &str {
    if !enable_thinking {
        return buf;
    }
    buf.find(THINK_CLOSE)
        .map_or(buf, |idx| &buf[idx + THINK_CLOSE.len()..])
}

/// Parse model tool-call output into structured calls; None when none found.
/// Shape-based + engine-agnostic; never panics (each block parsed independently,
/// malformed blocks skipped).
pub fn parse_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    if text.contains("<tool_call>") {
        // Qwen/Hermes: each <tool_call>…</tool_call> block.
        let mut rest = text;
        while let Some(start) = rest.find("<tool_call>") {
            let after = &rest[start + "<tool_call>".len()..];
            let inner = after.trim_start();
            // JSON Hermes: scan a brace-balanced (string/escape-aware) object so a
            // literal "</tool_call>" inside an argument value can't truncate the
            // block. Fall back to the close-tag delimiter for the XML dialect.
            if inner.starts_with('{') || inner.starts_with('[') {
                let off = after.len() - inner.len();
                if let Some(len) = balanced_json_end(&after[off..]) {
                    if let Ok(v) =
                        serde_json::from_str::<serde_json::Value>(after[off..off + len].trim())
                    {
                        if let Some(c) = call_from_value(&v) {
                            calls.push(c);
                        }
                    }
                    let tail = &after[off + len..];
                    rest = match tail.find("</tool_call>") {
                        Some(e) => &tail[e + "</tool_call>".len()..],
                        None => tail,
                    };
                    continue;
                }
            }
            let Some(end) = after.find("</tool_call>") else {
                break;
            };
            // Two non-JSON dialects share this delimiter. Try the Qwen3/MoE
            // `<function=…><parameter=…>` shape first, then GLM's bare-name +
            // `<arg_key>`/`<arg_value>` shape. Order is irrelevant to
            // correctness — each requires a marker the other never emits — but
            // keeping the older one first leaves its behaviour untouched.
            if let Some(c) = call_from_xml_function(&after[..end])
                .or_else(|| call_from_glm_arg_kv(&after[..end]))
            {
                calls.push(c);
            }
            rest = &after[end + "</tool_call>".len()..];
        }
    } else {
        // Llama-3.1: <|python_tag|> is an explicit tool-call marker. A BARE
        // whole-output JSON (no marker) is only treated as a call if it carries
        // `arguments`/`parameters` — a real Llama call always does (`parameters`,
        // even `{}`), so a plain JSON DATA answer with a stray `name` key (e.g.
        // {"name":"Alice","age":30}) isn't misread as a tool call.
        let trimmed = text.trim();
        let explicit = trimmed.starts_with("<|python_tag|>");
        let candidate = trimmed
            .strip_prefix("<|python_tag|>")
            .map(str::trim)
            .unwrap_or(trimmed);
        let looks_json = candidate.starts_with('{') || candidate.starts_with('[');
        if explicit || looks_json {
            let tool_shaped = |v: &serde_json::Value| {
                explicit || v.get("arguments").is_some() || v.get("parameters").is_some()
            };
            if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(candidate) {
                for v in &arr {
                    if tool_shaped(v) {
                        if let Some(c) = call_from_value(v) {
                            calls.push(c);
                        }
                    }
                }
            } else if let Ok(v) = serde_json::from_str::<serde_json::Value>(candidate) {
                if tool_shaped(&v) {
                    if let Some(c) = call_from_value(&v) {
                        calls.push(c);
                    }
                }
            }
        }
    }
    (!calls.is_empty()).then_some(calls)
}

/// Byte length of the balanced JSON value starting at `s[0]` (`{` or `[`),
/// string-/escape-aware so brackets inside string values don't miscount.
/// `None` if `s` doesn't start with a bracket or never balances.
fn balanced_json_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let (open, close) = match bytes.first()? {
        b'{' => (b'{', b'}'),
        b'[' => (b'[', b']'),
        _ => return None,
    };
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
        } else if b == b'"' {
            in_str = true;
        } else if b == open {
            depth += 1;
        } else if b == close {
            depth -= 1;
            if depth == 0 {
                return Some(i + 1);
            }
        }
    }
    None
}

/// One `ToolCall` from a parsed JSON value. `name` required (else None → skip).
/// `arguments` (preferred) or `parameters` (Llama alias); default "{}"; an
/// already-string arguments value passes through.
fn call_from_value(v: &serde_json::Value) -> Option<ToolCall> {
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = match v.get("arguments").or_else(|| v.get("parameters")) {
        None => "{}".to_string(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).ok()?,
    };
    Some(ToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        r#type: "function".to_string(),
        function: FunctionCall { name, arguments },
    })
}

/// Parse the Qwen3 `<function=NAME>…<parameter=K>V</parameter>…</function>` XML
/// tool-call dialect emitted inside a `<tool_call>` block by some Qwen3/MoE
/// templates (instead of the JSON Hermes form). Parameter values are kept as
/// strings (the model emits text); arguments is serialised to a JSON object.
fn call_from_xml_function(block: &str) -> Option<ToolCall> {
    let fstart = block.find("<function=")?;
    let after = &block[fstart + "<function=".len()..];
    let nend = after.find('>')?;
    let name = after[..nend].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut args = serde_json::Map::new();
    let mut rest = &after[nend + 1..];
    while let Some(ps) = rest.find("<parameter=") {
        let pa = &rest[ps + "<parameter=".len()..];
        let Some(ke) = pa.find('>') else { break };
        let key = pa[..ke].trim().to_string();
        let val_rest = &pa[ke + 1..];
        let Some(ve) = val_rest.find("</parameter>") else {
            break;
        };
        if !key.is_empty() {
            args.insert(
                key,
                serde_json::Value::String(val_rest[..ve].trim().to_string()),
            );
        }
        rest = &val_rest[ve + "</parameter>".len()..];
    }
    Some(ToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        r#type: "function".to_string(),
        function: FunctionCall {
            name,
            arguments: serde_json::to_string(&serde_json::Value::Object(args)).ok()?,
        },
    })
}

/// Shape test for a bare tool-call name with no arguments: `^[A-Za-z_][A-Za-z0-9_.-]*$`.
/// Prose disqualifies itself — it contains spaces or punctuation this pattern
/// excludes — so this is what separates GLM's zero-argument call shape
/// (`<tool_call>get_current_time\n</tool_call>`) from arbitrary text that
/// happens to land inside a `<tool_call>` block without `<arg_key>`.
fn is_bare_function_name(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

/// GLM-4.5+/GLM-5 `<tool_call>` dialect: the function name sits bare right
/// after the open tag, followed by `<arg_key>`/`<arg_value>` pairs.
///
/// ```text
/// <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>
/// ```
///
/// Neither of the other dialects matches this: [`call_from_xml_function`]
/// requires `<function=`, and the JSON arm requires the block to start with a
/// brace. Without this the block parses as nothing and the whole generation
/// ships as ordinary content — a tool call the model genuinely made, silently
/// dropped. Measured on a GLM-5 fleet: the model's own `chat_template.jinja`
/// contains `arg_key`/`arg_value` and neither `<function=` nor `<|python_tag|>`,
/// so tool calling could never work for that model.
///
/// Values follow the template's own encoding — `v | tojson if v is not string
/// else v` — so a value that parses as JSON is kept as that JSON type and
/// anything else stays a string. Parsing a bare `Paris` as JSON fails, which is
/// correct: it is a string.
///
/// A zero-argument call has no `<arg_key>` at all — just the bare name, e.g.
/// `<tool_call>get_current_time\n</tool_call>` — and is emitted with
/// `arguments: "{}"`. See [`is_bare_function_name`] for how that's told apart
/// from prose landing in a block with no `<arg_key>`.
fn call_from_glm_arg_kv(block: &str) -> Option<ToolCall> {
    // No `<arg_key>` at all is either a different dialect, or GLM's
    // zero-argument shape: `<tool_call>name\n</tool_call>`, a bare function
    // name and nothing else. Distinguish by shape rather than bailing
    // outright — arbitrary prose must still be rejected, but prose always
    // contains whitespace/punctuation a bare identifier doesn't, so the shape
    // test alone tells them apart without inventing a call from free text.
    let Some(first_key) = block.find("<arg_key>") else {
        let name = block.trim();
        return is_bare_function_name(name).then(|| ToolCall {
            id: format!("call_{}", Uuid::new_v4().simple()),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        });
    };
    let name = block[..first_key].trim();
    if name.is_empty() || name.contains('<') {
        return None;
    }
    let mut args = serde_json::Map::new();
    let mut rest = &block[first_key..];
    while let Some(ks) = rest.find("<arg_key>") {
        let ka = &rest[ks + "<arg_key>".len()..];
        let Some(ke) = ka.find("</arg_key>") else {
            break;
        };
        let key = ka[..ke].trim().to_string();
        let after_key = &ka[ke + "</arg_key>".len()..];
        let Some(vs) = after_key.find("<arg_value>") else {
            break;
        };
        let va = &after_key[vs + "<arg_value>".len()..];
        let Some(ve) = va.find("</arg_value>") else {
            break;
        };
        let raw = va[..ve].trim();
        if !key.is_empty() {
            let value = serde_json::from_str::<serde_json::Value>(raw)
                .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()));
            args.insert(key, value);
        }
        rest = &va[ve + "</arg_value>".len()..];
    }
    Some(ToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        r#type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: serde_json::to_string(&serde_json::Value::Object(args)).ok()?,
        },
    })
}

/// `AppState`-free chat-prompt renderer for in-process callers. Parses the
/// chat template once in [`new`](Self::new); reuse across requests.
#[derive(Clone)]
pub struct ChatPromptRenderer {
    env: Option<Arc<minijinja::Environment<'static>>>,
    bos_token: String,
    eos_token: String,
}

impl ChatPromptRenderer {
    /// Compile the chat template once. An unparseable template is treated as
    /// absent so [`render`](Self::render) falls back to the legacy formatter.
    pub fn new(cfg: &ChatTemplateConfig) -> Self {
        let env = cfg
            .template
            .as_deref()
            .and_then(|src| match build_chat_env(src) {
                Ok(env) => {
                    report_chat_template_smoke(&env);
                    Some(Arc::new(env))
                }
                Err(e) => {
                    warn!(error = %e, "chat_template failed to parse; using legacy formatter");
                    None
                }
            });
        Self {
            env,
            bos_token: cfg.bos_token.clone().unwrap_or_default(),
            eos_token: cfg.eos_token.clone().unwrap_or_default(),
        }
    }

    /// Render `messages` with optional `tools`. `enable_thinking = true` — use
    /// [`render_with_opts`](Self::render_with_opts) to control the template's
    /// thinking branch.
    pub fn render_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<String, PromptRenderError> {
        self.render_with_opts(messages, tools, true)
    }

    /// Render with an explicit `enable_thinking` for hybrid-reasoning
    /// templates (GLM/Qwen3 convention): the value lands in the template's
    /// `enable_thinking` variable, same as the server's `render_prompt` path.
    pub fn render_with_opts(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        enable_thinking: bool,
    ) -> Result<String, PromptRenderError> {
        self.render_with_effort(messages, tools, enable_thinking, None)
    }

    /// As [`render_with_opts`](Self::render_with_opts) but also setting the
    /// template's `reasoning_effort`.
    ///
    /// GLM-5's template resolves an undefined `reasoning_effort` to `'max'` and
    /// injects a `Reasoning Effort: Max` system line, so leaving it unset is not
    /// neutral — it is the most expensive setting, chosen by omission.
    pub fn render_with_effort(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        enable_thinking: bool,
        reasoning_effort: Option<&str>,
    ) -> Result<String, PromptRenderError> {
        render_or_fallback(
            self.env.as_deref(),
            messages,
            &self.bos_token,
            &self.eos_token,
            enable_thinking,
            reasoning_effort,
            tools,
        )
    }

    /// Render `messages` with no tools. Falls back to
    /// [`render_prompt_legacy`] when no template is set or rendering fails;
    /// still errors if the template rejected the messages outright.
    pub fn render(&self, messages: &[ChatMessage]) -> Result<String, PromptRenderError> {
        self.render_with_tools(messages, None)
    }
}

/// One-shot chat-prompt render for standalone / embedded callers that hold a
/// [`ChatTemplateConfig`] directly (e.g. an embedded shard backend) and render
/// outside the router's `AppState`. Builds a [`ChatPromptRenderer`] per call;
/// callers that render repeatedly should construct one [`ChatPromptRenderer`]
/// and reuse it. Falls back to [`render_prompt_legacy`] on parse/render failure.
pub fn render_chat_prompt(
    cfg: &ChatTemplateConfig,
    messages: &[ChatMessage],
) -> Result<String, PromptRenderError> {
    ChatPromptRenderer::new(cfg).render(messages)
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let task_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let enable_thinking = req.effective_enable_thinking();
    // effective_reasoning_effort() hardcodes GLM's high/max vocabulary and is
    // applied to every model this server serves, not just GLM-5. Latent
    // today because no other served template (Qwen3, minimax-m2, r1-distill)
    // reads `reasoning_effort`, but a future template that honours a
    // low/medium/high scale would see e.g. a client's "low" silently
    // escalated to "high". Template-gating this is a follow-up, not this fix.
    let prompt = match render_prompt(
        &state,
        &req.messages,
        enable_thinking,
        Some(req.effective_reasoning_effort()),
        req.tools.as_deref(),
    ) {
        Ok(p) => p,
        // A tools request that cannot be templated must not return a 200: the
        // model would answer without ever seeing the tools, and the caller
        // cannot tell that from "the model chose not to call one".
        Err(e @ PromptRenderError::Rejected(_)) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
        Err(e @ PromptRenderError::Failed(_)) => {
            tracing::error!(error = %e, "chat_template render failed for a request that needs it");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };
    // Degenerate input (no messages, or a render that collapses to nothing)
    // is a client error. Reject here with 400 rather than admitting an empty
    // prompt to the engine, which would generate nothing and return a 200
    // with empty content — indistinguishable from a real failure.
    if prompt.trim().is_empty() {
        count_rejected("empty_prompt");
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "no prompt content: `messages` is empty or rendered to an empty prompt"
            })),
        )
            .into_response();
    }
    if prompt.len() > state.max_prompt_bytes {
        count_rejected("prompt_too_large");
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(serde_json::json!({
                "error": format!(
                    "prompt is {} bytes; max allowed is {} (max_prompt_bytes)",
                    prompt.len(),
                    state.max_prompt_bytes,
                )
            })),
        )
            .into_response();
    }
    let task = GenerationTask {
        task_id: task_id.clone(),
        prompt,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        logprobs: req.logprobs_count(),
        sampling: req.sampling_params(),
        enable_thinking,
        trust_remote_code: false,
    };

    // Acquire a request slot before touching the engine. Contended streams
    // no longer block a tokio worker (#122: they park on a failed `try_lock`,
    // and submits go out via `spawn_blocking`), so this gate is not what
    // keeps the runtime alive — it bounds the two resources that are still
    // finite: the engine's own queue/slot depth, and the blocking pool the
    // submits land on. Backpressure is 503; clients should retry with
    // backoff.
    let permit = match state.permits.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            count_rejected("capacity");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "engine at capacity; retry after current requests complete"
                })),
            )
                .into_response();
        }
    };

    // Past the gate = an admitted request. Bump the cumulative counter and
    // hold an in-flight guard for the request's lifetime (the guard's Drop
    // decrements the gauge on any exit path).
    state.stats.requests_total.fetch_add(1, Ordering::Relaxed);
    let inflight = InFlightGuard::new(state.stats.clone());

    if req.stream {
        let include_usage = req
            .stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false);
        let tool_choice = req.tool_choice.clone();
        let tools_present = req.tools.as_ref().is_some_and(|t| !t.is_empty());
        return stream_completion(
            state,
            req.model,
            task,
            permit,
            inflight,
            include_usage,
            tool_choice,
            tools_present,
        )
        .await
        .into_response();
    }

    // Non-streaming: collect full output. Hold the permit + in-flight guard
    // until the task completes; drop frees the slot.
    let _permit = permit;
    let _inflight = inflight;
    let mut chunk_stream = match state.runner.generate_async(task.clone()).await {
        Ok(s) => s,
        Err(err) => return engine_error_response(err),
    };
    let mut buf = String::new();
    let mut completion_tokens: u32 = 0;
    let mut prompt_tokens: u32 = 0;
    // OpenAI `finish_reason`: `length` when the engine hit max_tokens, else
    // `stop`. Engines that don't distinguish leave it None → "stop".
    let mut finish_reason: &'static str = "stop";
    let mut logprobs_content: Vec<cascadia_types::TokenLogprobs> = Vec::new();
    while let Some(chunk) = chunk_stream.next().await {
        // A failed task carries `error` on its final chunk. Without this
        // branch the empty text below would build a normal 200 with empty
        // content — indistinguishable from "the model said nothing". Fail
        // loud with a 5xx instead (e.g. a sharded chain poisoned by a
        // handshake/manifest mismatch).
        if let Some(reason) = &chunk.error {
            warn!(task = %task_id, reason = %reason, "engine failed task; returning 503");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": reason })),
            )
                .into_response();
        }
        // Count tokens for EVERY chunk including the final one — engines
        // like ov-genai deliver their whole output on a single final
        // chunk, and spec-decode carries its last round there too; gating
        // on `!is_final` dropped them (tokens_total stuck at 0 for
        // ov-genai). chunk_token_count contributes 0 for the empty final
        // markers other engines emit, so there's no phantom over-count.
        buf.push_str(&chunk.text);
        completion_tokens += chunk_token_count(&chunk);
        if let Some(lp) = &chunk.logprobs {
            logprobs_content.push(lp.clone());
        }
        if chunk.is_final {
            prompt_tokens = chunk.prompt_tokens.unwrap_or(0);
            if let Some(fr) = chunk.finish_reason {
                finish_reason = fr.as_openai_str();
            }
        }
    }
    state
        .stats
        .tokens_total
        .fetch_add(completion_tokens as u64, Ordering::Relaxed);

    // Honor `stop` sequences by trimming the assembled text at the earliest
    // match. The engines don't early-stop on strings yet (an engine-side
    // follow-up would also save compute), so this enforces the OpenAI output
    // contract at the API boundary and forces finish_reason = "stop".
    let stops = req.sampling_params().stop;
    if !stops.is_empty() {
        if let Some(cut) = stops
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| buf.find(s.as_str()))
            .min()
        {
            buf.truncate(cut);
            finish_reason = "stop";
        }
    }

    let (message, choice_finish) = build_choice(
        buf,
        &req.tool_choice,
        req.tools.as_ref().is_some_and(|t| !t.is_empty()),
        enable_thinking,
    );
    // A parsed tool call overrides the streamed length/stop detection;
    // otherwise keep `finish_reason` (e.g. "length" when max_tokens hit).
    if choice_finish == "tool_calls" {
        finish_reason = "tool_calls";
    }
    Json(ChatCompletionResponse {
        id: task_id,
        object: "chat.completion",
        created: now_unix(),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message,
            finish_reason,
            logprobs: if logprobs_content.is_empty() {
                None
            } else {
                Some(ChatLogprobs {
                    content: logprobs_content,
                })
            },
        }],
        usage: Usage {
            // From the engine's final chunk; 0 when the engine can't tell
            // (e.g. an engine that reports no prompt-token count).
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response()
}

/// OpenAI-compatible legacy `/v1/completions`. A raw-`prompt` sibling of
/// `/v1/chat/completions` — no chat-template render, returns `text` instead
/// of a `message`. Reuses the same sampling/finish_reason/usage path.
async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> axum::response::Response {
    // Multi-prompt batching needs cross-task batching infra the engines don't
    // have yet (one task per step). Reject the array form with a clean 400.
    let prompt = match &req.prompt {
        PromptSpec::Single(s) => s.clone(),
        PromptSpec::Multiple(_) => {
            count_rejected("multi_prompt");
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "multi-prompt batching not yet supported; submit one prompt per request"
                })),
            )
                .into_response();
        }
    };
    let task_id = format!("cmpl-{}", Uuid::new_v4().simple());
    if prompt.trim().is_empty() {
        count_rejected("empty_prompt");
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "no prompt content: `prompt` is empty" })),
        )
            .into_response();
    }
    if prompt.len() > state.max_prompt_bytes {
        count_rejected("prompt_too_large");
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(serde_json::json!({
                "error": format!(
                    "prompt is {} bytes; max allowed is {} (max_prompt_bytes)",
                    prompt.len(),
                    state.max_prompt_bytes,
                )
            })),
        )
            .into_response();
    }
    let task = GenerationTask {
        task_id: task_id.clone(),
        // Raw prompt — the legacy endpoint does NOT apply a chat template.
        prompt: prompt.clone(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        logprobs: req.logprobs_count(),
        sampling: req.sampling_params(),
        enable_thinking: false,
        trust_remote_code: false,
    };

    let permit = match state.permits.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            count_rejected("capacity");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "engine at capacity; retry after current requests complete"
                })),
            )
                .into_response();
        }
    };
    state.stats.requests_total.fetch_add(1, Ordering::Relaxed);
    let inflight = InFlightGuard::new(state.stats.clone());

    if req.stream {
        let include_usage = req
            .stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false);
        let echo_prefix = if req.echo { Some(prompt) } else { None };
        return stream_text_completion(
            state,
            req.model,
            task,
            echo_prefix,
            permit,
            inflight,
            include_usage,
        )
        .await
        .into_response();
    }

    let _permit = permit;
    let _inflight = inflight;
    let mut chunk_stream = match state.runner.generate_async(task).await {
        Ok(s) => s,
        Err(err) => return engine_error_response(err),
    };
    let mut buf = String::new();
    let mut completion_tokens: u32 = 0;
    let mut prompt_tokens: u32 = 0;
    let mut finish_reason: &'static str = "stop";
    let mut logprobs_content: Vec<cascadia_types::TokenLogprobs> = Vec::new();
    while let Some(chunk) = chunk_stream.next().await {
        if let Some(reason) = &chunk.error {
            warn!(task = %task_id, reason = %reason, "engine failed task; returning 503");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": reason })),
            )
                .into_response();
        }
        buf.push_str(&chunk.text);
        completion_tokens += chunk_token_count(&chunk);
        if let Some(lp) = &chunk.logprobs {
            logprobs_content.push(lp.clone());
        }
        if chunk.is_final {
            prompt_tokens = chunk.prompt_tokens.unwrap_or(0);
            if let Some(fr) = chunk.finish_reason {
                finish_reason = fr.as_openai_str();
            }
        }
    }
    state
        .stats
        .tokens_total
        .fetch_add(completion_tokens as u64, Ordering::Relaxed);

    // Stop-sequence truncation (same as the chat path).
    let stops = req.sampling_params().stop;
    if !stops.is_empty() {
        if let Some(cut) = stops
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| buf.find(s.as_str()))
            .min()
        {
            buf.truncate(cut);
            finish_reason = "stop";
        }
    }

    // `echo`: prepend the (raw) prompt to the returned text.
    let text = if req.echo {
        format!("{prompt}{buf}")
    } else {
        buf
    };

    Json(CompletionResponse {
        id: task_id,
        object: "text_completion",
        created: now_unix(),
        model: req.model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason,
            logprobs: if logprobs_content.is_empty() {
                None
            } else {
                Some(ChatLogprobs {
                    content: logprobs_content,
                })
            },
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response()
}

/// SSE variant of `/v1/completions`: emits `text_completion` chunks with
/// `choices[].text` deltas. `echo_prefix`, when set, is prepended to the very
/// first content frame.
async fn stream_text_completion(
    state: AppState,
    model: String,
    task: GenerationTask,
    echo_prefix: Option<String>,
    permit: tokio::sync::OwnedSemaphorePermit,
    inflight: InFlightGuard,
    include_usage: bool,
) -> axum::response::Response {
    let task_id = task.task_id.clone();
    let chunk_stream = match state.runner.generate_async(task).await {
        Ok(s) => s,
        Err(err) => {
            warn!(error = %err, "completions-stream: generate failed");
            return engine_error_response(err);
        }
    };
    let stats = state.stats.clone();
    let usage_completion = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let usage_prompt = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let final_completion = usage_completion.clone();
    let final_prompt = usage_prompt.clone();
    let usage_model = model.clone();
    let usage_task_id = task_id.clone();
    // First-frame flag for `echo` (prepend the prompt once).
    let echo_pending = Arc::new(std::sync::Mutex::new(echo_prefix));
    let permit_carrier = StreamWithPermit {
        inner: chunk_stream,
        _permit: permit,
        _inflight: inflight,
    };
    let body_stream = permit_carrier
        .then(move |chunk| {
            let model = model.clone();
            let task_id = task_id.clone();
            let stats = stats.clone();
            let usage_completion = usage_completion.clone();
            let usage_prompt = usage_prompt.clone();
            let echo_pending = echo_pending.clone();
            async move {
                if chunk.error.is_none() {
                    let n = chunk_token_count(&chunk);
                    stats.tokens_total.fetch_add(n as u64, Ordering::Relaxed);
                    usage_completion.fetch_add(n, Ordering::Relaxed);
                    if chunk.is_final {
                        usage_prompt.store(chunk.prompt_tokens.unwrap_or(0), Ordering::Relaxed);
                    }
                }
                if let Some(reason) = &chunk.error {
                    warn!(task = %task_id, reason = %reason, "engine failed task mid-stream; emitting SSE error");
                    let payload = serde_json::json!({
                        "id": task_id,
                        "object": "error",
                        "error": { "message": reason, "type": "engine_error" },
                    });
                    tokio::task::yield_now().await;
                    return Ok::<Bytes, std::convert::Infallible>(Bytes::from(format!("data: {payload}\n\n")));
                }
                // Prepend the echo prefix to the first content frame, once.
                let mut text = chunk.text.clone();
                if let Some(prefix) = echo_pending.lock().unwrap().take() {
                    text = format!("{prefix}{text}");
                }
                let payload = serde_json::json!({
                    "id": task_id,
                    "object": "text_completion",
                    "created": now_unix(),
                    "model": model,
                    "n_tokens": chunk.n_tokens.unwrap_or(1),
                    "choices": [{
                        "index": 0,
                        "text": text,
                        "finish_reason": if chunk.is_final {
                            Some(chunk.finish_reason.map(|f| f.as_openai_str()).unwrap_or("stop"))
                        } else {
                            None
                        },
                    }],
                });
                let line = format!("data: {payload}\n\n");
                tokio::task::yield_now().await;
                Ok::<Bytes, std::convert::Infallible>(Bytes::from(line))
            }
        })
        .chain(stream::once(async move {
            let mut out = String::new();
            if include_usage {
                let prompt = final_prompt.load(Ordering::Relaxed);
                let completion = final_completion.load(Ordering::Relaxed);
                let usage = serde_json::json!({
                    "id": usage_task_id,
                    "object": "text_completion",
                    "created": now_unix(),
                    "model": usage_model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "total_tokens": prompt + completion,
                    },
                });
                out.push_str(&format!("data: {usage}\n\n"));
            }
            out.push_str("data: [DONE]\n\n");
            Ok::<Bytes, std::convert::Infallible>(Bytes::from(out))
        }));

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(body_stream))
        .unwrap()
        .into_response()
}

// Streaming carries the usage handles (inflight/include_usage) and the tool
// gate (tool_choice/tools_present) alongside the task plumbing.
#[allow(clippy::too_many_arguments)]
async fn stream_completion(
    state: AppState,
    model: String,
    task: GenerationTask,
    permit: tokio::sync::OwnedSemaphorePermit,
    inflight: InFlightGuard,
    include_usage: bool,
    tool_choice: Option<serde_json::Value>,
    tools_present: bool,
) -> axum::response::Response {
    let task_id = task.task_id.clone();
    // Captured before `task` moves into `generate` below — arms the tool-call
    // scan the same way the non-streaming path does (see `tool_call_scan_text`).
    let enable_thinking = task.enable_thinking;
    let _ = SystemTime::now();

    let chunk_stream = match state.runner.generate_async(task).await {
        Ok(s) => s,
        Err(err) => {
            warn!(error = %err, "ov-stream: generate failed");
            return engine_error_response(err);
        }
    };
    // Buffer + emit one tool delta only when tools are present and not disabled.
    let tools_active =
        tools_present && tool_choice.as_ref().and_then(|v| v.as_str()) != Some("none");
    if tools_active {
        let _permit = permit; // hold the slot until generation completes
        let _inflight = inflight; // hold the in-flight gauge until generation completes
        let mut buf = String::new();
        let mut err: Option<String> = None;
        let mut stream = chunk_stream;
        while let Some(chunk) = stream.next().await {
            if let Some(reason) = &chunk.error {
                err = Some(reason.clone());
                break;
            }
            buf.push_str(&chunk.text);
        }
        let frames: Vec<Bytes> = if let Some(reason) = err {
            warn!(task = %task_id, reason = %reason, "engine failed tool task mid-stream; SSE error");
            vec![Bytes::from(format!(
                "data: {}\n\n",
                serde_json::json!({
                    "id": task_id.clone(), "object": "error",
                    "error": { "message": reason, "type": "engine_error" },
                })
            ))]
        } else if let Some(calls) = parse_tool_calls(tool_call_scan_text(&buf, enable_thinking)) {
            let tool_calls: Vec<serde_json::Value> = calls.iter().enumerate().map(|(i, c)| {
                serde_json::json!({ "index": i, "id": c.id.clone(), "type": c.r#type.clone(),
                    "function": { "name": c.function.name.clone(), "arguments": c.function.arguments.clone() } })
            }).collect();
            let delta = serde_json::json!({ "id": task_id.clone(), "object": "chat.completion.chunk",
                "created": now_unix(), "model": model.clone(), "choices": [{ "index": 0,
                "delta": { "role": "assistant", "tool_calls": tool_calls }, "finish_reason": serde_json::Value::Null }] });
            let finish = serde_json::json!({ "id": task_id.clone(), "object": "chat.completion.chunk",
                "created": now_unix(), "model": model.clone(), "choices": [{ "index": 0,
                "delta": {}, "finish_reason": "tool_calls" }] });
            vec![
                Bytes::from(format!("data: {delta}\n\n")),
                Bytes::from(format!("data: {finish}\n\n")),
            ]
        } else {
            let delta = serde_json::json!({ "id": task_id.clone(), "object": "chat.completion.chunk",
                "created": now_unix(), "model": model.clone(), "choices": [{ "index": 0,
                "delta": { "role": "assistant", "content": buf }, "finish_reason": "stop" }] });
            vec![Bytes::from(format!("data: {delta}\n\n"))]
        };
        let body = stream::iter(
            frames
                .into_iter()
                .map(Ok::<Bytes, std::convert::Infallible>),
        )
        .chain(stream::once(async {
            Ok::<Bytes, std::convert::Infallible>(Bytes::from_static(b"data: [DONE]\n\n"))
        }));
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header("x-accel-buffering", "no")
            .body(Body::from_stream(body))
            .unwrap()
            .into_response();
    }
    // Clone the counter handle for per-chunk token accounting in the
    // formatter closure below.
    let stats = state.stats.clone();
    // Per-request token tallies for the OpenAI `stream_options.include_usage`
    // final chunk (the global `stats.tokens_total` is cross-request). The
    // `final_*` handles are read by the trailing usage frame after the body
    // stream drains; the per-chunk closure gets the originals.
    let usage_completion = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let usage_prompt = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let final_completion = usage_completion.clone();
    let final_prompt = usage_prompt.clone();
    let usage_model = model.clone();
    let usage_task_id = task_id.clone();
    // Move the permit + in-flight guard into the stream so they're released
    // only when the body is dropped (client disconnect or final chunk).
    let permit_carrier = StreamWithPermit {
        inner: chunk_stream,
        _permit: permit,
        _inflight: inflight,
    };
    // Format each chunk as a raw SSE frame `data: <json>\n\n` and send
    // it as a one-byte-shy-of-MTU frame via `Body::from_stream`.
    //
    // Why raw bytes instead of `axum::response::sse::Sse`:
    // `Sse::new(stream).keep_alive(KeepAlive::default())` wraps the
    // stream in a KeepAlive forwarder that, per measurement at the
    // raw-socket level, accumulates ~16 events (~3500 B) before the
    // body framer flushes — so an audience watching the chat sees
    // mega-bursts every ~1 s instead of one token at the engine's
    // ~12 tok/s natural cadence. Going straight to `Body::from_stream`
    // makes each chunk its own HTTP/1.1 chunked-encoding frame, which
    // Hyper writes immediately (combined with TCP_NODELAY set on the
    // accepted connection in cascadia-cli's serve loop, this lands the
    // per-token chunk on the wire as a separate ~220 B packet).
    let body_stream = permit_carrier
        .then(move |chunk| {
            let model = model.clone();
            let task_id = task_id.clone();
            let stats = stats.clone();
            let usage_completion = usage_completion.clone();
            let usage_prompt = usage_prompt.clone();
            async move {
                // Count model tokens as they stream so the dashboard's
                // tokens_total advances live (not just at request end).
                // Counts the final chunk too (ov-genai emits its whole
                // output there); chunk_token_count yields 0 for empty
                // markers, so no phantom token.
                if chunk.error.is_none() {
                    let n = chunk_token_count(&chunk);
                    stats.tokens_total.fetch_add(n as u64, Ordering::Relaxed);
                    usage_completion.fetch_add(n, Ordering::Relaxed);
                    if chunk.is_final {
                        usage_prompt
                            .store(chunk.prompt_tokens.unwrap_or(0), Ordering::Relaxed);
                    }
                }
                // A failed task carries `error` on its final chunk. The 200
                // headers are already on the wire by the time the body
                // streams, so unlike the non-streaming path we cannot
                // downgrade to a 5xx here. Surface the failure as an explicit
                // SSE error event instead of a silent empty delta + [DONE].
                if let Some(reason) = &chunk.error {
                    warn!(task = %task_id, reason = %reason, "engine failed task mid-stream; emitting SSE error");
                    let payload = serde_json::json!({
                        "id": task_id,
                        "object": "error",
                        "error": { "message": reason, "type": "engine_error" },
                    });
                    tokio::task::yield_now().await;
                    return Ok::<Bytes, std::convert::Infallible>(Bytes::from(format!("data: {payload}\n\n")));
                }
                // Custom (non-OpenAI) `n_tokens` field: how many model
                // tokens this chunk carries. Spec-decode emits 1..=K+1
                // tokens per chunk; downstream tok/s would be wrong if
                // it counted chunks. Standard clients ignore unknown
                // fields; our orchestrator reads it.
                let payload = serde_json::json!({
                    "id": task_id,
                    "object": "chat.completion.chunk",
                    "created": now_unix(),
                    "model": model,
                    "n_tokens": chunk.n_tokens.unwrap_or(1),
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": chunk.text,
                        },
                        "finish_reason": if chunk.is_final {
                            Some(chunk.finish_reason.map(|f| f.as_openai_str()).unwrap_or("stop"))
                        } else {
                            None
                        },
                    }],
                });
                let line = format!("data: {payload}\n\n");
                // Yield between chunks so the body-sink task can drain
                // this frame to TCP. cascadia_runner::ChunkStream::poll_next
                // calls engine.step() inside a parking_lot::Mutex, which
                // blocks the tokio worker for ~80 ms per chunk without
                // yielding. Without an explicit yield here, tokio's
                // cooperative scheduling lets this task hog its worker
                // for ~1 s at a time, and the per-chunk body writes
                // accumulate in Hyper's buffer and flush as one ~3500 B
                // burst — visible in the demo as 16 tokens at once
                // every ~1 s. Yielding here lets the body sink run
                // between chunks so each frame goes to TCP individually.
                tokio::task::yield_now().await;
                Ok::<Bytes, std::convert::Infallible>(Bytes::from(line))
            }
        })
        .chain(stream::once(async move {
            // OpenAI: when stream_options.include_usage is set, emit one final
            // chunk with empty `choices` carrying `usage`, just before [DONE].
            // Combined into a single HTTP frame with [DONE] (SSE events are
            // delimited by \n\n regardless of HTTP chunk boundaries).
            let mut out = String::new();
            if include_usage {
                let prompt = final_prompt.load(Ordering::Relaxed);
                let completion = final_completion.load(Ordering::Relaxed);
                let usage = serde_json::json!({
                    "id": usage_task_id,
                    "object": "chat.completion.chunk",
                    "created": now_unix(),
                    "model": usage_model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "total_tokens": prompt + completion,
                    },
                });
                out.push_str(&format!("data: {usage}\n\n"));
            }
            out.push_str("data: [DONE]\n\n");
            Ok::<Bytes, std::convert::Infallible>(Bytes::from(out))
        }));

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(body_stream))
        .unwrap()
        .into_response()
}

/// Wraps a [`cascadia_runner::ChunkStream`] so it carries its concurrency
/// permit and the permit is released when the stream is dropped (which
/// happens on client disconnect — axum drops the SSE response).
struct StreamWithPermit {
    inner: cascadia_runner::ChunkStream,
    _permit: tokio::sync::OwnedSemaphorePermit,
    /// Decrements `requests_in_flight` when the stream body is dropped.
    _inflight: InFlightGuard,
}

impl Stream for StreamWithPermit {
    type Item = cascadia_types::Chunk;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

fn engine_error_response(err: cascadia_engine::EngineError) -> axum::response::Response {
    use cascadia_engine::EngineError;
    // An engine-queue-full 503 is the same "at capacity" experience as the
    // permit-gate 503 (engines cap pending tasks BELOW the default permit
    // count, so this path is reachable at default config) — count it under
    // the same rejection reason or capacity pressure hides from the metric.
    //
    // A prompt the engine cannot window is a rejection too, and the one an
    // operator most needs to see: at 8 packed slots on a 1024-context export
    // the per-slot region is 127 tokens, so it is the common path, not an
    // edge case, and the fix is a sizing change (fewer slots, wider context).
    // Its own reason rather than `prompt_too_large` — that is the API's byte
    // limit on the request body, a different knob with a different owner.
    // Exhaustive on purpose — no `_` arm. Every variant reaching here is a
    // request rejected before generation started, which is exactly what this
    // metric promises to count, and a catch-all is how PromptTooLong stayed
    // invisible until someone went looking. A new EngineError variant should
    // fail to compile until it is classified, not silently vanish.
    //
    // Reasons name the knob, not the status: `engine_unavailable` means bring
    // the stage or its peer up, `engine_error` means read that node's logs,
    // `invalid_request` means the caller sent something this worker can't
    // serve. Without these a node failing 100% of requests looked identical
    // to a healthy idle one on every metric here.
    count_rejected(match &err {
        EngineError::QueueFull { .. } => "capacity",
        EngineError::PromptTooLong(_) => "prompt_over_window",
        EngineError::NotLoaded | EngineError::NotConnected => "engine_unavailable",
        EngineError::Backend(_)
        | EngineError::Io(_)
        | EngineError::Task { .. }
        | EngineError::BatchAborted(_) => "engine_error",
        EngineError::InvalidConfig(_)
        | EngineError::PeerRejected(_)
        | EngineError::ShardRejected(_)
        | EngineError::ModelNotFound(_) => "invalid_request",
    });
    let status = match &err {
        EngineError::QueueFull { .. } => StatusCode::SERVICE_UNAVAILABLE,
        EngineError::NotLoaded | EngineError::NotConnected => StatusCode::SERVICE_UNAVAILABLE,
        // Permanent for this request: retrying sends the identical prompt into
        // the identical window. 413 is what every SDK treats as non-retryable.
        EngineError::PromptTooLong(_) => StatusCode::PAYLOAD_TOO_LARGE,
        EngineError::InvalidConfig(_)
        | EngineError::PeerRejected(_)
        | EngineError::ShardRejected(_) => StatusCode::BAD_REQUEST,
        EngineError::ModelNotFound(_) => StatusCode::NOT_FOUND,
        EngineError::Backend(_) | EngineError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
        // A task-attributed step failure: the engine abandoned a task; the
        // underlying cause is a backend/transport failure.
        EngineError::Task { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        // A pipeline stage abandoned the in-flight batch. Server-side and
        // this node's to explain, like any other engine failure.
        EngineError::BatchAborted(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(serde_json::json!({"error": err.to_string()}))).into_response()
}

async fn cancel(State(state): State<AppState>, Path(task_id): Path<String>) -> impl IntoResponse {
    state.runner.cancel(&task_id);
    info!(task = %task_id, "cancelled");
    (
        StatusCode::OK,
        Json(serde_json::json!({"cancelled": task_id})),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use cascadia_engine_mock::MockBuilder;
    use cascadia_types::{Chunk, PeerLayout, ShardSpec};
    use serde_json::Value;
    use tower::ServiceExt;

    #[test]
    fn sanitizes_numeric_dot_index_for_minijinja() {
        // `.N` subscript -> `[N]`; float literals (digit before `.`) untouched.
        assert_eq!(
            sanitize_numeric_dot_index("m.content.0.type"),
            "m.content[0].type"
        );
        assert_eq!(
            sanitize_numeric_dot_index("a.content.12 is mapping"),
            "a.content[12] is mapping"
        );
        assert_eq!(
            sanitize_numeric_dot_index("temp = 1.5 or 0.0"),
            "temp = 1.5 or 0.0"
        );
        assert_eq!(sanitize_numeric_dot_index("plain text"), "plain text");
        // A template using the Python `.N` subscript (as GLM-5's tool handling
        // does) must now parse instead of falling back to the legacy formatter.
        let src = "{%- if m.content.0.type == 'x' -%}{{ m.content.0.output }}{%- endif -%}";
        assert!(
            build_chat_env(src).is_ok(),
            "`.N` subscript template should parse after dot-index sanitization"
        );
        // minijinja rejects the raw form, proving the sanitizer is what fixes it.
        let mut raw = minijinja::Environment::new();
        assert!(raw.add_template_owned("t", src.to_owned()).is_err());
    }

    // Guards the contract the ov-genai usage fix (#55) depends on: when an
    // engine sets `n_tokens` on a single text-bearing final chunk, that count
    // is authoritative (not the "1 per chunk" fallback), so `usage` reflects
    // the real generated-token count rather than 0/1.
    /// A prompt that cannot fit the engine's per-request window is a permanent,
    /// deterministic rejection of THIS request — not backpressure. It must not
    /// map to 503: the official SDKs retry 5xx with backoff, so a request that
    /// can never succeed would be sent three times before surfacing, and the
    /// operator could not tell it apart from genuine "engine at capacity".
    #[test]
    fn prompt_too_long_maps_to_413_not_503() {
        let r = engine_error_response(cascadia_engine::EngineError::PromptTooLong(
            "prompt is 207 tokens but this worker's per-slot KV region holds 127".into(),
        ));
        assert_eq!(r.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    /// The window rejection is a rejection, and it must be countable on its
    /// own: at 8 packed slots on a 1024-context export the per-slot region is
    /// 127 tokens, so an operator watching a worker shed traffic needs to see
    /// "resize the slots" separately from `capacity` (add workers) and from
    /// `prompt_too_large` (raise the API's byte limit). Folding it into
    /// either would point at the wrong knob.
    /// A request that fails at `submit()` never builds a `ChunkStream`, so
    /// the runner's `tasks_failed_total` cannot see it. If these aren't
    /// counted as rejections either, a worker failing 100% of requests is
    /// indistinguishable from a healthy idle one on every metric we export.
    ///
    /// Strictly-greater deltas throughout: `reason` has no isolating label
    /// and the registry is process-global, so other tests in this binary
    /// drive the same arms concurrently.
    #[test]
    fn every_engine_error_books_a_rejection_reason() {
        use cascadia_engine::EngineError;
        // (error, expected reason) — one per arm of the classifier.
        let cases: Vec<(EngineError, &str)> = vec![
            (EngineError::QueueFull { queued: 8, cap: 8 }, "capacity"),
            (EngineError::PromptTooLong("x".into()), "prompt_over_window"),
            (EngineError::NotLoaded, "engine_unavailable"),
            (EngineError::NotConnected, "engine_unavailable"),
            (EngineError::Backend("boom".into()), "engine_error"),
            (EngineError::BatchAborted("x".into()), "engine_error"),
            (EngineError::InvalidConfig("bad".into()), "invalid_request"),
            (EngineError::ModelNotFound("nope".into()), "invalid_request"),
        ];
        for (err, reason) in cases {
            let before = cascadia_metrics::API_REJECTED_TOTAL
                .with_label_values(&[reason])
                .get();
            let status = engine_error_response(err).status();
            let after = cascadia_metrics::API_REJECTED_TOTAL
                .with_label_values(&[reason])
                .get();
            assert!(
                after > before,
                "{status} was returned but booked no {reason} rejection"
            );
        }
    }

    /// Strictly-greater, not an exact delta: the registry is process-global
    /// and `prompt_too_long_maps_to_413_not_503` drives the same arm on
    /// another test thread, so an `== before + 1` assertion races it. That
    /// this is NOT also booked as `capacity` needs no assertion — the two
    /// live in exclusive arms of one `match`.
    #[test]
    fn prompt_over_window_counts_its_own_rejection_reason() {
        let before = cascadia_metrics::API_REJECTED_TOTAL
            .with_label_values(&["prompt_over_window"])
            .get();
        let r = engine_error_response(cascadia_engine::EngineError::PromptTooLong(
            "prompt is 207 tokens but this worker's per-slot KV region holds 127".into(),
        ));
        assert_eq!(r.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let after = cascadia_metrics::API_REJECTED_TOTAL
            .with_label_values(&["prompt_over_window"])
            .get();
        assert!(
            after > before,
            "a windowed-out prompt must count as a rejection ({before} -> {after})"
        );
    }

    #[test]
    fn chunk_token_count_honors_engine_count() {
        // ov-genai's shape: whole response on one final chunk, n_tokens set.
        let c = Chunk::final_marker("t", "The capital of France is Paris.").with_n_tokens(8);
        assert_eq!(chunk_token_count(&c), 8);

        // Same shape WITHOUT n_tokens falls back to 1 (the pre-#55 behavior
        // that under-counted ov-genai) — documents why the engine must set it.
        let c = Chunk::final_marker("t", "The capital of France is Paris.");
        assert_eq!(chunk_token_count(&c), 1);

        // Empty final marker (mock/runtime) contributes 0 — no phantom token.
        let c = Chunk::final_marker("t", "");
        assert_eq!(chunk_token_count(&c), 0);

        // A normal per-token chunk counts as 1.
        let c = Chunk::token("t", 123, "Paris");
        assert_eq!(chunk_token_count(&c), 1);
    }

    // The R1-family chat template the exporter (tools/export_deepseek_v4.py,
    // DSV4_CHAT_TEMPLATE) writes for DeepSeek-V4-Flash, which ships no jinja
    // template of its own. Kept byte-identical here so this test guards the
    // exact string that reaches production.
    const DSV4_CHAT_TEMPLATE: &str = concat!(
        "{% if not add_generation_prompt is defined %}",
        "{% set add_generation_prompt = false %}{% endif %}",
        "{{ bos_token }}",
        "{% for message in messages %}",
        "{% if message['role'] == 'system' %}{{ message['content'] }}",
        "{% elif message['role'] == 'user' %}",
        "{{ '<｜User｜>' + message['content'] }}",
        "{% elif message['role'] == 'assistant' %}",
        "{{ '<｜Assistant｜>' + message['content'] + '<｜end▁of▁sentence｜>' }}",
        "{% endif %}{% endfor %}",
        "{% if add_generation_prompt %}{{ '<｜Assistant｜>' }}{% endif %}",
    );

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    // Proves the dsv4 chat fix at the render boundary: WITHOUT a template the
    // API degenerates an instruct model into a "role: content" join; WITH the
    // R1 template it emits the <｜User｜>/<｜Assistant｜> markers the model was
    // aligned on. Same messages through both paths — the diff is the fix.
    #[test]
    fn dsv4_chat_template_renders_r1_format_not_legacy_join() {
        let msgs = [
            msg("user", "What is 2+2?"),
            msg("assistant", "4"),
            msg("user", "And 3+3?"),
        ];

        // No template (a dsv4 export before the fix): generic formatter, no
        // R1 turn markers — the state that degrades instruct chat.
        let legacy = ChatPromptRenderer::new(&ChatTemplateConfig::default())
            .render(&msgs)
            .unwrap();
        assert!(
            !legacy.contains("<｜User｜>") && !legacy.contains("<｜Assistant｜>"),
            "legacy formatter must NOT emit R1 markers, got: {legacy}"
        );

        // With the R1 template + the model's real bos/eos tokens.
        let cfg = ChatTemplateConfig {
            template: Some(DSV4_CHAT_TEMPLATE.to_string()),
            bos_token: Some("<｜begin▁of▁sentence｜>".to_string()),
            eos_token: Some("<｜end▁of▁sentence｜>".to_string()),
        };
        let r1 = ChatPromptRenderer::new(&cfg).render(&msgs).unwrap();
        let want = "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜>4<｜end▁of▁sentence｜><｜User｜>And 3+3?<｜Assistant｜>";
        assert_eq!(r1, want, "R1 render mismatch");
    }

    // Proves the CLI's sparse-moe root-layout path (load_chat_template_config_at,
    // which the SparseMoe arm now uses) resolves the template the exporter
    // writes to the model root, and extracts bos/eos from the AddedToken object
    // shape dsv4's tokenizer_config uses.
    #[test]
    fn load_chat_template_at_reads_root_jinja_and_addedtoken_tokens() {
        let dir = std::env::temp_dir().join(format!("dsv4_tpl_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // dsv4 export layout: tokenizer_config.json with AddedToken bos/eos and
        // NO inline chat_template, plus a sibling chat_template.jinja.
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"add_bos_token":false,"bos_token":{"__type":"AddedToken","content":"<｜begin▁of▁sentence｜>"},"eos_token":{"__type":"AddedToken","content":"<｜end▁of▁sentence｜>"}}"#,
        )
        .unwrap();
        std::fs::write(dir.join("chat_template.jinja"), DSV4_CHAT_TEMPLATE).unwrap();

        let cfg = load_chat_template_config_at(&dir);
        assert!(cfg.template.is_some(), "template must load from root jinja");
        assert_eq!(cfg.bos_token.as_deref(), Some("<｜begin▁of▁sentence｜>"));
        assert_eq!(cfg.eos_token.as_deref(), Some("<｜end▁of▁sentence｜>"));

        // ...and it renders end-to-end through the same public renderer.
        let out = ChatPromptRenderer::new(&cfg)
            .render(&[msg("user", "hi")])
            .unwrap();
        assert_eq!(out, "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜>");

        let _ = std::fs::remove_dir_all(&dir);
    }

    async fn make_app() -> Router {
        let mut runner = Runner::new(Box::new(MockBuilder::new()));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("mock-model", "CPU"),
            )
            .await
            .unwrap();
        make_router(Arc::new(runner), "mock-model")
    }

    /// A router whose model carries a (minimal) tools-capable chat template.
    /// Tool requests need one: without a template there is nothing that can
    /// render tool definitions, and the API refuses rather than answering as
    /// though none were sent.
    async fn make_tool_app() -> Router {
        let mut runner = Runner::new(Box::new(MockBuilder::new()));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("mock-model", "CPU"),
            )
            .await
            .unwrap();
        let cfg = Config {
            chat_template: ChatTemplateConfig {
                template: Some(
                    "{% if tools %}{{ tools | tojson }}{% endif %}\
                     {% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
                        .into(),
                ),
                bos_token: None,
                eos_token: None,
            },
            ..Config::default()
        };
        make_router_with_config(Arc::new(runner), "mock-model", cfg)
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let app = make_app().await;
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["status"], "ok");
    }

    #[tokio::test]
    async fn list_models_returns_configured_id() {
        let app = make_app().await;
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["data"][0]["id"], "mock-model");
    }

    #[tokio::test]
    async fn chat_completion_returns_assistant_message() {
        let app = make_app().await;
        let payload = serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "alpha bravo charlie"}],
            "max_tokens": 2,
            "stream": false,
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        // Mock yields words from the prompt; check non-empty content.
        let content = v["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(!content.is_empty(), "completion content was empty");
    }

    // Helper: POST a chat-completions request body, return (status, parsed JSON).
    async fn post_chat(app: Router, payload: serde_json::Value) -> (StatusCode, Value) {
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), 16384).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        (status, v)
    }

    #[tokio::test]
    async fn finish_reason_length_when_max_tokens_hit() {
        // Mock echoes the prompt's words; with 5 words and max_tokens 2 it hits
        // the cap before exhausting the prompt → finish_reason "length".
        let (status, v) = post_chat(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "alpha bravo charlie delta echo"}],
                "max_tokens": 2,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn finish_reason_stop_when_prompt_exhausted() {
        // 2 words, generous cap → the mock exhausts the prompt → "stop".
        let (status, v) = post_chat(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "alpha bravo"}],
                "max_tokens": 64,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn stop_sequence_truncates_output() {
        // Full output would be "alpha bravo charlie delta "; stop ["charlie"]
        // trims it to "alpha bravo " and forces finish_reason "stop".
        let (status, v) = post_chat(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "alpha bravo charlie delta"}],
                "max_tokens": 64,
                "stop": "charlie",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let content = v["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(
            !content.contains("charlie"),
            "stop text not trimmed: {content:?}"
        );
        assert_eq!(content.trim(), "alpha bravo");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn full_sampling_param_set_is_accepted() {
        // Every new OpenAI field present + well-typed must parse (no 400).
        let (status, _v) = post_chat(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "alpha bravo"}],
                "max_tokens": 4,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "seed": 1234,
                "stop": ["zzz", "qqq"],
                "frequency_penalty": 0.5,
                "presence_penalty": -0.25,
                "logprobs": true,
                "top_logprobs": 3,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn streaming_usage_emitted_when_requested() {
        let response = make_app()
            .await
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model": "mock-model",
                            "messages": [{"role": "user", "content": "alpha bravo charlie"}],
                            "max_tokens": 8,
                            "stream": true,
                            "stream_options": {"include_usage": true},
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 65536).await.unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("\"usage\""),
            "no usage frame in stream: {text}"
        );
        assert!(text.contains("\"completion_tokens\""));
        assert!(text.trim_end().ends_with("data: [DONE]"));
    }

    async fn post_completions(app: Router, payload: serde_json::Value) -> (StatusCode, Value) {
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), 16384).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        (status, v)
    }

    #[tokio::test]
    async fn completions_returns_text_completion() {
        // Mock echoes prompt words verbatim (no chat template applied).
        let (status, v) = post_completions(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "prompt": "alpha bravo charlie",
                "max_tokens": 64,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["object"], "text_completion");
        let text = v["choices"][0]["text"].as_str().unwrap();
        assert_eq!(text.trim(), "alpha bravo charlie");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["completion_tokens"], 3);
    }

    #[tokio::test]
    async fn completions_echo_prepends_prompt() {
        let (status, v) = post_completions(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "prompt": "alpha bravo",
                "max_tokens": 64,
                "echo": true,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        // echo prepends the raw prompt to the (prompt-stripped) continuation.
        let text = v["choices"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("alpha bravo"), "echo missing: {text:?}");
    }

    #[tokio::test]
    async fn completions_streaming_emits_text_completion_and_done() {
        let response = make_app()
            .await
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model": "mock-model",
                            "prompt": "alpha bravo charlie",
                            "max_tokens": 8,
                            "stream": true,
                            "stream_options": {"include_usage": true},
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 65536).await.unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("\"text_completion\""), "wrong object: {text}");
        assert!(text.contains("\"text\""), "no text delta: {text}");
        assert!(text.contains("\"usage\""));
        assert!(text.trim_end().ends_with("data: [DONE]"));
    }

    #[tokio::test]
    async fn completions_multi_prompt_array_is_400() {
        let (status, _v) = post_completions(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "prompt": ["one", "two"],
                "max_tokens": 4,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn completions_honor_stop_and_finish_reason() {
        let (status, v) = post_completions(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "prompt": "alpha bravo charlie delta",
                "max_tokens": 64,
                "stop": ["charlie"],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let text = v["choices"][0]["text"].as_str().unwrap();
        assert_eq!(text.trim(), "alpha bravo");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn completions_finish_reason_length_when_capped() {
        // 5 prompt words, max_tokens 2 → hits the cap → "length" (the forked
        // finish_reason path must preserve length just like the chat path).
        let (status, v) = post_completions(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "prompt": "alpha bravo charlie delta echo",
                "max_tokens": 2,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn completions_engine_error_returns_503() {
        // The mock fails a task whose prompt contains `__engine_error__`. The
        // forked non-streaming error branch must surface that as a 5xx, not a
        // 200 with empty text.
        let (status, _v) = post_completions(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "prompt": "__engine_error__",
                "max_tokens": 8,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn health_reflects_pipeline_and_recovers() {
        // /health must track the pipeline, not return a hardcoded 200: a
        // completion that fails 5xx (stand-in for a dropped peer link) degrades
        // it; a subsequent success recovers it. The `ready` flag is shared
        // across router clones (same AppState Arc), so sequential oneshots see
        // each other's effect.
        async fn health(app: Router) -> (StatusCode, Value) {
            let r = app
                .oneshot(
                    Request::builder()
                        .uri("/health")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let st = r.status();
            let b = to_bytes(r.into_body(), 1024).await.unwrap();
            (st, serde_json::from_slice(&b).unwrap())
        }
        async fn chat(app: Router, content: String) -> StatusCode {
            let payload = serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": content}],
                "stream": false,
            });
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
            .status()
        }

        let app = make_app().await;
        // fresh → ready
        let (s, v) = health(app.clone()).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["status"], "ok");
        // a 5xx completion → degraded
        assert_eq!(
            chat(app.clone(), "__engine_error__".into()).await,
            StatusCode::SERVICE_UNAVAILABLE
        );
        let (s, v) = health(app.clone()).await;
        assert_eq!(
            s,
            StatusCode::SERVICE_UNAVAILABLE,
            "health degrades after a 5xx"
        );
        assert_eq!(v["status"], "degraded");
        // a successful completion → ready again
        assert_eq!(chat(app.clone(), "hello".into()).await, StatusCode::OK);
        let (s, _) = health(app.clone()).await;
        assert_eq!(s, StatusCode::OK, "health recovers after a success");
    }

    #[tokio::test]
    async fn empty_messages_returns_400_not_empty_200() {
        // Degenerate input (no messages → empty rendered prompt) is a client
        // error: reject with 400 at the API rather than admitting it to the
        // engine and returning a 200 with empty content.
        let app = make_app().await;
        let payload = serde_json::json!({
            "model": "mock-model",
            "messages": [],
            "stream": false,
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn failed_task_returns_5xx_not_empty_200() {
        // C1 regression: a task the engine FAILS (here via the mock's
        // `__engine_error__` sentinel, standing in for a qwen36 handshake
        // NAK) must surface as a 5xx, not a 200 with empty content.
        let app = make_app().await;
        let payload = serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "__engine_error__"}],
            "stream": false,
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert!(
            v["error"].as_str().unwrap().contains("mock injected"),
            "5xx body should carry the engine's failure reason, got {v}"
        );
    }

    #[test]
    fn load_chat_template_falls_back_to_jinja_sibling() {
        // Gemma 3/4 (and recent Llama) ship the template as a sibling
        // chat_template.jinja file with no inline `chat_template` field in
        // tokenizer_config.json. The loader must pick up the .jinja file.
        let dir = std::env::temp_dir().join(format!("cascadia_g4_tmpl_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir); // robust to leftover dirt (PID reuse)
        let tok = dir.join("tokenizer");
        std::fs::create_dir_all(&tok).unwrap();
        std::fs::write(
            tok.join("tokenizer_config.json"),
            r#"{"bos_token": "<bos>", "eos_token": {"content": "<eos>"}}"#,
        )
        .unwrap();
        std::fs::write(
            tok.join("chat_template.jinja"),
            "{%- macro greet() -%}hello{%- endmacro -%}{{ greet() }}",
        )
        .unwrap();
        let cfg = load_chat_template_config(&dir);
        std::fs::remove_dir_all(&dir).ok();
        assert!(
            cfg.template.as_deref().unwrap_or("").contains("macro"),
            "expected template loaded from chat_template.jinja"
        );
        assert_eq!(cfg.bos_token.as_deref(), Some("<bos>"));
        assert_eq!(cfg.eos_token.as_deref(), Some("<eos>"));
    }

    /// Gemma-style chat template for the macro/renderer tests.
    const MACRO_TEMPLATE: &str = "{%- macro turn(role, text) -%}<start_of_turn>{{ role }}\n\
                    {{ text }}<end_of_turn>\n{% endmacro -%}\
                    {{ bos_token }}{% for m in messages %}{{ turn(m.role, m.content) }}{% endfor %}\
                    {% if add_generation_prompt %}<start_of_turn>model\n{% endif %}";

    #[test]
    fn chat_env_supports_macro_templates() {
        // Guards the minijinja `macros` feature; without it Gemma-style
        // instruct templates fail to parse and silently fall back to legacy.
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let env = build_chat_env(MACRO_TEMPLATE).expect("macro-based template must parse");
        let out = render_with_chat_env(&env, &msgs, "<bos>", "<eos>", true, None, None)
            .expect("macro-based template must render");
        assert!(out.contains("<bos>"), "out={out}");
        assert!(out.contains("<start_of_turn>user"), "out={out}");
        assert!(out.contains("hi"), "out={out}");
        assert!(out.contains("<start_of_turn>model"), "out={out}");
    }

    #[test]
    fn renderer_opts_control_the_thinking_branch() {
        // GLM/Qwen3 hybrid-reasoning templates branch on `enable_thinking`.
        // `render_with_tools` must keep its historical thinking-on default;
        // `render_with_opts(.., false)` must take the off branch.
        const T: &str = "{% for m in messages %}{{ m.content }}{% endfor %}\
            {% if enable_thinking %}<think>{% else %}<think></think>{% endif %}";
        let cfg = ChatTemplateConfig {
            template: Some(T.into()),
            bos_token: None,
            eos_token: None,
        };
        let renderer = ChatPromptRenderer::new(&cfg);
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let on = renderer.render_with_tools(&msgs, None).unwrap();
        assert!(on.ends_with("<think>"), "thinking-on render: {on}");
        let off = renderer.render_with_opts(&msgs, None, false).unwrap();
        assert!(
            off.ends_with("<think></think>"),
            "thinking-off render: {off}"
        );
    }

    #[test]
    fn tool_call_arguments_render_as_object_for_hf_templates() {
        // Regression: HF tool templates (Qwen3 etc.) iterate `tool_call.arguments | items`,
        // which needs a JSON object. OpenAI ships `arguments` as a string; passing it through
        // verbatim makes minijinja error ("cannot convert value into pairs") and the whole
        // request silently falls back to the legacy `role: content` formatter.
        const T: &str = "{%- for m in messages -%}{%- if m.tool_calls -%}\
            {%- for tc in m.tool_calls -%}fn={{ tc.function.name }};\
            {%- for k, v in tc.function.arguments | items -%}{{ k }}={{ v }};{%- endfor -%}\
            {%- endfor -%}{%- endif -%}{%- endfor -%}";
        let msgs = [ChatMessage {
            role: "assistant".into(),
            content: String::new(),
            name: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".into(),
                r#type: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: r#"{"city":"Tokyo"}"#.into(),
                },
            }]),
            tool_call_id: None,
        }];
        let env = build_chat_env(T).expect("template parses");
        let out = render_with_chat_env(&env, &msgs, "<bos>", "<eos>", true, None, None)
            .expect("arguments|items must not fail the render");
        assert!(out.contains("fn=get_weather"), "out={out}");
        assert!(out.contains("city=Tokyo"), "out={out}");
    }

    #[test]
    fn chat_prompt_renderer_renders_then_falls_back() {
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        // Template present: renders through the parsed env (parsed once in `new`).
        let r = ChatPromptRenderer::new(&ChatTemplateConfig {
            template: Some(MACRO_TEMPLATE.into()),
            bos_token: Some("<bos>".into()),
            eos_token: Some("<eos>".into()),
        });
        let out = r.render(&msgs).unwrap();
        assert!(out.contains("<bos>"), "out={out}");
        assert!(out.contains("<start_of_turn>user"), "out={out}");

        // No template: falls back to the legacy formatter.
        let none = ChatPromptRenderer::new(&ChatTemplateConfig::default());
        assert_eq!(none.render(&msgs).unwrap(), render_prompt_legacy(&msgs));

        // Unparseable template is treated as absent → legacy fallback, not a panic.
        let broken = ChatPromptRenderer::new(&ChatTemplateConfig {
            template: Some("{% for %}".into()),
            bos_token: None,
            eos_token: None,
        });
        assert_eq!(broken.render(&msgs).unwrap(), render_prompt_legacy(&msgs));
    }

    #[test]
    fn empty_jinja_template_falls_back_to_none() {
        // A present-but-empty chat_template.jinja must not be treated as a
        // usable template — rendering through it would yield an empty prompt,
        // worse than the legacy formatter. The loader must return None so the
        // request falls back. Guards #115.
        let dir = std::env::temp_dir().join(format!("cascadia_g4_empty_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir); // robust to leftover dirt (PID reuse)
        let tok = dir.join("tokenizer");
        std::fs::create_dir_all(&tok).unwrap();
        std::fs::write(
            tok.join("tokenizer_config.json"),
            r#"{"bos_token": "<bos>"}"#,
        )
        .unwrap();
        // whitespace-only — the worst case (passes a naive is_empty check)
        std::fs::write(tok.join("chat_template.jinja"), "   \n\t  ").unwrap();
        let cfg = load_chat_template_config(&dir);
        std::fs::remove_dir_all(&dir).ok();
        assert!(
            cfg.template.is_none(),
            "empty/whitespace chat_template.jinja must yield template=None, got {:?}",
            cfg.template
        );
    }

    #[test]
    fn request_accepts_tools_and_tool_choice() {
        let raw = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tool_choice":"auto","tools":[{"type":"function","function":{"name":"get_weather",
            "description":"Get weather","parameters":{"type":"object",
            "properties":{"city":{"type":"string"}}}}}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
        let tools = req.tools.expect("tools parse");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].r#type, "function");
        assert_eq!(tools[0].function["name"], "get_weather");
        assert_eq!(req.tool_choice.unwrap(), serde_json::json!("auto"));
    }

    #[test]
    fn message_accepts_null_content_and_tool_calls() {
        let raw = r#"{"role":"assistant","content":null,
            "tool_calls":[{"id":"call_abc","type":"function",
            "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]}"#;
        let m: ChatMessage = serde_json::from_str(raw).unwrap();
        assert!(m.content.is_empty(), "null content -> empty string");
        let calls = m.tool_calls.expect("tool_calls parse");
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"city\":\"Paris\"}");
    }

    #[test]
    fn message_accepts_tool_role_with_tool_call_id() {
        let raw = r#"{"role":"tool","content":"{\"temp_c\":18}",
            "tool_call_id":"call_abc","name":"get_weather"}"#;
        let m: ChatMessage = serde_json::from_str(raw).unwrap();
        assert_eq!(m.role, "tool");
        assert_eq!(m.tool_call_id.as_deref(), Some("call_abc"));
        assert_eq!(m.name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn non_tool_choice_message_serializes_content_as_string() {
        let msg = ChatChoiceMessage {
            role: "assistant",
            content: Some("hi".into()),
            tool_calls: None,
        };
        let v = serde_json::to_value(&msg).unwrap();
        assert_eq!(v["content"], "hi");
        assert!(
            v.get("tool_calls").is_none(),
            "no tool_calls key on non-tool message"
        );
    }

    #[test]
    fn parse_llama_python_tag_single() {
        let calls = parse_tool_calls(
            "<|python_tag|>{\"name\":\"get_weather\",\"parameters\":{\"city\":\"Paris\"}}",
        )
        .expect("one");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap()
                ["city"],
            "Paris"
        );
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].r#type, "function");
    }

    #[test]
    fn parse_bare_json_requires_tool_shape() {
        // M3(b): a plain JSON DATA answer with a stray `name` key (no marker, no
        // args/params) must NOT be misread as a tool call.
        assert!(parse_tool_calls(r#"{"name":"Alice","age":30}"#).is_none());
        assert!(parse_tool_calls(r#"[{"name":"Bob"}]"#).is_none());
        // A real bare Llama call (carries `parameters`, even `{}`) still parses.
        let calls = parse_tool_calls(r#"{"name":"get_weather","parameters":{"city":"Paris"}}"#)
            .expect("bare call");
        assert_eq!(calls[0].function.name, "get_weather");
        let calls = parse_tool_calls(r#"{"name":"ping","arguments":{}}"#).expect("bare empty-args");
        assert_eq!(calls[0].function.name, "ping");
        // The explicit <|python_tag|> marker stays lenient (no params needed).
        let calls = parse_tool_calls(r#"<|python_tag|>{"name":"now"}"#).expect("explicit no-arg");
        assert_eq!(calls[0].function.name, "now");
    }

    #[test]
    fn parse_qwen_hermes_single() {
        let calls = parse_tool_calls("<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>").expect("one");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap()
                ["city"],
            "Paris"
        );
    }

    #[test]
    fn parse_qwen3_xml_function_dialect() {
        // Qwen3.6-MoE emits the <function=NAME><parameter=K>V</parameter></function>
        // XML dialect inside <tool_call> (not the JSON Hermes form). Live rig: the MoE
        // produced exactly this, finish=stop because the parser didn't recognise it.
        let calls = parse_tool_calls(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
        )
        .expect("one tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap()
                ["city"],
            "Paris"
        );
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].r#type, "function");
    }

    #[test]
    fn parse_qwen3_xml_multi_param() {
        let calls = parse_tool_calls(
            "<tool_call><function=f><parameter=a>1</parameter><parameter=b>two</parameter></function></tool_call>",
        )
        .expect("one");
        let args = serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap();
        assert_eq!(args["a"], "1");
        assert_eq!(args["b"], "two");
    }

    #[test]
    fn parse_tool_call_arg_containing_close_tag() {
        // An argument value that literally contains "</tool_call>" must not
        // truncate the block (brace-balanced scan, not first-close-tag).
        let calls = parse_tool_calls(
            "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"</tool_call> bye\"}}</tool_call>",
        )
        .expect("one call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "echo");
        let args = serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap();
        assert_eq!(args["text"], "</tool_call> bye");
    }

    #[test]
    fn tool_call_drafted_before_think_close_is_not_parsed() {
        // A call drafted in the scratchpad while deliberating must not fire —
        // armed (thinking on).
        let buf = "<tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call></think>no calls needed";
        assert!(parse_tool_calls(tool_call_scan_text(buf, true)).is_none());
    }

    #[test]
    fn tool_call_after_think_close_is_parsed() {
        let buf = "reasoning...</think><tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>";
        let calls = parse_tool_calls(tool_call_scan_text(buf, true)).expect("one");
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn only_tool_call_after_think_close_is_parsed_when_both_present() {
        let buf = "<tool_call>{\"name\":\"draft\",\"arguments\":{}}</tool_call></think><tool_call>{\"name\":\"real\",\"arguments\":{}}</tool_call>";
        let calls = parse_tool_calls(tool_call_scan_text(buf, true)).expect("one");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "real");
    }

    #[test]
    fn tool_call_with_no_think_close_is_parsed_unchanged() {
        // Non-reasoning models never emit `</think>`; behaviour must be unchanged.
        let buf = "<tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>";
        let calls = parse_tool_calls(tool_call_scan_text(buf, true)).expect("one");
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn tool_call_scan_starts_after_first_think_close_not_last() {
        // find, not rfind: the scan must resume right after the FIRST close.
        let buf =
            "</think>middle</think><tool_call>{\"name\":\"real\",\"arguments\":{}}</tool_call>";
        assert_eq!(
            tool_call_scan_text(buf, true),
            "middle</think><tool_call>{\"name\":\"real\",\"arguments\":{}}</tool_call>"
        );
    }

    // The regression the reviewer proved by execution: a REAL call whose own
    // arguments contain the literal "</think>" must not be dropped. Disarmed
    // (thinking off) is the exact case that was unbounded before — no
    // scratchpad exists, so scanning must be a no-op and the whole buffer
    // must reach `parse_tool_calls` unmodified.
    #[test]
    fn disarmed_scan_keeps_real_call_whose_argument_contains_think_close() {
        let buf = r#"<tool_call>{"name":"echo","arguments":{"text":"</think>"}}</tool_call>"#;
        let calls = parse_tool_calls(tool_call_scan_text(buf, false)).expect("one");
        assert_eq!(calls[0].function.name, "echo");
    }

    #[test]
    fn disarmed_scan_keeps_real_call_whose_argument_mentions_think_close_in_prose() {
        let buf =
            r#"<tool_call>{"name":"write","arguments":{"body":"strip </think> tags"}}</tool_call>"#;
        let calls = parse_tool_calls(tool_call_scan_text(buf, false)).expect("one");
        assert_eq!(calls[0].function.name, "write");
    }

    #[test]
    fn disarmed_scan_keeps_real_call_when_think_close_trails_it() {
        let buf = r#"<tool_call>{"name":"a","arguments":{}}</tool_call> aside: </think> ends it"#;
        let calls = parse_tool_calls(tool_call_scan_text(buf, false)).expect("one");
        assert_eq!(calls[0].function.name, "a");
    }

    #[test]
    fn parse_tool_calls_glm_arg_kv_dialect() {
        // EXACTLY the shape GLM-4.5+/GLM-5's own chat_template.jinja emits:
        //   '<tool_call>' + tc.name, then per argument
        //   <arg_key>k</arg_key><arg_value>v</arg_value>, then </tool_call>.
        // Before this dialect was handled the block parsed as nothing and the
        // whole generation shipped as ordinary content — a call the model made,
        // silently dropped, with finish_reason "stop".
        let calls = parse_tool_calls(
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        )
        .expect("GLM arg_key/arg_value dialect must parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args = serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap();
        // A bare string value is NOT valid JSON, so it must stay a string
        // rather than being dropped.
        assert_eq!(args["city"], "Paris");

        // The template JSON-encodes non-string values (`v | tojson if v is not
        // string else v`), so those must come back as their JSON type, not as
        // the string "7".
        let calls = parse_tool_calls(
            "<tool_call>book<arg_key>days</arg_key><arg_value>7</arg_value>\
             <arg_key>opts</arg_key><arg_value>{\"bags\":2}</arg_value></tool_call>",
        )
        .expect("multi-argument GLM block must parse");
        let args = serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap();
        assert_eq!(args["days"], 7);
        assert_eq!(args["opts"]["bags"], 2);

        // Prose inside a <tool_call> block with no <arg_key> is not a call —
        // inventing a zero-argument call from arbitrary text would be worse
        // than dropping it.
        assert!(parse_tool_calls("<tool_call>just some prose</tool_call>").is_none());

        // The older dialects must be untouched by the added fallback.
        let calls = parse_tool_calls(
            "<tool_call><function=lookup><parameter=id>42</parameter></function></tool_call>",
        )
        .expect("Qwen3 XML dialect must still parse");
        assert_eq!(calls[0].function.name, "lookup");
    }

    #[test]
    fn parse_tool_calls_glm_zero_argument_call() {
        // EXACTLY the shape GLM's chat_template emits for a call with no
        // arguments: the bare function name, no `<arg_key>` at all. Before
        // this shape test existed, the missing `<arg_key>` bailed out
        // entirely and the raw markup shipped as assistant content with
        // finish_reason "stop" — every zero-argument tool (get_current_time,
        // list_files, ...) was unusable.
        let calls = parse_tool_calls("<tool_call>get_current_time</tool_call>")
            .expect("bare zero-argument call must parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_current_time");
        assert_eq!(calls[0].function.arguments, "{}");

        // Matching the template's actual output: newline before the close tag.
        let calls = parse_tool_calls("<tool_call>get_current_time\n</tool_call>")
            .expect("bare call with trailing newline must parse");
        assert_eq!(calls[0].function.name, "get_current_time");
        assert_eq!(calls[0].function.arguments, "{}");

        // Whitespace on both sides too.
        let calls = parse_tool_calls("<tool_call>\n  list_files  \n</tool_call>")
            .expect("bare call with surrounding whitespace must parse");
        assert_eq!(calls[0].function.name, "list_files");
        assert_eq!(calls[0].function.arguments, "{}");

        // The guard against fabricating calls from prose must still hold:
        // multi-word text has no `<arg_key>` either but must NOT parse.
        assert!(parse_tool_calls("<tool_call>this is not a function name</tool_call>").is_none());

        // A zero-arg call alongside a normal call with arguments: both must
        // parse, in order, and the fallback must not consume the second block.
        let calls = parse_tool_calls(
            "<tool_call>get_current_time</tool_call>\
             <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        )
        .expect("both calls must parse");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_current_time");
        assert_eq!(calls[0].function.arguments, "{}");
        assert_eq!(calls[1].function.name, "get_weather");
        let args = serde_json::from_str::<serde_json::Value>(&calls[1].function.arguments).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[test]
    fn parse_tool_calls_brace_balance_stress() {
        // 1) nested object/array args + a '}' and a '{' inside a string value +
        //    an escaped quote — exact JSON must survive the balanced scan.
        let calls = parse_tool_calls(
            r#"<tool_call>{"name":"f","arguments":{"o":{"k":1},"a":[1,2],"s":"a{b}c \" }"}}</tool_call>"#,
        )
        .expect("nested");
        assert_eq!(calls.len(), 1);
        let args = serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap();
        assert_eq!(args["o"]["k"], 1);
        assert_eq!(args["a"][1], 2);
        assert_eq!(args["s"], "a{b}c \" }");

        // 2) two blocks; the FIRST carries '</tool_call>' AND '}' inside a value.
        //    Both calls must survive (no truncation, no bleed into the next).
        let calls = parse_tool_calls(
            r#"<tool_call>{"name":"a","arguments":{"t":"</tool_call> } x"}}</tool_call><tool_call>{"name":"b","arguments":{}}</tool_call>"#,
        )
        .expect("two");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
        assert_ne!(calls[0].id, calls[1].id);

        // 3) XML dialect still parses through the fallback (regression).
        let calls = parse_tool_calls(
            "<tool_call><function=g><parameter=x>v</parameter></function></tool_call>",
        )
        .expect("xml");
        assert_eq!(calls[0].function.name, "g");

        // 4) truncated/unbalanced JSON block => skipped, no panic; a later valid
        //    block still parses.
        let calls = parse_tool_calls(
            r#"<tool_call>{"name":"bad","arguments":{"x":1</tool_call><tool_call>{"name":"ok","arguments":{}}</tool_call>"#,
        )
        .expect("recovers");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "ok");

        // 5) a lone unbalanced block never panics and yields nothing.
        assert!(parse_tool_calls(r#"<tool_call>{"name":"x","arguments":{"a":"#).is_none());
    }

    #[test]
    fn parse_qwen_multiple_distinct_ids() {
        let calls = parse_tool_calls(
            "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\
            <tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>",
        )
        .expect("two");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[0].function.arguments, "{}");
        assert_eq!(calls[1].function.name, "b");
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn parse_skips_malformed_first_block_keeps_valid() {
        let calls = parse_tool_calls(
            "<tool_call>{bad</tool_call><tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>",
        )
        .expect("one");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "b");
    }

    #[test]
    fn parse_plain_prose_is_none() {
        assert!(parse_tool_calls("The weather in Paris is sunny.").is_none());
        assert!(parse_tool_calls("you could call <tool_call> here").is_none());
    }

    #[test]
    fn parse_missing_name_and_truncated_no_panic() {
        assert!(parse_tool_calls("<tool_call>{\"arguments\":{}}</tool_call>").is_none());
        assert!(parse_tool_calls("<tool_call>{\"name\":\"x\",\"argum").is_none());
        assert!(parse_tool_calls("<|python_tag|>{\"name\":\"x\"").is_none());
    }

    #[test]
    fn parse_arguments_already_string_passthrough() {
        let calls = parse_tool_calls(
            "<tool_call>{\"name\":\"f\",\"arguments\":\"{\\\"k\\\":1}\"}</tool_call>",
        )
        .expect("one");
        assert_eq!(calls[0].function.arguments, "{\"k\":1}");
    }

    const TOOL_TEMPLATE: &str = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\
        {% if m.tool_calls %}[CALLS:{% for c in m.tool_calls %}{{ c.function.name }}{% endfor %}]{% endif %}\
        {% if m.tool_call_id %}[TCID:{{ m.tool_call_id }}]{% endif %}\n{% endfor %}\
        {% if tools %}[TOOLS:{% for t in tools %}{{ t.function.name }}{% endfor %}]{% endif %}";

    #[test]
    fn renderer_forwards_tools_into_template() {
        let env = build_chat_env(TOOL_TEMPLATE).unwrap();
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let tools = vec![Tool {
            r#type: "function".into(),
            function: serde_json::json!({"name":"get_weather"}),
        }];
        let out = render_with_chat_env(&env, &msgs, "", "", true, None, Some(&tools)).unwrap();
        assert!(
            out.contains("[TOOLS:get_weather]"),
            "tools not forwarded: {out}"
        );
    }

    #[test]
    fn chat_env_supports_tojson_filter() {
        // Guards the minijinja `json` feature. Real Llama-3.1/Qwen instruct
        // templates serialize the tool schema via `{{ tool | tojson }}`; without
        // `json` the filter is unknown, render_with_chat_env errors, and
        // render_with_tools silently falls back to the legacy formatter —
        // dropping tools from the prompt so the model never sees them.
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let tools = vec![Tool {
            r#type: "function".into(),
            function: serde_json::json!({"name":"get_weather"}),
        }];
        let env = build_chat_env("{% if tools %}{{ tools | tojson }}{% endif %}")
            .expect("tojson template must parse");
        let out = render_with_chat_env(&env, &msgs, "", "", true, None, Some(&tools))
            .expect("tojson template must render (needs minijinja `json` feature)");
        assert!(
            out.contains("get_weather"),
            "tojson did not serialize tools: {out}"
        );
    }

    /// Run `py_tojson` over a JSON literal. Expectations in these tests are
    /// verbatim `json.dumps` output — transformers substitutes exactly that for
    /// Jinja2's `tojson`, so it is the byte-level reference for every HF
    /// template.
    fn tojson_of(
        v: serde_json::Value,
        kwargs: &[(&str, minijinja::value::Value)],
    ) -> Result<String, minijinja::Error> {
        let args = minijinja::value::Kwargs::from_iter(kwargs.iter().map(|(k, v)| (*k, v.clone())));
        py_tojson(minijinja::value::Value::from_serialize(&v), None, args)
            .map(|out| out.to_string())
    }

    #[test]
    fn tojson_matches_python_separators_and_leaves_punctuation_alone() {
        // minijinja's builtin writes `{"a":1}` and escapes `'`/`<`/`&` into
        // \uXXXX for HTML safety. json.dumps does neither, so the builtin puts
        // tool schemas on the wire in bytes no model was trained on.
        // Keys are written alphabetically here so the expectation isolates
        // separators and escaping: `serde_json::Value` is a BTreeMap, which
        // re-sorts client key order independently of this filter. See
        // `tool_schema_key_order_is_alphabetical_not_client_order`.
        let out = tojson_of(
            serde_json::json!([{
                "function": {
                    "description": "the user's city <b> & more",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "type": "object"
                    }
                },
                "type": "function"
            }]),
            &[],
        )
        .expect("tojson must render");
        assert_eq!(
            out,
            r#"[{"function": {"description": "the user's city <b> & more", "name": "get_weather", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"], "type": "object"}}, "type": "function"}]"#
        );
    }

    #[test]
    fn tojson_number_formatting_matches_python() {
        // Schemas carry numeric bounds (minimum/maximum/multipleOf), so number
        // formatting is on the byte-parity path too. serde_json emits
        // shortest-roundtrip via ryu; Python uses repr. They agree except that
        // Python zero-pads the exponent to two digits: `1e-07` vs our `1e-7`.
        // Left as-is — rewriting exponents in serialized JSON risks mangling
        // string contents, and it takes a bound like `multipleOf: 1e-7` in a
        // tool schema to reach it.
        let v: serde_json::Value =
            serde_json::from_str(r#"{"a":1e30,"b":1.0,"c":0.1,"d":1e-7}"#).unwrap();
        let out = tojson_of(v, &[]).unwrap();
        assert_eq!(out, r#"{"a": 1e+30, "b": 1.0, "c": 0.1, "d": 1e-7}"#);
    }

    #[test]
    fn tojson_empty_containers_match_python() {
        let out = tojson_of(serde_json::json!({"a": [], "b": {}}), &[]).unwrap();
        assert_eq!(out, r#"{"a": [], "b": {}}"#);
    }

    #[test]
    fn tojson_ensure_ascii_false_keeps_utf8_and_true_escapes_it() {
        let v = serde_json::json!({"city": "Köln", "emoji": "🌦"});
        let raw = tojson_of(v.clone(), &[("ensure_ascii", false.into())]).unwrap();
        assert_eq!(raw, "{\"city\": \"Köln\", \"emoji\": \"🌦\"}");

        // Above the BMP json.dumps emits a surrogate pair, not one \U escape.
        let escaped = tojson_of(v, &[("ensure_ascii", true.into())]).unwrap();
        let expected = concat!(
            "{\"city\": \"K",
            "\\u00f6",
            "ln\", \"emoji\": \"",
            "\\ud83c\\udf26",
            "\"}"
        );
        assert_eq!(escaped, expected);
    }

    #[test]
    fn tojson_indent_matches_python() {
        let out = tojson_of(
            serde_json::json!({"a": 1, "b": [1, 2]}),
            &[("indent", 4.into())],
        )
        .unwrap();
        assert_eq!(
            out,
            "{\n    \"a\": 1,\n    \"b\": [\n        1,\n        2\n    ]\n}"
        );
    }

    #[test]
    fn tojson_refuses_unsupported_kwargs_rather_than_ignoring_them() {
        // Consumed so the render doesn't die on an unknown kwarg, but honoured
        // or refused — never silently dropped.
        let err = tojson_of(
            serde_json::json!({"b": 1, "a": 2}),
            &[("sort_keys", true.into())],
        )
        .expect_err("sort_keys=True must not be silently ignored");
        assert!(err.to_string().contains("sort_keys"), "{err}");

        assert!(
            tojson_of(serde_json::json!({}), &[("sort_keys", false.into())]).is_ok(),
            "sort_keys=False is the default and must be accepted"
        );
    }

    #[test]
    fn chat_env_renders_glm_style_tojson_ensure_ascii_kwarg() {
        // Regression: GLM-5's tool block calls `tojson(ensure_ascii=False)`.
        // minijinja's builtin rejects the kwarg at RENDER time — inside a
        // branch only taken when tools are present — so the template parsed
        // clean, every tool-less request worked, and every tool request
        // silently fell back to a formatter that cannot express tools at all.
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let tools = vec![Tool {
            r#type: "function".into(),
            function: serde_json::json!({"name": "get_weather", "description": "user's city"}),
        }];
        let env = build_chat_env(
            "{% if tools %}{{ tools[0].function | tojson(ensure_ascii=False) }}{% endif %}",
        )
        .expect("template must parse");
        let out = render_with_chat_env(&env, &msgs, "", "", true, None, Some(&tools))
            .expect("tojson(ensure_ascii=False) must render");
        assert_eq!(
            out,
            r#"{"description": "user's city", "name": "get_weather"}"#
        );
    }

    #[test]
    fn smoke_battery_catches_a_tools_only_render_failure_at_load() {
        // The exact shape of the GLM-5 incident: parses clean, renders clean
        // for every tool-less request, fails only once tools are present.
        let env = build_chat_env("ok{% if tools %}{{ boom.nope.deeper }}{% endif %}").unwrap();
        let failures = chat_template_smoke_failures(&env);
        let cases: Vec<_> = failures.iter().map(|(c, _)| *c).collect();
        assert!(cases.contains(&"tools"), "battery missed it: {cases:?}");
        assert!(
            !cases.contains(&"plain"),
            "tool-less shapes must still pass: {cases:?}"
        );
    }

    #[test]
    fn smoke_battery_is_quiet_for_a_healthy_template_and_for_rejections() {
        let ok = build_chat_env(
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\
             {% if tools %}{{ tools | tojson(ensure_ascii=False) }}{% endif %}",
        )
        .unwrap();
        assert!(chat_template_smoke_failures(&ok).is_empty());

        // A template refusing a system role is validating input, not broken —
        // it must not show up as a startup defect.
        let strict = build_chat_env(
            r#"{% for m in messages %}{% if m.role == "system" %}\
{{ raise_exception("no system role") }}{% endif %}{{ m.content }}{% endfor %}"#,
        )
        .unwrap();
        let cases: Vec<_> = chat_template_smoke_failures(&strict)
            .iter()
            .map(|(c, _)| *c)
            .collect();
        assert!(!cases.contains(&"system"), "rejection reported: {cases:?}");
    }

    #[test]
    fn tools_request_errors_instead_of_silently_falling_back() {
        // The incident this whole path exists for: a render failure inside a
        // tools-only branch dropped to a formatter that cannot express tools,
        // and the caller got a confident 200 answering a question the model was
        // never asked.
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let tools = vec![Tool {
            r#type: "function".into(),
            function: serde_json::json!({"name": "get_weather"}),
        }];
        // Renders fine without tools; explodes inside the tools branch.
        let env = build_chat_env("{% if tools %}{{ nope.missing.deeper }}{% endif %}ok").unwrap();

        let err = render_or_fallback(Some(&env), &msgs, "", "", true, None, Some(&tools))
            .expect_err("a tools request must not fall back");
        assert!(matches!(err, PromptRenderError::Failed(_)), "{err:?}");

        // Same broken template, no tools: fallback is still allowed, because a
        // degraded prompt beats a dead endpoint.
        let out = render_or_fallback(Some(&env), &msgs, "", "", true, None, None)
            .expect("tool-less requests keep the fallback");
        assert_eq!(out, "ok");
    }

    #[test]
    fn raw_template_resolves_undefined_reasoning_effort_to_max() {
        // GLM-5's own template, reduced to the branch that matters:
        //
        //   {%- set effective = 'high' if reasoning_effort is defined
        //        and reasoning_effort == 'high' else 'max' -%}
        //
        // Leaving it unset is NOT neutral at this layer — it selects Max, the
        // most expensive setting, by omission. This is ground truth about the
        // template, not the server's behavior: `ChatCompletionRequest::
        // effective_reasoning_effort` maps omission (and everything but an
        // explicit high-ish request) to `"high"` before it ever reaches here.
        const T: &str = "{%- set effective = 'high' if reasoning_effort is defined and reasoning_effort == 'high' else 'max' -%}Reasoning Effort: {{ effective }}";
        let env = build_chat_env(T).unwrap();
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let unset = render_with_chat_env(&env, &msgs, "", "", true, None, None).unwrap();
        assert_eq!(unset, "Reasoning Effort: max", "omission selects max");

        let high = render_with_chat_env(&env, &msgs, "", "", true, Some("high"), None).unwrap();
        assert_eq!(high, "Reasoning Effort: high");

        // Only the literal "high" does anything; every other value is max, so a
        // typo silently costs maximum reasoning.
        let typo = render_with_chat_env(&env, &msgs, "", "", true, Some("High"), None).unwrap();
        assert_eq!(typo, "Reasoning Effort: max", "template is case-sensitive");
    }

    /// Deserializes a `ChatCompletionRequest` from a JSON fragment, filling in
    /// the required `model`/`messages` fields so callers only spell out the
    /// keys the test cares about.
    fn chat_request(extra: serde_json::Value) -> ChatCompletionRequest {
        let mut v = serde_json::json!({ "model": "x", "messages": [] });
        v.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn reasoning_effort_maps_openai_vocabulary_to_glm_levels() {
        // Omitted must resolve to "high", not undefined (Max) — the defect
        // this mapping exists to close.
        assert_eq!(
            chat_request(serde_json::json!({})).effective_reasoning_effort(),
            "high"
        );
        // "none" belongs in the conservative bucket, not with xhigh/max: it
        // signals *less* than minimal effort, and grouping it with the
        // expensive values would silently escalate whenever thinking stays on
        // despite "none" (see reasoning_effort_none_plus_explicit_thinking_stays_high).
        for cheap in ["minimal", "low", "medium", "high", "none"] {
            assert_eq!(
                chat_request(serde_json::json!({ "reasoning_effort": cheap }))
                    .effective_reasoning_effort(),
                "high",
                "{cheap} should map to high"
            );
        }
        // An explicit choice of xhigh/max is distinguishable from omission at
        // the Rust type level (a literal "max"), even though both currently
        // render the same template output — see effective_reasoning_effort's
        // doc comment.
        for expensive in ["xhigh", "max"] {
            assert_eq!(
                chat_request(serde_json::json!({ "reasoning_effort": expensive }))
                    .effective_reasoning_effort(),
                "max",
                "{expensive} should map to max"
            );
        }
        // Unrecognised: conservative default, never silently escalate to Max.
        assert_eq!(
            chat_request(serde_json::json!({ "reasoning_effort": "bogus" }))
                .effective_reasoning_effort(),
            "high"
        );
    }

    #[test]
    fn reasoning_effort_none_disables_thinking_unless_explicit() {
        // "none" with no explicit toggle: thinking goes off.
        assert!(
            !chat_request(serde_json::json!({ "reasoning_effort": "none" }))
                .effective_enable_thinking()
        );

        // "none" cannot override an explicit `enable_thinking: true` — the
        // direct instruction from a client that has the toggle wins.
        assert!(chat_request(serde_json::json!({
            "reasoning_effort": "none",
            "enable_thinking": true,
        }))
        .effective_enable_thinking());

        // Omitted reasoning_effort, omitted enable_thinking: still on.
        assert!(chat_request(serde_json::json!({})).effective_enable_thinking());

        // Explicit enable_thinking: false stands on its own, no reasoning_effort needed.
        assert!(
            !chat_request(serde_json::json!({ "enable_thinking": false }))
                .effective_enable_thinking()
        );
    }

    #[test]
    fn reasoning_effort_none_plus_explicit_thinking_stays_high_not_max() {
        // A client that asks for "none" but separately keeps thinking on via
        // an explicit toggle asked for less reasoning, not more — the template
        // value must not fall into the Max bucket just because "none" also
        // disables thinking in the common (no-override) case.
        let req = chat_request(serde_json::json!({
            "reasoning_effort": "none",
            "enable_thinking": true,
        }));
        assert!(req.effective_enable_thinking(), "explicit toggle wins");
        assert_eq!(
            req.effective_reasoning_effort(),
            "high",
            "\"none\" must not silently escalate to Max once thinking stays on"
        );
    }

    #[test]
    fn chat_template_kwargs_enable_thinking_outranks_everything() {
        // kwargs alone.
        assert!(!chat_request(serde_json::json!({
            "chat_template_kwargs": { "enable_thinking": false }
        }))
        .effective_enable_thinking());

        // kwargs beats the legacy top-level toggle.
        assert!(chat_request(serde_json::json!({
            "chat_template_kwargs": { "enable_thinking": true },
            "enable_thinking": false,
        }))
        .effective_enable_thinking());

        // kwargs beats "none" too — a client that used the vLLM/SGLang
        // toggle gets what it asked for, not the OpenAI-shaped off-switch.
        assert!(chat_request(serde_json::json!({
            "chat_template_kwargs": { "enable_thinking": true },
            "reasoning_effort": "none",
        }))
        .effective_enable_thinking());

        // Unrelated members must not error and must not affect resolution —
        // the map is free-form in the ecosystem.
        assert!(chat_request(serde_json::json!({
            "chat_template_kwargs": { "some_other_kwarg": "value" }
        }))
        .effective_enable_thinking());

        // Absent entirely: unchanged from before this field existed.
        assert!(chat_request(serde_json::json!({})).effective_enable_thinking());
    }

    #[test]
    fn tools_without_a_chat_template_are_refused() {
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let tools = vec![Tool {
            r#type: "function".into(),
            function: serde_json::json!({"name": "get_weather"}),
        }];
        let err = render_or_fallback(None, &msgs, "", "", true, None, Some(&tools))
            .expect_err("no template + tools must not be answered");
        assert!(matches!(err, PromptRenderError::Rejected(_)), "{err:?}");

        assert!(
            render_or_fallback(None, &msgs, "", "", true, None, None).is_ok(),
            "tool-less requests still render without a template"
        );
    }

    #[test]
    fn raise_exception_is_a_client_error_not_a_fallback() {
        // A template validating its own input (Gemma on a system message,
        // Mistral on non-alternating roles) is not an engine bug, and other
        // engines surface it. Swallowing it into the legacy formatter turns a
        // 400 into a plausible-looking 200.
        let env = build_chat_env(r#"{{ raise_exception("system role not supported") }}"#).unwrap();
        let msgs = [ChatMessage {
            role: "system".into(),
            content: "be nice".into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        match render_or_fallback(Some(&env), &msgs, "", "", true, None, None) {
            Err(PromptRenderError::Rejected(m)) => {
                assert!(m.contains("system role not supported"), "{m}")
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tools_without_template_return_400_not_200() {
        let app = make_app().await; // no chat template
        let payload = serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "stream": false
        });
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn tool_schema_key_order_is_alphabetical_not_client_order() {
        // Pins a known divergence rather than hiding it. `Tool.function` is a
        // `serde_json::Value`, whose map is a BTreeMap without the
        // `preserve_order` feature, so a client's `{name, description}` reaches
        // the prompt as `{description, name}`. transformers and vLLM preserve
        // client order. This test fails the day that feature is enabled, which
        // is the point — the goldens above must be regenerated with it.
        let tools = vec![Tool {
            r#type: "function".into(),
            function: serde_json::json!({"name": "b", "description": "a"}),
        }];
        let env = build_chat_env("{{ tools[0].function | tojson }}").unwrap();
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let out = render_with_chat_env(&env, &msgs, "", "", true, None, Some(&tools)).unwrap();
        assert_eq!(out, r#"{"description": "a", "name": "b"}"#);
    }

    #[test]
    fn strftime_now_returns_a_real_date_and_rejects_bad_formats() {
        // A stubbed empty string here reads as a model quirk downstream, not a
        // bug: Llama-3.2 templates put the result straight into the system
        // prompt.
        let env = build_chat_env(r#"{{ strftime_now("%Y") }}"#).unwrap();
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let out = render_with_chat_env(&env, &msgs, "", "", true, None, None).unwrap();
        assert_eq!(out.len(), 4, "expected a 4-digit year, got {out:?}");
        assert!(out.chars().all(|c| c.is_ascii_digit()), "{out:?}");

        let bad = build_chat_env(r#"{{ strftime_now("%Q") }}"#).unwrap();
        assert!(
            render_with_chat_env(&bad, &msgs, "", "", true, None, None).is_err(),
            "an unsupported strftime specifier must error, not render empty"
        );
    }

    #[test]
    fn renderer_forwards_assistant_tool_calls_and_tool_result() {
        let env = build_chat_env(TOOL_TEMPLATE).unwrap();
        let msgs = [
            ChatMessage {
                role: "assistant".into(),
                content: "".into(),
                name: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".into(),
                    r#type: "function".into(),
                    function: FunctionCall {
                        name: "get_weather".into(),
                        arguments: "{}".into(),
                    },
                }]),
                tool_call_id: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: "18C".into(),
                name: Some("get_weather".into()),
                tool_calls: None,
                tool_call_id: Some("call_1".into()),
            },
        ];
        let out = render_with_chat_env(&env, &msgs, "", "", true, None, None).unwrap();
        assert!(
            out.contains("[CALLS:get_weather]"),
            "assistant tool_calls not rendered: {out}"
        );
        assert!(
            out.contains("[TCID:call_1]"),
            "tool_call_id not rendered: {out}"
        );
    }

    #[test]
    fn build_choice_emits_tool_calls_when_parsed() {
        let (m, fr) = build_choice(
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>"
                .into(),
            &Some(serde_json::json!("auto")),
            true,
            true,
        );
        assert_eq!(fr, "tool_calls");
        assert!(m.content.is_none());
        assert_eq!(m.tool_calls.unwrap()[0].function.name, "get_weather");
    }

    #[test]
    fn build_choice_none_skips_parse() {
        let (m, fr) = build_choice(
            "<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>".into(),
            &Some(serde_json::json!("none")),
            true,
            true,
        );
        assert_eq!(fr, "stop");
        assert!(m.tool_calls.is_none());
        assert!(m.content.is_some());
    }

    #[test]
    fn build_choice_no_tools_skips_parse() {
        // tools absent -> never parse, even if output looks like a tool call.
        let (m, fr) = build_choice(
            "<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>".into(),
            &Some(serde_json::json!("auto")),
            false,
            true,
        );
        assert_eq!(fr, "stop");
        assert!(m.tool_calls.is_none());
        assert!(m.content.is_some());
    }

    #[test]
    fn build_choice_disarmed_still_parses_call_with_think_close_in_arguments() {
        // End-to-end (through build_choice) version of the regression test:
        // thinking off must not lose a real call whose arguments literally
        // contain "</think>".
        let (m, fr) = build_choice(
            r#"<tool_call>{"name":"echo","arguments":{"text":"</think>"}}</tool_call>"#.into(),
            &Some(serde_json::json!("auto")),
            true,
            false,
        );
        assert_eq!(fr, "tool_calls");
        assert_eq!(m.tool_calls.unwrap()[0].function.name, "echo");
    }

    #[test]
    fn build_choice_plain_is_stop() {
        let (m, fr) = build_choice("hello".into(), &None, true, true);
        assert_eq!(fr, "stop");
        assert!(m.tool_calls.is_none());
        assert_eq!(m.content.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn tool_request_returns_tool_calls_finish_e2e() {
        let app = make_tool_app().await;
        let tc =
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>";
        let payload = serde_json::json!({"model":"mock-model","messages":[{"role":"user","content":tc}],
            "tool_choice":"auto","tools":[{"type":"function","function":{"name":"get_weather"}}],"stream":false});
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 8192).await.unwrap()).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        assert!(v["choices"][0]["message"]["content"].is_null());
        assert_eq!(
            v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
    }

    #[tokio::test]
    async fn streaming_tool_request_emits_indexed_tool_delta() {
        let app = make_tool_app().await;
        let tc =
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>";
        let payload = serde_json::json!({"model":"mock-model","messages":[{"role":"user","content":tc}],
            "tool_choice":"auto","tools":[{"type":"function","function":{"name":"get_weather"}}],"stream":true});
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body =
            String::from_utf8(to_bytes(resp.into_body(), 65536).await.unwrap().to_vec()).unwrap();
        let (mut saw_tool, mut saw_finish) = (false, false);
        for line in body.lines().filter_map(|l| l.strip_prefix("data: ")) {
            if line.trim() == "[DONE]" {
                continue;
            }
            let v: Value = serde_json::from_str(line).unwrap();
            let d = &v["choices"][0]["delta"];
            if d.get("tool_calls").is_some() {
                saw_tool = true;
                assert_eq!(d["tool_calls"][0]["index"], 0);
                assert_eq!(d["tool_calls"][0]["function"]["name"], "get_weather");
                assert_eq!(d["role"], "assistant");
            }
            if v["choices"][0]["finish_reason"] == "tool_calls" {
                saw_finish = true;
            }
        }
        assert!(saw_tool, "no tool_calls delta: {body}");
        assert!(saw_finish, "no tool_calls finish: {body}");
    }

    #[tokio::test]
    async fn streaming_non_tool_is_per_token_with_stop() {
        let app = make_app().await;
        let payload = serde_json::json!({"model":"mock-model",
            "messages":[{"role":"user","content":"alpha bravo charlie"}],"stream":true});
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body =
            String::from_utf8(to_bytes(resp.into_body(), 65536).await.unwrap().to_vec()).unwrap();
        assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
        assert!(
            !body.contains("tool_calls"),
            "non-tool stream must not mention tool_calls: {body}"
        );
    }

    #[tokio::test]
    async fn streaming_tool_engine_error_emits_error_event() {
        let app = make_tool_app().await;
        let payload = serde_json::json!({"model":"mock-model",
            "messages":[{"role":"user","content":"__engine_error__"}],
            "tool_choice":"auto","tools":[{"type":"function","function":{"name":"x"}}],"stream":true});
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body =
            String::from_utf8(to_bytes(resp.into_body(), 65536).await.unwrap().to_vec()).unwrap();
        assert!(
            body.contains("\"object\":\"error\""),
            "expected SSE error event: {body}"
        );
        assert!(!body.contains("tool_calls"), "{body}");
    }

    // Helper: GET /metrics, return the exposition text.
    async fn scrape_metrics(app: Router) -> (StatusCode, String, String) {
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        (
            status,
            content_type,
            String::from_utf8(body.to_vec()).unwrap(),
        )
    }

    // #16 smoke test: real requests populate the request-, generation-, and
    // engine-level metrics, and /metrics serves them in text exposition
    // format. Uses a DEDICATED model id — the registry is process-global and
    // other tests in this binary run concurrently, so assertions must be
    // isolated by label, and on counters only assert presence (not exact
    // values shared with other tests' traffic).
    #[tokio::test]
    async fn metrics_endpoint_exposes_request_and_generation_metrics() {
        let runner = Runner::new(Box::new(MockBuilder::new()));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("metrics-probe-model", "CPU"),
            )
            .await
            .unwrap();
        let app = make_router(Arc::new(runner), "metrics-probe-model");

        // Before traffic: the endpoint serves and the label-less families
        // are already registered.
        let (status, content_type, text) = scrape_metrics(app.clone()).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/plain; version=0.0.4");
        assert!(text.contains("cascadia_inflight_tasks"), "{text}");

        // One real (mock-engine) chat completion.
        let (status, _) = post_chat(
            app.clone(),
            serde_json::json!({
                "model": "metrics-probe-model",
                "messages": [{"role": "user", "content": "alpha bravo charlie delta"}],
                "max_tokens": 2,
                "stream": false,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (_, _, text) = scrape_metrics(app).await;
        // Request-level: counted against the MATCHED route + status. (The
        // prometheus text encoder emits label pairs sorted by name.)
        assert!(
            text.contains(
                "cascadia_http_requests_total{endpoint=\"/v1/chat/completions\",status=\"200\"}"
            ),
            "{text}"
        );
        assert!(
            text.contains("cascadia_http_request_duration_seconds_count"),
            "{text}"
        );
        // Generation-level: tokens + TTFT + duration under the model label.
        assert!(
            text.contains("cascadia_tokens_generated_total{model=\"metrics-probe-model\"}"),
            "{text}"
        );
        assert!(
            text.contains("cascadia_generation_ttft_seconds_count{model=\"metrics-probe-model\"}"),
            "{text}"
        );
        assert!(
            text.contains(
                "cascadia_generation_duration_seconds_count{finish_reason=\"length\",model=\"metrics-probe-model\"}"
            ),
            "{text}"
        );
        // Engine-level: load + warmup gauges recorded by Runner::start.
        assert!(
            text.contains(
                "cascadia_engine_model_load_duration_seconds{device=\"CPU\",model=\"metrics-probe-model\"}"
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "cascadia_engine_warmup_duration_seconds{device=\"CPU\",model=\"metrics-probe-model\"}"
            ),
            "{text}"
        );
    }

    /// Engine whose queue is always full: submit() fails QueueFull, the
    /// same "at capacity" 503 the permit gate produces.
    struct QueueFullEngine;

    impl cascadia_engine::Engine for QueueFullEngine {
        fn warmup(&mut self) {}
        fn submit(
            &mut self,
            _task: cascadia_types::GenerationTask,
        ) -> Result<(), cascadia_engine::EngineError> {
            Err(cascadia_engine::EngineError::QueueFull { queued: 8, cap: 8 })
        }
        fn step(
            &mut self,
        ) -> Result<Vec<(cascadia_types::TaskId, Chunk)>, cascadia_engine::EngineError> {
            Ok(Vec::new())
        }
    }

    struct QueueFullBuilder;

    #[::async_trait::async_trait]
    impl cascadia_engine::Builder for QueueFullBuilder {
        async fn connect(
            &mut self,
            _peers: PeerLayout,
        ) -> Result<(), cascadia_engine::EngineError> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> Result<cascadia_engine::LoadStream, cascadia_engine::EngineError> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn build(
            self: Box<Self>,
        ) -> Result<Box<dyn cascadia_engine::Engine>, cascadia_engine::EngineError> {
            Ok(Box::new(QueueFullEngine))
        }
    }

    // An engine-queue-full 503 must count toward the same capacity-rejection
    // reason as the permit-gate 503 — engines cap pending tasks below the
    // default permit count, so this path is reachable at default config.
    #[tokio::test]
    async fn engine_queue_full_counts_capacity_rejection() {
        let capacity_before = cascadia_metrics::API_REJECTED_TOTAL
            .with_label_values(&["capacity"])
            .get();
        let runner = Runner::new(Box::new(QueueFullBuilder));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("queuefull-model", "CPU"),
            )
            .await
            .unwrap();
        let app = make_router(Arc::new(runner), "queuefull-model");
        let (status, _) = post_chat(
            app,
            serde_json::json!({
                "model": "queuefull-model",
                "messages": [{"role": "user", "content": "hi"}],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let capacity_after = cascadia_metrics::API_REJECTED_TOTAL
            .with_label_values(&["capacity"])
            .get();
        assert!(
            capacity_after > capacity_before,
            "QueueFull 503 must count as a capacity rejection ({capacity_before} -> {capacity_after})"
        );
    }

    /// Engine that emits a token every step and NEVER finishes. Only
    /// `Runner::close()` can end a generation from it, which is what makes
    /// the teardown test deterministic: the no-progress wedge guard trips
    /// after 3 consecutive EMPTY steps, so an engine that stalls instead
    /// would race that guard rather than the shutdown.
    struct NeverEndingEngine {
        task: Option<cascadia_types::TaskId>,
        steps: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl cascadia_engine::Engine for NeverEndingEngine {
        fn warmup(&mut self) {}
        fn submit(
            &mut self,
            task: cascadia_types::GenerationTask,
        ) -> Result<(), cascadia_engine::EngineError> {
            self.task = Some(task.task_id);
            Ok(())
        }
        fn step(
            &mut self,
        ) -> Result<Vec<(cascadia_types::TaskId, Chunk)>, cascadia_engine::EngineError> {
            self.steps.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(match &self.task {
                Some(id) => vec![(id.clone(), Chunk::token(id, 0, "tok "))],
                None => Vec::new(),
            })
        }
    }

    struct NeverEndingBuilder(Arc<std::sync::atomic::AtomicUsize>);

    #[::async_trait::async_trait]
    impl cascadia_engine::Builder for NeverEndingBuilder {
        async fn connect(
            &mut self,
            _peers: PeerLayout,
        ) -> Result<(), cascadia_engine::EngineError> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: cascadia_types::ShardSpec,
        ) -> Result<cascadia_engine::LoadStream, cascadia_engine::EngineError> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn build(
            self: Box<Self>,
        ) -> Result<Box<dyn cascadia_engine::Engine>, cascadia_engine::EngineError> {
            Ok(Box::new(NeverEndingEngine {
                task: None,
                steps: self.0.clone(),
            }))
        }
    }

    /// Server teardown mid-generation must reach the CLIENT as a failure.
    ///
    /// The non-streaming handler seeds `finish_reason = "stop"` and drains
    /// with `while let Some(chunk)`, so a bare end-of-stream builds a 200
    /// carrying the partial text — telling the caller the model finished
    /// normally while the server was going away. Anything persisting
    /// completions then records truncated output as complete.
    ///
    /// The runner unit tests assert the metric on both poll/drop orderings;
    /// this one covers what those cannot — the HTTP surface, on the
    /// NON-streaming path (the local SSE probe covered streaming only).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn teardown_mid_generation_fails_the_request_and_books_teardown() {
        const MODEL: &str = "api-teardown-model";
        let steps = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runner = Arc::new(Runner::new(Box::new(NeverEndingBuilder(steps.clone()))));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage(MODEL, "CPU"),
            )
            .await
            .unwrap();
        let app = make_router(runner.clone(), MODEL);

        let req = tokio::spawn(post_chat(
            app,
            serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}],
            }),
        ));

        // Wait until the generation is genuinely under way before shutting
        // down — otherwise the close could land before admission and the
        // request would fail as NotLoaded, which is a different path.
        while steps.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        runner.close();

        let (status, body) = req.await.unwrap();
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "teardown must not return 200: {body}"
        );
        assert!(
            body["error"]
                .as_str()
                .unwrap_or_default()
                .contains("shutting down"),
            "expected a shutdown error, got {body}"
        );

        let (_, buf) = cascadia_metrics::encode_text();
        let text = String::from_utf8(buf).unwrap();
        let needle = format!(
            "cascadia_generation_duration_seconds_count{{finish_reason=\"teardown\",model=\"{MODEL}\"}} 1"
        );
        assert!(text.contains(&needle), "missing {needle} in:\n{text}");
        // A restart is neither a client cancellation nor an engine fault.
        for reason in ["cancelled", "error"] {
            assert_eq!(
                cascadia_metrics::GENERATION_DURATION_SECONDS
                    .with_label_values(&[MODEL, reason])
                    .get_sample_count(),
                0,
                "teardown must not be booked as {reason}"
            );
        }
    }

    // Rejections that never reach the engine are counted by reason.
    #[tokio::test]
    async fn metrics_count_pre_engine_rejections() {
        let app = make_app().await;
        let (status, _) = post_chat(
            app.clone(),
            serde_json::json!({ "model": "mock-model", "messages": [] }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (_, _, text) = scrape_metrics(app).await;
        assert!(
            text.contains("cascadia_api_rejected_total{reason=\"empty_prompt\"}"),
            "{text}"
        );
    }
}
