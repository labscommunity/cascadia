//! OpenAI-compatible HTTP server.
//!
//! Mirrors the subset of `cascadia/api/server.py` needed for parity with PR
//! #2's e2e bench: `/health`, `/v1/models`, `/v1/chat/completions`
//! (non-streaming + SSE streaming). Tools, logprobs, /events, /state,
//! Ollama dialect deferred — see Phase 5 follow-up.

use std::sync::atomic::{AtomicU64, Ordering};
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

/// Live request/token counters bumped on the chat hot path. Shared
/// (via `Arc`) with the dashboard's `/api/stats` so the cluster view
/// reflects activity in real time. All `Relaxed` — these are coarse
/// monotonic gauges, not synchronization points.
#[derive(Default)]
pub struct ApiStats {
    /// Cumulative chat-completion requests admitted (past the permit gate).
    pub requests_total: AtomicU64,
    /// Requests currently executing (incremented on admit, decremented when
    /// the response/stream is fully drained or dropped).
    pub requests_in_flight: AtomicU64,
    /// Cumulative model tokens emitted across all requests.
    pub tokens_total: AtomicU64,
}

/// RAII guard: bumps `requests_in_flight` on construction and decrements it
/// on drop, so the gauge is correct even on early return, client disconnect,
/// or a mid-stream engine error.
struct InFlightGuard(Arc<ApiStats>);

impl InFlightGuard {
    fn new(stats: Arc<ApiStats>) -> Self {
        stats.requests_in_flight.fetch_add(1, Ordering::Relaxed);
        InFlightGuard(stats)
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.requests_in_flight.fetch_sub(1, Ordering::Relaxed);
    }
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
                Ok(env) => Some(Arc::new(env)),
                Err(e) => {
                    warn!(error = %e, "chat_template failed to parse at startup; using legacy formatter");
                    None
                }
            }),
        bos_token: Arc::from(cfg.chat_template.bos_token.unwrap_or_default()),
        eos_token: Arc::from(cfg.chat_template.eos_token.unwrap_or_default()),
        defer_template_on_thinking: cfg.defer_template_on_thinking,
    };
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/cancel/:task_id", post(cancel))
        .with_state(state)
        // Cap the JSON body so an attacker can't OOM the server with
        // a multi-GB request. Apply at router level so it applies to
        // every route, not just chat_completions.
        .layer(DefaultBodyLimit::max(cfg.max_body_bytes))
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
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

#[derive(Deserialize, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
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
    pub stream: bool,
    /// Hybrid-reasoning switch (Qwen3+ convention). Default true =
    /// model-default behavior; false asks the engine to skip the
    /// <think> block (engines that can't, ignore it).
    #[serde(default = "default_true")]
    pub enable_thinking: bool,
}

fn default_true() -> bool {
    true
}

fn default_max_tokens() -> u32 {
    256
}

#[derive(Serialize)]
struct ChatChoiceMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatChoiceMessage,
    finish_reason: &'static str,
    logprobs: Option<()>,
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
fn render_prompt_legacy(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

/// Build the minijinja environment for a model's HF chat template ONCE, with
/// the template pre-parsed and the HF-compat shims installed. The template is
/// fixed at model load, so this is built at router construction and reused
/// across every request — the ~17 KB Gemma 3/4 macro template is parsed once,
/// not on each `/v1/chat/completions` call. Returns Err on parse failure so the
/// caller can fall back to the legacy formatter.
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
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });
    // Some templates (Llama 3) call strftime_now() to embed the current
    // date in the system prompt. We don't actually need a real date for
    // inference correctness — a fixed empty string is a safe stand-in.
    env.add_function("strftime_now", |_fmt: String| -> String { String::new() });

    // add_template_owned (not add_template) so the Environment owns the source
    // and is 'static — it can then live in AppState behind an Arc.
    env.add_template_owned("chat", template_src.to_owned())
        .map_err(|e| format!("template parse: {e}"))?;
    Ok(env)
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
) -> Result<String, String> {
    use minijinja::context;
    use minijinja::value::Value;

    let tmpl = env
        .get_template("chat")
        .map_err(|e| format!("template lookup: {e}"))?;

    let messages_value: Vec<Value> = messages
        .iter()
        .map(|m| {
            Value::from_serialize(serde_json::json!({
                "role": m.role,
                "content": m.content,
            }))
        })
        .collect();

    let ctx = context! {
        messages => messages_value,
        add_generation_prompt => true,
        // Hybrid-reasoning templates (Qwen3.x) read this and inject the
        // empty think block themselves when false.
        enable_thinking => enable_thinking,
        bos_token => bos_token,
        eos_token => eos_token,
    };

    tmpl.render(ctx)
        .map_err(|e| format!("template render: {e}"))
}

fn render_prompt(state: &AppState, messages: &[ChatMessage], enable_thinking: bool) -> String {
    // Hybrid: for engines that own native templating (ov-genai), only render
    // here when thinking is OFF (to inject the empty-think block). With thinking
    // ON, emit the legacy join so the engine applies its own template natively —
    // leaving the already-working path byte-identical.
    let defer_to_engine = state.defer_template_on_thinking && enable_thinking;
    if !defer_to_engine {
        if let Some(env) = &state.chat_env {
            match render_with_chat_env(
                env,
                messages,
                &state.bos_token,
                &state.eos_token,
                enable_thinking,
            ) {
                Ok(s) => return s,
                Err(e) => {
                    warn!(error = %e, "chat_template render failed; falling back to legacy formatter");
                }
            }
        }
    }
    render_prompt_legacy(messages)
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
                Ok(env) => Some(Arc::new(env)),
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

    /// Render `messages`, falling back to [`render_prompt_legacy`] when no
    /// template is set or rendering fails.
    pub fn render(&self, messages: &[ChatMessage]) -> String {
        if let Some(env) = &self.env {
            match render_with_chat_env(env, messages, &self.bos_token, &self.eos_token, true) {
                Ok(s) => return s,
                Err(e) => {
                    warn!(error = %e, "chat_template render failed; falling back to legacy formatter");
                }
            }
        }
        render_prompt_legacy(messages)
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let task_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let prompt = render_prompt(&state, &req.messages, req.enable_thinking);
    // Degenerate input (no messages, or a render that collapses to nothing)
    // is a client error. Reject here with 400 rather than admitting an empty
    // prompt to the engine, which would generate nothing and return a 200
    // with empty content — indistinguishable from a real failure.
    if prompt.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "no prompt content: `messages` is empty or rendered to an empty prompt"
            })),
        )
            .into_response();
    }
    if prompt.len() > state.max_prompt_bytes {
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
        logprobs: 0,
        enable_thinking: req.enable_thinking,
        trust_remote_code: false,
    };

    // Acquire a request slot before touching the engine. Without this
    // a flood of concurrent SSE callers would hammer one engine mutex
    // and starve everyone (the `MAX_CONSECUTIVE_EMPTY_STEPS=3` guard
    // in the runner would then truncate streams). Backpressure is
    // 503; clients should retry with backoff.
    let permit = match state.permits.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
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
        return stream_completion(state, req.model, task, permit, inflight)
            .await
            .into_response();
    }

    // Non-streaming: collect full output. Hold the permit + in-flight guard
    // until the task completes; drop frees the slot.
    let _permit = permit;
    let _inflight = inflight;
    let mut chunk_stream = match state.runner.generate(task.clone()) {
        Ok(s) => s,
        Err(err) => return engine_error_response(err),
    };
    let mut buf = String::new();
    let mut completion_tokens: u32 = 0;
    let mut prompt_tokens: u32 = 0;
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
        if !chunk.is_final {
            buf.push_str(&chunk.text);
            completion_tokens += chunk.n_tokens.unwrap_or(1);
        } else {
            if !chunk.text.is_empty() {
                buf.push_str(&chunk.text);
            }
            prompt_tokens = chunk.prompt_tokens.unwrap_or(0);
        }
    }
    state
        .stats
        .tokens_total
        .fetch_add(completion_tokens as u64, Ordering::Relaxed);

    Json(ChatCompletionResponse {
        id: task_id,
        object: "chat.completion",
        created: now_unix(),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatChoiceMessage {
                role: "assistant",
                content: buf,
            },
            finish_reason: "stop",
            logprobs: None,
        }],
        usage: Usage {
            // From the engine's final chunk; 0 when the engine can't tell
            // (e.g. ov-genai tokenizes inside the pipeline).
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response()
}

async fn stream_completion(
    state: AppState,
    model: String,
    task: GenerationTask,
    permit: tokio::sync::OwnedSemaphorePermit,
    inflight: InFlightGuard,
) -> axum::response::Response {
    let task_id = task.task_id.clone();
    let _ = SystemTime::now();

    let chunk_stream = match state.runner.generate(task) {
        Ok(s) => s,
        Err(err) => {
            warn!(error = %err, "ov-stream: generate failed");
            return engine_error_response(err);
        }
    };
    // Clone the counter handle for per-chunk token accounting in the
    // formatter closure below.
    let stats = state.stats.clone();
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
            async move {
                // Count model tokens as they stream so the dashboard's
                // tokens_total advances live (not just at request end).
                if chunk.error.is_none() && !chunk.is_final {
                    stats
                        .tokens_total
                        .fetch_add(chunk.n_tokens.unwrap_or(1) as u64, Ordering::Relaxed);
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
                        "finish_reason": if chunk.is_final { Some("stop") } else { None },
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
        .chain(stream::once(async {
            Ok::<Bytes, std::convert::Infallible>(Bytes::from_static(b"data: [DONE]\n\n"))
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
    let status = match &err {
        EngineError::QueueFull { .. } => StatusCode::SERVICE_UNAVAILABLE,
        EngineError::NotLoaded | EngineError::NotConnected => StatusCode::SERVICE_UNAVAILABLE,
        EngineError::InvalidConfig(_)
        | EngineError::PeerRejected(_)
        | EngineError::ShardRejected(_) => StatusCode::BAD_REQUEST,
        EngineError::ModelNotFound(_) => StatusCode::NOT_FOUND,
        EngineError::Backend(_) | EngineError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
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
    use cascadia_types::{PeerLayout, ShardSpec};
    use serde_json::Value;
    use tower::ServiceExt;

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
        }];
        let env = build_chat_env(MACRO_TEMPLATE).expect("macro-based template must parse");
        let out = render_with_chat_env(&env, &msgs, "<bos>", "<eos>", true)
            .expect("macro-based template must render");
        assert!(out.contains("<bos>"), "out={out}");
        assert!(out.contains("<start_of_turn>user"), "out={out}");
        assert!(out.contains("hi"), "out={out}");
        assert!(out.contains("<start_of_turn>model"), "out={out}");
    }

    #[test]
    fn chat_prompt_renderer_renders_then_falls_back() {
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];

        // Template present: renders through the parsed env (parsed once in `new`).
        let r = ChatPromptRenderer::new(&ChatTemplateConfig {
            template: Some(MACRO_TEMPLATE.into()),
            bos_token: Some("<bos>".into()),
            eos_token: Some("<eos>".into()),
        });
        let out = r.render(&msgs);
        assert!(out.contains("<bos>"), "out={out}");
        assert!(out.contains("<start_of_turn>user"), "out={out}");

        // No template: falls back to the legacy formatter.
        let none = ChatPromptRenderer::new(&ChatTemplateConfig::default());
        assert_eq!(none.render(&msgs), render_prompt_legacy(&msgs));

        // Unparseable template is treated as absent → legacy fallback, not a panic.
        let broken = ChatPromptRenderer::new(&ChatTemplateConfig {
            template: Some("{% for %}".into()),
            bos_token: None,
            eos_token: None,
        });
        assert_eq!(broken.render(&msgs), render_prompt_legacy(&msgs));
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
}
