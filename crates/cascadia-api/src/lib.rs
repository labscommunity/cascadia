//! OpenAI-compatible HTTP server.
//!
//! Mirrors the subset of `cascadia/api/server.py` needed for parity with PR
//! #2's e2e bench: `/health`, `/v1/models`, `/v1/chat/completions`
//! (non-streaming + SSE streaming). Tools, logprobs, /events, /state,
//! Ollama dialect deferred — see Phase 5 follow-up.

mod generation_info;
#[cfg(test)]
mod spec_schema;
pub use generation_info::{read_model_generation_info, ModelGenerationInfo};

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use cascadia_runner::Runner;
use cascadia_types::{ChatTurn, GenerationTask};
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

/// Reconcile per-request and global streaming counters together. A final
/// authoritative count replaces estimates, including when it is lower.
fn record_stream_tokens(
    chunk: &cascadia_types::Chunk,
    stats: &ApiStats,
    completion: &std::sync::atomic::AtomicU32,
) {
    let before = completion.load(Ordering::Relaxed);
    let mut after = before;
    chunk.accumulate_tokens(&mut after);
    completion.store(after, Ordering::Relaxed);
    if after >= before {
        stats
            .tokens_total
            .fetch_add(u64::from(after - before), Ordering::Relaxed);
    } else {
        stats
            .tokens_total
            .fetch_sub(u64::from(before - after), Ordering::Relaxed);
    }
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
    /// The engine applies the model's chat template itself (sycl-llama: the
    /// llama-server child renders the GGUF template on the structured turns
    /// the task carries). Render here is a placeholder so the empty-prompt
    /// guard stays honest; tools flow through GenerationTask, so the
    /// no-template refusal does not apply.
    pub engine_applies_template: bool,
    /// The chat template distinguishes OpenAI's `reasoning_effort` words
    /// (probed at load: "low" and "medium" render differently). When true the
    /// caller's own word reaches the template; when false the GLM high/max
    /// mapping applies. See [`EffortArgs`].
    pub effort_words: bool,
    /// Special-token message framing to translate back into text delimiters
    /// (Inkling). See [`MarkerDialect`].
    pub marker_dialect: Option<MarkerDialect>,
    /// Pipeline readiness for `/health`. Starts `true`; a completion that fails
    /// with a 5xx (e.g. the distributed pipeline dropped a peer link) flips it
    /// `false`, a success flips it back `true` — so `/health` reflects the real
    /// state of the chain instead of always returning 200 while requests 503.
    pub ready: Arc<std::sync::atomic::AtomicBool>,
    /// Values for the sampling fields a request leaves unset. See
    /// [`GenerationDefaults`].
    pub generation_defaults: GenerationDefaults,
    /// Idle time after which a stream sends an SSE keep-alive comment. See
    /// [`SSE_KEEPALIVE_INTERVAL`].
    pub sse_keepalive: Duration,
}

/// How long a stream may go without sending anything before it sends the SSE
/// comment `: keep-alive` (M1 D11). One-shot engines produce no bytes until
/// their whole answer is done; proxies with a 60 s idle timeout would cut
/// such a stream. A comment, not a `data:` frame: OpenAI clients skip it.
pub const SSE_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(15);

/// The SSE keep-alive comment line (see [`SSE_KEEPALIVE_INTERVAL`]).
const SSE_KEEPALIVE_COMMENT: &str = ": keep-alive\n\n";

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
    /// See [`AppState::engine_applies_template`]. Set by the CLI for sycl-llama.
    pub engine_applies_template: bool,
    /// See [`MarkerDialect`]; the CLI loads it with [`MarkerDialect::load`].
    pub marker_dialect: Option<MarkerDialect>,
    /// What a request that leaves `max_tokens`/`temperature`/`top_p` unset
    /// gets. See [`GenerationDefaults`]; the CLI resolves it per model.
    pub generation_defaults: GenerationDefaults,
    /// See [`AppState::sse_keepalive`]; [`SSE_KEEPALIVE_INTERVAL`] by default.
    pub sse_keepalive: Duration,
}

/// Output-token limit for a request that sets none, when the model's context
/// window is unknown (the repo's default context: sycl-llama `--llama-ctx`,
/// the sparse-moe staged max sequence).
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Server-wide values for the sampling fields a request leaves unset (or
/// sends as `null`). Decision 2: no limit means "until the model stops", sent
/// to the engines as one context window (`max_tokens`); `temperature`/`top_p`
/// come from the model's `generation_config.json`, else the OpenAI default 1.
/// The CLI resolves these per model (`--default-max-tokens`,
/// `--default-temperature`); the defaults here are the no-model fallback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationDefaults {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    /// The model's context window when a real source gives it (the engine's
    /// own limit or `config.json`), never the [`DEFAULT_MAX_TOKENS`] fallback
    /// or `--default-max-tokens`. An explicit request limit above it is sent
    /// as the window (M1 D13), so the answer ends with `length` there.
    pub context_window: Option<u32>,
}

impl Default for GenerationDefaults {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKENS,
            temperature: 1.0,
            top_p: 1.0,
            context_window: None,
        }
    }
}

impl GenerationDefaults {
    /// The limit sent to the engine for a request that asked for
    /// `requested`: the server default when unset, else the request's own
    /// limit, at most the known context window.
    pub fn max_tokens_for(&self, requested: Option<u32>) -> u32 {
        match (requested, self.context_window) {
            (None, _) => self.max_tokens,
            (Some(n), Some(window)) => n.min(window),
            (Some(n), None) => n,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT,
            max_prompt_bytes: DEFAULT_MAX_PROMPT_BYTES,
            chat_template: ChatTemplateConfig::default(),
            defer_template_on_thinking: false,
            engine_applies_template: false,
            marker_dialect: None,
            generation_defaults: GenerationDefaults::default(),
            sse_keepalive: SSE_KEEPALIVE_INTERVAL,
        }
    }
}

/// Message framing made of SPECIAL tokens (Inkling): the model emits
/// `<|message_model|><|content_thinking|>…<|end_message|>` for its scratchpad,
/// `<|message_model|><|content_text|>…<|end_message|>` for the answer and
/// `<|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":…}<|end_message|>`
/// for a tool call. Engines decode with special tokens stripped, so that
/// framing vanishes from the text stream and the reasoning is glued to the
/// answer. The API reads the marker ids off each chunk instead and re-inserts
/// the textual delimiters every other served template uses (`<think>…</think>`,
/// Hermes `<tool_call>{"name","arguments"}</tool_call>`), so the response side
/// keeps a single dialect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarkerDialect {
    pub message_model: i64,
    pub content_text: i64,
    pub content_thinking: i64,
    pub end_message: i64,
    pub invoke_tool_json: i64,
}

impl MarkerDialect {
    /// Read the marker ids from `<tok_dir>/tokenizer_config.json`
    /// (`added_tokens_decoder`) when `template` frames its output with
    /// `<|content_thinking|>`; `None` for every other template.
    pub fn load(tok_dir: &std::path::Path, template: Option<&str>) -> Option<Self> {
        if !template?.contains("<|content_thinking|>") {
            // This template does not frame its output with special tokens;
            // there is nothing to translate. Silent None is correct here.
            return None;
        }
        // From here the template DOES frame with markers, so failing to resolve
        // the ids is a broken export, not an ordinary non-marker template. Warn
        // instead of silently returning None (which strips the markers from the
        // stream and glues the model's reasoning into the visible answer).
        let cfg = tok_dir.join("tokenizer_config.json");
        let v = match std::fs::read(&cfg)
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        {
            Some(v) => v,
            None => {
                warn!(
                    path = %cfg.display(),
                    "chat template frames output with <|content_thinking|> but tokenizer_config.json could not be read/parsed; marker translation disabled (reasoning may be merged into the answer)"
                );
                return None;
            }
        };
        let table = match v.get("added_tokens_decoder").and_then(|t| t.as_object()) {
            Some(t) => t,
            None => {
                warn!(
                    "marker-framed template but tokenizer_config.json has no added_tokens_decoder object; marker translation disabled (reasoning may be merged into the answer)"
                );
                return None;
            }
        };
        let id_of = |name: &str| -> Option<i64> {
            table.iter().find_map(|(id, tok)| {
                (tok.get("content")?.as_str()? == name).then(|| id.parse::<i64>().ok())?
            })
        };
        let need = |name: &str| -> Option<i64> {
            let id = id_of(name);
            if id.is_none() {
                warn!(
                    token = name,
                    "marker-framed template but this marker token id is missing/unparseable in added_tokens_decoder; marker translation disabled (reasoning may be merged into the answer)"
                );
            }
            id
        };
        let d = Self {
            message_model: need("<|message_model|>")?,
            content_text: need("<|content_text|>")?,
            content_thinking: need("<|content_thinking|>")?,
            end_message: need("<|end_message|>")?,
            invoke_tool_json: need("<|content_invoke_tool_json|>")?,
        };
        info!(
            ?d,
            "chat template frames output with special tokens; translating markers"
        );
        Some(d)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MarkerMode {
    Idle,
    /// After `<|message_model|>`: text until the content-type marker (a tool
    /// call's name lives here).
    Header,
    Thinking,
    Text,
    /// After `<|content_invoke_tool_json|>`: the call's JSON until `<|end_message|>`.
    Tool,
}

/// Per-response translator for a [`MarkerDialect`]: feed every engine chunk,
/// take the text the response should carry. Text arriving in the `Header` and
/// `Tool` states is withheld until the marker that types it, so a streamed
/// tool call surfaces as one `<tool_call>` block.
pub struct MarkerFilter {
    d: MarkerDialect,
    mode: MarkerMode,
    header: String,
    tool_json: String,
}

impl MarkerFilter {
    pub fn new(d: MarkerDialect) -> Self {
        Self {
            d,
            mode: MarkerMode::Idle,
            header: String::new(),
            tool_json: String::new(),
        }
    }

    fn on_marker(&mut self, id: i64, out: &mut String) -> bool {
        let d = &self.d;
        if id == d.message_model {
            self.mode = MarkerMode::Header;
            self.header.clear();
        } else if id == d.content_thinking {
            out.push_str("<think>");
            self.mode = MarkerMode::Thinking;
        } else if id == d.content_text {
            // Header text before a text body is not a name; keep it.
            out.push_str(&std::mem::take(&mut self.header));
            self.mode = MarkerMode::Text;
        } else if id == d.invoke_tool_json {
            self.mode = MarkerMode::Tool;
            self.tool_json.clear();
        } else if id == d.end_message {
            match self.mode {
                MarkerMode::Thinking => out.push_str("</think>"),
                MarkerMode::Tool => out.push_str(&self.tool_call_block()),
                MarkerMode::Header => out.push_str(&std::mem::take(&mut self.header)),
                _ => {}
            }
            self.mode = MarkerMode::Idle;
        } else {
            return false;
        }
        true
    }

    /// `NAME<|content_invoke_tool_json|>{"name":…,"args":…}` → the Hermes
    /// block `parse_tool_calls` already understands.
    fn tool_call_block(&mut self) -> String {
        let name_hdr = std::mem::take(&mut self.header);
        let raw = std::mem::take(&mut self.tool_json);
        let name_hdr = name_hdr.trim();
        let body = match serde_json::from_str::<serde_json::Value>(raw.trim()) {
            Ok(v) => {
                let name = v
                    .get("name")
                    .and_then(|n| n.as_str())
                    .map(str::to_owned)
                    .unwrap_or_else(|| name_hdr.to_owned());
                let args = v
                    .get("args")
                    .or_else(|| v.get("arguments"))
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                serde_json::json!({ "name": name, "arguments": args }).to_string()
            }
            // Unparseable: pass it through inside the block; parse_tool_calls
            // skips a malformed block rather than failing the response.
            Err(_) => raw,
        };
        format!("<tool_call>{body}</tool_call>")
    }

    /// Translate one chunk. Marker ids (empty text after special-token
    /// stripping) drive the state; other text is routed by the current state.
    /// A chunk carrying several ids (spec-decode) routes its text by the state
    /// after all of them.
    pub fn feed(&mut self, chunk: &cascadia_types::Chunk) -> String {
        let mut out = String::new();
        let ids: Vec<i64> = if chunk.token_ids.is_empty() {
            vec![chunk.token_id]
        } else {
            chunk.token_ids.clone()
        };
        let mut all_markers = !ids.is_empty();
        for id in ids {
            if !self.on_marker(id, &mut out) {
                all_markers = false;
            }
        }
        if all_markers && chunk.text.is_empty() {
            return out;
        }
        match self.mode {
            MarkerMode::Header => self.header.push_str(&chunk.text),
            MarkerMode::Tool => self.tool_json.push_str(&chunk.text),
            _ => out.push_str(&chunk.text),
        }
        out
    }

    /// Flush anything still withheld (the stream ended mid-header / mid-call).
    pub fn finish(&mut self) -> String {
        let mut out = std::mem::take(&mut self.header);
        if self.mode == MarkerMode::Tool && !self.tool_json.is_empty() {
            out.push_str(&self.tool_call_block());
        } else if self.mode == MarkerMode::Thinking {
            out.push_str("</think>");
        }
        self.mode = MarkerMode::Idle;
        out
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
        bos_token: Arc::from(cfg.chat_template.bos_token.clone().unwrap_or_default()),
        eos_token: Arc::from(cfg.chat_template.eos_token.clone().unwrap_or_default()),
        defer_template_on_thinking: cfg.defer_template_on_thinking,
        engine_applies_template: cfg.engine_applies_template,
        effort_words: false,
        marker_dialect: cfg.marker_dialect.clone(),
        ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        generation_defaults: cfg.generation_defaults,
        sse_keepalive: cfg.sse_keepalive,
    };
    let state = AppState {
        effort_words: state
            .chat_env
            .as_deref()
            .map(|env| template_speaks_effort_words(env, &state.bos_token, &state.eos_token))
            .unwrap_or(false),
        ..state
    };
    if state.effort_words {
        info!(
            "chat_template distinguishes OpenAI reasoning_effort words; passing the caller's own"
        );
    }
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

/// Marks a 5xx raised before the request ever reached the engine, so
/// [`track_pipeline_health`] does not read it as the pipeline failing.
///
/// The readiness flip is a latch with an asymmetry: a 5xx that keeps serving
/// other traffic (a capacity 503) heals on the next 2xx, but one that a load
/// balancer gates on `/health` stops the traffic that would have healed it.
/// A chat-template render failure is request-shape-determined and never
/// touches the pipeline, so letting it latch drains a healthy node
/// permanently on a single request.
#[derive(Clone, Copy)]
struct PreEngineFailure;

/// Response middleware: for completion routes, flip `AppState.ready` from the
/// response status (5xx → not ready, 2xx → ready) so `/health` reflects whether
/// the pipeline is actually serving. Non-completion routes don't touch it.
/// 5xx responses tagged [`PreEngineFailure`] are excluded — they say nothing
/// about the pipeline.
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
        if s.is_server_error() && resp.extensions().get::<PreEngineFailure>().is_none() {
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
    #[serde(default, deserialize_with = "de_null_default")]
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

/// `null` -> the type's default, for fields the spec marks nullable: message
/// `content` (assistant tool-call turns send `null`), `stream`, `logprobs`,
/// `echo`. Pair with `#[serde(default)]` so an absent field is the same.
fn de_null_default<'de, D, T>(d: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(d)?.unwrap_or_default())
}

/// `seed`: a signed or an unsigned 64-bit integer (or `null`), as the
/// engine's `u64`; negative values wrap (`-1` → `u64::MAX`), a stable
/// mapping so the same seed always gives the same stream.
fn de_seed<'de, D>(d: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum AnySeed {
        Unsigned(u64),
        Signed(i64),
    }
    Ok(Option::<AnySeed>::deserialize(d)?.map(|s| match s {
        AnySeed::Unsigned(u) => u,
        AnySeed::Signed(i) => i as u64,
    }))
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
    /// `null` = unset = false.
    #[serde(default, deserialize_with = "de_null_default")]
    pub include_usage: bool,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletionRequest {
    pub model: String,
    /// Legacy `function_call` / role `function` turns arrive already mapped
    /// onto tool calls (see [`de_chat_messages`]).
    #[serde(deserialize_with = "de_chat_messages")]
    pub messages: Vec<ChatMessage>,
    /// Deprecated in the OpenAI spec in favour of `max_completion_tokens`;
    /// still honoured when that field is absent. `null` = unset.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// OpenAI's current output-token limit; wins over `max_tokens` when both
    /// are sent. `null` = unset.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    /// `null` = unset = the server default ([`GenerationDefaults`]).
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Any integer from `i64::MIN` to `u64::MAX` (the spec's `seed` is a
    /// plain integer); a negative one reaches the engine as its
    /// two's-complement `u64` (`-1` → `u64::MAX`). See [`de_seed`].
    #[serde(default, deserialize_with = "de_seed")]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopSpec>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// OpenAI `logprobs` (bool). When true, per-token logprobs are returned
    /// (engine-permitting). `null` = unset.
    #[serde(default, deserialize_with = "de_null_default")]
    pub logprobs: bool,
    /// OpenAI `top_logprobs` (0..=20): how many alternatives per position.
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    /// `null` = unset.
    #[serde(default, deserialize_with = "de_null_default")]
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
    /// Deprecated `functions` (bare function definitions): moved onto `tools`
    /// by [`ChatCompletionRequest::apply_legacy_functions`].
    #[serde(default)]
    pub functions: Option<Vec<serde_json::Value>>,
    /// Deprecated `function_call` (`"none"`, `"auto"` or `{"name": N}`):
    /// moved onto `tool_choice` the same way.
    #[serde(default)]
    pub function_call: Option<serde_json::Value>,
    /// The request used `functions`, so the answer takes the legacy shape:
    /// `function_call` instead of `tool_calls` (M1 D5).
    #[serde(skip)]
    legacy_functions: bool,
}

/// A request message as sent: today's [`ChatMessage`] fields plus the
/// deprecated assistant `function_call`. `ChatMessage` itself stays
/// unchanged because callers outside this crate build it as a struct literal.
/// The fields are spelled out (no `#[serde(flatten)]`) so a 422 names the
/// field's full path, e.g. `messages[0].content`.
#[derive(Deserialize)]
struct WireChatMessage {
    role: String,
    #[serde(default, deserialize_with = "de_null_default")]
    content: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    function_call: Option<FunctionCall>,
}

/// Prefix of the ids synthesised for legacy calls: no client sends it, so a
/// synthesised id never collides with a client's own `call_0`.
const LEGACY_CALL_ID_PREFIX: &str = "call_fn_";

/// `messages`, with the deprecated function-calling turns mapped onto tool
/// calls: an assistant `function_call` becomes one `tool_calls` entry with a
/// synthesised id (`call_fn_0`, `call_fn_1`, … in message order), and the
/// `function`-role reply right after it becomes a `tool` turn carrying that
/// id. An assistant turn that also sends non-empty `tool_calls` keeps those
/// (the reply then answers their first call); an empty `tool_calls` is the
/// same as none. A `function` turn with no call before it still becomes a
/// `tool` turn, with a fresh id of its own: templates that require an id get
/// one, and it matches no call, which is what the history says.
fn de_chat_messages<'de, D>(d: D) -> Result<Vec<ChatMessage>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let wire = Vec::<WireChatMessage>::deserialize(d)?;
    let mut next_id = 0usize;
    let mut fresh_id = || {
        let id = format!("{LEGACY_CALL_ID_PREFIX}{next_id}");
        next_id += 1;
        id
    };
    // The id of the call made by the message just before, awaiting its reply.
    let mut awaiting: Option<String> = None;
    let mut messages = Vec::with_capacity(wire.len());
    for w in wire {
        let mut message = ChatMessage {
            role: w.role,
            content: w.content,
            name: w.name,
            tool_calls: w.tool_calls.filter(|c| !c.is_empty()),
            tool_call_id: w.tool_call_id,
        };
        let call_id = awaiting.take();
        if message.role == "function" {
            message.role = "tool".into();
            if message.tool_call_id.is_none() {
                message.tool_call_id = Some(call_id.unwrap_or_else(&mut fresh_id));
            }
        } else if let Some(function) = w.function_call {
            if message.role == "assistant" {
                match &message.tool_calls {
                    Some(calls) => awaiting = Some(calls[0].id.clone()),
                    None => {
                        let id = fresh_id();
                        awaiting = Some(id.clone());
                        message.tool_calls = Some(vec![ToolCall {
                            id,
                            r#type: "function".into(),
                            function,
                        }]);
                    }
                }
            }
        }
        messages.push(message);
    }
    Ok(messages)
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
    /// Moves the deprecated `functions` / `function_call` onto `tools` /
    /// `tool_choice` (M1 D5), before [`validate_tool_choice`](Self::validate_tool_choice).
    /// Mixing a legacy field with its replacement is a 400, so the answer's
    /// shape (legacy `function_call` or `tool_calls`) is never ambiguous.
    fn apply_legacy_functions(&mut self) -> Result<(), String> {
        if self.functions.is_some() && self.tools.is_some() {
            return Err(
                "`functions` and `tools` cannot be sent together; send only `tools` \
                 (`functions` is deprecated)"
                    .into(),
            );
        }
        if self.function_call.is_some() && self.tool_choice.is_some() {
            return Err(
                "`function_call` and `tool_choice` cannot be sent together; send only \
                 `tool_choice` (`function_call` is deprecated)"
                    .into(),
            );
        }
        if let Some(functions) = self.functions.take() {
            self.tools = Some(
                functions
                    .into_iter()
                    .map(|function| Tool {
                        r#type: "function".into(),
                        function,
                    })
                    .collect(),
            );
            self.legacy_functions = true;
        }
        let Some(call) = self.function_call.take() else {
            return Ok(());
        };
        if matches!(call.as_str(), Some("none" | "auto")) {
            self.tool_choice = Some(call);
            return Ok(());
        }
        let name = call
            .get("name")
            .and_then(|n| n.as_str())
            .filter(|n| !n.is_empty())
            .ok_or_else(|| {
                "function_call must be \"none\", \"auto\" or {\"name\":...}".to_string()
            })?;
        let declared = self
            .tools
            .iter()
            .flatten()
            .any(|t| t.function.get("name").and_then(|n| n.as_str()) == Some(name));
        if !declared {
            let list = if self.legacy_functions {
                "functions"
            } else {
                "tools"
            };
            return Err(format!(
                "function_call names function '{name}', which is not in {list}"
            ));
        }
        self.tool_choice =
            Some(serde_json::json!({"type": "function", "function": {"name": name}}));
        Ok(())
    }

    /// The output-token limit the client asked for, if any:
    /// `max_completion_tokens` wins over the deprecated `max_tokens` (as vLLM).
    fn requested_max_tokens(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    /// Build the engine-facing `SamplingParams`: an omitted `top_p` takes the
    /// server default, omitted penalties the OpenAI default 0.0.
    fn sampling_params(&self, defaults: &GenerationDefaults) -> cascadia_types::SamplingParams {
        cascadia_types::SamplingParams {
            top_p: self.top_p.unwrap_or(defaults.top_p),
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

    /// `tool_choice` must be one of the OpenAI shapes: `"auto"`, `"none"`,
    /// `"required"`, or `{"type":"function","function":{"name":N}}` naming a
    /// function present in `tools`. Anything else is the caller's mistake
    /// (a 400), not something an engine should guess at.
    fn validate_tool_choice(&self) -> Result<(), String> {
        let Some(choice) = &self.tool_choice else {
            return Ok(());
        };
        if let Some(s) = choice.as_str() {
            return match s {
                "auto" | "none" | "required" => Ok(()),
                other => Err(format!(
                    "tool_choice '{other}' is not one of \"auto\", \"none\", \"required\""
                )),
            };
        }
        let name = choice
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .filter(|n| choice.get("type").and_then(|t| t.as_str()) == Some("function") && !n.is_empty())
            .ok_or_else(|| {
                "tool_choice must be a string or {\"type\":\"function\",\"function\":{\"name\":...}}"
                    .to_string()
            })?;
        let declared = self
            .tools
            .iter()
            .flatten()
            .any(|t| t.function.get("name").and_then(|n| n.as_str()) == Some(name));
        if declared {
            Ok(())
        } else {
            Err(format!(
                "tool_choice names function '{name}', which is not in tools"
            ))
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

#[derive(Serialize)]
struct ChatChoiceMessage {
    role: &'static str,
    /// `Option` so a tool-call turn emits `content: null` (OpenAI shape).
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    /// The legacy single call, answering a `functions` request (M1 D5).
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<FunctionCall>,
}

impl ChatChoiceMessage {
    /// The legacy shape: the first tool call as `function_call` (the legacy
    /// field holds one call), no `tool_calls`.
    fn into_legacy(mut self) -> Self {
        self.function_call = self
            .tool_calls
            .take()
            .and_then(|calls| calls.into_iter().next())
            .map(|c| c.function);
        self
    }
}

/// OpenAI chat `logprobs` object: one entry per generated token under
/// `content`. An object only when the request set `logprobs: true` AND the
/// engine emitted per-token logprobs; otherwise the choice carries `null`.
#[derive(Serialize)]
struct ChatLogprobs {
    content: Vec<cascadia_types::TokenLogprobs>,
}

#[derive(Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatChoiceMessage,
    finish_reason: &'static str,
    /// Required (nullable) by the OpenAI schema: serialized as `null` when absent.
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
    /// `null` = unset = the server default ([`GenerationDefaults`]). This
    /// endpoint has no `max_completion_tokens` (OpenAI's legacy schema).
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// `null` = unset = the server default ([`GenerationDefaults`]).
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Any integer from `i64::MIN` to `u64::MAX` (the spec's `seed` is a
    /// plain integer); a negative one reaches the engine as its
    /// two's-complement `u64` (`-1` → `u64::MAX`). See [`de_seed`].
    #[serde(default, deserialize_with = "de_seed")]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopSpec>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub logprobs: Option<u32>,
    /// `null` = unset.
    #[serde(default, deserialize_with = "de_null_default")]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Prepend the prompt to the returned text (OpenAI `echo`). `null` = unset.
    #[serde(default, deserialize_with = "de_null_default")]
    pub echo: bool,
}

impl CompletionRequest {
    /// See [`ChatCompletionRequest::sampling_params`].
    fn sampling_params(&self, defaults: &GenerationDefaults) -> cascadia_types::SamplingParams {
        cascadia_types::SamplingParams {
            top_p: self.top_p.unwrap_or(defaults.top_p),
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
    // converting to the legacy schema is the follow-up. The field is `null`
    // when empty (the OpenAI schema requires it, nullable), so a legacy client
    // sees no malformed object today.
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
/// serde_json formatter with Python `json.dumps` separators: the default
/// `(", ", ": ")`, or whatever the template passed as `separators=` (Inkling's
/// template asks for the compact `(",", ":")`). serde_json's compact formatter
/// omits both spaces, which is why the default is not simply the builtin.
struct PyDumpsCompact {
    item: Vec<u8>,
    key: Vec<u8>,
}

impl Default for PyDumpsCompact {
    fn default() -> Self {
        Self {
            item: b", ".to_vec(),
            key: b": ".to_vec(),
        }
    }
}

impl serde_json::ser::Formatter for PyDumpsCompact {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(&self.item)
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
            w.write_all(&self.item)
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&self.key)
    }
}

/// `json.dumps(sort_keys=True)`: rebuild every object with its keys in sorted
/// order (recursively), leaving arrays and scalars untouched.
fn sort_json_keys(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(m) => {
            let mut entries: Vec<(String, serde_json::Value)> = m.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let mut out = serde_json::Map::new();
            for (k, val) in entries {
                out.insert(k, sort_json_keys(val));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(a) => {
            serde_json::Value::Array(a.into_iter().map(sort_json_keys).collect())
        }
        other => other,
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
    // transformers' shim also accepts sort_keys and separators (Inkling's
    // template renders tool specs and tool calls with
    // `tojson(sort_keys=true, separators=(",", ":"))`). Both are honoured with
    // json.dumps semantics; anything else unknown still fails the render
    // rather than silently producing a prompt the model never saw.
    let sort_keys = args.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    let separators = match args.get::<Option<Value>>("separators")? {
        None => None,
        Some(v) => {
            let parts: Vec<String> = v
                .try_iter()
                .ok()
                .map(|it| it.map(|x| x.to_string()).collect())
                .unwrap_or_default();
            if parts.len() != 2 {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "tojson: separators= must be a (item, key) pair",
                ));
            }
            Some((parts[0].clone(), parts[1].clone()))
        }
    };
    args.assert_all_used()?;
    // Python honours an explicit separators tuple under indent (items end in
    // the item separator, keys use the key separator); serde_json's pretty
    // formatter has fixed separators. Refuse the combination rather than
    // silently emitting bytes the template did not ask for.
    if indent.is_some() && separators.is_some() {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "tojson: separators= together with indent= is not supported",
        ));
    }

    // Route through serde_json::Value when keys must be sorted (a serde_json
    // Map may preserve insertion order when the `preserve_order` feature is
    // unified into the build; sorting explicitly keeps the output stable).
    let sorted: Option<serde_json::Value> = if sort_keys {
        Some(sort_json_keys(serde_json::to_value(&value).map_err(
            |err| {
                Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(err)
            },
        )?))
    } else {
        None
    };
    let mut out = Vec::<u8>::new();
    let res = match indent {
        // Python switches to `(',', ': ')` under indent, which is exactly what
        // serde_json's pretty formatter already emits.
        Some(n) => {
            let pad = " ".repeat(n);
            let fmt = serde_json::ser::PrettyFormatter::with_indent(pad.as_bytes());
            let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
            match &sorted {
                Some(sv) => serde::Serialize::serialize(sv, &mut ser),
                None => serde::Serialize::serialize(&value, &mut ser),
            }
        }
        None => {
            let fmt = match &separators {
                Some((item, key)) => PyDumpsCompact {
                    item: item.as_bytes().to_vec(),
                    key: key.as_bytes().to_vec(),
                },
                None => PyDumpsCompact::default(),
            };
            let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
            match &sorted {
                Some(sv) => serde::Serialize::serialize(sv, &mut ser),
                None => serde::Serialize::serialize(&value, &mut ser),
            }
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

    #[allow(clippy::type_complexity)]
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
        bos_token => bos_token,
        eos_token => eos_token,
        tools => tools_value,
    };
    // `reasoning_effort` is added only when Some: a `None` value renders as
    // Jinja `none`, which `is defined` / `default()` treat as a caller's choice.
    // Left out entirely, the template's own default applies (GLM-5 resolves
    // undefined to 'max'; Qwen3.8 to 'xhigh').
    let ctx = match reasoning_effort {
        Some(effort) => context! { reasoning_effort => effort, ..ctx },
        None => ctx,
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
/// What the template should see as `reasoning_effort`, and how it was chosen.
#[derive(Clone, Copy, Debug, Default)]
pub struct EffortArgs<'a> {
    /// The API's word: OpenAI vocabulary mapped onto GLM's high/max levels
    /// (`effective_reasoning_effort`).
    pub mapped: Option<&'a str>,
    /// The caller's own word, verbatim.
    pub raw: Option<&'a str>,
    /// The caller set `enable_thinking` explicitly (a toggle beats an effort
    /// word that would switch thinking off).
    pub explicit_thinking: bool,
    /// The template distinguishes the OpenAI words (see
    /// [`template_speaks_effort_words`]).
    pub template_words: bool,
}

/// Does `env` treat OpenAI's `reasoning_effort` words as distinct levels?
/// Probed once at load by rendering the same message with "low" and "medium":
/// Inkling (none/minimal/low/medium/high/max → a numeric thinking level) and
/// Qwen3.8 (xhigh/medium/low) render them differently; GLM-5 (`'high' if
/// reasoning_effort == 'high' else 'max'`) and templates that ignore the
/// effort render them identically, and keep the GLM mapping.
pub fn template_speaks_effort_words(
    env: &minijinja::Environment<'static>,
    bos_token: &str,
    eos_token: &str,
) -> bool {
    let msgs = [ChatMessage {
        role: "user".into(),
        content: "probe".into(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    let render =
        |w: &str| render_with_chat_env(env, &msgs, bos_token, eos_token, true, Some(w), None);
    match (render("low"), render("medium")) {
        (Ok(a), Ok(b)) => a != b,
        // A probe render that errors is indeterminate, not "identical" — a
        // template that genuinely speaks the effort words but throws on the
        // bare probe input (e.g. it requires a system turn) would otherwise be
        // silently mapped GLM-style, escalating a thinking-off/low request.
        // Fall back to the mapping but leave a trace so the mis-detection is
        // visible.
        (Err(e), _) | (_, Err(e)) => {
            warn!(
                error = %e,
                "reasoning_effort probe could not render the chat template; assuming GLM-style high/max mapping (effort words will not reach the template)"
            );
            false
        }
    }
}

fn render_or_fallback(
    env: Option<&minijinja::Environment<'static>>,
    messages: &[ChatMessage],
    bos_token: &str,
    eos_token: &str,
    enable_thinking: bool,
    effort: EffortArgs<'_>,
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
    // Which `reasoning_effort` the template sees, in order of preference.
    //
    // Templates that distinguish OpenAI's words (`template_words`, probed at
    // load — Inkling, Qwen3.8) get the caller's own word: the GLM mapping
    // would silently turn "none" or "low" into "high" (measured on Inkling:
    // a thinking-off request rendered at level 0.9). Thinking off with no
    // effort given sends "none" (Inkling switches thinking through the
    // effort). An explicit `enable_thinking: true` beats a "none" — the
    // documented toggle-wins rule — by sending the mapped word instead. Then
    // the mapped word, then undefined (the template's default) so Qwen3.8's
    // rejection of "high" lands on its `default('xhigh')`.
    //
    // Every other template (GLM-5's high/max vocabulary, or one that ignores
    // the effort) sees the mapped word first, as before; on a rejection, the
    // caller's own word if it differs, else undefined.
    //
    // A word the caller chose that the template rejects stays a rejection
    // (the 400 below) — unless it is also the API's mapped word, which is the
    // API's to discard. Serving a rejected caller word at another level would
    // drop a parameter the caller believed had taken effect.
    //
    // Substring match, coupled to the templates' `raise_exception` prose:
    // Qwen3.8 says "Unexpected reasoning effort ...", Inkling "Unknown
    // reasoning_effort: ...", and the underscore spelling covers a template
    // that names the parameter instead.
    let rejects_effort = |r: &Result<String, PromptRenderError>| {
        matches!(r, Err(PromptRenderError::Rejected(m)) if {
            let m = m.to_ascii_lowercase();
            m.contains("reasoning effort") || m.contains("reasoning_effort")
        })
    };
    let EffortArgs {
        mapped,
        raw,
        explicit_thinking,
        template_words,
    } = effort;
    fn push<'a>(order: &mut Vec<Option<&'a str>>, w: Option<&'a str>) {
        if !order.contains(&w) {
            order.push(w);
        }
    }
    let mut order: Vec<Option<&str>> = Vec::with_capacity(4);
    if template_words {
        match raw {
            Some("none") if enable_thinking && explicit_thinking => push(&mut order, mapped),
            Some(r) => push(&mut order, Some(r)),
            None if !enable_thinking => push(&mut order, Some("none")),
            None => push(&mut order, mapped),
        }
        push(&mut order, mapped);
    } else {
        push(&mut order, mapped);
        if let Some(r) = raw {
            push(&mut order, Some(r));
        }
    }
    push(&mut order, None);
    // The caller's own word, when it is not also the API's.
    let callers = raw.filter(|r| Some(*r) != mapped);
    let mut first = Err(PromptRenderError::Failed("no render attempted".into()));
    for (i, candidate) in order.iter().enumerate() {
        first = render_with_chat_env(
            env,
            messages,
            bos_token,
            eos_token,
            enable_thinking,
            *candidate,
            tools,
        );
        if !rejects_effort(&first) {
            break;
        }
        // The caller's own word was rejected: that is their 400 to see.
        if callers.is_some() && *candidate == callers {
            break;
        }
        if i + 1 < order.len() {
            tracing::info!(
                rejected = ?candidate,
                next = ?order[i + 1],
                "chat_template rejected a reasoning_effort word; rendering again"
            );
        }
    }
    match first {
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
    effort: EffortArgs<'_>,
    tools: Option<&[Tool]>,
) -> Result<String, PromptRenderError> {
    // ov-genai applies the model's own template, so the engine gets raw text.
    // Tools would be dropped on the floor here, so refuse rather than pretend.
    // `reasoning_effort` (the caller's parameter above) is also dropped on
    // this branch, intentionally: ov-genai applies its own template with the
    // effort undefined, so there is nowhere for the mapped value to go.
    if state.engine_applies_template {
        // the child renders the GGUF chat template on the forwarded turns;
        // the prompt is a fallback and must simply be non-empty
        return Ok(render_prompt_legacy(messages));
    }
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
        EffortArgs {
            template_words: state.effort_words,
            ..effort
        },
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
                    function_call: None,
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
            function_call: None,
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

/// Opening marker of the `<tool_call>…</tool_call>` block dialects
/// (Hermes/Qwen JSON, Qwen3 `<function=…>` XML, GLM arg-kv).
const TOOL_CALL_OPEN: &str = "<tool_call>";

/// Llama-3.1's explicit whole-output tool-call marker.
const PYTHON_TAG: &str = "<|python_tag|>";

/// Incremental content/tool-call split for a streamed chat reply with tools
/// active. Fed the same translated text the non-streaming path accumulates,
/// it releases every byte that can no longer become part of a tool call and
/// withholds the rest, so content streams live while the final decision is
/// still exactly `parse_tool_calls(tool_call_scan_text(full_text, …))`.
///
/// Withheld from the hold point on, where scanning starts at 0 or (thinking
/// on) just past the first `</think>`, mirroring `tool_call_scan_text`:
/// * from the scan start, while the first non-whitespace text there is
///   undecided or starts a whole-output call (`{`, `[`, `<|python_tag|>` or
///   a prefix of it) — the bare-JSON / Llama-3.1 dialects parse the whole
///   scan text, so none of it is content until the reply ends;
/// * otherwise from the first `<tool_call>`, or from a trailing proper
///   prefix of it (a marker split across chunks).
///
/// Before `</think>` arrives the rules apply from 0; once it does, text held
/// only for a scratchpad draft is released. Released text is never taken
/// back, and every boundary sits on an ASCII byte, so no UTF-8 char splits.
struct ToolStreamSplitter {
    buf: String,
    /// Bytes of `buf` already released as content.
    emitted: usize,
    enable_thinking: bool,
    /// Where tool-call scanning starts: 0, or just past the first `</think>`.
    scan_start: usize,
    /// Thinking on, no `</think>` yet: where the next search for it starts.
    think_from: Option<usize>,
    /// First non-whitespace byte at/after `scan_start` (`buf.len()` while
    /// only whitespace has arrived). Advanced per feed, so a long leading
    /// whitespace run is scanned once, not on every feed.
    lead: usize,
}

/// The decision at the end of a tools-active stream.
enum ToolStreamEnd<'a> {
    /// No call: the text not yet released, so the content deltas add up to
    /// the whole reply.
    Content(&'a str),
    /// Calls parsed: `content` is the text still releasable before the hold
    /// point; everything from the hold point on is dropped.
    ToolCalls {
        content: &'a str,
        calls: Vec<ToolCall>,
    },
}

impl ToolStreamSplitter {
    fn new(enable_thinking: bool) -> Self {
        Self {
            buf: String::new(),
            emitted: 0,
            enable_thinking,
            scan_start: 0,
            think_from: enable_thinking.then_some(0),
            lead: 0,
        }
    }

    /// Append `text`; return the newly releasable content (possibly empty).
    fn feed(&mut self, text: &str) -> &str {
        self.push(text);
        let start = self.emitted;
        self.emitted = self.hold_point().max(start);
        &self.buf[start..self.emitted]
    }

    /// Append the final `text` and decide: calls, or the rest as content.
    fn finish(&mut self, text: &str) -> ToolStreamEnd<'_> {
        self.push(text);
        let start = self.emitted;
        let calls = parse_tool_calls(tool_call_scan_text(&self.buf, self.enable_thinking));
        let end = if calls.is_some() {
            self.hold_point().max(start)
        } else {
            self.buf.len()
        };
        self.emitted = self.buf.len();
        let content = &self.buf[start..end];
        match calls {
            Some(calls) => ToolStreamEnd::ToolCalls { content, calls },
            None => ToolStreamEnd::Content(content),
        }
    }

    fn push(&mut self, text: &str) {
        self.buf.push_str(text);
        if let Some(from) = self.think_from {
            match self.buf[from..].find(THINK_CLOSE) {
                Some(i) => {
                    self.scan_start = from + i + THINK_CLOSE.len();
                    self.think_from = None;
                    self.lead = self.scan_start;
                }
                // Only the last `len - 1` bytes can start a delimiter that the
                // next feed completes.
                None => {
                    let mut next = self.buf.len().saturating_sub(THINK_CLOSE.len() - 1);
                    while !self.buf.is_char_boundary(next) {
                        next -= 1;
                    }
                    self.think_from = Some(next.max(from));
                }
            }
        }
        let rest = &self.buf[self.lead..];
        self.lead += rest.len() - rest.trim_start().len();
    }

    /// First byte that may still belong to a tool call.
    fn hold_point(&self) -> usize {
        let start = self.scan_start;
        let lead = &self.buf[self.lead..];
        if lead.is_empty()
            || lead.starts_with(['{', '['])
            || lead.starts_with(PYTHON_TAG)
            || PYTHON_TAG.starts_with(lead)
        {
            return start;
        }
        // Released text holds no marker at/after `start`, so the search can
        // resume at the release point.
        let from = start.max(self.emitted);
        let rest = &self.buf[from..];
        from + rest.find(TOOL_CALL_OPEN).unwrap_or_else(|| {
            (1..TOOL_CALL_OPEN.len())
                .rev()
                .find(|&k| rest.ends_with(&TOOL_CALL_OPEN[..k]))
                .map_or(rest.len(), |k| rest.len() - k)
        })
    }
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
        // A malformed pair means the block does not say what the model asked
        // for, so no call is emitted. These used to `break`, which fell through
        // to the unconditional `Some(..)` below and fired the call with the
        // arguments parsed so far — `send(to = "a@b.com")` with `body` silently
        // gone, reported as a clean `finish_reason: "tool_calls"`. Truncation
        // here is not rare: an `<arg_value>` containing the literal
        // `</tool_call>` ends the block early, which is routine when an agent
        // writes about the tool-call format itself.
        //
        // The loop only runs on a remaining `<arg_key>`, so trailing junk is
        // ignored unless it opens one — in which case a complete set of pairs
        // is dropped too. Without the request's tool schemas there is no way to
        // tell that apart from a truncation, and dropping fires nothing wrong
        // where the old behaviour fired something wrong.
        let ka = &rest[ks + "<arg_key>".len()..];
        let ke = ka.find("</arg_key>")?;
        let key = ka[..ke].trim().to_string();
        let after_key = &ka[ke + "</arg_key>".len()..];
        let vs = after_key.find("<arg_value>")?;
        let va = &after_key[vs + "<arg_value>".len()..];
        let ve = va.find("</arg_value>")?;
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
            EffortArgs {
                mapped: reasoning_effort,
                ..Default::default()
            },
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

/// The engine-side structured turns from a chat request: tool fields travel
/// as the OpenAI wire JSON a downstream chat server (llama-server's jinja)
/// expects verbatim.
fn chat_turns(req: &ChatCompletionRequest) -> Vec<ChatTurn> {
    req.messages
        .iter()
        .map(|m| ChatTurn {
            role: m.role.clone(),
            content: m.content.clone(),
            tool_calls: m
                .tool_calls
                .as_ref()
                .and_then(|cs| serde_json::to_value(cs).ok()),
            tool_call_id: m.tool_call_id.clone(),
            name: m.name.clone(),
        })
        .collect()
}

/// The request's tool definitions as wire JSON for the engine.
fn chat_tools(req: &ChatCompletionRequest) -> Option<serde_json::Value> {
    req.tools
        .as_ref()
        .and_then(|ts| serde_json::to_value(ts).ok())
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(mut req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let task_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    if let Err(e) = req
        .apply_legacy_functions()
        .and_then(|()| req.validate_tool_choice())
    {
        count_rejected("invalid_request");
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e })),
        )
            .into_response();
    }
    let enable_thinking = req.effective_enable_thinking();
    // effective_reasoning_effort() hardcodes GLM's high/max vocabulary and is
    // applied to every model this server serves, not just GLM-5. A template
    // with its own scale (Qwen3.8: xhigh/medium/low) rejects the mapped word;
    // render_or_fallback then retries with the caller's own word, so a
    // client's "low" reaches such a template unescalated — or, when the caller
    // sent nothing of their own, with the effort undefined (the template's
    // default). A caller word the template also rejects is a 400.
    let prompt = match render_prompt(
        &state,
        &req.messages,
        enable_thinking,
        EffortArgs {
            mapped: Some(req.effective_reasoning_effort()),
            raw: req.reasoning_effort.as_deref(),
            explicit_thinking: req.enable_thinking.is_some(),
            template_words: state.effort_words,
        },
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
            // Stays a 5xx: a template that cannot render is a server-side fault
            // an operator needs to see, not the caller's mistake. But it is
            // raised before the engine is involved, so it must not latch
            // readiness — see `PreEngineFailure`.
            let mut resp = (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
            resp.extensions_mut().insert(PreEngineFailure);
            return resp;
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
    // Same bound on the structured turns: tool-call results ride as
    // messages and never count toward prompt.len(), so a request could
    // pass the prompt check while shipping a megabyte of tool output.
    let messages_bytes: usize = req
        .messages
        .iter()
        .map(|m| {
            m.content.len()
                + m.tool_calls
                    .as_ref()
                    .map_or(0, |cs| serde_json::to_string(cs).map_or(0, |s| s.len()))
                + m.tool_call_id.as_ref().map_or(0, String::len)
                + m.name.as_ref().map_or(0, String::len)
        })
        .sum();
    if messages_bytes > state.max_prompt_bytes {
        count_rejected("prompt_too_large");
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(serde_json::json!({
                "error": format!(
                    "messages are {messages_bytes} bytes (content + tool fields); \
                     max allowed is {} (max_prompt_bytes)",
                    state.max_prompt_bytes,
                )
            })),
        )
            .into_response();
    }
    let task = GenerationTask {
        task_id: task_id.clone(),
        prompt,
        // Structured turns for engines that forward the conversation to a
        // downstream chat server applying its own template (sycl-llama:
        // llama-server renders the GGUF chat template). Other engines ignore
        // the field and keep consuming `prompt`.
        messages: chat_turns(&req),
        tools: chat_tools(&req),
        tool_choice: req.tool_choice.clone(),
        // Unset limit = one context window (Decision 2, D2), an explicit one
        // at most the window when it is known (D13); unset temperature/top_p
        // = the model's generation_config.json, else 1 (D3).
        max_tokens: state
            .generation_defaults
            .max_tokens_for(req.requested_max_tokens()),
        temperature: req
            .temperature
            .unwrap_or(state.generation_defaults.temperature),
        logprobs: req.logprobs_count(),
        sampling: req.sampling_params(&state.generation_defaults),
        enable_thinking,
        trust_remote_code: false,
        // H.1b: hardcoded, never read from the request. The tenant is a KV-namespace boundary; a
        // client that could set it from request JSON could name another tenant's namespace and
        // resume off its prefix. The real value is assigned server-side from the admitted identity.
        tenant: String::new(),
        // Never client-settable, same class as `tenant`: a client that could
        // seed it from request JSON would inject arbitrary token-ids into the
        // model's context PAST the rendered prompt. Only the scheduler's
        // rescue path sets it, engine-side of this API.
        resume_token_ids: None,
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
            req.legacy_functions,
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
    let mut marker = state.marker_dialect.clone().map(MarkerFilter::new);
    let mut buf = String::new();
    let mut completion_tokens: u32 = 0;
    let mut prompt_tokens: u32 = 0;
    // OpenAI `finish_reason`: `length` when the engine hit max_tokens, else
    // `stop`. Engines that don't distinguish leave it None → "stop".
    let mut finish_reason: &'static str = "stop";
    let mut logprobs_content: Vec<cascadia_types::TokenLogprobs> = Vec::new();
    let stops = req.sampling_params(&state.generation_defaults).stop;
    let mut stop = StopMatcher::new(&stops);
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
        // Count every chunk, but replace the running sum when the final
        // carries authoritative usage (including hidden thinking/tools).
        let text = match marker.as_mut() {
            Some(f) => {
                let mut t = f.feed(&chunk);
                if chunk.is_final {
                    t.push_str(&f.finish());
                }
                t
            }
            None => chunk.text.clone(),
        };
        buf.push_str(&stop.push(&text));
        chunk.accumulate_tokens(&mut completion_tokens);
        if let Some(lp) = &chunk.logprobs {
            logprobs_content.push(lp.clone());
        }
        // The latest count any chunk carries: token-streaming engines report
        // it on their first chunk too, so it is known before an early stop
        // (D14).
        if let Some(n) = chunk.prompt_tokens {
            prompt_tokens = n;
        }
        if chunk.is_final {
            if let Some(fr) = chunk.finish_reason {
                finish_reason = fr.as_openai_str();
            }
        }
        // Early stop (M1 D11): end at the match. Dropping the stream below
        // cancels the engine task; usage counts the tokens received so far.
        if stop.stopped() {
            finish_reason = "stop";
            break;
        }
    }
    drop(chunk_stream);
    buf.push_str(&stop.finish());
    state
        .stats
        .tokens_total
        .fetch_add(completion_tokens as u64, Ordering::Relaxed);

    // Final guard on `stop`: the matcher above already ended the generation
    // at the earliest match; trimming the assembled text again keeps the
    // OpenAI output contract whatever reached `buf`.
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

    let (mut message, choice_finish) = build_choice(
        buf,
        &req.tool_choice,
        req.tools.as_ref().is_some_and(|t| !t.is_empty()),
        enable_thinking,
    );
    // A parsed tool call overrides the streamed length/stop detection;
    // otherwise keep `finish_reason` (e.g. "length" when max_tokens hit).
    if choice_finish == "tool_calls" {
        finish_reason = "tool_calls";
        if req.legacy_functions {
            message = message.into_legacy();
            finish_reason = "function_call";
        }
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
        // Prompt-only: no structured turns on the /v1/completions path.
        messages: Vec::new(),
        tools: None,
        tool_choice: None,
        // Same defaults and window clamp as chat (D4, D13).
        max_tokens: state.generation_defaults.max_tokens_for(req.max_tokens),
        temperature: req
            .temperature
            .unwrap_or(state.generation_defaults.temperature),
        logprobs: req.logprobs_count(),
        sampling: req.sampling_params(&state.generation_defaults),
        enable_thinking: false,
        trust_remote_code: false,
        // H.1b: hardcoded server-side; see the note on the chat endpoint's task above.
        tenant: String::new(),
        // Never client-settable; see the chat endpoint's note.
        resume_token_ids: None,
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
    let mut marker = state.marker_dialect.clone().map(MarkerFilter::new);
    let mut buf = String::new();
    let mut completion_tokens: u32 = 0;
    let mut prompt_tokens: u32 = 0;
    let mut finish_reason: &'static str = "stop";
    let mut logprobs_content: Vec<cascadia_types::TokenLogprobs> = Vec::new();
    let stops = req.sampling_params(&state.generation_defaults).stop;
    let mut stop = StopMatcher::new(&stops);
    while let Some(chunk) = chunk_stream.next().await {
        if let Some(reason) = &chunk.error {
            warn!(task = %task_id, reason = %reason, "engine failed task; returning 503");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": reason })),
            )
                .into_response();
        }
        let text = match marker.as_mut() {
            Some(f) => {
                let mut t = f.feed(&chunk);
                if chunk.is_final {
                    t.push_str(&f.finish());
                }
                t
            }
            None => chunk.text.clone(),
        };
        buf.push_str(&stop.push(&text));
        chunk.accumulate_tokens(&mut completion_tokens);
        if let Some(lp) = &chunk.logprobs {
            logprobs_content.push(lp.clone());
        }
        // The latest count any chunk carries: token-streaming engines report
        // it on their first chunk too, so it is known before an early stop
        // (D14).
        if let Some(n) = chunk.prompt_tokens {
            prompt_tokens = n;
        }
        if chunk.is_final {
            if let Some(fr) = chunk.finish_reason {
                finish_reason = fr.as_openai_str();
            }
        }
        // Early stop (M1 D11): end at the match. Dropping the stream below
        // cancels the engine task; usage counts the tokens received so far.
        if stop.stopped() {
            finish_reason = "stop";
            break;
        }
    }
    drop(chunk_stream);
    buf.push_str(&stop.finish());
    state
        .stats
        .tokens_total
        .fetch_add(completion_tokens as u64, Ordering::Relaxed);

    // Stop-sequence final guard (same as the chat path).
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
    let stop = task.sampling.stop.clone();
    let chunk_stream = match state.runner.generate_async(task).await {
        Ok(s) => s,
        Err(err) => {
            warn!(error = %err, "completions-stream: generate failed");
            return engine_error_response(err);
        }
    };
    let text_stream = TextStream {
        chunks: Some(spawn_engine_forwarder(StreamWithPermit {
            inner: chunk_stream,
            _permit: permit,
            _inflight: inflight,
        })),
        keepalive: KeepAlive::new(state.sse_keepalive),
        kind: TextStreamKind::Completion { echo: echo_prefix },
        // The raw-prompt endpoint streams the engine's text as is.
        marker: None,
        stop: StopMatcher::new(&stop),
        stats: state.stats.clone(),
        completion: std::sync::atomic::AtomicU32::new(0),
        held_tokens: 0,
        prompt_tokens: 0,
        finished: false,
        phase: ToolStreamPhase::Streaming,
        task_id,
        model,
        include_usage,
    };
    sse_response(stream::unfold(text_stream, |mut s| async move {
        let frame = s.next_frame().await?;
        s.keepalive.sent();
        Some((frame, s))
    }))
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
    legacy_functions: bool,
) -> axum::response::Response {
    let task_id = task.task_id.clone();
    // Captured before `task` moves into `generate` below — arms the tool-call
    // scan the same way the non-streaming path does (see `tool_call_scan_text`).
    let enable_thinking = task.enable_thinking;
    let stop = StopMatcher::new(&task.sampling.stop);

    let chunk_stream = match state.runner.generate_async(task).await {
        Ok(s) => s,
        Err(err) => {
            warn!(error = %err, "ov-stream: generate failed");
            return engine_error_response(err);
        }
    };
    // Move the permit + in-flight guard into the engine stream, which its
    // forwarder task owns: they are released when the forwarder ends (the
    // stream finished, or the body dropped its receiver on a client
    // disconnect or a stop match).
    let chunks = Some(spawn_engine_forwarder(StreamWithPermit {
        inner: chunk_stream,
        _permit: permit,
        _inflight: inflight,
    }));
    // Tools present and not disabled: content streams as the splitter
    // releases it; whether the reply is a tool call is decided at the final
    // chunk, on the same text and parse as the non-streaming path.
    let tools_active =
        tools_present && tool_choice.as_ref().and_then(|v| v.as_str()) != Some("none");
    if tools_active {
        let tool_stream = ToolStream {
            chunks,
            keepalive: KeepAlive::new(state.sse_keepalive),
            marker: state.marker_dialect.clone().map(MarkerFilter::new),
            stop,
            splitter: ToolStreamSplitter::new(enable_thinking),
            stats: state.stats.clone(),
            completion: std::sync::atomic::AtomicU32::new(0),
            held_tokens: 0,
            prompt_tokens: 0,
            finished: false,
            phase: ToolStreamPhase::Streaming,
            task_id,
            model,
            include_usage,
            legacy_functions,
        };
        return sse_response(stream::unfold(tool_stream, |mut s| async move {
            let frame = s.next_frame().await?;
            s.keepalive.sent();
            Some((frame, s))
        }));
    }
    // Each chunk becomes a raw SSE frame `data: <json>\n\n`, sent as its
    // own body chunk via `Body::from_stream`.
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
    // per-token chunk on the wire as a separate ~220 B packet). Our own
    // keep-alive ([`KeepAlive`]) times out the engine `next()` instead.
    //
    // Each frame is followed by a `yield_now`: cascadia_runner's
    // ChunkStream::poll_next calls engine.step() inside a
    // parking_lot::Mutex, which blocks the tokio worker for ~80 ms per
    // chunk without yielding. Without an explicit yield, tokio's
    // cooperative scheduling lets this task hog its worker for ~1 s at a
    // time, and the per-chunk body writes accumulate in Hyper's buffer and
    // flush as one ~3500 B burst — visible in the demo as 16 tokens at once
    // every ~1 s. Yielding lets the body sink run between chunks so each
    // frame goes to TCP individually.
    let text_stream = TextStream {
        chunks,
        keepalive: KeepAlive::new(state.sse_keepalive),
        kind: TextStreamKind::Chat,
        marker: state.marker_dialect.clone().map(MarkerFilter::new),
        stop,
        stats: state.stats.clone(),
        completion: std::sync::atomic::AtomicU32::new(0),
        held_tokens: 0,
        prompt_tokens: 0,
        finished: false,
        phase: ToolStreamPhase::Streaming,
        task_id,
        model,
        include_usage,
    };
    sse_response(stream::unfold(text_stream, |mut s| async move {
        let frame = s.next_frame().await?;
        s.keepalive.sent();
        Some((frame, s))
    }))
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

/// Finds `stop` sequences in generated text as it arrives (M1 D11), so the
/// API can end a generation at the match instead of trimming a whole window
/// of output afterwards. Pushed text comes back once it can no longer be the
/// start of a stop sequence: only a tail that is a prefix of some stop is
/// held, so without a match nothing is delayed.
///
/// The cut is at the first stop to complete; of several completing on the
/// same byte, the one that starts earliest. Stop strings are client input of
/// any length, so the work is linear: each stop is a KMP automaton fed byte
/// by byte (O(stops) per byte, amortized), and the held tail is an offset
/// into a buffer compacted as it is released, never rescanned.
#[derive(Debug, Default)]
struct StopMatcher {
    stops: Vec<StopAutomaton>,
    /// `buf[start..]` is the text not yet released.
    buf: String,
    start: usize,
    stopped: bool,
}

/// One stop string as a Knuth-Morris-Pratt automaton over its bytes.
#[derive(Debug)]
struct StopAutomaton {
    pattern: Vec<u8>,
    /// `fail[i]`: length of the longest proper prefix of `pattern[..=i]`
    /// that is also its suffix.
    fail: Vec<usize>,
    /// Bytes of `pattern` matched by the end of the text seen so far.
    matched: usize,
}

impl StopAutomaton {
    fn new(stop: &str) -> Self {
        let pattern = stop.as_bytes().to_vec();
        let mut fail = vec![0; pattern.len()];
        let mut k = 0;
        for i in 1..pattern.len() {
            while k > 0 && pattern[i] != pattern[k] {
                k = fail[k - 1];
            }
            if pattern[i] == pattern[k] {
                k += 1;
            }
            fail[i] = k;
        }
        Self {
            pattern,
            fail,
            matched: 0,
        }
    }

    /// Advances by one byte; true when the whole pattern has just matched.
    fn step(&mut self, b: u8) -> bool {
        while self.matched > 0 && self.pattern[self.matched] != b {
            self.matched = self.fail[self.matched - 1];
        }
        if self.pattern[self.matched] == b {
            self.matched += 1;
        }
        if self.matched == self.pattern.len() {
            self.matched = self.fail[self.matched - 1];
            return true;
        }
        false
    }
}

impl StopMatcher {
    /// Empty stop strings match nothing and are dropped.
    fn new(stops: &[String]) -> Self {
        Self {
            stops: stops
                .iter()
                .filter(|s| !s.is_empty())
                .map(|s| StopAutomaton::new(s))
                .collect(),
            ..Self::default()
        }
    }

    fn is_active(&self) -> bool {
        !self.stops.is_empty()
    }

    /// A stop sequence matched: the text before it has been released and
    /// every later push returns nothing.
    fn stopped(&self) -> bool {
        self.stopped
    }

    /// Adds `text`; returns what can be released now. On a match that is the
    /// text up to the match, and [`stopped`](Self::stopped) turns true.
    fn push(&mut self, text: &str) -> String {
        if self.stopped {
            return String::new();
        }
        if self.stops.is_empty() {
            return text.to_owned();
        }
        let base = self.buf.len();
        self.buf.push_str(text);
        for (j, &b) in text.as_bytes().iter().enumerate() {
            let end = base + j + 1;
            let mut cut: Option<usize> = None;
            for stop in &mut self.stops {
                if stop.step(b) {
                    let at = end - stop.pattern.len();
                    cut = Some(cut.map_or(at, |c| c.min(at)));
                }
            }
            if let Some(cut) = cut {
                // `cut` starts a stop string, so it is a char boundary.
                self.stopped = true;
                let out = self.buf[self.start..cut].to_owned();
                self.buf = String::new();
                self.start = 0;
                return out;
            }
        }
        // Hold the longest partial match: the bytes that may begin a stop.
        // They equal a stop's prefix, so they start on a char boundary.
        let hold = self.stops.iter().map(|s| s.matched).max().unwrap_or(0);
        let mut keep = self.buf.len() - hold;
        while !self.buf.is_char_boundary(keep) {
            keep -= 1;
        }
        let keep = keep.max(self.start);
        let out = self.buf[self.start..keep].to_owned();
        self.start = keep;
        // Compact once the released prefix outgrows the held tail, so each
        // byte is moved O(1) times overall.
        if self.start > self.buf.len() - self.start {
            self.buf.drain(..self.start);
            self.start = 0;
        }
        out
    }

    /// Releases whatever is held (the generation ended without a match).
    fn finish(&mut self) -> String {
        let out = self.buf[self.start..].to_owned();
        self.buf = String::new();
        self.start = 0;
        out
    }
}

/// What a stream got while waiting on the engine.
enum StreamNext {
    Chunk(Option<cascadia_types::Chunk>),
    /// Nothing was sent for the keep-alive interval: send the comment.
    Idle,
}

/// Idle timer of an SSE stream (see [`SSE_KEEPALIVE_INTERVAL`]). The timeout
/// wraps the engine stream's `next()`, which is cancel-safe; axum's
/// `KeepAlive` is not used for the batching reason noted in
/// [`stream_completion`].
struct KeepAlive {
    interval: Duration,
    last_sent: tokio::time::Instant,
}

impl KeepAlive {
    fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_sent: tokio::time::Instant::now(),
        }
    }

    /// Something was just sent.
    fn sent(&mut self) {
        self.last_sent = tokio::time::Instant::now();
    }

    /// The next engine chunk, or [`StreamNext::Idle`] once the interval
    /// passes with nothing sent. `None` chunks (no stream) end the body.
    async fn next(&mut self, chunks: Option<&mut EngineChunks>) -> StreamNext {
        let Some(chunks) = chunks else {
            return StreamNext::Chunk(None);
        };
        let deadline = self.last_sent + self.interval;
        // Checked first: an engine that keeps returning chunks that send
        // nothing (progress markers, held text) never lets the timeout fire.
        if tokio::time::Instant::now() >= deadline {
            self.sent();
            return StreamNext::Idle;
        }
        match tokio::time::timeout_at(deadline, chunks.recv()).await {
            Ok(chunk) => StreamNext::Chunk(chunk),
            Err(_) => {
                self.sent();
                StreamNext::Idle
            }
        }
    }
}

/// The receiving end of an engine stream run by [`spawn_engine_forwarder`].
type EngineChunks = tokio::sync::mpsc::Receiver<cascadia_types::Chunk>;

/// Chunks the forwarder may hold ahead of the SSE body. One keeps the
/// engine paced by the body as before (it can be at most a chunk or two
/// ahead), so a stop match or a disconnect still stops it within a step.
const FORWARD_BUFFER: usize = 1;

/// Drives `chunks` on its own task and forwards each chunk into a bounded
/// channel. `ChunkStream::poll_next` runs `engine.step()` inline, so a long
/// one-shot step blocks whichever task polls it; polled here, it blocks this
/// task's worker, and the SSE body, reading the channel on another worker,
/// still sends keep-alive comments (D11).
///
/// Disconnect and cancel are unchanged: dropping the receiver (the body
/// dropped on a client disconnect, or a stop match) ends the forwarder at
/// its next poll, which drops the engine stream — that cancels the task and
/// releases the permit and in-flight guard, as dropping the body did. A
/// step already running cannot be interrupted either way.
fn spawn_engine_forwarder(mut chunks: StreamWithPermit) -> EngineChunks {
    let (tx, rx) = tokio::sync::mpsc::channel(FORWARD_BUFFER);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                () = tx.closed() => break,
                chunk = chunks.next() => match chunk {
                    Some(chunk) => {
                        if tx.send(chunk).await.is_err() {
                            break;
                        }
                    }
                    None => break,
                },
            }
        }
        drop(chunks);
    });
    rx
}

/// The OpenAI `include_usage` frame (successful completions only) + `[DONE]`.
fn usage_tail(task_id: &str, model: &str, object: &str, usage: Option<(u32, u32)>) -> Bytes {
    let mut out = String::new();
    if let Some((prompt, completion)) = usage {
        let usage = serde_json::json!({
            "id": task_id,
            "object": object,
            "created": now_unix(),
            "model": model,
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
    Bytes::from(out)
}

/// SSE `data:` frame for an engine error. The 200 headers are already on the
/// wire, so the failure travels as an explicit error event.
fn error_frame(task_id: &str, reason: &str) -> Bytes {
    let payload = serde_json::json!({
        "id": task_id,
        "object": "error",
        "error": { "message": reason, "type": "engine_error" },
    });
    Bytes::from(format!("data: {payload}\n\n"))
}

/// Which wire shape a [`TextStream`] sends.
enum TextStreamKind {
    /// `chat.completion.chunk` deltas (`/v1/chat/completions` without tools).
    Chat,
    /// `text_completion` chunks (`/v1/completions`); `echo` is prepended to
    /// the first frame, once.
    Completion { echo: Option<String> },
}

/// Body state of a plain (no tools) chat stream or a `/v1/completions`
/// stream: one frame per engine chunk, `stop` applied as the text arrives,
/// keep-alive comments while idle. Owns the permit + in-flight guard through
/// `chunks`; dropping `chunks` (client disconnect, or a stop match) releases
/// both and cancels the engine task.
struct TextStream {
    chunks: Option<EngineChunks>,
    keepalive: KeepAlive,
    kind: TextStreamKind,
    marker: Option<MarkerFilter>,
    stop: StopMatcher,
    stats: Arc<ApiStats>,
    completion: std::sync::atomic::AtomicU32,
    /// Model tokens whose text the stop matcher still holds.
    held_tokens: u32,
    prompt_tokens: u32,
    /// Completed normally (not failed, not cancelled): gates the usage frame.
    finished: bool,
    phase: ToolStreamPhase,
    task_id: String,
    model: String,
    include_usage: bool,
}

impl TextStream {
    /// Next SSE frame; `None` once `[DONE]` has been sent.
    async fn next_frame(&mut self) -> Option<Bytes> {
        loop {
            match self.phase {
                ToolStreamPhase::Streaming => {}
                ToolStreamPhase::Tail => {
                    self.phase = ToolStreamPhase::Done;
                    return Some(self.tail());
                }
                ToolStreamPhase::Done => return None,
            }
            let chunk = match self.keepalive.next(self.chunks.as_mut()).await {
                StreamNext::Idle => {
                    return Some(Bytes::from_static(SSE_KEEPALIVE_COMMENT.as_bytes()))
                }
                StreamNext::Chunk(Some(chunk)) => chunk,
                // Ended without a final chunk (cancelled): no usage. Text
                // the stop matcher held never matched, so it goes out first
                // (as the tools stream settles what it holds).
                StreamNext::Chunk(None) => {
                    self.phase = ToolStreamPhase::Done;
                    let rest = self.stop.finish();
                    if rest.is_empty() {
                        return Some(self.tail());
                    }
                    let n_tokens = std::mem::take(&mut self.held_tokens);
                    let mut out = self.frame(rest, n_tokens, None, None).into_bytes();
                    out.extend_from_slice(&self.tail());
                    return Some(Bytes::from(out));
                }
            };
            // A progress marker (a long prompt still going down the
            // pipeline): nothing to send, nothing to count. Yield so a run of
            // them cannot hold this worker.
            if chunk.is_progress() {
                tokio::task::yield_now().await;
                continue;
            }
            if let Some(reason) = &chunk.error {
                warn!(task = %self.task_id, reason = %reason, "engine failed task mid-stream; emitting SSE error");
                tokio::task::yield_now().await;
                return Some(error_frame(&self.task_id, reason));
            }
            record_stream_tokens(&chunk, &self.stats, &self.completion);
            // Latest count from any chunk, before the stop check (D14).
            if let Some(n) = chunk.prompt_tokens {
                self.prompt_tokens = n;
            }
            if chunk.is_final {
                self.finished =
                    chunk.finish_reason != Some(cascadia_types::FinishReason::Cancelled);
            }
            // Special-token framing (Inkling) → textual delimiters, same
            // translation as the non-streaming path.
            let text = match self.marker.as_mut() {
                Some(f) => {
                    let mut t = f.feed(&chunk);
                    if chunk.is_final {
                        t.push_str(&f.finish());
                    }
                    t
                }
                None => chunk.text.clone(),
            };
            let mut released = self.stop.push(&text);
            let n_tokens = std::mem::take(&mut self.held_tokens) + chunk.token_count();
            let finish = if self.stop.stopped() {
                // Cut at the match: end the choice here and drop the engine
                // stream, which cancels the task (M1 D11).
                self.finished = true;
                self.chunks = None;
                self.phase = ToolStreamPhase::Tail;
                Some("stop")
            } else if chunk.is_final {
                released.push_str(&self.stop.finish());
                self.phase = ToolStreamPhase::Tail;
                Some(
                    chunk
                        .finish_reason
                        .map(|f| f.as_openai_str())
                        .unwrap_or("stop"),
                )
            } else {
                None
            };
            if finish.is_none() && self.stop.is_active() && released.is_empty() {
                // All of it may begin a stop sequence: hold it (and its
                // token count) for the next frame.
                self.held_tokens = n_tokens;
                continue;
            }
            let completion_tokens = if self.stop.stopped() {
                None
            } else {
                chunk.completion_tokens
            };
            let frame = self.frame(released, n_tokens, completion_tokens, finish);
            // Yield between chunks so the body-sink task can drain this
            // frame to TCP (see the note in `stream_completion`).
            tokio::task::yield_now().await;
            return Some(Bytes::from(frame));
        }
    }

    fn frame(
        &mut self,
        mut text: String,
        n_tokens: u32,
        completion_tokens: Option<u32>,
        finish_reason: Option<&str>,
    ) -> String {
        match &mut self.kind {
            TextStreamKind::Chat => content_frame(
                &self.task_id,
                &self.model,
                &text,
                n_tokens,
                completion_tokens,
                finish_reason,
            ),
            TextStreamKind::Completion { echo } => {
                if let Some(prefix) = echo.take() {
                    text.insert_str(0, &prefix);
                }
                let payload = serde_json::json!({
                    "id": self.task_id,
                    "object": "text_completion",
                    "created": now_unix(),
                    "model": self.model,
                    "n_tokens": n_tokens,
                    "completion_tokens": completion_tokens,
                    "choices": [{
                        "index": 0,
                        "text": text,
                        "finish_reason": finish_reason,
                    }],
                });
                format!("data: {payload}\n\n")
            }
        }
    }

    fn tail(&self) -> Bytes {
        let object = match self.kind {
            TextStreamKind::Chat => "chat.completion.chunk",
            TextStreamKind::Completion { .. } => "text_completion",
        };
        let usage = (self.include_usage && self.finished)
            .then(|| (self.prompt_tokens, self.completion.load(Ordering::Relaxed)));
        usage_tail(&self.task_id, &self.model, object, usage)
    }
}

/// The SSE response around a stream body. Raw `Body::from_stream`, not
/// axum's `Sse`: see the batching note in [`stream_completion`].
fn sse_response<S>(body: S) -> axum::response::Response
where
    S: Stream<Item = Bytes> + Send + 'static,
{
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(
            body.map(Ok::<Bytes, std::convert::Infallible>),
        ))
        .unwrap()
        .into_response()
}

/// Body state of a tools-active chat stream. Owns the permit + in-flight
/// guard through `chunks`, so dropping the body (client disconnect) releases
/// both and cancels the engine task, exactly like the plain stream.
struct ToolStream {
    /// `None` once a stop match has dropped (and so cancelled) the task.
    chunks: Option<EngineChunks>,
    keepalive: KeepAlive,
    marker: Option<MarkerFilter>,
    /// Applied before the splitter, as the non-streaming path trims before
    /// parsing.
    stop: StopMatcher,
    splitter: ToolStreamSplitter,
    stats: Arc<ApiStats>,
    completion: std::sync::atomic::AtomicU32,
    /// Model tokens not yet reported in a frame's `n_tokens` (their text is
    /// still held).
    held_tokens: u32,
    prompt_tokens: u32,
    /// Completed normally (not failed, not cancelled): gates the usage frame.
    finished: bool,
    phase: ToolStreamPhase,
    task_id: String,
    model: String,
    include_usage: bool,
    /// Answer a `functions` request: `delta.function_call` (M1 D5).
    legacy_functions: bool,
}

enum ToolStreamPhase {
    Streaming,
    /// The choice is closed (or failed); the usage frame + `[DONE]` are next.
    Tail,
    Done,
}

impl ToolStream {
    /// Next SSE frame; `None` once `[DONE]` has been sent.
    async fn next_frame(&mut self) -> Option<Bytes> {
        loop {
            match self.phase {
                ToolStreamPhase::Streaming => {}
                ToolStreamPhase::Tail => {
                    self.phase = ToolStreamPhase::Done;
                    return Some(self.tail());
                }
                ToolStreamPhase::Done => return None,
            }
            let chunk = match self.keepalive.next(self.chunks.as_mut()).await {
                StreamNext::Idle => {
                    return Some(Bytes::from_static(SSE_KEEPALIVE_COMMENT.as_bytes()))
                }
                StreamNext::Chunk(Some(chunk)) => chunk,
                StreamNext::Chunk(None) => {
                    // Ended without a final chunk (cancelled): settle what
                    // is held instead of dropping it.
                    self.phase = ToolStreamPhase::Tail;
                    let rest = self.stop.finish();
                    return Some(self.close_choice(&rest, None, "stop"));
                }
            };
            // Progress chunks carry nothing to send or count.
            if chunk.is_progress() {
                tokio::task::yield_now().await;
                continue;
            }
            // The 200 headers are already on the wire: surface the failure
            // as an SSE error event. No tool_calls or usage follow it.
            if let Some(reason) = &chunk.error {
                warn!(task = %self.task_id, reason = %reason, "engine failed tool task mid-stream; SSE error");
                self.phase = ToolStreamPhase::Tail;
                tokio::task::yield_now().await;
                return Some(error_frame(&self.task_id, reason));
            }
            record_stream_tokens(&chunk, &self.stats, &self.completion);
            self.held_tokens += chunk.token_count();
            // Latest count from any chunk, before the stop check (D14).
            if let Some(n) = chunk.prompt_tokens {
                self.prompt_tokens = n;
            }
            let translated = self.marker.as_mut().map(|f| {
                let mut t = f.feed(&chunk);
                if chunk.is_final {
                    t.push_str(&f.finish());
                }
                t
            });
            let mut text = self.stop.push(translated.as_deref().unwrap_or(&chunk.text));
            let frame = if self.stop.stopped() {
                // Cut at the match: close the choice on what came before it
                // and drop the engine stream, which cancels the task (M1 D11).
                self.finished = true;
                self.chunks = None;
                self.phase = ToolStreamPhase::Tail;
                Some(self.close_choice(&text, None, "stop"))
            } else if chunk.is_final {
                text.push_str(&self.stop.finish());
                self.finished =
                    chunk.finish_reason != Some(cascadia_types::FinishReason::Cancelled);
                self.phase = ToolStreamPhase::Tail;
                let reason = chunk
                    .finish_reason
                    .map(|f| f.as_openai_str())
                    .unwrap_or("stop");
                Some(self.close_choice(&text, chunk.completion_tokens, reason))
            } else {
                let content = self.splitter.feed(&text);
                if content.is_empty() {
                    None
                } else {
                    let n_tokens = std::mem::take(&mut self.held_tokens);
                    Some(Bytes::from(content_frame(
                        &self.task_id,
                        &self.model,
                        content,
                        n_tokens,
                        chunk.completion_tokens,
                        None,
                    )))
                }
            };
            // Same per-chunk yield as the plain stream (see the comment
            // there): lets the body sink drain each frame to TCP.
            tokio::task::yield_now().await;
            if let Some(frame) = frame {
                return Some(frame);
            }
        }
    }

    /// Close the choice: the rest as content with the engine's
    /// finish_reason, or the still-releasable content, one indexed
    /// tool_calls delta, and a `tool_calls` finish — for a `functions`
    /// request, one `function_call` delta (the first call) and a
    /// `function_call` finish instead.
    fn close_choice(
        &mut self,
        text: &str,
        completion_tokens: Option<u32>,
        finish_reason: &str,
    ) -> Bytes {
        let n_tokens = std::mem::take(&mut self.held_tokens);
        let (task_id, model) = (&self.task_id, &self.model);
        let legacy = self.legacy_functions;
        let out = match self.splitter.finish(text) {
            ToolStreamEnd::Content(content) => content_frame(
                task_id,
                model,
                content,
                n_tokens,
                completion_tokens,
                Some(finish_reason),
            ),
            ToolStreamEnd::ToolCalls { content, calls } => {
                // The held tokens are the call's: they ride the tool_calls
                // delta, so each model token is counted in exactly one frame.
                let mut out = String::new();
                if !content.is_empty() {
                    out.push_str(&content_frame(task_id, model, content, 0, None, None));
                }
                let (call_delta, finish_reason) = if legacy {
                    let first = calls.into_iter().next().map(|c| c.function);
                    (
                        serde_json::json!({ "role": "assistant", "function_call": first }),
                        "function_call",
                    )
                } else {
                    let tool_calls: Vec<serde_json::Value> = calls
                        .into_iter()
                        .enumerate()
                        .map(|(i, c)| {
                            serde_json::json!({
                                "index": i, "id": c.id, "type": c.r#type,
                                "function": { "name": c.function.name, "arguments": c.function.arguments },
                            })
                        })
                        .collect();
                    (
                        serde_json::json!({ "role": "assistant", "tool_calls": tool_calls }),
                        "tool_calls",
                    )
                };
                let delta = serde_json::json!({
                    "id": task_id, "object": "chat.completion.chunk",
                    "created": now_unix(), "model": model,
                    "n_tokens": n_tokens, "completion_tokens": completion_tokens,
                    "choices": [{ "index": 0,
                        "delta": call_delta,
                        "finish_reason": serde_json::Value::Null }],
                });
                let finish = serde_json::json!({
                    "id": task_id, "object": "chat.completion.chunk",
                    "created": now_unix(), "model": model,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason }],
                });
                out.push_str(&format!("data: {delta}\n\ndata: {finish}\n\n"));
                out
            }
        };
        Bytes::from(out)
    }

    /// OpenAI `include_usage` frame (successful completions only) + `[DONE]`.
    fn tail(&self) -> Bytes {
        let usage = (self.include_usage && self.finished)
            .then(|| (self.prompt_tokens, self.completion.load(Ordering::Relaxed)));
        usage_tail(&self.task_id, &self.model, "chat.completion.chunk", usage)
    }
}

/// One `chat.completion.chunk` content delta, in the plain stream's shape.
fn content_frame(
    task_id: &str,
    model: &str,
    content: &str,
    n_tokens: u32,
    completion_tokens: Option<u32>,
    finish_reason: Option<&str>,
) -> String {
    let payload = serde_json::json!({
        "id": task_id,
        "object": "chat.completion.chunk",
        "created": now_unix(),
        "model": model,
        "n_tokens": n_tokens,
        "completion_tokens": completion_tokens,
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant", "content": content },
            "finish_reason": finish_reason,
        }],
    });
    format!("data: {payload}\n\n")
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
        assert_eq!(c.token_count(), 8);

        // Same shape WITHOUT n_tokens falls back to 1 (the pre-#55 behavior
        // that under-counted ov-genai) — documents why the engine must set it.
        let c = Chunk::final_marker("t", "The capital of France is Paris.");
        assert_eq!(c.token_count(), 1);

        // Empty final marker (mock/runtime) contributes 0 — no phantom token.
        let c = Chunk::final_marker("t", "");
        assert_eq!(c.token_count(), 0);

        // A normal per-token chunk counts as 1.
        let c = Chunk::token("t", 123, "Paris");
        assert_eq!(c.token_count(), 1);
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

    // Qwen3.x templates group consecutive tool results into one user turn with
    // `loop.previtem` / `loop.nextitem`. Without minijinja's
    // `adjacent_loop_items` feature both are undefined: a tool result followed
    // by anything failed to render (HTTP 500), and the first tool result lost
    // its `<|im_start|>user` header.
    #[test]
    fn tool_results_group_with_adjacent_loop_items() {
        let env = build_chat_env(concat!(
            "{%- for message in messages %}",
            "{%- if message.role == 'tool' %}",
            "{%- if loop.previtem and loop.previtem.role != 'tool' %}{{- '<|im_start|>user' }}{%- endif %}",
            "{{- '\\n<tool_response>' + message.content + '</tool_response>' }}",
            "{%- if not loop.last and loop.nextitem.role != 'tool' %}{{- '<|im_end|>\\n' }}",
            "{%- elif loop.last %}{{- '<|im_end|>\\n' }}{%- endif %}",
            "{%- else %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}",
            "{%- endif %}{%- endfor %}",
        ))
        .unwrap();
        let render = |msgs: &[ChatMessage]| {
            render_with_chat_env(&env, msgs, "", "", false, None, None).unwrap()
        };

        // A tool result followed by a user turn.
        let out = render(&[
            msg("user", "q"),
            msg("assistant", ""),
            msg("tool", "r1"),
            msg("user", "next"),
        ]);
        assert!(
            out.contains("<|im_start|>user\n<tool_response>r1</tool_response><|im_end|>\n<|im_start|>user\nnext"),
            "{out}"
        );

        // Two tool results in a row share one user turn.
        let out = render(&[
            msg("user", "q"),
            msg("assistant", ""),
            msg("tool", "r1"),
            msg("tool", "r2"),
        ]);
        assert!(
            out.ends_with("<|im_start|>user\n<tool_response>r1</tool_response>\n<tool_response>r2</tool_response><|im_end|>\n"),
            "{out}"
        );

        // Two tool rounds.
        let out = render(&[
            msg("user", "q"),
            msg("assistant", ""),
            msg("tool", "r1"),
            msg("assistant", ""),
            msg("tool", "r2"),
        ]);
        assert_eq!(
            out.matches("<|im_start|>user\n<tool_response>").count(),
            2,
            "{out}"
        );
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
        make_template_app(
            "{% if tools %}{{ tools | tojson }}{% endif %}\
             {% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}",
        )
        .await
    }

    /// A mock router whose model carries chat template `template`.
    async fn make_template_app(template: &str) -> Router {
        let runner = Runner::new(Box::new(MockBuilder::new()));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("mock-model", "CPU"),
            )
            .await
            .unwrap();
        let cfg = Config {
            chat_template: ChatTemplateConfig {
                template: Some(template.into()),
                bos_token: None,
                eos_token: None,
            },
            ..Config::default()
        };
        make_router_with_config(Arc::new(runner), "mock-model", cfg)
    }

    /// A subprocess-style stream: final usage includes hidden thinking/tools;
    /// transport fragments deliberately contribute zero model-token counts.
    struct UsageTailEngine {
        text: String,
        fail: bool,
        pending: Option<cascadia_types::TaskId>,
    }

    impl cascadia_engine::Engine for UsageTailEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, task: GenerationTask) -> cascadia_engine::EngineResult<()> {
            self.pending = Some(task.task_id);
            Ok(())
        }
        fn step(&mut self) -> cascadia_engine::EngineResult<Vec<(cascadia_types::TaskId, Chunk)>> {
            let Some(id) = self.pending.take() else {
                return Ok(Vec::new());
            };
            let final_chunk = if self.fail {
                Chunk::error(&id, "child failed before usage")
            } else {
                Chunk::final_marker(&id, "")
                    .with_n_tokens(0)
                    .with_prompt_tokens(42)
                    .with_completion_tokens(38)
            };
            Ok(vec![
                (id.clone(), Chunk::progress(id.clone())),
                (
                    id.clone(),
                    Chunk::token(&id, 0, &self.text).with_n_tokens(0),
                ),
                (id, final_chunk),
            ])
        }
    }

    struct UsageTailBuilder {
        text: String,
        fail: bool,
    }

    #[::async_trait::async_trait]
    impl cascadia_engine::Builder for UsageTailBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> cascadia_engine::EngineResult<()> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> cascadia_engine::EngineResult<cascadia_engine::LoadStream> {
            Ok(Box::pin(stream::empty()))
        }
        fn build(
            self: Box<Self>,
        ) -> cascadia_engine::EngineResult<Box<dyn cascadia_engine::Engine>> {
            Ok(Box::new(UsageTailEngine {
                text: self.text,
                fail: self.fail,
                pending: None,
            }))
        }
    }

    async fn usage_tail_app(tool: bool, fail: bool) -> (Router, Arc<ApiStats>) {
        let text = if tool {
            r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#
        } else {
            "visible answer"
        };
        let runner = Runner::new(Box::new(UsageTailBuilder {
            text: text.into(),
            fail,
        }));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("mock-model", "CPU"),
            )
            .await
            .unwrap();
        let stats = Arc::new(ApiStats::default());
        let cfg = Config {
            chat_template: ChatTemplateConfig {
                template: Some("{% for m in messages %}{{ m.content }}{% endfor %}".into()),
                bos_token: None,
                eos_token: None,
            },
            ..Config::default()
        };
        (
            make_router_with_stats(Arc::new(runner), "mock-model", cfg, stats.clone()),
            stats,
        )
    }

    #[tokio::test]
    async fn authoritative_usage_reaches_chat_legacy_tools_and_stats() {
        for (legacy, tool, streaming) in [
            (false, false, false),
            (false, false, true),
            (true, false, false),
            (true, false, true),
            (false, true, false),
            (false, true, true),
        ] {
            let (app, stats) = usage_tail_app(tool, false).await;
            let mut request = serde_json::json!({
                "model": "mock-model", "stream": streaming,
                "stream_options": {"include_usage": true},
            });
            if legacy {
                request["prompt"] = serde_json::json!("hi");
            } else {
                request["messages"] = serde_json::json!([{"role":"user","content":"hi"}]);
            }
            if tool {
                request["tools"] =
                    serde_json::json!([{"type":"function","function":{"name":"get_weather"}}]);
            }
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(if legacy {
                            "/v1/completions"
                        } else {
                            "/v1/chat/completions"
                        })
                        .header("content-type", "application/json")
                        .body(Body::from(request.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = to_bytes(response.into_body(), 65536).await.unwrap();
            let usage = if streaming {
                let body = String::from_utf8(bytes.to_vec()).unwrap();
                assert!(body.trim_end().ends_with("data: [DONE]"));
                let events: Vec<Value> = body
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter(|line| *line != "[DONE]")
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
                let tails: Vec<_> = events
                    .iter()
                    .filter(|event| event.get("usage").is_some())
                    .collect();
                assert_eq!(tails.len(), 1, "{body}");
                assert!(tails[0]["choices"].as_array().unwrap().is_empty());
                assert_eq!(events.last().unwrap()["usage"], tails[0]["usage"]);
                if tool {
                    assert!(body.contains("\"finish_reason\":\"tool_calls\""), "{body}");
                }
                tails[0]["usage"].clone()
            } else {
                let response: Value = serde_json::from_slice(&bytes).unwrap();
                if tool {
                    assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
                }
                response["usage"].clone()
            };
            assert_eq!(
                usage,
                serde_json::json!({"prompt_tokens":42,"completion_tokens":38,"total_tokens":80})
            );
            assert_eq!(stats.tokens_total.load(Ordering::Relaxed), 38);
        }
    }

    #[tokio::test]
    async fn failed_streams_never_emit_successful_usage_tail() {
        for (legacy, tool) in [(false, false), (true, false), (false, true)] {
            let (app, stats) = usage_tail_app(tool, true).await;
            let mut request = serde_json::json!({"model":"mock-model","stream":true,
                "stream_options":{"include_usage":true}});
            if legacy {
                request["prompt"] = serde_json::json!("hi");
            } else {
                request["messages"] = serde_json::json!([{"role":"user","content":"hi"}]);
            }
            if tool {
                request["tools"] =
                    serde_json::json!([{"type":"function","function":{"name":"get_weather"}}]);
            }
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(if legacy {
                            "/v1/completions"
                        } else {
                            "/v1/chat/completions"
                        })
                        .header("content-type", "application/json")
                        .body(Body::from(request.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            let body = String::from_utf8(
                to_bytes(response.into_body(), 65536)
                    .await
                    .unwrap()
                    .to_vec(),
            )
            .unwrap();
            assert!(body.contains("\"object\":\"error\""), "{body}");
            assert!(!body.contains("\"usage\""), "{body}");
            assert_eq!(stats.tokens_total.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn streaming_stats_replace_not_add_lower_authoritative_total() {
        let stats = ApiStats::default();
        let completion = std::sync::atomic::AtomicU32::new(0);
        record_stream_tokens(
            &Chunk::token("t", 0, "partial").with_n_tokens(5),
            &stats,
            &completion,
        );
        record_stream_tokens(
            &Chunk::final_marker("t", "").with_completion_tokens(3),
            &stats,
            &completion,
        );
        assert_eq!(completion.load(Ordering::Relaxed), 3);
        assert_eq!(stats.tokens_total.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn render_failure_5xx_does_not_latch_readiness() {
        // A template that cannot render is a server-side fault worth a 5xx, but
        // it is raised before the engine is touched, so it says nothing about
        // pipeline health. Letting it flip `ready` drains a healthy node for
        // good: behind a load balancer gating on /health, the traffic that
        // would have produced the recovering 2xx never arrives.
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
                // Renders only when tools are present — and then fails, because
                // there is no such filter. Tool-less requests still fall back to
                // the legacy formatter, which is why the request below sends one.
                template: Some("{% if tools %}{{ tools | no_such_filter }}{% endif %}".into()),
                bos_token: None,
                eos_token: None,
            },
            ..Config::default()
        };
        let app = make_router_with_config(Arc::new(runner), "mock-model", cfg);

        let payload = serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object"}}
            }],
            "stream": false,
        });
        let resp = app
            .clone()
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
        assert_eq!(
            resp.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "a render failure should stay a 5xx so operators see it"
        );

        let health = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            health.status(),
            StatusCode::OK,
            "a pre-engine 5xx must not latch readiness"
        );
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

    /// Posts a 5-word chat prompt with the given limit fields and asserts the
    /// mock stopped at `expected` tokens for `finish_reason`, with a body that
    /// matches the spec.
    async fn assert_chat_limit(limits: serde_json::Value, expected: u64, finish: &str) {
        let mut payload = serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "alpha bravo charlie delta echo"}],
        });
        payload
            .as_object_mut()
            .unwrap()
            .extend(limits.as_object().unwrap().clone());
        let (status, v) = post_chat(make_app().await, payload).await;
        assert_eq!(status, StatusCode::OK, "{limits}: {v}");
        assert_eq!(v["usage"]["completion_tokens"], expected, "{limits}: {v}");
        assert_eq!(v["choices"][0]["finish_reason"], finish, "{limits}: {v}");
        crate::spec_schema::assert_spec("CreateChatCompletionResponse", &v);
    }

    #[tokio::test]
    async fn max_completion_tokens_limits_chat_completion() {
        assert_chat_limit(serde_json::json!({"max_completion_tokens": 2}), 2, "length").await;
    }

    #[tokio::test]
    async fn max_tokens_still_limits_chat_completion() {
        assert_chat_limit(serde_json::json!({"max_tokens": 2}), 2, "length").await;
    }

    #[tokio::test]
    async fn max_completion_tokens_wins_over_max_tokens() {
        // D1: the current field beats the deprecated one.
        assert_chat_limit(
            serde_json::json!({"max_tokens": 4, "max_completion_tokens": 2}),
            2,
            "length",
        )
        .await;
    }

    #[tokio::test]
    async fn null_token_limits_mean_unset() {
        // D9: `null` is the same as omitting the field.
        assert_chat_limit(
            serde_json::json!({"max_tokens": 2, "max_completion_tokens": null}),
            2,
            "length",
        )
        .await;
        assert_chat_limit(
            serde_json::json!({"max_tokens": null, "max_completion_tokens": 2}),
            2,
            "length",
        )
        .await;
        // Both unset: the 5-word echo finishes well under the fallback limit.
        assert_chat_limit(
            serde_json::json!({"max_tokens": null, "max_completion_tokens": null}),
            5,
            "stop",
        )
        .await;
    }

    #[test]
    fn requested_max_tokens_prefers_max_completion_tokens() {
        let limit = |extra| chat_request(extra).requested_max_tokens();
        assert_eq!(limit(serde_json::json!({})), None);
        assert_eq!(limit(serde_json::json!({"max_tokens": 4})), Some(4));
        assert_eq!(
            limit(serde_json::json!({"max_completion_tokens": 2})),
            Some(2)
        );
        assert_eq!(
            limit(serde_json::json!({"max_tokens": 4, "max_completion_tokens": 2})),
            Some(2)
        );
    }

    #[tokio::test]
    async fn streaming_honours_max_completion_tokens() {
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
                            "messages": [{"role": "user", "content": "alpha bravo charlie delta echo"}],
                            "max_completion_tokens": 2,
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
        let events = sse_events(&text);
        let content_frames = events
            .iter()
            .filter(|e| {
                e["choices"][0]["delta"]["content"]
                    .as_str()
                    .is_some_and(|s| !s.is_empty())
            })
            .count();
        assert_eq!(content_frames, 2, "{text}");
        let usage = events
            .iter()
            .find(|e| e["usage"].is_object())
            .expect("usage frame");
        assert_eq!(usage["usage"]["completion_tokens"], 2, "{text}");
        let finish = events
            .iter()
            .find_map(|e| e["choices"][0]["finish_reason"].as_str())
            .expect("finish_reason frame");
        assert_eq!(finish, "length", "{text}");
    }

    // Response bodies must match the pinned OpenAI spec (see spec_schema.rs).

    #[tokio::test]
    async fn chat_completion_response_matches_openai_spec() {
        let (status, v) = post_chat(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "alpha bravo charlie"}],
                "max_tokens": 2,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        crate::spec_schema::assert_spec("CreateChatCompletionResponse", &v);
    }

    #[tokio::test]
    async fn completion_response_matches_openai_spec() {
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
                            "max_tokens": 2,
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 16384).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        crate::spec_schema::assert_spec("CreateCompletionResponse", &v);
    }

    #[tokio::test]
    async fn list_models_response_matches_openai_spec() {
        let response = make_app()
            .await
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        crate::spec_schema::assert_spec("ListModelsResponse", &v);
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
            function_call: None,
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

        // A truncated pair must drop the whole call, not fire it with the
        // arguments gathered so far. Each of the three exit points gets a case.
        // The realistic trigger is the second one: an <arg_value> holding the
        // literal </tool_call> ends the block early, so `body` never closes.
        // Each case is a properly CLOSED block whose contents are cut short —
        // the block only exists at all because a `</tool_call>` was found, so
        // truncating that terminator away would test nothing.
        for truncated in [
            // missing </arg_key>
            "<tool_call>send<arg_key>to</arg_key><arg_value>a@b.com</arg_value>\
             <arg_key>body</tool_call>",
            // <arg_value> never opens
            "<tool_call>send<arg_key>to</arg_key><arg_value>a@b.com</arg_value>\
             <arg_key>body</arg_key></tool_call>",
            // missing </arg_value>, because the value itself contained the
            // literal </tool_call> and ended the block early — what happens
            // whenever an agent writes about the tool-call format itself
            "<tool_call>send<arg_key>to</arg_key><arg_value>a@b.com</arg_value>\
             <arg_key>body</arg_key><arg_value>see </tool_call> the docs</arg_value></tool_call>",
        ] {
            assert!(
                parse_tool_calls(truncated).is_none(),
                "a truncated argument pair must drop the call rather than fire it \
                 with `body` missing: {truncated}"
            );
        }

        // The complete form of that same call still parses — the guard above
        // rejects truncation, not the arguments themselves.
        let calls = parse_tool_calls(
            "<tool_call>send<arg_key>to</arg_key><arg_value>a@b.com</arg_value>\
             <arg_key>body</arg_key><arg_value>hello</arg_value></tool_call>",
        )
        .expect("a complete two-argument block must still parse");
        let args = serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap();
        assert_eq!(args["to"], "a@b.com");
        assert_eq!(args["body"], "hello");

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
    fn tojson_honours_sort_keys_and_separators_kwargs() {
        // json.dumps semantics: sort_keys orders every object's keys
        // (recursively), separators=(item, key) replaces the default
        // (", ", ": "). Inkling's template uses both for tool specs.
        let out = tojson_of(
            serde_json::json!({"b": {"d": 1, "c": [1, {"z": 0, "y": 0}]}, "a": 2}),
            &[("sort_keys", true.into())],
        )
        .expect("sort_keys=True renders");
        assert_eq!(
            out,
            "{\"a\": 2, \"b\": {\"c\": [1, {\"y\": 0, \"z\": 0}], \"d\": 1}}"
        );
        assert!(
            tojson_of(serde_json::json!({}), &[("sort_keys", false.into())]).is_ok(),
            "sort_keys=False is the default and must be accepted"
        );
        let out = tojson_of(
            serde_json::json!({"a": [1, 2], "b": "x"}),
            &[("separators", minijinja::value::Value::from(vec![",", ":"]))],
        )
        .expect("separators renders");
        assert_eq!(out, "{\"a\":[1,2],\"b\":\"x\"}");
        let err = tojson_of(
            serde_json::json!({}),
            &[("separators", minijinja::value::Value::from(vec![","]))],
        )
        .expect_err("a separators tuple must have two entries");
        assert!(err.to_string().contains("separators"), "{err}");
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

        let err = render_or_fallback(
            Some(&env),
            &msgs,
            "",
            "",
            true,
            EffortArgs {
                mapped: None,
                raw: None,
                ..Default::default()
            },
            Some(&tools),
        )
        .expect_err("a tools request must not fall back");
        assert!(matches!(err, PromptRenderError::Failed(_)), "{err:?}");

        // Same broken template, no tools: fallback is still allowed, because a
        // degraded prompt beats a dead endpoint.
        let out = render_or_fallback(
            Some(&env),
            &msgs,
            "",
            "",
            true,
            EffortArgs {
                mapped: None,
                raw: None,
                ..Default::default()
            },
            None,
        )
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
    fn chat_turns_forward_tool_fields_as_wire_json() {
        let req = chat_request(serde_json::json!({
            "tools": [{"type":"function","function":{"name":"get_weather",
                "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
            "messages": [
                {"role":"assistant","content":null,
                 "tool_calls":[{"id":"call_1","type":"function",
                    "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
                {"role":"tool","tool_call_id":"call_1","name":"get_weather",
                 "content":"{\"temp_c\":18}"},
                {"role":"user","content":"thanks"}
            ]
        }));
        let turns = chat_turns(&req);
        assert_eq!(turns.len(), 3);
        let calls = turns[0].tool_calls.as_ref().expect("tool_calls forwarded");
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(
            calls[0]["function"]["arguments"], "{\"city\":\"Paris\"}",
            "arguments stay the OpenAI string form"
        );
        assert_eq!(turns[1].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(turns[1].name.as_deref(), Some("get_weather"));
        let tools = chat_tools(&req).expect("tools forwarded");
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
    }

    #[tokio::test]
    async fn oversized_tool_turns_get_413() {
        // max_prompt_bytes small enough that the rendered prompt passes but
        // the tool-result turn does not — content + tool fields must count.
        let runner = Runner::new(Box::new(MockBuilder::new()));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("mock-model", "CPU"),
            )
            .await
            .unwrap();
        let cfg = Config {
            max_prompt_bytes: 128,
            // roles-only template: the rendered prompt stays tiny while the
            // tool-result turn is huge, so the messages check is what fires
            chat_template: ChatTemplateConfig {
                template: Some(
                    "{% if tools %}{{ tools | tojson }}{% endif %}\
                     {% for m in messages %}{{ m.role }}\n{% endfor %}"
                        .into(),
                ),
                bos_token: None,
                eos_token: None,
            },
            ..Config::default()
        };
        let app = make_router_with_config(Arc::new(runner), "mock-model", cfg);
        let big = "x".repeat(4096);
        let payload = serde_json::json!({
            "model": "mock-model",
            "tools": [{"type":"function","function":{"name":"f"}}],
            "messages": [
                {"role":"user","content":"hi"},
                {"role":"tool","tool_call_id":"c1","content": big},
            ],
        });
        let (status, body) = post_chat(app, payload).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
        assert!(
            body["error"].as_str().unwrap_or("").contains("messages"),
            "error must name the messages payload: {body}"
        );
    }

    /// Qwen3.8's template accepts only xhigh/medium/low and raises for anything
    /// else; the API's GLM-mapped default ("high") must not turn into a 400.
    #[test]
    fn template_rejecting_the_mapped_effort_is_rendered_with_it_undefined() {
        const T: &str = "{%- if reasoning_effort is defined and reasoning_effort not in ['xhigh', 'medium', 'low'] -%}\
{{ raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}\
{%- endif -%}effort={{ reasoning_effort | default('xhigh') }}";
        let cfg = ChatTemplateConfig {
            template: Some(T.to_string()),
            bos_token: None,
            eos_token: None,
        };
        let renderer = ChatPromptRenderer::new(&cfg);
        let msgs = [msg("user", "hi")];
        // The mapped default is rejected by the template → rendered undefined.
        let out = renderer
            .render_with_effort(&msgs, None, true, Some("high"))
            .expect("second render with effort undefined");
        assert_eq!(out, "effort=xhigh");
        // An accepted value goes through untouched.
        let out = renderer
            .render_with_effort(&msgs, None, true, Some("low"))
            .unwrap();
        assert_eq!(out, "effort=low");
        // Through the API: the caller said "low", the GLM mapping made it
        // "high"; the template rejects "high" and gets the caller's "low".
        let env = build_chat_env(T).unwrap();
        let out = render_or_fallback(
            Some(&env),
            &msgs,
            "",
            "",
            true,
            EffortArgs {
                mapped: Some("high"),
                raw: Some("low"),
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(out, "effort=low");
        // The caller sent the API's own word: nothing of theirs is lost by
        // falling through to the template default.
        let out = render_or_fallback(
            Some(&env),
            &msgs,
            "",
            "",
            true,
            EffortArgs {
                mapped: Some("high"),
                raw: Some("high"),
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(out, "effort=xhigh");
        // A caller word of their own that the template rejects is a 400, not a
        // silent downgrade to the template default: "hgih" is a typo, and
        // answering at xhigh hides that the parameter was discarded.
        let err = render_or_fallback(
            Some(&env),
            &msgs,
            "",
            "",
            true,
            EffortArgs {
                mapped: Some("high"),
                raw: Some("hgih"),
                ..Default::default()
            },
            None,
        )
        .expect_err("the caller's own rejected word must surface");
        assert!(
            matches!(&err, PromptRenderError::Rejected(m) if m.contains("hgih")),
            "expected the template's message about the caller's word: {err:?}"
        );
        // A rejection that is not about the effort is still surfaced.
        let env = build_chat_env("{{ raise_exception('no tools here') }}").unwrap();
        assert!(matches!(
            render_or_fallback(
                Some(&env),
                &msgs,
                "",
                "",
                true,
                EffortArgs {
                    mapped: Some("high"),
                    raw: Some("low"),
                    ..Default::default()
                },
                None
            ),
            Err(PromptRenderError::Rejected(_))
        ));
    }

    /// End to end over HTTP: the handler must hand `req.reasoning_effort` to
    /// `render_or_fallback` as the raw word, or a caller's "low" is lost when
    /// the family's template rejects the GLM mapping. The mock engine echoes
    /// the prompt back word by word, so the completion text *is* the rendered
    /// prompt.
    #[tokio::test]
    async fn http_request_reaches_the_template_with_the_callers_effort() {
        const T: &str = "{%- if reasoning_effort is defined and reasoning_effort not in ['xhigh', 'medium', 'low'] -%}\
{{ raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}\
{%- endif -%}effort={{ reasoning_effort | default('xhigh') }}";
        let runner = Runner::new(Box::new(MockBuilder::new()));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("mock-model", "CPU"),
            )
            .await
            .unwrap();
        let cfg = Config {
            chat_template: ChatTemplateConfig {
                template: Some(T.to_string()),
                bos_token: None,
                eos_token: None,
            },
            ..Config::default()
        };
        let app = make_router_with_config(Arc::new(runner), "mock-model", cfg);

        let (status, v) = post_chat(
            app,
            serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "low",
                "stream": false,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "body: {v}");
        let content = v["choices"][0]["message"]["content"].as_str().unwrap();
        assert_eq!(
            content.trim(),
            "effort=low",
            "the template saw the mapped word, not the caller's"
        );
    }

    /// The rejection match is a substring test against the template's own
    /// prose. A template that spells the parameter `reasoning_effort` must
    /// trigger the same retry, not the blanket 400 this ladder exists to fix.
    #[test]
    fn underscore_wording_in_the_rejection_still_triggers_the_retry() {
        const T: &str = "{%- if reasoning_effort is defined and reasoning_effort not in ['xhigh', 'medium', 'low'] -%}\
{{ raise_exception('reasoning_effort ' ~ reasoning_effort ~ ' is not supported') }}\
{%- endif -%}effort={{ reasoning_effort | default('xhigh') }}";
        let env = build_chat_env(T).unwrap();
        let msgs = [msg("user", "hi")];
        let out = render_or_fallback(
            Some(&env),
            &msgs,
            "",
            "",
            true,
            EffortArgs {
                mapped: Some("high"),
                raw: None,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(out, "effort=xhigh");
    }

    /// Inkling's template renders tool specs with
    /// `tojson(sort_keys=true, separators=(",", ":"))`; the filter must honour
    /// both (json.dumps semantics) instead of failing the render.
    #[test]
    fn tojson_honours_sort_keys_and_separators() {
        const T: &str = "{%- set spec = {\"type\": \"function\", \"name\": tools[0].function.name, \"description\": \"d\", \"parameters\": {\"b\": 1, \"a\": [2, 3]}} -%}\
{{ [spec] | tojson(sort_keys=true, separators=(\",\", \":\")) }}|{{ {\"b\": 1, \"a\": 2} | tojson(sort_keys=true) }}";
        let env = build_chat_env(T).unwrap();
        let tools = vec![Tool {
            r#type: "function".into(),
            function: serde_json::json!({"name": "get_weather"}),
        }];
        let msgs = [msg("user", "hi")];
        let out = render_or_fallback(
            Some(&env),
            &msgs,
            "",
            "",
            true,
            EffortArgs {
                mapped: None,
                raw: None,
                ..Default::default()
            },
            Some(&tools),
        )
        .unwrap();
        assert_eq!(
            out,
            "[{\"description\":\"d\",\"name\":\"get_weather\",\"parameters\":{\"a\":[2,3],\"b\":1},\"type\":\"function\"}]\
|{\"a\": 2, \"b\": 1}"
        );
    }

    /// Inkling's template turns the effort word into a numeric level and
    /// switches thinking off at 0: the caller's own word must reach it
    /// (the GLM mapping would send "none" as "high"), thinking-off with no
    /// effort must render level 0, and an omitted effort keeps the mapped
    /// default.
    #[test]
    fn effort_word_reaches_a_template_with_its_own_scale() {
        const T: &str = "{%- set effort_map = {\"none\": 0.0, \"minimal\": 0.1, \"low\": 0.2, \"medium\": 0.7, \"high\": 0.9, \"max\": 0.99} -%}\
{%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else \"high\" -%}\
{%- if eff not in effort_map -%}{{ raise_exception(\"Unknown reasoning_effort: \" ~ eff) }}{%- endif -%}\
level={{ effort_map[eff] }}";
        let env = build_chat_env(T).unwrap();
        let msgs = [msg("user", "hi")];
        assert!(
            template_speaks_effort_words(&env, "", ""),
            "the probe must classify an Inkling-shaped template as word-speaking"
        );
        let r = |thinking: bool, explicit: bool, mapped: Option<&str>, raw: Option<&str>| {
            let effort = EffortArgs {
                mapped,
                raw,
                explicit_thinking: explicit,
                template_words: true,
            };
            render_or_fallback(Some(&env), &msgs, "", "", thinking, effort, None)
        };
        // caller said "none" (thinking off by that alone): honoured, not
        // escalated to the mapped "high"
        assert_eq!(
            r(false, false, Some("high"), Some("none")).unwrap(),
            "level=0.0"
        );
        assert_eq!(
            r(true, false, Some("high"), Some("low")).unwrap(),
            "level=0.2"
        );
        // thinking off, no effort given: level 0
        assert_eq!(r(false, true, Some("high"), None).unwrap(), "level=0.0");
        // omitted effort, thinking on: the mapped default
        assert_eq!(r(true, false, Some("high"), None).unwrap(), "level=0.9");
        // explicit enable_thinking:true beats "none": the toggle wins
        assert_eq!(
            r(true, true, Some("high"), Some("none")).unwrap(),
            "level=0.9"
        );
        // a caller word the template rejects is the caller's 400
        assert!(matches!(
            r(true, false, Some("high"), Some("bogus")),
            Err(PromptRenderError::Rejected(_))
        ));
    }

    /// GLM-5's template never rejects: anything but the literal "high" is
    /// Max. The probe must classify it as NOT word-speaking so the mapping
    /// keeps "low"/"none"/"medium" at High (the regression the review caught).
    #[test]
    fn glm_style_template_keeps_the_mapping() {
        const T: &str = "effort={{ 'high' if reasoning_effort is defined and reasoning_effort == 'high' else 'max' }}";
        let env = build_chat_env(T).unwrap();
        assert!(!template_speaks_effort_words(&env, "", ""));
        let msgs = [msg("user", "hi")];
        for raw in [
            Some("low"),
            Some("medium"),
            Some("none"),
            Some("bogus"),
            None,
        ] {
            let effort = EffortArgs {
                mapped: Some("high"),
                raw,
                explicit_thinking: false,
                template_words: false,
            };
            assert_eq!(
                render_or_fallback(Some(&env), &msgs, "", "", true, effort, None).unwrap(),
                "effort=high",
                "raw {raw:?} must stay at the mapped level on a GLM-style template"
            );
        }
        let effort = EffortArgs {
            mapped: Some("max"),
            ..Default::default()
        };
        assert_eq!(
            render_or_fallback(Some(&env), &msgs, "", "", true, effort, None).unwrap(),
            "effort=max"
        );
    }

    /// Inkling's special-token framing → the textual delimiters the rest of
    /// the API understands: `<think>…</think>` and a Hermes tool-call block.
    #[test]
    fn marker_filter_translates_thinking_text_and_tool_calls() {
        use cascadia_types::Chunk;
        let d = MarkerDialect {
            message_model: 200001,
            content_text: 200004,
            content_thinking: 200008,
            end_message: 200010,
            invoke_tool_json: 200049,
        };
        let tok = |id: i64, text: &str| {
            let mut c = Chunk::token(String::from("t"), id, text.to_string());
            c.token_ids = vec![id];
            c
        };
        // <|message_model|><|content_thinking|>plan<|end_message|><|message_model|><|content_text|>Paris<|end_message|>
        let mut f = MarkerFilter::new(d.clone());
        let mut out = String::new();
        for c in [
            tok(200001, ""),
            tok(200008, ""),
            tok(7, "plan"),
            tok(200010, ""),
            tok(200001, ""),
            tok(200004, ""),
            tok(9, "Paris"),
            tok(200010, ""),
        ] {
            out.push_str(&f.feed(&c));
        }
        out.push_str(&f.finish());
        assert_eq!(out, "<think>plan</think>Paris");
        // tool call: NAME<|content_invoke_tool_json|>{"name":…,"args":…}<|end_message|>
        let mut f = MarkerFilter::new(d.clone());
        let mut out = String::new();
        for c in [
            tok(200001, ""),
            tok(11, "get_"),
            tok(12, "weather"),
            tok(200049, ""),
            tok(13, "{\"name\":\"get_weather\","),
            tok(14, "\"args\":{\"city\":\"Paris\"}}"),
            tok(200010, ""),
        ] {
            let piece = f.feed(&c);
            // withheld until the closing marker: no partial name/JSON leaks
            if c.token_id != 200010 {
                assert_eq!(piece, "", "leaked {piece:?}");
            }
            out.push_str(&piece);
        }
        assert_eq!(
            out,
            "<tool_call>{\"arguments\":{\"city\":\"Paris\"},\"name\":\"get_weather\"}</tool_call>"
        );
        let calls = parse_tool_calls(&out).expect("Hermes block parses");
        assert_eq!(calls[0].function.name, "get_weather");
        // thinking-off answer with no framing markers passes straight through
        let mut f = MarkerFilter::new(d);
        assert_eq!(f.feed(&tok(9, "42")), "42");
        assert_eq!(f.finish(), "");
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
        let err = render_or_fallback(
            None,
            &msgs,
            "",
            "",
            true,
            EffortArgs {
                mapped: None,
                raw: None,
                ..Default::default()
            },
            Some(&tools),
        )
        .expect_err("no template + tools must not be answered");
        assert!(matches!(err, PromptRenderError::Rejected(_)), "{err:?}");

        assert!(
            render_or_fallback(
                None,
                &msgs,
                "",
                "",
                true,
                EffortArgs {
                    mapped: None,
                    raw: None,
                    ..Default::default()
                },
                None
            )
            .is_ok(),
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
        match render_or_fallback(
            Some(&env),
            &msgs,
            "",
            "",
            true,
            EffortArgs {
                mapped: None,
                raw: None,
                ..Default::default()
            },
            None,
        ) {
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

    #[tokio::test]
    async fn tool_choice_shape_is_validated_before_the_engine() {
        // a templated app, so the only thing that can 400 here is tool_choice
        let app = make_tool_app().await;
        for (choice, want) in [
            (serde_json::json!("sometimes"), StatusCode::BAD_REQUEST),
            (
                serde_json::json!({"type": "function", "function": {"name": "get_time"}}),
                StatusCode::BAD_REQUEST,
            ),
            (
                serde_json::json!({"type": "function"}),
                StatusCode::BAD_REQUEST,
            ),
            (
                serde_json::json!({"type": "function", "function": {"name": "get_weather"}}),
                StatusCode::OK,
            ),
            (serde_json::json!("required"), StatusCode::OK),
        ] {
            let payload = serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "weather in Paris?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                "tool_choice": choice,
                "stream": false
            });
            let resp = app
                .clone()
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
            assert_eq!(resp.status(), want, "{choice}");
        }
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
        // The call's own text is withheld, never sent as content.
        assert!(!body.contains("<tool_call>"), "marker leaked: {body}");
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

    /// Feed every piece but the last, finish with the last. Returns each
    /// released content piece (the finish content last) and parsed calls.
    fn run_splitter(pieces: &[&str], thinking: bool) -> (Vec<String>, Option<Vec<ToolCall>>) {
        let mut s = ToolStreamSplitter::new(thinking);
        let (last, body) = pieces.split_last().unwrap();
        let mut out: Vec<String> = body.iter().map(|p| s.feed(p).to_owned()).collect();
        match s.finish(last) {
            ToolStreamEnd::Content(c) => {
                out.push(c.to_owned());
                (out, None)
            }
            ToolStreamEnd::ToolCalls { content, calls } => {
                out.push(content.to_owned());
                (out, Some(calls))
            }
        }
    }

    const WEATHER_CALL: &str =
        r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#;

    #[test]
    fn tool_splitter_streams_plain_text_as_it_arrives() {
        let (out, calls) = run_splitter(&["Hello ", "world", "!", ""], false);
        assert_eq!(out, ["Hello ", "world", "!", ""]);
        assert!(calls.is_none());
        // A `<` that turns out not to open a marker is held, then released.
        let (out, calls) = run_splitter(&["a <", "b>bold</b>", ""], false);
        assert_eq!(out, ["a ", "<b>bold</b>", ""]);
        assert!(calls.is_none());
        // Leading whitespace is held across feeds until the first real
        // byte decides, then released with it.
        let (out, calls) = run_splitter(&[" ", "\n", "  ", "Hi", " there", ""], false);
        assert_eq!(out, ["", "", "", " \n  Hi", " there", ""]);
        assert!(calls.is_none());
    }

    #[test]
    fn tool_splitter_streams_text_before_a_split_marker_and_parses_the_call() {
        let call_rest = &WEATHER_CALL["<tool_".len()..];
        let (out, calls) = run_splitter(&["Let me check. <tool_", call_rest, "\n"], false);
        assert_eq!(out, ["Let me check. ", "", ""]);
        let calls = calls.expect("call parsed");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
            serde_json::json!({"city":"Paris"})
        );
    }

    #[test]
    fn tool_splitter_releases_a_malformed_call_as_content() {
        let pieces = ["Try <tool_call>{\"name\":", "\"oops\"", " and more"];
        let (out, calls) = run_splitter(&pieces, false);
        assert!(calls.is_none());
        assert_eq!(out[..2], ["Try ", ""]);
        assert_eq!(out.concat(), pieces.concat());
    }

    #[test]
    fn tool_splitter_holds_whole_output_dialects_until_the_end() {
        // Bare-JSON call: nothing streams, the call is parsed at the end.
        let (out, calls) = run_splitter(
            &["\n", "{\"name\":\"get_weather\",", "\"arguments\":{}}", ""],
            false,
        );
        assert_eq!(out, ["", "", "", ""]);
        assert_eq!(
            calls.expect("bare JSON call")[0].function.name,
            "get_weather"
        );
        // Bare JSON that is data, not a call: held, then released whole.
        let (out, calls) = run_splitter(&["{\"name\":", "\"Alice\",\"age\":30}", ""], false);
        assert!(calls.is_none());
        assert_eq!(out, ["", "", "{\"name\":\"Alice\",\"age\":30}"]);
        // Llama-3.1 `<|python_tag|>`, split mid-marker.
        let (out, calls) = run_splitter(
            &[
                "<|py",
                "thon_tag|>{\"name\":\"f\",",
                "\"parameters\":{}}",
                "",
            ],
            false,
        );
        assert_eq!(out, ["", "", "", ""]);
        assert_eq!(calls.expect("python_tag call")[0].function.name, "f");
    }

    #[test]
    fn tool_splitter_releases_a_scratchpad_draft_and_parses_the_real_call() {
        let draft = r#"<tool_call>{"name":"draft","arguments":{}}</tool_call>"#;
        // Draft held while reasoning, released once `</think>` (split across
        // chunks) closes the scratchpad; no real call follows.
        let pieces = ["<think>maybe ", draft, " no.</thi", "nk>Answer: 42", ""];
        let (out, calls) = run_splitter(&pieces, true);
        assert!(calls.is_none());
        assert_eq!(out[0], "<think>maybe ");
        assert_eq!(out[1], "");
        assert_eq!(out[2], "");
        assert_eq!(out[3], format!("{draft} no.</think>Answer: 42"));
        assert_eq!(out.concat(), pieces.concat());
        // Draft in the scratchpad, real call after it: only the real call.
        let real = r#"<tool_call>{"name":"real","arguments":{}}</tool_call>"#;
        let (out, calls) = run_splitter(&["<think>", draft, "</think>", real, ""], true);
        let calls = calls.expect("real call parsed");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "real");
        assert_eq!(out.concat(), format!("<think>{draft}</think>"));
        // Thinking off: `</think>` means nothing, the first call is the call.
        let (_, calls) = run_splitter(&["<think>", draft, "</think>", real, ""], false);
        assert_eq!(calls.expect("calls")[0].function.name, "draft");
    }

    #[test]
    fn tool_splitter_keeps_multibyte_chars_whole() {
        let call_rest = &WEATHER_CALL["<tool".len()..];
        let (out, calls) = run_splitter(&["héllo ", "日本", "語 <tool", call_rest, ""], false);
        assert_eq!(out.concat(), "héllo 日本語 ");
        assert!(calls.is_some());
        let (out, calls) = run_splitter(&["日", "本語 <", "ü>", ""], false);
        assert!(calls.is_none());
        assert_eq!(out, ["日", "本語 ", "<ü>", ""]);
    }

    #[test]
    fn tool_splitter_decision_matches_the_buffered_parse_at_every_split() {
        let draft = r#"<tool_call>{"name":"draft","arguments":{}}</tool_call>"#;
        let inputs = [
            "plain answer with <b>tags</b>, 日本 and a lone <".to_owned(),
            "Use <tool_call> tags like this: <tool_call>{broken</tool_call>".to_owned(),
            format!("Sure — {WEATHER_CALL} trailing"),
            format!("<think>plan {draft}</think>Done."),
            format!("<think>plan</think>\n{WEATHER_CALL}"),
            "  {\"name\":\"Alice\",\"age\":30}".to_owned(),
            "<|python_tag|>{\"name\":\"f\",\"parameters\":{}}".to_owned(),
            "<think>plan</think> \n {\"name\":\"f\",\"arguments\":{}}".to_owned(),
            "<think> plan</think>  \n then text".to_owned(),
        ];
        // Call ids are random; the decision is the (name, arguments) list.
        let functions = |calls: Option<Vec<ToolCall>>| {
            calls.map(|c| {
                c.into_iter()
                    .map(|c| (c.function.name, c.function.arguments))
                    .collect::<Vec<_>>()
            })
        };
        for input in &inputs {
            for thinking in [false, true] {
                let expected = functions(parse_tool_calls(tool_call_scan_text(input, thinking)));
                for i in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
                    let (out, calls) = run_splitter(&[&input[..i], &input[i..], ""], thinking);
                    let streamed = out.concat();
                    assert_eq!(functions(calls), expected, "{input:?} split at {i}");
                    if expected.is_none() {
                        assert_eq!(&streamed, input, "split at {i}");
                    } else {
                        assert!(input.starts_with(&streamed), "{input:?} split at {i}");
                        let scanned = input.len() - tool_call_scan_text(input, thinking).len();
                        assert!(
                            streamed.len() <= scanned
                                || !streamed[scanned..].contains(TOOL_CALL_OPEN),
                            "{input:?} split at {i} leaked a marker: {streamed:?}"
                        );
                    }
                }
            }
        }
    }

    /// Emits `parts[0]`, then nothing until `release` is set, then the rest
    /// and a final chunk (an error chunk when `fail`), so a test can read
    /// frames while generation is provably still running.
    struct GatedEngine {
        parts: Vec<String>,
        release: Arc<std::sync::atomic::AtomicBool>,
        fail: bool,
        pending: Option<(cascadia_types::TaskId, usize)>,
    }

    impl cascadia_engine::Engine for GatedEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, task: GenerationTask) -> cascadia_engine::EngineResult<()> {
            self.pending = Some((task.task_id, 0));
            Ok(())
        }
        fn step(&mut self) -> cascadia_engine::EngineResult<Vec<(cascadia_types::TaskId, Chunk)>> {
            let Some((id, i)) = self.pending.take() else {
                return Ok(Vec::new());
            };
            if i > 0 && !self.release.load(Ordering::SeqCst) {
                self.pending = Some((id, i));
                return Ok(Vec::new());
            }
            if let Some(part) = self.parts.get(i) {
                let chunk = Chunk::token(&id, i as i64, part.clone());
                self.pending = Some((id.clone(), i + 1));
                return Ok(vec![(id, chunk)]);
            }
            let last = if self.fail {
                Chunk::error(&id, "gated engine failure")
            } else {
                Chunk::final_marker(&id, "")
                    .with_finish_reason(cascadia_types::FinishReason::Stop)
                    .with_prompt_tokens(5)
                    .with_completion_tokens(self.parts.len() as u32)
            };
            Ok(vec![(id, last)])
        }
    }

    struct GatedBuilder(GatedEngine);

    #[::async_trait::async_trait]
    impl cascadia_engine::Builder for GatedBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> cascadia_engine::EngineResult<()> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> cascadia_engine::EngineResult<cascadia_engine::LoadStream> {
            Ok(Box::pin(stream::empty()))
        }
        fn build(
            self: Box<Self>,
        ) -> cascadia_engine::EngineResult<Box<dyn cascadia_engine::Engine>> {
            Ok(Box::new(self.0))
        }
    }

    /// Router over a [`GatedEngine`] with a template that renders tools, so
    /// tool requests are accepted. Returns the gate.
    async fn gated_app(parts: &[&str], fail: bool) -> (Router, Arc<std::sync::atomic::AtomicBool>) {
        let release = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let runner = Runner::new(Box::new(GatedBuilder(GatedEngine {
            parts: parts.iter().map(|p| (*p).to_owned()).collect(),
            release: release.clone(),
            fail,
            pending: None,
        })));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("mock-model", "CPU"),
            )
            .await
            .unwrap();
        let cfg = Config {
            chat_template: ChatTemplateConfig {
                template: Some("{% for m in messages %}{{ m.content }}{% endfor %}".into()),
                bos_token: None,
                eos_token: None,
            },
            ..Config::default()
        };
        (
            make_router_with_config(Arc::new(runner), "mock-model", cfg),
            release,
        )
    }

    fn tool_stream_request() -> Request<Body> {
        let payload = serde_json::json!({"model":"mock-model",
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"get_weather"}}],
            "stream":true,"stream_options":{"include_usage":true}});
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap()
    }

    /// JSON events of an SSE body, `[DONE]` excluded.
    fn sse_events(body: &str) -> Vec<Value> {
        body.lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|l| l.trim() != "[DONE]")
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn streamed_content(events: &[Value]) -> String {
        events
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect()
    }

    #[tokio::test]
    async fn streaming_tools_sends_content_before_generation_ends() {
        let (app, release) = gated_app(&["Checking ", "the weather. ", WEATHER_CALL], false).await;
        let resp = app.oneshot(tool_stream_request()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body = resp.into_body().into_data_stream();
        // The engine is gated after its first token: this frame can only
        // arrive if content streams while generation is still running.
        let first = loop {
            let frame = tokio::time::timeout(std::time::Duration::from_secs(10), body.next())
                .await
                .expect("content frame while generation is gated")
                .expect("body still open")
                .unwrap();
            if !frame.is_empty() {
                break String::from_utf8(frame.to_vec()).unwrap();
            }
        };
        let events = sse_events(&first);
        assert_eq!(events.len(), 1, "{first}");
        assert_eq!(events[0]["choices"][0]["delta"]["content"], "Checking ");
        assert!(
            events[0]["choices"][0]["finish_reason"].is_null(),
            "{first}"
        );
        release.store(true, Ordering::SeqCst);
        let mut rest = String::new();
        while let Some(frame) = body.next().await {
            rest.push_str(std::str::from_utf8(&frame.unwrap()).unwrap());
        }
        assert!(rest.trim_end().ends_with("data: [DONE]"), "{rest}");
        assert!(!rest.contains("<tool_call>"), "marker leaked: {rest}");
        let events = sse_events(&rest);
        assert_eq!(streamed_content(&events), "the weather. ");
        let call_events: Vec<_> = events
            .iter()
            .filter(|e| e["choices"][0]["delta"].get("tool_calls").is_some())
            .collect();
        assert_eq!(call_events.len(), 1, "{rest}");
        let call = &call_events[0]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(call["index"], 0);
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(),
            serde_json::json!({"city":"Paris"})
        );
        let finishes: Vec<_> = events
            .iter()
            .filter_map(|e| e["choices"][0]["finish_reason"].as_str())
            .collect();
        assert_eq!(finishes, ["tool_calls"], "{rest}");
        let usage = events.last().unwrap();
        assert!(usage["choices"].as_array().unwrap().is_empty(), "{rest}");
        assert_eq!(
            usage["usage"],
            serde_json::json!({"prompt_tokens":5,"completion_tokens":3,"total_tokens":8})
        );
    }

    /// Sum of the custom per-frame `n_tokens` across an SSE body.
    fn frame_tokens(events: &[Value]) -> u64 {
        events.iter().filter_map(|e| e["n_tokens"].as_u64()).sum()
    }

    #[tokio::test]
    async fn streaming_tools_counts_every_token_in_exactly_one_frame() {
        let pieces: Vec<&str> = WEATHER_CALL.split_inclusive([',', ':']).collect();
        assert!(pieces.len() >= 5, "{pieces:?}");
        let mut with_text = vec!["Checking ", "the weather. "];
        with_text.extend(&pieces);
        for parts in [pieces.clone(), with_text] {
            let (app, release) = gated_app(&parts, false).await;
            release.store(true, Ordering::SeqCst);
            let resp = app.oneshot(tool_stream_request()).await.unwrap();
            let body = String::from_utf8(to_bytes(resp.into_body(), 65536).await.unwrap().to_vec())
                .unwrap();
            assert!(body.contains("\"finish_reason\":\"tool_calls\""), "{body}");
            let events = sse_events(&body);
            // The gated engine emits one token per part.
            assert_eq!(frame_tokens(&events), parts.len() as u64, "{body}");
        }
    }

    #[tokio::test]
    async fn streaming_tools_sends_two_calls_in_one_indexed_delta() {
        let second = r#"<tool_call>{"name":"get_time","arguments":{"tz":"CET"}}</tool_call>"#;
        let (app, release) = gated_app(&[WEATHER_CALL, "\n", second], false).await;
        release.store(true, Ordering::SeqCst);
        let resp = app.oneshot(tool_stream_request()).await.unwrap();
        let body =
            String::from_utf8(to_bytes(resp.into_body(), 65536).await.unwrap().to_vec()).unwrap();
        let events = sse_events(&body);
        let call_events: Vec<_> = events
            .iter()
            .filter(|e| e["choices"][0]["delta"].get("tool_calls").is_some())
            .collect();
        assert_eq!(call_events.len(), 1, "{body}");
        let calls = call_events[0]["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .unwrap();
        let summary: Vec<_> = calls
            .iter()
            .map(|c| {
                (
                    c["index"].as_u64().unwrap(),
                    c["function"]["name"].as_str().unwrap(),
                )
            })
            .collect();
        assert_eq!(summary, [(0, "get_weather"), (1, "get_time")], "{body}");
        assert_ne!(calls[0]["id"], calls[1]["id"], "{body}");
        let finishes: Vec<_> = events
            .iter()
            .filter_map(|e| e["choices"][0]["finish_reason"].as_str())
            .collect();
        assert_eq!(finishes, ["tool_calls"], "{body}");
        assert_eq!(streamed_content(&events), "", "{body}");
    }

    #[tokio::test]
    async fn streaming_tools_without_a_call_streams_every_chunk() {
        let (app, release) = gated_app(&["Hello ", "world", "!"], false).await;
        release.store(true, Ordering::SeqCst);
        let resp = app.oneshot(tool_stream_request()).await.unwrap();
        let body =
            String::from_utf8(to_bytes(resp.into_body(), 65536).await.unwrap().to_vec()).unwrap();
        assert!(!body.contains("tool_calls"), "{body}");
        let events = sse_events(&body);
        let pieces: Vec<_> = events
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .filter(|c| !c.is_empty())
            .collect();
        assert_eq!(pieces, ["Hello ", "world", "!"], "{body}");
        let finishes: Vec<_> = events
            .iter()
            .filter_map(|e| e["choices"][0]["finish_reason"].as_str())
            .collect();
        assert_eq!(finishes, ["stop"], "{body}");
        assert_eq!(events.last().unwrap()["usage"]["total_tokens"], 8, "{body}");
    }

    #[tokio::test]
    async fn streaming_tools_error_after_content_ends_with_error_event() {
        let (app, release) = gated_app(&["Checking ", "more "], true).await;
        release.store(true, Ordering::SeqCst);
        let resp = app.oneshot(tool_stream_request()).await.unwrap();
        let body =
            String::from_utf8(to_bytes(resp.into_body(), 65536).await.unwrap().to_vec()).unwrap();
        assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
        let events = sse_events(&body);
        assert_eq!(streamed_content(&events), "Checking more ");
        assert_eq!(events.last().unwrap()["object"], "error", "{body}");
        assert!(!body.contains("tool_calls"), "{body}");
        assert!(!body.contains("\"usage\""), "{body}");
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

    // --- Generation defaults (1.6, D2/D3/D4/D9) ---------------------------

    /// A mock router whose server-wide generation defaults are `defaults`.
    async fn make_app_with_defaults(defaults: GenerationDefaults) -> Router {
        let runner = Runner::new(Box::new(MockBuilder::new()));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("mock-model", "CPU"),
            )
            .await
            .unwrap();
        let cfg = Config {
            generation_defaults: defaults,
            ..Config::default()
        };
        make_router_with_config(Arc::new(runner), "mock-model", cfg)
    }

    /// `n` distinct words: the mock echoes one per token, so a prompt of `n`
    /// words finishes with `stop` after `n` tokens unless a limit cuts it.
    fn words(n: usize) -> String {
        (0..n)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    async fn post_chat_big(app: Router, payload: serde_json::Value) -> (StatusCode, Value) {
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
        let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        (status, v)
    }

    async fn post_completions_big(app: Router, payload: serde_json::Value) -> (StatusCode, Value) {
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
        let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        (status, v)
    }

    #[tokio::test]
    async fn unset_chat_limit_runs_past_256_to_the_models_stop() {
        // Decision 2: no limit = until the model stops (here: the 300-word echo
        // ends), not the old silent 256-token cut.
        let (status, v) = post_chat_big(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": words(300)}],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["usage"]["completion_tokens"], 300, "{v}");
        assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
        crate::spec_schema::assert_spec("CreateChatCompletionResponse", &v);
    }

    #[tokio::test]
    async fn default_max_tokens_256_restores_the_old_cap_on_chat() {
        // `--default-max-tokens 256` reaches the API as this default.
        let defaults = GenerationDefaults {
            max_tokens: 256,
            ..GenerationDefaults::default()
        };
        let (status, v) = post_chat_big(
            make_app_with_defaults(defaults).await,
            serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": words(300)}],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["usage"]["completion_tokens"], 256, "{v}");
        assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
        crate::spec_schema::assert_spec("CreateChatCompletionResponse", &v);
    }

    #[tokio::test]
    async fn unset_completions_limit_runs_past_256_to_the_models_stop() {
        // D4: /v1/completions follows the same rule as chat.
        let (status, v) = post_completions_big(
            make_app().await,
            serde_json::json!({"model": "mock-model", "prompt": words(300)}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["usage"]["completion_tokens"], 300, "{v}");
        assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
        crate::spec_schema::assert_spec("CreateCompletionResponse", &v);
    }

    #[tokio::test]
    async fn default_max_tokens_256_restores_the_old_cap_on_completions() {
        let defaults = GenerationDefaults {
            max_tokens: 256,
            ..GenerationDefaults::default()
        };
        let (status, v) = post_completions_big(
            make_app_with_defaults(defaults).await,
            serde_json::json!({"model": "mock-model", "prompt": words(300)}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["usage"]["completion_tokens"], 256, "{v}");
        assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
        crate::spec_schema::assert_spec("CreateCompletionResponse", &v);
    }

    #[tokio::test]
    async fn completions_ignores_max_completion_tokens() {
        // /v1/completions has no `max_completion_tokens` in the spec: only
        // `max_tokens` limits it.
        let (status, v) = post_completions_big(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "prompt": "alpha bravo charlie delta echo",
                "max_tokens": 2,
                "max_completion_tokens": 4,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["usage"]["completion_tokens"], 2, "{v}");
        assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
        crate::spec_schema::assert_spec("CreateCompletionResponse", &v);
    }

    /// Records every task the API submits, then answers with one token.
    struct CaptureEngine {
        seen: Arc<parking_lot::Mutex<Vec<GenerationTask>>>,
        pending: Option<cascadia_types::TaskId>,
    }

    impl cascadia_engine::Engine for CaptureEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, task: GenerationTask) -> cascadia_engine::EngineResult<()> {
            self.pending = Some(task.task_id.clone());
            self.seen.lock().push(task);
            Ok(())
        }
        fn step(&mut self) -> cascadia_engine::EngineResult<Vec<(cascadia_types::TaskId, Chunk)>> {
            let Some(id) = self.pending.take() else {
                return Ok(Vec::new());
            };
            Ok(vec![
                (id.clone(), Chunk::token(&id, 0, "ok")),
                (id.clone(), Chunk::final_marker(&id, "")),
            ])
        }
    }

    struct CaptureBuilder {
        seen: Arc<parking_lot::Mutex<Vec<GenerationTask>>>,
    }

    #[::async_trait::async_trait]
    impl cascadia_engine::Builder for CaptureBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> cascadia_engine::EngineResult<()> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> cascadia_engine::EngineResult<cascadia_engine::LoadStream> {
            Ok(Box::pin(stream::empty()))
        }
        fn build(
            self: Box<Self>,
        ) -> cascadia_engine::EngineResult<Box<dyn cascadia_engine::Engine>> {
            Ok(Box::new(CaptureEngine {
                seen: self.seen,
                pending: None,
            }))
        }
    }

    /// Non-default values, so a test can tell "the server default" from any
    /// hard-coded fallback (1.0, 0.0, 256, 4096).
    const TEST_DEFAULTS: GenerationDefaults = GenerationDefaults {
        max_tokens: 777,
        temperature: 0.7,
        top_p: 0.9,
        context_window: Some(777),
    };

    /// Sends one request (`/v1/chat/completions` when `chat`, else
    /// `/v1/completions`) with `extra` merged in, through a capturing engine
    /// whose server defaults are [`TEST_DEFAULTS`]; returns the task the
    /// engine received.
    async fn captured_task(chat: bool, extra: serde_json::Value) -> GenerationTask {
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let runner = Runner::new(Box::new(CaptureBuilder { seen: seen.clone() }));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("mock-model", "CPU"),
            )
            .await
            .unwrap();
        let cfg = Config {
            generation_defaults: TEST_DEFAULTS,
            ..Config::default()
        };
        let app = make_router_with_config(Arc::new(runner), "mock-model", cfg);
        let mut payload = if chat {
            serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
            })
        } else {
            serde_json::json!({"model": "mock-model", "prompt": "hi"})
        };
        payload
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let (status, v) = if chat {
            post_chat_big(app, payload).await
        } else {
            post_completions_big(app, payload).await
        };
        assert_eq!(status, StatusCode::OK, "{extra}: {v}");
        let schema = if chat {
            "CreateChatCompletionResponse"
        } else {
            "CreateCompletionResponse"
        };
        crate::spec_schema::assert_spec(schema, &v);
        let tasks = seen.lock();
        assert_eq!(tasks.len(), 1, "{extra}");
        tasks[0].clone()
    }

    async fn assert_unset_fields_take_server_defaults(chat: bool) {
        let t = captured_task(chat, serde_json::json!({})).await;
        assert_eq!(t.max_tokens, TEST_DEFAULTS.max_tokens);
        assert_eq!(t.temperature, TEST_DEFAULTS.temperature);
        assert_eq!(t.sampling.top_p, TEST_DEFAULTS.top_p);
    }

    async fn assert_explicit_fields_beat_server_defaults(chat: bool) {
        // An explicit 0 is a value (greedy), not "unset".
        let t = captured_task(
            chat,
            serde_json::json!({"temperature": 0, "top_p": 0.5, "max_tokens": 3}),
        )
        .await;
        assert_eq!(t.temperature, 0.0);
        assert_eq!(t.sampling.top_p, 0.5);
        assert_eq!(t.max_tokens, 3);
    }

    async fn assert_null_fields_are_unset(chat: bool) {
        // D9: `null` is the same as omitting the field.
        let t = captured_task(
            chat,
            serde_json::json!({"temperature": null, "top_p": null, "max_tokens": null}),
        )
        .await;
        assert_eq!(t.max_tokens, TEST_DEFAULTS.max_tokens);
        assert_eq!(t.temperature, TEST_DEFAULTS.temperature);
        assert_eq!(t.sampling.top_p, TEST_DEFAULTS.top_p);
    }

    #[tokio::test]
    async fn chat_unset_fields_take_server_defaults() {
        assert_unset_fields_take_server_defaults(true).await;
    }

    #[tokio::test]
    async fn chat_explicit_fields_beat_server_defaults() {
        assert_explicit_fields_beat_server_defaults(true).await;
    }

    #[tokio::test]
    async fn chat_null_fields_are_unset() {
        assert_null_fields_are_unset(true).await;
        let t = captured_task(true, serde_json::json!({"max_completion_tokens": null})).await;
        assert_eq!(t.max_tokens, TEST_DEFAULTS.max_tokens);
    }

    #[tokio::test]
    async fn completions_unset_fields_take_server_defaults() {
        assert_unset_fields_take_server_defaults(false).await;
    }

    #[tokio::test]
    async fn completions_explicit_fields_beat_server_defaults() {
        assert_explicit_fields_beat_server_defaults(false).await;
    }

    #[tokio::test]
    async fn completions_null_fields_are_unset() {
        assert_null_fields_are_unset(false).await;
    }

    #[test]
    fn generation_defaults_default_is_one_window_fallback_and_spec_sampling() {
        assert_eq!(
            GenerationDefaults::default(),
            GenerationDefaults {
                max_tokens: 4096,
                temperature: 1.0,
                top_p: 1.0,
                context_window: None,
            }
        );
        assert_eq!(
            Config::default().generation_defaults,
            GenerationDefaults::default()
        );
    }

    // --- Accepted client fields and real client bodies (1.9, D6/D9) -------

    /// POSTs a chat body; returns the status and the raw (JSON or SSE) body.
    async fn post_chat_text(app: Router, payload: &Value) -> (StatusCode, String) {
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
        let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, String::from_utf8_lossy(&body).into_owned())
    }

    /// Sends `payload` to `/v1/chat/completions` and checks the answer has
    /// `tokens` completion tokens ending in `finish`. A streamed answer must
    /// carry that many content frames, the usage frame when the body asked
    /// for one, and `[DONE]`; a plain one must match the OpenAI spec.
    async fn assert_chat_body(app: Router, payload: Value, tokens: u64, finish: &str) {
        let (status, text) = post_chat_text(app, &payload).await;
        assert_eq!(status, StatusCode::OK, "{payload}: {text}");
        if payload["stream"] == true {
            let events = sse_events(&text);
            let content_frames = events
                .iter()
                .filter(|e| {
                    e["choices"][0]["delta"]["content"]
                        .as_str()
                        .is_some_and(|s| !s.is_empty())
                })
                .count() as u64;
            assert_eq!(content_frames, tokens, "{text}");
            let finish_seen = events
                .iter()
                .find_map(|e| e["choices"][0]["finish_reason"].as_str())
                .expect("finish_reason frame");
            assert_eq!(finish_seen, finish, "{text}");
            if payload["stream_options"]["include_usage"] == true {
                let usage = events
                    .iter()
                    .find(|e| e["usage"].is_object())
                    .expect("usage frame");
                assert_eq!(usage["usage"]["completion_tokens"], tokens, "{text}");
            }
            assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
        } else {
            let v: Value = serde_json::from_str(&text).unwrap();
            assert_eq!(v["usage"]["completion_tokens"], tokens, "{v}");
            assert_eq!(v["choices"][0]["finish_reason"], finish, "{v}");
            crate::spec_schema::assert_spec("CreateChatCompletionResponse", &v);
        }
    }

    /// The OpenAI fields that change nothing on a local server (1.9, D6),
    /// with representative values: they are accepted and ignored, never 422.
    fn harmless_openai_fields() -> Value {
        serde_json::json!({
            "user": "user-1234",
            "safety_identifier": "sid-1234",
            "prompt_cache_key": "session-1",
            "prompt_cache_options": {"mode": "auto"},
            "prompt_cache_retention": "24h",
            "metadata": {"run": "t4"},
            "store": true,
            "service_tier": "auto",
            "verbosity": "low",
            "moderation": "auto",
            "prediction": {"type": "content", "content": "alpha bravo"},
        })
    }

    fn five_word_chat(extra: Value) -> Value {
        let mut payload = serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "alpha bravo charlie delta echo"}],
        });
        payload
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        payload
    }

    #[tokio::test]
    async fn chat_accepts_and_ignores_harmless_openai_fields() {
        // Guard (D6): nothing parses these, and nothing may start rejecting them.
        let fields = harmless_openai_fields();
        assert_chat_body(make_app().await, five_word_chat(fields.clone()), 5, "stop").await;
        let mut not_stored = fields.clone();
        not_stored["store"] = false.into();
        assert_chat_body(make_app().await, five_word_chat(not_stored), 5, "stop").await;
        let mut streamed = fields;
        streamed["stream"] = true.into();
        streamed["stream_options"] =
            serde_json::json!({"include_usage": true, "include_obfuscation": false});
        assert_chat_body(make_app().await, five_word_chat(streamed), 5, "stop").await;
    }

    #[tokio::test]
    async fn completions_accepts_and_ignores_user_and_metadata() {
        let (status, v) = post_completions(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "prompt": "alpha bravo charlie",
                "user": "user-1234",
                "metadata": {"run": "t4"},
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["usage"]["completion_tokens"], 3, "{v}");
        crate::spec_schema::assert_spec("CreateCompletionResponse", &v);
    }

    #[tokio::test]
    async fn null_stream_and_logprobs_mean_unset_on_chat() {
        // D9: the spec marks both nullable; `null` = omitted = false.
        assert_chat_body(
            make_app().await,
            five_word_chat(serde_json::json!({"stream": null, "logprobs": null})),
            5,
            "stop",
        )
        .await;
    }

    #[tokio::test]
    async fn null_stream_and_echo_mean_unset_on_completions() {
        let (status, v) = post_completions(
            make_app().await,
            serde_json::json!({
                "model": "mock-model",
                "prompt": "alpha bravo charlie",
                "stream": null,
                "echo": null,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["usage"]["completion_tokens"], 3, "{v}");
        assert_eq!(v["choices"][0]["text"], "alpha bravo charlie ", "{v}");
        crate::spec_schema::assert_spec("CreateCompletionResponse", &v);
    }

    // Real client bodies (scenario "Clients"). Fields and values come from the
    // client notes in the openai-compat clients survey; each test names its note.

    #[tokio::test]
    async fn langchain_and_n8n_streaming_chat_body_is_answered() {
        // frameworks.md (langchain-openai: ChatOpenAI always sends
        // max_completion_tokens) and platforms.md (n8n Chat Model: temperature
        // 0.7, top_p 1, stream + stream_options.include_usage via LangChain;
        // LibreChat on LangChain injects a top-level `user`).
        let body = serde_json::json!({
            "model": "mock-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "alpha bravo charlie delta echo"},
            ],
            "temperature": 0.7,
            "top_p": 1,
            "max_completion_tokens": 1024,
            "stream": true,
            "stream_options": {"include_usage": true},
            "user": "user-42",
        });
        assert_chat_body(make_app().await, body, 5, "stop").await;
    }

    #[tokio::test]
    async fn opencode_chat_body_is_answered() {
        // cli.md "What opencode actually sends for a generic custom provider":
        // stream true, max_completion_tokens (generic provider default),
        // stream_options.include_usage on, tools with `strict: true`,
        // tool_choice with active tools, no `store`. The template renders the
        // messages before the tools, so the mock's echo is plain text, and the
        // limit (opencode sends the model's configured output limit) is below
        // the prompt's length, so the count is exactly the limit.
        let body = serde_json::json!({
            "model": "mock-model",
            "messages": [
                {"role": "system", "content": "You are opencode."},
                {"role": "user", "content": "alpha bravo charlie delta echo"},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"filePath": {"type": "string"}},
                        "required": ["filePath"],
                        "additionalProperties": false,
                    },
                    "strict": true,
                },
            }],
            "tool_choice": "auto",
            "stream": true,
            "temperature": 0,
            "max_completion_tokens": 3,
            "stream_options": {"include_usage": true},
        });
        let app = make_template_app(
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\
             {% if tools %}{{ tools | tojson }}{% endif %}",
        )
        .await;
        assert_chat_body(app, body, 3, "length").await;
    }

    #[tokio::test]
    async fn ai_sdk_openai_compatible_chat_body_is_answered() {
        // cli.md "Upstream vercel/ai @ai-sdk/openai-compatible": `user`,
        // `max_tokens` (from maxOutputTokens), temperature, passthrough provider
        // options (here `store: true`), and stream + stream_options.include_usage
        // when the provider is created with `includeUsage: true`.
        let body = serde_json::json!({
            "model": "mock-model",
            "user": "user-42",
            "max_tokens": 2,
            "temperature": 0.5,
            "store": true,
            "messages": [{"role": "user", "content": "alpha bravo charlie delta echo"}],
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        assert_chat_body(make_app().await, body, 2, "length").await;
    }

    fn browser_use_body() -> Value {
        // agents.md (browser-use): non-streaming, temperature 0.2,
        // frequency_penalty 0.3, max_completion_tokens 4096 by default.
        serde_json::json!({
            "model": "mock-model",
            "messages": [
                {"role": "system", "content": "You are a browser agent."},
                {"role": "user", "content": "alpha bravo charlie delta echo"},
            ],
            "temperature": 0.2,
            "frequency_penalty": 0.3,
            "max_completion_tokens": 4096,
        })
    }

    #[tokio::test]
    async fn browser_use_chat_body_is_answered() {
        assert_chat_body(make_app().await, browser_use_body(), 5, "stop").await;
        // Its sampling values reach the engine.
        let mut extra = browser_use_body();
        extra.as_object_mut().unwrap().remove("messages");
        let t = captured_task(true, extra).await;
        assert_eq!(t.temperature, 0.2);
        assert_eq!(t.sampling.frequency_penalty, 0.3);
        // Its 4096 is above the 777-token window: sent as the window (D13).
        assert_eq!(t.max_tokens, 777);
    }

    #[tokio::test]
    async fn explicit_limits_are_clamped_to_a_known_window() {
        for chat in [true, false] {
            let t = captured_task(chat, serde_json::json!({"max_tokens": 5000})).await;
            assert_eq!(t.max_tokens, 777, "chat={chat}");
            let t = captured_task(chat, serde_json::json!({"max_tokens": 776})).await;
            assert_eq!(t.max_tokens, 776, "chat={chat}");
        }
        let t = captured_task(true, serde_json::json!({"max_completion_tokens": 9999})).await;
        assert_eq!(t.max_tokens, 777);
    }

    #[tokio::test]
    async fn default_max_tokens_does_not_cap_an_explicit_limit() {
        // `--default-max-tokens 256` bounds requests that set no limit; it is
        // not the window, so an explicit 1000 passes.
        let defaults = GenerationDefaults {
            max_tokens: 256,
            context_window: Some(32768),
            ..GenerationDefaults::default()
        };
        let (status, v) = post_chat_big(
            make_app_with_defaults(defaults).await,
            serde_json::json!({"model": "mock-model",
                "messages": [{"role": "user", "content": words(300)}],
                "max_tokens": 1000}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["usage"]["completion_tokens"], 300, "{v}");
        // No window known: an explicit limit is sent as is.
        let defaults = GenerationDefaults {
            max_tokens: 256,
            ..GenerationDefaults::default()
        };
        let (_, v) = post_completions_big(
            make_app_with_defaults(defaults).await,
            serde_json::json!({"model": "mock-model", "prompt": words(300), "max_tokens": 280}),
        )
        .await;
        assert_eq!(v["usage"]["completion_tokens"], 280, "{v}");
    }

    // --- Deprecated `functions` / `function_call` / role `function` (1.17, D5) --

    /// A chat request already through the legacy-field mapping, as the
    /// handler sees it before `validate_tool_choice`.
    fn legacy_request(extra: serde_json::Value) -> ChatCompletionRequest {
        let mut req = chat_request(extra);
        req.apply_legacy_functions().unwrap();
        req
    }

    #[test]
    fn legacy_functions_reach_the_template_as_tools() {
        // (a) `functions` entries are function definitions; they reach the
        // template as `tools` of type function.
        let req = legacy_request(serde_json::json!({
            "functions": [{"name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}],
            "messages": [{"role": "user", "content": "weather?"}],
        }));
        assert!(req.legacy_functions);
        let env = build_chat_env(TOOL_TEMPLATE).unwrap();
        let out = render_with_chat_env(
            &env,
            &req.messages,
            "",
            "",
            true,
            None,
            req.tools.as_deref(),
        )
        .unwrap();
        assert!(
            out.contains("[TOOLS:get_weather]"),
            "functions not forwarded: {out}"
        );
        let tools = chat_tools(&req).expect("tools forwarded to the engine");
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn legacy_function_call_maps_to_tool_choice() {
        // (b) "none" / "auto" / {"name"} → the matching tool_choice.
        for (fc, want) in [
            (serde_json::json!("none"), serde_json::json!("none")),
            (serde_json::json!("auto"), serde_json::json!("auto")),
            (
                serde_json::json!({"name": "get_weather"}),
                serde_json::json!({"type": "function", "function": {"name": "get_weather"}}),
            ),
        ] {
            let req = legacy_request(serde_json::json!({
                "functions": [{"name": "get_weather"}],
                "function_call": fc,
            }));
            assert_eq!(req.tool_choice.as_ref(), Some(&want), "{fc}");
            req.validate_tool_choice().unwrap();
        }
        // A request without the legacy fields is left alone.
        let req = legacy_request(serde_json::json!({
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }));
        assert!(!req.legacy_functions);
        assert!(req.tool_choice.is_none());
    }

    #[tokio::test]
    async fn legacy_function_errors_name_the_legacy_field() {
        // (b) unknown name, (c) functions + tools, function_call + tool_choice,
        // and a malformed function_call: all 400 before the engine, each
        // naming the legacy field the client sent.
        let functions = serde_json::json!([{"name": "get_weather"}]);
        let tools = serde_json::json!([{"type": "function", "function": {"name": "get_weather"}}]);
        for (extra, field) in [
            (
                serde_json::json!({"functions": functions, "function_call": {"name": "get_time"}}),
                "function_call",
            ),
            (
                serde_json::json!({"functions": functions, "tools": tools}),
                "functions",
            ),
            (
                serde_json::json!({"functions": functions, "function_call": "auto",
                    "tool_choice": "auto"}),
                "function_call",
            ),
            (
                serde_json::json!({"functions": functions, "function_call": "sometimes"}),
                "function_call",
            ),
        ] {
            let mut payload = serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "weather in Paris?"}],
            });
            payload
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let (status, v) = post_chat(make_tool_app().await, payload).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{extra}: {v}");
            let msg = v["error"].as_str().unwrap_or_default();
            assert!(
                msg.contains(field),
                "{extra}: error must name `{field}`: {v}"
            );
        }
    }

    #[test]
    fn legacy_function_history_becomes_tool_call_and_tool_turn() {
        // (d) assistant `function_call` + the `function` reply → one tool call
        // and a `tool` turn sharing a synthesised id.
        let req = legacy_request(serde_json::json!({
            "functions": [{"name": "get_weather"}],
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {"role": "assistant", "content": null,
                 "function_call": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}},
                {"role": "function", "name": "get_weather", "content": "{\"temp_c\":18}"},
                {"role": "assistant", "content": null,
                 "function_call": {"name": "get_weather", "arguments": "{\"city\":\"Rome\"}"}},
                {"role": "function", "name": "get_weather", "content": "{\"temp_c\":25}"},
            ],
        }));
        let m = &req.messages;
        let call = &m[1]
            .tool_calls
            .as_ref()
            .expect("function_call → tool_calls")[0];
        assert_eq!(call.id, "call_fn_0");
        assert_eq!(call.r#type, "function");
        assert_eq!(call.function.name, "get_weather");
        assert_eq!(call.function.arguments, "{\"city\":\"Paris\"}");
        assert_eq!(m[2].role, "tool");
        assert_eq!(m[2].tool_call_id.as_deref(), Some("call_fn_0"));
        assert_eq!(m[2].name.as_deref(), Some("get_weather"));
        assert_eq!(m[2].content, "{\"temp_c\":18}");
        assert_eq!(m[3].tool_calls.as_ref().unwrap()[0].id, "call_fn_1");
        assert_eq!(m[4].tool_call_id.as_deref(), Some("call_fn_1"));
        let env = build_chat_env(TOOL_TEMPLATE).unwrap();
        let out = render_with_chat_env(&env, m, "", "", true, None, req.tools.as_deref()).unwrap();
        assert!(out.contains("<|assistant|>[CALLS:get_weather]"), "{out}");
        assert!(
            out.contains("<|tool|>{\"temp_c\":18}[TCID:call_fn_0]"),
            "{out}"
        );
        // The structured turns (sycl-llama) carry the same mapping.
        let turns = chat_turns(&req);
        assert_eq!(turns[1].tool_calls.as_ref().unwrap()[0]["id"], "call_fn_0");
        assert_eq!(turns[2].role, "tool");
        assert_eq!(turns[2].tool_call_id.as_deref(), Some("call_fn_0"));
    }

    #[test]
    fn orphan_function_turn_gets_its_own_id() {
        // A `function` turn with no assistant `function_call` before it still
        // becomes a `tool` turn, with a fresh id no call shares.
        let req = legacy_request(serde_json::json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "function", "name": "get_weather", "content": "18"},
            ],
        }));
        assert_eq!(req.messages[1].role, "tool");
        assert_eq!(req.messages[1].tool_call_id.as_deref(), Some("call_fn_0"));
        // Modern messages are untouched.
        let req = legacy_request(serde_json::json!({
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [{"id": "abc",
                    "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "abc", "content": "1"},
            ],
        }));
        assert_eq!(req.messages[0].tool_calls.as_ref().unwrap()[0].id, "abc");
        assert_eq!(req.messages[1].tool_call_id.as_deref(), Some("abc"));
    }

    #[tokio::test]
    async fn legacy_functions_answer_in_the_legacy_shape() {
        // (e) + (g): a `functions` request gets `message.function_call` and
        // finish_reason `function_call`; the spec accepts it.
        let (status, v) = post_chat(
            make_tool_app().await,
            serde_json::json!({"model": "mock-model",
                "messages": [{"role": "user", "content": WEATHER_CALL}],
                "functions": [{"name": "get_weather"}], "function_call": "auto"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        let choice = &v["choices"][0];
        assert_eq!(choice["finish_reason"], "function_call", "{v}");
        assert_eq!(
            choice["message"]["function_call"],
            serde_json::json!({"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}),
            "{v}"
        );
        assert!(choice["message"].get("tool_calls").is_none(), "{v}");
        assert!(choice["message"]["content"].is_null(), "{v}");
        crate::spec_schema::assert_spec("CreateChatCompletionResponse", &v);
    }

    #[tokio::test]
    async fn legacy_shape_carries_only_the_first_call() {
        let two = format!(
            "{WEATHER_CALL}<tool_call>{{\"name\":\"get_time\",\"arguments\":{{}}}}</tool_call>"
        );
        let (status, v) = post_chat(
            make_tool_app().await,
            serde_json::json!({"model": "mock-model",
                "messages": [{"role": "user", "content": two}],
                "functions": [{"name": "get_weather"}, {"name": "get_time"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["choices"][0]["finish_reason"], "function_call", "{v}");
        assert_eq!(
            v["choices"][0]["message"]["function_call"]["name"], "get_weather",
            "{v}"
        );
        crate::spec_schema::assert_spec("CreateChatCompletionResponse", &v);
    }

    #[tokio::test]
    async fn tools_request_keeps_the_tool_calls_shape() {
        // (e) the same request with `tools` is unchanged, and (Task 1 review)
        // the modern tool-call response passes the spec too.
        let (status, v) = post_chat(
            make_tool_app().await,
            serde_json::json!({"model": "mock-model",
                "messages": [{"role": "user", "content": WEATHER_CALL}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                "tool_choice": "auto"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        let choice = &v["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls", "{v}");
        assert_eq!(
            choice["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert!(choice["message"].get("function_call").is_none(), "{v}");
        crate::spec_schema::assert_spec("CreateChatCompletionResponse", &v);
    }

    #[tokio::test]
    async fn streaming_legacy_functions_emit_function_call_delta() {
        // (f) streaming: `delta.function_call`, then finish_reason `function_call`.
        let (status, body) = post_chat_text(
            make_tool_app().await,
            &serde_json::json!({"model": "mock-model",
                "messages": [{"role": "user", "content": WEATHER_CALL}],
                "functions": [{"name": "get_weather"}], "stream": true}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let events = sse_events(&body);
        let deltas: Vec<&Value> = events
            .iter()
            .map(|e| &e["choices"][0]["delta"])
            .filter(|d| d.get("function_call").is_some())
            .collect();
        assert_eq!(deltas.len(), 1, "one function_call delta: {body}");
        assert_eq!(deltas[0]["role"], "assistant");
        assert_eq!(
            deltas[0]["function_call"],
            serde_json::json!({"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"})
        );
        let finishes: Vec<&Value> = events
            .iter()
            .map(|e| &e["choices"][0]["finish_reason"])
            .filter(|f| !f.is_null())
            .collect();
        assert_eq!(finishes, [&serde_json::json!("function_call")], "{body}");
        assert!(
            !body.contains("tool_calls"),
            "legacy stream mentions tool_calls: {body}"
        );
        assert!(!body.contains("<tool_call>"), "marker leaked: {body}");
        for event in &events {
            crate::spec_schema::assert_spec("CreateChatCompletionStreamResponse", event);
        }
    }

    // --- Early stop on `stop` and SSE keep-alive (M1 D11) -----------------

    /// POSTs `payload` to `uri`; returns the status and the raw body.
    async fn post_text(app: Router, uri: &str, payload: &Value) -> (StatusCode, String) {
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, String::from_utf8_lossy(&body).into_owned())
    }

    /// `body` with each `"created":<digits>` set to 0 and each response id
    /// set to `ID`, so two runs compare byte for byte.
    fn normalize_sse(body: &str) -> String {
        let mut out = String::new();
        let mut rest = body;
        while let Some(i) = rest.find("\"created\":") {
            let (head, tail) = rest.split_at(i + "\"created\":".len());
            out.push_str(head);
            out.push('0');
            rest = tail.trim_start_matches(|c: char| c.is_ascii_digit());
        }
        out.push_str(rest);
        let mut norm = String::new();
        let mut rest = out.as_str();
        while let Some(i) = rest.find("\"id\":\"") {
            let (head, tail) = rest.split_at(i + "\"id\":\"".len());
            norm.push_str(head);
            norm.push_str("ID");
            rest = &tail[tail.find('"').unwrap()..];
        }
        norm.push_str(rest);
        norm
    }

    /// The frames a stream without `stop` sends, recorded before the stop
    /// matcher and keep-alive were added (a9ca3349): they must not change.
    /// (Only the usage frame's prompt count moved, from 0 to 2, when the mock
    /// engine began reporting its prompt's words, M1 D14.)
    #[tokio::test]
    async fn streams_without_stop_are_byte_identical_to_before() {
        let chat = serde_json::json!({"model": "mock-model",
            "messages": [{"role": "user", "content": "alpha bravo"}],
            "max_tokens": 8, "stream": true,
            "stream_options": {"include_usage": true}});
        let (status, body) = post_text(make_app().await, "/v1/chat/completions", &chat).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(normalize_sse(&body), "data: {\"choices\":[{\"delta\":{\"content\":\"alpha \",\"role\":\"assistant\"},\"finish_reason\":null,\"index\":0}],\"completion_tokens\":null,\"created\":0,\"id\":\"ID\",\"model\":\"mock-model\",\"n_tokens\":1,\"object\":\"chat.completion.chunk\"}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"bravo \",\"role\":\"assistant\"},\"finish_reason\":null,\"index\":0}],\"completion_tokens\":null,\"created\":0,\"id\":\"ID\",\"model\":\"mock-model\",\"n_tokens\":1,\"object\":\"chat.completion.chunk\"}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"\",\"role\":\"assistant\"},\"finish_reason\":\"stop\",\"index\":0}],\"completion_tokens\":null,\"created\":0,\"id\":\"ID\",\"model\":\"mock-model\",\"n_tokens\":0,\"object\":\"chat.completion.chunk\"}\n\ndata: {\"choices\":[],\"created\":0,\"id\":\"ID\",\"model\":\"mock-model\",\"object\":\"chat.completion.chunk\",\"usage\":{\"completion_tokens\":2,\"prompt_tokens\":2,\"total_tokens\":4}}\n\ndata: [DONE]\n\n", "chat");
        let text = serde_json::json!({"model": "mock-model", "prompt": "alpha bravo",
            "max_tokens": 8, "stream": true, "echo": true,
            "stream_options": {"include_usage": true}});
        let (status, body) = post_text(make_app().await, "/v1/completions", &text).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(normalize_sse(&body), "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"text\":\"alpha bravoalpha \"}],\"completion_tokens\":null,\"created\":0,\"id\":\"ID\",\"model\":\"mock-model\",\"n_tokens\":1,\"object\":\"text_completion\"}\n\ndata: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"text\":\"bravo \"}],\"completion_tokens\":null,\"created\":0,\"id\":\"ID\",\"model\":\"mock-model\",\"n_tokens\":1,\"object\":\"text_completion\"}\n\ndata: {\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"text\":\"\"}],\"completion_tokens\":null,\"created\":0,\"id\":\"ID\",\"model\":\"mock-model\",\"n_tokens\":0,\"object\":\"text_completion\"}\n\ndata: {\"choices\":[],\"created\":0,\"id\":\"ID\",\"model\":\"mock-model\",\"object\":\"text_completion\",\"usage\":{\"completion_tokens\":2,\"prompt_tokens\":2,\"total_tokens\":4}}\n\ndata: [DONE]\n\n", "completions");
    }

    /// Emits `parts` one per step (`gate` after the first holds the rest,
    /// sending progress markers, until it is set), then a final chunk; records every task it was told
    /// to cancel and how many parts it emitted.
    struct ScriptEngine {
        parts: Vec<String>,
        /// Text of the final chunk (which always carries `prompt_tokens`).
        final_text: String,
        /// Report `prompt_tokens` on the first part too, as token-streaming
        /// engines do (M1 D14).
        prompt_first: bool,
        /// Each step blocks this long first, like a one-shot engine's step.
        step_delay: Duration,
        gate: Option<Arc<std::sync::atomic::AtomicBool>>,
        log: Arc<ScriptLog>,
        pending: Option<(cascadia_types::TaskId, usize)>,
    }

    #[derive(Default)]
    struct ScriptLog {
        cancelled: std::sync::Mutex<Vec<String>>,
        emitted: std::sync::atomic::AtomicUsize,
    }

    impl cascadia_engine::Engine for ScriptEngine {
        fn warmup(&mut self) {}
        fn cancel(&mut self, task_id: &cascadia_types::TaskId) {
            self.log.cancelled.lock().unwrap().push(task_id.to_string());
            if self.pending.as_ref().is_some_and(|(id, _)| id == task_id) {
                self.pending = None;
            }
        }
        fn submit(&mut self, task: GenerationTask) -> cascadia_engine::EngineResult<()> {
            self.pending = Some((task.task_id, 0));
            Ok(())
        }
        fn step(&mut self) -> cascadia_engine::EngineResult<Vec<(cascadia_types::TaskId, Chunk)>> {
            let Some((id, i)) = self.pending.take() else {
                return Ok(Vec::new());
            };
            if !self.step_delay.is_zero() {
                std::thread::sleep(self.step_delay);
            }
            if i > 0
                && self
                    .gate
                    .as_ref()
                    .is_some_and(|g| !g.load(Ordering::SeqCst))
            {
                // Busy but silent, like a long prefill: progress markers
                // only (an empty step would trip the runner's stall guard).
                std::thread::sleep(Duration::from_millis(2));
                self.pending = Some((id.clone(), i));
                return Ok(vec![(id.clone(), Chunk::progress(id))]);
            }
            if let Some(part) = self.parts.get(i) {
                self.log.emitted.fetch_add(1, Ordering::SeqCst);
                let mut chunk = Chunk::token(&id, i as i64, part.clone());
                if i == 0 && self.prompt_first {
                    chunk = chunk.with_prompt_tokens(5);
                }
                self.pending = Some((id.clone(), i + 1));
                return Ok(vec![(id, chunk)]);
            }
            let last = Chunk::final_marker(&id, self.final_text.clone())
                .with_finish_reason(cascadia_types::FinishReason::Length)
                .with_prompt_tokens(5);
            Ok(vec![(id, last)])
        }
    }

    struct ScriptBuilder(Option<ScriptEngine>);

    #[::async_trait::async_trait]
    impl cascadia_engine::Builder for ScriptBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> cascadia_engine::EngineResult<()> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> cascadia_engine::EngineResult<cascadia_engine::LoadStream> {
            Ok(Box::pin(stream::empty()))
        }
        fn build(
            mut self: Box<Self>,
        ) -> cascadia_engine::EngineResult<Box<dyn cascadia_engine::Engine>> {
            Ok(Box::new(self.0.take().unwrap()))
        }
    }

    /// Router over a [`ScriptEngine`] (tools-capable template), its log, and
    /// the gate when `gated`; `keepalive` overrides the SSE idle interval.
    async fn script_app(
        parts: &[&str],
        gated: bool,
        keepalive: Duration,
    ) -> (Router, Arc<ScriptLog>, Arc<std::sync::atomic::AtomicBool>) {
        script_app_full(parts, "", true, gated, keepalive, Duration::ZERO).await
    }

    /// [`script_app`] with the final chunk's text and whether the first
    /// part reports `prompt_tokens`.
    async fn script_app_full(
        parts: &[&str],
        final_text: &str,
        prompt_first: bool,
        gated: bool,
        keepalive: Duration,
        step_delay: Duration,
    ) -> (Router, Arc<ScriptLog>, Arc<std::sync::atomic::AtomicBool>) {
        let log = Arc::new(ScriptLog::default());
        let gate = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let runner = Runner::new(Box::new(ScriptBuilder(Some(ScriptEngine {
            parts: parts.iter().map(|p| (*p).to_owned()).collect(),
            final_text: final_text.to_owned(),
            prompt_first,
            step_delay,
            gate: gated.then(|| gate.clone()),
            log: log.clone(),
            pending: None,
        }))));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("mock-model", "CPU"),
            )
            .await
            .unwrap();
        let cfg = Config {
            chat_template: ChatTemplateConfig {
                template: Some("{% for m in messages %}{{ m.content }}{% endfor %}".into()),
                bos_token: None,
                eos_token: None,
            },
            sse_keepalive: keepalive,
            ..Config::default()
        };
        (
            make_router_with_config(Arc::new(runner), "mock-model", cfg),
            log,
            gate,
        )
    }

    /// Twenty filler parts after `head`: an engine that ran to its end would
    /// emit all of them.
    fn script(head: &[&'static str]) -> Vec<&'static str> {
        let mut parts = head.to_vec();
        parts.extend(std::iter::repeat_n("filler ", 20));
        parts
    }

    fn stop_matcher_run(stops: &[&str], pieces: &[&str]) -> (Vec<String>, bool, String) {
        let stops: Vec<String> = stops.iter().map(|s| (*s).to_owned()).collect();
        let mut m = StopMatcher::new(&stops);
        let out = pieces.iter().map(|p| m.push(p)).collect();
        let stopped = m.stopped();
        (out, stopped, m.finish())
    }

    #[test]
    fn stop_matcher_cuts_at_a_match_split_across_pieces() {
        let (out, stopped, rest) = stop_matcher_run(
            &["\nObservation:"],
            &["Thought: x\nObs", "ervation: y", "z"],
        );
        assert_eq!(out, ["Thought: x", "", ""]);
        assert!(stopped);
        assert_eq!(rest, "");
    }

    #[test]
    fn stop_matcher_releases_text_that_cannot_start_a_stop() {
        // No added latency: nothing is held unless it may begin a stop.
        let (out, stopped, rest) = stop_matcher_run(&["STOP"], &["alpha ", "bravo S", "ix"]);
        assert_eq!(out, ["alpha ", "bravo ", "Six"]);
        assert!(!stopped);
        assert_eq!(rest, "");
        let (out, _, rest) = stop_matcher_run(&["STOP"], &["alpha ST"]);
        assert_eq!(out, ["alpha "]);
        assert_eq!(rest, "ST", "an unmatched prefix comes back at the end");
        let (out, _, _) = stop_matcher_run(&[], &["a", "b"]);
        assert_eq!(out, ["a", "b"], "no stop: everything passes through");
        let (out, _, _) = stop_matcher_run(&[""], &["a"]);
        assert_eq!(out, ["a"], "an empty stop matches nothing");
    }

    #[test]
    fn stop_matcher_handles_a_stop_at_the_start_and_the_earliest_of_several() {
        let (out, stopped, _) = stop_matcher_run(&["STOP"], &["STOP and more"]);
        assert_eq!(out, [""]);
        assert!(stopped);
        let (out, _, _) = stop_matcher_run(&["zeta", "beta"], &["alpha beta gamma zeta"]);
        assert_eq!(out, ["alpha "], "earliest match wins, not first listed");
        let (out, _, _) = stop_matcher_run(&["ab", "b"], &["xa", "b"]);
        assert_eq!(out, ["x", ""]);
        let (out, stopped, _) = stop_matcher_run(&["STOP"], &["a", "b"]);
        assert_eq!(out, ["a", "b"]);
        assert!(!stopped);
        let mut m = StopMatcher::new(&["x".into()]);
        assert_eq!(m.push("ax"), "a");
        assert_eq!(m.push("more"), "", "nothing after the match");
    }

    #[test]
    fn stop_matcher_keeps_multibyte_chars_whole() {
        let (out, stopped, rest) = stop_matcher_run(&["日本語"], &["xx日本", "語の"]);
        assert_eq!(out, ["xx", ""]);
        assert!(stopped);
        assert_eq!(rest, "");
        // The hold-back window starts mid-char here; it must not split one.
        let (out, stopped, rest) = stop_matcher_run(&["é!!"], &["ééé", "é"]);
        assert_eq!(out, ["éé", "é"]);
        assert!(!stopped);
        assert_eq!(rest, "é");
    }

    #[test]
    fn stop_matcher_is_linear_in_a_huge_client_stop() {
        // A ~200 KB stop is within the body limit. The matcher must not
        // rescan the held tail per offset (quadratic in the stop's length).
        let stop = format!("{}b", "a".repeat(200_000));
        let mut m = StopMatcher::new(&[stop, "zz".into()]);
        let chunk = "a".repeat(1000);
        let started = std::time::Instant::now();
        let mut released = 0;
        for _ in 0..400 {
            released += m.push(&chunk).len();
        }
        // Everything but the 200 000-byte partial match has gone out.
        assert_eq!(released, 400_000 - 200_000);
        assert!(!m.stopped());
        assert_eq!(m.push("b"), "", "the held tail was the match itself");
        assert!(m.stopped(), "the stop completes on the final byte");
        // Generous bound: the quadratic version took hours here.
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
        // The same stream with a near-miss byte releases the held tail.
        let mut m = StopMatcher::new(&[format!("{}b", "a".repeat(200_000))]);
        assert_eq!(m.push(&"a".repeat(250_000)).len(), 50_000);
        assert_eq!(m.push("c").len(), 200_001);
    }

    #[test]
    fn stop_matcher_handles_self_overlapping_stops() {
        // KMP fallbacks: a failed partial match may still end in a shorter one.
        let (out, stopped, _) = stop_matcher_run(&["aab"], &["aa", "aab"]);
        assert_eq!(out, ["", "aa"], "\"aaaab\" matches at byte 2");
        assert!(stopped);
        let (out, stopped, _) = stop_matcher_run(&["abab"], &["xaba", "bab"]);
        assert_eq!(out, ["x", ""]);
        assert!(stopped);
        let (out, _, rest) = stop_matcher_run(&["abac"], &["ababa"]);
        assert_eq!(out, ["ab"]);
        assert_eq!(rest, "aba");
    }

    async fn assert_cancelled(log: &ScriptLog) {
        // The cancel is applied when the engine stream drops; a body read to
        // its end has dropped it.
        for _ in 0..100 {
            if !log.cancelled.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            log.cancelled.lock().unwrap().len(),
            1,
            "engine saw no cancel"
        );
        assert!(
            log.emitted.load(Ordering::SeqCst) < 10,
            "engine kept generating: {} parts",
            log.emitted.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn non_streaming_chat_stops_at_the_match_and_cancels_the_engine() {
        let (app, log, _) = script_app(
            &script(&["alpha ", "bravo\nObs", "ervation: x"]),
            false,
            SSE_KEEPALIVE_INTERVAL,
        )
        .await;
        let (status, v) = post_chat(
            app,
            serde_json::json!({"model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stop": ["zzz", "\nObservation:"]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["choices"][0]["message"]["content"], "alpha bravo");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["completion_tokens"], 3, "tokens received so far");
        // Reported on the first chunk (D14), so known before the cut.
        assert_eq!(v["usage"]["prompt_tokens"], 5, "{v}");
        assert_eq!(v["usage"]["total_tokens"], 8, "{v}");
        crate::spec_schema::assert_spec("CreateChatCompletionResponse", &v);
        assert_cancelled(&log).await;
    }

    #[tokio::test]
    async fn non_streaming_completions_stop_at_the_match_and_cancel_the_engine() {
        let (app, log, _) =
            script_app(&script(&["STOP right away"]), false, SSE_KEEPALIVE_INTERVAL).await;
        let (status, v) = post_completions(
            app,
            serde_json::json!({"model": "mock-model", "prompt": "hi", "stop": "STOP"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["choices"][0]["text"], "");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["prompt_tokens"], 5, "{v}");
        crate::spec_schema::assert_spec("CreateCompletionResponse", &v);
        assert_cancelled(&log).await;
    }

    #[tokio::test]
    async fn streaming_chat_cuts_at_the_match_and_cancels_the_engine() {
        let (app, log, _) = script_app(
            &script(&["héllo wör", "ld ST", "OP tail"]),
            false,
            SSE_KEEPALIVE_INTERVAL,
        )
        .await;
        let (status, body) = post_text(
            app,
            "/v1/chat/completions",
            &serde_json::json!({"model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stop": "STOP", "stream": true,
                "stream_options": {"include_usage": true}}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
        let events = sse_events(&body);
        for event in &events {
            crate::spec_schema::assert_spec("CreateChatCompletionStreamResponse", event);
        }
        let contents: Vec<&str> = events
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        // "héllo wör" goes out at once; "ST" is held until "OP" completes the stop.
        assert_eq!(contents, ["héllo wör", "ld ", ""], "{body}");
        let finishes: Vec<&Value> = events
            .iter()
            .map(|e| &e["choices"][0]["finish_reason"])
            .filter(|f| !f.is_null())
            .collect();
        assert_eq!(finishes, [&serde_json::json!("stop")], "{body}");
        let usage = &events.last().unwrap()["usage"];
        assert_eq!(usage["completion_tokens"], 3, "{body}");
        assert_eq!(usage["prompt_tokens"], 5, "{body}");
        assert_cancelled(&log).await;
    }

    #[tokio::test]
    async fn streaming_completions_cut_at_the_match_and_cancel_the_engine() {
        let (app, log, _) = script_app(
            &script(&["one ", "two END three"]),
            false,
            SSE_KEEPALIVE_INTERVAL,
        )
        .await;
        let (status, body) = post_text(
            app,
            "/v1/completions",
            &serde_json::json!({"model": "mock-model", "prompt": "P: ",
                "echo": true, "stop": ["END"], "stream": true,
                "stream_options": {"include_usage": true}}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let events = sse_events(&body);
        let texts: Vec<&str> = events
            .iter()
            .filter_map(|e| e["choices"][0]["text"].as_str())
            .collect();
        assert_eq!(texts, ["P: one ", "two "], "{body}");
        let n = events.len();
        assert_eq!(events[n - 2]["choices"][0]["finish_reason"], "stop");
        assert_eq!(events[n - 1]["usage"]["prompt_tokens"], 5, "{body}");
        assert_eq!(events[n - 1]["usage"]["completion_tokens"], 2, "{body}");
        assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
        assert_cancelled(&log).await;
    }

    #[tokio::test]
    async fn streaming_tool_requests_cut_at_the_match_too() {
        let (app, log, _) = script_app(
            &script(&["Sure, ", "checking STOP now"]),
            false,
            SSE_KEEPALIVE_INTERVAL,
        )
        .await;
        let (status, body) = post_text(
            app,
            "/v1/chat/completions",
            &serde_json::json!({"model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                "stop": "STOP", "stream": true,
                "stream_options": {"include_usage": true}}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let events = sse_events(&body);
        for event in &events {
            crate::spec_schema::assert_spec("CreateChatCompletionStreamResponse", event);
        }
        assert_eq!(streamed_content(&events), "Sure, checking ", "{body}");
        let n = events.len();
        assert_eq!(events[n - 2]["choices"][0]["finish_reason"], "stop");
        assert_eq!(events[n - 1]["usage"]["prompt_tokens"], 5, "{body}");
        assert_eq!(events[n - 1]["usage"]["completion_tokens"], 2, "{body}");
        assert_cancelled(&log).await;
    }

    #[tokio::test]
    async fn stop_inside_the_final_chunk_keeps_its_prompt_count() {
        // A one-shot engine: everything, and the only prompt count, on the
        // final chunk. The count must be read before the stop check.
        for tools in [false, true] {
            let (app, _log, _) = script_app_full(
                &["Sure, "],
                "done STOP tail",
                false,
                false,
                SSE_KEEPALIVE_INTERVAL,
                Duration::ZERO,
            )
            .await;
            let mut body = serde_json::json!({"model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stop": "STOP", "stream": true,
                "stream_options": {"include_usage": true}});
            if tools {
                body["tools"] =
                    serde_json::json!([{"type": "function", "function": {"name": "f"}}]);
            }
            let (status, text) = post_text(app, "/v1/chat/completions", &body).await;
            assert_eq!(status, StatusCode::OK);
            let events = sse_events(&text);
            assert_eq!(
                streamed_content(&events),
                "Sure, done ",
                "tools={tools}: {text}"
            );
            let n = events.len();
            assert_eq!(
                events[n - 2]["choices"][0]["finish_reason"],
                "stop",
                "{text}"
            );
            assert_eq!(
                events[n - 1]["usage"]["prompt_tokens"],
                5,
                "tools={tools}: {text}"
            );
        }
        // Non-streaming, both endpoints.
        let (app, _log, _) = script_app_full(
            &["Sure, "],
            "done STOP tail",
            false,
            false,
            SSE_KEEPALIVE_INTERVAL,
            Duration::ZERO,
        )
        .await;
        let (_, v) = post_chat(
            app,
            serde_json::json!({"model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}], "stop": "STOP"}),
        )
        .await;
        assert_eq!(v["choices"][0]["message"]["content"], "Sure, done ", "{v}");
        assert_eq!(v["usage"]["prompt_tokens"], 5, "{v}");
        let (app, _log, _) = script_app_full(
            &["Sure, "],
            "done STOP tail",
            false,
            false,
            SSE_KEEPALIVE_INTERVAL,
            Duration::ZERO,
        )
        .await;
        let (_, v) = post_completions(
            app,
            serde_json::json!({"model": "mock-model", "prompt": "hi", "stop": "STOP"}),
        )
        .await;
        assert_eq!(v["choices"][0]["text"], "Sure, done ", "{v}");
        assert_eq!(v["usage"]["prompt_tokens"], 5, "{v}");
    }

    /// Reads `uri`'s stream from a gated engine until a keep-alive comment
    /// arrives, then opens the gate and reads the rest.
    async fn read_with_keepalive(uri: &str, payload: Value) -> String {
        let (app, _log, gate) =
            script_app(&["first ", "second "], true, Duration::from_millis(50)).await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body = resp.into_body().into_data_stream();
        let mut text = String::new();
        while !text.contains(": keep-alive") {
            let frame = tokio::time::timeout(Duration::from_secs(10), body.next())
                .await
                .expect("a keep-alive comment while the engine is idle")
                .unwrap_or_else(|| panic!("body ended before a keep-alive: {text}"))
                .unwrap();
            text.push_str(std::str::from_utf8(&frame).unwrap());
        }
        gate.store(true, Ordering::SeqCst);
        while let Some(frame) = body.next().await {
            text.push_str(std::str::from_utf8(&frame.unwrap()).unwrap());
        }
        text
    }

    fn assert_keepalive_body(body: &str, object: &str) {
        assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
        // SSE events are separated by a blank line: each one is either the
        // comment or a single `data:` line, never a mix.
        let events: Vec<&str> = body.split("\n\n").filter(|e| !e.is_empty()).collect();
        assert!(events.contains(&": keep-alive"), "{body}");
        for event in events {
            if event == ": keep-alive" || event == "data: [DONE]" {
                continue;
            }
            let json = event
                .strip_prefix("data: ")
                .unwrap_or_else(|| panic!("{event:?}"));
            assert!(!json.contains("keep-alive"), "{event}");
            let v: Value = serde_json::from_str(json).unwrap();
            assert_eq!(v["object"], object, "{event}");
            if object == "chat.completion.chunk" {
                crate::spec_schema::assert_spec("CreateChatCompletionStreamResponse", &v);
            }
        }
    }

    #[tokio::test]
    async fn idle_streams_send_keepalive_comments() {
        let chat = serde_json::json!({"model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}], "stream": true});
        let body = read_with_keepalive("/v1/chat/completions", chat.clone()).await;
        assert_keepalive_body(&body, "chat.completion.chunk");
        assert_eq!(
            streamed_content(&sse_events(&body.replace(": keep-alive\n\n", ""))),
            "first second "
        );
        let mut tools = chat;
        tools["tools"] = serde_json::json!([{"type": "function", "function": {"name": "f"}}]);
        let body = read_with_keepalive("/v1/chat/completions", tools).await;
        assert_keepalive_body(&body, "chat.completion.chunk");
        let text = serde_json::json!({"model": "mock-model", "prompt": "hi", "stream": true});
        let body = read_with_keepalive("/v1/completions", text).await;
        assert_keepalive_body(&body, "text_completion");
    }

    // --- Legacy functions polish (fix wave C) ------------------------------

    #[test]
    fn unknown_function_call_names_the_list_the_request_used() {
        let mut req = chat_request(serde_json::json!({
            "functions": [{"name": "get_weather"}],
            "function_call": {"name": "get_time"},
        }));
        let e = req.apply_legacy_functions().unwrap_err();
        assert!(e.contains("not in functions"), "{e}");
        let mut req = chat_request(serde_json::json!({
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "function_call": {"name": "get_time"},
        }));
        let e = req.apply_legacy_functions().unwrap_err();
        assert!(e.contains("not in tools"), "{e}");
        assert!(e.contains("function_call"), "{e}");
    }

    #[test]
    fn synthesised_ids_cannot_collide_with_a_clients_call_n() {
        // A client's own `call_0` next to a legacy call: the synthesised id
        // uses a prefix no client sends.
        let req = legacy_request(serde_json::json!({
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [{"id": "call_0",
                    "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_0", "content": "1"},
                {"role": "assistant", "content": null,
                 "function_call": {"name": "f", "arguments": "{}"}},
                {"role": "function", "name": "f", "content": "2"},
            ],
        }));
        let m = &req.messages;
        let id = &m[2].tool_calls.as_ref().unwrap()[0].id;
        assert_ne!(id, "call_0");
        assert_eq!(m[3].tool_call_id.as_ref(), Some(id));
    }

    #[test]
    fn empty_tool_calls_beside_function_call_maps_the_function_call() {
        let req = legacy_request(serde_json::json!({
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [],
                 "function_call": {"name": "f", "arguments": "{\"a\":1}"}},
                {"role": "function", "name": "f", "content": "2"},
            ],
        }));
        let calls = req.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{\"a\":1}");
        assert_eq!(req.messages[1].tool_call_id.as_ref(), Some(&calls[0].id));
    }

    #[test]
    fn function_reply_after_tool_calls_takes_the_first_call_id() {
        // Both sent: `tool_calls` wins as the history, and the `function`
        // reply answers its first call instead of an id nothing matches.
        let req = legacy_request(serde_json::json!({
            "messages": [
                {"role": "assistant", "content": null,
                 "tool_calls": [
                    {"id": "abc", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                    {"id": "def", "type": "function", "function": {"name": "g", "arguments": "{}"}}],
                 "function_call": {"name": "f", "arguments": "{}"}},
                {"role": "function", "name": "f", "content": "2"},
            ],
        }));
        assert_eq!(req.messages[0].tool_calls.as_ref().unwrap().len(), 2);
        assert_eq!(req.messages[1].tool_call_id.as_deref(), Some("abc"));
    }

    #[tokio::test]
    async fn bad_message_field_422_names_its_path() {
        let (status, body) = post_chat_text(
            make_app().await,
            &serde_json::json!({"model": "mock-model",
                "messages": [{"role": "user", "content": 5}]}),
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert!(body.contains("messages[0].content"), "{body}");
        let (status, body) = post_chat_text(
            make_app().await,
            &serde_json::json!({"model": "mock-model",
                "messages": [{"role": "assistant", "content": null,
                    "function_call": {"name": 1}}]}),
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert!(body.contains("messages[0].function_call.name"), "{body}");
    }

    #[tokio::test]
    async fn negative_seed_is_accepted_and_reaches_the_engine() {
        for chat in [true, false] {
            let t = captured_task(chat, serde_json::json!({"seed": -1})).await;
            assert_eq!(t.sampling.seed, Some(u64::MAX), "chat={chat}");
            let t = captured_task(chat, serde_json::json!({"seed": 1234})).await;
            assert_eq!(t.sampling.seed, Some(1234), "chat={chat}");
            // Above i64::MAX, as main accepted (u64).
            let t = captured_task(chat, serde_json::json!({"seed": u64::MAX})).await;
            assert_eq!(t.sampling.seed, Some(u64::MAX), "chat={chat}");
            let t = captured_task(chat, serde_json::json!({"seed": null})).await;
            assert_eq!(t.sampling.seed, None, "chat={chat}");
        }
    }

    // --- Keep-alive during a blocking engine step (fix round 2) -----------

    /// One `step()` that blocks well past the keep-alive interval: the comment
    /// must still go out, from the body task, while the engine is stuck.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn keepalive_goes_out_during_a_blocking_engine_step() {
        for (uri, payload) in [
            (
                "/v1/chat/completions",
                serde_json::json!({"model": "mock-model", "stream": true,
                    "messages": [{"role": "user", "content": "hi"}]}),
            ),
            (
                "/v1/chat/completions",
                serde_json::json!({"model": "mock-model", "stream": true,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function", "function": {"name": "f"}}]}),
            ),
            (
                "/v1/completions",
                serde_json::json!({"model": "mock-model", "prompt": "hi", "stream": true}),
            ),
        ] {
            let (app, _log, _) = script_app_full(
                &["only "],
                "",
                true,
                false,
                Duration::from_millis(50),
                Duration::from_millis(400),
            )
            .await;
            let (status, body) = post_text(app, uri, &payload).await;
            assert_eq!(status, StatusCode::OK);
            let first_data = body.find("data: ").expect("a data frame");
            let first_ping = body
                .find(": keep-alive")
                .unwrap_or_else(|| panic!("{uri}: no keep-alive while the step blocked: {body}"));
            assert!(first_ping < first_data, "{uri}: {body}");
            assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
        }
    }

    /// A client that disconnects while a step blocks still cancels the task
    /// once the step returns, and frees its slot.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn disconnect_during_a_blocking_step_still_cancels() {
        let (app, log, _) = script_app_full(
            &script(&["a "]),
            "",
            true,
            false,
            Duration::from_millis(20),
            Duration::from_millis(100),
        )
        .await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"model": "mock-model", "stream": true,
                            "messages": [{"role": "user", "content": "hi"}]})
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body = resp.into_body().into_data_stream();
        // A keep-alive first (the first step blocks), then disconnect.
        let frame = tokio::time::timeout(Duration::from_secs(10), body.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(&frame[..], b": keep-alive\n\n");
        drop(body);
        assert_cancelled(&log).await;
    }

    /// A stream that ends without a final chunk (the task was cancelled)
    /// still sends the text the stop matcher was holding: it never matched.
    #[tokio::test]
    async fn cancelled_stream_releases_a_held_stop_prefix() {
        for tools in [false, true] {
            let (app, _log, gate) =
                script_app(&["first ST", "never sent"], true, SSE_KEEPALIVE_INTERVAL).await;
            let mut payload = serde_json::json!({"model": "mock-model", "stream": true,
                "messages": [{"role": "user", "content": "hi"}], "stop": "STOP"});
            if tools {
                payload["tools"] =
                    serde_json::json!([{"type": "function", "function": {"name": "f"}}]);
            }
            let resp = app
                .clone()
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
            let mut body = resp.into_body().into_data_stream();
            let mut text = String::new();
            while !text.contains("first ") {
                let frame = tokio::time::timeout(Duration::from_secs(10), body.next())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                text.push_str(std::str::from_utf8(&frame).unwrap());
            }
            let id = sse_events(&text)[0]["id"].as_str().unwrap().to_owned();
            let cancel = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(format!("/v1/cancel/{id}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(cancel.status(), StatusCode::OK);
            while let Some(frame) = body.next().await {
                text.push_str(std::str::from_utf8(&frame.unwrap()).unwrap());
            }
            drop(gate);
            assert_eq!(
                streamed_content(&sse_events(&text)),
                "first ST",
                "tools={tools}: {text}"
            );
            assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
        }
    }

    #[tokio::test]
    async fn null_include_usage_means_false() {
        let (status, body) = post_text(
            make_app().await,
            "/v1/chat/completions",
            &serde_json::json!({"model": "mock-model", "stream": true,
                "messages": [{"role": "user", "content": "alpha"}],
                "stream_options": {"include_usage": null}}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(!body.contains("\"usage\""), "{body}");
        let (status, body) = post_text(
            make_app().await,
            "/v1/completions",
            &serde_json::json!({"model": "mock-model", "prompt": "alpha", "stream": true,
                "stream_options": {"include_usage": null}}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }

    #[tokio::test]
    async fn non_integer_seed_is_still_rejected() {
        let (status, _) = post_chat(
            make_app().await,
            serde_json::json!({"model": "mock-model", "seed": 1.5,
                "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    }
}
