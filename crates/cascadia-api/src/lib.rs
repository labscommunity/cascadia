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

/// Tokens a chunk contributes to the cumulative counter. The engine's
/// `n_tokens` is authoritative when set (spec-decode reports 1..=K+1;
/// ov-genai sets it on its single final chunk so `usage` reflects the
/// real generated count — issue #55); otherwise the per-chunk convention
/// is "one token per non-empty chunk" (see `Chunk::n_tokens` docs). An
/// empty chunk — the no-text final marker mock/runtime engines emit —
/// contributes 0 so it isn't counted as a phantom token.
fn chunk_token_count(chunk: &cascadia_types::Chunk) -> u32 {
    chunk
        .n_tokens
        .unwrap_or(if chunk.text.is_empty() { 0 } else { 1 })
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
        .route("/v1/completions", post(completions))
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
    /// Hybrid-reasoning switch (Qwen3+ convention). Default true =
    /// model-default behavior; false asks the engine to skip the
    /// <think> block (engines that can't, ignore it).
    #[serde(default = "default_true")]
    pub enable_thinking: bool,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
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
    tools: Option<&[Tool]>,
) -> Result<String, String> {
    use minijinja::context;
    use minijinja::value::Value;
    let tmpl = env.get_template("chat").map_err(|e| format!("template lookup: {e}"))?;
    let messages_value: Vec<Value> = messages
        .iter()
        .map(|m| {
            let mut obj = serde_json::json!({ "role": m.role, "content": m.content });
            if let Some(tc) = &m.tool_calls {
                obj["tool_calls"] = serde_json::to_value(tc).unwrap_or_default();
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
    tmpl.render(ctx).map_err(|e| format!("template render: {e}"))
}

fn render_prompt(state: &AppState, messages: &[ChatMessage], enable_thinking: bool, tools: Option<&[Tool]>) -> String {
    let defer_to_engine = state.defer_template_on_thinking && enable_thinking;
    if !defer_to_engine {
        if let Some(env) = &state.chat_env {
            match render_with_chat_env(env, messages, &state.bos_token, &state.eos_token, enable_thinking, tools) {
                Ok(s) => return s,
                Err(e) => warn!(error = %e, "chat_template render failed; falling back to legacy formatter"),
            }
        }
    }
    render_prompt_legacy(messages)
}

/// Decide the response message + finish_reason from accumulated text.
/// `tool_choice == Some("none")` skips parsing (always "stop").
fn build_choice(buf: String, tool_choice: &Option<serde_json::Value>) -> (ChatChoiceMessage, &'static str) {
    let parse_enabled = tool_choice.as_ref().and_then(|v| v.as_str()) != Some("none");
    if parse_enabled {
        if let Some(calls) = parse_tool_calls(&buf) {
            return (ChatChoiceMessage { role: "assistant", content: None, tool_calls: Some(calls) }, "tool_calls");
        }
    }
    (ChatChoiceMessage { role: "assistant", content: Some(buf), tool_calls: None }, "stop")
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
            let Some(end) = after.find("</tool_call>") else { break };
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(after[..end].trim()) {
                if let Some(c) = call_from_value(&v) {
                    calls.push(c);
                }
            }
            rest = &after[end + "</tool_call>".len()..];
        }
    } else {
        // Llama-3.1: require <|python_tag|> prefix OR whole-output JSON.
        let trimmed = text.trim();
        let candidate = trimmed.strip_prefix("<|python_tag|>").map(str::trim).unwrap_or(trimmed);
        let looks_json = candidate.starts_with('{') || candidate.starts_with('[');
        if trimmed.starts_with("<|python_tag|>") || looks_json {
            if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(candidate) {
                for v in &arr {
                    if let Some(c) = call_from_value(v) {
                        calls.push(c);
                    }
                }
            } else if let Ok(v) = serde_json::from_str::<serde_json::Value>(candidate) {
                if let Some(c) = call_from_value(&v) {
                    calls.push(c);
                }
            }
        }
    }
    (!calls.is_empty()).then_some(calls)
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

    /// Render `messages` with optional `tools`, falling back to
    /// [`render_prompt_legacy`] when no template is set or rendering fails.
    pub fn render_with_tools(&self, messages: &[ChatMessage], tools: Option<&[Tool]>) -> String {
        if let Some(env) = &self.env {
            match render_with_chat_env(env, messages, &self.bos_token, &self.eos_token, true, tools) {
                Ok(s) => return s,
                Err(e) => warn!(error = %e, "chat_template render failed; falling back to legacy formatter"),
            }
        }
        render_prompt_legacy(messages)
    }

    /// Render `messages`, falling back to [`render_prompt_legacy`] when no
    /// template is set or rendering fails.
    pub fn render(&self, messages: &[ChatMessage]) -> String {
        self.render_with_tools(messages, None)
    }
}

/// One-shot chat-prompt render for standalone / embedded callers that hold a
/// [`ChatTemplateConfig`] directly (e.g. an embedded shard backend) and render
/// outside the router's `AppState`. Builds a [`ChatPromptRenderer`] per call;
/// callers that render repeatedly should construct one [`ChatPromptRenderer`]
/// and reuse it. Falls back to [`render_prompt_legacy`] on parse/render failure.
pub fn render_chat_prompt(cfg: &ChatTemplateConfig, messages: &[ChatMessage]) -> String {
    ChatPromptRenderer::new(cfg).render(messages)
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let task_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let prompt = render_prompt(&state, &req.messages, req.enable_thinking, req.tools.as_deref());
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
        logprobs: req.logprobs_count(),
        sampling: req.sampling_params(),
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
        let include_usage = req
            .stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false);
        return stream_completion(state, req.model, task, permit, inflight, include_usage)
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

    let (message, choice_finish) = build_choice(buf, &req.tool_choice);
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
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "no prompt content: `prompt` is empty" })),
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
    let mut chunk_stream = match state.runner.generate(task) {
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
    let chunk_stream = match state.runner.generate(task) {
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

async fn stream_completion(
    state: AppState,
    model: String,
    task: GenerationTask,
    permit: tokio::sync::OwnedSemaphorePermit,
    inflight: InFlightGuard,
    include_usage: bool,
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
    let status = match &err {
        EngineError::QueueFull { .. } => StatusCode::SERVICE_UNAVAILABLE,
        EngineError::NotLoaded | EngineError::NotConnected => StatusCode::SERVICE_UNAVAILABLE,
        EngineError::InvalidConfig(_)
        | EngineError::PeerRejected(_)
        | EngineError::ShardRejected(_) => StatusCode::BAD_REQUEST,
        EngineError::ModelNotFound(_) => StatusCode::NOT_FOUND,
        EngineError::Backend(_) | EngineError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
        // A task-attributed step failure: the engine abandoned a task; the
        // underlying cause is a backend/transport failure.
        EngineError::Task { .. } => StatusCode::INTERNAL_SERVER_ERROR,
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

    // Guards the contract the ov-genai usage fix (#55) depends on: when an
    // engine sets `n_tokens` on a single text-bearing final chunk, that count
    // is authoritative (not the "1 per chunk" fallback), so `usage` reflects
    // the real generated-token count rather than 0/1.
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
        let out = render_with_chat_env(&env, &msgs, "<bos>", "<eos>", true, None)
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
        let msg = ChatChoiceMessage { role: "assistant", content: Some("hi".into()), tool_calls: None };
        let v = serde_json::to_value(&msg).unwrap();
        assert_eq!(v["content"], "hi");
        assert!(v.get("tool_calls").is_none(), "no tool_calls key on non-tool message");
    }

    #[test]
    fn parse_llama_python_tag_single() {
        let calls = parse_tool_calls("<|python_tag|>{\"name\":\"get_weather\",\"parameters\":{\"city\":\"Paris\"}}").expect("one");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap()["city"], "Paris");
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].r#type, "function");
    }

    #[test]
    fn parse_qwen_hermes_single() {
        let calls = parse_tool_calls("<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>").expect("one");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap()["city"], "Paris");
    }

    #[test]
    fn parse_qwen_multiple_distinct_ids() {
        let calls = parse_tool_calls("<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\
            <tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>").expect("two");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[0].function.arguments, "{}");
        assert_eq!(calls[1].function.name, "b");
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn parse_skips_malformed_first_block_keeps_valid() {
        let calls = parse_tool_calls("<tool_call>{bad</tool_call><tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>").expect("one");
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
        let calls = parse_tool_calls("<tool_call>{\"name\":\"f\",\"arguments\":\"{\\\"k\\\":1}\"}</tool_call>").expect("one");
        assert_eq!(calls[0].function.arguments, "{\"k\":1}");
    }

    const TOOL_TEMPLATE: &str = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\
        {% if m.tool_calls %}[CALLS:{% for c in m.tool_calls %}{{ c.function.name }}{% endfor %}]{% endif %}\
        {% if m.tool_call_id %}[TCID:{{ m.tool_call_id }}]{% endif %}\n{% endfor %}\
        {% if tools %}[TOOLS:{% for t in tools %}{{ t.function.name }}{% endfor %}]{% endif %}";

    #[test]
    fn renderer_forwards_tools_into_template() {
        let env = build_chat_env(TOOL_TEMPLATE).unwrap();
        let msgs = [ChatMessage { role: "user".into(), content: "hi".into(),
            name: None, tool_calls: None, tool_call_id: None }];
        let tools = vec![Tool { r#type: "function".into(), function: serde_json::json!({"name":"get_weather"}) }];
        let out = render_with_chat_env(&env, &msgs, "", "", true, Some(&tools)).unwrap();
        assert!(out.contains("[TOOLS:get_weather]"), "tools not forwarded: {out}");
    }

    #[test]
    fn renderer_forwards_assistant_tool_calls_and_tool_result() {
        let env = build_chat_env(TOOL_TEMPLATE).unwrap();
        let msgs = [
            ChatMessage { role: "assistant".into(), content: "".into(), name: None,
                tool_calls: Some(vec![ToolCall { id: "call_1".into(), r#type: "function".into(),
                    function: FunctionCall { name: "get_weather".into(), arguments: "{}".into() } }]),
                tool_call_id: None },
            ChatMessage { role: "tool".into(), content: "18C".into(), name: Some("get_weather".into()),
                tool_calls: None, tool_call_id: Some("call_1".into()) },
        ];
        let out = render_with_chat_env(&env, &msgs, "", "", true, None).unwrap();
        assert!(out.contains("[CALLS:get_weather]"), "assistant tool_calls not rendered: {out}");
        assert!(out.contains("[TCID:call_1]"), "tool_call_id not rendered: {out}");
    }

    #[test]
    fn build_choice_emits_tool_calls_when_parsed() {
        let (m, fr) = build_choice(
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>".into(),
            &Some(serde_json::json!("auto")));
        assert_eq!(fr, "tool_calls");
        assert!(m.content.is_none(), "tool-call message content must be null");
        assert_eq!(m.tool_calls.unwrap()[0].function.name, "get_weather");
    }

    #[test]
    fn build_choice_none_skips_parse() {
        let (m, fr) = build_choice(
            "<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>".into(),
            &Some(serde_json::json!("none")));
        assert_eq!(fr, "stop");
        assert!(m.tool_calls.is_none());
        assert!(m.content.is_some(), "non-parsed content preserved");
    }

    #[test]
    fn build_choice_plain_is_stop() {
        let (m, fr) = build_choice("hello".into(), &None);
        assert_eq!(fr, "stop");
        assert!(m.tool_calls.is_none());
        assert_eq!(m.content.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn tool_request_returns_tool_calls_finish_e2e() {
        let app = make_app().await;
        let tc = "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>";
        let payload = serde_json::json!({"model":"mock-model","messages":[{"role":"user","content":tc}],
            "tool_choice":"auto","tools":[{"type":"function","function":{"name":"get_weather"}}],"stream":false});
        let resp = app.oneshot(Request::builder().method("POST").uri("/v1/chat/completions")
            .header("content-type","application/json").body(Body::from(payload.to_string())).unwrap()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: Value = serde_json::from_slice(&to_bytes(resp.into_body(), 8192).await.unwrap()).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        assert!(v["choices"][0]["message"]["content"].is_null());
        assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "get_weather");
    }
}
