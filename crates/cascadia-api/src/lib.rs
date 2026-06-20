//! OpenAI-compatible HTTP server.
//!
//! Mirrors the subset of `cascadia/api/server.py` needed for parity with PR
//! #2's e2e bench: `/health`, `/v1/models`, `/v1/chat/completions`
//! (non-streaming + SSE streaming). Tools, logprobs, /events, /state,
//! Ollama dialect deferred — see Phase 5 follow-up.

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

#[derive(Clone)]
pub struct AppState {
    pub runner: Arc<Runner>,
    pub model_id: String,
    pub permits: Arc<Semaphore>,
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
    let state = AppState {
        runner,
        model_id: model_id.into(),
        permits: Arc::new(Semaphore::new(cfg.max_concurrent_requests)),
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

#[derive(Deserialize, Debug, Clone, PartialEq)]
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
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
}

fn default_true() -> bool {
    true
}

fn default_max_tokens() -> u32 {
    256
}

/// Request-surface structured-output spec. Upstream twin of enterprise
/// `cascadia_protocol::ResponseFormat`; bridged like `Tool`. Only `JsonSchema` in slice 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResponseFormat {
    JsonSchema { name: String, schema: String, strict: bool },
}

/// Parse+validate a partner `response_format` Value. `Ok(None)` for absent/null/text;
/// `Err` on a malformed json_schema (stronger contract than tool_choice). Shared by the
/// standalone server and the enterprise gateway.
pub fn validate_response_format(v: Option<&serde_json::Value>) -> Result<Option<ResponseFormat>, String> {
    let v = match v {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(v) => v,
    };
    match v.get("type").and_then(|t| t.as_str()).unwrap_or("") {
        "text" => Ok(None),
        "json_schema" => {
            let js = v.get("json_schema").and_then(|j| j.as_object())
                .ok_or("response_format.json_schema must be an object")?;
            let name = js.get("name").and_then(|n| n.as_str())
                .ok_or("response_format.json_schema.name is required")?.to_string();
            let schema = js.get("schema").filter(|s| s.is_object())
                .ok_or("response_format.json_schema.schema must be an object")?;
            let strict = js.get("strict").and_then(|s| s.as_bool()).unwrap_or(false);
            Ok(Some(ResponseFormat::JsonSchema {
                name,
                schema: serde_json::to_string(schema).map_err(|e| e.to_string())?,
                strict,
            }))
        }
        other => Err(format!("unsupported response_format.type: {other}")),
    }
}

/// Derive the backend-neutral engine grammar (slice 1: response_format-only).
pub fn grammar_from_response_format(rf: Option<&ResponseFormat>) -> Option<cascadia_types::GrammarSpec> {
    match rf {
        Some(ResponseFormat::JsonSchema { schema, .. }) => Some(cascadia_types::GrammarSpec {
            kind: cascadia_types::GrammarKind::JsonSchema,
            body: schema.clone(),
        }),
        None => None,
    }
}

/// Best-effort required/forced nudge text. `None` ⇒ no directive (auto/none).
pub fn tool_call_directive(forced_name: Option<&str>, is_required: bool) -> Option<String> {
    match forced_name {
        Some(name) => Some(format!("You must call the `{name}` tool.")),
        None if is_required => Some("You must call one of the provided tools.".to_string()),
        None => None,
    }
}

/// Normalize a partner `tool_choice` Value → (forced_name, is_required).
pub fn normalize_tool_choice_value(tc: Option<&serde_json::Value>) -> (Option<String>, bool) {
    match tc {
        Some(serde_json::Value::String(s)) if s == "required" => (None, true),
        Some(serde_json::Value::Object(o)) => (
            o.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()).map(str::to_string),
            true,
        ),
        _ => (None, false),
    }
}

/// Receipt-safe: append the required/forced directive to a render-local messages copy.
pub fn apply_tool_directive(msgs: &mut Vec<ChatMessage>, forced_name: Option<&str>, is_required: bool) {
    if let Some(d) = tool_call_directive(forced_name, is_required) {
        msgs.push(ChatMessage { role: "system".to_string(), content: d, name: None, tool_calls: None, tool_call_id: None });
    }
}

#[derive(Serialize)]
struct ChatChoiceMessage {
    role: &'static str,
    /// `Option` so a tool-call turn emits `content: null` (OpenAI shape).
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
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

/// Last-resort formatting when no chat template is available. Matches
/// the Python "no chat template" engines and what cascadia shipped pre-
/// minijinja. Coherent on permissive base models, brittle on instruct
/// models that expect their own prompt format.
fn render_prompt_legacy(messages: &[ChatMessage]) -> String {
    let mut buf = String::new();
    for m in messages {
        if !buf.is_empty() {
            buf.push('\n');
        }
        buf.push_str(&m.role);
        buf.push_str(": ");
        buf.push_str(&m.content);
    }
    buf
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
    let tmpl = env
        .get_template("chat")
        .map_err(|e| format!("template lookup: {e}"))?;
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
    tmpl.render(ctx)
        .map_err(|e| format!("template render: {e}"))
}

fn render_prompt(
    state: &AppState,
    messages: &[ChatMessage],
    enable_thinking: bool,
    tools: Option<&[Tool]>,
) -> String {
    let defer_to_engine = state.defer_template_on_thinking && enable_thinking;
    if !defer_to_engine {
        if let Some(env) = &state.chat_env {
            match render_with_chat_env(
                env,
                messages,
                &state.bos_token,
                &state.eos_token,
                enable_thinking,
                tools,
            ) {
                Ok(s) => return s,
                Err(e) => {
                    warn!(error = %e, "chat_template render failed; falling back to legacy formatter")
                }
            }
        }
    }
    render_prompt_legacy(messages)
}

/// Decide the response message + finish_reason from accumulated text.
/// `tool_choice == Some("none")` skips parsing (always "stop").
fn build_choice(
    buf: String,
    tool_choice: &Option<serde_json::Value>,
    tools_present: bool,
) -> (ChatChoiceMessage, &'static str) {
    let parse_enabled =
        tools_present && tool_choice.as_ref().and_then(|v| v.as_str()) != Some("none");
    if parse_enabled {
        if let Some(calls) = parse_tool_calls(&buf) {
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
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(after[off..off + len].trim())
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
            if let Some(c) = call_from_xml_function(&after[..end]) {
                // Qwen3/MoE <function=…><parameter=…> XML dialect (non-JSON).
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
            match render_with_chat_env(env, messages, &self.bos_token, &self.eos_token, true, tools)
            {
                Ok(s) => return s,
                Err(e) => {
                    warn!(error = %e, "chat_template render failed; falling back to legacy formatter")
                }
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

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let task_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let prompt = render_prompt(
        &state,
        &req.messages,
        req.enable_thinking,
        req.tools.as_deref(),
    );
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
        grammar: None,
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

    if req.stream {
        let tool_choice = req.tool_choice.clone();
        let tools_present = req.tools.as_ref().is_some_and(|t| !t.is_empty());
        return stream_completion(state, req.model, task, permit, tool_choice, tools_present)
            .await
            .into_response();
    }

    // Non-streaming: collect full output. Hold the permit until the
    // task completes; drop frees the slot.
    let _permit = permit;
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

    let (message, finish_reason) = build_choice(
        buf,
        &req.tool_choice,
        req.tools.as_ref().is_some_and(|t| !t.is_empty()),
    );
    Json(ChatCompletionResponse {
        id: task_id,
        object: "chat.completion",
        created: now_unix(),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message,
            finish_reason,
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
    tool_choice: Option<serde_json::Value>,
    tools_present: bool,
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
    // Buffer + emit one tool delta only when tools are present and not disabled.
    let tools_active =
        tools_present && tool_choice.as_ref().and_then(|v| v.as_str()) != Some("none");
    if tools_active {
        let _permit = permit; // hold the slot until generation completes
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
        } else if let Some(calls) = parse_tool_calls(&buf) {
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
    // Move the permit into the stream so it's released only when the
    // body is dropped (client disconnect or final chunk).
    let permit_carrier = StreamWithPermit {
        inner: chunk_stream,
        _permit: permit,
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
            async move {
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
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap()["city"],
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
        let calls =
            parse_tool_calls("<tool_call><function=g><parameter=x>v</parameter></function></tool_call>")
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
        let out = render_with_chat_env(&env, &msgs, "", "", true, Some(&tools)).unwrap();
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
        let out = render_with_chat_env(&env, &msgs, "", "", true, Some(&tools))
            .expect("tojson template must render (needs minijinja `json` feature)");
        assert!(
            out.contains("get_weather"),
            "tojson did not serialize tools: {out}"
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
        let out = render_with_chat_env(&env, &msgs, "", "", true, None).unwrap();
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
        );
        assert_eq!(fr, "stop");
        assert!(m.tool_calls.is_none());
        assert!(m.content.is_some());
    }

    #[test]
    fn build_choice_plain_is_stop() {
        let (m, fr) = build_choice("hello".into(), &None, true);
        assert_eq!(fr, "stop");
        assert!(m.tool_calls.is_none());
        assert_eq!(m.content.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn tool_request_returns_tool_calls_finish_e2e() {
        let app = make_app().await;
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
        let app = make_app().await;
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
        let app = make_app().await;
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

    #[test]
    fn chat_completion_request_carries_response_format() {
        let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}}"#;
        let req: ChatCompletionRequest = serde_json::from_str(body).unwrap();
        assert!(validate_response_format(req.response_format.as_ref()).unwrap().is_some());
    }

    #[test]
    fn validate_response_format_accept_reject_none() {
        let ok = serde_json::json!({"type":"json_schema","json_schema":{"name":"a","schema":{"type":"object"},"strict":true}});
        let rf = validate_response_format(Some(&ok)).unwrap().unwrap();
        let ResponseFormat::JsonSchema { name, schema, strict } = rf;
        assert_eq!(name, "a");
        assert!(strict);
        assert_eq!(serde_json::from_str::<serde_json::Value>(&schema).unwrap()["type"], "object");

        assert!(validate_response_format(None).unwrap().is_none());
        assert!(validate_response_format(Some(&serde_json::json!({"type":"text"}))).unwrap().is_none());
        assert!(validate_response_format(Some(&serde_json::json!({"type":"json_schema"}))).is_err());
        assert!(validate_response_format(Some(&serde_json::json!({"type":"json_schema","json_schema":{"schema":{"type":"object"}}}))).is_err());
        assert!(validate_response_format(Some(&serde_json::json!({"type":"json_schema","json_schema":{"name":"x","schema":"nope"}}))).is_err());
        assert!(validate_response_format(Some(&serde_json::json!({"type":"json_obj"}))).is_err());
    }

    #[test]
    fn grammar_from_response_format_maps_json_schema() {
        let rf = ResponseFormat::JsonSchema { name: "x".into(), schema: r#"{"type":"object"}"#.into(), strict: true };
        let g = grammar_from_response_format(Some(&rf)).unwrap();
        assert!(matches!(g.kind, cascadia_types::GrammarKind::JsonSchema));
        assert_eq!(g.body, r#"{"type":"object"}"#);
        assert!(grammar_from_response_format(None).is_none());
    }

    #[test]
    fn tool_directive_helpers() {
        assert_eq!(tool_call_directive(None, false), None);
        assert_eq!(tool_call_directive(None, true).as_deref(), Some("You must call one of the provided tools."));
        assert_eq!(tool_call_directive(Some("get_weather"), true).as_deref(), Some("You must call the `get_weather` tool."));
        // normalize from the partner Value
        assert_eq!(normalize_tool_choice_value(Some(&serde_json::json!("required"))), (None, true));
        assert_eq!(
            normalize_tool_choice_value(Some(&serde_json::json!({"type":"function","function":{"name":"f"}}))),
            (Some("f".to_string()), true)
        );
        assert_eq!(normalize_tool_choice_value(Some(&serde_json::json!("auto"))), (None, false));
        assert_eq!(normalize_tool_choice_value(None), (None, false));
    }

    #[test]
    fn apply_tool_directive_appends_only_when_required() {
        let base = vec![ChatMessage { role: "user".into(), content: "hi".into(), name: None, tool_calls: None, tool_call_id: None }];
        let mut m = base.clone();
        apply_tool_directive(&mut m, None, false);
        assert_eq!(m, base); // auto/none: byte-identical, no injection
        let mut m2 = base.clone();
        apply_tool_directive(&mut m2, None, true);
        assert_eq!(m2.len(), 2);
        assert_eq!(m2[1].role, "system");
        assert_eq!(m2[1].content, "You must call one of the provided tools.");
        assert_eq!(m2[0], base[0]); // original user message preserved
    }
}
