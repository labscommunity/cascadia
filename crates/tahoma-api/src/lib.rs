//! OpenAI-compatible HTTP server.
//!
//! Mirrors the subset of `tahoma/api/server.py` needed for parity with PR
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
use chrono::Utc;
use futures::stream::{self, Stream};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tahoma_runner::Runner;
use tahoma_types::GenerationTask;
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
    /// Optional Jinja2 chat template (from tokenizer_config.json's
    /// `chat_template` field). When set, /v1/chat/completions renders
    /// messages through it; when None, falls back to the legacy
    /// "role: content\n…" join.
    pub chat_template: Option<Arc<str>>,
    pub bos_token: Arc<str>,
    pub eos_token: Arc<str>,
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT,
            max_prompt_bytes: DEFAULT_MAX_PROMPT_BYTES,
            chat_template: ChatTemplateConfig::default(),
        }
    }
}

/// Read tokenizer_config.json from the shard's `tokenizer/` subdir and
/// extract the chat_template + bos/eos token strings. Returns Default
/// (all None) if any step fails — the caller falls back to the legacy
/// "role: content" formatting in that case. Some HF tokenizer configs
/// store special tokens as objects ({"content": "...", ...}) rather than
/// strings; both shapes are handled.
pub fn load_chat_template_config(model_dir: &std::path::Path) -> ChatTemplateConfig {
    let p = model_dir.join("tokenizer").join("tokenizer_config.json");
    let Ok(bytes) = std::fs::read(&p) else {
        return ChatTemplateConfig::default();
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return ChatTemplateConfig::default();
    };
    let template = v
        .get("chat_template")
        .and_then(|t| t.as_str())
        .map(|s| s.to_owned());
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
        chat_template: cfg.chat_template.template.map(Arc::from),
        bos_token: Arc::from(cfg.chat_template.bos_token.unwrap_or_default()),
        eos_token: Arc::from(cfg.chat_template.eos_token.unwrap_or_default()),
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
            owned_by: "tahoma",
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

/// Last-resort formatting when no chat template is available. Matches
/// the Python "no chat template" engines and what tahoma shipped pre-
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

/// Render messages through the model's HF Jinja2 chat_template. Returns
/// Err on any template parse/render error so the caller can fall back to
/// [`render_prompt_legacy`] rather than fail the request entirely.
fn render_prompt_with_template(
    template_src: &str,
    messages: &[ChatMessage],
    bos_token: &str,
    eos_token: &str,
) -> Result<String, String> {
    use minijinja::value::Value;
    use minijinja::{context, Environment, Error, ErrorKind};

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

    env.add_template("chat", template_src)
        .map_err(|e| format!("template parse: {e}"))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| format!("template lookup: {e}"))?;

    let messages_value: Vec<Value> = messages
        .iter()
        .map(|m| {
            Value::from_serialize(&serde_json::json!({
                "role": m.role,
                "content": m.content,
            }))
        })
        .collect();

    let ctx = context! {
        messages => messages_value,
        add_generation_prompt => true,
        bos_token => bos_token,
        eos_token => eos_token,
    };

    tmpl.render(ctx)
        .map_err(|e| format!("template render: {e}"))
}

fn render_prompt(state: &AppState, messages: &[ChatMessage]) -> String {
    if let Some(tmpl) = &state.chat_template {
        match render_prompt_with_template(tmpl, messages, &state.bos_token, &state.eos_token) {
            Ok(s) => return s,
            Err(e) => {
                warn!(error = %e, "chat_template render failed; falling back to legacy formatter");
            }
        }
    }
    render_prompt_legacy(messages)
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let task_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let prompt = render_prompt(&state, &req.messages);
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
        enable_thinking: false,
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

    if req.stream {
        return stream_completion(state, req.model, task, permit)
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
    while let Some(chunk) = chunk_stream.next().await {
        if !chunk.is_final {
            buf.push_str(&chunk.text);
            completion_tokens += 1;
        } else if !chunk.text.is_empty() {
            buf.push_str(&chunk.text);
        }
    }

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
            prompt_tokens: 0, // tahoma engines don't surface this today
            completion_tokens,
            total_tokens: completion_tokens,
        },
    })
    .into_response()
}

async fn stream_completion(
    state: AppState,
    model: String,
    task: GenerationTask,
    permit: tokio::sync::OwnedSemaphorePermit,
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
    // accepted connection in tahoma-cli's serve loop, this lands the
    // per-token chunk on the wire as a separate ~220 B packet).
    let body_stream = permit_carrier
        .then(move |chunk| {
            let model = model.clone();
            let task_id = task_id.clone();
            async move {
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
                // this frame to TCP. tahoma_runner::ChunkStream::poll_next
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

/// Wraps a [`tahoma_runner::ChunkStream`] so it carries its concurrency
/// permit and the permit is released when the stream is dropped (which
/// happens on client disconnect — axum drops the SSE response).
struct StreamWithPermit {
    inner: tahoma_runner::ChunkStream,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl Stream for StreamWithPermit {
    type Item = tahoma_types::Chunk;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

fn engine_error_response(err: tahoma_engine::EngineError) -> axum::response::Response {
    use tahoma_engine::EngineError;
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
    use serde_json::Value;
    use tahoma_engine_mock::MockBuilder;
    use tahoma_types::{PeerLayout, ShardSpec};
    use tower::ServiceExt;

    async fn make_app() -> Router {
        let runner = Runner::new(Box::new(MockBuilder::new()));
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
    async fn chat_completion_omitting_stream_defaults_to_non_streaming() {
        // Regression guard: OpenAI clients that never set `stream` must
        // continue to get a single JSON body. `stream` defaults to false
        // via `#[serde(default)]` on the bool field.
        let app = make_app().await;
        let payload = serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "alpha"}],
            "max_tokens": 1,
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
        // Non-streaming path returns application/json, not text/event-stream.
        let ct = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|h| h.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.starts_with("application/json"),
            "expected JSON content-type, got {ct:?}"
        );
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["object"], "chat.completion");
    }

    #[tokio::test]
    async fn chat_completion_stream_emits_openai_sse_format() {
        // End-to-end SSE shape check. We don't pin token text (the mock
        // engine echoes prompt words) — we pin the OpenAI envelope:
        //   - Content-Type: text/event-stream
        //   - One or more `data: {json}\n\n` frames whose JSON has
        //     `object: "chat.completion.chunk"` and a `choices[0].delta`
        //     with role/content + finish_reason on the last chunk.
        //   - Terminator frame `data: [DONE]\n\n`.
        let app = make_app().await;
        let payload = serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "alpha bravo charlie"}],
            "max_tokens": 3,
            "stream": true,
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

        let ct = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|h| h.to_str().ok())
            .unwrap_or("");
        assert_eq!(
            ct, "text/event-stream",
            "stream=true must return SSE content-type"
        );
        let cache = response
            .headers()
            .get(header::CACHE_CONTROL)
            .and_then(|h| h.to_str().ok())
            .unwrap_or("");
        assert!(
            cache.contains("no-cache"),
            "SSE response must disable caching, got {cache:?}"
        );

        // Collect the full body; the mock engine produces only a handful
        // of frames so a generous 64 KiB cap is plenty.
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let text = std::str::from_utf8(&body).expect("SSE body must be UTF-8");

        // Frames are separated by the SSE delimiter "\n\n". Split and
        // drop the trailing empty element from the final delimiter.
        let frames: Vec<&str> = text.split("\n\n").filter(|s| !s.is_empty()).collect();
        assert!(
            frames.len() >= 2,
            "expected at least one data frame plus [DONE], got {frames:?}"
        );
        // Last frame must be the OpenAI terminator.
        assert_eq!(
            *frames.last().unwrap(),
            "data: [DONE]",
            "stream must end with `data: [DONE]\\n\\n`"
        );

        // Every non-terminator frame must be a `data: <json>` line whose
        // JSON parses and has the chat.completion.chunk shape.
        let mut saw_content = false;
        let mut saw_finish = false;
        for frame in &frames[..frames.len() - 1] {
            let payload = frame
                .strip_prefix("data: ")
                .unwrap_or_else(|| panic!("frame missing `data: ` prefix: {frame:?}"));
            let v: Value = serde_json::from_str(payload)
                .unwrap_or_else(|e| panic!("frame is not JSON ({e}): {payload:?}"));
            assert_eq!(v["object"], "chat.completion.chunk");
            assert_eq!(v["model"], "mock-model");
            let choice = &v["choices"][0];
            assert_eq!(choice["index"], 0);
            // delta.role MUST be assistant per OpenAI streaming spec.
            assert_eq!(choice["delta"]["role"], "assistant");
            if choice["delta"]["content"]
                .as_str()
                .is_some_and(|s| !s.is_empty())
            {
                saw_content = true;
            }
            if choice["finish_reason"] == "stop" {
                saw_finish = true;
            }
        }
        assert!(saw_content, "expected at least one chunk with content");
        assert!(
            saw_finish,
            "expected at least one chunk with finish_reason=stop"
        );
    }
}
