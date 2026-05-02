//! OpenAI-compatible HTTP server.
//!
//! Mirrors the subset of `tahoma/api/server.py` needed for parity with PR
//! #2's e2e bench: `/health`, `/v1/models`, `/v1/chat/completions`
//! (non-streaming + SSE streaming). Tools, logprobs, /events, /state,
//! Ollama dialect deferred — see Phase 5 follow-up.

use std::sync::Arc;
use std::time::SystemTime;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::Utc;
use futures::stream::{self, Stream};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tahoma_runner::Runner;
use tahoma_types::GenerationTask;
use tracing::info;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub runner: Arc<Runner>,
    pub model_id: String,
}

pub fn make_router(runner: Arc<Runner>, model_id: impl Into<String>) -> Router {
    let state = AppState {
        runner,
        model_id: model_id.into(),
    };
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/cancel/:task_id", post(cancel))
        .with_state(state)
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

fn render_prompt(messages: &[ChatMessage]) -> String {
    // Minimal chat formatting — caller's model is expected to accept the
    // raw concatenation. Matches the Python "no chat template" engines.
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

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let task_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let prompt = render_prompt(&req.messages);
    let task = GenerationTask {
        task_id: task_id.clone(),
        prompt,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        logprobs: 0,
        enable_thinking: false,
        trust_remote_code: false,
    };

    if req.stream {
        return stream_completion(state, req.model, task).await.into_response();
    }

    // Non-streaming: collect full output.
    let mut chunk_stream = match state.runner.generate(task.clone()) {
        Ok(s) => s,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
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
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let task_id = task.task_id.clone();
    let _ = SystemTime::now();

    let chunk_stream = state.runner.generate(task).expect("generate");
    let mapped = chunk_stream.map(move |chunk| {
        let payload = serde_json::json!({
            "id": task_id,
            "object": "chat.completion.chunk",
            "created": now_unix(),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": chunk.text,
                },
                "finish_reason": if chunk.is_final { Some("stop") } else { None },
            }],
        });
        Ok(Event::default().data(payload.to_string()))
    });
    let final_event = stream::once(async {
        Ok(Event::default().data("[DONE]".to_string()))
    });
    Sse::new(mapped.chain(final_event)).keep_alive(KeepAlive::default())
}

async fn cancel(
    State(state): State<AppState>,
    Path(task_id): Path<String>,
) -> impl IntoResponse {
    state.runner.cancel(&task_id);
    info!(task = %task_id, "cancelled");
    (StatusCode::OK, Json(serde_json::json!({"cancelled": task_id})))
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
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
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
}
