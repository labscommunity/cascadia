//! `/v1/responses`: OpenAI's Responses API over the chat path.
//!
//! A Responses request is mapped onto a [`ChatCompletionRequest`] and admitted
//! by the same code as `/v1/chat/completions` (`admit_chat`), so the chat
//! template, sampling, image decoding, limits and every refusal there apply
//! here unchanged. Only the request and response shapes differ.
//!
//! Cascadia keeps no state and serves no tools on this endpoint, so the fields
//! that need either (`previous_response_id`, `store: true`, `conversation`,
//! `background`, `tools`, file ids) are refused with a 400 naming the field,
//! never ignored: a dropped `previous_response_id` would have the model answer
//! without the conversation the caller thinks it sent.

use super::*;
use serde_json::{json, Value};

/// OpenAI `POST /v1/responses` body. The fields after `metadata` are
/// accepted only to be refused by name (see [`refuse_unsupported`]); unknown
/// fields are ignored, as on the chat endpoint.
#[derive(Deserialize, Debug)]
pub struct ResponsesRequest {
    pub model: String,
    /// A string (one user message) or an array of input items.
    pub input: Value,
    /// Becomes a leading `system` message.
    #[serde(default)]
    pub instructions: Option<String>,
    /// `max_tokens` on the chat path (same default when omitted).
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    /// Omitted = 0.0, the chat endpoint's default.
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    /// Only `false` (or absent): nothing is stored.
    #[serde(default)]
    pub store: Option<bool>,
    /// `reasoning.effort` is the chat endpoint's `reasoning_effort`.
    #[serde(default)]
    pub reasoning: Option<ReasoningParam>,
    /// Only `{"format": {"type": "text"}}`; structured output is refused.
    #[serde(default)]
    pub text: Option<Value>,
    /// Only `"disabled"`: a prompt over the window is a 413, never trimmed.
    #[serde(default)]
    pub truncation: Option<String>,
    /// Only `"none"` or `"auto"`, which with no tools mean the same.
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// Echoed back on the response object.
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub previous_response_id: Option<Value>,
    #[serde(default)]
    pub conversation: Option<Value>,
    #[serde(default)]
    pub background: Option<Value>,
    #[serde(default)]
    pub include: Option<Value>,
    #[serde(default)]
    pub prompt: Option<Value>,
    #[serde(default)]
    pub top_logprobs: Option<Value>,
    /// Not a Responses API parameter; refused rather than half-honored.
    #[serde(default)]
    pub stop: Option<Value>,
}

/// OpenAI `reasoning`: `{effort, summary}`.
#[derive(Deserialize, Debug, Default)]
pub struct ReasoningParam {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub summary: Option<Value>,
}

/// A 400 naming the request field at fault.
#[derive(Debug, PartialEq)]
struct Refusal {
    param: String,
    message: String,
}

impl Refusal {
    fn new(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            param: param.into(),
            message: message.into(),
        }
    }

    fn into_response(self) -> Response {
        count_rejected("unsupported_field");
        api_error_for(
            StatusCode::BAD_REQUEST,
            self.message,
            Some(&self.param),
            None,
        )
    }
}

const STATELESS: &str = "this server keeps no responses; send the whole conversation in `input`";

/// Refuse every request field this endpoint cannot honor, by name.
fn refuse_unsupported(req: &ResponsesRequest) -> Result<(), Refusal> {
    if req.store == Some(true) {
        return Err(Refusal::new(
            "store",
            format!("`store: true` is not supported: {STATELESS}, with `store: false`"),
        ));
    }
    if req.previous_response_id.is_some() {
        return Err(Refusal::new(
            "previous_response_id",
            format!("`previous_response_id` is not supported: {STATELESS}"),
        ));
    }
    if req.conversation.is_some() {
        return Err(Refusal::new(
            "conversation",
            format!("`conversation` is not supported: {STATELESS}"),
        ));
    }
    if req.background.as_ref().is_some_and(|b| b != &json!(false)) {
        return Err(Refusal::new(
            "background",
            "`background` is not supported: responses are generated within the request",
        ));
    }
    if req.tools.as_ref().is_some_and(|t| t != &json!([])) {
        return Err(Refusal::new(
            "tools",
            "`tools` are not supported on `/v1/responses`; use `/v1/chat/completions` \
             for function calling",
        ));
    }
    if let Some(choice) = &req.tool_choice {
        if !matches!(choice.as_str(), Some("none" | "auto")) {
            return Err(Refusal::new(
                "tool_choice",
                "`tool_choice` other than `none` or `auto` is not supported: \
                 `/v1/responses` serves no tools",
            ));
        }
    }
    if req.include.as_ref().is_some_and(|i| i != &json!([])) {
        return Err(Refusal::new(
            "include",
            "`include` is not supported: responses carry no extra output data",
        ));
    }
    if req.prompt.is_some() {
        return Err(Refusal::new(
            "prompt",
            "prompt templates (`prompt`) are not supported; send the text in `input`",
        ));
    }
    if req.top_logprobs.as_ref().is_some_and(|n| n != &json!(0)) {
        return Err(Refusal::new(
            "top_logprobs",
            "`top_logprobs` is not supported on `/v1/responses`; use `/v1/chat/completions` \
             with `logprobs`",
        ));
    }
    if req.stop.is_some() {
        return Err(Refusal::new(
            "stop",
            "`stop` is not a Responses API parameter; use `/v1/chat/completions` \
             for stop sequences",
        ));
    }
    if let Some(text) = &req.text {
        let kind = text.pointer("/format/type").and_then(Value::as_str);
        if kind.is_some_and(|k| k != "text") {
            return Err(Refusal::new(
                "text.format",
                format!(
                    "`text.format` of type `{}` is not supported: only plain `text` output",
                    kind.unwrap_or_default()
                ),
            ));
        }
    }
    if req.reasoning.as_ref().is_some_and(|r| r.summary.is_some()) {
        return Err(Refusal::new(
            "reasoning.summary",
            "`reasoning.summary` is not supported: reasoning stays in the output text",
        ));
    }
    if req.truncation.as_deref().is_some_and(|t| t != "disabled") {
        return Err(Refusal::new(
            "truncation",
            "only `truncation: \"disabled\"` is supported: a prompt over the model's \
             window is refused, not trimmed",
        ));
    }
    Ok(())
}

/// `input` as chat messages, `instructions` first as a `system` message.
/// Each item's content parts are rewritten into the chat endpoint's part
/// types and parsed by its own [`parse_content`], so text joining and image
/// placement match `/v1/chat/completions` exactly.
fn input_messages(req: &ResponsesRequest) -> Result<Vec<ChatMessage>, Refusal> {
    let mut out = Vec::new();
    if let Some(instructions) = &req.instructions {
        out.push(text_message("system", instructions.clone()));
    }
    let items = match &req.input {
        Value::String(s) => {
            out.push(text_message("user", s.clone()));
            return Ok(out);
        }
        Value::Array(items) => items,
        _ => {
            return Err(Refusal::new(
                "input",
                "`input` must be a string or an array of input items",
            ))
        }
    };
    for (i, item) in items.iter().enumerate() {
        let at = format!("input[{i}]");
        // `type` is optional on the shorthand message form `{role, content}`.
        let kind = item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        if kind != "message" {
            return Err(Refusal::new(
                &at,
                format!(
                    "input item type `{kind}` is not supported; only `message` items are \
                     accepted (no tools or stored items)"
                ),
            ));
        }
        let role = match item.get("role").and_then(Value::as_str) {
            Some(r @ ("user" | "assistant" | "system")) => r,
            // The chat templates know no `developer` turn; OpenAI ranks the
            // two alike.
            Some("developer") => "system",
            Some(other) => {
                return Err(Refusal::new(
                    format!("{at}.role"),
                    format!("role `{other}` is not supported"),
                ))
            }
            None => {
                return Err(Refusal::new(
                    format!("{at}.role"),
                    "input message is missing its `role`",
                ))
            }
        };
        let content = match item.get("content") {
            Some(Value::String(s)) => Value::String(s.clone()),
            Some(Value::Array(parts)) => Value::Array(
                parts
                    .iter()
                    .enumerate()
                    .map(|(j, p)| chat_part(p, &format!("{at}.content[{j}]")))
                    .collect::<Result<_, _>>()?,
            ),
            _ => {
                return Err(Refusal::new(
                    format!("{at}.content"),
                    "`content` must be a string or an array of content parts",
                ))
            }
        };
        let (content, parts) =
            parse_content(Some(content)).map_err(|e| Refusal::new(format!("{at}.content"), e))?;
        out.push(ChatMessage {
            role: role.to_owned(),
            content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            parts,
        });
    }
    Ok(out)
}

fn text_message(role: &str, content: String) -> ChatMessage {
    ChatMessage {
        role: role.to_owned(),
        content,
        name: None,
        tool_calls: None,
        tool_call_id: None,
        parts: Vec::new(),
    }
}

/// One Responses content part as the chat endpoint's `text` / `image_url`
/// part. `output_text` and `refusal` are a prior assistant turn's text.
fn chat_part(part: &Value, at: &str) -> Result<Value, Refusal> {
    let kind = part.get("type").and_then(Value::as_str).unwrap_or("");
    let field = |name: &str| part.get(name).and_then(Value::as_str);
    match kind {
        "input_text" | "output_text" | "refusal" => {
            let key = if kind == "refusal" { "refusal" } else { "text" };
            match field(key) {
                Some(text) => Ok(json!({"type": "text", "text": text})),
                None => Err(Refusal::new(
                    format!("{at}.{key}"),
                    format!("content part of type `{kind}` needs a string `{key}`"),
                )),
            }
        }
        "input_image" => match (field("image_url"), part.get("file_id")) {
            (Some(url), _) => Ok(json!({"type": "image_url", "image_url": {
                "url": url, "detail": field("detail")}})),
            (None, Some(f)) if !f.is_null() => Err(Refusal::new(
                format!("{at}.file_id"),
                "`file_id` images are not supported: this server keeps no files; \
                 send the image as a `data:` URL in `image_url`",
            )),
            _ => Err(Refusal::new(
                format!("{at}.image_url"),
                "content part of type `input_image` needs `image_url`",
            )),
        },
        "input_file" => Err(Refusal::new(
            at,
            "file inputs (`input_file`) are not supported",
        )),
        "input_audio" => Err(Refusal::new(
            at,
            "audio inputs (`input_audio`) are not supported",
        )),
        "" => Err(Refusal::new(
            format!("{at}.type"),
            "content part is missing its `type`",
        )),
        other => Err(Refusal::new(
            format!("{at}.type"),
            format!(
                "content part type `{other}` is not supported; only `input_text` and \
                 `input_image` parts are accepted"
            ),
        )),
    }
}

/// The chat request this Responses request is served as.
fn chat_request(req: &ResponsesRequest, messages: Vec<ChatMessage>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        max_tokens: req.max_output_tokens.unwrap_or_else(default_max_tokens),
        temperature: req.temperature.unwrap_or_default(),
        top_p: req.top_p,
        top_k: None,
        seed: None,
        stop: None,
        frequency_penalty: None,
        presence_penalty: None,
        logprobs: false,
        top_logprobs: None,
        stream: req.stream,
        stream_options: None,
        enable_thinking: None,
        reasoning_effort: req.reasoning.as_ref().and_then(|r| r.effort.clone()),
        chat_template_kwargs: None,
        tools: None,
        tool_choice: None,
    }
}

/// The request-derived fields of the response object, shared by the
/// non-streaming body and every streamed snapshot of it.
struct ResponseShell {
    id: String,
    message_id: String,
    created_at: i64,
    model: String,
    instructions: Option<String>,
    /// The cap in force, echoed even when the caller left it to the default:
    /// an `incomplete` response names it.
    max_output_tokens: u32,
    temperature: f32,
    top_p: f32,
    effort: Option<String>,
    metadata: Value,
}

impl ResponseShell {
    fn new(req: &ResponsesRequest, id: String) -> Self {
        Self {
            id,
            message_id: format!("msg_{}", Uuid::new_v4().simple()),
            created_at: now_unix(),
            model: req.model.clone(),
            instructions: req.instructions.clone(),
            max_output_tokens: req.max_output_tokens.unwrap_or_else(default_max_tokens),
            temperature: req.temperature.unwrap_or_default(),
            top_p: req.top_p.unwrap_or(1.0),
            effort: req.reasoning.as_ref().and_then(|r| r.effort.clone()),
            metadata: req.metadata.clone().unwrap_or_else(|| json!({})),
        }
    }

    /// The OpenAI `response` object. `outcome` is `None` while in progress.
    fn object(&self, status: &str, output: Value, outcome: Option<&Outcome>) -> Value {
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "completed_at": outcome.filter(|_| status == "completed").map(|_| now_unix()),
            "error": outcome.and_then(|o| o.error.as_ref()).map(|e| json!({
                "code": "server_error", "message": e})),
            "incomplete_details": (status == "incomplete")
                .then(|| json!({"reason": "max_output_tokens"})),
            "instructions": self.instructions,
            "max_output_tokens": self.max_output_tokens,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": false,
            "previous_response_id": null,
            "reasoning": {"effort": self.effort, "summary": null},
            "store": false,
            "temperature": self.temperature,
            "text": {"format": {"type": "text"}},
            "tool_choice": "none",
            "tools": [],
            "top_p": self.top_p,
            "truncation": "disabled",
            "usage": outcome.map(|o| json!({
                "input_tokens": o.prompt_tokens,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": o.completion_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": o.prompt_tokens + o.completion_tokens,
            })),
            "user": null,
            "metadata": self.metadata,
        })
    }

    /// The one output item: an assistant message holding the text.
    fn message(&self, status: &str, text: Option<&str>) -> Value {
        json!({
            "id": self.message_id,
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": text.map(|t| vec![output_text(t)]).unwrap_or_default(),
        })
    }
}

fn output_text(text: &str) -> Value {
    json!({"type": "output_text", "text": text, "annotations": [], "logprobs": []})
}

/// How a generation ended.
struct Outcome {
    prompt_tokens: u32,
    completion_tokens: u32,
    /// The engine's failure, for a `failed` response.
    error: Option<String>,
}

/// How the generation ended, in Responses terms.
#[derive(Clone, Copy, Debug, PartialEq)]
enum End {
    Completed,
    /// Stopped at `max_output_tokens`.
    TokenCap,
    /// Cancelled (`/v1/cancel`, or the engine's `Cancelled`), or the stream
    /// ended without a final chunk: the text is partial either way, so it is
    /// never reported `completed`.
    Cancelled,
}

impl End {
    /// From the final chunk's reason; `None` = no final chunk.
    fn from_final(reason: Option<cascadia_types::FinishReason>) -> Self {
        use cascadia_types::FinishReason;
        match reason {
            Some(FinishReason::Stop) => End::Completed,
            Some(FinishReason::Length) => End::TokenCap,
            Some(FinishReason::Cancelled) | None => End::Cancelled,
        }
    }

    /// The response's `status`: `incomplete` at the token cap and
    /// `cancelled` for a cancelled turn, as OpenAI's `ResponseStatus` has them.
    fn status(self) -> &'static str {
        match self {
            End::Completed => "completed",
            End::TokenCap => "incomplete",
            End::Cancelled => "cancelled",
        }
    }

    /// The message item's `status` (items have no `cancelled`).
    fn item_status(self) -> &'static str {
        match self {
            End::Completed => "completed",
            End::TokenCap | End::Cancelled => "incomplete",
        }
    }

    /// The terminal stream event. OpenAI defines none for a cancelled
    /// response, so it ends on `response.incomplete`, carrying `status:
    /// "cancelled"`: never `response.completed` for partial text.
    fn event(self) -> &'static str {
        match self {
            End::Completed => "response.completed",
            End::TokenCap | End::Cancelled => "response.incomplete",
        }
    }
}

/// `POST /v1/responses`.
pub(crate) async fn create_response(
    State(state): State<AppState>,
    OpenAiJson(req): OpenAiJson<ResponsesRequest>,
) -> Response {
    let messages = match refuse_unsupported(&req).and_then(|()| input_messages(&req)) {
        Ok(m) => m,
        Err(refusal) => return refusal.into_response(),
    };
    let chat = chat_request(&req, messages);
    // The response id doubles as the task id, so `/v1/cancel/<id>` works.
    let shell = ResponseShell::new(&req, format!("resp_{}", Uuid::new_v4().simple()));
    let admitted = match admit_chat(
        &state,
        &chat,
        shell.id.clone(),
        "`input` is empty or rendered to an empty prompt",
    )
    .await
    {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    if req.stream {
        return stream_response(state, shell, admitted).await;
    }
    let c = match collect_generation(&state, admitted, &[]).await {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let end = End::from_final(c.end);
    let outcome = Outcome {
        prompt_tokens: c.prompt_tokens,
        completion_tokens: c.completion_tokens,
        error: None,
    };
    Json(shell.object(
        end.status(),
        json!([shell.message(end.item_status(), Some(&c.text))]),
        Some(&outcome),
    ))
    .into_response()
}

/// A streamed response's state between SSE frames.
struct ResponseStream {
    shell: ResponseShell,
    chunks: StreamWithPermit,
    marker: Option<MarkerFilter>,
    stats: Arc<ApiStats>,
    text: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    sequence: u64,
    started: bool,
    done: bool,
}

impl ResponseStream {
    /// Append one SSE event: `event: <type>` plus the payload carrying its
    /// `type` and `sequence_number`, as OpenAI streams them.
    fn emit(&mut self, out: &mut String, kind: &str, mut payload: Value) {
        payload["type"] = json!(kind);
        payload["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        out.push_str(&format!("event: {kind}\ndata: {payload}\n\n"));
    }

    /// The part/item address every text event carries.
    fn at(&self) -> Value {
        json!({"item_id": self.shell.message_id, "output_index": 0, "content_index": 0})
    }

    fn with_at(&self, extra: Value) -> Value {
        let mut v = self.at();
        if let (Some(v), Value::Object(extra)) = (v.as_object_mut(), extra) {
            v.extend(extra);
        }
        v
    }

    /// The next frame: the opening events, then one per engine chunk, the
    /// closing events riding on the final one.
    async fn next_frame(&mut self) -> String {
        let mut out = String::new();
        if !self.started {
            self.started = true;
            let created = self.shell.object("in_progress", json!([]), None);
            self.emit(&mut out, "response.created", json!({"response": created}));
            let progress = self.shell.object("in_progress", json!([]), None);
            self.emit(
                &mut out,
                "response.in_progress",
                json!({"response": progress}),
            );
            let item = self.shell.message("in_progress", None);
            self.emit(
                &mut out,
                "response.output_item.added",
                json!({"output_index": 0, "item": item}),
            );
            let part = self.with_at(json!({"part": output_text("")}));
            self.emit(&mut out, "response.content_part.added", part);
            return out;
        }
        let Some(chunk) = self.chunks.next().await else {
            // No final chunk: the runner ends a cancelled task's stream so.
            self.finish(&mut out, End::Cancelled);
            return out;
        };
        // The 200 is on the wire already, so a failed task is reported the
        // Responses way: a `response.failed` event, as the chat endpoint's
        // SSE error event does for its stream.
        if let Some(reason) = &chunk.error {
            warn!(task = %self.shell.id, reason = %reason, "engine failed task mid-stream; emitting response.failed");
            let outcome = Outcome {
                prompt_tokens: self.prompt_tokens,
                completion_tokens: self.completion_tokens,
                error: Some(reason.clone()),
            };
            let failed = self.shell.object("failed", json!([]), Some(&outcome));
            self.emit(&mut out, "response.failed", json!({"response": failed}));
            self.done = true;
            return out;
        }
        let n = chunk_token_count(&chunk);
        self.stats
            .tokens_total
            .fetch_add(n as u64, Ordering::Relaxed);
        self.completion_tokens += n;
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
        if !text.is_empty() {
            self.text.push_str(&text);
            let delta = self.with_at(json!({"delta": text, "logprobs": []}));
            self.emit(&mut out, "response.output_text.delta", delta);
        }
        if chunk.is_final {
            self.prompt_tokens = chunk.prompt_tokens.unwrap_or(0);
            let reason = chunk
                .finish_reason
                .unwrap_or(cascadia_types::FinishReason::Stop);
            self.finish(&mut out, End::from_final(Some(reason)));
        }
        // Let the body sink drain this frame first (see `stream_completion`).
        tokio::task::yield_now().await;
        out
    }

    /// The closing events: text, part and item done, then the response.
    fn finish(&mut self, out: &mut String, end: End) {
        let text = self.text.clone();
        let done = self.with_at(json!({"text": text, "logprobs": []}));
        self.emit(out, "response.output_text.done", done);
        let part = self.with_at(json!({"part": output_text(&text)}));
        self.emit(out, "response.content_part.done", part);
        let item = self.shell.message(end.item_status(), Some(&text));
        self.emit(
            out,
            "response.output_item.done",
            json!({"output_index": 0, "item": item}),
        );
        let outcome = Outcome {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            error: None,
        };
        let response = self
            .shell
            .object(end.status(), json!([item]), Some(&outcome));
        self.emit(out, end.event(), json!({"response": response}));
        self.done = true;
    }
}

/// SSE variant: OpenAI's Responses event sequence, raw frames written per
/// chunk like the chat stream's (see `stream_completion` on why not `Sse`).
/// No `[DONE]` sentinel: the Responses stream ends after its terminal event.
async fn stream_response(
    state: AppState,
    shell: ResponseShell,
    AdmittedChat {
        task,
        permit,
        inflight,
    }: AdmittedChat,
) -> Response {
    let chunks = match state.runner.generate_async(task).await {
        Ok(s) => s,
        Err(err) => {
            warn!(error = %err, "responses-stream: generate failed");
            return engine_error_response(err);
        }
    };
    let st = ResponseStream {
        shell,
        // The permit + in-flight guard ride in the stream, released when the
        // body is dropped (client disconnect or the terminal event).
        chunks: StreamWithPermit {
            inner: chunks,
            _permit: permit,
            _inflight: inflight,
        },
        marker: state.marker_dialect.clone().map(MarkerFilter::new),
        stats: state.stats.clone(),
        text: String::new(),
        prompt_tokens: 0,
        completion_tokens: 0,
        sequence: 0,
        started: false,
        done: false,
    };
    let body = stream::unfold(st, |mut st| async move {
        if st.done {
            return None;
        }
        let frame = st.next_frame().await;
        Some((
            Ok::<Bytes, std::convert::Infallible>(Bytes::from(frame)),
            st,
        ))
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(body))
        .unwrap()
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{make_app, make_image_app, png_url};
    use axum::body::to_bytes;
    use axum::http::Request;
    use tower::ServiceExt;

    async fn post(app: Router, payload: Value) -> (StatusCode, String) {
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
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

    async fn post_json(app: Router, payload: Value) -> (StatusCode, Value) {
        let (status, body) = post(app, payload).await;
        (status, serde_json::from_str(&body).unwrap_or(Value::Null))
    }

    fn request(v: Value) -> ResponsesRequest {
        serde_json::from_value(v).unwrap()
    }

    /// `input` as a string is one user message; as items, each part maps to
    /// the chat part it names, `instructions` leads as `system`, and
    /// `developer` reads as `system`.
    #[test]
    fn input_maps_onto_chat_messages() {
        let m = input_messages(&request(json!({"model": "m", "input": "hi there"}))).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(
            (m[0].role.as_str(), m[0].content.as_str()),
            ("user", "hi there")
        );

        let m = input_messages(&request(json!({"model": "m",
        "instructions": "be brief",
        "input": [
            {"role": "developer", "content": "terse"},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "what"},
                {"type": "input_image", "image_url": "data:x", "detail": "low"},
                {"type": "input_text", "text": "is this"}]},
            {"role": "assistant", "id": "msg_1", "status": "completed", "content": [
                {"type": "output_text", "text": "a dot", "annotations": []}]},
        ]})))
        .unwrap();
        let roles: Vec<_> = m.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "system", "user", "assistant"]);
        assert_eq!(m[0].content, "be brief");
        assert_eq!(m[2].content, "whatis this");
        assert_eq!(
            m[2].parts,
            vec![
                ContentPart::Text("what".into()),
                ContentPart::ImageUrl {
                    url: "data:x".into(),
                    detail: Some("low".into())
                },
                ContentPart::Text("is this".into()),
            ]
        );
        assert_eq!(m[3].content, "a dot");
        assert_eq!(m[3].parts, vec![ContentPart::Text("a dot".into())]);

        let chat = chat_request(
            &request(json!({"model": "m", "input": "x", "max_output_tokens": 7,
                "temperature": 0.5, "top_p": 0.9, "reasoning": {"effort": "low"}})),
            Vec::new(),
        );
        assert_eq!(chat.max_tokens, 7);
        assert_eq!(chat.temperature, 0.5);
        assert_eq!(chat.top_p, Some(0.9));
        assert_eq!(chat.reasoning_effort.as_deref(), Some("low"));
        let chat = chat_request(&request(json!({"model": "m", "input": "x"})), Vec::new());
        assert_eq!(chat.max_tokens, default_max_tokens());
    }

    #[tokio::test]
    async fn non_streaming_returns_a_response_object() {
        let (status, v) = post_json(
            make_app().await,
            json!({"model": "mock-model", "input": "alpha bravo charlie",
                "instructions": "be brief", "max_output_tokens": 64, "store": false}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert!(v["id"].as_str().unwrap().starts_with("resp_"));
        assert_eq!(v["object"], "response");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["model"], "mock-model");
        assert_eq!(v["instructions"], "be brief");
        assert_eq!(v["store"], false);
        assert!(v["incomplete_details"].is_null());
        let item = &v["output"][0];
        assert_eq!(v["output"].as_array().unwrap().len(), 1);
        assert_eq!(item["type"], "message");
        assert_eq!(item["role"], "assistant");
        assert_eq!(item["status"], "completed");
        assert!(item["id"].as_str().unwrap().starts_with("msg_"));
        let part = &item["content"][0];
        assert_eq!(part["type"], "output_text");
        assert_eq!(part["annotations"], json!([]));
        // The mock echoes prompt words.
        assert!(part["text"].as_str().unwrap().contains("alpha"), "{part}");
        let usage = &v["usage"];
        let (i, o) = (
            usage["input_tokens"].as_u64().unwrap(),
            usage["output_tokens"].as_u64().unwrap(),
        );
        assert!(o > 0);
        assert_eq!(usage["total_tokens"].as_u64().unwrap(), i + o);
    }

    /// Stopping at `max_output_tokens` is `incomplete`, as OpenAI reports it.
    #[tokio::test]
    async fn token_cap_is_an_incomplete_response() {
        let (status, v) = post_json(
            make_app().await,
            json!({"model": "mock-model", "input": "alpha bravo charlie delta echo",
                "max_output_tokens": 2}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["status"], "incomplete");
        assert_eq!(v["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(v["output"][0]["status"], "incomplete");
    }

    /// `(event, data)` pairs of an SSE body.
    fn events(body: &str) -> Vec<(String, Value)> {
        body.split("\n\n")
            .filter(|f| !f.trim().is_empty())
            .map(|f| {
                let mut lines = f.lines();
                let event = lines.next().unwrap().strip_prefix("event: ").unwrap();
                let data = lines.next().unwrap().strip_prefix("data: ").unwrap();
                (event.to_owned(), serde_json::from_str(data).unwrap())
            })
            .collect()
    }

    #[tokio::test]
    async fn streaming_emits_the_responses_event_sequence() {
        let (status, body) = post(
            make_app().await,
            json!({"model": "mock-model", "input": "one two three four",
                "max_output_tokens": 64, "stream": true}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(!body.contains("[DONE]"));
        let ev = events(&body);
        let kinds: Vec<_> = ev.iter().map(|(k, _)| k.as_str()).collect();
        let n = kinds.len();
        assert_eq!(
            kinds[..4],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added"
            ]
        );
        assert_eq!(
            kinds[n - 4..],
            [
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed"
            ]
        );
        assert!(kinds[4..n - 4]
            .iter()
            .all(|k| *k == "response.output_text.delta"));
        assert!(n > 8, "no deltas: {kinds:?}");
        for (i, (kind, data)) in ev.iter().enumerate() {
            assert_eq!(data["type"], kind.as_str());
            assert_eq!(data["sequence_number"], i as u64);
        }
        let created = &ev[0].1["response"];
        assert_eq!(created["status"], "in_progress");
        assert_eq!(created["output"], json!([]));
        let deltas: String = ev[4..n - 4]
            .iter()
            .map(|(_, d)| d["delta"].as_str().unwrap())
            .collect();
        let item_id = ev[2].1["item"]["id"].clone();
        assert!(ev[4..n - 2].iter().all(|(_, d)| d["item_id"] == item_id));
        assert_eq!(ev[n - 2].1["item"]["id"], item_id);
        assert_eq!(ev[n - 4].1["text"], deltas.as_str());
        let done = &ev[n - 1].1["response"];
        assert_eq!(done["status"], "completed");
        assert_eq!(done["id"], created["id"]);
        assert_eq!(done["output"][0]["content"][0]["text"], deltas.as_str());
        assert!(done["usage"]["output_tokens"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn streaming_token_cap_ends_with_response_incomplete() {
        let (_, body) = post(
            make_app().await,
            json!({"model": "mock-model", "input": "alpha bravo charlie delta echo",
                "max_output_tokens": 2, "stream": true}),
        )
        .await;
        let ev = events(&body);
        let (kind, data) = ev.last().unwrap();
        assert_eq!(kind, "response.incomplete");
        assert_eq!(
            data["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );
    }

    fn image_input(text: &str, url: &str) -> Value {
        json!([{"role": "user", "content": [
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": url, "detail": "auto"}]}])
    }

    /// An `input_image` is served like a chat `image_url` part: its
    /// placeholder lands where the caller put it (the mock echoes the prompt).
    #[tokio::test]
    async fn image_input_is_served_with_its_placeholder_in_place() {
        let (status, v) = post_json(
            make_image_app().await,
            json!({"model": "mock-model", "input": image_input("describe", &png_url(8, 8)),
                "max_output_tokens": 16}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        let text = v["output"][0]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("describe") && text.contains("<ov_genai_image_0>"),
            "placeholder missing from the echoed prompt: {text:?}"
        );
        // `detail: "original"` (the SDKs' full-resolution value) is served.
        let (status, v) = post_json(
            make_image_app().await,
            json!({"model": "mock-model", "max_output_tokens": 16, "input": [
                {"role": "user", "content": [
                    {"type": "input_text", "text": "describe"},
                    {"type": "input_image", "image_url": png_url(8, 8), "detail": "original"}]}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
    }

    /// The chat endpoint's image refusals hold here too: a text-only model,
    /// an undecodable image, an image outside a `user` turn, and reserved
    /// placeholder text.
    #[tokio::test]
    async fn image_refusals_match_the_chat_endpoint() {
        let png = png_url(4, 4);
        let (status, v) = post_json(
            make_app().await,
            json!({"model": "mock-model", "input": image_input("what?", &png)}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("does not accept image input"));
        let cases = [
            (
                image_input("x", "data:image/png;base64,aGVsbG8="),
                "image 0: unsupported image format",
            ),
            (
                json!([{"role": "developer", "content": [
                    {"type": "input_image", "image_url": png}]}]),
                "only accepted in `user` messages",
            ),
            (
                json!("see <ov_genai_image_0>"),
                "reserved for image placeholders",
            ),
        ];
        for (input, needle) in cases {
            let (status, v) = post_json(
                make_image_app().await,
                json!({"model": "mock-model", "input": input}),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
            assert_eq!(v["error"]["type"], "invalid_request_error");
            let msg = v["error"]["message"].as_str().unwrap_or_default();
            assert!(msg.contains(needle), "expected {needle:?} in {msg:?}");
        }
    }

    /// `instructions` and every message's text are checked for image tags on
    /// a server that takes images, images attached or not.
    #[tokio::test]
    async fn reserved_tags_are_refused_in_text_only_input() {
        for body in [
            json!({"instructions": "say <ov_genai_image_0>", "input": "hi"}),
            json!({"input": [{"role": "developer", "content": [
                {"type": "input_text", "text": "<ov_genai_image_1>"}]},
                {"role": "user", "content": "hi"}]}),
        ] {
            let mut body = body;
            body["model"] = "mock-model".into();
            let (status, v) = post_json(make_image_app().await, body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
            let msg = v["error"]["message"].as_str().unwrap();
            assert!(msg.contains("reserved for image placeholders"), "{msg}");
        }
    }

    /// Every feature this endpoint cannot honor is an OpenAI 400 whose
    /// `param` names the field, before anything reaches the engine.
    #[tokio::test]
    async fn unsupported_fields_are_refused_by_name() {
        let fn_tool = json!([{"type": "function", "name": "f", "parameters": {}}]);
        let cases = [
            (json!({"store": true}), "store"),
            (
                json!({"previous_response_id": "resp_1"}),
                "previous_response_id",
            ),
            (json!({"conversation": "conv_1"}), "conversation"),
            (json!({"background": true}), "background"),
            (json!({"tools": fn_tool}), "tools"),
            (json!({"tool_choice": "required"}), "tool_choice"),
            (
                json!({"include": ["message.output_text.logprobs"]}),
                "include",
            ),
            (json!({"prompt": {"id": "pmpt_1"}}), "prompt"),
            (json!({"top_logprobs": 3}), "top_logprobs"),
            (json!({"stop": ["\n"]}), "stop"),
            (
                json!({"text": {"format": {"type": "json_schema", "name": "x", "schema": {}}}}),
                "text.format",
            ),
            (
                json!({"reasoning": {"summary": "auto"}}),
                "reasoning.summary",
            ),
            (json!({"truncation": "auto"}), "truncation"),
            (json!({"input": 5}), "input"),
            (
                json!({"input": [{"type": "function_call_output", "call_id": "c", "output": "1"}]}),
                "input[0]",
            ),
            (
                json!({"input": [{"type": "item_reference", "id": "msg_1"}]}),
                "input[0]",
            ),
            (
                json!({"input": [{"role": "tool", "content": "x"}]}),
                "input[0].role",
            ),
            (
                json!({"input": [{"role": "user", "content": [
                    {"type": "input_file", "file_id": "file-1"}]}]}),
                "input[0].content[0]",
            ),
            (
                json!({"input": [{"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}]}]}),
                "input[0].content[0]",
            ),
            (
                json!({"input": [{"role": "user", "content": [
                    {"type": "input_image", "file_id": "file-1"}]}]}),
                "input[0].content[0].file_id",
            ),
        ];
        for (extra, param) in cases {
            let mut body = json!({"model": "mock-model", "input": "hello"});
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let (status, v) = post_json(make_image_app().await, body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{param}: {v}");
            assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
            assert_eq!(v["error"]["param"], param, "{v}");
            assert!(!v["error"]["message"].as_str().unwrap().is_empty());
        }
    }

    /// The no-op forms of the refused fields are accepted.
    #[tokio::test]
    async fn neutral_values_of_refused_fields_are_accepted() {
        let (status, v) = post_json(
            make_app().await,
            json!({"model": "mock-model", "input": "hello there", "store": false,
                "background": false, "tools": [], "tool_choice": "auto", "include": [],
                "previous_response_id": null, "truncation": "disabled",
                "text": {"format": {"type": "text"}}, "metadata": {"k": "v"}}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["metadata"], json!({"k": "v"}));
    }

    #[tokio::test]
    async fn empty_input_is_a_400_naming_input() {
        let (status, v) = post_json(
            make_app().await,
            json!({"model": "mock-model", "input": []}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
        assert!(v["error"]["message"].as_str().unwrap().contains("`input`"));
    }

    /// Echoes `tok ` for its one task. With `cancel_after: Some(n)` it ends
    /// the task after `n` tokens with a `Cancelled` final marker; with `None`
    /// it never ends on its own, so only `/v1/cancel` stops it (and the
    /// runner then ends the stream with no final chunk at all). With
    /// `fail_after: Some(n)` it fails the task after `n` tokens instead (a
    /// final chunk with `error` set).
    struct ScriptEngine {
        task: Option<cascadia_types::TaskId>,
        cancel_after: Option<usize>,
        fail_after: Option<usize>,
        emitted: Arc<std::sync::atomic::AtomicUsize>,
        seen: Arc<parking_lot::Mutex<Option<cascadia_types::TaskId>>>,
    }

    impl cascadia_engine::Engine for ScriptEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, task: GenerationTask) -> Result<(), cascadia_engine::EngineError> {
            *self.seen.lock() = Some(task.task_id.clone());
            self.task = Some(task.task_id);
            Ok(())
        }
        fn step(
            &mut self,
        ) -> Result<
            Vec<(cascadia_types::TaskId, cascadia_types::Chunk)>,
            cascadia_engine::EngineError,
        > {
            use std::sync::atomic::Ordering::SeqCst;
            let Some(id) = self.task.clone() else {
                return Ok(Vec::new());
            };
            let n = self.emitted.load(SeqCst);
            if self.fail_after.is_some_and(|max| n >= max) {
                self.task = None;
                let end = cascadia_types::Chunk::error(&id, "peer link dropped");
                return Ok(vec![(id, end)]);
            }
            if self.cancel_after.is_some_and(|max| n >= max) {
                self.task = None;
                let end = cascadia_types::Chunk::final_marker(&id, "")
                    .with_finish_reason(cascadia_types::FinishReason::Cancelled);
                return Ok(vec![(id, end)]);
            }
            self.emitted.fetch_add(1, SeqCst);
            Ok(vec![(
                id.clone(),
                cascadia_types::Chunk::token(&id, n as i64, "tok "),
            )])
        }
    }

    struct ScriptBuilder(Option<ScriptEngine>);

    #[::async_trait::async_trait]
    impl cascadia_engine::Builder for ScriptBuilder {
        async fn connect(
            &mut self,
            _peers: cascadia_types::PeerLayout,
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
            mut self: Box<Self>,
        ) -> Result<Box<dyn cascadia_engine::Engine>, cascadia_engine::EngineError> {
            Ok(Box::new(self.0.take().expect("built once")))
        }
    }

    /// Run one request against a [`ScriptEngine`]. Without `cancel_after`,
    /// the request is cancelled through `/v1/cancel/<id>` once a token is out.
    async fn scripted(cancel_after: Option<usize>, stream: bool) -> (StatusCode, String) {
        scripted_with(cancel_after, None, stream).await
    }

    async fn scripted_with(
        cancel_after: Option<usize>,
        fail_after: Option<usize>,
        stream: bool,
    ) -> (StatusCode, String) {
        use std::sync::atomic::Ordering::SeqCst;
        let emitted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = Arc::new(parking_lot::Mutex::new(None));
        let runner = Arc::new(Runner::new(Box::new(ScriptBuilder(Some(ScriptEngine {
            task: None,
            cancel_after,
            fail_after,
            emitted: emitted.clone(),
            seen: seen.clone(),
        })))));
        runner
            .start(
                cascadia_types::PeerLayout::single_stage(),
                cascadia_types::ShardSpec::single_stage("script-model", "CPU"),
            )
            .await
            .unwrap();
        let app = make_router(runner.clone(), "script-model");
        let req = tokio::spawn(post(
            app.clone(),
            json!({"model": "script-model", "input": "hello", "stream": stream}),
        ));
        if cancel_after.is_none() && fail_after.is_none() {
            while emitted.load(SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
            let id = seen.lock().clone().unwrap();
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(format!("/v1/cancel/{id}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        req.await.unwrap()
    }

    fn assert_cancelled(response: &Value) {
        assert_eq!(response["status"], "cancelled", "{response}");
        assert!(response["completed_at"].is_null());
        let item = &response["output"][0];
        assert_eq!(item["status"], "incomplete");
        assert!(item["content"][0]["text"]
            .as_str()
            .unwrap()
            .starts_with("tok"));
    }

    /// A turn the engine ends with `Cancelled`, and one cut off by
    /// `/v1/cancel` (whose stream ends with no final chunk), carry partial
    /// text: neither is ever reported `completed`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cancelled_turns_are_not_completed() {
        for cancel_after in [Some(2), None] {
            let (status, body) = scripted(cancel_after, false).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_cancelled(&serde_json::from_str(&body).unwrap());

            let (status, body) = scripted(cancel_after, true).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            let ev = events(&body);
            assert!(ev.iter().all(|(k, _)| k != "response.completed"), "{body}");
            let (kind, data) = ev.last().unwrap();
            assert_eq!(kind, "response.incomplete");
            assert_cancelled(&data["response"]);
        }
    }

    /// A task that fails after the 200 is on the wire ends the stream with
    /// one `response.failed` event: status `failed`, the engine's reason in
    /// `error`, the tokens so far in `usage`, and nothing after it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn mid_stream_failure_ends_with_response_failed() {
        let (status, body) = scripted_with(None, Some(2), true).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let ev = events(&body);
        let kinds: Vec<_> = ev.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            kinds,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.failed",
            ],
            "{body}"
        );
        let (_, data) = ev.last().unwrap();
        assert_eq!(data["type"], "response.failed");
        assert_eq!(data["sequence_number"], ev.len() as u64 - 1);
        let response = &data["response"];
        assert_eq!(response["status"], "failed", "{response}");
        assert_eq!(response["error"]["code"], "server_error");
        assert_eq!(response["error"]["message"], "peer link dropped");
        assert!(response["completed_at"].is_null());
        assert_eq!(response["usage"]["output_tokens"], 2);
    }

    /// The cap in force is echoed even when the caller left it to the
    /// default, so an `incomplete` response names a limit the caller can see.
    #[tokio::test]
    async fn the_effective_token_cap_is_echoed() {
        let (_, v) = post_json(
            make_app().await,
            json!({"model": "mock-model", "input": "hello"}),
        )
        .await;
        assert_eq!(v["max_output_tokens"], default_max_tokens());
    }

    /// Refusals book their own rejection reason, apart from bodies that do
    /// not parse.
    #[tokio::test]
    async fn refusals_count_as_unsupported_field() {
        let count = || {
            cascadia_metrics::API_REJECTED_TOTAL
                .with_label_values(&["unsupported_field"])
                .get()
        };
        let before = count();
        let (status, _) = post_json(
            make_app().await,
            json!({"model": "mock-model", "input": "hi", "store": true}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(count() > before);
    }
}
