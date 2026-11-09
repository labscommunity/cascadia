//! `/v1/responses`: OpenAI's Responses API over the chat path.
//!
//! A Responses request is mapped onto a [`ChatCompletionRequest`] and admitted
//! by the same code as `/v1/chat/completions` (`admit_chat`), so the chat
//! template, sampling, image decoding, limits and every refusal there apply
//! here unchanged. Only the request and response shapes differ.
//!
//! `tools` are mapped onto chat tools (see [`tools`]); tools this server
//! cannot serve (OpenAI-hosted ones, and the client-side `shell`,
//! `apply_patch` and `tool_search` types) are dropped with a warning.
//! Replayed reasoning and tool-call items become one assistant turn (see
//! [`input`]); the output is split into reasoning, message and call items
//! (see [`output`]). `store` (absent = `true`) keeps the output items in an
//! in-memory cache so a later `item_reference` resolves (see [`store`]); no
//! response is stored. The fields that need state Cascadia does not keep
//! (`previous_response_id`, `conversation`, `background`, file ids) are
//! refused with a 400 naming the field, never ignored: a dropped
//! `previous_response_id` would have the model answer without the
//! conversation the caller thinks it sent. Fields Codex and the SDKs send
//! that change nothing here (`client_metadata`, `prompt_cache_key`,
//! `service_tier`, `text.verbosity`, ...) are accepted and ignored.

use super::*;
use serde_json::{json, Value};

mod input;
mod output;
mod store;
mod tools;
use input::input_messages;
pub(crate) use input::{fold_system_turns, Fold};
use output::{OutputBuilder, OutputEvent};
pub(crate) use store::ItemCache;
use tools::{check_include, check_tool_choice, ToolMap};

/// OpenAI `POST /v1/responses` body. The fields after `max_tool_calls` are
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
    /// Nullable: `null` is the default, `false`.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Echoed; absent is OpenAI's default, `true`.
    #[serde(default)]
    pub store: Option<bool>,
    /// `reasoning.effort` is the chat endpoint's `reasoning_effort`;
    /// `reasoning.summary` is accepted, but no summary is generated.
    #[serde(default)]
    pub reasoning: Option<ReasoningParam>,
    /// Only `{"format": {"type": "text"}}`; structured output is refused.
    #[serde(default)]
    pub text: Option<Value>,
    /// Only `"disabled"`: a prompt over the window is a 400
    /// `context_length_exceeded`, never trimmed.
    #[serde(default)]
    pub truncation: Option<String>,
    /// Only `"none"` or `"auto"` (see [`check_tool_choice`]).
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// Echoed back on the response object; held to OpenAI's shape (see
    /// [`check_metadata`]).
    #[serde(default)]
    pub metadata: Option<Value>,
    /// See [`ToolMap::from_request`].
    #[serde(default)]
    pub tools: Option<Value>,
    /// Checked against OpenAI's values (see [`check_include`]).
    #[serde(default)]
    pub include: Option<Value>,
    /// Echoed (absent = `true`).
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Echoed; it caps hosted tool calls, and none runs here. A
    /// non-negative integer or null (see [`refuse_unsupported`]).
    #[serde(default)]
    pub max_tool_calls: Option<Value>,
    #[serde(default)]
    pub previous_response_id: Option<Value>,
    #[serde(default)]
    pub conversation: Option<Value>,
    #[serde(default)]
    pub background: Option<Value>,
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

/// An error (a 400 unless `status` says otherwise) naming the request field
/// at fault.
#[derive(Debug, PartialEq)]
pub(super) struct Refusal {
    status: StatusCode,
    param: String,
    message: String,
    code: Option<&'static str>,
    /// The `cascadia_api_rejected_total` reason: a field this endpoint does
    /// not serve, unless the value itself is malformed.
    reason: &'static str,
}

impl Refusal {
    fn new(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            param: param.into(),
            message: message.into(),
            code: None,
            reason: "unsupported_field",
        }
    }

    /// A served field whose value is not what the API takes.
    fn malformed(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            reason: "malformed_body",
            ..Self::new(param, message)
        }
    }

    /// A 413: expanding `input`'s `item_reference`s would pass the
    /// `max_bytes` prompt budget.
    pub(super) fn too_large(param: impl Into<String>, max_bytes: usize) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            code: Some("prompt_too_large"),
            reason: "prompt_too_large",
            ..Self::new(
                param,
                format!(
                    "expanding `item_reference` items would pass {max_bytes} bytes \
                     (max_prompt_bytes)"
                ),
            )
        }
    }

    /// The machine-readable `code` other servers give this refusal.
    fn with_code(self, code: &'static str) -> Self {
        Self {
            code: Some(code),
            ..self
        }
    }

    fn into_response(self) -> Response {
        count_rejected(self.reason);
        api_error_for(self.status, self.message, Some(&self.param), self.code)
    }
}

const STATELESS: &str = "this server keeps no responses; send the whole conversation in `input`";

/// OpenAI's limits on `metadata`: up to 16 pairs, keys up to 64
/// characters, string values up to 512.
const MAX_METADATA_PAIRS: usize = 16;
const MAX_METADATA_KEY: usize = 64;
const MAX_METADATA_VALUE: usize = 512;

/// Refuse `metadata` OpenAI would refuse: anything but an object of at most
/// [`MAX_METADATA_PAIRS`] string pairs within the length limits. It is
/// echoed on every response object (and every streamed event's), so an
/// unbounded one would be repeated for the whole stream.
fn check_metadata(metadata: Option<&Value>) -> Result<(), Refusal> {
    let refuse = |m: String| Err(Refusal::malformed("metadata", m));
    let map = match metadata {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::Object(map)) => map,
        Some(_) => return refuse("`metadata` must be an object of string keys and values".into()),
    };
    if map.len() > MAX_METADATA_PAIRS {
        return refuse(format!(
            "`metadata` has {} pairs; at most {MAX_METADATA_PAIRS} are allowed",
            map.len()
        ));
    }
    for (k, v) in map {
        if k.chars().count() > MAX_METADATA_KEY {
            return refuse(format!(
                "`metadata` keys must be at most {MAX_METADATA_KEY} characters"
            ));
        }
        match v.as_str() {
            Some(s) if s.chars().count() <= MAX_METADATA_VALUE => {}
            Some(_) => {
                return refuse(format!(
                    "`metadata` values must be at most {MAX_METADATA_VALUE} characters \
                     (`{k}` is longer)"
                ))
            }
            None => return refuse(format!("`metadata` values must be strings (`{k}` is not)")),
        }
    }
    Ok(())
}

/// Refuse every request field this endpoint cannot honor, by name. `true`
/// when `include` asks for reasoning items' `encrypted_content`.
fn refuse_unsupported(req: &ResponsesRequest) -> Result<bool, Refusal> {
    check_metadata(req.metadata.as_ref())?;
    let encrypted = check_include(req.include.as_ref())?;
    // Echoed on the response, so held to the schema's integer.
    if req
        .max_tool_calls
        .as_ref()
        .is_some_and(|n| n.as_u64().is_none())
    {
        return Err(Refusal::malformed(
            "max_tool_calls",
            "`max_tool_calls` must be a non-negative integer or null",
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
    if req.truncation.as_deref().is_some_and(|t| t != "disabled") {
        return Err(Refusal::new(
            "truncation",
            "only `truncation: \"disabled\"` is supported: a prompt over the model's \
             window is refused, not trimmed",
        ));
    }
    Ok(encrypted)
}

/// The chat request this Responses request is served as, with `map`'s chat
/// tools and the checked `tool_choice`.
fn chat_request(
    req: &ResponsesRequest,
    messages: Vec<ChatMessage>,
    map: &ToolMap,
    choice: Option<&str>,
) -> ChatCompletionRequest {
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
        stream: req.stream.unwrap_or(false),
        stream_options: None,
        enable_thinking: None,
        reasoning_effort: req.reasoning.as_ref().and_then(|r| r.effort.clone()),
        chat_template_kwargs: None,
        tools: (!map.chat_tools.is_empty()).then(|| map.chat_tools.clone()),
        tool_choice: choice.map(|c| json!(c)),
    }
}

/// The request-derived fields of the response object, shared by the
/// non-streaming body and every streamed snapshot of it.
struct ResponseShell {
    id: String,
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
    store: bool,
    tool_choice: &'static str,
    parallel_tool_calls: bool,
    max_tool_calls: Value,
    /// The served tools, Responses-shaped ([`ToolMap::echo`]).
    tools: Vec<Value>,
}

impl ResponseShell {
    fn new(
        req: &ResponsesRequest,
        id: String,
        map: &ToolMap,
        choice: Option<&'static str>,
    ) -> Self {
        Self {
            id,
            created_at: now_unix(),
            model: req.model.clone(),
            instructions: req.instructions.clone(),
            max_output_tokens: req.max_output_tokens.unwrap_or_else(default_max_tokens),
            temperature: req.temperature.unwrap_or_default(),
            top_p: req.top_p.unwrap_or(1.0),
            effort: req.reasoning.as_ref().and_then(|r| r.effort.clone()),
            metadata: req.metadata.clone().unwrap_or_else(|| json!({})),
            store: req.store.unwrap_or(true),
            tool_choice: choice.unwrap_or("auto"),
            parallel_tool_calls: req.parallel_tool_calls.unwrap_or(true),
            max_tool_calls: req.max_tool_calls.clone().unwrap_or(Value::Null),
            tools: map.echo.clone(),
        }
    }

    /// The OpenAI `response` object. `outcome` is `None` while in progress.
    fn object(&self, status: &str, output: Value, outcome: Option<&Outcome>) -> Value {
        json!({
            "id": self.id,
            "object": "response",
            "access_programs": null,
            "background": false,
            "created_at": self.created_at,
            "status": status,
            "completed_at": outcome.filter(|_| status == "completed").map(|_| now_unix()),
            "error": outcome.and_then(|o| o.error.as_ref()).map(|e| json!({
                "code": "server_error", "message": e})),
            "incomplete_details": (status == "incomplete")
                .then(|| json!({"reason": "max_output_tokens"})),
            "instructions": self.instructions,
            "max_output_tokens": self.max_output_tokens,
            "max_tool_calls": self.max_tool_calls,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": self.parallel_tool_calls,
            "previous_response_id": null,
            "reasoning": {"effort": self.effort, "summary": null},
            "store": self.store,
            "temperature": self.temperature,
            "text": {"format": {"type": "text"}},
            "tool_choice": self.tool_choice,
            "tools": self.tools,
            "top_p": self.top_p,
            "truncation": "disabled",
            "usage": outcome.map(|o| json!({
                "input_tokens": o.prompt_tokens,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": o.completion_tokens,
                "output_tokens_details": {"reasoning_tokens": o.reasoning_tokens},
                "total_tokens": o.prompt_tokens + o.completion_tokens,
            })),
            "user": null,
            "metadata": self.metadata,
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
    /// Of `completion_tokens`, those spent reasoning.
    reasoning_tokens: u32,
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

/// Looks `item_reference`s up in the item cache; an item past what is left
/// of the prompt budget is a 413 (the walk names the reference).
fn resolver(
    state: &AppState,
) -> impl Fn(&str, usize) -> Result<Option<(Value, usize)>, Refusal> + '_ {
    move |id, budget| {
        state
            .item_cache
            .get_within(id, budget)
            .map_err(|_| Refusal::too_large("input", state.max_prompt_bytes))
    }
}

/// `POST /v1/responses`.
pub(crate) async fn create_response(
    State(state): State<AppState>,
    OpenAiJson(req): OpenAiJson<ResponsesRequest>,
) -> Response {
    let checked = refuse_unsupported(&req).and_then(|encrypted| {
        let map = ToolMap::from_request(req.tools.as_ref())?;
        let choice = check_tool_choice(req.tool_choice.as_ref())?;
        Ok((
            encrypted,
            map,
            choice,
            input_messages(&req, &resolver(&state), state.max_prompt_bytes)?,
        ))
    });
    let (wants_encrypted, map, choice, messages) = match checked {
        Ok(c) => c,
        Err(refusal) => return refusal.into_response(),
    };
    if !map.dropped.is_empty() {
        warn!(types = ?map.dropped, "responses: hosted tools are not served; dropped");
    }
    let chat = chat_request(&req, messages, &map, choice);
    // The response id doubles as the task id, so `/v1/cancel/<id>` works.
    let id = format!("resp_{}", Uuid::new_v4().simple());
    let shell = ResponseShell::new(&req, id, &map, choice);
    let admitted = match admit_chat(
        &state,
        &chat,
        shell.id.clone(),
        "`input` is empty or rendered to an empty prompt",
        AdmitOpts {
            api_render_on_thinking: true,
            fold_refused_system_turns: true,
            reasoning_as_text: true,
        },
    )
    .await
    {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    let task = &admitted.task;
    let opened = opens_think(
        state.engine_applies_template,
        state.defer_template_on_thinking,
        task.enable_thinking,
        state.template_opens_think,
        task,
    );
    let mut builder = OutputBuilder::new(
        task.enable_thinking,
        opened,
        map.tools_active(choice),
        Arc::new(map),
        req.parallel_tool_calls.unwrap_or(true),
        wants_encrypted,
    );
    // Where the final items are kept for `item_reference`; `store: false`
    // keeps nothing.
    let cache = (req.store != Some(false)).then(|| state.item_cache.clone());
    if req.stream.unwrap_or(false) {
        return stream_response(state, shell, admitted, builder, cache).await;
    }
    let collected = collect_generation_with(&state, admitted, &[], "input", &mut |text, n| {
        builder.feed(text, n);
    })
    .await;
    let c = match collected {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let end = End::from_final(c.end);
    builder.finish(end.item_status());
    let outcome = Outcome {
        prompt_tokens: c.prompt_tokens,
        completion_tokens: c.completion_tokens,
        reasoning_tokens: builder.reasoning_tokens(c.completion_tokens),
        error: None,
    };
    let items = builder.items();
    if let Some(cache) = &cache {
        cache.insert_all(&items);
    }
    Json(shell.object(end.status(), json!(items), Some(&outcome))).into_response()
}

/// A streamed response's state between SSE frames.
struct ResponseStream {
    shell: ResponseShell,
    chunks: StreamWithPermit,
    marker: Option<MarkerFilter>,
    stats: Arc<ApiStats>,
    /// The output items, built as the chunks arrive (as the non-streaming
    /// path builds them), and the events announcing them.
    builder: OutputBuilder,
    /// Where the final items are kept (`None` for `store: false`).
    cache: Option<Arc<ItemCache>>,
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

    /// The address of a text event: the item at `index`, its one part.
    fn at(index: usize, item_id: &Value, extra: Value) -> Value {
        let mut v = json!({"item_id": item_id, "output_index": index, "content_index": 0});
        if let (Some(v), Value::Object(extra)) = (v.as_object_mut(), extra) {
            v.extend(extra);
        }
        v
    }

    /// Render the builder's events as Responses stream events (spec D11).
    fn render(&mut self, out: &mut String, events: Vec<OutputEvent>) {
        for e in events {
            match e {
                OutputEvent::ItemAdded { index, item } => {
                    let part = match item["type"].as_str() {
                        Some("reasoning") => Some(json!({"type": "reasoning_text", "text": ""})),
                        Some("message") => Some(output_text("")),
                        _ => None,
                    };
                    let id = item["id"].clone();
                    self.emit(
                        out,
                        "response.output_item.added",
                        json!({"output_index": index, "item": item}),
                    );
                    if let Some(part) = part {
                        let added = Self::at(index, &id, json!({"part": part}));
                        self.emit(out, "response.content_part.added", added);
                    }
                }
                OutputEvent::ReasoningDelta {
                    index,
                    item_id,
                    delta,
                } => {
                    let d = Self::at(index, &json!(item_id), json!({"delta": delta}));
                    self.emit(out, "response.reasoning_text.delta", d);
                }
                OutputEvent::TextDelta {
                    index,
                    item_id,
                    delta,
                } => {
                    let d = Self::at(
                        index,
                        &json!(item_id),
                        json!({"delta": delta, "logprobs": []}),
                    );
                    self.emit(out, "response.output_text.delta", d);
                }
                OutputEvent::CallArgs {
                    index,
                    item_id,
                    name,
                    custom,
                    full,
                } => {
                    let (prefix, field) = if custom {
                        ("response.custom_tool_call_input", "input")
                    } else {
                        ("response.function_call_arguments", "arguments")
                    };
                    let at = json!({"item_id": item_id, "output_index": index});
                    let mut delta = at.clone();
                    delta["delta"] = json!(full);
                    self.emit(out, &format!("{prefix}.delta"), delta);
                    let mut done = at;
                    done[field] = json!(full);
                    // OpenAI's arguments `done` names the function too.
                    if !custom {
                        done["name"] = json!(name);
                    }
                    self.emit(out, &format!("{prefix}.done"), done);
                }
                OutputEvent::ItemDone { index, item } => {
                    let text_done = match item["type"].as_str() {
                        Some("reasoning") => Some(("response.reasoning_text.done", json!({}))),
                        Some("message") => {
                            Some(("response.output_text.done", json!({"logprobs": []})))
                        }
                        _ => None,
                    };
                    if let Some((kind, mut extra)) = text_done {
                        let part = item["content"][0].clone();
                        extra["text"] = part["text"].clone();
                        let done = Self::at(index, &item["id"], extra);
                        self.emit(out, kind, done);
                        let done = Self::at(index, &item["id"], json!({"part": part}));
                        self.emit(out, "response.content_part.done", done);
                    }
                    self.emit(
                        out,
                        "response.output_item.done",
                        json!({"output_index": index, "item": item}),
                    );
                }
            }
        }
    }

    fn outcome(&self, error: Option<String>) -> Outcome {
        Outcome {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            reasoning_tokens: self.builder.reasoning_tokens(self.completion_tokens),
            error,
        }
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
            return out;
        }
        let Some(chunk) = self.chunks.next().await else {
            // No final chunk: the runner ends a cancelled task's stream so.
            self.finish(&mut out, End::Cancelled);
            return out;
        };
        // The 200 is on the wire already, so a failed task is reported the
        // Responses way: a `response.failed` event, as the chat endpoint's
        // SSE error event does for its stream. No closing events precede
        // it; its `output` has every item announced so far, the open one
        // `incomplete`.
        if let Some(reason) = &chunk.error {
            warn!(task = %self.shell.id, reason = %reason, "engine failed task mid-stream; emitting response.failed");
            let outcome = self.outcome(Some(reason.clone()));
            let output = json!(self.builder.items_closing_open("incomplete"));
            let failed = self.shell.object("failed", output, Some(&outcome));
            self.emit(&mut out, "response.failed", json!({"response": failed}));
            self.done = true;
            return out;
        }
        // Count every chunk, but take the final chunk's authoritative total
        // when it has one (as `collect_generation_with` and the chat stream
        // do); the global counter moves by the difference, down included.
        let n = chunk_token_count(&chunk);
        let before = self.completion_tokens;
        chunk.accumulate_tokens(&mut self.completion_tokens);
        move_tokens_total(&self.stats, before, self.completion_tokens);
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
        let events = self.builder.feed(&text, n);
        self.render(&mut out, events);
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

    /// The closing events: the open item's (and any parsed calls'), then
    /// the response. The items are stored before the terminal event is
    /// written, so a client that has read it finds them.
    fn finish(&mut self, out: &mut String, end: End) {
        let events = self.builder.finish(end.item_status());
        self.render(out, events);
        let outcome = self.outcome(None);
        let items = self.builder.items();
        if let Some(cache) = &self.cache {
            cache.insert_all(&items);
        }
        let output = json!(items);
        let response = self.shell.object(end.status(), output, Some(&outcome));
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
    builder: OutputBuilder,
    cache: Option<Arc<ItemCache>>,
) -> Response {
    let chunks = match state.runner.generate_async(task).await {
        Ok(s) => s,
        Err(err) => {
            warn!(error = %err, "responses-stream: generate failed");
            return engine_error_response(err, "input");
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
        builder,
        cache,
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
    use crate::spec_schema::{assert_event, assert_spec};
    use crate::tests::{make_app, make_image_app, png_url};
    use axum::body::to_bytes;
    use axum::http::Request;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use tower::ServiceExt;

    async fn post(app: Router, payload: Value) -> (StatusCode, String) {
        post_to(app, "/v1/responses", payload).await
    }

    async fn post_to(app: Router, uri: &str, payload: Value) -> (StatusCode, String) {
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

    async fn post_json(app: Router, payload: Value) -> (StatusCode, Value) {
        let (status, body) = post(app, payload).await;
        (status, serde_json::from_str(&body).unwrap_or(Value::Null))
    }

    fn request(v: Value) -> ResponsesRequest {
        serde_json::from_value(v).unwrap()
    }

    /// `input_messages` over `v` with no stored items and a 1 MiB budget.
    fn walk_request(v: Value) -> Result<Vec<ChatMessage>, Refusal> {
        input_messages(&request(v), &|_, _| Ok(None), 1 << 20)
    }

    /// `input` as a string is one user message; as items, each part maps to
    /// the chat part it names, `instructions` leads as `system`, and
    /// `developer` reads as `system` (merged into the `instructions`, D22).
    #[test]
    fn input_maps_onto_chat_messages() {
        let m = walk_request(json!({"model": "m", "input": "hi there"})).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(
            (m[0].role.as_str(), m[0].content.as_str()),
            ("user", "hi there")
        );

        let m = walk_request(json!({"model": "m",
        "instructions": "be brief",
        "input": [
            {"role": "developer", "content": "terse"},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "what"},
                {"type": "input_image", "image_url": "data:x", "detail": "low"},
                {"type": "input_text", "text": "is this"}]},
            {"role": "assistant", "id": "msg_1", "status": "completed", "content": [
                {"type": "output_text", "text": "a dot", "annotations": []}]},
        ]}))
        .unwrap();
        let roles: Vec<_> = m.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "assistant"]);
        assert_eq!(m[0].content, "be brief\n\nterse");
        assert_eq!(m[1].content, "whatis this");
        assert_eq!(
            m[1].parts,
            vec![
                ContentPart::Text("what".into()),
                ContentPart::ImageUrl {
                    url: "data:x".into(),
                    detail: Some("low".into())
                },
                ContentPart::Text("is this".into()),
            ]
        );
        assert_eq!(m[2].content, "a dot");
        assert_eq!(m[2].parts, vec![ContentPart::Text("a dot".into())]);

        let chat = chat_request(
            &request(json!({"model": "m", "input": "x", "max_output_tokens": 7,
                "temperature": 0.5, "top_p": 0.9, "reasoning": {"effort": "low"}})),
            Vec::new(),
            &ToolMap::default(),
            None,
        );
        assert_eq!(chat.max_tokens, 7);
        assert_eq!(chat.temperature, 0.5);
        assert_eq!(chat.top_p, Some(0.9));
        assert_eq!(chat.reasoning_effort.as_deref(), Some("low"));
        let chat = chat_request(
            &request(json!({"model": "m", "input": "x"})),
            Vec::new(),
            &ToolMap::default(),
            None,
        );
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
        assert_spec("Response", &v);
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
        assert_spec("Response", &v);
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
            assert_event(data);
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
        ev.iter().for_each(|(_, data)| assert_event(data));
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
        let cases = [
            (
                json!({"previous_response_id": "resp_1"}),
                "previous_response_id",
            ),
            (json!({"conversation": "conv_1"}), "conversation"),
            (json!({"background": true}), "background"),
            (json!({"tool_choice": "required"}), "tool_choice"),
            (json!({"include": ["bogus"]}), "include"),
            (json!({"prompt": {"id": "pmpt_1"}}), "prompt"),
            (json!({"top_logprobs": 3}), "top_logprobs"),
            (json!({"stop": ["\n"]}), "stop"),
            (
                json!({"text": {"format": {"type": "json_schema", "name": "x", "schema": {}}}}),
                "text.format",
            ),
            (json!({"truncation": "auto"}), "truncation"),
            (json!({"max_tool_calls": "many"}), "max_tool_calls"),
            (json!({"max_tool_calls": {}}), "max_tool_calls"),
            (json!({"max_tool_calls": -1}), "max_tool_calls"),
            (json!({"max_tool_calls": 1.5}), "max_tool_calls"),
            (json!({"input": 5}), "input"),
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
                "metadata": {"k": "v"}, "reasoning": {"summary": "auto"},
                "client_metadata": {"x-codex-installation-id": "id"},
                "prompt_cache_key": "k", "prompt_cache_options": {}, "prompt_cache_retention": "24h",
                "safety_identifier": "s", "user": "u", "service_tier": "auto",
                "text": {"format": {"type": "text"}, "verbosity": "low"},
                "stream_options": {"include_obfuscation": false}, "access_programs": null,
                "context_management": [{"type": "compaction", "compact_threshold": 1000}],
                "moderation": null, "parallel_tool_calls": false}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_spec("Response", &v);
        assert_eq!(v["metadata"], json!({"k": "v"}));
        let (status, v) = post_json(
            make_app().await,
            json!({"model": "mock-model", "input": "hello there", "store": true}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["store"], true);
    }

    /// `metadata` is held to OpenAI's shape: an object of at most 16
    /// string pairs, keys up to 64 characters and values up to 512.
    #[tokio::test]
    async fn metadata_is_held_to_openais_limits() {
        let pairs = |n: usize| -> Value {
            (0..n)
                .map(|i| (format!("k{i}"), json!("v")))
                .collect::<serde_json::Map<_, _>>()
                .into()
        };
        let long_key = json!({ "k".repeat(65): "v" });
        let long_value = json!({"k": "é".repeat(513)});
        for bad in [
            json!("text"),
            json!([1]),
            json!({"k": 1}),
            json!({"k": {"nested": "v"}}),
            pairs(17),
            long_key,
            long_value,
        ] {
            let (status, v) = post_json(
                make_app().await,
                json!({"model": "mock-model", "input": "hi", "metadata": bad}),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {v}");
            assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
            assert_eq!(v["error"]["param"], "metadata", "{v}");
        }
        let at_limits = {
            let mut m = pairs(15);
            m["k".repeat(64)] = json!("é".repeat(512));
            m
        };
        for (good, echoed) in [
            (at_limits.clone(), at_limits),
            (json!({}), json!({})),
            (Value::Null, json!({})),
        ] {
            let (status, v) = post_json(
                make_app().await,
                json!({"model": "mock-model", "input": "hi", "metadata": good}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{good}: {v}");
            assert_eq!(v["metadata"], echoed, "{v}");
        }
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

    /// What a [`RecordingEngine`] writes.
    enum Text {
        /// These chunk texts, one token each, then a final chunk (unless
        /// `never_end`).
        Script(Vec<String>),
        /// `tok ` forever.
        Tok,
    }

    /// A test engine recording every submitted [`GenerationTask`]; its
    /// emitted count resets on every submit. With `cancel_after: Some(n)` it
    /// ends the task after `n` tokens with a `Cancelled` final marker; with
    /// `fail_after: Some(n)` it fails the task after `n` tokens instead (a
    /// final chunk with `error` set). A [`Text::Tok`] engine with neither
    /// never ends on its own, so only `/v1/cancel` stops it (and the runner
    /// then ends the stream with no final chunk at all).
    struct RecordingEngine {
        text: Text,
        /// No final chunk once the script is out.
        never_end: bool,
        cancel_after: Option<usize>,
        fail_after: Option<usize>,
        tasks: Arc<parking_lot::Mutex<Vec<GenerationTask>>>,
        /// Tokens out so far; tests wait on it before `/v1/cancel`.
        emitted: Arc<AtomicUsize>,
        task: Option<cascadia_types::TaskId>,
        /// The submitted task's `max_tokens`: a [`Text::Script`] stops there
        /// with `Length`, as an engine at the cap does.
        max_tokens: usize,
        /// `n_tokens` on every token chunk (`None`: one each).
        chunk_tokens: Option<u32>,
        /// `completion_tokens` on a [`Text::Script`]'s `Stop` final chunk,
        /// the engine's authoritative total (as llama.cpp reports it).
        final_tokens: Option<u32>,
    }

    impl RecordingEngine {
        fn new(text: Text) -> Self {
            Self {
                text,
                never_end: false,
                cancel_after: None,
                fail_after: None,
                tasks: Arc::default(),
                emitted: Arc::default(),
                task: None,
                max_tokens: usize::MAX,
                chunk_tokens: None,
                final_tokens: None,
            }
        }

        fn script(chunks: &[&str]) -> Self {
            Self::new(Text::Script(chunks.iter().map(|c| c.to_string()).collect()))
        }

        fn tok() -> Self {
            Self::new(Text::Tok)
        }
    }

    impl cascadia_engine::Engine for RecordingEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, task: GenerationTask) -> Result<(), cascadia_engine::EngineError> {
            self.emitted.store(0, SeqCst);
            self.task = Some(task.task_id.clone());
            self.max_tokens = task.max_tokens as usize;
            self.tasks.lock().push(task);
            Ok(())
        }
        fn step(
            &mut self,
        ) -> Result<
            Vec<(cascadia_types::TaskId, cascadia_types::Chunk)>,
            cascadia_engine::EngineError,
        > {
            use cascadia_types::{Chunk, FinishReason};
            let Some(id) = self.task.clone() else {
                return Ok(Vec::new());
            };
            let n = self.emitted.load(SeqCst);
            if self.fail_after.is_some_and(|max| n >= max) {
                self.task = None;
                return Ok(vec![(id.clone(), Chunk::error(&id, "peer link dropped"))]);
            }
            if self.cancel_after.is_some_and(|max| n >= max) {
                self.task = None;
                let end = Chunk::final_marker(&id, "").with_finish_reason(FinishReason::Cancelled);
                return Ok(vec![(id, end)]);
            }
            let text = match &self.text {
                Text::Tok => "tok ",
                Text::Script(_) if n >= self.max_tokens => {
                    self.task = None;
                    let end = Chunk::final_marker(&id, "").with_finish_reason(FinishReason::Length);
                    return Ok(vec![(id, end)]);
                }
                Text::Script(chunks) => match chunks.get(n) {
                    Some(c) => c.as_str(),
                    None if self.never_end => return Ok(Vec::new()),
                    None => {
                        self.task = None;
                        let mut end =
                            Chunk::final_marker(&id, "").with_finish_reason(FinishReason::Stop);
                        if let Some(n) = self.final_tokens {
                            end = end.with_completion_tokens(n);
                        }
                        return Ok(vec![(id, end)]);
                    }
                },
            };
            let mut chunk = Chunk::token(&id, n as i64, text);
            if let Some(k) = self.chunk_tokens {
                chunk = chunk.with_n_tokens(k);
            }
            self.emitted.fetch_add(1, SeqCst);
            Ok(vec![(id, chunk)])
        }
    }

    struct RecordingBuilder(Option<RecordingEngine>);

    #[::async_trait::async_trait]
    impl cascadia_engine::Builder for RecordingBuilder {
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

    /// Byte limits of 1 MiB and the mock-tools fixture's chat template
    /// (tests/fixtures/mock-tools-model), which renders tools and tool calls.
    fn tools_config() -> Config {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/mock-tools-model");
        let template = load_chat_template_config(&dir).template;
        assert!(
            template.is_some(),
            "mock-tools fixture has no chat_template"
        );
        Config {
            max_prompt_bytes: 1 << 20,
            max_body_bytes: 1 << 20,
            chat_template: ChatTemplateConfig {
                template,
                bos_token: None,
                eos_token: Some("<|im_end|>".into()),
            },
            ..Config::default()
        }
    }

    /// A router over `engine`, with the engine's task log and emitted count.
    async fn app_with(
        engine: RecordingEngine,
        cfg: Config,
    ) -> (
        Router,
        Arc<parking_lot::Mutex<Vec<GenerationTask>>>,
        Arc<AtomicUsize>,
    ) {
        let (tasks, emitted) = (engine.tasks.clone(), engine.emitted.clone());
        let runner = Arc::new(Runner::new(Box::new(RecordingBuilder(Some(engine)))));
        runner
            .start(
                cascadia_types::PeerLayout::single_stage(),
                cascadia_types::ShardSpec::single_stage("script-model", "CPU"),
            )
            .await
            .unwrap();
        (
            make_router_with_config(runner, "script-model", cfg),
            tasks,
            emitted,
        )
    }

    /// The mock engine behind [`tools_config`]: it answers a tools request
    /// carrying `__mock_tool_call__` with a tool call.
    async fn mock_tools_app() -> Router {
        let runner = Runner::new(Box::new(cascadia_engine_mock::MockBuilder::new()));
        runner
            .start(
                cascadia_types::PeerLayout::single_stage(),
                cascadia_types::ShardSpec::single_stage("mock-model", "CPU"),
            )
            .await
            .unwrap();
        make_router_with_config(Arc::new(runner), "mock-model", tools_config())
    }

    /// Run one request against a [`RecordingEngine::tok`]. Without
    /// `cancel_after`, the request is cancelled through `/v1/cancel/<id>`
    /// once a token is out.
    async fn scripted(cancel_after: Option<usize>, stream: bool) -> (StatusCode, String) {
        scripted_with(cancel_after, None, stream, Config::default()).await
    }

    async fn scripted_with(
        cancel_after: Option<usize>,
        fail_after: Option<usize>,
        stream: bool,
        cfg: Config,
    ) -> (StatusCode, String) {
        let engine = RecordingEngine {
            cancel_after,
            fail_after,
            ..RecordingEngine::tok()
        };
        let (app, tasks, emitted) = app_with(engine, cfg).await;
        let req = tokio::spawn(post(
            app.clone(),
            json!({"model": "script-model", "input": "hello", "stream": stream}),
        ));
        if cancel_after.is_none() && fail_after.is_none() {
            while emitted.load(SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
            let id = tasks.lock().last().unwrap().task_id.clone();
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

    /// A [`RecordingEngine::script`] writes its chunks, then ends, and the
    /// task it got is recorded.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recording_engine_plays_its_script() {
        let (app, tasks, _) =
            app_with(RecordingEngine::script(&["al", "pha"]), Config::default()).await;
        let (status, v) = post_json(app, json!({"model": "script-model", "input": "hi"})).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["status"], "completed", "{v}");
        assert_eq!(v["output"][0]["content"][0]["text"], "alpha");
        assert_eq!(tasks.lock().len(), 1);
    }

    /// `body` against `engine` (built afresh per call), streamed (the
    /// terminal response) and not.
    async fn usage_both_ways(engine: impl Fn() -> RecordingEngine, body: Value) -> (Value, Value) {
        let mut streamed = body.clone();
        streamed["stream"] = json!(true);
        let (app, _, _) = app_with(engine(), Config::default()).await;
        let (status, text) = post(app, streamed).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let done = events(&text).last().unwrap().1["response"].clone();
        let (app, _, _) = app_with(engine(), Config::default()).await;
        let (status, v) = post_json(app, body).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        (done, v)
    }

    /// An engine whose token chunks carry no count and whose final chunk
    /// carries the authoritative total (llama.cpp's shape): both ways report
    /// that total as `output_tokens`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn usage_takes_the_final_chunks_total() {
        let engine = || RecordingEngine {
            chunk_tokens: Some(0),
            final_tokens: Some(7),
            ..RecordingEngine::script(&["hel", "lo"])
        };
        let (done, v) =
            usage_both_ways(engine, json!({"model": "script-model", "input": "hi"})).await;
        assert_eq!(done["usage"]["output_tokens"], 7, "{done}");
        assert_eq!(v["usage"]["output_tokens"], 7, "{v}");
    }

    /// The same engine shape reasoning: the authoritative total is split by
    /// the reasoning share of the bytes (19 of 25, tags included: 7.6 → 8).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reasoning_tokens_split_the_final_chunks_total() {
        let engine = || RecordingEngine {
            chunk_tokens: Some(0),
            final_tokens: Some(10),
            ..RecordingEngine::script(&["<think>plan</think>", "answer"])
        };
        let body = json!({"model": "script-model", "input": "hi",
            "reasoning": {"effort": "low"}});
        let (done, v) = usage_both_ways(engine, body).await;
        for r in [&done, &v] {
            assert_eq!(r["output"][0]["type"], "reasoning", "{r}");
            assert_eq!(r["usage"]["output_tokens"], 10, "{r}");
            assert_eq!(
                r["usage"]["output_tokens_details"]["reasoning_tokens"], 8,
                "{r}"
            );
        }
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
            let v: Value = serde_json::from_str(&body).unwrap();
            assert_spec("Response", &v);
            assert_cancelled(&v);

            let (status, body) = scripted(cancel_after, true).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            let ev = events(&body);
            ev.iter().for_each(|(_, data)| assert_event(data));
            assert!(ev.iter().all(|(k, _)| k != "response.completed"), "{body}");
            let (kind, data) = ev.last().unwrap();
            assert_eq!(kind, "response.incomplete");
            assert_cancelled(&data["response"]);
            // The open message was closed `incomplete` by its done events.
            let done: Vec<_> = ev
                .iter()
                .filter(|(k, _)| k == "response.output_item.done")
                .map(|(_, d)| d["item"].clone())
                .collect();
            assert_eq!(json!(done), data["response"]["output"], "{body}");
        }
    }

    /// A task that fails after the 200 is on the wire ends the stream with
    /// one `response.failed` event: status `failed`, the engine's reason in
    /// `error`, the tokens so far in `usage`, and nothing after it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn mid_stream_failure_ends_with_response_failed() {
        let (status, body) = scripted_with(None, Some(2), true, Config::default()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let ev = events(&body);
        ev.iter().for_each(|(_, data)| assert_event(data));
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
        // The open message is in `output`, closed `incomplete`, though no
        // closing event announced it.
        let output = response["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "{response}");
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["status"], "incomplete");
        assert_eq!(output[0]["id"], ev[2].1["item"]["id"]);
        assert_eq!(output[0]["content"][0]["text"], "tok tok ");
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
            json!({"model": "mock-model", "input": "hi", "previous_response_id": "resp_1"}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(count() > before);
    }

    /// A query string such as `?beta=true`, which some clients add, leaves
    /// the route as it is.
    #[tokio::test]
    async fn beta_query_is_served() {
        let (status, body) = post_to(
            make_app().await,
            "/v1/responses?beta=true",
            json!({"model": "mock-model", "input": "hi"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }

    /// Codex CLI 0.166.0's first request of a turn, trimmed (fake ids, short
    /// texts, four of its tools), as `tests/fixtures` keeps it.
    fn codex_turn() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/codex-0.166.0-turn1.json")).unwrap()
    }

    /// Codex 0.166.0's second request of the turn: the turn-1 fixture plus
    /// the replayed reasoning, `exec_command` call and its output.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn codex_turn2_replays_reasoning_and_tool_items() {
        let body: Value =
            serde_json::from_str(include_str!("../tests/fixtures/codex-0.166.0-turn2.json"))
                .unwrap();
        let (app, tasks, _) = app_with(RecordingEngine::script(&["done"]), tools_config()).await;
        let (status, text) = post(app, body).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let ev = events(&text);
        assert!(!ev.is_empty(), "{text}");
        ev.iter().for_each(|(_, data)| assert_event(data));
        let prompt = tasks.lock()[0].prompt.clone();
        assert!(prompt.contains("<think>I should list.</think>"), "{prompt}");
        // The system prompt names `<tool_call>` too: some block must hold
        // the replayed call.
        let replayed = prompt
            .split("<tool_call>")
            .skip(1)
            .filter_map(|rest| rest.split_once("</tool_call>"))
            .any(|(call, _)| call.contains("\"exec_command\""));
        assert!(replayed, "{prompt}");
        assert!(prompt.contains("a.txt"), "{prompt}");
    }

    /// The echoed `tools` of a Codex turn: its two functions and the
    /// namespace, never the hosted `web_search`.
    fn assert_codex_tools(response: &Value) {
        let tools = response["tools"].as_array().unwrap();
        let shape: Vec<_> = tools
            .iter()
            .map(|t| (t["type"].as_str().unwrap(), t["name"].as_str().unwrap()))
            .collect();
        assert_eq!(
            shape,
            [
                ("function", "exec_command"),
                ("function", "view_image"),
                ("namespace", "multi_agent_v1")
            ],
            "{response}"
        );
        assert_eq!(tools[2]["tools"].as_array().unwrap().len(), 2);
    }

    /// A Codex turn as captured (streamed) is served: every event is in the
    /// spec's shape and the terminal response echoes the served tools.
    #[tokio::test]
    async fn codex_turn_is_served_streaming() {
        let body = codex_turn();
        assert_eq!(body["stream"], true);
        let (status, text) = post(mock_tools_app().await, body).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let ev = events(&text);
        ev.iter().for_each(|(_, data)| assert_event(data));
        let (kind, data) = ev.last().unwrap();
        assert!(
            kind == "response.completed" || kind == "response.incomplete",
            "{kind}"
        );
        assert_codex_tools(&data["response"]);
    }

    #[tokio::test]
    async fn codex_turn_is_served_non_streaming() {
        let mut body = codex_turn();
        body["stream"] = json!(false);
        let (status, v) = post_json(mock_tools_app().await, body).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_spec("Response", &v);
        assert_codex_tools(&v);
        assert_eq!(v["store"], false);
        assert_eq!(v["parallel_tool_calls"], true);
        assert_eq!(v["tool_choice"], "auto");
        assert!(v["reasoning"]["summary"].is_null());
    }

    /// Served tools reach the chat template: a namespace's under their
    /// flattened names, a custom tool as a function, a hosted one not at all.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tools_reach_the_template() {
        let (app, tasks, _) = app_with(RecordingEngine::script(&["ok"]), tools_config()).await;
        let (status, v) = post_json(
            app,
            json!({"model": "script-model", "input": "hi", "tools": [
                {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
                {"type": "namespace", "name": "mcp__x", "description": "x",
                    "tools": [{"type": "function", "name": "click"}]},
                {"type": "custom", "name": "apply_patch", "description": "Patch files."},
                {"type": "web_search"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_spec("Response", &v);
        let prompt = tasks.lock()[0].prompt.clone();
        for name in ["get_weather", "mcp__x__click", "apply_patch"] {
            assert!(prompt.contains(name), "{name} missing: {prompt}");
        }
        assert!(!prompt.contains("web_search"), "{prompt}");
    }

    /// A namespace's `description: null` is echoed as `""`, as absent is:
    /// the schema wants a string.
    #[tokio::test]
    async fn null_namespace_description_is_echoed_empty() {
        let (status, v) = post_json(
            mock_tools_app().await,
            json!({"model": "mock-model", "input": "hi", "tools": [
                {"type": "namespace", "name": "mcp__x", "description": null,
                    "tools": [{"type": "function", "name": "click"}]}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_spec("Response", &v);
        assert_eq!(v["tools"][0]["description"], "", "{v}");
    }

    /// A function tool as n8n sends it (no `strict`) is echoed with
    /// `strict: null`, as the schema requires the key.
    #[tokio::test]
    async fn tool_without_strict_is_echoed_with_null() {
        let (status, v) = post_json(
            mock_tools_app().await,
            json!({"model": "mock-model", "input": "hi", "tools": [{"type": "function",
                "name": "calculator", "description": "Do math.",
                "parameters": {"type": "object", "properties": {"input": {"type": "string"}}}}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_spec("Response", &v);
        assert!(v["tools"][0]["strict"].is_null(), "{v}");
        assert_eq!(v["tools"][0]["name"], "calculator");
    }

    /// The response object echoes what was asked: `store` (true when
    /// absent, OpenAI's default), `max_tool_calls`, `tool_choice` (`auto`
    /// when absent), `parallel_tool_calls` (true when absent), and a
    /// `reasoning.summary` of null (none is generated).
    #[tokio::test]
    async fn request_fields_are_echoed() {
        let echo = |extra: Value| async move {
            let mut body = json!({"model": "mock-model", "input": "hi"});
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let (status, v) = post_json(make_app().await, body).await;
            assert_eq!(status, StatusCode::OK, "{v}");
            assert_spec("Response", &v);
            v
        };
        let v = echo(json!({})).await;
        assert_eq!(v["store"], true);
        assert_eq!(v["tool_choice"], "auto");
        assert_eq!(v["parallel_tool_calls"], true);
        assert!(v["max_tool_calls"].is_null());
        assert_eq!(v["tools"], json!([]));
        let v = echo(
            json!({"store": false, "max_tool_calls": 5, "tool_choice": "none",
            "parallel_tool_calls": false, "reasoning": {"effort": "low", "summary": "auto"}}),
        )
        .await;
        assert_eq!(v["store"], false);
        assert_eq!(v["max_tool_calls"], 5);
        assert_eq!(v["tool_choice"], "none");
        assert_eq!(v["parallel_tool_calls"], false);
        assert_eq!(v["reasoning"], json!({"effort": "low", "summary": null}));
        let v = echo(json!({"max_tool_calls": null, "stream": null})).await;
        assert!(v["max_tool_calls"].is_null(), "{v}");
    }

    /// `stream: null` is the spec's nullable default: not streamed.
    #[tokio::test]
    async fn null_stream_is_not_streamed() {
        let (status, v) = post_json(
            make_app().await,
            json!({"model": "mock-model", "input": "hi", "stream": null}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["object"], "response", "{v}");
        assert_spec("Response", &v);
    }

    /// Tool and tool-choice refusals name the field, before the engine.
    #[tokio::test]
    async fn tool_refusals_name_the_field() {
        for (extra, param) in [
            (json!({"tools": [{"type": "foo"}]}), "tools[0].type"),
            (json!({"tools": [{"type": "function"}]}), "tools[0].name"),
            (
                json!({"tools": [{"type": "function", "name": "f"}],
                    "tool_choice": {"type": "function", "name": "f"}}),
                "tool_choice",
            ),
            (
                json!({"tool_choice": {"type": "web_search_preview"}}),
                "tool_choice",
            ),
        ] {
            let mut body = json!({"model": "mock-model", "input": "hi"});
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let (status, v) = post_json(mock_tools_app().await, body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{param}: {v}");
            assert_eq!(v["error"]["param"], param, "{v}");
        }
    }

    fn get_weather() -> Value {
        json!({"type": "function", "name": "get_weather", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city", "days"]}})
    }

    /// A non-streamed turn that reasons, then calls a tool, comes back as a
    /// reasoning item and a `function_call`, the reasoning tokens counted.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn non_streaming_returns_reasoning_and_a_call() {
        let engine = RecordingEngine::script(&[
            "<think>",
            "plan",
            "</think>",
            "<tool_call>",
            "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}",
            "</tool_call>",
        ]);
        let (app, _, _) = app_with(engine, tools_config()).await;
        let (status, v) = post_json(
            app,
            json!({"model": "script-model", "input": "weather?", "tools": [get_weather()]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_spec("Response", &v);
        let kinds: Vec<_> = v["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["type"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, ["reasoning", "function_call"], "{v}");
        assert_eq!(v["output"][0]["content"][0]["text"], "plan");
        assert_eq!(v["output"][1]["name"], "get_weather");
        assert_eq!(v["status"], "completed");
        assert_eq!(
            v["usage"]["output_tokens_details"]["reasoning_tokens"], 3,
            "{v}"
        );
    }

    /// The client tool loop against the mock engine: a call, then the
    /// answer to its output, sending the same tool again.
    #[tokio::test]
    async fn mock_tool_loop_round_trips() {
        let app = mock_tools_app().await;
        let (status, v) = post_json(
            app.clone(),
            json!({"model": "mock-model", "input": "weather __mock_tool_call__",
                "tools": [get_weather()]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_spec("Response", &v);
        let output = v["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "{v}");
        let call = &output[0];
        assert_eq!(call["type"], "function_call");
        assert_eq!(call["name"], "get_weather");
        let args: Value = serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({"city": "mock", "days": 1}));

        let (status, v) = post_json(
            app,
            json!({"model": "mock-model", "tools": [get_weather()], "input": [
                {"role": "user", "content": "weather __mock_tool_call__"},
                {"type": "function_call", "call_id": "call_1", "name": "get_weather",
                    "arguments": "{\"city\":\"mock\",\"days\":1}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "sunny"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_spec("Response", &v);
        assert_eq!(v["status"], "completed");
        let output = v["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "{v}");
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["content"][0]["text"], "mock: got sunny");
    }

    /// The events of a stream, each checked against its schema, numbered
    /// from 0 and carrying its own `type`.
    fn checked_events(body: &str) -> Vec<(String, Value)> {
        let ev = events(body);
        for (i, (kind, data)) in ev.iter().enumerate() {
            assert_event(data);
            assert_eq!(data["type"], kind.as_str(), "{body}");
            assert_eq!(data["sequence_number"], i as u64, "{body}");
        }
        ev
    }

    fn kinds(ev: &[(String, Value)]) -> Vec<&str> {
        ev.iter().map(|(k, _)| k.as_str()).collect()
    }

    /// `output` with each item's generated ids dropped.
    fn without_ids(output: &Value) -> Value {
        let mut out = output.clone();
        for item in out.as_array_mut().unwrap() {
            let item = item.as_object_mut().unwrap();
            item.remove("id");
            item.remove("call_id");
        }
        out
    }

    /// `body` against a [`RecordingEngine::script`] of `chunks`, streamed
    /// (checked events) and not (a checked response).
    async fn both_ways(chunks: &[&str], body: Value) -> (Vec<(String, Value)>, Value) {
        let mut streamed = body.clone();
        streamed["stream"] = json!(true);
        let (app, _, _) = app_with(RecordingEngine::script(chunks), tools_config()).await;
        let (status, text) = post(app, streamed).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let ev = checked_events(&text);
        let (app, _, _) = app_with(RecordingEngine::script(chunks), tools_config()).await;
        let (status, v) = post_json(app, body).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_spec("Response", &v);
        (ev, v)
    }

    /// A streamed turn that reasons, then calls a tool: the reasoning item's
    /// events, then the call's, and the terminal output is the non-streamed
    /// one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn streaming_returns_reasoning_and_a_call() {
        let (ev, v) = both_ways(
            &[
                "<think>",
                "pl",
                "an",
                "</think>",
                "<tool_call>",
                "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}",
                "</tool_call>",
            ],
            json!({"model": "script-model", "input": "weather?", "tools": [get_weather()]}),
        )
        .await;
        assert_eq!(
            kinds(&ev),
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_eq!(ev[2].1["item"]["type"], "reasoning");
        assert_eq!(
            ev[3].1["part"],
            json!({"type": "reasoning_text", "text": ""})
        );
        assert_eq!(ev[6].1["text"], "plan");
        assert_eq!(ev[9].1["item"]["type"], "function_call");
        assert_eq!(ev[9].1["item"]["arguments"], "");
        let call_id = ev[9].1["item"]["id"].clone();
        for (i, (_, d)) in ev[2..13].iter().enumerate() {
            let index = if i < 7 { 0 } else { 1 };
            assert_eq!(d["output_index"], index, "{d}");
            if i < 7 {
                if let Some(c) = d.get("content_index") {
                    assert_eq!(c, 0, "{d}");
                }
            }
        }
        assert_eq!(ev[10].1["item_id"], call_id);
        let args: Value = serde_json::from_str(ev[11].1["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({"city": "Paris"}));
        assert_eq!(ev[11].1["name"], "get_weather", "{:?}", ev[11]);
        let done = &ev.last().unwrap().1["response"];
        assert_eq!(done["status"], "completed");
        assert_eq!(done["output"][0], ev[8].1["item"]);
        assert_eq!(done["output"][1], ev[12].1["item"]);
        assert_eq!(without_ids(&done["output"]), without_ids(&v["output"]));
        assert_eq!(
            done["usage"]["output_tokens_details"]["reasoning_tokens"],
            v["usage"]["output_tokens_details"]["reasoning_tokens"]
        );
    }

    /// A custom tool's call streams its `input` with the custom events.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn streaming_a_custom_tool_call() {
        let (ev, v) = both_ways(
            &[
                "<tool_call>",
                "{\"name\":\"apply_patch\",\"arguments\":{\"input\":\"*** Begin\"}}",
                "</tool_call>",
            ],
            json!({"model": "script-model", "input": "patch it", "tools": [
                {"type": "custom", "name": "apply_patch", "description": "Patch files."}]}),
        )
        .await;
        assert_eq!(
            kinds(&ev),
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.custom_tool_call_input.delta",
                "response.custom_tool_call_input.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_eq!(ev[2].1["item"]["type"], "custom_tool_call");
        assert_eq!(ev[2].1["item"]["input"], "");
        assert_eq!(ev[3].1["delta"], "*** Begin");
        assert_eq!(ev[4].1["input"], "*** Begin");
        let done = &ev.last().unwrap().1["response"];
        assert_eq!(done["output"][0]["input"], "*** Begin");
        assert_eq!(without_ids(&done["output"]), without_ids(&v["output"]));
    }

    /// Whitespace before a call opens no message: a pure tool-call turn has
    /// no message events.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn streaming_call_after_whitespace_has_no_message() {
        let (ev, v) = both_ways(
            &[
                "\n",
                "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"}}</tool_call>",
            ],
            json!({"model": "script-model", "input": "weather?", "tools": [get_weather()]}),
        )
        .await;
        assert!(
            ev.iter().all(|(k, d)| !k.contains("output_text")
                && !k.contains("content_part")
                && d["item"]["type"] != "message"),
            "{ev:?}"
        );
        let done = &ev.last().unwrap().1["response"];
        let output = done["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "{done}");
        assert_eq!(output[0]["type"], "function_call");
        assert_eq!(without_ids(&done["output"]), without_ids(&v["output"]));
    }

    /// The mock engine's tool-call sentinel, streamed: one call, its events.
    #[tokio::test]
    async fn mock_tool_call_streams() {
        let (status, text) = post(
            mock_tools_app().await,
            json!({"model": "mock-model", "input": "weather __mock_tool_call__",
                "tools": [get_weather()], "stream": true}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let ev = checked_events(&text);
        let k = kinds(&ev);
        assert!(
            k.contains(&"response.function_call_arguments.delta"),
            "{k:?}"
        );
        assert!(
            k.contains(&"response.function_call_arguments.done"),
            "{k:?}"
        );
        let (kind, data) = ev.last().unwrap();
        assert_eq!(kind, "response.completed");
        let output = data["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "{data}");
        assert_eq!(output[0]["type"], "function_call");
        let args: Value = serde_json::from_str(output[0]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({"city": "mock", "days": 1}));
    }

    /// `include: ["reasoning.encrypted_content"]` fills the reasoning item's
    /// `encrypted_content` with a blob that decodes to its text; without it
    /// the key is absent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn encrypted_reasoning_is_returned_when_included() {
        let body = json!({"model": "script-model", "input": "hi",
            "include": ["reasoning.encrypted_content"]});
        let (app, _, _) = app_with(
            RecordingEngine::script(&["<think>r</think>hi"]),
            tools_config(),
        )
        .await;
        let (status, v) = post_json(app, body).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_spec("Response", &v);
        let r = &v["output"][0];
        assert_eq!(r["type"], "reasoning", "{v}");
        let blob = r["encrypted_content"].as_str().expect("encrypted_content");
        assert_eq!(input::decode_reasoning(blob).as_deref(), Some("r"));
        assert_eq!(v["output"][1]["content"][0]["text"], "hi");

        let (app, _, _) = app_with(
            RecordingEngine::script(&["<think>r</think>hi"]),
            tools_config(),
        )
        .await;
        let (status, v) = post_json(app, json!({"model": "script-model", "input": "hi"})).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_spec("Response", &v);
        assert_eq!(v["output"][0]["type"], "reasoning", "{v}");
        assert!(v["output"][0].get("encrypted_content").is_none(), "{v}");
    }

    /// A generation with no text is a completed response with no items,
    /// streamed (no item events) or not.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn empty_generation_has_no_output() {
        let (ev, v) = both_ways(&[], json!({"model": "script-model", "input": "hi"})).await;
        assert_eq!(
            kinds(&ev),
            [
                "response.created",
                "response.in_progress",
                "response.completed"
            ]
        );
        let done = &ev[2].1["response"];
        assert_eq!(done["status"], "completed");
        assert_eq!(done["output"], json!([]));
        assert_spec("Response", done);
        assert_eq!(v["status"], "completed");
        assert_eq!(v["output"], json!([]));
    }

    /// A follow-up request sending `refs` as `item_reference`s between two
    /// user turns.
    async fn send_refs(app: Router, refs: &[&Value]) -> (StatusCode, Value) {
        let mut input = vec![json!({"role": "user", "content": "first"})];
        input.extend(
            refs.iter()
                .map(|id| json!({"type": "item_reference", "id": id})),
        );
        input.push(json!({"role": "user", "content": "again"}));
        post_json(app, json!({"model": "script-model", "input": input})).await
    }

    /// The script the stored-item tests answer request 1 with.
    const REASON_THEN_HELLO: &[&str] = &["<think>", "r", "</think>", "hello"];

    /// Request 1's output items (non-streamed), with `store` as given.
    async fn first_output(app: Router, store: Option<bool>) -> Vec<Value> {
        let mut body = json!({"model": "script-model", "input": "hi"});
        if let Some(store) = store {
            body["store"] = json!(store);
        }
        let (status, v) = post_json(app, body).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_spec("Response", &v);
        v["output"].as_array().unwrap().clone()
    }

    /// With `store` absent or `true`, a response's output items resolve as
    /// `item_reference`s in a later request: the reasoning and the answer
    /// reach the prompt as the assistant turn they were.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stored_items_resolve_by_reference() {
        for store in [None, Some(true)] {
            let (app, tasks, _) =
                app_with(RecordingEngine::script(REASON_THEN_HELLO), tools_config()).await;
            let output = first_output(app.clone(), store).await;
            let (rs, msg) = (&output[0]["id"], &output[1]["id"]);
            assert!(rs.as_str().unwrap().starts_with("rs_"), "{output:?}");
            assert!(msg.as_str().unwrap().starts_with("msg_"), "{output:?}");
            let (status, v) = send_refs(app, &[rs, msg]).await;
            assert_eq!(status, StatusCode::OK, "{store:?}: {v}");
            let prompt = tasks.lock()[1].prompt.clone();
            assert!(prompt.contains("<think>r</think>hello"), "{prompt}");
        }
    }

    /// `store: false` keeps nothing: a reference to its item is a 400
    /// naming the reference, its message pointing at `store: false`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unstored_items_are_not_found() {
        let (app, _, _) =
            app_with(RecordingEngine::script(REASON_THEN_HELLO), tools_config()).await;
        let output = first_output(app.clone(), Some(false)).await;
        let (status, v) = send_refs(app, &[&output[0]["id"]]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
        assert_eq!(v["error"]["param"], "input[1]", "{v}");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(message.contains("store: false"), "{message}");
    }

    /// A streamed response's items are stored by the time its terminal
    /// event is read.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn streamed_items_resolve_after_the_terminal_event() {
        let (app, tasks, _) =
            app_with(RecordingEngine::script(REASON_THEN_HELLO), tools_config()).await;
        let (status, text) = post(
            app.clone(),
            json!({"model": "script-model", "input": "hi", "stream": true}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let ev = checked_events(&text);
        let (kind, data) = ev.last().unwrap();
        assert_eq!(kind, "response.completed");
        let output = &data["response"]["output"];
        let (status, v) = send_refs(app, &[&output[0]["id"], &output[1]["id"]]).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        let prompt = tasks.lock()[1].prompt.clone();
        assert!(prompt.contains("<think>r</think>hello"), "{prompt}");
    }

    /// A `failed` response's items are not stored; a cancelled one's are.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn failed_items_are_not_stored_cancelled_ones_are() {
        for (engine, terminal, found) in [
            (
                RecordingEngine {
                    fail_after: Some(2),
                    ..RecordingEngine::tok()
                },
                "response.failed",
                false,
            ),
            (
                RecordingEngine {
                    cancel_after: Some(2),
                    ..RecordingEngine::tok()
                },
                "response.incomplete",
                true,
            ),
        ] {
            let (app, tasks, _) = app_with(engine, tools_config()).await;
            let (status, text) = post(
                app.clone(),
                json!({"model": "script-model", "input": "hi", "stream": true}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{text}");
            let ev = checked_events(&text);
            let (kind, data) = ev.last().unwrap();
            assert_eq!(kind, terminal, "{text}");
            let id = &data["response"]["output"][0]["id"];
            assert!(id.is_string(), "no output item id: {text}");
            let (status, v) = send_refs(app, &[id]).await;
            if found {
                assert_eq!(data["response"]["status"], "cancelled");
                assert_eq!(status, StatusCode::OK, "{v}");
                let prompt = tasks.lock()[1].prompt.clone();
                assert!(prompt.contains("tok tok"), "{prompt}");
            } else {
                assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
                assert_eq!(v["error"]["param"], "input[1]", "{v}");
            }
        }
    }

    /// A client that disconnects mid-stream leaves nothing stored: the
    /// announced item never resolves.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_disconnected_stream_stores_nothing() {
        use futures::StreamExt;
        let (app, _, _) = app_with(RecordingEngine::tok(), tools_config()).await;
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"model": "script-model", "input": "hi", "stream": true}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body().into_data_stream();
        let mut text = String::new();
        let id = loop {
            let frame = body.next().await.expect("stream ended").unwrap();
            text.push_str(&String::from_utf8_lossy(&frame));
            let added = events(&text)
                .into_iter()
                .find(|(k, _)| k == "response.output_item.added");
            if let Some((_, data)) = added {
                break data["item"]["id"].clone();
            }
        };
        drop(body);
        let (status, v) = send_refs(app, &[&id]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
        assert_eq!(v["error"]["param"], "input[1]", "{v}");
    }

    /// References past `max_prompt_bytes` are a 413 from the walk itself
    /// (`code: prompt_too_large`, naming the reference that passed it),
    /// before any item past the budget is cloned.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn references_past_the_prompt_budget_are_refused() {
        let text = "x".repeat(480);
        let cfg = Config {
            max_prompt_bytes: 1000,
            ..Config::default()
        };
        let (app, _, _) = app_with(RecordingEngine::script(&[&text]), cfg).await;
        let output = first_output(app.clone(), None).await;
        let size = serde_json::to_vec(&output[0]).unwrap().len();
        assert!((501..1000).contains(&size), "{size}");
        let id = &output[0]["id"];
        let (status, v) = send_refs(app, &[id, id, id]).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{v}");
        assert_eq!(v["error"]["code"], "prompt_too_large", "{v}");
        assert_eq!(v["error"]["param"], "input[2]", "{v}");
    }

    /// A turn that says something, calls a tool and then hits
    /// `max_output_tokens`: the message is complete, the call parsed, the
    /// response `incomplete`. (The [`RecordingEngine`] stops a script at the
    /// task's `max_tokens` with `Length`.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn streaming_a_call_then_the_token_cap() {
        let (app, _, _) = app_with(
            RecordingEngine::script(&[
                "Let me check",
                "<tool_call>",
                "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}",
                "</tool_call>",
                "never sent",
            ]),
            tools_config(),
        )
        .await;
        let (status, text) = post(
            app,
            json!({"model": "script-model", "input": "weather?", "tools": [get_weather()],
                "max_output_tokens": 4, "stream": true}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let ev = checked_events(&text);
        let (kind, data) = ev.last().unwrap();
        assert_eq!(kind, "response.incomplete", "{text}");
        let response = &data["response"];
        assert_eq!(response["status"], "incomplete");
        assert_eq!(
            response["incomplete_details"]["reason"],
            "max_output_tokens"
        );
        let output = response["output"].as_array().unwrap();
        assert_eq!(output.len(), 2, "{response}");
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["status"], "completed", "{response}");
        assert_eq!(output[0]["content"][0]["text"], "Let me check");
        assert_eq!(output[1]["type"], "function_call");
        assert_eq!(output[1]["name"], "get_weather");
        let args: Value = serde_json::from_str(output[1]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({"city": "Paris"}));
        let done = ev
            .iter()
            .find(|(k, _)| k == "response.function_call_arguments.done")
            .expect("arguments done");
        assert_eq!(done.1["name"], "get_weather");
    }

    /// A stream cancelled through `/v1/cancel` while reasoning ends on
    /// `response.incomplete`, status `cancelled`, the reasoning item
    /// `incomplete` in `output`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cancel_while_reasoning() {
        let engine = RecordingEngine {
            never_end: true,
            ..RecordingEngine::script(&["<think>", "hmm"])
        };
        let (app, tasks, emitted) = app_with(engine, tools_config()).await;
        let req = tokio::spawn(post(
            app.clone(),
            json!({"model": "script-model", "input": "hi", "stream": true}),
        ));
        while emitted.load(SeqCst) < 2 {
            tokio::task::yield_now().await;
        }
        let id = tasks.lock().last().unwrap().task_id.clone();
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
        let (status, text) = req.await.unwrap();
        assert_eq!(status, StatusCode::OK, "{text}");
        let ev = checked_events(&text);
        let (kind, data) = ev.last().unwrap();
        assert_eq!(kind, "response.incomplete", "{text}");
        let response = &data["response"];
        assert_eq!(response["status"], "cancelled", "{response}");
        let output = response["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "{response}");
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["status"], "incomplete", "{response}");
    }

    /// Gemma 3 / Mistral style: a leading system turn only, then user and
    /// assistant alternate (tool turns aside).
    const STRICT_TEMPLATE: &str = concat!(
        "{%- set ns = namespace(next='user') -%}",
        "{%- for m in messages -%}",
        "{%- if m.role == 'system' -%}",
        "{%- if not loop.first -%}{{ raise_exception('System role must come first') }}{%- endif -%}",
        "<sys>{{ m.content }}</sys>",
        "{%- elif m.role == 'tool' -%}<tool>{{ m.content }}</tool>",
        "{%- else -%}",
        "{%- if m.role != ns.next -%}",
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}",
        "{%- endif -%}",
        "<{{ m.role }}>{{ m.content }}</{{ m.role }}>",
        "{%- set ns.next = 'assistant' if m.role == 'user' else 'user' -%}",
        "{%- endif -%}",
        "{%- endfor -%}",
        "{%- if add_generation_prompt -%}<assistant>{%- endif -%}",
    );

    /// Gemma 2 / Gemma 1 style: no system role at all.
    const NO_SYSTEM_TEMPLATE: &str = concat!(
        "{%- for m in messages -%}",
        "{%- if m.role == 'system' -%}{{ raise_exception('System role not supported') }}{%- endif -%}",
        "<{{ m.role }}>{{ m.content }}</{{ m.role }}>",
        "{%- endfor -%}",
        "{%- if add_generation_prompt -%}<assistant>{%- endif -%}",
    );

    /// Takes any role anywhere (Gemma 4, Qwen).
    const PERMISSIVE_TEMPLATE: &str = concat!(
        "{%- for m in messages -%}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{%- endfor -%}",
        "{%- if add_generation_prompt -%}<assistant>{%- endif -%}",
    );

    fn template_config(src: &str) -> Config {
        Config {
            chat_template: ChatTemplateConfig {
                template: Some(src.into()),
                bos_token: None,
                eos_token: None,
            },
            ..Config::default()
        }
    }

    /// Codex's shape: instructions, a user turn, a call and its output, then
    /// a `developer` turn and the next user turn.
    fn developer_mid_history() -> Value {
        json!({"model": "script-model", "instructions": "be brief", "input": [
            {"role": "user", "content": "list files"},
            {"type": "function_call", "call_id": "call_1", "name": "ls", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "a.txt"},
            {"role": "developer", "content": "now be terse"},
            {"role": "user", "content": "and now?"},
        ]})
    }

    /// The prompt the engine got for `body` under `template`, with the
    /// status and body.
    async fn prompt_under(template: &str, body: Value) -> (StatusCode, String, String) {
        let (app, tasks, _) =
            app_with(RecordingEngine::script(&["ok"]), template_config(template)).await;
        let (status, body) = post(app, body).await;
        let prompt = tasks.lock().first().map(|t| t.prompt.clone());
        (status, body, prompt.unwrap_or_default())
    }

    /// A template that takes only a leading system turn gets the later
    /// `developer` turn folded into the next user turn, its text first.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_late_system_turn_folds_into_the_next_user_turn() {
        let (status, body, prompt) = prompt_under(STRICT_TEMPLATE, developer_mid_history()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(prompt.matches("<sys>").count(), 1, "{prompt}");
        assert!(prompt.starts_with("<sys>be brief</sys>"), "{prompt}");
        assert!(
            prompt.ends_with("<user>now be terse\n\nand now?</user><assistant>"),
            "{prompt}"
        );
    }

    /// A `developer` turn between a tool output and the next call stays
    /// there, as a user turn, so the history keeps its order and alternates.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_developer_turn_before_a_call_folds_in_place() {
        let (status, body, prompt) = prompt_under(
            STRICT_TEMPLATE,
            json!({"model": "script-model", "instructions": "be brief", "input": [
                {"role": "user", "content": "list files"},
                {"type": "function_call", "call_id": "c1", "name": "ls", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
                {"role": "developer", "content": "now be terse"},
                {"type": "function_call", "call_id": "c2", "name": "ls", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c2", "output": "b.txt"},
                {"role": "user", "content": "and now?"},
            ]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            prompt,
            "<sys>be brief</sys><user>list files</user><assistant></assistant>\
             <tool>a.txt</tool><user>now be terse</user><assistant></assistant>\
             <tool>b.txt</tool><user>and now?</user><assistant>"
        );
    }

    /// The fold is judged with the request's tools in the render: a strict
    /// template that prints them still folds the late system turn.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_late_system_turn_folds_with_tools() {
        let template = format!(
            "{{%- if tools -%}}<tools>{{{{ tools | tojson }}}}</tools>{{%- endif -%}}{STRICT_TEMPLATE}"
        );
        let mut body = developer_mid_history();
        body["tools"] = json!([{"type": "function", "name": "get_weather",
            "parameters": {"type": "object"}}]);
        let (status, body, prompt) = prompt_under(&template, body).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(prompt.starts_with("<tools>"), "{prompt}");
        assert!(prompt.contains("get_weather"), "{prompt}");
        assert_eq!(prompt.matches("<sys>").count(), 1, "{prompt}");
        assert!(
            prompt.ends_with("<user>now be terse\n\nand now?</user><assistant>"),
            "{prompt}"
        );
    }

    /// (role, content) of each turn forwarded to the engine.
    fn forwarded(task: &GenerationTask) -> Vec<(String, String)> {
        task.messages
            .iter()
            .map(|t| (t.role.clone(), t.content.clone()))
            .collect()
    }

    /// The folded turns `developer_mid_history` reaches the engine as.
    fn folded_turns() -> Vec<(String, String)> {
        [
            ("system", "be brief"),
            ("user", "list files"),
            ("assistant", ""),
            ("tool", "a.txt"),
            ("user", "now be terse\n\nand now?"),
        ]
        .map(|(r, c)| (r.to_owned(), c.to_owned()))
        .to_vec()
    }

    /// An engine that renders the template itself from the forwarded turns
    /// (sycl-llama) is sent the folded turns: the API never renders the
    /// template there, so it probes it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_engine_that_templates_itself_gets_the_folded_turns() {
        let cfg = Config {
            engine_applies_template: true,
            ..template_config(STRICT_TEMPLATE)
        };
        let (app, tasks, _) = app_with(RecordingEngine::script(&["ok"]), cfg).await;
        let (status, body) = post(app, developer_mid_history()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(forwarded(&tasks.lock()[0]), folded_turns());
    }

    /// An ov-genai-style engine with thinking on (the API renders the
    /// prompt) folds too, in the prompt and the forwarded turns.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_thinking_request_on_a_deferring_engine_folds() {
        let cfg = Config {
            defer_template_on_thinking: true,
            ..template_config(STRICT_TEMPLATE)
        };
        let (app, tasks, _) = app_with(RecordingEngine::script(&["ok"]), cfg).await;
        let (status, body) = post(app, developer_mid_history()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let task = tasks.lock()[0].clone();
        assert!(task.enable_thinking && task.prompt_rendered);
        assert!(
            task.prompt
                .ends_with("<user>now be terse\n\nand now?</user><assistant>"),
            "{}",
            task.prompt
        );
        assert_eq!(forwarded(&task), folded_turns());
    }

    /// An image request folds with its image where it was: the system text
    /// leads the user turn, before the image's placeholder.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_image_request_folds_with_its_image_in_place() {
        let cfg = Config {
            media: Some(MediaConfig::ov_genai(None)),
            ..template_config(NO_SYSTEM_TEMPLATE)
        };
        let (app, tasks, _) = app_with(RecordingEngine::script(&["ok"]), cfg).await;
        let (status, body) = post(
            app,
            json!({"model": "script-model", "instructions": "be brief", "input": [
                {"role": "user", "content": [
                    {"type": "input_text", "text": "describe"},
                    {"type": "input_image", "image_url": png_url(8, 8)}]}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let task = tasks.lock()[0].clone();
        assert_eq!(task.media.len(), 1);
        assert!(!task.prompt.contains("<system>"), "{}", task.prompt);
        let (sys, text, image) = (
            task.prompt.find("be brief").expect("system text"),
            task.prompt.find("describe").expect("user text"),
            task.prompt.find("<ov_genai_image_0>").expect("placeholder"),
        );
        assert!(task.prompt.starts_with("<user>"), "{}", task.prompt);
        assert!(sys < text && text < image, "{}", task.prompt);
        assert_eq!(
            forwarded(&task),
            [("user".to_owned(), "be brief\n\ndescribe".to_owned())]
        );
    }

    /// On an engine that templates itself with no template here
    /// (sycl-llama), the task asks for reasoning as text, and reasoning sent
    /// as `<think>`…`</think>` ahead of the answer is a reasoning item; with
    /// thinking off, or on the chat endpoint, it does not ask.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reasoning_sent_as_text_is_a_reasoning_item() {
        let cfg = || Config {
            engine_applies_template: true,
            ..Config::default()
        };
        let script = ["<think>", "17 times 3 is 51", "</think>", "51"];
        let (app, tasks, _) = app_with(RecordingEngine::script(&script), cfg()).await;
        let (status, v) = post_json(
            app,
            json!({"model": "script-model", "input": "What is 17 times 3?"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert!(tasks.lock()[0].reasoning_as_text);
        let kinds: Vec<_> = v["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["type"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, ["reasoning", "message"], "{v}");
        assert_eq!(v["output"][0]["content"][0]["text"], "17 times 3 is 51");
        assert_eq!(v["output"][1]["content"][0]["text"], "51");

        // Thinking off: the reply is not split, so reasoning is not asked
        // for as text (a model that reasons anyway would show its markers).
        let (app, tasks, _) = app_with(RecordingEngine::script(&["51"]), cfg()).await;
        let (status, v) = post_json(
            app,
            json!({"model": "script-model", "input": "What is 17 times 3?",
                "reasoning": {"effort": "none"}}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert!(!tasks.lock()[0].enable_thinking);
        assert!(!tasks.lock()[0].reasoning_as_text);

        let (app, tasks, _) = app_with(RecordingEngine::script(&["51"]), cfg()).await;
        let (status, body) = post_to(
            app,
            "/v1/chat/completions",
            json!({"model": "script-model",
                "messages": [{"role": "user", "content": "What is 17 times 3?"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(!tasks.lock()[0].reasoning_as_text);
    }

    /// A template that takes no system role gets every system turn folded,
    /// the instructions into the first user turn.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn every_system_turn_folds_when_the_template_has_no_system_role() {
        let (status, body, prompt) =
            prompt_under(NO_SYSTEM_TEMPLATE, developer_mid_history()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(!prompt.contains("<system>"), "{prompt}");
        assert!(
            prompt.starts_with("<user>be brief\n\nlist files</user>"),
            "{prompt}"
        );
        assert!(
            prompt.contains("<user>now be terse\n\nand now?</user>"),
            "{prompt}"
        );
    }

    /// A template that takes system turns anywhere sees them as sent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_template_that_takes_late_system_turns_is_unchanged() {
        let (status, body, prompt) =
            prompt_under(PERMISSIVE_TEMPLATE, developer_mid_history()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            prompt,
            "<system>be brief</system><user>list files</user><assistant></assistant>\
             <tool>a.txt</tool><system>now be terse</system><user>and now?</user><assistant>"
        );
    }

    /// A template that refuses for another reason still refuses: folding
    /// does not help, so the template's own message is the 400.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_template_refusing_for_another_reason_stays_a_400() {
        let (status, body, _) = prompt_under(
            "{{ raise_exception('no thanks') }}",
            developer_mid_history(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            body.contains("chat template rejected the request: no thanks"),
            "{body}"
        );
    }

    /// A `developer` turn with no user turn after it becomes the last user
    /// turn.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_trailing_developer_turn_becomes_a_user_turn() {
        let (status, body, prompt) = prompt_under(
            STRICT_TEMPLATE,
            json!({"model": "script-model", "input": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yes"},
                {"role": "developer", "content": "be terse"},
            ]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            prompt,
            "<user>hi</user><assistant>yes</assistant><user>be terse</user><assistant>"
        );
    }

    /// `/v1/chat/completions` does not fold: a late system message under
    /// the strict template is the template's 400, as before.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn chat_completions_does_not_fold_system_turns() {
        let (app, _, _) = app_with(
            RecordingEngine::script(&["ok"]),
            template_config(STRICT_TEMPLATE),
        )
        .await;
        let (status, body) = post_to(
            app,
            "/v1/chat/completions",
            json!({"model": "script-model", "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yes"},
                {"role": "system", "content": "late"},
                {"role": "user", "content": "again"},
            ]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            body.contains("chat template rejected the request: System role must come first"),
            "{body}"
        );
    }
}
