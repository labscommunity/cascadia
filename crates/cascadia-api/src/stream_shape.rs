//! Shared pieces of the OpenAI response wire shape: the `system_fingerprint`
//! value, the `usage` object and the streamed chunk frames.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::Serialize;

/// OpenAI's `system_fingerprint`: `fp_` plus 10 lowercase hex digits. It is
/// the same on every response of one server and changes when the release,
/// the engine or the served model changes, which is what clients use it to
/// notice. FNV-1a is written out because std's `DefaultHasher` may change
/// between Rust releases, and the value must not.
pub(crate) fn fingerprint(version: &str, engine: &str, model_id: &str) -> String {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let key = format!("cascadia/{version}/{engine}/{model_id}");
    let hash = key
        .bytes()
        .fold(OFFSET, |h, b| (h ^ u64::from(b)).wrapping_mul(PRIME));
    // The top 40 bits are the first 10 of the 16 zero-padded hex digits.
    format!("fp_{:010x}", hash >> 24)
}

/// `usage.prompt_tokens_details`. A count is sent only when it is measured;
/// prompt caching is not counted yet, so the object is empty rather than
/// claiming 0 cached tokens.
#[derive(Serialize, Default)]
pub(crate) struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cached_tokens: Option<u32>,
}

/// `usage.completion_tokens_details`. Reasoning tokens are not counted
/// apart from the answer yet, so the object is empty rather than claiming 0.
#[derive(Serialize, Default)]
pub(crate) struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_tokens: Option<u32>,
}

/// The OpenAI `usage` object, details included: OpenAI sends both details
/// objects for text models and some clients read them.
#[derive(Serialize)]
pub(crate) struct Usage {
    pub(crate) prompt_tokens: u32,
    pub(crate) completion_tokens: u32,
    pub(crate) total_tokens: u32,
    pub(crate) prompt_tokens_details: PromptTokensDetails,
    pub(crate) completion_tokens_details: CompletionTokensDetails,
}

impl Usage {
    pub(crate) fn new(prompt: u32, completion: u32) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt.saturating_add(completion),
            prompt_tokens_details: PromptTokensDetails::default(),
            completion_tokens_details: CompletionTokensDetails::default(),
        }
    }
}

/// [`Usage`] as a JSON value, for frames built with `serde_json::json!`.
pub(crate) fn usage_json(prompt: u32, completion: u32) -> serde_json::Value {
    let u = Usage::new(prompt, completion);
    let mut prompt_details = serde_json::Map::new();
    if let Some(n) = u.prompt_tokens_details.cached_tokens {
        prompt_details.insert("cached_tokens".into(), n.into());
    }
    let mut completion_details = serde_json::Map::new();
    if let Some(n) = u.completion_tokens_details.reasoning_tokens {
        completion_details.insert("reasoning_tokens".into(), n.into());
    }
    serde_json::json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
        "prompt_tokens_details": prompt_details,
        "completion_tokens_details": completion_details,
    })
}

/// What a stream's `usage` key carries, from `stream_options`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct UsageMode {
    /// `include_usage`: a closing `choices: []` chunk with the usage, and
    /// `usage: null` on every chunk before it.
    pub(crate) include: bool,
    /// `continuous_usage_stats` (with `include_usage` only): every chunk
    /// carries the usage so far instead of null.
    pub(crate) continuous: bool,
}

impl UsageMode {
    pub(crate) fn new(include_usage: bool, continuous_usage_stats: bool) -> Self {
        Self {
            include: include_usage,
            continuous: include_usage && continuous_usage_stats,
        }
    }
}

/// What every chunk of one stream repeats, plus whether the assistant role
/// has been sent yet. One per request: `created` is taken once, so every
/// chunk of a stream carries the same value.
pub(crate) struct StreamMeta {
    pub(crate) id: String,
    pub(crate) model: String,
    pub(crate) created: i64,
    pub(crate) fingerprint: String,
    pub(crate) usage: UsageMode,
    /// Set by the first chat chunk, which alone carries `role`.
    role_sent: bool,
}

/// Token counts at the time a chunk is built: the latest prompt count (0
/// until one is known) and the stream's running completion count.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Counts {
    pub(crate) prompt: u32,
    pub(crate) completion: u32,
}

impl StreamMeta {
    pub(crate) fn new(
        id: String,
        model: String,
        created: i64,
        fingerprint: String,
        usage: UsageMode,
    ) -> Self {
        Self {
            id,
            model,
            created,
            fingerprint,
            usage,
            role_sent: false,
        }
    }

    /// One `chat.completion.chunk` frame (`data: …\n\n`) with one choice.
    /// The first chat chunk of the stream gets `role: assistant`, and
    /// `content: ""` when it carries neither content nor a call, as
    /// OpenAI's first chunk does; no later one carries `role`.
    pub(crate) fn chat_chunk(
        &mut self,
        delta: serde_json::Map<String, serde_json::Value>,
        finish: Option<&str>,
        logprobs: serde_json::Value,
        counts: Counts,
    ) -> String {
        let mut delta = delta;
        if !self.role_sent {
            self.role_sent = true;
            let says_something = ["content", "tool_calls", "function_call"]
                .iter()
                .any(|k| delta.contains_key(*k));
            if !says_something {
                delta.insert("content".into(), "".into());
            }
            delta.insert("role".into(), "assistant".into());
        }
        let choice = serde_json::json!({
            "index": 0,
            "delta": delta,
            "logprobs": logprobs,
            "finish_reason": finish,
        });
        self.frame(
            "chat.completion.chunk",
            vec![choice],
            self.chunk_usage(counts),
        )
    }

    /// One `text_completion` chunk frame with one choice.
    pub(crate) fn completion_chunk(
        &self,
        text: &str,
        finish: Option<&str>,
        logprobs: serde_json::Value,
        counts: Counts,
    ) -> String {
        let choice = serde_json::json!({
            "index": 0,
            "text": text,
            "logprobs": logprobs,
            "finish_reason": finish,
        });
        self.frame("text_completion", vec![choice], self.chunk_usage(counts))
    }

    /// The `include_usage` chunk: no choices, the final usage.
    pub(crate) fn usage_chunk(&self, object: &str, counts: Counts) -> String {
        self.frame(
            object,
            Vec::new(),
            Some(usage_json(counts.prompt, counts.completion)),
        )
    }

    /// The `usage` value of a chunk with choices: absent without
    /// `include_usage`, else null, or the usage so far in continuous mode.
    fn chunk_usage(&self, counts: Counts) -> Option<serde_json::Value> {
        if self.usage.continuous {
            Some(usage_json(counts.prompt, counts.completion))
        } else if self.usage.include {
            Some(serde_json::Value::Null)
        } else {
            None
        }
    }

    fn frame(
        &self,
        object: &str,
        choices: Vec<serde_json::Value>,
        usage: Option<serde_json::Value>,
    ) -> String {
        let mut payload = serde_json::json!({
            "id": self.id,
            "object": object,
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.fingerprint,
            "choices": choices,
        });
        if let (Some(usage), Some(obj)) = (usage, payload.as_object_mut()) {
            obj.insert("usage".into(), usage);
        }
        format!("data: {payload}\n\n")
    }
}

/// A chat delta with just `content`.
pub(crate) fn content_delta(content: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut delta = serde_json::Map::new();
    delta.insert("content".into(), content.into());
    delta
}

/// A logprob as sent: OpenAI writes `-9999.0` for a probability too small
/// to represent, and serde_json would write a non-finite value as `null`,
/// which the schema rejects. A finite f32 goes through its own shortest
/// decimal form, so `-0.1` is not sent as `-0.10000000149011612`.
fn wire_logprob(logprob: f32) -> f64 {
    if logprob.is_finite() {
        logprob.to_string().parse().unwrap_or(f64::from(logprob))
    } else {
        -9999.0
    }
}

/// One entry of OpenAI's `logprobs.content`: the token, its logprob, its
/// UTF-8 bytes and the alternatives in the same shape. The engine's token
/// ids stay off the wire.
pub(crate) fn spec_logprob(lp: &cascadia_types::TokenLogprobs) -> serde_json::Value {
    let top: Vec<serde_json::Value> = lp
        .top_logprobs
        .iter()
        .map(|t| {
            serde_json::json!({
                "token": t.token,
                "logprob": wire_logprob(t.logprob),
                "bytes": t.token.as_bytes(),
            })
        })
        .collect();
    serde_json::json!({
        "token": lp.token,
        "logprob": wire_logprob(lp.logprob),
        "bytes": lp.token.as_bytes(),
        "top_logprobs": top,
    })
}

/// OpenAI's choice `logprobs` object over `items`.
pub(crate) fn spec_logprobs(items: &[cascadia_types::TokenLogprobs]) -> serde_json::Value {
    let content: Vec<serde_json::Value> = items.iter().map(spec_logprob).collect();
    serde_json::json!({ "content": content, "refusal": null })
}

/// The logprob items of a stream whose text has not gone out yet. Text can
/// be held back (by the `stop` matcher, or while it may start a tool call)
/// and released later in other pieces, so each item is kept with the byte
/// offset its token's text starts at in the stream's text, and goes out with
/// the frame that releases that byte. Items of text that is never released
/// (cut by a `stop` match, or tool-call markup) are dropped by
/// [`finish`](Self::finish).
pub(crate) struct LogprobQueue {
    /// How many alternatives each item carries when the request asked for
    /// logprobs; `None` otherwise, and every frame says `null`. The engine
    /// may be asked for more than this, so its lists are cut to it.
    top: Option<usize>,
    items: std::collections::VecDeque<(usize, cascadia_types::TokenLogprobs)>,
    /// Bytes of text the engine has produced so far.
    pushed: usize,
    /// Bytes of that text released so far.
    released: usize,
}

impl LogprobQueue {
    /// `top`: the alternatives asked for per item, `None` for no logprobs.
    pub(crate) fn new(top: Option<u32>) -> Self {
        Self {
            top: top.map(|n| n as usize),
            items: std::collections::VecDeque::new(),
            pushed: 0,
            released: 0,
        }
    }

    /// Records the next `len` bytes of text and, when the engine gave one,
    /// their item.
    pub(crate) fn push(&mut self, len: usize, item: Option<&cascadia_types::TokenLogprobs>) {
        if let (Some(top), Some(item)) = (self.top, item) {
            let mut item = item.clone();
            item.top_logprobs.truncate(top);
            self.items.push_back((self.pushed, item));
        }
        self.pushed += len;
    }

    /// The next `len` bytes go out: the `logprobs` value of their frame, the
    /// items whose text starts in them, or `null` when there are none. A
    /// frame with no text carries `null`.
    pub(crate) fn release(&mut self, len: usize) -> serde_json::Value {
        if len == 0 {
            return serde_json::Value::Null;
        }
        self.released += len;
        let mut out = Vec::new();
        while let Some((start, _)) = self.items.front() {
            if *start >= self.released {
                break;
            }
            if let Some((_, item)) = self.items.pop_front() {
                out.push(item);
            }
        }
        if out.is_empty() {
            serde_json::Value::Null
        } else {
            spec_logprobs(&out)
        }
    }

    /// Drops the items held: their text will never be released.
    pub(crate) fn discard(&mut self) {
        self.items.clear();
    }

    /// Items waiting for their text, for tests.
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        self.items.len()
    }

    /// The last `len` bytes go out; nothing after them ever will, so the
    /// items still held are dropped.
    pub(crate) fn finish(&mut self, len: usize) -> serde_json::Value {
        let out = self.release(len);
        self.items.clear();
        out
    }
}

/// The OpenAI error object an error event carries, as the caller classified
/// the failure: a prompt over the window is a client fault
/// (`invalid_request_error`, the prompt's `param`, `context_length_exceeded`),
/// an engine or pipeline failure is [`StreamError::server`].
pub(crate) struct StreamError<'a> {
    /// The error object's `type`.
    pub(crate) kind: &'a str,
    pub(crate) message: &'a str,
    pub(crate) param: Option<&'a str>,
    pub(crate) code: Option<&'a str>,
}

impl<'a> StreamError<'a> {
    /// A failure of the server or its engine: `server_error`, with no
    /// `param` and no `code`, as a 5xx error body has.
    pub(crate) fn server(message: &'a str) -> Self {
        Self {
            kind: "server_error",
            message,
            param: None,
            code: None,
        }
    }
}

/// The SSE event that ends a stream which failed after its 200 headers went
/// out: the OpenAI error object under `error`, so openai-python and -node
/// raise `APIError` on it. `id` and `object: "error"` stay for clients that
/// key on them; no usage chunk follows it, only `[DONE]`.
pub(crate) fn error_frame(id: &str, err: &StreamError<'_>) -> String {
    let payload = serde_json::json!({
        "id": id,
        "object": "error",
        "error": {
            "message": err.message,
            "type": err.kind,
            "param": err.param,
            "code": err.code,
        },
    });
    format!("data: {payload}\n\n")
}

/// The message of the error event sent when an engine stream ends with no
/// final chunk and nobody cancelled it.
pub(crate) const UNEXPLAINED_END: &str = "engine stream ended without a final chunk";

/// The task ids of the streams being sent, each with whether
/// `POST /v1/cancel` named it. A cancelled task's engine stream ends with no
/// final chunk, as one whose engine went away does; this tells them apart.
/// Entries live only as long as their stream ([`LiveStream`]), and a cancel
/// flags only an id already here, so the map never outgrows the open streams.
#[derive(Clone, Default)]
pub(crate) struct LiveStreams(Arc<Mutex<HashMap<String, bool>>>);

impl LiveStreams {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, bool>> {
        // The map holds plain flags: a panic elsewhere cannot leave it torn.
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Registers the stream of `id` until the returned guard drops.
    pub(crate) fn register(&self, id: &str) -> LiveStream {
        self.lock().insert(id.to_owned(), false);
        LiveStream {
            streams: self.clone(),
            id: id.to_owned(),
        }
    }

    /// Flags the stream of `id` as cancelled; an id with no open stream adds
    /// nothing.
    pub(crate) fn cancel(&self, id: &str) {
        if let Some(flag) = self.lock().get_mut(id) {
            *flag = true;
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().len()
    }
}

/// One open stream's entry in [`LiveStreams`], removed on drop: when the
/// stream body ends or is dropped on a client disconnect.
pub(crate) struct LiveStream {
    streams: LiveStreams,
    id: String,
}

impl LiveStream {
    /// `POST /v1/cancel` named this stream's task.
    pub(crate) fn cancelled(&self) -> bool {
        self.streams.lock().get(&self.id).copied().unwrap_or(false)
    }
}

impl Drop for LiveStream {
    fn drop(&mut self) {
        self.streams.lock().remove(&self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_stable_and_spec_shaped() {
        let a = fingerprint("0.2.7", "ov-genai", "Qwen3-8B");
        assert_eq!(a, fingerprint("0.2.7", "ov-genai", "Qwen3-8B"));
        // Pins the hash: FNV-1a 64 over "cascadia/0.2.7/ov-genai/Qwen3-8B".
        assert_eq!(a, "fp_9a87287c6f");
        assert_eq!(a.len(), 13);
        assert!(a.starts_with("fp_"));
        assert!(a[3..]
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)));
        assert_ne!(a, fingerprint("0.2.7", "sycl-llama", "Qwen3-8B"));
        assert_ne!(a, fingerprint("0.2.7", "ov-genai", "Qwen3-4B"));
        assert_ne!(a, fingerprint("0.2.8", "ov-genai", "Qwen3-8B"));
        let empty = fingerprint("", "", "");
        assert_eq!(empty.len(), 13);
    }

    #[test]
    fn usage_json_leaves_uncounted_details_empty() {
        assert_eq!(
            usage_json(5, 3),
            serde_json::json!({
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
                "prompt_tokens_details": {},
                "completion_tokens_details": {},
            })
        );
    }

    #[test]
    fn usage_json_matches_the_serialized_usage() {
        for (p, c) in [(0, 0), (5, 3), (u32::MAX, 1)] {
            assert_eq!(
                usage_json(p, c),
                serde_json::to_value(Usage::new(p, c)).unwrap()
            );
        }
    }
}

#[cfg(test)]
mod chunk_tests {
    use super::*;

    fn meta(include: bool, continuous: bool) -> StreamMeta {
        StreamMeta::new(
            "chatcmpl-1".into(),
            "m".into(),
            42,
            "fp_0123456789".into(),
            UsageMode::new(include, continuous),
        )
    }

    fn parse(frame: &str) -> serde_json::Value {
        let data = frame
            .strip_prefix("data: ")
            .and_then(|f| f.strip_suffix("\n\n"))
            .unwrap();
        serde_json::from_str(data).unwrap()
    }

    #[test]
    fn first_chat_chunk_alone_carries_the_role() {
        let mut m = meta(false, false);
        let counts = Counts::default();
        let first = parse(&m.chat_chunk(
            serde_json::Map::new(),
            None,
            serde_json::Value::Null,
            counts,
        ));
        assert_eq!(
            first["choices"][0]["delta"],
            serde_json::json!({"role": "assistant", "content": ""})
        );
        let next = parse(&m.chat_chunk(
            content_delta("hi"),
            Some("stop"),
            serde_json::Value::Null,
            counts,
        ));
        assert_eq!(
            next["choices"][0]["delta"],
            serde_json::json!({"content": "hi"})
        );
        assert_eq!(next["choices"][0]["finish_reason"], "stop");
        assert!(next["choices"][0]["logprobs"].is_null());
        assert_eq!(next["created"], 42);
        assert_eq!(next["system_fingerprint"], "fp_0123456789");
        assert!(next.get("usage").is_none());
    }

    #[test]
    fn continuous_usage_needs_include_usage() {
        assert_eq!(UsageMode::new(false, true), UsageMode::default());
        let counts = Counts {
            prompt: 2,
            completion: 3,
        };
        let null =
            parse(&meta(true, false).completion_chunk("x", None, serde_json::Value::Null, counts));
        assert!(null["usage"].is_null() && null.get("usage").is_some());
        let running =
            parse(&meta(true, true).completion_chunk("x", None, serde_json::Value::Null, counts));
        assert_eq!(running["usage"], usage_json(2, 3));
        let tail = parse(&meta(true, false).usage_chunk("text_completion", counts));
        assert_eq!(tail["choices"], serde_json::json!([]));
        assert_eq!(tail["usage"], usage_json(2, 3));
    }
}

#[cfg(test)]
mod logprob_tests {
    use super::*;
    use cascadia_types::{TokenLogprobs, TopLogprob};

    fn item(token: &str, logprob: f32) -> TokenLogprobs {
        TokenLogprobs {
            token: token.into(),
            token_id: 7,
            logprob,
            top_logprobs: vec![
                TopLogprob {
                    token: token.into(),
                    token_id: 7,
                    logprob,
                },
                TopLogprob {
                    token: "zz".into(),
                    token_id: 8,
                    logprob: f32::NEG_INFINITY,
                },
            ],
        }
    }

    #[test]
    fn spec_logprob_has_bytes_and_no_token_id() {
        assert_eq!(
            spec_logprob(&item("é", -0.5)),
            serde_json::json!({
                "token": "é", "logprob": -0.5, "bytes": [195, 169],
                "top_logprobs": [
                    {"token": "é", "logprob": -0.5, "bytes": [195, 169]},
                    {"token": "zz", "logprob": -9999.0, "bytes": [122, 122]},
                ],
            })
        );
    }

    #[test]
    fn non_finite_logprobs_become_the_floor_and_f32_prints_short() {
        for bad in [f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            assert_eq!(spec_logprob(&item("a", bad))["logprob"], -9999.0);
        }
        // An f32 widened to f64 prints as -0.10000000149011612; the wire
        // carries the f32's own shortest form.
        assert_eq!(
            spec_logprob(&item("a", -0.1))["logprob"].to_string(),
            "-0.1"
        );
    }

    #[test]
    fn spec_logprobs_wraps_items_with_a_null_refusal() {
        let v = spec_logprobs(&[item("a", -1.0), item("b", -2.0)]);
        assert_eq!(v["refusal"], serde_json::Value::Null);
        assert!(v.get("refusal").is_some());
        let tokens: Vec<&str> = v["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["token"].as_str().unwrap())
            .collect();
        assert_eq!(tokens, ["a", "b"]);
    }

    fn tokens(v: &serde_json::Value) -> Vec<String> {
        v["content"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .map(|i| i["token"].as_str().unwrap().to_owned())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn queue_sends_each_item_with_the_frame_that_releases_its_first_byte() {
        let mut q = LogprobQueue::new(Some(20));
        q.push(3, Some(&item("ab ", -1.0)));
        q.push(2, Some(&item(" S", -1.0)));
        // " S": only " " released, the "S" held: the item goes out with it.
        assert_eq!(tokens(&q.release(4)), ["ab ", " S"]);
        q.push(1, Some(&item("x", -1.0)));
        assert_eq!(tokens(&q.release(2)), ["x"]);
        // Nothing released: nothing sent, the item waits.
        q.push(2, Some(&item("ST", -1.0)));
        assert!(q.release(0).is_null());
        // The text from here on was cut: its item is dropped.
        q.push(2, Some(&item("OP", -1.0)));
        assert!(q.finish(0).is_null());
        assert!(q.release(4).is_null());
    }

    #[test]
    fn queue_skips_chunks_without_items_and_is_inert_when_not_asked() {
        let mut q = LogprobQueue::new(Some(20));
        q.push(2, None);
        q.push(1, Some(&item("c", -1.0)));
        assert!(q.release(2).is_null());
        assert_eq!(tokens(&q.finish(1)), ["c"]);
        let mut off = LogprobQueue::new(None);
        off.push(1, Some(&item("c", -1.0)));
        assert!(off.release(1).is_null());
    }
    #[test]
    fn error_frame_carries_the_classification_it_is_given() {
        let parse = |frame: &str| -> serde_json::Value {
            let json = frame.strip_prefix("data: ").unwrap();
            assert!(json.ends_with("\n\n"), "{frame:?}");
            serde_json::from_str(json.trim_end()).unwrap()
        };
        assert_eq!(
            parse(&error_frame("c1", &StreamError::server("boom"))),
            serde_json::json!({"id": "c1", "object": "error", "error": {
                "message": "boom", "type": "server_error", "param": null, "code": null}})
        );
        let over = error_frame(
            "c2",
            &StreamError {
                kind: "invalid_request_error",
                message: "too long",
                param: Some("messages"),
                code: Some("context_length_exceeded"),
            },
        );
        assert_eq!(
            parse(&over),
            serde_json::json!({"id": "c2", "object": "error", "error": {
                "message": "too long", "type": "invalid_request_error",
                "param": "messages", "code": "context_length_exceeded"}})
        );
    }
}
