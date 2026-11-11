//! Shared pieces of the OpenAI response wire shape: the `system_fingerprint`
//! value, the `usage` object and the streamed chunk frames.

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
