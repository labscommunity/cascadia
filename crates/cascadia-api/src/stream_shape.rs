//! Shared pieces of the OpenAI response wire shape: the `system_fingerprint`
//! value and the `usage` object.

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
