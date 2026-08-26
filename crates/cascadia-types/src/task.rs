use serde::{Deserialize, Serialize};

/// Stable per-request identifier propagated end-to-end.
pub type TaskId = String;

/// OpenAI-compatible sampling knobs, bundled so they thread through the API,
/// `GenerationTask`, and every engine as one unit. Engines that don't support
/// a given field ignore it. `temperature` and `max_tokens` stay on
/// `GenerationTask` itself (they predate this struct and have many call sites).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SamplingParams {
    /// Nucleus sampling. 1.0 (or 0.0) disables. Maps to OpenAI `top_p`.
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Top-k truncation. 0 disables. Maps to OpenAI `top_k` (non-standard but
    /// widely supported).
    #[serde(default)]
    pub top_k: u32,
    /// PRNG seed. None → system entropy. Maps to OpenAI `seed`.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Penalize tokens by how often they already appeared (count-scaled).
    /// 0.0 disables. Maps to OpenAI `frequency_penalty` (range -2.0..=2.0).
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Penalize tokens that appeared at all (presence, not count). 0.0
    /// disables. Maps to OpenAI `presence_penalty` (range -2.0..=2.0).
    #[serde(default)]
    pub presence_penalty: f32,
    /// Stop sequences. Generation halts (and the stop text is trimmed) when
    /// any of these is produced. Maps to OpenAI `stop` (string or array).
    #[serde(default)]
    pub stop: Vec<String>,
}

fn default_top_p() -> f32 {
    1.0
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            top_p: 1.0,
            top_k: 0,
            seed: None,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop: Vec::new(),
        }
    }
}

/// One generation request, end-to-end.
///
/// Engines that don't support a given option silently ignore it — the API
/// layer is responsible for surfacing degradation to the caller.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GenerationTask {
    pub task_id: TaskId,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: f32,
    /// Number of top logprobs to return per token. 0 = disabled.
    #[serde(default)]
    pub logprobs: u32,
    /// OpenAI-compatible sampling knobs (top_p / top_k / seed / penalties /
    /// stop). `temperature` above and `max_tokens` stay separate for
    /// historical reasons.
    #[serde(default)]
    pub sampling: SamplingParams,
    /// When true, the engine should expose chain-of-thought / reasoning
    /// tokens if the model supports it (DeepSeek V3.1, Qwen3, GLM-4.7,
    /// etc.).
    #[serde(default)]
    pub enable_thinking: bool,
    /// When true, the engine may execute code from the model repository.
    /// Treat as a security boundary — opt in only.
    #[serde(default)]
    pub trust_remote_code: bool,
    /// Issue-34 H.1b: the tenant this turn belongs to. The engine namespaces its KV cache by it at
    /// BOTH ends — `capture` tags the entry, `take_warm` only resumes from an entry with the same
    /// value — so one tenant can never warm-resume off another's prefix.
    ///
    /// Defaults to `""` (`kv_coordination::LOCAL_NS`), which is exactly today's single-namespace
    /// behaviour, so plumbing this through is inert until the plane starts asserting a real tenant.
    /// Landing the plane's assertion while captures are still tagged `""` would send cross-node warm
    /// pull cold for that tenant — fail-closed, but a silent performance cliff (design §12.10.0), so
    /// the two must flip together.
    #[serde(default)]
    pub tenant: String,
    /// Option B resumable generation: already-emitted assistant token-ids to
    /// force-prefix. An engine that supports resume APPENDS these to the
    /// rendered prompt-ids and accounts prefix + new against `max_tokens`
    /// (by pre-seeding its accumulator or by subtracting — see
    /// `resume_generated_seed`); one that cannot honor a forced prefix
    /// declines with a `resume_unsupported:`-prefixed error. Either way the
    /// ids are NEVER re-emitted, and never silently ignored.
    #[serde(default)]
    pub resume_token_ids: Option<Vec<i32>>,
}

fn default_max_tokens() -> u32 {
    256
}

/// Append the caller's already-emitted assistant token-ids to the engine's
/// rendered prompt-ids. No-op when not resuming. Concatenation, NOT replacement —
/// the rendered prompt is KEPT as context.
pub fn append_resume_ids(prompt_ids: &mut Vec<i64>, resume: Option<&[i32]>) {
    if let Some(r) = resume {
        prompt_ids.extend(r.iter().map(|&t| t as i64));
    }
}

/// Tokens to pre-seed the engine's `generated` accumulator with on a resumed
/// turn: makes `generated.len() >= max_tokens` bound total = prefix + new, and
/// makes tail tokens decode WITH the prefix as context. Empty for a normal turn.
pub fn resume_generated_seed(resume: Option<&[i32]>) -> Vec<i64> {
    resume
        .map(|r| r.iter().map(|&t| t as i64).collect())
        .unwrap_or_default()
}

/// Peer-supplied resume ids are indices into the embedding table, so they must
/// be validated BEFORE any engine feeds them to a native runtime: a negative or
/// out-of-vocab id is an out-of-range gather in OpenVINO — crash/UB, not a typed
/// error. `vocab_size: None` skips the upper bound for engines that cannot
/// cheaply know it; the sign check always applies.
pub fn validate_resume_ids(ids: &[i32], vocab_size: Option<u32>) -> Result<(), String> {
    for (i, &id) in ids.iter().enumerate() {
        if id < 0 {
            return Err(format!("resume_token_ids[{i}] = {id} is negative"));
        }
        if let Some(v) = vocab_size {
            if id as u32 >= v {
                return Err(format!("resume_token_ids[{i}] = {id} >= vocab_size {v}"));
            }
        }
    }
    Ok(())
}

impl GenerationTask {
    /// The normalized resume prefix: `Some(&ids)` only when non-empty. Engines
    /// MUST branch on this, never on `resume_token_ids.is_some()` — a wire-legal
    /// `Some([])` is "not resuming", and predicate drift here already produced
    /// one engine forcing greedy decode on a non-resumed turn.
    pub fn resume_ids(&self) -> Option<&[i32]> {
        self.resume_token_ids.as_deref().filter(|r| !r.is_empty())
    }

    pub fn new(task_id: impl Into<TaskId>, prompt: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            prompt: prompt.into(),
            max_tokens: default_max_tokens(),
            temperature: 0.0,
            logprobs: 0,
            sampling: SamplingParams::default(),
            enable_thinking: false,
            trust_remote_code: false,
            tenant: String::new(), // LOCAL_NS — see the field doc
            resume_token_ids: None,
        }
    }

    /// Issue-34 H.1b: set the tenant this turn's KV is namespaced under (both capture and resume).
    #[must_use]
    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = tenant.into();
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_sampling(mut self, sampling: SamplingParams) -> Self {
        self.sampling = sampling;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resume_token_ids_defaults_none_and_is_additive() {
        let t = GenerationTask {
            task_id: "t1".to_string(),
            prompt: "hello".to_string(),
            max_tokens: default_max_tokens(),
            temperature: 0.0,
            logprobs: 0,
            sampling: SamplingParams::default(),
            enable_thinking: false,
            trust_remote_code: false,
            tenant: String::new(),
            resume_token_ids: None,
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: GenerationTask = serde_json::from_str(&json).unwrap();
        assert_eq!(back.resume_token_ids, None);

        // Additivity: a payload encoded WITHOUT the field still decodes,
        // defaulting to None.
        let without_field = serde_json::json!({
            "task_id": "t1",
            "prompt": "hello",
        });
        let back: GenerationTask = serde_json::from_value(without_field).unwrap();
        assert_eq!(back.resume_token_ids, None);
    }

    #[test]
    fn append_resume_ids_concatenates_and_is_noop_when_absent() {
        let mut ids = vec![1i64, 2, 3];
        append_resume_ids(&mut ids, None);
        assert_eq!(ids, vec![1, 2, 3]);
        append_resume_ids(&mut ids, Some(&[11, 12]));
        assert_eq!(ids, vec![1, 2, 3, 11, 12]);
    }

    #[test]
    fn resume_generated_seed_maps_and_defaults_empty() {
        assert_eq!(resume_generated_seed(None), Vec::<i64>::new());
        assert_eq!(resume_generated_seed(Some(&[5, 6, 7])), vec![5i64, 6, 7]);
    }
    #[test]
    fn resume_ids_normalizes_empty_to_none() {
        let mut t = GenerationTask::new("t", "p");
        assert_eq!(t.resume_ids(), None);
        t.resume_token_ids = Some(vec![]);
        assert_eq!(t.resume_ids(), None, "wire-legal Some([]) is NOT resuming");
        t.resume_token_ids = Some(vec![7]);
        assert_eq!(t.resume_ids(), Some(&[7][..]));
    }

    #[test]
    fn validate_resume_ids_bounds() {
        assert!(validate_resume_ids(&[0, 1, 2], Some(3)).is_ok());
        assert!(
            validate_resume_ids(&[3], Some(3)).is_err(),
            ">= vocab rejected"
        );
        assert!(
            validate_resume_ids(&[-1], None).is_err(),
            "negative always rejected"
        );
        assert!(
            validate_resume_ids(&[i32::MAX], None).is_ok(),
            "no vocab => only sign-checked"
        );
    }
}
