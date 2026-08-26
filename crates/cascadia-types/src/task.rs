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
}

fn default_max_tokens() -> u32 {
    256
}

impl GenerationTask {
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
