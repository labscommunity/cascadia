use serde::{Deserialize, Serialize};

/// Stable per-request identifier propagated end-to-end.
pub type TaskId = String;

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
    /// When true, the engine should expose chain-of-thought / reasoning
    /// tokens if the model supports it (DeepSeek V3.1, Qwen3, GLM-4.7,
    /// etc.).
    #[serde(default)]
    pub enable_thinking: bool,
    /// When true, the engine may execute code from the model repository.
    /// Treat as a security boundary — opt in only.
    #[serde(default)]
    pub trust_remote_code: bool,
    /// Guided-decoding constraint; enforced by engines that support it, else ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<GrammarSpec>,
}

/// Backend-neutral guided-decoding constraint. `None` = unconstrained; engines
/// that don't enforce it ignore it (like `logprobs`). `body` is a String (never
/// serde_json::Value — ADR-001 bincode).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GrammarSpec {
    pub kind: GrammarKind,
    pub body: String,
}

/// Only `JsonSchema` in slice 1; `Regex`/`Lark` are additive future variants.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GrammarKind {
    JsonSchema,
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
            enable_thinking: false,
            trust_remote_code: false,
            grammar: None,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_grammar(mut self, grammar: GrammarSpec) -> Self {
        self.grammar = Some(grammar);
        self
    }
}
