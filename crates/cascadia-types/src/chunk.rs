use serde::{Deserialize, Serialize};

use crate::TaskId;

/// One (token, logprob) pair in a top-k logprobs response.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TopLogprob {
    pub token: String,
    pub token_id: i64,
    pub logprob: f32,
}

/// Logprob payload for a single emitted token. OpenAI-compatible.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TokenLogprobs {
    pub token: String,
    pub token_id: i64,
    pub logprob: f32,
    #[serde(default)]
    pub top_logprobs: Vec<TopLogprob>,
}

/// One token (or final marker) produced by an engine for a task.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Chunk {
    pub task_id: TaskId,
    pub token_id: i64,
    pub text: String,
    #[serde(default)]
    pub is_final: bool,
    /// Filled when the engine supports it AND the task asked for it.
    #[serde(default)]
    pub logprobs: Option<TokenLogprobs>,
    /// Number of model tokens condensed into this chunk's `text`. Most
    /// engines emit one token per chunk so this is None (consumers
    /// treat None as 1). Spec-decode emits 1..=K+1 tokens per chunk
    /// (one full spec round) and sets this so downstream tok/s metrics
    /// don't undercount.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_tokens: Option<u32>,
    /// Prompt token count, set on the FINAL chunk by engines that know
    /// it (the API's usage block reads it; None = engine can't tell).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
}

impl Chunk {
    pub fn token(task_id: impl Into<TaskId>, token_id: i64, text: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            token_id,
            text: text.into(),
            is_final: false,
            logprobs: None,
            n_tokens: None,
            prompt_tokens: None,
        }
    }

    pub fn final_marker(task_id: impl Into<TaskId>, text: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            token_id: 0,
            text: text.into(),
            is_final: true,
            logprobs: None,
            n_tokens: None,
            prompt_tokens: None,
        }
    }

    pub fn with_prompt_tokens(mut self, n: u32) -> Self {
        self.prompt_tokens = Some(n);
        self
    }
}
