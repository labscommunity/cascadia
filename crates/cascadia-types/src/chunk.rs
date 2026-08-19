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

/// Why a generation ended. Set on the FINAL chunk by engines that can tell
/// the difference; maps to OpenAI's `finish_reason`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// EOS token emitted or a stop sequence matched.
    Stop,
    /// `max_tokens` reached before a natural stop.
    Length,
    /// Client disconnected or an explicit cancel arrived.
    Cancelled,
}

impl FinishReason {
    /// OpenAI wire string. `Cancelled` has no OpenAI equivalent — report it
    /// as `stop` per the streaming spec (a server SHOULD end cleanly when the
    /// client goes away); surface `cancelled` only in metrics.
    pub fn as_openai_str(self) -> &'static str {
        match self {
            FinishReason::Stop | FinishReason::Cancelled => "stop",
            FinishReason::Length => "length",
        }
    }
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
    /// Set when a task terminates by FAILURE rather than completion. The
    /// `step()` API returns `Vec<(TaskId, Chunk)>` — not a `Result` — so a
    /// final chunk is the only "task is done" signal an engine can emit.
    /// Without this, a failed task is indistinguishable from a successful
    /// empty completion: consumers see `is_final && text == ""` and answer
    /// HTTP 200. An error chunk still sets `is_final` (old consumers end the
    /// stream as before, no regression); new consumers check this and fail
    /// loud (5xx). `None` on every normal token/final chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Why generation ended. Set on the FINAL chunk by engines that can tell
    /// `length` (hit max_tokens) from `stop` (EOS / stop sequence). `None` on
    /// non-final chunks and on engines that don't distinguish — the API then
    /// falls back to `"stop"` (the historical behavior).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
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
            error: None,
            finish_reason: None,
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
            error: None,
            finish_reason: None,
        }
    }

    /// A final chunk that marks the task FAILED (not completed). Carries the
    /// failure reason so the API layer can return a 5xx instead of a 200 with
    /// empty content. `is_final` is set so legacy consumers still terminate.
    pub fn error(task_id: impl Into<TaskId>, reason: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            token_id: 0,
            text: String::new(),
            is_final: true,
            logprobs: None,
            n_tokens: None,
            prompt_tokens: None,
            error: Some(reason.into()),
            finish_reason: None,
        }
    }

    pub fn with_prompt_tokens(mut self, n: u32) -> Self {
        self.prompt_tokens = Some(n);
        self
    }

    /// Tag the final chunk with why generation stopped (length vs stop).
    pub fn with_finish_reason(mut self, reason: FinishReason) -> Self {
        self.finish_reason = Some(reason);
        self
    }

    /// Record how many model tokens this chunk's `text` condenses. Engines
    /// that deliver a whole response on one chunk (ov-genai) set this so the
    /// API's `usage.completion_tokens` reflects the real count instead of
    /// the "one token per chunk" fallback. See `Chunk::n_tokens` docs.
    pub fn with_n_tokens(mut self, n: u32) -> Self {
        self.n_tokens = Some(n);
        self
    }

    /// Tokens this chunk contributes to a cumulative count — the ONE
    /// convention every consumer (API `usage`, dashboard counters,
    /// Prometheus metrics) must share. The engine's `n_tokens` is
    /// authoritative when set (spec-decode reports 1..=K+1; ov-genai sets
    /// it on its single final chunk — issue #55); otherwise one token per
    /// non-empty chunk, so the empty final markers mock/runtime engines
    /// emit contribute 0 rather than a phantom token.
    pub fn token_count(&self) -> u32 {
        self.n_tokens
            .unwrap_or(if self.text.is_empty() { 0 } else { 1 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every openvino/sparse-moe engine reports token usage over this same
    /// seam (`prompt_tokens` + `token_count()`), not a per-engine field. Two
    /// shapes are in use: (a) one token per streamed `Chunk`, finished by an
    /// empty marker carrying `prompt_tokens` (dist_spec, gemma4, ov-runtime's
    /// static path, most sparse-moe paths), and (b) a single final chunk
    /// carrying the whole response with an explicit `n_tokens` (ov-genai,
    /// spec-decode's packed path, PipelineEngine). Pins that a consumer
    /// summing `token_count()` across the stream recovers the real prompt
    /// and completion counts under both shapes.
    #[test]
    fn per_token_stream_reports_fed_prompt_len_and_emitted_count() {
        let fed_prompt_len = 5u32;
        let emitted = ["The", " quick", " fox"];
        let mut chunks: Vec<Chunk> = emitted
            .iter()
            .enumerate()
            .map(|(i, t)| Chunk::token("t1", i as i64, *t))
            .collect();
        chunks.push(Chunk::final_marker("t1", "").with_prompt_tokens(fed_prompt_len));

        let completion_tokens: u32 = chunks.iter().map(Chunk::token_count).sum();
        let prompt_tokens = chunks
            .iter()
            .find(|c| c.is_final)
            .and_then(|c| c.prompt_tokens);

        assert_eq!(completion_tokens, emitted.len() as u32);
        assert_eq!(prompt_tokens, Some(fed_prompt_len));
    }

    #[test]
    fn single_final_chunk_reports_fed_prompt_len_and_emitted_count() {
        let fed_prompt_len = 12u32;
        let generated_tokens = 7u32;
        let chunk = Chunk::final_marker("t1", "the quick fox jumps")
            .with_n_tokens(generated_tokens)
            .with_prompt_tokens(fed_prompt_len);

        assert_eq!(chunk.token_count(), generated_tokens);
        assert_eq!(chunk.prompt_tokens, Some(fed_prompt_len));
    }

    /// An engine that can't tell the prompt length (or hasn't reached the
    /// final chunk yet) leaves `prompt_tokens` at `None` — the API falls back
    /// to 0 rather than fabricating a count.
    #[test]
    fn prompt_tokens_defaults_to_none() {
        assert_eq!(Chunk::token("t1", 0, "hi").prompt_tokens, None);
        assert_eq!(Chunk::final_marker("t1", "").prompt_tokens, None);
    }
}
