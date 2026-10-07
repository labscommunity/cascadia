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
    /// Number of known model tokens condensed into this chunk. Most engines
    /// emit one token per chunk and leave this None (non-empty text counts
    /// as one). Spec-decode sets its exact batch count. A transport fragment
    /// with unknown token boundaries sets Some(0) and reports authoritative
    /// `completion_tokens` on the final chunk instead of guessing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_tokens: Option<u32>,
    /// Prompt token count, set on the FINAL chunk by engines that know
    /// it (the API's usage block reads it; None = engine can't tell).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    /// Authoritative total model tokens generated for this task, including
    /// hidden reasoning and tool calls. Only valid on a successful FINAL
    /// chunk; replaces the accumulated per-chunk count, never adds to it.
    /// None means no authoritative total is available (including failures
    /// and early cancellation). SSE events are not model-token boundaries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
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
    /// Every generated token-id in this chunk, in order, so a resume source can
    /// accumulate EXACT ids. Engines that know the ids fill it on EVERY chunk —
    /// including single-token chunks, where `token_id == 0` with an empty vec is
    /// indistinguishable from an id-less chunk (token 0 is a legal sample).
    /// Chunks whose text is not token-addressed (whole-turn ov-genai bursts,
    /// synthetic markers with `token_id = 0`) leave it empty; such streams are
    /// not resume sources. `#[serde(default)]` keeps the wire additive.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ids: Vec<i64>,
    /// Raw per-option logits of a decision readout, on the FINAL chunk only
    /// (length = `ReadoutHead::output_len`). `None` on every generation chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_logits: Option<Vec<f32>>,
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
            completion_tokens: None,
            error: None,
            finish_reason: None,
            token_ids: Vec::new(),
            decision_logits: None,
        }
    }

    /// A progress marker: no text, no token (`n_tokens` = 0), not final. An
    /// engine emits one from a step that did real work for the task but has
    /// nothing to say yet (a long prompt going down a pipeline one window at
    /// a time), so the runner's no-progress watchdog sees the work and the
    /// API sends nothing for it. Consumers: [`Self::is_progress`].
    pub fn progress(task_id: TaskId) -> Self {
        Self {
            task_id,
            token_id: 0,
            text: String::new(),
            token_ids: Vec::new(),
            is_final: false,
            logprobs: None,
            n_tokens: Some(0),
            prompt_tokens: None,
            completion_tokens: None,
            error: None,
            finish_reason: None,
            decision_logits: None,
        }
    }

    /// See [`Self::progress`].
    pub fn is_progress(&self) -> bool {
        !self.is_final && self.error.is_none() && self.text.is_empty() && self.n_tokens == Some(0)
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
            completion_tokens: None,
            error: None,
            finish_reason: None,
            token_ids: Vec::new(),
            decision_logits: None,
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
            completion_tokens: None,
            error: Some(reason.into()),
            finish_reason: None,
            token_ids: Vec::new(),
            decision_logits: None,
        }
    }

    pub fn with_prompt_tokens(mut self, n: u32) -> Self {
        self.prompt_tokens = Some(n);
        self
    }

    /// Set the whole-generation total on a successful final chunk.
    pub fn with_completion_tokens(mut self, n: u32) -> Self {
        self.completion_tokens = Some(n);
        self
    }

    /// Tag the final chunk with why generation stopped (length vs stop).
    pub fn with_finish_reason(mut self, reason: FinishReason) -> Self {
        self.finish_reason = Some(reason);
        self
    }

    /// Stamp the exact generated ids this chunk carries (see `token_ids` docs:
    /// per-token producers stamp every chunk so a legal token-0 sample can't
    /// read as id-less).
    pub fn with_token_ids(mut self, ids: Vec<i64>) -> Self {
        self.token_ids = ids;
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

    /// Attach a decision readout's raw per-option logits (FINAL chunk only).
    pub fn with_decision_logits(mut self, logits: Vec<f32>) -> Self {
        self.decision_logits = Some(logits);
        self
    }

    /// Known tokens contributed by this chunk (not a whole-generation total).
    /// `n_tokens` wins when set; otherwise non-empty text contributes one and
    /// empty final markers contribute zero. Cumulative usage consumers must
    /// use [`Self::accumulate_tokens`] to reconcile authoritative final totals.
    pub fn token_count(&self) -> u32 {
        self.n_tokens
            .unwrap_or(if self.text.is_empty() { 0 } else { 1 })
    }

    /// Update a task's cumulative count without double-counting a final
    /// authoritative total. Failures and cancelled tasks have no final total.
    pub fn accumulate_tokens(&self, total: &mut u32) {
        if self.error.is_some() {
            return;
        }
        if self.is_final && self.finish_reason != Some(FinishReason::Cancelled) {
            if let Some(n) = self.completion_tokens {
                *total = n;
                return;
            }
        }
        *total += self.token_count();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_marker_is_not_a_token_nor_final() {
        let p = Chunk::progress("t".into());
        assert!(p.is_progress());
        assert!(!p.is_final);
        assert!(p.text.is_empty());
        assert_eq!(p.n_tokens, Some(0));
        assert!(!Chunk::token("t".to_string(), 1, "a").is_progress());
        assert!(!Chunk::final_marker("t".to_string(), "").is_progress());
        let mut e = Chunk::progress("t".into());
        e.error = Some("x".into());
        assert!(!e.is_progress());
    }

    #[test]
    fn chunk_token_ids_defaults_empty_and_is_additive() {
        let c = Chunk {
            task_id: "t1".to_string(),
            token_id: 42,
            text: "hi".to_string(),
            is_final: false,
            logprobs: None,
            n_tokens: None,
            prompt_tokens: None,
            completion_tokens: None,
            error: None,
            finish_reason: None,
            token_ids: Vec::new(),
            decision_logits: None,
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: Chunk = serde_json::from_str(&json).unwrap();
        assert_eq!(back.token_ids, Vec::<i64>::new());

        // Additivity: a payload encoded WITHOUT the field still decodes,
        // defaulting to an empty vec.
        let without_field = serde_json::json!({
            "task_id": "t1",
            "token_id": 42,
            "text": "hi",
        });
        let back: Chunk = serde_json::from_value(without_field).unwrap();
        assert_eq!(back.token_ids, Vec::<i64>::new());
        assert_eq!(back.completion_tokens, None);
    }

    #[test]
    fn authoritative_completion_replaces_partial_counts_only_on_success() {
        let mut total = 0;
        Chunk::progress("t".into()).accumulate_tokens(&mut total);
        assert_eq!(total, 0);
        Chunk::token("t", 1, "partial").accumulate_tokens(&mut total);
        Chunk::final_marker("t", "last")
            .with_n_tokens(2)
            .with_completion_tokens(38)
            .accumulate_tokens(&mut total);
        assert_eq!(total, 38);
        Chunk::error("t", "failed")
            .with_completion_tokens(99)
            .accumulate_tokens(&mut total);
        assert_eq!(total, 38);
        Chunk::final_marker("t", "")
            .with_finish_reason(FinishReason::Cancelled)
            .with_completion_tokens(99)
            .accumulate_tokens(&mut total);
        assert_eq!(total, 38);
        Chunk::final_marker("t", "")
            .with_completion_tokens(0)
            .accumulate_tokens(&mut total);
        assert_eq!(total, 0);
        let final_chunk = Chunk::final_marker("t", "").with_completion_tokens(38);
        let round_trip: Chunk =
            serde_json::from_str(&serde_json::to_string(&final_chunk).unwrap()).unwrap();
        assert_eq!(round_trip.completion_tokens, Some(38));
    }
}
