//! Ending a running single-stage one-shot generation early.
//!
//! The three single-stage paths (`SparseMoEEngine`, `OvMoeEngine`,
//! `PipelineEngine`) each run a whole generation in one blocking runner call
//! inside `step()`, which holds the engine lock. The runner's `*_polled`
//! generate calls a `should_stop` closure with the ids generated so far before
//! each prefill token, each decode token and each speculative round. This module
//! is the part of that closure the three engines share: the lock-free cancel
//! flag ([`InflightCancel`]) and the bounded in-engine `stop` check
//! ([`OneShotStop`]) on a suffix decode of the generated ids.

use std::sync::Arc;

use cascadia_engine::{InflightCancel, OneShotStop, VisibleArm};
use cascadia_types::{Chunk, FinishReason, TaskId};
use tracing::warn;

/// Ids decoded past the longest stop's byte length, so a run of 0-byte
/// (skipped special) tokens doesn't push a stop out of the window.
pub(crate) const TAIL_SLACK_IDS: usize = 16;

/// Calls [`InflightCancel::end`] when dropped, so every path out of a begun
/// task (success, error, a panic unwinding through `step`) frees the handle.
pub(crate) struct EndInflight(pub(crate) Arc<InflightCancel>);

impl Drop for EndInflight {
    fn drop(&mut self) {
        self.0.end();
    }
}

/// The final chunk for a cancelled task: empty, `Cancelled`, with
/// the prompt's token count when it is known, so the runner books the prompt.
/// No text and no `n_tokens`: no TTFT sample, 0 generated tokens.
pub(crate) fn cancelled_chunk(task_id: &TaskId, prompt_tokens: Option<u32>) -> Chunk {
    let mut chunk =
        Chunk::final_marker(task_id.clone(), "").with_finish_reason(FinishReason::Cancelled);
    chunk.prompt_tokens = prompt_tokens;
    chunk
}

/// Decodes a window of generated ids with the engine's final-decode options.
type TailDecode<'a, T> = Box<dyn Fn(&[T]) -> Option<String> + 'a>;

/// Why a polled generation was told to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PollEnd {
    Cancelled,
    Stopped,
}

/// The `should_stop` decision of one task, generic over the runner's id type.
///
/// [`Self::should_stop`] is true once the task is cancelled, or (with the
/// check on and a stop active) once a decode of the last
/// `max_stop_bytes() + TAIL_SLACK_IDS + extra_slack` generated ids holds a
/// stop. `extra_slack` is how many ids one poll interval can add beyond one
/// (a speculative round: `draft_k + 1`), so they can't push a stop out of the
/// window; 0 for a runner polled per token. `decode`
/// must use the tokenizer and options of the engine's final decode: a decode of
/// a suffix of the ids is then a substring of the final text (OneShotStop's
/// input contract). With no active stop nothing is decoded.
pub(crate) struct OneShotPoll<'a, T> {
    inflight: &'a InflightCancel,
    stop: OneShotStop,
    window: usize,
    decode: TailDecode<'a, T>,
    /// When the check may run; armed at once unless
    /// [`Self::with_visible_arm`] says the answer strips text.
    arm: VisibleArm,
    ended: Option<PollEnd>,
}

impl<'a, T> OneShotPoll<'a, T> {
    pub(crate) fn new(
        inflight: &'a InflightCancel,
        stops: &[String],
        engine_stop_check: bool,
        extra_slack: usize,
        decode: impl Fn(&[T]) -> Option<String> + 'a,
    ) -> Self {
        let stop = if engine_stop_check {
            OneShotStop::new(stops)
        } else {
            OneShotStop::new(&[])
        };
        let window = if stop.is_active() {
            stop.max_stop_bytes() + TAIL_SLACK_IDS + extra_slack
        } else {
            0
        };
        Self {
            inflight,
            stop,
            window,
            decode: Box::new(decode),
            arm: VisibleArm::always(),
            ended: None,
        }
    }

    /// For an engine whose answer is `strip_prefix(prompt)` + `trim_start`
    /// of its decode: arm the stop check only on text that answer keeps, so
    /// an engine stop never lands in a stripped prompt echo or leading
    /// whitespace (see [`VisibleArm`]).
    pub(crate) fn with_visible_arm(mut self, prompt: &str) -> Self {
        self.arm = VisibleArm::new(prompt, self.window);
        self
    }

    /// Whether the runner should end the generation now. Latches: once true,
    /// it stays true.
    pub(crate) fn should_stop(&mut self, generated: &[T]) -> bool {
        if self.ended.is_some() {
            return true;
        }
        if self.inflight.cancelled() {
            self.ended = Some(PollEnd::Cancelled);
            return true;
        }
        if self.window > 0 && !generated.is_empty() {
            let tail = &generated[generated.len().saturating_sub(self.window)..];
            if let Some(text) = (self.decode)(tail) {
                if self.arm.armed(generated.len(), &text) && self.stop.matches_tail(&text) {
                    self.ended = Some(PollEnd::Stopped);
                    return true;
                }
            }
        }
        false
    }

    /// The generation ended on a stop match (not a cancel).
    pub(crate) fn stopped_on_match(&self) -> bool {
        self.ended == Some(PollEnd::Stopped)
    }
}

/// After an engine stop: warns when the returned text holds no stop. The check
/// armed only on text that text keeps, so this means the stop-text invariant
/// broke (a detokenizer cleanup rule rewrote text based on later tokens); the
/// hardware check fails on that line.
pub(crate) fn check_text_after_stop(task_id: &TaskId, stops: &[String], text: &str) {
    if !has_stop(stops, text) {
        warn_d8_invariant(task_id);
    }
}

fn has_stop(stops: &[String], text: &str) -> bool {
    OneShotStop::new(stops).push(text)
}

fn warn_d8_invariant(task_id: &TaskId) {
    warn!(
        task = %task_id,
        "engine stopped on a stop the returned text lacks (stop-text invariant)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A decode stand-in: id `n` is the letter `'a' + n`, joined with no
    /// separator; it records each window it was asked for.
    fn letters(ids: &[u32]) -> String {
        ids.iter().map(|&i| (b'a' + i as u8) as char).collect()
    }

    fn stops(s: &[&str]) -> Vec<String> {
        s.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn no_stop_and_no_cancel_never_stops_and_never_decodes() {
        let h = InflightCancel::new();
        let calls = std::cell::Cell::new(0);
        let mut p = OneShotPoll::new(&h, &[], true, 0, |ids: &[u32]| {
            calls.set(calls.get() + 1);
            Some(letters(ids))
        });
        for n in 0..50u32 {
            let ids: Vec<u32> = (0..n).collect();
            assert!(!p.should_stop(&ids));
        }
        assert_eq!(calls.get(), 0, "no active stop: nothing is decoded");
        assert!(!p.stopped_on_match());
    }

    #[test]
    fn a_cancel_stops_at_the_next_poll_and_latches() {
        let h = InflightCancel::new();
        let id = TaskId::from("t");
        assert!(!h.begin(&id));
        let mut p = OneShotPoll::new(&h, &stops(&["zz"]), true, 0, |ids: &[u32]| {
            Some(letters(ids))
        });
        assert!(!p.should_stop(&[0, 1]));
        h.cancel(&id);
        assert!(p.should_stop(&[0, 1, 2]));
        assert!(!p.stopped_on_match(), "a cancel is not a stop match");
        h.end();
        assert!(p.should_stop(&[0, 1, 2, 3]), "latched");
    }

    #[test]
    fn a_cancel_during_prefill_stops_with_no_ids() {
        let h = InflightCancel::new();
        let id = TaskId::from("t");
        h.begin(&id);
        h.cancel(&id);
        let mut p = OneShotPoll::new(&h, &[], true, 0, |_: &[i64]| None);
        assert!(p.should_stop(&[]));
    }

    #[test]
    fn a_stop_in_the_decoded_tail_stops() {
        let h = InflightCancel::new();
        // "cd" = ids [2, 3].
        let mut p = OneShotPoll::new(&h, &stops(&["cd"]), true, 0, |ids: &[u32]| {
            Some(letters(ids))
        });
        assert!(!p.should_stop(&[0]));
        assert!(!p.should_stop(&[0, 1]));
        assert!(!p.should_stop(&[0, 1, 2]));
        assert!(p.should_stop(&[0, 1, 2, 3]));
        assert!(p.stopped_on_match());
    }

    #[test]
    fn the_window_is_the_longest_stop_plus_slack() {
        let h = InflightCancel::new();
        let seen = std::cell::RefCell::new(Vec::new());
        let mut p = OneShotPoll::new(&h, &stops(&["x", "abc"]), true, 0, |ids: &[u32]| {
            seen.borrow_mut().push(ids.len());
            Some(letters(ids))
        });
        let ids: Vec<u32> = vec![25; 100];
        assert!(!p.should_stop(&ids));
        assert_eq!(*seen.borrow(), vec![3 + TAIL_SLACK_IDS]);
    }

    /// A speculative round can accept up to K drafts plus the bonus id in one
    /// poll interval; the window grows by that much so the round can't push a
    /// stop out of it.
    #[test]
    fn extra_slack_widens_the_window() {
        let h = InflightCancel::new();
        let seen = std::cell::RefCell::new(Vec::new());
        let mut p = OneShotPoll::new(&h, &stops(&["abc"]), true, 4 + 1, |ids: &[u32]| {
            seen.borrow_mut().push(ids.len());
            Some(letters(ids))
        });
        assert!(!p.should_stop(&[25; 100]));
        assert_eq!(*seen.borrow(), vec![3 + TAIL_SLACK_IDS + 5]);
    }

    #[test]
    fn a_stop_pushed_out_of_the_window_is_missed() {
        // Only latency: the API still cuts at the end.
        let h = InflightCancel::new();
        let mut p = OneShotPoll::new(&h, &stops(&["a"]), true, 0, |ids: &[u32]| {
            Some(letters(ids))
        });
        let mut ids = vec![0u32];
        ids.extend(std::iter::repeat_n(25, 1 + TAIL_SLACK_IDS));
        assert!(!p.should_stop(&ids));
    }

    #[test]
    fn the_check_off_ignores_stops_but_not_a_cancel() {
        let h = InflightCancel::new();
        let id = TaskId::from("t");
        h.begin(&id);
        let mut p = OneShotPoll::new(&h, &stops(&["cd"]), false, 0, |ids: &[u32]| {
            Some(letters(ids))
        });
        assert!(!p.should_stop(&[2, 3]));
        h.cancel(&id);
        assert!(p.should_stop(&[2, 3]));
    }

    #[test]
    fn a_failed_decode_is_no_match() {
        let h = InflightCancel::new();
        let mut p = OneShotPoll::new(&h, &stops(&["cd"]), true, 0, |_: &[u32]| None);
        assert!(!p.should_stop(&[2, 3]));
    }

    #[test]
    fn a_replacement_char_stop_is_never_checked() {
        let h = InflightCancel::new();
        let mut p = OneShotPoll::new(&h, &stops(&["\u{FFFD}"]), true, 0, |_: &[u32]| {
            Some("\u{FFFD}".to_string())
        });
        assert!(!p.should_stop(&[1]));
    }

    /// The K2.6 single-stage flow over a token table: poll after every
    /// generated id (as the runner does), stop when told, then the visible
    /// text `step_single_stage` returns (strip the prompt, trim).
    fn k26_answer(prompt: &str, toks: &[&str], stops_: &[&str], check: bool) -> (String, usize) {
        let h = InflightCancel::new();
        let decode = |ids: &[u32]| Some(ids.iter().map(|&i| toks[i as usize]).collect::<String>());
        let mut p = OneShotPoll::new(&h, &stops(stops_), check, 0, decode).with_visible_arm(prompt);
        let mut ids: Vec<u32> = Vec::new();
        for i in 0..toks.len() as u32 {
            ids.push(i);
            if p.should_stop(&ids) {
                break;
            }
        }
        let raw = decode(&ids).unwrap();
        let visible = match raw.strip_prefix(prompt) {
            Some(s) => s.trim_start().to_string(),
            None => raw,
        };
        (visible, ids.len())
    }

    /// An empty prompt: the strip always applies. The leading "\n"s are not
    /// in the answer, so no engine stop there: the visible text is
    /// the check-off text, and the API cut gives "Hi".
    #[test]
    fn an_empty_prompt_does_not_stop_in_leading_whitespace() {
        let toks = ["\n", "\n", "Hi", "\n", "There"];
        let (on, n) = k26_answer("", &toks, &["\n"], true);
        assert_eq!(on, "Hi\nThere");
        assert_eq!(n, toks.len());
        assert_eq!(on, k26_answer("", &toks, &["\n"], false).0);
    }

    #[test]
    fn a_prompt_echo_keeps_the_visible_answer() {
        let toks = ["Hi", "\n", "There", "\n"];
        let (on, _) = k26_answer("Hi", &toks, &["\n"], true);
        assert_eq!(on, "There\n");
        assert_eq!(on, k26_answer("Hi", &toks, &["\n"], false).0);
    }

    #[test]
    fn a_chat_prompt_arms_at_once() {
        let (on, n) = k26_answer("<|user|>count", &["one", "\n", "two", "\n"], &["\n"], true);
        assert_eq!((on.as_str(), n), ("one\n", 2));
    }

    /// An echo longer than the window: never armed, latency only.
    #[test]
    fn a_long_echo_never_arms() {
        let echo: Vec<String> = (0..20).map(|i| format!("w{i} ")).collect();
        let prompt = echo.concat();
        let mut toks: Vec<&str> = echo.iter().map(String::as_str).collect();
        toks.extend(["x", " y", " x", " z"]);
        // Stop "x": window 1 + TAIL_SLACK_IDS = 17 ids; the echo is 20.
        let (on, n) = k26_answer(&prompt, &toks, &["x"], true);
        assert_eq!(n, toks.len());
        assert_eq!(on, "x y x z");
        assert_eq!(on, k26_answer(&prompt, &toks, &["x"], false).0);
    }
}
