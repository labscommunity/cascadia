//! M18: ending a running single-stage one-shot generation early (D3, D4, D6).
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

use cascadia_engine::{InflightCancel, OneShotStop};
use cascadia_types::{Chunk, FinishReason, TaskId};
use tracing::warn;

/// Ids decoded past the longest stop's byte length, so a run of 0-byte
/// (skipped special) tokens doesn't push a stop out of the window (D6).
pub(crate) const TAIL_SLACK_IDS: usize = 16;

/// Calls [`InflightCancel::end`] when dropped, so every path out of a begun
/// task (success, error, a panic unwinding through `step`) frees the handle.
pub(crate) struct EndInflight(pub(crate) Arc<InflightCancel>);

impl Drop for EndInflight {
    fn drop(&mut self) {
        self.0.end();
    }
}

/// The final chunk for a cancelled task: empty, `Cancelled` (M18 D4).
pub(crate) fn cancelled_chunk(task_id: &TaskId) -> Chunk {
    Chunk::final_marker(task_id.clone(), "").with_finish_reason(FinishReason::Cancelled)
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
/// `max_stop_bytes() + TAIL_SLACK_IDS` generated ids holds a stop. `decode`
/// must use the tokenizer and options of the engine's final decode: a decode of
/// a suffix of the ids is then a substring of the final text (OneShotStop's
/// input contract; M18 D8). With no active stop nothing is decoded.
pub(crate) struct OneShotPoll<'a, T> {
    inflight: &'a InflightCancel,
    stop: OneShotStop,
    window: usize,
    decode: TailDecode<'a, T>,
    ended: Option<PollEnd>,
}

impl<'a, T> OneShotPoll<'a, T> {
    pub(crate) fn new(
        inflight: &'a InflightCancel,
        stops: &[String],
        engine_stop_check: bool,
        decode: impl Fn(&[T]) -> Option<String> + 'a,
    ) -> Self {
        let stop = if engine_stop_check {
            OneShotStop::new(stops)
        } else {
            OneShotStop::new(&[])
        };
        let window = if stop.is_active() {
            stop.max_stop_bytes() + TAIL_SLACK_IDS
        } else {
            0
        };
        Self {
            inflight,
            stop,
            window,
            decode: Box::new(decode),
            ended: None,
        }
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
                if self.stop.matches_tail(&text) {
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

/// The text to return after an engine stop, for an engine that post-processes
/// its decode (`raw`) into `visible` (a prompt-echo strip, a leading trim).
///
/// That processing can remove the text the stop matched in; the API would then
/// find no stop and the client would get an answer cut short with nothing to
/// show why. So `raw` is returned when only it holds a stop (OpenAI: stop applies
/// to the generated output). When neither does, the D8 invariant broke (a
/// detokenizer cleanup rule rewrote text based on later tokens): keep `visible`
/// and warn; the hardware check fails on that line (M18 D8).
pub(crate) fn text_after_stop(
    task_id: &TaskId,
    stops: &[String],
    visible: String,
    raw: String,
) -> String {
    let has_stop = |t: &str| OneShotStop::new(stops).push(t);
    if has_stop(&visible) {
        return visible;
    }
    if has_stop(&raw) {
        return raw;
    }
    warn!(
        task = %task_id,
        "engine stopped on a stop the returned text lacks (D8 invariant)"
    );
    visible
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
        let mut p = OneShotPoll::new(&h, &[], true, |ids: &[u32]| {
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
        let mut p = OneShotPoll::new(&h, &stops(&["zz"]), true, |ids: &[u32]| Some(letters(ids)));
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
        let mut p = OneShotPoll::new(&h, &[], true, |_: &[i64]| None);
        assert!(p.should_stop(&[]));
    }

    #[test]
    fn a_stop_in_the_decoded_tail_stops() {
        let h = InflightCancel::new();
        // "cd" = ids [2, 3].
        let mut p = OneShotPoll::new(&h, &stops(&["cd"]), true, |ids: &[u32]| Some(letters(ids)));
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
        let mut p = OneShotPoll::new(&h, &stops(&["x", "abc"]), true, |ids: &[u32]| {
            seen.borrow_mut().push(ids.len());
            Some(letters(ids))
        });
        let ids: Vec<u32> = vec![25; 100];
        assert!(!p.should_stop(&ids));
        assert_eq!(*seen.borrow(), vec![3 + TAIL_SLACK_IDS]);
    }

    #[test]
    fn a_stop_pushed_out_of_the_window_is_missed() {
        // Only latency: the API still cuts at the end (D6).
        let h = InflightCancel::new();
        let mut p = OneShotPoll::new(&h, &stops(&["a"]), true, |ids: &[u32]| Some(letters(ids)));
        let mut ids = vec![0u32];
        ids.extend(std::iter::repeat_n(25, 1 + TAIL_SLACK_IDS));
        assert!(!p.should_stop(&ids));
    }

    #[test]
    fn the_check_off_ignores_stops_but_not_a_cancel() {
        let h = InflightCancel::new();
        let id = TaskId::from("t");
        h.begin(&id);
        let mut p = OneShotPoll::new(&h, &stops(&["cd"]), false, |ids: &[u32]| Some(letters(ids)));
        assert!(!p.should_stop(&[2, 3]));
        h.cancel(&id);
        assert!(p.should_stop(&[2, 3]));
    }

    #[test]
    fn a_failed_decode_is_no_match() {
        let h = InflightCancel::new();
        let mut p = OneShotPoll::new(&h, &stops(&["cd"]), true, |_: &[u32]| None);
        assert!(!p.should_stop(&[2, 3]));
    }

    #[test]
    fn a_replacement_char_stop_is_never_checked() {
        let h = InflightCancel::new();
        let mut p = OneShotPoll::new(&h, &stops(&["\u{FFFD}"]), true, |_: &[u32]| {
            Some("\u{FFFD}".to_string())
        });
        assert!(!p.should_stop(&[1]));
    }

    #[test]
    fn text_after_stop_prefers_visible_then_raw_then_warns() {
        let id = TaskId::from("t");
        let s = stops(&["END"]);
        assert_eq!(
            text_after_stop(&id, &s, "a END".into(), "p a END".into()),
            "a END"
        );
        // The strip removed the match: the raw text goes out.
        assert_eq!(
            text_after_stop(&id, &s, "a".into(), "END a".into()),
            "END a"
        );
        // Neither holds it: keep the visible text.
        assert_eq!(text_after_stop(&id, &s, "a".into(), "b".into()), "a");
    }
}
