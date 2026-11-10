//! A cancel signal that reaches an engine parked inside `step()`.
//!
//! The runner holds the engine behind one mutex, and `step()` holds that mutex for the whole call.
//! An engine that runs a whole generation inside one step (one blocking FFI call, one runner loop)
//! can therefore never be handed `Engine::cancel` while it runs. This handle is the way around the
//! lock: the engine hands it out once, through [`crate::Engine::inflight_cancel`], and the runner
//! signals it without taking the engine lock. The engine polls [`InflightCancel::cancelled`] from
//! inside its step.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use cascadia_types::TaskId;

/// How many cancels for tasks that are not running yet the handle remembers. Oldest first out.
pub const MAX_EARLY_CANCELS: usize = 4096;

#[derive(Default)]
struct State {
    current: Option<TaskId>,
    early: VecDeque<TaskId>,
    /// Set by [`InflightCancel::shutdown`]: every `begin` reports a cancel until `reopen`.
    closed: bool,
}

/// Lock-free-to-poll cancel state for the one task an engine is running.
///
/// The engine brackets each task's run with [`begin`](Self::begin) and [`end`](Self::end), and polls
/// [`cancelled`](Self::cancelled) in between. Anyone may call [`cancel`](Self::cancel) at any time,
/// from any thread.
///
/// A cancel for a task that isn't running (still queued, or popped but not yet begun) is kept in a
/// bounded FIFO set, so [`begin`](Self::begin) can report it. That closes the race between the
/// engine popping a task and starting it.
///
/// An engine that drops a queued task in its `Engine::cancel` should call [`forget`](Self::forget)
/// for it, so the task's early entry (the runner signals this handle on every cancel) doesn't sit in
/// the set until it is evicted.
///
/// Invariant: a cancel for task A never sets the flag while task B is current. `cancel` and `begin`
/// both decide under the same mutex, and `begin` resets the flag for the incoming task.
#[derive(Default)]
pub struct InflightCancel {
    state: Mutex<State>,
    flag: AtomicBool,
}

impl InflightCancel {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        // A panic while holding this lock can't leave `State` half-updated in a way that matters
        // (each method is a couple of field writes), so a poisoned lock is still usable.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Mark `task_id` as the running task. Returns `true` when it was cancelled before it began; the
    /// engine should then not run it. Either way the engine must call [`end`](Self::end) after.
    pub fn begin(&self, task_id: &TaskId) -> bool {
        let mut s = self.lock();
        let early = match s.early.iter().position(|t| t == task_id) {
            Some(i) => {
                s.early.remove(i);
                true
            }
            None => false,
        } || s.closed;
        s.current = Some(task_id.clone());
        // Under the lock, so a `cancel` for the previous task can't land between the reset and the
        // new `current`.
        self.flag.store(early, Ordering::Release);
        early
    }

    /// Ask for `task_id` to stop. Sets the flag if it is the running task, else remembers it for
    /// its [`begin`](Self::begin).
    pub fn cancel(&self, task_id: &TaskId) {
        let mut s = self.lock();
        if s.current.as_ref() == Some(task_id) {
            self.flag.store(true, Ordering::Release);
            return;
        }
        if s.early.iter().any(|t| t == task_id) {
            return;
        }
        if s.early.len() >= MAX_EARLY_CANCELS {
            s.early.pop_front();
        }
        s.early.push_back(task_id.clone());
    }

    /// The engine is shutting down: stop the running task, and make every later
    /// [`begin`](Self::begin) report a cancel, until [`reopen`](Self::reopen). A queued task
    /// that wins the engine after this, or one popped but not yet begun, then never runs.
    pub fn shutdown(&self) {
        let mut s = self.lock();
        s.closed = true;
        if s.current.is_some() {
            self.flag.store(true, Ordering::Release);
        }
    }

    /// Whether [`shutdown`](Self::shutdown) is in effect.
    pub fn is_shut(&self) -> bool {
        self.lock().closed
    }

    /// Undo [`shutdown`](Self::shutdown): the engine is in service again.
    pub fn reopen(&self) {
        self.lock().closed = false;
    }

    /// The running task is over: clear it and reset the flag.
    pub fn end(&self) {
        let mut s = self.lock();
        s.current = None;
        self.flag.store(false, Ordering::Release);
    }

    /// Drop `task_id` from the early-cancel set, if it is there. For an engine's `Engine::cancel` of a
    /// task that is queued and will now never begin. Leaves the running task and the flag alone.
    pub fn forget(&self, task_id: &TaskId) {
        let mut s = self.lock();
        if let Some(i) = s.early.iter().position(|t| t == task_id) {
            s.early.remove(i);
        }
    }

    /// Whether the running task has been cancelled. One `Acquire` load; cheap enough per token.
    pub fn cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn early_len(&self) -> usize {
        self.lock().early.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn id(s: &str) -> TaskId {
        s.to_string()
    }

    #[test]
    fn shutdown_cancels_the_running_task_and_every_later_begin() {
        let ic = InflightCancel::new();
        assert!(!ic.begin(&id("a")));
        ic.shutdown();
        assert!(ic.cancelled(), "the running task is cancelled");
        ic.end();
        assert!(
            ic.begin(&id("b")),
            "a task begun after the shutdown is cancelled"
        );
        assert!(ic.cancelled());
        ic.end();
        assert_eq!(ic.early_len(), 0, "nothing is parked in the early set");
        assert!(ic.is_shut());
        ic.reopen();
        assert!(!ic.is_shut());
        assert!(!ic.begin(&id("c")), "reopened: tasks run again");
        assert!(!ic.cancelled());
        ic.end();
    }

    #[test]
    fn a_cancel_before_begin_is_reported_by_begin() {
        let ic = InflightCancel::new();
        ic.cancel(&id("a"));
        assert!(
            ic.begin(&id("a")),
            "a cancel that landed before begin was lost"
        );
        assert!(ic.cancelled());
        ic.end();
        assert!(!ic.cancelled());
        // It is consumed: the same id begun again is not cancelled.
        assert!(!ic.begin(&id("a")));
        ic.end();
    }

    #[test]
    fn a_cancel_for_the_running_task_sets_the_flag() {
        let ic = InflightCancel::new();
        assert!(!ic.begin(&id("a")));
        assert!(!ic.cancelled());
        ic.cancel(&id("a"));
        assert!(ic.cancelled());
        ic.end();
        assert!(!ic.cancelled());
    }

    #[test]
    fn a_cancel_for_another_task_never_sets_the_flag() {
        let ic = InflightCancel::new();
        assert!(!ic.begin(&id("b")));
        ic.cancel(&id("a"));
        assert!(!ic.cancelled(), "a cancel for A set the flag while B ran");
        ic.end();
        // A's cancel was kept for A.
        assert!(ic.begin(&id("a")));
        ic.end();
    }

    /// Two threads cancel A without pause while the engine hands over from A to B, 10 000 times.
    /// B must never see the flag.
    #[test]
    fn hammered_cancels_for_a_never_cancel_b_across_a_handover() {
        let ic = Arc::new(InflightCancel::new());
        let stop = Arc::new(AtomicBool::new(false));
        let hammers: Vec<_> = (0..2)
            .map(|_| {
                let ic = ic.clone();
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let a = id("A");
                    while !stop.load(Ordering::Relaxed) {
                        ic.cancel(&a);
                    }
                })
            })
            .collect();
        let (a, b) = (id("A"), id("B"));
        let mut b_cancelled = 0usize;
        for _ in 0..10_000 {
            let _ = ic.begin(&a);
            ic.end();
            if ic.begin(&b) {
                b_cancelled += 1;
            }
            for _ in 0..8 {
                if ic.cancelled() {
                    b_cancelled += 1;
                    break;
                }
                std::hint::spin_loop();
            }
            ic.end();
        }
        stop.store(true, Ordering::Relaxed);
        for h in hammers {
            h.join().unwrap();
        }
        assert_eq!(b_cancelled, 0, "B was cancelled by cancels meant for A");
    }

    #[test]
    fn forget_drops_an_early_cancel_and_nothing_else() {
        let ic = InflightCancel::new();
        ic.cancel(&id("queued"));
        ic.cancel(&id("other"));
        ic.forget(&id("queued"));
        assert_eq!(ic.early_len(), 1, "forget left the id in the early set");
        assert!(
            !ic.begin(&id("queued")),
            "a forgotten cancel was still reported"
        );
        ic.end();
        assert!(
            ic.begin(&id("other")),
            "forget removed another task's cancel"
        );
        ic.end();
        // An id that isn't there is a no-op, and the running task's flag is untouched.
        assert!(!ic.begin(&id("run")));
        ic.cancel(&id("run"));
        ic.forget(&id("run"));
        ic.forget(&id("never"));
        assert!(ic.cancelled(), "forget cleared the running task's flag");
        ic.end();
    }

    #[test]
    fn the_early_set_is_deduped_and_bounded_oldest_out() {
        let ic = InflightCancel::new();
        ic.cancel(&id("same"));
        ic.cancel(&id("same"));
        assert_eq!(
            ic.early_len(),
            1,
            "a repeated cancel must not take two slots"
        );
        ic.end();

        let ic = InflightCancel::new();
        let n = MAX_EARLY_CANCELS + 100;
        for i in 0..n {
            ic.cancel(&format!("t{i}"));
        }
        assert_eq!(ic.early_len(), MAX_EARLY_CANCELS);
        // The oldest were dropped, the newest kept.
        assert!(
            !ic.begin(&id("t0")),
            "the oldest cancel should have been evicted"
        );
        ic.end();
        assert!(
            ic.begin(&format!("t{}", n - 1)),
            "the newest cancel was lost"
        );
        ic.end();
    }
}
