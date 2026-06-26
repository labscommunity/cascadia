//! Shared step-failure WARN rate-limiter (cascadia-enterprise issue #30).
//!
//! A persistently-failing `Engine::step()` driven by the relay loop logs
//! one WARN per call (~90k/s when the upstream connection drops and recv
//! fast-errors), which grew a node's chain.log to 770 GB. Used by the
//! relay-loop engines that fast-error without their own backoff —
//! `OvRuntimeEngine` and `Gemma4Engine` — so both bound the flood
//! identically. (`qwen36` already self-throttles with a sleep; `dist_spec`
//! is the speculative-decode engine.)

/// Rate-limits the per-call "step failed" WARN. First failure of a streak
/// logs in full; later ones are counted and surfaced as a periodic "still
/// failing, N suppressed" summary. Success closes the streak so the next
/// failure logs immediately again.
#[derive(Default)]
pub(crate) struct StepWarnLimiter {
    /// `Some(last WARN instant)` while in a failing streak.
    last_warn: Option<std::time::Instant>,
    suppressed: u64,
}

#[derive(Debug)]
pub(crate) enum StepWarn {
    First,
    StillFailing { suppressed: u64 },
}

impl StepWarnLimiter {
    pub(crate) const INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

    /// What to log for a failure at `now`, if anything.
    pub(crate) fn on_failure(&mut self, now: std::time::Instant) -> Option<StepWarn> {
        match self.last_warn {
            None => {
                self.last_warn = Some(now);
                Some(StepWarn::First)
            }
            Some(last) if now.duration_since(last) >= Self::INTERVAL => {
                self.last_warn = Some(now);
                let suppressed = std::mem::take(&mut self.suppressed);
                Some(StepWarn::StillFailing { suppressed })
            }
            Some(_) => {
                self.suppressed += 1;
                None
            }
        }
    }

    /// Closes a failing streak; returns the suppressed count to report.
    pub(crate) fn on_success(&mut self) -> Option<u64> {
        self.last_warn
            .take()
            .map(|_| std::mem::take(&mut self.suppressed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> std::time::Duration {
        std::time::Duration::from_millis(n)
    }

    #[test]
    fn logs_first_failure_then_suppresses() {
        let mut l = StepWarnLimiter::default();
        let t0 = std::time::Instant::now();
        assert!(matches!(l.on_failure(t0), Some(StepWarn::First)));
        assert!(l.on_failure(t0 + ms(1)).is_none());
        assert!(l.on_failure(t0 + ms(2)).is_none());
    }

    #[test]
    fn emits_periodic_summary_with_suppressed_count() {
        let mut l = StepWarnLimiter::default();
        let t0 = std::time::Instant::now();
        l.on_failure(t0);
        for i in 1..=5 {
            assert!(l.on_failure(t0 + ms(i)).is_none());
        }
        let later = t0 + StepWarnLimiter::INTERVAL + ms(10);
        match l.on_failure(later) {
            Some(StepWarn::StillFailing { suppressed }) => assert_eq!(suppressed, 5),
            other => panic!("expected StillFailing summary, got {other:?}"),
        }
        // Counter resets after each summary; the next interval reports
        // only what was suppressed since.
        assert!(l.on_failure(later + ms(1)).is_none());
        match l.on_failure(later + StepWarnLimiter::INTERVAL + ms(10)) {
            Some(StepWarn::StillFailing { suppressed }) => assert_eq!(suppressed, 1),
            other => panic!("expected second summary, got {other:?}"),
        }
    }

    #[test]
    fn success_ends_streak_and_relatches() {
        let mut l = StepWarnLimiter::default();
        let t0 = std::time::Instant::now();
        // No streak → nothing to report.
        assert!(l.on_success().is_none());
        l.on_failure(t0);
        l.on_failure(t0 + ms(1));
        // Streak ends: report the one suppressed failure.
        assert_eq!(l.on_success(), Some(1));
        // Latch cleared: a new failure logs immediately again.
        assert!(matches!(l.on_failure(t0 + ms(2)), Some(StepWarn::First)));
        // Streak with nothing suppressed still closes (count 0).
        assert_eq!(l.on_success(), Some(0));
        assert!(l.on_success().is_none());
    }
}
