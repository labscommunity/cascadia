//! Per-pass timings of a decision pass on the single-box qwen35 path, logged at debug level on
//! target `cascadia::perf` (`RUST_LOG=info,cascadia::perf=debug`). Accumulating costs a few
//! `Instant::now()` calls per stage per span (tens of ns against ~10 ms stage infers), so it
//! is always on and only the log line is level-gated.
//!
//! Only the single-box path (`step_local`) fills it. The pipeline path (`step` with
//! `total > 1`) is not instrumented.

use std::time::Duration;

#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct PassTiming {
    pub prompt: usize,
    pub warm_prefix: usize,
    pub spans: u32,
    pub span_max: usize,
    /// Pop to prefill start: scrub, probe, restore, refusals, snapshot planning.
    pub admit: Duration,
    /// The admission's own `reset_all`, inside `admit`: a cold pass's reset, which after a
    /// decision pass is the scrub that pass deferred (a recreate after a warm pass, Task 55).
    /// Zero on a warm hit.
    pub admit_reset: Duration,
    /// The whole restore, priming fold included (what `qwen35 prefix-cache hit` logs).
    pub restore: Duration,
    pub prime: Duration,
    pub set_state: Duration,
    pub snapshot: Duration,
    pub embed: Duration,
    /// Per local stage, the sum of its `infer()` calls in this pass (the priming fold's too).
    pub stage_infer: Vec<Duration>,
    /// `set_input` (with the LE conversions) plus `output()` and `f32_from_le`, all stages.
    pub host_copy: Duration,
    pub head: Duration,
    /// The latest `reset_all` of this pass. Since Task 55 the single-box finish defers its scrub
    /// to the next admission, so this is the admission's own (`admit_reset`): zero on a warm hit in the default
    /// restore mode (`CASCADIA_QWEN36_RESTORE_CLEAR` unset).
    pub reset: Duration,
    /// That reset recreated the requests (they held a restored state). False on a warm hit in the
    /// default restore mode.
    pub recreate: bool,
    pub total: Duration,
}

impl PassTiming {
    pub fn new(n_stages: usize, prompt: usize) -> Self {
        Self {
            prompt,
            stage_infer: vec![Duration::ZERO; n_stages],
            ..Default::default()
        }
    }
    pub fn add_span(&mut self, n: usize) {
        self.spans += 1;
        self.span_max = self.span_max.max(n);
    }
    pub fn prefilled(&self) -> usize {
        self.prompt.saturating_sub(self.warm_prefix)
    }
    pub fn infer(&self) -> Duration {
        self.stage_infer.iter().sum()
    }
    /// `stage_ms` field: per-stage infer milliseconds, one decimal, comma-separated.
    pub fn stage_ms(&self) -> String {
        self.stage_infer
            .iter()
            .map(|d| format!("{:.1}", d.as_secs_f64() * 1e3))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Milliseconds rounded to one decimal, for the log fields.
pub(crate) fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1e4).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_prefilled_and_stage_sums() {
        let mut t = PassTiming::new(3, 300);
        t.warm_prefix = 171;
        t.add_span(129);
        t.add_span(7);
        t.stage_infer[0] = Duration::from_millis(10);
        t.stage_infer[2] = Duration::from_micros(2500);
        assert_eq!((t.spans, t.span_max, t.prefilled()), (2, 129, 129));
        assert_eq!(t.infer(), Duration::from_micros(12_500));
        assert_eq!(t.stage_ms(), "10.0,0.0,2.5");
        assert_eq!(ms(Duration::from_micros(1234)), 1.2);
    }

    #[test]
    fn prefilled_never_underflows() {
        let mut t = PassTiming::new(1, 5);
        t.warm_prefix = 9;
        assert_eq!(t.prefilled(), 0);
    }
}
