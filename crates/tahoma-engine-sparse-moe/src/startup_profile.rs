//! Startup phase profiler for the sparse-MoE engine.
//!
//! The K2.6 model is ~553 GB on disk; cold load takes ~5 min on miner
//! (subsequent loads ~60 s once the OS page cache is warm). Without
//! per-phase instrumentation it's impossible to tell whether the time
//! is going to safetensors mmap, per-shell int4 quantization, head IR
//! compile, or tokenizer parse.
//!
//! This module records (phase_name, wall_duration) tuples into a
//! process-global vector as each phase of `Builder::load` / `Runner::load`
//! completes. The recorder is intentionally tiny — one `Mutex<Vec<...>>`
//! plus an `Instant` per phase — because the cost of recording must be
//! deeply negligible compared to a phase whose units are tenths of
//! seconds and up.
//!
//! Usage from inside the engine:
//!
//! ```ignore
//! use crate::startup_profile::PhaseTimer;
//! let _t = PhaseTimer::start("manifest_load");
//! // ... do the work ...
//! drop(_t); // records on drop
//! ```
//!
//! At the CLI seam (after `Runner::start_with_listen` returns), call
//! [`drain_report`] to fetch the ordered phase list. The CLI's
//! `--profile-startup` flag controls whether the human-readable table
//! is printed to stderr.
//!
//! Why not just rely on `tracing::info_span!` event timings? The fmt
//! subscriber doesn't print span open/close durations by default; the
//! ENTER/EXIT events that `with_span_events(FmtSpan::CLOSE)` enables
//! are visually noisy and don't aggregate. A purpose-built recorder
//! gives the operator a single coherent table they can paste into a
//! commit message. We still emit `info_span!` + an `info!` line per
//! phase so `RUST_LOG=info` users see the same data, just unaggregated.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// One phase of `Builder::load`'s startup sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseRecord {
    /// Phase name (e.g. `"manifest_load"`, `"shell_quantize_L3"`).
    pub name: String,
    /// Wall-clock duration of the phase.
    pub elapsed: Duration,
}

/// Process-global recorder. Initialised lazily on first
/// [`PhaseTimer::start`]; persists for the life of the process.
fn recorder() -> &'static Mutex<Vec<PhaseRecord>> {
    static REC: OnceLock<Mutex<Vec<PhaseRecord>>> = OnceLock::new();
    REC.get_or_init(|| Mutex::new(Vec::new()))
}

/// RAII timer. Recording happens in `Drop` so the same `let _t = …`
/// pattern that scopes a `tracing::Span` works for the recorder too,
/// and recording can't be skipped by an early `?` return inside the
/// timed block.
pub struct PhaseTimer {
    name: String,
    started: Instant,
}

impl PhaseTimer {
    /// Begin timing a named phase. The recorded duration is the wall
    /// time between `start` and `Drop`. Phases are recorded in `Drop`
    /// order, which matches phase-completion order in practice.
    pub fn start(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            started: Instant::now(),
        }
    }

    /// Wall-clock elapsed so far. Useful for in-flight logging without
    /// closing the timer (e.g. shell-loading loops that emit progress
    /// every N shells).
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

impl Drop for PhaseTimer {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        // Lock-poison should be impossible here (we only push, never
        // panic with the lock held), but if some other thread did
        // panic with the lock we'd rather lose a phase record than
        // poison the rest of startup. Swallow poison errors.
        if let Ok(mut g) = recorder().lock() {
            g.push(PhaseRecord {
                name: std::mem::take(&mut self.name),
                elapsed,
            });
        }
    }
}

/// Drain the current set of recorded phases, leaving the recorder
/// empty. Intended to be called once after `Runner::start_with_listen`
/// returns. Returns the phases in completion (Drop) order.
pub fn drain_report() -> Vec<PhaseRecord> {
    match recorder().lock() {
        Ok(mut g) => std::mem::take(&mut *g),
        Err(_) => Vec::new(),
    }
}

/// Render a list of phases as a human-readable table. Lines are
/// indented two spaces so the table sits naturally under a `tracing`
/// info line. The final `TOTAL` row is the simple sum of the per-phase
/// durations — not wall-clock end-to-end — so it slightly undercounts
/// (it misses any wall time spent between phases, e.g. tokio
/// suspensions). In practice the gap is bounded by sub-millisecond
/// scheduling overhead and the operator can tell it's an aggregate
/// because the row says `SUM`.
pub fn format_report(phases: &[PhaseRecord]) -> String {
    if phases.is_empty() {
        return "  (no phases recorded — startup profiler saw zero events)".to_string();
    }
    let mut out = String::new();
    let name_w = phases
        .iter()
        .map(|p| p.name.len())
        .max()
        .unwrap_or(0)
        .max("phase".len());
    let total: Duration = phases.iter().map(|p| p.elapsed).sum();
    out.push_str(&format!(
        "  {:name_w$}    elapsed_ms    pct\n",
        "phase",
        name_w = name_w
    ));
    out.push_str(&format!(
        "  {:-<name_w$}    ----------    ----\n",
        "",
        name_w = name_w
    ));
    let total_ms = total.as_secs_f64() * 1000.0;
    for p in phases {
        let ms = p.elapsed.as_secs_f64() * 1000.0;
        let pct = if total_ms > 0.0 {
            100.0 * ms / total_ms
        } else {
            0.0
        };
        out.push_str(&format!(
            "  {:name_w$}    {:>10.1}    {:>4.1}\n",
            p.name,
            ms,
            pct,
            name_w = name_w
        ));
    }
    out.push_str(&format!(
        "  {:-<name_w$}    ----------    ----\n",
        "",
        name_w = name_w
    ));
    out.push_str(&format!(
        "  {:name_w$}    {:>10.1}    SUM\n",
        "TOTAL",
        total_ms,
        name_w = name_w
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::thread;

    /// Cargo runs unit tests in parallel by default; the four tests in
    /// this module that exercise the *global* recorder all observe
    /// shared state, so they'd flake under parallel scheduling (one
    /// test's `drain_report` would eat another's records). Serialise
    /// them with a module-local mutex. Tests that only inspect their
    /// own values (`format_report_*`) skip the guard.
    fn global_guard() -> std::sync::MutexGuard<'static, ()> {
        static M: OnceLock<StdMutex<()>> = OnceLock::new();
        let m = M.get_or_init(|| StdMutex::new(()));
        // Recover from a poisoned mutex — a previous test panicking
        // shouldn't sink the rest of the suite.
        match m.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        }
    }

    /// Each global-state test starts by draining to clear any
    /// leftovers from earlier runs.
    fn fresh_recorder() {
        let _ = drain_report();
    }

    #[test]
    fn phases_recorded_in_completion_order() {
        let _g = global_guard();
        fresh_recorder();
        {
            let _a = PhaseTimer::start("first");
            thread::sleep(Duration::from_millis(2));
        }
        {
            let _b = PhaseTimer::start("second");
            thread::sleep(Duration::from_millis(1));
        }
        let phases = drain_report();
        assert_eq!(phases.len(), 2, "expected 2 phases, got {:?}", phases);
        assert_eq!(phases[0].name, "first");
        assert_eq!(phases[1].name, "second");
    }

    #[test]
    fn drain_leaves_recorder_empty() {
        let _g = global_guard();
        fresh_recorder();
        {
            let _a = PhaseTimer::start("only");
        }
        let first = drain_report();
        assert_eq!(first.len(), 1);
        let second = drain_report();
        assert!(
            second.is_empty(),
            "second drain should be empty; got {:?}",
            second
        );
    }

    #[test]
    fn timer_records_nonzero_duration() {
        let _g = global_guard();
        fresh_recorder();
        {
            let _a = PhaseTimer::start("sleeper");
            thread::sleep(Duration::from_millis(3));
        }
        let phases = drain_report();
        assert_eq!(phases.len(), 1);
        // A 3ms sleep should always show > 0; allow generous slack
        // for slow CI VMs where sleep granularity is coarser.
        assert!(
            phases[0].elapsed >= Duration::from_millis(1),
            "expected elapsed >= 1ms, got {:?}",
            phases[0].elapsed
        );
    }

    #[test]
    fn format_report_has_total_row() {
        let phases = vec![
            PhaseRecord {
                name: "a".into(),
                elapsed: Duration::from_millis(100),
            },
            PhaseRecord {
                name: "b".into(),
                elapsed: Duration::from_millis(300),
            },
        ];
        let s = format_report(&phases);
        assert!(s.contains("TOTAL"), "report missing TOTAL row:\n{}", s);
        assert!(s.contains("400.0"), "report missing sum 400.0:\n{}", s);
        // Phases appear in order in the rendered table.
        let pa = s.find(" a ").expect("phase a not in report");
        let pb = s.find(" b ").expect("phase b not in report");
        assert!(pa < pb, "phase order wrong in report:\n{}", s);
    }

    #[test]
    fn format_report_with_no_phases_is_explicit() {
        let s = format_report(&[]);
        assert!(
            s.contains("no phases recorded"),
            "empty report should explain itself; got:\n{}",
            s
        );
    }

    #[test]
    fn timer_elapsed_during_run_grows_monotonically() {
        let _g = global_guard();
        // Drain so the unrecorded-by-design phase we create below
        // doesn't leak into a sibling test.
        fresh_recorder();
        let t = PhaseTimer::start("growing");
        let e1 = t.elapsed();
        thread::sleep(Duration::from_millis(2));
        let e2 = t.elapsed();
        assert!(e2 >= e1, "elapsed went backwards: {:?} -> {:?}", e1, e2);
        drop(t);
        // Drain to keep the recorder clean for sibling tests in any order.
        let _ = drain_report();
    }
}
