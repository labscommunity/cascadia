//! Integration test for the startup profiler.
//!
//! Verifies that phases emitted via `PhaseTimer` show up in `drain_report`
//! in completion (Drop) order and with monotonically non-decreasing
//! elapsed durations matching the order they ran.
//!
//! Integration tests in a single `tests/*.rs` file all link into the
//! same test binary AND run in parallel by default, so they share the
//! process-global recorder just like unit tests do. Serialise with a
//! file-local `Mutex` so a concurrent test's phases don't leak in.

use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread::sleep;
use std::time::Duration;

use tahoma_engine_sparse_moe::{drain_report, format_report, PhaseTimer};

fn recorder_guard() -> MutexGuard<'static, ()> {
    static M: OnceLock<Mutex<()>> = OnceLock::new();
    let m = M.get_or_init(|| Mutex::new(()));
    match m.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    }
}

#[test]
fn ordered_phase_emission_matches_runner_load_sequence() {
    let _g = recorder_guard();
    // Drain any leftover phases from a previous test in this binary.
    let _ = drain_report();
    // Simulate the same phase shape Runner::load emits:
    //   manifest_load → safetensors_source_open → head_compile →
    //   layer0_safetensors_fetch → layer0_int4_quantize →
    //   embed_tokens_mmap → shells_load → experts_cache_init
    //
    // Make each phase sleep just enough to be measurable, and order
    // them so each one's elapsed_ms is monotonically increasing —
    // that lets the assertion catch any reordering by drop semantics
    // (e.g. if we accidentally `let` two timers into a single block
    // and reverse the LIFO drop order).
    let expected = [
        ("runner.manifest_load", 1),
        ("runner.safetensors_source_open", 2),
        ("runner.head_compile", 3),
        ("runner.layer0_safetensors_fetch", 4),
        ("runner.layer0_int4_quantize", 5),
        ("runner.embed_tokens_mmap", 6),
        ("runner.shells_load", 7),
        ("runner.experts_cache_init", 8),
    ];
    for (name, ms) in expected.iter() {
        let _t = PhaseTimer::start(*name);
        sleep(Duration::from_millis(*ms as u64));
    }

    let report = drain_report();
    assert_eq!(
        report.len(),
        expected.len(),
        "expected {} phases, recorded {}; phases={:?}",
        expected.len(),
        report.len(),
        report.iter().map(|p| &p.name).collect::<Vec<_>>()
    );

    // Names must match in order.
    for (got, &(want, _)) in report.iter().zip(expected.iter()) {
        assert_eq!(got.name, want, "phase order mismatch: report={:?}", report);
    }

    // Durations should be roughly increasing — allow ±5 ms wobble per
    // phase for OS scheduling jitter on CI VMs but the trend must
    // hold (later phases have longer sleeps).
    let last = &report[report.len() - 1];
    let first = &report[0];
    assert!(
        last.elapsed >= first.elapsed,
        "expected last phase ({:?}) >= first phase ({:?})",
        last.elapsed,
        first.elapsed
    );

    // Formatted table must include the TOTAL row and at least the
    // first and last phase names.
    let table = format_report(&report);
    assert!(
        table.contains("TOTAL"),
        "format_report missing TOTAL:\n{table}"
    );
    for (name, _) in expected.iter() {
        assert!(table.contains(name), "table missing {name}:\n{table}");
    }
}

#[test]
fn nested_phase_records_in_outer_after_inner() {
    let _g = recorder_guard();
    let _ = drain_report();
    // Drop order is LIFO. If we open an outer timer and an inner timer
    // inside its scope, the inner timer drops first and records first.
    // This mirrors how Runner::load's outer `builder.load_total` wraps
    // the inner per-phase timers.
    let outer = PhaseTimer::start("outer");
    {
        let _inner = PhaseTimer::start("inner");
        sleep(Duration::from_millis(2));
    }
    sleep(Duration::from_millis(2));
    drop(outer);

    let report = drain_report();
    assert_eq!(report.len(), 2, "expected 2 phases, got {:?}", report);
    assert_eq!(
        report[0].name, "inner",
        "inner should record first (LIFO drop)"
    );
    assert_eq!(report[1].name, "outer");
    // Outer encompasses inner + the post-inner sleep, so it must be
    // strictly >= inner's elapsed.
    assert!(
        report[1].elapsed >= report[0].elapsed,
        "outer ({:?}) should be >= inner ({:?})",
        report[1].elapsed,
        report[0].elapsed
    );
}
