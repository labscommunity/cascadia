//! Opt-in per-token profiling for the qwen35 staged engine (`CASCADIA_QWEN36_PROFILE=1`).
//!
//! The engine records, per decode step and on every rank, the microseconds spent in each phase
//! it controls — embedding lookup (rank 0 / single-box), each local stage's `set_input` / `infer`
//! / `output` copy, the blocking transport waits (rank 0: the downstream round trip and its wire
//! share; relays: the wait for the next upstream frame and their own downstream round trip),
//! argmax sampling, and the GPU keep-alive's synthetic infers — and logs a `key=p50/p95us`
//! summary every [`SUMMARY_EVERY`] decode steps and at task end. With the stages compiled under
//! `--ov-config PERF_COUNT=YES`, one per-stage OpenVINO per-op summary is logged after the
//! [`PERF_DUMP_AFTER_STEPS`]th decode step of a task (the same aggregation `runtime.rs`'s
//! `dump_decode_profile` prints for the single-model engine).
//!
//! The engine reads the variable once at build; when it is unset the accumulators are `None`
//! and no `Instant::now()` runs on the hot path, so the unprofiled engine is byte-identical.
//!
//! Log keys (µs): `embed`, `s<j>.set` / `s<j>.infer` / `s<j>.out` per local stage `j`,
//! `down.rtt` / `down.wire` (rank 0 and middle ranks: FORWARD→TOKEN round trip and RTT minus the
//! peer-reported infer time), `wire.net` (rank 0: the chain's wire share from the step's RTT),
//! `up.wait` (relays: blocked on the upstream frame), `sample`, `ka.infer` (keep-alive).

use std::collections::BTreeMap;

/// Profiling env var; `1` enables it, anything else leaves the engine unprofiled.
pub const PROFILE_ENV: &str = "CASCADIA_QWEN36_PROFILE";

/// Decode steps between two `qwen36 profile (p50/p95 per decode step)` summary lines.
pub const SUMMARY_EVERY: usize = 32;

/// The decode step after which the one-off `PERF_COUNT` per-op dump fires (late enough that
/// every stage has run its steady-state decode kernels, early enough to see it on a short task).
pub const PERF_DUMP_AFTER_STEPS: usize = 8;

/// Node types listed in the per-stage OpenVINO per-op summary.
pub const OV_PROFILE_TOP_TYPES: usize = 12;

/// `CASCADIA_QWEN36_PROFILE=1` in this process' environment.
pub fn profile_enabled() -> bool {
    std::env::var(PROFILE_ENV).ok().as_deref() == Some("1")
}

/// `p`-th percentile of an ASCENDING sample list (nearest rank; `p` in `[0, 1]`).
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p.clamp(0.0, 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Named sample series (microseconds) behind the per-step summaries.
#[derive(Clone, Debug, Default)]
pub struct ProfileAcc {
    samples: BTreeMap<String, Vec<f64>>,
}

impl ProfileAcc {
    pub fn record(&mut self, key: &str, us: f64) {
        match self.samples.get_mut(key) {
            Some(v) => v.push(us),
            None => {
                self.samples.insert(key.to_string(), vec![us]);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Samples recorded under `key`.
    pub fn count(&self, key: &str) -> usize {
        self.samples.get(key).map_or(0, Vec::len)
    }

    /// `key=p50/p95us` per series, space-separated, keys sorted.
    pub fn summary(&self) -> String {
        let mut parts = Vec::with_capacity(self.samples.len());
        for (k, v) in &self.samples {
            let mut s = v.clone();
            s.sort_by(f64::total_cmp);
            parts.push(format!(
                "{k}={:.0}/{:.0}us",
                percentile(&s, 0.50),
                percentile(&s, 0.95)
            ));
        }
        parts.join(" ")
    }
}

/// Per-step timing state the engine carries while profiling is on.
#[derive(Debug, Default)]
pub struct ProfileState {
    pub acc: ProfileAcc,
    /// Decode steps since the last summary line.
    pub since_summary: usize,
    /// Decode steps of the current task (rank 0 / single-box) or since the last RESET (relays).
    pub task_steps: usize,
    /// The last chain pass's per-local-stage `(set_input, infer, output)` µs, folded into `acc`
    /// by the step hook (a prefill chunk's pass is overwritten by the next, never recorded).
    pub stage_last: Vec<(u64, u64, u64)>,
    /// The last `embed_seq` µs (rank 0 / single-box); 0 = none pending.
    pub embed_last_us: u64,
    /// The `PERF_COUNT` dump already ran for this task.
    pub perf_dumped: bool,
}

impl ProfileState {
    /// Fold the pending embed / per-stage timings and `extra` samples into the accumulator and
    /// count the step. Returns `(summary_due, perf_dump_due)`: whether a periodic summary line
    /// is owed (the counter is reset here) and whether the one-off `PERF_COUNT` dump should fire
    /// now (`perf_count` says the stages were compiled for it; marked done here).
    pub fn end_step(&mut self, extra: &[(&str, f64)], perf_count: bool) -> (bool, bool) {
        if self.embed_last_us > 0 {
            self.acc.record("embed", self.embed_last_us as f64);
            self.embed_last_us = 0;
        }
        for (j, (set, infer, out)) in std::mem::take(&mut self.stage_last).iter().enumerate() {
            self.acc.record(&format!("s{j}.set"), *set as f64);
            self.acc.record(&format!("s{j}.infer"), *infer as f64);
            self.acc.record(&format!("s{j}.out"), *out as f64);
        }
        for (k, v) in extra {
            self.acc.record(k, *v);
        }
        self.since_summary += 1;
        self.task_steps += 1;
        let summary_due = self.since_summary >= SUMMARY_EVERY;
        if summary_due {
            self.since_summary = 0;
        }
        let dump_due = perf_count && !self.perf_dumped && self.task_steps >= PERF_DUMP_AFTER_STEPS;
        if dump_due {
            self.perf_dumped = true;
        }
        (summary_due, dump_due)
    }

    /// Clear everything at a task boundary (the caller logs the final summary first).
    pub fn reset(&mut self) {
        self.acc.clear();
        self.since_summary = 0;
        self.task_steps = 0;
        self.perf_dumped = false;
        self.stage_last.clear();
        self.embed_last_us = 0;
    }
}

/// Summarise one `Runtime::profiling()` TSV (`node_name\tnode_type\texec_type\treal_us\tcpu_us`
/// per line): total real vs cpu time, nodes with non-zero real time ("executed"), and the top
/// `top` node types by real time — one line per row.
pub fn summarize_ov_profile(tsv: &str, top: usize) -> Vec<String> {
    let mut by_type: BTreeMap<&str, (u64, u64, u32)> = BTreeMap::new();
    let (mut total_real, mut total_cpu, mut executed, mut nodes) = (0u64, 0u64, 0u32, 0u32);
    for line in tsv.lines() {
        let mut it = line.split('\t');
        let (_name, ntype) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
        let _exec = it.next();
        let real = it.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        let cpu = it.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        if ntype.is_empty() && real == 0 && cpu == 0 {
            continue;
        }
        nodes += 1;
        total_real += real;
        total_cpu += cpu;
        if real > 0 {
            executed += 1;
        }
        let e = by_type.entry(ntype).or_default();
        e.0 += real;
        e.1 += cpu;
        e.2 += 1;
    }
    let mut rows: Vec<_> = by_type.into_iter().collect();
    rows.sort_by_key(|(_, (real, _, _))| std::cmp::Reverse(*real));
    let mut out = vec![format!(
        "total real {total_real} us, cpu {total_cpu} us; executed {executed} of {nodes} nodes"
    )];
    for (ntype, (real, cpu, count)) in rows.iter().take(top) {
        let share = if total_real > 0 {
            100.0 * *real as f64 / total_real as f64
        } else {
            0.0
        };
        out.push(format!(
            "{ntype:<28} real {real:>9} us ({share:>5.1}%)  cpu {cpu:>9} us  x{count}"
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_nearest_rank() {
        let s = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&s, 0.5), 6.0);
        assert_eq!(percentile(&s, 0.95), 10.0);
        assert_eq!(percentile(&s, 0.0), 1.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn summary_lists_sorted_keys_with_p50_p95() {
        let mut acc = ProfileAcc::default();
        for i in 1..=20 {
            acc.record("s0.infer", i as f64 * 100.0);
        }
        acc.record("embed", 50.0);
        let s = acc.summary();
        assert!(s.starts_with("embed=50/50us "), "{s}");
        assert!(s.contains("s0.infer=1100/1900us"), "{s}");
        assert_eq!(acc.count("s0.infer"), 20);
        acc.clear();
        assert!(acc.is_empty());
    }

    /// The step hook folds the pending embed and per-stage timings exactly once, counts the
    /// step, owes a summary every `SUMMARY_EVERY` steps and the PERF_COUNT dump once per task.
    #[test]
    fn end_step_folds_pending_timings_and_paces_summaries() {
        let mut p = ProfileState::default();
        p.embed_last_us = 120;
        p.stage_last = vec![(10, 33_000, 70), (12, 31_000, 80)];
        let (summary, dump) = p.end_step(&[("sample", 300.0)], true);
        assert!(!summary && !dump);
        assert_eq!(p.acc.count("embed"), 1);
        assert_eq!(p.acc.count("s1.infer"), 1);
        assert_eq!(p.acc.count("sample"), 1);
        assert_eq!(p.embed_last_us, 0, "folded once");
        assert!(p.stage_last.is_empty(), "folded once");
        // Steps 2..8: the dump fires on the 8th step, exactly once.
        let mut dumps = 0;
        for _ in 1..PERF_DUMP_AFTER_STEPS {
            if p.end_step(&[], true).1 {
                dumps += 1;
            }
        }
        assert_eq!(dumps, 1);
        assert_eq!(p.task_steps, PERF_DUMP_AFTER_STEPS);
        assert!(!p.end_step(&[], true).1, "dumped once per task");
        // The summary is owed on the 32nd step and the counter restarts.
        let mut summaries = 0;
        while p.task_steps < 2 * SUMMARY_EVERY {
            if p.end_step(&[], true).0 {
                summaries += 1;
            }
        }
        assert_eq!(summaries, 2);
        assert_eq!(p.since_summary, 0);
        // Stages compiled without PERF_COUNT never ask for the dump.
        let mut q = ProfileState::default();
        for _ in 0..2 * PERF_DUMP_AFTER_STEPS {
            assert!(!q.end_step(&[], false).1);
        }
        p.reset();
        assert!(p.acc.is_empty());
        assert_eq!((p.task_steps, p.since_summary), (0, 0));
        assert!(!p.perf_dumped);
    }

    #[test]
    fn ov_profile_summary_totals_and_top_types() {
        let tsv = "a\tMatMul\tjit\t500\t450\nb\tMatMul\tjit\t300\t280\nc\tSoftmax\tref\t100\t90\n\
                   d\tReshape\tunknown\t0\t0\n";
        let rows = summarize_ov_profile(tsv, 2);
        assert_eq!(
            rows[0],
            "total real 900 us, cpu 820 us; executed 3 of 4 nodes"
        );
        assert_eq!(rows.len(), 3, "header + top 2");
        assert!(rows[1].starts_with("MatMul"), "{}", rows[1]);
        assert!(rows[1].contains("800 us ( 88.9%)"), "{}", rows[1]);
        assert!(rows[2].starts_with("Softmax"), "{}", rows[2]);
        assert_eq!(summarize_ov_profile("", 5).len(), 1, "header only");
    }
}
