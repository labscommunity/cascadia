//! Per-section decode profiler for GLM-5.2, gated by `CASCADIA_GLM5_PROFILE`.
//!
//! Mirrors dsv4's `DSV4_PROFILE`. When the env var is unset, [`add`]/[`dump`] are
//! no-ops (a cached-bool check plus an unused `Instant`), so it is safe to leave
//! wired into the hot path in release builds. When set, each decoded token
//! prints the cumulative per-section wall-time split on stderr — how the decode
//! budget divides across attention, MoE routing, and the (I/O-bound) expert
//! FFNs, which is the first thing you want when diagnosing tok/s on this
//! NVMe-streaming workload.
//!
//! Usage: `CASCADIA_GLM5_PROFILE=1 cascadia worker …` → `GLM5_PROF …` lines, one
//! dump per forwarded token (cumulative; the last line before teardown is the
//! whole-run total). In the pipeline each rank profiles its OWN layer slice, so
//! read the split per rank.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

pub const ATTN: usize = 0; // attention: proj GEMVs + DSA indexer + sparse softmax
pub const ROUTER: usize = 1; // MoE gate: router logits + noaux_tc top-k select
pub const EXPERTS: usize = 2; // routed + shared expert FFNs (streams int4 weights)
pub const WALL: usize = 3; // whole per-layer decode — the denominator
const N: usize = 4;
const NAMES: [&str; N] = ["attn", "router", "experts", "WALL"];
static NS: [AtomicU64; N] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Cached `CASCADIA_GLM5_PROFILE` check (read once).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CASCADIA_GLM5_PROFILE").is_ok())
}

/// Accumulate `since.elapsed()` into `bucket` (no-op when disabled).
#[inline]
pub fn add(bucket: usize, since: Instant) {
    if enabled() {
        NS[bucket].fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Print the cumulative per-section split as a share of WALL (no-op when
/// disabled). `attn + router + experts` are subsets of each layer's wall; the
/// remainder ("other") is rmsnorm/rope/residual and the dense-layer MLPs.
pub fn dump(tag: &str) {
    if !enabled() {
        return;
    }
    let wall = NS[WALL].load(Ordering::Relaxed).max(1);
    let mut summed = 0u64;
    eprintln!("GLM5_PROF {tag} wall_ms={:.1}", wall as f64 / 1e6);
    for i in 0..WALL {
        let v = NS[i].load(Ordering::Relaxed);
        summed += v;
        eprintln!(
            "  {:8} {:9.1} ms  {:4.1}%",
            NAMES[i],
            v as f64 / 1e6,
            v as f64 / wall as f64 * 100.0
        );
    }
    let resid = wall.saturating_sub(summed);
    eprintln!(
        "  {:8} {:9.1} ms  {:4.1}%  (norm/rope/residual/dense-mlp)",
        "other",
        resid as f64 / 1e6,
        resid as f64 / wall as f64 * 100.0
    );
}
