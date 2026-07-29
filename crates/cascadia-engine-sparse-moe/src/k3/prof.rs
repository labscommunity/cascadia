//! Per-token decode profiler for K3, gated by `CASCADIA_K3_PROFILE`.
//!
//! Mirrors glm5's `CASCADIA_GLM5_PROFILE`. Every entry point is a cached-bool
//! check when the env var is unset, so it is safe to leave wired into the hot
//! path.
//!
//! K3-specific sections. The `attnres` bucket exists because Attention
//! Residuals are new to this family and nobody has cost data for them: the
//! mixture runs TWICE per layer over a stack that grows to 8 entries, so at 93
//! layers that is 186 softmaxes over up to 9x7168 floats per token. If it turns
//! out to be material, that is worth knowing before optimising anything else.
//!
//! Residency is reported from a direct page probe, not an I/O counter: expert
//! bytes arrive via mmap page faults, which are not billed to the process read
//! counter, so a `read_bytes`-style delta reads ~0 and reports a bogus 100% hit.
//!
//! Usage: `CASCADIA_K3_PROFILE=1 cascadia worker …` → cumulative `K3_PROF …`
//! lines on stderr, one per forwarded token. In a pipeline each rank profiles
//! its OWN layer slice, so read the split per rank.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub const KDA: usize = 0; // linear-attention layers: conv + gate + recurrence
pub const MLA: usize = 1; // full-attention layers: proj + softmax + output gate
pub const ATTNRES: usize = 2; // the two per-layer AttnRes mixtures
pub const ROUTER: usize = 3; // gate logits + noaux_tc top-k
pub const EXPERTS: usize = 4; // routed + shared FFNs (streams fp4 weights)
pub const WALL: usize = 5; // whole per-token decode — the denominator
const N: usize = 6;
const NAMES: [&str; N] = ["kda", "mla", "attnres", "router", "experts", "WALL"];

static NS: [AtomicU64; N] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static TOKENS: AtomicU64 = AtomicU64::new(0);
/// Bytes of routed-expert weight touched, at 0% cache hit.
static ROUTED_BYTES: AtomicU64 = AtomicU64::new(0);
/// Resident / probed pages across the routed experts this run.
static PAGES_RESIDENT: AtomicU64 = AtomicU64::new(0);
static PAGES_PROBED: AtomicU64 = AtomicU64::new(0);
/// Routed experts repeated from the previous token (amortisation headroom).
static REUSED: AtomicU64 = AtomicU64::new(0);
static SELECTED: AtomicU64 = AtomicU64::new(0);

fn last_set() -> &'static Mutex<HashSet<(u32, u32)>> {
    static S: OnceLock<Mutex<HashSet<(u32, u32)>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Cached once — an unset env var makes every hook below a branch and nothing else.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CASCADIA_K3_PROFILE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
    })
}

#[inline]
pub fn add(bucket: usize, since: Instant) {
    if enabled() {
        NS[bucket].fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Record the routed experts this token selected, for reuse and IO accounting.
pub fn record_routing(layer: u32, experts: &[u32], expert_bytes: usize) {
    if !enabled() {
        return;
    }
    ROUTED_BYTES.fetch_add((experts.len() * expert_bytes) as u64, Ordering::Relaxed);
    SELECTED.fetch_add(experts.len() as u64, Ordering::Relaxed);
    let mut prev = last_set().lock().unwrap();
    for &e in experts {
        if prev.contains(&(layer, e)) {
            REUSED.fetch_add(1, Ordering::Relaxed);
        }
    }
    for &e in experts {
        prev.insert((layer, e));
    }
}

/// Fold in a residency sample: how many of `probed` pages were RAM-resident.
pub fn record_residency(resident: usize, probed: usize) {
    if enabled() {
        PAGES_RESIDENT.fetch_add(resident as u64, Ordering::Relaxed);
        PAGES_PROBED.fetch_add(probed as u64, Ordering::Relaxed);
    }
}

/// Clear the cross-token reuse window (call on reset / new sequence).
pub fn new_sequence() {
    if enabled() {
        last_set().lock().unwrap().clear();
    }
}

/// Dump cumulative counters. Called once per forwarded token.
pub fn dump(tag: &str) {
    if !enabled() {
        return;
    }
    let toks = TOKENS.fetch_add(1, Ordering::Relaxed) + 1;
    let wall = NS[WALL].load(Ordering::Relaxed).max(1);
    let mut split = String::new();
    for i in 0..N {
        let ns = NS[i].load(Ordering::Relaxed);
        split.push_str(&format!(
            " {}={:.1}ms({:.0}%)",
            NAMES[i],
            ns as f64 / 1e6 / toks as f64,
            100.0 * ns as f64 / wall as f64
        ));
    }

    let bytes = ROUTED_BYTES.load(Ordering::Relaxed);
    let gb_tok = bytes as f64 / 1e9 / toks as f64;
    let exp_ns = NS[EXPERTS].load(Ordering::Relaxed).max(1);
    let eff_mbs = bytes as f64 / 1e6 / (exp_ns as f64 / 1e9);

    let probed = PAGES_PROBED.load(Ordering::Relaxed);
    let hit = if probed > 0 {
        100.0 * PAGES_RESIDENT.load(Ordering::Relaxed) as f64 / probed as f64
    } else {
        f64::NAN
    };
    let sel = SELECTED.load(Ordering::Relaxed).max(1);
    let reuse = 100.0 * REUSED.load(Ordering::Relaxed) as f64 / sel as f64;

    eprintln!(
        "K3_PROF [{tag}] tok={toks}{split} | routed={gb_tok:.2}GB/tok \
         eff={eff_mbs:.0}MB/s hit={hit:.1}% reuse={reuse:.1}%"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_hooks_are_inert() {
        // With the env var unset nothing is recorded and dump prints nothing.
        assert!(!enabled(), "CASCADIA_K3_PROFILE must be unset under test");
        let before = NS[WALL].load(Ordering::Relaxed);
        add(WALL, Instant::now());
        record_routing(0, &[1, 2, 3], 1024);
        record_residency(1, 2);
        dump("test");
        assert_eq!(NS[WALL].load(Ordering::Relaxed), before);
        assert_eq!(ROUTED_BYTES.load(Ordering::Relaxed), 0);
    }
}
