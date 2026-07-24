//! Per-token decode profiler for GLM-5.2, gated by `CASCADIA_GLM5_PROFILE`.
//!
//! Mirrors dsv4's `DSV4_PROFILE`. When the env var is unset, every entry point
//! here is a no-op (a cached-bool check), so it is safe to leave wired into the
//! hot path in release builds. When set, each decoded token prints, on stderr:
//!   * the cumulative per-section **wall-time** split (attn / router / experts /
//!     other) — how the decode budget divides, and
//!   * the **residency** counters that actually decide tok/s on this
//!     NVMe-streaming workload: bytes streamed at 0%-hit (routed GB/tok), the
//!     achieved expert throughput (eff MB/s = routed_bytes / experts-time), the
//!     true expert-cache **hit %** (fraction of routed experts' pages resident in
//!     RAM, probed directly via the working set), and cross-token routed-expert
//!     **reuse %** (how much of one token's routed set repeats the last token's —
//!     the number that decides whether batched / speculative decode can amortise
//!     expert reads).
//!
//! Why a working-set probe and not a process I/O counter: mmap page faults are
//! not billed to the process read counter on Windows (paging I/O is charged
//! elsewhere), so a `read_bytes`-style delta reads ~0 and reports a bogus 100%
//! hit. Probing page residency directly (`QueryWorkingSetEx` on Windows,
//! `mincore` on unix) gives the real resident fraction on every fleet OS.
//!
//! Usage: `CASCADIA_GLM5_PROFILE=1 cascadia worker …` → `GLM5_PROF …` lines, one
//! dump per forwarded token (cumulative; the last line before teardown is the
//! whole-run total). In the pipeline each rank profiles its OWN layer slice, so
//! read the split — and the hit % — per rank.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
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

// --- residency counters (cumulative; all gated by `enabled()`) ----------------
/// Bytes we WOULD stream at 0% cache hit: Σ on-disk int4 size of every routed +
/// shared expert touched. The residency-independent denominator.
static ROUTED_BYTES: AtomicU64 = AtomicU64::new(0);
/// Working-set probe totals: Σ resident pages / Σ probed pages across routed +
/// shared experts. hit% = RESIDENT_PAGES / PROBED_PAGES.
static RESIDENT_PAGES: AtomicU64 = AtomicU64::new(0);
static PROBED_PAGES: AtomicU64 = AtomicU64::new(0);
/// Decoded tokens seen at a dump boundary (the per-token averaging base).
static TOKENS: AtomicU64 = AtomicU64::new(0);
/// Σ routed-expert selections (top_k × layers × tokens on this rank).
static SELECTIONS: AtomicU64 = AtomicU64::new(0);
/// Cross-token routed-set overlap, accumulated as intersection / union of
/// consecutive tokens' `(layer,expert)` sets — a running Jaccard of expert reuse.
static OVL_INTER: AtomicU64 = AtomicU64::new(0);
static OVL_UNION: AtomicU64 = AtomicU64::new(0);

// --- LOOKAHEAD recall counters (measurement spike; gated by `enabled()`) ----------
/// Routed selections this rank made that a next-layer LOOKAHEAD prediction had
/// already named (the hits), over all selections whose layer WAS predicted (the
/// total). recall = LOOKAHEAD_HIT / LOOKAHEAD_TOTAL — how often predicting layer L+1's
/// experts from `rmsnorm(out_L, post_ln_{L+1})` (attention-free proxy) names the
/// experts L+1 actually fires. This is the number that gates whether real-router
/// prefetch can convert misses to hits at all on this workload.
static LOOKAHEAD_HIT: AtomicU64 = AtomicU64::new(0);
static LOOKAHEAD_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Per-token `layer -> predicted routed-expert set`, keyed by the layer's GLOBAL
/// index (same key `note_selection` uses), so a prediction issued for layer L+1
/// during layer L's step is matched when L+1 actually selects. Cleared each token
/// in [`roll_token`].
fn predicted() -> &'static Mutex<HashMap<u32, HashSet<u32>>> {
    static S: OnceLock<Mutex<HashMap<u32, HashSet<u32>>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record LOOKAHEAD's predicted routed-expert set for `layer` (global index) this
/// token (no-op when disabled). Called from the prefetch site before `layer`
/// computes; matched against the real selections in [`note_selection`].
pub fn note_predicted(layer: u32, experts: &[u32]) {
    if !enabled() {
        return;
    }
    if let Ok(mut m) = predicted().lock() {
        m.insert(layer, experts.iter().copied().collect());
    }
}

/// This token's routed `(layer,expert)` set (key = `layer<<32 | expert`), rolled
/// into `PREV_SET` at each dump to measure cross-token reuse.
fn cur_set() -> &'static Mutex<HashSet<u64>> {
    static S: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashSet::new()))
}
fn prev_set() -> &'static Mutex<HashSet<u64>> {
    static S: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashSet::new()))
}

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

/// Record the on-disk int4 size of one expert touched this token (no-op when
/// disabled). The 0%-hit streaming baseline.
#[inline]
pub fn note_expert_bytes(bytes: usize) {
    if enabled() {
        ROUTED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

/// Record a working-set residency sample `(resident, probed)` for one expert
/// (no-op when disabled). Probe BEFORE the expert is computed this token, so it
/// reflects whether the access hit RAM.
#[inline]
pub fn note_residency(resident: usize, probed: usize) {
    if enabled() {
        RESIDENT_PAGES.fetch_add(resident as u64, Ordering::Relaxed);
        PROBED_PAGES.fetch_add(probed as u64, Ordering::Relaxed);
    }
}

/// Record one routed selection `(layer,expert)` this token (no-op when disabled).
/// Feeds the selection count and the cross-token reuse set.
#[inline]
pub fn note_selection(layer: u32, expert: u32) {
    if !enabled() {
        return;
    }
    SELECTIONS.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut s) = cur_set().lock() {
        s.insert(((layer as u64) << 32) | expert as u64);
    }
    // LOOKAHEAD recall: only score layers that carried a prediction this token, so a
    // baseline run (no LOOKAHEAD) leaves both counters at 0 (recall=n/a).
    if let Ok(p) = predicted().lock() {
        if let Some(set) = p.get(&layer) {
            LOOKAHEAD_TOTAL.fetch_add(1, Ordering::Relaxed);
            if set.contains(&expert) {
                LOOKAHEAD_HIT.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Roll this token's routed set into the reuse accumulators. Called once per
/// token, from [`dump`].
fn roll_token() {
    if let (Ok(mut c), Ok(mut p)) = (cur_set().lock(), prev_set().lock()) {
        let inter = c.iter().filter(|k| p.contains(*k)).count() as u64;
        let uni = (c.len() + p.len()) as u64 - inter;
        OVL_INTER.fetch_add(inter, Ordering::Relaxed);
        OVL_UNION.fetch_add(uni, Ordering::Relaxed);
        *p = std::mem::take(&mut *c); // prev <- cur, cur <- empty
    }
    // Predictions are per-token; drop them so next token starts clean.
    if let Ok(mut m) = predicted().lock() {
        m.clear();
    }
    TOKENS.fetch_add(1, Ordering::Relaxed);
}

/// Print the cumulative per-section split as a share of WALL, plus the residency
/// counters (no-op when disabled). `attn + router + experts` are subsets of each
/// layer's wall; the remainder ("other") is rmsnorm/rope/residual and the
/// dense-layer MLPs.
pub fn dump(tag: &str) {
    if !enabled() {
        return;
    }
    roll_token();
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

    // Residency. `experts` wall-time is the denominator for MB/s (the expert FFNs
    // are where the streaming happens).
    let routed = ROUTED_BYTES.load(Ordering::Relaxed);
    let toks = TOKENS.load(Ordering::Relaxed).max(1);
    let exp_s = (NS[EXPERTS].load(Ordering::Relaxed) as f64 / 1e9).max(1e-9); // ns -> s
    let eff_mbs = routed as f64 / 1e6 / exp_s;
    let probed = PROBED_PAGES.load(Ordering::Relaxed);
    let hit_str = if probed > 0 {
        let hit = 100.0 * RESIDENT_PAGES.load(Ordering::Relaxed) as f64 / probed as f64;
        format!("hit={hit:.1}%")
    } else {
        "hit=n/a".to_string()
    };
    eprintln!(
        "  io       routed={:.2} GB ({:.1} MB/tok)  {}  eff={:.0} MB/s",
        routed as f64 / 1e9,
        routed as f64 / 1e6 / toks as f64,
        hit_str,
        eff_mbs,
    );
    let sel = SELECTIONS.load(Ordering::Relaxed);
    let inter = OVL_INTER.load(Ordering::Relaxed);
    let union = OVL_UNION.load(Ordering::Relaxed).max(1);
    eprintln!(
        "  moe      sel={}  ({:.1}/tok)  reuse={:.1}%  tokens={}",
        sel,
        sel as f64 / toks as f64,
        100.0 * inter as f64 / union as f64,
        toks,
    );
    // Lookahead recall (only meaningful under CASCADIA_GLM5_LOOKAHEAD; n/a otherwise).
    let ph = LOOKAHEAD_HIT.load(Ordering::Relaxed);
    let pt = LOOKAHEAD_TOTAL.load(Ordering::Relaxed);
    if pt > 0 {
        eprintln!(
            "  lookahead    recall={:.1}%  ({ph}/{pt} predicted-and-fired)",
            100.0 * ph as f64 / pt as f64,
        );
    }
}
