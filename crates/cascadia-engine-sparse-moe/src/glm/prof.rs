//! Per-token decode profiler for GLM-5.2, gated by `CASCADIA_GLM5_PROFILE`.
//!
//! Mirrors dsv4's `DSV4_PROFILE`. When the env var is unset, every entry point
//! here is a no-op (a cached-bool check), so it is safe to leave wired into the
//! hot path in release builds. When set, each decoded token prints, on stderr:
//!   * the cumulative per-section **wall-time** split (attn / router / experts /
//!     other) — how the decode budget divides, and
//!   * the **I/O + residency** counters that actually decide tok/s on this
//!     NVMe-streaming workload: bytes streamed at 0%-hit vs bytes that actually
//!     hit the disk, the derived expert-cache **hit %**, the achieved expert
//!     throughput (MB/s), and the cross-token expert **reuse %** (how much of
//!     one token's routed set repeats the last token's — the number that decides
//!     whether batched / speculative decode can amortise expert reads).
//!
//! Why these and not `mincore`: the disk-byte counter comes from the OS process
//! I/O accounting (`/proc/self/io` `read_bytes` on Linux, `GetProcessIoCounters`
//! on Windows), which counts only bytes that actually reached the block layer —
//! page-cache hits do not increment it. So `hit% = 1 − disk_bytes/routed_bytes`
//! needs no per-page residency probe and is portable across the fleet's OSes.
//! Where the OS counter is unavailable (e.g. the macOS dev host) the disk-based
//! fields print `n/a`, but the residency-independent numbers (routed GB/token,
//! effective MB/s, expert reuse) still work.
//!
//! Usage: `CASCADIA_GLM5_PROFILE=1 cascadia worker …` → `GLM5_PROF …` lines, one
//! dump per forwarded token (cumulative; the last line before teardown is the
//! whole-run total). In the pipeline each rank profiles its OWN layer slice and
//! its OWN process I/O, so read the split — and the hit % — per rank.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

// --- I/O + residency counters (cumulative; all gated by `enabled()`) ----------
/// Bytes we WOULD stream at 0% cache hit: Σ on-disk int4 size of every routed +
/// shared expert touched. The residency-independent denominator.
static ROUTED_BYTES: AtomicU64 = AtomicU64::new(0);
/// Bytes that actually reached the disk (OS process read-counter deltas). The
/// gap between this and `ROUTED_BYTES` is what the page cache / pins absorbed.
static DISK_BYTES: AtomicU64 = AtomicU64::new(0);
/// Decoded tokens seen at a dump boundary (the per-token averaging base).
static TOKENS: AtomicU64 = AtomicU64::new(0);
/// Σ routed-expert selections (top_k × layers × tokens on this rank).
static SELECTIONS: AtomicU64 = AtomicU64::new(0);
/// Cross-token routed-set overlap, accumulated as intersection / union of
/// consecutive tokens' `(layer,expert)` sets — a running Jaccard of expert reuse.
static OVL_INTER: AtomicU64 = AtomicU64::new(0);
static OVL_UNION: AtomicU64 = AtomicU64::new(0);
/// Last OS read-byte sample (`u64::MAX` = not yet sampled → first delta skipped,
/// so model-load reads before the first token are excluded).
static LAST_IO: AtomicU64 = AtomicU64::new(u64::MAX);
/// Whether the OS read-byte counter is available on this platform/process.
static IO_OK: AtomicBool = AtomicBool::new(false);

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
}

/// Process cumulative disk-read bytes, or `None` where the OS doesn't expose it.
/// Linux: `/proc/self/io` `read_bytes` (block-layer bytes; cache hits excluded).
/// Windows: `GetProcessIoCounters().ReadTransferCount`.
#[cfg(target_os = "linux")]
fn proc_read_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/io").ok()?;
    s.lines().find_map(|l| {
        l.strip_prefix("read_bytes:")
            .and_then(|v| v.trim().parse::<u64>().ok())
    })
}
#[cfg(windows)]
fn proc_read_bytes() -> Option<u64> {
    #[repr(C)]
    struct IoCounters {
        read_ops: u64,
        write_ops: u64,
        other_ops: u64,
        read_bytes: u64,
        write_bytes: u64,
        other_bytes: u64,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessIoCounters(process: isize, counters: *mut IoCounters) -> i32;
    }
    let mut c = IoCounters {
        read_ops: 0,
        write_ops: 0,
        other_ops: 0,
        read_bytes: 0,
        write_bytes: 0,
        other_bytes: 0,
    };
    // SAFETY: `GetProcessIoCounters` writes the six u64 fields of `c` for the
    // current-process pseudo handle; nonzero return means the struct is valid.
    unsafe {
        if GetProcessIoCounters(GetCurrentProcess(), &mut c) != 0 {
            Some(c.read_bytes)
        } else {
            None
        }
    }
}
#[cfg(not(any(target_os = "linux", windows)))]
fn proc_read_bytes() -> Option<u64> {
    None
}

/// Sample the OS read counter and fold the per-token delta into `DISK_BYTES`;
/// roll this token's routed set into the reuse accumulators. Called once per
/// token, from [`dump`].
fn roll_token() {
    if let Some(cur) = proc_read_bytes() {
        IO_OK.store(true, Ordering::Relaxed);
        let prev = LAST_IO.swap(cur, Ordering::Relaxed);
        if prev != u64::MAX && cur >= prev {
            DISK_BYTES.fetch_add(cur - prev, Ordering::Relaxed);
        }
    }
    if let (Ok(mut c), Ok(mut p)) = (cur_set().lock(), prev_set().lock()) {
        let inter = c.iter().filter(|k| p.contains(*k)).count() as u64;
        let uni = (c.len() + p.len()) as u64 - inter;
        OVL_INTER.fetch_add(inter, Ordering::Relaxed);
        OVL_UNION.fetch_add(uni, Ordering::Relaxed);
        *p = std::mem::take(&mut *c); // prev <- cur, cur <- empty
    }
    TOKENS.fetch_add(1, Ordering::Relaxed);
}

/// Print the cumulative per-section split as a share of WALL, plus the I/O /
/// residency counters (no-op when disabled). `attn + router + experts` are
/// subsets of each layer's wall; the remainder ("other") is
/// rmsnorm/rope/residual and the dense-layer MLPs.
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

    // I/O + residency. `experts` wall-time is the denominator for MB/s (the
    // expert FFNs are where the streaming happens).
    let routed = ROUTED_BYTES.load(Ordering::Relaxed);
    let disk = DISK_BYTES.load(Ordering::Relaxed);
    let toks = TOKENS.load(Ordering::Relaxed).max(1);
    let exp_s = (NS[EXPERTS].load(Ordering::Relaxed) as f64 / 1e9).max(1e-9); // ns -> s
    let routed_gb = routed as f64 / 1e9;
    let eff_mbs = routed as f64 / 1e6 / exp_s;
    if IO_OK.load(Ordering::Relaxed) {
        let disk_gb = disk as f64 / 1e9;
        let hit = if routed > 0 {
            (100.0 * (1.0 - disk as f64 / routed as f64)).clamp(0.0, 100.0)
        } else {
            0.0
        };
        let disk_mbs = disk as f64 / 1e6 / exp_s;
        eprintln!(
            "  io       routed={:.2} GB ({:.1} MB/tok)  disk={:.2} GB  hit={:.1}%  eff={:.0} MB/s  disk={:.0} MB/s",
            routed_gb,
            routed as f64 / 1e6 / toks as f64,
            disk_gb,
            hit,
            eff_mbs,
            disk_mbs,
        );
    } else {
        eprintln!(
            "  io       routed={:.2} GB ({:.1} MB/tok)  disk=n/a (no OS counter)  eff={:.0} MB/s",
            routed_gb,
            routed as f64 / 1e6 / toks as f64,
            eff_mbs,
        );
    }
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
}
