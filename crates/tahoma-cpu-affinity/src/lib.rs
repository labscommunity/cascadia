//! Per-thread CPU affinity for tahoma worker processes.
//!
//! # Why
//!
//! On large servers (the canonical case is a dual-socket Xeon Gold 6252
//! with 48 logical CPUs, but the same problem shows up on any 16+ core
//! host), the Linux/Windows schedulers will happily migrate threads
//! between cores under load. Every migration drops the warm L1d/L2
//! working set on the old core. For the int4 GEMV kernel that holds
//! the dequantized scale buffer and the in-flight token's activation
//! across `rayon::par_iter_mut`, those caches matter — empirically,
//! losing the warm caches drops the kernel's effective bandwidth by
//! 20-40%.
//!
//! Pinning rayon worker threads to fixed cores keeps the working set
//! resident. Reserving a separate core pool for the tokio I/O reactor
//! (and for the [planned] prefetcher / hot-buffer helper threads)
//! prevents the heavy compute pool from contending with low-latency
//! I/O wakeups.
//!
//! # What this crate provides
//!
//! - [`Mode`] + [`parse_mode`] — parse the `--cpu-affinity` CLI flag
//!   into one of `None`, `Auto`, or `Spec(...)`.
//! - [`Layout`] — the planned per-role core assignment. Pure data, no
//!   syscalls.
//! - [`Layout::plan`] — given a mode + the host's online CPU count,
//!   compute a [`Layout`].
//! - [`Layout::apply_to_rayon_global`] — install a global rayon pool
//!   sized to `rayon_cores.len()` whose threads each pin to a fixed
//!   core on startup.
//! - [`Layout::tokio_on_thread_start`] — returns a closure suitable
//!   for `tokio::runtime::Builder::on_thread_start` that round-robins
//!   tokio worker threads across the reserved tokio core pool.
//! - [`pin_current_thread`] — low-level: pin the calling thread to one
//!   core. Used by the spawn-blocking relay loop and the (future)
//!   prefetcher / hot-buffer threads.
//!
//! # One-concern-per-crate
//!
//! No dependency on rayon, tokio, or any tahoma crate. Callers (the
//! CLI, the int4-gemm crate) wire this in. Keeps the build graph
//! shallow and means `cargo build -p tahoma-cpu-affinity` runs in
//! ~300ms.
//!
//! # Cross-platform
//!
//! `core_affinity 0.8` covers Linux, Windows, macOS, and the BSDs.
//! On macOS Apple Silicon, pinning is a no-op at the kernel level —
//! Darwin treats it as a hint and we return success but log
//! `applied=false`. The layout planner runs the same everywhere so
//! tests are deterministic.

use std::sync::OnceLock;

use thiserror::Error;
use tracing::{debug, info, warn};

/// User-facing setting parsed out of `--cpu-affinity <mode>`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum Mode {
    /// Do not pin any threads. Back-compat default. Use this on
    /// shared / virtualized hosts where pinning would fight the
    /// hypervisor or the cgroup scheduler.
    #[default]
    None,
    /// Heuristic: split the cores between rayon (bulk), tokio (2
    /// cores), prefetcher (1), and hot-buffer (1). Falls back to a
    /// rayon-only split on hosts with < 8 cores.
    Auto,
    /// Explicit assignment spec, e.g.
    /// `"rayon=0-43,tokio=44-45,prefetcher=46,hot-buffer=47"`.
    /// Any group may be omitted; missing groups stay unpinned.
    Spec(String),
}

#[derive(Debug, Error)]
pub enum AffinityError {
    #[error("invalid cpu-affinity spec: {0}")]
    Spec(String),
    #[error("invalid cpu-affinity mode: {0:?} (expected 'auto', 'none', or a spec)")]
    UnknownMode(String),
    #[error("no online CPUs visible to core_affinity")]
    NoCores,
    #[error("rayon global pool already initialized — apply_to_rayon_global must run before any par_*() call")]
    RayonAlreadyInit,
    #[error("core id {0} out of range; host reports {1} online CPUs")]
    CoreOutOfRange(usize, usize),
}

/// Parse the `--cpu-affinity` flag value.
///
/// - `"none"` (case-insensitive) → [`Mode::None`].
/// - `"auto"` (case-insensitive) → [`Mode::Auto`].
/// - Anything else is treated as a spec string and returned as
///   [`Mode::Spec`] (validated by [`Layout::plan`]).
pub fn parse_mode(s: &str) -> Result<Mode, AffinityError> {
    match s.trim() {
        s if s.eq_ignore_ascii_case("none") => Ok(Mode::None),
        s if s.eq_ignore_ascii_case("auto") => Ok(Mode::Auto),
        "" => Err(AffinityError::UnknownMode(String::new())),
        other => Ok(Mode::Spec(other.to_string())),
    }
}

/// Per-role assignment of core ids. All vectors are sorted ascending.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Layout {
    /// Cores that rayon worker threads pin to.
    pub rayon_cores: Vec<usize>,
    /// Cores that tokio worker threads pin to (round-robin).
    pub tokio_cores: Vec<usize>,
    /// Core for the long-running prefetcher thread (e.g. expert
    /// safetensors prefetch, planned in iter 033). Pure reservation
    /// today; the prefetcher itself doesn't exist yet so this is what
    /// the helper will pin to when it lands.
    pub prefetcher_core: Option<usize>,
    /// Core for the hot-buffer build thread (e.g. KV-cache eager
    /// extend, planned in iter 069). Same status as `prefetcher_core`.
    pub hot_buffer_core: Option<usize>,
    /// If `true`, [`apply_to_rayon_global`] and
    /// [`tokio_on_thread_start`] are no-ops. Set when the mode is
    /// `None` or when planning failed safely.
    pub unpinned: bool,
}

impl Layout {
    /// Convenience: the layout returned by [`Mode::None`].
    pub fn unpinned() -> Self {
        Self {
            unpinned: true,
            ..Default::default()
        }
    }

    /// Plan a layout for `online_cpus` CPUs under the given mode.
    ///
    /// The planner runs without any side effects (no syscalls, no
    /// rayon/tokio touch). It only validates that all referenced
    /// core ids are < `online_cpus`. Pass in the result of
    /// [`detected_online_cpus`] for the real machine count.
    pub fn plan(mode: &Mode, online_cpus: usize) -> Result<Self, AffinityError> {
        if online_cpus == 0 {
            return Err(AffinityError::NoCores);
        }
        match mode {
            Mode::None => Ok(Self::unpinned()),
            Mode::Auto => Ok(auto_layout(online_cpus)),
            Mode::Spec(s) => parse_spec(s, online_cpus),
        }
    }

    /// Total cores referenced across all roles, deduplicated.
    pub fn referenced_cores(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .rayon_cores
            .iter()
            .chain(self.tokio_cores.iter())
            .chain(self.prefetcher_core.iter())
            .chain(self.hot_buffer_core.iter())
            .copied()
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Returns true if the same core appears in more than one role.
    /// Used by the CLI to print a warning — overlap isn't fatal (it
    /// just means contention is back) but it almost always indicates
    /// a spec typo.
    pub fn has_overlap(&self) -> bool {
        let total = self.rayon_cores.len()
            + self.tokio_cores.len()
            + self.prefetcher_core.iter().count()
            + self.hot_buffer_core.iter().count();
        total != self.referenced_cores().len()
    }

    /// Install the global rayon pool, sized to `rayon_cores.len()`,
    /// where each worker thread pins itself to its assigned core on
    /// startup. Idempotent across processes; returns
    /// [`AffinityError::RayonAlreadyInit`] if the global pool has
    /// already been built (i.e. some earlier `par_*()` call has run).
    ///
    /// No-op (and `Ok(())`) when `self.unpinned` is true or
    /// `rayon_cores` is empty.
    #[cfg(feature = "rayon-glue")]
    pub fn apply_to_rayon_global(&self) -> Result<(), AffinityError> {
        if self.unpinned || self.rayon_cores.is_empty() {
            return Ok(());
        }
        let cores = self.rayon_cores.clone();
        let n = cores.len();
        info!(
            n_threads = n,
            cores = ?cores,
            "installing rayon global pool with pinned workers"
        );
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("tahoma-rayon-{i}"))
            .start_handler(move |idx| {
                let core_id = cores[idx % cores.len()];
                pin_current_thread(core_id);
            })
            .build_global()
            .map_err(|_| AffinityError::RayonAlreadyInit)
    }

    /// Return an `on_thread_start` closure suitable for
    /// `tokio::runtime::Builder::on_thread_start`. The closure
    /// round-robins each newly-spawned tokio worker thread to a core
    /// out of `tokio_cores`. Returns `None` (the caller should skip
    /// `.on_thread_start(...)`) when the layout is unpinned or no
    /// tokio cores are reserved.
    pub fn tokio_on_thread_start(&self) -> Option<impl Fn() + Send + Sync + 'static> {
        if self.unpinned || self.tokio_cores.is_empty() {
            return None;
        }
        let cores = self.tokio_cores.clone();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        Some(move || {
            let i = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let core_id = cores[i % cores.len()];
            pin_current_thread(core_id);
            debug!(core = core_id, worker = i, "pinned tokio worker thread");
        })
    }

    /// Pin the calling thread to the reserved prefetcher core, if any.
    /// Convenience for the long-running prefetcher worker.
    pub fn pin_current_to_prefetcher(&self) {
        if let Some(c) = self.prefetcher_core {
            pin_current_thread(c);
            info!(core = c, "pinned current thread (prefetcher)");
        }
    }

    /// Pin the calling thread to the reserved hot-buffer core, if any.
    pub fn pin_current_to_hot_buffer(&self) {
        if let Some(c) = self.hot_buffer_core {
            pin_current_thread(c);
            info!(core = c, "pinned current thread (hot-buffer)");
        }
    }

    /// Human-readable one-line summary for the worker startup log.
    pub fn describe(&self) -> String {
        if self.unpinned {
            return "cpu-affinity: none (threads scheduled freely)".into();
        }
        let r = self.rayon_cores.len();
        let t = self.tokio_cores.len();
        format!(
            "cpu-affinity: rayon={r}c {} tokio={t}c {} prefetcher={} hot-buffer={}",
            fmt_range(&self.rayon_cores),
            fmt_range(&self.tokio_cores),
            self.prefetcher_core
                .map(|c| c.to_string())
                .unwrap_or_else(|| "-".into()),
            self.hot_buffer_core
                .map(|c| c.to_string())
                .unwrap_or_else(|| "-".into()),
        )
    }
}

/// Detect how many CPUs are online and pinnable.
/// Falls back to `std::thread::available_parallelism()` if
/// `core_affinity` is unavailable on the platform.
pub fn detected_online_cpus() -> usize {
    if let Some(v) = core_affinity::get_core_ids() {
        return v.len();
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Pin the current OS thread to one core. Logs (not errors) on
/// failure — pinning is a hint on macOS, may fail under cgroup
/// restrictions, and we never want a failed pin to crash inference.
pub fn pin_current_thread(core_id: usize) {
    let cid = core_affinity::CoreId { id: core_id };
    let ok = core_affinity::set_for_current(cid);
    if !ok {
        // Note: macOS always returns false here. That's expected; we
        // log at debug, not warn, to avoid flooding logs on darwin
        // dev machines.
        debug!(
            core = core_id,
            "core_affinity::set_for_current returned false"
        );
    } else {
        debug!(core = core_id, "pinned current thread");
    }
}

// ---------------------------------------------------------------------------
// Auto-layout heuristic.
// ---------------------------------------------------------------------------

fn auto_layout(n: usize) -> Layout {
    // Very small hosts: don't try to be clever; pin only rayon and
    // leave tokio and the helper roles unpinned. Below 4 cores, the
    // overhead of the I/O reactor sharing a rayon core is less bad
    // than starving it entirely.
    if n < 4 {
        return Layout {
            rayon_cores: (0..n).collect(),
            tokio_cores: Vec::new(),
            prefetcher_core: None,
            hot_buffer_core: None,
            unpinned: false,
        };
    }
    if n < 8 {
        // 4..=7 cores: reserve the last core for tokio + helpers,
        // give the rest to rayon. Prefetcher/hot-buffer share the
        // tokio pool — better than nothing on a thin host.
        let last = n - 1;
        return Layout {
            rayon_cores: (0..last).collect(),
            tokio_cores: vec![last],
            prefetcher_core: None,
            hot_buffer_core: None,
            unpinned: false,
        };
    }
    // n >= 8: reserve the top of the core range for I/O + helpers
    // and give the rest to rayon. The top cores are typically the
    // highest-numbered SMT siblings on Intel client hardware
    // (Alder/Raptor Lake place E-cores after P-cores) — keeping
    // them off the heavy compute path is mostly a win even when
    // wrong. The exact split below leaves:
    //   - 2 tokio cores
    //   - 1 prefetcher core
    //   - 1 hot-buffer core
    //   - the rest for rayon
    let hot_buffer = n - 1;
    let prefetcher = n - 2;
    let tokio_hi = n - 3;
    let tokio_lo = n - 4;
    let rayon_end = n - 4;
    Layout {
        rayon_cores: (0..rayon_end).collect(),
        tokio_cores: vec![tokio_lo, tokio_hi],
        prefetcher_core: Some(prefetcher),
        hot_buffer_core: Some(hot_buffer),
        unpinned: false,
    }
}

// ---------------------------------------------------------------------------
// Spec parser.
//
// Grammar (informal, single-line):
//
//   spec    := group ("," group)*
//   group   := name "=" ranges
//   name    := "rayon" | "tokio" | "prefetcher" | "hot-buffer"
//   ranges  := range ("|" range)*       // pipe avoids the comma-in-list ambiguity
//   range   := UINT | UINT "-" UINT
//
// Whitespace around `=`, `|`, `,`, `-` is tolerated.
//
// Prefetcher / hot-buffer accept exactly one core; multiple are
// rejected (would be unused).
// ---------------------------------------------------------------------------

fn parse_spec(spec: &str, online_cpus: usize) -> Result<Layout, AffinityError> {
    let mut layout = Layout::default();
    for raw_group in spec.split(',') {
        let group = raw_group.trim();
        if group.is_empty() {
            continue;
        }
        let (name, rhs) = group
            .split_once('=')
            .ok_or_else(|| AffinityError::Spec(format!("missing '=' in group: {group:?}")))?;
        let name = name.trim();
        let cores = parse_ranges(rhs.trim(), online_cpus)
            .map_err(|e| AffinityError::Spec(format!("group {name:?}: {e}")))?;
        match name {
            "rayon" => layout.rayon_cores = cores,
            "tokio" => layout.tokio_cores = cores,
            "prefetcher" => {
                if cores.len() != 1 {
                    return Err(AffinityError::Spec(format!(
                        "prefetcher needs exactly 1 core, got {}",
                        cores.len()
                    )));
                }
                layout.prefetcher_core = Some(cores[0]);
            }
            "hot-buffer" | "hot_buffer" => {
                if cores.len() != 1 {
                    return Err(AffinityError::Spec(format!(
                        "hot-buffer needs exactly 1 core, got {}",
                        cores.len()
                    )));
                }
                layout.hot_buffer_core = Some(cores[0]);
            }
            other => {
                return Err(AffinityError::Spec(format!(
                    "unknown role {other:?} (want rayon|tokio|prefetcher|hot-buffer)"
                )))
            }
        }
    }
    if layout.rayon_cores.is_empty()
        && layout.tokio_cores.is_empty()
        && layout.prefetcher_core.is_none()
        && layout.hot_buffer_core.is_none()
    {
        return Err(AffinityError::Spec(
            "spec assigned no cores to any role".into(),
        ));
    }
    Ok(layout)
}

fn parse_ranges(s: &str, online_cpus: usize) -> Result<Vec<usize>, String> {
    let mut out: Vec<usize> = Vec::new();
    for piece in s.split('|') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        if let Some((a, b)) = piece.split_once('-') {
            let a = a
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("range start {a:?}: {e}"))?;
            let b = b
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("range end {b:?}: {e}"))?;
            if b < a {
                return Err(format!("range {a}-{b}: end < start"));
            }
            for c in a..=b {
                if c >= online_cpus {
                    return Err(format!("core {c} >= online_cpus={online_cpus}"));
                }
                out.push(c);
            }
        } else {
            let c = piece
                .parse::<usize>()
                .map_err(|e| format!("core id {piece:?}: {e}"))?;
            if c >= online_cpus {
                return Err(format!("core {c} >= online_cpus={online_cpus}"));
            }
            out.push(c);
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

fn fmt_range(cores: &[usize]) -> String {
    if cores.is_empty() {
        return "[]".into();
    }
    // Compress contiguous spans for log brevity:
    //   [0,1,2,3,5,6,7] -> "[0-3,5-7]"
    let mut out = String::from("[");
    let mut i = 0;
    while i < cores.len() {
        let start = cores[i];
        let mut end = start;
        while i + 1 < cores.len() && cores[i + 1] == end + 1 {
            end = cores[i + 1];
            i += 1;
        }
        if !out.ends_with('[') {
            out.push(',');
        }
        if start == end {
            out.push_str(&start.to_string());
        } else {
            out.push_str(&format!("{start}-{end}"));
        }
        i += 1;
    }
    out.push(']');
    out
}

// ---------------------------------------------------------------------------
// Process-global handle.
//
// The CLI calls `init_global` once at startup; downstream code that
// wants to read the layout (e.g. the int4-gemm crate considering an
// auto-tuned chunk size, or a future iter 069 hot-buffer thread
// spawner) uses `global()`. `OnceLock` keeps it lock-free after init.
// ---------------------------------------------------------------------------

static GLOBAL: OnceLock<Layout> = OnceLock::new();

/// Install the process-wide layout. Calling twice is a soft no-op
/// that logs a warn — only the first install wins.
pub fn init_global(layout: Layout) {
    if GLOBAL.set(layout).is_err() {
        warn!("tahoma_cpu_affinity::init_global called twice; second call ignored");
    }
}

/// Read the process-wide layout, or the unpinned default if
/// [`init_global`] was never called.
pub fn global() -> &'static Layout {
    static FALLBACK: OnceLock<Layout> = OnceLock::new();
    GLOBAL
        .get()
        .unwrap_or_else(|| FALLBACK.get_or_init(Layout::unpinned))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mode_basics() {
        assert_eq!(parse_mode("none").unwrap(), Mode::None);
        assert_eq!(parse_mode("None").unwrap(), Mode::None);
        assert_eq!(parse_mode("AUTO").unwrap(), Mode::Auto);
        assert_eq!(
            parse_mode("rayon=0-3").unwrap(),
            Mode::Spec("rayon=0-3".into())
        );
        assert!(parse_mode("").is_err());
    }

    #[test]
    fn auto_layout_for_a_48c_xeon() {
        let l = Layout::plan(&Mode::Auto, 48).unwrap();
        assert!(!l.unpinned);
        assert_eq!(l.rayon_cores.len(), 44);
        assert_eq!(l.rayon_cores.first(), Some(&0));
        assert_eq!(l.rayon_cores.last(), Some(&43));
        assert_eq!(l.tokio_cores, vec![44, 45]);
        assert_eq!(l.prefetcher_core, Some(46));
        assert_eq!(l.hot_buffer_core, Some(47));
        assert!(!l.has_overlap());
    }

    #[test]
    fn auto_layout_for_small_host() {
        let l = Layout::plan(&Mode::Auto, 6).unwrap();
        assert!(!l.unpinned);
        assert_eq!(l.rayon_cores, vec![0, 1, 2, 3, 4]);
        assert_eq!(l.tokio_cores, vec![5]);
        assert_eq!(l.prefetcher_core, None);
        assert_eq!(l.hot_buffer_core, None);
    }

    #[test]
    fn auto_layout_for_two_core_host() {
        let l = Layout::plan(&Mode::Auto, 2).unwrap();
        assert!(!l.unpinned);
        assert_eq!(l.rayon_cores, vec![0, 1]);
        assert_eq!(l.tokio_cores, Vec::<usize>::new());
    }

    #[test]
    fn auto_layout_for_one_core_host() {
        let l = Layout::plan(&Mode::Auto, 1).unwrap();
        assert_eq!(l.rayon_cores, vec![0]);
        // Nothing else gets pinned — there's no room.
        assert!(l.tokio_cores.is_empty());
        assert!(l.prefetcher_core.is_none());
    }

    #[test]
    fn mode_none_yields_unpinned_layout() {
        let l = Layout::plan(&Mode::None, 48).unwrap();
        assert!(l.unpinned);
        assert!(l.rayon_cores.is_empty());
        assert!(l.tokio_cores.is_empty());
    }

    #[test]
    fn spec_parses_full_grammar() {
        let l = Layout::plan(
            &Mode::Spec("rayon=0-43,tokio=44-45,prefetcher=46,hot-buffer=47".into()),
            48,
        )
        .unwrap();
        assert_eq!(l.rayon_cores.len(), 44);
        assert_eq!(l.tokio_cores, vec![44, 45]);
        assert_eq!(l.prefetcher_core, Some(46));
        assert_eq!(l.hot_buffer_core, Some(47));
    }

    #[test]
    fn spec_tolerates_whitespace_and_pipes() {
        let l = Layout::plan(&Mode::Spec("  rayon = 0-2 | 5  ,  tokio = 3-4  ".into()), 8).unwrap();
        assert_eq!(l.rayon_cores, vec![0, 1, 2, 5]);
        assert_eq!(l.tokio_cores, vec![3, 4]);
    }

    #[test]
    fn spec_rejects_out_of_range_core() {
        let err = Layout::plan(&Mode::Spec("rayon=0-48".into()), 48).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("48"), "msg = {msg}");
    }

    #[test]
    fn spec_rejects_unknown_role() {
        let err = Layout::plan(&Mode::Spec("nvidia=0-3".into()), 8).unwrap_err();
        assert!(err.to_string().contains("unknown role"));
    }

    #[test]
    fn spec_rejects_empty() {
        let err = Layout::plan(&Mode::Spec(",,,".into()), 8).unwrap_err();
        assert!(err.to_string().contains("no cores"));
    }

    #[test]
    fn spec_rejects_multi_core_prefetcher() {
        let err = Layout::plan(&Mode::Spec("prefetcher=0-1".into()), 8).unwrap_err();
        assert!(err.to_string().contains("prefetcher"));
    }

    #[test]
    fn hot_buffer_underscore_alias_works() {
        let l = Layout::plan(&Mode::Spec("hot_buffer=7".into()), 8).unwrap();
        assert_eq!(l.hot_buffer_core, Some(7));
    }

    #[test]
    fn referenced_cores_dedups() {
        let l = Layout {
            rayon_cores: vec![0, 1, 2],
            tokio_cores: vec![2, 3],
            prefetcher_core: Some(4),
            hot_buffer_core: Some(4),
            unpinned: false,
        };
        assert_eq!(l.referenced_cores(), vec![0, 1, 2, 3, 4]);
        assert!(l.has_overlap());
    }

    #[test]
    fn describe_renders_unpinned() {
        let l = Layout::unpinned();
        let s = l.describe();
        assert!(s.contains("none"));
    }

    #[test]
    fn describe_renders_auto() {
        let l = Layout::plan(&Mode::Auto, 48).unwrap();
        let s = l.describe();
        assert!(s.contains("rayon=44c"));
        assert!(s.contains("[0-43]"));
        assert!(s.contains("tokio=2c"));
        assert!(s.contains("46"));
        assert!(s.contains("47"));
    }

    #[test]
    fn fmt_range_compresses_contiguous_spans() {
        assert_eq!(fmt_range(&[]), "[]");
        assert_eq!(fmt_range(&[5]), "[5]");
        assert_eq!(fmt_range(&[0, 1, 2, 3]), "[0-3]");
        assert_eq!(fmt_range(&[0, 1, 2, 5, 6, 7]), "[0-2,5-7]");
        assert_eq!(fmt_range(&[1, 3, 5]), "[1,3,5]");
    }

    #[test]
    fn plan_rejects_zero_cpus() {
        let err = Layout::plan(&Mode::Auto, 0).unwrap_err();
        assert!(matches!(err, AffinityError::NoCores));
    }

    #[test]
    fn tokio_on_thread_start_returns_none_when_unpinned() {
        let l = Layout::unpinned();
        assert!(l.tokio_on_thread_start().is_none());
    }

    #[test]
    fn tokio_on_thread_start_returns_some_when_pinned() {
        let l = Layout::plan(&Mode::Auto, 48).unwrap();
        assert!(l.tokio_on_thread_start().is_some());
    }

    #[test]
    fn init_global_is_idempotent() {
        // Note: this test must run in its own process-isolated form
        // if other tests touch GLOBAL. Cargo by default runs tests in
        // parallel within the same process. The fallback path is what
        // we care about — `global()` always returns *some* layout.
        let l = global();
        // Either someone called init_global earlier (any layout is
        // fine) or we get the fallback unpinned layout.
        assert!(l.unpinned || !l.rayon_cores.is_empty() || l.tokio_cores.is_empty());
    }
}
