//! Expert residency — the lever that actually decides K3 throughput.
//!
//! At ~1.45 TB of routed experts against any plausible host RAM, decode is
//! bound by how much of the expert set is resident, not by arithmetic. This
//! module supplies the three pieces glm5 found mattered:
//!
//! * a **budget**: how many experts a rank may pin without pushing the box into
//!   swap,
//! * a **usage histogram**: which (layer, expert) pairs actually get routed, so
//!   the pin set is learned from real traffic rather than guessed,
//! * **pinning**: `mlock` / `VirtualLock` over the chosen sub-ranges of the
//!   per-layer expert maps, so the hottest experts stop being evicted.
//!
//! Pinning is opt-in (`CASCADIA_K3_AUTOPIN`) because an over-large pin set is
//! worse than none — it evicts the page cache that was serving the cold tail.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

/// Physically-available RAM in bytes, or 0 when it cannot be determined.
/// Page-cache reserve. Starving it drops buffered pread from ~800 to ~180 MB/s.
pub const PAGE_CACHE_RESERVE: u64 = 2_500_000_000;
/// Fraction of available RAM the pin budget may claim.
pub const RAM_FRACTION: f64 = 0.88;
/// Batch-union prefill touches a block of distinct experts; leave room for it.
pub const WORKING_SET_BLOCK: u64 = 64;
/// No pinning below this many recorded selections.
pub const AUTOPIN_MIN_SELECTIONS: u64 = 5_000;
/// Confidence reaches 1.0 at this many selections.
pub const AUTOPIN_FULL_SELECTIONS: f64 = 200_000.0;

pub fn mem_available() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    if let Some(kb) = rest.split_whitespace().next() {
                        return kb.parse::<u64>().unwrap_or(0) * 1024;
                    }
                }
            }
        }
        0
    }
    #[cfg(windows)]
    {
        // The fleet runs Windows. Returning 0 here made autopin_budget return 0,
        // so `if budget > 0` never fired and CASCADIA_K3_AUTOPIN silently pinned
        // nothing on the machines that need it most — the same shape of bug as
        // the cfg(unix)-only memory hints below.
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        extern "system" {
            fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        }
        // SAFETY: a zeroed MEMORYSTATUSEX with `length` set to its own size, as
        // the API requires; the call only writes into it.
        let mut ms: MemoryStatusEx = unsafe { std::mem::zeroed() };
        ms.length = std::mem::size_of::<MemoryStatusEx>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut ms) } != 0 {
            return ms.avail_phys;
        }
    }
    // A query failure must not read as "no memory available" — that silently
    // disables pinning. Be conservative instead.
    8_000_000_000
}

/// How many experts fit in `budget_bytes`, leaving `reserve_bytes` for the
/// shell, KV/recurrent state and the OS.
pub fn pin_budget_experts(budget_bytes: u64, reserve_bytes: u64, expert_bytes: u64) -> usize {
    if expert_bytes == 0 {
        return 0;
    }
    // Leave the page cache intact. Starving it is measured to drop buffered
    // pread from ~800 to ~180 MB/s on this class of host — pinning that takes
    // the cache's memory can therefore cost more than the pins are worth, which
    // is precisely the failure this module's header warns about.
    let slack = PAGE_CACHE_RESERVE + WORKING_SET_BLOCK * expert_bytes;
    let usable = (budget_bytes as f64 * RAM_FRACTION) as u64;
    usable.saturating_sub(reserve_bytes.saturating_add(slack)) as usize / expert_bytes as usize
}

/// Confidence ramp: pin nothing until [`AUTOPIN_MIN_SELECTIONS`], then rise
/// linearly to half the budget by [`AUTOPIN_FULL_SELECTIONS`].
///
/// A histogram with a handful of observations picks near-arbitrary experts and
/// spends the budget on them, evicting page-cache entries that were doing more
/// good. Confidence is earned rather than assumed.
pub fn autopin_count(total_selections: u64, budget_experts: usize) -> usize {
    if total_selections < AUTOPIN_MIN_SELECTIONS {
        return 0;
    }
    let conf = (total_selections as f64 / AUTOPIN_FULL_SELECTIONS).min(1.0);
    (budget_experts as f64 * 0.5 * conf) as usize
}

/// The pin budget for this rank, from `CASCADIA_K3_PIN_BYTES` or available RAM.
///
/// Returns 0 unless `CASCADIA_K3_AUTOPIN` is set — pinning the wrong set is
/// worse than pinning nothing, so it never engages by default.
pub fn autopin_budget(expert_bytes: u64, reserve_bytes: u64) -> usize {
    if std::env::var("CASCADIA_K3_AUTOPIN")
        .map(|v| v == "0" || v.is_empty())
        .unwrap_or(true)
    {
        return 0;
    }
    let budget = std::env::var("CASCADIA_K3_PIN_BYTES")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or_else(mem_available);
    pin_budget_experts(budget, reserve_bytes, expert_bytes)
}

/// Routed-expert hit counts, keyed by `(layer, expert)`.
///
/// Persisted between runs so a deployment's pin set improves with use — the
/// first run learns the distribution, later runs start warm.
#[derive(Default, Debug, Clone)]
pub struct UsageStats {
    counts: HashMap<(u32, u32), u64>,
}

impl UsageStats {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn record(&mut self, layer: u32, expert: u32) {
        *self.counts.entry((layer, expert)).or_insert(0) += 1;
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    pub fn total(&self) -> u64 {
        self.counts.values().sum()
    }

    /// The `n` hottest experts of one layer, hottest first. Ties break toward
    /// the lower expert id so a pin set is reproducible across runs.
    pub fn hottest_for(&self, layer: u32, n: usize) -> Vec<u32> {
        let mut v: Vec<(u32, u64)> = self
            .counts
            .iter()
            .filter(|((l, _), _)| *l == layer)
            .map(|((_, e), &c)| (*e, c))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(n);
        v.into_iter().map(|(e, _)| e).collect()
    }

    /// `layer expert count` per line — plain text so it can be inspected and
    /// hand-edited on a node without tooling.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        // Write-then-rename: this runs from `Drop`, the moment a kill is most
        // likely, and `load` skips malformed lines silently — so a torn write
        // would degrade the pin set with no error anywhere.
        let mut out = String::new();
        for ((l, e), c) in &self.counts {
            out.push_str(&format!("{l} {e} {c}\n"));
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, out)?;
        std::fs::rename(&tmp, path)
    }

    pub fn load(&mut self, path: &Path) -> std::io::Result<()> {
        let s = std::fs::read_to_string(path)?;
        for line in s.lines() {
            let mut it = line.split_whitespace();
            if let (Some(l), Some(e), Some(c)) = (it.next(), it.next(), it.next()) {
                if let (Ok(l), Ok(e), Ok(c)) =
                    (l.parse::<u32>(), e.parse::<u32>(), c.parse::<u64>())
                {
                    *self.counts.entry((l, e)).or_insert(0) += c;
                }
            }
        }
        Ok(())
    }
}

/// Process-global routing histogram — one model per process, so a global beats
/// threading a handle through the whole layer API. Recorded unconditionally so
/// autopin has data even on runs where pinning was off.
fn global() -> &'static Mutex<UsageStats> {
    static U: OnceLock<Mutex<UsageStats>> = OnceLock::new();
    U.get_or_init(|| Mutex::new(UsageStats::new()))
}

/// Record one layer's routed selection.
#[inline]
pub fn record_selection(layer: u32, experts: &[u32]) {
    if let Ok(mut u) = global().lock() {
        for &e in experts {
            u.record(layer, e);
        }
    }
}

/// Copy of the histogram so far.
pub fn snapshot() -> UsageStats {
    global().lock().map(|u| u.clone()).unwrap_or_default()
}

/// Merge a saved histogram into the global one (called at load).
pub fn load_global(path: &Path) -> std::io::Result<()> {
    let mut u = global()
        .lock()
        .map_err(|_| std::io::Error::other("usage lock poisoned"))?;
    u.load(path)
}

/// Persist the global histogram so the next run starts warm.
pub fn save_global(path: &Path) -> std::io::Result<()> {
    let u = global()
        .lock()
        .map_err(|_| std::io::Error::other("usage lock poisoned"))?;
    u.save(path)
}

/// Where a model dir keeps its learned histogram.
pub fn usage_path(model_dir: &Path) -> std::path::PathBuf {
    model_dir.join(".k3_usage")
}

/// Windows equivalents of the Unix memory hints.
///
/// The fleet deployment target is Windows, where `madvise`/`mlock` do not exist.
/// Without these every hint below was a silent no-op on exactly the machines they
/// were written for. Mirrors the block `dsv4::expert_mmap` already proves, kept
/// local so the k3 shell does not reach into a sibling backend for platform glue.
#[cfg(windows)]
mod win {
    use core::ffi::c_void;

    #[repr(C)]
    pub struct Win32MemoryRangeEntry {
        pub virtual_address: *mut c_void,
        pub number_of_bytes: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        pub fn VirtualLock(address: *const c_void, size: usize) -> i32;
        pub fn GetCurrentProcess() -> isize;
        pub fn PrefetchVirtualMemory(
            process: isize,
            number_of_entries: usize,
            addresses: *const Win32MemoryRangeEntry,
            flags: u32,
        ) -> i32;
    }
}

/// Lock `len` bytes at `addr` into RAM. Best-effort: a failure (rlimit, lack of
/// privilege) is reported, never fatal — the run still works, just colder.
pub fn pin_range(addr: usize, len: usize) -> bool {
    #[cfg(unix)]
    {
        use core::ffi::c_void;
        extern "C" {
            fn mlock(addr: *const c_void, len: usize) -> i32;
        }
        // SAFETY: [addr, addr+len) is a live read-only mapping owned by the caller.
        unsafe { mlock(addr as *const c_void, len) == 0 }
    }
    #[cfg(windows)]
    {
        // SAFETY: a live mapping owned by the caller. VirtualLock pins it into
        // the working set; the process quota bounds how much can be locked.
        unsafe { win::VirtualLock(addr as *const core::ffi::c_void, len) != 0 }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (addr, len);
        false
    }
}

/// Round a byte range out to whole pages.
///
/// `madvise` on Linux rejects a non-page-aligned address with EINVAL, while
/// macOS quietly accepts it — so an unaligned call is a hint that silently does
/// nothing on the platform we actually deploy to. K3's expert stride happens to
/// be 4284 pages exactly, so today every expert offset is aligned by luck; a
/// change to moe_intermediate_size or the group size would break that with no
/// error, just the speedup gone. Aligning here removes the landmine.
#[cfg(unix)]
fn page_range(addr: usize, len: usize) -> (usize, usize) {
    extern "C" {
        fn getpagesize() -> i32;
    }
    // SAFETY: getpagesize takes no arguments and cannot fail.
    let ps = unsafe { getpagesize() } as usize;
    debug_assert!(ps.is_power_of_two());
    let start = addr & !(ps - 1);
    let end = (addr + len + ps - 1) & !(ps - 1);
    (start, end - start)
}

/// Tell the kernel this mapping is read in random order, disabling readahead.
///
/// An expert is a ~17.6 MB contiguous slice, but the 16 experts a layer routes to
/// are scattered through a ~15.7 GB mapping. Sequential-readahead heuristics see
/// each slice as the start of a stream and fetch well past its end, so the disk
/// delivers several times the bytes the model asked for. On the Xeon host a
/// decode token needed 25.8 GB of expert weights and read ~100 GB — roughly 4x.
///
/// `MADV_RANDOM` is 1 on Linux and on the BSDs/macOS. Best-effort: failure just
/// leaves the default heuristics in place.
pub fn advise_random(addr: usize, len: usize) -> bool {
    #[cfg(unix)]
    {
        use core::ffi::c_void;
        const MADV_RANDOM: i32 = 1;
        extern "C" {
            fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        }
        let (a, l) = page_range(addr, len);
        // SAFETY: [a, a+l) covers the caller's live mapping, rounded out to whole
        // pages. madvise only changes kernel readahead policy; it never writes.
        unsafe { madvise(a as *mut c_void, l, MADV_RANDOM) == 0 }
    }
    #[cfg(not(unix))]
    {
        let _ = (addr, len);
        false
    }
}

/// Queue a read of `[addr, addr+len)` and return without waiting for it.
///
/// The expert loop is serial — read one expert, compute it, read the next — so
/// the drive sits idle for a seek between every expert. Advising all of a layer's
/// routed slices up front lets the block layer queue and reorder them, which is
/// where the win comes from: not overlapping I/O with compute (compute is 1% of
/// wall time here) but keeping the request queue deep.
///
/// `MADV_WILLNEED` is 3 on Linux and on the BSDs/macOS. Best-effort and purely a
/// hint — it can never change what the model computes.
pub fn advise_willneed(addr: usize, len: usize) -> bool {
    #[cfg(unix)]
    {
        use core::ffi::c_void;
        const MADV_WILLNEED: i32 = 3;
        extern "C" {
            fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        }
        let (a, l) = page_range(addr, len);
        // SAFETY: [a, a+l) covers the caller's live mapping, rounded out to whole
        // pages. madvise only starts read-ahead; it never writes to the range.
        unsafe { madvise(a as *mut c_void, l, MADV_WILLNEED) == 0 }
    }
    #[cfg(windows)]
    {
        // PrefetchVirtualMemory is the analogue: an async read-ahead over an
        // explicit range, which is in fact a closer fit than madvise's advisory
        // flag. No page alignment requirement.
        let entry = win::Win32MemoryRangeEntry {
            virtual_address: addr as *mut core::ffi::c_void,
            number_of_bytes: len,
        };
        // SAFETY: `entry` describes a live mapping owned by the caller;
        // PrefetchVirtualMemory only reads the descriptor and queues a read.
        unsafe { win::PrefetchVirtualMemory(win::GetCurrentProcess(), 1, &entry, 0) != 0 }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (addr, len);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_divides_and_reserves() {
        // 100 GB budget, 12 GB reserved, 17.5 MB experts:
        //   usable = 100e9 * 0.88                       = 88.00 GB
        //   slack  = 2.5 GB + 64 * 17.5 MB              =  3.62 GB
        //   (88.00 - 12.00 - 3.62) GB / 17.547 MB       =  4124
        // Below the naive (budget - reserve) / expert = 5015, deliberately:
        // starving the page cache costs more than the extra pins gain.
        assert_eq!(
            pin_budget_experts(100_000_000_000, 12_000_000_000, 17_547_264),
            4124
        );
        // a reserve larger than the budget pins nothing rather than underflowing
        assert_eq!(pin_budget_experts(1_000, 2_000, 10), 0);
        assert_eq!(pin_budget_experts(1_000, 0, 0), 0);
    }

    #[test]
    fn hottest_is_ordered_and_tie_broken_by_id() {
        let mut u = UsageStats::new();
        for _ in 0..5 {
            u.record(1, 7);
        }
        for _ in 0..5 {
            u.record(1, 3); // ties with expert 7 -> lower id wins
        }
        u.record(1, 9);
        u.record(2, 0); // a different layer must not leak in
        assert_eq!(u.hottest_for(1, 3), vec![3, 7, 9]);
        assert_eq!(u.hottest_for(1, 1), vec![3]);
        assert_eq!(u.total(), 12);
    }

    #[test]
    fn usage_roundtrips_through_a_file() {
        let mut u = UsageStats::new();
        u.record(0, 1);
        u.record(0, 1);
        u.record(3, 2);
        let p = std::env::temp_dir().join("k3_usage_roundtrip.txt");
        u.save(&p).expect("save");
        let mut back = UsageStats::new();
        back.load(&p).expect("load");
        assert_eq!(back.hottest_for(0, 2), vec![1]);
        assert_eq!(back.total(), u.total());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn autopin_is_off_unless_asked() {
        // never engage by default: a wrong pin set evicts the cache serving the
        // cold tail and is worse than not pinning at all
        std::env::remove_var("CASCADIA_K3_AUTOPIN");
        assert_eq!(autopin_budget(17_547_264, 0), 0);
    }
}

#[cfg(all(test, unix))]
mod advise_tests {
    use super::advise_random;

    #[test]
    fn advise_random_succeeds_on_a_real_mapping() {
        // a private anonymous mapping is enough to exercise the syscall path
        let len = 1 << 20;
        let mut v = vec![0u8; len];
        let addr = v.as_mut_ptr() as usize;
        assert!(advise_random(addr, len), "madvise(MADV_RANDOM) failed");
    }

    #[test]
    fn mem_available_never_reports_zero() {
        // Returning 0 on an unhandled platform silently disabled autopin: the
        // budget became 0 and the `if budget > 0` guard never fired. A query
        // failure must degrade to a conservative number, not to "no memory".
        assert!(
            super::mem_available() > 0,
            "mem_available must never be 0 — that silently disables pinning"
        );
    }

    #[test]
    fn budget_leaves_the_page_cache_alone() {
        let eb = 17_547_264u64;
        // A budget that only just covers the reserve must not pin anything: the
        // page-cache reserve and working-set block come out first.
        assert_eq!(
            super::pin_budget_experts(10_000_000_000, 9_000_000_000, eb),
            0
        );
        // A large budget still holds back the reserve rather than spending all.
        let big = super::pin_budget_experts(200_000_000_000, 10_000_000_000, eb);
        let naive = (200_000_000_000u64 - 10_000_000_000) / eb;
        assert!(
            big > 0 && (big as u64) < naive,
            "got {big}, naive would be {naive}"
        );
    }

    #[test]
    fn autopin_ramps_with_confidence() {
        // Below the floor: nothing, however large the budget.
        assert_eq!(super::autopin_count(0, 1000), 0);
        assert_eq!(
            super::autopin_count(super::AUTOPIN_MIN_SELECTIONS - 1, 1000),
            0
        );
        // At the floor it engages, but only a sliver.
        let low = super::autopin_count(super::AUTOPIN_MIN_SELECTIONS, 1000);
        assert!(low > 0 && low < 50, "expected a small ramp, got {low}");
        // Fully confident caps at half the budget, never the whole thing.
        let full = super::autopin_count(1_000_000, 1000);
        assert_eq!(full, 500);
    }

    #[test]
    fn usage_save_is_atomic_and_round_trips() {
        let dir = std::env::temp_dir().join("k3_usage_atomic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("usage");
        let mut u = super::UsageStats::new();
        u.record(3, 7);
        u.record(3, 7);
        u.record(1, 2);
        u.save(&path).unwrap();
        // the temp file must not survive a successful save
        assert!(!path.with_extension("tmp").exists(), "left a .tmp behind");
        let mut back = super::UsageStats::new();
        back.load(&path).unwrap();
        assert_eq!(back.total(), u.total());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn advise_willneed_succeeds_on_a_real_mapping() {
        let len = 1 << 20;
        let mut v = vec![0u8; len];
        let addr = v.as_mut_ptr() as usize;
        assert!(
            super::advise_willneed(addr, len),
            "madvise(MADV_WILLNEED) failed"
        );
    }

    #[test]
    fn advise_works_on_a_deliberately_unaligned_address() {
        // The regression this guards: Linux madvise rejects an unaligned addr
        // with EINVAL while macOS accepts it, so this only ever failed on the
        // platform we deploy to.
        let len = 1 << 20;
        let mut v = vec![0u8; len];
        let base = v.as_mut_ptr() as usize;
        assert!(
            super::advise_willneed(base + 3, len - 8),
            "unaligned WILLNEED must still be applied"
        );
        assert!(super::advise_random(base + 1, len - 8));
    }

    #[test]
    fn advise_random_reports_failure_on_a_bad_range() {
        // length 0 at a bogus address must not panic; it just reports false/true
        // without touching memory. The point is that it never aborts the run.
        let _ = advise_random(0, 0);
    }
}
