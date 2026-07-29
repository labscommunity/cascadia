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

/// Physically-available RAM in bytes, or 0 when it cannot be determined.
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
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// How many experts fit in `budget_bytes`, leaving `reserve_bytes` for the
/// shell, KV/recurrent state and the OS.
pub fn pin_budget_experts(budget_bytes: u64, reserve_bytes: u64, expert_bytes: u64) -> usize {
    if expert_bytes == 0 {
        return 0;
    }
    budget_bytes.saturating_sub(reserve_bytes) as usize / expert_bytes as usize
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
        let mut s = String::new();
        let mut keys: Vec<_> = self.counts.iter().collect();
        keys.sort_by_key(|(k, _)| **k);
        for ((l, e), c) in keys {
            s.push_str(&format!("{l} {e} {c}\n"));
        }
        std::fs::write(path, s)
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
    #[cfg(not(unix))]
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
        // 100 GB budget, 12 GB reserved, 17.5 MB experts
        assert_eq!(
            pin_budget_experts(100_000_000_000, 12_000_000_000, 17_547_264),
            5015
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
