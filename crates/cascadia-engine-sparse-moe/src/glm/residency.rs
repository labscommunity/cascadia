//! Per-node RAM residency for the mmap int4 expert path.
//!
//! Cascadia mmaps expert bins and leans on the OS page cache as the LRU. On a
//! node where experts (~89 GB) far exceed RAM (~32 GB), cold experts thrash the
//! NVMe. Residency adds two levers:
//!   1. a **budget** ([`pin_expert_count`], ported from `cap_for_ram`) for how
//!      many experts RAM can afford to keep pinned, after reserving the dense
//!      set, KV pools, the batch-union working set, activations, and a
//!      load-bearing page-cache reserve;
//!   2. a persistent **routing histogram** ([`UsageStats`], the `.coli_usage`
//!      file) that records which experts fire, so the hottest can be `mlock`'d
//!      (AUTOPIN) — "gets faster the more you use it".
//!
//! This module is the pure budget + bookkeeping; the mlock application and the
//! per-forward usage hook wire it into the runner separately.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

/// Page-cache reserve. Starving it drops buffered pread from ~800 to ~180 MB/s;
/// the budget always leaves it intact, even under `O_DIRECT`.
pub const PAGE_CACHE_RESERVE: u64 = 2_500_000_000;
/// Activations + logits + misc overhead.
pub const ACTIVATION_SLACK: u64 = 1_200_000_000;
/// Fraction of MemAvailable-at-boot the budget may use.
pub const RAM_FRACTION: f64 = 0.88;
/// Batch-union working set is a block of 64 unique experts.
pub const WORKING_SET_BLOCK: u64 = 64;
/// AUTOPIN gate: no pinning below this many recorded selections.
pub const AUTOPIN_MIN_SELECTIONS: u64 = 5_000;
/// AUTOPIN confidence reaches 1.0 at this many selections.
pub const AUTOPIN_FULL_SELECTIONS: f64 = 200_000.0;

/// How many int4 experts this node may `mlock`, after reserving the dense
/// resident set (`resident`), the KV pools (`kv_bytes`), the batch-union
/// working set, activations, and the page-cache reserve. Returns 0 when nothing fits (the caller
/// then relies purely on the OS page cache).
pub fn pin_expert_count(mem_available: u64, resident: u64, kv_bytes: u64, expert_bytes: u64) -> usize {
    if expert_bytes == 0 {
        return 0;
    }
    let ram = (mem_available as f64 * RAM_FRACTION) as u64;
    let slack = ACTIVATION_SLACK + PAGE_CACHE_RESERVE + WORKING_SET_BLOCK * expert_bytes + kv_bytes;
    let avail = ram.saturating_sub(resident.saturating_add(slack));
    (avail / expert_bytes) as usize
}

/// AUTOPIN confidence ramp: pin nothing until
/// [`AUTOPIN_MIN_SELECTIONS`], then ramp linearly to 50% of the budget as the
/// recorded history reaches [`AUTOPIN_FULL_SELECTIONS`]. Few data points pick
/// the wrong experts and steal LRU slots, so confidence is earned.
pub fn autopin_count(total_selections: u64, budget_experts: usize) -> usize {
    if total_selections < AUTOPIN_MIN_SELECTIONS {
        return 0;
    }
    let conf = (total_selections as f64 / AUTOPIN_FULL_SELECTIONS).min(1.0);
    (budget_experts as f64 * 0.5 * conf) as usize
}

/// Persistent routing histogram: `(layer, expert) -> selection count`. Mirrors
/// the `.coli_usage` file (line format `"<layer> <expert> <count>"`).
#[derive(Default)]
pub struct UsageStats {
    counts: HashMap<(u32, u32), u64>,
    pub total: u64,
}

impl UsageStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one routed selection.
    pub fn record(&mut self, layer: u32, expert: u32) {
        *self.counts.entry((layer, expert)).or_insert(0) += 1;
        self.total += 1;
    }

    /// Load + ACCUMULATE from a `.coli_usage` file (counts resume across
    /// sessions). A missing file is not an error.
    pub fn load(&mut self, path: &Path) -> std::io::Result<()> {
        let s = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e),
        };
        for line in s.lines() {
            let mut it = line.split_whitespace();
            if let (Some(l), Some(e), Some(c)) = (it.next(), it.next(), it.next()) {
                if let (Ok(l), Ok(e), Ok(c)) = (l.parse::<u32>(), e.parse::<u32>(), c.parse::<u64>()) {
                    *self.counts.entry((l, e)).or_insert(0) += c;
                    self.total += c;
                }
            }
        }
        Ok(())
    }

    /// Atomically write the histogram (tmp + rename), so a crash mid-write
    /// leaves the previous file intact.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let tmp = path.with_extension("coli_usage.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            for (&(l, e), &c) in &self.counts {
                if c > 0 {
                    writeln!(f, "{l} {e} {c}")?;
                }
            }
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)
    }

    /// The `n` hottest `(layer, expert)` pairs, most-used first (ties -> lower
    /// (layer, expert) for determinism).
    pub fn hottest(&self, n: usize) -> Vec<(u32, u32)> {
        let mut v: Vec<((u32, u32), u64)> = self.counts.iter().map(|(&k, &c)| (k, c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(n);
        v.into_iter().map(|(k, _)| k).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_reserves_slack_and_page_cache() {
        // 32 GB node, ~2.5 GB resident dense, ~1 GB KV, 18.9 MB experts.
        let eb = 18_900_000u64;
        let n = pin_expert_count(32_000_000_000, 2_500_000_000, 1_000_000_000, eb);
        // 0.88*32G=28.16G; slack = 1.2G + 2.5G + 64*18.9M(~1.2G) + 1G = ~5.9G;
        // avail = 28.16 - 2.5 - 5.9 = ~19.76G; /18.9M = ~1045 experts.
        assert!((900..1200).contains(&n), "unexpected pin count {n}");
        // Degenerate: nothing fits -> 0, no panic.
        assert_eq!(pin_expert_count(4_000_000_000, 2_000_000_000, 1_000_000_000, eb), 0);
        assert_eq!(pin_expert_count(32_000_000_000, 0, 0, 0), 0);
    }

    #[test]
    fn autopin_ramps_with_confidence() {
        assert_eq!(autopin_count(4_999, 1000), 0); // below the gate
        assert_eq!(autopin_count(200_000, 1000), 500); // full confidence -> 50%
        assert_eq!(autopin_count(400_000, 1000), 500); // capped at 50%
        assert_eq!(autopin_count(100_000, 1000), 250); // half confidence -> 25%
    }

    #[test]
    fn usage_records_hottest_and_roundtrips() {
        let mut u = UsageStats::new();
        for _ in 0..10 {
            u.record(3, 7);
        }
        for _ in 0..5 {
            u.record(3, 2);
        }
        u.record(4, 0);
        assert_eq!(u.total, 16);
        assert_eq!(u.hottest(2), vec![(3, 7), (3, 2)]);

        let dir = std::env::temp_dir().join(format!("glm5_usage_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(".coli_usage");
        u.save(&p).unwrap();
        let mut v = UsageStats::new();
        v.load(&p).unwrap();
        assert_eq!(v.total, 16);
        assert_eq!(v.hottest(1), vec![(3, 7)]);
        // load ACCUMULATES.
        v.load(&p).unwrap();
        assert_eq!(v.total, 32);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
