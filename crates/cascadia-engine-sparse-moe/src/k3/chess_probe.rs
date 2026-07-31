//! Measures how much of an expert's `w3` section lane sparsity could skip.
//!
//! CHESS-style lane sparsity says a lane whose gate activation is negligible
//! contributes nothing, so that lane's `w3` row (and the matching `w2` column)
//! never has to be read. The catch is that the unit of I/O is a 4 KiB page, not
//! a row: dead rows scattered across the section still fault in every page they
//! touch, and the saving is exactly zero. The idea only pays where the dead
//! lanes are dense enough that WHOLE pages are dead.
//!
//! So this counts both numbers — the raw dead-lane fraction and the fully-dead
//! PAGE fraction — at several thresholds. The gap between them is the point:
//! the first is what the sparsity claim promises, the second is what the storage
//! stack would actually deliver.
//!
//! The liveness signal is `g`, the `w1` output the SiTU gate consumes. `w1` has
//! to be read whichever way this goes, so a decision made on `g` is free; a
//! decision made on `u` would require reading the very section it is meant to
//! skip.
//!
//! `CASCADIA_K3_CHESS_PROBE=1` turns it on; it is off and branch-cheap
//! otherwise. Nothing here feeds back into routing or arithmetic — `g` is read,
//! never written — so a threshold that is far too aggressive costs a counter,
//! not a token. Results print via [`report`] alongside the profiler dump.

use crate::k3::expert_fp4::{section_bytes, GROUP};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Fault granularity. Nothing smaller than this can be skipped, whether the
/// bytes arrive by page fault or by `pread`.
const PAGE: usize = 4096;

/// A lane is dead at threshold `t` when `|g[lane]| <= t`.
///
/// `0.0` is the lossless floor — lanes whose gate is exactly zero contribute
/// exactly nothing, so whatever it reports is available for free. The other
/// three walk a decade at a time up to `1e-1`, which is the point where a lane's
/// contribution stops being lost in bf16 rounding of the accumulated output and
/// the trade starts costing real precision. Anything past that would be
/// measuring a model we would not ship, so it is not worth a counter.
pub const THRESHOLDS: [f32; 4] = [0.0, 1e-3, 1e-2, 1e-1];

static DEAD_LANES: [AtomicU64; THRESHOLDS.len()] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static DEAD_PAGES: [AtomicU64; THRESHOLDS.len()] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
/// Lanes examined, the denominator for [`DEAD_LANES`].
static TOTAL_LANES: AtomicU64 = AtomicU64::new(0);
/// `w3` pages examined, the denominator for [`DEAD_PAGES`].
static TOTAL_PAGES: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CASCADIA_K3_CHESS_PROBE").as_deref() == Ok("1"))
}

/// The byte ranges of `w3` that lane `lane` alone owns.
///
/// Two, not one: the packed section is `[inter * latent / 2]` nibble bytes
/// FOLLOWED BY `[inter * (latent / GROUP)]` E8M0 scale bytes, so a lane's row is
/// split across two regions rather than being one contiguous stride. Skipping a
/// lane only helps if BOTH of its pieces go unread, and the scale region packs
/// 16x more lanes per page than the nibble region, so it is by far the harder
/// one to clear — leaving it out would flatter the result badly.
fn lane_ranges(lane: usize, inter: usize, latent: usize) -> [(usize, usize); 2] {
    let nib_bytes = inter * latent / 2;
    [
        (lane * (latent / 2), latent / 2),
        (nib_bytes + lane * (latent / GROUP), latent / GROUP),
    ]
}

/// `(fully dead pages, total pages)` of the `w3` section.
///
/// A page counts as dead only if NO live lane touches it, which is the honest
/// bar: one live lane anywhere on a page faults the whole page in and the other
/// rows on it come along at no extra cost.
///
/// Split out from [`observe`] so it can be tested: the env gate is a
/// process-wide `OnceLock`, so a test cannot turn the probe on, and an
/// unverified probe is worth less than no probe.
fn dead_pages(dead: &[bool], inter: usize, latent: usize) -> (u64, u64) {
    let total = section_bytes(inter, latent).div_ceil(PAGE);
    let mut touched = vec![false; total];
    for (lane, &is_dead) in dead.iter().enumerate() {
        if is_dead {
            continue;
        }
        for (start, len) in lane_ranges(lane, inter, latent) {
            if len == 0 {
                continue;
            }
            let (first, last) = (start / PAGE, (start + len - 1) / PAGE);
            for t in touched[first..=last].iter_mut() {
                *t = true;
            }
        }
    }
    (touched.iter().filter(|&&t| !t).count() as u64, total as u64)
}

/// Dead-lane mask at threshold `t`.
fn dead_mask(g: &[f32], t: f32) -> Vec<bool> {
    g.iter().map(|v| v.abs() <= t).collect()
}

/// Record one expert forward. `g` is the `w1` output, `[inter]`.
pub fn observe(g: &[f32], inter: usize, latent: usize) {
    if !enabled() || g.len() != inter {
        return;
    }
    for (i, &t) in THRESHOLDS.iter().enumerate() {
        let mask = dead_mask(g, t);
        DEAD_LANES[i].fetch_add(
            mask.iter().filter(|&&d| d).count() as u64,
            Ordering::Relaxed,
        );
        let (dead, total) = dead_pages(&mask, inter, latent);
        DEAD_PAGES[i].fetch_add(dead, Ordering::Relaxed);
        if i == 0 {
            TOTAL_PAGES.fetch_add(total, Ordering::Relaxed);
        }
    }
    TOTAL_LANES.fetch_add(g.len() as u64, Ordering::Relaxed);
}

/// One field per threshold: dead lanes, then the `w3` pages that saves.
///
/// Read the two side by side. If `pages` tracks `lanes` the sparsity is dense
/// enough to act on; if `pages` stays near zero while `lanes` climbs, the dead
/// lanes are scattered and skipping them would read exactly the same bytes.
pub fn report() -> Option<String> {
    if !enabled() {
        return None;
    }
    let lanes = TOTAL_LANES.load(Ordering::Relaxed);
    let pages = TOTAL_PAGES.load(Ordering::Relaxed);
    if lanes == 0 || pages == 0 {
        return None;
    }
    let mut s = format!("K3_CHESS_PROBE lanes={lanes} pages={pages}");
    for (i, t) in THRESHOLDS.iter().enumerate() {
        s.push_str(&format!(
            " t={t}:lanes={:.1}%/pages={:.1}%",
            100.0 * DEAD_LANES[i].load(Ordering::Relaxed) as f64 / lanes as f64,
            100.0 * DEAD_PAGES[i].load(Ordering::Relaxed) as f64 / pages as f64,
        ));
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// K3's real expert shape, which is what the page arithmetic has to hold for.
    const INTER: usize = 3072;
    const LATENT: usize = 3584;

    #[test]
    fn thresholds_start_lossless_and_ascend() {
        // t = 0 is the free case: those lanes contribute exactly nothing, so
        // whatever it reports needs no accuracy argument at all.
        assert_eq!(THRESHOLDS[0], 0.0);
        assert!(THRESHOLDS.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(DEAD_LANES.len(), THRESHOLDS.len());
        assert_eq!(DEAD_PAGES.len(), THRESHOLDS.len());
    }

    #[test]
    fn lane_ranges_match_the_packed_section_layout() {
        // The nibble row is latent/2 bytes and the scale row is latent/GROUP,
        // with the scale region starting after ALL the nibbles. Getting the
        // second range wrong is the easy mistake, and it would silently inflate
        // the dead-page count.
        let [(n0, nl), (s0, sl)] = lane_ranges(0, INTER, LATENT);
        assert_eq!((n0, nl), (0, LATENT / 2), "nibble row stride is latent/2");
        assert_eq!(s0, INTER * LATENT / 2, "scales follow every nibble byte");
        assert_eq!(sl, LATENT / GROUP);

        let [(n1, _), (s1, _)] = lane_ranges(1, INTER, LATENT);
        assert_eq!(n1 - n0, LATENT / 2);
        assert_eq!(s1 - s0, LATENT / GROUP);

        // and the last lane must land inside the section, not past it
        let [_, (last, len)] = lane_ranges(INTER - 1, INTER, LATENT);
        assert_eq!(last + len, section_bytes(INTER, LATENT));
    }

    #[test]
    fn scattered_dead_lanes_free_no_pages() {
        // Every other lane dead — 50% of the rows — but no page is ever clear
        // of a live lane, so the section still has to be read in full. This is
        // the failure mode the whole measurement exists to expose.
        let dead: Vec<bool> = (0..INTER).map(|l| l % 2 == 0).collect();
        let (d, total) = dead_pages(&dead, INTER, LATENT);
        assert!(total > 0);
        assert_eq!(d, 0, "alternating dead lanes must save nothing");
    }

    #[test]
    fn contiguous_dead_lanes_free_pages() {
        // Same 50% dead, this time in one run. Now whole pages clear, and the
        // contrast with the scattered case is the entire finding.
        let dead: Vec<bool> = (0..INTER).map(|l| l >= INTER / 2).collect();
        let (d, total) = dead_pages(&dead, INTER, LATENT);
        assert!(
            d > 0,
            "a contiguous dead half must clear pages: {d} of {total}"
        );
        // The dead-lane fraction is the ceiling, since a page shared with a live
        // lane never clears. At these dims both regions divide evenly into pages
        // and the run starts on a boundary, so the ceiling is actually reached.
        assert!(d <= total / 2, "{d} of {total} exceeds the dead-lane share");
        assert_eq!(d, total / 2, "page-aligned run should give the full half");
    }

    #[test]
    fn all_lanes_dead_frees_every_page_and_none_dead_frees_none() {
        let (all, total) = dead_pages(&vec![true; INTER], INTER, LATENT);
        assert_eq!(all, total, "an entirely dead section is entirely skippable");
        let (none, _) = dead_pages(&vec![false; INTER], INTER, LATENT);
        assert_eq!(none, 0);
    }

    #[test]
    fn dead_mask_is_inclusive_and_sign_blind() {
        // Liveness is a magnitude test: a large negative gate is as live as a
        // large positive one, and a lane sitting exactly on the threshold is
        // dead, matching the `<=` in the doc comment.
        let g = [0.0f32, -0.5, 0.01, -0.01, 2.0];
        assert_eq!(dead_mask(&g, 0.0), [true, false, false, false, false]);
        assert_eq!(dead_mask(&g, 0.01), [true, false, true, true, false]);
        assert_eq!(dead_mask(&g, 1.0), [true, true, true, true, false]);
    }

    #[test]
    fn disabled_probe_records_nothing_and_reports_nothing() {
        // The env is unset under test, so the entry point must no-op. If this
        // regressed, every expert forward in production would allocate four
        // lane masks and a page bitmap.
        assert!(!enabled());
        observe(&vec![0.0f32; 32], 32, 32);
        assert_eq!(TOTAL_LANES.load(Ordering::Relaxed), 0);
        assert!(report().is_none());
    }
}
