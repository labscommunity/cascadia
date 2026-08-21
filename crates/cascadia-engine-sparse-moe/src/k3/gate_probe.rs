//! Measures whether one layer's hidden predicts the NEXT layer's routing.
//!
//! Cross-layer gate prediction only pays if a prediction made one layer early
//! catches enough of the true top-k to be worth prefetching. Prefetching `M`
//! experts to catch `recall * top_k` of them costs `M - top_k` wasted fetches,
//! so a low recall makes the feature cost more I/O than it saves.
//!
//! This answers that with counters instead of a feature: score layer `L`'s gate
//! on layer `L-1`'s hidden, take the top-`M`, and count how much of layer `L`'s
//! real selection lands inside it. Nothing here changes routing or output — it
//! only observes, so a wrong prediction costs a counter, not a token.
//!
//! `CASCADIA_K3_GATE_PROBE=1` turns it on; it is off and branch-cheap otherwise.
//! Results print via [`report`] at the same time as the profiler dump.
//!
//! Prior work reports high cross-layer routing similarity, but those numbers are
//! from models with far fewer experts and a much larger top-k fraction. K3 picks
//! 16 of 896 — 1.8% — so the numbers do not transfer and have to be measured
//! here before anything is built on them.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

/// The candidate widths to score. Prefetching more than this is self-defeating:
/// at `M = 32` we would already be fetching twice the bytes a token needs.
pub const WIDTHS: [usize; 4] = [16, 20, 24, 32];

static HITS: [AtomicU64; WIDTHS.len()] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
/// Total true selections compared, the denominator for every width.
static TOTAL: AtomicU64 = AtomicU64::new(0);
/// Set once a prediction is staged, cleared once consumed, so the first MoE
/// layer of a token (which has no predecessor) is not counted as a miss.
static ARMED: AtomicBool = AtomicBool::new(false);

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CASCADIA_K3_GATE_PROBE").as_deref() == Ok("1"))
}

thread_local! {
    /// Top-`max(WIDTHS)` expert ids predicted for the next MoE layer, best first.
    static PREDICTED: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Score `gate` against `hidden` and stage the ranked candidates.
///
/// `gate` is the NEXT layer's router weights, `[n_experts * hidden]`, and
/// `hidden` is the CURRENT layer's activation — that one-layer offset is the
/// whole point, since a prediction is only useful if it lands before the layer
/// it describes.
pub fn predict(gate: &[f32], hidden: &[f32], n_experts: usize) {
    if !enabled() {
        return;
    }
    let ranked = rank_experts(gate, hidden, n_experts, WIDTHS[WIDTHS.len() - 1]);
    PREDICTED.with(|p| *p.borrow_mut() = ranked);
    ARMED.store(true, Ordering::Relaxed);
}

/// The top-`k` expert ids by gate score, best first.
///
/// Split out from [`predict`] so it can be tested: the env gate is a process-wide
/// `OnceLock`, so a test cannot turn the probe on, and an unverified probe is
/// worth less than no probe.
fn rank_experts(gate: &[f32], hidden: &[f32], n_experts: usize, k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = gate
        .chunks_exact(hidden.len())
        .take(n_experts)
        .enumerate()
        .map(|(e, row)| {
            let s: f32 = row.iter().zip(hidden).map(|(&a, &b)| a * b).sum();
            (s, e as u32)
        })
        .collect();
    let k = k.min(scored.len());
    if k == 0 {
        return Vec::new();
    }
    // Only the head is ever read, so order just that far.
    scored.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
    scored.truncate(k);
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().map(|(_, e)| e).collect()
}

/// Hits per width: how many of `actual` appear in the first `M` of `ranked`.
fn count_hits(ranked: &[u32], actual: &[u32]) -> [u64; WIDTHS.len()] {
    let mut out = [0u64; WIDTHS.len()];
    for (i, &m) in WIDTHS.iter().enumerate() {
        let head = &ranked[..m.min(ranked.len())];
        out[i] = actual.iter().filter(|e| head.contains(e)).count() as u64;
    }
    out
}

/// Compare a layer's real selection against the staged prediction.
pub fn observe(actual: &[u32]) {
    if !enabled() || !ARMED.swap(false, Ordering::Relaxed) {
        return;
    }
    PREDICTED.with(|p| {
        for (i, h) in count_hits(&p.borrow(), actual).iter().enumerate() {
            HITS[i].fetch_add(*h, Ordering::Relaxed);
        }
    });
    TOTAL.fetch_add(actual.len() as u64, Ordering::Relaxed);
}

/// One line per width: recall of the true selection within the top-`M`.
///
/// Read it against the break-even: catching `recall * top_k` experts is only
/// worth it if the misses saved beat the `M - top_k` fetches wasted. Below
/// ~80% at `M = 24` the idea should be dropped rather than built.
pub fn report() -> Option<String> {
    if !enabled() {
        return None;
    }
    let total = TOTAL.load(Ordering::Relaxed);
    if total == 0 {
        return None;
    }
    let mut s = format!("K3_GATE_PROBE n={total}");
    for (i, m) in WIDTHS.iter().enumerate() {
        let hits = HITS[i].load(Ordering::Relaxed);
        s.push_str(&format!(
            " top{m}={:.1}%",
            100.0 * hits as f64 / total as f64
        ));
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_are_ascending_and_start_at_the_real_top_k() {
        // top-16 is K3's actual selection width: recall there is the "free"
        // case, prefetching exactly as many experts as the token will use.
        assert_eq!(WIDTHS[0], 16);
        assert!(WIDTHS.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(HITS.len(), WIDTHS.len());
    }

    /// A gate whose expert `e` scores `e * hidden[0]`, so the true ranking is
    /// simply descending expert id — a known answer to check the sort against.
    fn ramp_gate(n_experts: usize, hidden_len: usize) -> Vec<f32> {
        let mut g = vec![0.0f32; n_experts * hidden_len];
        for e in 0..n_experts {
            g[e * hidden_len] = e as f32;
        }
        g
    }

    #[test]
    fn ranking_is_by_descending_score_and_stops_at_k() {
        let (n, hl) = (40usize, 4usize);
        let hidden = {
            let mut h = vec![0.0f32; hl];
            h[0] = 1.0;
            h
        };
        let ranked = rank_experts(&ramp_gate(n, hl), &hidden, n, 32);
        assert_eq!(ranked.len(), 32, "must return exactly k");
        let expect: Vec<u32> = (0..n as u32).rev().take(32).collect();
        assert_eq!(ranked, expect, "highest-scoring expert must come first");
    }

    #[test]
    fn ranking_handles_fewer_experts_than_k() {
        // A short layer must not panic or over-read; select_nth on k-1 is a
        // panic if k exceeds the length, which is why k is clamped.
        let (n, hl) = (3usize, 4usize);
        let hidden = vec![1.0f32, 0.0, 0.0, 0.0];
        let ranked = rank_experts(&ramp_gate(n, hl), &hidden, n, 32);
        assert_eq!(ranked, vec![2, 1, 0]);
    }

    #[test]
    fn hits_are_counted_per_width_against_rank_position() {
        // ranked = [0,1,2,...]; an actual pick at position p counts for every
        // width M > p and no width below it. This is the whole measurement, so
        // an off-by-one here would silently flatter or bury the result.
        let ranked: Vec<u32> = (0..32).collect();
        // one inside top-16, one only inside top-24, one only inside top-32
        let hits = count_hits(&ranked, &[3, 20, 30]);
        assert_eq!(hits[0], 1, "top16 catches only expert 3");
        assert_eq!(hits[1], 1, "top20 still catches only expert 3");
        assert_eq!(hits[2], 2, "top24 adds expert 20");
        assert_eq!(hits[3], 3, "top32 adds expert 30");

        // an expert outside every width counts nowhere
        assert_eq!(count_hits(&ranked, &[99]), [0, 0, 0, 0]);
        // perfect prediction saturates at the selection size
        assert_eq!(count_hits(&ranked, &[0, 1, 2])[0], 3);
    }

    #[test]
    fn disabled_probe_records_nothing_and_reports_nothing() {
        // The env is unset under test, so every entry point must no-op. If this
        // ever regresses, the probe would be scoring a full gate matmul per
        // layer per token in production.
        assert!(!enabled());
        predict(&[1.0; 8], &[1.0; 4], 2);
        observe(&[0, 1]);
        assert_eq!(TOTAL.load(Ordering::Relaxed), 0);
        assert!(report().is_none());
    }
}
