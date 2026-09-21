//! Opt-in selection counts from the actual router, including prefill and decode.
//! This is diagnostic only: no sampling, pruning, or changes to routing weights.
use super::{gate::GateOut, moe::RouteObserver};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Counts {
    rows: u64,
    selections: Vec<u64>,
}

impl Counts {
    fn observe(&mut self, gate: &GateOut) {
        self.rows += 1;
        for &id in &gate.idx {
            self.selections[id] += 1;
        }
    }

    fn summary(&self) -> [u64; 5] {
        let mut sorted = self.selections.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let total = sorted.iter().sum::<u64>().max(1);
        let ppm = |n| sorted.iter().take(n).sum::<u64>() * 1_000_000 / total;
        [
            ppm(16),
            ppm(32),
            ppm(64),
            ppm(1),
            sorted.iter().filter(|&&n| n > 0).count() as u64,
        ]
    }
}

pub(super) fn observer(layer: usize, experts: usize) -> RouteObserver {
    let counts = Mutex::new(Counts {
        rows: 0,
        selections: vec![0; experts],
    });
    // A changing tag lets the beacon retain later cumulative measurements.
    // At 1024 rows the whole fleet emits 64 short lines about once a minute.
    Arc::new(move |gate| {
        let mut counts = counts.lock().unwrap_or_else(|e| e.into_inner());
        counts.observe(gate);
        if counts.rows.is_multiple_of(1024) {
            let [top16, top32, top64, max, distinct] = counts.summary();
            let rows = counts.rows;
            println!(
                "EU{layer}_{rows} probe stage profile layer={layer} rows={rows} top16_ppm={top16} top32_ppm={top32} top64_ppm={top64} max_ppm={max} distinct={distinct}"
            );
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_routed_selections_and_excludes_shared_weights() {
        let mut counts = Counts {
            rows: 0,
            selections: vec![0; 256],
        };
        for i in 0..256 {
            counts.observe(&GateOut {
                idx: vec![i],
                w: vec![0.25],
                gammas: vec![0.5, 0.25],
            });
        }
        assert_eq!(counts.rows, 256);
        assert_eq!(counts.summary(), [62500, 125000, 250000, 3906, 256]);
        for _ in 0..256 {
            counts.observe(&GateOut {
                idx: vec![0],
                w: vec![0.25],
                gammas: vec![],
            });
        }
        assert_eq!(counts.summary(), [531250, 562500, 625000, 501953, 256]);
    }
}
