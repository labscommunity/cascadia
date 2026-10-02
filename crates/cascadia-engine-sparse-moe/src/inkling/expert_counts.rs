//! Opt-in selection counts from the actual router, including prefill and decode.
//! This is diagnostic only: no sampling, pruning, or changes to routing weights.
use super::{gate::GateOut, moe::RouteObserver};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

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

#[derive(Default)]
pub(super) struct Reporter {
    layers: Vec<(usize, Weak<Mutex<Counts>>)>,
}

impl Reporter {
    pub(super) fn observer(&mut self, layer: usize, experts: usize) -> RouteObserver {
        let counts = Arc::new(Mutex::new(Counts {
            rows: 0,
            selections: vec![0; experts],
        }));
        self.layers.push((layer, Arc::downgrade(&counts)));
        Arc::new(move |gate| {
            counts
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .observe(gate);
        })
    }

    fn next_line(&mut self, tick: usize) -> Option<String> {
        self.layers.retain(|(_, counts)| counts.strong_count() > 0);
        let (layer, counts) = self.layers.get(tick % self.layers.len().max(1))?;
        let counts = counts.upgrade()?;
        let counts = counts.lock().unwrap_or_else(|e| e.into_inner());
        if counts.rows == 0 {
            return None;
        }
        let [top16, top32, top64, max, distinct] = counts.summary();
        let rows = counts.rows;
        Some(format!(
            "EU{layer}_{rows} probe stage profile layer={layer} rows={rows} top16_ppm={top16} top32_ppm={top32} top64_ppm={top64} max_ppm={max} distinct={distinct}"
        ))
    }

    pub(super) fn start(mut self) {
        if self.layers.is_empty() {
            return;
        }
        // The beacon polls the latest profile line every five seconds. Printing
        // all layers at once loses five of six; rotate one short line every seven
        // seconds instead. Repeated rounds also tolerate ordinary stage profiles
        // replacing an occasional report. No logging/sleep in the inference path.
        // Weak references let this diagnostic stop after its stage is dropped.
        if let Err(error) = std::thread::Builder::new()
            .name("expert-counts".into())
            .spawn(move || {
                for tick in 0.. {
                    std::thread::sleep(Duration::from_secs(7));
                    if let Some(line) = self.next_line(tick) {
                        println!("{line}");
                    }
                    if self.layers.is_empty() {
                        break;
                    }
                }
            })
        {
            eprintln!("[inkling] expert-count reporting disabled: {error}");
        }
    }
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

    #[test]
    fn reports_every_layer_and_does_not_keep_a_dropped_stage_alive() {
        let mut reporter = Reporter::default();
        let observers: Vec<_> = (6..12).map(|l| reporter.observer(l, 256)).collect();
        assert!(reporter.next_line(0).is_none());
        for observe in &observers {
            observe(&GateOut {
                idx: vec![7],
                w: vec![1.0],
                gammas: vec![],
            });
        }
        for tick in 0..12 {
            let line = reporter.next_line(tick).unwrap();
            assert!(line.contains(&format!("layer={} rows=1 ", 6 + tick % 6)));
            assert!(line.contains("top64_ppm=1000000"));
        }
        drop(observers);
        assert!(reporter.next_line(12).is_none());
        assert!(reporter.layers.is_empty());
    }
}
