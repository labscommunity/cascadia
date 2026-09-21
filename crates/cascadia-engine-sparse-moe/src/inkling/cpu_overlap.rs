//! Before-serving diagnostic: can the real CPU attention/router kernels run
//! during an independent fused-expert call? This prices an optimistic overlap
//! bound; it does not implement a legal inter-frame scheduler or change serving.
use super::{model::Layer, ov_moe::OvMoe};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    layers: &[Layer],
    lo: usize,
    moe: &Arc<OvMoe>,
    hidden: usize,
    routed: usize,
    shared: usize,
    top_k: usize,
) -> Result<(), String> {
    let Some((offset, layer)) = layers
        .iter()
        .enumerate()
        .find(|(i, l)| l.moe().is_some() && moe.has_layer((lo + i) as u32))
    else {
        return Ok(());
    };
    let lid = (lo + offset) as u32;
    for rows in [1usize, 2] {
        let x: Vec<f32> = (0..rows * hidden)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
            .collect();
        let k = top_k + shared;
        let mut ids = Vec::new();
        for row in 0..rows {
            ids.extend((0..top_k).map(|j| ((row * 17 + j * 43) % routed) as i32));
            ids.extend((0..shared).map(|s| (routed + s) as i32));
        }
        let weights = vec![1.0 / k as f32; rows * k];
        let device = || moe.forward(lid, &x, rows, &ids, &weights);
        let expected = device().ok_or("CPU-overlap probe requires working GPU")?;
        for context in [128usize, 512] {
            let mut cpu = layer.cpu_overlap_work(rows, context);
            for _ in 0..3 {
                cpu();
                std::hint::black_box(device().ok_or("CPU-overlap GPU call failed")?);
            }
            let n = 41;
            let start = Instant::now();
            for _ in 0..n {
                cpu();
            }
            let cpu_us = start.elapsed().as_micros() / n;
            let start = Instant::now();
            for _ in 0..n {
                std::hint::black_box(device().ok_or("CPU-overlap GPU call failed")?);
            }
            let device_us = start.elapsed().as_micros() / n;
            let start = Instant::now();
            for _ in 0..n {
                cpu();
                std::hint::black_box(device().ok_or("CPU-overlap GPU call failed")?);
            }
            let serial_us = start.elapsed().as_micros() / n;
            let start = Instant::now();
            // One worker thread for the whole trial, avoiding per-call spawn
            // costs. Barriers start matched CPU/device iterations together.
            let barrier = std::sync::Barrier::new(2);
            let same = std::thread::scope(|scope| {
                scope.spawn(|| {
                    for _ in 0..n {
                        barrier.wait();
                        cpu();
                        barrier.wait();
                    }
                });
                let mut same = true;
                for _ in 0..n {
                    barrier.wait();
                    let result = device();
                    same &= result.as_ref() == Some(&expected);
                    barrier.wait();
                }
                same
            });
            let overlap_us = start.elapsed().as_micros() / n;
            let line=format!("CO{lid}R{rows}C{context} probe stage profile layer={lid} rows={rows} context={context} cpu_us={cpu_us} device_us={device_us} serial_us={serial_us} overlap_us={overlap_us} same={}",u8::from(same));
            for _ in 0..7 {
                println!("{line}");
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::dsv4::loader::ExpertsMode;
    use crate::inkling::loader::{load_stage, read_manifest, ExpertSet};
    #[test]
    fn diagnostic_cpu_work_preserves_live_sequence_state() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export");
        let count = read_manifest(&dir).unwrap().num_layers;
        assert!(count > 0);
        let mut actual = load_stage(
            &dir,
            128,
            0,
            count,
            true,
            true,
            ExpertsMode::Mmap,
            ExpertSet::All,
        )
        .unwrap();
        let mut control = load_stage(
            &dir,
            128,
            0,
            count,
            true,
            true,
            ExpertsMode::Mmap,
            ExpertSet::All,
        )
        .unwrap();
        let hidden = actual.manifest.hidden_size;
        let input = vec![0.1; hidden];
        for (a, b) in actual.layers.iter_mut().zip(&mut control.layers) {
            assert_eq!(a.forward_token(&input), b.forward_token(&input));
            for rows in [1, 2] {
                let mut probe = a.cpu_overlap_work(rows, 32);
                for _ in 0..5 {
                    probe();
                }
            }
            assert_eq!(a.forward_token(&input), b.forward_token(&input));
        }
    }
}
