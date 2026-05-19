//! Bench compression ratio + CPU cost on synthetic K2.6 hidden states.
//!
//! Runs three workloads (each meant to bracket a real-world
//! distribution) for each of None / Zstd / Lz4:
//!   1. `gaussian-like` — pseudo-Gaussian f32 in [-2, 2], mean ≈ 0
//!   2. `decode-tail` — single-token hidden state, K2.6 (7168 f32)
//!   3. `prefill-burst` — 16-token batch (114688 f32)
//!
//! For each (workload × scheme) it prints:
//!   - compression ratio (compressed_bytes / raw_bytes)
//!   - encode µs (single core)
//!   - decode µs
//!   - wire-time-saved-vs-None on the matias 117 ms RTT chain
//!     (approximated as (raw_bytes - comp_bytes) / 60 MB/s tunnel)
//!
//! Run:
//!   cargo run --release -p tahoma-transport --bin bench_wire_compression

use std::time::Instant;

use tahoma_transport::Compression;

/// Approximate sustained bandwidth of matias's SSH-tunnel chain
/// (iter 030 = 117 ms RTT). 60 MB/s is a conservative steady-state
/// throughput observed on multi-MB transfers. Tunneled traffic is
/// CPU-bound on the SSH process, not link-bound, so the number is
/// lower than the underlying LAN.
const MATIAS_TUNNEL_BW_MBPS: f64 = 60.0;

/// LAN bandwidth ceiling on the cascadia fleet (1 GbE).
const LAN_BW_MBPS: f64 = 110.0;

fn synth_gaussian(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    let mut x = 0.1f32;
    for i in 0..n {
        // pseudo-Gaussian-ish: tanh of a wandering sum scales to [-1, 1]
        x = (x * 1.31 + (i as f32) * 0.0007).sin();
        let v = (x * 1.2 + 0.4 * ((i as f32) * 0.13).cos()).tanh() * 1.8;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn synth_hidden_state(tokens: usize, hidden: usize) -> Vec<u8> {
    let n = tokens * hidden;
    let mut out = Vec::with_capacity(n * 4);
    // Per-token: mostly-small values with the occasional outlier.
    // Roughly matches what we've observed peeking at saved hidden
    // states (rainier scripts/dump_hidden.py).
    for ti in 0..tokens {
        let mut x = 0.1 + 0.013 * ti as f32;
        for di in 0..hidden {
            x = (x * 1.21 + (di as f32) * 0.0009).sin();
            let v = x * (0.5 + 0.4 * ((di as f32) * 0.0021).cos());
            // 1% outlier
            let v = if (ti * 7 + di) % 97 == 0 { v * 4.0 } else { v };
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn measure_one(name: &str, raw: &[u8], scheme: Compression, iters: usize) {
    // Warm.
    let comp = scheme.compress(raw).expect("compress warmup");
    let _ = scheme
        .decompress(&comp, raw.len())
        .expect("decompress warmup");

    let t0 = Instant::now();
    let mut last = Vec::new();
    for _ in 0..iters {
        last = scheme.compress(raw).expect("compress");
    }
    let encode_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = scheme.decompress(&last, raw.len()).expect("decompress");
    }
    let decode_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

    let ratio = last.len() as f64 / raw.len() as f64;
    let saved_bytes = raw.len() as f64 - last.len() as f64;
    let tunnel_saved_us = saved_bytes / (MATIAS_TUNNEL_BW_MBPS * 1e6) * 1e6;
    let lan_saved_us = saved_bytes / (LAN_BW_MBPS * 1e6) * 1e6;
    let net_tunnel_us = tunnel_saved_us - encode_us - decode_us;
    let net_lan_us = lan_saved_us - encode_us - decode_us;

    println!(
        "  {scheme:<5} ratio={ratio:.3} raw={raw:>7}B wire={wire:>7}B \
         enc={encode_us:>7.1}µs dec={decode_us:>7.1}µs \
         tunnel_net={net_tunnel_us:+8.1}µs lan_net={net_lan_us:+8.1}µs",
        scheme = format!("{scheme:?}"),
        raw = raw.len(),
        wire = last.len(),
    );
    let _ = name;
}

fn distribution_stats(raw: &[u8]) -> (f32, f32, f32, f32) {
    let mut vals: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let n = vals.len() as f32;
    let mean = vals.iter().sum::<f32>() / n;
    let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = vals[0];
    let max = *vals.last().unwrap();
    (mean, std, min, max)
}

fn run_workload(name: &str, raw: Vec<u8>, iters: usize) {
    let (mean, std, min, max) = distribution_stats(&raw);
    println!(
        "\n=== {name} ({} elements, {} B raw) — mean={mean:+.3} std={std:.3} min={min:+.3} max={max:+.3}",
        raw.len() / 4,
        raw.len()
    );
    for scheme in [Compression::None, Compression::Zstd, Compression::Lz4] {
        measure_one(name, &raw, scheme, iters);
    }
}

fn main() {
    println!("bench_wire_compression — synthetic K2.6 hidden states");
    println!("  matias-tunnel bandwidth = {MATIAS_TUNNEL_BW_MBPS} MB/s (iter 030, 117 ms RTT)");
    println!("  lan bandwidth          = {LAN_BW_MBPS} MB/s (1 GbE ceiling)");
    println!("  tunnel_net = wire_saved_us - encode - decode  (positive = win on matias tunnel)");
    println!("  lan_net    = wire_saved_us - encode - decode  (positive = win on cascadia LAN)");

    run_workload("gaussian-like 7168 f32", synth_gaussian(7168), 200);
    run_workload(
        "K2.6 single-token (1 × 7168 f32)",
        synth_hidden_state(1, 7168),
        200,
    );
    run_workload(
        "K2.6 prefill burst (16 × 7168 f32)",
        synth_hidden_state(16, 7168),
        50,
    );
}
