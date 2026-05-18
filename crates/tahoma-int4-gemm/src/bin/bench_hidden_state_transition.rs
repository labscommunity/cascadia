//! Bench the inter-layer hidden-state transition for K2.6.
//!
//! Background (iter 049): testing whether converting the inter-layer
//! hidden state from f32 to bf16 would meaningfully reduce per-token
//! latency. The hypothesis is that 60 × 28 KB = 1.7 MB / token in
//! f32 hidden-state traffic is the bottleneck and bf16 (14 KB) would
//! double inter-layer bandwidth.
//!
//! Two transitions measured:
//!
//! 1. **baseline (current path)** — in-place f32 mutation:
//!    `for j in 0..HIDDEN { h_f32[j] = residual[j] + shared[j] + moe[j] }`
//!    This is what `forward_shells` does at the end of each layer
//!    (runner.rs:670). The next layer's `shell_forward_decode_int4`
//!    reads `h_f32` directly. No copy, no conversion.
//!
//! 2. **hypothetical bf16** — same fused accumulate but the resulting
//!    hidden state is stored as bf16 (downcast at write, upcast at
//!    next layer's read). Total inter-layer cost = downcast + upcast.
//!
//! The bench reports per-iter ns AND derives the per-token total
//! across 60 layers, to compare against the iter 032 attention budget
//! (mean 687 ms/layer = 41.2 s/token).

use std::hint::black_box;
use std::time::Instant;

use half::bf16;

/// K2.6 hidden dimension.
const HIDDEN: usize = 7168;
/// K2.6 layer count.
const NUM_LAYERS: usize = 60;

/// Allocate three random-ish input vectors of size HIDDEN, plus an
/// output buffer. Returns (residual, shared, moe, out).
fn make_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut residual = vec![0.0f32; HIDDEN];
    let mut shared = vec![0.0f32; HIDDEN];
    let mut moe = vec![0.0f32; HIDDEN];
    let mut out = vec![0.0f32; HIDDEN];
    // Fill with non-trivial values so the compiler can't const-fold.
    for i in 0..HIDDEN {
        residual[i] = (i as f32 + 0.1) * 1e-3;
        shared[i] = ((i % 17) as f32) * 1e-4;
        moe[i] = ((i % 31) as f32) * 1e-4;
        out[i] = 0.0;
    }
    (residual, shared, moe, out)
}

/// Baseline: in-place f32 accumulate. The output buffer is the same
/// `h_f32` that gets passed into the next layer's shell forward.
fn transition_f32(residual: &[f32], shared: &[f32], moe: &[f32], h_f32: &mut [f32]) {
    for j in 0..HIDDEN {
        h_f32[j] = residual[j] + shared[j] + moe[j];
    }
}

/// Hypothetical: accumulate into bf16, then up-convert into f32 for
/// the next layer. Returns the cost of the two halves separately so
/// we can attribute the conversion cost honestly.
fn transition_bf16_downcast(residual: &[f32], shared: &[f32], moe: &[f32], h_bf16: &mut [u16]) {
    for j in 0..HIDDEN {
        let v = residual[j] + shared[j] + moe[j];
        h_bf16[j] = bf16::from_f32(v).to_bits();
    }
}

fn transition_bf16_upcast(h_bf16: &[u16], h_f32: &mut [f32]) {
    for j in 0..HIDDEN {
        h_f32[j] = f32::from_bits((h_bf16[j] as u32) << 16);
    }
}

/// Bench a closure that takes the precomputed inputs as black_box and
/// returns its body's per-iter ns averaged across `iters`.
fn bench_f32(name: &str, iters: usize) -> f64 {
    let (residual, shared, moe, mut out) = make_inputs();
    // Warm.
    transition_f32(&residual, &shared, &moe, &mut out);
    let t0 = Instant::now();
    for _ in 0..iters {
        transition_f32(
            black_box(&residual),
            black_box(&shared),
            black_box(&moe),
            black_box(&mut out),
        );
    }
    let dt_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!(
        "{name:<35}  {dt_ns:8.1} ns/iter ({:>6.1} GB/s write)",
        (HIDDEN * 4) as f64 / dt_ns
    );
    dt_ns
}

fn bench_bf16(name: &str, iters: usize) -> f64 {
    let (residual, shared, moe, _) = make_inputs();
    let mut h_bf16 = vec![0u16; HIDDEN];
    let mut h_f32 = vec![0.0f32; HIDDEN];
    // Warm.
    transition_bf16_downcast(&residual, &shared, &moe, &mut h_bf16);
    transition_bf16_upcast(&h_bf16, &mut h_f32);
    let t0 = Instant::now();
    for _ in 0..iters {
        transition_bf16_downcast(
            black_box(&residual),
            black_box(&shared),
            black_box(&moe),
            black_box(&mut h_bf16),
        );
        transition_bf16_upcast(black_box(&h_bf16), black_box(&mut h_f32));
    }
    let dt_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!(
        "{name:<35}  {dt_ns:8.1} ns/iter ({:>6.1} GB/s write+read)",
        (HIDDEN * 4 + HIDDEN * 2 + HIDDEN * 4) as f64 / dt_ns
    );
    dt_ns
}

/// Just the down-cast half (for attribution).
fn bench_bf16_downcast_only(name: &str, iters: usize) -> f64 {
    let (residual, shared, moe, _) = make_inputs();
    let mut h_bf16 = vec![0u16; HIDDEN];
    transition_bf16_downcast(&residual, &shared, &moe, &mut h_bf16);
    let t0 = Instant::now();
    for _ in 0..iters {
        transition_bf16_downcast(
            black_box(&residual),
            black_box(&shared),
            black_box(&moe),
            black_box(&mut h_bf16),
        );
    }
    let dt_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("{name:<35}  {dt_ns:8.1} ns/iter",);
    dt_ns
}

/// Just the up-cast half.
fn bench_bf16_upcast_only(name: &str, iters: usize) -> f64 {
    let mut h_bf16 = vec![0u16; HIDDEN];
    let mut h_f32 = vec![0.0f32; HIDDEN];
    for (i, slot) in h_bf16.iter_mut().enumerate() {
        *slot = bf16::from_f32(i as f32 * 1e-3).to_bits();
    }
    transition_bf16_upcast(&h_bf16, &mut h_f32);
    let t0 = Instant::now();
    for _ in 0..iters {
        transition_bf16_upcast(black_box(&h_bf16), black_box(&mut h_f32));
    }
    let dt_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("{name:<35}  {dt_ns:8.1} ns/iter",);
    dt_ns
}

fn main() {
    let iters = 200_000;
    println!("=== K2.6 inter-layer hidden-state transition bench ===");
    println!(
        "HIDDEN = {HIDDEN} ({} KB f32, {} KB bf16)",
        HIDDEN * 4 / 1024,
        HIDDEN * 2 / 1024
    );
    println!("Iterations per bench: {iters}");
    println!();

    let f32_ns = bench_f32("f32 in-place accumulate", iters);
    let bf16_ns = bench_bf16("bf16 down+upcast pair", iters);
    let dc_ns = bench_bf16_downcast_only("  bf16 downcast (accum + store)", iters);
    let uc_ns = bench_bf16_upcast_only("  bf16 upcast (read + store)", iters);

    println!();
    println!("=== Per-token totals across {NUM_LAYERS} layers ===");
    let f32_us = f32_ns * NUM_LAYERS as f64 / 1000.0;
    let bf16_us = bf16_ns * NUM_LAYERS as f64 / 1000.0;
    let dc_us = dc_ns * NUM_LAYERS as f64 / 1000.0;
    let uc_us = uc_ns * NUM_LAYERS as f64 / 1000.0;
    println!("f32 baseline:    {f32_us:8.1} us / token");
    println!("bf16 hypothetical: {bf16_us:8.1} us / token");
    println!("  downcast only: {dc_us:8.1} us / token");
    println!("  upcast only:   {uc_us:8.1} us / token");

    println!();
    println!("=== Comparison to iter 032 attention budget ===");
    let attn_us_per_layer = 687_000.0; // iter 032 mean shell_attn_us
    let attn_us_per_token = attn_us_per_layer * NUM_LAYERS as f64;
    let f32_pct = f32_us / attn_us_per_token * 100.0;
    let bf16_pct = bf16_us / attn_us_per_token * 100.0;
    let diff_us = bf16_us - f32_us;
    println!(
        "attention compute / token: {:.1} ms ({} layers × {:.0} ms)",
        attn_us_per_token / 1000.0,
        NUM_LAYERS,
        attn_us_per_layer / 1000.0
    );
    println!("f32 transition is  {:.4}% of attention compute", f32_pct);
    println!("bf16 transition is {:.4}% of attention compute", bf16_pct);
    println!();
    if diff_us > 0.0 {
        println!(
            "VERDICT: bf16 would ADD {:.1} us/token ({:.4}% of attention).",
            diff_us,
            diff_us / attn_us_per_token * 100.0
        );
        println!("         Hidden-state bf16 is NOT net-positive in single-stage.");
    } else {
        println!("VERDICT: bf16 would SAVE {:.1} us/token.", -diff_us);
    }

    println!();
    println!("=== Distributed (pipeline-parallel) consideration ===");
    let wire_rtt_ms = 22.4f64; // cascadia_fleet_deploy.md
    let wire_size_f32 = (HIDDEN * 4) as f64;
    let wire_size_bf16 = (HIDDEN * 2) as f64;
    println!("28 KB hidden state, measured RTT (LAN): {wire_rtt_ms} ms / hop");
    println!("On a 1 Gbps LAN, raw byte-transmit time:");
    println!(
        "  f32:  {:.3} ms ({:.0} bytes @ 125 MB/s)",
        wire_size_f32 / 125e6 * 1000.0,
        wire_size_f32
    );
    println!(
        "  bf16: {:.3} ms ({:.0} bytes @ 125 MB/s)",
        wire_size_bf16 / 125e6 * 1000.0,
        wire_size_bf16
    );
    println!(
        "RTT is {} the byte-transmit time — wire is latency-bound, not bandwidth-bound.",
        if wire_rtt_ms > wire_size_f32 / 125e6 * 1000.0 * 10.0 {
            "MUCH greater than"
        } else {
            "comparable to"
        }
    );
}
