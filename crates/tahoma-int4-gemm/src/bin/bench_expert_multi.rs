#![allow(clippy::too_many_arguments)]
//! Microbench: per-token expert_forward loop vs batched expert_forward_multi.
//!
//! Iter 051 (expert batching) validation. Compares two paths on a K2.6
//! expert FFN (HIDDEN=7168, INTERMEDIATE=2048):
//!   1. Per-token reference: `num_tokens` calls of [`expert_forward`]
//!      (each one loads ~21 MB of int4 weights from DRAM → `num_tokens × 21 MB`).
//!   2. Batched: one call of [`expert_forward_multi`] (weights loaded
//!      once and reused across `num_tokens` input rows → `~21 MB`).
//!
//! Run on the miner via:
//!
//! ```text
//! cargo run --release --bin bench_expert_multi -- --tokens 1,2,4,8 --iters 20
//! ```
//!
//! Per-iter ms and the multi-vs-per-token speedup are printed per
//! `num_tokens`. The point is the cross-token weight-reuse win — the
//! amortization should mirror what iter 042's shell-projection bench
//! showed (~1.4-4.75x at seq=2-16 depending on shape).
//!
//! For K2.6 spec-decode at K=4 verify width with ~50% expert reuse,
//! the engine-level effective num_tokens-per-expert is in the 1-4
//! range; this bench's num_tokens=2-4 numbers are the relevant ones
//! for spec-decode acceleration.

use std::time::Instant;

use half::bf16;
use tahoma_int4_gemm::{expert_forward, expert_forward_multi};

fn parse_args() -> (Vec<usize>, usize) {
    let mut args = std::env::args().skip(1);
    let mut tokens: Vec<usize> = vec![1, 2, 4, 8];
    let mut iters: usize = 20;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tokens" => {
                if let Some(s) = args.next() {
                    tokens = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                }
            }
            "--iters" => {
                iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters);
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    (tokens, iters)
}

const GROUP_SIZE: usize = 32;
const HIDDEN: usize = 7168;
const INTERMEDIATE: usize = 2048;

/// Returns (gate_packed, gate_scale, up_packed, up_scale, down_packed, down_scale).
/// Same deterministic pattern as `bench_int4_multi.rs::make_data`.
fn make_expert() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let (g_p, g_s) = make_w(INTERMEDIATE, HIDDEN, 1);
    let (u_p, u_s) = make_w(INTERMEDIATE, HIDDEN, 2);
    let (d_p, d_s) = make_w(HIDDEN, INTERMEDIATE, 3);
    (g_p, g_s, u_p, u_s, d_p, d_s)
}

fn make_w(n_rows: usize, k_cols: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    assert!(k_cols.is_multiple_of(GROUP_SIZE));
    let n_groups = k_cols / GROUP_SIZE;
    let mut packed = vec![0u8; n_rows * k_cols / 2];
    for r in 0..n_rows {
        for c in 0..(k_cols / 2) {
            let v = ((r
                .wrapping_mul(31)
                .wrapping_add(c)
                .wrapping_add(seed as usize))
                & 0xFF) as u8;
            packed[r * (k_cols / 2) + c] = v;
        }
    }
    let mut scales = vec![0u8; n_rows * n_groups * 2];
    for r in 0..n_rows {
        for g in 0..n_groups {
            let s = 0.05f32 + (((r * 7 + g * 3 + seed as usize) % 7) as f32) * 0.01;
            let bits = bf16_round(s);
            let off = (r * n_groups + g) * 2;
            scales[off] = (bits & 0xFF) as u8;
            scales[off + 1] = (bits >> 8) as u8;
        }
    }
    (packed, scales)
}

fn bf16_round(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn make_xs(num_tokens: usize, hidden: usize) -> Vec<bf16> {
    let mut xs = vec![bf16::ZERO; num_tokens * hidden];
    for t in 0..num_tokens {
        for i in 0..hidden {
            let v = (((t * 13 + i * 5) as f32).sin()) * 0.5;
            xs[t * hidden + i] = bf16::from_f32(v);
        }
    }
    xs
}

fn time_per_token_loop(
    xs: &[bf16],
    g_p: &[u8],
    g_s: &[u8],
    u_p: &[u8],
    u_s: &[u8],
    d_p: &[u8],
    d_s: &[u8],
    num_tokens: usize,
    ys: &mut [bf16],
    iters: usize,
) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iters {
        for t in 0..num_tokens {
            let x_t = &xs[t * HIDDEN..(t + 1) * HIDDEN];
            let y_t = &mut ys[t * HIDDEN..(t + 1) * HIDDEN];
            expert_forward(x_t, g_p, g_s, u_p, u_s, d_p, d_s, y_t);
        }
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn time_batched(
    xs: &[bf16],
    g_p: &[u8],
    g_s: &[u8],
    u_p: &[u8],
    u_s: &[u8],
    d_p: &[u8],
    d_s: &[u8],
    num_tokens: usize,
    ys: &mut [bf16],
    iters: usize,
) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iters {
        expert_forward_multi(xs, g_p, g_s, u_p, u_s, d_p, d_s, num_tokens, ys);
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    let (tokens, iters) = parse_args();
    eprintln!("bench_expert_multi: tokens={tokens:?} iters={iters}");
    #[cfg(target_arch = "x86_64")]
    {
        let avx512 = is_x86_feature_detected!("avx512f");
        let bw = is_x86_feature_detected!("avx512bw");
        let vl = is_x86_feature_detected!("avx512vl");
        eprintln!("  cpu features: avx512f={avx512} bw={bw} vl={vl}");
    }
    eprintln!("  rayon threads: {}", rayon::current_num_threads());
    eprintln!("  expert dims:   HIDDEN={HIDDEN} INTERMEDIATE={INTERMEDIATE}");
    eprintln!(
        "  weight bytes:  gate={:.2} MB, up={:.2} MB, down={:.2} MB, total={:.2} MB",
        (INTERMEDIATE * HIDDEN / 2) as f64 / 1e6,
        (INTERMEDIATE * HIDDEN / 2) as f64 / 1e6,
        (HIDDEN * INTERMEDIATE / 2) as f64 / 1e6,
        (3 * INTERMEDIATE * HIDDEN / 2) as f64 / 1e6,
    );
    println!();
    println!(
        "{:>8} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
        "tokens", "per_tok/ms", "batched/ms", "speedup", "max_diff", "ns/tok_b"
    );

    let (g_p, g_s, u_p, u_s, d_p, d_s) = make_expert();

    for &nt in &tokens {
        let xs = make_xs(nt, HIDDEN);
        let mut y_ref = vec![bf16::ZERO; nt * HIDDEN];
        let mut y_batched = vec![bf16::ZERO; nt * HIDDEN];

        // Warmup.
        let _ = time_per_token_loop(&xs, &g_p, &g_s, &u_p, &u_s, &d_p, &d_s, nt, &mut y_ref, 1);
        let _ = time_batched(
            &xs,
            &g_p,
            &g_s,
            &u_p,
            &u_s,
            &d_p,
            &d_s,
            nt,
            &mut y_batched,
            1,
        );

        // Measure.
        let t_ref = time_per_token_loop(
            &xs, &g_p, &g_s, &u_p, &u_s, &d_p, &d_s, nt, &mut y_ref, iters,
        );
        let t_b = time_batched(
            &xs,
            &g_p,
            &g_s,
            &u_p,
            &u_s,
            &d_p,
            &d_s,
            nt,
            &mut y_batched,
            iters,
        );

        // Correctness check (per cell).
        let mut max_diff = 0.0f32;
        for i in 0..(nt * HIDDEN) {
            let a = y_ref[i].to_f32();
            let b = y_batched[i].to_f32();
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        let ns_per_token_batched = t_b * 1e6 / nt as f64;
        println!(
            "{:>8} | {:>10.3} | {:>10.3} | {:>10.3}x | {:>10.4} | {:>10.0}",
            nt,
            t_ref,
            t_b,
            t_ref / t_b,
            max_diff,
            ns_per_token_batched,
        );
    }
}
