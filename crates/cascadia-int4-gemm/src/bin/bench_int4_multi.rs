#![allow(clippy::too_many_arguments)]
//! Microbench: scalar per-token loop vs AVX-512 tiled multi-token GEMM.
//!
//! Iter 042 prototype validation. Compares two paths:
//!   1. Scalar reference: `seq` calls of `dequant_gemv_int4_auto`
//!      (the iter 041 `_multi` implementation — `seq × W` weight motion).
//!   2. Tiled multi: one call of `dequant_gemm_int4_multi_auto` (weights
//!      stay in registers across all `seq` tokens — `~W` weight motion).
//!
//! Run on the miner via:
//!
//! ```text
//! cargo run --release --bin bench_int4_multi -- --shape qproj --iters 50
//! ```
//!
//! Shapes covered (K2.6 dims):
//!   - qproj    : N=1536, K=7168   (Q_LORA_RANK x HIDDEN; q_a_proj)
//!   - kvproj   : N=576,  K=7168   (KV_LORA_RANK+QK_ROPE x HIDDEN)
//!   - oproj    : N=7168, K=8192   (HIDDEN x NUM_HEADS*V_HEAD_DIM)
//!   - shared_gate : N=2048, K=7168
//!   - shared_down : N=7168, K=2048
//!   - tile     : N=64, K=64 (single-tile microbench, no parallel)
//!
//! For each shape, iterates seq in {1, 2, 4, 8, 16} and prints per-iter
//! ms and the multi-vs-scalar speedup.

use std::time::Instant;

use cascadia_int4_gemm::dequant_gemm_int4_multi_auto;
use cascadia_int4_gemm::dequant_gemm_int4_multi_blocked_auto;
use cascadia_int4_gemm::dequant_gemv_int4_auto;

fn parse_args() -> (String, usize, Vec<usize>) {
    let mut args = std::env::args().skip(1);
    let mut shape: String = "qproj".into();
    let mut iters: usize = 30;
    let mut seqs: Vec<usize> = vec![1, 2, 4, 8, 16];
    while let Some(a) = args.next() {
        match a.as_str() {
            "--shape" => shape = args.next().unwrap_or(shape),
            "--iters" => {
                iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters);
            }
            "--seqs" => {
                if let Some(s) = args.next() {
                    seqs = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                }
            }
            "--all" => shape = "all".into(),
            other => panic!("unknown arg: {other}"),
        }
    }
    (shape, iters, seqs)
}

/// (label, n_rows, k_cols)
fn shapes(name: &str) -> Vec<(&'static str, usize, usize)> {
    match name {
        "qproj" => vec![("qproj", 1536, 7168)],
        "kvproj" => vec![("kvproj", 576, 7168)],
        "oproj" => vec![("oproj", 7168, 8192)],
        "shared_gate" => vec![("shared_gate", 2048, 7168)],
        "shared_down" => vec![("shared_down", 7168, 2048)],
        "tile" => vec![("tile_64x64", 64, 64)],
        // Qwen3.6-35B-A3B expert shapes (qwen36 spec M2'-0 probe):
        // gate/up = [moe_intermediate=512, hidden=2048], down = transpose.
        "qwen36_expert" => vec![
            ("qwen36_gate_up", 512, 2048),
            ("qwen36_down", 2048, 512),
        ],
        "all" => vec![
            ("tile_64x64", 64, 64),
            ("qproj", 1536, 7168),
            ("kvproj", 576, 7168),
            ("shared_gate", 2048, 7168),
            ("shared_down", 7168, 2048),
            ("oproj", 7168, 8192),
        ],
        other => panic!(
            "unknown shape: {other} (use qproj|kvproj|oproj|shared_gate|shared_down|tile|qwen36_expert|all)"
        ),
    }
}

fn make_data(n_rows: usize, k_cols: usize, seq: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    const GROUP_SIZE: usize = 32;
    assert!(k_cols.is_multiple_of(GROUP_SIZE));
    let n_groups = k_cols / GROUP_SIZE;
    let mut packed = vec![0u8; n_rows * k_cols / 2];
    for r in 0..n_rows {
        for c in 0..(k_cols / 2) {
            let v = ((r.wrapping_mul(31).wrapping_add(c)) & 0xFF) as u8;
            packed[r * (k_cols / 2) + c] = v;
        }
    }
    let mut scales = vec![0u8; n_rows * n_groups * 2];
    for r in 0..n_rows {
        for g in 0..n_groups {
            let s = 0.5f32 + (((r * 7 + g * 3) % 11) as f32) * 0.1;
            let bits = bf16_round(s);
            let off = (r * n_groups + g) * 2;
            scales[off] = (bits & 0xFF) as u8;
            scales[off + 1] = (bits >> 8) as u8;
        }
    }
    let mut xs = vec![0.0f32; seq * k_cols];
    for t in 0..seq {
        for c in 0..k_cols {
            xs[t * k_cols + c] = ((t * 17 + c * 5) as f32).sin() * 0.5;
        }
    }
    (packed, scales, xs)
}

fn bf16_round(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn time_scalar_loop(
    packed: &[u8],
    scales: &[u8],
    xs: &[f32],
    n_rows: usize,
    k_cols: usize,
    seq: usize,
    ys: &mut [f32],
    iters: usize,
) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iters {
        for t in 0..seq {
            let x_t = &xs[t * k_cols..(t + 1) * k_cols];
            let y_t = &mut ys[t * n_rows..(t + 1) * n_rows];
            dequant_gemv_int4_auto(packed, scales, x_t, n_rows, k_cols, y_t);
        }
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn time_multi_tile(
    packed: &[u8],
    scales: &[u8],
    xs: &[f32],
    n_rows: usize,
    k_cols: usize,
    seq: usize,
    ys: &mut [f32],
    iters: usize,
) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iters {
        dequant_gemm_int4_multi_auto(packed, scales, xs, n_rows, k_cols, seq, ys);
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn time_blocked_tile(
    packed: &[u8],
    scales: &[u8],
    xs: &[f32],
    n_rows: usize,
    k_cols: usize,
    seq: usize,
    ys: &mut [f32],
    iters: usize,
) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iters {
        dequant_gemm_int4_multi_blocked_auto(packed, scales, xs, n_rows, k_cols, seq, ys);
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    let (shape_name, iters, seqs) = parse_args();
    let shape_list = shapes(&shape_name);
    eprintln!("bench_int4_multi: iters={iters}, seqs={seqs:?}");
    #[cfg(target_arch = "x86_64")]
    {
        let avx512 = is_x86_feature_detected!("avx512f");
        let bw = is_x86_feature_detected!("avx512bw");
        let vl = is_x86_feature_detected!("avx512vl");
        let vnni = is_x86_feature_detected!("avx512vnni");
        eprintln!("  cpu features: avx512f={avx512} bw={bw} vl={vl} vnni={vnni}");
    }
    let rayon_threads = rayon::current_num_threads();
    eprintln!("  rayon threads: {rayon_threads}");

    for (label, n_rows, k_cols) in shape_list {
        println!();
        println!(
            "=== shape {label}: N={n_rows} K={k_cols} (weight: {:.2} MB int4) ===",
            (n_rows * k_cols / 2) as f64 / (1024.0 * 1024.0)
        );
        println!(
            "{:>6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8}",
            "seq", "scalar/ms", "multi/ms", "blkd/ms", "mult_vs_sc", "blk_vs_mlt", "max_diff"
        );
        for &seq in &seqs {
            let (packed, scales, xs) = make_data(n_rows, k_cols, seq);
            let mut y_scalar = vec![0.0f32; seq * n_rows];
            let mut y_multi = vec![0.0f32; seq * n_rows];
            let mut y_blocked = vec![0.0f32; seq * n_rows];

            // Warmup: one run of each.
            {
                for t in 0..seq {
                    dequant_gemv_int4_auto(
                        &packed,
                        &scales,
                        &xs[t * k_cols..(t + 1) * k_cols],
                        n_rows,
                        k_cols,
                        &mut y_scalar[t * n_rows..(t + 1) * n_rows],
                    );
                }
                dequant_gemm_int4_multi_auto(
                    &packed,
                    &scales,
                    &xs,
                    n_rows,
                    k_cols,
                    seq,
                    &mut y_multi,
                );
                dequant_gemm_int4_multi_blocked_auto(
                    &packed,
                    &scales,
                    &xs,
                    n_rows,
                    k_cols,
                    seq,
                    &mut y_blocked,
                );
            }

            // Sanity check across all three paths. Scalar vs multi may
            // differ due to parallel reduction ordering across rows;
            // blocked vs multi should be bit-identical (same FMA sum
            // order per output cell).
            let mut max_diff: f32 = 0.0;
            for i in 0..(seq * n_rows) {
                let d = (y_scalar[i] - y_multi[i]).abs();
                if d > max_diff {
                    max_diff = d;
                }
                let d2 = (y_blocked[i] - y_multi[i]).abs();
                if d2 > max_diff {
                    max_diff = d2;
                }
            }

            // Time each path.
            let scalar_ms = time_scalar_loop(
                &packed,
                &scales,
                &xs,
                n_rows,
                k_cols,
                seq,
                &mut y_scalar,
                iters,
            );
            let multi_ms = time_multi_tile(
                &packed,
                &scales,
                &xs,
                n_rows,
                k_cols,
                seq,
                &mut y_multi,
                iters,
            );
            let blocked_ms = time_blocked_tile(
                &packed,
                &scales,
                &xs,
                n_rows,
                k_cols,
                seq,
                &mut y_blocked,
                iters,
            );
            let multi_vs_sc = scalar_ms / multi_ms;
            let blk_vs_mlt = multi_ms / blocked_ms;
            println!(
                "{:>6} | {:>10.3} | {:>10.3} | {:>10.3} | {:>9.2}x | {:>9.2}x | {:>8.2e}",
                seq, scalar_ms, multi_ms, blocked_ms, multi_vs_sc, blk_vs_mlt, max_diff,
            );
        }
    }
}
