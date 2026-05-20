//! Microbench: dense vs sparse FFN forward at PowerInfer-port thresholds.
//!
//! Iterates `--iters` calls of `expert_forward` (dense baseline) and
//! `expert_forward_sparse` at thresholds `0.05`, `0.10`, `0.15`, `0.20`,
//! reporting:
//!
//!   - ns per call
//!   - speedup vs dense
//!   - average active-lane fraction
//!
//! Synthetic weights — runtime numbers are representative of K2.6
//! expert sizes (HIDDEN=7168, INTERMEDIATE=2048) but the actual
//! sparsity curve depends on real activations. Use this bench to
//! validate the *speedup* of skipping inactive lanes; for the
//! quality / accuracy curve see the K26 end-to-end eval test
//! (`tests/k26_layer0_eval.rs` with `K26_MODEL_DIR` set).
//!
//! Run:
//!
//! ```text
//! cargo run --release --bin bench_ffn_sparsity -- --iters 200
//! ```
//!
//! On miner (Cascade Lake AVX-512+VNNI) the AVX-512 path is taken;
//! on Apple Silicon the scalar path runs (much slower, but
//! correctness-checks).

use std::time::Instant;

use cascadia_int4_gemm::{
    dequant_gemv_int4_auto, dequant_gemv_int4_rows_subset_auto, expert_forward,
    expert_forward_sparse, GROUP_SIZE, HIDDEN, INTERMEDIATE,
};
use half::bf16;

fn parse_args() -> (usize, Vec<f32>) {
    let mut iters: usize = 200;
    let mut thresholds: Vec<f32> = vec![0.00, 0.05, 0.10, 0.15, 0.20];
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--iters" => {
                iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters);
            }
            "--thresholds" => {
                if let Some(s) = args.next() {
                    thresholds = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                }
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    (iters, thresholds)
}

/// Generate a synthetic packed int4 weight matrix. The byte pattern
/// is chosen to produce roughly-uniform nibble values, so the
/// dequantized weights span the full [-8, 7] range — gives the gate
/// matmul a realistic-looking output distribution.
fn make_synthetic_packed(n_rows: usize, k_cols: usize) -> Vec<u8> {
    let n_bytes = n_rows * k_cols / 2;
    (0..n_bytes)
        .map(|i| {
            let lo = (i * 31 + 7) & 0x0F;
            let hi = (i * 53 + 11) & 0x0F;
            ((hi << 4) | lo) as u8
        })
        .collect()
}

fn make_synthetic_scale_bits(n_rows: usize, n_groups: usize) -> Vec<u8> {
    // bf16 1.0 = 0x3F80, written little-endian as [0x80, 0x3F].
    vec![0x80u8, 0x3Fu8].repeat(n_rows * n_groups)
}

fn make_synthetic_x(hidden: usize) -> Vec<bf16> {
    // Small-magnitude inputs centered near zero, like a residual
    // stream after LayerNorm.
    (0..hidden)
        .map(|i| bf16::from_f32(((i as f32) * 0.013 - 0.5) * 0.1))
        .collect()
}

fn fmt_us(secs: f64) -> String {
    let us = secs * 1.0e6;
    if us >= 1000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else if us >= 1.0 {
        format!("{:.2} µs", us)
    } else {
        format!("{:.0} ns", us * 1000.0)
    }
}

fn main() {
    let (iters, thresholds) = parse_args();
    println!("# bench_ffn_sparsity");
    println!(
        "iters={iters} | HIDDEN={HIDDEN} | INTERMEDIATE={INTERMEDIATE} | GROUP_SIZE={GROUP_SIZE}"
    );
    println!();

    let n_in_groups = HIDDEN / GROUP_SIZE;
    let n_mid_groups = INTERMEDIATE / GROUP_SIZE;

    let gate_packed = make_synthetic_packed(INTERMEDIATE, HIDDEN);
    let gate_scale = make_synthetic_scale_bits(INTERMEDIATE, n_in_groups);
    let up_packed = make_synthetic_packed(INTERMEDIATE, HIDDEN);
    let up_scale = make_synthetic_scale_bits(INTERMEDIATE, n_in_groups);
    let down_packed = make_synthetic_packed(HIDDEN, INTERMEDIATE);
    let down_scale = make_synthetic_scale_bits(HIDDEN, n_mid_groups);
    let x = make_synthetic_x(HIDDEN);

    // Pre-warm to avoid first-call cold-cache cost in the timing.
    let mut warm = vec![bf16::ZERO; HIDDEN];
    expert_forward(
        &x,
        &gate_packed,
        &gate_scale,
        &up_packed,
        &up_scale,
        &down_packed,
        &down_scale,
        &mut warm,
    );

    // --- Dense baseline ---
    let mut out_dense = vec![bf16::ZERO; HIDDEN];
    let t0 = Instant::now();
    for _ in 0..iters {
        expert_forward(
            &x,
            &gate_packed,
            &gate_scale,
            &up_packed,
            &up_scale,
            &down_packed,
            &down_scale,
            &mut out_dense,
        );
    }
    let dense_elapsed = t0.elapsed().as_secs_f64();
    let dense_per_call = dense_elapsed / iters as f64;
    println!("| threshold | per-call    | speedup | active-frac |");
    println!("| --------- | ----------- | ------- | ----------- |");
    println!(
        "| dense     | {:>11} | 1.00×   | 1.000       |",
        fmt_us(dense_per_call)
    );

    // --- Sparse paths ---
    for &tau in &thresholds {
        let mut out = vec![bf16::ZERO; HIDDEN];
        let mut frac_sum = 0.0f64;
        let t0 = Instant::now();
        for _ in 0..iters {
            let f = expert_forward_sparse(
                &x,
                &gate_packed,
                &gate_scale,
                &up_packed,
                &up_scale,
                &down_packed,
                &down_scale,
                &mut out,
                tau,
            );
            frac_sum += f as f64;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_call = elapsed / iters as f64;
        let speedup = dense_per_call / per_call;
        let frac_avg = frac_sum / iters as f64;
        println!(
            "| τ={:<6.3} | {:>11} | {:<5.2}×  | {:<11.3} |",
            tau,
            fmt_us(per_call),
            speedup,
            frac_avg
        );
    }
    println!();
    println!("Notes:");
    println!(" - τ=0.00 path falls through to dense; speedup ≈1.00 confirms no overhead.");
    println!(" - Active-frac is the per-token fraction of intermediate lanes computed by");
    println!("   the up + down phases; (1 - active-frac) is the dropped work.");
    println!(" - On synthetic uniform weights the gate output distribution is flat, so");
    println!("   no lanes fall below the relative threshold. Real K2.6 activations have a");
    println!("   heavy tail that yields 30–50% sparsity at τ=0.10 (CATS/CHESS), making");
    println!("   the kernel speedup numbers below the practical upper bound.");
    println!();
    println!("---");
    println!();
    println!("# Direct kernel bench: dense GEMV vs sparse-rows GEMV");
    println!();
    println!(
        "Up matmul shape: rows=INTERMEDIATE={}, cols=HIDDEN={}.",
        INTERMEDIATE, HIDDEN
    );
    println!("Active fractions emulate the sparsity that real models exhibit.");
    println!();

    // Re-use the same up-projection weights for this bench.
    let mut y = vec![0.0f32; INTERMEDIATE];
    let x_f32: Vec<f32> = x.iter().map(|b| b.to_f32()).collect();
    // Dense baseline.
    let t0 = Instant::now();
    for _ in 0..iters {
        dequant_gemv_int4_auto(&up_packed, &up_scale, &x_f32, INTERMEDIATE, HIDDEN, &mut y);
    }
    let dense_kernel = t0.elapsed().as_secs_f64() / iters as f64;
    println!("| active-frac | per-call    | speedup |");
    println!("| ----------- | ----------- | ------- |");
    println!("| 1.00 (dense)| {:>11} | 1.00×   |", fmt_us(dense_kernel));
    for &frac in &[0.50f32, 0.30, 0.10] {
        let n_active = ((INTERMEDIATE as f32) * frac).round() as usize;
        let stride = INTERMEDIATE / n_active.max(1);
        let active: Vec<u32> = (0..n_active).map(|i| (i * stride) as u32).collect();
        let mut y = vec![0.0f32; INTERMEDIATE];
        let t0 = Instant::now();
        for _ in 0..iters {
            dequant_gemv_int4_rows_subset_auto(
                &up_packed,
                &up_scale,
                &x_f32,
                INTERMEDIATE,
                HIDDEN,
                &mut y,
                &active,
            );
        }
        let elapsed = t0.elapsed().as_secs_f64() / iters as f64;
        let speedup = dense_kernel / elapsed;
        println!(
            "| {:<11.2} | {:>11} | {:<5.2}×  |",
            frac,
            fmt_us(elapsed),
            speedup
        );
    }
    println!();
    println!("Direct kernel speedup ≈ 1 / active-frac (rayon parallelizes over active");
    println!("rows; the constant overhead floor sets the lower bound on per-call time).");
}
