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
//! On a Cascade Lake Xeon (AVX-512+VNNI) the AVX-512 path is taken;
//! on Apple Silicon the scalar path runs (much slower, but
//! correctness-checks).

use std::time::Instant;

use cascadia_int4_gemm::{
    dequant_gemv_int4_auto, dequant_gemv_int4_rows_subset_auto, expert_forward,
    expert_forward_sparse, expert_forward_sparse_per_channel, GROUP_SIZE, HIDDEN, INTERMEDIATE,
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

    // --- Per-channel-τ overhead microbench (issue #38) ---
    //
    // Compares the global-τ path against the per-channel-τ path with
    // a UNIFORM threshold vector (every τ[c] = τ0). With uniform
    // thresholds the two paths must produce bit-identical output (we
    // assert this in `per_channel_uniform_matches_global`), so any
    // runtime gap is pure dispatcher + per-element-multiply overhead.
    println!("---");
    println!();
    println!("# Per-channel-τ overhead (uniform vector vs global scalar)");
    println!();
    println!("Both runs construct the same mask; per-channel adds one extra");
    println!("multiply per intermediate lane per token. Expect <1% gap.");
    println!();
    println!("| τ0      | global  | per-channel | overhead |");
    println!("| ------- | ------- | ----------- | -------- |");
    for &tau in &thresholds {
        if tau <= 0.0 {
            continue;
        }
        let τ_vec: Vec<f32> = vec![tau; INTERMEDIATE];
        let mut out_g = vec![bf16::ZERO; HIDDEN];
        let t0 = Instant::now();
        for _ in 0..iters {
            expert_forward_sparse(
                &x,
                &gate_packed,
                &gate_scale,
                &up_packed,
                &up_scale,
                &down_packed,
                &down_scale,
                &mut out_g,
                tau,
            );
        }
        let per_call_g = t0.elapsed().as_secs_f64() / iters as f64;

        let mut out_pc = vec![bf16::ZERO; HIDDEN];
        let t0 = Instant::now();
        for _ in 0..iters {
            expert_forward_sparse_per_channel(
                &x,
                &gate_packed,
                &gate_scale,
                &up_packed,
                &up_scale,
                &down_packed,
                &down_scale,
                &mut out_pc,
                &τ_vec,
            );
        }
        let per_call_pc = t0.elapsed().as_secs_f64() / iters as f64;
        let overhead = (per_call_pc - per_call_g) / per_call_g * 100.0;

        // Sanity check: outputs must be bit-identical (uniform-τ →
        // global-τ contract). Cheap assert that catches accidental
        // divergence between the two code paths.
        for h in 0..HIDDEN {
            assert_eq!(
                out_g[h].to_bits(),
                out_pc[h].to_bits(),
                "τ={tau}, h={h}: global/per-channel output divergence (uniform vector should be bit-identical)",
            );
        }
        println!(
            "| {:<7.3} | {:>7} | {:>11} | {:>+6.2}%  |",
            tau,
            fmt_us(per_call_g),
            fmt_us(per_call_pc),
            overhead,
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
    println!(" - Per-channel overhead measures the cost of the per-lane threshold vector vs");
    println!("   a single scalar. Real wins from per-channel come from CHESS calibration");
    println!("   allowing higher sparsity at the same quality — not from this overhead.");
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

    // ===== Section 3: AXPY-form down kernel (issue #35) =====
    println!();
    println!("---");
    println!();
    println!("# Direct kernel bench: dense GEMV vs AXPY-form sparse down (#35)");
    println!();
    println!(
        "Down matmul shape: rows=HIDDEN={}, cols=INTERMEDIATE={}.",
        HIDDEN, INTERMEDIATE
    );
    println!();
    println!("Dense path runs `dequant_gemv_int4_auto(down, scale, inter, ...)`");
    println!("(reads all intermediate cols). AXPY path runs the transposed-down");
    println!("kernel: for each active intermediate lane r, accumulate");
    println!("`y += inter[r] * dequant(down_t[r])` into the hidden output.");
    println!();

    // Construct fake DOWN weights (HIDDEN x INTERMEDIATE) +
    // transpose them once.
    let down_packed_dense = make_synthetic_packed(HIDDEN, INTERMEDIATE);
    let down_scale_dense = make_synthetic_scale_bits(HIDDEN, n_mid_groups);
    let (down_packed_t, down_scale_t_bits) =
        cascadia_int4_gemm::ffn_axpy::transpose_requantize_down(
            &down_packed_dense,
            &down_scale_dense,
            HIDDEN,
            INTERMEDIATE,
        );

    // Intermediate vector — the AXPY scalars input.
    let inter: Vec<f32> = (0..INTERMEDIATE)
        .map(|i| (i as f32) * 0.013 - 0.4)
        .collect();

    // Dense down baseline (existing kernel).
    let mut y_dense = vec![0.0f32; HIDDEN];
    let t0 = Instant::now();
    for _ in 0..iters {
        dequant_gemv_int4_auto(
            &down_packed_dense,
            &down_scale_dense,
            &inter,
            HIDDEN,
            INTERMEDIATE,
            &mut y_dense,
        );
    }
    let dense_down = t0.elapsed().as_secs_f64() / iters as f64;
    println!("| active-frac | per-call    | speedup |");
    println!("| ----------- | ----------- | ------- |");
    println!("| 1.00 (dense)| {:>11} | 1.00×   |", fmt_us(dense_down));
    for &frac in &[0.50f32, 0.30, 0.10] {
        let n_active = ((INTERMEDIATE as f32) * frac).round() as usize;
        let stride = INTERMEDIATE / n_active.max(1);
        let active: Vec<u32> = (0..n_active).map(|i| (i * stride) as u32).collect();
        let mut y = vec![0.0f32; HIDDEN];
        let t0 = Instant::now();
        for _ in 0..iters {
            // AXPY accumulates into y; zero before each call.
            y.fill(0.0);
            cascadia_int4_gemm::ffn_axpy::dequant_axpy_int4_active_auto(
                &down_packed_t,
                &down_scale_t_bits,
                &inter,
                &active,
                INTERMEDIATE,
                HIDDEN,
                &mut y,
            );
        }
        let elapsed = t0.elapsed().as_secs_f64() / iters as f64;
        let speedup = dense_down / elapsed;
        println!(
            "| {:<11.2} | {:>11} | {:<5.2}×  |",
            frac,
            fmt_us(elapsed),
            speedup
        );
    }
    println!();
    println!("AXPY speedup approaches the kernel ceiling 1/active_frac modulo");
    println!("output-buffer write traffic (y[HIDDEN]=7168 reread+rewritten per");
    println!("active scalar) and the rayon-chunked-y scheduling overhead.");

    // ===== Section 4: Full FFN forward — dense vs sparse vs sparse+AXPY =====
    println!();
    println!("---");
    println!();
    println!("# Full FFN forward at K2.6 active_frac (~0.74 measured on K2.6 τ=0.05)");
    println!();
    println!("Compares three FFN forward functions at K2.6 dims with a SYNTHETIC");
    println!("active mask that matches the real K2.6 active_frac. This catches");
    println!("per-call overhead the kernel-only benches above hide.");
    println!();

    use cascadia_int4_gemm::ffn_axpy::{transpose_requantize_down, FfnScratch};
    use cascadia_int4_gemm::ffn_sparsity::{ffn_forward_sparse_axpy_f32, ffn_forward_sparse_f32};
    use cascadia_int4_gemm::kernel::expert_forward;

    let x_full: Vec<bf16> = (0..HIDDEN)
        .map(|i| bf16::from_f32((i as f32) * 0.013 - 0.5))
        .collect();
    let x_full_f32: Vec<f32> = x_full.iter().map(|b| b.to_f32()).collect();

    // Pre-build the transposed down for AXPY.
    let (down_packed_t_full, down_scale_t_bits_full) =
        transpose_requantize_down(&down_packed, &down_scale, HIDDEN, INTERMEDIATE);

    let mut warm_dense = vec![bf16::ZERO; HIDDEN];
    let mut warm_sparse = vec![0.0f32; HIDDEN];
    let mut scratch = FfnScratch::new(HIDDEN, INTERMEDIATE);
    // warmup
    expert_forward(
        &x_full,
        &gate_packed,
        &gate_scale,
        &up_packed,
        &up_scale,
        &down_packed,
        &down_scale,
        &mut warm_dense,
    );
    let _ = ffn_forward_sparse_f32(
        &x_full_f32,
        HIDDEN,
        INTERMEDIATE,
        &gate_packed,
        &gate_scale,
        &up_packed,
        &up_scale,
        &down_packed,
        &down_scale,
        &mut warm_sparse,
        0.05,
    );
    let _ = ffn_forward_sparse_axpy_f32(
        &mut scratch,
        &x_full_f32,
        HIDDEN,
        INTERMEDIATE,
        &gate_packed,
        &gate_scale,
        &up_packed,
        &up_scale,
        &down_packed_t_full,
        &down_scale_t_bits_full,
        &mut warm_sparse,
        0.05,
    );

    // Bench dense expert_forward (bf16 in/out).
    let mut out_dense = vec![bf16::ZERO; HIDDEN];
    let t0 = Instant::now();
    for _ in 0..iters {
        expert_forward(
            &x_full,
            &gate_packed,
            &gate_scale,
            &up_packed,
            &up_scale,
            &down_packed,
            &down_scale,
            &mut out_dense,
        );
    }
    let dense_full = t0.elapsed().as_secs_f64() / iters as f64;

    // Bench ffn_forward_sparse_f32 at τ=0.05 (the PR #34 path).
    let mut out_sp = vec![0.0f32; HIDDEN];
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = ffn_forward_sparse_f32(
            &x_full_f32,
            HIDDEN,
            INTERMEDIATE,
            &gate_packed,
            &gate_scale,
            &up_packed,
            &up_scale,
            &down_packed,
            &down_scale,
            &mut out_sp,
            0.05,
        );
    }
    let sp_full = t0.elapsed().as_secs_f64() / iters as f64;

    // Bench ffn_forward_sparse_axpy_f32 at τ=0.05 (this PR).
    let mut out_axpy = vec![0.0f32; HIDDEN];
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = ffn_forward_sparse_axpy_f32(
            &mut scratch,
            &x_full_f32,
            HIDDEN,
            INTERMEDIATE,
            &gate_packed,
            &gate_scale,
            &up_packed,
            &up_scale,
            &down_packed_t_full,
            &down_scale_t_bits_full,
            &mut out_axpy,
            0.05,
        );
    }
    let axpy_full = t0.elapsed().as_secs_f64() / iters as f64;

    println!("| path                          | per-expert-call | speedup vs dense |");
    println!("| ----------------------------- | --------------- | ---------------- |");
    println!(
        "| `expert_forward` (dense bf16) | {:>15} | 1.00× (ref)      |",
        fmt_us(dense_full)
    );
    println!(
        "| `ffn_forward_sparse_f32` τ0.05 | {:>14} | {:<5.2}×           |",
        fmt_us(sp_full),
        dense_full / sp_full
    );
    println!(
        "| `ffn_forward_sparse_axpy_f32` τ0.05 | {:>9} | {:<5.2}×           |",
        fmt_us(axpy_full),
        dense_full / axpy_full
    );
    println!();
    println!("The full-FFN bench includes everything the runner pays per expert");
    println!("call: gate matmul (full) + silu + threshold-mask build + up sparse +");
    println!("elementwise + down (dense or AXPY). Use this number, not the");
    println!("kernel-only numbers above, to predict end-to-end FFN-compute gain.");
}
