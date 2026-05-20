//! Bench bf16 GEMV at the shapes used by K2.6 shell projections.

use std::time::Instant;

use cascadia_int4_gemm::kernel_bf16::bf16_gemv_auto;

fn bench(name: &str, n: usize, k: usize, iters: usize) {
    let weight = vec![0u8; n * k * 2];
    let x = vec![0.0f32; k];
    let mut y = vec![0.0f32; n];
    // Warm
    bf16_gemv_auto(&weight, &x, n, k, &mut y);
    let t0 = Instant::now();
    for _ in 0..iters {
        bf16_gemv_auto(&weight, &x, n, k, &mut y);
    }
    let dt = t0.elapsed().as_secs_f64();
    let bytes_read = (n * k * 2) as f64 * iters as f64;
    let gbps = bytes_read / dt / 1e9;
    let ms_per = dt / iters as f64 * 1000.0;
    println!("{name:<25}  [{n}x{k}]  {ms_per:6.2} ms/iter  ({gbps:5.1} GB/s read)");
}

fn main() {
    let iters = 50;
    println!("=== K2.6 shell projections (bf16 weight) ===");
    bench("q_a_proj", 1536, 7168, iters);
    bench("q_b_proj", 12288, 1536, iters);
    bench("kv_a_proj_with_mqa", 576, 7168, iters);
    bench("kv_b_proj", 16384, 512, iters);
    bench("o_proj", 7168, 8192, iters);
    bench("mlp.gate (router)", 384, 7168, iters);
    bench("shared.gate_proj", 2048, 7168, iters);
    bench("shared.up_proj", 2048, 7168, iters);
    bench("shared.down_proj", 7168, 2048, iters);

    let total_bytes_per_shell: u64 = (1536u64 * 7168
        + 12288u64 * 1536
        + 576u64 * 7168
        + 16384u64 * 512
        + 7168u64 * 8192
        + 384u64 * 7168
        + 2048u64 * 7168
        + 2048u64 * 7168
        + 7168u64 * 2048)
        * 2;
    println!(
        "\ntotal per shell:        {} MB",
        total_bytes_per_shell / 1_000_000
    );
    println!(
        "for 60 shells:          {} GB",
        60 * total_bytes_per_shell / 1_000_000_000
    );
}
