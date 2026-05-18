//! Bench bf16 and int4 GEMV at the shapes used by K2.6 shell projections.
//!
//! The bf16 numbers are kept as a historical reference — production
//! shells run through `shell_int4` and use the int4 kernel for every
//! big projection (q/kv/o, router, shared expert). The int4 line for
//! the router shape `[384, 7168]` is the apples-to-apples comparison
//! for the iter 055 router-quantization speedup.

use std::time::Instant;

use tahoma_int4_gemm::kernel_avx512::dequant_gemv_int4_auto;
use tahoma_int4_gemm::kernel_bf16::bf16_gemv_auto;
use tahoma_int4_gemm::GROUP_SIZE;

fn bench_bf16(name: &str, n: usize, k: usize, iters: usize) {
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
    println!("{name:<28}  [{n}x{k}]  {ms_per:6.2} ms/iter  ({gbps:5.1} GB/s read)");
}

fn bench_int4(name: &str, n: usize, k: usize, iters: usize) {
    // Zero weights are fine for cycle-count purposes — the kernel does
    // the same per-byte work regardless of nibble value, and the byte
    // value 0x88 corresponds to signed-zero under the +8 unpack.
    let packed = vec![0x88u8; n * k / 2];
    // 0x3f80 in LE = bf16(1.0), so each group's scale = 1.0 — keeps
    // the bench loop hot without inducing denormals.
    let n_groups = k / GROUP_SIZE;
    let mut scale = vec![0u8; n * n_groups * 2];
    for i in 0..(n * n_groups) {
        scale[i * 2] = 0x80;
        scale[i * 2 + 1] = 0x3f;
    }
    let x = vec![0.0f32; k];
    let mut y = vec![0.0f32; n];
    dequant_gemv_int4_auto(&packed, &scale, &x, n, k, &mut y);
    let t0 = Instant::now();
    for _ in 0..iters {
        dequant_gemv_int4_auto(&packed, &scale, &x, n, k, &mut y);
    }
    let dt = t0.elapsed().as_secs_f64();
    // Bytes read = packed (0.5 byte/elem) + scale (2 bytes per group of 32 elems)
    let weight_bytes = (n * k / 2) as f64 + (n * n_groups * 2) as f64;
    let bytes_read = weight_bytes * iters as f64;
    let gbps = bytes_read / dt / 1e9;
    let ms_per = dt / iters as f64 * 1000.0;
    println!("{name:<28}  [{n}x{k}]  {ms_per:6.2} ms/iter  ({gbps:5.1} GB/s read)");
}

fn main() {
    let iters = 50;
    println!("=== K2.6 shell projections (bf16 weight) ===");
    bench_bf16("q_a_proj (bf16)", 1536, 7168, iters);
    bench_bf16("q_b_proj (bf16)", 12288, 1536, iters);
    bench_bf16("kv_a_proj_with_mqa (bf16)", 576, 7168, iters);
    bench_bf16("kv_b_proj (bf16)", 16384, 512, iters);
    bench_bf16("o_proj (bf16)", 7168, 8192, iters);
    bench_bf16("mlp.gate router (bf16)", 384, 7168, iters);
    bench_bf16("shared.gate_proj (bf16)", 2048, 7168, iters);
    bench_bf16("shared.up_proj (bf16)", 2048, 7168, iters);
    bench_bf16("shared.down_proj (bf16)", 7168, 2048, iters);

    println!("\n=== Same shapes via int4 + bf16 scales (group=32) ===");
    println!("(production shell path runs every projection through this kernel)");
    bench_int4("q_a_proj (int4)", 1536, 7168, iters);
    bench_int4("q_b_proj (int4)", 12288, 1536, iters);
    bench_int4("kv_a_proj_with_mqa (int4)", 576, 7168, iters);
    bench_int4("kv_b_proj (int4)", 16384, 512, iters);
    bench_int4("o_proj (int4)", 7168, 8192, iters);
    bench_int4("mlp.gate router (int4)", 384, 7168, iters);
    bench_int4("shared.gate_proj (int4)", 2048, 7168, iters);
    bench_int4("shared.up_proj (int4)", 2048, 7168, iters);
    bench_int4("shared.down_proj (int4)", 7168, 2048, iters);

    let total_bytes_per_shell_bf16: u64 = (1536u64 * 7168
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
        "\nbf16 total per shell:        {} MB",
        total_bytes_per_shell_bf16 / 1_000_000
    );
    println!(
        "bf16 total for 60 shells:    {} GB",
        60 * total_bytes_per_shell_bf16 / 1_000_000_000
    );

    // int4: 0.5 byte/elem packed + 2 bytes per group of 32 elems = 0.5 + 2/32 = 0.5625 byte/elem.
    let total_bytes_per_shell_int4: u64 = total_bytes_per_shell_bf16 * 9 / 32;
    println!(
        "int4 total per shell:        {} MB",
        total_bytes_per_shell_int4 / 1_000_000
    );
    println!(
        "int4 total for 60 shells:    {} GB",
        60 * total_bytes_per_shell_int4 / 1_000_000_000
    );

    // Router-specific calls per token: 1 router GEMV per layer × 60 layers.
    let router_bytes_bf16 = 384u64 * 7168 * 2;
    let router_bytes_int4 = 384u64 * 7168 / 2 + 384u64 * (7168 / 32) * 2;
    println!(
        "\nrouter bytes per call: bf16 = {} kB, int4 = {} kB ({:.2}× reduction)",
        router_bytes_bf16 / 1024,
        router_bytes_int4 / 1024,
        router_bytes_bf16 as f64 / router_bytes_int4 as f64,
    );
    println!(
        "per-token router traffic (60 layers): bf16 = {} MB, int4 = {} MB",
        router_bytes_bf16 * 60 / 1_000_000,
        router_bytes_int4 * 60 / 1_000_000,
    );
}
