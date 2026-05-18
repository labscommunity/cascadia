//! CLI: pull one (layer, expert) from a tahoma model_dir's safetensors,
//! run the int4 expert forward N times, then re-quantize to int2 and
//! run that N times, and print per-iter timings. Lets us measure the
//! kernel speedup independently of the full sparse-MoE engine.
//!
//! Usage::
//!
//!     bench_int2_vs_int4 --model <dir> --layer 30 --expert 5 --iters 50
//!
//! Output (stderr):
//!
//!     bench_int2_vs_int4 layer=30 expert=5 iters=50
//!       int4: total=... ms  per-iter=... ms
//!       int2: total=... ms  per-iter=... ms (quant=... ms)
//!       speedup: ...x

use std::path::PathBuf;
use std::time::Instant;

use half::bf16;
use tahoma_int4_gemm::kernel_int2::{expert_forward_int2, Int2Expert};
use tahoma_int4_gemm::{
    expert_forward as int4_expert_forward, SafetensorsExpertSource, HIDDEN, INTERMEDIATE,
};

fn parse_args() -> (PathBuf, u32, u32, usize) {
    let mut args = std::env::args().skip(1);
    let mut model: Option<PathBuf> = None;
    let mut layer: u32 = 30;
    let mut expert: u32 = 0;
    let mut iters: usize = 20;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next().map(PathBuf::from),
            "--layer" => layer = args.next().and_then(|s| s.parse().ok()).unwrap_or(layer),
            "--expert" => expert = args.next().and_then(|s| s.parse().ok()).unwrap_or(expert),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters),
            other => panic!("unknown arg: {other}"),
        }
    }
    (
        model.expect("--model required (path to model dir with safetensors/)"),
        layer,
        expert,
        iters,
    )
}

fn main() {
    let (model_dir, layer, expert, iters) = parse_args();
    let st_dir = model_dir.join("safetensors");
    let st_dir = if st_dir.exists() {
        st_dir
    } else {
        model_dir.clone()
    };
    let source = SafetensorsExpertSource::open(st_dir).expect("open safetensors");

    eprintln!("loading expert layer={layer} expert={expert}…");
    let e = source.expert(layer, expert).expect("get expert");

    // Input vector: deterministic ramp.
    let mut x = vec![0.0f32; HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = (i as f32 * 0.001).sin() * 0.5;
    }
    let x_bf16: Vec<bf16> = x.iter().map(|v| bf16::from_f32(*v)).collect();
    let mut out_bf16 = vec![bf16::ZERO; HIDDEN];

    // Warm int4.
    int4_expert_forward(
        &x_bf16,
        e.gate_packed,
        e.gate_scale,
        e.up_packed,
        e.up_scale,
        e.down_packed,
        e.down_scale,
        &mut out_bf16,
    );
    let t0 = Instant::now();
    for _ in 0..iters {
        int4_expert_forward(
            &x_bf16,
            e.gate_packed,
            e.gate_scale,
            e.up_packed,
            e.up_scale,
            e.down_packed,
            e.down_scale,
            &mut out_bf16,
        );
    }
    let int4_total = t0.elapsed();

    // Quantize to int2.
    eprintln!("re-quantizing expert to int2…");
    let t_q = Instant::now();
    let i2 = Int2Expert::from_int4(
        e.gate_packed,
        e.gate_scale,
        e.up_packed,
        e.up_scale,
        e.down_packed,
        e.down_scale,
        INTERMEDIATE,
        HIDDEN,
    );
    let quant_ms = t_q.elapsed().as_secs_f64() * 1000.0;

    // Warm int2.
    let mut out2_bf16 = vec![bf16::ZERO; HIDDEN];
    expert_forward_int2(
        &x_bf16,
        &i2.gate_packed,
        &i2.gate_scale,
        &i2.up_packed,
        &i2.up_scale,
        &i2.down_packed,
        &i2.down_scale,
        &mut out2_bf16,
    );
    let t1 = Instant::now();
    for _ in 0..iters {
        expert_forward_int2(
            &x_bf16,
            &i2.gate_packed,
            &i2.gate_scale,
            &i2.up_packed,
            &i2.up_scale,
            &i2.down_packed,
            &i2.down_scale,
            &mut out2_bf16,
        );
    }
    let int2_total = t1.elapsed();

    // Compare outputs (cosine, max abs diff).
    let mut dot = 0.0f64;
    let mut n1 = 0.0f64;
    let mut n2 = 0.0f64;
    let mut max_diff: f32 = 0.0;
    for i in 0..HIDDEN {
        let a = out_bf16[i].to_f32() as f64;
        let b = out2_bf16[i].to_f32() as f64;
        dot += a * b;
        n1 += a * a;
        n2 += b * b;
        let d = (out_bf16[i].to_f32() - out2_bf16[i].to_f32()).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    let cos = dot / (n1.sqrt() * n2.sqrt() + 1e-30);

    let footprint_int4_mb = (e.gate_packed.len()
        + e.gate_scale.len()
        + e.up_packed.len()
        + e.up_scale.len()
        + e.down_packed.len()
        + e.down_scale.len()) as f64
        / (1024.0 * 1024.0);
    let footprint_int2_mb = i2.footprint_bytes() as f64 / (1024.0 * 1024.0);

    eprintln!("bench_int2_vs_int4 layer={layer} expert={expert} iters={iters}");
    eprintln!("  footprint: int4 = {footprint_int4_mb:.2} MB, int2 = {footprint_int2_mb:.2} MB");
    eprintln!(
        "  int4: total={:.2} ms  per-iter={:.3} ms",
        int4_total.as_secs_f64() * 1000.0,
        int4_total.as_secs_f64() * 1000.0 / iters as f64,
    );
    eprintln!(
        "  int2: total={:.2} ms  per-iter={:.3} ms (quant={:.1} ms)",
        int2_total.as_secs_f64() * 1000.0,
        int2_total.as_secs_f64() * 1000.0 / iters as f64,
        quant_ms,
    );
    let speedup = int4_total.as_secs_f64() / int2_total.as_secs_f64();
    eprintln!("  speedup (int4 / int2): {speedup:.2}x");
    eprintln!("  output similarity: cosine={cos:.6}, max_abs_diff={max_diff:.4}");
}
