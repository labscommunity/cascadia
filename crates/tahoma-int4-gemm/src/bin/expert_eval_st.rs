//! CLI variant of expert_eval that reads weights directly from the
//! safetensors shards in the K2.6 model dir — no on-disk dup.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use half::bf16;
use tahoma_int4_gemm::{expert_forward, SafetensorsExpertSource, HIDDEN};

fn parse_args() -> (PathBuf, u32, u32, PathBuf, PathBuf, usize) {
    let mut args = std::env::args().skip(1);
    let mut model_dir: Option<PathBuf> = None;
    let mut layer: Option<u32> = None;
    let mut expert: Option<u32> = None;
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut iters: usize = 1;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            "--layer" => layer = args.next().and_then(|s| s.parse().ok()),
            "--expert" => expert = args.next().and_then(|s| s.parse().ok()),
            "--input" => input = args.next().map(PathBuf::from),
            "--output" => output = args.next().map(PathBuf::from),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            other => panic!("unknown arg: {other}"),
        }
    }
    (
        model_dir.expect("--model-dir required"),
        layer.expect("--layer required"),
        expert.expect("--expert required"),
        input.expect("--input required"),
        output.expect("--output required"),
        iters,
    )
}

fn main() {
    let (model_dir, layer, expert, input_path, output_path, iters) = parse_args();

    let t0 = Instant::now();
    let source = SafetensorsExpertSource::open(&model_dir).expect("open safetensors");
    eprintln!(
        "opened source in {:.2}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    let t0 = Instant::now();
    let w = source.expert(layer, expert).expect("fetch expert");
    eprintln!(
        "fetched expert L{}/{} in {:.2}ms (mmap'd lazy)",
        layer,
        expert,
        t0.elapsed().as_secs_f64() * 1000.0
    );

    let input_bytes = fs::read(&input_path).expect("read input");
    assert_eq!(input_bytes.len(), HIDDEN * 4);
    let x_f32: &[f32] =
        unsafe { std::slice::from_raw_parts(input_bytes.as_ptr() as *const f32, HIDDEN) };
    let x_bf16: Vec<bf16> = x_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    let mut out_bf16 = vec![bf16::ZERO; HIDDEN];

    // Warm
    expert_forward(
        &x_bf16,
        w.gate_packed,
        w.gate_scale,
        w.up_packed,
        w.up_scale,
        w.down_packed,
        w.down_scale,
        &mut out_bf16,
    );

    let t0 = Instant::now();
    for _ in 0..iters {
        expert_forward(
            &x_bf16,
            w.gate_packed,
            w.gate_scale,
            w.up_packed,
            w.up_scale,
            w.down_packed,
            w.down_scale,
            &mut out_bf16,
        );
    }
    let elapsed = t0.elapsed();
    eprintln!(
        "expert_forward: {} iters in {:.2}ms ({:.3}ms/iter)",
        iters,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / iters as f64,
    );

    let out_f32: Vec<f32> = out_bf16.iter().map(|b| b.to_f32()).collect();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(out_f32.as_ptr() as *const u8, out_f32.len() * 4) };
    fs::write(&output_path, bytes).expect("write output");
}
