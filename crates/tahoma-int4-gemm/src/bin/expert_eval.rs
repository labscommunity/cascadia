//! CLI: load one expert.bin, run forward on an input vector, print
//! output as raw f32 bytes (no JSON, no framing). Used by the Python
//! cross-validation harness on the miner.
//!
//! Usage::
//!
//!     expert_eval --expert <path.bin> --input <path.bin> --output <path.bin>
//!
//! Input file: HIDDEN=7168 f32 little-endian values.
//! Output file: HIDDEN=7168 f32 little-endian values written by this
//! binary.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use half::bf16;
use tahoma_int4_gemm::{expert_forward, ExpertWeights, HIDDEN};

fn parse_args() -> (PathBuf, PathBuf, PathBuf, usize) {
    let mut args = std::env::args().skip(1);
    let mut expert: Option<PathBuf> = None;
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut iters: usize = 1;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--expert" => expert = args.next().map(PathBuf::from),
            "--input" => input = args.next().map(PathBuf::from),
            "--output" => output = args.next().map(PathBuf::from),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            other => panic!("unknown arg: {other}"),
        }
    }
    (
        expert.expect("--expert required"),
        input.expect("--input required"),
        output.expect("--output required"),
        iters,
    )
}

fn main() {
    let (expert_path, input_path, output_path, iters) = parse_args();

    let expert = ExpertWeights::open(&expert_path).expect("open expert");
    let input_bytes = fs::read(&input_path).expect("read input");
    assert_eq!(
        input_bytes.len(),
        HIDDEN * 4,
        "input must be exactly HIDDEN * 4 bytes (f32)"
    );

    // f32 input -> bf16 (kernel takes bf16 in).
    let x_f32: &[f32] =
        unsafe { std::slice::from_raw_parts(input_bytes.as_ptr() as *const f32, HIDDEN) };
    let x_bf16: Vec<bf16> = x_f32.iter().map(|v| bf16::from_f32(*v)).collect();

    let mut out_bf16 = vec![bf16::ZERO; HIDDEN];

    // Warm
    expert_forward(
        &x_bf16,
        expert.gate_packed_bytes(),
        expert.gate_scale_bits(),
        expert.up_packed_bytes(),
        expert.up_scale_bits(),
        expert.down_packed_bytes(),
        expert.down_scale_bits(),
        &mut out_bf16,
    );

    let t0 = Instant::now();
    for _ in 0..iters {
        expert_forward(
            &x_bf16,
            expert.gate_packed_bytes(),
            expert.gate_scale_bits(),
            expert.up_packed_bytes(),
            expert.up_scale_bits(),
            expert.down_packed_bytes(),
            expert.down_scale_bits(),
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

    // Write f32 output (convert from bf16 for Python comparison).
    let out_f32: Vec<f32> = out_bf16.iter().map(|b| b.to_f32()).collect();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(out_f32.as_ptr() as *const u8, out_f32.len() * 4) };
    fs::write(&output_path, bytes).expect("write output");
}
