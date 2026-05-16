//! CLI: load shell weights for a layer, run shell_forward_decode, print
//! per-output stats and write the 7 output tensors as raw f32 / i64 to
//! files for cross-validation against the OV reference.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use tahoma_int4_gemm::shell::{shell_forward_decode, HIDDEN, NUM_HEADS, QK_HEAD_DIM, V_HEAD_DIM};
use tahoma_int4_gemm::SafetensorsExpertSource;

fn parse_args() -> (PathBuf, u32, PathBuf, PathBuf, usize, usize) {
    let mut args = std::env::args().skip(1);
    let mut model_dir: Option<PathBuf> = None;
    let mut layer: Option<u32> = None;
    let mut input: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut past_seq_len: usize = 0;
    let mut iters: usize = 1;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            "--layer" => layer = args.next().and_then(|s| s.parse().ok()),
            "--input" => input = args.next().map(PathBuf::from),
            "--out-dir" => out_dir = args.next().map(PathBuf::from),
            "--past-seq-len" => {
                past_seq_len = args.next().and_then(|s| s.parse().ok()).unwrap_or(0)
            }
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            other => panic!("unknown arg: {other}"),
        }
    }
    (
        model_dir.expect("--model-dir required"),
        layer.expect("--layer required"),
        input.expect("--input required"),
        out_dir.expect("--out-dir required"),
        past_seq_len,
        iters,
    )
}

fn write_f32(path: &PathBuf, v: &[f32]) {
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    fs::write(path, bytes).expect("write f32");
}
fn write_i64(path: &PathBuf, v: &[i64]) {
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) };
    fs::write(path, bytes).expect("write i64");
}

fn main() {
    let (model_dir, layer, input_path, out_dir, past_seq_len, iters) = parse_args();
    fs::create_dir_all(&out_dir).unwrap();

    let t0 = Instant::now();
    let source = SafetensorsExpertSource::open(&model_dir).expect("open safetensors");
    eprintln!(
        "opened source in {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    let t0 = Instant::now();
    let shell = source.shell(layer).expect("fetch shell");
    eprintln!(
        "fetched shell L{} ({} tensors) in {:.1}ms",
        layer,
        14,
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Input: f32 [HIDDEN]
    let input_bytes = fs::read(&input_path).expect("read input");
    assert_eq!(input_bytes.len(), HIDDEN * 4);
    let x: &[f32] =
        unsafe { std::slice::from_raw_parts(input_bytes.as_ptr() as *const f32, HIDDEN) };

    // past_k / past_v: zeros for now (just measure decode at fresh).
    let past_k = vec![0.0f32; NUM_HEADS * past_seq_len * QK_HEAD_DIM];
    let past_v = vec![0.0f32; NUM_HEADS * past_seq_len * V_HEAD_DIM];

    // Warm
    let _ = shell_forward_decode(&shell, x, &past_k, &past_v, past_seq_len);

    let t0 = Instant::now();
    let mut last: Option<_> = None;
    for _ in 0..iters {
        last = Some(shell_forward_decode(
            &shell,
            x,
            &past_k,
            &past_v,
            past_seq_len,
        ));
    }
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "shell_forward: {} iters in {:.1}ms ({:.3}ms/iter)",
        iters,
        dt * 1000.0,
        dt / iters as f64 * 1000.0,
    );

    let o = last.unwrap();
    write_f32(
        &out_dir.join("attn_out_post_norm.f32.bin"),
        &o.attn_out_post_norm,
    );
    write_f32(&out_dir.join("attn_residual.f32.bin"), &o.attn_residual);
    write_f32(
        &out_dir.join("shared_expert_out.f32.bin"),
        &o.shared_expert_out,
    );
    write_i64(&out_dir.join("routing_ids.i64.bin"), &o.routing_ids);
    write_f32(&out_dir.join("routing_weights.f32.bin"), &o.routing_weights);
    write_f32(&out_dir.join("present_k.f32.bin"), &o.present_k);
    write_f32(&out_dir.join("present_v.f32.bin"), &o.present_v);

    eprintln!(
        "outputs (pos 0): attn_residual min={:.4} max={:.4} mean={:.4}",
        o.attn_residual
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        o.attn_residual
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
        o.attn_residual.iter().sum::<f32>() / o.attn_residual.len() as f32,
    );
    eprintln!("routing_ids: {:?}", &o.routing_ids[..8],);
    eprintln!("routing_weights: {:?}", &o.routing_weights[..8],);
}
