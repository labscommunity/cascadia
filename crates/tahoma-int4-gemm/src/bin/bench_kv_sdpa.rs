//! Microbench: bf16 vs int4 KV SDPA inner loops at K2.6 attention shapes.
//!
//! Compares the per-token attention cost of:
//!   1. f32 KV (pre-iter-032 baseline)
//!   2. bf16-as-u16 KV (iter 032 / A8 — current main)
//!   3. int4-packed-with-bf16-scale KV (this iter, scalar reference)
//!
//! For each KV format we measure the SDPA inner loop a single attention
//! head would do: `past_seq_len` × `QK_HEAD_DIM` k-dot-products + one
//! softmax + `past_seq_len` × `V_HEAD_DIM` v-accumulations.
//!
//! Output is per-call ns + per-element ns + GB/s read from cache. Run
//! with `cargo run --release --bin bench_kv_sdpa -p tahoma-int4-gemm`.

use std::time::Instant;

use tahoma_int4_gemm::int4_kv::{
    dequant_kv_accum_f32, dequant_kv_dot_f32, packed_bytes, quantize_kv_row,
};

const QK_HEAD_DIM: usize = 192;
const V_HEAD_DIM: usize = 128;

/// Round f32 → bf16 bits (RTE) — local copy so this binary does not
/// depend on the runner crate's `f32_to_bf16_bits`.
#[inline]
fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn deterministic_rand(n: usize, seed: u64) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let mut s = seed;
    for slot in out.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *slot = (((s >> 32) as u32 as f32) / 4_294_967_296.0 - 0.5) * 0.3;
    }
    out
}

struct Caches {
    past_k_f32: Vec<f32>,
    past_v_f32: Vec<f32>,
    past_k_bf16: Vec<u16>,
    past_v_bf16: Vec<u16>,
    past_k_int4_packed: Vec<u8>,
    past_k_int4_scales: Vec<u16>,
    past_v_int4_packed: Vec<u8>,
    past_v_int4_scales: Vec<u16>,
}

fn build_caches(past_seq_len: usize) -> Caches {
    let past_k_f32 = deterministic_rand(past_seq_len * QK_HEAD_DIM, 0x1111_1111);
    let past_v_f32 = deterministic_rand(past_seq_len * V_HEAD_DIM, 0x2222_2222);

    let past_k_bf16: Vec<u16> = past_k_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    let past_v_bf16: Vec<u16> = past_v_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();

    let mut past_k_int4_packed = Vec::with_capacity(past_seq_len * (QK_HEAD_DIM / 2));
    let mut past_k_int4_scales = Vec::with_capacity(past_seq_len * (QK_HEAD_DIM / 32));
    for j in 0..past_seq_len {
        let row = &past_k_f32[j * QK_HEAD_DIM..(j + 1) * QK_HEAD_DIM];
        let (packed, scales) = quantize_kv_row(row);
        past_k_int4_packed.extend_from_slice(&packed);
        past_k_int4_scales.extend_from_slice(&scales);
    }
    let mut past_v_int4_packed = Vec::with_capacity(past_seq_len * (V_HEAD_DIM / 2));
    let mut past_v_int4_scales = Vec::with_capacity(past_seq_len * (V_HEAD_DIM / 32));
    for j in 0..past_seq_len {
        let row = &past_v_f32[j * V_HEAD_DIM..(j + 1) * V_HEAD_DIM];
        let (packed, scales) = quantize_kv_row(row);
        past_v_int4_packed.extend_from_slice(&packed);
        past_v_int4_scales.extend_from_slice(&scales);
    }

    Caches {
        past_k_f32,
        past_v_f32,
        past_k_bf16,
        past_v_bf16,
        past_k_int4_packed,
        past_k_int4_scales,
        past_v_int4_packed,
        past_v_int4_scales,
    }
}

/// One head's SDPA forward over `past_seq_len` cached rows. Returns the
/// final accumulator so the optimizer can't elide the loop.
fn sdpa_f32(c: &Caches, q: &[f32], past_seq_len: usize) -> f32 {
    let scale = 1.0f32 / (QK_HEAD_DIM as f32).sqrt();
    let mut scores = vec![0.0f32; past_seq_len];
    for j in 0..past_seq_len {
        let k_row = &c.past_k_f32[j * QK_HEAD_DIM..(j + 1) * QK_HEAD_DIM];
        let mut s = 0.0f32;
        for i in 0..QK_HEAD_DIM {
            s += q[i] * k_row[i];
        }
        scores[j] = s * scale;
    }
    softmax_inplace(&mut scores);
    let mut out = vec![0.0f32; V_HEAD_DIM];
    for j in 0..past_seq_len {
        let v_row = &c.past_v_f32[j * V_HEAD_DIM..(j + 1) * V_HEAD_DIM];
        let w = scores[j];
        for i in 0..V_HEAD_DIM {
            out[i] += w * v_row[i];
        }
    }
    out.iter().sum()
}

fn sdpa_bf16(c: &Caches, q: &[f32], past_seq_len: usize) -> f32 {
    let scale = 1.0f32 / (QK_HEAD_DIM as f32).sqrt();
    let mut scores = vec![0.0f32; past_seq_len];
    for j in 0..past_seq_len {
        let k_row = &c.past_k_bf16[j * QK_HEAD_DIM..(j + 1) * QK_HEAD_DIM];
        let mut s = 0.0f32;
        for i in 0..QK_HEAD_DIM {
            let kf = f32::from_bits((k_row[i] as u32) << 16);
            s += q[i] * kf;
        }
        scores[j] = s * scale;
    }
    softmax_inplace(&mut scores);
    let mut out = vec![0.0f32; V_HEAD_DIM];
    for j in 0..past_seq_len {
        let v_row = &c.past_v_bf16[j * V_HEAD_DIM..(j + 1) * V_HEAD_DIM];
        let w = scores[j];
        for i in 0..V_HEAD_DIM {
            let vf = f32::from_bits((v_row[i] as u32) << 16);
            out[i] += w * vf;
        }
    }
    out.iter().sum()
}

fn sdpa_int4(c: &Caches, q: &[f32], past_seq_len: usize) -> f32 {
    let scale = 1.0f32 / (QK_HEAD_DIM as f32).sqrt();
    let k_packed_bytes = QK_HEAD_DIM / 2;
    let k_scale_count = QK_HEAD_DIM / 32;
    let v_packed_bytes = V_HEAD_DIM / 2;
    let v_scale_count = V_HEAD_DIM / 32;
    let mut scores = vec![0.0f32; past_seq_len];
    for j in 0..past_seq_len {
        let packed = &c.past_k_int4_packed[j * k_packed_bytes..(j + 1) * k_packed_bytes];
        let scales = &c.past_k_int4_scales[j * k_scale_count..(j + 1) * k_scale_count];
        scores[j] = dequant_kv_dot_f32(q, packed, scales) * scale;
    }
    softmax_inplace(&mut scores);
    let mut out = vec![0.0f32; V_HEAD_DIM];
    for j in 0..past_seq_len {
        let packed = &c.past_v_int4_packed[j * v_packed_bytes..(j + 1) * v_packed_bytes];
        let scales = &c.past_v_int4_scales[j * v_scale_count..(j + 1) * v_scale_count];
        dequant_kv_accum_f32(&mut out, scores[j], packed, scales);
    }
    out.iter().sum()
}

fn softmax_inplace(scores: &mut [f32]) {
    let mut max_s = scores[0];
    for &v in scores.iter().skip(1) {
        if v > max_s {
            max_s = v;
        }
    }
    let mut sum = 0.0f32;
    for v in scores.iter_mut() {
        *v = (*v - max_s).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in scores.iter_mut() {
        *v *= inv;
    }
}

fn bench<F: FnMut() -> f32>(name: &str, mut f: F, iters: usize, bytes_per_iter: usize) {
    // Warm
    for _ in 0..10 {
        let _ = f();
    }
    let t0 = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..iters {
        acc += f();
    }
    let dt = t0.elapsed().as_secs_f64();
    let per_iter_us = dt / iters as f64 * 1.0e6;
    let gbps = bytes_per_iter as f64 * iters as f64 / dt / 1e9;
    // Touch acc so the optimizer can't elide the calls.
    let sink = if acc.is_nan() { 1 } else { 0 };
    println!("{name:<14}  {per_iter_us:8.2} us/iter  {gbps:6.2} GB/s read   (sink={sink})");
}

fn run_at(past_seq_len: usize) {
    println!(
        "\n=== past_seq_len = {past_seq_len} (QK_HEAD_DIM={QK_HEAD_DIM}, V_HEAD_DIM={V_HEAD_DIM}) ==="
    );
    let caches = build_caches(past_seq_len);
    let q = deterministic_rand(QK_HEAD_DIM, 0x3333_3333);

    // Bytes read per head per SDPA forward
    let k_bytes_f32 = past_seq_len * QK_HEAD_DIM * 4;
    let v_bytes_f32 = past_seq_len * V_HEAD_DIM * 4;
    let k_bytes_bf16 = past_seq_len * QK_HEAD_DIM * 2;
    let v_bytes_bf16 = past_seq_len * V_HEAD_DIM * 2;
    let k_bytes_int4 = past_seq_len * packed_bytes(QK_HEAD_DIM);
    let v_bytes_int4 = past_seq_len * packed_bytes(V_HEAD_DIM);

    println!(
        "  per-head bytes read:  f32={:>5}  bf16={:>5}  int4={:>5}",
        k_bytes_f32 + v_bytes_f32,
        k_bytes_bf16 + v_bytes_bf16,
        k_bytes_int4 + v_bytes_int4,
    );

    let iters = if past_seq_len <= 64 {
        200_000
    } else if past_seq_len <= 256 {
        50_000
    } else {
        10_000
    };
    bench(
        "f32 SDPA",
        || sdpa_f32(&caches, &q, past_seq_len),
        iters,
        k_bytes_f32 + v_bytes_f32,
    );
    bench(
        "bf16 SDPA",
        || sdpa_bf16(&caches, &q, past_seq_len),
        iters,
        k_bytes_bf16 + v_bytes_bf16,
    );
    bench(
        "int4 SDPA",
        || sdpa_int4(&caches, &q, past_seq_len),
        iters,
        k_bytes_int4 + v_bytes_int4,
    );
}

fn main() {
    println!("=== K2.6 KV SDPA inner-loop bench ===");
    println!("Single-head, scalar kernels. Cache lives in L1/L2 at small past_seq_len.");
    println!(
        "`bytes read` is K + V per call; per-element cost scales with past_seq_len × HEAD_DIM."
    );

    // Steady-state decode: a 16-token prompt + a few generated tokens.
    run_at(16);
    // Just past iter 032's INITIAL_KV_CAPACITY = 32 (first cap grow).
    run_at(64);
    // The shape where bandwidth starts mattering — past_seq=256 K row alone
    // is 256 * 192 * 2 = 96 KB bf16 (fits in L2 on Lunar Lake / mostly fits
    // on the miner's 1 MB L2 per core).
    run_at(256);
    // Long context — typical of the autolab eval at mt=128.
    run_at(1024);
    // The shape where iter 050's eval lived (4 K context, code prompts).
    run_at(4096);
}
