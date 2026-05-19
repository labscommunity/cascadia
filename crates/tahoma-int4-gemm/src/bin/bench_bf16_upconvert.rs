//! Microbench: how much of iter 032's bf16-as-u16 KV SDPA is the inline
//! upconvert (`f32::from_bits((bits as u32) << 16)`) vs the f32 dot/accum?
//!
//! Iter 032 stores K/V as `Vec<u16>` (bf16 bit pattern). The SDPA inner
//! loop upconverts each element to f32 inline before the multiply:
//!
//! ```text
//! for i in 0..QK_HEAD_DIM {
//!     let kf = f32::from_bits((k_row[i] as u32) << 16);
//!     s += q_h[i] * kf;
//! }
//! ```
//!
//! The proposed iter 064 work was a *native bf16 SDPA* using
//! `VDPBF16PS` (AVX-512 BF16). That extension is NOT available on either
//! target host:
//!   - miner = Xeon Gold 6252 (Cascade Lake) → no `avx512_bf16`
//!   - AI PC fleet = Lunar Lake → no AVX-512 at all
//!
//! So the only achievable win is restructuring the inline upconvert so
//! the compiler vectorises it cleanly into `VPMOVZXWD + VPSLLD +
//! VFMADD231PS` (or the AVX2 equivalent `VPMOVZXWD + VPSLLD +
//! VFMADD231PS` with 256-bit lanes). This binary measures *exactly how
//! much room there is* — if the inline upconvert is already a few percent
//! of total SDPA, there is no point shipping a hand-rolled intrinsic
//! version.
//!
//! Compares five variants at K2.6 attention shapes (QK_HEAD_DIM = 192,
//! V_HEAD_DIM = 128) across `past_seq_len ∈ {16, 64, 256, 1024, 4096}`:
//!
//! - `f32 SDPA`        — KV already f32, pure dot/accum (pre-iter-032)
//! - `bf16 inline`     — iter 032 baseline: scalar inline upconvert
//! - `bf16 split`      — pass 1 upconverts whole row into a scratch f32
//!   buffer (one tight `cvt` loop), pass 2 f32 dot —
//!   isolates "is the inline cvt blocking SIMD?"
//! - `bf16 upcvt only` — *just* the upconvert pass (k + v) — bytes-only
//!   cost of dequant, no math
//! - `f32 dot only`    — the dot/accum on f32 data, no cvt — what's left
//!   after you remove cvt entirely
//!
//! Read this output as:
//!   `(bf16 inline) - (f32 dot only) ≈ upconvert cost as observed
//!   in the fused loop`
//!   `(bf16 upcvt only)              ≈ upconvert cost in isolation`
//!   `(bf16 split) - (f32 dot only)  ≈ upconvert cost when given its
//!   own loop (best the compiler can do without intrinsics)`
//!
//! Decision rule:
//!   - if `(bf16 inline)` is within ~2% of `(bf16 split)` and within ~5%
//!     of `(f32 dot only) + (bf16 upcvt only)`: the inline upconvert is
//!     already cheap, do NOT ship intrinsics (kill the iter)
//!   - if `(bf16 split)` is noticeably faster than `(bf16 inline)`:
//!     restructure the SDPA to pre-cvt, no intrinsics needed
//!   - if neither (e.g. compute-bound by the FMA itself): document and
//!     stop, intrinsics won't help on this hardware
//!
//! Run with:
//!   `cargo run --release --bin bench_bf16_upconvert -p tahoma-int4-gemm`

// Inner SDPA loops are written `for i in 0..QK_HEAD_DIM` deliberately to
// mirror the byte-for-byte structure of the SDPA in `shell_int4.rs` /
// `layer0_int4.rs` — refactoring to `for (i, kf) in ...` would change
// what the compiler sees and would no longer be measuring the same
// thing. Same convention as the rest of `tahoma-int4-gemm` (see e.g.
// `shell.rs:193`, `shell_int4.rs:377`).
#![allow(clippy::needless_range_loop)]

use std::hint::black_box;
use std::time::Instant;

const QK_HEAD_DIM: usize = 192;
const V_HEAD_DIM: usize = 128;

/// Round f32 → bf16 bits (RTE).
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
}

fn build_caches(past_seq_len: usize) -> Caches {
    let past_k_f32 = deterministic_rand(past_seq_len * QK_HEAD_DIM, 0x1111_1111);
    let past_v_f32 = deterministic_rand(past_seq_len * V_HEAD_DIM, 0x2222_2222);
    // Round-trip via bf16 so the f32 path operates on byte-identical
    // values to the bf16 path. Without this, the f32 baseline reads
    // values with twice the mantissa width and could in theory get a
    // slightly different cache-line layout (it won't here — sizes are
    // disjoint — but it keeps the numerics directly comparable when
    // we check the dot-product output).
    let past_k_bf16: Vec<u16> = past_k_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    let past_v_bf16: Vec<u16> = past_v_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    let past_k_f32: Vec<f32> = past_k_bf16
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let past_v_f32: Vec<f32> = past_v_bf16
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();

    Caches {
        past_k_f32,
        past_v_f32,
        past_k_bf16,
        past_v_bf16,
    }
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

// ---------------------------------------------------------------------------
// f32 SDPA — pre-iter-032 reference. No upconvert in the inner loops.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// bf16 inline — iter 032 baseline. Upconvert lives inside the dot/accum.
// ---------------------------------------------------------------------------
fn sdpa_bf16_inline(c: &Caches, q: &[f32], past_seq_len: usize) -> f32 {
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

// ---------------------------------------------------------------------------
// bf16 split — pass 1 upconverts one row into a scratch f32 buffer, pass 2
// runs the f32 dot. The scratch buffer is per-row (sized to head_dim) so it
// stays in L1.
// ---------------------------------------------------------------------------
fn upconvert_row_bf16_to_f32(src: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    for i in 0..src.len() {
        dst[i] = f32::from_bits((src[i] as u32) << 16);
    }
}

fn sdpa_bf16_split(c: &Caches, q: &[f32], past_seq_len: usize) -> f32 {
    let scale = 1.0f32 / (QK_HEAD_DIM as f32).sqrt();
    let mut scores = vec![0.0f32; past_seq_len];
    let mut k_scratch = vec![0.0f32; QK_HEAD_DIM];
    for j in 0..past_seq_len {
        let k_row = &c.past_k_bf16[j * QK_HEAD_DIM..(j + 1) * QK_HEAD_DIM];
        upconvert_row_bf16_to_f32(k_row, &mut k_scratch);
        let mut s = 0.0f32;
        for i in 0..QK_HEAD_DIM {
            s += q[i] * k_scratch[i];
        }
        scores[j] = s * scale;
    }
    softmax_inplace(&mut scores);
    let mut out = vec![0.0f32; V_HEAD_DIM];
    let mut v_scratch = vec![0.0f32; V_HEAD_DIM];
    for j in 0..past_seq_len {
        let v_row = &c.past_v_bf16[j * V_HEAD_DIM..(j + 1) * V_HEAD_DIM];
        upconvert_row_bf16_to_f32(v_row, &mut v_scratch);
        let w = scores[j];
        for i in 0..V_HEAD_DIM {
            out[i] += w * v_scratch[i];
        }
    }
    out.iter().sum()
}

// ---------------------------------------------------------------------------
// bf16 upcvt only — just walk the bf16 buffers and upconvert into a scratch.
// No multiplies, no softmax. Isolates the dequant cost.
// ---------------------------------------------------------------------------
fn upcvt_only(c: &Caches, past_seq_len: usize) -> f32 {
    let mut k_scratch = vec![0.0f32; QK_HEAD_DIM];
    let mut v_scratch = vec![0.0f32; V_HEAD_DIM];
    let mut sink = 0.0f32;
    for j in 0..past_seq_len {
        let k_row = &c.past_k_bf16[j * QK_HEAD_DIM..(j + 1) * QK_HEAD_DIM];
        upconvert_row_bf16_to_f32(k_row, &mut k_scratch);
        sink += k_scratch[0]; // anti-DCE
    }
    for j in 0..past_seq_len {
        let v_row = &c.past_v_bf16[j * V_HEAD_DIM..(j + 1) * V_HEAD_DIM];
        upconvert_row_bf16_to_f32(v_row, &mut v_scratch);
        sink += v_scratch[0];
    }
    sink
}

// ---------------------------------------------------------------------------
// f32 dot only — same shape of math as f32 SDPA but with the softmax and
// the v-accumulate fused into one read pass so the optimizer doesn't
// disappear the work. Used to subtract from `bf16 inline` to isolate
// upconvert cost.
// ---------------------------------------------------------------------------
fn f32_dot_only(c: &Caches, q: &[f32], past_seq_len: usize) -> f32 {
    // Same arithmetic as sdpa_f32, just inlined and with a single anti-DCE
    // accumulator at the end.
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

fn bench<F: FnMut() -> f32>(name: &str, mut f: F, iters: usize, bytes_per_iter: usize) -> f64 {
    // Warm
    for _ in 0..10 {
        black_box(f());
    }
    let t0 = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..iters {
        acc += black_box(f());
    }
    let dt = t0.elapsed().as_secs_f64();
    let per_iter_us = dt / iters as f64 * 1.0e6;
    let gbps = bytes_per_iter as f64 * iters as f64 / dt / 1e9;
    // Touch acc so the optimizer can't elide the calls.
    let sink = if acc.is_nan() { 1 } else { 0 };
    println!("  {name:<18}  {per_iter_us:8.3} us/iter  {gbps:6.2} GB/s read   (sink={sink})");
    per_iter_us
}

fn run_at(past_seq_len: usize) {
    println!(
        "\n=== past_seq_len = {past_seq_len} (QK_HEAD_DIM={QK_HEAD_DIM}, V_HEAD_DIM={V_HEAD_DIM}) ==="
    );
    let caches = build_caches(past_seq_len);
    let q = deterministic_rand(QK_HEAD_DIM, 0x3333_3333);

    let k_bytes_f32 = past_seq_len * QK_HEAD_DIM * 4;
    let v_bytes_f32 = past_seq_len * V_HEAD_DIM * 4;
    let k_bytes_bf16 = past_seq_len * QK_HEAD_DIM * 2;
    let v_bytes_bf16 = past_seq_len * V_HEAD_DIM * 2;

    let iters = if past_seq_len <= 64 {
        200_000
    } else if past_seq_len <= 256 {
        50_000
    } else if past_seq_len <= 1024 {
        20_000
    } else {
        5_000
    };

    let f32_us = bench(
        "f32 SDPA",
        || sdpa_f32(&caches, &q, past_seq_len),
        iters,
        k_bytes_f32 + v_bytes_f32,
    );
    let inline_us = bench(
        "bf16 inline",
        || sdpa_bf16_inline(&caches, &q, past_seq_len),
        iters,
        k_bytes_bf16 + v_bytes_bf16,
    );
    let split_us = bench(
        "bf16 split",
        || sdpa_bf16_split(&caches, &q, past_seq_len),
        iters,
        k_bytes_bf16 + v_bytes_bf16,
    );
    let upcvt_us = bench(
        "bf16 upcvt only",
        || upcvt_only(&caches, past_seq_len),
        iters,
        k_bytes_bf16 + v_bytes_bf16,
    );
    let dot_us = bench(
        "f32 dot only",
        || f32_dot_only(&caches, &q, past_seq_len),
        iters,
        k_bytes_f32 + v_bytes_f32,
    );

    // Decomposition summary — what fraction of `bf16 inline` is the
    // upconvert (estimated two ways).
    let inline_minus_dot = (inline_us - dot_us).max(0.0);
    let frac_inline_minus_dot = inline_minus_dot / inline_us * 100.0;
    let frac_upcvt = upcvt_us / inline_us * 100.0;
    let split_vs_inline = (inline_us - split_us) / inline_us * 100.0;
    println!("  ---");
    println!(
        "  upconvert cost (inline - f32 dot only): {inline_minus_dot:7.3} us  ({frac_inline_minus_dot:5.1}% of bf16 inline)"
    );
    println!(
        "  upconvert cost (isolated upcvt only):   {upcvt_us:7.3} us  ({frac_upcvt:5.1}% of bf16 inline)"
    );
    if split_vs_inline >= 0.0 {
        println!(
            "  split-pass speedup vs inline:           {split_vs_inline:7.2}% (positive = split is faster)"
        );
    } else {
        println!(
            "  split-pass slowdown vs inline:          {:7.2}% (negative = split is slower)",
            split_vs_inline
        );
    }
    println!("  f32 (raw, no cvt):                      {f32_us:7.3} us");
}

fn main() {
    println!("=== K2.6 KV SDPA: bf16 upconvert cost decomposition ===");
    println!("Single-head, scalar Rust. Cache lives in L1/L2 at small past_seq_len.");
    println!("Goal: quantify the inline `f32::from_bits((u16 as u32) << 16)` upconvert cost");
    println!("in iter 032's SDPA loop. Decision rule in source-file doc comment.");

    run_at(16);
    run_at(64);
    run_at(256);
    run_at(1024);
    run_at(4096);
}
