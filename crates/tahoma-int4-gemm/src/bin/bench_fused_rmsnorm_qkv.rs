//! Cost analysis for the proposed fused RMSNorm + projection kernel
//! ([perf/fused-rmsnorm-qkv-053]).
//!
//! Hypothesis under test: in the K2.6 shell, fusing the RMSNorm into the
//! q_a_proj + kv_a_proj kernels (so the normed hidden state never
//! materialises in memory) is a meaningful decode-time win.
//!
//! Counter-hypothesis: the normed buffer is 28 KB (fits in L1d), the int4
//! GEMV weight traffic per projection is 2–6 MB (dominates DRAM time by
//! 100x). The save should be <2% of the unfused projection time and
//! invisible at the engine level. This bench produces the data to decide.
//!
//! We model TWO fusion sites that the K2.6 shell has, since the spec was
//! written against a "Q/K/V projection" pattern that K2.6's MLA path does
//! not literally have:
//!
//!   Site A: `input_norm` feeds `q_a_proj` and `kv_a_proj`
//!           (normed read 2× — NOT 3× like standard QKV)
//!   Site B: `post_norm`  feeds `router`, `shared_gate`, `shared_up`
//!           (normed read 3×, but the matmuls are smaller-aggregate)
//!
//! Use the AVX-512 / scalar auto-dispatched kernel so the measurement
//! tracks what production decode actually pays. On the M4 dev box the
//! scalar fallback runs; on Linux production (Xeon Gold 6252) the
//! AVX-512 path runs. Either way, the *ratio* of (norm time + intermediate
//! write/read) to (projection time) is what tells us whether fusion is
//! worth doing.
//!
//! Output: per-site numbers and a verdict. A "verdict" of <2% saved means
//! fusion is not worth the implementation cost (lose the modularity of
//! the standalone `rmsnorm_apply` + `dequant_gemv_int4_auto` building
//! blocks for a sub-percent decode speedup).

use std::time::Instant;

use tahoma_int4_gemm::kernel_avx512::dequant_gemv_int4_auto;
use tahoma_int4_gemm::shell::{
    rmsnorm_apply_pub, HIDDEN, INTERMEDIATE_SHARED, KV_LORA_RANK, N_ROUTED_EXPERTS,
    QK_ROPE_HEAD_DIM, Q_LORA_RANK,
};

const GROUP_SIZE: usize = 32;

/// Local re-implementation of the (pub(crate)) `quantize_int4_group` from
/// `shell_int4.rs`. Symmetric int4 with group_size=32 and bf16 scales.
/// Lifted here to keep the bench bin from forcing a public API change.
fn quantize_int4_group(weight_bf16: &[u8], n_rows: usize, k_cols: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(weight_bf16.len(), n_rows * k_cols * 2);
    assert!(k_cols.is_multiple_of(GROUP_SIZE));
    let n_groups = k_cols / GROUP_SIZE;
    let mut packed = vec![0u8; n_rows * k_cols / 2];
    let mut scales = vec![0u8; n_rows * n_groups * 2];
    for r in 0..n_rows {
        for g in 0..n_groups {
            let mut max_abs = 0.0f32;
            for k in 0..GROUP_SIZE {
                let c = g * GROUP_SIZE + k;
                let off = (r * k_cols + c) * 2;
                let bits = ((weight_bf16[off + 1] as u32) << 8) | (weight_bf16[off] as u32);
                let w = f32::from_bits(bits << 16);
                if w.abs() > max_abs {
                    max_abs = w.abs();
                }
            }
            let scale = if max_abs == 0.0 {
                1.0e-10
            } else {
                max_abs / 7.0
            };
            let scale_bits = {
                let b = scale.to_bits();
                let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
                (r >> 16) as u16
            };
            let s_off = (r * n_groups + g) * 2;
            scales[s_off] = (scale_bits & 0xFF) as u8;
            scales[s_off + 1] = (scale_bits >> 8) as u8;
            let scale_q = f32::from_bits((scale_bits as u32) << 16);
            let inv = 1.0 / scale_q;
            for k in 0..GROUP_SIZE {
                let c = g * GROUP_SIZE + k;
                let w_off = (r * k_cols + c) * 2;
                let bits = ((weight_bf16[w_off + 1] as u32) << 8) | (weight_bf16[w_off] as u32);
                let w = f32::from_bits(bits << 16);
                let q = (w * inv).round().clamp(-8.0, 7.0) as i32;
                let nibble = ((q + 8) & 0x0F) as u8;
                let p_off = (r * k_cols + c) / 2;
                if c.is_multiple_of(2) {
                    packed[p_off] = (packed[p_off] & 0xF0) | nibble;
                } else {
                    packed[p_off] = (packed[p_off] & 0x0F) | (nibble << 4);
                }
            }
        }
    }
    (packed, scales)
}

/// Make a deterministic bf16 "norm weight" buffer for the given dim.
/// All weights are 1.0 so RMSNorm degenerates to pure normalisation;
/// the timing is unaffected by the weight value.
fn make_norm_weight(dim: usize) -> Vec<u8> {
    let mut w = vec![0u8; dim * 2];
    // bf16(1.0) = 0x3f80 LE
    for i in 0..dim {
        w[i * 2] = 0x80;
        w[i * 2 + 1] = 0x3f;
    }
    w
}

/// Make a deterministic bf16 weight matrix [n_rows, k_cols], quantise
/// to int4 + bf16 scales. Returns (packed, scales). Uses a small smooth
/// pattern so the quantiser has something to compress.
fn make_int4_weight(n_rows: usize, k_cols: usize) -> (Vec<u8>, Vec<u8>) {
    let mut w = vec![0u8; n_rows * k_cols * 2];
    for r in 0..n_rows {
        for c in 0..k_cols {
            // Smooth small-ish value: sin-ish via cheap arithmetic.
            let raw = (((r * 7919 + c * 31 + 17) & 0xFF) as f32) / 256.0 - 0.5;
            // bf16: top 16 bits of f32.
            let bits = raw.to_bits();
            let bf = (bits >> 16) as u16;
            w[(r * k_cols + c) * 2] = (bf & 0xFF) as u8;
            w[(r * k_cols + c) * 2 + 1] = (bf >> 8) as u8;
        }
    }
    quantize_int4_group(&w, n_rows, k_cols)
}

fn make_input(dim: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; dim];
    for (i, v) in x.iter_mut().enumerate() {
        // Bound the magnitude so RMS scale stays well-defined.
        *v = ((i as f32) * 0.001).sin() * 0.5 + 0.1;
    }
    x
}

struct SiteBench {
    name: &'static str,
    /// Norm input dim (= HIDDEN for both K2.6 sites).
    norm_dim: usize,
    /// Output dims of each projection that consumes the normed buffer.
    proj_out_dims: Vec<usize>,
}

fn bench_site(site: &SiteBench, iters: usize) {
    println!("\n=== {} ===", site.name);
    println!(
        "  norm_dim={}, projections={:?}",
        site.norm_dim, site.proj_out_dims
    );

    let x = make_input(site.norm_dim);
    let norm_weight = make_norm_weight(site.norm_dim);

    // Build all projection weights up-front so allocation / setup is out
    // of the timing.
    let proj_weights: Vec<(Vec<u8>, Vec<u8>)> = site
        .proj_out_dims
        .iter()
        .map(|&n_out| make_int4_weight(n_out, site.norm_dim))
        .collect();
    let mut outs: Vec<Vec<f32>> = site
        .proj_out_dims
        .iter()
        .map(|&n_out| vec![0.0f32; n_out])
        .collect();

    // ---- A) Just the RMSNorm (norm_dim writes + reads, single pass) ----
    // Warm
    let _ = rmsnorm_apply_pub(&x, &norm_weight, site.norm_dim);
    let t0 = Instant::now();
    let mut norm_out = Vec::new();
    for _ in 0..iters {
        norm_out = rmsnorm_apply_pub(&x, &norm_weight, site.norm_dim);
    }
    let norm_dt = t0.elapsed().as_secs_f64();
    let norm_us = norm_dt / iters as f64 * 1e6;
    println!("  rmsnorm alone:           {norm_us:8.2} us/iter");

    // ---- B) Just the projections, given a pre-computed normed input ----
    // Warm
    for ((packed, scales), out) in proj_weights.iter().zip(outs.iter_mut()) {
        dequant_gemv_int4_auto(packed, scales, &norm_out, out.len(), site.norm_dim, out);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        for ((packed, scales), out) in proj_weights.iter().zip(outs.iter_mut()) {
            dequant_gemv_int4_auto(packed, scales, &norm_out, out.len(), site.norm_dim, out);
        }
    }
    let proj_dt = t0.elapsed().as_secs_f64();
    let proj_us = proj_dt / iters as f64 * 1e6;
    let n_projs = site.proj_out_dims.len();
    println!("  projections alone ({n_projs}):   {proj_us:8.2} us/iter");

    // ---- C) Unfused total = rmsnorm + projections (current code path) ----
    let t0 = Instant::now();
    for _ in 0..iters {
        let n = rmsnorm_apply_pub(&x, &norm_weight, site.norm_dim);
        for ((packed, scales), out) in proj_weights.iter().zip(outs.iter_mut()) {
            dequant_gemv_int4_auto(packed, scales, &n, out.len(), site.norm_dim, out);
        }
    }
    let unfused_dt = t0.elapsed().as_secs_f64();
    let unfused_us = unfused_dt / iters as f64 * 1e6;
    println!("  unfused (current path):  {unfused_us:8.2} us/iter");

    // ---- Verdict: best-case fusion savings ----
    // Best case: fusion eliminates the rmsnorm write + the n_projs reads of
    // norm_out, replacing them with on-the-fly normalisation inside each
    // GEMV inner loop. The GEMV already pays for the input read (the
    // un-normed `x` is read once per row, just like the normed buffer is in
    // the unfused path), and the in-register multiply by inv_rms * weight
    // is essentially free. So the *upper bound* on the saving is the
    // standalone rmsnorm time minus an "input read" cost that the GEMV
    // would now pay (which is small — input is 28 KB f32, hot in L1).
    let saved_upper_us = norm_us;
    let percent_saved = 100.0 * saved_upper_us / unfused_us;
    println!(
        "  fusion upper-bound save: {saved_upper_us:8.2} us ({percent_saved:5.2}% of unfused)"
    );

    if percent_saved < 2.0 {
        println!("  VERDICT: NOT WORTH FUSING (< 2% theoretical max)");
    } else if percent_saved < 5.0 {
        println!("  VERDICT: marginal — implement only if other wins are exhausted");
    } else {
        println!("  VERDICT: candidate — worth a working PoC");
    }
}

fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    println!("=== Fused RMSNorm + projection cost analysis (iter 053) ===");
    println!("iters per measurement: {iters}");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("kernel: AVX-512");
        } else {
            println!("kernel: scalar (no AVX-512)");
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("kernel: scalar (non-x86_64 platform — production target is x86_64 AVX-512)");
    }
    println!("rayon threads: {}", rayon::current_num_threads());

    let sites = [
        // Site A: input_norm feeds q_a_proj + kv_a_proj. Normed buffer read 2x.
        SiteBench {
            name: "Site A: input_norm + (q_a_proj, kv_a_proj) [MLA Q+KV down-proj]",
            norm_dim: HIDDEN,
            proj_out_dims: vec![Q_LORA_RANK, KV_LORA_RANK + QK_ROPE_HEAD_DIM],
        },
        // Site B: post_norm feeds router + shared_gate + shared_up. Normed buffer
        // read 3x — the closest analog to "Q/K/V" in the spec.
        SiteBench {
            name: "Site B: post_norm + (router, shared_gate, shared_up)",
            norm_dim: HIDDEN,
            proj_out_dims: vec![N_ROUTED_EXPERTS, INTERMEDIATE_SHARED, INTERMEDIATE_SHARED],
        },
    ];
    for s in &sites {
        bench_site(s, iters);
    }

    println!("\n=== Per-shell sum (one shell forward, decode seq=1) ===");
    // Sum the unfused costs to put the fusion saving in shell-level context.
    println!(
        "  K2.6 has 60 shells -> per-token cost scales 60x. A saving of N us per\n  shell is N*60 us per token. Useful baseline: shell decode is currently\n  ~9 ms (~0.11 tok/s)."
    );
}
