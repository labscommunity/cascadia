//! Microbench: K2.6 router sigmoid vs everything else in one shell.
//!
//! Question (from the sparse-softmax investigation): is the
//! per-element sigmoid over the 384 routed-expert logits a meaningful
//! slice of decode time? If yes, a pre-sigmoid score threshold to
//! skip the `exp()` on low-magnitude logits is worth shipping.
//!
//! What this measures, on whatever box you run it on, with synthetic
//! random data sized to match the K2.6 numbers
//! (`HIDDEN=7168`, `N_ROUTED_EXPERTS=384`, `INTERMEDIATE_DENSE=18432`,
//! `TOPK=8`):
//!
//! 1. **Router sigmoid alone** — the literal
//!    `1 / (1 + exp(-x))` loop over `N_ROUTED_EXPERTS` floats.
//! 2. **Router GEMV** — `dequant_gemv_int4_auto` at
//!    `[N_ROUTED_EXPERTS, HIDDEN]`. Same kernel the production shell
//!    forward calls. Bandwidth-bound (~2.75 M int4 reads per call).
//! 3. **One expert MLP** — three int4 GEMVs (`gate`, `up`, `down`)
//!    at `HIDDEN×INTERMEDIATE_DENSE`. The dominant per-layer cost;
//!    8 of these run on the routed path each decode step.
//! 4. **Top-K selection** — argsort-by-score, take 8. Today's
//!    implementation in `shell_int4.rs::shell_forward_decode_int4`
//!    sorts the full 384, which is also a "skip me if I'm cheap"
//!    candidate.
//!
//! Output is six lines: ns/iter for each piece + the implied % of
//! one shell + one full layer (shell + 8 experts). The investigation
//! at `docs/architecture/sparse-softmax-router.md` reads these.
//!
//! Run:
//!
//! ```sh
//! cargo run --release -p tahoma-int4-gemm --bin bench_router_sigmoid
//! ```
//!
//! Synthetic weights, not byte-equivalent to a real shard. The point
//! is **ratios**, not absolute throughput.

use std::time::Instant;

use tahoma_int4_gemm::kernel_avx512::dequant_gemv_int4_auto;

const HIDDEN: usize = 7168;
const N_ROUTED_EXPERTS: usize = 384;
const INTERMEDIATE_DENSE: usize = 18432;
const TOPK: usize = 8;
const GROUP_SIZE: usize = 32;

/// Cheap deterministic PRNG so the bench is reproducible.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn random_f32_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        // Range roughly [-2, 2] — matches the order of magnitude we see
        // on real K2.6 router logits.
        let bits = lcg(&mut s);
        let u = (bits as f64) / (u32::MAX as f64); // [0, 1]
        v.push((u as f32) * 4.0 - 2.0);
    }
    v
}

fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    while v.len() + 4 <= n {
        v.extend_from_slice(&lcg(&mut s).to_le_bytes());
    }
    while v.len() < n {
        v.push((lcg(&mut s) & 0xFF) as u8);
    }
    v
}

/// Build a synthetic int4 GEMV operand set: packed nibbles + bf16
/// scales. Values are random nonsense but sized exactly to what the
/// kernel expects.
fn make_int4_operands(n_rows: usize, k_cols: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    assert!(k_cols.is_multiple_of(GROUP_SIZE));
    let packed = random_bytes(n_rows * k_cols / 2, seed);
    // Scales are bf16, stored little-endian as 2 bytes per group. Random
    // bf16 patterns produce a wide range of nonsense outputs — that's
    // fine, the kernel does the same number of FMAs either way.
    let scales = random_bytes(n_rows * (k_cols / GROUP_SIZE) * 2, seed.wrapping_add(1));
    (packed, scales)
}

#[inline(never)]
fn router_sigmoid(scores: &mut [f32], logits: &[f32]) {
    for i in 0..scores.len() {
        scores[i] = 1.0_f32 / (1.0_f32 + (-logits[i]).exp());
    }
}

#[inline(never)]
fn topk_argsort(scores: &[f32]) -> [usize; TOPK] {
    let mut idx_score: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    idx_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = [0usize; TOPK];
    for k in 0..TOPK {
        out[k] = idx_score[k].0;
    }
    out
}

/// Run `f` `iters` times after `warm` warmup iters, return median ns
/// per iter (median over `repeats` blocks).
fn time_ns<F: FnMut()>(name: &str, repeats: usize, iters: usize, warm: usize, mut f: F) -> f64 {
    for _ in 0..warm {
        f();
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        let dt = t0.elapsed();
        samples.push(dt.as_nanos() as f64 / iters as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = samples[samples.len() / 2];
    println!("{name:<32} {med:>10.1} ns/iter  ({iters} iters x {repeats} blocks)");
    med
}

fn main() {
    println!("# K2.6 router sigmoid microbench");
    println!(
        "# HIDDEN={HIDDEN}, N_ROUTED_EXPERTS={N_ROUTED_EXPERTS}, INTERMEDIATE_DENSE={INTERMEDIATE_DENSE}, TOPK={TOPK}"
    );
    println!("# Synthetic data — ratios matter, not absolute throughput.");
    println!();

    // Inputs reused across calls.
    let post = random_f32_vec(HIDDEN, 0xCAFEBABE);
    let logits = random_f32_vec(N_ROUTED_EXPERTS, 0xDEADBEEF);
    let mut scores = vec![0.0f32; N_ROUTED_EXPERTS];

    // ---- (1) Router sigmoid loop ----
    let sigmoid_ns = time_ns("router_sigmoid (384 expx)", 11, 10_000, 1000, || {
        router_sigmoid(&mut scores, &logits);
    });

    // ---- (2) Router GEMV [384, 7168] ----
    let (router_packed, router_scale) = make_int4_operands(N_ROUTED_EXPERTS, HIDDEN, 0x1234);
    let mut router_logits = vec![0.0f32; N_ROUTED_EXPERTS];
    let router_gemv_ns = time_ns("router_gemv int4 [384,7168]", 9, 200, 20, || {
        dequant_gemv_int4_auto(
            &router_packed,
            &router_scale,
            &post,
            N_ROUTED_EXPERTS,
            HIDDEN,
            &mut router_logits,
        );
    });

    // ---- (3) One expert MLP: gate + up + down ----
    let (gate_packed, gate_scale) = make_int4_operands(INTERMEDIATE_DENSE, HIDDEN, 0x5678);
    let (up_packed, up_scale) = make_int4_operands(INTERMEDIATE_DENSE, HIDDEN, 0x9ABC);
    let (down_packed, down_scale) = make_int4_operands(HIDDEN, INTERMEDIATE_DENSE, 0xDEF0);

    let mut gate_out = vec![0.0f32; INTERMEDIATE_DENSE];
    let mut up_out = vec![0.0f32; INTERMEDIATE_DENSE];
    let mut inter = vec![0.0f32; INTERMEDIATE_DENSE];
    let mut down_out = vec![0.0f32; HIDDEN];

    let expert_ns = time_ns("one_expert (gate+silu+up+down)", 7, 20, 5, || {
        dequant_gemv_int4_auto(
            &gate_packed,
            &gate_scale,
            &post,
            INTERMEDIATE_DENSE,
            HIDDEN,
            &mut gate_out,
        );
        dequant_gemv_int4_auto(
            &up_packed,
            &up_scale,
            &post,
            INTERMEDIATE_DENSE,
            HIDDEN,
            &mut up_out,
        );
        for i in 0..INTERMEDIATE_DENSE {
            let g = gate_out[i];
            let silu = g / (1.0_f32 + (-g).exp());
            inter[i] = silu * up_out[i];
        }
        dequant_gemv_int4_auto(
            &down_packed,
            &down_scale,
            &inter,
            HIDDEN,
            INTERMEDIATE_DENSE,
            &mut down_out,
        );
    });

    // ---- (4) Top-K argsort over 384 scores ----
    let topk_ns = time_ns("topk_argsort (384 -> 8)", 11, 1000, 100, || {
        let _ = std::hint::black_box(topk_argsort(&scores));
    });

    println!();
    println!("# Per-layer totals (decode, MoE layer)");

    // One "shell" = everything except the experts. Numbers below for the
    // bits this bench actually measured; the rest (attn projections,
    // SDPA, RMSNorms, shared expert) is *not* measured here. The
    // per-layer routed cost is what we compare router-sigmoid against.
    let routed_experts_ns = expert_ns * (TOPK as f64);
    println!("  router_gemv:              {router_gemv_ns:>10.1} ns");
    println!("  router_sigmoid:           {sigmoid_ns:>10.1} ns");
    println!("  topk_argsort:             {topk_ns:>10.1} ns");
    println!("  routed_experts (8 x):     {routed_experts_ns:>10.1} ns");

    let pct_sigmoid_of_router = 100.0 * sigmoid_ns / (sigmoid_ns + router_gemv_ns + topk_ns);
    let pct_sigmoid_of_routed_path =
        100.0 * sigmoid_ns / (sigmoid_ns + router_gemv_ns + topk_ns + routed_experts_ns);
    println!();
    println!("# Headlines");
    println!("  sigmoid / (router gemv + sigmoid + topk) = {pct_sigmoid_of_router:.2}%");
    println!("  sigmoid / (above + 8 expert MLPs)        = {pct_sigmoid_of_routed_path:.4}%");
    println!(
        "  topk_argsort / router stage              = {:.2}%",
        100.0 * topk_ns / (sigmoid_ns + router_gemv_ns + topk_ns)
    );
}
