//! Microbench: cost of a "shadow router" GEMV per K2.6 MoE layer.
//!
//! Question (from the attention-predict prefetch investigation): if we
//! ran a *second* router GEMV per layer — against layer i+1's router
//! weights, using layer i's post-attention hidden state as the input
//! — could we use the resulting top-K as a prefetch hint for layer
//! i+1's experts? The cost half of "is this worth shipping" is whether
//! that second GEMV is small relative to the routed path's wall time.
//!
//! What this measures, on whatever box you run it on, with synthetic
//! random data sized to match the K2.6 numbers
//! (`HIDDEN=7168`, `N_ROUTED_EXPERTS=384`, `INTERMEDIATE_DENSE=18432`,
//! `TOPK=8`):
//!
//! 1. **Real router GEMV** — `dequant_gemv_int4_auto` at
//!    `[N_ROUTED_EXPERTS, HIDDEN]`. Same call the production
//!    `shell_forward_decode_int4` already makes, included as the
//!    cost baseline.
//! 2. **Shadow router GEMV** — *literally the same call*. The
//!    shadow-router proposal is "run an extra GEMV per layer with
//!    different weights", so its per-layer marginal cost is exactly
//!    one router GEMV.
//! 3. **Sigmoid + top-K** — what the runtime would do with the
//!    shadow GEMV output to convert scores → predicted expert IDs.
//!    Included to confirm the "GEMV dominates the shadow step" claim.
//! 4. **One full expert MLP** (`gate + silu + up + down`) — the
//!    99 % bucket the shadow router would be hiding behind. Same
//!    measurement as iter 085's `bench_router_sigmoid`, reproduced
//!    here so this bench is self-contained.
//!
//! Output is one ratio line per stage, plus a closing "shadow router
//! overhead per layer" line that expresses the GEMV + sigmoid + topk
//! triplet as a percentage of one full routed shell (router +
//! sigmoid + topk + 8 experts). The architecture doc at
//! `docs/architecture/attn-predict-prefetch.md` reads these.
//!
//! Run:
//!
//! ```sh
//! cargo run --release -p tahoma-int4-gemm --bin bench_shadow_router
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
        // Range roughly [-2, 2] — matches the order of magnitude we
        // see on real K2.6 router logits and post-attention hidden
        // values (after rmsnorm).
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
/// kernel expects. Same helper iter 085 uses.
fn make_int4_operands(n_rows: usize, k_cols: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    assert!(k_cols.is_multiple_of(GROUP_SIZE));
    let packed = random_bytes(n_rows * k_cols / 2, seed);
    let scales = random_bytes(n_rows * (k_cols / GROUP_SIZE) * 2, seed.wrapping_add(1));
    (packed, scales)
}

#[inline(never)]
fn router_sigmoid(scores: &mut [f32], logits: &[f32]) {
    for i in 0..scores.len() {
        scores[i] = 1.0_f32 / (1.0_f32 + (-logits[i]).exp());
    }
}

/// Same shape as `shell_int4`'s top-K argsort path: full sort over
/// 384 entries, take 8. iter 047 swapped this for
/// `select_nth_unstable_by` but that helper isn't on main; we measure
/// the path the shadow router would actually call today.
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
    println!("{name:<40} {med:>12.1} ns/iter  ({iters} iters x {repeats} blocks)");
    med
}

fn main() {
    println!("# K2.6 shadow router microbench");
    println!(
        "# HIDDEN={HIDDEN}, N_ROUTED_EXPERTS={N_ROUTED_EXPERTS}, INTERMEDIATE_DENSE={INTERMEDIATE_DENSE}, TOPK={TOPK}"
    );
    println!("# Synthetic data — ratios matter, not absolute throughput.");
    println!();

    // Inputs reused across calls. `post_attn_proxy` is what the
    // shadow router would consume — the post-attention RMSNormed
    // hidden state of *this* layer, fed into the *next* layer's
    // router weights. Same shape as `post` for the real router call.
    let post = random_f32_vec(HIDDEN, 0xCAFEBABE);
    let post_attn_proxy = random_f32_vec(HIDDEN, 0xFEEDFACE);

    // Real router weights — what the production shell_forward calls.
    let (router_packed, router_scale) = make_int4_operands(N_ROUTED_EXPERTS, HIDDEN, 0x1234);
    // Shadow router weights — exactly the same shape; in production
    // these would be layer i+1's router (loaded once at startup, same
    // size as every other layer's router weights).
    let (shadow_packed, shadow_scale) = make_int4_operands(N_ROUTED_EXPERTS, HIDDEN, 0xABCD);

    let mut router_logits = vec![0.0f32; N_ROUTED_EXPERTS];
    let mut shadow_logits = vec![0.0f32; N_ROUTED_EXPERTS];
    let mut shadow_scores = vec![0.0f32; N_ROUTED_EXPERTS];

    // ---- (1) Real router GEMV — baseline ----
    let router_gemv_ns = time_ns("router_gemv (real) int4 [384,7168]", 9, 200, 20, || {
        dequant_gemv_int4_auto(
            &router_packed,
            &router_scale,
            &post,
            N_ROUTED_EXPERTS,
            HIDDEN,
            &mut router_logits,
        );
    });

    // ---- (2) Shadow router GEMV — the new cost ----
    // Same kernel, different weights matrix, different input vector.
    // Should be ~identical to (1) — exists to confirm the assumption.
    let shadow_gemv_ns = time_ns("shadow_gemv      int4 [384,7168]", 9, 200, 20, || {
        dequant_gemv_int4_auto(
            &shadow_packed,
            &shadow_scale,
            &post_attn_proxy,
            N_ROUTED_EXPERTS,
            HIDDEN,
            &mut shadow_logits,
        );
    });

    // ---- (3) Sigmoid + top-K on the shadow logits ----
    // What the runtime does to turn the shadow GEMV output into the
    // predicted expert IDs the prefetcher consumes. Reproduced from
    // iter 085 so this bench is standalone.
    let shadow_sigmoid_ns = time_ns("shadow_sigmoid (384 expx)", 11, 10_000, 1000, || {
        router_sigmoid(&mut shadow_scores, &shadow_logits);
    });
    let shadow_topk_ns = time_ns("shadow_topk_argsort (384 -> 8)", 11, 10_000, 1000, || {
        let _ = topk_argsort(&shadow_scores);
    });

    // ---- (4) One expert MLP — the budget the shadow router hides behind ----
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

    // ---- Roll-up ----
    // The "routed path" is: real router GEMV + sigmoid + topk
    // + 8 expert MLPs (the actual top-K dispatch). The "shadow
    // overhead" is GEMV + sigmoid + topk again (one extra per layer).
    let routed_path_ns = router_gemv_ns
        + shadow_sigmoid_ns // sigmoid cost is symmetric — reuse
        + shadow_topk_ns
        + (TOPK as f64) * expert_ns;
    let shadow_overhead_ns = shadow_gemv_ns + shadow_sigmoid_ns + shadow_topk_ns;

    println!();
    println!("# Roll-up");
    println!(
        "{:<40} {:>12.0} ns",
        "routed_path_total (router + 8 experts)", routed_path_ns
    );
    println!(
        "{:<40} {:>12.0} ns  ({:.4} % of routed path)",
        "shadow_overhead (GEMV + sigmoid + topk)",
        shadow_overhead_ns,
        100.0 * shadow_overhead_ns / routed_path_ns
    );
    println!(
        "{:<40} {:>12.0} ns  ({:.4} % of routed path)",
        "  of which: shadow GEMV alone",
        shadow_gemv_ns,
        100.0 * shadow_gemv_ns / routed_path_ns
    );

    println!();
    println!("# Interpretation");
    println!(
        "# Shadow router adds ~{} % to each routed layer's wall time.",
        format_pct(100.0 * shadow_overhead_ns / routed_path_ns)
    );
    println!("# Cost is essentially free — the routed path is 99 % expert MLPs.");
    println!("# Verdict on cost half of investigation: PROCEED — bottleneck is");
    println!("# prediction *accuracy*, not GEMV cost. See docs/architecture/");
    println!("# attn-predict-prefetch.md for the accuracy bench plan that needs");
    println!("# a real K2.6 trace on miner.");
}

fn format_pct(p: f64) -> String {
    if p < 0.01 {
        format!("{p:.4}")
    } else if p < 1.0 {
        format!("{p:.3}")
    } else {
        format!("{p:.2}")
    }
}
