//! T2 spike: Rust baseline for the glm5 MLA attention projection GEMMs.
//!
//! Times `linear_bf16_w` (the bf16 shell kernel the engine actually runs, rayon
//! row-parallel) over the five MLA projection shapes read from the target
//! model's manifest.json, at the spike's batch sizes. Prefill in this engine
//! appends KV per token in order, so the batch loop is sequential per row —
//! that IS the production shape, not a pessimisation.
//!
//! Companion of `tools/glm5_attn_ov_probe.py`; spec lives in the enterprise
//! repo (`docs/superpowers/specs/2026-08-09-glm5-attn-igpu-spike-spec.md`).
//!
//! Usage:
//!   cargo run --release -p cascadia-engine-sparse-moe \
//!     --example glm5_attn_bench -- <model_dir> [iters]

use std::time::Instant;

use cascadia_engine_sparse_moe::dsv4::math::linear_bf16_w;
use cascadia_engine_sparse_moe::glm::loader::read_manifest;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| {
        eprintln!("usage: glm5_attn_bench <model_dir> [iters]");
        std::process::exit(2);
    });
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);

    let m = read_manifest(std::path::Path::new(&dir)).expect("read manifest.json");
    let (h, ql, kvl) = (m.hidden_size, m.q_lora_rank, m.kv_lora_rank);
    let (nope, rope, vh, nh) = (
        m.qk_nope_head_dim,
        m.qk_rope_head_dim,
        m.v_head_dim,
        m.num_attention_heads,
    );
    let qk = nope + rope;
    // (name, out_dim, in_dim) — shapes per glm/attn.rs; keep in sync with the
    // python probe's PROJ table.
    let proj: [(&str, usize, usize); 5] = [
        ("wq_a", ql, h),
        ("wq_b", nh * qk, ql),
        ("wkv_a", kvl + rope, h),
        ("wkv_b", nh * (nope + vh), kvl),
        ("wo", h, nh * vh),
    ];
    println!("shapes: H={h} QL={ql} KVL={kvl} QK={qk} VH={vh} NH={nh}  iters={iters}");

    // Deterministic cheap PRNG; values are irrelevant to throughput.
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as u16
    };

    for batch in [1usize, 512, 1024, 2048] {
        println!("== batch {batch} ==");
        let mut total = 0.0f64;
        for (name, out_dim, in_dim) in proj {
            let w: Vec<u16> = (0..out_dim * in_dim).map(|_| next()).collect();
            let xs: Vec<f32> = (0..batch * in_dim)
                .map(|i| (i % 251) as f32 * 1e-3)
                .collect();
            let mut y = vec![0.0f32; out_dim];
            // warmup
            for row in xs.chunks(in_dim).take(2.min(batch)) {
                linear_bf16_w(row, &w, out_dim, in_dim, &mut y);
            }
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                for row in xs.chunks(in_dim) {
                    linear_bf16_w(row, &w, out_dim, in_dim, &mut y);
                }
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(|a, b| a.total_cmp(b));
            let med = times[times.len() / 2];
            let p95 = times[(times.len() as f64 * 0.95) as usize - 1];
            total += med;
            println!("  {name:6} [{out_dim:6}x{in_dim:6}]: median={med:9.3}ms p95={p95:9.3}ms");
        }
        println!("  five-projection total (median sum): {total:.3}ms");
    }
}
