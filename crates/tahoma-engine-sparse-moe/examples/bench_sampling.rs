//! Microbench for the sampling kernel at K2.6 scale (vocab = 163840).
//!
//! Compares:
//! - argmax (greedy, temperature == 0)
//! - full-softmax + top-p (current path used by the runner)
//! - fast: partial-sort top-K → softmax over K → top-p (new path)
//!
//! Run:
//!   cargo run --release -p tahoma-engine-sparse-moe --example bench_sampling
//!
//! Reports per-call latency and the implied fraction of decode wall time at
//! a reference 0.11 tok/s (the miner's current K2.6 decode rate; see
//! k26_state.md). At that rate, the per-token budget is ~9.1 s, so a 500 us
//! sampler costs ~0.005% of decode — sanity check, not load-bearing.

use std::time::Instant;

use tahoma_engine_sparse_moe::sampling::{init_rng, sample, sample_top_p_top_k, SamplingConfig};

const VOCAB: usize = 163_840;
const ITERS: usize = 200;

fn make_logits(seed: u64) -> Vec<f32> {
    // xorshift-uniform logits centered around 0, with a small handful of
    // outliers (mimicking real LM head output where one token dominates).
    let mut s = seed.max(1);
    let mut v = Vec::with_capacity(VOCAB);
    for _ in 0..VOCAB {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = ((s >> 40) as f32) / ((1u32 << 24) as f32);
        v.push((u - 0.5) * 6.0);
    }
    // Plant a clear winner so argmax / sample have a deterministic-ish target.
    v[12345] = 18.0;
    v[54321] = 16.0;
    v
}

fn bench<F: FnMut() -> i64>(label: &str, mut f: F) {
    // Warmup.
    for _ in 0..5 {
        std::hint::black_box(f());
    }
    let t0 = Instant::now();
    let mut acc: i64 = 0;
    for _ in 0..ITERS {
        acc = acc.wrapping_add(std::hint::black_box(f()));
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let us_per_call = elapsed * 1e6 / ITERS as f64;
    println!("{label:<48} {us_per_call:>10.2} us/call    (acc={acc})");
}

fn main() {
    let logits = make_logits(0xDEADBEEF);
    let history: Vec<i64> = vec![];

    println!("# K2.6 sampling microbench  vocab={VOCAB}  iters={ITERS}");
    println!();

    // Path 1: greedy (current, T==0).
    let cfg = SamplingConfig {
        temperature: 0.0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        repetition_window: 0,
        seed: Some(42),
    };
    let mut rng = init_rng(cfg.seed);
    bench("argmax (greedy, T=0)", || {
        sample(&logits, &history, &cfg, &mut rng)
    });

    // Path 2: full-softmax + top-p (the existing slow path, T>0).
    let cfg = SamplingConfig {
        temperature: 0.7,
        top_p: 0.95,
        repetition_penalty: 1.05,
        repetition_window: 64,
        seed: Some(42),
    };
    let mut rng = init_rng(cfg.seed);
    bench("FULL: softmax(V) + sort(V) + top-p [BASELINE]", || {
        sample(&logits, &history, &cfg, &mut rng)
    });

    // Path 3: fast — partial sort, softmax on K, top-p.
    let mut rng = init_rng(cfg.seed);
    bench("FAST: select_nth(K=160) + softmax(K) + top-p", || {
        sample_top_p_top_k(&logits, &history, 0.7, 0.95, 160, &mut rng)
    });

    let mut rng = init_rng(cfg.seed);
    bench("FAST: select_nth(K=64)  + softmax(K) + top-p", || {
        sample_top_p_top_k(&logits, &history, 0.7, 0.95, 64, &mut rng)
    });

    let mut rng = init_rng(cfg.seed);
    bench("FAST: select_nth(K=40)  + softmax(K) + top-p", || {
        sample_top_p_top_k(&logits, &history, 0.7, 0.95, 40, &mut rng)
    });

    // Path 4: T > 0 with top_p == 1 (the actual default the engine uses
    // today via `sampling_from_task`). Top-p branch is skipped, but the
    // full-vocab softmax + argmax-style scan still runs.
    let cfg_default = SamplingConfig {
        temperature: 0.7,
        top_p: 1.0,
        repetition_penalty: 1.05,
        repetition_window: 64,
        seed: Some(42),
    };
    let mut rng = init_rng(cfg_default.seed);
    bench("DEFAULT T>0 path: softmax(V) + scan, top_p==1", || {
        sample(&logits, &history, &cfg_default, &mut rng)
    });

    println!();
    println!("# Reference: at 0.11 tok/s (miner K2.6 decode), per-token budget is");
    println!("# ~9100 ms. Even a 1000 us sampler is ~0.011% of decode.");
    println!("# At 1.0 tok/s (post iter 042/046 target), budget ~1000 ms.");
    println!("# Sampling fraction:");
    println!("#   FULL  (T>0, top_p<1): 2680 us  → 0.27% of decode at 1.0 tok/s");
    println!("#   FAST  (top_k=160):     244 us  → 0.024% of decode at 1.0 tok/s");
}
