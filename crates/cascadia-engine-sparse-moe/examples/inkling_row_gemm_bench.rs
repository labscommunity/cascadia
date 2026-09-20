//! Row-batched int4 expert kernel bench: one expert serving `n` rows, the old
//! per-row loop (`swiglu_from` × n — the expert's 31.85 MB crosses the memory
//! bus n times) against the multi-input kernel (`swiglu_rows_from` — once).
//!
//! Real dims (hidden 6144, expert intermediate 3072, int4 group 32). Every
//! call takes the NEXT of `--experts` distinct random bins (64 = 2 GB), so the
//! weights stream from DRAM like the real 46 GB working set instead of sitting
//! in L3. `--par P` runs P distinct experts concurrently per call (the engine
//! visits a block's experts in rayon cohorts of 8, each kernel row-parallel
//! inside), reporting the time for the whole cohort.
//!
//! ```text
//! RAYON_NUM_THREADS=16 cargo run --release -p cascadia-engine-sparse-moe \
//!     --example inkling_row_gemm_bench -- --isa avx2 --gemv-rows 4
//! ```
//!
//! `--isa native|avx2|avx512|scalar` forces the kernel (the fleet is AVX2; an
//! AVX-512 box picks AVX-512 natively), `--gemv-rows 1|2|4` is the per-row
//! baseline's `CASCADIA_INT4_GEMV_ROWS` tiling (AVX2 only; the fleet runs 4),
//! `--gemm-rows 1|2|4|8` the multi-input kernel's tile
//! (`CASCADIA_INT4_GEMM_ROWS`). Before timing, each `n` asserts the two paths
//! agree bit for bit.

use std::time::Instant;

use cascadia_engine_sparse_moe::dsv4::expert_mmap::{Int4Isa, MmapExpert, GEMM_MAX_INPUTS};
use rayon::prelude::*;

const DIM: usize = 6144;
const INTER: usize = 3072;
const G: usize = 32;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn expert_bin(rng: &mut Rng) -> Vec<u8> {
    let mut bin = Vec::new();
    for (out_dim, in_dim) in [(INTER, DIM), (INTER, DIM), (DIM, INTER)] {
        let nib = out_dim * in_dim / 2;
        let start = bin.len();
        bin.resize(start + nib, 0);
        for chunk in bin[start..].chunks_mut(8) {
            let v = rng.next().to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
        for _ in 0..out_dim * in_dim / G {
            let s = (rng.next() % 255) as f32 / 8192.0 - 127.0 / 8192.0;
            bin.extend(half::bf16::from_f32(s).to_le_bytes());
        }
    }
    bin
}

fn arg(name: &str) -> Option<String> {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == name {
            return it.next();
        }
    }
    None
}

fn main() {
    let isa = match arg("--isa").as_deref().unwrap_or("native") {
        "native" => Int4Isa::active(),
        "avx2" => Int4Isa::Avx2,
        "avx512" => Int4Isa::Avx512,
        "scalar" => Int4Isa::Scalar,
        other => panic!("--isa {other}: native|avx2|avx512|scalar"),
    };
    assert!(isa.supported(), "{isa:?} is not supported on this CPU");
    let num = |name: &str, default: usize| {
        arg(name)
            .map(|s| s.parse::<usize>().unwrap_or_else(|_| panic!("{name} {s}")))
            .unwrap_or(default)
    };
    let gemv_rows = num("--gemv-rows", 1);
    let gemm_rows = num("--gemm-rows", 8);
    let n_experts = num("--experts", 64);
    let par = num("--par", 1).max(1);
    let budget_ms = num("--budget-ms", 3000) as f64;
    let ns: Vec<usize> = arg("--ns")
        .unwrap_or_else(|| "1,2,4,8,16,32".into())
        .split(',')
        .map(|s| s.trim().parse().expect("--ns"))
        .collect();

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let t = Instant::now();
    let bins: Vec<Vec<u8>> = (0..n_experts).map(|_| expert_bin(&mut rng)).collect();
    let bin_bytes = bins[0].len();
    let dir = std::env::temp_dir().join(format!("row_gemm_bench_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("layout.bin");
    std::fs::write(&path, &bins[0]).unwrap();
    let m = MmapExpert::open(&path, DIM, INTER).unwrap();
    println!(
        "# {isa:?} kernel, {} rayon threads, {n_experts} experts x {:.2} MB = {:.2} GB (built in {:.1}s), \
         per-row GEMV tile {gemv_rows}, GEMM tile {gemm_rows}, {par} expert(s) per call",
        rayon::current_num_threads(),
        bin_bytes as f64 / 1e6,
        (n_experts * bin_bytes) as f64 / 1e9,
        t.elapsed().as_secs_f64()
    );
    println!(
        "# bus GB/s = packed bytes actually streamed / s; old streams the bin once per row, new once per call"
    );
    println!(
        "{:>3} | {:>10} {:>9} {:>8} | {:>10} {:>9} {:>8} | {:>7}",
        "n", "old ms", "ms/row", "bus GB/s", "new ms", "ms/row", "bus GB/s", "speedup"
    );

    for &n in &ns {
        let xs: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..DIM)
                    .map(|_| (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = xs.iter().map(Vec::as_slice).collect();
        let old = |bin: &[u8]| -> Vec<f32> {
            let mut ys = Vec::with_capacity(n * DIM);
            for x in &refs {
                ys.extend_from_slice(&m.swiglu_from_forced(bin, x, isa, gemv_rows));
            }
            ys
        };
        let new = |bin: &[u8]| -> Vec<f32> {
            if n == 1 {
                // `swiglu_rows_from` hands one row to the single-input kernel.
                m.swiglu_from_forced(bin, refs[0], isa, gemv_rows)
            } else {
                m.swiglu_rows_from_forced(bin, &refs, isa, gemm_rows)
            }
        };
        let (a, b) = (old(&bins[0]), new(&bins[0]));
        assert!(
            a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
            "n={n}: multi-row output is not bit-identical to the per-row loop"
        );

        let mut cursor = 0usize;
        let mut time = |f: &(dyn Fn(&[u8]) -> Vec<f32> + Sync)| -> f64 {
            let call = |cursor: &mut usize| {
                let cohort: Vec<&Vec<u8>> =
                    (0..par).map(|i| &bins[(*cursor + i) % n_experts]).collect();
                *cursor += par;
                let t = Instant::now();
                if par == 1 {
                    std::hint::black_box(f(cohort[0]));
                } else {
                    let ys: Vec<Vec<f32>> = cohort.par_iter().map(|bin| f(bin)).collect();
                    std::hint::black_box(ys);
                }
                t.elapsed().as_secs_f64() * 1e3
            };
            let first = call(&mut cursor); // warm-up, sizes the run
            let calls = ((budget_ms / first.max(0.01)) as usize).clamp(6, 400);
            let mut ms: Vec<f64> = (0..calls).map(|_| call(&mut cursor)).collect();
            ms.sort_by(f64::total_cmp);
            ms[ms.len() / 2]
        };
        let old_ms = time(&old);
        let new_ms = time(&new);
        let gb = |passes: usize, ms: f64| (passes * par * bin_bytes) as f64 / 1e9 / (ms / 1e3);
        println!(
            "{:>3} | {:>10.2} {:>9.2} {:>8.1} | {:>10.2} {:>9.2} {:>8.1} | {:>6.2}x",
            n,
            old_ms,
            old_ms / n as f64,
            gb(n, old_ms),
            new_ms,
            new_ms / n as f64,
            gb(n.div_ceil(GEMM_MAX_INPUTS), new_ms),
            old_ms / new_ms
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}
