//! The multi-input int4 expert kernel (`MmapExpert::swiglu_rows_from`, the
//! `CASCADIA_INKLING_ROW_GEMM` path) against the single-input one it replaces:
//! every row of a block must come out BIT-IDENTICAL to `swiglu_from` on that
//! row alone — the property the multi-stream engine rests on (a stream decoded
//! in a batch equals the same stream decoded alone).
//!
//! Each kernel is forced explicitly (`Int4Isa`), so an AVX-512 build box also
//! runs the AVX2 path the Panther Lake fleet uses — against both of its
//! single-input schedules (`CASCADIA_INT4_GEMV_ROWS` = 1 and 4). The raw
//! pre-rounding dots (and the teeth check on summation order) are pinned by
//! the unit tests next to the kernel (`dsv4::expert_mmap::row_gemm_tests`).
//!
//! Own process: the MoE-layer test runs under the fleet's cache profile
//! (process-wide env, read once), so the block takes the freshly-read branch
//! on the first pass and the cache-hit branch on the second.

use std::path::{Path, PathBuf};

use cascadia_engine_sparse_moe::dsv4::expert_mmap::{Int4Isa, MmapExpert, GEMM_MAX_INPUTS};
use cascadia_engine_sparse_moe::inkling::ffn::{forward_rows, swiglu_mmap, AnyExpert};
use cascadia_engine_sparse_moe::inkling::moe::{MoeLayer, MoeWeights};

const G: usize = 32;
const NS: [usize; 6] = [1, 2, 3, 5, 8, 17];

// `production_entry_points_…` relies on 70 and 130 rows spanning two and three
// GEMM passes; fail the build if the pass size outgrows them.
const _: () = assert!(70 > GEMM_MAX_INPUTS && 130 > 2 * GEMM_MAX_INPUTS);

/// The fleet's read/cache profile. Set before anything in the library reads
/// its env (every test calls this first).
fn profile() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::env::set_var("CASCADIA_INKLING_EXPERT_CACHE_MIB", "64");
        std::env::set_var("CASCADIA_INKLING_REUSE_READ_BUFFERS", "1");
        std::env::set_var("CASCADIA_INKLING_PIPELINE_READS", "1");
        std::env::set_var("CASCADIA_INKLING_PREFILL_READS", "1");
        std::env::set_var("CASCADIA_INKLING_SKIP_BULK_PREFETCH", "1");
    });
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 16777216.0 - 0.5
    }
}

/// A random int4 expert bin (exporter layout: gate, up `[inter, dim]`, down
/// `[dim, inter]`, each nibbles then bf16-LE per-32 scales), scales of the real
/// model's order, both signs and zero.
fn expert_bin(rng: &mut Rng, dim: usize, inter: usize) -> Vec<u8> {
    let mut bin = Vec::new();
    for (out_dim, in_dim) in [(inter, dim), (inter, dim), (dim, inter)] {
        bin.extend((0..out_dim * in_dim / 2).map(|_| rng.next() as u8));
        for _ in 0..out_dim * in_dim / G {
            let s = (rng.next() % 255) as f32 / 8192.0 - 127.0 / 8192.0;
            bin.extend(half::bf16::from_f32(s).to_le_bytes());
        }
    }
    bin
}

fn open(dir: &Path, name: &str, bin: &[u8], dim: usize, inter: usize) -> MmapExpert {
    let path = dir.join(name);
    std::fs::write(&path, bin).unwrap();
    MmapExpert::open(&path, dim, inter).unwrap()
}

fn inputs(rng: &mut Rng, n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..dim).map(|_| 2.0 * rng.unit()).collect())
        .collect()
}

fn refs(xs: &[Vec<f32>]) -> Vec<&[f32]> {
    xs.iter().map(Vec::as_slice).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn isas() -> Vec<Int4Isa> {
    [Int4Isa::Scalar, Int4Isa::Avx2, Int4Isa::Avx512]
        .into_iter()
        .filter(|i| i.supported())
        .collect()
}

/// For every kernel this CPU has: the multi-input SwiGLU, per tile setting and
/// `n`, against the single-input SwiGLU on each row (AVX2: both GEMV schedules).
fn check_forced(dim: usize, inter: usize, tiles: &[usize], seed: u64) {
    let mut rng = Rng(seed);
    let dir = tempfile::tempdir().unwrap();
    let bin = expert_bin(&mut rng, dim, inter);
    let m = open(dir.path(), "e.bin", &bin, dim, inter);
    let xs = inputs(&mut rng, *NS.iter().max().unwrap(), dim);
    for isa in isas() {
        let gemv_rows: &[usize] = if isa == Int4Isa::Avx2 { &[1, 4] } else { &[1] };
        for &rows in gemv_rows {
            let want: Vec<Vec<f32>> = xs
                .iter()
                .map(|x| m.swiglu_from_forced(&bin, x, isa, rows))
                .collect();
            assert!(want[0].iter().any(|&v| v != 0.0) && want[0].iter().all(|v| v.is_finite()));
            for &tile in tiles {
                for n in NS {
                    let got = m.swiglu_rows_from_forced(&bin, &refs(&xs[..n]), isa, tile);
                    assert_eq!(got.len(), n * dim);
                    for j in 0..n {
                        assert_eq!(
                            bits(&got[j * dim..(j + 1) * dim]),
                            bits(&want[j]),
                            "{isa:?} {dim}x{inter} gemv_rows={rows} tile={tile} n={n} row {j}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn multi_row_swiglu_is_bit_identical_at_the_real_dims() {
    profile();
    check_forced(6144, 3072, &[8], 0xA11CE);
}

#[test]
fn multi_row_swiglu_is_bit_identical_at_tiny_dims_for_every_tile() {
    profile();
    check_forced(64, 32, &[1, 2, 4, 8], 0xB0B);
    check_forced(96, 160, &[1, 2, 4, 8], 0xC0DE);
    check_forced(32, 32, &[3, 5], 7); // odd tiles: a short last tile (32 % 3, 32 % 5)
}

/// The production entry points (active kernel, env-selected schedules): read
/// buffer and mapping, incl. blocks past one GEMM pass (`GEMM_MAX_INPUTS` = 64:
/// 70 rows run as 35 + 35, 130 as 44 + 43 + 43).
#[test]
fn production_entry_points_match_swiglu_from_and_swiglu_mmap() {
    profile();
    for (dim, inter, n) in [(6144usize, 3072usize, 5usize), (64, 32, 70), (64, 32, 130)] {
        let mut rng = Rng(0xD00D + n as u64);
        let dir = tempfile::tempdir().unwrap();
        let bin = expert_bin(&mut rng, dim, inter);
        let m = open(dir.path(), "e.bin", &bin, dim, inter);
        let xs = inputs(&mut rng, n, dim);
        let data = m.read_bytes().unwrap();
        let from_buffer = m.swiglu_rows_from(&data, &refs(&xs));
        let from_mapping = m.swiglu_rows(&refs(&xs));
        for (j, x) in xs.iter().enumerate() {
            let want = m.swiglu_from(&data, x);
            assert_eq!(
                bits(&from_buffer[j * dim..(j + 1) * dim]),
                bits(&want),
                "swiglu_rows_from {dim}x{inter} n={n} row {j}"
            );
            assert_eq!(
                bits(&from_mapping[j * dim..(j + 1) * dim]),
                bits(&swiglu_mmap(&m, x)),
                "swiglu_rows {dim}x{inter} n={n} row {j}"
            );
            // The forced baseline on the active kernel IS the production one.
            assert_eq!(
                bits(&m.swiglu_from_forced(&data, x, Int4Isa::active(), 1)),
                bits(&want)
            );
        }
        assert!(m.swiglu_rows_from(&data, &[]).is_empty());
        assert_eq!(
            bits(&m.swiglu_rows_from(&data, &[&xs[0]])),
            bits(&m.swiglu_from(&data, &xs[0]))
        );
    }
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export/experts/layer_01")
}

/// `forward_rows` per storage (mapped int4, owned int4, eager f32) against
/// `AnyExpert::forward` per row.
#[test]
fn forward_rows_matches_forward_for_every_storage() {
    profile();
    let (hidden, inter) = (64usize, 32usize);
    let open = |name: &str| MmapExpert::open(&fixture().join(name), hidden, inter).unwrap();
    let mut rng = Rng(0xFEED);
    let eager = AnyExpert::EagerF32 {
        wg: (0..inter * hidden).map(|_| rng.unit()).collect(),
        wu: (0..inter * hidden).map(|_| rng.unit()).collect(),
        wd: (0..inter * hidden).map(|_| rng.unit()).collect(),
    };
    let experts = [
        AnyExpert::Mmap(open("expert_shared0.bin")),
        AnyExpert::Mmap(open("expert_shared1.bin"))
            .into_owned_int4()
            .unwrap(),
        eager,
    ];
    let xs = inputs(&mut rng, 19, hidden);
    for (i, e) in experts.iter().enumerate() {
        for parallel in [false, true] {
            let got = forward_rows(e, &refs(&xs), hidden, inter, parallel);
            for (j, x) in xs.iter().enumerate() {
                assert_eq!(
                    bits(&got[j * hidden..(j + 1) * hidden]),
                    bits(&e.forward(x, hidden, inter)),
                    "storage {i} parallel={parallel} row {j}"
                );
            }
        }
    }
}

/// The batch-union block under the fleet's cache profile: every row of
/// `forward_batch` (multi-input kernel; freshly-read branch on pass 1,
/// cache-hit branch on pass 2) equals that row through the single-token
/// `forward` (single-input kernels) bit for bit.
#[test]
fn batched_block_equals_single_token_forward_on_miss_and_on_hit() {
    profile();
    let (hidden, inter, n_routed, n_shared, rows) = (64usize, 32usize, 8usize, 2usize, 41usize);
    let mut rng = Rng(0x5EA);
    let router_w: Vec<f32> = (0..(n_routed + n_shared) * hidden)
        .map(|_| rng.unit())
        .collect();
    let layer = |router_w: Vec<f32>| {
        let open = |name: String| MmapExpert::open(&fixture().join(name), hidden, inter).unwrap();
        MoeLayer::new(
            hidden,
            inter,
            3,
            1.0,
            MoeWeights {
                router_w,
                router_bias: vec![0.0; n_routed],
                global_scale: 8.0,
                experts: (0..n_routed)
                    .map(|e| AnyExpert::Mmap(open(format!("expert_{e:03}.bin"))))
                    .collect(),
                shared: vec![
                    AnyExpert::Mmap(open("expert_shared0.bin".into())),
                    AnyExpert::Mmap(open("expert_shared1.bin".into()))
                        .into_owned_int4()
                        .unwrap(),
                ],
            },
        )
    };
    let xs: Vec<f32> = (0..rows * hidden).map(|_| 4.0 * rng.unit()).collect();
    let single = layer(router_w.clone());
    let want: Vec<f32> = xs.chunks(hidden).flat_map(|x| single.forward(x)).collect();
    assert!(want.iter().any(|&v| v != 0.0));

    let batched = layer(router_w);
    assert!(
        batched.expert_cache_stats().capacity_bytes > 0,
        "cache profile not applied"
    );
    let miss = batched.forward_batch(&xs, rows);
    let after_miss = batched.expert_cache_stats();
    let hit = batched.forward_batch(&xs, rows);
    let after_hit = batched.expert_cache_stats();
    assert!(
        after_miss.admissions > 0,
        "pass 1 admitted nothing: {after_miss:?}"
    );
    assert!(
        after_hit.hits > after_miss.hits,
        "pass 2 never hit the cache: {after_hit:?}"
    );
    assert_eq!(bits(&miss), bits(&want), "freshly-read branch");
    assert_eq!(bits(&hit), bits(&want), "cache-hit branch");
}
