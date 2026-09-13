//! Synthetic resident-layer benchmark at Inkling 975B dimensions.
//!
//! Uses the production attention, routing, mmap expert and decoder paths.
//! Eight distinct int4 bins stand for the six selected + two shared experts;
//! the other router entries are suppressed. This is NOT a checkpoint benchmark
//! and layer_tokens_per_s is NOT whole-model generated tokens/s.
use std::hint::black_box;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cascadia_engine_sparse_moe::dsv4::expert_mmap::MmapExpert;
use cascadia_engine_sparse_moe::inkling::attn::{AttentionLayer, AttnDims, AttnWeights};
use cascadia_engine_sparse_moe::inkling::conv::ShortConv;
use cascadia_engine_sparse_moe::inkling::ffn::AnyExpert;
use cascadia_engine_sparse_moe::inkling::model::{Layer, LayerMlp};
use cascadia_engine_sparse_moe::inkling::moe::{MoeLayer, MoeWeights};
use cascadia_engine_sparse_moe::inkling::relpos::RelPos;

const H: usize = 6144;
const I: usize = 3072;
const Q: usize = 8192;
const KV: usize = 2048;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn f32(&mut self, scale: f32) -> f32 {
        ((self.next() >> 8) as f32 / 16777216.0 * 2.0 - 1.0) * scale
    }
    fn floats(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32(scale)).collect()
    }
    fn bf16(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.f32(0.02)).to_bits())
            .collect()
    }
}

fn experts(dir: &Path) -> Result<MoeLayer, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let section = H * I / 2 + H * I / 32 * 2;
    for id in 0..8 {
        let path = dir.join(format!("synthetic_{id}.bin"));
        if path.exists() {
            if path.metadata()?.len() != (3 * section) as u64 {
                return Err(format!("unexpected synthetic bin size: {}", path.display()).into());
            }
            continue;
        }
        let mut rng = Rng(900 + id);
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path)?);
        for _ in 0..3 {
            let packed: Vec<u8> = (0..H * I / 2).map(|_| rng.next() as u8).collect();
            file.write_all(&packed)?;
            for _ in 0..H * I / 32 {
                file.write_all(
                    &half::bf16::from_f32(0.002 + rng.f32(0.0005))
                        .to_bits()
                        .to_le_bytes(),
                )?;
            }
        }
        file.flush()?;
    }
    let open = |id: usize| -> Result<AnyExpert, Box<dyn std::error::Error>> {
        Ok(AnyExpert::Mmap(MmapExpert::open(
            &dir.join(format!("synthetic_{id}.bin")),
            H,
            I,
        )?))
    };
    let mut bias = vec![-100.0; 256];
    bias[..6].fill(0.0);
    let mut rng = Rng(704);
    let w = MoeWeights {
        router_w: rng.floats(258 * H, 0.01),
        router_bias: bias,
        global_scale: 1.0,
        experts: (0..256)
            .map(|id| open(id % 8))
            .collect::<Result<Vec<_>, _>>()?,
        shared: (6..8).map(open).collect::<Result<Vec<_>, _>>()?,
    };
    Ok(MoeLayer::new(H, I, 6, 1.0, w))
}

fn attention() -> AttentionLayer {
    let mut r = Rng(77);
    let w = AttnWeights {
        wq: r.bf16(Q * H),
        wk: r.bf16(KV * H),
        wv: r.bf16(KV * H),
        wr: r.bf16(64 * 16 * H),
        wo: r.bf16(H * Q),
        q_norm: vec![1.0; 128],
        k_norm: vec![1.0; 128],
    };
    let kc = ShortConv::new(r.floats(KV * 4, 0.05), KV, 4);
    let vc = ShortConv::new(r.floats(KV * 4, 0.05), KV, 4);
    AttentionLayer::from_parts(
        AttnDims::sliding(H, 64, 16, 128, 16, 512, 1e-6),
        w,
        kc,
        vc,
        RelPos::new(r.floats(16 * 512, 0.05), 16, 512),
    )
}

fn hash_update(mut hash: u64, values: &[f32]) -> u64 {
    for &v in values {
        assert!(v.is_finite(), "non-finite output");
        for b in v.to_bits().to_le_bytes() {
            hash = (hash ^ b as u64).wrapping_mul(1099511628211);
        }
    }
    hash
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |key: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == key)
            .and_then(|p| args.get(p + 1))
            .cloned()
            .unwrap_or_else(|| default.into())
    };
    let kind = arg("--kind", "layer");
    let tokens: usize = arg("--tokens", "32").parse()?;
    let samples: usize = arg("--samples", "5").parse()?;
    let warmup: usize = arg("--warmup", "16").parse()?;
    if tokens == 0 || samples == 0 || warmup > 4096 {
        return Err("invalid sample/warmup count".into());
    }
    let dir = PathBuf::from(arg("--expert-dir", "synthetic-experts"));
    cascadia_engine_sparse_moe::init_thread_pool();
    let mut rng = Rng(20260912);
    let inputs = rng.floats((tokens + warmup) * H, 1.0);
    let moe = experts(&dir)?;
    let mut layer = Layer::new(
        H,
        1e-6,
        vec![1.0; H],
        attention(),
        ShortConv::new(rng.floats(H * 4, 0.03), H, 4),
        vec![1.0; H],
        LayerMlp::Moe(moe),
        ShortConv::new(rng.floats(H * 4, 0.03), H, 4),
    );
    #[cfg(target_arch = "x86_64")]
    println!(
        "cpu_avx2={} cpu_fma={} cpu_avx512f={}",
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("fma"),
        is_x86_feature_detected!("avx512f")
    );
    println!("scope=synthetic_resident_layer hidden={H} inter={I} selected_experts=8 kind={kind} threads={} tokens={tokens} warmup={warmup}", rayon::current_num_threads());
    let mut timings = Vec::new();
    let mut hashes = Vec::new();
    for sample in 0..samples {
        layer.reset();
        for x in inputs[..warmup * H].chunks_exact(H) {
            black_box(layer.forward_token(x));
        }
        let start = Instant::now();
        let mut outputs = Vec::with_capacity(tokens * H);
        match kind.as_str() {
            "layer" => {
                for x in inputs[warmup * H..].chunks_exact(H) {
                    outputs.extend(layer.forward_token(black_box(x)));
                }
            }
            "moe" => {
                for x in inputs[warmup * H..].chunks_exact(H) {
                    outputs.extend(layer.moe().unwrap().forward(black_box(x)));
                }
            }
            "prefill" => outputs = layer.forward_prefill(black_box(&inputs[warmup * H..]), tokens),
            _ => return Err(format!("unsupported kind: {kind}").into()),
        }
        let elapsed = start.elapsed().as_secs_f64();
        let digest = hash_update(14695981039346656037, &outputs);
        hashes.push(digest);
        timings.push(elapsed / tokens as f64 * 1000.0);
        println!(
            "sample={sample} ms_per_layer_token={:.6} hash={digest:016x}",
            timings.last().unwrap()
        );
    }
    assert!(
        hashes.iter().all(|h| *h == hashes[0]),
        "non-deterministic output across repeats"
    );
    timings.sort_by(f64::total_cmp);
    let median = timings[timings.len() / 2];
    println!("layer_ms={median:.6}");
    println!("layer_tokens_per_s={:.6}", 1000.0 / median);
    println!("output_hash={:016x}", hashes[0]);
    let expected = arg("--expect-hash", "");
    if !expected.is_empty() && expected != format!("{:016x}", hashes[0]) {
        return Err(format!(
            "correctness gate: expected {expected}, got {:016x}",
            hashes[0]
        )
        .into());
    }
    println!("timing_samples_json={}", serde_json::to_string(&timings)?);
    Ok(())
}
