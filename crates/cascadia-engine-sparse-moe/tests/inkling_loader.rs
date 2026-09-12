//! Loader round-trip: `inkling::loader::load_model` reads a
//! `tools/export_inkling.py --tiny` layout (bf16/f32 shell safetensors + int4
//! expert bins) and must generate the exact greedy tokens transformers' native
//! `InklingForCausalLM` produces from the SAME (int4-dequantized) weights —
//! `reference.json` next to the export. Token-exact through int4 dequant
//! validates the exporter write, the loader read and the int4 numeric
//! contract together. The staged runner (rank 0 of 1) must match the
//! single-process model on the same prompt.
//!
//! Regenerate the export:
//!   python tools/export_inkling.py --tiny \
//!       crates/cascadia-engine-sparse-moe/tests/fixtures/inkling_export

use std::path::{Path, PathBuf};

use cascadia_engine_sparse_moe::dsv4::loader::ExpertsMode;
use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::inkling::loader::{
    load_layer, load_model, load_model_with, read_manifest, ExpertSet,
};
use cascadia_engine_sparse_moe::inkling::model::argmax;
use cascadia_engine_sparse_moe::inkling::stage::InklingRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;
use half::bf16;

fn export_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export")
}

struct Reference {
    prompt: Vec<u32>,
    greedy: Vec<u32>,
    first_argmax: Option<u32>,
}

fn reference() -> Option<Reference> {
    let p = export_dir().join("reference.json");
    if !p.exists() {
        eprintln!("inkling_export/reference.json missing; skipping (run export_inkling.py --tiny)");
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .unwrap_or_else(|| panic!("reference.json: {k}"))
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    Some(Reference {
        prompt: ids("prompt_ids"),
        greedy: ids("greedy_ids"),
        first_argmax: v["first_logits_argmax"].as_u64().map(|x| x as u32),
    })
}

#[test]
fn loader_greedy_matches_hf_reference() {
    let Some(r) = reference() else { return };
    let mut model = load_model(&export_dir(), 64).expect("load inkling export");
    if let Some(want) = r.first_argmax {
        let logits = model.prefill(&r.prompt);
        assert_eq!(argmax(&logits) as u32, want, "first-token argmax");
        model.reset();
    }
    let got = model.greedy(&r.prompt, r.greedy.len());
    assert_eq!(got, r.greedy, "loader greedy mismatch vs HF reference");
}

#[test]
fn staged_runner_single_rank_matches_model() {
    let Some(r) = reference() else { return };
    let mut runner =
        InklingRunner::load_staged(&export_dir(), 64, 0, 1, 0, 0, Some("eager".into()), None)
            .expect("load rank 0 of 1");
    runner.reset();
    // Token-by-token drive (the pipeline's decode path).
    let mut pos = 0usize;
    let mut next = 0u32;
    for &t in &r.prompt {
        let h = runner.embed_token(t);
        let h = runner.forward_layers(h, pos, None);
        next = argmax(&runner.head_logits(&h)) as u32;
        pos += 1;
    }
    let mut got = vec![next];
    for _ in 1..r.greedy.len() {
        let h = runner.embed_token(next);
        let h = runner.forward_layers(h, pos, None);
        next = argmax(&runner.head_logits(&h)) as u32;
        pos += 1;
        got.push(next);
    }
    assert_eq!(got, r.greedy, "staged token-by-token greedy mismatch");

    // Batched prefill (the pipeline's prefill path) must land on the same first token.
    runner.reset();
    let hs = runner.hidden_size();
    let mut rows = Vec::with_capacity(r.prompt.len() * hs);
    for &t in &r.prompt {
        rows.extend(runner.embed_token(t));
    }
    let out = runner.forward_layers_batch(rows, 0, r.prompt.len());
    let last = &out[(r.prompt.len() - 1) * hs..];
    assert_eq!(
        argmax(&runner.head_logits(last)) as u32,
        r.greedy[0],
        "batched prefill first token"
    );
}

#[test]
fn layer_split_covers_every_layer_exactly_once() {
    use cascadia_engine_sparse_moe::inkling::loader::read_manifest;
    use cascadia_engine_sparse_moe::inkling::stage::layer_split;
    let Ok(m) = read_manifest(&export_dir()) else {
        eprintln!("inkling_export/manifest.json missing; skipping");
        return;
    };
    for total in 1..=m.num_layers as u32 {
        let mut next = 0usize;
        for rank in 0..total {
            let (lo, hi) = layer_split(&m, rank, total).unwrap();
            assert_eq!(lo, next, "rank {rank}/{total} contiguity");
            assert!(hi > lo, "rank {rank}/{total} non-empty");
            next = hi;
        }
        assert_eq!(next, m.num_layers, "total {total} covers all layers");
    }
    // One rank more than layers: the LAST rank would own zero layers.
    let n = m.num_layers as u32;
    assert!(layer_split(&m, n, n + 1).is_err());
}

// ---- bf16 weights are read straight off the payload -----------------------

/// `StFile::bf16_bits` (a payload copy for BF16, a narrow for F32) must give
/// the same bits the old path did (`f32` then `bf16::from_f32`) for every
/// tensor of the tiny export — the edge tables, the projections and the f32
/// norms / convs / router alike.
#[test]
fn bf16_bits_equals_the_f32_path_narrowed_on_the_tiny_export() {
    let dir = export_dir();
    let Ok(m) = read_manifest(&dir) else {
        eprintln!("inkling_export/manifest.json missing; skipping");
        return;
    };
    let mut files = vec![dir.join("embed.safetensors"), dir.join("head.safetensors")];
    for li in 0..m.num_layers {
        files.push(dir.join(format!("shells/layer_{li:02}.safetensors")));
    }
    let (mut n_bf16, mut n_f32) = (0usize, 0usize);
    for f in files {
        let st = StFile::open(&f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        let mut names: Vec<&String> = st.tensors.keys().collect();
        names.sort();
        for name in names {
            let (sa, bits) = st.bf16_bits(name).unwrap();
            let (sb, f) = st.f32(name).unwrap();
            assert_eq!(sa, sb, "{name}: shape");
            let want: Vec<u16> = f.iter().map(|&x| bf16::from_f32(x).to_bits()).collect();
            assert_eq!(bits, want, "{name}: bf16 bits");
            match st.info(name).unwrap().dtype.as_str() {
                "BF16" => n_bf16 += 1,
                "F32" => n_f32 += 1,
                _ => {}
            }
        }
    }
    assert!(
        n_bf16 > 0 && n_f32 > 0,
        "bf16 {n_bf16} f32 {n_f32}: both dtypes must be covered"
    );
}

/// Minimal safetensors writer (header + concatenated payload) for the dtype test.
fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (name, dtype, shape, bytes) in tensors {
        let end = off + bytes.len();
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, end]}),
        );
        off = end;
    }
    let mut hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    while !hdr.len().is_multiple_of(8) {
        hdr.push(b' ');
    }
    let mut out = (hdr.len() as u64).to_le_bytes().to_vec();
    out.extend(hdr);
    for (_, _, _, bytes) in tensors {
        out.extend(bytes);
    }
    std::fs::write(path, out).unwrap();
}

#[test]
fn bf16_bits_copies_bf16_and_narrows_f32_and_f16() {
    let vals = [1.0f32, -2.5, 3.3125, 65504.0, 1e-3, -0.0, 0.1, 1234.5678];
    let f32_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let f16_bytes: Vec<u8> = vals
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect();
    let bf16_bytes: Vec<u8> = vals
        .iter()
        .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("dtypes.safetensors");
    write_safetensors(
        &p,
        &[
            ("a_f32", "F32", vec![2, 4], f32_bytes),
            ("b_f16", "F16", vec![8], f16_bytes),
            ("c_bf16", "BF16", vec![4, 2], bf16_bytes),
        ],
    );
    let st = StFile::open(&p).unwrap();
    let narrowed =
        |v: &[f32]| -> Vec<u16> { v.iter().map(|&x| bf16::from_f32(x).to_bits()).collect() };

    let (shape, bits) = st.bf16_bits("a_f32").unwrap();
    assert_eq!(shape, vec![2, 4]);
    assert_eq!(bits, narrowed(&vals));

    let (shape, bits) = st.bf16_bits("b_f16").unwrap();
    assert_eq!(shape, vec![8]);
    let via_f16: Vec<f32> = vals
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    assert_eq!(bits, narrowed(&via_f16));

    let (shape, bits) = st.bf16_bits("c_bf16").unwrap();
    assert_eq!(shape, vec![4, 2]);
    assert_eq!(bits, narrowed(&vals), "BF16 payload copied verbatim");
    assert_eq!(
        bits,
        narrowed(&st.f32("c_bf16").unwrap().1),
        "== f32 path narrowed"
    );
}

// ---- manifest contract -----------------------------------------------------

/// A complete, valid manifest (the tiny export's) as JSON the tests mutate.
fn base_manifest() -> serde_json::Value {
    serde_json::json!({
        "arch": "inkling",
        "num_layers": 4,
        "hidden_size": 64,
        "vocab_size": 128,
        "unpadded_vocab_size": 120,
        "num_attention_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 16,
        "d_rel": 4,
        "rel_extent": 8,
        "sliding_window": 4,
        "layer_types": ["sliding", "sliding", "sliding", "global"],
        "dense_layers": [0],
        "dense_intermediate": 64,
        "moe_intermediate": 32,
        "num_experts": 8,
        "top_k": 2,
        "n_shared_experts": 2,
        "route_scale": 8.0,
        "rms_norm_eps": 1e-6,
        "log_scaling_n_floor": 4,
        "log_scaling_alpha": 0.1,
        "logits_mup_width_multiplier": 24.0,
        "conv_kernel_size": 4,
        "eos_token_ids": [127]
    })
}

fn read(v: &serde_json::Value) -> Result<(), String> {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("manifest.json"),
        serde_json::to_string_pretty(v).unwrap(),
    )
    .unwrap();
    read_manifest(dir.path())
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn without(key: &str) -> serde_json::Value {
    let mut v = base_manifest();
    v.as_object_mut().unwrap().remove(key).expect(key);
    v
}

fn with(key: &str, val: serde_json::Value) -> serde_json::Value {
    let mut v = base_manifest();
    v[key] = val;
    v
}

#[test]
fn manifest_base_is_accepted() {
    read(&base_manifest()).expect("base manifest");
}

/// The three scales that silently default into coherent-looking garbage are
/// required — serde reports the missing field by name.
#[test]
fn manifest_requires_route_scale() {
    let e = read(&without("route_scale")).unwrap_err();
    assert!(e.contains("route_scale"), "{e}");
}

#[test]
fn manifest_requires_logits_mup_width_multiplier() {
    let e = read(&without("logits_mup_width_multiplier")).unwrap_err();
    assert!(e.contains("logits_mup_width_multiplier"), "{e}");
}

#[test]
fn manifest_requires_log_scaling_alpha() {
    let e = read(&without("log_scaling_alpha")).unwrap_err();
    assert!(e.contains("log_scaling_alpha"), "{e}");
    // ... even when log scaling is off (n_floor absent).
    let mut v = without("log_scaling_alpha");
    v.as_object_mut().unwrap().remove("log_scaling_n_floor");
    let e = read(&v).unwrap_err();
    assert!(e.contains("log_scaling_alpha"), "{e}");
}

#[test]
fn manifest_rejects_zero_moe_intermediate() {
    let e = read(&with("moe_intermediate", serde_json::json!(0))).unwrap_err();
    assert!(e.contains("moe_intermediate"), "{e}");
}

#[test]
fn manifest_rejects_zero_dense_intermediate_when_dense_layers_present() {
    let e = read(&with("dense_intermediate", serde_json::json!(0))).unwrap_err();
    assert!(e.contains("dense_intermediate"), "{e}");
    let mut v = with("dense_intermediate", serde_json::json!(0));
    v.as_object_mut().unwrap().remove("dense_intermediate");
    let e = read(&v).unwrap_err();
    assert!(
        e.contains("dense_intermediate"),
        "{e} (serde default 0 with dense layers)"
    );
}

#[test]
fn manifest_allows_zero_dense_intermediate_without_dense_layers() {
    let mut v = with("dense_intermediate", serde_json::json!(0));
    v["dense_layers"] = serde_json::json!([]);
    read(&v).expect("no dense layers -> dense_intermediate unused");
}

#[test]
fn manifest_rejects_nonpositive_mup() {
    for bad in [0.0f64, -24.0] {
        let e = read(&with("logits_mup_width_multiplier", serde_json::json!(bad))).unwrap_err();
        assert!(e.contains("logits_mup_width_multiplier"), "{e}");
    }
}

// ---- mmap experts: overlapped reads --------------------------------------

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// `MoeLayer::forward` on mmap'd experts prefetches the selection and reads
/// every selected bin concurrently, then runs the GEMVs from those buffers;
/// `forward_batch` prefetches and runs the same kernel straight off the mmap.
/// Same bytes, same kernel: bit-identical.
#[test]
fn mmap_overlapped_reads_match_the_mmap_kernel_bit_for_bit() {
    let dir = export_dir();
    let Ok(m) = read_manifest(&dir) else {
        eprintln!("inkling_export/manifest.json missing; skipping");
        return;
    };
    let li = (0..m.num_layers)
        .find(|li| !m.dense_layers.contains(li))
        .expect("a MoE layer");
    let layer =
        load_layer(&dir, &m, li, 64, ExpertsMode::Mmap, ExpertSet::All).expect("mmap layer");
    let moe = layer.moe().expect("MoE layer");
    let hidden = m.hidden_size;
    let rows = 6;
    let xs = Lcg(41).vec(rows * hidden);
    let batch = moe.forward_batch(&xs, rows);
    let mut per_row = Vec::with_capacity(rows * hidden);
    for x in xs.chunks_exact(hidden) {
        per_row.extend(moe.forward(x));
    }
    assert_eq!(
        batch, per_row,
        "overlapped-read forward vs mmap-kernel batch"
    );
    assert!(per_row.iter().any(|&v| v != 0.0));
    assert!(per_row.iter().all(|v| v.is_finite()));
}

/// The whole model on mmap experts: the HF reference greedy ids, prefill ==
/// per-token bit for bit (batch-union + prefetch vs overlapped reads), and the
/// eager (dequant-then-dot) model within the fused kernel's tolerance.
#[test]
fn mmap_model_matches_the_reference_and_the_eager_model() {
    let Some(r) = reference() else { return };
    let mut mm = load_model_with(&export_dir(), 64, ExpertsMode::Mmap).expect("mmap model");
    assert_eq!(
        mm.greedy(&r.prompt, r.greedy.len()),
        r.greedy,
        "mmap greedy vs HF reference"
    );
    mm.reset();
    let pre = mm.prefill(&r.prompt);
    mm.reset();
    let mut tok = Vec::new();
    for &t in &r.prompt {
        tok = mm.forward_token(t);
    }
    assert_eq!(pre, tok, "mmap prefill vs per-token");

    let mut eager = load_model(&export_dir(), 64).expect("eager model");
    let ea = eager.prefill(&r.prompt);
    assert_eq!(argmax(&pre), argmax(&ea), "mmap vs eager argmax");
    let scale = ea.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = pre
        .iter()
        .zip(&ea)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst <= 2e-2 * scale.max(1.0),
        "mmap vs eager logits: worst |diff| {worst} (scale {scale})"
    );
}
