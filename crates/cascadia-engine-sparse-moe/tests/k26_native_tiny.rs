//! End-to-end proof that the NATIVE sparse-MoE path (`shell_backend =
//! "rust_k26"` → `Runner` / `SparseMoEEngine`) loads and runs against a
//! synthetic tree, the native analogue of the `m2_tiny` OV fixture.
//!
//! Gated on `K26_TINY_DIR` because the fixture CANNOT be committed: the native
//! kernels pin K2.6 dims at compile time (`cascadia-int4-gemm/src/shell.rs`
//! HIDDEN=7168 … INTERMEDIATE_DENSE=18432, enforced by the
//! `assert_eq!(weight_bf16.len(), n_rows * k_cols * 2)` in
//! `shell_int4.rs::quantize_int4_group`), so the smallest shape-valid tree is
//! ~1.4 GiB. Produce one with:
//!
//! ```sh
//! python3 tools/export_kimi_k26.py --tiny --out /tmp/k26_tiny
//! K26_TINY_DIR=/tmp/k26_tiny cargo test -p cascadia-engine-sparse-moe \
//!     --features kv_coord --test k26_native_tiny -- --nocapture
//! ```
//!
//! The tree is loaded as a NON-LAST stage. That is not a shortcut: the native
//! head is OV-IR only (`runner.rs` `Runtime::compile(head_xml)`), and without
//! the `openvino` feature `Runtime::compile` is a stub that always errors —
//! `native_last_stage_requires_openvino` pins that wall.

use std::path::PathBuf;

use cascadia_engine::{Builder, Engine};
use cascadia_engine_sparse_moe::runner::LayerRange;
use cascadia_engine_sparse_moe::{
    Manifest, Runner, RunnerOptions, SparseMoEBuilder, SparseMoEBuilderConfig,
};
use cascadia_ov_genai_shim::PluginConfig;
use cascadia_types::{PeerLayout, ShardSpec};

fn tiny_dir() -> Option<PathBuf> {
    std::env::var("K26_TINY_DIR").ok().map(PathBuf::from)
}

/// `layer_end` covering every MoE layer in the manifest (ids are 1-based).
fn moe_end(m: &Manifest) -> u32 {
    m.moe_layer_ids().last().copied().unwrap_or(0) + 1
}

fn first_stage_shard(m: &Manifest, is_last: bool) -> ShardSpec {
    ShardSpec {
        model_id: "k26_tiny".into(),
        layer_start: 1,
        layer_end: moe_end(m),
        total_layers: m.num_layers,
        device: "CPU".into(),
        is_first_stage: true,
        is_last_stage: is_last,
        tp_size: 1,
        tp_rank: 0,
    }
}

/// The two predicates `SparseMoEBuilder::load` uses to pick an engine, checked
/// against the fixture so a passing load can't silently be the OvMoe or Dsv4
/// engine instead.
#[test]
fn fixture_selects_the_native_branch() {
    let Some(dir) = tiny_dir() else {
        eprintln!("K26_TINY_DIR not set; skipping");
        return;
    };
    let raw: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert_ne!(
        raw.get("arch").and_then(|a| a.as_str()),
        Some("deepseek_v4"),
        "would route to Dsv4Engine"
    );
    let m = Manifest::load(&dir).expect("manifest");
    assert!(!m.is_ov_shell(), "would route to OvMoeEngine");
    assert!(!dir.join("shells").exists(), "no OV shell IRs in this tree");
    assert!(!dir.join("head").exists(), "no OV head IR in this tree");
}

/// The wall: a last stage needs the OV-IR head, and `Runtime::compile` without
/// the `openvino` feature is a stub. Nothing in a native tree can satisfy it.
///
/// Mirrors the fixture into a temp dir (symlinks — the shard is >1 GiB) and
/// adds a `head/openvino_model.xml`, so the failure is the compile stub and
/// not merely the missing-file check that precedes it.
///
/// Unix-only: `std::os::unix::fs::symlink` does not exist on Windows, and an
/// ungated use is a compile error for the whole test binary on the fleet nodes.
#[test]
#[cfg(unix)]
fn native_last_stage_requires_openvino() {
    let Some(src) = tiny_dir() else {
        eprintln!("K26_TINY_DIR not set; skipping");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    for e in std::fs::read_dir(&src).unwrap() {
        let e = e.unwrap();
        std::os::unix::fs::symlink(e.path(), dir.join(e.file_name())).unwrap();
    }
    std::fs::create_dir(dir.join("head")).unwrap();
    std::fs::write(dir.join("head").join("openvino_model.xml"), b"<net/>").unwrap();

    let m = Manifest::load(&dir).expect("manifest");
    let r = Runner::load(
        dir.clone(),
        "CPU",
        PluginConfig::new(),
        LayerRange {
            layer_start: 1,
            layer_end: moe_end(&m),
            is_first: true,
            is_last: true,
        },
    );
    let s = match r {
        Ok(_) => panic!("last stage loaded without a real OV head"),
        Err(e) => e.to_string(),
    };
    assert!(
        s.contains("ov:"),
        "expected the OV compile wall (not the missing-file check), got {s:?}"
    );
    eprintln!("last-stage wall: {s}");
}

/// The load proof: `SparseMoEBuilder` reaches a live engine, and that engine
/// exposes the native `SparseMoeKvHolder`.
#[tokio::test]
async fn native_first_stage_builds_a_live_engine() {
    let Some(dir) = tiny_dir() else {
        eprintln!("K26_TINY_DIR not set; skipping");
        return;
    };
    let m = Manifest::load(&dir).expect("manifest");
    // rank 0 of 2: is_first = true, is_last = false (head skipped).
    let mut b = SparseMoEBuilder::new(
        SparseMoEBuilderConfig::new(dir.to_str().unwrap(), "CPU").with_rank(0, 2),
    );
    b.connect(PeerLayout {
        upstream: None,
        downstream: None,
    })
    .await
    .expect("connect");
    let _progress = b.load(first_stage_shard(&m, false)).await.expect("load");
    let engine: Box<dyn Engine> = Box::new(b).build().expect("build");

    #[cfg(feature = "kv_coord")]
    assert!(
        engine.kv_holder().is_some(),
        "native engine must expose a KvSnapshotHolder"
    );
    let _ = engine;
}

/// The genuinely smallest native tree: a MIDDLE rank (`is_first = false`,
/// `is_last = false`) needs neither layer 0 / embed nor the head, so the tree
/// is shells only. Generate with `--no-layer0 --no-experts`.
#[test]
fn native_middle_stage_loads_shells_only() {
    let Some(dir) = std::env::var("K26_TINY_MID_DIR").ok().map(PathBuf::from) else {
        eprintln!("K26_TINY_MID_DIR not set; skipping");
        return;
    };
    let m = Manifest::load(&dir).expect("manifest");
    let runner = Runner::load(
        dir.clone(),
        "CPU",
        PluginConfig::new(),
        LayerRange {
            layer_start: 1,
            layer_end: moe_end(&m),
            is_first: false,
            is_last: false,
        },
    )
    .expect("middle-stage Runner::load");
    assert_eq!(
        runner.kv_past_seq_lens().len(),
        m.moe_layer_ids().len(),
        "one KV layer per owned MoE shell, no layer 0"
    );
}

/// Determinism: the same token through layer 0 + the shells twice, from a
/// freshly reset KV cache, must be bit-identical.
#[test]
fn native_forward_is_deterministic() {
    let Some(dir) = tiny_dir() else {
        eprintln!("K26_TINY_DIR not set; skipping");
        return;
    };
    let m = Manifest::load(&dir).expect("manifest");
    let mut runner = Runner::load_with_options(
        dir.clone(),
        "CPU",
        PluginConfig::new(),
        LayerRange {
            layer_start: 1,
            layer_end: moe_end(&m),
            is_first: true,
            is_last: false,
        },
        RunnerOptions::default(),
    )
    .expect("native Runner::load");

    let hidden = m.hidden_size as usize;
    let mut run = || {
        runner.reset_kv();
        let mut out = Vec::new();
        for &tok in &[3i64, 7, 11] {
            let h = runner.forward_layer0_step(tok).expect("layer0");
            assert_eq!(h.len(), hidden);
            let past = runner.kv_past_seq_len();
            let y = runner
                .forward_shells(&h, &[1, 1, hidden], past.saturating_sub(1))
                .expect("shells");
            out.push(y);
        }
        out
    };

    let a = run();
    let b = run();
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let xb: Vec<u32> = x.iter().map(|v| v.to_bits()).collect();
        let yb: Vec<u32> = y.iter().map(|v| v.to_bits()).collect();
        assert_eq!(xb, yb, "step {i} diverged between identical runs");
    }
    assert!(
        a.iter().any(|v| v.iter().any(|x| *x != 0.0)),
        "all-zero output"
    );
    eprintln!("deterministic over {} steps, hidden={hidden}", a.len());
}
