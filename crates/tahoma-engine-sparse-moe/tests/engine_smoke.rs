//! Integration smoke tests that exercise the Engine + Builder trait
//! surface without an OpenVINO toolchain. Catches regressions in the
//! manifest, builder wiring, and tensor utilities that the per-module
//! unit tests don't see.

use std::fs;
use std::path::PathBuf;

use tahoma_engine::{Builder, EngineError};
use tahoma_engine_sparse_moe::{
    Manifest, SamplingConfig, SparseMoEBuilder, SparseMoEBuilderConfig,
};
use tahoma_types::{PeerLayout, ShardSpec};

fn write_manifest(dir: &PathBuf) {
    let m = serde_json::json!({
        "arch": "kimi_k2.6",
        "num_layers": 61,
        "dense_layers": [0],
        "num_experts": 384,
        "top_k": 8,
        "hidden_size": 7168,
        "num_kv_heads": 64,
        "qk_head_dim": 192,
        "v_head_dim": 128,
        "vocab_size": 163840,
        "eos_token_ids": [163585, 163586],
    });
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("manifest.json"), m.to_string()).unwrap();
}

#[tokio::test]
async fn builder_rejects_multi_stage_peers() {
    let dir = tempfile::tempdir().unwrap();
    write_manifest(&dir.path().to_path_buf());
    let mut b = SparseMoEBuilder::new(SparseMoEBuilderConfig::new(
        dir.path().to_str().unwrap(),
        "CPU",
    ));
    let peers = PeerLayout {
        upstream: Some(tahoma_types::PeerEndpoint {
            host: "x".into(),
            port: 1,
        }),
        downstream: None,
    };
    let err = b.connect(peers).await.unwrap_err();
    assert!(matches!(err, EngineError::PeerRejected(_)), "got {:?}", err);
}

#[tokio::test]
async fn builder_accepts_empty_peers() {
    let dir = tempfile::tempdir().unwrap();
    write_manifest(&dir.path().to_path_buf());
    let mut b = SparseMoEBuilder::new(SparseMoEBuilderConfig::new(
        dir.path().to_str().unwrap(),
        "CPU",
    ));
    let peers = PeerLayout {
        upstream: None,
        downstream: None,
    };
    b.connect(peers).await.expect("connect");
}

#[tokio::test]
async fn manifest_load_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    write_manifest(&dir.path().to_path_buf());
    let m = Manifest::load(dir.path()).expect("load manifest");
    assert_eq!(m.num_layers, 61);
    assert_eq!(m.num_experts, 384);
    assert_eq!(m.top_k, 8);
    assert_eq!(m.dense_layers, vec![0]);
    assert_eq!(m.moe_layer_ids().len(), 60);
    assert_eq!(m.moe_layer_ids()[0], 1);
}

#[tokio::test]
async fn load_returns_missing_file_for_empty_dir() {
    // No manifest → Manifest::load returns ManifestError::Missing,
    // surfaced via Builder::load as EngineError::Backend.
    let dir = tempfile::tempdir().unwrap();
    let mut b = SparseMoEBuilder::new(SparseMoEBuilderConfig::new(
        dir.path().to_str().unwrap(),
        "CPU",
    ));
    let r = b
        .load(ShardSpec {
            model_id: "test".into(),
            layer_start: 0,
            layer_end: 60,
            total_layers: 61,
            device: "CPU".into(),
            is_first_stage: true,
            is_last_stage: true,
            tp_size: 1,
            tp_rank: 0,
        })
        .await;
    assert!(r.is_err(), "expected load error on empty dir");
}

#[test]
fn sampling_config_default_is_greedy() {
    let cfg = SamplingConfig::default();
    assert_eq!(cfg.temperature, 0.0);
    assert_eq!(cfg.top_p, 1.0);
}
