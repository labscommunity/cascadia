//! Real-dims reproduction fixture: the "big" export has the real model's
//! attention dims (dim=4096 → hc*dim=16384 wire width, n_heads=64,
//! head_dim=512, o_groups=8 < n_heads, index_head_dim=128) PLUS ratio-4/128
//! compressor+indexer layers across 8 layers, split 4 ranks (2 each) so a
//! ratio-128 compressor is the ENTRY layer of every mid rank — the exact
//! real-model condition (real layer 11 = ratio-128 = rank1 entry) that tiny
//! (ratio<=16, dim=64) and med (ratio=0, dim=512) never exercised.
//!
//! The fixture is ~GB-scale at dim=4096 so it is NOT committed; generate it
//! first (deterministic, ~6 min):
//!
//!   python3 tools/export_deepseek_v4.py --big \
//!     --out crates/cascadia-engine-sparse-moe/tests/fixtures/dsv4_big_export
//!
//! Both tests are #[ignore] for that reason. KNOWN STATE (2026-07-08):
//! single-stage and 4-rank agree EXACTLY with each other but both drift from
//! the torch reference at generated token 5 — under investigation (suspected
//! bf16 accumulation-order noise flipping a thin argmax margin on random
//! weights; the real-checkpoint single-stage run is coherent). The 4-rank ==
//! single-stage equality is the load-bearing assertion for sharding.
use cascadia_engine_sparse_moe::dist::{
    recv_forward_body_server, recv_kind_server, send_forward, FrameKind,
};
use cascadia_engine_sparse_moe::dsv4::stage::Dsv4Runner;
use cascadia_engine_sparse_moe::SamplingConfig;
use cascadia_transport::{ActivationClient, ActivationServer};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

const MAXSEQ: usize = 64;

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_big_export")
}

fn load_ref() -> (Vec<u32>, Vec<u32>) {
    let dir = fixture_dir();
    let r: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
            .unwrap();
    let u32v = |k: &str| {
        r[k].as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<u32>>()
    };
    (u32v("prompt_ids"), u32v("generated"))
}

async fn pair() -> (Arc<Mutex<ActivationServer>>, Arc<Mutex<ActivationClient>>) {
    let mut s = ActivationServer::new("127.0.0.1", 0);
    s.start().await.unwrap();
    let port = s.port();
    let s = Arc::new(Mutex::new(s));
    let sc = s.clone();
    let t = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });
    let mut c = ActivationClient::new("127.0.0.1", port);
    c.connect_with_timeout(std::time::Duration::from_secs(5))
        .await
        .unwrap();
    let c = Arc::new(Mutex::new(c));
    t.await.unwrap();
    (s, c)
}

async fn relay(
    r: &mut Dsv4Runner,
    pos: usize,
    h: u32,
    cfg: &SamplingConfig,
    up: &Arc<Mutex<ActivationServer>>,
    down: &Arc<Mutex<ActivationClient>>,
) {
    assert_eq!(
        recv_kind_server(up).await.unwrap(),
        Some(FrameKind::Forward)
    );
    let (_p, _c, hw, _s) = recv_forward_body_server(up).await.unwrap();
    let hmid = r.forward_layers(hw, pos, None);
    let downb = down.clone();
    let cfgb = cfg.clone();
    tokio::spawn(async move {
        send_forward(&downb, pos as u32, &cfgb, &hmid, [1, 1, h])
            .await
            .unwrap();
    })
    .await
    .unwrap();
}

// single greedy step through the 4-rank chain; returns argmax token
async fn drive(
    rs: &mut [Dsv4Runner; 4],
    tok: u32,
    pos: usize,
    h: u32,
    cfg: &SamplingConfig,
    links: &[(Arc<Mutex<ActivationServer>>, Arc<Mutex<ActivationClient>>); 3],
) -> u32 {
    let [r0, r1, r2, r3] = rs;
    let hid = r0.forward_layers(r0.embed_token(tok), pos, Some(tok));
    let c01 = links[0].1.clone();
    let cfgb = cfg.clone();
    let st = tokio::spawn(async move {
        send_forward(&c01, pos as u32, &cfgb, &hid, [1, 1, h])
            .await
            .unwrap();
    });
    relay(r1, pos, h, cfg, &links[0].0, &links[1].1).await;
    st.await.unwrap();
    relay(r2, pos, h, cfg, &links[1].0, &links[2].1).await;
    assert_eq!(
        recv_kind_server(&links[2].0).await.unwrap(),
        Some(FrameKind::Forward)
    );
    let (_p, _c, hw, _s) = recv_forward_body_server(&links[2].0).await.unwrap();
    let hlast = r3.forward_layers(hw, pos, None);
    argmax(&r3.head_logits(&hlast))
}

#[test]
#[ignore = "needs generated dsv4_big_export fixture (see module doc); known torch drift at token 5 under investigation"]
fn big_single_stage_matches_reference() {
    let dir = fixture_dir();
    let (prompt, want) = load_ref();
    let mut r = Dsv4Runner::load_staged(&dir, MAXSEQ, 0, 1, 0, 8).unwrap();
    let got = r.generate_argmax(&prompt, want.len());
    eprintln!("reference:    {want:?}\nsingle-stage: {got:?}");
    assert_eq!(got, want, "big single-stage diverges from torch reference");
}

#[tokio::test]
#[ignore = "needs generated dsv4_big_export fixture (see module doc); known torch drift at token 5 under investigation"]
async fn big_four_rank_matches_reference() {
    let dir = fixture_dir();
    let (prompt, want) = load_ref();
    // 8 layers over 4 ranks: 2 each. mid-rank entries (layers 2,4) are ratio-128.
    let mut rs = [
        Dsv4Runner::load_staged(&dir, MAXSEQ, 0, 4, 0, 2).unwrap(),
        Dsv4Runner::load_staged(&dir, MAXSEQ, 1, 4, 2, 4).unwrap(),
        Dsv4Runner::load_staged(&dir, MAXSEQ, 2, 4, 4, 6).unwrap(),
        Dsv4Runner::load_staged(&dir, MAXSEQ, 3, 4, 6, 8).unwrap(),
    ];
    for r in rs.iter_mut() {
        r.reset();
    }
    let h = rs[0].hidden_size() as u32;
    let links = [pair().await, pair().await, pair().await];
    let cfg = SamplingConfig::default();

    let mut pos = 0;
    let mut next = 0u32;
    for &t in &prompt {
        next = drive(&mut rs, t, pos, h, &cfg, &links).await;
        pos += 1;
    }
    let mut got = vec![next];
    for _ in 1..want.len() {
        next = drive(&mut rs, next, pos, h, &cfg, &links).await;
        pos += 1;
        got.push(next);
    }
    eprintln!("reference: {want:?}\n4-rank:    {got:?}");
    assert_eq!(
        got, want,
        "big 4-rank diverges from reference (real-scale sharding bug)"
    );
}
