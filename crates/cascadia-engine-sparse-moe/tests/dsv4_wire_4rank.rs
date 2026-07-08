//! 4-rank chain over REAL loopback transport with TWO consecutive mid-relays —
//! the exact shape the real 4-node run uses (rank1 AND rank2 both mid) that the
//! 2/3-rank tests never cover. tiny has 4 layers (hash only on layer 0), so a
//! 4-way split is one layer per rank: r0=layer0(hash), r1=layer1(indexer,mid),
//! r2=layer2(compressor,mid), r3=layer3(indexer,last). Greedy must match the
//! single-process reference; divergence localizes any node garbage to the
//! sharded pipeline itself (fresh weights → not shard corruption).
use cascadia_engine_sparse_moe::dist::{
    recv_forward_body_server, recv_kind_server, send_forward, FrameKind,
};
use cascadia_engine_sparse_moe::dsv4::stage::Dsv4Runner;
use cascadia_engine_sparse_moe::SamplingConfig;
use cascadia_transport::{ActivationClient, ActivationServer};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
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

// One mid relay: recv Forward from upstream server, run my layers, send Forward
// to downstream client.
async fn relay_forward(
    r: &mut Dsv4Runner,
    pos: usize,
    h: u32,
    cfg: &SamplingConfig,
    up: &Arc<Mutex<ActivationServer>>,
    down: &Arc<Mutex<ActivationClient>>,
) {
    assert_eq!(recv_kind_server(up).await.unwrap(), Some(FrameKind::Forward));
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

#[allow(clippy::too_many_arguments)]
async fn drive(
    r0: &mut Dsv4Runner,
    r1: &mut Dsv4Runner,
    r2: &mut Dsv4Runner,
    r3: &mut Dsv4Runner,
    tok: u32,
    pos: usize,
    h: u32,
    cfg: &SamplingConfig,
    c01: &Arc<Mutex<ActivationClient>>,
    s01: &Arc<Mutex<ActivationServer>>,
    c12: &Arc<Mutex<ActivationClient>>,
    s12: &Arc<Mutex<ActivationServer>>,
    c23: &Arc<Mutex<ActivationClient>>,
    s23: &Arc<Mutex<ActivationServer>>,
) -> u32 {
    // rank0: embed + layers -> send to rank1
    let hid = r0.forward_layers(r0.embed_token(tok), pos, Some(tok));
    let c01b = c01.clone();
    let cfgb = cfg.clone();
    let st = tokio::spawn(async move {
        send_forward(&c01b, pos as u32, &cfgb, &hid, [1, 1, h])
            .await
            .unwrap();
    });
    // rank1 MID: recv from r0, forward, send to r2
    relay_forward(r1, pos, h, cfg, s01, c12).await;
    st.await.unwrap();
    // rank2 MID: recv from r1, forward, send to r3
    relay_forward(r2, pos, h, cfg, s12, c23).await;
    // rank3 LAST: recv from r2, forward, head, argmax
    assert_eq!(
        recv_kind_server(s23).await.unwrap(),
        Some(FrameKind::Forward)
    );
    let (_p, _c, hw, _s) = recv_forward_body_server(s23).await.unwrap();
    let hlast = r3.forward_layers(hw, pos, None);
    argmax(&r3.head_logits(&hlast))
}

#[tokio::test]
async fn dsv4_four_rank_two_mid_relays_matches_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_export");
    let r: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
            .unwrap();
    let prompt: Vec<u32> = r["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let want: Vec<u32> = r["generated"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    // tiny has 4 layers: rank r owns exactly layer r (even split of 4 over 4).
    let mut r0 = Dsv4Runner::load_staged(&dir, 64, 0, 4, 0, 1).unwrap();
    let mut r1 = Dsv4Runner::load_staged(&dir, 64, 1, 4, 1, 2).unwrap();
    let mut r2 = Dsv4Runner::load_staged(&dir, 64, 2, 4, 2, 3).unwrap();
    let mut r3 = Dsv4Runner::load_staged(&dir, 64, 3, 4, 3, 4).unwrap();
    r0.reset();
    r1.reset();
    r2.reset();
    r3.reset();
    let h = r0.hidden_size() as u32;
    let (s01, c01) = pair().await;
    let (s12, c12) = pair().await;
    let (s23, c23) = pair().await;
    let cfg = SamplingConfig::default();

    let mut pos = 0;
    let mut next = 0u32;
    for &t in &prompt {
        next = drive(
            &mut r0, &mut r1, &mut r2, &mut r3, t, pos, h, &cfg, &c01, &s01, &c12, &s12, &c23, &s23,
        )
        .await;
        pos += 1;
    }
    let mut got = vec![next];
    for _ in 1..want.len() {
        next = drive(
            &mut r0, &mut r1, &mut r2, &mut r3, next, pos, h, &cfg, &c01, &s01, &c12, &s12, &c23,
            &s23,
        )
        .await;
        pos += 1;
        got.push(next);
    }
    eprintln!("reference: {want:?}\n4-rank:    {got:?}");
    assert_eq!(got, want, "4-rank two-mid-relay over transport diverges");
}
