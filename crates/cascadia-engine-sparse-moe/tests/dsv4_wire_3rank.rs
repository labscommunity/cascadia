//! 3-rank chain over REAL loopback transport WITH a mid-relay stage — the
//! exact topology the 4-node run uses that the 2-rank test skipped
//! (rank0 -> [rank1 mid-relay] -> rank2 last). Greedy must match reference.
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

#[tokio::test]
async fn dsv4_three_rank_mid_relay_matches_reference() {
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
    // tiny has 4 layers: rank0=[0,2) first, rank1=[2,3) MID, rank2=[3,4) last
    let mut r0 = Dsv4Runner::load_staged(&dir, 64, 0, 3, 0, 2).unwrap();
    let mut r1 = Dsv4Runner::load_staged(&dir, 64, 1, 3, 2, 3).unwrap();
    let mut r2 = Dsv4Runner::load_staged(&dir, 64, 2, 3, 3, 4).unwrap();
    r0.reset();
    r1.reset();
    r2.reset();
    let h = r0.hidden_size() as u32;
    let (s01, c01) = pair().await; // rank0 -> rank1
    let (s12, c12) = pair().await; // rank1 -> rank2
    let cfg = SamplingConfig::default();

    async fn drive(
        r0: &mut Dsv4Runner,
        r1: &mut Dsv4Runner,
        r2: &mut Dsv4Runner,
        tok: u32,
        pos: usize,
        h: u32,
        cfg: &SamplingConfig,
        c01: &Arc<Mutex<ActivationClient>>,
        s01: &Arc<Mutex<ActivationServer>>,
        c12: &Arc<Mutex<ActivationClient>>,
        s12: &Arc<Mutex<ActivationServer>>,
    ) -> u32 {
        // rank0: embed+layers -> send to rank1
        let hid = r0.forward_layers(r0.embed_token(tok), pos, Some(tok));
        let c01b = c01.clone();
        let cfgb = cfg.clone();
        let st = tokio::spawn(async move {
            send_forward(&c01b, pos as u32, &cfgb, &hid, [1, 1, h])
                .await
                .unwrap();
        });
        // rank1 MID: recv, forward, send to rank2, recv token, relay up
        assert_eq!(
            recv_kind_server(s01).await.unwrap(),
            Some(FrameKind::Forward)
        );
        let (_p, _c, hw, _s) = recv_forward_body_server(s01).await.unwrap();
        st.await.unwrap();
        let hmid = r1.forward_layers(hw, pos, None);
        let c12b = c12.clone();
        let cfgb = cfg.clone();
        let st2 = tokio::spawn(async move {
            send_forward(&c12b, pos as u32, &cfgb, &hmid, [1, 1, h])
                .await
                .unwrap();
        });
        // rank2 LAST: recv, forward, head, argmax
        assert_eq!(
            recv_kind_server(s12).await.unwrap(),
            Some(FrameKind::Forward)
        );
        let (_p2, _c2, hw2, _s2) = recv_forward_body_server(s12).await.unwrap();
        st2.await.unwrap();
        let hlast = r2.forward_layers(hw2, pos, None);
        argmax(&r2.head_logits(&hlast))
    }
    let mut pos = 0;
    let mut next = 0u32;
    for &t in &prompt {
        next = drive(
            &mut r0, &mut r1, &mut r2, t, pos, h, &cfg, &c01, &s01, &c12, &s12,
        )
        .await;
        pos += 1;
    }
    let mut got = vec![next];
    for _ in 1..want.len() {
        next = drive(
            &mut r0, &mut r1, &mut r2, next, pos, h, &cfg, &c01, &s01, &c12, &s12,
        )
        .await;
        pos += 1;
        got.push(next);
    }
    eprintln!("reference: {want:?}\n3-rank:    {got:?}");
    assert_eq!(got, want, "3-rank mid-relay over transport diverges");
}
