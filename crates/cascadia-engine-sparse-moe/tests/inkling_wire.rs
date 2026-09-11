//! Inkling 2-rank pipeline over a REAL loopback cascadia-transport (the exact
//! send_forward/recv_forward path the distributed run uses). The greedy stream
//! from the split stages must match the single-process reference — proving the
//! N-rank layer split + the (verbatim-reused) dist wire are correct for inkling.
//!
//! Requires the tiny export (run tools/export_inkling.py --tiny ...); skips if absent.

use std::path::PathBuf;
use std::sync::Arc;

use cascadia_engine_sparse_moe::dist::{
    recv_forward_body_server, recv_kind_server, send_forward, FrameKind,
};
use cascadia_engine_sparse_moe::inkling::stage::InklingRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;
use cascadia_engine_sparse_moe::SamplingConfig;
use cascadia_transport::{ActivationClient, ActivationServer};
use tokio::sync::Mutex;

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

#[tokio::test]
async fn inkling_two_rank_over_real_transport_matches_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export");
    let refp = dir.join("reference.json");
    if !refp.exists() {
        eprintln!("inkling_export/reference.json missing; skipping");
        return;
    }
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refp).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    let prompt: Vec<u32> = ids("prompt_ids");
    let want: Vec<u32> = ids("greedy_ids"); // HF reference greedy on the exported weights

    let mut r0 = InklingRunner::load_staged(&dir, 64, 0, 2, 0, 0, Some("eager".into()), None)
        .expect("rank0");
    let mut r1 = InklingRunner::load_staged(&dir, 64, 1, 2, 0, 0, Some("eager".into()), None)
        .expect("rank1");
    r0.reset();
    r1.reset();
    let hsz = r0.hidden_size() as u32;

    // loopback: rank1 owns the server (upstream), rank0 owns the client (downstream)
    let mut server = ActivationServer::new("127.0.0.1", 0);
    server.start().await.unwrap();
    let port = server.port();
    let server = Arc::new(Mutex::new(server));
    let sc = server.clone();
    let atask = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });
    let mut client = ActivationClient::new("127.0.0.1", port);
    client
        .connect_with_timeout(std::time::Duration::from_secs(5))
        .await
        .unwrap();
    let client = Arc::new(Mutex::new(client));
    atask.await.unwrap();

    let cfg = SamplingConfig::default();
    async fn step(
        r0: &mut InklingRunner,
        r1: &mut InklingRunner,
        tok: u32,
        pos: usize,
        hsz: u32,
        cfg: &SamplingConfig,
        client: &Arc<Mutex<ActivationClient>>,
        server: &Arc<Mutex<ActivationServer>>,
    ) -> u32 {
        let h = r0.embed_token(tok);
        let h = r0.forward_layers(h, pos, None);
        let client2 = client.clone();
        let cfg2 = cfg.clone();
        let send_task = tokio::spawn(async move {
            send_forward(&client2, pos as u32, &cfg2, &h, [1, 1, hsz], true)
                .await
                .unwrap();
        });
        let k = recv_kind_server(server).await.unwrap();
        assert_eq!(k, Some(FrameKind::Forward));
        let (_p, _c, _ph, hw, _s) = recv_forward_body_server(server).await.unwrap();
        send_task.await.unwrap();
        let hw = r1.forward_layers(hw, pos, None);
        argmax(&r1.head_logits(&hw))
    }

    let mut pos = 0usize;
    let mut next = 0u32;
    for &t in &prompt {
        next = step(&mut r0, &mut r1, t, pos, hsz, &cfg, &client, &server).await;
        pos += 1;
    }
    let mut got = vec![next];
    for _ in 1..want.len() {
        next = step(&mut r0, &mut r1, next, pos, hsz, &cfg, &client, &server).await;
        pos += 1;
        got.push(next);
    }
    eprintln!("reference: {want:?}\ninkling wire 2-rank: {got:?}");
    assert_eq!(
        got, want,
        "inkling 2-rank over real transport diverges from reference"
    );
}
