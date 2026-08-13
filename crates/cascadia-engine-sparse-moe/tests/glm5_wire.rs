//! GLM-5.2 2-rank pipeline over a REAL loopback cascadia-transport (the exact
//! send_forward/recv_forward path the distributed run uses). The greedy stream
//! from the split stages must match the single-process reference — proving the
//! N-rank layer split + the (verbatim-reused) dist wire are correct for glm5.
//!
//! Requires the export fixture (run tools/glm5_ref/gen_fixtures.py).

use std::path::PathBuf;
use std::sync::Arc;

use cascadia_engine_sparse_moe::dist::{
    recv_forward_body_server, recv_kind_server, send_forward, FrameKind,
};
use cascadia_engine_sparse_moe::glm::stage::GlmRunner;
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
async fn glm5_two_rank_over_real_transport_matches_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export");
    let prompt: Vec<u32> = vec![1, 2, 3, 4];
    let want: Vec<u32> = vec![4, 10, 3, 15]; // loader/reference greedy

    let mut r0 = GlmRunner::load_staged(&dir, 32, 0, 2, 0, 0, Default::default()).expect("rank0");
    let mut r1 = GlmRunner::load_staged(&dir, 32, 1, 2, 0, 0, Default::default()).expect("rank1");
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
        r0: &mut GlmRunner,
        r1: &mut GlmRunner,
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
            send_forward(&client2, pos as u32, &cfg2, &h, [1, 1, hsz])
                .await
                .unwrap();
        });
        let k = recv_kind_server(server).await.unwrap();
        assert_eq!(k, Some(FrameKind::Forward));
        let (_p, _c, hw, _s) = recv_forward_body_server(server).await.unwrap();
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
    eprintln!("reference: {want:?}\nglm5 wire 2-rank: {got:?}");
    assert_eq!(
        got, want,
        "glm5 2-rank over real transport diverges from reference"
    );
}
