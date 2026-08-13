//! Distributed IndexShare: an 8-layer mixed-topology model (full at 0,1,2,6;
//! shared elsewhere), `index_topk=2`, decoded across M ranks over the REAL
//! loopback transport, must match single-process greedy at ctx > topk (where the
//! full layers actually prune and the shared layers reuse the carried top-k).
//!
//! The point: a naive even N=2 split boundary (layer 4) is a "shared" layer,
//! which would reset the carry across the rank boundary and diverge. Because
//! `load_staged` uses `index_aligned_split`, each rank instead starts on a full
//! layer, so the within-rank carry suffices — no cross-rank wire carry needed.
//!
//! Requires the fixture (run tools/glm5_ref/gen_fixtures.py).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use cascadia_engine_sparse_moe::dist::{
    recv_forward_body_server, recv_kind_server, send_forward, FrameKind,
};
use cascadia_engine_sparse_moe::glm::loader::load_model;
use cascadia_engine_sparse_moe::glm::stage::GlmRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;
use cascadia_engine_sparse_moe::SamplingConfig;
use cascadia_transport::{ActivationClient, ActivationServer};
use tokio::sync::Mutex;

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// M-rank greedy generation over M-1 loopback links (decode path).
async fn pipeline_generate(
    dir: &Path,
    max_seq: usize,
    m: usize,
    prompt: &[u32],
    n_gen: usize,
) -> Vec<u32> {
    let mut ranks: Vec<GlmRunner> = (0..m)
        .map(|r| {
            GlmRunner::load_staged(dir, max_seq, r as u32, m as u32, 0, 0, Default::default())
                .expect("load rank")
        })
        .collect();
    for r in &mut ranks {
        r.reset();
    }
    let hsz = ranks[0].hidden_size() as u32;

    let mut clients = Vec::new();
    let mut servers = Vec::new();
    for _ in 0..m.saturating_sub(1) {
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let server = Arc::new(Mutex::new(server));
        let sc = server.clone();
        let atask = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client
            .connect_with_timeout(Duration::from_secs(5))
            .await
            .unwrap();
        atask.await.unwrap();
        clients.push(Arc::new(Mutex::new(client)));
        servers.push(server);
    }

    let cfg = SamplingConfig::default();
    async fn step(
        ranks: &mut [GlmRunner],
        clients: &[Arc<Mutex<ActivationClient>>],
        servers: &[Arc<Mutex<ActivationServer>>],
        tok: u32,
        pos: usize,
        hsz: u32,
        cfg: &SamplingConfig,
    ) -> u32 {
        let m = ranks.len();
        let mut h = ranks[0].embed_token(tok);
        h = ranks[0].forward_layers(h, pos, None);
        for i in 0..m - 1 {
            let client = clients[i].clone();
            let cfg2 = cfg.clone();
            let hsend = h;
            let send = tokio::spawn(async move {
                send_forward(&client, pos as u32, &cfg2, &hsend, [1, 1, hsz])
                    .await
                    .unwrap();
            });
            let k = recv_kind_server(&servers[i]).await.unwrap();
            assert_eq!(k, Some(FrameKind::Forward));
            let (_p, _c, hw, _s) = recv_forward_body_server(&servers[i]).await.unwrap();
            send.await.unwrap();
            h = ranks[i + 1].forward_layers(hw, pos, None);
        }
        argmax(&ranks[m - 1].head_logits(&h))
    }

    let mut pos = 0usize;
    let mut next = 0u32;
    for &t in prompt {
        next = step(&mut ranks, &clients, &servers, t, pos, hsz, &cfg).await;
        pos += 1;
    }
    let mut out = vec![next];
    for _ in 1..n_gen {
        next = step(&mut ranks, &clients, &servers, next, pos, hsz, &cfg).await;
        pos += 1;
        out.push(next);
    }
    out
}

#[tokio::test]
async fn indexshare_pipeline_matches_single_process_beyond_topk() {
    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export_indexshare_ml");
    // prompt longer than index_topk(2) so full layers prune and shared reuse.
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let (n_gen, max_seq) = (4usize, 32usize);

    let want = load_model(&dir, max_seq)
        .expect("load_model")
        .generate(&prompt, n_gen);

    for m in [1usize, 2, 3] {
        let got = pipeline_generate(&dir, max_seq, m, &prompt, n_gen).await;
        assert_eq!(
            got, want,
            "M={m}: distributed IndexShare diverged from single-process (aligned split / carry bug)"
        );
    }
}
