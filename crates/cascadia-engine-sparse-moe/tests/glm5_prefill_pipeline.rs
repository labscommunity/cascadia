//! Batched-prefill across the pipeline over the REAL loopback transport, for
//! M ∈ {1,2,4} ranks (M=4 exercises middle-relay ranks). The whole prompt is
//! pushed through each rank's `forward_layers_batch` (per-position attention +
//! batch-union MoE) in one shot and relayed downstream as a batch frame; the
//! last rank's head logits at the final position must equal the single-process
//! `GlmModel::prefill`. This proves the distributed prefill carries batch-union
//! correctly — the throughput path a real N-node run uses.
//!
//! Requires the 8-layer fixture (`glm5_export_ml`); see glm5_pipeline_ml.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use cascadia_engine_sparse_moe::dist::{
    recv_forward_batch_body_server, recv_kind_server, send_forward_batch, FrameKind,
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

/// Push `prompt` through an `m`-rank pipeline as a single batched-prefill and
/// return the last rank's head logits at the final prompt position.
async fn pipeline_prefill(dir: &Path, max_seq: usize, m: usize, prompt: &[u32]) -> Vec<f32> {
    let rows = prompt.len();
    let mut ranks: Vec<GlmRunner> = (0..m)
        .map(|r| GlmRunner::load_staged(dir, max_seq, r as u32, m as u32, 0, 0).expect("load rank"))
        .collect();
    for r in &mut ranks {
        r.reset();
    }
    let hsz = ranks[0].hidden_size() as u32;

    // m-1 loopback links: rank i (client) -> rank i+1 (server).
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
        client.connect_with_timeout(Duration::from_secs(5)).await.unwrap();
        atask.await.unwrap();
        clients.push(Arc::new(Mutex::new(client)));
        servers.push(server);
    }

    // rank 0: embed the whole prompt, run its layer slice as one batch.
    let mut batch = vec![0.0f32; rows * hsz as usize];
    for (r, &t) in prompt.iter().enumerate() {
        batch[r * hsz as usize..(r + 1) * hsz as usize].copy_from_slice(&ranks[0].embed_token(t));
    }
    let mut h = ranks[0].forward_layers_batch(batch, 0, rows);

    // relay the batch hop by hop; each rank runs its slice batched.
    let cfg = SamplingConfig::default();
    for i in 0..m - 1 {
        let client = clients[i].clone();
        let cfg2 = cfg.clone();
        let hsend = h;
        let rows32 = rows as u32;
        let send = tokio::spawn(async move {
            send_forward_batch(&client, 0, rows32, &cfg2, &hsend, [1, rows32, hsz])
                .await
                .unwrap();
        });
        let k = recv_kind_server(&servers[i]).await.unwrap();
        assert_eq!(k, Some(FrameKind::ForwardBatch), "relay {i}: expected ForwardBatch");
        let (_start, count, _s, hw, _shape) =
            recv_forward_batch_body_server(&servers[i]).await.unwrap();
        assert_eq!(count as usize, rows);
        send.await.unwrap();
        h = ranks[i + 1].forward_layers_batch(hw, 0, rows);
    }

    // last rank: head logits at the final position.
    let last = (rows - 1) * hsz as usize;
    ranks[m - 1].head_logits(&h[last..last + hsz as usize])
}

#[tokio::test]
async fn glm5_batched_prefill_pipeline_matches_single_process() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export_ml");
    if !dir.join("manifest.json").exists() {
        eprintln!("SKIP: glm5_export_ml absent (run tools/glm5_ref/gen_fixtures.py)");
        return;
    }
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let max_seq = 32usize;

    // Single-process batched prefill is the reference (itself bit-exact vs
    // per-token, proven by glm5_model).
    let want = load_model(&dir, max_seq).expect("load_model").prefill(&prompt);
    let want_tok = argmax(&want);

    for m in [1usize, 2, 4] {
        let got = pipeline_prefill(&dir, max_seq, m, &prompt).await;
        assert_eq!(got.len(), want.len(), "M={m}: logit length mismatch");
        assert_eq!(
            got, want,
            "M={m}: pipeline batched prefill diverged from single-process prefill"
        );
        assert_eq!(argmax(&got), want_tok, "M={m}: first-token argmax mismatch");
    }
}
