//! GLM-5.2 M-rank pipeline parity over the REAL loopback transport, for
//! M ∈ {1,2,3,4}. Unlike `glm5_wire` (fixed 2 ranks), this drives a chain with
//! **middle-relay ranks** (rank 0 → middle → … → last) — the exact topology a
//! 4-AI-PC run uses, where a middle rank does recv-upstream → `forward_layers`
//! → send-downstream. Greedy output of every split must equal the
//! single-process `GlmModel`. This is the end-to-end integration dry-run:
//! `load_staged` × N + the verbatim dist wire + sampling, on real code paths,
//! before any hardware.
//!
//! Requires the 8-layer fixture (splits evenly for M=1,2,4; M=3 → 3/3/2):
//!   python3 -c "import sys;sys.path.insert(0,'tools');from pathlib import Path;\
//!     from export_glm5 import export_tiny; \
//!     export_tiny(Path('crates/cascadia-engine-sparse-moe/tests/fixtures/glm5_export_ml'), num_layers=8)"

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

/// First-maximum argmax (ties → lowest index), matching `GlmModel`/`torch.argmax`.
fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// One greedy decode step at `pos`, feeding `tok` into rank 0 and returning the
/// last rank's argmax. Each of the `M-1` links carries the hidden state one hop
/// over a real socket; every rank runs `forward_layers` exactly once (keeping
/// its internal KV position in lockstep).
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
        let hsend = h; // move; reassigned from the relayed hidden below
        let send_task = tokio::spawn(async move {
            send_forward(&client, pos as u32, &cfg2, &hsend, [1, 1, hsz])
                .await
                .unwrap();
        });
        let k = recv_kind_server(&servers[i]).await.unwrap();
        assert_eq!(
            k,
            Some(FrameKind::Forward),
            "relay {i}: expected Forward frame"
        );
        let (_p, _c, hw, _s) = recv_forward_body_server(&servers[i]).await.unwrap();
        send_task.await.unwrap();
        h = ranks[i + 1].forward_layers(hw, pos, None);
    }
    argmax(&ranks[m - 1].head_logits(&h))
}

/// Build an `m`-rank pipeline from `dir` and greedily generate `n_gen` tokens
/// after `prompt`, chaining hidden state over `m-1` loopback sockets.
async fn run_pipeline(
    dir: &Path,
    max_seq: usize,
    m: usize,
    prompt: &[u32],
    n_gen: usize,
) -> Vec<u32> {
    let mut ranks: Vec<GlmRunner> = (0..m)
        .map(|r| {
            GlmRunner::load_staged(dir, max_seq, r as u32, m as u32, 0, 0, Default::default())
                .unwrap_or_else(|e| panic!("load rank {r}/{m}: {e}"))
        })
        .collect();
    for r in &mut ranks {
        r.reset();
    }
    let hsz = ranks[0].hidden_size() as u32;

    // link[i]: rank i (client, downstream) → rank i+1 (server, upstream).
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
    let mut pos = 0usize;
    let mut next = 0u32;
    for &t in prompt {
        next = step(&mut ranks, &clients, &servers, t, pos, hsz, &cfg).await;
        pos += 1;
    }
    let mut got = vec![next];
    for _ in 1..n_gen {
        next = step(&mut ranks, &clients, &servers, next, pos, hsz, &cfg).await;
        pos += 1;
        got.push(next);
    }
    got
}

#[tokio::test]
async fn glm5_pipeline_matches_reference_for_1_2_3_4_ranks() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export_ml");
    let prompt: Vec<u32> = vec![1, 2, 3, 4];
    let n_gen = 4usize;
    let max_seq = 32usize;

    // Single-process greedy is the ground truth (self-consistent — no hardcoded
    // token vector to drift from the fixture).
    let want = {
        let mut model = load_model(&dir, max_seq).expect("load_model reference");
        model.generate(&prompt, n_gen)
    };

    for m in [1usize, 2, 3, 4] {
        let got = run_pipeline(&dir, max_seq, m, &prompt, n_gen).await;
        eprintln!("M={m}: {got:?}  (reference {want:?})");
        assert_eq!(
            got, want,
            "M={m}-rank pipeline diverges from single-process reference"
        );
    }
}
