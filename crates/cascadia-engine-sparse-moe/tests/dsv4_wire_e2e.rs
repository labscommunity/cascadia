//! Decisive test for the 4-node bug: drive two Dsv4Runner stages over a REAL
//! loopback cascadia-transport (the exact send_forward/recv_forward path the
//! distributed run uses), and require the greedy stream to match the
//! single-process reference. If this passes, the engine+wire are correct and
//! the real-node garbage is a deployment issue; if it diverges, the wire path
//! is the bug.

use std::path::PathBuf;
use std::sync::Arc;

use cascadia_engine_sparse_moe::dist::{
    recv_forward_batch_body_server, recv_forward_body_server, recv_kind_server, send_forward,
    send_forward_batch_prefill, FrameKind,
};
use cascadia_engine_sparse_moe::dsv4::stage::Dsv4Runner;
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

/// One per-token pipeline step over the wire: rank0 embed+layers -> send Forward
/// -> rank1 recv+layers+head -> argmax. Shared by the per-token and batched tests
/// (the batched test uses this for decode, after a one-shot batched prefill).
#[allow(clippy::too_many_arguments)]
async fn step(
    r0: &mut Dsv4Runner,
    r1: &mut Dsv4Runner,
    tok: u32,
    pos: usize,
    hsz: u32,
    cfg: &SamplingConfig,
    client: &Arc<Mutex<ActivationClient>>,
    server: &Arc<Mutex<ActivationServer>>,
) -> u32 {
    let h = r0.embed_token(tok);
    let h = r0.forward_layers(h, pos, Some(tok));
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

#[tokio::test]
async fn dsv4_two_rank_over_real_transport_matches_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_export");
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
            .unwrap();
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let want: Vec<u32> = reference["generated"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let mut r0 = Dsv4Runner::load_staged(&dir, 64, 0, 2, 0, 0).expect("rank0");
    let mut r1 = Dsv4Runner::load_staged(&dir, 64, 1, 2, 0, 0).expect("rank1");
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
    eprintln!("reference: {want:?}\nwire 2-rank: {got:?}");
    assert_eq!(
        got, want,
        "dsv4 2-rank over real transport diverges from reference"
    );
}

/// Same 2-rank loopback, but prefill is driven as ONE `ForwardBatchPrefill`
/// frame (the whole prompt in a single batch, MoE dedups overlapping experts)
/// instead of per-token — rank1 runs its layers batched and samples only the
/// last row (the first generated token). Decode then continues per-token. The
/// greedy stream must still match the single-process reference, proving the
/// batched-prefill wire path is bit-exact with the per-token path.
#[tokio::test]
async fn dsv4_two_rank_batched_prefill_matches_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_export");
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
            .unwrap();
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let want: Vec<u32> = reference["generated"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert!(
        prompt.len() >= 2,
        "batched-prefill test needs a multi-token prompt"
    );

    let mut r0 = Dsv4Runner::load_staged(&dir, 64, 0, 2, 0, 0).expect("rank0");
    let mut r1 = Dsv4Runner::load_staged(&dir, 64, 1, 2, 0, 0).expect("rank1");
    r0.reset();
    r1.reset();
    let hsz = r0.hidden_size() as u32;

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

    // --- batched prefill: one ForwardBatchPrefill for the whole prompt ---
    let h_batch = r0.prefill_batch_first(&prompt);
    let rows = prompt.len() as u32;
    let client2 = client.clone();
    let cfg2 = cfg.clone();
    let send_task = tokio::spawn(async move {
        send_forward_batch_prefill(&client2, 0, rows, &cfg2, &h_batch, [1, rows, hsz])
            .await
            .unwrap();
    });
    let k = recv_kind_server(&server).await.unwrap();
    assert_eq!(k, Some(FrameKind::ForwardBatchPrefill));
    let (_p, bc, _c, hw, _s) = recv_forward_batch_body_server(&server).await.unwrap();
    send_task.await.unwrap();
    let out = r1.forward_layers_prefill_batch(hw, bc as usize);
    let w = r1.hidden_size();
    let last = &out[(bc as usize - 1) * w..(bc as usize) * w];
    let mut next = argmax(&r1.head_logits(last));

    // --- decode continues per-token from pos = prompt.len() ---
    let mut got = vec![next];
    let mut pos = prompt.len();
    for _ in 1..want.len() {
        next = step(&mut r0, &mut r1, next, pos, hsz, &cfg, &client, &server).await;
        pos += 1;
        got.push(next);
    }
    eprintln!("reference: {want:?}\nbatched-prefill 2-rank: {got:?}");
    assert_eq!(
        got, want,
        "dsv4 2-rank batched prefill diverges from reference"
    );
}
