//! Integration tests for the KV migration wire frame.
//!
//! Exercises `FrameKind::KvMigration` on a real in-process
//! tahoma-transport socket pair, with no OpenVINO + no model artifacts.
//! Validates that the byte format on the wire round-trips between two
//! ranks without an actual Engine in between.
//!
//! See `docs/architecture/kv-migration.md` for the full design.

use std::sync::Arc;

use tahoma_engine_sparse_moe::dist::{
    recv_kind_client, recv_kind_server, recv_kv_migration_body_client,
    recv_kv_migration_body_server, send_kv_migration, FrameKind, KV_MIGRATION_HEADER_BYTES,
};
use tahoma_transport::{ActivationClient, ActivationServer};
use tokio::sync::Mutex;

async fn make_pair() -> (Arc<Mutex<ActivationServer>>, Arc<Mutex<ActivationClient>>) {
    let mut server = ActivationServer::new("127.0.0.1", 0);
    server.start().await.expect("server.start");
    let port = server.port();
    let server = Arc::new(Mutex::new(server));
    let server_clone = server.clone();
    let server_task = tokio::spawn(async move {
        server_clone
            .lock()
            .await
            .accept()
            .await
            .expect("server.accept");
    });
    let mut client = ActivationClient::new("127.0.0.1", port);
    client
        .connect_with_timeout(std::time::Duration::from_secs(5))
        .await
        .expect("client.connect");
    let client = Arc::new(Mutex::new(client));
    server_task.await.expect("server task panicked");
    (server, client)
}

/// Build a deterministic KV body for a single layer: `K then V`, f32 LE.
fn kv_body(num_heads: u32, past_seq_len: u32, qk_dim: u32, v_dim: u32) -> Vec<u8> {
    let nh = num_heads as usize;
    let ps = past_seq_len as usize;
    let qk = qk_dim as usize;
    let vd = v_dim as usize;
    let mut out = Vec::with_capacity(nh * ps * (qk + vd) * 4);
    // K block — values pattern: 0.001 * (h * 1000 + s * 10 + d).
    for h in 0..nh {
        for s in 0..ps {
            for d in 0..qk {
                let v = 0.001f32 * ((h * 1000 + s * 10 + d) as f32);
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    // V block — pattern: -0.001 * (h * 1000 + s * 10 + d).
    for h in 0..nh {
        for s in 0..ps {
            for d in 0..vd {
                let v = -0.001f32 * ((h * 1000 + s * 10 + d) as f32);
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    out
}

#[tokio::test]
async fn kv_migration_frame_round_trips_tiny_layer() {
    // Tiny shape so the test runs fast and the produced byte count is
    // small. Production K2.6 uses 64 heads / 192 qk_dim / 128 v_dim.
    let (server, client) = make_pair().await;
    let num_heads = 4u32;
    let past_seq_len = 3u32;
    let qk_head_dim = 8u32;
    let v_head_dim = 6u32;
    let body = kv_body(num_heads, past_seq_len, qk_head_dim, v_head_dim);
    let body_for_send = body.clone();

    let send_task = tokio::spawn(async move {
        send_kv_migration(
            &client,
            42,
            past_seq_len,
            num_heads,
            qk_head_dim,
            v_head_dim,
            &body_for_send,
        )
        .await
        .expect("send_kv_migration");
    });

    let kind = recv_kind_server(&server).await.expect("recv_kind");
    assert_eq!(kind, Some(FrameKind::KvMigration));
    let layer = recv_kv_migration_body_server(&server)
        .await
        .expect("recv_kv_migration_body_server");
    send_task.await.expect("send task");

    assert_eq!(layer.lid, 42);
    assert_eq!(layer.past_seq_len, past_seq_len);
    assert_eq!(layer.num_heads, num_heads);
    assert_eq!(layer.qk_head_dim, qk_head_dim);
    assert_eq!(layer.v_head_dim, v_head_dim);
    assert_eq!(layer.body_bytes, body);

    // Sanity: into_install_slab prepends a header sized matching the
    // KV_MIGRATION_HEADER_BYTES contract.
    let slab = layer.into_install_slab();
    assert_eq!(slab.len(), KV_MIGRATION_HEADER_BYTES + body.len());
}

#[tokio::test]
async fn kv_migration_frame_round_trips_zero_past_seq_len() {
    // Zero past_seq_len = brand-new session being shipped (no decoded
    // tokens yet). Verifies the body validator handles a zero-length
    // payload — `recv_tensor` accepts a `shape=[1,1,0]` I8 tensor with
    // 0 data bytes.
    let (server, client) = make_pair().await;
    let num_heads = 4u32;
    let past_seq_len = 0u32;
    let qk_head_dim = 8u32;
    let v_head_dim = 6u32;
    let body: Vec<u8> = Vec::new();
    let body_for_send = body.clone();

    let send_task = tokio::spawn(async move {
        send_kv_migration(
            &client,
            1,
            past_seq_len,
            num_heads,
            qk_head_dim,
            v_head_dim,
            &body_for_send,
        )
        .await
        .expect("send_kv_migration");
    });

    let kind = recv_kind_server(&server).await.expect("recv_kind");
    assert_eq!(kind, Some(FrameKind::KvMigration));
    let layer = recv_kv_migration_body_server(&server)
        .await
        .expect("recv_kv_migration_body_server");
    send_task.await.expect("send task");

    assert_eq!(layer.past_seq_len, 0);
    assert!(layer.body_bytes.is_empty());
}

#[tokio::test]
async fn kv_migration_two_layers_in_sequence() {
    // The protocol ships one layer per frame. Two consecutive frames
    // must round-trip without state leaking between them.
    let (server, client) = make_pair().await;
    let num_heads = 2u32;
    let past = 2u32;
    let qk = 4u32;
    let vd = 4u32;
    let body_a = kv_body(num_heads, past, qk, vd);
    let body_b = kv_body(num_heads, past, qk, vd);
    let body_a_for_send = body_a.clone();
    let body_b_for_send = body_b.clone();

    let send_task = tokio::spawn(async move {
        send_kv_migration(&client, 10, past, num_heads, qk, vd, &body_a_for_send)
            .await
            .unwrap();
        send_kv_migration(&client, 11, past, num_heads, qk, vd, &body_b_for_send)
            .await
            .unwrap();
    });

    // Layer 10.
    assert_eq!(
        recv_kind_server(&server).await.unwrap(),
        Some(FrameKind::KvMigration)
    );
    let layer_a = recv_kv_migration_body_server(&server).await.unwrap();
    assert_eq!(layer_a.lid, 10);
    assert_eq!(layer_a.body_bytes, body_a);

    // Layer 11.
    assert_eq!(
        recv_kind_server(&server).await.unwrap(),
        Some(FrameKind::KvMigration)
    );
    let layer_b = recv_kv_migration_body_server(&server).await.unwrap();
    assert_eq!(layer_b.lid, 11);
    assert_eq!(layer_b.body_bytes, body_b);

    send_task.await.unwrap();
}

#[tokio::test]
async fn kv_migration_rejects_wrong_body_length() {
    // The receiver should refuse a body whose declared past_seq_len
    // disagrees with the actual tensor byte count. Demonstrates the
    // shape-vs-bytes invariant we get for free from `recv_tensor` and
    // re-check in `validate_kv_body`.
    let (server, client) = make_pair().await;
    let num_heads = 4u32;
    let past_seq_len = 3u32;
    let qk_head_dim = 8u32;
    let v_head_dim = 6u32;
    // Build a body sized to a SMALLER past_seq_len than declared.
    let body = kv_body(num_heads, past_seq_len - 1, qk_head_dim, v_head_dim);

    let send_task = tokio::spawn(async move {
        // send_kv_migration will catch this BEFORE even putting bytes on
        // the wire — it cross-checks `kv_body.len()` against the
        // declared shape and returns an Io error.
        let err = send_kv_migration(
            &client,
            7,
            past_seq_len,
            num_heads,
            qk_head_dim,
            v_head_dim,
            &body,
        )
        .await
        .expect_err("expected length mismatch");
        let msg = format!("{err}");
        assert!(msg.contains("body length"), "got: {msg}");
    });
    send_task.await.unwrap();
    // Server-side recv never gets driven — the sender bailed before
    // putting bytes on the wire. Drain the connection to satisfy the
    // server's listening half.
    drop(server);
}

#[tokio::test]
async fn kv_migration_client_side_recv() {
    // Symmetric path: ranks can ship migration BOTH directions. Verify
    // recv_kv_migration_body_client works when the LAST rank pushes
    // upstream-bound (via its server-side socket — same as Token).
    let (server, client) = make_pair().await;
    let num_heads = 2u32;
    let past = 2u32;
    let qk = 4u32;
    let vd = 4u32;
    let body = kv_body(num_heads, past, qk, vd);
    let body_for_send = body.clone();

    // Sender owns the server-side socket (last rank); receiver owns
    // the client-side socket (upstream). Same direction Token uses.
    let send_task = tokio::spawn(async move {
        // Reuse send_kv_migration: under the hood it just writes a
        // FrameKind + header + I8 tensor; the *direction* is whichever
        // socket the caller holds.
        let server_as_client = server;
        let mut header = [0u8; 4 + KV_MIGRATION_HEADER_BYTES];
        header[0..4].copy_from_slice(&(FrameKind::KvMigration as u32).to_be_bytes());
        let mut hdr = [0u8; KV_MIGRATION_HEADER_BYTES];
        hdr[0..4].copy_from_slice(&77u32.to_be_bytes());
        hdr[4..8].copy_from_slice(&past.to_be_bytes());
        hdr[8..12].copy_from_slice(&num_heads.to_be_bytes());
        hdr[12..16].copy_from_slice(&qk.to_be_bytes());
        hdr[16..20].copy_from_slice(&vd.to_be_bytes());
        header[4..].copy_from_slice(&hdr);
        let mut guard = server_as_client.lock().await;
        guard.send_raw(&header).await.unwrap();
        let tensor = tahoma_transport::Tensor::new(
            tahoma_transport::DType::I8,
            [1, 1, body_for_send.len() as u32],
            body_for_send,
        );
        guard.send(&tensor).await.unwrap();
    });

    let kind = recv_kind_client(&client).await.unwrap();
    assert_eq!(kind, Some(FrameKind::KvMigration));
    let layer = recv_kv_migration_body_client(&client).await.unwrap();
    send_task.await.unwrap();

    assert_eq!(layer.lid, 77);
    assert_eq!(layer.body_bytes, body);
}
