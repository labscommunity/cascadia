//! QUIC transport integration tests (feature `quic`).
//!
//! These are the real data-plane proof for the QUIC backend: they push
//! actual tensors + control bytes through `ActivationServer`/`Client`
//! pinned to [`TransportKind::Quic`] and assert byte-exact round-trips,
//! including the exact `send_raw`/`recv_raw` + tensor interleave the
//! dist-spec engines use on the wire.
//!
//! Run with:  cargo test -p cascadia-transport --features quic
//! (the idle-survival test is `#[ignore]`d for speed — see its note).
#![cfg(feature = "quic")]

use std::time::Duration;

use cascadia_transport::{ActivationClient, ActivationServer, DType, Tensor, TransportKind};

fn quic_server(port: u16) -> ActivationServer {
    ActivationServer::new_with_kind("127.0.0.1", port, TransportKind::Quic)
}
fn quic_client(port: u16) -> ActivationClient {
    ActivationClient::new_with_kind("127.0.0.1", port, TransportKind::Quic)
}

#[tokio::test]
async fn quic_roundtrip_f32_2d() {
    let mut server = quic_server(0);
    server.start().await.unwrap();
    let port = server.port();

    let h = tokio::spawn(async move {
        server.accept().await.unwrap();
        let (got, _) = server.recv().await.unwrap();
        got
    });

    let mut client = quic_client(port);
    client.connect().await.unwrap();
    let payload = vec![0u8, 0, 128, 63, 0, 0, 0, 64]; // f32: 1.0, 2.0
    let tensor = Tensor::from_2d(DType::F32, 1, 2, payload.clone());
    client.send(&tensor).await.unwrap();

    let got = h.await.unwrap();
    assert_eq!(got.dtype, DType::F32);
    assert_eq!(got.shape, [1, 1, 2]);
    assert_eq!(got.data, payload);
}

#[tokio::test]
async fn quic_roundtrip_i64_3d_large() {
    let mut server = quic_server(0);
    server.start().await.unwrap();
    let port = server.port();
    let h = tokio::spawn(async move {
        server.accept().await.unwrap();
        let (got, _) = server.recv().await.unwrap();
        got
    });
    let mut client = quic_client(port);
    client.connect().await.unwrap();
    // A few hundred KiB to cross multiple QUIC packets/stream frames.
    let payload: Vec<u8> = (0..8 * 16 * 512 * 8).map(|i| (i % 251) as u8).collect();
    let tensor = Tensor::new(DType::I64, [8, 16, 512], payload.clone());
    client.send(&tensor).await.unwrap();
    let got = h.await.unwrap();
    assert_eq!(got.shape, [8, 16, 512]);
    assert_eq!(got.data, payload);
}

/// Exercises the exact dist-spec frame choreography over one QUIC bidi
/// stream: client sends control bytes (`send_raw`) then two tensors, the
/// server replies with a control byte + a tensor. Proves `send_raw` /
/// `recv_raw` interleave correctly with tensor frames in BOTH directions.
#[tokio::test]
async fn quic_control_byte_interleave_bidirectional() {
    const FORWARD: u32 = 1;
    const LOGITS: u32 = 2;
    let mut server = quic_server(0);
    server.start().await.unwrap();
    let port = server.port();

    // The real relay keeps the connection open for the whole session (the
    // run_relay_loop), tearing it down only after the last frame is
    // consumed. QUIC — unlike TCP — drops in-flight stream data on an
    // abrupt connection close, so the server must not return (and drop the
    // connection) until the client has read the reply. This oneshot models
    // that session lifetime; without it the test would race the close.
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
    let h = tokio::spawn(async move {
        server.accept().await.unwrap();
        // FORWARD frame: kind, logical_pos_start, attn, hidden.
        let kind = u32::from_be_bytes(server.recv_raw(4).await.unwrap().try_into().unwrap());
        assert_eq!(kind, FORWARD);
        let pos = u32::from_be_bytes(server.recv_raw(4).await.unwrap().try_into().unwrap());
        let (attn, _) = server.recv().await.unwrap();
        let (hidden, _) = server.recv().await.unwrap();
        // Reply: LOGITS kind + a logits tensor.
        server.send_raw(&LOGITS.to_be_bytes()).await.unwrap();
        let logits = Tensor::from_2d(DType::F32, 1, 4, vec![1u8; 16]);
        server.send(&logits).await.unwrap();
        let _ = done_rx.await; // hold the connection open until client is done
        (pos, attn, hidden)
    });

    let mut client = quic_client(port);
    client.connect().await.unwrap();
    client.send_raw(&FORWARD.to_be_bytes()).await.unwrap();
    client.send_raw(&7u32.to_be_bytes()).await.unwrap();
    let attn = Tensor::new(DType::I64, [1, 1, 3], vec![1u8; 24]);
    let hidden = Tensor::new(DType::F16, [1, 2, 4], vec![2u8; 16]);
    client.send(&attn).await.unwrap();
    client.send(&hidden).await.unwrap();
    let reply_kind = u32::from_be_bytes(client.recv_raw(4).await.unwrap().try_into().unwrap());
    assert_eq!(reply_kind, LOGITS);
    let (logits, _) = client.recv().await.unwrap();

    let _ = done_tx.send(()); // client done — server may now tear down
    let (pos, got_attn, got_hidden) = h.await.unwrap();
    assert_eq!(pos, 7);
    assert_eq!(got_attn.shape, [1, 1, 3]);
    assert_eq!(got_attn.data, vec![1u8; 24]);
    assert_eq!(got_hidden.shape, [1, 2, 4]);
    assert_eq!(got_hidden.data, vec![2u8; 16]);
    assert_eq!(logits.shape, [1, 1, 4]);
    assert_eq!(logits.data, vec![1u8; 16]);
}

/// A tensor sent over QUIC must decode to exactly what the same tensor
/// decodes to over TCP — the substrate is transparent to the framing.
#[tokio::test]
async fn quic_and_tcp_decode_identically() {
    async fn roundtrip(kind: TransportKind, tensor: &Tensor) -> Tensor {
        let mut server = ActivationServer::new_with_kind("127.0.0.1", 0, kind);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            server.recv().await.unwrap().0
        });
        let mut client = ActivationClient::new_with_kind("127.0.0.1", port, kind);
        client.connect().await.unwrap();
        client.send(tensor).await.unwrap();
        h.await.unwrap()
    }

    let tensor = Tensor::new(DType::F16, [1, 3, 5], (0..30).collect());
    let over_tcp = roundtrip(TransportKind::Tcp, &tensor).await;
    let over_quic = roundtrip(TransportKind::Quic, &tensor).await;
    assert_eq!(over_tcp, over_quic);
    assert_eq!(over_quic, tensor);
}

/// A pipeline stage is idle between requests. QUIC keep-alive must hold an
/// otherwise-silent connection open across a gap longer than the keep-alive
/// interval (10 s), so the next frame still arrives. `#[ignore]` because it
/// sleeps ~11 s; run with `--ignored` when touching the QUIC timeout config.
#[tokio::test]
#[ignore = "sleeps ~11s to cross the QUIC keep-alive interval"]
async fn quic_survives_idle_gap_past_keepalive() {
    let mut server = quic_server(0);
    server.start().await.unwrap();
    let port = server.port();
    let h = tokio::spawn(async move {
        server.accept().await.unwrap();
        server.recv().await
    });
    let mut client = quic_client(port);
    client.connect().await.unwrap();
    // Idle past the 10 s keep-alive interval (but under the 30 s idle cap).
    tokio::time::sleep(Duration::from_secs(11)).await;
    let tensor = Tensor::from_2d(DType::F32, 1, 2, vec![0, 0, 128, 63, 0, 0, 0, 64]);
    client.send(&tensor).await.unwrap();
    let got = h.await.unwrap();
    assert!(
        got.is_ok(),
        "keep-alive should hold the idle QUIC link open: {:?}",
        got.err()
    );
}
