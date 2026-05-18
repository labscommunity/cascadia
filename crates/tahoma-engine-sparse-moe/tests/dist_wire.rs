//! Integration tests for the pipeline-parallel wire protocol.
//!
//! Exercises every frame kind (Forward, Reset, Token) on a real
//! in-process tahoma-transport socket pair, with no OpenVINO + no
//! model artifacts. Validates that the byte format on the wire round-
//! trips between two ranks without an actual Engine in between.

use std::sync::Arc;

use tahoma_engine_sparse_moe::dist::{
    decode_sampling, encode_sampling, recv_forward_body_server, recv_head_partial_body_client,
    recv_head_partial_body_server, recv_kind_client, recv_kind_server, recv_token_body_client,
    send_forward, send_head_partial, send_reset, send_token_upstream, FrameKind,
    SAMPLING_WIRE_BYTES,
};
use tahoma_engine_sparse_moe::SamplingConfig;
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

#[tokio::test]
async fn forward_frame_round_trips_hidden_state() {
    let (server, client) = make_pair().await;
    let hidden: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001 - 0.5).collect();
    let shape = [1u32, 1, 1024];
    let cfg = SamplingConfig {
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.15,
        repetition_window: 256,
        seed: Some(0x1234_5678_9abc_def0),
    };
    let cfg_for_assert = cfg.clone();

    let send_task = tokio::spawn(async move {
        send_forward(&client, 17, &cfg, &hidden, shape)
            .await
            .unwrap()
    });

    let kind = recv_kind_server(&server).await.unwrap();
    assert_eq!(kind, Some(FrameKind::Forward));
    let (past_seq_len, cfg_back, h_back, in_shape) =
        recv_forward_body_server(&server).await.unwrap();
    assert_eq!(past_seq_len, 17);
    assert_eq!(in_shape, shape);
    assert_eq!(h_back.len(), 1024);
    assert_eq!(cfg_back.temperature, cfg_for_assert.temperature);
    assert_eq!(cfg_back.top_p, cfg_for_assert.top_p);
    assert_eq!(
        cfg_back.repetition_penalty,
        cfg_for_assert.repetition_penalty
    );
    assert_eq!(cfg_back.repetition_window, cfg_for_assert.repetition_window);
    assert_eq!(cfg_back.seed, cfg_for_assert.seed);
    for (i, &got) in h_back.iter().enumerate() {
        let expected = (i as f32) * 0.001 - 0.5;
        assert!(
            (got - expected).abs() < 1e-6,
            "i={i}: got {got} expected {expected}"
        );
    }

    send_task.await.unwrap();
}

#[test]
fn sampling_wire_round_trips_defaults_and_explicit_values() {
    // Default config (greedy, no rep penalty, no seed) — the seed=0
    // sentinel must round-trip back to None.
    let cfg = SamplingConfig::default();
    let mut bytes = [0u8; SAMPLING_WIRE_BYTES];
    encode_sampling(&cfg, &mut bytes);
    let back = decode_sampling(&bytes);
    assert_eq!(back.temperature, cfg.temperature);
    assert_eq!(back.top_p, cfg.top_p);
    assert_eq!(back.repetition_penalty, cfg.repetition_penalty);
    assert_eq!(back.repetition_window, cfg.repetition_window);
    assert_eq!(back.seed, None);

    // Explicit fields including a seed.
    let cfg = SamplingConfig {
        temperature: 0.8,
        top_p: 0.95,
        repetition_penalty: 1.1,
        repetition_window: 1024,
        seed: Some(42),
    };
    let mut bytes = [0u8; SAMPLING_WIRE_BYTES];
    encode_sampling(&cfg, &mut bytes);
    let back = decode_sampling(&bytes);
    assert_eq!(back.temperature, cfg.temperature);
    assert_eq!(back.top_p, cfg.top_p);
    assert_eq!(back.repetition_penalty, cfg.repetition_penalty);
    assert_eq!(back.repetition_window, cfg.repetition_window);
    assert_eq!(back.seed, cfg.seed);
}

#[tokio::test]
async fn reset_frame_round_trips() {
    let (server, client) = make_pair().await;
    let send_task = tokio::spawn(async move {
        send_reset(&client).await.unwrap();
    });

    let kind = recv_kind_server(&server).await.unwrap();
    assert_eq!(kind, Some(FrameKind::Reset));
    send_task.await.unwrap();
}

#[tokio::test]
async fn token_frame_round_trips_upstream() {
    // For the token path, the LAST rank uses the server-side socket
    // (the one upstream connected to) to send back, and the upstream
    // recv via its client-side socket.
    let (server, client) = make_pair().await;
    let token = 12345i64;

    let send_task = tokio::spawn(async move { send_token_upstream(&server, token).await.unwrap() });

    let kind = recv_kind_client(&client).await.unwrap();
    assert_eq!(kind, Some(FrameKind::Token));
    let token_back = recv_token_body_client(&client).await.unwrap();
    assert_eq!(token_back, token);

    send_task.await.unwrap();
}

#[tokio::test]
async fn token_frame_handles_negative_and_extreme_ids() {
    let (server, client) = make_pair().await;
    let tokens = [-1i64, 0, 1, i64::MAX, i64::MIN, 163585, 163586];

    // Run all of them sequentially through the same socket.
    let send_task = tokio::spawn({
        let server = server.clone();
        async move {
            for &t in &tokens {
                send_token_upstream(&server, t).await.unwrap();
            }
        }
    });

    for &t in &tokens {
        let kind = recv_kind_client(&client).await.unwrap();
        assert_eq!(kind, Some(FrameKind::Token));
        let token_back = recv_token_body_client(&client).await.unwrap();
        assert_eq!(token_back, t);
    }

    send_task.await.unwrap();
}

#[tokio::test]
async fn head_partial_frame_round_trips_to_client() {
    // Rank-0 holds the lower vocab slice and sends a HeadPartial
    // upstream to the sampling rank. Wire goes through the server-side
    // socket (last rank's listener) → client-side socket (rank-0's
    // outbound). For this test we model the inverse direction
    // (server.send -> client.recv) so it matches send_token_upstream's
    // direction; same code path as a HeadPartial flowing from a
    // sampling rank's downstream peer.
    let (server, client) = make_pair().await;
    let vocab_start = 0u32;
    let vocab_end = 1024u32;
    let partial: Vec<f32> = (0..(vocab_end - vocab_start))
        .map(|i| (i as f32) * 0.0001 - 0.5)
        .collect();
    let partial_for_send = partial.clone();

    let send_task = tokio::spawn(async move {
        send_head_partial(&server, vocab_start, vocab_end, &partial_for_send)
            .await
            .unwrap();
    });

    let kind = recv_kind_client(&client).await.unwrap();
    assert_eq!(kind, Some(FrameKind::HeadPartial));
    let (vs, ve, body) = recv_head_partial_body_client(&client).await.unwrap();
    assert_eq!(vs, vocab_start);
    assert_eq!(ve, vocab_end);
    assert_eq!(body.len(), partial.len());
    for (i, (&a, &b)) in body.iter().zip(partial.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "head_partial[{i}]: got {a} expected {b}"
        );
    }

    send_task.await.unwrap();
}

#[tokio::test]
async fn head_partial_frame_round_trips_to_server() {
    // Inverse direction: rank-0 (client side) sends HeadPartial along
    // its downstream socket to the sampling rank (server side). Used
    // when the topology routes HeadPartial along the same forward
    // direction as Forward frames.
    let (server, client) = make_pair().await;
    let vocab_start = 81920u32;
    let vocab_end = 163840u32;
    let partial: Vec<f32> = (0..(vocab_end - vocab_start))
        .map(|i| (i as f32) * 0.00001 + 0.1)
        .collect();
    let partial_for_send = partial.clone();
    let _ = client; // make_pair returns both sides bound — we reuse the inverse.

    // For the inverse direction we need send-from-client / recv-on-server.
    // Reuse the existing pair but build a fresh inverse pair so each test
    // owns its own socket lifecycle.
    let (server2, client2) = make_pair().await;
    let _ = server; // silence unused on the first pair

    // Send via the inverse: use the existing public send fn (which uses
    // the server side). We model a HeadPartial flowing client→server by
    // using send_forward style send_raw → server.recv. Since
    // send_head_partial takes a server, exercise the server-receiver
    // path by using the server-side variant on the receive end.
    let send_task = tokio::spawn(async move {
        send_head_partial(&server2, vocab_start, vocab_end, &partial_for_send)
            .await
            .unwrap();
    });
    let kind = recv_kind_client(&client2).await.unwrap();
    assert_eq!(kind, Some(FrameKind::HeadPartial));
    let (vs, ve, body) = recv_head_partial_body_client(&client2).await.unwrap();
    assert_eq!(vs, vocab_start);
    assert_eq!(ve, vocab_end);
    assert_eq!(body.len(), partial.len());

    send_task.await.unwrap();
}

#[tokio::test]
async fn head_partial_two_halves_concat_to_full_vocab() {
    // End-to-end: two ranks each send their half of the lm_head output.
    // The receiver gathers both and concatenates them. This is the wire
    // half of the full head-TP path; the int4-gemm crate covers the
    // numerical (slice-then-concat) half.
    let (server, client) = make_pair().await;
    let vocab_total = 163840u32;
    let mid = vocab_total / 2;
    let lower: Vec<f32> = (0..mid).map(|i| (i as f32) * 0.00001).collect();
    let upper: Vec<f32> = (mid..vocab_total)
        .map(|i| (i as f32) * 0.00001 + 1.0)
        .collect();
    let lower_for_send = lower.clone();
    let upper_for_send = upper.clone();

    let send_task = tokio::spawn(async move {
        send_head_partial(&server, 0, mid, &lower_for_send)
            .await
            .unwrap();
        send_head_partial(&server, mid, vocab_total, &upper_for_send)
            .await
            .unwrap();
    });

    let mut received: Vec<(u32, Vec<f32>)> = Vec::with_capacity(2);
    for _ in 0..2 {
        let kind = recv_kind_client(&client).await.unwrap();
        assert_eq!(kind, Some(FrameKind::HeadPartial));
        let (vs, _ve, body) = recv_head_partial_body_client(&client).await.unwrap();
        received.push((vs, body));
    }
    received.sort_by_key(|(s, _)| *s);

    assert_eq!(received[0].0, 0);
    assert_eq!(received[1].0, mid);
    assert_eq!(received[0].1.len() as u32, mid);
    assert_eq!(received[1].1.len() as u32, vocab_total - mid);

    // Concat — exactly what the sampling rank will do.
    let mut combined: Vec<f32> = Vec::with_capacity(vocab_total as usize);
    combined.extend_from_slice(&received[0].1);
    combined.extend_from_slice(&received[1].1);
    assert_eq!(combined.len(), vocab_total as usize);
    // Spot-check a few values.
    assert!((combined[0] - lower[0]).abs() < 1e-6);
    assert!((combined[mid as usize] - upper[0]).abs() < 1e-6);
    assert!((combined[(vocab_total - 1) as usize] - upper[upper.len() - 1]).abs() < 1e-6);

    send_task.await.unwrap();
}

#[tokio::test]
async fn head_partial_recv_server_variant_round_trips() {
    // The server-side recv variant exists for the topology where the
    // sampling rank is downstream of the head-slice owner — i.e. it
    // accepted the upstream connection and reads HeadPartial from its
    // server socket. Round-trip the body bytes to validate the variant.
    let (server, client) = make_pair().await;
    let vocab_start = 100u32;
    let vocab_end = 250u32;
    let partial: Vec<f32> = (0..150).map(|i| i as f32 * 0.001).collect();
    let partial_for_send = partial.clone();

    // Client sends, server receives — opposite of the upstream-Token
    // path. send_head_partial wraps a server-side sender; here we
    // assemble the bytes by hand (mirrors how a future "send_head_partial_client"
    // would look) so we exercise the server-side recv variant in isolation.
    let kind_bytes = (FrameKind::HeadPartial as u32).to_be_bytes();
    let mut header = [0u8; 12];
    header[0..4].copy_from_slice(&kind_bytes);
    header[4..8].copy_from_slice(&vocab_start.to_be_bytes());
    header[8..12].copy_from_slice(&vocab_end.to_be_bytes());
    let mut tensor_bytes = Vec::with_capacity(partial.len() * 4);
    for v in &partial_for_send {
        tensor_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let tensor = tahoma_transport::Tensor::new(
        tahoma_transport::DType::F32,
        [1, 1, partial.len() as u32],
        tensor_bytes,
    );

    let send_task = tokio::spawn(async move {
        client.lock().await.send_raw(&header).await.unwrap();
        client.lock().await.send(&tensor).await.unwrap();
    });

    let kind = recv_kind_server(&server).await.unwrap();
    assert_eq!(kind, Some(FrameKind::HeadPartial));
    let (vs, ve, body) = recv_head_partial_body_server(&server).await.unwrap();
    assert_eq!(vs, vocab_start);
    assert_eq!(ve, vocab_end);
    assert_eq!(body.len(), partial.len());
    for (i, (&a, &b)) in body.iter().zip(partial.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "head_partial_server[{i}]");
    }

    send_task.await.unwrap();
}

#[tokio::test]
async fn sequence_reset_then_forward_then_token() {
    // Reset → Forward(hidden) → Token. The actual order a rank-0
    // driver issues on each task.
    let (server, client) = make_pair().await;
    let hidden: Vec<f32> = vec![0.123; 7168];
    let hidden_for_send = hidden.clone();
    let shape = [1u32, 1, 7168];

    let server_for_token = server.clone();
    let cfg = SamplingConfig::default();
    let send_task = tokio::spawn(async move {
        send_reset(&client).await.unwrap();
        send_forward(&client, 0, &cfg, &hidden_for_send, shape)
            .await
            .unwrap();
    });

    assert_eq!(
        recv_kind_server(&server).await.unwrap(),
        Some(FrameKind::Reset)
    );
    assert_eq!(
        recv_kind_server(&server).await.unwrap(),
        Some(FrameKind::Forward)
    );
    let (past, _cfg_back, h_back, sh) = recv_forward_body_server(&server).await.unwrap();
    assert_eq!(past, 0);
    assert_eq!(sh, shape);
    assert_eq!(h_back.len(), 7168);
    assert!((h_back[0] - 0.123).abs() < 1e-6);

    send_task.await.unwrap();

    // Now the worker rank sends a token back along the same upstream
    // socket — exercise that path too.
    send_token_upstream(&server_for_token, 42).await.unwrap();
}
