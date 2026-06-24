//! Integration tests for the pipeline-parallel wire protocol.
//!
//! Exercises every frame kind (Forward, Reset, Token) on a real
//! in-process cascadia-transport socket pair, with no OpenVINO + no
//! model artifacts. Validates that the byte format on the wire round-
//! trips between two ranks without an actual Engine in between.

use std::sync::Arc;

use cascadia_engine_sparse_moe::dist::{
    decode_sampling, encode_sampling, recv_forward_batch_body_server, recv_forward_body_server,
    recv_kind_client, recv_kind_server, recv_token_batch_body_client, recv_token_body_client,
    send_forward, send_forward_batch, send_reset, send_token_batch_upstream, send_token_upstream,
    FrameKind, MAX_BATCH_COUNT, SAMPLING_WIRE_BYTES,
};
use cascadia_engine_sparse_moe::SamplingConfig;
use cascadia_transport::{ActivationClient, ActivationServer};
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
        ..SamplingConfig::default()
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

    // Explicit fields including a seed + the #14 sampling knobs.
    let cfg = SamplingConfig {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40,
        frequency_penalty: 0.5,
        presence_penalty: -0.25,
        repetition_penalty: 1.1,
        repetition_window: 1024,
        seed: Some(42),
    };
    let mut bytes = [0u8; SAMPLING_WIRE_BYTES];
    encode_sampling(&cfg, &mut bytes);
    let back = decode_sampling(&bytes);
    assert_eq!(back.temperature, cfg.temperature);
    assert_eq!(back.top_p, cfg.top_p);
    assert_eq!(back.top_k, cfg.top_k);
    assert_eq!(back.frequency_penalty, cfg.frequency_penalty);
    assert_eq!(back.presence_penalty, cfg.presence_penalty);
    assert_eq!(back.repetition_penalty, cfg.repetition_penalty);
    assert_eq!(back.repetition_window, cfg.repetition_window);
    assert_eq!(back.seed, cfg.seed);

    // The Forward frame kind was bumped to a new code alongside the wider
    // sampling block so a 28-byte-layout peer fails fast (#14).
    assert_eq!(
        FrameKind::from_code(0x53_4D_45_04),
        Some(FrameKind::Forward)
    );
    assert_eq!(
        FrameKind::from_code(0x53_4D_45_05),
        Some(FrameKind::ForwardBatch)
    );
    // The old 28-byte Forward/ForwardBatch codes are now unknown → rejected.
    assert_eq!(FrameKind::from_code(0x53_4D_45_02), None);
    assert_eq!(FrameKind::from_code(0x53_4D_45_03), None);
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

#[tokio::test]
async fn forward_batch_frame_round_trips_k4() {
    // The headline test: K=4 hiddens go down the wire in one frame
    // and come back as exactly the K=4 f32 vectors the sender shipped.
    // Mirrors the call site in `drive_generation_first_spec`.
    let (server, client) = make_pair().await;
    let hidden_size: usize = 1024;
    let k: u32 = 4;
    let total_elems = (hidden_size as u32) * k;
    let hidden: Vec<f32> = (0..total_elems as usize)
        .map(|i| (i as f32) * 0.0001 - 0.25)
        .collect();
    let shape = [1u32, k, hidden_size as u32];
    let cfg = SamplingConfig {
        temperature: 0.0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        repetition_window: 0,
        seed: Some(42),
        ..SamplingConfig::default()
    };
    let cfg_for_assert = cfg.clone();
    let past_seq_len_start: u32 = 13;

    let send_task = tokio::spawn(async move {
        send_forward_batch(&client, past_seq_len_start, k, &cfg, &hidden, shape)
            .await
            .unwrap()
    });

    let kind = recv_kind_server(&server).await.unwrap();
    assert_eq!(kind, Some(FrameKind::ForwardBatch));
    let (past_back, k_back, cfg_back, h_back, in_shape) =
        recv_forward_batch_body_server(&server).await.unwrap();
    assert_eq!(past_back, past_seq_len_start);
    assert_eq!(k_back, k);
    assert_eq!(in_shape, shape);
    assert_eq!(h_back.len(), total_elems as usize);
    assert_eq!(cfg_back.seed, cfg_for_assert.seed);
    assert_eq!(cfg_back.temperature, cfg_for_assert.temperature);
    // Spot-check a few hidden values across the batch dimension so a
    // misaligned reshape (K interleaved wrong) would be caught.
    for k_idx in 0..(k as usize) {
        let row_off = k_idx * hidden_size;
        for sample_idx in [0usize, 1, hidden_size / 2, hidden_size - 1] {
            let i = row_off + sample_idx;
            let expected = (i as f32) * 0.0001 - 0.25;
            assert!(
                (h_back[i] - expected).abs() < 1e-6,
                "K={k_idx} sample={sample_idx} got={} expected={expected}",
                h_back[i]
            );
        }
    }

    send_task.await.unwrap();
}

#[tokio::test]
async fn token_batch_frame_round_trips_k4() {
    // The response shape of a ForwardBatch verify. Worker sends K
    // sampled tokens in one frame; rank-0 driver reconstructs the
    // ordered vector.
    let (server, client) = make_pair().await;
    let tokens = vec![100i64, 200, 300, 400];
    let tokens_for_assert = tokens.clone();

    let send_task =
        tokio::spawn(async move { send_token_batch_upstream(&server, &tokens).await.unwrap() });

    let kind = recv_kind_client(&client).await.unwrap();
    assert_eq!(kind, Some(FrameKind::TokenBatch));
    let tokens_back = recv_token_batch_body_client(&client).await.unwrap();
    assert_eq!(tokens_back, tokens_for_assert);

    send_task.await.unwrap();
}

#[tokio::test]
async fn token_batch_handles_negative_and_extreme_ids() {
    // Bonus token can legitimately be any i64 (negative is valid per
    // the K2.6 sampler's argmax behavior on NaN logits → 0; extremes
    // catch sign-extension bugs in the encoder).
    let (server, client) = make_pair().await;
    let tokens = vec![i64::MIN, -1i64, 0, 1, i64::MAX, 163585, 163586, -163586];
    let tokens_for_assert = tokens.clone();
    let send_task =
        tokio::spawn(async move { send_token_batch_upstream(&server, &tokens).await.unwrap() });
    let kind = recv_kind_client(&client).await.unwrap();
    assert_eq!(kind, Some(FrameKind::TokenBatch));
    let tokens_back = recv_token_batch_body_client(&client).await.unwrap();
    assert_eq!(tokens_back, tokens_for_assert);
    send_task.await.unwrap();
}

#[tokio::test]
async fn forward_batch_then_token_batch_pair() {
    // Sequence: ForwardBatch(K) downstream → TokenBatch(K) upstream
    // along the SAME socket pair. Matches the
    // `forward_batch_first` round-trip in `drive_generation_first_spec`.
    let (server, client) = make_pair().await;
    let k: u32 = 3;
    let hidden_size: usize = 64;
    let hidden: Vec<f32> = (0..k as usize * hidden_size)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let shape = [1u32, k, hidden_size as u32];
    let cfg = SamplingConfig::default();

    let server_for_token = server.clone();
    let client_for_send = client.clone();
    let send_task = tokio::spawn(async move {
        send_forward_batch(&client_for_send, 7, k, &cfg, &hidden, shape)
            .await
            .unwrap();
    });

    let kind = recv_kind_server(&server).await.unwrap();
    assert_eq!(kind, Some(FrameKind::ForwardBatch));
    let (past, k_back, _cfg, h_back, sh) = recv_forward_batch_body_server(&server).await.unwrap();
    assert_eq!(past, 7);
    assert_eq!(k_back, k);
    assert_eq!(sh, shape);
    assert_eq!(h_back.len(), k as usize * hidden_size);
    send_task.await.unwrap();

    // Now reply with a TokenBatch upstream on the same socket.
    let response = vec![11i64, 22, 33];
    let response_for_assert = response.clone();
    let send_task = tokio::spawn(async move {
        send_token_batch_upstream(&server_for_token, &response)
            .await
            .unwrap();
    });
    let kind = recv_kind_client(&client).await.unwrap();
    assert_eq!(kind, Some(FrameKind::TokenBatch));
    let tokens_back = recv_token_batch_body_client(&client).await.unwrap();
    assert_eq!(tokens_back, response_for_assert);
    send_task.await.unwrap();
}

#[tokio::test]
async fn forward_batch_rejects_k_zero() {
    // K=0 has no defined meaning — sender should refuse to construct
    // such a frame so the receiver never has to worry about it.
    let (_server, client) = make_pair().await;
    let cfg = SamplingConfig::default();
    let res = send_forward_batch(&client, 0, 0, &cfg, &[], [1, 0, 64]).await;
    assert!(res.is_err(), "send_forward_batch(K=0) should error");
}

#[tokio::test]
async fn forward_batch_rejects_oversize_k() {
    // K > MAX_BATCH_COUNT (256) is a defense against a corrupt /
    // adversarial caller asking us to allocate K * hidden bytes from a
    // 4-byte header.
    let (_server, client) = make_pair().await;
    let cfg = SamplingConfig::default();
    let bad_k = MAX_BATCH_COUNT + 1;
    let res = send_forward_batch(&client, 0, bad_k, &cfg, &[], [1, bad_k, 64]).await;
    assert!(res.is_err(), "send_forward_batch(K>MAX) should error");
}

#[tokio::test]
async fn forward_batch_rejects_mismatched_shape() {
    // shape[1] must equal batch_count — a mismatch usually means the
    // caller built the hidden buffer wrong and we want a loud error
    // not a silently-truncated send.
    let (_server, client) = make_pair().await;
    let cfg = SamplingConfig::default();
    // K=4 claimed in batch_count, but shape says K=3.
    let hidden = vec![0.0f32; 4 * 64];
    let res = send_forward_batch(&client, 0, 4, &cfg, &hidden, [1, 3, 64]).await;
    assert!(
        res.is_err(),
        "send_forward_batch with K vs shape mismatch should error"
    );
}

#[test]
fn frame_kind_codes_remain_stable() {
    // This pins the wire codes so a future refactor can't drift them silently.
    // Reset/Token carry no sampling block and keep their original codes.
    // Forward/ForwardBatch were bumped 0x02→0x04 / 0x03→0x05 in #14 when the
    // sampling block grew 28→40 bytes, so a peer on the old layout fails fast
    // at parse_kind rather than mis-reading the wider frame.
    assert_eq!(FrameKind::Forward as u32, 0x53_4D_45_04);
    assert_eq!(FrameKind::Reset as u32, 0x53_4D_45_10);
    assert_eq!(FrameKind::Token as u32, 0x53_4D_45_20);
    assert_eq!(FrameKind::ForwardBatch as u32, 0x53_4D_45_05);
    assert_eq!(FrameKind::TokenBatch as u32, 0x53_4D_45_21);
    // The retired 28-byte-layout codes must now be rejected as unknown.
    assert_eq!(FrameKind::from_code(0x53_4D_45_02), None);
    assert_eq!(FrameKind::from_code(0x53_4D_45_03), None);

    // And round-trip every variant through from_code.
    for k in [
        FrameKind::Forward,
        FrameKind::Reset,
        FrameKind::Token,
        FrameKind::ForwardBatch,
        FrameKind::TokenBatch,
    ] {
        assert_eq!(FrameKind::from_code(k as u32), Some(k));
    }
    assert_eq!(FrameKind::from_code(0xDEAD_BEEF), None);
}
