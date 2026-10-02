//! #17 acceptance micro-benchmark: round-trip latency for pipeline-sized
//! tensor frames over a Unix domain socket vs loopback TCP on one host.
//!
//! Ignored by default (it's a timing measurement, not a pass/fail test).
//! Run manually:
//!
//! ```bash
//! cargo test -p cascadia-transport --release --test uds_vs_tcp_bench -- --ignored --nocapture
//! ```
//!
//! Results are recorded in docs/CLI.md ("Unix domain sockets").

#![cfg(unix)]

use std::time::Instant;

use cascadia_transport::{ActivationClient, ActivationServer, DType, Tensor};

const ROUNDS: usize = 300;
const WARMUP: usize = 30;

/// One echo server: recv a tensor, send it straight back, forever.
async fn echo_server(mut server: ActivationServer) {
    server.accept().await.unwrap();
    loop {
        let Ok((tensor, _)) = server.recv().await else {
            return;
        };
        if server.send(&tensor).await.is_err() {
            return;
        }
    }
}

/// Drive ROUNDS round trips of `frame_bytes`-sized frames; returns
/// (mean_us, p50_us, p99_us) per round trip.
async fn bench_roundtrips(client: &mut ActivationClient, frame_bytes: usize) -> (f64, f64, f64) {
    let cols = (frame_bytes / 4) as u32; // f32 elements
    let tensor = Tensor::from_2d(DType::F32, 1, cols, vec![0u8; cols as usize * 4]);
    for _ in 0..WARMUP {
        client.send(&tensor).await.unwrap();
        client.recv_reply().await.unwrap();
    }
    let mut samples_us = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let start = Instant::now();
        client.send(&tensor).await.unwrap();
        let (echoed, _) = client.recv_reply().await.unwrap();
        samples_us.push(start.elapsed().as_secs_f64() * 1e6);
        assert_eq!(echoed.data.len(), tensor.data.len());
    }
    samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = samples_us.iter().sum::<f64>() / samples_us.len() as f64;
    let p50 = samples_us[samples_us.len() / 2];
    let p99 = samples_us[samples_us.len() * 99 / 100];
    (mean, p50, p99)
}

#[tokio::test]
#[ignore = "timing benchmark; run with --ignored --nocapture"]
async fn uds_vs_tcp_roundtrip_latency() {
    // Frame sizes: 14 KiB ≈ a 7168-dim bf16 hidden state (K2.6-class decode
    // step), 1 MiB ≈ a large prefill/logits frame.
    for &frame in &[14 * 1024usize, 1024 * 1024] {
        // TCP over loopback.
        let mut tcp_server = ActivationServer::new("127.0.0.1", 0);
        tcp_server.start().await.unwrap();
        let port = tcp_server.port();
        tokio::spawn(echo_server(tcp_server));
        let mut tcp_client = ActivationClient::new("127.0.0.1", port);
        tcp_client.connect().await.unwrap();
        let (tcp_mean, tcp_p50, tcp_p99) = bench_roundtrips(&mut tcp_client, frame).await;

        // Unix domain socket.
        let sock = std::env::temp_dir().join(format!("cascadia-bench-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let addr = format!("unix:{}", sock.display());
        let mut uds_server = ActivationServer::new(addr.clone(), 0);
        uds_server.start().await.unwrap();
        tokio::spawn(echo_server(uds_server));
        let mut uds_client = ActivationClient::new(addr, 0);
        uds_client.connect().await.unwrap();
        let (uds_mean, uds_p50, uds_p99) = bench_roundtrips(&mut uds_client, frame).await;
        let _ = std::fs::remove_file(&sock);

        println!(
            "frame {:>7} B | tcp mean {:>8.1} us p50 {:>8.1} p99 {:>8.1} | uds mean {:>8.1} us p50 {:>8.1} p99 {:>8.1} | p50 win {:>5.1}%",
            frame,
            tcp_mean,
            tcp_p50,
            tcp_p99,
            uds_mean,
            uds_p50,
            uds_p99,
            (1.0 - uds_p50 / tcp_p50) * 100.0
        );
    }
}
