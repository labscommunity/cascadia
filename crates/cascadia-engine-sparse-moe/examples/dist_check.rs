//! Cross-host wire-protocol smoke test.
//!
//! Spawns either the server side (binds, recv Forward, send Token) or
//! the client side (connect, send Forward N times with synthetic
//! hidden state, recv Token, time the round-trip). Used to verify the
//! sparse-MoE pipeline-parallel wire protocol works over a real
//! cross-host network (e.g. a relayed VPN link between two Intel
//! AI PCs).
//!
//! ```text
//! # On the box that plays "last rank":
//! cargo run --release --example dist_check -- server 9100
//!
//! # On the box that plays "rank 0":
//! cargo run --release --example dist_check -- client <peer-ts-ip> 9100 100
//! ```

use std::env;
use std::sync::Arc;
use std::time::Instant;

use cascadia_engine_sparse_moe::dist::{
    recv_forward_body_server, recv_kind_client, recv_kind_server, recv_token_body_client,
    send_forward, send_reset, send_token_upstream, FrameKind,
};
use cascadia_transport::{ActivationClient, ActivationServer};
use tokio::sync::Mutex;

const HIDDEN_SIZE: usize = 7168;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("");
    match mode {
        "server" => {
            let port: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(9100);
            run_server(port).await?;
        }
        "client" => {
            let peer = args.get(2).cloned().ok_or("missing peer host")?;
            let port: u16 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(9100);
            let rounds: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10);
            run_client(&peer, port, rounds).await?;
        }
        _ => {
            eprintln!("usage:");
            eprintln!("  dist_check server <bind-port>");
            eprintln!("  dist_check client <peer-host> <peer-port> [rounds=10]");
            std::process::exit(2);
        }
    }
    Ok(())
}

async fn run_server(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let mut server = ActivationServer::new("0.0.0.0", port);
    server.start().await?;
    println!("[server] listening on 0.0.0.0:{port}");
    server.accept().await?;
    println!("[server] peer accepted");
    let server = Arc::new(Mutex::new(server));

    let mut frame_count = 0u32;
    let mut got_reset = false;
    loop {
        let kind = recv_kind_server(&server).await?;
        let Some(kind) = kind else {
            println!("[server] upstream closed cleanly after {frame_count} frames");
            break;
        };
        match kind {
            FrameKind::Reset => {
                got_reset = true;
                println!("[server] recv RESET");
            }
            FrameKind::Forward => {
                let (past_seq_len, _sampling, hidden, shape) =
                    recv_forward_body_server(&server).await?;
                let mean: f32 = hidden.iter().copied().sum::<f32>() / hidden.len() as f32;
                frame_count += 1;
                // Synthesise a "sampled" token from the input — XOR of
                // past_seq_len with the hidden mean's bit pattern.
                let token = (past_seq_len as i64) ^ (mean.to_bits() as i64);
                send_token_upstream(&server, token).await?;
                if frame_count <= 3 || frame_count.is_power_of_two() {
                    println!(
                        "[server] frame {frame_count} past_seq_len={past_seq_len} shape={:?} mean={:.4} → token={token}",
                        shape, mean
                    );
                }
            }
            FrameKind::Token => {
                println!("[server] unexpected TOKEN from upstream — ignoring");
            }
            FrameKind::ForwardNoSample => {
                // Prefill-intermediate: consume the body, ack with a dummy
                // Token(-1) — same shape the real worker uses.
                let (_p, _s, _h, _sh) = recv_forward_body_server(&server).await?;
                frame_count += 1;
                send_token_upstream(&server, -1).await?;
            }
            FrameKind::ForwardPrefill => {
                // Streamed prefill: consume the body, no ack (one-way).
                let (_p, _s, _h, _sh) = recv_forward_body_server(&server).await?;
                frame_count += 1;
            }
            FrameKind::ForwardBatch => {
                println!("[server] FORWARD_BATCH not handled in dist_check — ignoring");
            }
            FrameKind::TokenBatch => {
                println!("[server] unexpected TOKEN_BATCH from upstream — ignoring");
            }
            FrameKind::ForwardBatchPrefill => {
                println!("[server] FORWARD_BATCH_PREFILL not handled in dist_check — ignoring");
            }
        }
    }
    println!("[server] frames processed: {frame_count} (got_reset={got_reset})");
    Ok(())
}

async fn run_client(peer: &str, port: u16, rounds: u32) -> Result<(), Box<dyn std::error::Error>> {
    println!("[client] connecting to {peer}:{port}");
    let t_connect = Instant::now();
    let mut client = ActivationClient::new(peer, port);
    client
        .connect_with_timeout(std::time::Duration::from_secs(30))
        .await?;
    let connect_ms = t_connect.elapsed().as_secs_f64() * 1000.0;
    println!("[client] connected in {connect_ms:.1} ms");
    let client = Arc::new(Mutex::new(client));

    send_reset(&client).await?;
    println!("[client] sent RESET");

    let mut hidden = vec![0.0f32; HIDDEN_SIZE];
    let shape = [1u32, 1, HIDDEN_SIZE as u32];
    let mut rt_ms: Vec<f64> = Vec::with_capacity(rounds as usize);

    let total_t0 = Instant::now();
    for round in 0..rounds {
        // Fill hidden with a deterministic pattern that varies per round.
        for (i, v) in hidden.iter_mut().enumerate() {
            *v = (i as f32) * 0.0001 + (round as f32) * 0.01;
        }
        let cfg = cascadia_engine_sparse_moe::SamplingConfig::default();
        let t0 = Instant::now();
        send_forward(&client, round, &cfg, &hidden, shape).await?;
        let kind = recv_kind_client(&client).await?;
        let Some(FrameKind::Token) = kind else {
            return Err(format!("expected TOKEN, got {kind:?}").into());
        };
        let token = recv_token_body_client(&client).await?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        rt_ms.push(dt);
        if round < 3 || round.is_power_of_two() {
            println!("[client] round {round} rt={dt:.1} ms token={token}");
        }
    }
    let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;
    rt_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = rt_ms[rt_ms.len() / 2];
    let p99 = rt_ms[(rt_ms.len() * 99) / 100];
    let avg = rt_ms.iter().sum::<f64>() / rt_ms.len() as f64;
    let bytes_per_forward = 8 /* kind+past */ + 20 /* tensor header */ + HIDDEN_SIZE * 4 /* f32 */;
    let bytes_per_token_back = 12;
    let bytes_per_round = bytes_per_forward + bytes_per_token_back;
    println!("[client] {rounds} rounds in {total_ms:.1} ms");
    println!("[client] per-round rt avg={avg:.1} ms p50={p50:.1} ms p99={p99:.1} ms");
    println!(
        "[client] per-round wire bytes ≈ {} ({} forward + {} token)",
        bytes_per_round, bytes_per_forward, bytes_per_token_back
    );
    Ok(())
}
