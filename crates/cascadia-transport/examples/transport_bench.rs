//! Payload-sweep micro-benchmark for the activation transport (TCP vs QUIC).
//!
//! Isolates transport cost from model compute. A server echoes tensors; a
//! client ping-pongs fixed-size frames across a size sweep and reports p50/
//! p90/p99/mean round-trip latency + derived MB/s. Fitting `rtt ~ a + b*bytes`
//! per transport splits **per-frame** cost (intercept `a`: userspace datapath,
//! ACK/flow-control bookkeeping, pacing) from **per-byte** cost (slope `b`:
//! TLS AEAD crypto + userspace reassembly copies).
//!
//! Build:
//!   cargo build --release --example transport_bench --features quic
//! Run (server first, then client — same `<tcp|quic>` on both):
//!   transport_bench server <tcp|quic> <bind_host> <port>
//!   transport_bench client <tcp|quic> <server_host> <port>
//!
//! The connection is established once and reused across the whole sweep, so
//! the numbers are warm steady-state (handshake excluded).

use std::time::Instant;

use cascadia_transport::{ActivationClient, ActivationServer, DType, Tensor, TransportKind};

/// (payload_bytes, iters). Fewer iters at large sizes to bound wall-clock;
/// still enough samples for a stable p99.
const SWEEP: &[(usize, usize)] = &[
    (64, 5000),
    (1024, 5000),
    (4096, 3000),
    (16384, 2000),
    (65536, 1000),
    (262144, 500),
    (1048576, 200),
];

fn parse_kind(s: &str) -> TransportKind {
    match s {
        "quic" => TransportKind::Quic,
        "tcp" => TransportKind::Tcp,
        other => panic!("transport must be tcp|quic, got {other}"),
    }
}

/// i8 `[1,1,N]` => data length == N, so the wire payload is exactly N bytes
/// (recv validates shape * dtype_size == payload_len).
fn make_tensor(nbytes: usize) -> Tensor {
    Tensor::new(DType::I8, [1, 1, nbytes as u32], vec![0x5a; nbytes])
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: transport_bench <server|client> <tcp|quic> <host> <port>");
        std::process::exit(2);
    }
    let role = args[1].as_str();
    let kind = parse_kind(&args[2]);
    let host = args[3].clone();
    let port: u16 = args[4].parse().expect("port");

    match role {
        "server" => run_server(kind, &host, port).await,
        "client" => run_client(kind, &host, port).await,
        other => panic!("role must be server|client, got {other}"),
    }
}

async fn run_server(kind: TransportKind, host: &str, port: u16) {
    let mut server = ActivationServer::new_with_kind(host, port, kind);
    server.start().await.expect("server start");
    eprintln!("[server] {kind:?} listening on {host}:{}", server.port());
    server.accept().await.expect("server accept");
    eprintln!("[server] client connected; echoing until it closes");
    loop {
        match server.recv().await {
            Ok((t, _)) => {
                if let Err(e) = server.send(&t).await {
                    eprintln!("[server] send err: {e}");
                    break;
                }
            }
            Err(_) => {
                eprintln!("[server] client closed; exiting");
                break;
            }
        }
    }
}

async fn run_client(kind: TransportKind, host: &str, port: u16) {
    let mut client = ActivationClient::new_with_kind(host, port, kind);
    client.connect().await.expect("client connect");
    eprintln!("[client] {kind:?} connected to {host}:{port}");
    let kname = if kind == TransportKind::Quic {
        "quic"
    } else {
        "tcp"
    };
    println!("transport,bytes,iters,p50_us,p90_us,p99_us,mean_us,mb_per_s");
    for &(nbytes, iters) in SWEEP {
        let t = make_tensor(nbytes);
        let warm = (iters / 20).max(50);
        for _ in 0..warm {
            client.send(&t).await.unwrap();
            client.recv().await.unwrap();
        }
        let mut us = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            client.send(&t).await.unwrap();
            let _ = client.recv().await.unwrap();
            us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |q: f64| us[((us.len() as f64 * q) as usize).min(us.len() - 1)];
        let mean: f64 = us.iter().sum::<f64>() / us.len() as f64;
        // A round-trip moves the payload twice; report goodput over mean RTT.
        let mb_s = (2.0 * nbytes as f64) / (mean / 1e6) / (1024.0 * 1024.0);
        println!(
            "{kname},{nbytes},{iters},{:.1},{:.1},{:.1},{:.1},{:.1}",
            pct(0.50),
            pct(0.90),
            pct(0.99),
            mean,
            mb_s
        );
    }
    client.close().await;
    eprintln!("[client] done");
}
