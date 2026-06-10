//! Stress tests for the idle-tolerant frame-start recv (recv_exact_frame_start).
//! All run on a 1s activation timeout to keep wall-clock short; serialized via
//! a lock because the timeout is a process-global.

use cascadia_transport::{
    set_activation_timeout_secs, ActivationClient, ActivationServer, DType, Tensor,
};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn tensor(i: u8) -> Tensor {
    Tensor::new(DType::I8, [1, 1, 4], vec![i, i, i, i])
}

/// 30 frames over one connection with idle gaps straddling the timeout
/// (every 5th gap is 1.4s on a 1s timeout). All frames must arrive intact.
#[tokio::test]
async fn soak_frames_with_idle_gaps_survive_and_stay_in_sync() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    set_activation_timeout_secs(1);

    let mut server = ActivationServer::new("127.0.0.1", 0);
    server.start().await.unwrap();
    let port = server.port();
    let h = tokio::spawn(async move {
        server.accept().await.unwrap();
        let mut got = Vec::new();
        for _ in 0..30 {
            let (t, _) = server.recv().await?;
            got.push(t.data[0]);
        }
        Ok::<_, cascadia_transport::TransportError>(got)
    });

    let mut client = ActivationClient::new("127.0.0.1", port);
    client.connect().await.unwrap();
    for i in 0..30u8 {
        if i % 5 == 0 {
            tokio::time::sleep(Duration::from_millis(1400)).await; // > timeout
        }
        client.send(&tensor(i)).await.unwrap();
    }
    let got = h.await.unwrap();
    set_activation_timeout_secs(0);
    let got = got.expect("all frames must survive idle gaps");
    assert_eq!(got, (0..30u8).collect::<Vec<_>>(), "frames lost or desynced");
}

/// Full header + partial body then stall: mid-frame timeout must still fire.
#[tokio::test]
async fn header_then_partial_body_stall_times_out() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    set_activation_timeout_secs(1);

    let mut server = ActivationServer::new("127.0.0.1", 0);
    server.start().await.unwrap();
    let port = server.port();
    let h = tokio::spawn(async move {
        server.accept().await.unwrap();
        server.recv().await
    });

    let mut sock = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    // Valid 20-byte header: payload_len=4, dtype=I8(code 2), shape [1,1,4]
    let mut header = Vec::new();
    header.extend_from_slice(&4u32.to_be_bytes());
    header.extend_from_slice(&2u32.to_be_bytes());
    header.extend_from_slice(&1u32.to_be_bytes());
    header.extend_from_slice(&1u32.to_be_bytes());
    header.extend_from_slice(&4u32.to_be_bytes());
    sock.write_all(&header).await.unwrap();
    sock.write_all(&[7u8, 7]).await.unwrap(); // 2 of 4 body bytes, then stall
    sock.flush().await.unwrap();

    let got = h.await.unwrap();
    set_activation_timeout_secs(0);
    assert!(got.is_err(), "partial body then stall must time out");
}

/// 4 server/client pairs concurrently, each with idle gaps > timeout.
#[tokio::test]
async fn concurrent_pairs_with_idle_gaps() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    set_activation_timeout_secs(1);

    let mut tasks = Vec::new();
    for pair in 0..4u8 {
        tasks.push(tokio::spawn(async move {
            let mut server = ActivationServer::new("127.0.0.1", 0);
            server.start().await.unwrap();
            let port = server.port();
            let recv = tokio::spawn(async move {
                server.accept().await.unwrap();
                let mut got = Vec::new();
                for _ in 0..6 {
                    let (t, _) = server.recv().await?;
                    got.push(t.data[0]);
                }
                Ok::<_, cascadia_transport::TransportError>(got)
            });
            let mut client = ActivationClient::new("127.0.0.1", port);
            client.connect().await.unwrap();
            for i in 0..6u8 {
                if i % 2 == 0 {
                    tokio::time::sleep(Duration::from_millis(1300)).await;
                }
                client.send(&tensor(pair * 10 + i)).await.unwrap();
            }
            let got = recv.await.unwrap().expect("pair must survive gaps");
            assert_eq!(got, (0..6u8).map(|i| pair * 10 + i).collect::<Vec<_>>());
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
    set_activation_timeout_secs(0);
}

/// 500 rapid frames, no gaps: integrity + no perf cliff from the split read.
#[tokio::test]
async fn rapid_fire_integrity() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    set_activation_timeout_secs(1);

    let mut server = ActivationServer::new("127.0.0.1", 0);
    server.start().await.unwrap();
    let port = server.port();
    let h = tokio::spawn(async move {
        server.accept().await.unwrap();
        let mut sum: u64 = 0;
        for _ in 0..500 {
            let (t, _) = server.recv().await?;
            sum += t.data[0] as u64;
        }
        Ok::<_, cascadia_transport::TransportError>(sum)
    });
    let mut client = ActivationClient::new("127.0.0.1", port);
    client.connect().await.unwrap();
    let start = std::time::Instant::now();
    let mut expect: u64 = 0;
    for i in 0..500u32 {
        let b = (i % 251) as u8;
        expect += b as u64;
        client.send(&tensor(b)).await.unwrap();
    }
    let sum = h.await.unwrap().unwrap();
    let elapsed = start.elapsed();
    set_activation_timeout_secs(0);
    assert_eq!(sum, expect, "data corruption under rapid fire");
    assert!(
        elapsed < Duration::from_secs(10),
        "500 loopback frames took {elapsed:?} — perf cliff"
    );
}
