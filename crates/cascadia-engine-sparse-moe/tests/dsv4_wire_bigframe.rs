//! Round-trip the Forward hidden tensor over the REAL loopback transport at
//! the real dsv4 wire width (hc*dim = 16384 f32 = 65536 bytes, the 64 KB
//! boundary) and adjacent sizes. The tiny fixtures only ever send 256 floats,
//! so a size-dependent frame bug would hide there. Byte-exact required.
//! Also covers the ForwardNoSample kind: identical body, distinct kind code.
use cascadia_engine_sparse_moe::dist::{
    recv_forward_body_server, recv_kind_server, send_forward, send_forward_nosample, FrameKind,
};
use cascadia_engine_sparse_moe::SamplingConfig;
use cascadia_transport::{ActivationClient, ActivationServer};
use std::sync::Arc;
use tokio::sync::Mutex;

async fn pair() -> (Arc<Mutex<ActivationServer>>, Arc<Mutex<ActivationClient>>) {
    let mut s = ActivationServer::new("127.0.0.1", 0);
    s.start().await.unwrap();
    let port = s.port();
    let s = Arc::new(Mutex::new(s));
    let sc = s.clone();
    let t = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });
    let mut c = ActivationClient::new("127.0.0.1", port);
    c.connect_with_timeout(std::time::Duration::from_secs(5))
        .await
        .unwrap();
    let c = Arc::new(Mutex::new(c));
    t.await.unwrap();
    (s, c)
}

async fn roundtrip(n: usize, nosample: bool) {
    // distinctive, non-degenerate values so any drop/dup/reorder shows up
    let hidden: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let (s, c) = pair().await;
    let cfg = SamplingConfig::default();
    let hib = hidden.clone();
    let cb = c.clone();
    let cfgb = cfg.clone();
    let send = tokio::spawn(async move {
        if nosample {
            send_forward_nosample(&cb, 7, &cfgb, &hib, [1, 1, n as u32])
                .await
                .unwrap();
        } else {
            send_forward(&cb, 7, &cfgb, &hib, [1, 1, n as u32], true)
                .await
                .unwrap();
        }
    });
    let want_kind = if nosample {
        FrameKind::ForwardNoSample
    } else {
        FrameKind::Forward
    };
    assert_eq!(recv_kind_server(&s).await.unwrap(), Some(want_kind));
    let (pos, _cfg, _ph, got, shape) = recv_forward_body_server(&s).await.unwrap();
    send.await.unwrap();

    assert_eq!(pos, 7, "past_seq_len corrupted at n={n}");
    assert_eq!(shape, [1, 1, n as u32], "shape corrupted at n={n}");
    assert_eq!(got.len(), n, "length changed at n={n}: got {}", got.len());
    assert_eq!(got, hidden, "hidden bytes corrupted at n={n}");
}

#[tokio::test]
async fn forward_hidden_roundtrips_at_real_width_and_boundaries() {
    // 256 = tiny (known good). 16384 = real dsv4 hc*dim (65536 B, exactly
    // 64 KB). 16383/16385 straddle it; 65536 floats = 256 KB well past any
    // 16-bit cap.
    for n in [256usize, 16383, 16384, 16385, 20000, 65536] {
        roundtrip(n, false).await;
        eprintln!("ok n={n} ({} bytes)", n * 4);
    }
}

#[tokio::test]
async fn forward_nosample_roundtrips_and_is_distinct_kind() {
    roundtrip(16384, true).await;
    // kind codes must be distinct and reversible
    assert_ne!(FrameKind::Forward as u32, FrameKind::ForwardNoSample as u32);
    assert_eq!(
        FrameKind::from_code(FrameKind::ForwardNoSample as u32),
        Some(FrameKind::ForwardNoSample)
    );
}
