//! A refused remote fetch logs only why at `warn`; the host and the address
//! it resolved to appear at `debug` alone, on `cascadia_api::media_fetch`.
//!
//! Its own test binary with one test: `tracing` caches each log site's
//! interest process-wide, so a capture next to other tests' threads (which
//! run with no subscriber) can lose events.

use std::sync::{Arc, Mutex};

use cascadia_api::{fetch_request_media, ChatMessage, MediaConfig};

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Capture {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn fetch_failures_log_no_host_or_address_at_warn() {
    let buf = Capture::default();
    let writer = buf.clone();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .init();

    let mut media = MediaConfig::ov_genai(None);
    media.allow_remote_urls = true;
    let cfg = Arc::new(media);
    for url in ["https://10.1.2.3/a.png", "https://localhost/a.png"] {
        let messages: Vec<ChatMessage> = vec![serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": url}}]
        }))
        .unwrap()];
        let err = fetch_request_media(Arc::clone(&cfg), &messages)
            .await
            .unwrap_err();
        assert_eq!(err.message(), "image 0: image host is not a public address");
    }

    let out = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
    let warn: Vec<&str> = out.lines().filter(|l| l.contains(" WARN ")).collect();
    let debug: Vec<&str> = out.lines().filter(|l| l.contains(" DEBUG ")).collect();
    assert_eq!(warn.len(), 2, "{out}");
    for line in &warn {
        assert!(line.contains("cascadia_api::media_fetch"), "{line}");
        assert!(line.contains("reason=\"not_public\""), "{line}");
        for leak in ["10.1.2.3", "localhost", "127.0.0.1", "::1"] {
            assert!(!line.contains(leak), "warn names {leak}: {line}");
        }
    }
    let detail: Vec<&&str> = debug
        .iter()
        .filter(|l| l.contains("remote media fetch detail"))
        .collect();
    assert_eq!(detail.len(), 2, "{out}");
    assert!(detail[0].contains("10.1.2.3"), "{}", detail[0]);
    assert!(detail[1].contains("localhost"), "{}", detail[1]);
}
