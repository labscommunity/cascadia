//! End-to-end binary integration tests for the tahoma Rust port.
//!
//! Spawns the built `tahoma` binary, waits for /health, exercises the
//! OpenAI-compatible API, then tears down. Validates the full chain:
//! CLI parsing -> Runner.start() -> Engine -> API handlers -> wire JSON.
//!
//! Run via:
//!     cargo build -p tahoma
//!     cargo test -p tahoma-tests-e2e -- --test-threads=1

use std::net::TcpListener;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use tokio::process::{Child, Command};

/// Find a free TCP port to avoid collisions when tests run in parallel.
pub fn pick_free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

pub fn binary_path() -> PathBuf {
    // Tests run from the workspace root; cargo puts binaries here.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .join("target")
        .join(if cfg!(debug_assertions) { "debug" } else { "release" })
        .join("tahoma")
}

pub struct TahomaProc {
    pub port: u16,
    pub child: Child,
}

impl TahomaProc {
    pub async fn spawn_mock_with_api() -> Self {
        let port = pick_free_port();
        let bin = binary_path();
        assert!(
            bin.exists(),
            "tahoma binary not built — run `cargo build -p tahoma` first ({:?})",
            bin
        );
        let child = Command::new(&bin)
            .args([
                "worker",
                "--rank",
                "0",
                "--total",
                "1",
                "--engine",
                "mock",
                "--model",
                "mock-model",
                "--api",
                &format!(":{port}"),
                "--log-level",
                "warn",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .expect("spawn tahoma binary");
        Self { port, child }
    }

    pub async fn wait_for_health(&self, timeout: Duration) -> bool {
        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{}/health", self.port);
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Ok(r) = client.get(&url).send().await {
                if r.status().is_success() {
                    return true;
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        false
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{}", self.port, path)
    }
}

impl Drop for TahomaProc {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn binary_serves_health_then_models_then_chat() {
        let proc = TahomaProc::spawn_mock_with_api().await;
        assert!(
            proc.wait_for_health(Duration::from_secs(10)).await,
            "binary did not become healthy in 10s"
        );

        let client = reqwest::Client::new();

        // /v1/models
        let r = client
            .get(proc.url("/v1/models"))
            .send()
            .await
            .expect("get models");
        assert!(r.status().is_success(), "models status: {}", r.status());
        let v: serde_json::Value = r.json().await.unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["id"], "mock-model");

        // /v1/chat/completions (non-streaming)
        let body = serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "alpha bravo charlie"}],
            "max_tokens": 2,
            "stream": false,
        });
        let r = client
            .post(proc.url("/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .expect("post chat");
        assert!(r.status().is_success(), "chat status: {}", r.status());
        let v: serde_json::Value = r.json().await.unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string");
        assert!(!content.is_empty(), "completion content was empty");
    }

    #[tokio::test]
    async fn binary_supports_concurrent_requests() {
        let proc = TahomaProc::spawn_mock_with_api().await;
        assert!(proc.wait_for_health(Duration::from_secs(10)).await);

        let client = reqwest::Client::new();
        let url = proc.url("/v1/chat/completions");

        // Fire 4 requests in parallel; all should complete.
        let futures = (0..4).map(|i| {
            let client = client.clone();
            let url = url.clone();
            tokio::spawn(async move {
                let body = serde_json::json!({
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": format!("hello {}", i)}],
                    "max_tokens": 1,
                });
                client.post(&url).json(&body).send().await
            })
        });

        for fut in futures {
            let r = fut.await.unwrap().expect("request");
            assert!(r.status().is_success(), "concurrent req status {}", r.status());
        }
    }
}
