//! End-to-end binary integration tests for the cascadia Rust port.
//!
//! Spawns the built `cascadia` binary, waits for /health, exercises the
//! OpenAI-compatible API, then tears down. Validates the full chain:
//! CLI parsing -> Runner.start() -> Engine -> API handlers -> wire JSON.
//!
//! Run via:
//!     cargo build -p cascadia
//!     cargo test -p cascadia-tests-e2e
//!
//! Every test binds ephemeral ports (see [`pick_free_port`]), so they run
//! in parallel safely — `--test-threads=1` is not required.

use std::net::TcpListener;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use tokio::process::{Child, Command};

/// Find a free TCP port to avoid collisions when tests run in parallel.
///
/// There is an unavoidable TOCTOU window: the port is free when we drop the
/// listener but could be claimed by another process before the caller binds
/// it. The pick→spawn gap is small and the ephemeral range is large, so
/// collisions are vanishingly rare — this is the standard bind-to-`:0` trick.
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
        .join(if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        })
        // EXE_SUFFIX or the path misses `cascadia.exe` on Windows.
        .join(format!("cascadia{}", std::env::consts::EXE_SUFFIX))
}

pub struct CascadiaProc {
    pub port: u16,
    pub child: Child,
}

impl CascadiaProc {
    pub async fn spawn_mock_with_api() -> Self {
        let port = pick_free_port();
        let bin = binary_path();
        assert!(
            bin.exists(),
            "cascadia binary not built — run `cargo build -p cascadia` first ({:?})",
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
            .expect("spawn cascadia binary");
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

impl Drop for CascadiaProc {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

/// A multi-stage mock pipeline spawned on loopback (one process per stage).
///
/// Exercises the full sharded path — pipeline wiring, the activation transport
/// between stages, and request routing — with the mock engine, so it needs no
/// model weights or GPU and is deterministic. This is the single-machine
/// version of a sharded deployment; the cross-node version is driven by the
/// fleet (see `multi_node_pipeline_via_fleet`).
pub struct MockPipeline {
    pub api_port: u16,
    pub stages: Vec<Child>,
}

impl MockPipeline {
    /// Spawn an N-stage mock pipeline on 127.0.0.1, downstream-first (each
    /// upstream needs its downstream listening before it connects).
    pub async fn spawn(stages: usize) -> Self {
        let listen_addrs: Vec<String> = (0..stages)
            .map(|_| format!("127.0.0.1:{}", pick_free_port()))
            .collect();
        Self::spawn_with_links(stages, &listen_addrs).await
    }

    /// Spawn an N-stage mock pipeline linked over Unix domain sockets
    /// (`unix:/path.sock` --listen/--next, #17) instead of loopback TCP.
    /// Unix-only.
    #[cfg(unix)]
    pub async fn spawn_uds(stages: usize) -> Self {
        let dir = std::env::temp_dir().join("cascadia-e2e-uds");
        let _ = std::fs::create_dir_all(&dir);
        let pid = std::process::id();
        let listen_addrs: Vec<String> = (0..stages)
            .map(|rank| format!("unix:{}/p{pid}-r{rank}.sock", dir.display()))
            .collect();
        Self::spawn_with_links(stages, &listen_addrs).await
    }

    /// Shared spawner: `listen_addrs[rank]` is the address stage `rank`
    /// listens on (TCP `host:port` or `unix:/path.sock`); stage N-1's
    /// `--next` is `listen_addrs[N]`.
    async fn spawn_with_links(stages: usize, listen_addrs: &[String]) -> Self {
        assert!(stages >= 1, "need at least one stage");
        assert_eq!(listen_addrs.len(), stages);
        let bin = binary_path();
        assert!(bin.exists(), "cascadia binary not built ({:?})", bin);

        let api_port = pick_free_port();
        let mut procs = Vec::with_capacity(stages);

        for rank in (0..stages).rev() {
            let is_first = rank == 0;
            let is_last = rank == stages - 1;
            let mut args: Vec<String> = vec![
                "worker".into(),
                "--rank".into(),
                rank.to_string(),
                "--total".into(),
                stages.to_string(),
                "--engine".into(),
                "mock".into(),
                "--model".into(),
                "mock-model".into(),
                "--log-level".into(),
                "warn".into(),
            ];
            if !is_first {
                args.push("--listen".into());
                args.push(listen_addrs[rank].clone());
            }
            if !is_last {
                args.push("--next".into());
                args.push(listen_addrs[rank + 1].clone());
            }
            if is_first {
                args.push("--api".into());
                args.push(format!("127.0.0.1:{api_port}"));
            }
            let child = Command::new(&bin)
                .args(&args)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .kill_on_drop(true)
                .spawn()
                .expect("spawn pipeline stage");
            procs.push(child);
            // No inter-stage sleep needed: stages are spawned downstream-first
            // and the activation transport retries its connect (every 500ms up
            // to DEFAULT_CONNECT_TIMEOUT = 30s), so an upstream simply waits out
            // the gap until its downstream is bound and accepting. The same
            // retry contract covers a not-yet-bound unix socket path.
        }
        Self {
            api_port,
            stages: procs,
        }
    }

    pub async fn wait_for_health(&self, timeout: Duration) -> bool {
        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{}/health", self.api_port);
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Ok(r) = client.get(&url).send().await {
                if r.status().is_success() {
                    return true;
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        false
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{}", self.api_port, path)
    }
}

impl Drop for MockPipeline {
    fn drop(&mut self) {
        for c in &mut self.stages {
            let _ = c.start_kill();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn binary_serves_health_then_models_then_chat() {
        let proc = CascadiaProc::spawn_mock_with_api().await;
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

        // /metrics (#16): the real binary serves Prometheus text exposition
        // and the chat request above populated the request + generation
        // metrics.
        let r = client
            .get(proc.url("/metrics"))
            .send()
            .await
            .expect("get metrics");
        assert!(r.status().is_success(), "metrics status: {}", r.status());
        let content_type = r
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        assert!(
            content_type.starts_with("text/plain"),
            "metrics content-type: {content_type}"
        );
        let text = r.text().await.unwrap();
        assert!(
            text.contains(
                "cascadia_http_requests_total{endpoint=\"/v1/chat/completions\",status=\"200\"}"
            ),
            "metrics missing chat request counter:\n{text}"
        );
        assert!(
            text.contains("cascadia_tokens_generated_total"),
            "metrics missing token counter:\n{text}"
        );
        assert!(
            text.contains("cascadia_generation_ttft_seconds_bucket"),
            "metrics missing TTFT histogram:\n{text}"
        );
        assert!(
            text.contains("cascadia_inflight_tasks 0"),
            "in-flight gauge should be back to 0 after the request:\n{text}"
        );
    }

    #[tokio::test]
    async fn binary_supports_concurrent_requests() {
        let proc = CascadiaProc::spawn_mock_with_api().await;
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
            assert!(
                r.status().is_success(),
                "concurrent req status {}",
                r.status()
            );
        }
    }

    /// Sharded pipeline on one machine (loopback): two mock stages wired
    /// rank 0 (entry+API) -> rank 1. Exercises the activation transport and
    /// request routing through the chain without weights/GPU. Runs in CI.
    #[tokio::test]
    async fn multi_stage_mock_pipeline_loopback() {
        let pipe = MockPipeline::spawn(2).await;
        assert!(
            pipe.wait_for_health(Duration::from_secs(20)).await,
            "2-stage pipeline entry did not become healthy"
        );

        let client = reqwest::Client::new();
        let body = serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "alpha bravo charlie"}],
            "max_tokens": 2,
            "stream": false,
        });
        let r = client
            .post(pipe.url("/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .expect("post chat to pipeline entry");
        assert!(
            r.status().is_success(),
            "pipeline chat status: {}",
            r.status()
        );
        let v: serde_json::Value = r.json().await.unwrap();
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string");
        assert!(!content.is_empty(), "pipeline produced empty completion");
    }

    /// Ask a pipeline's API for one deterministic mock completion; returns
    /// the assistant content.
    async fn chat_once(pipe: &MockPipeline, prompt: &str) -> String {
        let client = reqwest::Client::new();
        let body = serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4,
            "stream": false,
        });
        let r = client
            .post(pipe.url("/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .expect("post chat");
        assert!(r.status().is_success(), "chat status: {}", r.status());
        let v: serde_json::Value = r.json().await.unwrap();
        v["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string")
            .to_string()
    }

    /// #17: a 2-stage worker chain configured with `unix:` --listen/--next
    /// comes up healthy and answers identically to the TCP chain.
    ///
    /// Scope honesty: the MOCK engine's connect/configure_listen are
    /// no-ops, so no activation tensors cross the socket here — this test
    /// pins the CLI surface (unix address parsing, worker startup, no
    /// bogus TCP probe bind, identical output). The actual frame protocol
    /// over a real UnixStream is exercised by
    /// `cascadia-engine-sparse-moe/tests/dist_wire.rs::
    /// uds_sequence_reset_then_forward_then_token` (engine wire) and the
    /// `uds_*` tests in cascadia-transport (framing/timeout semantics).
    #[cfg(unix)]
    #[tokio::test]
    async fn uds_pipeline_matches_tcp_pipeline_output() {
        let tcp = MockPipeline::spawn(2).await;
        let uds = MockPipeline::spawn_uds(2).await;
        assert!(
            tcp.wait_for_health(Duration::from_secs(20)).await,
            "TCP 2-stage pipeline did not become healthy"
        );
        assert!(
            uds.wait_for_health(Duration::from_secs(20)).await,
            "UDS 2-stage pipeline did not become healthy"
        );

        let prompt = "alpha bravo charlie delta echo foxtrot";
        let tcp_out = chat_once(&tcp, prompt).await;
        let uds_out = chat_once(&uds, prompt).await;
        assert!(!uds_out.is_empty(), "UDS chain produced empty completion");
        assert_eq!(
            tcp_out, uds_out,
            "UDS chain must produce byte-identical output to the TCP chain"
        );
    }

    /// Cross-node sharded e2e: brings up an N-stage topology across real fleet
    /// nodes via the control plane, runs a request through it, tears down.
    /// Gated behind CASCADIA_MULTINODE=1 (live fleet only). Requires env:
    ///   CASCADIA_FLEET_CONTROL  http://<control-plane>:8765
    ///   CASCADIA_FLEET_TOKEN    shared fleet token
    ///   CASCADIA_FLEET_NODES    ordered node_ids, comma-separated (rank 0 first)
    ///   CASCADIA_REMOTE_BIN     path to the cascadia binary on the nodes
    /// Optional (defaults to a mock run; set for a real sharded model):
    ///   CASCADIA_ENGINE  engine, default "mock" (e.g. "ov-runtime")
    ///   CASCADIA_MODEL   model id / shard dir on the nodes, default "mock-model"
    ///   CASCADIA_DEVICE  device for OV engines, e.g. "GPU" (omitted for mock)
    #[tokio::test]
    async fn multi_node_pipeline_via_fleet() {
        if std::env::var("CASCADIA_MULTINODE").ok().as_deref() != Some("1") {
            eprintln!("skipping multi_node_pipeline_via_fleet (set CASCADIA_MULTINODE=1)");
            return;
        }
        let control = std::env::var("CASCADIA_FLEET_CONTROL").expect("CASCADIA_FLEET_CONTROL");
        let token = std::env::var("CASCADIA_FLEET_TOKEN").expect("CASCADIA_FLEET_TOKEN");
        let nodes: Vec<String> = std::env::var("CASCADIA_FLEET_NODES")
            .expect("CASCADIA_FLEET_NODES")
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let remote_bin = std::env::var("CASCADIA_REMOTE_BIN")
            .unwrap_or_else(|_| "C:/cascadia/cascadia.exe".into());
        // Defaults give a mock run; set CASCADIA_ENGINE/MODEL/DEVICE for a real
        // sharded model (e.g. ov-runtime + a shard dir + GPU).
        let engine = std::env::var("CASCADIA_ENGINE").unwrap_or_else(|_| "mock".into());
        let model = std::env::var("CASCADIA_MODEL").unwrap_or_else(|_| "mock-model".into());
        let device_flag = match std::env::var("CASCADIA_DEVICE") {
            Ok(d) if !d.is_empty() => format!("--device {d}"),
            _ => String::new(),
        };
        let name = "e2e-multinode";
        let command = format!(
            "{remote_bin} worker --engine {engine} --model {model} {device_flag} --log-level warn \
             --rank {{rank}} --total {{total}} {{listen_flag}} {{next_flag}} {{api_flag}}"
        );
        let client = reqwest::Client::new();

        let up = client
            .post(format!("{control}/topology/up"))
            .bearer_auth(&token)
            .json(&serde_json::json!({"name": name, "nodes": nodes, "command": command}))
            .send()
            .await
            .expect("topology up");
        assert!(up.status().is_success(), "topology up: {}", up.status());
        let up: serde_json::Value = up.json().await.unwrap();
        let entry = up["entry_url"]
            .as_str()
            .expect("entry_url in response")
            .to_string();

        // Run the request, capturing the outcome so teardown always happens.
        let outcome: Result<(), String> = async {
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut healthy = false;
            while Instant::now() < deadline {
                if let Ok(r) = client.get(format!("{entry}/health")).send().await {
                    if r.status().is_success() {
                        healthy = true;
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            if !healthy {
                return Err(format!("pipeline entry {entry} never became healthy"));
            }
            let r = client
                .post(format!("{entry}/v1/chat/completions"))
                .json(&serde_json::json!({
                    "model": model,
                    "messages": [{"role": "user", "content": "hello world"}],
                    "max_tokens": 2,
                }))
                .send()
                .await
                .map_err(|e| e.to_string())?;
            if !r.status().is_success() {
                return Err(format!("chat status {}", r.status()));
            }
            Ok(())
        }
        .await;

        let _ = client
            .post(format!("{control}/topology/down"))
            .bearer_auth(&token)
            .json(&serde_json::json!({"name": name, "nodes": nodes}))
            .send()
            .await;

        outcome.expect("multi-node pipeline request failed");
    }
}
