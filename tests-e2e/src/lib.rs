//! End-to-end binary integration tests for the cascadia Rust port.
//!
//! Spawns the built `cascadia` binary, waits for /health, exercises the
//! OpenAI-compatible API, then tears down. Validates the full chain:
//! CLI parsing -> Runner.start() -> Engine -> API handlers -> wire JSON.
//!
//! Run via:
//!     cargo build -p cascadia
//!     cargo test -p cascadia-tests-e2e -- --test-threads=1

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
        .join(if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        })
        .join("cascadia")
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
        assert!(stages >= 1, "need at least one stage");
        let bin = binary_path();
        assert!(bin.exists(), "cascadia binary not built ({:?})", bin);

        let api_port = pick_free_port();
        let listen_ports: Vec<u16> = (0..stages).map(|_| pick_free_port()).collect();
        let mut procs = Vec::with_capacity(stages);

        for rank in (0..stages).rev() {
            let is_first = rank == 0;
            let is_last = rank == stages - 1;
            let mut args: Vec<String> = vec![
                "worker".into(),
                "--rank".into(), rank.to_string(),
                "--total".into(), stages.to_string(),
                "--engine".into(), "mock".into(),
                "--model".into(), "mock-model".into(),
                "--log-level".into(), "warn".into(),
            ];
            if !is_first {
                args.push("--listen".into());
                args.push(format!("127.0.0.1:{}", listen_ports[rank]));
            }
            if !is_last {
                args.push("--next".into());
                args.push(format!("127.0.0.1:{}", listen_ports[rank + 1]));
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
            // Give the stage a moment to bind before its upstream connects.
            tokio::time::sleep(Duration::from_millis(400)).await;
        }
        Self { api_port, stages: procs }
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
        assert!(r.status().is_success(), "pipeline chat status: {}", r.status());
        let v: serde_json::Value = r.json().await.unwrap();
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string");
        assert!(!content.is_empty(), "pipeline produced empty completion");
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
        let remote_bin =
            std::env::var("CASCADIA_REMOTE_BIN").unwrap_or_else(|_| "C:/cascadia/cascadia.exe".into());
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
        let entry = up["entry_url"].as_str().expect("entry_url in response").to_string();

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
                    "model": "mock-model",
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
