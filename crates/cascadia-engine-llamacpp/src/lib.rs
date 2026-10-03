//! External llama.cpp engine — subprocess-backed.
//!
//! Cascadia's other engines load weights in-process. This one spawns a
//! `llama-server` child (GGUF, SYCL backend) and proxies its OpenAI API
//! through the [`Engine`] contract:
//!
//! - `Builder::load`   -> spawn child, poll `/health` until ready
//! - `Engine::submit`  -> queue the task
//! - `Engine::step`    -> drive one SSE completion stream, emit Chunks
//! - `Engine::cancel`  -> mark the active task dead; `close` kills the child
//!
//! Elastic posture (paper contract, obligations O1-O4):
//! `--elastic` on this engine maps to `GGML_STREAM_WEIGHTS=1` on the child —
//! device-side O1: weights are never resident on the GPU, they are pread from
//! the GGUF into a fixed slot pool per forward pass. This is NOT the
//! host-heap interposer (`cascadia-elastic`); that posture still applies to
//! the child's host allocations via inherited LD_PRELOAD when the parent was
//! launched with --elastic. The two mechanisms are orthogonal and additive:
//! file-backed host pages + never-resident device weights.
//!
//! Known bounds (measured 2026-10-03, experiments/2026-10-03-elasticity-matrix):
//! decode scales as ~8.1 GB/s / model_GB; KV is still reserved (O2 open);
//! fused ops and SYCL graphs are disabled while streaming; single-device.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::stream;

/// Configuration for the llama.cpp subprocess engine.
pub struct LlamaCppConfig {
    /// Path to a llama-server binary (a build with the stream-weights patch
    /// for elastic mode; a stock binary runs fine with elastic off).
    pub llama_bin: PathBuf,
    /// GGUF model file.
    pub model: PathBuf,
    /// SYCL device name passed to `--device` (e.g. "SYCL0", "SYCL1").
    pub device: String,
    /// Context size (-c).
    pub ctx: u32,
    /// GPU layers (-ngl). Default 99 = full offload.
    pub ngl: u32,
    /// Elastic posture: GGML_STREAM_WEIGHTS=1 on the child.
    pub elastic: bool,
    /// Extra raw args appended verbatim to the server command line.
    pub extra_args: Vec<String>,
}

/// Minimal HTTP health probe — a GET /health against 127.0.0.1:port is a
/// success when the status line reads 200. Deliberately not reqwest: this
/// is called from `load()` on a tokio worker.
fn health_ok(port: u16) -> bool {
    let mut s = match TcpStream::connect(("127.0.0.1", port)) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = s.set_write_timeout(Some(Duration::from_secs(2)));
    if s.write_all(b"GET /health HTTP/1.0\r\nHost: x\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut buf = [0u8; 64];
    let n = s.read(&mut buf).unwrap_or(0);
    String::from_utf8_lossy(&buf[..n]).starts_with("HTTP/1.1 200")
        || String::from_utf8_lossy(&buf[..n]).starts_with("HTTP/1.0 200")
}

/// Builder: spawns and readiness-checks the llama-server child.
pub struct LlamaCppBuilder {
    cfg: LlamaCppConfig,
    port: u16,
    child: Option<Child>,
}

impl LlamaCppBuilder {
    pub fn new(cfg: LlamaCppConfig) -> Self {
        Self {
            cfg,
            port: 0,
            child: None,
        }
    }

    /// Pick a free loopback port by binding :0 and dropping the listener.
    fn pick_port() -> EngineResult<u16> {
        let l = TcpListener::bind("127.0.0.1:0")
            .map_err(|e| EngineError::Backend(format!("port probe: {e}")))?;
        Ok(l.local_addr().unwrap().port())
    }
}

#[async_trait]
impl Builder for LlamaCppBuilder {
    async fn connect(&mut self, peers: PeerLayout) -> EngineResult<()> {
        // Single-stage external server: no upstream/downstream peers.
        if peers.upstream.is_some() || peers.downstream.is_some() {
            return Err(EngineError::PeerRejected(
                "sycl-llama is single-stage; run with --total 1".into(),
            ));
        }
        Ok(())
    }

    async fn load(&mut self, _shard: ShardSpec) -> EngineResult<LoadStream> {
        self.port = Self::pick_port()?;
        let mut cmd = Command::new(&self.cfg.llama_bin);
        cmd.arg("-m")
            .arg(&self.cfg.model)
            .arg("--device")
            .arg(&self.cfg.device)
            .arg("-ngl")
            .arg(self.cfg.ngl.to_string())
            .arg("-c")
            .arg(self.cfg.ctx.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(self.port.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        if self.cfg.elastic {
            // Device-side O1: weights stay on disk, streamed per layer.
            cmd.env("GGML_STREAM_WEIGHTS", "1");
        }
        for a in &self.cfg.extra_args {
            cmd.arg(a);
        }
        let child = cmd
            .spawn()
            .map_err(|e| EngineError::Backend(format!("spawn llama-server: {e}")))?;
        self.child = Some(child);

        // Poll /health until the server is serving or the child exits.
        // Raw TcpStream — this runs on a tokio worker where reqwest::blocking
        // cannot (nested runtime panics); the SSE reader is a plain thread.
        let port = self.port;
        let t0 = Instant::now();
        let deadline = Duration::from_secs(120);
        loop {
            if let Some(c) = self.child.as_mut() {
                if let Ok(Some(st)) = c.try_wait() {
                    return Err(EngineError::Backend(format!(
                        "llama-server exited during load: {st}"
                    )));
                }
            }
            if health_ok(port) {
                break;
            }
            if t0.elapsed() > deadline {
                return Err(EngineError::Backend("llama-server health timeout".into()));
            }
            std::thread::sleep(Duration::from_secs(2));
        }
        tracing::info!(port, "llama-server ready");

        let evs = vec![
            LoadProgress::message("llama-server spawned"),
            LoadProgress::message(if self.cfg.elastic {
                "elastic posture: GGML_STREAM_WEIGHTS=1 (device O1 — weights not resident)"
            } else {
                "stock posture: weights resident"
            }),
            LoadProgress::message("health OK — engine ready"),
            LoadProgress::ready(),
        ];
        Ok(Box::pin(stream::iter(evs)))
    }

    fn build(mut self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        let child = self.child.take().ok_or(EngineError::NotLoaded)?;
        Ok(Box::new(LlamaCppEngine {
            base: format!("http://127.0.0.1:{}", self.port),
            child,
            pending: Vec::new(),
            active: None,
            rx: None,
            cancelled: Arc::new(AtomicBool::new(false)),
            socket: None,
        }))
    }

    fn close(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

impl Drop for LlamaCppBuilder {
    fn drop(&mut self) {
        self.close();
    }
}

/// Live engine: one queued task at a time against the child server.
/// A blocking SSE reader thread converts the stream into Chunks; step()
/// drains whatever has arrived.
pub struct LlamaCppEngine {
    base: String,
    child: Child,
    pending: Vec<GenerationTask>,
    active: Option<TaskId>,
    rx: Option<Receiver<EngineResult<(TaskId, Chunk)>>>,
    cancelled: Arc<AtomicBool>,
    socket: Option<TcpStream>,
}

impl LlamaCppEngine {
    /// Start the SSE reader for `task` against the child's completion API.
    /// Raw TcpStream + manual chunked decode: reqwest::blocking cannot be
    /// trusted anywhere near a process that also runs a tokio runtime (its
    /// inner runtime panics on ambient-runtime detection).
    fn start_stream(&mut self, task: GenerationTask) {
        let tid = task.task_id.clone();
        let port: u16 = self
            .base
            .rsplit(':')
            .next()
            .and_then(|p| p.parse().ok())
            .unwrap_or(0);
        let cancelled = self.cancelled.clone();
        cancelled.store(false, Ordering::Relaxed);
        let (tx, rx) = channel();
        self.rx = Some(rx);

        // Structured chat turns go to the child's chat endpoint so
        // llama-server renders the model's own (GGUF) chat template.
        // Prompt-only tasks (the legacy /v1/completions path) keep the
        // completions endpoint with the pre-rendered prompt.
        let (endpoint, body) = if task.messages.is_empty() {
            let body = serde_json::json!({
                "prompt": task.prompt,
                "n_predict": task.max_tokens,
                "temperature": task.temperature,
                "stream": true,
            });
            ("/v1/completions", body)
        } else {
            let messages: Vec<serde_json::Value> = task
                .messages
                .iter()
                .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
                .collect();
            let body = serde_json::json!({
                "messages": messages,
                "max_tokens": task.max_tokens,
                "temperature": task.temperature,
                "stream": true,
            });
            ("/v1/chat/completions", body)
        };
        let body = body.to_string();
        let mut s = match TcpStream::connect(("127.0.0.1", port)) {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.send(Err(EngineError::Backend(format!("connect: {e}"))));
                return;
            }
        };
        let _ = s.set_read_timeout(Some(Duration::from_secs(60)));
        self.socket = s.try_clone().ok();

        std::thread::spawn(move || {
            let send = |r: EngineResult<(TaskId, Chunk)>| {
                let _ = tx.send(r);
            };
            let req = format!(
                "POST {} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                endpoint,
                body.len(),
                body
            );
            if s.write_all(req.as_bytes()).is_err() {
                send(Err(EngineError::Backend("write request".into())));
                return;
            }
            let mut reader = BufReader::new(s);
            // Skip response headers.
            let mut chunked = false;
            let mut status = String::new();
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    send(Err(EngineError::Backend("no response".into())));
                    return;
                }
                if status.is_empty() {
                    status = line.trim().to_string();
                }
                if line.to_lowercase().contains("transfer-encoding:")
                    && line.to_lowercase().contains("chunked")
                {
                    chunked = true;
                }
                if line == "\r\n" {
                    break;
                }
            }
            if !status.contains(" 200") {
                send(Err(EngineError::Backend(format!("server {status}"))));
                return;
            }
            // Body is one byte stream of SSE `data: ...` lines, possibly
            // chunk-framed. Read chunk sizes as hex when chunked.
            let mut token_id: i64 = 0;
            let mut leftover = String::new();
            loop {
                if cancelled.load(Ordering::Relaxed) {
                    return;
                }
                let mut line = String::new();
                if chunked {
                    // read until CRLF-terminated hex size line
                    let mut sz = String::new();
                    if reader.read_line(&mut sz).unwrap_or(0) == 0 {
                        return;
                    }
                    let n = usize::from_str_radix(sz.trim(), 16).unwrap_or(0);
                    if n == 0 {
                        return; // terminal chunk
                    }
                    let mut buf = vec![0u8; n];
                    if reader.read_exact(&mut buf).is_err() {
                        return;
                    }
                    leftover.push_str(&String::from_utf8_lossy(&buf));
                    // swallow the chunk's trailing CRLF
                    let mut crlf = [0u8; 2];
                    let _ = reader.read_exact(&mut crlf);
                    while let Some(idx) = leftover.find('\n') {
                        line = leftover[..idx].to_string();
                        leftover = leftover[idx + 1..].to_string();
                        if !handle_sse_line(&line, &tid, &mut token_id, &send) {
                            return;
                        }
                    }
                    continue;
                }
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    return;
                }
                if !handle_sse_line(&line, &tid, &mut token_id, &send) {
                    return;
                }
            }
        });
    }
}

/// Parse one SSE `data: ...` line into a Chunk; returns false when the
/// stream is terminal (`[DONE]` or a stop chunk was emitted).
fn handle_sse_line(
    line: &str,
    tid: &TaskId,
    token_id: &mut i64,
    send: &dyn Fn(EngineResult<(TaskId, Chunk)>),
) -> bool {
    let line = line.trim_end_matches('\r');
    let Some(data) = line.strip_prefix("data: ") else {
        return true;
    };
    if data.trim() == "[DONE]" {
        let mut c = Chunk::token(tid.clone(), *token_id, "");
        c.is_final = true;
        send(Ok((tid.clone(), c)));
        return false;
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
        return true;
    };
    let text = v["choices"][0]["delta"]["content"].as_str().or(v["choices"][0]["text"].as_str()).unwrap_or("");
    let stop_str = v["choices"][0]["finish_reason"].as_str();
    let stop = stop_str.is_some();
    // Let stop-only chunks through even if empty
    if text.is_empty() && !stop {
        return true;
    }
    let mut c = Chunk::token(tid.clone(), *token_id, text.to_string());
    *token_id += 1;
    if stop {
        c.is_final = true;
        c.finish_reason = match stop_str {
            Some("length") => Some(cascadia_types::FinishReason::Length),
            Some("stop") | _ => Some(cascadia_types::FinishReason::Stop),
        };
    }
    send(Ok((tid.clone(), c)));
    !stop
}

impl Engine for LlamaCppEngine {
    fn warmup(&mut self) {}

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if self.pending.iter().any(|t| t.task_id == task.task_id)
            || self.active.as_ref() == Some(&task.task_id)
        {
            return Ok(());
        }
        self.pending.push(task);
        Ok(())
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        // Pick up a task if idle.
        if self.active.is_none() {
            if let Some(task) = self.pending.first().cloned() {
                self.active = Some(task.task_id.clone());
                self.pending.remove(0);
                self.start_stream(task);
            } else {
                return Ok(vec![]);
            }
        }
        // Wait for the next chunk. step() must not spin empty
        let mut out = Vec::new();
        let mut done = false;
        if let Some(rx) = &self.rx {
            match rx.recv_timeout(Duration::from_secs(300)) {
                Ok(item) => {
                    let (tid, chunk) = item?;
                    if chunk.is_final {
                        done = true;
                    }
                    out.push((tid.clone(), chunk));
                    // drain anything else already buffered for this task
                    while let Ok(item) = rx.try_recv() {
                        let (tid, chunk) = item?;
                        if chunk.is_final {
                            done = true;
                        }
                        out.push((tid, chunk));
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    let tid = self.active.take().unwrap();
                    let mut c = Chunk::token(tid.clone(), 0, "");
                    c.is_final = true;
                    c.error = Some("completion stream ended without [DONE]".into());
                    out.push((tid, c));
                    done = true;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    let tid = self.active.take().unwrap();
                    let mut c = Chunk::token(tid.clone(), 0, "");
                    c.is_final = true;
                    c.error = Some("completion stream stalled >300s".into());
                    out.push((tid, c));
                    done = true;
                }
            }
        }
        if done {
            self.active = None;
            self.rx = None;
        }
        // Child died mid-stream?
        if self.active.is_some() {
            if let Ok(Some(_)) = self.child.try_wait() {
                let tid = self.active.take().unwrap();
                let mut c = Chunk::token(tid.clone(), 0, "");
                c.is_final = true;
                c.error = Some("llama-server exited mid-task".into());
                out.push((tid, c));
                self.rx = None;
            }
        }
        Ok(out)
    }

    fn cancel(&mut self, task_id: &TaskId) {
        self.pending.retain(|t| &t.task_id != task_id);
        if self.active.as_ref() == Some(task_id) {
            self.cancelled.store(true, Ordering::Relaxed);
            if let Some(s) = self.socket.take() {
                let _ = s.shutdown(std::net::Shutdown::Both);
            }
            self.active = None;
            self.rx = None;
        }
    }

    fn close(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
        if let Some(s) = self.socket.take() {
            let _ = s.shutdown(std::net::Shutdown::Both);
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for LlamaCppEngine {
    fn drop(&mut self) {
        self.close();
    }
}
