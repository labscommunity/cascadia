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

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_types::{
    Chunk, FinishReason, GenerationTask, LoadProgress, PeerLayout, SamplingParams, ShardSpec,
    TaskId,
};
use futures::stream;

/// VRAM budget for `--elastic` partial weight streaming, mapped to the
/// child's `GGML_STREAM_VRAM_MB` env: `auto` = keep whatever fits resident,
/// a GiB number = that much resident weight budget, `0` = stream every
/// layer (the original all-or-nothing mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElasticVram {
    Auto,
    MiB(u64),
}

impl std::str::FromStr for ElasticVram {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        if raw.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        let gib: f64 = raw.parse().map_err(|_| {
            format!("invalid --elastic-vram '{raw}' (expected 'auto' or a GiB number)")
        })?;
        if !gib.is_finite() || gib < 0.0 {
            return Err(format!(
                "invalid --elastic-vram '{raw}' (expected 'auto' or a GiB number)"
            ));
        }
        Ok(Self::MiB((gib * 1024.0).round() as u64))
    }
}

impl std::fmt::Display for ElasticVram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::MiB(v) => write!(f, "{v}"),
        }
    }
}

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
    /// Resident-weight budget for elastic mode (GGML_STREAM_VRAM_MB).
    /// Only meaningful with `elastic`.
    pub elastic_vram: ElasticVram,
    /// Expected co-tenant count for elastic mode (GGML_STREAM_VRAM_SHARE):
    /// caps the child's automatic resident-weight budget at a 1/N share of
    /// the card. Only applies to `elastic_vram == Auto`; an explicit MiB
    /// budget or GGML_STREAM_RESIDENT_LAYERS wins on the child. This is a
    /// load-time cap, not a fairness guarantee: it does not account for
    /// KV, slot pools, expert caches, or already-running instances.
    pub elastic_share: Option<u32>,
    /// Experimental placement: layer tensors whose names match this regex
    /// live in the backend's pinned HOST buffer and the GPU computes on them
    /// in place (`-ot '<regex>=SYCL_Host'`, with mmap off so the loader
    /// honours the host buffer). Needs a llama-server built with
    /// `patches/llama.cpp/0003-sycl-host-buffer-compute.patch`. On a
    /// discrete card this reads the overflow over the link every token with
    /// stable pointers (fused ops and graphs stay on) instead of streaming
    /// it through a staging copy; on a UMA iGPU it is slower than device
    /// placement and frees no device-pool memory (see
    /// `docs/perf/sycl-elastic/placement-b390.md`).
    pub host_layers: Option<String>,
    /// Speculative decoding with the model's own MTP (nextn) head:
    /// `--spec-type draft-mtp` on the child. The target model verifies every
    /// drafted token, so one weight pass yields several tokens: this is the
    /// biggest decode lever when weights stream (each pass re-reads them).
    /// Needs a GGUF that carries nextn layers (e.g. Qwen3.8-27B).
    pub mtp: bool,
    /// Extra raw args appended verbatim to the server command line.
    pub extra_args: Vec<String>,
    /// Health deadline per load attempt. `None` = auto: 60 s + 8 s per GiB
    /// of model file (300 s when the file metadata is unreadable).
    pub load_timeout: Option<Duration>,
    /// Extra load attempts after the first fails (child exit or health
    /// timeout). xe copy-engine resets hang the load path intermittently;
    /// the next attempt on a fresh port nearly always succeeds. Default 1.
    pub load_retries: u32,
}

/// How long one `step()` waits for the child before handing the runtime
/// thread back with a progress marker.
const STEP_POLL: Duration = Duration::from_millis(50);
/// No chunk from the child for this long (wall clock since the last one, or
/// since the request started) fails the task.
const STALL_LIMIT: Duration = Duration::from_secs(300);

/// Tie the child's life to cascadia's: a cascadia that is SIGKILLed or
/// crashes must not leave a `llama-server` holding the model's VRAM and the
/// port (every later `--elastic-vram auto` load would then see less free
/// memory and silently keep fewer layers resident).
///
/// Linux: `PR_SET_PDEATHSIG` delivers SIGKILL when the *thread* that
/// spawned the child exits — not the process. Engine load/retry can run on
/// tokio blocking-pool threads that retire after seconds of idle, so a
/// naive pre_exec spawn would kill a healthy child whenever that caller
/// thread goes away. All Linux spawns therefore go through one
/// process-wide thread that never exits (`spawn_on_parent_thread`), making
/// the child's life track cascadia's own. Windows: a Job Object with
/// KILL_ON_JOB_CLOSE, closed by the OS with the last handle.
#[cfg(target_os = "linux")]
fn arm_death_guard(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    // SAFETY: prctl is async-signal-safe and the closure touches no heap.
    // parent_pid is the process pid (same on every thread); the child
    // re-checks it post-fork because a subreaper (systemd --user) can
    // reparent us between fork and prctl — test `!=`, not `== 1`.
    let parent_pid = std::process::id() as i32;
    unsafe {
        cmd.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            // The parent may already have died between fork and prctl.
            if libc::getppid() != parent_pid {
                libc::_exit(1);
            }
            Ok(())
        });
    }
}

/// Spawn `cmd` on the process-wide spawner thread (see above). The thread
/// is created once and never exits: it is what makes PR_SET_PDEATHSIG mean
/// "when cascadia dies" instead of "when the calling thread dies".
#[cfg(target_os = "linux")]
fn spawn_on_parent_thread(cmd: Command) -> std::io::Result<Child> {
    use std::io::Error;
    use std::sync::mpsc::{channel, Sender};
    use std::sync::OnceLock;
    type Job = (Command, Sender<std::io::Result<Child>>);
    static TX: OnceLock<Sender<Job>> = OnceLock::new();
    let tx = TX.get_or_init(|| {
        let (tx, rx) = channel::<Job>();
        std::thread::Builder::new()
            .name("llamacpp-child-spawner".into())
            .spawn(move || {
                while let Ok((mut cmd, reply)) = rx.recv() {
                    let r = cmd.spawn();
                    // caller went away; don't leak the child
                    if let Err(std::sync::mpsc::SendError(Ok(mut c))) = reply.send(r) {
                        let _ = c.kill();
                    }
                }
            })
            .expect("spawn llamacpp child spawner thread");
        tx
    });
    let (reply_tx, reply_rx) = channel();
    tx.send((cmd, reply_tx))
        .map_err(|_| Error::other("child spawner thread gone"))?;
    reply_rx
        .recv()
        .map_err(|_| Error::other("child spawner thread gone"))?
}

#[cfg(not(target_os = "linux"))]
fn arm_death_guard(_cmd: &mut Command) {}

/// Windows half of the death guard: the job the child is assigned to after
/// spawn. Keeping the handle in the engine ties the child's life to ours.
#[cfg(windows)]
struct ChildJob(windows_sys::Win32::Foundation::HANDLE);

// SAFETY: a job object handle is a kernel object reference, valid from any
// thread; the engine moves it between runtime threads but never shares it.
#[cfg(windows)]
unsafe impl Send for ChildJob {}

#[cfg(windows)]
impl ChildJob {
    fn new_kill_on_close() -> Option<Self> {
        use windows_sys::Win32::System::JobObjects::{
            CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        };
        // SAFETY: plain Win32 calls with a zeroed, correctly sized struct.
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return None;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 {
                windows_sys::Win32::Foundation::CloseHandle(job);
                return None;
            }
            Some(ChildJob(job))
        }
    }

    fn assign(&self, child: &Child) -> bool {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::JobObjects::AssignProcessToJobObject;
        // SAFETY: both handles are live for the call.
        unsafe { AssignProcessToJobObject(self.0, child.as_raw_handle() as _) != 0 }
    }
}

#[cfg(windows)]
impl Drop for ChildJob {
    fn drop(&mut self) {
        // SAFETY: the handle was returned by CreateJobObjectW.
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

/// Environment the host-side `--elastic` posture sets on cascadia itself
/// (#132) and must not reach the child: `llama-server` has its own device
/// side of the posture, its staging buffer must stay plain anonymous memory
/// (Level Zero cannot memcpy from file-mapped pages), and the interposer
/// would build a second retention pool in the child.
const HOST_ELASTIC_ENV: [&str; 5] = [
    "CASCADIA_ELASTIC_ACTIVE",
    "ELASTIC_DIR",
    "ELASTIC_MIN_MB",
    "ELASTIC_POOL_MB",
    "ELASTIC_SO_PATH",
];

/// Remove cascadia's interposer entries (`libcascadia_elastic.<pid>.so`,
/// written to TMPDIR by `cascadia_elastic::activate`) from an LD_PRELOAD
/// value; `None` when nothing else remains. glibc accepts both colons and
/// spaces as separators, so split on both — a space-joined list parsed as
/// one entry would match the filter and drop the user's own preloads too.
fn filter_ld_preload(v: &str) -> Option<String> {
    let kept: Vec<&str> = v
        .split([':', ' '])
        .filter(|e| {
            !e.is_empty()
                && !Path::new(e)
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("libcascadia_elastic."))
        })
        .collect();
    if kept.is_empty() {
        None
    } else {
        Some(kept.join(":"))
    }
}

/// Scrub the child's environment: the host interposer's variables never
/// cross, and ambient `GGML_STREAM_WEIGHTS` / `GGML_STREAM_VRAM_MB` cannot
/// stream without `--elastic` (the engine sets both itself when it is on).
/// `GGML_STREAM_RESIDENT_LAYERS` is a documented child knob that overrides
/// the budget, so it passes through — with a warning when it will.
/// `elastic_share` is the `--elastic-share` value, if any: when it is set the
/// engine writes `GGML_STREAM_VRAM_SHARE` itself (after this scrub), so an
/// ambient value is overridden rather than merely warned about.
fn scrub_child_env(cmd: &mut Command, elastic: bool, elastic_share: Option<u32>) {
    for k in HOST_ELASTIC_ENV {
        cmd.env_remove(k);
    }
    // LD_PRELOAD: strip only cascadia's interposer entries (the private
    // libcascadia_elastic.<pid>.so activate() prepends); a user's own
    // preloads still reach the child, and the var drops only when nothing
    // remains.
    if let Some(v) = std::env::var_os("LD_PRELOAD") {
        match filter_ld_preload(&v.to_string_lossy()) {
            Some(kept) => {
                cmd.env("LD_PRELOAD", kept);
            }
            None => {
                cmd.env_remove("LD_PRELOAD");
            }
        }
    }
    if !elastic {
        cmd.env_remove("GGML_STREAM_WEIGHTS");
        cmd.env_remove("GGML_STREAM_VRAM_MB");
        cmd.env_remove("GGML_STREAM_RESIDENT_LAYERS");
        cmd.env_remove("GGML_STREAM_VRAM_SHARE");
        if std::env::var_os("GGML_STREAM_VRAM_SHARE").is_some() {
            tracing::warn!(
                "sycl-llama: GGML_STREAM_VRAM_SHARE is set but --elastic is off; dropping it from the child"
            );
        }
    } else {
        if let Ok(n) = std::env::var("GGML_STREAM_RESIDENT_LAYERS") {
            if !n.trim().is_empty() {
                tracing::warn!(
                    "sycl-llama: GGML_STREAM_RESIDENT_LAYERS={n} in the environment overrides --elastic-vram on the child"
                );
            }
        }
        // an ambient GGML_STREAM_VRAM_SHARE passes through like
        // RESIDENT_LAYERS does; --elastic-share overrides it when given, so
        // only warn when no explicit share will overwrite it
        if elastic_share.is_none()
            && std::env::var("GGML_STREAM_VRAM_SHARE").is_ok_and(|v| !v.trim().is_empty())
        {
            tracing::warn!(
                "sycl-llama: ambient GGML_STREAM_VRAM_SHARE reaches the child; --elastic-share sets it explicitly"
            );
        }
    }
}

/// Byte string whose presence in llama-server or libggml-base marks a build
/// with the weight-streaming patch (`GGML_STREAM_WEIGHTS` env gate).
const STREAM_MARKER: &[u8] = b"GGML_STREAM_WEIGHTS";

/// Marker for router-aware MoE expert streaming (0002 patch): MoE models
/// stream only the experts the router selects instead of every expert.
const EXPERT_MARKER: &[u8] = b"GGML_STREAM_EXPERT_CACHE_MB";

/// Resolve the llama-server binary.
///
/// Order: `--llama-bin` > `$CASCADIA_LLAMA_BIN` > `llama-server`
/// (`llama-server.exe` on Windows) found on `PATH`. An explicit path that
/// does not exist is an error, not a fall-through — a typo must not silently
/// pick up a different build.
pub fn resolve_llama_bin(
    flag: Option<&str>,
    env: Option<OsString>,
    path_var: Option<OsString>,
) -> Result<PathBuf, String> {
    if let Some(p) = flag {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Ok(pb);
        }
        return Err(format!("--llama-bin {p}: no such file"));
    }
    if let Some(v) = env.filter(|v| !v.is_empty()) {
        let pb = PathBuf::from(&v);
        if pb.is_file() {
            return Ok(pb);
        }
        return Err(format!("CASCADIA_LLAMA_BIN={}: no such file", pb.display()));
    }
    let name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    if let Some(paths) = path_var {
        for dir in std::env::split_paths(&paths) {
            let cand = dir.join(name);
            if cand.is_file() {
                return Ok(cand);
            }
        }
    }
    Err(format!(
        "no llama-server binary found: pass --llama-bin <path>, set \
         CASCADIA_LLAMA_BIN, or put {name} on PATH; build one with \
         scripts/build-llama-stream.sh"
    ))
}

/// `-ot '<regex>=SYCL_Host' --load-mode none` for [`LlamaCppConfig::host_layers`],
/// or nothing. mmap must be off: with it on, the loader silently replaces a
/// host-buffer placement by the CPU buffer (llama-model-loader.cpp, "avoid
/// using a host buffer when using mmap") and the CPU backend computes the
/// layer. The regex is used as given; `=` and `,` are the override
/// syntax's separators, so they are rejected here rather than mis-parsed
/// by the child.
pub fn llama_host_layer_args(host_layers: Option<&str>) -> Vec<String> {
    match host_layers.map(str::trim).filter(|r| !r.is_empty()) {
        Some(re) if re.contains('=') || re.contains(',') => {
            eprintln!(
                "sycl-llama: --llama-host-layers {re:?} contains '=' or ',' (the -ot separators); ignored"
            );
            Vec::new()
        }
        // `--load-mode none` is what the deprecated `--no-mmap` maps to
        Some(re) => vec![
            "--load-mode".into(),
            "none".into(),
            "-ot".into(),
            format!("{re}=SYCL_Host"),
        ],
        None => Vec::new(),
    }
}

/// `--spec-type draft-mtp` for [`LlamaCppConfig::mtp`], or nothing. The
/// draft context runs against the same loaded weights, so no second model
/// is read. The target model decides every accepted token; its verification
/// runs as a small batch, so greedy text can differ from plain decoding at
/// near-ties (batched kernels round differently), as any batch-size change
/// does.
pub fn llama_spec_args(mtp: bool) -> Vec<String> {
    if mtp {
        vec!["--spec-type".into(), "draft-mtp".into()]
    } else {
        Vec::new()
    }
}

/// Map a cascadia `--device` value to llama.cpp `--device` args + the ngl
/// actually in force. `GPU` -> `SYCL0`, `GPU.N` -> `SYCLN`, `CPU` (any case)
/// -> `--device none` with ngl forced to 0; anything else (SYCL1, Vulkan0,
/// `SYCL0,SYCL1`, ...) passes verbatim.
pub fn llama_device_args(device: &str, ngl: u32) -> (Vec<String>, u32) {
    if device.eq_ignore_ascii_case("cpu") {
        return (vec!["--device".into(), "none".into()], 0);
    }
    let mapped = if device == "GPU" {
        Some("SYCL0".to_string())
    } else if let Some(idx) = device.strip_prefix("GPU.") {
        idx.parse::<u32>().ok().map(|n| format!("SYCL{n}"))
    } else {
        None
    };
    (
        vec!["--device".into(), mapped.unwrap_or_else(|| device.into())],
        ngl,
    )
}

/// Result of probing a llama-server build for weight-streaming support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamWeightsSupport {
    /// `GGML_STREAM_WEIGHTS` marker found in the binary or a libggml-base
    /// next to it — `--elastic` will do device-side streaming.
    Present,
    /// No libggml-base beside the binary and the binary lacks the marker —
    /// support could not be verified (static build? foreign layout?).
    Unknown,
}

fn file_contains(path: &Path, needle: &[u8]) -> bool {
    std::fs::read(path)
        .map(|b| b.windows(needle.len()).any(|w| w == needle))
        .unwrap_or(false)
}

/// Shared-library names that may carry the streaming markers beside the
/// binary, on any OS: Linux/macOS `libggml-{base,sycl}.*`, Windows
/// `ggml-{base,sycl}*.dll`.
fn is_ggml_stream_lib(name: &str) -> bool {
    name.starts_with("libggml-base")
        || name.starts_with("libggml-sycl")
        || ((name.starts_with("ggml-base") || name.starts_with("ggml-sycl"))
            && name.ends_with(".dll"))
}

/// Preflight: does this llama-server build understand GGML_STREAM_WEIGHTS?
/// Scans the binary itself plus `libggml-{base,sycl}*` / `ggml-*.dll` files
/// in the same directory (the env gate lives in libggml-base; distro builds
/// link it as a shared object).
///
/// - marker anywhere -> `Present`
/// - a ggml lib exists but nothing contains the marker -> `Err` (a stock
///   build: `--elastic` would silently not reduce VRAM)
/// - no ggml lib at all and the binary lacks the marker -> `Unknown`
///   (caller warns and continues)
pub fn probe_stream_weights(bin: &Path) -> Result<StreamWeightsSupport, String> {
    if file_contains(bin, STREAM_MARKER) {
        return Ok(StreamWeightsSupport::Present);
    }
    let dir = bin.parent().unwrap_or_else(|| Path::new("."));
    let mut found_lib = false;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if is_ggml_stream_lib(&name) {
                found_lib = true;
                if file_contains(&e.path(), STREAM_MARKER) {
                    return Ok(StreamWeightsSupport::Present);
                }
            }
        }
    }
    if found_lib {
        Err("this llama-server build has no weight-streaming support; \
             --elastic would not reduce VRAM. Build one with \
             scripts/build-llama-stream.sh (or drop --elastic)"
            .to_string())
    } else {
        Ok(StreamWeightsSupport::Unknown)
    }
}

/// Preflight: does this build also have router-aware MoE expert streaming
/// (the 0002 patch)? Purely informational — a 0001-only build still streams
/// correctly, it just reads every expert per token on MoE models.
pub fn probe_expert_streaming(bin: &Path) -> bool {
    if file_contains(bin, EXPERT_MARKER) {
        return true;
    }
    let dir = bin.parent().unwrap_or_else(|| Path::new("."));
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten().any(|e| {
                is_ggml_stream_lib(&e.file_name().to_string_lossy())
                    && file_contains(&e.path(), EXPERT_MARKER)
            })
        })
        .unwrap_or(false)
}

/// Auto load deadline: 60 s + 8 s per GiB of model file; 300 s when the
/// metadata is unreadable (a spawn error will surface the real cause).
fn auto_load_timeout(model: &Path) -> Duration {
    match std::fs::metadata(model) {
        Ok(m) => Duration::from_secs(60 + 8 * (m.len() >> 30)),
        Err(_) => Duration::from_secs(300),
    }
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
    let status = String::from_utf8_lossy(&buf[..n]);
    status.starts_with("HTTP/1.1 200") || status.starts_with("HTTP/1.0 200")
}

/// Last lines of child stderr, kept for error messages. The forwarding
/// thread pushes every line; the engine reads it when a load attempt fails.
type StderrTail = Arc<Mutex<VecDeque<String>>>;

/// `stream-weights:` lines the patched child prints during load. Kept
/// separately from the stderr tail: the tail is only 40 lines, so an early
/// split line would scroll out before ready on chatty models.
type StreamLines = Arc<Mutex<Vec<String>>>;

/// llama-server arguments the engine owns: llama-server's parser is
/// last-wins, so a `--llama-args` token could silently rebind the host or
/// port the health checks never reach, or point at another model/device.
/// Each entry maps the reserved flag to the cascadia-side way to say it.
const RESERVED_LLAMA_ARGS: &[(&str, &str)] = &[
    ("-m", "the model comes from the positional MODEL argument"),
    (
        "--model",
        "the model comes from the positional MODEL argument",
    ),
    ("-mu", "the model comes from the positional MODEL argument"),
    (
        "--model-url",
        "the model comes from the positional MODEL argument",
    ),
    ("-hf", "the model comes from the positional MODEL argument"),
    ("-hfr", "the model comes from the positional MODEL argument"),
    (
        "--hf-repo",
        "the model comes from the positional MODEL argument",
    ),
    ("--host", "the engine binds 127.0.0.1 itself"),
    ("--port", "the engine picks a free loopback port itself"),
    ("--device", "use cascadia --device instead"),
    ("-dev", "use cascadia --device instead"),
    ("-ngl", "use --llama-ngl instead"),
    ("--gpu-layers", "use --llama-ngl instead"),
    ("--n-gpu-layers", "use --llama-ngl instead"),
    ("-c", "use --llama-ctx instead"),
    ("--ctx-size", "use --llama-ctx instead"),
];

/// Reject `extra_args` tokens that shadow engine-owned flags — a token equal
/// to a reserved flag or of the `<flag>=<value>` form.
pub fn validate_extra_args(extra_args: &[String]) -> Result<(), String> {
    for arg in extra_args {
        for (reserved, hint) in RESERVED_LLAMA_ARGS {
            if arg == reserved || arg.starts_with(&format!("{reserved}=")) {
                return Err(format!("--llama-args '{arg}' shadows '{reserved}': {hint}"));
            }
        }
    }
    Ok(())
}

/// Builder: spawns and readiness-checks the llama-server child.
pub struct LlamaCppBuilder {
    cfg: LlamaCppConfig,
    port: u16,
    child: Option<Child>,
    /// Windows: the kill-on-close job the child is assigned to at spawn;
    /// handed to the engine with the child.
    #[cfg(windows)]
    child_job: Option<ChildJob>,
    stderr_tail: StderrTail,
    stream_lines: StreamLines,
    stderr_drain: Option<std::thread::JoinHandle<()>>,
}

impl LlamaCppBuilder {
    pub fn new(cfg: LlamaCppConfig) -> Self {
        Self {
            cfg,
            port: 0,
            child: None,
            #[cfg(windows)]
            child_job: None,
            stderr_tail: Arc::new(Mutex::new(VecDeque::new())),
            stream_lines: Arc::new(Mutex::new(Vec::new())),
            stderr_drain: None,
        }
    }

    /// Pick a free loopback port by binding :0 and dropping the listener.
    fn pick_port() -> EngineResult<u16> {
        let l = TcpListener::bind("127.0.0.1:0")
            .map_err(|e| EngineError::Backend(format!("port probe: {e}")))?;
        Ok(l.local_addr().unwrap().port())
    }

    /// Spawn the llama-server child on `self.port`. stderr is piped and a
    /// drainer thread forwards every line to our stderr while keeping the
    /// last 40 in a ring buffer for error messages — the drainer must run
    /// for the child's whole life or a full pipe would block the child.
    fn spawn_child(&mut self) -> Result<(), String> {
        let (dev_args, ngl) = llama_device_args(&self.cfg.device, self.cfg.ngl);
        let mut cmd = Command::new(&self.cfg.llama_bin);
        cmd.arg("-m")
            .arg(&self.cfg.model)
            .args(&dev_args)
            .arg("-ngl")
            .arg(ngl.to_string())
            .arg("-c")
            .arg(self.cfg.ctx.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(self.port.to_string())
            // tool calls only render through the jinja chat template; a
            // --no-jinja in --llama-args still wins (last flag applies)
            .arg("--jinja")
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        scrub_child_env(&mut cmd, self.cfg.elastic, self.cfg.elastic_share);
        arm_death_guard(&mut cmd);
        if self.cfg.elastic {
            // Device-side O1: weights stay on disk, streamed per layer.
            cmd.env("GGML_STREAM_WEIGHTS", "1");
            cmd.env("GGML_STREAM_VRAM_MB", self.cfg.elastic_vram.to_string());
            // --elastic-share caps only the 'auto' budget; the child
            // applies it to nothing else (explicit MiB wins, and an
            // ambient share passes through untouched when no flag is set)
            if let Some(n) = self.cfg.elastic_share {
                if matches!(self.cfg.elastic_vram, ElasticVram::Auto) {
                    cmd.env("GGML_STREAM_VRAM_SHARE", n.to_string());
                } else {
                    tracing::warn!(
                        "sycl-llama: --elastic-share {n} ignored: it caps only the 'auto' VRAM budget"
                    );
                }
            }
        } else if self.cfg.elastic_share.is_some() {
            tracing::warn!("sycl-llama: --elastic-share has no effect without --elastic");
        }
        // Router-aware MoE streaming lives entirely in the child (0002
        // patch): selective expert loading is automatic when streaming; a
        // hot-expert cache is opt-in via env so --elastic-vram's
        // resident-layer semantics stay exact.
        if let Some(mb) = std::env::var("CASCADIA_EXPERT_CACHE_MB")
            .ok()
            .filter(|v| !v.is_empty())
        {
            cmd.env("GGML_STREAM_EXPERT_CACHE_MB", mb);
        }
        cmd.args(llama_spec_args(self.cfg.mtp));
        cmd.args(llama_host_layer_args(self.cfg.host_layers.as_deref()));
        for a in self.cfg.extra_args.iter().filter(|a| !a.is_empty()) {
            cmd.arg(a);
        }
        #[cfg(target_os = "linux")]
        let mut child =
            spawn_on_parent_thread(cmd).map_err(|e| format!("spawn llama-server: {e}"))?;
        #[cfg(not(target_os = "linux"))]
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("spawn llama-server: {e}"))?;
        #[cfg(windows)]
        {
            match ChildJob::new_kill_on_close() {
                Some(job) if job.assign(&child) => self.child_job = Some(job),
                _ => tracing::warn!(
                    "sycl-llama: could not tie llama-server to a job object; a killed cascadia may orphan it"
                ),
            }
        }
        self.stderr_tail.lock().unwrap().clear();
        self.stream_lines.lock().unwrap().clear();
        if let Some(err) = child.stderr.take() {
            let tail = self.stderr_tail.clone();
            let slines = self.stream_lines.clone();
            self.stderr_drain = Some(std::thread::spawn(move || {
                for line in BufReader::new(err).lines().map_while(Result::ok) {
                    eprintln!("{line}");
                    if line.starts_with("stream-weights: warning:") {
                        tracing::warn!("llama-server: {line}");
                    } else if line.starts_with("stream-weights:") {
                        tracing::info!("llama-server: {line}");
                    }
                    if line.starts_with("stream-weights:") {
                        let mut s = slines.lock().unwrap();
                        if s.len() >= 8 {
                            s.remove(0);
                        }
                        s.push(line.clone());
                    }
                    let mut t = tail.lock().unwrap();
                    if t.len() >= 40 {
                        t.pop_front();
                    }
                    t.push_back(line);
                }
            }));
        }
        self.child = Some(child);
        Ok(())
    }

    /// Poll `/health` until the server answers, the child exits, or the
    /// deadline hits. Raw TcpStream — this runs on a tokio worker where
    /// reqwest::blocking cannot (nested runtime panics); the SSE reader
    /// is a plain thread.
    fn wait_healthy(&mut self, timeout: Duration) -> Result<(), String> {
        let port = self.port;
        let t0 = Instant::now();
        loop {
            if let Some(c) = self.child.as_mut() {
                if let Ok(Some(st)) = c.try_wait() {
                    return Err(format!("llama-server exited during load: {st}"));
                }
            }
            if health_ok(port) {
                return Ok(());
            }
            if t0.elapsed() > timeout {
                return Err(format!("llama-server health timeout ({timeout:?})"));
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    /// Kill + reap the child, then give the stderr drainer a short grace to
    /// reach EOF and flush the tail. A grandchild may still hold the pipe
    /// open, in which case the drainer is detached and keeps forwarding.
    fn kill_child(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        if let Some(t) = self.stderr_drain.take() {
            for _ in 0..50 {
                if t.is_finished() {
                    let _ = t.join();
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    /// The last `n` captured child stderr lines, joined.
    fn tail_lines(&self, n: usize) -> String {
        let t = self.stderr_tail.lock().unwrap();
        t.iter()
            .skip(t.len().saturating_sub(n))
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// `stream-weights:` lines captured by the stderr drainer (split + warnings).
fn stream_lines(buf: &[String]) -> Vec<String> {
    buf.iter()
        .filter(|l| l.starts_with("stream-weights:"))
        .cloned()
        .collect()
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
        let timeout = self
            .cfg
            .load_timeout
            .unwrap_or_else(|| auto_load_timeout(&self.cfg.model));
        if self.cfg.elastic && self.cfg.device.eq_ignore_ascii_case("cpu") {
            tracing::warn!(
                "sycl-llama --elastic on a CPU device: GGML_STREAM_WEIGHTS \
                 streams weights only on the device path — it does nothing \
                 for `--device none` (the host interposer still applies)"
            );
        }
        let attempts = 1 + self.cfg.load_retries;
        let mut last_err = String::new();
        for attempt in 1..=attempts {
            if attempt > 1 {
                tracing::warn!(
                    attempt,
                    "retrying llama-server load ({last_err}); intermittent load hangs \
                     correlate with xe copy-engine resets: check \
                     `dmesg | grep -i 'engine reset'`"
                );
                std::thread::sleep(Duration::from_secs(3));
            }
            self.port = Self::pick_port()?;
            match self.spawn_child().and_then(|()| self.wait_healthy(timeout)) {
                Ok(()) => break,
                Err(e) => {
                    self.kill_child();
                    let tail = self.tail_lines(20);
                    last_err = if tail.is_empty() {
                        e
                    } else {
                        format!("{e}; child stderr (last lines): {tail}")
                    };
                }
            }
            if attempt == attempts {
                return Err(EngineError::Backend(format!(
                    "llama-server failed to load after {attempts} attempt(s): {last_err}"
                )));
            }
        }
        tracing::info!(port = self.port, "llama-server ready");

        let mut evs = vec![
            LoadProgress::message("llama-server spawned"),
            LoadProgress::message(if self.cfg.elastic {
                format!(
                    "elastic posture: GGML_STREAM_WEIGHTS=1, \
                     GGML_STREAM_VRAM_MB={}{} (device weight streaming)",
                    self.cfg.elastic_vram,
                    self.cfg
                        .elastic_share
                        .map(|n| format!(", GGML_STREAM_VRAM_SHARE={n}"))
                        .unwrap_or_default()
                )
            } else {
                "stock posture: weights resident".to_string()
            }),
        ];
        // Surface the child's own streaming split lines (warning + split).
        for line in stream_lines(&self.stream_lines.lock().unwrap()) {
            evs.push(LoadProgress::message(line));
        }
        evs.extend([
            LoadProgress::message("health OK — engine ready"),
            LoadProgress::ready(),
        ]);
        Ok(Box::pin(stream::iter(evs)))
    }

    fn build(mut self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        let child = self.child.take().ok_or(EngineError::NotLoaded)?;
        // Detach the stderr drainer: it must keep draining until the child's
        // stderr hits EOF (child exit), so Drop must not join it here.
        let _ = self.stderr_drain.take();
        Ok(Box::new(LlamaCppEngine {
            base: format!("http://127.0.0.1:{}", self.port),
            child,
            pending: Vec::new(),
            active: None,
            rx: None,
            last_chunk_at: Instant::now(),
            #[cfg(windows)]
            child_job: self.child_job.take(),
            cancelled: Arc::new(AtomicBool::new(false)),
            socket: None,
        }))
    }

    fn close(&mut self) {
        self.kill_child();
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
    /// When the active task last produced a real chunk (or started): the
    /// 300 s stall limit is measured against this, not against one blocking
    /// receive, so `step()` can poll briefly and let the runtime thread go.
    last_chunk_at: Instant,
    cancelled: Arc<AtomicBool>,
    socket: Option<TcpStream>,
    /// Windows: the kill-on-close job the child lives in (see `ChildJob`);
    /// held only so its Drop closes the job with the engine.
    #[cfg(windows)]
    #[allow(dead_code)]
    child_job: Option<ChildJob>,
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
        self.last_chunk_at = Instant::now();

        // Structured chat turns go to the child's chat endpoint so
        // llama-server renders the model's own (GGUF) chat template.
        // Prompt-only tasks (the legacy /v1/completions path) keep the
        // completions endpoint with the pre-rendered prompt.
        let (endpoint, mut body) = if task.messages.is_empty() {
            let body = serde_json::json!({
                "prompt": task.prompt,
                "n_predict": task.max_tokens,
                "temperature": task.temperature,
                "stream": true,
                "stream_options": {"include_usage": true},
            });
            ("/v1/completions", body)
        } else {
            let messages: Vec<serde_json::Value> = task
                .messages
                .iter()
                .map(|m| {
                    let mut v = serde_json::json!({"role": m.role, "content": m.content});
                    // an assistant tool_call turn carries null content, not ""
                    if m.content.is_empty() && m.tool_calls.is_some() {
                        v["content"] = serde_json::Value::Null;
                    }
                    if let Some(calls) = &m.tool_calls {
                        v["tool_calls"] = calls.clone();
                    }
                    if let Some(id) = &m.tool_call_id {
                        v["tool_call_id"] = serde_json::json!(id);
                    }
                    if let Some(n) = &m.name {
                        v["name"] = serde_json::json!(n);
                    }
                    v
                })
                .collect();
            let mut body = serde_json::json!({
                "messages": messages,
                "max_tokens": task.max_tokens,
                "temperature": task.temperature,
                "stream": true,
                "stream_options": {"include_usage": true},
                // Relay the API-resolved thinking toggle to the child's own
                // template (Qwen-style templates read enable_thinking;
                // others ignore the unused kwarg).
                "chat_template_kwargs": {"enable_thinking": task.enable_thinking},
            });
            if let Some(tools) = &task.tools {
                body["tools"] = tools.clone();
                // tool_choice only means anything alongside tools. The API
                // already 400s a malformed one; this guards other callers and
                // fails the task as a final error chunk, like a connect
                // failure below.
                if let Err(e) = apply_tool_choice(&mut body, task.tool_choice.as_ref()) {
                    let _ = tx.send(Err(EngineError::InvalidConfig(e)));
                    return;
                }
            }
            ("/v1/chat/completions", body)
        };
        apply_sampling(&mut body, &task.sampling);
        let body = body.to_string();
        let mut s = match TcpStream::connect(("127.0.0.1", port)) {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.send(Err(EngineError::Backend(format!("connect: {e}"))));
                return;
            }
        };
        let _ = s.set_read_timeout(Some(Duration::from_secs(300)));
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
            let mut st = SseState::default();
            // byte buffer, decoded per complete line: a chunk boundary can
            // split a UTF-8 multibyte character, so never convert per chunk
            let mut leftover = Vec::<u8>::new();
            'body: loop {
                if cancelled.load(Ordering::Relaxed) {
                    return;
                }
                let mut line = String::new();
                if chunked {
                    // read until CRLF-terminated hex size line
                    let mut sz = String::new();
                    if reader.read_line(&mut sz).unwrap_or(0) == 0 {
                        break 'body;
                    }
                    let n = usize::from_str_radix(sz.trim(), 16).unwrap_or(0);
                    if n == 0 {
                        break 'body; // terminal chunk
                    }
                    let mut buf = vec![0u8; n];
                    if reader.read_exact(&mut buf).is_err() {
                        break 'body;
                    }
                    leftover.extend_from_slice(&buf);
                    // swallow the chunk's trailing CRLF
                    let mut crlf = [0u8; 2];
                    let _ = reader.read_exact(&mut crlf);
                    // '\n' (0x0A) never occurs inside a UTF-8 multibyte
                    // sequence, so byte-splitting on it is boundary-safe
                    while let Some(idx) = leftover.iter().position(|&b| b == b'\n') {
                        line = String::from_utf8_lossy(&leftover[..idx])
                            .trim_end()
                            .to_string();
                        leftover.drain(..idx + 1);
                        if !handle_sse_line(&line, &tid, &mut st, &send) {
                            return;
                        }
                    }
                    continue;
                }
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    break 'body;
                }
                if !handle_sse_line(&line, &tid, &mut st, &send) {
                    return;
                }
            }
            // Clean EOF without a [DONE] line: deliver a pending final so
            // finish_reason/usage still reach the caller.
            for block in st.tool_call_blocks() {
                send(Ok((
                    tid.clone(),
                    Chunk::token(tid.clone(), st.token_id, block),
                )));
            }
            if !st.sent_final && st.finish_reason.is_some() {
                let mut c = Chunk::token(tid.clone(), st.token_id, "");
                c.is_final = true;
                c.finish_reason = st.finish_reason;
                c.prompt_tokens = st.prompt_tokens;
                send(Ok((tid.clone(), c)));
            }
        });
    }
}

/// Forward the non-default sampling knobs into a request body. Omitted
/// fields let llama-server apply its own defaults (identical for these
/// keys, but keeps the wire clean for fields the child may treat
/// differently when present).
fn apply_sampling(body: &mut serde_json::Value, sp: &SamplingParams) {
    let d = SamplingParams::default();
    let Some(o) = body.as_object_mut() else {
        return;
    };
    if sp.top_p != d.top_p {
        o.insert("top_p".into(), sp.top_p.into());
    }
    if sp.top_k != d.top_k {
        o.insert("top_k".into(), sp.top_k.into());
    }
    if sp.frequency_penalty != d.frequency_penalty {
        o.insert("frequency_penalty".into(), sp.frequency_penalty.into());
    }
    if sp.presence_penalty != d.presence_penalty {
        o.insert("presence_penalty".into(), sp.presence_penalty.into());
    }
    if let Some(seed) = sp.seed {
        o.insert("seed".into(), seed.into());
    }
    if !sp.stop.is_empty() {
        o.insert("stop".into(), sp.stop.clone().into());
    }
}

/// Translate an OpenAI `tool_choice` into the form llama-server parses.
///
/// llama-server understands only the string forms `"auto"`, `"none"` and
/// `"required"`; a JSON object silently degrades to `"auto"`, so a caller
/// asking for one specific function would instead let the model call any
/// tool. A named choice is therefore narrowed to that single entry of
/// `body["tools"]` plus `"required"`. `"none"` drops `tools` altogether:
/// llama-server still renders them into the prompt under `"none"`, and the
/// model then writes a call as plain text (measured on Qwen2.5-1.5B), so
/// the only reliable "no tools" is a prompt without them (vLLM does the
/// same). `None` leaves `body` untouched (the child's own default is
/// `"auto"`). An unknown string or malformed object is an error, never a
/// silent pass-through. Call only when tools were sent.
fn apply_tool_choice(
    body: &mut serde_json::Value,
    choice: Option<&serde_json::Value>,
) -> Result<(), String> {
    let Some(choice) = choice else {
        return Ok(());
    };
    match choice {
        serde_json::Value::String(s) => match s.as_str() {
            "none" => {
                if let Some(obj) = body.as_object_mut() {
                    obj.remove("tools");
                }
                Ok(())
            }
            "auto" | "required" => {
                body["tool_choice"] = serde_json::json!(s);
                Ok(())
            }
            other => Err(format!(
                "tool_choice string '{other}' is not one of \"auto\", \"none\", \"required\""
            )),
        },
        serde_json::Value::Object(_) => {
            let name = choice
                .get("type")
                .and_then(|t| t.as_str())
                .filter(|t| *t == "function")
                .and_then(|_| choice.get("function"))
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .filter(|n| !n.is_empty())
                .ok_or_else(|| {
                    format!(
                        "tool_choice object {choice} is not the OpenAI \
                         {{\"type\":\"function\",\"function\":{{\"name\":N}}}} form"
                    )
                })?;
            let tools = body
                .get("tools")
                .and_then(|t| t.as_array())
                .ok_or_else(|| {
                    format!("tool_choice names function '{name}' but no tools were sent")
                })?;
            let kept: Vec<serde_json::Value> = tools
                .iter()
                .filter(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        == Some(name)
                })
                .cloned()
                .collect();
            if kept.is_empty() {
                return Err(format!(
                    "tool_choice names function '{name}', which is not in tools"
                ));
            }
            body["tools"] = serde_json::Value::Array(kept);
            body["tool_choice"] = serde_json::json!("required");
            Ok(())
        }
        other => Err(format!(
            "tool_choice {other} is not a string or an OpenAI function object"
        )),
    }
}

/// Per-stream SSE parsing state. The finish chunk is delayed until
/// `[DONE]` so the trailing `usage` object (stream_options) and the
/// finish_reason both land on the final Chunk the API reads.
#[derive(Default)]
struct SseState {
    token_id: i64,
    finish_reason: Option<FinishReason>,
    prompt_tokens: Option<u32>,
    sent_final: bool,
    /// OpenAI `delta.tool_calls` fragments accumulated by call index.
    tool_calls: std::collections::BTreeMap<u64, ToolCallParts>,
}

/// One streaming tool call assembled from `delta.tool_calls` fragments.
#[derive(Default)]
struct ToolCallParts {
    name: String,
    arguments: String,
}

impl SseState {
    /// The accumulated calls re-encoded as `<tool_call>` text — the Hermes
    /// form the API's parse_tool_calls already understands — in call order.
    /// Drains the map: the stream-end paths ([DONE] and clean EOF) can both
    /// run, and a call must never be emitted twice.
    fn tool_call_blocks(&mut self) -> Vec<String> {
        std::mem::take(&mut self.tool_calls)
            .values()
            .filter(|p| !p.name.is_empty())
            .map(|p| {
                // arguments as the parsed JSON object when it is one, else
                // the raw string — parse_tool_calls accepts both
                let args: serde_json::Value = serde_json::from_str(&p.arguments)
                    .unwrap_or_else(|_| serde_json::Value::String(p.arguments.clone()));
                format!(
                    "<tool_call>{}</tool_call>",
                    serde_json::json!({"name": p.name, "arguments": args})
                )
            })
            .collect()
    }
}

/// Parse one SSE `data: ...` line; returns false when the stream is
/// terminal (`[DONE]` was handled and the final chunk emitted).
fn handle_sse_line(
    line: &str,
    tid: &TaskId,
    st: &mut SseState,
    send: &dyn Fn(EngineResult<(TaskId, Chunk)>),
) -> bool {
    let line = line.trim_end_matches('\r');
    let Some(data) = line.strip_prefix("data: ") else {
        return true;
    };
    if data.trim() == "[DONE]" {
        // Emit any assembled tool calls as text just before the final
        // chunk so the API's tool-call machinery sees them in-order.
        for block in st.tool_call_blocks() {
            send(Ok((
                tid.clone(),
                Chunk::token(tid.clone(), st.token_id, block),
            )));
        }
        let mut c = Chunk::token(tid.clone(), st.token_id, "");
        c.is_final = true;
        c.finish_reason = st.finish_reason;
        c.prompt_tokens = st.prompt_tokens;
        st.sent_final = true;
        send(Ok((tid.clone(), c)));
        return false;
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
        return true;
    };
    // The child can emit an error object mid-stream; surface it instead of
    // letting the stream end as an ambiguous EOF.
    if v.get("error").is_some() {
        let msg = v["error"]["message"]
            .as_str()
            .or_else(|| v["error"].as_str())
            .unwrap_or("llama-server stream error")
            .to_string();
        send(Ok((tid.clone(), Chunk::error(tid.clone(), msg))));
        st.sent_final = true;
        return false;
    }
    // stream_options usage tail (empty choices) or a timings block on a
    // content chunk — either can carry the prompt token count.
    if let Some(n) = v["usage"]["prompt_tokens"].as_u64() {
        st.prompt_tokens = Some(n as u32);
    } else if let Some(n) = v["timings"]["prompt_n"].as_u64() {
        st.prompt_tokens = Some(n as u32);
    }
    let text = v["choices"][0]["delta"]["content"]
        .as_str()
        .or(v["choices"][0]["text"].as_str())
        .unwrap_or("");
    // Structured tool calls arrive as OpenAI delta.tool_calls fragments
    // (index + id + function.name + function.arguments pieces, usually
    // without delta.content); accumulate them for the stream-end emit.
    let mut active = false;
    if let Some(calls) = v["choices"][0]["delta"]["tool_calls"].as_array() {
        active = true;
        for call in calls {
            let idx = call["index"].as_u64().unwrap_or(0);
            let parts = st.tool_calls.entry(idx).or_default();
            if let Some(f) = call.get("function") {
                if let Some(n) = f["name"].as_str() {
                    parts.name.push_str(n);
                }
                if let Some(a) = f["arguments"].as_str() {
                    parts.arguments.push_str(a);
                }
            }
        }
    }
    // reasoning_content deltas carry no visible text but prove the child is
    // alive; the stall clock must hear them or a long reasoning-only stream
    // gets killed at the limit while still generating
    if v["choices"][0]["delta"]["reasoning_content"]
        .as_str()
        .is_some_and(|s| !s.is_empty())
        || v["choices"][0]["delta"]["reasoning"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
    {
        active = true;
    }
    let stop_str = v["choices"][0]["finish_reason"].as_str();
    if let Some(r) = stop_str {
        st.finish_reason = Some(if r == "length" {
            FinishReason::Length
        } else {
            FinishReason::Stop
        });
    }
    // Empty chunks carry no content; a stop-only chunk only updates the
    // recorded finish_reason (emitted on the [DONE] final). Anything that
    // carried real upstream activity still pings the stall clock.
    if text.is_empty() {
        if active {
            send(Ok((tid.clone(), Chunk::progress(tid.clone()))));
        }
        return true;
    }
    let c = Chunk::token(tid.clone(), st.token_id, text.to_string());
    st.token_id += 1;
    send(Ok((tid.clone(), c)));
    true
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
        // Wait briefly for the next chunk. `step()` runs on a runtime thread
        // (the runner's ChunkStream polls it there), so it must not block for
        // the child's whole prefill or inter-token gap: poll for STEP_POLL and
        // hand back a progress marker when nothing arrived — the runner's
        // no-progress watchdog counts it as work and the API sends nothing
        // for it — while the real stall limit is a wall clock since the last
        // chunk.
        let mut out = Vec::new();
        let mut done = false;
        if let Some(rx) = &self.rx {
            match rx.recv_timeout(STEP_POLL) {
                Ok(item) => {
                    self.last_chunk_at = Instant::now();
                    // A reader-thread failure becomes a final error chunk on
                    // the active task (the API maps it to a 5xx) rather than
                    // an orphaned step() error.
                    let err_chunk = |e: EngineError, active: &Option<TaskId>| {
                        let tid = active.clone().unwrap_or_default();
                        (tid.clone(), Chunk::error(tid, e.to_string()))
                    };
                    let (tid, chunk) = item.unwrap_or_else(|e| err_chunk(e, &self.active));
                    if chunk.is_final {
                        done = true;
                    }
                    out.push((tid.clone(), chunk));
                    // drain anything else already buffered for this task
                    while let Ok(item) = rx.try_recv() {
                        let (tid, chunk) = item.unwrap_or_else(|e| err_chunk(e, &self.active));
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
                    if self.last_chunk_at.elapsed() >= STALL_LIMIT {
                        let tid = self.active.take().unwrap();
                        let mut c = Chunk::token(tid.clone(), 0, "");
                        c.is_final = true;
                        c.error = Some(format!(
                            "completion stream stalled >{}s",
                            STALL_LIMIT.as_secs()
                        ));
                        out.push((tid, c));
                        done = true;
                    } else if let Some(tid) = self.active.clone() {
                        out.push((tid.clone(), Chunk::progress(tid)));
                    }
                }
            }
        }
        if done {
            self.active = None;
            self.rx = None;
            // Shut the request socket down so the SSE reader thread wakes
            // from read_line() now instead of lingering until the child's
            // own read timeout fires (300s) after a stall-side abort.
            if let Some(s) = self.socket.take() {
                let _ = s.shutdown(std::net::Shutdown::Both);
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(
        line: &str,
        tid: &TaskId,
        st: &mut SseState,
    ) -> (bool, Vec<EngineResult<(TaskId, Chunk)>>) {
        let out = std::cell::RefCell::new(Vec::new());
        let send = |r: EngineResult<(TaskId, Chunk)>| out.borrow_mut().push(r);
        let go = handle_sse_line(line, tid, st, &send);
        (go, out.into_inner())
    }

    #[test]
    fn sse_chat_delta_emits_token() {
        let (go, out) = collect(
            r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#,
            &"t1".to_string(),
            &mut SseState::default(),
        );
        assert!(go);
        let c = out[0].as_ref().unwrap().1.clone();
        assert_eq!(c.text, "Hello");
        assert!(!c.is_final);
    }

    #[test]
    fn sse_tool_call_fragments_emit_hermes_block_before_final() {
        let tid = "t1".to_string();
        let mut st = SseState::default();
        // fragmented delta.tool_calls, the way llama-server streams them
        collect(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}"#,
            &tid,
            &mut st,
        );
        collect(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]}}]}"#,
            &tid,
            &mut st,
        );
        collect(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\": \"Paris\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            &tid,
            &mut st,
        );
        let (_go, out) = collect("data: [DONE]", &tid, &mut st);
        let chunks: Vec<Chunk> = out.into_iter().map(|r| r.unwrap().1).collect();
        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0].text,
            r#"<tool_call>{"arguments":{"city":"Paris"},"name":"get_weather"}</tool_call>"#
        );
        assert!(!chunks[0].is_final);
        assert!(chunks[1].is_final);
        // unparseable arguments still surface as a string
        let tid2 = "t2".to_string();
        let mut st2 = SseState::default();
        collect(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"not-json"}}]}}]}"#,
            &tid2,
            &mut st2,
        );
        let (_go, out) = collect("data: [DONE]", &tid2, &mut st2);
        let text = &out[0].as_ref().unwrap().1.text;
        assert_eq!(
            text,
            r#"<tool_call>{"arguments":"not-json","name":"f"}</tool_call>"#
        );
    }

    /// [DONE] emits the assembled calls and drains them; the clean-EOF path
    /// that runs next must not emit them a second time.
    #[test]
    fn sse_tool_calls_emit_once_across_done_and_eof() {
        let tid = "t1".to_string();
        let mut st = SseState::default();
        collect(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
            &tid,
            &mut st,
        );
        let (_go, out) = collect("data: [DONE]", &tid, &mut st);
        assert_eq!(
            out.iter()
                .filter(|r| r.as_ref().unwrap().1.text.contains("<tool_call>"))
                .count(),
            1
        );
        // the EOF path emits whatever remains — nothing left now
        assert!(st.tool_call_blocks().is_empty());
    }

    #[test]
    fn sse_completion_text_emits_token() {
        let (go, out) = collect(
            r#"data: {"choices":[{"text":" world"}]}"#,
            &"t1".to_string(),
            &mut SseState::default(),
        );
        assert!(go);
        assert_eq!(out[0].as_ref().unwrap().1.text, " world");
    }

    #[test]
    fn sse_done_is_final_and_terminal() {
        let mut st = SseState {
            token_id: 3,
            ..Default::default()
        };
        let (go, out) = collect("data: [DONE]", &"t1".to_string(), &mut st);
        assert!(!go);
        let c = out[0].as_ref().unwrap().1.clone();
        assert!(c.is_final);
        assert_eq!(c.token_id, 3);
    }

    #[test]
    fn sse_finish_reason_maps_length_vs_stop() {
        // finish_reason is deferred onto the [DONE] final chunk
        let tid = "t1".to_string();
        let mut st = SseState::default();
        let (go, _out) = collect(
            r#"data: {"choices":[{"delta":{"content":"x"},"finish_reason":"length"}]}"#,
            &tid,
            &mut st,
        );
        assert!(go);
        let (go, out) = collect("data: [DONE]", &tid, &mut st);
        assert!(!go);
        let c = out[0].as_ref().unwrap().1.clone();
        assert!(c.is_final);
        assert_eq!(c.finish_reason, Some(FinishReason::Length));

        let mut st = SseState::default();
        let (go, _out) = collect(
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            &tid,
            &mut st,
        );
        assert!(go);
        let (go, out) = collect("data: [DONE]", &tid, &mut st);
        assert!(!go);
        assert_eq!(
            out[0].as_ref().unwrap().1.finish_reason,
            Some(FinishReason::Stop)
        );
    }

    #[test]
    fn sse_usage_lands_on_final_chunk() {
        let tid = "t1".to_string();
        let mut st = SseState::default();
        let (go, out) = collect(
            r#"data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
            &tid,
            &mut st,
        );
        assert!(go);
        assert_eq!(out.len(), 1);
        let (go, out) = collect(
            r#"data: {"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":1}}"#,
            &tid,
            &mut st,
        );
        assert!(go);
        assert!(out.is_empty());
        let (go, out) = collect("data: [DONE]", &tid, &mut st);
        assert!(!go);
        let c = out[0].as_ref().unwrap().1.clone();
        assert!(c.is_final);
        assert_eq!(c.prompt_tokens, Some(42));
        assert_eq!(c.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn sse_timings_prompt_n_fallback() {
        let tid = "t1".to_string();
        let mut st = SseState::default();
        let (_go, _out) = collect(
            r#"data: {"choices":[{"text":"x","finish_reason":"stop"}],"timings":{"prompt_n":7}}"#,
            &tid,
            &mut st,
        );
        let (_go, out) = collect("data: [DONE]", &tid, &mut st);
        assert_eq!(out[0].as_ref().unwrap().1.prompt_tokens, Some(7));
    }

    #[test]
    fn sse_non_data_and_malformed_lines_are_skipped() {
        let mut st = SseState::default();
        assert!(handle_sse_line(": comment", &"t".into(), &mut st, &|_| {
            panic!()
        }));
        assert!(handle_sse_line("", &"t".into(), &mut st, &|_| panic!()));
        assert!(handle_sse_line(
            "data: {not json",
            &"t".into(),
            &mut st,
            &|_| panic!()
        ));
        // an event line is not data either
        assert!(handle_sse_line("event: x", &"t".into(), &mut st, &|_| {
            panic!()
        }));
    }

    #[test]
    fn elastic_vram_parse() {
        use std::str::FromStr;
        assert_eq!(ElasticVram::from_str("auto").unwrap(), ElasticVram::Auto);
        assert_eq!(ElasticVram::from_str("AUTO").unwrap(), ElasticVram::Auto);
        assert_eq!(ElasticVram::from_str("0").unwrap(), ElasticVram::MiB(0));
        assert_eq!(
            ElasticVram::from_str("12").unwrap(),
            ElasticVram::MiB(12288)
        );
        assert_eq!(
            ElasticVram::from_str("7.5").unwrap(),
            ElasticVram::MiB(7680)
        );
        assert!(ElasticVram::from_str("garbage").is_err());
        assert!(ElasticVram::from_str("-1").is_err());
        assert!(ElasticVram::from_str("").is_err());
    }

    #[test]
    fn stream_lines_collects_warning_and_split() {
        let buf: Vec<String> = [
            "stream-weights: warning: 100 MiB free".to_string(),
            "stream-weights: resident layers 0/28 (0 MiB resident, 743 MiB streamed per token)"
                .to_string(),
            "not a stream line".to_string(),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            stream_lines(&buf),
            vec![
                "stream-weights: warning: 100 MiB free".to_string(),
                "stream-weights: resident layers 0/28 (0 MiB resident, 743 MiB streamed per token)"
                    .to_string(),
            ]
        );
        assert!(stream_lines(&[]).is_empty());
    }

    #[test]
    fn apply_sampling_omits_defaults_and_sends_non_defaults() {
        let mut body = serde_json::json!({"prompt": "p"});
        apply_sampling(&mut body, &SamplingParams::default());
        assert_eq!(body, serde_json::json!({"prompt": "p"}));

        let sp = SamplingParams {
            top_p: 0.9,
            top_k: 40,
            seed: Some(7),
            frequency_penalty: 0.5,
            presence_penalty: -0.5,
            stop: vec!["###".into()],
        };
        apply_sampling(&mut body, &sp);
        assert_eq!(body["top_p"].as_f64().unwrap(), 0.9f32 as f64);
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["seed"], 7);
        assert_eq!(body["frequency_penalty"].as_f64().unwrap(), 0.5f32 as f64);
        assert_eq!(body["presence_penalty"].as_f64().unwrap(), -0.5f32 as f64);
        assert_eq!(body["stop"], serde_json::json!(["###"]));
    }

    fn two_tools() -> serde_json::Value {
        serde_json::json!([
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "function", "function": {"name": "get_time"}},
        ])
    }

    #[test]
    fn apply_tool_choice_none_leaves_body_untouched() {
        let mut body = serde_json::json!({"tools": two_tools()});
        apply_tool_choice(&mut body, None).unwrap();
        assert!(body.get("tool_choice").is_none());
        assert_eq!(body["tools"], two_tools());
    }

    #[test]
    fn apply_tool_choice_strings_pass_through() {
        for s in ["auto", "required"] {
            let mut body = serde_json::json!({"tools": two_tools()});
            apply_tool_choice(&mut body, Some(&serde_json::json!(s))).unwrap();
            assert_eq!(body["tool_choice"], serde_json::json!(s), "{s}");
            assert_eq!(body["tools"], two_tools(), "{s}");
        }
    }

    #[test]
    fn apply_tool_choice_none_keeps_tools_out_of_the_prompt() {
        let mut body = serde_json::json!({"tools": two_tools()});
        apply_tool_choice(&mut body, Some(&serde_json::json!("none"))).unwrap();
        assert!(body.get("tools").is_none(), "{body}");
        assert!(body.get("tool_choice").is_none(), "{body}");
    }

    #[test]
    fn apply_tool_choice_named_narrows_tools_and_requires() {
        let mut body = serde_json::json!({"tools": two_tools()});
        let choice = serde_json::json!({"type": "function", "function": {"name": "get_time"}});
        apply_tool_choice(&mut body, Some(&choice)).unwrap();
        assert_eq!(body["tool_choice"], serde_json::json!("required"));
        assert_eq!(
            body["tools"],
            serde_json::json!([{"type": "function", "function": {"name": "get_time"}}])
        );
    }

    #[test]
    fn apply_tool_choice_unknown_name_errors() {
        let mut body = serde_json::json!({"tools": two_tools()});
        let choice = serde_json::json!({"type": "function", "function": {"name": "nope"}});
        let e = apply_tool_choice(&mut body, Some(&choice)).unwrap_err();
        assert!(e.contains("'nope'"), "{e}");
        assert!(e.contains("not in tools"), "{e}");
    }

    #[test]
    fn apply_tool_choice_bogus_string_errors() {
        let mut body = serde_json::json!({"tools": two_tools()});
        let e = apply_tool_choice(&mut body, Some(&serde_json::json!("sometimes"))).unwrap_err();
        assert!(e.contains("sometimes"), "{e}");
    }

    #[test]
    fn apply_tool_choice_malformed_choice_errors() {
        // malformed objects, a non-string/object scalar, and an object whose
        // named function is absent from `tools` all fail loudly
        for bad in [
            serde_json::json!({}),
            serde_json::json!({"type": "function"}),
            serde_json::json!({"function": {"name": "get_time"}}),
            serde_json::json!({"type": "function", "function": {"name": ""}}),
            serde_json::json!(42),
            serde_json::json!(true),
        ] {
            let mut body = serde_json::json!({"tools": two_tools()});
            assert!(apply_tool_choice(&mut body, Some(&bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn resolve_llama_bin_prefers_flag_then_env_then_path() {
        let dir = tempfile::tempdir().unwrap();
        let flag_bin = dir.path().join("flag-bin");
        let env_bin = dir.path().join("env-bin");
        let path_bin = dir.path().join(if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        });
        for p in [&flag_bin, &env_bin, &path_bin] {
            std::fs::write(p, b"x").unwrap();
        }
        let env = Some(OsString::from(&env_bin));
        let path = Some(std::env::join_paths([dir.path()]).unwrap());
        // flag wins over env and PATH
        assert_eq!(
            resolve_llama_bin(Some(flag_bin.to_str().unwrap()), env.clone(), path.clone()).unwrap(),
            flag_bin
        );
        // env wins over PATH
        assert_eq!(
            resolve_llama_bin(None, env.clone(), path.clone()).unwrap(),
            env_bin
        );
        // PATH lookup
        assert_eq!(resolve_llama_bin(None, None, path).unwrap(), path_bin);
        // nothing anywhere -> error naming all three sources
        let err = resolve_llama_bin(None, None, Some(OsString::from("/nonexistent"))).unwrap_err();
        assert!(
            err.contains("--llama-bin") && err.contains("CASCADIA_LLAMA_BIN"),
            "{err}"
        );
        assert!(err.contains("build-llama-stream.sh"), "{err}");
        // an explicit path that does not exist errors instead of falling through
        assert!(resolve_llama_bin(Some("/nonexistent/bin"), env, None)
            .is_err_and(|e| e.contains("--llama-bin")));
    }

    #[test]
    fn host_layer_args_only_with_a_regex() {
        assert!(llama_host_layer_args(None).is_empty());
        assert!(llama_host_layer_args(Some("  ")).is_empty());
        assert_eq!(
            llama_host_layer_args(Some(r"blk\.(6[2-4])\..*")),
            ["--load-mode", "none", "-ot", r"blk\.(6[2-4])\..*=SYCL_Host"]
        );
        // the override syntax's own separators cannot be part of the pattern
        assert!(llama_host_layer_args(Some("a=b")).is_empty());
        assert!(llama_host_layer_args(Some("a,b")).is_empty());
    }

    #[test]
    fn llama_device_args_maps_cascadia_devices() {
        let (a, ngl) = llama_device_args("GPU", 99);
        assert_eq!(a, ["--device", "SYCL0"]);
        assert_eq!(ngl, 99);
        let (a, ngl) = llama_device_args("GPU.1", 99);
        assert_eq!(a, ["--device", "SYCL1"]);
        assert_eq!(ngl, 99);
        for cpu in ["CPU", "cpu", "Cpu"] {
            let (a, ngl) = llama_device_args(cpu, 99);
            assert_eq!(a, ["--device", "none"], "{cpu}");
            assert_eq!(ngl, 0, "{cpu}");
        }
        for verbatim in ["SYCL1", "Vulkan0", "SYCL0,SYCL1", "GPU.x"] {
            let (a, ngl) = llama_device_args(verbatim, 12);
            assert_eq!(a, ["--device", verbatim], "{verbatim}");
            assert_eq!(ngl, 12, "{verbatim}");
        }
    }

    #[test]
    fn probe_stream_weights_detects_marker() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("llama-server");
        // marker inside the binary itself
        std::fs::write(&bin, b"bin GGML_STREAM_WEIGHTS").unwrap();
        assert_eq!(
            probe_stream_weights(&bin).unwrap(),
            StreamWeightsSupport::Present
        );

        // marker only in libggml-base next to the binary
        std::fs::write(&bin, b"bin").unwrap();
        std::fs::write(
            dir.path().join("libggml-base.so"),
            b"lib GGML_STREAM_WEIGHTS",
        )
        .unwrap();
        assert_eq!(
            probe_stream_weights(&bin).unwrap(),
            StreamWeightsSupport::Present
        );

        // libggml-base present, marker nowhere -> hard error
        std::fs::write(dir.path().join("libggml-base.so"), b"lib").unwrap();
        let err = probe_stream_weights(&bin).unwrap_err();
        assert!(err.contains("no weight-streaming support"), "{err}");

        // no libggml-base, no marker -> Unknown (warn and continue)
        std::fs::remove_file(dir.path().join("libggml-base.so")).unwrap();
        assert_eq!(
            probe_stream_weights(&bin).unwrap(),
            StreamWeightsSupport::Unknown
        );
    }

    #[test]
    fn probes_match_sycl_and_base_lib_names_on_both_oses() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("llama-server");
        std::fs::write(&bin, b"bin").unwrap();

        // the 0002 marker lives in libggml-sycl (the SYCL backend lib)
        std::fs::write(
            dir.path().join("libggml-sycl.so.0.19.0"),
            b"lib GGML_STREAM_EXPERT_CACHE_MB",
        )
        .unwrap();
        assert!(probe_expert_streaming(&bin));
        // and a 0001-only sycl lib answers "no 0002"
        std::fs::write(
            dir.path().join("libggml-sycl.so.0.19.0"),
            b"lib GGML_STREAM_WEIGHTS",
        )
        .unwrap();
        assert!(!probe_expert_streaming(&bin));
        // but the 0001 probe still reads it as a stream lib
        assert_eq!(
            probe_stream_weights(&bin).unwrap(),
            StreamWeightsSupport::Present
        );

        // Windows-style names: ggml-sycl.dll / ggml-base.dll
        std::fs::remove_file(dir.path().join("libggml-sycl.so.0.19.0")).unwrap();
        std::fs::write(
            dir.path().join("ggml-sycl.dll"),
            b"lib GGML_STREAM_EXPERT_CACHE_MB",
        )
        .unwrap();
        assert!(probe_expert_streaming(&bin));
        std::fs::remove_file(dir.path().join("ggml-sycl.dll")).unwrap();
        assert!(!probe_expert_streaming(&bin));

        // a ggml lib without the marker counts as "found" -> hard error
        std::fs::write(dir.path().join("libggml-sycl.so"), b"lib").unwrap();
        assert!(probe_stream_weights(&bin).is_err());
        std::fs::remove_file(dir.path().join("libggml-sycl.so")).unwrap();
        // unrelated libs are not scanned
        std::fs::write(dir.path().join("libggml-cpu.so"), b"lib").unwrap();
        assert_eq!(
            probe_stream_weights(&bin).unwrap(),
            StreamWeightsSupport::Unknown
        );
    }

    #[test]
    fn auto_load_timeout_scales_with_model_size() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.gguf");
        let f = std::fs::File::create(&model).unwrap();
        f.set_len(2 << 30).unwrap(); // 2 GiB sparse
        assert_eq!(auto_load_timeout(&model), Duration::from_secs(60 + 16));
        // unreadable metadata -> fixed 300 s
        assert_eq!(
            auto_load_timeout(Path::new("/nonexistent/m.gguf")),
            Duration::from_secs(300)
        );
    }

    #[test]
    fn extra_args_rejects_reserved_flags() {
        for arg in [
            "-m",
            "--model",
            "--model-url",
            "-mu",
            "-hf",
            "--host",
            "--port",
            "--device",
            "-dev",
            "-ngl",
            "--gpu-layers",
            "-c",
            "--ctx-size",
        ] {
            let err = validate_extra_args(&[arg.to_string()]).unwrap_err();
            assert!(err.contains(arg), "{err}");
        }
        // the <flag>=<value> form shadows just the same
        let err = validate_extra_args(&["--port=1".to_string()]).unwrap_err();
        assert!(err.contains("--port"), "{err}");
        let err = validate_extra_args(&["-c=8192".to_string()]).unwrap_err();
        assert!(err.contains("--llama-ctx"), "{err}");
    }

    #[test]
    fn extra_args_allows_non_reserved() {
        assert!(validate_extra_args(&[
            "-ctk".into(),
            "q8_0".into(),
            "-fa".into(),
            "on".into(),
            "--no-jinja".into(),
        ])
        .is_ok());
        assert!(validate_extra_args(&[]).is_ok());
    }
}

#[cfg(all(test, unix))]
mod unix_tests {
    use super::*;
    use cascadia_types::ChatTurn;
    use std::os::unix::fs::PermissionsExt;

    /// In-process mock of llama-server: accepts one connection, reads the
    /// request, hands it to `respond`, which writes the reply to the socket.
    fn mock_server(
        respond: impl FnOnce(&[u8], &mut TcpStream) + Send + 'static,
    ) -> (u16, std::thread::JoinHandle<()>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let t = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let _ = s.set_read_timeout(Some(Duration::from_secs(30)));
            let mut reader = BufReader::new(s.try_clone().unwrap());
            let mut req = Vec::new();
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    break;
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    content_length = v.trim().parse().unwrap_or(0);
                }
                req.extend_from_slice(line.as_bytes());
                if line == "\r\n" {
                    break;
                }
            }
            let mut body = vec![0u8; content_length];
            let _ = reader.read_exact(&mut body);
            req.extend_from_slice(&body);
            respond(&req, &mut s);
        });
        (port, t)
    }

    fn chunked(payload: &str) -> String {
        format!("{:x}\r\n{}\r\n", payload.len(), payload)
    }

    fn test_engine(port: u16) -> LlamaCppEngine {
        LlamaCppEngine {
            base: format!("http://127.0.0.1:{port}"),
            child: Command::new("sleep").arg("60").spawn().unwrap(),
            pending: Vec::new(),
            active: None,
            rx: None,
            last_chunk_at: Instant::now(),
            #[cfg(windows)]
            child_job: None,
            cancelled: Arc::new(AtomicBool::new(false)),
            socket: None,
        }
    }

    fn chat_task() -> GenerationTask {
        let mut t = GenerationTask::new("task-1", "ignored");
        t.messages = vec![ChatTurn {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        }];
        t
    }

    /// Run step() until a final chunk arrives; returns everything collected.
    fn drain(engine: &mut LlamaCppEngine) -> Vec<(TaskId, Chunk)> {
        let mut out = Vec::new();
        for _ in 0..64 {
            let batch = engine.step().unwrap();
            let fin = batch.iter().any(|(_, c)| c.is_final);
            out.extend(batch);
            if fin {
                return out;
            }
        }
        panic!("no final chunk");
    }

    #[test]
    fn engine_streams_chat_chunks_over_chunked_sse() {
        let (port, server) = mock_server(|req, s| {
            let req = String::from_utf8_lossy(req);
            assert!(req.starts_with("POST /v1/chat/completions"), "{req}");
            assert!(req.contains("\"role\":\"user\""), "{req}");
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
            s.write_all(head.as_bytes()).unwrap();
            for p in [
                "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":2}}\n",
                "data: [DONE]\n",
            ] {
                s.write_all(chunked(p).as_bytes()).unwrap();
            }
            s.write_all(b"0\r\n\r\n").unwrap();
        });
        let mut engine = test_engine(port);
        engine.submit(chat_task()).unwrap();
        let chunks = drain(&mut engine);
        let texts: Vec<&str> = chunks.iter().map(|(_, c)| c.text.as_str()).collect();
        assert_eq!(texts, ["Hello", " world", ""]);
        let last = &chunks.last().unwrap().1;
        assert!(last.is_final);
        assert_eq!(last.finish_reason, Some(FinishReason::Stop));
        assert_eq!(last.prompt_tokens, Some(11));
        server.join().unwrap();
    }

    #[test]
    fn engine_decodes_multibyte_split_across_chunks() {
        // "é" is 2 UTF-8 bytes; split it across two chunk boundaries to
        // prove the decoder buffers bytes per line, not per chunk
        let (port, server) = mock_server(|_, s| {
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
            s.write_all(head.as_bytes()).unwrap();
            let payload = "data: {\"choices\":[{\"delta\":{\"content\":\"h\u{e9}llo\"}}]}\n";
            let bytes = payload.as_bytes();
            let split = bytes
                .iter()
                .position(|&b| b == 0xC3)
                .expect("multibyte lead byte");
            let a = &bytes[..split + 1]; // ends mid-character (0xC3)
            let b = &bytes[split + 1..]; // starts with the trailing byte
            let frame = |p: &[u8]| {
                let mut f = format!("{:x}\r\n", p.len()).into_bytes();
                f.extend_from_slice(p);
                f.extend_from_slice(b"\r\n");
                f
            };
            s.write_all(&frame(a)).unwrap();
            s.write_all(&frame(b)).unwrap();
            s.write_all(b"0\r\n\r\n").unwrap();
        });
        let mut engine = test_engine(port);
        engine.submit(chat_task()).unwrap();
        let chunks = drain(&mut engine);
        let texts: Vec<&str> = chunks.iter().map(|(_, c)| c.text.as_str()).collect();
        assert_eq!(texts, ["h\u{e9}llo", ""], "no U+FFFD may appear");
        assert!(!texts.iter().any(|t| t.contains('\u{FFFD}')));
        server.join().unwrap();
    }

    #[test]
    fn engine_emits_error_chunk_on_server_500() {
        let (port, server) = mock_server(|_, s| {
            s.write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\noops")
                .unwrap();
        });
        let mut engine = test_engine(port);
        engine.submit(chat_task()).unwrap();
        let chunks = drain(&mut engine);
        let last = &chunks.last().unwrap().1;
        assert!(last.is_final);
        assert!(
            last.error.as_deref().unwrap_or("").contains("500"),
            "{last:?}"
        );
        server.join().unwrap();
    }

    /// A quiet child (prefill, a long inter-token gap) must not hold the
    /// runtime thread: `step()` returns within a poll interval with a
    /// progress marker (no text, `n_tokens` 0, not final) so the runner's
    /// watchdog sees work, and the stall limit is measured since the last
    /// real chunk, not per call.
    #[test]
    fn engine_step_yields_progress_while_child_is_quiet() {
        let (port, server) = mock_server(|_, s| {
            s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
                .unwrap();
            s.write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n")
                .unwrap();
            // then nothing: hold the stream open until the client goes away
            let mut buf = [0u8; 64];
            let _ = s.read(&mut buf);
        });
        let mut engine = test_engine(port);
        let task = chat_task();
        let tid = task.task_id.clone();
        engine.submit(task).unwrap();
        // The first real chunk arrives within a few polls.
        let mut got_text = false;
        for _ in 0..100 {
            let out = engine.step().unwrap();
            if out.iter().any(|(_, c)| c.text == "Hi") {
                got_text = true;
                break;
            }
            assert!(out.iter().all(|(_, c)| c.is_progress()), "{out:?}");
        }
        assert!(got_text);
        // Quiet child: every step returns promptly with one progress marker.
        for _ in 0..5 {
            let t0 = Instant::now();
            let out = engine.step().unwrap();
            assert!(t0.elapsed() < Duration::from_secs(2), "step blocked");
            assert_eq!(out.len(), 1, "{out:?}");
            assert_eq!(out[0].0, tid);
            assert!(out[0].1.is_progress(), "{:?}", out[0].1);
        }
        assert!(engine.active.is_some());
        // Past the stall limit since the last real chunk: a final error.
        engine.last_chunk_at = Instant::now() - STALL_LIMIT;
        let out = engine.step().unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].1.is_final);
        assert!(
            out[0].1.error.as_deref().unwrap().contains("stalled"),
            "{:?}",
            out[0].1
        );
        assert!(engine.active.is_none());
        drop(engine);
        server.join().unwrap();
    }

    /// Linux: PDEATHSIG fires on the spawning *thread's* exit — the reason
    /// children go through the never-exiting spawner thread. A child
    /// spawned via it from a caller thread that then exits must stay
    /// alive; only the process dying may take the child down.
    #[cfg(target_os = "linux")]
    #[test]
    fn spawner_thread_child_survives_short_lived_callers() {
        let (tx, rx) = std::sync::mpsc::channel();
        let t = std::thread::spawn(move || {
            let mut cmd = Command::new("sleep");
            cmd.arg("30");
            arm_death_guard(&mut cmd);
            let child = spawn_on_parent_thread(cmd).unwrap();
            tx.send(child.id()).unwrap();
            // the calling thread exits here: a naive PDEATHSIG spawn
            // would already have killed the child
        });
        let pid = rx.recv().unwrap();
        t.join().unwrap();
        std::thread::sleep(Duration::from_millis(300));
        let state = std::fs::read_to_string(format!("/proc/{pid}/status"))
            .unwrap_or_default()
            .lines()
            .find(|l| l.starts_with("State:"))
            .unwrap_or("")
            .to_string();
        assert!(
            !state.is_empty() && !state.contains('Z'),
            "child {pid} did not survive the caller thread: {state}"
        );
        let _ = Command::new("kill").arg("-9").arg(pid.to_string()).output();
    }

    /// The host interposer's environment (#132) never reaches the child,
    /// and ambient GGML_STREAM_* cannot stream without `--elastic`.
    #[test]
    fn scrub_child_env_drops_host_elastic_and_ambient_stream_vars() {
        fn removed(cmd: &Command) -> Vec<String> {
            cmd.get_envs()
                .filter(|(_, v)| v.is_none())
                .map(|(k, _)| k.to_string_lossy().into_owned())
                .collect()
        }
        let mut off = Command::new("true");
        scrub_child_env(&mut off, false, None);
        let r = removed(&off);
        for k in HOST_ELASTIC_ENV {
            assert!(r.iter().any(|x| x == k), "{k} not removed: {r:?}");
        }
        for k in [
            "GGML_STREAM_WEIGHTS",
            "GGML_STREAM_VRAM_MB",
            "GGML_STREAM_RESIDENT_LAYERS",
            "GGML_STREAM_VRAM_SHARE",
        ] {
            assert!(r.iter().any(|x| x == k), "{k} not removed: {r:?}");
        }
        // With --elastic the engine sets the two it owns after the scrub;
        // the documented child knob GGML_STREAM_RESIDENT_LAYERS passes
        // through (warned about when set).
        let mut on = Command::new("true");
        scrub_child_env(&mut on, true, None);
        on.env("GGML_STREAM_WEIGHTS", "1");
        let r = removed(&on);
        assert!(
            !r.iter().any(|x| x == "GGML_STREAM_RESIDENT_LAYERS"),
            "{r:?}"
        );
        // LD_PRELOAD keeps user entries and drops only cascadia's
        // interposer; the var goes away entirely when the interposer was
        // the only entry. (Pure fn — no process-env mutation in tests.)
        assert_eq!(
            filter_ld_preload("/tmp/libcascadia_elastic.42.so:/opt/user/libmine.so").as_deref(),
            Some("/opt/user/libmine.so")
        );
        assert_eq!(filter_ld_preload("/tmp/libcascadia_elastic.42.so"), None);
        assert_eq!(filter_ld_preload(""), None);
        // space-separated entries (glibc accepts both separators): the
        // interposer is dropped without taking the user's preload with it
        assert_eq!(
            filter_ld_preload("/opt/user.so /tmp/libcascadia_elastic.42.so").as_deref(),
            Some("/opt/user.so")
        );
        assert_eq!(
            filter_ld_preload("/opt/a.so /opt/b.so:/tmp/libcascadia_elastic.9.so").as_deref(),
            Some("/opt/a.so:/opt/b.so")
        );
        assert_eq!(
            filter_ld_preload("/opt/a.so:/opt/b.so").as_deref(),
            Some("/opt/a.so:/opt/b.so")
        );
        let set: Vec<(String, String)> = on
            .get_envs()
            // filtered LD_PRELOAD may or may not be set depending on the
            // ambient value; exclude it from the owned-env assertion
            .filter(|(k, _)| *k != "LD_PRELOAD")
            .filter_map(|(k, v)| {
                v.map(|v| {
                    (
                        k.to_string_lossy().into_owned(),
                        v.to_string_lossy().into_owned(),
                    )
                })
            })
            .collect();
        assert_eq!(set, [("GGML_STREAM_WEIGHTS".to_string(), "1".to_string())]);
    }

    #[test]
    fn engine_cancel_mid_stream() {
        let (port, server) = mock_server(|_, s| {
            s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
                .unwrap();
            s.write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n")
                .unwrap();
            // hold the stream open; the cancelled read returns at EOF/shutdown
            let mut buf = [0u8; 64];
            let _ = s.read(&mut buf);
        });
        let mut engine = test_engine(port);
        let task = chat_task();
        let tid = task.task_id.clone();
        engine.submit(task).unwrap();
        let first = engine.step().unwrap();
        assert_eq!(first[0].1.text, "Hello");
        engine.cancel(&tid);
        assert!(engine.step().unwrap().is_empty());
        drop(engine);
        server.join().unwrap();
    }

    fn dummy_shard() -> ShardSpec {
        ShardSpec {
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 0,
            total_layers: 0,
            device: "GPU".into(),
            is_first_stage: true,
            is_last_stage: true,
            tp_size: 1,
            tp_rank: 0,
        }
    }

    fn write_script(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        let mut perm = std::fs::metadata(&p).unwrap().permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(&p, perm).unwrap();
        p
    }

    fn cfg_for(bin: PathBuf, retries: u32) -> LlamaCppConfig {
        LlamaCppConfig {
            llama_bin: bin,
            model: PathBuf::from("/nonexistent.gguf"),
            device: "GPU".into(),
            ctx: 128,
            ngl: 0,
            elastic: false,
            elastic_vram: ElasticVram::Auto,
            elastic_share: None,
            host_layers: None,
            mtp: false,
            extra_args: vec![],
            load_timeout: Some(Duration::from_secs(1)),
            load_retries: retries,
        }
    }

    #[tokio::test]
    async fn load_retries_then_fails_with_child_stderr() {
        let dir = tempfile::tempdir().unwrap();
        let bin = write_script(
            dir.path(),
            "llama-server",
            "#!/bin/sh\necho 'boom-marker-xyz' >&2\nexit 1\n",
        );
        let mut b = LlamaCppBuilder::new(cfg_for(bin, 1));
        let err = b.load(dummy_shard()).await.err().unwrap().to_string();
        assert!(err.contains("after 2 attempt(s)"), "{err}");
        assert!(err.contains("boom-marker-xyz"), "{err}");
    }

    #[tokio::test]
    async fn load_health_timeout_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let bin = write_script(dir.path(), "llama-server", "#!/bin/sh\nsleep 30\n");
        let mut b = LlamaCppBuilder::new(cfg_for(bin, 0));
        let t0 = Instant::now();
        let err = b.load(dummy_shard()).await.err().unwrap().to_string();
        assert!(t0.elapsed() < Duration::from_secs(20));
        assert!(err.contains("health timeout"), "{err}");
    }
}
