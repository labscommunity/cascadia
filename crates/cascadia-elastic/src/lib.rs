//! Elastic-memory posture for cascadia workers — the ramlab exp 198 result
//! packaged as a first-class flag.
//!
//! # What it does
//!
//! An LLM server's dirty RAM is dominated by large, evictable buffers: the
//! engine's weight copies, KV state and per-inference scratch. The elastic
//! posture serves every allocation at or above a threshold from a file-backed
//! mapping (an unlinked temp file) instead of anonymous memory, so those pages
//! become clean/file-dirty and the kernel can write them back and reclaim them
//! under pressure. Freed big mappings are retained in a pool and reused without
//! unmap/zero, so the churn is a warm-up cost, not a per-inference tax
//! (ramlab D-012/D-013/D-015; the process-wide generalization is exp 198).
//!
//! # Why an interposer, not a Rust `#[global_allocator]`
//!
//! In an OpenVINO worker the bytes that matter are allocated by C++ (oneDNN
//! weight repacks, the GenAI KV cache). A Rust global allocator never sees
//! them. An allocator **interposer** loaded ahead of the C runtime does, which
//! is why this ships as a preloaded shared library rather than a Rust type.
//!
//! # Platform support
//!
//! * **Linux**: [`activate`] embeds the compiled interposer, writes it to a
//!   private temp file, sets `LD_PRELOAD`, and re-executes the current process
//!   once (guarded by an env marker, mirroring the low-RAM re-exec pattern).
//!   Measured on the deployment target: 4.4× less committed RAM at −9% decode
//!   on an unmodified OpenVINO stack (exp 198).
//! * **Windows**: there is no `LD_PRELOAD`, and redirecting the whole C runtime
//!   heap needs a trampoline-based redirector that catches every module's calls
//!   into `ucrtbase` (the approach mimalloc ships as a separate signed DLL, and
//!   the shape a Detours-based hook of the UCRT `malloc` family would take).
//!   That is a self-contained systems project and is NOT implemented here yet,
//!   so on Windows [`activate`] performs no interposition and returns
//!   [`Activation::UnsupportedPlatform`].
//!
//!   Note the OV-native knobs do **not** substitute: measured on Linux
//!   (ramlab exp 199), `ENABLE_MMAP=YES` + `CACHE_MODE=OPTIMIZE_SIZE` leave the
//!   committed footprint unchanged, because the dirty bytes are oneDNN's
//!   *repacked* weight copies, and OV exposes no property to disable those
//!   (that is exactly D-004, and the reason the interposer exists). `--elastic`
//!   still asserts `ENABLE_MMAP=YES` so the original weight blob stays clean
//!   file pages, but the repacked-copy reduction on Windows waits on the
//!   redirector.

use std::ffi::OsString;

/// Env marker set on the child so the re-exec happens at most once.
const GUARD: &str = "CASCADIA_ELASTIC_ACTIVE";
/// Env var the interposer reads for its threshold, in MB.
const MIN_MB: &str = "ELASTIC_MIN_MB";
/// Env var the interposer reads for the retained-mapping pool cap, in MB.
const POOL_MB: &str = "ELASTIC_POOL_MB";
/// Env var the interposer reads for the backing directory.
const DIR: &str = "ELASTIC_DIR";

/// The interposer shared library, embedded at build time (empty on non-Unix).
const ELASTIC_SO: &[u8] = include_bytes!(env!("ELASTIC_SO_PATH"));

/// Tunables for the elastic posture. Defaults mirror the exp 198 sweet spot
/// (1 MB threshold with the retention pool on).
#[derive(Debug, Clone)]
pub struct ElasticOpts {
    /// Route allocations at least this many MB through the file-backed pool.
    /// 1 = maximum RAM cut; 16 = weights-only, zero measured speed cost.
    pub min_mb: u32,
    /// Cap on retained (freed-but-mapped) bytes, in MB. 0 disables the pool.
    pub pool_mb: u32,
    /// Backing directory for the temp files. `None` = `$TMPDIR` or `/tmp`.
    pub dir: Option<OsString>,
}

impl Default for ElasticOpts {
    fn default() -> Self {
        Self { min_mb: 1, pool_mb: 8192, dir: None }
    }
}

/// Outcome of [`activate`].
#[derive(Debug)]
pub enum Activation {
    /// The interposer is already active in this process (we are the re-exec'd
    /// child, or a parent set `LD_PRELOAD` for us). Nothing to do.
    AlreadyActive,
    /// This platform has no supported interposition path. The caller should
    /// fall back to engine-level knobs. Carries a human-readable reason.
    UnsupportedPlatform(&'static str),
}

/// Errors that abort activation before the process is re-executed. A failure
/// here is non-fatal to the caller: run without the posture rather than not at
/// all.
#[derive(Debug)]
pub enum ActivateError {
    /// Writing the embedded interposer to a temp file failed.
    Io(std::io::Error),
    /// `execv` returned, i.e. the re-exec itself failed.
    Exec(std::io::Error),
}

impl std::fmt::Display for ActivateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivateError::Io(e) => write!(f, "writing elastic interposer: {e}"),
            ActivateError::Exec(e) => write!(f, "re-exec with elastic interposer: {e}"),
        }
    }
}
impl std::error::Error for ActivateError {}

/// Turn the elastic posture on for this process.
///
/// On Linux, on success this **does not return**: the process is replaced by a
/// copy of itself with the interposer preloaded. It returns `Ok(_)` only when
/// no re-exec was needed (already active) or the platform is unsupported, and
/// `Err(_)` if activation was attempted but failed (caller should continue
/// without the posture).
pub fn activate(opts: &ElasticOpts) -> Result<Activation, ActivateError> {
    if std::env::var_os(GUARD).is_some() {
        return Ok(Activation::AlreadyActive);
    }
    activate_impl(opts)
}

/// True when running inside an already-activated (re-exec'd) process. Lets the
/// CLI log "elastic: on" without attempting a second activation.
pub fn is_active() -> bool {
    std::env::var_os(GUARD).is_some()
}

#[cfg(unix)]
fn activate_impl(opts: &ElasticOpts) -> Result<Activation, ActivateError> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;

    // Write the interposer to a private temp file (0700). Keep it on disk for
    // the child's lifetime; the OS reclaims it when the process exits and the
    // fd/mapping close. We deliberately do not unlink before exec so the
    // dynamic loader can open it by path.
    let mut path = std::env::temp_dir();
    let unique = format!("libcascadia_elastic.{}.so", std::process::id());
    path.push(unique);
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .mode(0o700)
            .open(&path)
            .map_err(ActivateError::Io)?;
        f.write_all(ELASTIC_SO).map_err(ActivateError::Io)?;
    }

    // Prepend to any existing LD_PRELOAD rather than clobbering it.
    let mut preload = OsString::from(&path);
    if let Some(existing) = std::env::var_os("LD_PRELOAD") {
        if !existing.is_empty() {
            preload.push(":");
            preload.push(existing);
        }
    }

    let exe = std::env::current_exe().map_err(ActivateError::Io)?;
    let args: Vec<OsString> = std::env::args_os().collect();

    // SAFETY: execv replaces the image; on success nothing after it runs. We
    // set env with std (process still single-threaded intent aside — this is
    // called first thing in main, before tokio/rayon spin up).
    std::env::set_var("LD_PRELOAD", &preload);
    std::env::set_var(GUARD, "1");
    std::env::set_var(MIN_MB, opts.min_mb.to_string());
    std::env::set_var(POOL_MB, opts.pool_mb.to_string());
    if let Some(dir) = &opts.dir {
        std::env::set_var(DIR, dir);
    }

    // Build argv as C strings.
    let c_exe = std::ffi::CString::new(exe.as_os_str().as_bytes())
        .map_err(|e| ActivateError::Exec(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)))?;
    let c_args: Vec<std::ffi::CString> = args
        .iter()
        .map(|a| std::ffi::CString::new(a.as_bytes()).unwrap_or_default())
        .collect();
    let mut ptrs: Vec<*const libc::c_char> = c_args.iter().map(|a| a.as_ptr()).collect();
    ptrs.push(std::ptr::null());

    // SAFETY: c_exe/ptrs outlive the call; on success execv never returns.
    unsafe {
        libc::execv(c_exe.as_ptr(), ptrs.as_ptr());
    }
    // Only reached if execv failed.
    Err(ActivateError::Exec(std::io::Error::last_os_error()))
}

#[cfg(not(unix))]
fn activate_impl(_opts: &ElasticOpts) -> Result<Activation, ActivateError> {
    let _ = ELASTIC_SO; // silence unused on non-unix
    Ok(Activation::UnsupportedPlatform(
        "no allocator interposer on this platform yet — the committed-RAM cut \
         needs a UCRT-heap redirector (Detours/mimalloc-redirect class), which \
         is not implemented. OV-native knobs do NOT substitute (they cannot \
         disable oneDNN's dirty repacked weight copies — D-004); ENABLE_MMAP is \
         still seeded so the weight blob stays clean file pages",
    ))
}

/// The OpenVINO plugin properties `--elastic` seeds so the engine's own memory
/// is frugal too — used on every platform, and the ONLY memory lever on
/// Windows. Returned as `(key, value)` pairs the CLI merges into the property
/// set BEFORE user `--ov-config` (so the user can still override any of them).
///
/// * `ENABLE_MMAP=YES` — keep the weight blob memory-mapped (clean pages)
///   rather than copied into the compiled model. OV default is already YES for
///   reading, but we assert it so a stale cache mode cannot flip it.
/// * `CACHE_MODE=OPTIMIZE_SIZE` — when a compile cache is used, store a
///   weightless blob and re-read weights from the mmap'd IR, avoiding a second
///   resident copy.
pub fn ov_memory_props() -> Vec<(String, String)> {
    vec![
        ("ENABLE_MMAP".to_string(), "YES".to_string()),
        ("CACHE_MODE".to_string(), "OPTIMIZE_SIZE".to_string()),
    ]
}

#[cfg(all(unix, test))]
mod tests {
    use super::*;

    #[test]
    fn embedded_shim_is_nonempty_on_unix() {
        // build.rs must have compiled a real .so, not the placeholder.
        assert!(
            ELASTIC_SO.len() > 1024,
            "embedded interposer looks empty ({} bytes) — build.rs compile failed",
            ELASTIC_SO.len()
        );
        // ELF magic.
        assert_eq!(&ELASTIC_SO[..4], b"\x7fELF");
    }

    #[test]
    fn default_opts_match_exp198_sweet_spot() {
        let o = ElasticOpts::default();
        assert_eq!(o.min_mb, 1);
        assert!(o.pool_mb > 0, "pool on by default");
    }

    #[test]
    fn ov_memory_props_assert_mmap() {
        let p = ov_memory_props();
        assert!(p.iter().any(|(k, v)| k == "ENABLE_MMAP" && v == "YES"));
    }
}
