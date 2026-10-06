//! `cascadia doctor` — environment + hardware self-check.
//!
//! The single biggest onboarding hazard for an OpenVINO-backed,
//! Intel-native tool is the *silent* CPU-only fallback: a correct
//! OpenVINO + driver install can still leave the runtime seeing only
//! the CPU on the exact Core Ultra + Arc iGPU class Cascadia targets,
//! with no error anywhere. `clinfo` reporting a healthy GPU does NOT
//! predict whether OpenVINO's GPU plugin will find it. `doctor` makes
//! that failure loud and actionable instead of letting the operator
//! discover it as mysterious 10× slowness weeks later.
//!
//! It is also the recommended *first* command after build: it checks
//! the Rust/C++/Python toolchain, whether the binary was built with
//! `--features openvino`, the `INTEL_OPENVINO_DIR` env, and enumerates
//! the OV devices the runtime can actually reach.

use std::io::Read;
use std::process::{Command, Stdio};

use anyhow::Result;
use cascadia_engine_llamacpp::{
    probe_expert_streaming, probe_stream_weights, resolve_llama_bin, StreamWeightsSupport,
};
use clap::Parser;

/// Run environment + hardware checks and print a readable report.
#[derive(Parser, Debug, Clone)]
pub struct DoctorArgs {
    /// Exit non-zero if any check is in the WARN or FAIL state. Useful
    /// in CI / provisioning scripts that want to gate on a clean
    /// environment. Off by default so an interactive run is purely
    /// informational.
    #[arg(long, default_value_t = false)]
    pub strict: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    Ok,
    Warn,
    Fail,
    Info,
}

impl Level {
    fn glyph(self) -> &'static str {
        match self {
            Level::Ok => "✓",
            Level::Warn => "⚠",
            Level::Fail => "✗",
            Level::Info => "·",
        }
    }
}

struct Report {
    worst: Level,
}

impl Report {
    fn new() -> Self {
        Self { worst: Level::Ok }
    }

    fn line(&mut self, level: Level, label: &str, detail: &str) {
        // Track the worst non-info level for the strict exit code.
        match (self.worst, level) {
            (_, Level::Fail) => self.worst = Level::Fail,
            (Level::Ok, Level::Warn) => self.worst = Level::Warn,
            _ => {}
        }
        if detail.is_empty() {
            println!("  {} {label}", level.glyph());
        } else {
            println!("  {} {label} — {detail}", level.glyph());
        }
    }

    /// A continuation/remediation line under the previous check.
    fn note(&self, text: &str) {
        println!("      {text}");
    }
}

/// First line of `<cmd> <arg>` stdout/stderr, trimmed. None if the
/// command can't be spawned (not on PATH).
fn first_line_of(cmd: &str, arg: &str) -> Option<String> {
    let out = Command::new(cmd).arg(arg).output().ok()?;
    let text = if !out.stdout.is_empty() {
        String::from_utf8_lossy(&out.stdout)
    } else {
        String::from_utf8_lossy(&out.stderr)
    };
    text.lines().next().map(|l| l.trim().to_string())
}

/// Run `<bin> <arg>` capturing stdout+stderr, with a hard timeout so a
/// hung binary can't wedge the report. Returns the trimmed output (stderr
/// included so loader errors like a missing shared library show up).
fn run_capturing(bin: &std::path::Path, arg: &str, secs: u64) -> Result<String, String> {
    use std::sync::mpsc::channel;
    use std::time::{Duration, Instant};

    let mut child = Command::new(bin)
        .arg(arg)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn failed: {e}"))?;
    // Drain both pipes from the start on their own threads: a child that
    // fills the ~64 KiB pipe buffer would otherwise block mid-write and
    // never exit, deadlocking the wait loop below. Chunks go over a
    // channel rather than a join-on-exit: a probed binary that leaves a
    // daemon behind keeps the pipe's write end open forever, so joining
    // the readers would hang the report even though the child is gone.
    let (tx, rx) = channel::<(bool, Vec<u8>)>();
    let pipes: [(bool, Option<Box<dyn Read + Send>>); 2] = [
        (true, child.stdout.take().map(|p| Box::new(p) as _)),
        (false, child.stderr.take().map(|p| Box::new(p) as _)),
    ];
    for (is_out, pipe) in pipes {
        let Some(mut p) = pipe else { continue };
        let tx = tx.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            loop {
                match p.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if tx.send((is_out, buf[..n].to_vec())).is_err() {
                            break;
                        }
                    }
                }
            }
        });
    }
    drop(tx);
    for _ in 0..secs * 10 {
        match child.try_wait() {
            Ok(Some(_)) => {
                let _ = child.wait();
                let mut out = Vec::new();
                let mut err = Vec::new();
                // Readers EOF once every writer is gone; a daemon that
                // inherited the pipe may never let that happen, so bound
                // the drain and keep whatever arrived.
                let deadline = Instant::now() + Duration::from_secs(2);
                while let Some(rem) = deadline.checked_duration_since(Instant::now()) {
                    match rx.recv_timeout(rem) {
                        Ok((is_out, chunk)) => {
                            if is_out {
                                out.extend_from_slice(&chunk);
                            } else {
                                err.extend_from_slice(&chunk);
                            }
                        }
                        Err(_) => break,
                    }
                }
                let text = String::from_utf8_lossy(if !out.is_empty() { &out } else { &err });
                return Ok(text.trim().to_string());
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(e) => return Err(e.to_string()),
        }
    }
    // kill() alone leaves a zombie; wait() reaps it.
    let _ = child.kill();
    let _ = child.wait();
    Err(format!("timed out after {secs} s"))
}

/// sycl-llama: is there a usable llama-server, does it have the
/// weight-streaming patch, and which SYCL devices does it see.
fn check_sycl_llama(r: &mut Report) {
    let bin = match resolve_llama_bin(
        None,
        std::env::var_os("CASCADIA_LLAMA_BIN"),
        std::env::var_os("PATH"),
    ) {
        Ok(b) => b,
        Err(e) => {
            // Optional engine: a missing binary is a note, not a failure.
            r.line(Level::Info, "llama-server", "not found");
            r.note(&format!("{e} (needed only for `--engine sycl-llama`)"));
            return;
        }
    };
    r.line(Level::Ok, "llama-server", &format!("{}", bin.display()));

    match run_capturing(&bin, "--version", 10) {
        Ok(v) if v.contains("shared libraries") || v.contains("error while loading") => {
            r.line(Level::Warn, "llama-server --version", &v);
            r.note("oneAPI runtime missing: `source /opt/intel/oneapi/setvars.sh` before launching cascadia");
        }
        Ok(v) => r.line(Level::Ok, "llama-server --version", &v),
        Err(e) => r.line(Level::Warn, "llama-server --version", &e),
    }

    match probe_stream_weights(&bin) {
        Ok(StreamWeightsSupport::Present) => {
            r.line(Level::Ok, "weight streaming", "supported (--elastic works)")
        }
        Ok(StreamWeightsSupport::Unknown) => r.line(
            Level::Info,
            "weight streaming",
            "could not verify (no libggml-base next to the binary)",
        ),
        Err(e) => r.line(Level::Warn, "weight streaming", &e),
    }
    if probe_expert_streaming(&bin) {
        r.line(
            Level::Ok,
            "MoE expert streaming",
            "router-aware (--elastic streams selected experts only)",
        );
    } else {
        r.line(
            Level::Info,
            "MoE expert streaming",
            "not present: MoE models stream every expert per token",
        );
    }

    match run_capturing(&bin, "--list-devices", 10) {
        Ok(devs) if !devs.is_empty() => {
            r.line(Level::Ok, "llama-server devices", "");
            for l in devs.lines().take(10) {
                r.note(l);
            }
        }
        _ => {}
    }
}

fn check_rust(r: &mut Report) {
    match first_line_of("rustc", "--version") {
        Some(v) => r.line(Level::Ok, "Rust toolchain", &v),
        None => {
            // Informational, not a warning: a release bundle runs with no Rust
            // toolchain at all, and `--strict` must still pass on such a host.
            r.line(
                Level::Info,
                "Rust toolchain",
                "rustc not found on PATH (only needed to build from source)",
            );
            r.note("Install via https://rustup.rs then `rustup default stable` (need 1.89+).");
        }
    }
}

fn check_cpp(r: &mut Report) {
    // Only relevant for building with --features openvino. Probe the
    // usual suspects; on Windows the toolchain is MSVC (cl.exe), which
    // is only on PATH inside a Developer Prompt, so a miss there is a
    // soft warning, not a failure.
    let probe = ["c++", "g++", "clang++"]
        .into_iter()
        .find_map(|cc| first_line_of(cc, "--version").map(|v| (cc, v)));
    match probe {
        Some((cc, v)) => r.line(Level::Ok, "C++ compiler", &format!("{cc}: {v}")),
        None if cfg!(windows) => {
            r.line(
                Level::Info,
                "C++ compiler",
                "no g++/clang++ on PATH (expected on Windows; MSVC cl.exe is used)",
            );
            r.note(
                "For `--features openvino`, build from a \"Developer Command Prompt for VS 2022\".",
            );
        }
        None => {
            // Build-only, like rustc above: a release bundle needs no compiler.
            r.line(
                Level::Info,
                "C++ compiler",
                "no g++/clang++ on PATH (needed only for --features openvino)",
            );
            r.note("Linux: install g++ ≥ 12 (`sudo apt install g++`).");
        }
    }
}

fn check_python(r: &mut Report) {
    // Python is an EXPORT-time dependency (`cascadia shard`), not a runtime one.
    // Surface it here so users discover it before they hit sharding, but a miss
    // is only a warning. Resolve the interpreter the same way `cascadia shard`
    // does — one that can import the deps, not the first that answers --version.
    // `resolve_python` already probed the imports; don't pay for that twice.
    match crate::resolve_python(None, true) {
        Ok(env) => {
            let version = first_line_of(&env.path, "--version").unwrap_or_default();
            r.line(
                Level::Ok,
                "Python (export-time)",
                &format!("{version} ({})", env.path),
            );
            match env.deps {
                Some(_) => r.line(
                    Level::Ok,
                    "Export packages",
                    "torch/openvino/transformers present",
                ),
                None => {
                    // Export-only, like rustc/g++ above: a worker never needs
                    // Python, so this must not fail `--strict` on a bundle host.
                    r.line(
                        Level::Info,
                        "Export packages",
                        "missing (only needed for `cascadia shard`)",
                    );
                    r.note(&crate::export_pip_install_line(&env.path));
                }
            }
        }
        Err(_) => {
            r.line(
                Level::Info,
                "Python (export-time)",
                "no python3/python on PATH (only needed for `cascadia shard`)",
            );
            // Print the pins anyway: a bundle user has no tools/requirements.txt
            // to read, and this is the only place they can learn them.
            r.note("Install Python 3.10+, then:");
            r.note(&crate::export_pip_install_line("python"));
        }
    }
}

fn check_openvino_env(r: &mut Report) {
    match std::env::var("INTEL_OPENVINO_DIR") {
        Ok(v) if !v.trim().is_empty() => {
            let has_runtime = std::path::Path::new(&v).join("runtime/include").is_dir();
            if has_runtime {
                r.line(Level::Ok, "INTEL_OPENVINO_DIR", &v);
            } else {
                r.line(
                    Level::Warn,
                    "INTEL_OPENVINO_DIR",
                    &format!("{v} (no runtime/include/ — looks wrong)"),
                );
                r.note("Point it at the extracted SDK root (the dir containing `runtime/`).");
            }
        }
        _ => {
            r.line(
                Level::Info,
                "INTEL_OPENVINO_DIR",
                "unset (only needed to BUILD with --features openvino)",
            );
        }
    }
}

/// Which OpenVINO GenAI release is inside this binary — the SDK it was
/// compiled against and the runtime libraries the loader actually picked up.
/// Release bundles pin one version (see release.yml's `OV_VERSION`) and
/// variants built against another carry an `-ov<version>` suffix, so this is
/// how a bug report says which one it is. A skew between the two means the
/// bundle's `lib/` (or PATH on Windows) is serving a different SDK than the
/// binary was built for — flagged, since the C++ ABI is not stable across
/// OpenVINO releases.
fn check_ov_version(r: &mut Report) {
    if !cfg!(feature = "openvino") {
        return; // the stub line in check_ov_devices already says so
    }
    match cascadia_ov_genai_shim::ov_version() {
        Ok(v) => {
            let built = format!(
                "built against GenAI {}, core {}",
                v.built_genai, v.built_core
            );
            if v.genai_skew() {
                r.line(
                    Level::Warn,
                    "OpenVINO GenAI",
                    &format!("runtime {} but {built}", v.runtime_genai),
                );
                r.note(
                    "The loaded libraries are a different OpenVINO release than this binary was",
                );
                r.note("compiled for. Use the lib/ shipped in the same bundle (Linux) or the DLLs");
                r.note(
                    "beside cascadia.exe (Windows); to change the OpenVINO version, pick a bundle",
                );
                r.note("built for it or rebuild from source against that SDK (INSTALL.md).");
            } else {
                r.line(
                    Level::Ok,
                    "OpenVINO GenAI",
                    &format!("{} ({built})", v.runtime_genai),
                );
                r.note(&format!("core runtime {}", v.runtime_core));
            }
        }
        Err(e) => {
            r.line(
                Level::Fail,
                "OpenVINO GenAI",
                &format!("version query failed: {e}"),
            );
        }
    }
}

/// The heart of `doctor`: what devices can the OpenVINO runtime in THIS
/// binary actually reach? Only meaningful when built with the openvino
/// feature; the stub build reports that it can't check.
fn check_ov_devices(r: &mut Report) {
    if !cfg!(feature = "openvino") {
        r.line(
            Level::Info,
            "OpenVINO runtime",
            "this binary was built WITHOUT --features openvino (stub mode)",
        );
        r.note("Stub mode runs the `mock` engine only. Rebuild with --features openvino");
        r.note("for real inference on Intel hardware. See INSTALL.md.");
        return;
    }

    match cascadia_ov_genai_shim::list_devices() {
        Ok(devices) if devices.is_empty() => {
            r.line(
                Level::Fail,
                "OpenVINO devices",
                "runtime enumerated ZERO devices",
            );
            r.note("Even CPU is missing — the OpenVINO runtime libraries may not be on the");
            r.note("loader path. Ensure runtime/lib is reachable (LD_LIBRARY_PATH / PATH).");
        }
        Ok(devices) => {
            let has_accel = devices
                .iter()
                .any(|d| d.starts_with("GPU") || d.starts_with("NPU"));
            r.line(Level::Ok, "OpenVINO devices", &devices.join(", "));
            // Print the full device name for each — the GPU FULL_DEVICE_NAME
            // is how an operator confirms the iGPU vs a dGPU was picked up.
            for d in &devices {
                if let Ok(full) = cascadia_ov_genai_shim::device_full_name(d) {
                    r.note(&format!("{d}: {full}"));
                }
            }
            if !has_accel {
                // THE failure this command exists to catch.
                r.line(
                    Level::Warn,
                    "GPU/NPU acceleration",
                    "NOT visible to OpenVINO — only CPU is available",
                );
                r.note("This is the silent CPU-only fallback. Inference will work but be");
                r.note("several× slower than the iGPU/Arc this hardware has. clinfo reporting");
                r.note("a healthy GPU does NOT mean OpenVINO can see it. Likely fixes (Linux):");
                r.note("  • add yourself to the render group:  sudo usermod -a -G render $USER");
                r.note("    (then log out/in — group changes don't apply to the current shell)");
                r.note("  • install the GPU runtime packages: intel-opencl-icd,");
                r.note(
                    "    libze-intel-gpu1, libze1 + intel-opencl-icd, from Intel's graphics repo \
                     (`scripts/setup-openvino.sh` in a source checkout does this for you; \
                     the distro's own packages are too old for recent Intel GPUs)",
                );
                r.note("On Windows: install the latest Intel graphics driver, then reboot.");
            }
        }
        Err(e) => {
            r.line(
                Level::Fail,
                "OpenVINO runtime",
                &format!("device enumeration failed: {e}"),
            );
            r.note("The runtime libraries likely aren't loadable. On Linux, source the SDK env:");
            r.note("  source $INTEL_OPENVINO_DIR/setupvars.sh   (sets LD_LIBRARY_PATH)");
        }
    }
}

pub fn cmd_doctor(args: DoctorArgs) -> Result<()> {
    println!("cascadia doctor — environment + hardware self-check\n");

    let mut r = Report::new();
    println!("Toolchain:");
    check_rust(&mut r);
    check_cpp(&mut r);
    check_python(&mut r);

    println!("\nOpenVINO:");
    check_openvino_env(&mut r);
    check_ov_version(&mut r);
    check_ov_devices(&mut r);

    println!("\nsycl-llama:");
    check_sycl_llama(&mut r);

    println!();
    match r.worst {
        Level::Ok | Level::Info => {
            println!("All good. Try:  cascadia run <model-dir>   (export one first: cascadia shard --help)");
        }
        Level::Warn => {
            println!("Mostly OK with warnings above — see the remediation notes.");
        }
        Level::Fail => {
            println!("Problems found above. See INSTALL.md for the full setup.");
        }
    }

    if args.strict && matches!(r.worst, Level::Warn | Level::Fail) {
        anyhow::bail!("doctor: --strict and one or more checks were not OK");
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::time::{Duration, Instant};

    fn script(body: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("s.sh");
        // fs::write closes the handle — exec() of a still-open-for-write
        // file fails with ETXTBSY
        std::fs::write(&p, body).unwrap();
        let mut perm = std::fs::metadata(&p).unwrap().permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(&p, perm).unwrap();
        (dir, p)
    }

    /// A child that writes past the ~64 KiB pipe buffer must not deadlock
    /// the capture: both pipes drain from the start on their own threads.
    #[test]
    fn run_capturing_drains_output_larger_than_the_pipe() {
        let (_dir, f) =
            script("#!/bin/sh\nhead -c 200000 /dev/zero | tr '\\0' 'x'\necho err >&2\n");
        let out = run_capturing(&f, "", 10).expect("large output captured");
        assert_eq!(out.len(), 200000);
        assert!(out.chars().all(|c| c == 'x'));
    }

    /// A probed binary that leaves a daemon behind keeps the pipe's write
    /// end open; the capture must return when the foreground child exits
    /// instead of hanging on the reader threads' join.
    #[test]
    fn run_capturing_returns_when_grandchild_holds_the_pipe() {
        // the background sleep inherits stdout; the foreground exits at once
        let (_dir, f) = script("#!/bin/sh\nsleep 60 &\necho done\n");
        let t0 = Instant::now();
        let out = run_capturing(&f, "", 10).expect("capture returned");
        assert_eq!(out, "done");
        assert!(t0.elapsed() < Duration::from_secs(15));
    }

    /// A child sleeping past the deadline is killed AND reaped — kill()
    /// alone would leave it as a zombie child of this process.
    #[test]
    fn run_capturing_kills_and_reaps_on_timeout() {
        let (_dir, f) = script("#!/bin/sh\nsleep 60\n");
        let t0 = Instant::now();
        let err = run_capturing(&f, "", 1).unwrap_err();
        assert!(err.contains("timed out"), "{err}");
        assert!(t0.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(300));
        // An unreaped (or still-running) child is still listed among this
        // process's children; a reaped one is gone entirely.
        let children = Command::new("pgrep")
            .arg("-P")
            .arg(std::process::id().to_string())
            .output()
            .expect("pgrep");
        assert!(
            children.stdout.is_empty(),
            "timed-out child was left behind: {}",
            String::from_utf8_lossy(&children.stdout)
        );
    }
}
