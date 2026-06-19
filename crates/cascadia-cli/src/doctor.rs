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

use std::process::Command;

use anyhow::Result;
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

fn check_rust(r: &mut Report) {
    match first_line_of("rustc", "--version") {
        Some(v) => r.line(Level::Ok, "Rust toolchain", &v),
        None => {
            r.line(Level::Fail, "Rust toolchain", "rustc not found on PATH");
            r.note("Install via https://rustup.rs then `rustup default stable` (need 1.75+).");
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
            r.line(
                Level::Warn,
                "C++ compiler",
                "no g++/clang++ on PATH (needed only for --features openvino)",
            );
            r.note("Linux: install g++ ≥ 12 (`sudo apt install g++`).");
        }
    }
}

fn check_python(r: &mut Report) {
    // Python is an EXPORT-time dependency (`cascadia shard`), not a
    // runtime one. Surface it here so users discover it before they hit
    // sharding, but a miss is only a warning.
    let py = ["python3", "python"]
        .into_iter()
        .find_map(|p| first_line_of(p, "--version").map(|v| (p, v)));
    match py {
        Some((p, v)) => {
            r.line(Level::Ok, "Python (export-time)", &v);
            // Probe the export packages so the warning is specific.
            let probe = Command::new(p)
                .args([
                    "-c",
                    "import torch, openvino, transformers, safetensors, huggingface_hub",
                ])
                .output();
            match probe {
                Ok(o) if o.status.success() => {
                    r.line(
                        Level::Ok,
                        "Export packages",
                        "torch/openvino/transformers present",
                    );
                }
                _ => {
                    r.line(
                        Level::Warn,
                        "Export packages",
                        "missing (only needed for `cascadia shard`)",
                    );
                    r.note(
                        "pip install torch transformers openvino safetensors huggingface_hub nncf",
                    );
                }
            }
        }
        None => {
            r.line(
                Level::Warn,
                "Python (export-time)",
                "no python3/python on PATH (only needed for `cascadia shard`)",
            );
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
                    "    intel-level-zero-gpu, level-zero  (see INSTALL.md / setup-openvino.sh)",
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
    check_ov_devices(&mut r);

    println!();
    match r.worst {
        Level::Ok | Level::Info => {
            println!("All good. Try:  cascadia run <hf-model-id-or-path>");
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
