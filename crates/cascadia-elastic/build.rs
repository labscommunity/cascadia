//! Build the elastic allocator interposer for the target OS.
//!
//! * **Linux/macOS** — compile `src/elastic_unix.c` into a standalone
//!   `libcascadia_elastic.so` in `OUT_DIR`. The Rust side embeds it with
//!   `include_bytes!` and `LD_PRELOAD`s it via a one-shot re-exec.
//! * **Windows** — if `DETOURS_DIR` points at a built Microsoft Detours
//!   (`include/detours.h` + `lib.X64/detours.lib`), compile `src/elastic_win.cpp`
//!   and link it (plus Detours + psapi) INTO the binary; the Rust side calls
//!   `elastic_install_win` in-process. Detours is a required SDK-style build
//!   input here, exactly like `INTEL_OPENVINO_DIR` for the OV feature. Without
//!   it the Windows hook is compiled out (the flag still parses; it reports
//!   inactive at runtime).

use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    println!("cargo:rerun-if-changed=src/elastic_unix.c");
    println!("cargo:rerun-if-changed=src/elastic_win.cpp");
    println!("cargo:rerun-if-env-changed=DETOURS_DIR");
    // Custom cfg the Rust side gates the Windows hook on.
    println!("cargo:rustc-check-cfg=cfg(elastic_win_hook)");

    // Always produce the embedded-.so path so `include_bytes!` resolves.
    let so = out.join("libcascadia_elastic.so");

    if target_os == "linux" || target_os == "macos" {
        let compiler = cc::Build::new().get_compiler();
        let mut cmd = compiler.to_command();
        cmd.args(["-O2", "-fPIC", "-shared", "-o"])
            .arg(&so)
            .arg("src/elastic_unix.c")
            .args(["-ldl", "-lpthread"]);
        let status = cmd.status().expect("failed to spawn C compiler for elastic shim");
        assert!(status.success(), "elastic shim compile failed: {status}");
    } else {
        std::fs::write(&so, b"").unwrap(); // placeholder; never LD_PRELOADed
    }
    println!("cargo:rustc-env=ELASTIC_SO_PATH={}", so.display());

    if target_os == "windows" {
        build_windows_hook();
    }
}

fn build_windows_hook() {
    let detours = match std::env::var("DETOURS_DIR") {
        Ok(v) if !v.trim().is_empty() => PathBuf::from(v),
        _ => {
            println!(
                "cargo:warning=cascadia-elastic: DETOURS_DIR not set — the Windows \
                 --elastic hook is disabled (flag parses but reports inactive). \
                 Build Microsoft Detours and set DETOURS_DIR to enable it."
            );
            return;
        }
    };
    let inc = detours.join("include");
    let lib = detours.join("lib.X64");
    if !inc.join("detours.h").is_file() || !lib.join("detours.lib").is_file() {
        println!(
            "cargo:warning=cascadia-elastic: DETOURS_DIR={} is missing \
             include/detours.h or lib.X64/detours.lib — Windows hook disabled.",
            detours.display()
        );
        return;
    }

    cc::Build::new()
        .cpp(true)
        .file("src/elastic_win.cpp")
        .include(&inc)
        .flag_if_supported("/EHsc")
        .flag_if_supported("/O2")
        .compile("cascadia_elastic_win");

    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=detours");
    println!("cargo:rustc-link-lib=dylib=psapi");
    println!("cargo:rustc-cfg=elastic_win_hook");
}
