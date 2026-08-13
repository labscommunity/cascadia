//! Compile the elastic allocator interposer into a standalone shared library
//! (`libcascadia_elastic.so` on Linux) placed in `OUT_DIR`. The Rust side
//! embeds it with `include_bytes!` (see `src/lib.rs`), so there is no install
//! step and no runtime path discovery — `activate()` writes the bytes to a
//! private temp file and points `LD_PRELOAD` at it.
//!
//! On non-Unix targets there is no LD_PRELOAD, so we emit an empty placeholder
//! and the Rust side compiles the interposer path out; Windows uses the
//! OV-native knob route instead (documented in `lib.rs`).

use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    println!("cargo:rerun-if-changed=src/elastic_unix.c");

    if target_os == "linux" || target_os == "macos" {
        let src = "src/elastic_unix.c";
        let so = out.join("libcascadia_elastic.so");
        // Use the cc-selected compiler but drive the link ourselves: cc::Build
        // only makes static archives, and we need a real .so to LD_PRELOAD.
        let compiler = cc::Build::new().get_compiler();
        let mut cmd = compiler.to_command();
        cmd.args(["-O2", "-fPIC", "-shared", "-o"])
            .arg(&so)
            .arg(src)
            .args(["-ldl", "-lpthread"]);
        let status = cmd.status().expect("failed to spawn C compiler for elastic shim");
        assert!(status.success(), "elastic shim compile failed: {status}");
        println!("cargo:rustc-env=ELASTIC_SO_PATH={}", so.display());
    } else {
        // Placeholder so include_bytes! always resolves; never LD_PRELOADed.
        let so = out.join("libcascadia_elastic.so");
        std::fs::write(&so, b"").unwrap();
        println!("cargo:rustc-env=ELASTIC_SO_PATH={}", so.display());
    }
}
