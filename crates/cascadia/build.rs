//! Build script.
//!
//! macOS `--features openvino` only: embed the OpenVINO SDK's library dirs
//! as LC_RPATH in the `cascadia` binary, so it starts without
//! DYLD_LIBRARY_PATH (which `nohup` and other SIP-protected launchers strip).
//! The dirs come from the shim's `links` metadata; a dependency's
//! `rustc-link-arg` never reaches this crate's binary on its own. Every other
//! target, and the stub build, emits nothing.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    if let Some(dirs) = std::env::var_os("DEP_CASCADIA_OV_GENAI_SHIM_RPATH") {
        // The shim emits no empty or comma-bearing dir; skip them anyway rather than pass
        // `-Wl,-rpath,` a bad argument.
        for d in std::env::split_paths(&dirs) {
            let d = d.display().to_string();
            if !d.is_empty() && !d.contains(',') {
                println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{d}");
            }
        }
    }
}
