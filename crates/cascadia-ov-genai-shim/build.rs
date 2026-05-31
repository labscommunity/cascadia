//! Build script.
//!
//! Two paths:
//!
//! * `--features openvino` ON — compile `cpp/shim.cpp` against the OV
//!   GenAI C++ headers and link against `libopenvino_genai`. Requires
//!   `INTEL_OPENVINO_DIR` to point at an OV install with the genai SDK.
//! * default — emit a no-op build (the Rust stub implementation provides
//!   the full API surface returning runtime errors).

fn main() {
    if std::env::var_os("CARGO_FEATURE_OPENVINO").is_none() {
        // No-op stub build.
        return;
    }

    let ov_root = match std::env::var("INTEL_OPENVINO_DIR") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => panic!(
            "\n\
            ────────────────────────────────────────────────────────────\n\
            cascadia: building with `--features openvino` but INTEL_OPENVINO_DIR is not set.\n\n\
            Point it at an OpenVINO GenAI 2026.1+ SDK install, e.g.:\n\n\
            \x20 INTEL_OPENVINO_DIR=/opt/intel/openvino_genai_2026.1.0.0 \\\n\
            \x20   cargo build --release -p cascadia --features openvino\n\n\
            Don't have the SDK yet? See INSTALL.md (\"OpenVINO GenAI SDK\")\n\
            or run scripts/setup-openvino.sh (Linux) / setup-openvino.ps1 (Windows).\n\
            To build without OpenVINO (stub mode), drop `--features openvino`.\n\
            ────────────────────────────────────────────────────────────\n"
        ),
    };

    // Fail early with an actionable message if the path is set but wrong.
    // A bad INTEL_OPENVINO_DIR otherwise surfaces as an opaque
    // missing-header compile error or a link failure deep in cc/ld.
    let runtime_include_dir = format!("{ov_root}/runtime/include");
    if !std::path::Path::new(&runtime_include_dir).is_dir() {
        panic!(
            "\n\
            ────────────────────────────────────────────────────────────\n\
            cascadia: INTEL_OPENVINO_DIR={ov_root:?} does not look like an\n\
            OpenVINO GenAI SDK — expected to find `runtime/include/` under it.\n\n\
            Check the path points at the *extracted SDK root* (the directory\n\
            that contains `runtime/`, `setupvars.sh`, etc.), not a parent or\n\
            an archive. See INSTALL.md for the expected layout.\n\
            ────────────────────────────────────────────────────────────\n"
        );
    }

    let runtime_include = format!("{ov_root}/runtime/include");
    let genai_include = format!("{ov_root}/runtime/include/openvino/genai");
    // OpenVINO 2026.x Windows ships .lib files under lib/intel64/Release
    // (and lib/intel64/Debug); Linux ships them directly under
    // lib/intel64. Add both paths; the linker picks whichever resolves.
    let runtime_lib_root = format!("{ov_root}/runtime/lib/intel64");
    let runtime_lib_release = format!("{ov_root}/runtime/lib/intel64/Release");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("cpp/shim.cpp")
        .include(&runtime_include)
        .include(&genai_include)
        .compile("cascadia_ov_genai_shim");

    println!("cargo:rustc-link-search=native={runtime_lib_release}");
    println!("cargo:rustc-link-search=native={runtime_lib_root}");
    println!("cargo:rustc-link-lib=dylib=openvino_genai");
    println!("cargo:rustc-link-lib=dylib=openvino");
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-changed=cpp/shim.h");
    println!("cargo:rerun-if-env-changed=INTEL_OPENVINO_DIR");
}
