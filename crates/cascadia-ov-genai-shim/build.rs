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

    let ov_root = std::env::var("INTEL_OPENVINO_DIR")
        .expect("INTEL_OPENVINO_DIR must be set when building with --features openvino");

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
