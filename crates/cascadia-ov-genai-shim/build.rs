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
            Point it at an OpenVINO GenAI 2026.2+ SDK install, e.g.:\n\n\
            \x20 INTEL_OPENVINO_DIR=/opt/intel/openvino_genai_2026.2.0.0 \\\n\
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
    // openvino/core/parallel.hpp (used by the gemv-offload extension op)
    // includes TBB headers; the SDK ships them under runtime/3rdparty.
    let tbb_include = format!("{ov_root}/runtime/3rdparty/tbb/include");
    // OpenVINO 2026.x Windows ships .lib files under lib/intel64/Release
    // (and lib/intel64/Debug); Linux ships them directly under
    // lib/intel64. Add both paths; the linker picks whichever resolves.
    let runtime_lib_root = format!("{ov_root}/runtime/lib/intel64");
    let runtime_lib_release = format!("{ov_root}/runtime/lib/intel64/Release");

    // Optional oneDNN embedding for the gemv-offload op (the spike's
    // parity-throughput endgame): point DNNL_DIR at an extracted oneDNN
    // install (include/ + lib/dnnl.lib + bin/dnnl.dll — use a TBB-runtime
    // build so dnnl threads share the OV plugin's tbb12). Absent = the op
    // keeps its own kernels only.
    let dnnl_dir = std::env::var("DNNL_DIR")
        .ok()
        .filter(|v| !v.trim().is_empty());

    let mut build = cc::Build::new();
    build.cpp(true).std("c++17");
    // gemv_offload.cpp uses AVX2/FMA + AVX-VNNI intrinsics behind a runtime
    // guard; GCC/Clang additionally need the target flags at COMPILE time
    // (MSVC compiles intrinsics without arch flags). GCC >= 11 for -mavxvnni.
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("x86_64") && !target.contains("msvc") {
        build.flag("-mavx2").flag("-mfma").flag("-mavxvnni");
    }
    build
        .file("cpp/shim.cpp")
        .file("cpp/gemv_offload.cpp")
        .include(&runtime_include)
        .include(&genai_include)
        .include(&tbb_include);
    if let Some(d) = &dnnl_dir {
        build.include(format!("{d}/include"));
        build.define("CASCADIA_HAVE_DNNL", None);
    }
    build.compile("cascadia_ov_genai_shim");

    println!("cargo:rustc-link-search=native={runtime_lib_release}");
    println!("cargo:rustc-link-search=native={runtime_lib_root}");
    // ov::parallel_for (gemv-offload extension) inlines TBB calls; the SDK
    // ships the import libs under runtime/3rdparty/tbb/lib (Windows) and the
    // shared objects under .../tbb/lib/intel64 (Linux archives).
    println!("cargo:rustc-link-search=native={ov_root}/runtime/3rdparty/tbb/lib");
    println!("cargo:rustc-link-search=native={ov_root}/runtime/3rdparty/tbb/lib/intel64");
    println!("cargo:rustc-link-lib=dylib=openvino_genai");
    println!("cargo:rustc-link-lib=dylib=openvino");
    // Windows SDKs ship tbb12.lib; Linux OV archives ship libtbb.so.12
    // (linker name `tbb`).
    if std::env::var("TARGET")
        .unwrap_or_default()
        .contains("windows")
    {
        println!("cargo:rustc-link-lib=dylib=tbb12");
    } else {
        println!("cargo:rustc-link-lib=dylib=tbb");
    }
    if let Some(d) = &dnnl_dir {
        println!("cargo:rustc-link-search=native={d}/lib");
        println!("cargo:rustc-link-lib=dylib=dnnl");
    }
    println!("cargo:rerun-if-env-changed=DNNL_DIR");
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-changed=cpp/shim.h");
    println!("cargo:rerun-if-changed=cpp/gemv_offload.cpp");
    println!("cargo:rerun-if-changed=cpp/gemv_offload.hpp");
    println!("cargo:rerun-if-env-changed=INTEL_OPENVINO_DIR");
}
