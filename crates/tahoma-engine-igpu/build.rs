//! Build script for the iGPU scoping crate.
//!
//! There are two real-link paths planned (SYCL via OneAPI, and OpenCL
//! C); both are *off* until the corresponding feature flag is set, and
//! at this point in the timeline (PR `perf/igpu-oneapi-scoping-071`)
//! neither is implemented. The stub build always emits a no-op and the
//! Rust shim in `src/lib.rs` returns `Error::Stub` from every entry
//! point.
//!
//! Sketched-out flow for the real SYCL build, once the kernels in
//! `cpp/shim.cpp` exist:
//!
//! ```text
//! ONEAPI_ROOT=/opt/intel/oneapi              # default install location
//! source $ONEAPI_ROOT/setvars.sh             # exports CMPLR_ROOT, etc.
//! ICX=$CMPLR_ROOT/linux/bin/icpx             # SYCL-aware Clang front-end
//! cargo build -p tahoma-engine-igpu --features oneapi-sycl
//! ```
//!
//! The build.rs branch under `CARGO_FEATURE_ONEAPI_SYCL` would shell
//! out to `icpx -fsycl -fsycl-targets=spir64_gen` with the JIT off
//! (`-Xs "-device <arch>"` for AOT compile) so we ship a single .a per
//! supported Xe-LP / Xe-LPG arch and don't pay the per-process SPIR-V
//! JIT cost. AOT also catches kernel compile failures at build time
//! instead of at first launch.
//!
//! Linux package layout we'll need (verified at scoping time, May 2026):
//! - `intel-oneapi-compiler-dpcpp-cpp-2025.0` (provides `icpx`)
//! - `intel-level-zero-gpu-1.6+`, `level-zero-1.18+`, `intel-opencl-icd`
//! - `intel-i915-firmware` (kernel module — `i915` driver), MESA optional
//!
//! Windows (Arrow Lake / Lunar Lake / Panther Lake):
//! - "Intel Graphics Driver" 32.0.x+ (ships Level Zero + OpenCL runtime
//!   out of the box; OneAPI Base Toolkit installer adds `icpx.exe`)
//! - DDU + clean reinstall is the canonical "driver is weird" fix on
//!   Windows AI PC fleets — capture this in the runbook before users hit
//!   it for the third time
//!
//! Anything not listed above is a TODO at the top of this file (and a
//! matching TODO in the linked C++ shim once it's written).

fn main() {
    // Stub mode for both feature combos — we don't ship kernels yet.
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ONEAPI_SYCL");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_OPENCL_FALLBACK");

    if std::env::var_os("CARGO_FEATURE_ONEAPI_SYCL").is_some() {
        // Intentionally not implemented yet — surface a clear error so
        // an enthusiastic user doesn't think the link succeeded.
        panic!(
            "tahoma-engine-igpu: the `oneapi-sycl` feature is reserved \
             for the iGPU implementation PR; the scoping branch \
             `perf/igpu-oneapi-scoping-071` only ships stubs. \
             See crates/tahoma-engine-igpu/docs/IGPU_PLAN.md."
        );
    }
    if std::env::var_os("CARGO_FEATURE_OPENCL_FALLBACK").is_some() {
        panic!(
            "tahoma-engine-igpu: the `opencl-fallback` feature is \
             reserved for the OpenCL implementation PR; the scoping \
             branch `perf/igpu-oneapi-scoping-071` only ships stubs. \
             See crates/tahoma-engine-igpu/docs/IGPU_PLAN.md."
        );
    }
}
