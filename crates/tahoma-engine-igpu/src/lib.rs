//! Scoping crate for the Intel iGPU acceleration path on the K2.6
//! sparse-MoE engine.
//!
//! Status: **scoping only** (PR `perf/igpu-oneapi-scoping-071`,
//! 2026-05-18). This crate compiles, exports stubbed types, and is
//! wired into the workspace so subsequent PRs can land one kernel at a
//! time without re-litigating the API surface. Nothing here actually
//! talks to a GPU yet — every [`IGpuContext`] call returns
//! [`Error::Stub`].
//!
//! The detailed design lives in `docs/IGPU_PLAN.md` (same directory as
//! this `Cargo.toml`); the high-level path is summarized at the bottom
//! of this file under "Design at a glance".
//!
//! # Why this crate exists
//!
//! Today the sparse-MoE engine drives one Rust int4 GEMV kernel per
//! projection per layer plus 8 expert evaluations per token. All
//! kernels run on the CPU via [`tahoma_int4_gemm::kernel_avx512`].
//! On a Lunar Lake 140V iGPU the **same kernel signature** can land
//! inside a SYCL kernel and amortize:
//!
//! - Higher peak FP32 / FP16 throughput than the dual-issue AVX-512
//!   loop (iGPU XMX or DPAS-like fused MAC on Xe matrix engines).
//! - Higher effective bandwidth — Lunar Lake's iGPU shares LPDDR5x and
//!   wins from the 32-EU sustain when the CPU cores are blocked on
//!   `__builtin_ia32_vpdpbusd` issue stalls.
//! - **Async overlap** with the CPU: an expert dispatched to the iGPU
//!   leaves the AVX-512 lane free to compute the next shell projection
//!   in parallel. This is the same overlap insight that powered PR #5
//!   (`feat: async overlap of target wire with draft compute`).
//!
//! # API surface (planned)
//!
//! ```ignore
//! use tahoma_engine_igpu::{IGpuContext, IGpuDevice, Int4Gemv};
//! let ctx = IGpuContext::auto()?;     // L0 device probe + queue
//! let gemv = Int4Gemv::compile(&ctx)?; // AOT-cached SPIR-V or JIT
//! gemv.run(packed, scales, x, &mut y, n_rows, k_cols)?;
//! ```
//!
//! The trait surface mirrors the existing AVX-512 entry point
//! [`tahoma_int4_gemm::kernel_avx512::dequant_gemv_int4_auto`] so the
//! sparse-MoE engine can dispatch through a single
//! `dyn Int4GemvBackend` once both kernels exist.
//!
//! # Design at a glance
//!
//! - **Path**: Intel OneAPI Base Toolkit, SYCL via DPC++/`icpx`,
//!   Level Zero loader. C++ kernels in `cpp/shim.cpp`, FFI mirrored on
//!   the `tahoma-ov-genai-shim` C ABI pattern (see
//!   `crates/tahoma-ov-genai-shim/cpp/shim.cpp` in the workspace) —
//!   `extern "C"`, no exceptions across the boundary, every entry
//!   point catches `...`.
//! - **First op**: dequant int4 GEMV (`dequant_gemv_int4`). Hot in
//!   shells (router + Q/K/V/O projections) AND in the per-expert FFN.
//! - **Composition**: the sparse-MoE engine grows a `Backend` enum
//!   ({CPU, iGpu}); routing decisions stay on the CPU, dispatch
//!   distributes work across backends based on op size + queue depth.
//! - **Fallback**: every iGPU entry point returns `Result`; failure
//!   bubbles up and the CPU AVX-512 path runs in its place. No silent
//!   correctness loss, no fork in the engine code.

use std::fmt;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    /// The crate was built without the `oneapi-sycl` or
    /// `opencl-fallback` feature, so every iGPU call falls through to
    /// this branch. Match on this in the sparse-MoE engine and dispatch
    /// to the CPU path.
    #[error(
        "tahoma-engine-igpu is built without a real backend; rebuild \
         with `--features oneapi-sycl` (or `opencl-fallback`)"
    )]
    Stub,

    /// Surfaced by the Level Zero loader when no compatible iGPU is
    /// available (no Intel Xe-LP / Xe-LPG / Xe-HPG present, driver
    /// missing, or `level-zero` package not installed).
    #[error("no compatible Intel iGPU detected: {0}")]
    NoDevice(String),

    /// SYCL kernel compile / link failure. Catches both JIT (SPIR-V →
    /// device ISA) and AOT (precompiled `--features oneapi-sycl-aot`)
    /// errors uniformly.
    #[error("kernel compile failed: {0}")]
    Compile(String),

    /// Any runtime launch / submit / wait failure from the iGPU
    /// runtime. Includes OOM, OOR (out-of-range buffer access), and
    /// the dreaded "GPU hang, reset device".
    #[error("kernel launch failed: {0}")]
    Launch(String),

    /// Shape mismatch caught before we hit the device. Use this for
    /// debug-mode assertions on `n_rows`, `k_cols`, scale stride —
    /// catching them here gives a way better error message than the
    /// SYCL runtime would.
    #[error("shape mismatch: {0}")]
    Shape(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Handle to a configured iGPU device + command queue. Construct via
/// [`IGpuContext::auto`] (picks the first Intel GPU exposed by Level
/// Zero) or [`IGpuContext::by_index`].
///
/// Cheap to clone — the underlying `ze_device_handle_t` is reference
/// counted in the Level Zero loader. Heavy work (queue creation, JIT
/// compile cache load) happens once at construction.
#[derive(Clone)]
pub struct IGpuContext {
    _device_index: u32,
}

impl fmt::Debug for IGpuContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IGpuContext")
            .field("device_index", &self._device_index)
            .field("backend", &"stub")
            .finish()
    }
}

impl IGpuContext {
    /// Probe for an Intel iGPU and create the default execution
    /// context. Returns [`Error::Stub`] in the scoping build.
    pub fn auto() -> Result<Self> {
        Err(Error::Stub)
    }

    /// Construct the context for a specific Level Zero device index.
    /// Useful when a node has both an iGPU (`0`) and a dGPU (`1`).
    pub fn by_index(index: u32) -> Result<Self> {
        let _ = index;
        Err(Error::Stub)
    }

    /// Human-readable summary — driver version, EU count, peak FP32
    /// FLOPS — for the `tahoma devices` CLI (planned). Returns a
    /// stub message until the L0 probe lands.
    pub fn describe(&self) -> String {
        format!(
            "IGpuContext(device_index={}, backend=stub) — \
             rebuild with --features oneapi-sycl for the real probe",
            self._device_index
        )
    }
}

/// Stubbed int4 GEMV backend. Mirrors the shape of
/// [`tahoma_int4_gemm::kernel_avx512::dequant_gemv_int4_auto`] so the
/// sparse-MoE engine can switch backends without changing call sites.
///
/// Construction compiles + caches the kernel binary; `run` only
/// submits + waits. The real implementation will also accept an
/// optional submit-only path so the engine can pipeline an expert
/// dispatch behind a CPU op via `try_run_async` (TODO).
pub struct Int4Gemv {
    _ctx: IGpuContext,
}

impl Int4Gemv {
    /// AOT-compile (or load from disk cache) the int4 GEMV kernel for
    /// the bound device. Stubbed today.
    pub fn compile(ctx: &IGpuContext) -> Result<Self> {
        Ok(Self { _ctx: ctx.clone() })
    }

    /// Run `y = W @ x` where `W` is int4-packed (group_size=32, bf16
    /// scales). Same wire format as `tahoma_int4_gemm` — the GEMV
    /// kernel accepts the **same byte slices** with no host-side
    /// repack. This keeps the CPU and iGPU paths sharing one mmap'd
    /// expert weight on disk, which matters for the 384-expert × 60-layer
    /// K2.6 weight footprint.
    ///
    /// Returns [`Error::Stub`] in the scoping build; once landed, this
    /// will block until the iGPU finishes (use `run_async` for the
    /// overlap path).
    pub fn run(
        &self,
        packed: &[u8],
        scale_bits: &[u8],
        x: &[f32],
        n_rows: usize,
        k_cols: usize,
        y: &mut [f32],
    ) -> Result<()> {
        let _ = (packed, scale_bits, x, n_rows, k_cols, y);
        Err(Error::Stub)
    }
}

/// Convenience: returns whichever backend is preferred for the given
/// op size at runtime. Today always returns [`Backend::Cpu`] — the
/// crossover point is empirical, plan calls for measurement in PR
/// `perf/igpu-int4-gemv-073`. See `docs/IGPU_PLAN.md` § "Composition
/// story".
pub fn select_backend(_n_rows: usize, _k_cols: usize, _ctx: Option<&IGpuContext>) -> Backend {
    Backend::Cpu
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    IGpu,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_returns_stub_on_scoping_build() {
        match IGpuContext::auto() {
            Err(Error::Stub) => {}
            other => panic!("expected Error::Stub, got {:?}", other),
        }
    }

    #[test]
    fn select_backend_is_cpu_until_kernel_lands() {
        assert_eq!(select_backend(7168, 7168, None), Backend::Cpu);
        assert_eq!(select_backend(2048, 7168, None), Backend::Cpu);
    }

    #[test]
    fn compile_succeeds_but_run_returns_stub() {
        // The context constructor surfaces Stub; we can still produce
        // an Int4Gemv "for free" because compile() short-circuits when
        // there's no real backend wired up — the goal is to keep the
        // dispatch wiring in the engine ready to swap on PR landing.
        let ctx = IGpuContext { _device_index: 0 };
        let g = Int4Gemv::compile(&ctx).expect("stub compile is infallible");
        let packed = vec![0u8; 16];
        let scales = vec![0u8; 4];
        let x = vec![0.0f32; 32];
        let mut y = vec![0.0f32; 1];
        match g.run(&packed, &scales, &x, 1, 32, &mut y) {
            Err(Error::Stub) => {}
            other => panic!("expected Error::Stub, got {:?}", other),
        }
    }
}
