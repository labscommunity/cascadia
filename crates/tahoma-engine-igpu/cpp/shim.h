// C ABI for the iGPU acceleration shim.
//
// Status: header committed on the scoping branch
// `perf/igpu-oneapi-scoping-071` (2026-05-18). The matching .cpp does
// not exist yet — see `../docs/IGPU_PLAN.md` for the milestone
// breakdown that turns each of these prototypes into a SYCL kernel.
//
// Thread safety: handles are NOT thread-safe; serialise calls on the
// Rust side. Same convention as `tahoma-ov-genai-shim/cpp/shim.h`.
//
// Error reporting: every entry point returns 0 on success, non-zero on
// failure. The most recent error string is retrievable via
// `tahoma_igpu_last_error_message()`. C++ exceptions are caught with
// `catch (...)` at every entry point — exceptions never cross the FFI
// boundary into Rust.
//
// Build inputs (see `../build.rs`):
//   * `INTEL_ONEAPI_DIR`         → sysroot for `icpx` + DPC++ headers
//   * `LEVEL_ZERO_SDK_DIR`       → headers + lib for `libze_loader`
//   * `TAHOMA_IGPU_KERNEL_AOT`   → if set, compile SPIR-V ahead-of-time
//                                  for the target device arch (e.g.
//                                  `xe_lpg`, `xe_lp`, `xe_hpg`)

#ifndef TAHOMA_IGPU_SHIM_H
#define TAHOMA_IGPU_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tahoma_igpu_context_t tahoma_igpu_context_t;
typedef struct tahoma_igpu_int4_gemv_t tahoma_igpu_int4_gemv_t;
typedef struct tahoma_igpu_event_t tahoma_igpu_event_t;

/// Most recent error string. Static buffer; not thread-safe — call on
/// the same thread that triggered the error.
const char* tahoma_igpu_last_error_message();

// ---- Context lifecycle ----------------------------------------------------

/// Create a context bound to the first available Intel GPU exposed by
/// Level Zero. Returns 0 on success.
int32_t tahoma_igpu_context_create_auto(tahoma_igpu_context_t** out);

/// Create a context bound to a specific Level Zero device index.
int32_t tahoma_igpu_context_create_by_index(uint32_t index,
                                            tahoma_igpu_context_t** out);

/// Pretty-print device info into `buf` (driver version, EU count,
/// peak FP32 GFLOPS). Returns the bytes written, or -1 on overflow.
int32_t tahoma_igpu_context_describe(const tahoma_igpu_context_t* ctx,
                                     char* buf,
                                     size_t buf_len);

void tahoma_igpu_context_destroy(tahoma_igpu_context_t* ctx);

// ---- int4 GEMV ------------------------------------------------------------
//
// One handle = one compiled kernel ready to be re-launched with
// different (packed, scales, x, y) buffers. Compile cost is paid
// once; launch cost is a queue submit + (optional) wait.

int32_t tahoma_igpu_int4_gemv_compile(const tahoma_igpu_context_t* ctx,
                                      tahoma_igpu_int4_gemv_t** out);

/// Synchronous launch: submit kernel and wait for completion. Equivalent
/// to `tahoma_int4_gemm::kernel_avx512::dequant_gemv_int4_auto` —
/// `packed` is `n_rows * (k_cols / 2)` bytes, `scales` is
/// `n_rows * (k_cols / GROUP_SIZE) * 2` bytes, `x` is `k_cols` f32s,
/// `y` is `n_rows` f32s.
int32_t tahoma_igpu_int4_gemv_run(tahoma_igpu_int4_gemv_t* gemv,
                                  const uint8_t* packed,
                                  size_t packed_len,
                                  const uint8_t* scales,
                                  size_t scales_len,
                                  const float* x,
                                  size_t x_len,
                                  float* y,
                                  size_t y_len,
                                  uint32_t n_rows,
                                  uint32_t k_cols);

/// Async variant: submits the kernel, returns an event the caller can
/// wait on later (`tahoma_igpu_event_wait`). Use this for CPU↔iGPU
/// overlap: dispatch a router projection to the iGPU, run the next
/// shell linear on CPU, then wait.
///
/// The buffer pointers must remain valid until the event signals.
int32_t tahoma_igpu_int4_gemv_run_async(tahoma_igpu_int4_gemv_t* gemv,
                                        const uint8_t* packed,
                                        size_t packed_len,
                                        const uint8_t* scales,
                                        size_t scales_len,
                                        const float* x,
                                        size_t x_len,
                                        float* y,
                                        size_t y_len,
                                        uint32_t n_rows,
                                        uint32_t k_cols,
                                        tahoma_igpu_event_t** out_event);

void tahoma_igpu_int4_gemv_destroy(tahoma_igpu_int4_gemv_t* gemv);

// ---- Events ---------------------------------------------------------------

/// Block until the event signals (kernel done, data visible to host).
int32_t tahoma_igpu_event_wait(tahoma_igpu_event_t* event);

void tahoma_igpu_event_destroy(tahoma_igpu_event_t* event);

#ifdef __cplusplus
}
#endif

#endif // TAHOMA_IGPU_SHIM_H
