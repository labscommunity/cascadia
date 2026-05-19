# C++ SYCL shim — scope marker

This directory is intentionally empty of `.cpp` source on the scoping
branch (`perf/igpu-oneapi-scoping-071`, 2026-05-18).

The header `shim.h` describes the C ABI that the implementation PR
will fulfil. It is committed now so the Rust FFI in `../src/lib.rs`
can be reviewed end-to-end alongside the kernel signatures, and so
future contributors building this crate get a clear "you also need the
OneAPI Base Toolkit + Level Zero loader" build error rather than a
silent miscompile.

When the kernels land, the build chain will be:

```bash
# Linux
source /opt/intel/oneapi/setvars.sh
cargo build -p tahoma-engine-igpu --features oneapi-sycl --release

# Windows (cascadia matias-* fleet)
& "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
cargo build -p tahoma-engine-igpu --features oneapi-sycl --release
```

`cpp/shim.cpp` will hold:
- `tahoma_igpu_context_create` / `tahoma_igpu_context_destroy`
- `tahoma_igpu_int4_gemv_compile` / `..._destroy`
- `tahoma_igpu_int4_gemv_run` (sync) / `..._run_async` (returns event)
- `tahoma_igpu_event_wait` / `..._destroy`
- `tahoma_igpu_last_error_message`

These mirror the patterns already in
`crates/tahoma-ov-genai-shim/cpp/shim.h` — the same `int32_t` return
codes, the same opaque handle types, and the same `catch (...)`
guarding so an SYCL exception can't unwind into Rust UB.

See `../docs/IGPU_PLAN.md` for the full design + rollout plan.
