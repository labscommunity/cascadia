# Intel iGPU acceleration plan for K2.6 sparse-MoE

Scoping doc for branch `perf/igpu-oneapi-scoping-071`, written 2026-05-18.
This is **not** the implementation; it's the design + milestone plan
the implementation PRs will follow.

The mission line is unchanged: take a model that doesn't fit on one
Intel laptop and run it across two or three of them with usable
tok/s. The CPU-only sparse-MoE engine currently delivers ~0.06 tok/s
single-stage on a Lunar Lake 140V (32 GB, disk-bound on cold expert
pages) and ~0.11 tok/s on the Xeon Gold 6252 miner. Adding the iGPU
to the mix should buy back another **2–5×** on the active compute
fraction (gemv-bound projections inside the shells), without changing
the model layout or the wire protocol between ranks.

## TL;DR

| Decision | Choice | Why (short) |
|---|---|---|
| API surface | **OneAPI / SYCL** via `icpx -fsycl` | First-party Intel toolchain; AOT compile means no per-launch JIT; binds against the Level Zero loader that already ships in the Intel Graphics driver. |
| FFI shape | C ABI shim in `cpp/shim.cpp` | Same pattern as `tahoma-ov-genai-shim`; no exceptions cross the boundary; `extern "C"` keeps Rust binding work mechanical. |
| First op | int4 GEMV (`dequant_gemv_int4`) | Hottest op by call count (router + Q/K/V/O × 60 layers + 8 experts × 3 linears × 60 layers = ~1900 calls/tok); same weight bytes as the CPU path. |
| Fallback | iGPU returns `Result`; on `Err` engine routes the op to AVX-512 | No silent correctness loss. CPU path stays the canonical reference. |
| Composition | Backend enum in the sparse-MoE engine; dispatch decides per-op | Lets us land the kernel and prove it on, say, the router projection before migrating all 480 expert calls. |
| Target hardware | Lunar Lake `matias-02` / `matias-03` (cascadia fleet) | These are the headline AI PCs; both already build `tahoma --features openvino` and are reachable over tailscale (see [`cascadia_fleet_deploy.md`](../../../docs/deploy/README.md) for the SSH path). |
| Effort | **6–10 weeks** for the first working kernel + integration | See "Milestones" below. |

## Hot-path inventory (where the cycles go)

Numbers below are per **decoder step** at `seq_len=1` decode, K2.6 dims
(HIDDEN=7168, NUM_HEADS=64, QK_HEAD_DIM=192, V_HEAD_DIM=128,
INTERMEDIATE_SHARED=2048, N_ROUTED_EXPERTS=384, TOPK=8). Counts come
from `crates/tahoma-int4-gemm/src/shell_int4.rs` (shell forward) and
`crates/tahoma-engine-sparse-moe/src/runner.rs` (engine wiring).

| Op | n_rows | k_cols | bytes/weight | calls/layer | layers | calls/tok | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `q_a_proj` | 1536 | 7168 | int4 (5.3 MB) | 1 | 60 | 60 | Inside shell, attention pre-proj |
| `q_b_proj` | 64·192=12288 | 1536 | int4 (9.2 MB) | 1 | 60 | 60 | Wider out, narrow in |
| `kv_a_proj` | 576 | 7168 | int4 (2.0 MB) | 1 | 60 | 60 | KV down-proj + rope channel |
| `kv_b_proj` | 64·(128+128)=16384 | 512 | int4 (4.0 MB) | 1 | 60 | 60 | Widest output dim |
| `o_proj` | 7168 | 64·128=8192 | int4 (27 MB) | 1 | 60 | 60 | Heaviest single GEMV |
| `router` | 384 | 7168 | int4 (1.3 MB) | 1 | 60 | 60 | Tiny — likely CPU-only |
| `shared_gate/up/down` | 2048/2048/7168 | 7168/7168/2048 | int4 (5.2/5.2/5.2 MB) | 3 | 60 | 180 | "Always-on" expert |
| `expert_{gate,up,down}` | 2048/2048/7168 | 7168/7168/2048 | int4 (5.2/5.2/5.2 MB) | 3·8 | 60 | 1440 | Routed top-8; the **bulk of the work** |
| Embedding lookup | 1 row | HIDDEN | bf16 (14 KB) | – | – | 1 | Tiny, stays on CPU |
| SDPA | small | small | – | 1 | 60 | 60 | Quadratic in past_seq_len; on iGPU only if context > a few hundred tokens |
| `lm_head` | 163840 | HIDDEN | bf16 (~2 GB) | – | – | 1 | Big single GEMV but per-tok cost dominated by bandwidth, not compute |

**Total int4 GEMV calls / decode step ≈ 1900.** Of those, **76% are
expert FFN calls** (the routed experts) and another **9% are the
shared expert**. Even partial iGPU offload of just the expert path
should move the needle.

CPU-side dispatch cost (rayon worker thread overhead) per `dequant_gemv_int4_auto`
call is ~50 µs at the K2.6 dims on the miner Xeon — not the bottleneck,
but it's the floor an iGPU kernel needs to beat for `kernel_launch_overhead
+ device_compute` to be a net win.

## Expected speedup per op (back-of-envelope, Lunar Lake 140V iGPU)

The 140V iGPU has **8 Xe-cores × 16 vector engines × 8 SIMD lanes =
1024 simultaneous FP32 ops/clock**, peak ~2.5 GHz boost, so peak
~5.0 TFLOPS FP32. Hard cap from LPDDR5x-8533 quad-channel = ~136
GB/s shared with the CPU.

| Op | Arithmetic intensity (ops/byte) | Bottleneck on iGPU | Expected speedup vs miner AVX-512 |
|---|---:|---|---:|
| Big expert GEMV (2048×7168 down-proj) | ~0.5 (2 ops / 4 bits) | DRAM bandwidth | **2–3×** — iGPU's headroom is in the 100 GB/s sustained band, vs the Xeon's ~140 GB/s but with rayon contention; on Lunar Lake CPU specifically the iGPU is faster because the LP cores throttle at 4× peak in sustained AVX-512 |
| Medium GEMV (1536×7168 q_a) | similar | DRAM bandwidth | **1.5–2.5×** |
| Tiny GEMV (384×7168 router) | bandwidth-bound | launch-overhead-bound | **likely SLOWDOWN** until we batch routers across layers |
| SDPA seq=1, past=4096 | bandwidth-bound (read past KV) | DRAM bandwidth | **2–3×** if we keep `past_k`/`past_v` in iGPU device memory between layers, **~1×** if we copy host↔device per layer |
| Embedding gather | trivial | host overhead | **stays on CPU** |
| `lm_head` (163k×7168 bf16) | bandwidth-bound | iGPU bandwidth vs CPU bandwidth | **1.5–2×**, but only worth it if we can fuse with sampling to avoid the host transfer |

**Net expected end-to-end win on Lunar Lake**: 2–3× decode tok/s, with
upside from CPU↔iGPU overlap (next section).

## Composition story (the actual reason this is worth doing)

The CPU AVX-512 path issues kernels strictly sequentially. The iGPU
exposes a queue, so we get **two compute units running in parallel**
for free. The pattern that scales:

```text
                 │ CPU (AVX-512)          │ iGPU (SYCL queue)
─────────────────┼────────────────────────┼─────────────────────────
layer L step 0   │ q_a_proj               │ (idle)
layer L step 1   │ q_b_proj               │ kv_a_proj            ← overlap starts
layer L step 2   │ kv_b_proj              │ o_proj               ← already submitted
layer L step 3   │ router + SDPA          │ (waiting on o_proj)
layer L step 4   │ shared_expert path     │ expert[0] FFN        ← parallel
layer L step 5   │ expert[1] FFN          │ expert[2] FFN        ← 2-wide
layer L step 6   │ expert[3] FFN          │ expert[4] FFN
                 │  ...                   │  ...
```

The scheduling problem is identical to the dist-spec overlap in
PR #5 (`fix: async overlap of target wire with draft compute`), and
the implementation pattern carries over: `submit` returns a future
the dispatcher awaits at the consumer of the result. We already have
a tested rayon-based job queue in `kernel_avx512`; the iGPU side
adds one more queue.

Composition rules the engine will follow:

1. **Routing always on CPU.** `router_logits` → top-k indices is
   trivial compute on a single hidden vector. No win in moving it.
2. **Big GEMVs on iGPU when async-submittable.** If the next CPU op
   doesn't depend on the GEMV output, submit to iGPU and let the CPU
   run ahead. This is most of the win on the shell forward (Q/K/V/O
   pre-projections are independent).
3. **Expert dispatch fans across BOTH backends.** Of 8 routed experts,
   send 4 to iGPU, run 4 on CPU. Wait at the weighted-sum reduction.
4. **Fallback is a `select_backend()` decision per call**, not a
   compile-time switch. A device hang reset (`Error::Launch`) marks
   the iGPU as degraded for the rest of the generation and the engine
   reverts to pure CPU. Logged, not silent.

## Path comparison — why SYCL + OneAPI

| Option | Verdict | Why |
|---|---|---|
| **OneAPI / SYCL** (Intel DPC++, `icpx -fsycl`) | ✅ recommended | First-party. AOT compile to SPIR-V or device ISA. Native Level Zero. Same toolchain Intel uses for OpenVINO's GPU plugin (the path that already works in tahoma's `ov-genai`/`ov-runtime`/`ov-dist-spec`). |
| **OpenCL C** (`cl_intel_subgroups`, packaged in the Intel Graphics driver) | 🟡 fallback | Zero extra install (driver ships the runtime). Lower-level — manual kernel compile cache, no auto vectorization. Good for "OneAPI not installed" deployments. Keep as `--features opencl-fallback`, ship after the SYCL path proves out. |
| **wgpu (Vulkan compute)** | ❌ defer | Pure-Rust appeal is real, but: (1) we'd lose access to Intel's matrix engine (XMX / DPAS) intrinsics that beat naive shader-FMA by 2–4× on int4, (2) `wgpu`'s buffer-mapping latency on Intel Windows is ~5× SYCL's on the same hardware (measured in rainier 2025-Q4), (3) no AOT path, every node pays JIT. Revisit if/when the Vulkan compute backend on Intel ships dpas4 intrinsics. |
| **Direct Level Zero in Rust** (e.g. `level-zero-sys`) | ❌ defer | We'd hand-write the SPIR-V. SYCL gives us most of the same control with 10× less ceremony for the same generated code. |
| **OpenVINO custom op** | ❌ defer | Possible — write the op in C++ and register it as an OV plugin extension — but it forces us back through the OV graph compilation tax that the sparse-MoE engine was built to escape (the whole point of int4 GEMM in Rust). Not the right abstraction for per-op offload. |

**Cross-platform note.** SYCL via DPC++ runs on Linux AND Windows
(matias-02/03 already use the MSVC toolchain for tahoma; the OneAPI
installer adds `icpx.exe` to a separate vsdevcmd-style activation
script). The same C++ source compiles for both. We will need separate
AOT-compiled kernel blobs per OS at release time, but Cargo's
`cfg(target_os)` already handles that.

## FFI / build chain

Build inputs (defined in `crates/tahoma-engine-igpu/build.rs`):

| Variable | Path |
|---|---|
| `INTEL_ONEAPI_DIR` | Linux `/opt/intel/oneapi` (sourced via `setvars.sh`), Windows `C:\Program Files (x86)\Intel\oneAPI` |
| `LEVEL_ZERO_SDK_DIR` | Linux `/usr/include/level_zero` + `libze_loader`; Windows ships with the Intel Graphics driver |
| `TAHOMA_IGPU_KERNEL_AOT` | optional — when set, compile SPIR-V → device ISA AOT for the listed Xe arch tags (`xe_lpg`, `xe_lp`, `xe_hpg`) |

Build flow when the `oneapi-sycl` feature is enabled:

```bash
# Linux runtime deps (Ubuntu/Debian package names verified at scoping time):
sudo apt install intel-oneapi-compiler-dpcpp-cpp-2025.0 \
                 intel-level-zero-gpu \
                 level-zero \
                 libze1 \
                 intel-opencl-icd

# Activate the toolchain (sets CMPLR_ROOT, MKLROOT, ICX, etc.)
source /opt/intel/oneapi/setvars.sh

# Build with the iGPU feature
INTEL_ONEAPI_DIR=/opt/intel/oneapi \
  cargo build --release -p tahoma-engine-igpu --features oneapi-sycl

# All-in: bring up sparse-moe with iGPU backend
INTEL_ONEAPI_DIR=/opt/intel/oneapi INTEL_OPENVINO_DIR=$OV_ROOT \
  cargo build --release -p tahoma --features openvino,igpu
```

On Windows (matias fleet):

```powershell
# After Intel OneAPI Base Toolkit install:
& "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 vs2022

$env:INTEL_ONEAPI_DIR = "C:\Program Files (x86)\Intel\oneAPI"
$env:INTEL_OPENVINO_DIR = "C:\Users\devcloud\openvino_genai_windows_2026.1.0.0_x86_64"

cargo build --release -p tahoma --features openvino,igpu
```

The Rust shim mirrors `tahoma-ov-genai-shim`: opaque handles
(`tahoma_igpu_context_t`, `..._int4_gemv_t`, `..._event_t`), `int32_t`
error returns, static-buffer error message, every exception caught at
the boundary. See `cpp/shim.h` for the prototype list.

## Driver / version blockers to surface early

The single biggest schedule risk for an iGPU push isn't kernel code,
it's driver. Capture these in the README before users hit them:

1. **Intel Graphics Driver version.** Need 32.0.x or later on Windows
   for Level Zero 1.6+. Older drivers ship Level Zero 1.3 with a
   different `ze_command_queue_desc_t` ABI; trying to load 1.6
   symbols on a 1.3 runtime throws `ZE_RESULT_ERROR_INVALID_NULL_HANDLE`
   with no useful message. Lunar Lake systems shipped with 31.x; many
   in-the-field AI PCs need a driver update before the SYCL path even
   loads.
2. **i915 firmware** (Linux). Lunar Lake DG2 requires `firmware-misc-nonfree`
   ≥ 20241210 + a kernel ≥ 6.6. Older kernels boot fine but `i915`
   refuses to bind to the iGPU → no Level Zero device found.
3. **Compute Runtime version vs OneAPI**. The Intel Compute Runtime
   (intel-opencl-icd / level-zero-gpu) ships in lockstep with the
   user-mode driver. OneAPI 2025.0's DPC++ runtime requires Compute
   Runtime ≥ 24.50.x; older Ubuntu LTS packages ship 23.x. The fix is
   pinning the Intel `oneapi-base` repo, not the distro repo.
4. **WSL2 caveat**. Lunar Lake iGPU passthrough to WSL2 works on
   Windows 11 25H2 + Intel Graphics Driver 32.0.101.6299+, but only
   when `/dev/dxg` is mapped (default). Earlier driver versions expose
   the iGPU only to Windows-side processes. CI on WSL2 is not viable
   until this is sorted; for now, real-hardware testing happens on the
   matias boxes directly.
5. **Kernel-launch overhead vs op size**. Level Zero queue submit is
   ~50–80 µs on Lunar Lake (measured in rainier 2025-Q3 on a similar
   setup). Ops smaller than ~1 ms of CPU compute aren't worth
   offloading. The router projection (384×7168) lands in the ~700 µs
   CPU compute bucket — we'd need to batch routers across multiple
   layers or fuse with the post-norm to make it pay.
6. **Async copy granularity**. Host-pinned memory for fast device
   transfer requires `sycl::malloc_host` (or `zeMemAllocHost`). The
   K2.6 expert mmap can't be host-pinned without breaking the mmap
   semantics. First implementation will copy `packed`/`scales` into a
   pinned staging buffer per launch (~30 µs for a 5 MB expert) until
   we add a device-resident weight cache (next-next PR).

## Composition with the existing CPU pipeline

The sparse-MoE engine today calls into `kernel_avx512` directly. The
plan extends it as follows (Rust pseudocode):

```rust
// Existing in crates/tahoma-engine-sparse-moe/src/runner.rs (pre-iGPU):
crate::kernel_avx512::dequant_gemv_int4_auto(packed, scales, x, n_rows, k_cols, &mut y);

// Post-iGPU dispatch wrapper:
match select_backend(n_rows, k_cols, ctx) {
    Backend::IGpu => {
        if gemv.run(packed, scales, x, n_rows, k_cols, &mut y).is_err() {
            // Surface a single warn-once and quietly run on CPU.
            kernel_avx512::dequant_gemv_int4_auto(packed, scales, x, n_rows, k_cols, &mut y);
        }
    }
    Backend::Cpu => kernel_avx512::dequant_gemv_int4_auto(packed, scales, x, n_rows, k_cols, &mut y),
}
```

The `select_backend` logic starts out as a fixed shape→backend table
(big enough → iGPU) and graduates to a measurement-tuned policy after
the first benchmark numbers land. Sparse-MoE engine config grows one
field — `use_igpu: bool` — defaulted off so existing eval runs
reproduce exactly. The CLI exposes `--igpu` once correctness is
validated on the matias fleet.

For the async-overlap path:

```rust
// Pipeline the Q/K/V/O projections — independent inputs.
let kv_a_evt = gemv.run_async(kv_a_packed, ...)?;     // iGPU
let q_a_y = kernel_avx512::dequant_gemv_int4_auto(...); // CPU running in parallel
kv_a_evt.wait();                                       // sync at consumer
```

Drop-in compatible with the current sequential runner.

## Milestones (rough estimates)

The whole thing is a multi-month implementation. This scoping branch
is **week 0**.

| Week | PR (planned) | Deliverable | Acceptance |
|---:|---|---|---|
| **0** | `perf/igpu-oneapi-scoping-071` | **This branch.** Crate skeleton + docs + ABI header. | `cargo test -p tahoma-engine-igpu` passes (stub tests); workspace still builds with stub feature. |
| 1–2 | `perf/igpu-context-072` | Real `IGpuContext::auto` against Level Zero loader; `cargo build --features oneapi-sycl` on Linux + Windows. | `tahoma devices` lists the iGPU. |
| 3–4 | `perf/igpu-int4-gemv-073` | First `Int4Gemv` kernel landing — matches `kernel_avx512` byte-for-byte on a synthesized weight matrix. | Numerical agreement against existing CPU kernel within 1e-4 over 1k random inputs. |
| 5 | `perf/igpu-int4-gemv-bench-074` | Benchmark harness, criterion. | We have numbers; we know the crossover point per op size. |
| 6–7 | `perf/igpu-router-projection-075` | Wire the iGPU GEMV into the sparse-MoE engine for the router + Q/K/V/O projections only (no experts yet). `--igpu` CLI flag. | matias-02 single-stage decode: end-to-end correctness vs CPU + measured tok/s improvement. |
| 8–9 | `perf/igpu-expert-overlap-076` | Async overlap on the expert path: half experts on CPU, half on iGPU per layer, sync at the weighted sum. | matias-02: 2×+ decode tok/s improvement on K2.6. |
| 10 | `perf/igpu-2box-077` | Validate iGPU + pipeline-parallel together (matias-02 rank 0 ↔ matias-03 rank 1). | 3-prompt K2.6 eval (Paris/Pacific/four) still passes; tok/s on the 2-box path is ≥ 1.5× the CPU-only 2-box baseline. |

After milestone 10 we get into the "long tail" PRs: SDPA on iGPU,
embedding/lm_head fusion, device-resident weight cache, OpenCL
fallback, Arrow Lake + Panther Lake validation, etc. Those are not
scoped here.

## What we will explicitly NOT do in the first wave

To keep the rollout boring:

- **No NPU.** Intel Lunar Lake NPU is interesting but is a separate
  Movidius VPU runtime, separate driver, separate kernels. Park it.
- **No Arc dGPU.** The codebase calls this out as "later" in
  `CLAUDE.md`. The iGPU plan should work transparently for Arc once
  the SYCL path lands (same DPC++ toolchain), but we will not optimize
  for it or claim support until tested.
- **No model re-quantization.** All work targets the existing on-disk
  int4 format (`group_size=32`, symmetric, bf16 scales). The same
  mmap'd weight bytes feed both backends. We deliberately do not
  invent a new device-friendly layout on PR #1.
- **No mixed-dtype kernels.** First kernel is `int4 weight × f32
  activation → f32 output`, mirroring the CPU path exactly. f16
  activations are a follow-up that should drop another ~20% off the
  iGPU bandwidth bill but adds a numerical risk surface we don't need
  on day one.

## Open questions to validate in the first implementation PR

These need empirical answers; flagging here so reviewers can flush
them out alongside the code:

1. **What's actual Level Zero queue submit overhead on Lunar Lake**
   (vs the ~50 µs rainier measurement on a different SKU)? If it's
   >150 µs we need to batch aggressively.
2. **Does `sycl::malloc_shared` (USM unified) work without copy on
   Lunar Lake iGPU?** If yes, we can skip the pinned-staging step for
   the activation vector `x` and the output `y`. Weights stay mmap'd.
3. **Driver hang recovery.** A misbehaving kernel can wedge the iGPU
   for ~2 seconds while Windows TDR resets the device. Does Level
   Zero surface that as a clean `ZE_RESULT_ERROR_DEVICE_LOST`, or do
   we need a watchdog?
4. **Power thermal cap.** On a passively cooled Lunar Lake laptop,
   sustained iGPU load thermal-throttles within ~30 s. We need
   long-context decode numbers, not just first-second peaks.

These answers feed back into the `select_backend()` policy table.

## References inside the repo

- `crates/tahoma-int4-gemm/src/kernel_avx512.rs` — the kernel to match
  on iGPU
- `crates/tahoma-int4-gemm/src/shell_int4.rs` — call sites for the
  shell forward (router + projections)
- `crates/tahoma-engine-sparse-moe/src/runner.rs` — engine wiring +
  expert dispatch (line 410+ is the `dispatch_expert` per-expert
  block; that's where the iGPU fan-out lands)
- `crates/tahoma-ov-genai-shim/cpp/shim.cpp` — the canonical pattern
  for a C ABI shim wrapping an Intel C++ SDK
- `crates/tahoma-ov-genai-shim/build.rs` — the canonical pattern for
  a `cargo` build script linking against an Intel toolchain
- `docs/SHARDING.md` § "Picking `--quantization`" — explains why
  `group_size=128` is the sweet spot on Intel iGPU (and why we will
  evaluate switching the sparse-MoE int4 to that group size in a
  follow-up — the current K2.6 export uses 32)
