# tahoma Rust port — status

Tracking the Python → Rust hard rewrite landed under `rust/` on the
`feat/rust-port` branch. Updated whenever a phase lands.

## Engine port status (alpha + charlie hardware)

| Engine | Port | Mac unit tests | Real OV build | Real-model e2e | A/B vs Python |
|---|---|:-:|:-:|:-:|---|
| `ov-genai` | ✅ | ✅ 5 | ✅ alpha + charlie | ✅ | **at parity** (Rust 20.5 vs Python 20.3 plain; 22.3 vs 22.6 FastDraft; 29.3 vs 29.4 PL) |
| `ov-runtime` | ✅ | ✅ 3 | ✅ alpha + charlie | ⏳ no v3 shards on hand | n/a |
| `ov-dist-spec` | ✅ | ✅ 9 | ✅ alpha + charlie | ✅ alpha+charlie/TB4 | **Rust 1.29× slower** (21.70 vs 28.05 tok/s in `--release`; was 2.21× in debug) |

### Perf-gap investigation (commits b396709, 655f6b8)

Original assumption ("FFI memcpy is the bottleneck") was **wrong** —
empirical instrumentation per spec round on alpha+charlie/TB4 found:

| Component (debug profile) | Time/round | % of round |
|---|---:|---:|
| Driver stage_0 GPU infer | 27 ms | 25% |
| Worker stage_1 GPU compute (waited for) | 32 ms | 30% |
| TCP send + recv | 1 ms | 1% |
| **f16 → f32 of 1 MB logits payload** | **70 ms** | **65%** ← root cause |
| Per-round total | ~110 ms | |

The half crate's `f16::to_f32` runs unoptimized in debug builds (no
SIMD); release builds use AVX2/F16C intrinsics and the same conversion
takes 2 ms instead of 70 ms. Building tahoma with `--release` collapses
the dist-spec perf gap from 2.21× to 1.29×.

### Building for performance — IMPORTANT

The dist-spec engine **must be built with `--release`** for production
use. Default `cargo build` produces a 2× slower binary on this engine.

```powershell
cd C:\Users\cascadia\tahoma-rust
cargo build --release -p tahoma --features openvino
.\target\release\tahoma.exe ...
```

The single-node `ov-genai` engine is much less affected by the build
profile because most of its work is inside the openvino-genai C++
library; LLMPipeline takes only one `generate()` call per task so
per-call Rust overhead is amortized.

### Remaining ~23% gap on dist-spec

Profiled in release mode; remaining costs:

* **35 ms/round** waiting for the worker's stage_1 inference (intrinsic
  to OV + iGPU; not a Rust-side issue)
* **2 ms/round** for the 1 MB f16 → f32 logit conversion (could be
  eliminated by doing argmax directly on f16; saves ~36 ms/task ~ 1%)
* Smaller (per-call OV overhead, per-call ov::Tensor allocation in the
  FFI shim) that would each save single-digit % at most

The **structural fix** for the next ~10-20% would be the zero-copy
borrowing-tensor constructor in the C++ shim:

```cpp
// Current shim (alloc + memcpy):
ov::Tensor t(element_type, shape);
memcpy(t.data(), input, byte_size);
request->set_tensor(name, t);

// Zero-copy alternative:
ov::Tensor t(element_type, shape, input);   // borrows; no alloc, no copy
request->set_tensor(name, t);
```

The wrapping constructor is documented and stable in OV; the safety
contract is that `input` must outlive the inference call. This is
straightforward to satisfy by holding the input bytes (`Vec<u8>` from
the Rust side) for the duration of `infer()` and dropping after
`get_output_tensor()` is consumed.

### Decision: Python tree removal

The 1.29× gap on distributed is no longer a hard "most performant for
the user" blocker; it's a tradeoff. Single-binary deployment + memory
safety + tokio concurrency vs 23% slower distributed inference. Bench
the user's actual workload mix before deciding. For now Python remains
on this branch.

## What's working today

`cargo test --workspace` on macOS — **63 passing, 0 failures.**

| Crate | Tests | What it does |
|---|---:|---|
| `tahoma-types` | 13 | GenerationTask, Chunk, ShardSpec, ShardPlan, PeerLayout |
| `tahoma-topology` | 4 | NodeInfo, EdgeMetrics, in-memory graph |
| `tahoma-transport` | 5 | Async tokio TCP relay; **wire-format identical to Python** |
| `tahoma-engine` | 0 | Engine + Builder traits |
| `tahoma-engine-mock` | 4 | Deterministic word-echo engine |
| `tahoma-ov-genai-shim` | 3 | C++ FFI shim around openvino-genai |
| `tahoma-engine-openvino` | 5 | OvGenaiEngine using the shim |
| `tahoma-runner` | 3 | Per-stage Runner; concurrent `generate()` |
| `tahoma-api` | 3 | axum: /health, /v1/models, /v1/chat/completions (+SSE) |
| `tahoma-cli` | 0 | clap CLI; `tahoma worker` flag set |
| `tahoma-discovery` | 2 | mDNS via mdns-sd; populates Topology |
| `tahoma-download` | 3 | Local registry + HF snapshot pull (hf-hub) |
| `tahoma` | — | Binary entry point |
| `tahoma-tests-e2e` | 2 | **Real binary spawn** — built `tahoma` exe, /health poll, /v1/models, /v1/chat/completions, concurrent request fan-out |

End-to-end smoke: the `tahoma` binary serves valid OpenAI
chat-completions JSON against the mock engine.

```bash
cd rust && cargo build -p tahoma
./target/debug/tahoma worker --rank 0 --total 1 \
    --model mock --engine mock --api :8000

curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"mock","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'
```

## What's deferred

In rough priority order:

1. **`ov-runtime` + `ov-dist-spec` engines.** Need lower-level `openvino`
   crate (Core/CompiledModel/InferRequest) + the v5-stage-shard wire
   protocol. Significant work; tracked separately.
2. **Real OpenVINO build on alpha.** Blocked on Windows toolchain
   prereqs (see Windows Setup below).
3. **Full pytest parity.** Currently 45 Rust tests vs Python's 123.
   Most gaps are in the engines + dist-spec protocol not yet ported.
4. **A/B benchmark Python vs Rust** on alpha — needs the real OV build
   first.
5. **Discovery + download integration tests.** Crates compile + unit-test
   today; full mDNS round-trip + HF pull e2e are runtime tests deferred.

## Windows setup (alpha + charlie)

### What's already installed (autonomous, this session)

* **Rust toolchain 1.95.0** on both alpha and charlie via headless
  `rustup-init.exe -y --default-toolchain stable`.
* **MSVC C++ build tools** on alpha (Visual Studio 2022 BuildTools
  with `Microsoft.VisualStudio.Workload.VCTools;includeRecommended`
  + Windows 11 SDK 22621) installed via the silent
  `vs_buildtools.exe modify` path. Verified `cl.exe` + `vcvars64.bat`
  present.
* **OpenVINO GenAI 2026.1 Windows SDK** (~208 MB, includes C++ headers
  for `llm_pipeline.hpp`, `generation_config.hpp`, `tokenizer.hpp` plus
  `openvino_genai.lib` import library) downloaded from
  `https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.1/windows/openvino_genai_windows_2026.1.0.0_x86_64.zip`
  and extracted to
  `C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\`.
  Set `INTEL_OPENVINO_DIR` to that directory before building.
* **Tahoma source synced** to `C:\Users\cascadia\tahoma-rust\` (also at
  `C:\tahoma\rust\` but that path is SAC-blocked — see below).

### Hard blockers requiring user intervention

These prevent the autonomous run from completing on-hardware e2e
validation. Both alpha and charlie are **Windows 11 Home** with
**Smart App Control** (SAC) enforced (Code Integrity status `2` on
both). SAC blocks every unsigned `.exe` cargo produces in
`target/debug/build/...` with `(os error 4551)` — including the
`thiserror` and `getrandom` proc-macro build scripts that every Rust
crate depends on.

Tested paths that all hit the same SAC block:

* Build under `C:\tahoma\rust\` — blocked.
* Build under `C:\Users\cascadia\tahoma-rust\` — blocked.
* `cargo check` (no link, just type-check) — also blocked because
  proc-macro crates still need build-script execution.

Possible fixes the user needs to choose:

1. **Disable Smart App Control** on at least one AI PC.
   *Caveat:* SAC is one-way. Once disabled it cannot be re-enabled
   without reinstalling Windows. Settings → Privacy & security →
   Windows Security → App & browser control → Smart App Control →
   "Off". This is the **fastest unblock**.
2. **Use WSL2** instead of native Windows. WSL Linux processes are
   not subject to SAC. `Ubuntu-24.04` is registered on alpha but the
   ext4 disk is missing (`HCS/ERROR_PATH_NOT_FOUND`); `wsl --install
   -d Ubuntu-24.04` started but hit an `HCS_E_CONNECTION_TIMEOUT`
   creating the VM. A Windows reboot typically clears that. After
   reboot:
   ```powershell
   wsl --unregister Ubuntu-24.04
   wsl --install -d Ubuntu-24.04
   wsl -d Ubuntu-24.04 -- bash
   # inside Ubuntu:
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   ```
   Then download the **Linux** OpenVINO 2026.1 archive (separate URL,
   `openvino_genai_ubuntu24_2026.1.0.0_x86_64.tgz`) and build with
   `--features openvino` from inside WSL.
3. **Use a different machine** without SAC (any Windows 11 Pro/
   Enterprise install, any Linux box).

### Build command (once unblocked)

```powershell
$env:INTEL_OPENVINO_DIR = 'C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64'
cd C:\Users\cascadia\tahoma-rust
& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
cargo build -p tahoma --release --features openvino
```

### Run (once built)

```powershell
.\target\release\tahoma.exe worker --rank 0 --total 1 `
    --engine ov-genai --device GPU `
    --model C:\cascadia\models\llama-3.1-8b-int4 `
    --ov-cache-dir C:\cascadia\ov_cache_genai `
    --api :8000
```

Once any of the unblock paths above is taken, the e2e validation
(equivalent to the Python PR #2 e2e matrix) becomes scriptable from
Mac via SSH.

## Architectural notes

* Workspace layout follows cascadia's "one concern per crate"
  discipline (see `/Users/tatef/Workspaces/cascadia/Cargo.toml`).
  `tahoma-types` plays the role cascadia's `cascadia-protocol` plays
  (zero-dep wire/value types).
* Trait bounds: `Engine` and `Builder` are `Send` (not `Sync`); Runner
  wraps them in `parking_lot::Mutex` so the runner itself is `Sync`
  and shareable across axum handlers via `Arc<Runner>`.
* The OpenVINO C++ FFI shim defaults to **stub mode** (no link, runtime
  errors only) so dev iteration on macOS / CI Linux without OpenVINO
  installed stays fast. Real link is gated behind `--features openvino`.
* Wire format for activation transport is **byte-identical to
  `tahoma/worker/transport.py`** — Python and Rust ranks can interop
  during the migration. dtype codes: 0=f32, 1=f16, 2=i8, 3=i32, 4=i64.

## Decoupling from cascadia

Per the design constraint ("cascadia may depend on tahoma; tahoma may
not depend on cascadia"):

* **No cascadia imports anywhere in `rust/`.** Verified by grep.
* Tahoma's crates have stable public APIs (no `pub(crate)` on items
  that should be exported); cascadia could add tahoma crates as
  Cargo dependencies in the future.
* Specifically `tahoma-types`, `tahoma-topology`, `tahoma-transport`,
  `tahoma-engine` are the most reusable surface — they encode patterns
  cascadia's full-model-per-node design doesn't have today.
