# tahoma Rust port — status

Tracking the Python → Rust hard rewrite landed under `rust/` on the
`feat/rust-port` branch. Updated whenever a phase lands.

## What's working today

`cargo test --workspace` on macOS — **45 passing, 0 failures.**

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

To validate the C++ FFI shim against real OpenVINO, the AI PCs need:

1. **Rust toolchain** (~150 MB):
   ```powershell
   Invoke-WebRequest https://win.rustup.rs/x86_64 -OutFile rustup-init.exe
   .\rustup-init.exe -y --default-toolchain stable
   ```

2. **MSVC C++ build tools** (~7 GB, requires admin):
   - VS 2022 Build Tools is partially installed at
     `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`
     but the **C++ workload is missing**.
   - Add it via Visual Studio Installer → Modify → "Desktop development
     with C++" workload (includes MSVC compiler, Windows SDK,
     CMake support).
   - Or scripted: `vs_BuildTools.exe --add Microsoft.VisualStudio.Workload.VCTools --quiet`

3. **OpenVINO 2026.x SDK** (~3 GB) with C++ headers:
   - Download from intel.com/openvino → "OpenVINO Toolkit" archive
     (not the runtime-only pip package).
   - Extract somewhere stable, e.g. `C:\openvino_2026\`.
   - Set `INTEL_OPENVINO_DIR` env var to that root.
   - The C++ shim's build.rs reads this env var to locate
     `runtime/include` (headers) and `runtime/lib/intel64`
     (libopenvino_genai.lib).

4. **Build with the openvino feature**:
   ```powershell
   cd C:\tahoma\rust
   cargo build -p tahoma --release --features openvino
   ```

5. **Run**:
   ```powershell
   .\target\release\tahoma.exe worker --rank 0 --total 1 `
       --engine ov-genai --device GPU `
       --model C:\cascadia\models\llama-3.1-8b-int4 `
       --ov-cache-dir C:\cascadia\ov_cache_genai `
       --api :8000
   ```

Once steps 1–3 are completed by the user, the e2e validation
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
