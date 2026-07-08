# Installing Cascadia

Three ways to run Cascadia, depending on what you want:

| Goal | How | Needs |
| --- | --- | --- |
| Run it on Intel hardware | **Prebuilt bundle** (below) | an Intel driver; (Linux GPU) the runtime stack |
| Try it / develop / CI | **Stub mode** build | Rust only |
| Build real inference yourself | **OpenVINO mode** build | Rust + C++ toolchain + OpenVINO GenAI SDK + (Linux) GPU runtime |

After any install, run **`cascadia doctor`** — it checks your toolchain and tells you whether OpenVINO can actually see your GPU (a step that fails silently otherwise).

---

## Prebuilt binaries (recommended)

Each [GitHub Release](https://github.com/labscommunity/cascadia/releases) ships self-contained bundles with the OpenVINO runtime included — no SDK install, no `INTEL_OPENVINO_DIR`:

- **Windows** — `cascadia-<ver>-windows-x86_64.zip`: unzip, run `cascadia.exe doctor`. Needs only a current Intel graphics driver (the OpenCL/GPU runtime ships inside it).
- **Linux** — `cascadia-<ver>-linux-x86_64.tar.gz`: untar, run `./cascadia doctor`. Bundled libraries load from `lib/` beside the binary. Needs glibc 2.35+ (Ubuntu 22.04 or newer) and, for GPU inference, the Intel GPU runtime stack below (`./scripts/setup-openvino.sh`).

Python is not needed to run anything — only `cascadia shard` uses it (see "Export-time Python" below).

---

## Stub mode (no OpenVINO)

The fast path. Builds on macOS / Linux / Windows with nothing but Rust. The OpenVINO-backed engines return a clean runtime error; the `mock` engine works, so you can exercise the full API, transport, and multi-stage plumbing.

```bash
rustup default stable        # need 1.85+
cargo build --release -p cascadia
./target/release/cascadia doctor
```

---

## OpenVINO mode (real inference)

Three things have to be in place. `cascadia doctor` verifies all three.

### 1. C++ build toolchain

The FFI shim (`cascadia-ov-genai-shim`) compiles C++ against the OpenVINO GenAI headers.

- **Linux:** `g++` ≥ 12 (`sudo apt install g++`)
- **Windows:** Visual Studio 2022 Build Tools; build from a *Developer Command Prompt for VS 2022*

### 2. OpenVINO GenAI SDK

Download the **OpenVINO GenAI 2026.2+** archive for your platform from Intel:

- Archives: <https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/>
- Or the Intel download center: <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html>

Pick the build matching your OS (e.g. `openvino_genai_ubuntu24_2026.2.0.0_x86_64.tar.gz`). Extract it and point `INTEL_OPENVINO_DIR` at the **extracted SDK root** — the directory that contains `runtime/`, `setupvars.sh`, etc.:

```bash
export INTEL_OPENVINO_DIR=/opt/intel/openvino_genai_2026.2.0.0
```

> If you build with `--features openvino` and this isn't set (or points at the wrong place), the build now fails fast with an explanatory message instead of an opaque compile/link error.

### 3. (Linux GPU only) the GPU runtime stack

This is the step most people miss. OpenVINO GPU inference is **not** a single install — the GPU plugin needs the Intel Compute Runtime, OpenCL ICD, and Level-Zero loader, and your user must be in the `render` group. **Without these, OpenVINO silently sees only the CPU**, even with a working driver and a healthy `clinfo`.

Run the helper (Ubuntu 22.04 / 24.04):

```bash
./scripts/setup-openvino.sh
```

…or do it by hand:

```bash
sudo apt-get install -y ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero
sudo usermod -a -G render "$USER"   # then LOG OUT/IN — group changes don't apply to the current shell
```

On **Windows** the OpenCL runtime ships inside the Intel graphics driver, so just install the latest driver and reboot. `scripts/setup-openvino.ps1` checks your driver, MSVC, and SDK env.

On **macOS** there is no Intel GPU runtime — use stub mode for dev only.

### Build and verify

```bash
INTEL_OPENVINO_DIR=/opt/intel/openvino_genai_2026.2.0.0 \
  cargo build --release -p cascadia --features openvino

./target/release/cascadia doctor   # should list a GPU device, not just CPU
```

The binary is statically linked apart from the OpenVINO dynamic libraries. To run it elsewhere, copy the binary plus `INTEL_OPENVINO_DIR/runtime/lib/intel64/` (Linux) or `runtime/bin/intel64/Release/` + `runtime/3rdparty/tbb/bin/` (Windows) onto the target's library path.

---

## Export-time Python (only for `cascadia shard`)

Sharding a HuggingFace model into per-stage IRs runs a bundled Python exporter. This is needed **only** when you run `cascadia shard` — not at inference time, and not on workers. Install once, anywhere with the RAM:

```bash
pip install torch transformers openvino safetensors huggingface_hub nncf
```

`nncf` is optional (INT4 quantization); without it, sharding falls back to FP16. `cascadia doctor` reports whether these are present.

---

## Docker (reference image)

A `Dockerfile` bundles the OpenVINO + Level-Zero stack so you can skip the host install. GPU inference requires passing the host GPU in (`--device /dev/dri`) and a matching host driver. See the header comments in `Dockerfile`.

---

## Troubleshooting

Run `cascadia doctor` first — it diagnoses most of these. See also the README's **Troubleshooting** section for runtime issues (config.json, peer connect timeouts, Windows SSH).
