# Quickstart

First, how you got `cascadia` decides what you need:

| You have | To run real models you need | Start at |
|---|---|---|
| A **release bundle** ([Releases](https://github.com/labscommunity/cascadia/releases)) | nothing to build — the OpenVINO runtime is inside it (on Linux, GPU inference still needs the Intel GPU runtime stack: [INSTALL.md](INSTALL.md)) | [§2](#2-real-inference-on-one-intel-machine) (skip the build) |
| A **clone of the repo** | Rust; plus the OpenVINO GenAI SDK + a C++ toolchain for real inference ([INSTALL.md](INSTALL.md)) | [§1](#1-five-minutes-no-openvino-works-anywhere) |

From a bundle, run the binary from inside the unpacked directory (`./cascadia`,
or `.\cascadia.exe` on Windows) — it is not on your PATH. Everything below is
written as `cascadia …`.

---

## 1. Five minutes, no OpenVINO (works anywhere)

This proves your build is good and shows you the API/CLI surface, using
the built-in `mock` engine (deterministic word-echo — no model download,
no GPU).

```bash
# Build (stub mode — Rust only):
cargo build --release -p cascadia

# Check your environment:
./target/release/cascadia doctor

# Run the mock engine on a single machine with an OpenAI-compatible API:
./target/release/cascadia run mock-model --engine mock
```

In another terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "mock-model",
  "messages": [{"role": "user", "content": "Capital of France?"}]
}'
```

You should get a JSON chat-completion response back. That's the whole
request path working end to end. ✅

---

## 2. Real inference on one Intel machine

**From a release bundle?** Skip straight to `cascadia doctor` below — the
OpenVINO runtime ships inside the bundle, so there is nothing to build and no
SDK to install.

**Building from source?** Get OpenVINO set up first — see
**[INSTALL.md](INSTALL.md)** (C++ toolchain, the GenAI SDK, and on Linux the
GPU runtime stack) — then build with the feature enabled:

```bash
INTEL_OPENVINO_DIR=/path/to/openvino_genai_2026.2.0.0 \
  cargo build --release -p cascadia --features openvino
```

Either way, from here on it's the same:

```bash
# Confirm OpenVINO can see your GPU (not just CPU):
cascadia doctor

# Export a model to a 1-stage INT4 shard. Cascadia serves models from
# disk — it does not download or convert at run time. `cascadia shard`
# is what fetches from HuggingFace (needs the export deps, see INSTALL.md).
cascadia shard --model unsloth/Meta-Llama-3.1-8B-Instruct \
             --output-dir ~/cascadia/llama-8b-1stage \
             --num-stages 1 --quantization int4

cascadia run ~/cascadia/llama-8b-1stage --engine ov-runtime --device GPU
```

The export runs once; subsequent starts hit the OpenVINO kernel cache and
are fast. Hit the same `:8000` endpoint as above.

> Already have a whole-model OpenVINO IR (e.g. a pre-exported `*-int4-ov`
> directory)? Point `cascadia run <dir>` at it and it serves through the
> `ov-genai` engine — see [docs/engines/ov-genai.md](docs/engines/ov-genai.md).

> `cascadia run` is single-machine sugar — it picks the `ov-genai` engine,
> `GPU` device, and an API on `:8000`. For full control use `cascadia
> worker` (see `--help`).

---

## 3. Across two machines (the killer demo)

Run a model bigger than one laptop's RAM by splitting it across two. See
the **Two machines** section of the [README](README.md#two-machines-pipeline-parallel),
which walks through `cascadia shard` + one `cascadia worker` command per
node. Use `cascadia discover` to find peers' addresses on your LAN.

---

## Where to go next

- **[INSTALL.md](INSTALL.md)** — full setup, OpenVINO, Docker.
- **[docs/CLI.md](docs/CLI.md)** — every command and flag.
- **[docs/SHARDING.md](docs/SHARDING.md)** — supported models, picking stage counts.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — building, testing, the crate layout.
- `cascadia <command> --help` — every command is self-documenting.
