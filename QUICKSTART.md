# Quickstart

Two paths: a 5-minute stub run that works on any machine, and the real
thing on Intel hardware. If you just cloned the repo, start here.

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
curl http://localhost:8000/v1/chat/completions -d '{
  "model": "mock-model",
  "messages": [{"role": "user", "content": "Capital of France?"}]
}'
```

You should get a JSON chat-completion response back. That's the whole
request path working end to end. ✅

---

## 2. Real inference on one Intel machine

First get OpenVINO set up — see **[INSTALL.md](INSTALL.md)** (C++ toolchain,
the GenAI SDK, and on Linux the GPU runtime stack). Then:

```bash
INTEL_OPENVINO_DIR=/path/to/openvino_genai_2026.1.0.0 \
  cargo build --release -p cascadia --features openvino

# Confirm OpenVINO can see your GPU (not just CPU):
cascadia doctor

# Run a real model (auto-downloads from HuggingFace on first use,
# cached under ~/.cache/cascadia/models/):
cascadia run unsloth/Meta-Llama-3.1-8B-Instruct
```

The model is fetched once and cached; subsequent runs are fast. Hit the
same `:8000` endpoint as above.

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
- **[docs/SHARDING.md](docs/SHARDING.md)** — supported models, picking stage counts.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — building, testing, the crate layout.
- `cascadia <command> --help` — every command is self-documenting.
