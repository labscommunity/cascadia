# Tahoma

[![ci](https://github.com/labscommunity/tahoma/actions/workflows/ci.yml/badge.svg)](https://github.com/labscommunity/tahoma/actions/workflows/ci.yml)

> Run any model on Intel hardware.

Tahoma distributes LLM inference across Intel laptops, desktops, and AI PCs. Shard a model across the machines you already have and serve it through an OpenAI-compatible API — no cloud, no NVIDIA GPUs required.

## Status

**Pre-alpha.** Working on Intel AI PCs (Lunar Lake / Arrow Lake / Panther Lake / Battlemage Arc). Single Rust binary per node; no Python runtime. Four engines: `mock`, `ov-genai`, `ov-runtime`, `ov-dist-spec`. Intel Arc-only Xeon CPU-only nodes are on the roadmap.

The Rust port is the only implementation as of 2026-05-02. The earlier Python prototype was removed at the end of Phase 12.

## Why Tahoma

Frontier models don't fit on a single laptop. Cloud APIs are expensive, opaque, and require sending your data offsite. Tahoma lets you point a few Intel machines at each other and run models that none of them could handle alone.

## Build from source

Tahoma is a Cargo workspace at the repo root. Two build modes:

```bash
# Stub mode — no OpenVINO link required. Good for dev / CI on macOS / Linux.
# Engines that need OV will return a clean runtime error.
cargo build --release -p tahoma

# Real OV mode — links against openvino-genai 2026.1.0+. Required to run
# the OV engines on real Intel hardware.
INTEL_OPENVINO_DIR=/path/to/openvino_genai_<platform>_2026.1.0.0_x86_64 \
  cargo build --release -p tahoma --features openvino
```

The resulting binary lands at `target/release/tahoma` (`tahoma.exe` on Windows). It is statically linked apart from the OV dynamic libraries — copy the binary plus `INTEL_OPENVINO_DIR/runtime/bin/intel64/Release/` and `runtime/3rdparty/tbb/bin/` to the worker host's PATH.

Build prerequisites:

- Rust 1.75+ (`rustup default stable`)
- For `--features openvino`: a Visual Studio 2022 Build Tools install (Windows) or `g++` ≥ 12 (Linux), plus the OpenVINO GenAI 2026.1+ SDK download from intel.com

## Quick start

### Single machine

```bash
# Single-stage OV-GenAI engine (auto-export from HF model id):
tahoma worker --rank 0 --total 1 --engine ov-genai --device GPU \
              --model unsloth/Meta-Llama-3.1-8B-Instruct \
              --api :8000

# In another terminal:
curl http://localhost:8000/v1/chat/completions -d '{
  "model": "llama-3.1-8b",
  "messages": [{"role": "user", "content": "Capital of France?"}]
}'
```

### Two machines (pipeline parallel)

Pre-export per-stage shards once with rainier's exporter, then:

```bash
# On node B (last stage, listens for activations):
tahoma worker --rank 1 --total 2 --engine ov-runtime --device GPU \
              --model /path/to/shards \
              --listen 10.0.0.2:9100

# On node A (first stage, serves the API):
tahoma worker --rank 0 --total 2 --engine ov-runtime --device GPU \
              --model /path/to/shards \
              --next 10.0.0.2:9100 --api :8000
```

Add `--engine ov-dist-spec --draft-model unsloth/Llama-3.2-1B-Instruct --spec-k 4` on rank 0 for distributed speculative decoding (see [docs/engines/ov-dist-spec.md](docs/engines/ov-dist-spec.md)).

### List engines

```
$ tahoma engines
  mock           deterministic word-echo engine for tests
  ov-genai       single-stage openvino_genai.LLMPipeline; FastDraft + Prompt Lookup
  ov-runtime     multi-stage stateful KV cache; pre-exported per-stage v3+ shards
  ov-dist-spec   multi-stage spec decode (mask-based KV rewind); v5 shards
```

## Architecture

See [`CLAUDE.md`](CLAUDE.md) for design rationale. Key crates:

- `tahoma-api/` — OpenAI-compatible HTTP (axum)
- `tahoma-runner/` — Per-stage runner; concurrent-safe chunk streaming
- `tahoma-engine/` — `Engine` + `Builder` traits — the plugin seam
- `tahoma-engine-openvino/` — Three OV engines (`ov-genai`, `ov-runtime`, `ov-dist-spec`)
- `tahoma-ov-genai-shim/` — C++ FFI shim wrapping `openvino-genai`
- `tahoma-transport/` — TCP activation relay (length-prefixed tensor wire format)
- `tahoma-topology/` — Per-link latency + bandwidth measurements
- `tahoma-discovery/` — mDNS peer discovery on `_tahoma._tcp.local.`

Per-engine guides:

- [ov-runtime](docs/engines/ov-runtime.md) — multi-stage stateful KV cache
- [ov-dist-spec](docs/engines/ov-dist-spec.md) — multi-stage spec with mask-based rewind

## Cluster

- Auto-discovery: `--discover` (mDNS) advertises this node and browses for siblings in the same `TAHOMA_NAMESPACE`.
- Master election: lowest-lexicographic node id in a namespace wins; no explicit messaging.
- Tensor parallelism: foundation only — see [docs/architecture/tensor-parallelism.md](docs/architecture/tensor-parallelism.md).

## Deploying

Tahoma does not daemonize itself — run it under systemd / NSSM / launchd. See [`docs/deploy/`](docs/deploy/) for a systemd unit template and Windows / macOS recipes. Tahoma handles `SIGTERM` cleanly (`runner.close()` then exit 0).

**Security**: the HTTP API and inter-stage TCP relay are plaintext and unauthenticated. Bind only to trusted networks (LAN, loopback) or terminate TLS + auth at a reverse proxy in front of `--api`. See [`docs/STATUS.md`](docs/STATUS.md) "Security model" for the threat model and Phase 14 hardening details.

## Troubleshooting

**`config.json not in <model dir>`** — `ov-runtime` reads the HF model `config.json` from the shard's tokenizer dir to derive rotary parameters. v5 shards from rainier do not bundle `config.json`; copy it from the source model's HF cache (`~/.cache/huggingface/hub/models--<repo>/snapshots/<sha>/config.json`) into the shards root.

**`could not connect to … within 30s`** — Start the downstream worker first; the upstream waits for the upstream socket to bind. Check `--listen` on the downstream matches `--next` on the upstream and that the host's firewall allows the port.

**Worker dies silently when SSH session closes** — On Windows OpenSSH the child process is tied to the SSH parent. Run workers under systemd / NSSM / Task Scheduler in production; for ad-hoc testing keep the SSH session attached.

## License

Apache-2.0 (target). The repository is private during incubation.
