<p align="center">
  <img src="docs/assets/logo.svg" alt="Cascadia" width="520">
</p>

> Run any model on Intel hardware.

Cascadia distributes LLM inference across Intel laptops, desktops, and AI PCs. Shard a model across the machines you already have and serve it through an OpenAI-compatible API — no cloud, no NVIDIA GPUs required.

## Status

[![ci](https://github.com/labscommunity/cascadia/actions/workflows/ci.yml/badge.svg)](https://github.com/labscommunity/cascadia/actions/workflows/ci.yml)

**Pre-alpha.** Working on Intel AI PCs (Lunar Lake / Arrow Lake / Panther Lake / Battlemage Arc). Single Rust binary per node; no Python runtime at the worker. Five engines: `mock`, `ov-genai`, `ov-runtime`, `ov-dist-spec`, and a CPU-targeted `sparse-moe` engine for large mixture-of-experts models like Kimi K2.6. Intel Arc A-series discrete GPUs and Xeon CPU-only servers are on the roadmap.

## Why Cascadia

Frontier models don't fit on a single laptop. Cloud APIs are expensive, opaque, and require sending your data offsite. Cascadia lets you point a few Intel machines at each other and run models that none of them could handle alone.

## Build from source

Cascadia is a Cargo workspace at the repo root. Two build modes:

```bash
# Stub mode — no OpenVINO link required. Good for dev / CI on macOS / Linux.
# Engines that need OV will return a clean runtime error.
cargo build --release -p cascadia

# Real OV mode — links against openvino-genai 2026.1.0+. Required to run
# the OV engines on real Intel hardware.
INTEL_OPENVINO_DIR=/path/to/openvino_genai_<platform>_2026.2.0.0 \
  cargo build --release -p cascadia --features openvino
```

The resulting binary lands at `target/release/cascadia` (`cascadia.exe` on Windows). It is statically linked apart from the OV dynamic libraries — copy the binary plus `INTEL_OPENVINO_DIR/runtime/lib/intel64/` (Linux) or `runtime/bin/intel64/Release/` + `runtime/3rdparty/tbb/bin/` (Windows) to the worker host's library path.

**New here? Start with [QUICKSTART.md](QUICKSTART.md)** — a 5-minute stub run that needs nothing but Rust. Full setup (C++ toolchain, the OpenVINO GenAI SDK, and on Linux the GPU runtime stack) is in **[INSTALL.md](INSTALL.md)**; on Linux, `./scripts/setup-openvino.sh` installs the GPU runtime packages for you.

Build prerequisites:

- Rust 1.85+ (`rustup default stable`)
- For `--features openvino`: a Visual Studio 2022 Build Tools install (Windows) or `g++` ≥ 12 (Linux), plus the OpenVINO GenAI 2026.1+ SDK ([INSTALL.md](INSTALL.md) has the download links and the Linux GPU runtime steps)

> After building, run **`cascadia doctor`**. It checks your toolchain and — crucially — tells you whether OpenVINO can actually see your GPU. On Intel AI PCs the GPU can be invisible to OpenVINO even with a working driver, and that failure is otherwise silent (you just get slow CPU inference). `doctor` makes it loud and tells you how to fix it.

## Quick start

See [QUICKSTART.md](QUICKSTART.md) for the full walk-through (including a
no-OpenVINO path that runs on any machine).

### Single machine

```bash
# 0. Check the environment + that OpenVINO sees your GPU:
cascadia doctor

# 1. Run a model. `run` is single-machine sugar — it picks the ov-genai
#    engine, GPU device, and an OpenAI API on :8000. The model is fetched
#    from HuggingFace on first use and cached under ~/.cache/cascadia/models/.
cascadia run unsloth/Meta-Llama-3.1-8B-Instruct

# 2. In another terminal:
curl http://localhost:8000/v1/chat/completions -d '{
  "model": "llama-3.1-8b",
  "messages": [{"role": "user", "content": "Capital of France?"}]
}'
```

You should get back a JSON chat-completion whose `choices[0].message.content`
answers the question — that's the full path (tokenizer → engine → API)
working. For full control over engine, device, ports, and the speculative
/ sparsity knobs, use `cascadia worker` (`cascadia worker --help`):

```bash
cascadia worker --rank 0 --total 1 --engine ov-genai --device GPU \
              --model unsloth/Meta-Llama-3.1-8B-Instruct \
              --api :8000
```

### Two machines (pipeline parallel)

Cascadia ships its own sharder. One time, on whichever machine has the
RAM + a Python install (any node, or your laptop):

```bash
# Pip-install the export-time deps once (~3 GB; not needed at runtime):
pip install torch transformers openvino safetensors huggingface_hub nncf

# Shard a HuggingFace model into 2 stages with INT4 weights:
cascadia shard --model unsloth/Meta-Llama-3.1-8B-Instruct \
             --output-dir ~/cascadia/llama-8b-2stage \
             --num-stages 2 --quantization int4
```

This produces `~/cascadia/llama-8b-2stage/` with `pipeline_config.json`,
`tokenizer/`, and `stage_0/` + `stage_1/`. Copy the directory to each
node (`scp -r` / `rsync`), or re-shard separately on each node — pick
whichever is faster on your network.

Then run **one worker command per node** (start the last stage first so
the first stage finds it; if it isn't up yet the first stage prints a
clear "waiting for downstream peer" line and retries rather than hanging
silently):

```bash
# Node B (last stage, listens for activations):
cascadia worker --rank 1 --total 2 --engine ov-runtime --device GPU \
              --model ~/cascadia/llama-8b-2stage \
              --listen :9100

# Node A (first stage, serves the API):
cascadia worker --rank 0 --total 2 --engine ov-runtime --device GPU \
              --model ~/cascadia/llama-8b-2stage \
              --next 10.0.0.2:9100 --api :8000
```

Not sure of a node's address? Run `cascadia discover` on either box to
list Cascadia peers on the LAN and the `host:port` to pass to `--next`.

Add `--engine ov-dist-spec --draft-model unsloth/Llama-3.2-1B-Instruct --spec-k 4` on rank 0 for distributed speculative decoding (see [docs/engines/ov-dist-spec.md](docs/engines/ov-dist-spec.md)).

### Supported model families

`cascadia shard` works today with Llama (1–3.3), Mistral (7B, NeMo,
Small 3.x text), Qwen2 / Qwen2.5, Qwen3 dense, DeepSeek R1 Distills (Qwen
and Llama variants), Phi-3, Phi-4 / Phi-4-mini (partial rotary), and
Gemma 1 / Gemma 2 (Gemma 2 with logit softcapping + the 4-norm structure;
sliding-window attention is treated as full-causal, so output is exact
within the window). Gemma 4 (E2B / E4B / 31B) exports through a dedicated
path (`tools/export_gemma4.py`, auto-dispatched by `cascadia shard`).

Mixture-of-experts and other architecturally-incompatible families —
Llama 4, Qwen3-MoE, Mixtral, gpt-oss, full DeepSeek-V2/V3, Gemma 3, the
Gemma 4 26B-A4B MoE variant, and Mamba / hybrids — are detected and
rejected up front with a clear error. See [docs/SHARDING.md](docs/SHARDING.md)
and [docs/architectures/](docs/architectures/) for the full per-family
status table and per-family deep-dives.

### List engines

```
$ cascadia engines
  mock           deterministic word-echo engine for tests
  ov-genai       single-stage openvino_genai.LLMPipeline; FastDraft + Prompt Lookup
  ov-runtime     multi-stage stateful KV cache; pre-exported per-stage v3+ shards
  ov-dist-spec   multi-stage spec decode (mask-based KV rewind); v5 shards
  sparse-moe     Kimi K2.6 sparse top-8 dispatch; AVX-512 int4 GEMM + Rust shells
```

## Architecture

See [`CLAUDE.md`](CLAUDE.md) for design rationale. Key crates:

- `cascadia-api/` — OpenAI-compatible HTTP (axum)
- `cascadia-runner/` — Per-stage runner; concurrent-safe chunk streaming
- `cascadia-engine/` — `Engine` + `Builder` traits — the plugin seam
- `cascadia-engine-openvino/` — Three OV engines (`ov-genai`, `ov-runtime`, `ov-dist-spec`)
- `cascadia-engine-sparse-moe/` — Kimi K2.6-style sparse-MoE engine; routes only the top-k experts per token
- `cascadia-int4-gemm/` — hand-rolled AVX-512 INT4 GEMM kernels for the MoE expert path
- `cascadia-ov-genai-shim/` — C++ FFI shim wrapping `openvino-genai`
- `cascadia-transport/` — TCP activation relay (length-prefixed tensor wire format)
- `cascadia-topology/` — Per-link latency + bandwidth measurements
- `cascadia-discovery/` — mDNS peer discovery on `_cascadia._tcp.local.`

Per-engine guides:

- [ov-genai](docs/engines/ov-genai.md) — single-stage `openvino_genai.LLMPipeline`; FastDraft + Prompt Lookup
- [ov-runtime](docs/engines/ov-runtime.md) — multi-stage stateful KV cache
- [ov-dist-spec](docs/engines/ov-dist-spec.md) — multi-stage spec with mask-based rewind
- sparse-moe — CPU-targeted sparse MoE (Kimi K2.6). Tuning: [docs/A3_TOPK_REDUCTION.md](docs/A3_TOPK_REDUCTION.md), [docs/perf/CHESS_PER_CHANNEL.md](docs/perf/CHESS_PER_CHANNEL.md). Consumes a `manifest.json` + per-expert shard tree (not `cascadia shard` output).

## Cluster

- **Placement is manual today.** Operators set `--rank` / `--total` / `--listen` / `--next host:port` on each worker. `cascadia discover` browses the LAN and lists peer `host:port`s to plug into `--next`, but workers still need explicit ranks — full auto-ring formation (rank assignment + stage-count agreement + pipeline ordering) is not yet wired into `cascadia worker` (tracked in [#52](https://github.com/labscommunity/cascadia/issues/52); see [docs/STATUS.md](docs/STATUS.md) "Known limitations").
- **Device profiling.** `cascadia profile-devices --model <dir>` benchmarks each OV device (iGPU / NPU / CPU) on a host — cold-compile time + decode tok/s — and writes `device_profile.json`. Use it to pick `--device` today; it's step 1 toward automatic placement. See [docs/perf/DEVICE_PROFILE.md](docs/perf/DEVICE_PROFILE.md).
- **Tensor parallelism:** type-system plumbing only; no engine implements it yet. See [docs/architecture/tensor-parallelism.md](docs/architecture/tensor-parallelism.md).

## Deploying

Cascadia does not daemonize itself — run it under systemd / NSSM / launchd. See [`docs/deploy/`](docs/deploy/) for a systemd unit template and Windows / macOS recipes. Cascadia handles `SIGTERM` cleanly (`runner.close()` then exit 0).

**Security**: the HTTP API and inter-stage TCP relay are plaintext and unauthenticated. Bind only to trusted networks (LAN, loopback) or terminate TLS + auth at a reverse proxy in front of `--api`. See [`docs/STATUS.md`](docs/STATUS.md) "Security model" for the threat model and Phase 14 hardening details.

## Troubleshooting

**`config.json not in <model dir>`** — `ov-runtime` reads the HF model `config.json` from the shard's tokenizer dir to derive rotary parameters. v5 shards from rainier do not bundle `config.json`; copy it from the source model's HF cache (`~/.cache/huggingface/hub/models--<repo>/snapshots/<sha>/config.json`) into the shards root.

**`could not connect to … within 30s`** — Start the downstream worker first; the upstream waits for the upstream socket to bind. Check `--listen` on the downstream matches `--next` on the upstream and that the host's firewall allows the port.

**Worker dies silently when SSH session closes** — On Windows OpenSSH the child process is tied to the SSH parent. Run workers under systemd / NSSM / Task Scheduler in production; for ad-hoc testing keep the SSH session attached.

`cascadia doctor` diagnoses most environment/hardware issues (toolchain, OpenVINO device visibility) before they bite.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the build/test gate, crate layout, and commit conventions.

## License

Apache-2.0 (target). The repository is private during incubation.
