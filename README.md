<p align="center">
  <img src="docs/assets/logo.svg" alt="Cascadia" width="520">
</p>

<p align="center"><b>Run any model on Intel hardware.</b></p>

<p align="center">
  <a href="https://github.com/labscommunity/cascadia/actions/workflows/ci.yml"><img src="https://github.com/labscommunity/cascadia/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="license"></a>
  <img src="https://img.shields.io/badge/rust-1.89%2B-orange.svg" alt="rust 1.89+">
  <img src="https://img.shields.io/badge/status-pre--alpha-red.svg" alt="pre-alpha">
</p>

---

Cascadia distributes LLM inference across Intel laptops, desktops, and AI PCs. Shard a model across the machines you already have and serve it through an OpenAI-compatible API. No cloud or NVIDIA GPUs required.

Frontier models don't fit on a single laptop. Cloud APIs are expensive, opaque, and require sending your data offsite. Cascadia lets you point a few Intel machines at each other and run models that none of them could handle alone.

## Features

- **OpenAI-compatible API**: `/v1/chat/completions` with SSE streaming; point existing clients at it unchanged
- **Pipeline parallelism**: shard a model into stages and run each stage on a different machine, activations relayed over TCP
- **Built-in sharder**: `cascadia shard` cuts a HuggingFace model into INT4 per-stage shards; no external tooling
- **Seven engines**: `mock`, `ov-genai`, `ov-runtime`, `ov-dist-spec` (distributed speculative decoding), `gemma4`, a CPU-targeted `sparse-moe` engine for large mixture-of-experts models like Kimi K2.6 and MiniMax-M2, and `qwen36-moe` for Qwen3.6's hybrid MoE
- **Single static binary per node**: Rust only at runtime; no Python on workers
- **Zero-config peer discovery**: `cascadia discover` finds LAN peers over mDNS
- **`cascadia doctor`**: diagnoses the one failure everyone hits: OpenVINO silently not seeing your GPU

> [!NOTE]
> Cascadia is in **pre-alpha status**. It works on Intel AI PCs (Lunar Lake / Arrow Lake / Panther Lake) and Arc B-series (Battlemage) discrete GPUs. Intel Arc A-series discrete GPUs and Xeon CPU-only servers are on the roadmap.

## Quick start

**On an Intel machine?** Grab a self-contained bundle from [Releases](https://github.com/labscommunity/cascadia/releases): OpenVINO runtime included, no build, no SDK. Unpack it and run the binary from inside — it is not installed on your PATH:

```bash
# Linux
tar -xzf cascadia-<ver>-linux-x86_64.tar.gz && cd cascadia-<ver>-linux-x86_64
./cascadia doctor
```

```powershell
# Windows
Expand-Archive cascadia-<ver>-windows-x86_64.zip -DestinationPath .
cd cascadia-<ver>-windows-x86_64
.\cascadia.exe doctor
```

That's the whole install: no Rust, no OpenVINO SDK, no `INTEL_OPENVINO_DIR`. Commands below are written as `cascadia …`; from a bundle use `./cascadia` (`.\cascadia.exe` on Windows), or add the directory to your PATH.

Or build from source. You must have Rust installed; OpenVINO isn't required, and Cascadia will mock responses:

```bash
cargo build --release -p cascadia
./target/release/cascadia doctor
./target/release/cascadia run mock-model --engine mock
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "mock-model",
  "messages": [{"role": "user", "content": "Capital of France?"}]
}'
```

If you receive a JSON chat-completion back, it means the full path (API → engine → streaming) works.

There are follow-up steps for performing real inference in **[QUICKSTART.md](QUICKSTART.md)**.

## Installation

**Prebuilt bundles** for Linux and Windows x86_64 are on the [Releases page](https://github.com/labscommunity/cascadia/releases): binary + OpenVINO runtime libraries, ready to run. Building from source instead has two modes:

```bash
# Stub mode. Rust only. Good for dev / CI on macOS / Linux / Windows.
# Engines that need OpenVINO return a clean runtime error.
cargo build --release -p cascadia

# Real OpenVINO mode. Links against openvino-genai 2026.2.0+. Required
# for inference on real Intel hardware.
INTEL_OPENVINO_DIR=/path/to/openvino_genai_<platform>_2026.2.0.0 \
  cargo build --release -p cascadia --features openvino
```

Prerequisites: 
- Rust 1.89+; for `--features openvino`
- A C++ toolchain (VS 2022 Build Tools on Windows, `g++` ≥ 12 on Linux) and the OpenVINO GenAI SDK

**[INSTALL.md](INSTALL.md)** has download links, the Linux GPU-runtime steps (`scripts/setup-openvino.sh` automates them from a source checkout), and the Docker image.

> [!IMPORTANT]
> After building, run **`cascadia doctor`**. On Intel AI PCs, the GPU can be invisible to OpenVINO even with a working driver. That failure is otherwise silent (you would just get slow CPU inference). `doctor` detects the problem and tells you how to fix it.

## Usage

Every command and flag is catalogued in [docs/CLI.md](docs/CLI.md).

### Single machine

Cascadia serves models from a local directory — it does not download or convert at run time. Only `cascadia shard` fetches from HuggingFace (caching under `~/.cache/cascadia/models/`). Export once, then serve:

```bash
# Export to a 1-stage INT4 shard (export deps: see Installation above).
cascadia shard --model unsloth/Meta-Llama-3.1-8B-Instruct \
             --output-dir ~/cascadia/llama-8b-1stage \
             --num-stages 1 --quantization int4

# `run` is single-machine sugar: one stage, OpenAI API on :8000.
cascadia run ~/cascadia/llama-8b-1stage --engine ov-runtime --device GPU
```

Or serve a whole-model **OpenVINO IR** through the `ov-genai` engine, which adds FastDraft speculative decode and prompt-lookup. That layout comes from Intel's exporter, not `cascadia shard` — download a pre-exported INT4 IR (Intel publishes many under the [OpenVINO](https://huggingface.co/OpenVINO) org) or build one with `optimum-cli`, see [docs/engines/ov-genai.md](docs/engines/ov-genai.md):

```bash
cascadia run ~/models/llama-3.1-8b-int4-ov   # defaults to --engine ov-genai --device GPU
```

For full control over engine, device, ports, and the speculative / sparsity knobs, use `cascadia worker` (`cascadia worker --help`):

```bash
cascadia worker --rank 0 --total 1 --engine ov-genai --device GPU \
              --model ~/models/llama-3.1-8b-int4-ov \
              --api :8000
```

### Two machines (pipeline parallel)

Shard once, on whichever machine has the RAM and a Python install:

```bash
# Export-time deps (~3 GB; not needed at runtime). From a source checkout:
pip install -r tools/requirements.txt
# From a release bundle there is no tools/ — `cascadia doctor` prints the pinned line.

# Shard a HuggingFace model into 2 stages with INT4 weights:
cascadia shard --model unsloth/Meta-Llama-3.1-8B-Instruct \
             --output-dir ~/cascadia/llama-8b-2stage \
             --num-stages 2 --quantization int4
```

Copy the output directory to each node (`scp -r` / `rsync`), or re-shard separately on each node, whichever is faster on your network. Then run one worker per node: **start the last stage first** so the first stage finds it (if it isn't up yet, the first stage prints a clear "waiting for downstream peer" line and retries):

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

Not sure of a node's address? `cascadia discover` lists Cascadia peers on the LAN and the `host:port` to pass to `--next`.

Two stages on the **same host** (e.g. iGPU + dGPU in one box)? Link them over a Unix domain socket instead of loopback TCP — `--listen unix:/tmp/cascadia-1.sock` on the downstream, the same path as `--next` on the upstream. Measured 85–92 % lower per-frame round-trip latency for decode-sized frames (Unix only; see [docs/CLI.md](docs/CLI.md#unix-domain-sockets)).

For distributed speculative decoding, run **every** rank with `--engine ov-dist-spec` (they share a wire protocol) and give rank 0 `--draft-model ~/models/llama-3.2-1b-int4-ov --spec-k 4` — the draft is a local OpenVINO IR directory, not an HF id. See ([docs/engines/ov-dist-spec.md](docs/engines/ov-dist-spec.md)).

### Engines

```console
$ cascadia engines
  mock           deterministic word-echo engine for tests
  ov-genai       single-stage openvino_genai.LLMPipeline; FastDraft + Prompt Lookup
  ov-runtime     multi-stage stateful KV cache; pre-exported per-stage v3+ shards
  ov-dist-spec   multi-stage spec decode (mask-based KV rewind); v5 shards
  gemma4         Gemma 4 multi-stage (per-layer-type attn, KV-sharing, PLI); gemma4_cached_v1 shards
  sparse-moe     Kimi K2.6 (AVX-512 int4 GEMM + Rust MLA shells) or MiniMax-M2 (OV-IR shells); single-stage top-k expert dispatch
  qwen36-moe     Qwen3.6-35B-A3B staged chain (GatedDeltaNet + MoE); qwen3_5_moe IR-surgery shards
```

`sparse-moe` consumes a `manifest.json` + per-expert artefact tree, not `cascadia shard` output, see [docs/architectures/minimax-m2.md](docs/architectures/minimax-m2.md) and [docs/architectures/moe.md](docs/architectures/moe.md). MiniMax-M2 is the in-repo export path (`tools/export_minimax_m2.py`); the Kimi K2.6 artefacts come from an external pipeline that is not part of this repo. Tuning: [docs/perf/A3_TOPK_REDUCTION.md](docs/perf/A3_TOPK_REDUCTION.md), [docs/perf/CHESS_PER_CHANNEL.md](docs/perf/CHESS_PER_CHANNEL.md).

### Supported model families

`cascadia shard` works today with Llama (1–3.3), Mistral (7B, NeMo, Small 3.x text), Qwen2 / Qwen2.5, Qwen3 dense, DeepSeek R1 Distills (Qwen and Llama variants), Phi-3, Phi-4 / Phi-4-mini (partial rotary), and Gemma 1 / Gemma 2 (logit softcapping + the 4-norm structure; sliding-window attention is treated as full-causal, so output is exact within the window). Gemma 4 (E2B / E4B / 31B) exports through a dedicated path (`tools/export_gemma4.py`, auto-dispatched by `cascadia shard`).

Qwen3.5/3.6 hybrid MoE (`model_type: qwen3_5_moe`) is special-cased: `cascadia shard` dispatches it to a dedicated IR-surgery exporter and it serves through the `qwen36-moe` engine ([docs/architectures/qwen36-moe-support.md](docs/architectures/qwen36-moe-support.md)). Other mixture-of-experts and architecturally-incompatible families like Llama 4, Qwen3-MoE, Mixtral, gpt-oss, full DeepSeek-V2/V3, Gemma 3, the Gemma 4 26B-A4B MoE variant, and Mamba hybrids are detected and rejected up front with a clear error. See [docs/SHARDING.md](docs/SHARDING.md) and [docs/architectures/](docs/architectures/) for the full per-family status table and deep-dives.

## Architecture

Cascadia is a Cargo workspace; one concern per crate. The `Engine` + `Builder` traits in `cascadia-engine` are the plugin seam. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for design rationale and per-crate responsibilities. Key crates:

- `cascadia-api/`: OpenAI-compatible HTTP (axum)
- `cascadia-runner/`: Per-stage runner; concurrent-safe chunk streaming
- `cascadia-engine/`: `Engine` + `Builder` traits (the plugin seam)
- `cascadia-engine-openvino/`: Five OV engines (`ov-genai`, `ov-runtime`, `ov-dist-spec`, `gemma4`, `qwen36-moe`)
- `cascadia-engine-sparse-moe/`: Sparse-MoE engine; routes only the top-k experts per token
- `cascadia-int4-gemm/`: hand-rolled AVX-512 INT4 GEMM kernels for the MoE expert path
- `cascadia-ov-genai-shim/`: C++ FFI shim wrapping `openvino-genai`
- `cascadia-transport/`: TCP activation relay (length-prefixed tensor wire format)
- `cascadia-topology/`: Per-link latency + bandwidth measurements
- `cascadia-discovery/`: mDNS peer discovery on `_cascadia._tcp.local.`

### Cluster status

- **Placement is manual today.** Operators set `--rank` / `--total` / `--listen` / `--next host:port` on each worker. `cascadia discover` browses the LAN, but workers still need explicit ranks, full auto-ring formation is not yet wired into `cascadia worker` (tracked in [#89](https://github.com/labscommunity/cascadia/issues/89)).
- **Device profiling.** `cascadia profile-devices --model <dir>` benchmarks each OV device (iGPU / NPU / CPU) on a host and writes `device_profile.json`, step 1 toward automatic placement. See [docs/perf/DEVICE_PROFILE.md](docs/perf/DEVICE_PROFILE.md).
- **Tensor parallelism:** type-system plumbing only; no engine implements it yet. See [docs/TENSOR_PARALLELISM.md](docs/TENSOR_PARALLELISM.md).

## Deploying

Cascadia does not daemonize itself, so it needs to be run under systemd / NSSM / launchd. See [`docs/deploy/`](docs/deploy/) for a systemd unit template and Windows / macOS recipes. Cascadia handles `SIGTERM` cleanly.

**Security**: the HTTP API and inter-stage TCP relay are plaintext and unauthenticated. Bind only to trusted networks (LAN, loopback) or terminate TLS + auth at a reverse proxy in front of `--api`. See [SECURITY.md](SECURITY.md) for the threat model and built-in hardening.

## Troubleshooting

**`config.json not in <model dir>`**: `ov-runtime` reads the HF model `config.json` from the shard's tokenizer dir to derive rotary parameters. Older shard exports may not bundle `config.json`; copy it from the source model's HF cache (`~/.cache/huggingface/hub/models--<repo>/snapshots/<sha>/config.json`) into the shards root. Shards produced by `cascadia shard` bundle it automatically.

**`could not connect to downstream peer within timeout`** (engines wait 60 s): start the downstream worker first; check `--listen` on the downstream matches `--next` on the upstream and that the host's firewall allows the port.

**Worker dies silently when SSH session closes**: on Windows OpenSSH the child process is tied to the SSH parent. Run workers under systemd / NSSM / Task Scheduler in production.

`cascadia doctor` diagnoses most other environment/hardware issues.

## Documentation

| Doc | What's in it |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | 5-minute stub run → real inference |
| [INSTALL.md](INSTALL.md) | Full setup: OpenVINO SDK, GPU runtime, Docker |
| [docs/CLI.md](docs/CLI.md) | Every command and flag |
| [docs/SHARDING.md](docs/SHARDING.md) | Sharding flow + per-model-family support table |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Design decisions + crate responsibilities |
| [docs/engines/](docs/engines/) | Per-engine deep dives |
| [docs/architectures/](docs/architectures/) | Per-model-family export/support notes |
| [docs/perf/](docs/perf/) | Performance investigations and tuning |
| [SECURITY.md](SECURITY.md) | Threat model + vulnerability reporting |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the build/test gate, crate layout, and commit conventions.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache-2.0 (see [LICENSE](LICENSE)). Third-party attributions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
