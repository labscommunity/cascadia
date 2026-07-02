# Architecture

Cascadia is a Cargo workspace at the repo root. Each crate has a single responsibility and a stable interface; engines and discovery backends are swappable.

## Design decisions

- **Engine plurality: OpenVINO-first, pluggable.** The `Engine` + `Builder` traits live in `cascadia-engine`; seven engines ship behind them (`mock`, `ov-genai`, `ov-runtime`, `ov-dist-spec`, `gemma4`, `sparse-moe`, `qwen36-moe`). Future engines (IPEX, OneAPI direct) plug behind the same trait.
- **Discovery: zero-config peer-to-peer.** Workers find each other over mDNS; no central control plane.
- **Topology stores measured latency + bandwidth.** Latency is the dominant placement signal on Intel fleets — a 50 ms WAN hop drops throughput 65% — so Cascadia's topology graph stores per-link measurements, not just edge types.
- **Rust-only workers.** One static binary per node; no runtime Python dependency, no pip install on workers. Python is only needed at export time (`cascadia shard`).

## `cascadia-api`

OpenAI-compatible HTTP server (axum). Routes: `/health`, `/v1/models`, `/v1/chat/completions` (non-streaming + SSE streaming), `/v1/cancel/<task_id>`. Backpressure via a concurrent-request semaphore (default 16); request body cap (default 64 KiB) and prompt cap (default 32 KiB) enforce 413 / 503 responses on oversized or over-capacity input.

## `cascadia-runner`

Per-stage `Runner`. Connects upstream + downstream transports, loads weights, builds the engine, warms it up, and exposes `submit` / `generate` / `cancel`. Concurrent-safe — multiple `generate()` callers share one engine through a `Mutex`; chunks for other tasks emitted during one caller's `step()` are buffered for their owners.

## `cascadia-engine`

Two trait definitions — the plugin seam:

- `Engine`: `warmup`, `submit`, `step`, `cancel`, `close`. `submit` returns `EngineError::QueueFull` when the per-engine pending cap is reached.
- `Builder`: `configure_listen`, `connect`, `load`, `build`, `close`.

## `cascadia-engine-openvino`

Five engines:

- `ov-genai` — single-stage `openvino_genai.LLMPipeline` via the C++ FFI shim. FastDraft + Prompt Lookup variants.
- `ov-runtime` — multi-stage stateful KV cache. Pre-exported per-stage v3+ shards; each stage owns its layer range and runs SDPA attention with internal RoPE.
- `ov-dist-spec` — multi-stage spec decode with mask-based KV-cache rewind on rejected drafts. v5 shards (canonical optimum-style inputs).
- `gemma4` — Gemma 4 multi-stage: per-layer-type attention, KV-sharing, per-layer-input embeddings. `gemma4_cached_v1` shards.
- `qwen36-moe` — Qwen3.6-35B-A3B staged chain (GatedDeltaNet + MoE) from `qwen3_5_moe` IR-surgery shards; single-box or N-rank pipeline. See [architectures/qwen36-moe-support.md](architectures/qwen36-moe-support.md).

## `cascadia-engine-mock`

Deterministic word-echo engine — splits the prompt and emits one word per `step()`. Used by API / runner / CLI tests.

## `cascadia-engine-sparse-moe`

CPU-targeted sparse mixture-of-experts engine (Kimi K2.6-style models, MiniMax-M2). Runs attention/norm shells natively in Rust (default; OV IR shells are an optional backend) and dispatches only the top-k experts the router selects each step. Experts execute as per-(layer, expert) OV IRs by default, or through the `cascadia-int4-gemm` AVX-512 kernels against packed int4 weight binaries (`int4_bin` backend).

## `cascadia-int4-gemm`

Hand-rolled AVX-512 INT4 GEMM kernels for the sparse-MoE expert path — group-32 symmetric quantization with bf16 scales, matching the compressed-tensors on-disk format.

## `cascadia-dashboard`

Dashboard HTTP routes (`/api/topology`, `/api/stats`, `/api/events` SSE) and an embedded Vite SPA (behind the `embed-spa` feature) for visualizing a cluster. Kept separate from `cascadia-api` so the OpenAI surface doesn't grow a topology dependency or bundled static assets.

## `cascadia-ov-genai-shim`

C++ FFI shim wrapping `openvino-genai`. `extern "C"` only; every entry point catches `...` so a C++ exception cannot unwind into Rust UB. Stub mode (no link) is the default for dev / CI; `--features openvino` links against the real OV GenAI 2026.2+ SDK.

## `cascadia-types`

Zero-dependency wire/value types shared by every crate: generation tasks and chunks, shard descriptions, peer layout. Keeping them dependency-free lets engines and transports evolve without version-lockstep.

## `cascadia-transport`

TCP activation relay between pipeline stages. Wire format: 20-byte header (`payload_len`, `dtype`, `dim0`, `dim1`, `dim2`) then row-major payload. dtype codes: `0=f32, 1=f16, 2=i8, 3=i32, 4=i64`. Caps incoming payloads at 256 MiB and applies a 60 s read timeout per recv.

## `cascadia-topology`

Topology graph with per-link latency and bandwidth measurements. This is where Cascadia diverges from exo, whose topology only tracks edge type (Socket vs RDMA). Empirically, latency is the dominant placement signal on Intel fleets — a 50 ms WAN hop drops throughput 65%.

## `cascadia-discovery`

mDNS peer discovery via the `mdns-sd` crate. Advertises `_cascadia._tcp.local.` and browses for siblings in the same namespace (a TXT-record field; peers in other namespaces are ignored). Zero-config: spin up workers on the same LAN and they find each other.

## `cascadia-download`

Model registry plus on-demand HuggingFace pull. Registry lives at `~/.cache/cascadia/registry.json`; writes are atomic (`.tmp` + `fsync` + rename). Symlinks at the registry path are rejected to prevent path-substitution attacks.

## `cascadia-cli` + `cascadia`

`cascadia worker --rank N --total M --engine <name> --model <path|hf_id> ...` is the core serving subcommand; `run` is its single-machine sugar. Other subcommands: `shard` (bundled exporter), `doctor` (environment checks), `discover` (mDNS browse), `engines`, `profile-devices` / `profile-stages` / `place` / `run-placement` (placement tooling). The `cascadia` crate is the binary entry point and depends on `cascadia-cli`.
