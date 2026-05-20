# Architecture

Cascadia is a Rust workspace under `rust/`. Each crate has a single responsibility and a stable interface; engines and discovery backends are swappable.

## `cascadia-api`

OpenAI-compatible HTTP server (axum). Routes: `/health`, `/v1/models`, `/v1/chat/completions` (non-streaming + SSE streaming), `/v1/cancel/<task_id>`. Backpressure via a concurrent-request semaphore (default 16); request body cap (default 64 KiB) and prompt cap (default 32 KiB) enforce 413 / 503 responses on oversized or over-capacity input.

## `cascadia-runner`

Per-stage `Runner`. Connects upstream + downstream transports, loads weights, builds the engine, warms it up, and exposes `submit` / `generate` / `cancel`. Concurrent-safe — multiple `generate()` callers share one engine through a `Mutex`; chunks for other tasks emitted during one caller's `step()` are buffered for their owners.

## `cascadia-engine`

Two trait definitions — the plugin seam:

- `Engine`: `warmup`, `submit`, `step`, `cancel`, `close`. `submit` returns `EngineError::QueueFull` when the per-engine pending cap is reached.
- `Builder`: `configure_listen`, `connect`, `load`, `build`, `close`.

## `cascadia-engine-openvino`

Three engines:

- `ov-genai` — single-stage `openvino_genai.LLMPipeline` via the C++ FFI shim. FastDraft + Prompt Lookup variants.
- `ov-runtime` — multi-stage stateful KV cache. Pre-exported per-stage v3+ shards; each stage owns its layer range and runs SDPA attention with internal RoPE.
- `ov-dist-spec` — multi-stage spec decode with mask-based KV-cache rewind on rejected drafts. v5 shards (canonical optimum-style inputs).

## `cascadia-engine-mock`

Deterministic word-echo engine — splits the prompt and emits one word per `step()`. Used by API / runner / CLI tests.

## `cascadia-ov-genai-shim`

C++ FFI shim wrapping `openvino-genai`. `extern "C"` only; every entry point catches `...` so a C++ exception cannot unwind into Rust UB. Stub mode (no link) is the default for dev / CI; `--features openvino` links against the real OV GenAI 2026.1+ SDK.

## `cascadia-transport`

TCP activation relay between pipeline stages. Wire format is byte-identical to rainier's reference Python relay: 20-byte header (`payload_len`, `dtype`, `dim0`, `dim1`, `dim2`) then row-major payload. dtype codes: `0=f32, 1=f16, 2=i8, 3=i32, 4=i64`. Caps incoming payloads at 256 MiB and applies a 60 s read timeout per recv.

## `cascadia-topology`

Topology graph with per-link latency and bandwidth measurements. This is where Cascadia diverges from exo, whose topology only tracks edge type (Socket vs RDMA). Empirically, latency is the dominant placement signal on Intel fleets — a 50 ms WAN hop drops throughput 65%.

## `cascadia-discovery`

mDNS peer discovery via the `mdns-sd` crate. Advertises `_cascadia._tcp.local.` and browses for siblings in the same `CASCADIA_NAMESPACE`. Zero-config: spin up a worker and the master finds it.

## `cascadia-download`

Model registry plus on-demand HuggingFace pull. Registry lives at `~/.cache/cascadia/registry.json`; writes are atomic (`.tmp` + `fsync` + rename). Symlinks at the registry path are rejected to prevent path-substitution attacks.

## `cascadia-cli` + `cascadia`

`cascadia worker --rank N --total M --engine <name> --model <path|hf_id> ...` is the only subcommand that does work. `cascadia engines` lists registered engines. The `cascadia` crate is the binary entry point and depends on `cascadia-cli`.
