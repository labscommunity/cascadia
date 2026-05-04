# tahoma — current status

Snapshot of what works today. Updated whenever something material
ships on `main`.

## What it is

Distributed LLM inference for Intel hardware. Single Rust binary per
node. Sharding + serving + tokenizer + HTTP API are all in the same
~7 MB executable. Pipeline parallelism over TCP between nodes. The
OSS / hobbyist counterpart to cascadia (the enterprise track —
multi-tenant, auth, fault tolerance, separate repo).

Closest comparable in the ecosystem: [exo](https://github.com/exo-explore/exo),
but for Intel devices (Lunar Lake / Arrow Lake / Panther Lake / Battlemage
Arc) using OpenVINO instead of MLX.

## Engines

All four engines build clean, run on real hardware (Intel Arc B390 +
Lunar Lake 140V iGPU connected over Thunderbolt 4), and are exercised
end-to-end in CI plus on-hardware benches.

| Engine | Stages | Spec decode | Shards required | Status |
|--------|-------:|-------------|-----------------|--------|
| `mock` | 1 | – | none | reference / test fixture |
| `ov-genai` | 1 only | optional (FastDraft / Prompt Lookup) | off-the-shelf HF or `tahoma shard` | production-ready |
| `ov-runtime` | N | – | v3 OR v5 (`tahoma shard` produces v5) | production-ready |
| `ov-dist-spec` | N | required (chain spec) | v5 (`tahoma shard`) | production-ready |

`ov-runtime` auto-detects v3 vs v5 IR layouts and dispatches to the
right input-binding path (added in PR #6 so user-sharded models work
with both `ov-runtime` and `ov-dist-spec`).

## What you can do today

### One-command sharding (PR #6, May 2026)

```bash
pip install torch transformers openvino safetensors huggingface_hub nncf
tahoma shard --model unsloth/Meta-Llama-3.1-8B-Instruct \
             --output-dir ~/tahoma/llama-8b-2stage \
             --num-stages 2 --quantization int4
```

Drops the previous dependency on rainier (the sister Python repo). The
exporter is bundled into the tahoma binary via `include_str!` and
written to a temp file at runtime — no extra files to deploy.

Architectures explicitly tested: Llama (1, 2, 3, 3.1, 3.2), Mistral 7B+,
Qwen2 / 2.5. Phi-3 and Gemma 2 work best-effort. Mixtral and other MoE
models are not yet supported. Full table + tuning guide in
[docs/SHARDING.md](SHARDING.md).

### Distributed serving

```bash
# Node B (last stage):
tahoma worker --rank 1 --total 2 --engine ov-dist-spec --device GPU \
              --model ~/tahoma/llama-8b-2stage --listen :9100

# Node A (first stage + API):
tahoma worker --rank 0 --total 2 --engine ov-dist-spec --device GPU \
              --model ~/tahoma/llama-8b-2stage \
              --draft-model unsloth/Llama-3.2-1B-Instruct --spec-k 5 \
              --next 10.0.0.2:9100 --api :8000
```

OpenAI-compatible `/v1/chat/completions` endpoint with SSE streaming.

### Single-node serving (no sharding)

```bash
tahoma worker --rank 0 --total 1 --engine ov-genai --device GPU \
              --model unsloth/Meta-Llama-3.1-8B-Instruct \
              --draft-model unsloth/Llama-3.2-1B-Instruct --spec-k 5 \
              --api :8000
```

## Performance (on-hardware bench, May 2026)

Llama 3.1 8B Instruct INT4 on alpha (Battlemage Arc B390 dGPU, 12 GB) +
charlie (Lunar Lake 140V iGPU) over Thunderbolt 4. Llama 3.2 1B INT4
draft model. Numbers are tok/s, single trial each unless noted.

| Engine | Workload | Tokens | tok/s |
|--------|----------|-------:|------:|
| `ov-genai` (single-node, no draft) | factual | 605 (EOS) | 20.83 |
| `ov-genai` (single-node, FastDraft 150M K=5) | factual | 605 (EOS) | **28.04** |
| `ov-genai` (single-node, 1B INT4 draft K=5) | factual | 605 (EOS) | 26.93 |
| `ov-dist-spec` 2-stage (1B INT4 draft, K=5) | factual short | 256 | 18.90 |
| `ov-dist-spec` 2-stage (1B INT4 draft, K=5) | factual long | 1024 | 23.85 |
| `ov-dist-spec` 2-stage (1B INT4 draft, K=5) | **factual long-form** | **4096** | **29.62** |
| `ov-dist-spec` 2-stage (1B INT4 draft, K=1) | factual short | 256 | 18.98 (+19% vs main pre-PR-#5) |

PR #5 added async overlap of the target wire round-trip with the draft
compute step (`feed_send_async` / `feed_recv_async` + a speculative
`draft.feed` during the network wait). Benefit ranges from +6% (K=5
short prompts) to +19% (K=1) over the prior sequential implementation.

### When distributed beats single-node

Distributed `ov-dist-spec` (29.62 tok/s) beats the best single-node
config (`ov-genai` + FastDraft 150M, 28.04 tok/s) only on **long-form
generation** (≥ 1000 tokens). Below that, single-node wins because the
per-round network coordination overhead doesn't amortize. See
[experiments on the autolab branch](https://github.com/labscommunity/tahoma/tree/autolab/distributed-perf)
for the full investigation.

## Recent changes

* **PR #6 (`feat/standalone-sharding`, May 3, 2026)** — adds `tahoma
  shard` subcommand, drops the rainier dependency. Generalized
  exporter for Llama-family architectures. `ov-runtime` engine now
  handles both v3 and v5 IR input layouts. New `docs/SHARDING.md`.
* **PR #5 (`perf/dist-spec-async-overlap`, May 3, 2026)** — async
  overlap of target wire with draft compute in `ov-dist-spec`. +9%
  on long-form factual K=5, +19% on K=1. Per-task timing instrumentation
  (`target_alpha_setup_ms`, `target_alpha_infer_ms`,
  `target_alpha_output_ms`, `target_wire_ms`) added to the
  `spec_decode timing` log line.
* **PR #3 (`feat/rust-port`, May 2, 2026)** — Rust port of the
  Python prototype landed and the Python tree was removed. Phase 14
  hardening (security model below).

## Tests

`cargo test --workspace` on macOS — **49 passing, 0 failures.**

Per-crate breakdown:

| Crate | Tests |
|---|---:|
| `tahoma-types` | 13 |
| `tahoma-topology` | 4 |
| `tahoma-transport` | 5 |
| `tahoma-engine-mock` | 4 |
| `tahoma-ov-genai-shim` | 3 |
| `tahoma-engine-openvino` | 5 |
| `tahoma-runner` | 3 |
| `tahoma-api` | 3 |
| `tahoma-discovery` | 2 |
| `tahoma-download` | 3 |
| `tahoma-tests-e2e` | 2 |

The e2e crate spawns the real `tahoma` binary, polls `/health`, and
issues concurrent fan-out requests against the mock engine.

## Build

Two profiles:

```bash
# Stub mode (no OV link). Engines that need OV return a clean runtime error.
cargo build --release -p tahoma

# Real OV (Intel hardware). Requires the OpenVINO GenAI 2026.1 SDK download.
INTEL_OPENVINO_DIR=/path/to/openvino_genai_<platform>_2026.1.0.0_x86_64 \
  cargo build --release -p tahoma --features openvino
```

Binary at `target/release/tahoma` (`tahoma.exe` on Windows). Static
apart from the OV dynamic libraries; copy the binary plus
`INTEL_OPENVINO_DIR/runtime/bin/intel64/Release/` and
`runtime/3rdparty/tbb/bin/` to the worker host's PATH.

## Security model

Tahoma's network surface is designed for **trusted LAN deployment** —
think a closet or rack of Intel AI PCs on an isolated subnet, not the
public internet. The Phase 14 hardening (PR #3) closed the worst
failure modes (panics, OOM, crash on malformed input) but does NOT
add authentication or transport encryption.

**What you get out of the box:**
* HTTP API: 64 KiB request body cap, 32 KiB prompt cap, 16 concurrent
  request semaphore. Oversized prompt → 413; over-capacity → 503.
  Engine errors map cleanly to 5xx (no panics).
* Engine queue: 256-task pending cap; `EngineError::QueueFull` → 503.
* TCP relay: 256 MiB cap on tensor payloads, 64 KiB cap on raw control
  recvs, shape × dtype overflow check before alloc, 60 s read timeout
  on every recv. Wedged or hostile peer can't pin a worker thread or
  trigger a multi-GB allocation.
* C++ shim: null pointer guards on every exported function, bounded
  property dicts (256 pairs max), uniform `catch (...)` so C++
  exceptions can't unwind into Rust UB, tensor-shape overflow check.
* Numerics: NaN-aware `argmax` (warns instead of silently returning
  token 0 on a broken forward pass). Rotary `compute()` clamps `start`
  to 16 M positions and `seq_len` to 1 M tokens.
* Registry (`tahoma-download`): atomic write (tmp + fsync + rename),
  reject symlink at registry path, 16 MiB cap, parse errors are hard
  failures.

**What you do NOT get:**
* No TLS on either the HTTP API or the inter-stage TCP relay.
* No client authentication on the HTTP API.
* No mDNS authentication.
* No supply-chain pinning beyond `Cargo.lock`.

For untrusted-network exposure: terminate TLS + auth at a reverse
proxy in front of the `--api` port, firewall the inter-stage TCP ports
(`--listen` / `--next`) so only sibling workers can reach them. The
threat model and defense-in-depth limits are summarized in the
binary's `--help` long_about.

Cascadia (the enterprise repo) is where multi-tenancy, auth, fault
tolerance, and the rest of the production-grade story live. Tahoma's
job is to be the hobbyist-friendly OSS substrate.

## Known limitations / not yet done

In rough order of "how often this comes up":

1. **mDNS auto-discovery is built but not wired** into the worker CLI.
   `tahoma-discovery` advertises `_tahoma._tcp.local.` and populates
   `tahoma-topology` (latency + bandwidth per edge), but `tahoma
   worker` still requires explicit `--listen` / `--next host:port`. A
   follow-up PR will let workers auto-resolve peers by node id.
2. **No automatic placement** across discovered nodes. The operator
   picks which rank goes where.
3. **Pure-Rust exporter** would drop the Python pip-install
   requirement for `tahoma shard`. Significant effort; deferred until
   the supported-architecture set settles.
4. **Tensor parallelism** (`tp_size > 1`) is in the type system but no
   engine implements it — pure pipeline parallelism only.
5. **No multi-tenant batching across requests.** Engines process one
   task at a time. Concurrency happens at the API admission layer
   (16-request semaphore) but not inside the engine. Acceptable for
   the single-user OSS target; cascadia will own the batching story.
6. **Mixtral / MoE export** is not supported by the bundled exporter
   (the script assumes one MLP per layer). Llama / Mistral / Qwen2 /
   Phi-3 / Gemma 2 cover most current OSS deployments.
7. **NNCF can hang on tied-embedding INT4** (Llama 3.2 1B/3B). Workaround:
   `--quantization fp16` (documented in SHARDING.md). Llama 3.1 8B and
   most other production targets don't tie embeddings, so INT4 works
   normally there.
8. **Performance gap to optimum-intel ov-genai** on identical workloads
   is ~5% in `--release` builds (single-node, single-stage). Acceptable
   for the OSS target.

## Architecture position

Per the design constraint (cascadia may depend on tahoma; tahoma may
not depend on cascadia):

* No cascadia imports anywhere in the workspace.
* Stable public APIs on the most-reusable crates (`tahoma-types`,
  `tahoma-topology`, `tahoma-transport`, `tahoma-engine`).
* The OpenVINO C++ FFI shim defaults to **stub mode** (no link, runtime
  errors only) so dev iteration on macOS / CI Linux without OpenVINO
  installed stays fast. Real link is gated behind `--features openvino`.
* Wire format for activation transport is **byte-identical to the
  removed Python prototype's `tahoma/worker/transport.py`** — Python
  ranks could in principle still interop. dtype codes: 0=f32, 1=f16,
  2=i8, 3=i32, 4=i64.

## Where to look next

* [README.md](../README.md) — quickstart and links
* [docs/SHARDING.md](SHARDING.md) — the full sharding flow + arch table
* [docs/engines/ov-dist-spec.md](engines/ov-dist-spec.md) — the
  speculative-decode engine in depth (if present)
* [docs/ARCHITECTURE.md](ARCHITECTURE.md) — workspace layout + crate
  responsibilities
* [docs/ROADMAP.md](ROADMAP.md) — what's coming
