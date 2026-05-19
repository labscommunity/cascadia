# Cascadia

Distributed LLM inference for Intel hardware. Think exo (github.com/exo-explore/exo), but Intel-native.

## Mission

Run any model on Intel hardware. Shard models across Intel AI PCs using pipeline parallelism, serve them through an OpenAI-compatible API.

**Killer demo:** Take a model that does not fit on one Intel laptop and run it across two or three of them with usable tok/s.

## Hardware scope

- **Initial target:** Intel AI PCs — Lunar Lake, Arrow Lake, Panther Lake. iGPU + NPU + CPU.
- **Later:** Intel Arc A/B-series discrete GPUs, Xeon CPU-only servers, generic x86 with no AI accelerator.
- **Out of scope (for now):** Apple Silicon, NVIDIA, AMD.

## Architecture

Cascadia is a Cargo workspace at the repo root. The Python source tree
was removed at the end of Phase 12 (2026-05-02); the `rust/`
subdirectory was hoisted to the root in Phase 12.1. Module seams
(one concern per crate):

- `crates/cascadia-api/` — OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/models`, `/health`)
- `crates/cascadia-runner/` — Per-stage Runner: connects transports, drives the engine, streams chunks
- `crates/cascadia-engine/` — `Engine` + `Builder` traits (the plugin seam)
- `crates/cascadia-engine-openvino/` — Three OV engines (`ov-genai`, `ov-runtime`, `ov-dist-spec`)
- `crates/cascadia-engine-mock/` — Deterministic test engine
- `crates/cascadia-ov-genai-shim/` — C++ FFI shim wrapping `openvino-genai`
- `crates/cascadia-transport/` — TCP activation relay (length-prefixed tensor wire format)
- `crates/cascadia-topology/` — Topology graph with per-link latency + bandwidth
- `crates/cascadia-types/` — Zero-dep wire/value types
- `crates/cascadia-discovery/` — mDNS peer discovery (`_cascadia._tcp.local.`)
- `crates/cascadia-download/` — Model registry, HuggingFace pull
- `crates/cascadia-cli/` — `cascadia worker`, `cascadia engines`
- `crates/cascadia/` — `cascadia` binary entry point

## Design decisions (locked 2026-05-01)

- **Engine plurality: OpenVINO-first, pluggable.** `Engine` + `Builder` traits live in `cascadia-engine`; ship `mock`, `ov-genai`, `ov-runtime`, and `ov-dist-spec` from day one. Future engines (IPEX, OneAPI direct) plug behind the same trait.
- **Discovery: libp2p-style, zero-config peer-to-peer.** No central control plane in OSS — that's the productization angle (handled in the cascadia-enterprise track, not here).
- **Topology stores measured latency + bandwidth.** Exo's topology graph tracks edges as `SocketConnection` / `RDMAConnection` but does not store latency or bandwidth. Empirically (1,200+ rainier experiments), latency drives placement on Intel fleets — a 50 ms WAN hop drops throughput 65%. Cascadia's topology stores per-link measurements.
- **Rust-only.** Compiles to a single static binary per node; no runtime Python dep, no pip install on workers.
- **License:** Apache-2.0 target; repo is private during incubation.
- **No co-authors on commits.** See "Commit conventions" below — this is non-negotiable for this project.

## Source of truth for inference internals

The OpenVINO export + pipeline-parallel patterns originated in rainier
(`/Users/tatef/Workspaces/rainier`). Cascadia re-implements them in
Rust; rainier's Python is the reference for shape semantics and
attention-mask conventions:

- Per-stage INT4 OV IR export via `torch.jit.trace + nncf` (`scripts/export_cached_shards_v3.py`, `_v5.py`, `_v7.py`)
- Stateful per-stage shards with KV cache
- Speculative decode with mask-based KV-cache rewind (v5)
- Activation relay over TCP (byte-identical wire format)

Documented in rainier's `DISCOVERIES.md` and `docs/PRODUCTION_LEARNINGS.md`.

## Commit conventions

- Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`.
- **Never include `Co-Authored-By` lines in commit messages. Ever.** This applies to every commit on this repo, including AI-assisted ones. Do not add Claude, do not add any other tool, do not add any other author. Single-author commits only.
- Do not skip hooks (`--no-verify`).
- One logical change per commit. Smaller is better.

## Code style

- Rust 1.75+ (workspace edition 2021).
- `cargo fmt` + `cargo clippy --workspace` clean before commit.
- One concern per crate; `Engine` and `Builder` traits in `cascadia-engine` are the plugin seam — engines should not depend on each other.
- C++ in `cascadia-ov-genai-shim/cpp/` is `extern "C"` only (no exceptions across the boundary, every entry point catches `...`).
- No runtime cascadia-enterprise / rainier imports. Reference their algorithms; copy structure where useful; do not link.
