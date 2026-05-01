# Tahoma

Distributed LLM inference for Intel hardware. Think exo (github.com/exo-explore/exo), but Intel-native.

## Mission

Run any model on Intel hardware. Shard models across Intel AI PCs using pipeline parallelism, serve them through an OpenAI-compatible API.

**Killer demo:** Take a model that does not fit on one Intel laptop and run it across two or three of them with usable tok/s.

## Hardware scope

- **Initial target:** Intel AI PCs — Lunar Lake, Arrow Lake, Panther Lake. iGPU + NPU + CPU.
- **Later:** Intel Arc A/B-series discrete GPUs, Xeon CPU-only servers, generic x86 with no AI accelerator.
- **Out of scope (for now):** Apple Silicon, NVIDIA, AMD.

## Architecture

Mirrors exo's module seams. Seven concerns:

- `tahoma/api/` — OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)
- `tahoma/master/` — control plane: placement, instance lifecycle, leader election
- `tahoma/worker/` — runner + Engine plugin, executes the assigned shard
- `tahoma/routing/` — internal pub/sub message bus
- `tahoma/shared/` — topology graph, types, utilities
- `tahoma/discovery/` — peer discovery (libp2p / mDNS, zero-config)
- `tahoma/download/` — model registry, on-demand HuggingFace pull

## Design decisions (locked 2026-05-01)

- **Engine plurality: OpenVINO-first, pluggable.** Define `Engine` and `Builder` ABCs from day one (modeled on exo). Ship only `OpenVINOEngine` initially. `IPEXEngine` and others are future contributions.
- **Discovery: libp2p-style, zero-config peer-to-peer.** No central control plane in OSS — that's the productization angle (handled in the cascadia-fleet track, not here).
- **Topology stores measured latency + bandwidth.** Exo's topology graph tracks edges as `SocketConnection` / `RDMAConnection` but does not store latency or bandwidth. Empirically (1,200+ rainier experiments), latency drives placement on Intel fleets — a 50 ms WAN hop drops throughput 65%. Tahoma's topology stores per-link measurements.
- **License:** Apache-2.0 target; repo is private during incubation.
- **No co-authors on commits.** See "Commit conventions" below — this is non-negotiable for this project.

## Source of truth for inference internals

We do not reinvent the OpenVINO export and pipeline-parallel work. Rainier (`/Users/tatef/Workspaces/rainier`) has the production-tested implementations:

- Selective safetensors loader (`cascadia/model/loader.py`)
- Per-stage INT4 OV IR export via `torch.jit.trace + nncf` (`scripts/export_shards_dynamo.py`)
- Stateful per-stage shards with KV cache
- Multi-stream micro-batching for 1.38× free throughput
- Speculative decode with mask-based KV-cache rewind
- Activation relay over TCP

Documented in rainier's `DISCOVERIES.md` and `docs/PRODUCTION_LEARNINGS.md`. When wiring up `tahoma/worker/engines/openvino/`, port these in — do not rewrite from scratch.

## Commit conventions

- Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`.
- **Never include `Co-Authored-By` lines in commit messages. Ever.** This applies to every commit on this repo, including AI-assisted ones. Do not add Claude, do not add any other tool, do not add any other author. Single-author commits only.
- Do not skip hooks (`--no-verify`).
- One logical change per commit. Smaller is better.

## Code style

- Python 3.11+
- Type hints throughout
- Dataclasses for config
- No runtime imports of rainier — copy what's needed and adapt to this repo's types.
