# Tahoma

[![ci](https://github.com/labscommunity/tahoma/actions/workflows/ci.yml/badge.svg)](https://github.com/labscommunity/tahoma/actions/workflows/ci.yml)

> Run any model on Intel hardware.

Tahoma distributes LLM inference across Intel laptops, desktops, and AI PCs. Shard a model across the machines you already have and serve it through an OpenAI-compatible API — no cloud, no NVIDIA GPUs required.

## Status

**Pre-alpha.** Working on Intel AI PCs (Lunar Lake / Arrow Lake / Panther Lake). The runtime ships five engines today; see `tahoma engines`. Intel Arc discrete GPUs and Xeon CPU-only nodes are on the roadmap.

## Why Tahoma

Frontier models don't fit on a single laptop. Cloud APIs are expensive, opaque, and require sending your data offsite. Tahoma lets you point a few Intel machines at each other and run models that none of them could handle alone.

## Install

```bash
pip install -e .             # core (PyTorch path)
pip install -e '.[ov]'       # adds OpenVINO + optimum-intel for the OV engines
```

Requires Python 3.11+. The OpenVINO engines run on Intel iGPU/NPU/CPU; PyTorch paths work on any platform.

## Quick start

### Single machine, one shot

```bash
# Local OpenAI-style int4 export auto-pulled on first run.
tahoma worker --rank 0 --total 1 --engine ov-optimum --device GPU \
              --model unsloth/Meta-Llama-3.1-8B-Instruct \
              --api :8000

# In another terminal:
curl http://localhost:8000/v1/chat/completions -d '{
  "model": "llama-3.1-8b",
  "messages": [{"role": "user", "content": "Capital of France?"}]
}'
```

### Two machines (pipeline parallel)

Pre-export per-stage shards once with `optimum-cli` or rainier's exporter, then:

```bash
# On node B (last stage, listens for activations):
tahoma worker --rank 1 --total 2 --engine ov-runtime --device GPU \
              --model /path/to/shards_2stage_v5_beam \
              --listen 10.0.0.2:9100

# On node A (first stage, serves the API):
tahoma worker --rank 0 --total 2 --engine ov-runtime --device GPU \
              --model /path/to/shards_2stage_v5_beam \
              --next 10.0.0.2:9100 --api :8000
```

Add `--engine ov-dist-spec --draft-model unsloth/Llama-3.2-1B-Instruct --spec-k 4` on rank 0 for distributed speculative decoding (see [docs/engines/ov-dist-spec.md](docs/engines/ov-dist-spec.md)).

### List engines

```
$ tahoma engines
  ov-dist-spec    multi-stage OV spec decode with mask-based rewind; v5 shards
  ov-optimum      single-stage OV via optimum-intel; auto-export
  ov-runtime      multi-stage OV with stateful KV cache; pre-exported pipeline dir
  ov-spec         single-stage OV spec decode; requires --draft-model
  pytorch         distributed PyTorch (default)
```

## Architecture

See [`CLAUDE.md`](CLAUDE.md) for design rationale. Key seams:

- `tahoma/api/` — OpenAI-compatible HTTP
- `tahoma/worker/` — runner + Engine plugin (one per engine in `worker/engines/`)
- `tahoma/worker/engines/registry.py` — register a new engine in three lines
- `tahoma/shared/` — topology graph (per-link latency + bandwidth) and message types

## Per-engine guides

- [ov-runtime](docs/engines/ov-runtime.md) — multi-stage stateful KV cache
- [ov-spec](docs/engines/ov-spec.md) — single-stage speculative decoding
- [ov-dist-spec](docs/engines/ov-dist-spec.md) — multi-stage spec with mask-based rewind

## Cluster

- Auto-discovery: `tahoma worker --discover ...` (requires `pip install tahoma[discovery]`) advertises this node via mDNS on `_tahoma._tcp.local.` and browses for siblings in the same `TAHOMA_NAMESPACE`. Discovered peers feed `/state` and `/instance/previews` for placement suggestions.
- Master election: lowest-lexicographic node id in a namespace wins; no explicit messaging.
- Placement: `POST /instance/previews` returns the top-N pipeline orderings ranked by sum-of-edge-latency (using the per-link measurements in the topology graph — exo's graph stores socket vs RDMA *types* but no measurements).
- Tensor parallelism: foundation only — see [docs/architecture/tensor-parallelism.md](docs/architecture/tensor-parallelism.md). Engines opt in by calling `TPGroup.all_reduce_sum_inplace` after attention and MLP outputs; today's v5 shards aren't TP-split, so no built-in engine ships TP yet.

## Deploying

Tahoma does not daemonize itself — run it under systemd / NSSM / launchd. See [`docs/deploy/`](docs/deploy/) for a systemd unit template and Windows / macOS recipes. Tahoma handles `SIGTERM` cleanly (`runner.close()` then exit 0) and supports `--pid-file` for supervisor integration.

## Troubleshooting

**`Tokenizer class TokenizersBackend does not exist`** — Some shard exports bundle a tokenizer config that current `transformers` can't import. Tahoma falls back to the model id's HF cache; ensure you've run `hf download <model-id> --include tokenizer.json tokenizer_config.json special_tokens_map.json` on each node.

**`could not connect to … within 30s`** — Start the downstream worker first; the upstream waits for the upstream socket to bind. Check `--listen` on the downstream matches `--next` on the upstream and that the host's firewall allows the port.

**Worker dies silently when SSH session closes** — On Windows OpenSSH the python child runs in Session 0 and is tied to the SSH parent. Run workers under systemd / nssm / Task Scheduler in production; for ad-hoc testing, keep the SSH session attached or use `nohup` / `screen`.

## License

Apache-2.0 (target). The repository is private during incubation.
