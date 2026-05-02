# d1: Thunderbolt 4 link characterization (alpha ↔ charlie)

## Setup
- alpha 10.10.10.1 (Battlemage host, "Ethernet 4" = TB4)
- charlie 10.10.10.2 (Lunar Lake, "Ethernet 4" = TB4)
- Direct TB4 link, no switch
- TCP socket python bench (no iperf3 on either Windows host)

## Results

| Metric | Value |
|--------|------:|
| TCP throughput (1 GB transfer) | **1094 MB/s = 8.75 Gbps** |
| Round-trip latency (1000 ping-pongs of 8 bytes) | **0.142 ms** (one-way ≈ 71 µs) |

## Implication for distributed LLM inference

For an 8B-class model with `hidden_dim=4096`, each per-token activation
transferred between pipeline stages is:
  4096 floats × 2 bytes (FP16) = **8 KB**

Per-token activation transfer latency:
  - serialization at 8.75 Gbps: ~7 µs
  - one-way latency: ~71 µs
  - **total per-token transfer: ~78 µs**

For a target of 30 tok/s (~33 ms per token), the network accounts for
**< 0.3% of total per-token time**. Network is NOT the bottleneck for
single-prompt distributed inference at this hidden-dim.

## Theoretical pipeline-parallel ceiling

If compute time is split evenly across 2 stages: per-token = max(stage_0, stage_1) + transfer.

Single-node 8B INT4 = ~33 ms / forward pass (30 tok/s after spec etc.).
Pipeline-parallel: each stage does ~half the layers = ~16.5 ms per stage + 78 µs network.
**Theoretical PP ceiling = ~60 tok/s on alpha+charlie at 8B INT4.**

Caveat: this ignores per-stage overhead (KV cache management, position id
updates, OV plugin compile differences, etc).

## Observed (Phase 1 single-machine bench)

ov-dist-spec K=4 v5 shards: 17.59 tok/s (claimed in c5 — likely cap-inflated).

If real distributed rate is ~3.5x lower than the PP ceiling, the bottleneck
is in the OV runtime / pre-exported shards, not the network. d2 will
investigate.

## Tools left for next campaigns

- `experiments/d1-network/net_server.py` and `net_client.py` for re-running
  bandwidth/latency on demand.
- `iperf3` would be nicer; install if needed (didn't want to add deps in
  this campaign).
