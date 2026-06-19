# Roadmap

## MVP (alpha)

The minimum that demonstrates the killer story: serve a model that does not fit on one Intel laptop across two of them, with reasonable tok/s.

- `worker/engines/openvino/` — port rainier's per-stage INT4 OV IR pipeline
- `worker/runner/` — stage worker that loads its shard and serves activations
- `routing/` + activation relay — TCP point-to-point (lift from rainier)
- `master/placement.py` — simple: even split across discovered nodes
- `discovery/` — mDNS / libp2p, zero-config peer find
- `api/` — OpenAI `/v1/chat/completions` (non-streaming first, then streaming)
- CLI — `cascadia run <model>` and `cascadia serve`

**Acceptance test:** Two Intel laptops on the same LAN. One command per machine. `curl localhost:8000/v1/chat/completions` and get a response from a model that exceeds one machine's RAM.

## Beta

- Topology with measured per-link latency / bandwidth
- Topology-aware placement (heuristics from rainier's production data)
- Multi-stream micro-batching (1.38× throughput, ported from rainier)
- Speculative decoding with mask-based KV rewind
- Streaming responses
- Model registry with on-demand HF pull
- Web UI (single-page, vanilla JS — match rainier's demo style)

## v1

- 3+ node pipelines
- Spec decode + micro-batch composed
- Intel Arc discrete GPU support
- Sparse / FP16 activation compression for WAN deployments
- Documentation, tutorials, contribution guide
- Public release
