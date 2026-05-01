# Architecture

Tahoma's seven modules mirror exo's seams. Each has a single responsibility and a stable interface; engines and discovery backends are swappable.

## `tahoma/api/`

OpenAI-compatible HTTP server. `/v1/chat/completions`, `/v1/completions`, `/v1/models`. Non-streaming first; streaming after MVP. Talks to `master/` to schedule generation tasks.

## `tahoma/master/`

The control plane. Decides which nodes run which shards of which model (placement), tracks instance lifecycle (loading, ready, generating), and runs leader election so any node can become master if the current master goes down.

Placement scoring: `(download_score, available_RAM, link_latency)`. `download_score` favors nodes that already have weights cached (avoid re-pull). `link_latency` is measured, not assumed — see `shared/topology.py`.

## `tahoma/worker/`

The execution side. A `Runner` supervises an `Engine` plugin. The runner handles the lifecycle (connect to peers, load shard, warmup, generate); the engine handles the per-token math.

`worker/engines/base.py` defines two ABCs:

- `Engine`: `warmup`, `submit`, `step`, `close`, `serve_prefill`
- `Builder`: `connect`, `load`, `build`, `close`

`worker/engines/openvino/` is the only engine in the MVP. It ports rainier's per-stage INT4 OV IR pipeline.

## `tahoma/routing/`

Internal pub/sub message bus. Topic-based; used for control messages between master and workers.

## `tahoma/shared/`

Common types, the `Topology` graph, leader election protocol, logging.

The topology graph stores per-link latency and bandwidth — measured, not assumed. This is where Tahoma diverges from exo, whose topology only tracks edge type (Socket vs RDMA).

## `tahoma/discovery/`

Peer discovery. libp2p or mDNS-based. Zero-config: spin up a worker and the master finds it automatically. No tokens, no IPs to configure (in OSS — the productized version layers on top with auth).

## `tahoma/download/`

Model registry plus on-demand HuggingFace pull. Each node only pulls the weight ranges it owns.
