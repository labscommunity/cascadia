"""Cluster topology types.

Two layers:

- :class:`PeerEndpoint` + :class:`PeerLayout` describe a single pipeline
  stage's view of its immediate neighbours. This is the minimum needed for
  pipeline-parallel transport.

- :class:`Topology` is a full directed graph of nodes + measured edges. Each
  edge stores latency (ms) and bandwidth (MB/s) — exo's topology graph stores
  only socket vs RDMA *types*, but rainier's 1,200+ experiments showed
  latency drives placement on Intel fleets (a 50 ms WAN hop drops throughput
  ~65%). The placement module reads this graph to suggest pipeline splits.

The graph is intentionally simple: an in-memory dict keyed by node id, with
``add_node`` / ``add_edge`` mutators. Discovery populates it via mDNS;
benchmarking refines the edge measurements over time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PeerEndpoint:
    """Network address of a single peer."""

    host: str
    port: int


@dataclass(frozen=True)
class PeerLayout:
    """A pipeline stage's view of its neighbors.

    `upstream` sends activations to us (None on the first stage). `downstream`
    receives activations from us (None on the last stage).
    """

    upstream: PeerEndpoint | None
    downstream: PeerEndpoint | None


@dataclass
class NodeInfo:
    """One node's advertised capacity. Populated by discovery + heartbeats."""

    node_id: str
    host: str
    port: int
    namespace: str = "default"
    device: str = "CPU"           # CPU / GPU / NPU
    memory_mb: int = 0            # available device memory at advertise time
    engines: list[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)


@dataclass
class EdgeMetrics:
    """Per-link measurements; populated by occasional probes."""

    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    last_measured: float = 0.0


@dataclass
class Topology:
    """In-memory directed graph of nodes + measured edges.

    Edges are keyed (src, dst) and are *measured*, not declared — a missing
    edge means we never probed that link. Placement should treat unmeasured
    edges pessimistically (assume slow) rather than as missing.
    """

    nodes: dict[str, NodeInfo] = field(default_factory=dict)
    edges: dict[tuple[str, str], EdgeMetrics] = field(default_factory=dict)

    def add_node(self, info: NodeInfo) -> None:
        info.last_seen = time.time()
        self.nodes[info.node_id] = info

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.edges = {
            (s, d): m for (s, d), m in self.edges.items()
            if s != node_id and d != node_id
        }

    def measure(self, src: str, dst: str, *, latency_ms: float,
                bandwidth_mbps: float) -> None:
        self.edges[(src, dst)] = EdgeMetrics(
            latency_ms=latency_ms,
            bandwidth_mbps=bandwidth_mbps,
            last_measured=time.time(),
        )

    def expire_stale(self, max_age_s: float = 60.0) -> list[str]:
        """Drop nodes we haven't heard from recently. Returns the dropped ids."""
        cutoff = time.time() - max_age_s
        stale = [nid for nid, info in self.nodes.items() if info.last_seen < cutoff]
        for nid in stale:
            self.remove_node(nid)
        return stale
