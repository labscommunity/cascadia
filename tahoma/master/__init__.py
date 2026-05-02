"""Cluster master — leader election and automatic placement.

Two responsibilities:

1. **Election.** All nodes in a namespace see the same :class:`Topology`
   (populated by mDNS discovery). The lowest-lexicographic node id is the
   master; everyone else is a worker. No explicit messaging — election is
   purely a function of the visible peer set.

2. **Placement.** Given a model's per-stage memory + compute requirements
   plus the topology graph, return a :class:`PlacementProposal`: which node
   runs which rank, and the latency/bandwidth-cost for the chosen
   pipeline order. This is the engine behind ``/instance/previews``.

Placement uses two simple heuristics:

- **Capacity:** discard nodes whose advertised ``memory_mb`` is below the
  per-stage requirement.
- **Adjacency cost:** sum the measured latency between adjacent stages on
  the chosen ordering. Unmeasured edges are assigned a pessimistic default
  (``UNMEASURED_LATENCY_MS``) so we don't blindly favour them.

This is intentionally a "good enough" placement; the harder optimal
ordering (TSP-shaped) is deferred to a follow-up. For 2- and 3-node
clusters the brute-force search here returns the global optimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from tahoma.shared.topology import EdgeMetrics, NodeInfo, Topology

# When an edge has not been probed, treat it as a slow-but-not-broken link.
# Pessimistic on purpose — we'd rather prefer a known-fast link.
UNMEASURED_LATENCY_MS = 50.0
UNMEASURED_BW_MBPS = 100.0


@dataclass(frozen=True)
class StageRequirement:
    """Per-stage demand. Driven by the model's pipeline_config.json."""

    rank: int
    min_memory_mb: int
    needs_engines: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageAssignment:
    rank: int
    node_id: str
    host: str
    port: int


@dataclass(frozen=True)
class PlacementProposal:
    """A concrete proposal: rank → node, plus the cost we computed."""

    assignments: tuple[StageAssignment, ...]
    total_latency_ms: float
    min_bandwidth_mbps: float
    namespace: str

    def as_dict(self) -> dict:
        return {
            "namespace": self.namespace,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "min_bandwidth_mbps": round(self.min_bandwidth_mbps, 3),
            "assignments": [
                {"rank": a.rank, "node_id": a.node_id, "host": a.host, "port": a.port}
                for a in self.assignments
            ],
        }


def elect_master(topology: Topology, namespace: str = "default") -> str | None:
    """Return the master's node id for a namespace, or None if no peers exist."""
    candidates = [n.node_id for n in topology.nodes.values() if n.namespace == namespace]
    if not candidates:
        return None
    return min(candidates)


def is_master(topology: Topology, my_node_id: str, namespace: str = "default") -> bool:
    return elect_master(topology, namespace) == my_node_id


def _edge_or_default(edges: dict[tuple[str, str], EdgeMetrics],
                     src: str, dst: str) -> EdgeMetrics:
    return edges.get((src, dst)) or edges.get((dst, src)) or EdgeMetrics(
        latency_ms=UNMEASURED_LATENCY_MS,
        bandwidth_mbps=UNMEASURED_BW_MBPS,
        last_measured=0.0,
    )


def _candidate_nodes(
    topology: Topology,
    namespace: str,
    requirements: list[StageRequirement],
) -> list[NodeInfo]:
    nodes: list[NodeInfo] = []
    needed_engines = {e for r in requirements for e in r.needs_engines}
    min_memory = max((r.min_memory_mb for r in requirements), default=0)
    for n in topology.nodes.values():
        if n.namespace != namespace:
            continue
        if n.memory_mb < min_memory:
            continue
        if needed_engines and not (set(n.engines) & needed_engines):
            continue
        nodes.append(n)
    # Stable order so placement is deterministic for a given topology snapshot.
    return sorted(nodes, key=lambda n: n.node_id)


def propose_placements(
    topology: Topology,
    requirements: list[StageRequirement],
    *,
    namespace: str = "default",
    max_proposals: int = 5,
) -> list[PlacementProposal]:
    """Return the top-N placements ordered by ascending total latency.

    Brute-force enumerates every assignment of nodes to ranks, scores each
    by sum-of-adjacent-latencies, and returns the cheapest few. For
    n-node / m-rank with m ≤ n the search space is n!/(n-m)! which is fine
    up to ~7 nodes. Beyond that we fall back to a greedy heuristic.
    """
    if not requirements:
        return []
    nodes = _candidate_nodes(topology, namespace, requirements)
    if len(nodes) < len(requirements):
        return []

    proposals: list[PlacementProposal] = []
    if len(nodes) <= 7:
        ordered_perms = permutations(nodes, len(requirements))
    else:
        ordered_perms = _greedy_perms(nodes, requirements, topology, namespace)

    for perm in ordered_perms:
        assignments = tuple(
            StageAssignment(
                rank=req.rank, node_id=node.node_id,
                host=node.host, port=node.port,
            )
            for req, node in zip(requirements, perm)
        )
        if any(a.node_id == b.node_id for i, a in enumerate(assignments)
               for b in assignments[i + 1:]):
            continue  # one node can hold at most one stage in pipeline-only mode
        latency = 0.0
        min_bw = float("inf")
        for left, right in zip(assignments, assignments[1:]):
            e = _edge_or_default(topology.edges, left.node_id, right.node_id)
            latency += e.latency_ms
            if e.bandwidth_mbps < min_bw:
                min_bw = e.bandwidth_mbps
        if min_bw == float("inf"):
            min_bw = 0.0
        proposals.append(PlacementProposal(
            assignments=assignments,
            total_latency_ms=latency,
            min_bandwidth_mbps=min_bw,
            namespace=namespace,
        ))

    proposals.sort(key=lambda p: (p.total_latency_ms, -p.min_bandwidth_mbps))
    return proposals[:max_proposals]


def _greedy_perms(nodes, requirements, topology, namespace):  # type: ignore[no-untyped-def]
    """Fallback when there are too many nodes for full permutation search.

    Picks the lowest-id node for rank 0, then at each subsequent rank takes
    the node that minimises latency to the previous pick. Yields a single
    assignment so the caller still sees a PlacementProposal.
    """
    chosen = [nodes[0]]
    remaining = nodes[1:]
    for _ in requirements[1:]:
        if not remaining:
            return iter(())
        prev = chosen[-1]
        remaining.sort(key=lambda n: _edge_or_default(
            topology.edges, prev.node_id, n.node_id,
        ).latency_ms)
        chosen.append(remaining.pop(0))
    return iter([tuple(chosen)])


__all__ = [
    "PlacementProposal",
    "StageAssignment",
    "StageRequirement",
    "UNMEASURED_BW_MBPS",
    "UNMEASURED_LATENCY_MS",
    "elect_master",
    "is_master",
    "propose_placements",
]
