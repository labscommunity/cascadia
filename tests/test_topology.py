"""Topology graph: add/remove nodes, measure edges, prune stale entries."""

from __future__ import annotations

import time

from tahoma.shared.topology import EdgeMetrics, NodeInfo, Topology


def _node(nid: str, host: str = "10.0.0.1", port: int = 9100,
          device: str = "GPU", mem: int = 16000) -> NodeInfo:
    return NodeInfo(node_id=nid, host=host, port=port, device=device, memory_mb=mem)


def test_add_node_records_last_seen() -> None:
    topo = Topology()
    n = _node("a")
    topo.add_node(n)
    assert "a" in topo.nodes
    assert topo.nodes["a"].last_seen > 0


def test_remove_node_drops_associated_edges() -> None:
    topo = Topology()
    topo.add_node(_node("a"))
    topo.add_node(_node("b"))
    topo.measure("a", "b", latency_ms=2.0, bandwidth_mbps=900.0)
    topo.measure("b", "a", latency_ms=2.0, bandwidth_mbps=900.0)

    topo.remove_node("a")

    assert "a" not in topo.nodes
    assert ("a", "b") not in topo.edges
    assert ("b", "a") not in topo.edges


def test_measure_records_edge_metrics() -> None:
    topo = Topology()
    topo.add_node(_node("a"))
    topo.add_node(_node("b"))
    topo.measure("a", "b", latency_ms=1.5, bandwidth_mbps=1200.0)

    e = topo.edges[("a", "b")]
    assert isinstance(e, EdgeMetrics)
    assert e.latency_ms == 1.5
    assert e.bandwidth_mbps == 1200.0
    assert e.last_measured > 0


def test_expire_stale_drops_old_nodes() -> None:
    topo = Topology()
    fresh = _node("fresh")
    stale = _node("stale")
    stale.last_seen = time.time() - 120  # 2 min ago
    topo.add_node(fresh)
    topo.nodes["stale"] = stale  # bypass add_node so we keep the manual last_seen

    dropped = topo.expire_stale(max_age_s=60.0)
    assert dropped == ["stale"]
    assert "stale" not in topo.nodes
    assert "fresh" in topo.nodes
