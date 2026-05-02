"""Master election + automatic placement."""

from __future__ import annotations

from tahoma.master import (
    StageRequirement,
    elect_master,
    is_master,
    propose_placements,
)
from tahoma.shared.topology import NodeInfo, Topology


def _node(nid: str, host: str = "10.0.0.1", port: int = 9100,
          device: str = "GPU", mem: int = 16000,
          namespace: str = "default",
          engines: tuple[str, ...] = ("ov-runtime", "ov-dist-spec")) -> NodeInfo:
    return NodeInfo(
        node_id=nid, host=host, port=port, device=device, memory_mb=mem,
        namespace=namespace, engines=list(engines),
    )


def test_elect_master_returns_lowest_id() -> None:
    topo = Topology()
    for nid in ("zeta", "alpha", "mu"):
        topo.add_node(_node(nid))
    assert elect_master(topo) == "alpha"


def test_elect_master_respects_namespace_isolation() -> None:
    topo = Topology()
    topo.add_node(_node("alpha", namespace="prod"))
    topo.add_node(_node("beta", namespace="dev"))
    topo.add_node(_node("gamma", namespace="dev"))
    assert elect_master(topo, namespace="prod") == "alpha"
    assert elect_master(topo, namespace="dev") == "beta"
    assert elect_master(topo, namespace="missing") is None


def test_is_master_helper() -> None:
    topo = Topology()
    topo.add_node(_node("alpha"))
    topo.add_node(_node("beta"))
    assert is_master(topo, "alpha")
    assert not is_master(topo, "beta")


def test_propose_placements_two_node_optimal() -> None:
    topo = Topology()
    topo.add_node(_node("alpha", host="10.0.0.1"))
    topo.add_node(_node("beta", host="10.0.0.2"))
    topo.measure("alpha", "beta", latency_ms=1.0, bandwidth_mbps=2000.0)
    topo.measure("beta", "alpha", latency_ms=1.0, bandwidth_mbps=2000.0)

    reqs = [
        StageRequirement(rank=0, min_memory_mb=8000, needs_engines=("ov-runtime",)),
        StageRequirement(rank=1, min_memory_mb=8000, needs_engines=("ov-runtime",)),
    ]
    proposals = propose_placements(topo, reqs)
    assert len(proposals) >= 1
    p = proposals[0]
    assert {a.rank for a in p.assignments} == {0, 1}
    assert {a.node_id for a in p.assignments} == {"alpha", "beta"}
    assert p.total_latency_ms == 1.0
    assert p.min_bandwidth_mbps == 2000.0


def test_propose_placements_prefers_low_latency_link() -> None:
    """Three nodes; two have a fast link, one is slow. Best 2-stage uses the fast pair."""
    topo = Topology()
    for nid, host in (("alpha", "10.0.0.1"), ("beta", "10.0.0.2"), ("gamma", "10.0.0.3")):
        topo.add_node(_node(nid, host=host))
    topo.measure("alpha", "beta", latency_ms=1.0, bandwidth_mbps=2000.0)
    topo.measure("beta", "alpha", latency_ms=1.0, bandwidth_mbps=2000.0)
    topo.measure("alpha", "gamma", latency_ms=50.0, bandwidth_mbps=200.0)
    topo.measure("gamma", "alpha", latency_ms=50.0, bandwidth_mbps=200.0)
    topo.measure("beta", "gamma", latency_ms=50.0, bandwidth_mbps=200.0)
    topo.measure("gamma", "beta", latency_ms=50.0, bandwidth_mbps=200.0)

    reqs = [
        StageRequirement(rank=0, min_memory_mb=8000),
        StageRequirement(rank=1, min_memory_mb=8000),
    ]
    best = propose_placements(topo, reqs)[0]
    chosen = {a.node_id for a in best.assignments}
    assert chosen == {"alpha", "beta"}, "should pick the fast pair"


def test_propose_placements_filters_undersized_nodes() -> None:
    topo = Topology()
    topo.add_node(_node("alpha", mem=4000))   # too small
    topo.add_node(_node("beta", mem=16000))
    topo.add_node(_node("gamma", mem=16000))

    reqs = [
        StageRequirement(rank=0, min_memory_mb=8000),
        StageRequirement(rank=1, min_memory_mb=8000),
    ]
    p = propose_placements(topo, reqs)[0]
    assert "alpha" not in {a.node_id for a in p.assignments}


def test_propose_placements_filters_missing_engines() -> None:
    topo = Topology()
    topo.add_node(_node("alpha", engines=("pytorch",)))
    topo.add_node(_node("beta", engines=("ov-runtime",)))
    topo.add_node(_node("gamma", engines=("ov-runtime",)))

    reqs = [
        StageRequirement(rank=0, min_memory_mb=0, needs_engines=("ov-runtime",)),
        StageRequirement(rank=1, min_memory_mb=0, needs_engines=("ov-runtime",)),
    ]
    p = propose_placements(topo, reqs)[0]
    assert "alpha" not in {a.node_id for a in p.assignments}


def test_propose_placements_returns_empty_on_undersized_cluster() -> None:
    topo = Topology()
    topo.add_node(_node("alpha"))  # only one node
    reqs = [
        StageRequirement(rank=0, min_memory_mb=0),
        StageRequirement(rank=1, min_memory_mb=0),
    ]
    assert propose_placements(topo, reqs) == []


def test_unmeasured_edges_get_pessimistic_default() -> None:
    """Two-node cluster with no edge measurements → placement still works,
    but the cost is the unmeasured-default latency."""
    from tahoma.master import UNMEASURED_LATENCY_MS

    topo = Topology()
    topo.add_node(_node("alpha"))
    topo.add_node(_node("beta"))
    reqs = [StageRequirement(rank=0, min_memory_mb=0),
            StageRequirement(rank=1, min_memory_mb=0)]
    p = propose_placements(topo, reqs)[0]
    assert p.total_latency_ms == UNMEASURED_LATENCY_MS
