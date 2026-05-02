"""Discovery helpers — pure-Python unit tests, no real mDNS sockets.

Full integration with zeroconf is exercised manually on the cluster; here
we just check the property serialisation and the namespace-isolation
filter that controls which adverts make it into the topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tahoma.discovery import _info_from_service, _node_props
from tahoma.shared.topology import NodeInfo


@dataclass
class _FakeServiceInfo:
    """Minimal stand-in for zeroconf.ServiceInfo for the few attrs we read."""

    name: str
    server: str
    port: int
    properties: dict[bytes, bytes]
    addresses: list[str]

    def parsed_addresses(self) -> list[str]:
        return self.addresses


def _node(nid: str = "n1", **kw: Any) -> NodeInfo:
    return NodeInfo(
        node_id=nid, host="10.0.0.1", port=9100,
        device="GPU", memory_mb=16000,
        engines=["ov-runtime"], **kw,
    )


def test_node_props_round_trips_through_info_from_service() -> None:
    info = _node("alpha", namespace="prod")
    props = _node_props(info)
    fake = _FakeServiceInfo(
        name="alpha._tahoma._tcp.local.",
        server="tahoma-alpha.local.",
        port=info.port,
        properties=props,
        addresses=[info.host],
    )
    parsed = _info_from_service(fake, namespace="prod")
    assert parsed is not None
    assert parsed.node_id == "alpha"
    assert parsed.namespace == "prod"
    assert parsed.host == "10.0.0.1"
    assert parsed.port == 9100
    assert parsed.device == "GPU"
    assert parsed.memory_mb == 16000
    assert parsed.engines == ["ov-runtime"]


def test_info_from_service_drops_other_namespaces() -> None:
    info = _node("alpha", namespace="prod")
    fake = _FakeServiceInfo(
        name="alpha._tahoma._tcp.local.", server="x", port=9100,
        properties=_node_props(info), addresses=[info.host],
    )
    assert _info_from_service(fake, namespace="dev") is None


def test_info_from_service_handles_missing_properties() -> None:
    fake = _FakeServiceInfo(
        name="x._tahoma._tcp.local.", server="x", port=9100,
        properties={}, addresses=["10.0.0.1"],
    )
    assert _info_from_service(fake, namespace="default") is None


def test_info_from_service_handles_str_property_values() -> None:
    """Some zeroconf versions decode TXT values to str eagerly; we accept either."""
    fake = _FakeServiceInfo(
        name="alpha._tahoma._tcp.local.", server="x", port=9100,
        properties={
            "node_id": "alpha", "namespace": "default",
            "device": "GPU", "memory_mb": "8000", "engines": "ov-spec,ov-runtime",
        },
        addresses=["10.0.0.5"],
    )
    parsed = _info_from_service(fake, namespace="default")
    assert parsed is not None
    assert parsed.engines == ["ov-spec", "ov-runtime"]
    assert parsed.memory_mb == 8000
