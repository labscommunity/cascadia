"""Zero-config peer discovery for tahoma.

Uses mDNS (via the ``zeroconf`` library) to advertise this node and browse
for peers on the same LAN. Each node publishes a service of type
``_tahoma._tcp.local.`` whose TXT record carries the node id, namespace,
advertised device, available memory, and supported engines.

Discovery populates a :class:`tahoma.shared.topology.Topology` graph; the
master / placement module reads from that graph to compute pipeline splits.

The dependency on ``zeroconf`` is intentionally optional — installing it is
only required when you actually want auto-discovery. The library is small
(<200 KB), pure-python, and works on Linux / macOS / Windows.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any

from tahoma.shared.topology import NodeInfo, Topology

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_tahoma._tcp.local."


def _local_ip() -> str:
    """Best-effort: the IP this host uses to reach the LAN gateway."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def _node_props(info: NodeInfo) -> dict[bytes, bytes]:
    return {
        b"node_id": info.node_id.encode(),
        b"namespace": info.namespace.encode(),
        b"device": info.device.encode(),
        b"memory_mb": str(info.memory_mb).encode(),
        b"engines": ",".join(info.engines).encode(),
    }


def _info_from_service(srv: Any, namespace: str) -> NodeInfo | None:
    if srv is None or not srv.properties:
        return None
    props = {
        (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v or "")
        for k, v in srv.properties.items()
    }
    if props.get("namespace", "default") != namespace:
        return None  # different cluster on the same LAN
    addresses = srv.parsed_addresses() if hasattr(srv, "parsed_addresses") else []
    host = addresses[0] if addresses else srv.server.rstrip(".")
    return NodeInfo(
        node_id=props.get("node_id", srv.name),
        host=host,
        port=srv.port,
        namespace=props.get("namespace", "default"),
        device=props.get("device", "CPU"),
        memory_mb=int(props.get("memory_mb", "0") or 0),
        engines=[e for e in props.get("engines", "").split(",") if e],
    )


class DiscoveryService:
    """Combined advertise + browse for a single node.

    Lifecycle::

        svc = DiscoveryService(my_node_info, topology, namespace="default")
        svc.start()
        ...
        svc.close()

    While running, ``topology`` is kept in sync with the visible peers in the
    same namespace. Stale nodes (no advert for ``max_age_s`` seconds) are
    pruned via :meth:`Topology.expire_stale` on each browser callback.
    """

    def __init__(
        self,
        node: NodeInfo,
        topology: Topology,
        *,
        namespace: str = "default",
        max_age_s: float = 60.0,
    ):
        self._node = node
        self._topology = topology
        self._namespace = namespace
        self._max_age_s = max_age_s
        self._zc: Any = None
        self._service_info: Any = None
        self._browser: Any = None
        self._lock = threading.Lock()

    def start(self) -> None:
        try:
            from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, Zeroconf
        except ImportError as err:  # pragma: no cover
            raise RuntimeError(
                "tahoma.discovery requires the zeroconf package; "
                "install with `pip install tahoma[discovery]` or `pip install zeroconf`",
            ) from err

        self._zc = Zeroconf(ip_version=IPVersion.V4Only)
        host = self._node.host or _local_ip()
        self._service_info = ServiceInfo(
            type_=SERVICE_TYPE,
            name=f"{self._node.node_id[:12]}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(host)],
            port=self._node.port,
            properties=_node_props(self._node),
            server=f"tahoma-{self._node.node_id[:12]}.local.",
        )
        self._zc.register_service(self._service_info)
        # Browser keeps callbacks invoked on its own thread.
        self._browser = ServiceBrowser(self._zc, SERVICE_TYPE, handlers=[self._on_service])
        # Make sure our own node is in the topology even before the browser
        # round-trips back.
        self._topology.add_node(self._node)
        logger.info(
            "discovery: advertising node_id=%s on %s:%d (namespace=%s)",
            self._node.node_id, host, self._node.port, self._namespace,
        )

    def _on_service(self, zc: Any, service_type: str, name: str, state_change: Any) -> None:
        from zeroconf import ServiceStateChange

        if state_change == ServiceStateChange.Removed:
            with self._lock:
                for nid in list(self._topology.nodes):
                    if name.startswith(nid[:12]):
                        self._topology.remove_node(nid)
            return
        try:
            srv = zc.get_service_info(service_type, name, timeout=1000)
        except Exception as err:  # noqa: BLE001
            logger.warning("discovery: get_service_info(%s) failed: %s", name, err)
            return
        info = _info_from_service(srv, namespace=self._namespace)
        if info is None:
            return
        with self._lock:
            self._topology.add_node(info)
            self._topology.expire_stale(self._max_age_s)
        logger.info(
            "discovery: peer %s @ %s:%d device=%s engines=%s",
            info.node_id[:12], info.host, info.port, info.device, info.engines,
        )

    def heartbeat(self) -> None:
        """Refresh our own advert's last_seen so the browser keeps it alive."""
        with self._lock:
            self._node.last_seen = time.time()
            self._topology.add_node(self._node)

    def close(self) -> None:
        if self._zc is None:
            return
        try:
            if self._service_info is not None:
                self._zc.unregister_service(self._service_info)
            self._zc.close()
        except Exception as err:  # noqa: BLE001
            logger.warning("discovery: close failed: %s", err)
        finally:
            self._zc = None
            self._service_info = None
            self._browser = None


__all__ = ["DiscoveryService", "SERVICE_TYPE"]
