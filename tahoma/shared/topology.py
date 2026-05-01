"""Cluster topology types.

`PeerEndpoint` and `PeerLayout` describe a single pipeline stage's view of
its neighbors — the minimum needed for pipeline-parallel transport.

A full `Topology` graph (with measured per-link latency and bandwidth) lives
elsewhere; see `docs/ARCHITECTURE.md`.
"""

from __future__ import annotations

from dataclasses import dataclass


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
