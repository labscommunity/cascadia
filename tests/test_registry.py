"""Tests for the engine registry — every built-in must be discoverable and
must validate its arguments correctly."""

from __future__ import annotations

import argparse

import pytest

from tahoma.worker.engines import registry


def _ns(**kwargs: object) -> argparse.Namespace:
    """Build a minimal namespace with sensible defaults for validate()."""
    defaults: dict[str, object] = {
        "model": "test-model",
        "device": "CPU",
        "rank": 0,
        "total": 1,
        "draft_model": None,
        "spec_k": 4,
        "ov_weight_format": "int4",
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_built_ins_registered() -> None:
    expected = {"pytorch", "ov-optimum", "ov-runtime", "ov-spec", "ov-dist-spec"}
    assert set(registry.names()) >= expected


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown engine"):
        registry.get("does-not-exist")


def test_register_duplicate_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registry.get("pytorch"))


def test_ov_optimum_requires_single_stage() -> None:
    spec = registry.get("ov-optimum")
    with pytest.raises(SystemExit, match="single-stage only"):
        spec.validate(_ns(total=2))
    spec.validate(_ns(total=1))  # ok


def test_ov_spec_requires_draft() -> None:
    spec = registry.get("ov-spec")
    with pytest.raises(SystemExit, match="--draft-model"):
        spec.validate(_ns(total=1))
    spec.validate(_ns(total=1, draft_model="draft-id"))


def test_ov_dist_spec_requires_two_stages_and_draft_on_rank_0() -> None:
    spec = registry.get("ov-dist-spec")
    with pytest.raises(SystemExit, match="--total >= 2"):
        spec.validate(_ns(total=1))
    with pytest.raises(SystemExit, match="--draft-model"):
        spec.validate(_ns(total=2, rank=0, draft_model=None))
    # Worker (rank > 0) does not need a draft on its CLI.
    spec.validate(_ns(total=2, rank=1, draft_model=None))
    spec.validate(_ns(total=2, rank=0, draft_model="draft-id"))


def test_pytorch_no_constraints() -> None:
    spec = registry.get("pytorch")
    spec.validate(_ns(total=4, rank=2))  # any combination is fine
