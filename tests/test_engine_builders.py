"""Builder error / lifecycle tests that don't require real models or hardware.

These exercise the validation guards (`is_first_stage`, `is_last_stage`,
`stage_<rank>` directory layout, missing required configs) — the parts that
fail-fast before any GPU work happens.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tahoma.shared.shard import ShardSpec
from tahoma.shared.topology import PeerEndpoint, PeerLayout
from tahoma.worker.engines.openvino.dist_spec import (
    OVDistributedSpecBuilder,
    OVDistSpecWorkerBuilder,
)


def _shard(first: bool = True, last: bool = True) -> ShardSpec:
    return ShardSpec(
        model_id="fake", layer_start=0, layer_end=0, total_layers=0,
        device="CPU", is_first_stage=first, is_last_stage=last,
    )


def _write_v3_pipeline(tmp: Path) -> Path:
    (tmp / "stage_0").mkdir()
    (tmp / "pipeline_config.json").write_text(json.dumps({
        "model_id": "fake",
        "hidden_size": 16,
        "num_stages": 2,
        "export_version": "v3_kv_cached",
    }))
    (tmp / "stage_0" / "stage_config.json").write_text(json.dumps({
        "stage": 0, "has_embed": True, "has_head": False,
    }))
    return tmp


def _write_v5_no_embed(tmp: Path) -> Path:
    (tmp / "stage_0").mkdir()
    (tmp / "pipeline_config.json").write_text(json.dumps({
        "model_id": "fake",
        "hidden_size": 16,
        "num_stages": 2,
        "export_version": "v5_canonical_inputs_paged_attention",
    }))
    (tmp / "stage_0" / "stage_config.json").write_text(json.dumps({
        "stage": 0, "has_embed": False, "has_head": False,
    }))
    return tmp


def test_driver_connect_requires_downstream(tmp_path: Path) -> None:
    builder = OVDistributedSpecBuilder(
        pipeline_dir=str(tmp_path), draft_model_path="x", device="CPU",
    )
    with pytest.raises(RuntimeError, match="downstream peer"):
        builder.connect(PeerLayout(upstream=None, downstream=None))


def test_driver_connect_rejects_upstream(tmp_path: Path) -> None:
    builder = OVDistributedSpecBuilder(
        pipeline_dir=str(tmp_path), draft_model_path="x", device="CPU",
    )
    with pytest.raises(RuntimeError, match="should not have an upstream"):
        builder.connect(PeerLayout(
            upstream=PeerEndpoint(host="localhost", port=1234),
            downstream=PeerEndpoint(host="localhost", port=1234),
        ))


def test_driver_load_rejects_v3_shards(tmp_path: Path) -> None:
    _write_v3_pipeline(tmp_path)
    builder = OVDistributedSpecBuilder(
        pipeline_dir=str(tmp_path), draft_model_path="x", device="CPU",
    )
    with pytest.raises(RuntimeError, match="v5 shards"):
        list(builder.load(_shard()))


def test_driver_load_rejects_stage0_without_embed(tmp_path: Path) -> None:
    _write_v5_no_embed(tmp_path)
    builder = OVDistributedSpecBuilder(
        pipeline_dir=str(tmp_path), draft_model_path="x", device="CPU",
    )
    with pytest.raises(RuntimeError, match="has_embed"):
        list(builder.load(_shard()))


def test_worker_connect_requires_upstream(tmp_path: Path) -> None:
    builder = OVDistSpecWorkerBuilder(
        pipeline_dir=str(tmp_path), rank=1, total=2, device="CPU",
    )
    builder.configure_listen("127.0.0.1", 0)
    with pytest.raises(RuntimeError, match="upstream peer"):
        builder.connect(PeerLayout(upstream=None, downstream=None))


def test_worker_connect_requires_configure_listen(tmp_path: Path) -> None:
    builder = OVDistSpecWorkerBuilder(
        pipeline_dir=str(tmp_path), rank=1, total=2, device="CPU",
    )
    with pytest.raises(RuntimeError, match="configure_listen"):
        builder.connect(PeerLayout(
            upstream=PeerEndpoint(host="0.0.0.0", port=9100),
            downstream=None,
        ))


def test_worker_load_rejects_v3_shards(tmp_path: Path) -> None:
    (tmp_path / "stage_1").mkdir()
    (tmp_path / "stage_1" / "stage_config.json").write_text(json.dumps({
        "stage": 1, "has_head": True,
        "export_version": "v3_kv_cached",
    }))
    builder = OVDistSpecWorkerBuilder(
        pipeline_dir=str(tmp_path), rank=1, total=2, device="CPU",
    )
    with pytest.raises(RuntimeError, match="v5 shards"):
        list(builder.load(_shard(first=False, last=True)))


def test_worker_load_rejects_last_stage_without_head(tmp_path: Path) -> None:
    (tmp_path / "stage_1").mkdir()
    (tmp_path / "stage_1" / "stage_config.json").write_text(json.dumps({
        "stage": 1, "has_head": False,
        "export_version": "v5_canonical_inputs",
    }))
    builder = OVDistSpecWorkerBuilder(
        pipeline_dir=str(tmp_path), rank=1, total=2, device="CPU",
    )
    with pytest.raises(RuntimeError, match="has_head"):
        list(builder.load(_shard(first=False, last=True)))


def test_worker_load_falls_back_to_flat_layout(tmp_path: Path) -> None:
    """Worker can be pointed at a directory that IS the stage dir
    (no `stage_<rank>/` wrapper) — common when only one stage was scp'd."""
    (tmp_path / "stage_config.json").write_text(json.dumps({
        "stage": 1, "has_head": True,
        "export_version": "v5_canonical_inputs",
    }))
    builder = OVDistSpecWorkerBuilder(
        pipeline_dir=str(tmp_path), rank=1, total=2, device="CPU",
    )
    # Will fail at compile_model (no real openvino_model.xml), but the layout
    # check should pass first.
    with pytest.raises(Exception) as exc:
        list(builder.load(_shard(first=False, last=True)))
    # Should NOT be the "v5 shards" or "has_head" guard — those would mean
    # the fallback layout wasn't picked up.
    assert "v5 shards" not in str(exc.value)
    assert "has_head" not in str(exc.value)


def test_build_before_connect_raises(tmp_path: Path) -> None:
    builder = OVDistributedSpecBuilder(
        pipeline_dir=str(tmp_path), draft_model_path="x", device="CPU",
    )
    with pytest.raises(RuntimeError, match="call (connect|load)"):
        builder.build()
