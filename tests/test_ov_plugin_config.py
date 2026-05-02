"""Tests for the shared OV plugin-config helper.

This is the small dict builder that every OV-backed engine uses to
plumb CLI knobs (--ov-cache-dir, --ov-kv-precision, --ov-dyn-quant-group)
into the OpenVINO compile_model / LLMPipeline calls.
"""

from __future__ import annotations

from tahoma.worker.engines.openvino._plugin import build_plugin_config


def test_empty_returns_empty_dict() -> None:
    assert build_plugin_config() == {}
    assert build_plugin_config(None, None, None) == {}


def test_cache_dir_only() -> None:
    cfg = build_plugin_config(cache_dir="/tmp/ov_cache")
    assert cfg == {"CACHE_DIR": "/tmp/ov_cache"}


def test_all_three_set() -> None:
    cfg = build_plugin_config(
        cache_dir="/tmp/ov_cache",
        kv_cache_precision="u8",
        dyn_quant_group="32",
    )
    assert cfg == {
        "CACHE_DIR": "/tmp/ov_cache",
        "KV_CACHE_PRECISION": "u8",
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
    }


def test_falsy_values_excluded() -> None:
    """Empty strings should NOT generate dict entries — that would override
    OV's default behaviour with a meaningless value."""
    assert build_plugin_config(cache_dir="") == {}
    assert build_plugin_config(kv_cache_precision="") == {}


def test_distributed_engine_builders_accept_plugin_kwargs() -> None:
    """Smoke: the new kwargs are accepted by ov-runtime + ov-dist-spec
    builders without exploding at construction time. The actual
    compile_model integration is exercised by the on-device e2e."""
    from tahoma.worker.engines.openvino.dist_spec import (
        OVDistributedSpecBuilder,
        OVDistSpecWorkerBuilder,
    )
    from tahoma.worker.engines.openvino.ov_runtime import OVRuntimeBuilder

    OVRuntimeBuilder(
        pipeline_dir="/fake", rank=0, total=1, device="CPU",
        cache_dir="/tmp/ov_cache", kv_cache_precision="u8", dyn_quant_group="32",
    )
    OVDistributedSpecBuilder(
        pipeline_dir="/fake", draft_model_path="/fake-draft", device="CPU",
        cache_dir="/tmp/ov_cache",
    )
    OVDistSpecWorkerBuilder(
        pipeline_dir="/fake", rank=1, total=2, device="CPU",
        cache_dir="/tmp/ov_cache",
    )
