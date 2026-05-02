"""Exports from `tahoma` are stable public API and must remain importable."""

from __future__ import annotations


def test_top_level_exports() -> None:
    import tahoma

    expected = {
        "Builder",
        "Chunk",
        "Engine",
        "GenerationTask",
        "LoadProgress",
        "Runner",
        "TaskId",
        "__version__",
    }
    assert expected.issubset(set(tahoma.__all__))
    for name in expected:
        assert hasattr(tahoma, name), f"tahoma.{name} missing"


def test_version_string() -> None:
    import tahoma
    assert isinstance(tahoma.__version__, str)
    assert tahoma.__version__.count(".") >= 2
