"""Unit tests for export_shards' config.json fetch + its failure warning.

Run from the repo root with:

    python -m pytest tools/tests/test_config_fetch.py -v

`fetch_config_json` downloads into a `local_dir`, which makes
hf_hub_download bypass the HF hub cache — so an HF_HUB_OFFLINE=1 run over a
fully-cached model used to raise, main() swallowed it, and the config-first
Gemma 4 / Qwen3.5 dispatch was skipped silently: the generic path then
refused the model as unsupported. These cover the cache-only retry and the
warning that replaced the silent fall-through.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# Add tools/ to sys.path so we can import export_shards as a module.
_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

try:
    import export_shards
except ImportError as exc:
    if "torch" in str(exc) or "numpy" in str(exc):
        pytest.skip(
            f"export_shards.py needs torch/numpy at import time: {exc}",
            allow_module_level=True,
        )
    raise


def _patch_hf_hub_download(monkeypatch, fn):
    """Patch the hf_hub_download `fetch_config_json` imports at call time."""
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "hf_hub_download", fn)


def test_fetch_config_json_falls_back_to_hub_cache(tmp_path, monkeypatch):
    cached = tmp_path / "config.json"
    cached.write_text(json.dumps({"model_type": "gemma4"}))

    def _download(repo_id, filename, local_dir=None, local_files_only=False, revision=None):
        if local_dir is not None:
            raise OSError("offline: local_dir bypasses the hub cache")
        assert local_files_only
        return str(cached)

    _patch_hf_hub_download(monkeypatch, _download)
    assert export_shards.fetch_config_json("google/gemma-4-E2B-it") == str(cached)


def test_fetch_config_json_reraises_the_first_error(monkeypatch):
    def _download(repo_id, filename, local_dir=None, local_files_only=False, revision=None):
        raise OSError("first" if local_dir is not None else "second")

    _patch_hf_hub_download(monkeypatch, _download)
    with pytest.raises(OSError, match="first"):
        export_shards.fetch_config_json("google/gemma-4-E2B-it")


def test_main_warns_when_config_fetch_fails(tmp_path, monkeypatch, capsys):
    def _boom(_model, _revision=None):
        raise OSError("offline and not cached")

    monkeypatch.setattr(export_shards, "fetch_config_json", _boom)
    # No hub lookup for the sidecar's revision either.
    monkeypatch.setattr(export_shards, "_source_identity", lambda a: {
        "model_id": "gemma-4-E2B-it", "repo": a.model, "revision": None})
    # Stop main() right after the skipped dispatch: the generic path needs torch.
    monkeypatch.setattr(export_shards, "torch", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_shards.py",
            "--model", "google/gemma-4-E2B-it",
            "--output-dir", str(tmp_path / "out"),
            "--num-stages", "2",
        ],
    )
    with pytest.raises(SystemExit):
        export_shards.main()
    err = capsys.readouterr().err
    assert "google/gemma-4-E2B-it" in err
    assert "offline and not cached" in err
    assert "config-first model-type dispatch was skipped" in err
