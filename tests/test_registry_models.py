"""Model registry — persistence, listing, search shim, pull progress events.

The hub-touching paths (search, pull) are tested with a temporary
``TAHOMA_REGISTRY_DIR`` so we never touch the real cache. The hub itself is
not contacted; we monkey-patch ``snapshot_download`` for pull tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tahoma import download as registry


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAHOMA_REGISTRY_DIR", str(tmp_path))


def test_register_and_list_round_trip() -> None:
    e = registry.ModelEntry(id="org/foo", source="huggingface",
                            local_path="/x", size_bytes=42)
    registry.register(e)
    out = registry.list_models()
    assert len(out) == 1
    assert out[0].id == "org/foo"
    assert out[0].size_bytes == 42


def test_list_returns_sorted_by_id() -> None:
    for nid in ("z/x", "a/y", "m/z"):
        registry.register(registry.ModelEntry(id=nid))
    out = [e.id for e in registry.list_models()]
    assert out == ["a/y", "m/z", "z/x"]


def test_get_model_missing_returns_none() -> None:
    assert registry.get_model("nope/none") is None


def test_unregister_removes_entry() -> None:
    registry.register(registry.ModelEntry(id="x/y"))
    assert registry.unregister("x/y") is True
    assert registry.unregister("x/y") is False
    assert registry.list_models() == []


def test_persistence_survives_reload(tmp_path: Path) -> None:
    registry.register(registry.ModelEntry(id="persist/me", size_bytes=1234))
    # Force a reload by clearing any in-process cache (we don't keep one,
    # but this test guards against accidentally adding one).
    out = registry.list_models()
    assert out[0].size_bytes == 1234
    # Inspect raw JSON to confirm we wrote what we expected.
    blob = json.loads((tmp_path / "registry.json").read_text())
    assert blob["models"][0]["id"] == "persist/me"


def test_to_openai_shape() -> None:
    e = registry.ModelEntry(id="x/y", source="huggingface", pulled_at=10.0)
    payload = e.to_openai()
    assert payload["id"] == "x/y"
    assert payload["object"] == "model"
    assert payload["owned_by"] == "huggingface"
    assert payload["created"] == 10


def test_search_hf_returns_empty_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search swallows hub errors and returns []."""
    import tahoma.download

    def boom(*_args, **_kwargs):
        raise RuntimeError("hub down")
    monkeypatch.setattr("huggingface_hub.list_models", boom)
    assert tahoma.download.search_hf("anything") == []


def test_pull_emits_done_event_on_success(monkeypatch: pytest.MonkeyPatch,
                                          tmp_path: Path) -> None:
    """Pull yields a 'pulling' tick and a final 'done' tick; registers entry."""
    fake_snapshot = tmp_path / "fake-model"
    fake_snapshot.mkdir()
    (fake_snapshot / "config.json").write_text("{}")

    import tahoma.download
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_k: str(fake_snapshot),
    )

    events = list(tahoma.download.pull("org/fake"))
    assert events[0].status == "pulling"
    assert events[-1].status == "done"
    assert events[-1].file == str(fake_snapshot)
    # And the entry is now in the registry.
    e = registry.get_model("org/fake")
    assert e is not None
    assert e.local_path == str(fake_snapshot)


def test_pull_emits_error_event_on_hub_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import tahoma.download

    def boom(*_a, **_k):
        raise RuntimeError("404 model not found")
    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)

    events = list(tahoma.download.pull("org/missing"))
    assert events[-1].status == "error"
    assert "404" in events[-1].error
    # No registration on failure.
    assert registry.get_model("org/missing") is None


# --------- HTTP route tests --------------------------------------------------


pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from collections.abc import Iterator  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from tahoma.api.server import make_app  # noqa: E402
from tahoma.shared.types import Chunk, GenerationTask  # noqa: E402


@dataclass
class FakeRunner:
    text_chunks: list[str] = field(default_factory=lambda: ["x"])
    submitted: list[GenerationTask] = field(default_factory=list)

    def generate(self, task: GenerationTask) -> Iterator[Chunk]:
        self.submitted.append(task)
        for i, t in enumerate(self.text_chunks):
            yield Chunk(task_id=task.task_id, token_id=i, text=t,
                        is_final=(i == len(self.text_chunks) - 1))


def _client() -> TestClient:
    return TestClient(make_app(FakeRunner(), model_id="served-model"))


def test_get_models_endpoint_includes_registry_entries() -> None:
    registry.register(registry.ModelEntry(id="extra/model"))
    r = _client().get("/models")
    body = r.json()
    ids = {m["id"] for m in body["models"]}
    assert "extra/model" in ids


def test_v1_models_includes_served_model_first() -> None:
    registry.register(registry.ModelEntry(id="extra/model"))
    r = _client().get("/v1/models")
    data = r.json()["data"]
    assert data[0]["id"] == "served-model"
    ids = [m["id"] for m in data]
    assert "extra/model" in ids


def test_get_unknown_model_404() -> None:
    r = _client().get("/models/does-not-exist")
    assert r.status_code == 404


def test_delete_unregisters() -> None:
    registry.register(registry.ModelEntry(id="rm/me"))
    r = _client().delete("/models/rm/me")
    assert r.status_code == 200
    assert registry.get_model("rm/me") is None


def test_pull_blocking_returns_final_event(monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: Path) -> None:
    fake = tmp_path / "snap"
    fake.mkdir()
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_k: str(fake),
    )
    r = _client().post("/models/pull", json={"model": "org/x", "stream": False})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done"
    assert body["file"] == str(fake)
