"""Model registry — track locally-cached HuggingFace models and pull new ones.

Why a registry on top of HuggingFace's own cache:

- ``hf_hub`` knows what's been downloaded but not whether each download is
  *registered as servable* by tahoma.
- Pull progress needs to stream to API consumers (OpenAI ``/models``,
  Ollama ``/api/pull``, future TUIs); we wrap ``snapshot_download`` to
  emit progress events.
- Multiple workers on the same machine should agree on a single index
  rather than each duplicate-downloading.

The registry persists at ``~/.cache/tahoma/registry.json`` (override with
``TAHOMA_REGISTRY_DIR``). It's a plain file lock + JSON dict, no SQLite —
the access pattern is a few writes per pull and many cheap reads.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _registry_dir() -> Path:
    base = os.environ.get("TAHOMA_REGISTRY_DIR")
    if base:
        return Path(base)
    return Path(os.path.expanduser("~/.cache/tahoma"))


def _registry_path() -> Path:
    return _registry_dir() / "registry.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ModelEntry:
    """One registered model. Mirrors the OpenAI ``models`` payload + ours."""

    id: str
    source: str = "huggingface"          # "huggingface" / "local" / "exported"
    local_path: str | None = None        # filesystem path of the cached snapshot
    pulled_at: float = field(default_factory=time.time)
    size_bytes: int = 0
    revision: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "model",
            "owned_by": self.source,
            "created": int(self.pulled_at),
        }


# ---------------------------------------------------------------------------
# Registry persistence
# ---------------------------------------------------------------------------


_LOCK = threading.RLock()


def _load() -> dict[str, ModelEntry]:
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        logger.warning("registry: %s unreadable (%s); starting empty", path, err)
        return {}
    return {
        entry["id"]: ModelEntry(**{
            k: v for k, v in entry.items() if k in ModelEntry.__dataclass_fields__
        })
        for entry in raw.get("models", [])
    }


def _save(entries: dict[str, ModelEntry]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"models": [asdict(e) for e in entries.values()]}
    # Atomic write so a crashed write doesn't truncate the file.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_models() -> list[ModelEntry]:
    """Return all locally-registered models, sorted by id for stable output."""
    with _LOCK:
        entries = _load()
    return sorted(entries.values(), key=lambda e: e.id)


def get_model(model_id: str) -> ModelEntry | None:
    with _LOCK:
        return _load().get(model_id)


def register(entry: ModelEntry) -> ModelEntry:
    """Add or overwrite an entry. Returns the stored value."""
    with _LOCK:
        entries = _load()
        entries[entry.id] = entry
        _save(entries)
    return entry


def unregister(model_id: str) -> bool:
    """Remove an entry. Returns True if removed, False if not present.

    Does NOT delete the on-disk snapshot — that's a separate concern."""
    with _LOCK:
        entries = _load()
        if model_id not in entries:
            return False
        entries.pop(model_id)
        _save(entries)
    return True


def search_hf(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search the HF hub. Returns a list of {id, downloads, likes, ...}.

    Wraps ``huggingface_hub.list_models``. Best-effort: if the call fails
    (no network, hub down) we return an empty list rather than raising.
    """
    try:
        from huggingface_hub import list_models as hf_list_models
    except ImportError:
        logger.warning("search_hf: huggingface_hub not installed")
        return []
    try:
        results = hf_list_models(search=query, limit=limit, full=False)
    except Exception as err:  # noqa: BLE001
        logger.warning("search_hf: hub query failed: %s", err)
        return []
    out = []
    for m in results:
        out.append({
            "id": m.id,
            "downloads": getattr(m, "downloads", 0) or 0,
            "likes": getattr(m, "likes", 0) or 0,
            "pipeline_tag": getattr(m, "pipeline_tag", None),
            "tags": getattr(m, "tags", []) or [],
        })
    return out


@dataclass
class PullEvent:
    """One progress tick emitted while pulling a model."""

    status: str                   # "pulling" / "done" / "error"
    progress_bytes: int = 0
    total_bytes: int = 0
    file: str | None = None
    error: str | None = None


def pull(model_id: str, *, revision: str | None = None) -> Iterator[PullEvent]:
    """Pull ``model_id`` from the HuggingFace hub.

    Yields :class:`PullEvent` ticks suitable for streaming over an HTTP
    response (Ollama ``/api/pull`` ndjson, OpenAI Tahoma extension, etc.).
    Final event is either ``status="done"`` or ``status="error"``.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        yield PullEvent(status="error", error="huggingface_hub not installed")
        return

    yield PullEvent(status="pulling", file="(starting)")

    try:
        local_path = snapshot_download(model_id, revision=revision)
    except Exception as err:  # noqa: BLE001
        logger.warning("pull(%s) failed: %s", model_id, err)
        yield PullEvent(status="error", error=str(err))
        return

    size = _dir_size(Path(local_path))
    register(ModelEntry(
        id=model_id, source="huggingface",
        local_path=local_path, size_bytes=size,
        revision=revision,
    ))
    yield PullEvent(status="done", progress_bytes=size, total_bytes=size, file=local_path)


def _dir_size(p: Path) -> int:
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        return 0


def discover_local_snapshots() -> Iterable[ModelEntry]:
    """Walk the HF hub cache and yield every cached snapshot we see.

    Useful for the first-run case where the registry is empty but the
    user already has models cached locally.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return
    cache = Path(HF_HUB_CACHE)
    if not cache.is_dir():
        return
    for model_dir in cache.glob("models--*"):
        bare = model_dir.name.removeprefix("models--").replace("--", "/", 1)
        snapshots = model_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        latest: Path | None = None
        for snap in snapshots.iterdir():
            if not snap.is_dir():
                continue
            if latest is None or snap.stat().st_mtime > latest.stat().st_mtime:
                latest = snap
        if latest is None:
            continue
        yield ModelEntry(
            id=bare, source="huggingface",
            local_path=str(latest), size_bytes=_dir_size(latest),
            pulled_at=latest.stat().st_mtime,
        )


__all__ = [
    "ModelEntry",
    "PullEvent",
    "discover_local_snapshots",
    "get_model",
    "list_models",
    "pull",
    "register",
    "search_hf",
    "unregister",
]
