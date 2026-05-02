"""Shared HuggingFace hub helpers used by every OpenVINO engine.

Two recurring problems these helpers solve:

1. **Recent transformers ignores `local_files_only`.** Functions like
   `_patch_mistral_regex` and optimum-cli's `_infer_task_from_model_name_or_path`
   call `huggingface_hub.model_info()` directly when given a hub-style id
   (`org/name`), bypassing the offline flag. Resolving to a local snapshot path
   makes their `_is_local` check succeed and skips the hub call entirely.

2. **Bundled tokenizers from rainier-style shards reference
   `tokenizer_class: TokenizersBackend`** which isn't importable in any current
   `transformers` install. We try the bundled dir if present, but fall back to
   the model_id snapshot rather than crashing.

Use these helpers from every engine that touches a tokenizer or model id; do
not call `snapshot_download` or `AutoTokenizer.from_pretrained` directly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def force_offline() -> None:
    """Set ``HF_HUB_OFFLINE=1`` for the rest of the process.

    Only affects future imports — modules that already cached the constant
    won't pick it up. Useful as a belt-and-suspenders measure on workers.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"


def resolve_local_snapshot(model_id: str) -> str:
    """Return a local filesystem path for ``model_id``.

    If ``model_id`` already exists on disk it's returned as-is. Otherwise we
    look up the cached snapshot via :func:`huggingface_hub.snapshot_download`
    in offline mode. Raises if the model isn't already cached.
    """
    if Path(model_id).exists():
        return model_id
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, local_files_only=True)


def load_tokenizer(
    model_id: str,
    bundled_dir: Path | None = None,
) -> Any:
    """Load a tokenizer that works in offline / restricted environments.

    Order of attempts:
    1. ``bundled_dir`` if provided and a directory — useful when the shard
       export bundles its own tokenizer.
    2. The local HF snapshot for ``model_id``.

    Falls back from (1) to (2) on ``ValueError``/``KeyError``/``ImportError``,
    which covers the ``TokenizersBackend`` case and any tokenizer_class the
    local transformers install can't import.
    """
    from transformers import AutoTokenizer

    if bundled_dir is not None and bundled_dir.is_dir():
        try:
            return AutoTokenizer.from_pretrained(str(bundled_dir))
        except (ValueError, KeyError, ImportError) as err:
            logger.info(
                "bundled tokenizer at %s unusable (%s); falling back to %s",
                bundled_dir, type(err).__name__, model_id,
            )

    snapshot = resolve_local_snapshot(model_id)
    return AutoTokenizer.from_pretrained(snapshot)
