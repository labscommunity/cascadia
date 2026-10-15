"""export_gemma4.py --vision: the traceable image splice matches HF's
masked_scatter, and steps without images leave the embedding untouched.

    python -m pytest tools/tests/test_gemma4_vision_splice.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import export_gemma4  # noqa: E402

IMG = 7


def _hf(h, ids, feats):
    mask = (ids == IMG).unsqueeze(-1).expand_as(h)
    return h.masked_scatter(mask, feats)


@pytest.mark.parametrize(
    "ids",
    [
        [1, IMG, IMG, 2, IMG, 3],  # two images, split runs
        [IMG, IMG, IMG],  # all image
        [IMG],  # one image token
    ],
)
def test_splice_matches_masked_scatter(ids):
    torch.manual_seed(0)
    ids = torch.tensor([ids])
    h = torch.randn(1, ids.shape[1], 5)
    feats = torch.randn(int((ids == IMG).sum()), 5)
    got = export_gemma4.splice_image_rows(h, ids == IMG, feats)
    assert torch.equal(got, _hf(h, ids, feats))


def test_no_images_leaves_the_embedding_untouched():
    h = torch.randn(1, 4, 5)
    ids = torch.tensor([[1, 2, 3, 4]])
    got = export_gemma4.splice_image_rows(h, ids == IMG, torch.zeros(0, 5))
    assert torch.equal(got, h)
