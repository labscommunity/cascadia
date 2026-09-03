"""Unit tests for the Gemma 4 attention mask (`export_gemma4.build_attention_mask`).

Run from the repo root with:

    python -m pytest tools/tests/test_gemma4_sliding_window.py -v

Gemma 4 alternates five `sliding_attention` layers with one `full_attention`
layer. The exporter originally built a plain causal mask for every layer, so a
sliding layer could attend to the entire prefix. That agrees with HF while the
sequence is shorter than the window (1024 for 31B-it), which is why short
prompts look correct; past the window the exported graph and HF diverge and
output degrades. These tests pin the window semantics so that cannot regress
silently.

The mask is the whole contract, so the tests check membership directly rather
than comparing against a golden tensor: key `j` is visible to query `i` exactly
when `0 <= (past + i) - j < sliding_window`.
"""

from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch", reason="mask construction is a torch op")

# Add tools/ to sys.path so we can import export_gemma4 as a module.
_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import export_gemma4  # noqa: E402


def visible(mask):
    """{(query, key)} pairs the mask leaves unmasked."""
    finite = torch.isfinite(mask)
    return {(int(i), int(j)) for i, j in finite.nonzero(as_tuple=False)}


def expected(seq_len, full_seq_len, window):
    past = full_seq_len - seq_len
    out = set()
    for i in range(seq_len):
        for j in range(full_seq_len):
            delta = (past + i) - j
            if delta < 0:
                continue  # causal: no peeking at future keys
            if window is not None and delta >= window:
                continue  # outside the sliding window
            out.add((i, j))
    return out


@pytest.mark.parametrize(
    "seq_len,full_seq_len,window",
    [
        (8, 8, None),  # prefill, full attention
        (8, 8, 4),  # prefill, window shorter than the prompt
        (1, 9, None),  # decode step, full attention
        (1, 9, 4),  # decode step, window bites
        (1, 9, 32),  # window wider than the cache is a no-op
        (5, 12, 3),  # chunked prefill against an existing cache
    ],
)
def test_mask_matches_window_rule(seq_len, full_seq_len, window):
    mask = export_gemma4.build_attention_mask(
        seq_len, full_seq_len, window, torch.device("cpu"), torch.float32
    )
    assert mask.shape == (seq_len, full_seq_len)
    assert visible(mask) == expected(seq_len, full_seq_len, window)


def test_window_is_a_strict_subset_of_causal():
    """A windowed layer never sees more than the same layer without a window."""
    causal = visible(
        export_gemma4.build_attention_mask(
            6, 20, None, torch.device("cpu"), torch.float32
        )
    )
    windowed = visible(
        export_gemma4.build_attention_mask(
            6, 20, 4, torch.device("cpu"), torch.float32
        )
    )
    assert windowed < causal


def test_every_query_keeps_at_least_its_own_key():
    """No row may be fully masked -- softmax over an all -inf row is NaN."""
    for window in (1, 2, 1024):
        mask = export_gemma4.build_attention_mask(
            4, 4096, window, torch.device("cpu"), torch.float32
        )
        assert torch.isfinite(mask).any(dim=1).all()


def test_window_of_one_is_diagonal_only():
    mask = export_gemma4.build_attention_mask(
        3, 10, 1, torch.device("cpu"), torch.float32
    )
    assert visible(mask) == {(0, 7), (1, 8), (2, 9)}
