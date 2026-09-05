"""Unit tests for Gemma 4 sliding-window masking in `export_gemma4`.

Run from the repo root with:

    python -m pytest tools/tests/test_gemma4_sliding_window.py -v

Gemma 4 alternates five `sliding_attention` layers with one `full_attention`
layer. The exporter originally built a plain causal mask for every layer, so a
sliding layer could attend to the entire prefix. That agrees with HF while the
sequence is shorter than the window (1024 for 31B-it), which is why short
prompts look correct; past the window the exported graph and HF diverge and
output degrades. These tests pin the window semantics so that cannot regress
silently.

Three layers of coverage:

* `build_attention_mask` -- the mask is the whole contract, so membership is
  checked directly rather than against a golden tensor: key `j` is visible to
  query `i` exactly when `0 <= (past + i) - j < sliding_window`. A window
  `<= 0` must raise (it would mask every key and softmax would return NaN).
* `resolve_sliding_window` -- how `text_config.sliding_window` becomes the
  baked value: 0/absent disables, negative is a configuration error.
* `cached_gemma4_layer_forward` / `cached_gemma4_shared_layer_forward` -- the
  production call sites, driven with a tiny fake decoder layer, so a forward
  that stops passing the window (or re-inlines a plain causal mask) fails
  here rather than only in a long-context eval.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

torch = pytest.importorskip("torch", reason="mask construction is a torch op")

# Add tools/ to sys.path so we can import export_gemma4 as a module.
_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import export_gemma4  # noqa: E402

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# build_attention_mask
# ---------------------------------------------------------------------------


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
        (4, 8, 8),  # window == full_seq_len: the band just vanishes
        (1, 9, None),  # decode step, full attention
        (1, 9, 4),  # decode step, window bites
        (1, 9, 32),  # window wider than the cache is a no-op
        (5, 12, 3),  # chunked prefill against an existing cache
    ],
)
def test_mask_matches_window_rule(seq_len, full_seq_len, window):
    mask = export_gemma4.build_attention_mask(
        seq_len, full_seq_len, window, CPU, torch.float32
    )
    assert mask.shape == (seq_len, full_seq_len)
    assert visible(mask) == expected(seq_len, full_seq_len, window)


def test_window_is_a_strict_subset_of_causal():
    """A windowed layer never sees more than the same layer without a window."""
    causal = visible(
        export_gemma4.build_attention_mask(6, 20, None, CPU, torch.float32)
    )
    windowed = visible(
        export_gemma4.build_attention_mask(6, 20, 4, CPU, torch.float32)
    )
    assert windowed < causal


def test_every_query_keeps_at_least_its_own_key():
    """No row may be fully masked -- softmax over an all -inf row is NaN."""
    for window in (1, 2, 1024):
        mask = export_gemma4.build_attention_mask(
            4, 4096, window, CPU, torch.float32
        )
        assert torch.isfinite(mask).any(dim=1).all()


def test_window_of_one_is_diagonal_only():
    mask = export_gemma4.build_attention_mask(3, 10, 1, CPU, torch.float32)
    assert visible(mask) == {(0, 7), (1, 8), (2, 9)}


@pytest.mark.parametrize("window", [0, -1, -1024])
def test_window_must_be_positive(window):
    """A zero/negative band would mask every key; refuse instead of NaN."""
    with pytest.raises(ValueError, match="sliding_window must be >= 1"):
        export_gemma4.build_attention_mask(3, 10, window, CPU, torch.float32)


# ---------------------------------------------------------------------------
# resolve_sliding_window (text_config -> baked value)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,resolved",
    [
        (1024, 1024),  # 31B-it
        (512, 512),  # E2B / E4B
        (0, None),  # 0 disables the bound
        (None, None),  # explicit null disables the bound
    ],
)
def test_resolve_sliding_window_reads_text_config(raw, resolved):
    cfg = types.SimpleNamespace(sliding_window=raw)
    assert export_gemma4.resolve_sliding_window(cfg) == resolved


def test_resolve_sliding_window_absent_disables():
    cfg = types.SimpleNamespace()
    assert export_gemma4.resolve_sliding_window(cfg) is None


@pytest.mark.parametrize("raw", [-1, -512])
def test_resolve_sliding_window_rejects_negative(raw):
    cfg = types.SimpleNamespace(sliding_window=raw)
    with pytest.raises(ValueError, match="text_config.sliding_window"):
        export_gemma4.resolve_sliding_window(cfg)


# ---------------------------------------------------------------------------
# Layer forwards (the production call sites)
# ---------------------------------------------------------------------------

HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 16, 4, 2, 8
WINDOW = 4
# Explicit everywhere: another test module in this suite flips torch's default
# dtype to float16 (as export_shards.main() does), and the exporter's rotary
# always emits float32 tables.
DTYPE = torch.float32


def make_fake_layer(seed=0):
    """The attribute surface the cached layer forwards touch.

    Identity stands in for every RMSNorm; the projections and MLP are small
    seeded `Linear`s so the test is deterministic. No `layer_scalar`, and
    `per_layer_input` is always None, so the PLI branch is skipped.
    """
    torch.manual_seed(seed)
    nn = torch.nn

    def linear(n_in, n_out):
        return nn.Linear(n_in, n_out, bias=False, dtype=DTYPE)

    self_attn = types.SimpleNamespace(
        q_proj=linear(HIDDEN, NUM_HEADS * HEAD_DIM),
        k_proj=linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM),
        v_proj=linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM),
        o_proj=linear(NUM_HEADS * HEAD_DIM, HIDDEN),
        q_norm=nn.Identity(),
        k_norm=nn.Identity(),
        v_norm=nn.Identity(),
    )
    return types.SimpleNamespace(
        input_layernorm=nn.Identity(),
        self_attn=self_attn,
        post_attention_layernorm=nn.Identity(),
        pre_feedforward_layernorm=nn.Identity(),
        mlp=linear(HIDDEN, HIDDEN),
        post_feedforward_layernorm=nn.Identity(),
    )


def rope_tables(seq_len, past):
    """cos/sin as (batch, seq, head_dim) for absolute positions
    past..past+seq_len-1, from the exporter's own rotary module."""
    Rotary = export_gemma4._make_traced_rotary_class()
    rotary = Rotary(HEAD_DIM, 10000.0)
    position_ids = torch.arange(past, past + seq_len).unsqueeze(0)
    return rotary(position_ids, DTYPE)


def step_inputs(seq_len, past, seed=1):
    """Deterministic hidden states + past KV for one step."""
    torch.manual_seed(seed)
    x = torch.randn(1, seq_len, HIDDEN, dtype=DTYPE)
    past_k = torch.randn(1, NUM_KV_HEADS, past, HEAD_DIM, dtype=DTYPE)
    past_v = torch.randn(1, NUM_KV_HEADS, past, HEAD_DIM, dtype=DTYPE)
    cos, sin = rope_tables(seq_len, past)
    return x, cos, sin, past_k, past_v


def run_layer(layer, seq_len, past, sliding_window):
    x, cos, sin, past_k, past_v = step_inputs(seq_len, past)
    with torch.no_grad():
        return export_gemma4.cached_gemma4_layer_forward(
            layer,
            x,
            cos,
            sin,
            past_k,
            past_v,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            None,
            sliding_window=sliding_window,
        )


def run_shared_layer(layer, seq_len, past, source_k, source_v, sliding_window):
    x, cos, sin, _, _ = step_inputs(seq_len, past)
    with torch.no_grad():
        return export_gemma4.cached_gemma4_shared_layer_forward(
            layer,
            x,
            cos,
            sin,
            source_k,
            source_v,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            None,
            sliding_window=sliding_window,
        )


@pytest.mark.parametrize(
    "seq_len,past",
    [
        (12, 0),  # prefill longer than the window
        (1, 10),  # decode step against a cache longer than the window
        (5, 7),  # chunked prefill, past + seq_len > window
    ],
)
def test_layer_forward_applies_window_when_cache_exceeds_it(seq_len, past):
    layer = make_fake_layer()
    h_full, k_full, v_full = run_layer(layer, seq_len, past, None)
    h_win, k_win, v_win = run_layer(layer, seq_len, past, WINDOW)
    assert torch.isfinite(h_win).all()
    assert not torch.allclose(h_full, h_win)
    # The window changes what the queries read, not what gets cached.
    assert torch.equal(k_full, k_win)
    assert torch.equal(v_full, v_win)


@pytest.mark.parametrize(
    "seq_len,past,window",
    [
        (3, 0, WINDOW),  # prefill shorter than the window
        (4, 0, WINDOW),  # prefill exactly the window
        (1, 3, WINDOW),  # decode with past + 1 == window
        (12, 0, 64),  # window wider than the whole cache
    ],
)
def test_layer_forward_window_is_noop_when_cache_fits(seq_len, past, window):
    layer = make_fake_layer()
    h_full, _, _ = run_layer(layer, seq_len, past, None)
    h_win, _, _ = run_layer(layer, seq_len, past, window)
    assert torch.allclose(h_full, h_win)


def test_shared_layer_forward_applies_window():
    """KV-shared layers (E2B/E4B) mask the borrowed KV by their own type."""
    layer = make_fake_layer()
    seq_len = 12
    # Source KV = what a non-shared layer of the same type just produced for
    # these tokens (full_seq_len includes the current tokens, past = 0).
    _, k_src, v_src = run_layer(layer, seq_len, 0, None)
    h_full = run_shared_layer(layer, seq_len, 0, k_src, v_src, None)
    h_win = run_shared_layer(layer, seq_len, 0, k_src, v_src, WINDOW)
    h_big = run_shared_layer(layer, seq_len, 0, k_src, v_src, 64)
    assert torch.isfinite(h_win).all()
    assert not torch.allclose(h_full, h_win)
    assert torch.allclose(h_full, h_big)


def test_shared_layer_forward_window_is_noop_when_cache_fits():
    layer = make_fake_layer()
    seq_len = 3
    _, k_src, v_src = run_layer(layer, seq_len, 0, None)
    h_full = run_shared_layer(layer, seq_len, 0, k_src, v_src, None)
    h_win = run_shared_layer(layer, seq_len, 0, k_src, v_src, WINDOW)
    assert torch.allclose(h_full, h_win)
