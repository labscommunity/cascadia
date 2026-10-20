"""Unit tests for Gemma 4 sliding-window masking in `export_gemma4`.

Run from the repo root with:

    python -m pytest tools/tests/test_gemma4_sliding_window.py -v

Gemma 4 interleaves runs of `sliding_attention` layers with a `full_attention`
layer (5:1 on E4B/31B, 4:1 on E2B). The exporter originally built a plain causal
mask for every layer, so a sliding layer could attend to the entire prefix. That
agrees with HF while the sequence is shorter than the window (1024 for 31B-it),
which is why short prompts look correct; past the window the exported graph and
HF diverge and output degrades. These tests pin the window semantics so that
cannot regress silently.

Five layers of coverage:

* `build_attention_mask` -- the mask is the whole contract, so membership is
  checked directly rather than against a golden tensor: key `j` is visible to
  query `i` exactly when `0 <= (past + i) - j < sliding_window`. A window
  `<= 0` must raise (it would mask every key and softmax would return NaN).
* `resolve_sliding_window` -- how `text_config.sliding_window` becomes the
  baked value: 0/absent disables, negative is a configuration error.
* `cached_gemma4_layer_forward` / `cached_gemma4_shared_layer_forward` --
  driven with a tiny fake decoder layer, including an exact oracle (a
  windowed step equals a full-attention step over the truncated cache), so
  a forward that stops passing the window, re-inlines a plain causal mask,
  or widens the band by one fails here rather than only in a long-context
  eval.
* `build_cached_wrapper` -- the production call site, where per-layer windows
  and the per-(window, cache source) mask cache live: a fake model checked
  against a manual loop over the layer forwards (single stage, and a KV-shared
  second stage borrowing an external cache), plus a tiny random-init HF
  `Gemma4ForCausalLM` compared with HF's own logits past the window (that one
  skips when transformers lacks Gemma 4).
* `run_export` / `export_single_stage` / `_verify_stage` -- the config-first
  guards, the window reaching every stage and the tree metadata, and the
  self-check that must cross the window, reject a non-finite output and carry
  that failure out of the stage export instead of logging it.
"""

from __future__ import annotations

import json
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
        (1, 1, 4),  # first token, empty cache: the band must not clip it
        (1, 1, None),  # same, unwindowed
    ],
)
def test_mask_matches_window_rule(seq_len, full_seq_len, window):
    mask = export_gemma4.build_attention_mask(
        seq_len, full_seq_len, window, CPU, torch.float32
    )
    assert mask.shape == (seq_len, full_seq_len)
    assert visible(mask) == expected(seq_len, full_seq_len, window)


@pytest.mark.parametrize(
    "seq_len,full_seq_len,window",
    [(8, 8, 4), (1, 9, 4), (5, 12, 3), (4, 8, 8), (3, 10, 1)],
)
def test_mask_matches_hf_mask_function(seq_len, full_seq_len, window):
    """The hand-written oracle and the exporter could share one misreading of
    HF's boundary; pin both to transformers' own mask function."""
    mu = pytest.importorskip("transformers.masking_utils")
    fn = mu.sliding_window_causal_mask_function(window)
    past = full_seq_len - seq_len
    hf = {
        (i, j)
        for i in range(seq_len)
        for j in range(full_seq_len)
        if bool(fn(0, 0, torch.tensor(past + i), torch.tensor(j)))
    }
    mask = export_gemma4.build_attention_mask(
        seq_len, full_seq_len, window, CPU, torch.float32
    )
    assert visible(mask) == hf


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


@pytest.mark.parametrize("raw", [True, 0.5, "1024"])
def test_resolve_sliding_window_rejects_non_integers(raw):
    """`int(True)` is 1 and `int(0.5)` is 0: refuse instead of coercing."""
    cfg = types.SimpleNamespace(sliding_window=raw)
    with pytest.raises(ValueError, match="integer"):
        export_gemma4.resolve_sliding_window(cfg)


@pytest.mark.parametrize(
    "raw", [[512, 1024], float("nan"), float("inf"), "wide"]
)
def test_resolve_sliding_window_rejects_unconvertible_values(raw):
    """`int()` raises for a per-layer list (TypeError), nan (ValueError
    "cannot convert float NaN") and inf (OverflowError); those must surface
    as the prefixed error naming the config key, not as a bare conversion
    error the operator has to trace back to `sliding_window`."""
    cfg = types.SimpleNamespace(sliding_window=raw)
    with pytest.raises(ValueError, match="must be an integer"):
        export_gemma4.resolve_sliding_window(cfg)


def test_resolve_sliding_window_propagates_unrelated_read_errors():
    """Only transformers' per-layer-override error is relabelled; anything
    else the attribute read raises keeps its own message."""

    class Cfg:
        @property
        def sliding_window(self):
            raise RuntimeError("remote config backend unreachable")

    with pytest.raises(RuntimeError, match="remote config backend"):
        export_gemma4.resolve_sliding_window(Cfg())


SF = ["sliding_attention", "sliding_attention", "full_attention"]


@pytest.mark.parametrize("raw", [0, None])
def test_resolve_sliding_window_refuses_windowless_sliding_layers(raw):
    """A config with sliding layers but no window is the pre-v1.1 bug; the
    exporter must not stamp such a tree v1.1."""
    cfg = types.SimpleNamespace(sliding_window=raw)
    with pytest.raises(ValueError, match="sliding_attention layers but no"):
        export_gemma4.resolve_sliding_window(cfg, SF)
    with pytest.raises(ValueError, match="sliding_attention layers but no"):
        export_gemma4.resolve_sliding_window(types.SimpleNamespace(), SF)


def test_resolve_sliding_window_allows_windowless_full_only_models():
    cfg = types.SimpleNamespace(sliding_window=None)
    assert export_gemma4.resolve_sliding_window(cfg, ["full_attention"]) is None
    assert export_gemma4.resolve_sliding_window(
        types.SimpleNamespace(sliding_window=1024), SF
    ) == 1024


# ---------------------------------------------------------------------------
# Layer forwards (the production call sites)
# ---------------------------------------------------------------------------

HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 16, 4, 2, 8
WINDOW = 4
# Explicit so these tests never depend on the process-wide default dtype, and
# because the exporter's rotary always emits float32 tables.
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


def run_layer_with_cache(layer, seq_len, past, past_k, past_v, sliding_window):
    """Like `run_layer` (same seeded hidden states + rope for `past`), but with
    an explicit cache — so a truncated cache can stand in for a window."""
    x, cos, sin, _, _ = step_inputs(seq_len, past)
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


@pytest.mark.parametrize("window", [1, 2, 4])
def test_layer_forward_window_equals_full_attention_over_truncated_cache(window):
    """Exact oracle: a decode step with window W over a long cache must equal a
    full-attention step over only the last W-1 cached keys (+ the new token).
    "Some masking happened" tests pass a band that is one key too wide; this
    does not."""
    layer = make_fake_layer()
    past = 10
    x, cos, sin, past_k, past_v = step_inputs(1, past)
    h_win, _, _ = run_layer_with_cache(layer, 1, past, past_k, past_v, window)
    keep = window - 1
    trunc_k = past_k[:, :, past - keep :, :]
    trunc_v = past_v[:, :, past - keep :, :]
    with torch.no_grad():
        h_ref, _, _ = export_gemma4.cached_gemma4_layer_forward(
            layer, x, cos, sin, trunc_k, trunc_v, NUM_HEADS, NUM_KV_HEADS,
            HEAD_DIM, None, sliding_window=None,
        )
    assert torch.allclose(h_win, h_ref, atol=1e-6)
    # Teeth: one key wider is a different answer.
    h_wide, _, _ = run_layer_with_cache(layer, 1, past, past_k, past_v, window + 1)
    assert not torch.allclose(h_wide, h_ref, atol=1e-6)


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


def test_shared_layer_forward_decode_against_long_source_cache():
    """E2B/E4B decode: the borrowed source cache is `past + seq_len` long, so
    the shared forward must derive `past` from the source length, not from
    the query block. Exact oracle as for the own-KV layer."""
    layer = make_fake_layer()
    past, window = 10, WINDOW
    _, k_src, v_src = run_layer(layer, 1, past, None)  # length past + 1
    assert k_src.shape[2] == past + 1
    h_win = run_shared_layer(layer, 1, past, k_src, v_src, window)
    keep = window  # last W-1 cached keys + the current token
    h_ref = run_shared_layer(
        layer, 1, past, k_src[:, :, -keep:, :], v_src[:, :, -keep:, :], None
    )
    assert torch.allclose(h_win, h_ref, atol=1e-6)
    h_full = run_shared_layer(layer, 1, past, k_src, v_src, None)
    assert not torch.allclose(h_win, h_full)


# ---------------------------------------------------------------------------
# build_cached_wrapper vs HF (the production call site, end to end)
# ---------------------------------------------------------------------------

SLIDING, FULL = "sliding_attention", "full_attention"


def _tiny_gemma4():
    """Random-init Gemma 4 small enough to run on CPU in a second, with a
    4-token window so a 12-token prefill crosses it on every sliding layer."""
    cfg_mod = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")
    mdl_mod = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    layer_types = [SLIDING, SLIDING, FULL, SLIDING, SLIDING, FULL, SLIDING, FULL]
    # Only the import is optional: a transformers that HAS Gemma 4 but cannot
    # build this config must fail, not skip -- a swallowed TypeError here
    # would retire the one test that checks the wrapper against HF.
    cfg = cfg_mod.Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=2,
        layer_types=layer_types,
        sliding_window=4,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=64,
        final_logit_softcapping=None,
        tie_word_embeddings=False,
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
    )
    cfg._attn_implementation = "eager"
    # transformers >= 5.14 registers head_dim/num_key_value_heads as
    # per-layer attributes and refuses global reads; this config is
    # homogeneous, so the exporter's global reads are exact.
    if hasattr(cfg, "allow_global_per_layer_attribute_access"):
        cfg.allow_global_per_layer_attribute_access = True
    torch.manual_seed(0)
    model = mdl_mod.Gemma4ForCausalLM(cfg).eval()
    export_gemma4.fix_zero_dim_buffers(model)
    return cfg, model


def _wrapper_logits(cfg, model, ids, sliding_window):
    plan = export_gemma4.compute_stage_plan(cfg.num_hidden_layers, 1)[0]
    wrapper, head_dims, nkv, share = export_gemma4.build_cached_wrapper(
        model, cfg, plan, sliding_window=sliding_window
    )
    kv = []
    for i, shared in enumerate(share["is_shared"]):
        if not shared:
            kv.append(torch.zeros(1, nkv[i], 0, head_dims[i]))
            kv.append(torch.zeros(1, nkv[i], 0, head_dims[i]))
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    with torch.no_grad():
        return wrapper(ids, pos, *kv)[0]


def test_wrapper_matches_hf_past_the_window():
    """The whole stage wrapper (layer-type gating, mask hoisting, both call
    sites) against HF's own forward, on a prompt three times the window."""
    cfg, model = _tiny_gemma4()
    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    ours = _wrapper_logits(cfg, model, ids, sliding_window=cfg.sliding_window)
    assert torch.allclose(ours, ref, atol=1e-4), (ours - ref).abs().max()
    # Teeth: exporting the sliding layers unbounded (the v1 behaviour) differs.
    unbounded = _wrapper_logits(cfg, model, ids, sliding_window=None)
    assert not torch.allclose(unbounded, ref, atol=1e-4)


# ---------------------------------------------------------------------------
# build_cached_wrapper with a fake model (no transformers needed)
# ---------------------------------------------------------------------------

VOCAB = 32


class FakeLayer(torch.nn.Module):
    """`make_fake_layer`'s attribute surface as a real `nn.Module`, which is
    what `build_cached_wrapper` needs to put stage layers in an `nn.ModuleList`.
    """

    def __init__(self, seed=0):
        super().__init__()
        for name, value in vars(make_fake_layer(seed)).items():
            setattr(self, name, value)


def fake_model(layer_types, num_kv_shared_layers=0):
    """The surface `build_cached_wrapper` reads: a text backbone at
    `model.model` (`layers` / `embed_tokens` / `norm`) plus `model.lm_head`,
    and the `text_config` attributes it looks up. Returns (model, config).

    Both head_dims and both KV-head counts are equal so one `FakeLayer` shape
    serves `sliding_attention` and `full_attention` alike, and both rope
    thetas are equal so a single rotary table reproduces the wrapper's.
    """
    nn = torch.nn
    layers = [FakeLayer(seed=i) for i in range(len(layer_types))]
    tm = types.SimpleNamespace(
        layers=layers,
        embed_tokens=nn.Embedding(VOCAB, HIDDEN, dtype=DTYPE),
        norm=nn.Identity(),
    )
    model = types.SimpleNamespace(
        model=tm, lm_head=nn.Linear(HIDDEN, VOCAB, bias=False, dtype=DTYPE)
    )
    cfg = types.SimpleNamespace(
        num_hidden_layers=len(layer_types),
        hidden_size=HIDDEN,
        hidden_size_per_layer_input=0,
        layer_types=list(layer_types),
        head_dim=HEAD_DIM,
        global_head_dim=HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        num_global_key_value_heads=NUM_KV_HEADS,
        num_kv_shared_layers=num_kv_shared_layers,
        rope_theta=10000.0,
        rope_local_base_freq=10000.0,
        final_logit_softcapping=0.0,
        vocab_size=VOCAB,
    )
    return model, cfg


def empty_kv():
    return torch.zeros(1, NUM_KV_HEADS, 0, HEAD_DIM, dtype=DTYPE)


def run_stage(model, cfg, plan, sliding_window, main_input, pos, ext_kv=()):
    """Drive one stage through the production wrapper with an empty own-KV
    cache. Returns the wrapper's output tuple."""
    wrapper, _, _, share = export_gemma4.build_cached_wrapper(
        model, cfg, plan, sliding_window=sliding_window
    )
    past = []
    for shared in share["is_shared"]:
        if not shared:
            past.extend([empty_kv(), empty_kv()])
    with torch.no_grad():
        return wrapper(main_input, pos, *ext_kv, *past)


def manual_stage_logits(model, ids, windows):
    """The same single-stage forward written out by hand: embed, one
    `cached_gemma4_layer_forward` per layer with the window this test dictates,
    norm, head."""
    layers = model.model.layers
    cos, sin = rope_tables(ids.shape[1], 0)
    h = model.model.embed_tokens(ids)
    with torch.no_grad():
        for layer, window in zip(layers, windows):
            h, _, _ = export_gemma4.cached_gemma4_layer_forward(
                layer, h, cos, sin, empty_kv(), empty_kv(),
                NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, None,
                sliding_window=window,
            )
        return model.lm_head(model.model.norm(h))


def test_wrapper_windows_sliding_layers_only():
    """`_layer_windows` + `mask_for`: the wrapper must bake the window into
    the `sliding_attention` layers and nothing else."""
    layer_types = [SLIDING, SLIDING, FULL, SLIDING, SLIDING, FULL]
    model, cfg = fake_model(layer_types)
    plan = export_gemma4.compute_stage_plan(len(layer_types), 1)[0]
    torch.manual_seed(3)
    ids = torch.randint(0, VOCAB, (1, 12))  # three times the window
    pos = torch.arange(ids.shape[1]).unsqueeze(0)

    ours = run_stage(model, cfg, plan, WINDOW, ids, pos)[0]
    by_type = manual_stage_logits(
        model, ids, [WINDOW if lt == SLIDING else None for lt in layer_types]
    )
    assert torch.allclose(ours, by_type, atol=1e-6)

    # Teeth on both sides of the gate: windowing every layer, or none of
    # them (the v1 behaviour), are different answers.
    every = manual_stage_logits(model, ids, [WINDOW] * len(layer_types))
    none = manual_stage_logits(model, ids, [None] * len(layer_types))
    assert not torch.allclose(ours, every, atol=1e-6)
    assert not torch.allclose(ours, none, atol=1e-6)
    # And sliding_window=None reproduces the unwindowed export exactly.
    unbounded = run_stage(model, cfg, plan, None, ids, pos)[0]
    assert torch.allclose(unbounded, none, atol=1e-6)


def test_wrapper_windows_a_shared_layer_borrowing_external_kv():
    """Second stage of a 2-stage plan with KV sharing: one own-KV layer, one
    sliding layer borrowing KV from the *previous stage* (the "ext" mask key,
    whose length comes from the source cache, not from the own cache), and one
    full layer borrowing from inside the stage."""
    layer_types = [SLIDING, SLIDING, FULL, FULL, SLIDING, FULL]
    model, cfg = fake_model(layer_types, num_kv_shared_layers=2)
    plan = export_gemma4.compute_stage_plan(len(layer_types), 2)[1]
    share = export_gemma4.resolve_kv_sharing(cfg, plan)
    assert share["is_shared"] == [False, True, True]
    assert share["source_local"][1] < 0  # layer 4 borrows across the stage
    assert share["source_local"][2] == 0  # layer 5 borrows layer 3
    assert len(share["external_shared_sources"]) == 1

    seq_len, past = 3, 7
    torch.manual_seed(5)
    h_in = torch.randn(1, seq_len, HIDDEN, dtype=DTYPE)
    # The borrowed cache is the full past + seq the earlier stage produced,
    # while this stage's own KV state is still empty.
    src_k = torch.randn(1, NUM_KV_HEADS, past + seq_len, HEAD_DIM, dtype=DTYPE)
    src_v = torch.randn(1, NUM_KV_HEADS, past + seq_len, HEAD_DIM, dtype=DTYPE)
    pos = torch.arange(past, past + seq_len).unsqueeze(0)

    ours = run_stage(
        model, cfg, plan, WINDOW, h_in, pos, ext_kv=(src_k, src_v)
    )[0]

    cos, sin = rope_tables(seq_len, past)
    layers = model.model.layers
    with torch.no_grad():
        h, own_k, own_v = export_gemma4.cached_gemma4_layer_forward(
            layers[3], h_in, cos, sin, empty_kv(), empty_kv(),
            NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, None, sliding_window=None,
        )
        h = export_gemma4.cached_gemma4_shared_layer_forward(
            layers[4], h, cos, sin, src_k, src_v,
            NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, None, sliding_window=WINDOW,
        )
        h = export_gemma4.cached_gemma4_shared_layer_forward(
            layers[5], h, cos, sin, own_k, own_v,
            NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, None, sliding_window=None,
        )
        ref = model.lm_head(model.model.norm(h))
    assert torch.allclose(ours, ref, atol=1e-6)

    # Teeth: the borrowed cache is longer than the window, so leaving the
    # shared sliding layer unbounded is a different answer. This call also
    # pins the "own"/"ext" mask keys apart -- with one entry per window, the
    # external layer would be handed the own layers' 3-key mask and the add
    # would not even broadcast.
    unbounded = run_stage(
        model, cfg, plan, None, h_in, pos, ext_kv=(src_k, src_v)
    )[0]
    assert not torch.allclose(ours, unbounded, atol=1e-6)


# ---------------------------------------------------------------------------
# run_export -- the config-first guards and the metadata contract
# ---------------------------------------------------------------------------


def fake_text_config(**overrides):
    """Just the attributes `run_export` reads off `text_config`, so a guard
    can be driven without a checkpoint or a multi-minute model load."""
    fields = dict(
        layer_types=[SLIDING, SLIDING, FULL, SLIDING],
        num_hidden_layers=4,
        sliding_window=1024,
        enable_moe_block=False,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def patch_autoconfig(monkeypatch, text_config):
    """`run_export` does `from transformers import AutoConfig` at call time,
    so replacing the attribute on the module is enough.

    `apply_patches` runs ahead of the guards and replaces openvino's
    `torch_tensor_to_ov_const` for the rest of the pytest process; the fake
    config has no 0-dim buffers to patch, so stub it out rather than mutate
    an imported module out from under every later test.
    """
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(export_gemma4, "apply_patches", lambda: None)
    full = types.SimpleNamespace(
        text_config=text_config, to_json_file=lambda path: None
    )
    monkeypatch.setattr(
        transformers,
        "AutoConfig",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: full),
    )


def patch_run_export(monkeypatch, text_config):
    """Stub every heavy step past the guards -- no checkpoint, no tokenizer,
    no stage export -- so `run_export`'s own logic is what runs. Returns the
    list the fake `export_single_stage` records its kwargs into."""
    transformers = pytest.importorskip("transformers")
    patch_autoconfig(monkeypatch, text_config)
    model = types.SimpleNamespace(eval=lambda: None, named_buffers=lambda: [])
    monkeypatch.setattr(
        transformers,
        "AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: model),
    )
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        types.SimpleNamespace(
            from_pretrained=lambda *a, **k: types.SimpleNamespace(
                save_pretrained=lambda d: None
            )
        ),
    )
    calls = []

    def record_stage(*args, **kwargs):
        calls.append(kwargs)
        return 0.0

    monkeypatch.setattr(export_gemma4, "export_single_stage", record_stage)
    return calls


@pytest.mark.parametrize(
    "layer_types",
    [
        [SLIDING, FULL],  # short: used to IndexError inside the first stage
        [SLIDING, SLIDING, FULL, SLIDING, FULL, FULL],  # long: tail ignored
    ],
)
def test_run_export_rejects_layer_types_length_mismatch(
    layer_types, monkeypatch, tmp_path
):
    """`num_hidden_layers` and `layer_types` must agree before the load: the
    stage slices index by layer, so a short list failed minutes in and a long
    one exported the wrong types without a word."""
    patch_autoconfig(monkeypatch, fake_text_config(layer_types=layer_types))
    with pytest.raises(RuntimeError, match="layer_types has"):
        export_gemma4.run_export(
            model_id="fake", output_dir=str(tmp_path / "out"), num_stages=1
        )


def write_pipeline_config(out_dir, export_version):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pipeline_config.json").write_text(
        json.dumps({"export_version": export_version})
    )


def test_run_export_refuses_stage_reexport_of_another_version(
    monkeypatch, tmp_path
):
    """`--stage N` rewrites pipeline_config.json but only that one stage, so
    re-exporting into a v1 tree would stamp the root v1.1 while the stages
    nobody touched keep attending past the window."""
    out = tmp_path / "tree"
    write_pipeline_config(out, "gemma4_cached_v1")
    calls = patch_run_export(monkeypatch, fake_text_config())
    with pytest.raises(RuntimeError, match="re-export every stage"):
        export_gemma4.run_export(
            model_id="fake", output_dir=str(out), num_stages=2, stage=1
        )
    assert calls == []  # refused before any stage ran


@pytest.mark.parametrize("existing", [None, export_gemma4.EXPORT_VERSION])
def test_run_export_allows_stage_reexport_of_a_matching_tree(
    existing, monkeypatch, tmp_path
):
    out = tmp_path / "tree"
    out.mkdir()
    if existing is not None:
        write_pipeline_config(out, existing)
    calls = patch_run_export(monkeypatch, fake_text_config())
    export_gemma4.run_export(
        model_id="fake", output_dir=str(out), num_stages=2, stage=1
    )
    assert len(calls) == 1


def test_run_export_refuses_stage_reexport_into_another_split(monkeypatch, tmp_path):
    """Same version, other stage count: stage_0 of a 4-stage tree holds
    layer 0 only, so a 2-stage stage 1 (layers 2..3) beside it would leave
    layer 1 out of the tree."""
    out = tmp_path / "tree"
    write_pipeline_config(out, export_gemma4.EXPORT_VERSION)
    (out / "stage_0").mkdir()
    (out / "stage_0" / "stage_config.json").write_text(json.dumps(
        {"layer_start": 0, "layer_end": 1, "export_version": export_gemma4.EXPORT_VERSION}))
    calls = patch_run_export(monkeypatch, fake_text_config())
    with pytest.raises(SystemExit, match="different cut"):
        export_gemma4.run_export(
            model_id="fake", output_dir=str(out), num_stages=2, stage=1
        )
    assert calls == []


def test_run_export_full_export_overwrites_an_older_tree(monkeypatch, tmp_path):
    """Only `--stage` is refused: a full export rewrites every stage, so it
    may replace a v1 tree, and leaves it uniformly v1.1."""
    out = tmp_path / "tree"
    write_pipeline_config(out, "gemma4_cached_v1")
    calls = patch_run_export(monkeypatch, fake_text_config())
    export_gemma4.run_export(model_id="fake", output_dir=str(out), num_stages=2)
    assert len(calls) == 2
    written = json.loads((out / "pipeline_config.json").read_text())
    assert written["export_version"] == export_gemma4.EXPORT_VERSION


def test_run_export_records_the_window_and_hands_it_to_every_stage(
    monkeypatch, tmp_path
):
    """The `run_export -> export_single_stage -> build_cached_wrapper` kwarg
    chain and the tree metadata: the resolved window must reach every stage
    and be readable off the tree, under the version that promises it."""
    out = tmp_path / "tree"
    calls = patch_run_export(monkeypatch, fake_text_config())
    export_gemma4.run_export(model_id="fake", output_dir=str(out), num_stages=2)
    meta = json.loads((out / "pipeline_config.json").read_text())
    assert meta["sliding_window"] == 1024
    assert meta["export_version"] == "gemma4_cached_v1.1"
    assert [c["sliding_window"] for c in calls] == [1024, 1024]


@pytest.mark.parametrize("raw", [0, None])
def test_run_export_refuses_a_windowless_config_with_sliding_layers(
    raw, monkeypatch, tmp_path
):
    """Through `run_export`, not just the resolver: such a tree would be the
    v1 behaviour wearing the v1.1 stamp."""
    patch_autoconfig(monkeypatch, fake_text_config(sliding_window=raw))
    with pytest.raises(ValueError, match="sliding_attention layers but no"):
        export_gemma4.run_export(
            model_id="fake", output_dir=str(tmp_path / "out"), num_stages=1
        )


def test_run_export_rejects_unknown_layer_types(monkeypatch, tmp_path):
    layer_types = [SLIDING, "linear_attention", FULL, SLIDING]
    patch_autoconfig(monkeypatch, fake_text_config(layer_types=layer_types))
    with pytest.raises(RuntimeError, match="are not supported"):
        export_gemma4.run_export(
            model_id="fake", output_dir=str(tmp_path / "out"), num_stages=1
        )


@pytest.mark.parametrize(
    "overrides",
    [{"use_bidirectional_attention": "all"}, {"is_causal": False}],
)
def test_run_export_rejects_bidirectional_variants(
    overrides, monkeypatch, tmp_path
):
    patch_autoconfig(monkeypatch, fake_text_config(**overrides))
    with pytest.raises(RuntimeError, match="bidirectional"):
        export_gemma4.run_export(
            model_id="fake", output_dir=str(tmp_path / "out"), num_stages=1
        )


# ---------------------------------------------------------------------------
# _verify_stage -- the self-check that must cross the window and fail on NaN
# ---------------------------------------------------------------------------

np = pytest.importorskip("numpy")
OUT_PORT = "output0"


class FakeInferRequest:
    """Records what `_verify_stage` asks the compiled model to run. No state,
    so `query_state()` is empty and the zero-init loop is skipped."""

    def __init__(self, output_for):
        self.output_for = output_for
        self.inputs = []

    def reset_state(self):
        pass

    def query_state(self):
        return []

    def infer(self, inputs):
        self.inputs.append(inputs)
        return {OUT_PORT: self.output_for(inputs)}


def patch_ov_core(monkeypatch, request):
    """`_verify_stage` does `import openvino as ov` and then `ov.Core()`, so
    replacing the attribute on the module intercepts the whole compile path."""
    ov = pytest.importorskip("openvino")
    compiled = types.SimpleNamespace(
        create_infer_request=lambda: request, output=lambda i: OUT_PORT
    )
    core = types.SimpleNamespace(compile_model=lambda model, device: compiled)
    monkeypatch.setattr(ov, "Core", lambda: core)


def finite_output(inputs):
    return np.zeros((1, inputs[0].shape[1], 4), dtype=np.float32)


def verify(request, monkeypatch, sliding_window):
    patch_ov_core(monkeypatch, request)
    export_gemma4._verify_stage(
        ov_model=object(),
        stage_idx=0,
        has_embed=True,
        num_kv_heads=NUM_KV_HEADS,
        pli_dim=0,
        hidden_dim=HIDDEN,
        num_layers=2,
        downstream_pli_count=0,
        external_shared_sources=[],
        device="CPU",
        sliding_window=sliding_window,
    )


def test_verify_stage_prefills_past_the_window(monkeypatch):
    """The prefill must be `sliding_window + 4` tokens with contiguous
    positions, then decode the next one: a 3-token prefill never reaches the
    exported band, so a broken window would verify clean."""
    request = FakeInferRequest(finite_output)
    verify(request, monkeypatch, sliding_window=8)
    prefill, decode = request.inputs
    assert prefill[0].shape == (1, 12)
    assert prefill[1].tolist() == [list(range(12))]
    assert decode[0].shape == (1, 1)
    assert decode[1].tolist() == [[12]]


def test_verify_stage_rejects_a_non_finite_prefill(monkeypatch):
    """A window that masks whole rows surfaces as NaN, which must abort."""
    request = FakeInferRequest(lambda inputs: finite_output(inputs) * np.nan)
    with pytest.raises(RuntimeError, match="non-finite prefill"):
        verify(request, monkeypatch, sliding_window=8)


# ---------------------------------------------------------------------------
# export_single_stage -- a failed self-verify must abort the export
# ---------------------------------------------------------------------------


def export_one_stage(tmp_path, device_verify):
    """Run the real exporter over the fake model -- trace, convert, make
    stateful, save -- as a single stage carrying both embed and head."""
    pytest.importorskip("openvino")
    layer_types = [SLIDING, FULL]
    model, cfg = fake_model(layer_types)
    for layer in model.model.layers:
        # The projections hang off a SimpleNamespace, so they are not
        # registered parameters and torch.jit.trace has to fold them in as
        # constants, which it refuses to do while they require grad.
        for module in vars(layer.self_attn).values():
            module.requires_grad_(False)
    export_gemma4.export_single_stage(
        model,
        cfg,
        export_gemma4.compute_stage_plan(len(layer_types), 1)[0],
        str(tmp_path),
        "fp32",
        device_verify=device_verify,
        sliding_window=WINDOW,
    )


def test_export_single_stage_propagates_a_failed_self_verify(
    monkeypatch, tmp_path
):
    """The self-verify is all that stands between a mis-built mask and a
    v1.1-stamped NaN shard, so its failure must abort the export rather than
    log a line and return the stage's size."""
    def raiser(*args, **kwargs):
        raise RuntimeError("non-finite prefill")

    monkeypatch.setattr(export_gemma4, "_verify_stage", raiser)
    with pytest.raises(RuntimeError, match="non-finite prefill"):
        export_one_stage(tmp_path, device_verify="CPU")


def test_export_single_stage_without_a_device_skips_the_self_verify(
    monkeypatch, tmp_path
):
    """`device_verify=None` is the documented opt-out; the stage is still
    written."""
    calls = []
    monkeypatch.setattr(
        export_gemma4, "_verify_stage", lambda *a, **k: calls.append(k)
    )
    export_one_stage(tmp_path, device_verify=None)
    assert calls == []
    meta = json.loads((tmp_path / "stage_0" / "stage_config.json").read_text())
    assert meta["sliding_window"] == WINDOW
