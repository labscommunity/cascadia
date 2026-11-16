"""export_gemma4.py --vision: stage port naming and the self-verify image path.

    python -m pytest tools/tests/test_gemma4_vision_ports.py -v

The traced stage's ports come out of ov.convert_model positionally;
name_stage_ports gives them the names the gemma4 engine binds, with the
image-splicing stage 0 taking a trailing `image_features` [M, hidden] whose
M stays dynamic (0 on every step without images). _verify_stage then feeds
that port zero rows on the prefill and the decode step, so a stage that
cannot take a step without images fails the export. Small synthetic IRs stand
in for the traced stages.
"""

from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")
ov = pytest.importorskip("openvino")

_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import export_gemma4  # noqa: E402
from openvino import opset13 as ops  # noqa: E402

H, KVH, HD, VOCAB, SEQ = 8, 1, 4, 16, 4


def _c(v, dt=np.int64):
    return ops.constant(np.array(v, dtype=dt))


def converted_stage(has_embed, splice_images, n_external, n_cross, n_own):
    """Ports as ov.convert_model leaves them: unnamed, at the trace shapes
    (seq 4, past 1, two image rows), in the wrapper's argument order."""
    main = (ops.parameter([1, SEQ], ov.Type.i64) if has_embed
            else ops.parameter([1, SEQ, H], ov.Type.f32))
    pos = ops.parameter([1, SEQ], ov.Type.i64)
    ext = [ops.parameter([1, KVH, 1, HD], ov.Type.f32) for _ in range(2 * n_external)]
    own = [ops.parameter([1, KVH, 1, HD], ov.Type.f32) for _ in range(2 * n_own)]
    params = [main, pos, *ext, *own]
    if has_embed:
        table = np.random.default_rng(0).standard_normal((VOCAB, H)).astype(np.float32)
        h = ops.gather(ops.constant(table), main, _c(0))
    else:
        h = main
    h = ops.add(h, ops.unsqueeze(ops.convert(pos, ov.Type.f32), _c(2)))
    if splice_images:
        feats = ops.parameter([2, H], ov.Type.f32)
        params.append(feats)
        h = ops.add(h, ops.reduce_sum(feats, _c([0]), True))
    results = [ops.result(h)]
    # cross_kv outputs re-emit the first own pairs; present the own KV. (An
    # op between port and Result keeps the two tensors' names apart.)
    results += [ops.result(ops.relu(own[k])) for k in range(2 * n_cross)]
    results += [ops.result(ops.relu(p)) for p in own]
    return ov.Model(results, params, "converted_stage")


@pytest.mark.parametrize(
    "has_embed,has_head,splice,n_external,n_cross,n_own",
    [
        (True, False, True, 0, 1, 2),  # image stage 0 of a multi-stage tree
        (True, True, True, 0, 0, 1),  # one-stage image tree
        (False, True, False, 1, 0, 2),  # later stage reading shared KV
    ],
)
def test_ports_are_named_and_freed(has_embed, has_head, splice, n_external, n_cross, n_own):
    m = converted_stage(has_embed, splice, n_external, n_cross, n_own)
    export_gemma4.name_stage_ports(m, has_embed, has_head, splice, n_external, n_cross)
    m.validate_nodes_and_infer_types()

    want_in = [("input_ids", [1, -1]) if has_embed else ("hidden_states", [1, -1, H]),
               ("position_ids", [1, -1])]
    for e in range(n_external):
        want_in += [(f"external_kv.{e}.{k}", [1, KVH, -1, HD]) for k in ("key", "value")]
    for o in range(n_own):
        want_in += [(f"past_key_values.{o}.{k}", [1, KVH, -1, HD]) for k in ("key", "value")]
    if splice:
        want_in.append(("image_features", [-1, H]))
    got_in = [(i.get_any_name(), [d.get_length() if d.is_static else -1
                                  for d in i.get_partial_shape()]) for i in m.inputs]
    assert got_in == want_in

    want_out = ["logits" if has_head else "hidden_states"]
    for c in range(n_cross):
        want_out += [f"cross_kv.{c}.key", f"cross_kv.{c}.value"]
    for o in range(n_own):
        want_out += [f"present.{o}.key", f"present.{o}.value"]
    assert [o.get_any_name() for o in m.outputs] == want_out


def image_stage(splice):
    """A named, compilable image stage 0: `splice` is splice_image_rows' OV
    shape (image rows plus an appended zero row, gathered by image slot);
    otherwise the stage averages `image_features` (sum over row count),
    which is 0/0 on a step without images."""
    ids = ops.parameter([1, -1], ov.Type.i64)
    pos = ops.parameter([1, -1], ov.Type.i64)
    feats = ops.parameter([-1, H], ov.Type.f32)
    table = np.random.default_rng(1).standard_normal((VOCAB, H)).astype(np.float32)
    h = ops.gather(ops.constant(table), ids, _c(0))
    is_img = ops.equal(ids, _c(VOCAB - 1))
    if splice:
        rows = ops.concat([feats, ops.constant(np.zeros((1, H), np.float32))], 0)
        slot = ops.maximum(ops.subtract(ops.cumsum(ops.convert(is_img, ov.Type.i64), _c(1)),
                                        _c(1)), _c(0))
        picked = ops.gather(rows, slot, _c(0))
        h = ops.select(ops.unsqueeze(is_img, _c(2)), picked, h)
    else:
        rows = ops.convert(ops.gather(ops.shape_of(feats), _c([0]), _c(0)), ov.Type.f32)
        h = ops.add(h, ops.divide(ops.reduce_sum(feats, _c([0]), True), rows))
    h = ops.add(h, ops.unsqueeze(ops.convert(pos, ov.Type.f32), _c(2)))
    m = ov.Model([ops.result(h)], [ids, pos, feats], "image_stage")
    export_gemma4.name_stage_ports(m, True, False, True, 0, 0)
    m.validate_nodes_and_infer_types()
    return m


def verify(model):
    export_gemma4._verify_stage(
        model, stage_idx=0, has_embed=True, num_kv_heads=KVH, pli_dim=0, hidden_dim=H,
        num_layers=1, downstream_pli_count=0, external_shared_sources=[], device="CPU",
        splice_images=True)


def test_verify_stage_feeds_zero_image_rows():
    verify(image_stage(splice=True))


def test_verify_stage_rejects_a_stage_that_needs_image_rows():
    # Zero image rows average to NaN: the step without images is broken.
    with pytest.raises(RuntimeError, match="non-finite prefill"):
        verify(image_stage(splice=False))
