"""Unit tests for tools/vlm_surgery/static_variant.py on a synthetic stage.

Run from the repo root with:

    python -m pytest tools/tests/test_vlm_static_variant.py -v

The stage is a two-layer toy with the cut-stage KV shape (ReadValue ->
Gather(beam_idx) -> Concat -> Assign per key and value). The test converts it
and checks the static contract ov-runtime's ring relies on: port names and
order, f16 KV, static shapes, and that one stateless step with the stateful
model's past fed explicitly gives the same output and present rows.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
ov = pytest.importorskip("openvino")

_SURGERY = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "vlm_surgery"))
if _SURGERY not in sys.path:
    sys.path.insert(0, _SURGERY)

import static_variant as sv  # noqa: E402
from openvino import opset13 as ops  # noqa: E402

HIDDEN, KVH, HD, LAYERS = 8, 2, 4, 2


def toy_stage(seed: int = 0, head_dims=(HD, HD)) -> ov.Model:
    rng = np.random.default_rng(seed)
    x = ops.parameter([-1, -1, HIDDEN], ov.Type.f32, name="inputs_embeds")
    x.output(0).set_names({"inputs_embeds"})
    mask = ops.parameter([-1, -1], ov.Type.i64, name="attention_mask")
    mask.output(0).set_names({"attention_mask"})
    pos = ops.parameter([-1, -1], ov.Type.i64, name="position_ids")
    pos.output(0).set_names({"position_ids"})
    beam = ops.parameter([-1], ov.Type.i32, name="beam_idx")
    beam.output(0).set_names({"beam_idx"})
    h = x.output(0)
    sinks = []
    # Variables numbered by global layer, deliberately out of order.
    for layer, hd in zip((21, 20), head_dims):
        outs = {}
        for kind in ("key", "value"):
            w = ops.constant(rng.standard_normal((HIDDEN, KVH * hd)).astype(np.float32))
            new = ops.matmul(h, w, False, False)
            new = ops.reshape(new, ops.constant(np.array([0, 0, KVH, hd], np.int64)), True)
            new = ops.transpose(new, ops.constant(np.array([0, 2, 1, 3], np.int64)))
            vid = f"past_key_values.{layer}.{kind}present.{layer}.{kind}"
            info = ov.op.util.VariableInfo()
            info.variable_id = vid
            info.data_shape = ov.PartialShape([-1, KVH, -1, hd])
            info.data_type = ov.Type.f32
            var = ov.op.util.Variable(info)
            init = ops.broadcast(
                ops.constant(np.float32(0)),
                ops.constant(np.array([1, KVH, 0, hd], np.int64)))
            rv = ops.read_value(init, var)
            past = ops.gather(rv, beam, ops.constant(np.int64(0)))
            cat = ops.concat([past, new], 2)
            sinks.append(ops.assign(cat, var))
            outs[kind] = cat
        # Toy attention stand-in: mix the summed cache back into the stream.
        k = ops.reduce_sum(outs["key"], ops.constant(np.array([1, 2, 3], np.int64)), True)
        v = ops.reduce_sum(outs["value"], ops.constant(np.array([1, 2, 3], np.int64)), True)
        kv = ops.reshape(ops.add(k, v), ops.constant(np.array([-1, 1, 1], np.int64)), False)
        m = ops.convert(ops.reduce_sum(mask, ops.constant(np.array([1], np.int64)), True),
                        ov.Type.f32)
        p = ops.convert(ops.reduce_sum(pos, ops.constant(np.array([1], np.int64)), True),
                        ov.Type.f32)
        bias = ops.reshape(ops.add(m, p), ops.constant(np.array([-1, 1, 1], np.int64)), False)
        h = ops.add(ops.add(h, ops.multiply(kv, ops.constant(np.float32(0.01)))), bias).output(0)
    out = ops.result(h)
    return ov.Model([out], sinks, [x, mask, pos, beam], "toy_stage")


def test_ports_and_shapes():
    m = toy_stage()
    geom = sv.make_stateless(m, has_head=False)
    assert geom == [(KVH, HD)] * LAYERS
    assert not m.get_variables() and not m.get_sinks()
    names = [i.get_any_name() for i in m.inputs]
    assert "beam_idx" not in names
    assert [n for n in names if n.startswith("past_key_values.")] == [
        "past_key_values.0.key", "past_key_values.0.value",
        "past_key_values.1.key", "past_key_values.1.value"]
    outs = [o.get_any_name() for o in m.outputs]
    assert outs == ["hidden_states", "present.0.key", "present.0.value",
                    "present.1.key", "present.1.value"]
    for o in m.outputs[1:]:
        assert o.get_element_type() == ov.Type.f16
    sv.pin_shapes(m, 1, 16)
    shapes = {i.get_any_name(): list(i.get_shape()) for i in m.inputs}
    assert shapes["inputs_embeds"] == [1, 1, HIDDEN]
    assert shapes["attention_mask"] == [1, 16]
    assert shapes["position_ids"] == [1, 1]
    assert shapes["past_key_values.0.key"] == [1, KVH, 15, HD]
    assert list(m.outputs[1].get_shape()) == [1, KVH, 16, HD]


def test_stateless_step_matches_stateful():
    core = ov.Core()
    rng = np.random.default_rng(1)
    prompt = rng.standard_normal((1, 3, HIDDEN)).astype(np.float32)
    nxt = rng.standard_normal((1, 1, HIDDEN)).astype(np.float32)

    stateful = core.compile_model(toy_stage(), "CPU").create_infer_request()
    beam = np.array([0], np.int32)
    stateful.infer({"inputs_embeds": prompt, "attention_mask": np.ones((1, 3), np.int64),
                    "position_ids": np.arange(3, dtype=np.int64)[None], "beam_idx": beam})
    states = {s.name: s.state.data.copy() for s in stateful.query_state()}
    want = stateful.infer({"inputs_embeds": nxt, "attention_mask": np.ones((1, 4), np.int64),
                           "position_ids": np.array([[3]], np.int64), "beam_idx": beam})
    want_h = want[stateful.model_outputs[0]]

    m = toy_stage()
    sv.make_stateless(m, has_head=False)
    req = core.compile_model(m, "CPU").create_infer_request()
    feed = {"inputs_embeds": nxt, "attention_mask": np.ones((1, 4), np.int64),
            "position_ids": np.array([[3]], np.int64)}
    # Global layers 20, 21 are local 0, 1.
    for local, layer in enumerate((20, 21)):
        for kind in ("key", "value"):
            past = states[f"past_key_values.{layer}.{kind}present.{layer}.{kind}"]
            feed[f"past_key_values.{local}.{kind}"] = past.astype(np.float16)
    got = req.infer(feed)
    np.testing.assert_allclose(got["hidden_states"], want_h, rtol=2e-3, atol=2e-3)
    for local, layer in enumerate((20, 21)):
        present = got[f"present.{local}.key"]
        assert present.shape == (1, KVH, 4, HD)
        np.testing.assert_allclose(
            present[:, :, :3].astype(np.float32),
            states[f"past_key_values.{layer}.keypresent.{layer}.key"], rtol=2e-3, atol=2e-3)


def test_unslice_last_position_logits():
    x = ops.parameter([1, -1, 4], ov.Type.f32, name="hidden_states")
    sl = ops.slice(x, ops.constant(np.array([-1], np.int64)),
                   ops.constant(np.array([np.iinfo(np.int64).max], np.int64)),
                   ops.constant(np.array([1], np.int64)), ops.constant(np.array([1], np.int64)))
    m = ov.Model([ops.result(sl)], [x], "head")
    assert sv.unslice_logits(m)
    m.validate_nodes_and_infer_types()
    assert m.outputs[0].get_partial_shape().to_string() == "[1,?,4]"


def test_mixed_kv_geometry_is_listed_per_layer():
    # Gemma 4 style: layers differ in head_dim; each keeps its own, in the
    # local order of the ports (global layer 20 is local 0).
    m = toy_stage(head_dims=(HD, HD * 2))
    assert sv.make_stateless(m, has_head=False) == [(KVH, HD * 2), (KVH, HD)]
    sv.pin_shapes(m, 1, 16)
    shapes = {i.get_any_name(): list(i.get_shape()) for i in m.inputs}
    assert shapes["past_key_values.0.key"] == [1, KVH, 15, HD * 2]
    assert shapes["past_key_values.1.value"] == [1, KVH, 15, HD]


def toy_qwen_stage(deepstack_layers: int = 0) -> ov.Model:
    """A stage 0 like the Qwen VL language model's: M-RoPE position_ids
    `[3, ?, ?]`, and (Qwen3-VL) the DeepStack inputs. Each input is touched so
    its port survives make_stateless / reshape."""
    rng = np.random.default_rng(0)
    x = ops.parameter([-1, -1, HIDDEN], ov.Type.f32, name="inputs_embeds")
    x.output(0).set_names({"inputs_embeds"})
    mask = ops.parameter([-1, -1], ov.Type.i64, name="attention_mask")
    mask.output(0).set_names({"attention_mask"})
    pos = ops.parameter([3, -1, -1], ov.Type.i64, name="position_ids")
    pos.output(0).set_names({"position_ids"})
    beam = ops.parameter([-1], ov.Type.i32, name="beam_idx")
    beam.output(0).set_names({"beam_idx"})
    extra = []
    h = x.output(0)
    # Fold a scalar of the M-RoPE positions back into the stream so the port
    # is reachable from the result (else reshape would drop it).
    pbias = ops.convert(ops.reduce_sum(pos, ops.constant(np.array([0, 1, 2], np.int64)), False),
                        ov.Type.f32)
    h = ops.add(h, ops.reshape(pbias, ops.constant(np.array([1, 1, 1], np.int64)), False)).output(0)
    if deepstack_layers:
        vpm = ops.parameter([-1, -1], ov.Type.i64, name="visual_pos_masks")
        vpm.output(0).set_names({"visual_pos_masks"})
        dse = ops.parameter([-1, -1, HIDDEN], ov.Type.f32, name="deepstack_visual_embeds")
        dse.output(0).set_names({"deepstack_visual_embeds"})
        extra = [vpm, dse]
        # Gather layer 0's rows at a cumsum index of the mask, exactly the
        # export's `max(0, cumsum(mask) - 1)` gather, so N may exceed the
        # image-token count without the padded rows ever being read.
        cum = ops.cumsum(ops.reshape(vpm, ops.constant(np.array([-1], np.int64)), False),
                         ops.constant(np.int64(0)))
        idx = ops.maximum(ops.subtract(cum, ops.constant(np.int64(1))),
                          ops.constant(np.int64(0)))
        layer0 = ops.gather(dse, ops.constant(np.int64(0)), ops.constant(np.int64(0)))
        rows = ops.gather(layer0, idx, ops.constant(np.int64(0)))
        h = ops.add(h, ops.reshape(rows, ops.constant(np.array([1, -1, HIDDEN], np.int64)),
                                   False)).output(0)
    sinks = []
    for layer in (21, 20):
        outs = {}
        for kind in ("key", "value"):
            w = ops.constant(rng.standard_normal((HIDDEN, KVH * HD)).astype(np.float32))
            new = ops.matmul(h, w, False, False)
            new = ops.reshape(new, ops.constant(np.array([0, 0, KVH, HD], np.int64)), True)
            new = ops.transpose(new, ops.constant(np.array([0, 2, 1, 3], np.int64)))
            vid = f"past_key_values.{layer}.{kind}present.{layer}.{kind}"
            info = ov.op.util.VariableInfo()
            info.variable_id = vid
            info.data_shape = ov.PartialShape([-1, KVH, -1, HD])
            info.data_type = ov.Type.f32
            var = ov.op.util.Variable(info)
            init = ops.broadcast(ops.constant(np.float32(0)),
                                 ops.constant(np.array([1, KVH, 0, HD], np.int64)))
            rv = ops.read_value(init, var)
            past = ops.gather(rv, beam, ops.constant(np.int64(0)))
            cat = ops.concat([past, new], 2)
            sinks.append(ops.assign(cat, var))
            outs[kind] = cat
        k = ops.reduce_sum(outs["key"], ops.constant(np.array([1, 2, 3], np.int64)), True)
        kv = ops.reshape(k, ops.constant(np.array([-1, 1, 1], np.int64)), False)
        h = ops.add(h, ops.multiply(kv, ops.constant(np.float32(0.01)))).output(0)
    params = [x, mask, pos, beam] + extra
    return ov.Model([ops.result(h)], sinks, params, "toy_qwen")


def test_mrope_position_ids_pinned_to_three_rows():
    m = toy_qwen_stage()
    sv.make_stateless(m, has_head=False)
    sv.pin_shapes(m, 1, 16)
    shapes = {i.get_any_name(): list(i.get_shape()) for i in m.inputs}
    # [rows, batch, seq] — the shape the host feeds M-RoPE positions in.
    assert shapes["position_ids"] == [3, 1, 1]
    sv.pin_shapes(m, 5, 20)  # a prefill window
    assert list(next(i for i in m.inputs
                     if i.get_any_name() == "position_ids").get_shape()) == [3, 1, 5]


def test_deepstack_inputs_pinned_to_seq_width():
    m = toy_qwen_stage(deepstack_layers=3)
    sv.make_stateless(m, has_head=False)
    sv.pin_shapes(m, 4, 16, deepstack_layers=3)
    shapes = {i.get_any_name(): list(i.get_shape()) for i in m.inputs}
    assert shapes["position_ids"] == [3, 1, 4]
    assert shapes["visual_pos_masks"] == [1, 4]
    # [layers, seq, hidden] — N pinned to the query width, padded rows unread.
    assert shapes["deepstack_visual_embeds"] == [3, 4, HIDDEN]


def test_token_type_ids_pinned_to_seq_width():
    # Gemma 3 / Gemma 4 stages take a per-step token_type_ids frame; pin_shapes
    # must pin it to [1, seq] (a dropped branch here once made static conversion
    # of every token_type_ids tree fail with "unexpected stage input").
    x = ops.parameter([1, -1, HIDDEN], ov.Type.f32, name="inputs_embeds")
    x.output(0).set_names({"inputs_embeds"})
    pos = ops.parameter([1, -1], ov.Type.i64, name="position_ids")
    pos.output(0).set_names({"position_ids"})
    tt = ops.parameter([1, -1], ov.Type.i64, name="token_type_ids")
    tt.output(0).set_names({"token_type_ids"})
    bias = ops.convert(
        ops.reduce_sum(ops.add(pos, tt), ops.constant(np.array([1], np.int64)), True),
        ov.Type.f32)
    h = ops.add(x, ops.unsqueeze(bias, ops.constant(np.array([2], np.int64))))
    m = ov.Model([ops.result(h)], [x, pos, tt], "toy_token_types")
    sv.pin_shapes(m, 1, 16)
    shapes = {i.get_any_name(): list(i.get_shape()) for i in m.inputs}
    assert shapes["token_type_ids"] == [1, 1]
    assert shapes["inputs_embeds"] == [1, 1, HIDDEN]
    sv.pin_shapes(m, 5, 20)  # a prefill window
    assert list(next(i for i in m.inputs
                     if i.get_any_name() == "token_type_ids").get_shape()) == [1, 5]


def test_shared_prefill_ir_takes_any_query_width():
    # CPU/GPU: one IR, past fixed, query width (and mask length) dynamic.
    m = toy_stage()
    sv.make_stateless(m, has_head=False)
    sv.pin_shapes(m, -1, 16, past=15)
    shapes = {i.get_any_name(): i.get_partial_shape() for i in m.inputs}
    assert str(shapes["past_key_values.0.key"]) == f"[1,{KVH},15,{HD}]"
    assert str(shapes["inputs_embeds"]) == f"[1,?,{HIDDEN}]"
    assert str(shapes["attention_mask"]) == "[1,?]"
    req = ov.Core().compile_model(m, "CPU").create_infer_request()
    past = {f"past_key_values.{l}.{k}": np.zeros((1, KVH, 15, HD), np.float16)
            for l in range(LAYERS) for k in ("key", "value")}
    for seq in (1, 3):
        out = req.infer({"inputs_embeds": np.ones((1, seq, HIDDEN), np.float32),
                         "attention_mask": np.ones((1, 15 + seq), np.int64),
                         "position_ids": np.arange(seq, dtype=np.int64)[None], **past})
        assert out["present.0.key"].shape == (1, KVH, 15 + seq, HD)


# ---- --packed-only: lifting the Gemma 4 sliding-window cap ----


def test_window_error_cap_and_packed_only():
    # Within the window: fine either way (Gemma 4 280-row prefill, 1024 window).
    assert sv.window_error(736, 288, 1024, True, False) is None
    # Past it: refused, and on Gemma 4 the message names the flag.
    err = sv.window_error(1280, 288, 1024, True, False)
    assert "exceeds the sliding window (1024)" in err and "--packed-only" in err
    assert "--packed-slots" in err
    # --packed-only lifts the cap on Gemma 4 only.
    assert sv.window_error(1280, 288, 1024, True, True) is None
    other = sv.window_error(1280, 288, 1024, False, False)
    assert "exceeds" in other and "--packed-only" not in other
    assert "Gemma 4 trees only" in sv.window_error(1280, 288, 1024, False, True)
    assert "Gemma 4 trees only" in sv.window_error(16, 0, None, False, True)
    # No window: nothing to cap.
    assert sv.window_error(4096, 256, None, False, False) is None


def test_is_gemma4():
    assert sv.is_gemma4({"vision": {"family": "gemma4"}}, {})
    assert sv.is_gemma4({}, {"model_type": "gemma4_text"})
    assert not sv.is_gemma4({"vision": {"family": "gemma3"}}, {"model_type": "gemma3"})


def _toy_tree(root, window=8):
    tree = root / "tree"
    stage = tree / "stage_0"
    stage.mkdir(parents=True)
    ov.save_model(toy_stage(), str(stage / "openvino_model.xml"), compress_to_fp16=False)
    (stage / "stage_config.json").write_text(json.dumps({
        "stateful": True, "has_embed": True, "has_head": False,
        "layer_start": 0, "layer_end": LAYERS}))
    (tree / "pipeline_config.json").write_text(json.dumps({
        "num_stages": 1, "stages": [{"layer_start": 0, "layer_end": LAYERS}],
        "vision": {"family": "gemma4"}}))
    (tree / "config.json").write_text(json.dumps({
        "model_type": "gemma4", "sliding_window": window}))
    return tree


def _run(monkeypatch, tree, out, *extra):
    monkeypatch.setattr(sys, "argv", [
        "static_variant.py", "--tree", str(tree), "--output-dir", str(out),
        "--context", "12", "--prefill-seq", "4", "--shared-prefill", *extra])
    sv.main()


def test_main_refuses_past_the_window_without_packed_only(tmp_path, monkeypatch):
    tree = _toy_tree(tmp_path)
    with pytest.raises(SystemExit, match="--packed-only"):
        _run(monkeypatch, tree, tmp_path / "out")
    assert not (tmp_path / "out" / "pipeline_config.json").exists()


def test_main_packed_only_marks_every_stage_and_the_pipeline(tmp_path, monkeypatch):
    tree = _toy_tree(tmp_path)
    out = tmp_path / "out"
    _run(monkeypatch, tree, out, "--packed-only")
    stage = json.loads((out / "stage_0" / "stage_config.json").read_text())
    pipeline = json.loads((out / "pipeline_config.json").read_text())
    assert stage["packed_only"] is True and pipeline["packed_only"] is True
    assert stage["static_context"] == 12 and stage["static_prefill_seq"] == 4
    # A rebuild within the window, without the flag, drops the marker.
    _toy_tree(tmp_path / "again", window=64)
    monkeypatch.setattr(sys, "argv", [
        "static_variant.py", "--tree", str(tmp_path / "again" / "tree"),
        "--output-dir", str(out), "--context", "12", "--prefill-seq", "4", "--shared-prefill"])
    sv.main()
    assert "packed_only" not in json.loads((out / "stage_0" / "stage_config.json").read_text())
    assert "packed_only" not in json.loads((out / "pipeline_config.json").read_text())
