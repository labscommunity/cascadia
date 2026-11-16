"""export_gemma4_text.slice_stages(vision=True) on a toy gemma-4 language model.

Run from the repo root with:

    python -m pytest tools/tests/test_gemma4_vision_slice.py -v

A two-layer stateful toy with gemma-4's VLM inputs (inputs_embeds,
token_type_ids, attention_mask, position_ids, beam_idx) goes through
vision_frontend and a two-stage slice. The tree must carry the vision IRs and
processor configs rank 0 runs, the VLM config as config.vlm.json beside the
flat text config, and a `vision` block saying whether the stages take
token_type_ids; a stage that lost token_type_ids fails the export, with no
pipeline_config.json written.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
ov = pytest.importorskip("openvino")

_SURGERY = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "gemma4_surgery"))
if _SURGERY not in sys.path:
    sys.path.insert(0, _SURGERY)

import export_gemma4_text as g  # noqa: E402
from openvino import opset13 as ops  # noqa: E402

H, KVH, HD, VOCAB, LAYERS = 8, 2, 4, 11, 2
SCOPE = "__module.model.language_model.layers"
IMAGE_TOKEN = 258880


def named(node, name):
    node.set_friendly_name(name)
    return node


def toy_lm(types_layers=range(LAYERS), seed: int = 0) -> ov.Model:
    """Decoder layers cut at `...layers.N.input_layernorm/aten::pow/Power`, KV
    as ReadValue -> Gather(beam) -> Concat -> Assign; the layers in
    `types_layers` read token_type_ids (gemma-4's sliding masks do)."""
    rng = np.random.default_rng(seed)
    x = ops.parameter([-1, -1, H], ov.Type.f32, name="inputs_embeds")
    tt = ops.parameter([-1, -1], ov.Type.i64, name="token_type_ids")
    mask = ops.parameter([-1, -1], ov.Type.i64, name="attention_mask")
    pos = ops.parameter([-1, -1], ov.Type.i64, name="position_ids")
    beam = ops.parameter([-1], ov.Type.i32, name="beam_idx")
    params = [x, tt, mask, pos, beam]
    for p in params:
        p.output(0).set_names({p.get_friendly_name()})
    msum = ops.convert(ops.reduce_sum(mask, ops.constant(np.array([1], np.int64)), True),
                       ov.Type.f32)
    bias = ops.add(ops.unsqueeze(ops.convert(pos, ov.Type.f32), ops.constant(np.int64(2))),
                   ops.reshape(msum, ops.constant(np.array([-1, 1, 1], np.int64)), False))
    types = ops.unsqueeze(ops.convert(tt, ov.Type.f32), ops.constant(np.int64(2)))
    h = x.output(0)
    sinks = []
    for layer in range(LAYERS):
        sc = f"{SCOPE}.{layer}"
        pw = named(ops.power(h, ops.constant(np.float32(2))), f"{sc}.input_layernorm/aten::pow/Power")
        n = ops.multiply(h, ops.add(ops.reduce_mean(pw, ops.constant(np.int64(-1)), True),
                                    ops.constant(np.float32(1))))
        outs = {}
        for kind in ("key", "value"):
            w = ops.constant(rng.standard_normal((H, KVH * HD)).astype(np.float32) * 0.1)
            new = ops.reshape(ops.matmul(n, w, False, False),
                              ops.constant(np.array([0, 0, KVH, HD], np.int64)), True)
            new = ops.transpose(new, ops.constant(np.array([0, 2, 1, 3], np.int64)))
            info = ov.op.util.VariableInfo()
            info.variable_id = f"past_key_values.{layer}.{kind}present.{layer}.{kind}"
            info.data_shape = ov.PartialShape([-1, KVH, -1, HD])
            info.data_type = ov.Type.f32
            var = ov.op.util.Variable(info)
            init = ops.broadcast(ops.constant(np.float32(0)),
                                 ops.constant(np.array([1, KVH, 0, HD], np.int64)))
            rv = ops.read_value(init, var)
            past = ops.gather(rv, beam, ops.constant(np.int64(0)))
            cat = named(ops.concat([past, new], 2), f"{sc}.self_attn/aten::cat/Concat_{kind}")
            sinks.append(ops.assign(cat, var))
            outs[kind] = cat
        kv = ops.reduce_mean(ops.add(outs["key"], outs["value"]),
                             ops.constant(np.array([1, 2, 3], np.int64)), True)
        kv = ops.reshape(kv, ops.constant(np.array([-1, 1, 1], np.int64)), False)
        side = ops.add(ops.multiply(bias, ops.constant(np.float32(0.01))), kv)
        if layer in types_layers:
            side = ops.add(side, ops.multiply(types, ops.constant(np.float32(0.5))))
        wo = ops.constant(rng.standard_normal((H, H)).astype(np.float32) * 0.1)
        h = named(ops.add(ops.add(h, ops.matmul(n, wo, False, False)), side),
                  f"{sc}/aten::add/Add").output(0)
    head = ops.constant(rng.standard_normal((H, VOCAB)).astype(np.float32))
    logits = ops.matmul(h, head, False, False)
    logits.output(0).set_names({"logits"})
    return ov.Model([ops.result(logits)], sinks, params, "toy_gemma4_vlm_lm")


VLM_CONFIG = {
    "model_type": "gemma4", "image_token_id": IMAGE_TOKEN, "eos_token_id": [1, 106],
    "vision_config": {"model_type": "gemma4_vision"},
    "text_config": {"model_type": "gemma4_text", "num_hidden_layers": LAYERS,
                    "hidden_size": H, "num_attention_heads": KVH,
                    "num_key_value_heads": KVH, "head_dim": HD, "vocab_size": VOCAB},
}


def export_dir(root, types_layers=range(LAYERS), skip=()):
    root.mkdir()
    ov.save_model(toy_lm(types_layers), str(root / "openvino_language_model.xml"),
                  compress_to_fp16=False)
    (root / "config.json").write_text(json.dumps(VLM_CONFIG))
    (root / "tokenizer.json").write_text("{}")
    for fn in g.VISION_AUX_COPY:
        if fn not in skip:
            (root / fn).write_text(f"<{fn}>")
    return root


def slice_vision(src, out):
    grafted, info = g.vision_frontend(str(src))
    g.slice_stages(grafted, str(src), str(out), num_stages=2, num_layers=LAYERS,
                   hidden_size=H, num_kv_shared_layers=0,
                   suffix=g.DEFAULT_LAYERNORM_SUFFIX, last_logits_only=True,
                   skip_tokenizer_regen=True, num_kv_heads=KVH, head_dim=HD,
                   vision=True, vision_token_types=info["token_type_ids"])


def stage_inputs(out, i):
    m = ov.Core().read_model(str(out / f"stage_{i}" / "openvino_model.xml"))
    return {n for p in m.get_parameters() for n in p.output(0).get_names()}


def test_vision_slice_writes_the_image_path(tmp_path):
    src, out = export_dir(tmp_path / "vlm"), tmp_path / "tree"
    slice_vision(src, out)

    for fn in g.VISION_AUX_COPY:
        assert (out / fn).read_text() == f"<{fn}>", fn
    # Rank 0 reads the VLM original; config.json is the flat text config.
    assert json.loads((out / "config.vlm.json").read_text()) == VLM_CONFIG
    flat = json.loads((out / "config.json").read_text())
    assert flat["num_hidden_layers"] == LAYERS and "vision_config" not in flat

    pipe = json.loads((out / "pipeline_config.json").read_text())
    assert pipe["export_version"] == g.EXPORT_VERSION_VISION
    assert pipe["vision"] == {
        "family": "gemma4",
        "text_embeddings": "openvino_text_embeddings_model.xml",
        "vision_embeddings": "openvino_vision_embeddings_model.xml",
        "inputs_embeds": True,
        "token_type_ids": True,
    }
    assert "inputs_embeds" in stage_inputs(out, 0)
    assert "hidden_states" in stage_inputs(out, 1)
    for i in range(2):
        assert "token_type_ids" in stage_inputs(out, i), f"stage_{i}"
        cfg = json.loads((out / f"stage_{i}" / "stage_config.json").read_text())
        assert "token_type_ids" in cfg["inputs"]


def test_vision_slice_without_token_types_says_so(tmp_path):
    # An IR whose token_type_ids feeds nothing: image tokens attend causally,
    # the stages take no third frame, and the block tells the runtime.
    src, out = export_dir(tmp_path / "vlm", types_layers=()), tmp_path / "tree"
    slice_vision(src, out)
    assert json.loads((out / "pipeline_config.json").read_text())["vision"]["token_type_ids"] \
        is False
    for i in range(2):
        assert "token_type_ids" not in stage_inputs(out, i)


def test_vision_slice_refuses_a_stage_that_lost_token_types(tmp_path):
    # Only layer 0 reads token_type_ids, so stage_1 (layer 1) loses it.
    src, out = export_dir(tmp_path / "vlm", types_layers=(0,)), tmp_path / "tree"
    with pytest.raises(SystemExit, match="stage_1 lost token_type_ids"):
        slice_vision(src, out)
    assert not (out / "pipeline_config.json").exists()


def test_vision_slice_needs_the_vision_files(tmp_path):
    src = export_dir(tmp_path / "vlm", skip=("processor_config.json",))
    out = tmp_path / "tree"
    with pytest.raises(SystemExit, match="--vision needs processor_config.json"):
        slice_vision(src, out)
    assert not (out / "pipeline_config.json").exists()


def test_one_stage_export_needs_no_temp_copy(tmp_path):
    # --stage 1 on a node with room for its stage only: cut from the graft in
    # memory (no _grafted/ copy) and give the tree its own pipeline_config.
    src, out = export_dir(tmp_path / "vlm"), tmp_path / "tree"
    grafted, info = g.vision_frontend(str(src))
    g.slice_stages(grafted, str(src), str(out), num_stages=2, num_layers=LAYERS,
                   hidden_size=H, num_kv_shared_layers=0,
                   suffix=g.DEFAULT_LAYERNORM_SUFFIX, last_logits_only=True,
                   only_stage=1, skip_tokenizer_regen=True, num_kv_heads=KVH,
                   head_dim=HD, vision=True, vision_token_types=info["token_type_ids"])
    assert not (out / "_grafted").exists()
    assert not (out / "stage_0").exists()
    assert {"hidden_states", "token_type_ids"} <= stage_inputs(out, 1)
    pipe = json.loads((out / "pipeline_config.json").read_text())
    assert pipe["num_stages"] == 2 and pipe["only_stage"] == 1
    assert [s["stage"] for s in pipe["stages"]] == [1]
    assert pipe["vision"]["family"] == "gemma4"
    # The other stage into the same tree completes it.
    g.slice_stages(g.vision_frontend(str(src))[0], str(src), str(out), num_stages=2,
                   num_layers=LAYERS, hidden_size=H, num_kv_shared_layers=0,
                   suffix=g.DEFAULT_LAYERNORM_SUFFIX, last_logits_only=True,
                   only_stage=0, skip_tokenizer_regen=True, num_kv_heads=KVH,
                   head_dim=HD, vision=True, vision_token_types=info["token_type_ids"])
    pipe = json.loads((out / "pipeline_config.json").read_text())
    assert "only_stage" not in pipe
    assert [s["stage"] for s in pipe["stages"]] == [0, 1]
    assert "inputs_embeds" in stage_inputs(out, 0)
