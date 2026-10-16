"""Synthetic-graph tests for the ov-dist-spec VLM tools.

Run from the repo root with:

    python -m pytest tools/tests/test_vlm_distspec.py -v

* `tools/vlm_surgery/make_text_draft.py` composes a VLM export's text
  embedding IR into its `inputs_embeds` language model. The composed draft
  must take `input_ids` (+ mask, positions, beam) and give the same logits and
  KV state, step for step, as the two IRs run back to back.
* `tools/vlm_surgery/cut_vlm_lm.py --all-logits` keeps a logits row per fed
  position (ov-dist-spec verifies every drafted token); without it the head is
  sliced to the last position. The last stage records which in its config.
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

import make_text_draft as mtd  # noqa: E402
from openvino import opset13 as ops  # noqa: E402

VOCAB, HIDDEN, KVH, HD = 32, 8, 2, 4


def toy_embeddings(seed: int = 0, ids_type=ov.Type.i64) -> ov.Model:
    """optimum's text embedding IR: one ids input -> [B, T, HIDDEN] rows."""
    table = np.random.default_rng(seed).standard_normal((VOCAB, HIDDEN)).astype(np.float32)
    ids = ops.parameter([-1, -1], ids_type, name="input")
    ids.output(0).set_names({"input"})
    rows = ops.gather(ops.constant(table), ids, ops.constant(np.int64(0)))
    out = ops.result(rows)
    out.output(0).set_names({"inputs_embeds"})
    return ov.Model([out], [], [ids], "toy_text_embeddings")


def toy_lm(seed: int = 1) -> ov.Model:
    """A one-layer stateful language model shaped like optimum's VLM export:
    inputs_embeds/attention_mask/position_ids/beam_idx in, logits for every
    position out, KV as ReadValue -> Gather(beam) -> Concat -> Assign."""
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
    sinks, outs = [], {}
    for kind in ("key", "value"):
        w = ops.constant(rng.standard_normal((HIDDEN, KVH * HD)).astype(np.float32))
        new = ops.matmul(h, w, False, False)
        new = ops.reshape(new, ops.constant(np.array([0, 0, KVH, HD], np.int64)), True)
        new = ops.transpose(new, ops.constant(np.array([0, 2, 1, 3], np.int64)))
        info = ov.op.util.VariableInfo()
        info.variable_id = f"past_key_values.0.{kind}present.0.{kind}"
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
    v = ops.reduce_sum(outs["value"], ops.constant(np.array([1, 2, 3], np.int64)), True)
    kv = ops.reshape(ops.add(k, v), ops.constant(np.array([-1, 1, 1], np.int64)), False)
    m = ops.convert(ops.reduce_sum(mask, ops.constant(np.array([1], np.int64)), True), ov.Type.f32)
    p = ops.convert(pos, ov.Type.f32)
    p = ops.unsqueeze(p, ops.constant(np.int64(2)))
    bias = ops.add(ops.reshape(m, ops.constant(np.array([-1, 1, 1], np.int64)), False), p)
    h = ops.add(ops.add(h, ops.multiply(kv, ops.constant(np.float32(0.01)))), bias)
    head = ops.constant(rng.standard_normal((HIDDEN, VOCAB)).astype(np.float32))
    logits = ops.matmul(h, head, False, False)
    out = ops.result(logits)
    out.output(0).set_names({"logits"})
    return ov.Model([out], sinks, [x, mask, pos, beam], "toy_language_model")


def _feed(ids, start):
    n = len(ids)
    return {
        "attention_mask": np.ones((1, start + n), np.int64),
        "position_ids": np.arange(start, start + n, dtype=np.int64)[None],
        "beam_idx": np.array([0], np.int32),
    }


@pytest.mark.parametrize("ids_type", [ov.Type.i64, ov.Type.i32])
def test_composed_draft_matches_embeddings_then_lm(ids_type):
    core = ov.Core()
    composed = mtd.compose(toy_embeddings(ids_type=ids_type), toy_lm())
    names = [p.get_friendly_name() for p in composed.get_parameters()]
    assert names == ["input_ids", "attention_mask", "position_ids", "beam_idx"]
    assert composed.input("input_ids").get_element_type() == ov.Type.i64
    assert len(composed.get_sinks()) == 2
    assert composed.outputs[0].get_any_name() == "logits"

    draft = core.compile_model(composed, "CPU").create_infer_request()
    emb = core.compile_model(toy_embeddings(ids_type=ids_type), "CPU").create_infer_request()
    lm = core.compile_model(toy_lm(), "CPU").create_infer_request()
    ids_np = np.int64 if ids_type == ov.Type.i64 else np.int32
    start = 0
    # Prefill (an expanded prompt, image placeholder id 10 as plain text), then
    # two decode steps: the KV state must carry identically.
    for span in ([1, 10, 10, 10, 5], [7], [3]):
        got = draft.infer({"input_ids": np.array([span], np.int64), **_feed(span, start)})
        rows = emb.infer({"input": np.array([span], ids_np)})[emb.model_outputs[0]]
        want = lm.infer({"inputs_embeds": rows, **_feed(span, start)})
        g, w = got[draft.model_outputs[0]], want[lm.model_outputs[0]]
        assert g.shape == (1, len(span), VOCAB)
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)
        start += len(span)


def test_compose_refuses_a_model_without_inputs_embeds():
    with pytest.raises(SystemExit, match="inputs_embeds"):
        mtd.compose(toy_embeddings(), toy_embeddings())


def test_run_writes_a_loadable_draft_dir(tmp_path):
    src, out = tmp_path / "vlm", tmp_path / "draft"
    src.mkdir()
    ov.save_model(toy_lm(), str(src / mtd.LM_XML), compress_to_fp16=False)
    ov.save_model(toy_embeddings(), str(src / mtd.EMBED_XML), compress_to_fp16=False)
    (src / "config.json").write_text(json.dumps({
        "model_type": "mistral3", "eos_token_id": 2,
        "text_config": {"vocab_size": VOCAB, "hidden_size": HIDDEN}}))
    (src / "tokenizer.json").write_text("{}")
    mtd.run(str(src), str(out))
    m = ov.Core().read_model(str(out / mtd.OUT_XML))
    assert "input_ids" in [p.get_friendly_name() for p in m.get_parameters()]
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["vocab_size"] == VOCAB and cfg["eos_token_id"] == 2
    assert (out / "tokenizer.json").exists() and (out / "config.vlm.json").exists()


@pytest.mark.parametrize("all_logits", [False, True])
def test_cut_all_logits_keeps_every_row(tmp_path, all_logits):
    import cut_vlm_lm as cut

    src, out = tmp_path / "vlm", tmp_path / "tree"
    src.mkdir()
    ov.save_model(toy_lm(), str(src / "openvino_language_model.xml"), compress_to_fp16=False)
    for fn in ("openvino_text_embeddings_model.xml", "openvino_vision_embeddings_model.xml"):
        ov.save_model(toy_embeddings(), str(src / fn), compress_to_fp16=False)
    (src / "config.json").write_text(json.dumps({
        "model_type": "llava", "image_token_index": 10,
        "vision_config": {"model_type": "pixtral"},
        "text_config": {"num_hidden_layers": 1, "hidden_size": HIDDEN,
                        "num_attention_heads": KVH, "num_key_value_heads": KVH,
                        "vocab_size": VOCAB}}))
    (src / "tokenizer.json").write_text("{}")
    cut.run(str(src), str(out), 1, "pixtral", cut.g.DEFAULT_LAYERNORM_SUFFIX, None, all_logits)

    want = "all" if all_logits else "last"
    stage_cfg = json.loads((out / "stage_0" / "stage_config.json").read_text())
    assert stage_cfg["logits"] == want
    pipe = json.loads((out / "pipeline_config.json").read_text())
    assert pipe["stages"][-1]["logits"] == want
    assert pipe["vision"]["family"] == "pixtral"

    req = ov.Core().compile_model(str(out / "stage_0" / "openvino_model.xml"),
                                  "CPU").create_infer_request()
    span = [1, 10, 10, 5]
    rows = np.zeros((1, len(span), HIDDEN), np.float32)
    got = req.infer({"inputs_embeds": rows, **_feed(span, 0)})[req.model_outputs[0]]
    assert got.shape == ((1, len(span), VOCAB) if all_logits else (1, 1, VOCAB))
