"""A one-stage cut into a tree that already holds stages, and the stage
configs the cut and static tools write.

Run from the repo root with:

    python -m pytest tools/tests/test_stage_tree_reuse.py -v

`cut_vlm_lm.py --stage I`, `export_gemma4_text.py --stage I` and
`static_variant.py --stage I` write one stage beside the others already in the
output directory. A tree whose other stages came from a cut with another
stage count would load with layers missing or run twice, so the tools refuse
it. Also covered: the static tree's head records `"logits": "all"` (its
last-position slice is dropped), and `head_dim` comes from the IR when
config.json omits it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

np = pytest.importorskip("numpy")
ov = pytest.importorskip("openvino")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SURGERY = os.path.join(_HERE, os.pardir, "vlm_surgery")
for _d in (_HERE, _SURGERY):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import cut_vlm_lm as cut  # noqa: E402
import static_variant as sv  # noqa: E402
from test_vlm_qwen_cut import HD, KVH, export_dir, toy_lm  # noqa: E402

g = cut.g
SUFFIX = g.DEFAULT_LAYERNORM_SUFFIX


def _cut(src, out, stages, stage=None):
    cut.run(str(src), str(out), stages, "qwen2_5_vl", SUFFIX, stage)


def _read(path):
    return json.loads(path.read_text())


def _ranges(out):
    return [(s["layer_start"], s["layer_end"])
            for s in _read(out / "pipeline_config.json")["stages"]]


# --- cut_vlm_lm.py --stage --------------------------------------------------

def test_cut_stage_refuses_a_tree_cut_with_another_stage_count(tmp_path):
    src = export_dir(tmp_path, "qwen2_5_vl", layers=6, deepstack=0)
    out = tmp_path / "tree"
    _cut(src, out, 3)
    before = (out / "pipeline_config.json").read_text()
    # stage_1 of a 2-stage cut is layers 3..5; the tree's stage_0 is 0..1.
    with pytest.raises(SystemExit, match="different cut"):
        _cut(src, out, 2, stage=1)
    # Refused before anything was touched: the 3-stage tree still loads.
    assert (out / "pipeline_config.json").read_text() == before
    assert _read(out / "stage_1" / "stage_config.json")["layer_start"] == 2


def test_cut_stage_refuses_stages_from_another_export(tmp_path):
    src = export_dir(tmp_path, "qwen2_5_vl", layers=4, deepstack=0)
    out = tmp_path / "tree"
    _cut(src, out, 2)
    (out / "pipeline_config.json").unlink()  # a run that died before the marker
    cfg = out / "stage_0" / "stage_config.json"
    cfg.write_text(json.dumps({**_read(cfg), "export_version": "gemma4_text_surgery_v1"}))
    with pytest.raises(SystemExit, match="export_version"):
        _cut(src, out, 2, stage=1)


def test_cut_stage_into_its_own_tree_writes_a_contiguous_marker(tmp_path):
    src = export_dir(tmp_path, "qwen2_5_vl", layers=6, deepstack=0)
    out = tmp_path / "tree"
    _cut(src, out, 3, stage=0)
    _cut(src, out, 3, stage=2)
    _cut(src, out, 3, stage=1)
    assert _ranges(out) == [(0, 2), (2, 4), (4, 6)]


def test_head_dim_comes_from_the_ir_when_config_omits_it(tmp_path):
    # transformers drops default values on save: Gemma 3 12B's config has no
    # head_dim (256), and hidden_size / num_attention_heads is 240.
    src = export_dir(tmp_path, "qwen2_5_vl", layers=4, deepstack=0)
    cfg = _read(src / "config.json")
    del cfg["text_config"]["head_dim"]
    cfg["text_config"]["num_attention_heads"] = 4  # 8 / 4 = 2, the IR has 4
    cfg["text_config"]["num_key_value_heads"] = 1
    (src / "config.json").write_text(json.dumps(cfg))
    out = tmp_path / "tree"
    _cut(src, out, 2)
    for i in (0, 1):
        sc = _read(out / f"stage_{i}" / "stage_config.json")
        assert (sc["head_dim"], sc["num_kv_heads"]) == (HD, KVH)
    assert _read(out / "pipeline_config.json")["stages"][0]["head_dim"] == HD


# --- the shared checks -------------------------------------------------------

@pytest.mark.parametrize("stages,num_layers,match", [
    ([(0, 2), (3, 6)], 6, "skip or repeat"),      # gap: layer 2 never runs
    ([(0, 3), (2, 6)], 6, "skip or repeat"),      # overlap: layer 2 runs twice
    ([(0, 3), (3, 5)], 6, "end at layer 5"),       # short
    ([(1, 3), (3, 6)], 6, "skip or repeat"),      # does not start at 0
])
def test_check_contiguous_refuses(stages, num_layers, match):
    with pytest.raises(SystemExit, match=match):
        g.check_contiguous([{"layer_start": a, "layer_end": b} for a, b in stages],
                           num_layers, len(stages))


def test_check_contiguous_counts_stages():
    with pytest.raises(SystemExit, match="lists 1 stages"):
        g.check_contiguous([{"layer_start": 0, "layer_end": 6}], 6, 2)
    g.check_contiguous([{"layer_start": 0, "layer_end": 3},
                        {"layer_start": 3, "layer_end": 6}], 6, 2)


# --- export_gemma4_text.py --stage (the only_stage merge) ---------------------

def _gemma_src(tmp_path, layers):
    src = tmp_path / "gsrc"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"num_hidden_layers": layers}))
    (src / "tokenizer.json").write_text("{}")
    return src


def _slice(src, out, layers, stages, stage):
    from test_vlm_qwen_cut import H
    g.slice_stages(toy_lm(layers, 0), str(src), str(out), stages, layers, H, 0, SUFFIX,
                   True, only_stage=stage, skip_tokenizer_regen=True)


def test_gemma4_stage_merge_refuses_another_stage_count(tmp_path):
    src, out = _gemma_src(tmp_path, 6), tmp_path / "gtree"
    _slice(src, out, 6, 3, 0)
    with pytest.raises(SystemExit, match="num_stages 3, this cut 2"):
        _slice(src, out, 6, 2, 1)
    pipe = _read(out / "pipeline_config.json")
    assert pipe["num_stages"] == 3 and pipe["only_stage"] == 0


def test_gemma4_stage_merge_refuses_a_complete_tree_of_another_layout(tmp_path):
    src, out = _gemma_src(tmp_path, 6), tmp_path / "gtree"
    for i in range(3):
        _slice(src, out, 6, 3, i)
    pipe = _read(out / "pipeline_config.json")
    assert "only_stage" not in pipe
    assert [(s["layer_start"], s["layer_end"]) for s in pipe["stages"]] == \
        [(0, 2), (2, 4), (4, 6)]
    # The complete marker was kept as-is before; now the mismatch is refused.
    with pytest.raises(SystemExit, match="different cut"):
        _slice(src, out, 6, 2, 1)


# --- static_variant.py ---------------------------------------------------------

def _static(monkeypatch, tree, out, *extra):
    monkeypatch.setattr(sys, "argv", [
        "static_variant.py", "--tree", str(tree), "--output-dir", str(out),
        "--context", "8", *extra])
    sv.main()


def test_static_head_records_all_logits(tmp_path, monkeypatch):
    src = export_dir(tmp_path, "qwen2_5_vl", layers=4, deepstack=0)
    tree, out = tmp_path / "tree", tmp_path / "static"
    _cut(src, tree, 2)
    assert _read(tree / "stage_1" / "stage_config.json")["logits"] == "last"
    _static(monkeypatch, tree, out)
    assert _read(out / "stage_1" / "stage_config.json")["logits"] == "all"
    assert _read(out / "pipeline_config.json")["stages"][-1]["logits"] == "all"
    assert "logits" not in _read(out / "stage_0" / "stage_config.json")
    # The IR agrees: the head returns a row per fed position.
    m = ov.Core().read_model(str(out / "stage_1" / "openvino_model.xml"))
    assert sv.last_position_slice(m.get_results()[0]) is None


def test_static_stage_refuses_an_output_of_another_layout(tmp_path, monkeypatch):
    src = export_dir(tmp_path, "qwen2_5_vl", layers=6, deepstack=0)
    three, two, out = tmp_path / "three", tmp_path / "two", tmp_path / "static"
    _cut(src, three, 3)
    _cut(src, two, 2)
    _static(monkeypatch, three, out, "--stage", "0")
    with pytest.raises(SystemExit, match="different cut"):
        _static(monkeypatch, two, out, "--stage", "1")
    # The same layout is accepted, stage by stage.
    _static(monkeypatch, three, out, "--stage", "2")
    assert _ranges(out) == [(0, 2), (2, 4), (4, 6)]


def test_static_converts_a_nodes_one_stage_tree(tmp_path, monkeypatch):
    # export_gemma4_text.py --stage I leaves a node a tree that lists only its
    # own stage ("only_stage"); static_variant converts that stage alone.
    src = export_dir(tmp_path, "qwen2_5_vl", layers=4, deepstack=0)
    tree, out = tmp_path / "tree", tmp_path / "static"
    _cut(src, tree, 2)
    pipe = _read(tree / "pipeline_config.json")
    pipe["stages"] = [s for s in pipe["stages"] if s["stage"] == 1]
    pipe["only_stage"] = 1
    (tree / "pipeline_config.json").write_text(json.dumps(pipe))
    shutil.rmtree(tree / "stage_0")
    with pytest.raises(SystemExit, match="holds only stage"):
        _static(monkeypatch, tree, out)
    with pytest.raises(SystemExit, match="holds only stage"):
        _static(monkeypatch, tree, out, "--stage", "0")
    _static(monkeypatch, tree, out, "--stage", "1")
    got = _read(out / "pipeline_config.json")
    assert got["only_stage"] == 1
    assert [(s["layer_start"], s["layer_end"]) for s in got["stages"]] == [(2, 4)]
    assert _read(out / "stage_1" / "stage_config.json")["logits"] == "all"
