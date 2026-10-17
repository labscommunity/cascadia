"""tools/decision/cut_lm_head.py: in-place LM-head cut of an existing tree, with backup and restore.

    python -m pytest tools/tests/test_lm_head_cut.py -v
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "qwen36_surgery"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "decision"))
sys.path.insert(0, os.path.dirname(__file__))
import cut_lm_head as clh  # noqa: E402
import weight_swap as ws  # noqa: E402
from test_weight_swap import _head_model, _run  # noqa: E402

MANIFEST = {"arch": "qwen3_5", "family": "qwen3_5", "hidden_size": 4, "num_layers": 1,
            "layer_types": ["full_attention"], "source": "src", "last_logits_only": True,
            "final_hidden_output": True,
            "stages": [{"stage": 0, "layer_start": 0, "layer_end": 0, "has_embed": True, "has_head": True,
                        "inputs": ["stage_hidden"], "state_vars": []}]}


def _tree(tmp_path, manifest=MANIFEST, lora=False):
    ov, model = _head_model(lora=lora, tmp_path=tmp_path)
    tree = tmp_path / "tree"
    (tree / "stage0").mkdir(parents=True)
    ov.save_model(model, str(tree / "stage0" / "stage.xml"), compress_to_fp16=False)
    with open(tree / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return ov, tree


def _bytes(tree):
    return {n: (tree / n).read_bytes() for n in ("stage0/stage.xml", "stage0/stage.bin", "manifest.json")}


@pytest.mark.parametrize("lora", [False, True])
def test_cut_then_restore(tmp_path, capsys, lora):
    ov, tree = _tree(tmp_path, lora=lora)
    before = _bytes(tree)
    x = np.random.default_rng(0).normal(size=(1, 5, 4)).astype(np.float32)
    lb, hb = _run(ov, ov.Core().read_model(str(tree / "stage0" / "stage.xml")), x)

    assert clh.cut(str(tree)) is True
    model = ov.Core().read_model(str(tree / "stage0" / "stage.xml"))
    assert ws.cut_lm_head_to_last_position(model) is False, "the saved stage is cut"
    la, ha = _run(ov, model, x)
    np.testing.assert_allclose(la, lb, rtol=1e-6)
    np.testing.assert_array_equal(ha, hb)
    for n in ("stage0/stage.xml", "stage0/stage.bin", "manifest.json"):
        assert (tree / (n + ".pre-cut")).read_bytes() == before[n]
    assert not (tree / "stage0" / "stage.cut.xml").exists()
    m = json.loads((tree / "manifest.json").read_text())
    assert m["lm_head_last_row"] is True
    assert list(m)[-3:] == ["final_hidden_output", "lm_head_last_row", "stages"], "where build_manifest puts it"

    capsys.readouterr()
    assert clh.cut(str(tree)) is False
    assert "already cut" in capsys.readouterr().out

    clh.restore(str(tree))
    assert _bytes(tree) == before
    assert not any(p.name.endswith(".pre-cut") for p in tree.rglob("*"))


def test_cut_manifest_matches_a_fresh_export(tmp_path):
    """The patched manifest equals what run_export writes with lm_head_last_row=True."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "qwen36_surgery"))
    import export_qwen36_moe as ex
    spec = ex.ModelSpec(model_type="qwen3_5", hidden=4, num_layers=1, layer_types=["full_attention"])
    old = ex.build_manifest(spec, "src", final_hidden=True)
    old["stages"] = MANIFEST["stages"]
    new = ex.build_manifest(spec, "src", final_hidden=True, lm_head_last_row=True)
    new["stages"] = MANIFEST["stages"]
    assert json.dumps(clh.with_key(old), indent=2) == json.dumps(new, indent=2)


def test_cut_refuses_a_last_stage_without_head(tmp_path):
    m = json.loads(json.dumps(MANIFEST))
    m["stages"][0]["has_head"] = False
    _, tree = _tree(tmp_path, manifest=m)
    with pytest.raises(SystemExit, match="no head"):
        clh.cut(str(tree))


def test_cut_refuses_stale_backups(tmp_path):
    _, tree = _tree(tmp_path)
    (tree / "stage0" / "stage.bin.pre-cut").write_bytes(b"old")
    before = _bytes(tree)
    with pytest.raises(SystemExit, match="backups exist"):
        clh.cut(str(tree))
    assert _bytes(tree) == before


def test_restore_without_backup_refuses(tmp_path):
    _, tree = _tree(tmp_path)
    with pytest.raises(SystemExit, match="no backup"):
        clh.restore(str(tree))


def test_cut_refuses_a_directory_without_manifest(tmp_path):
    with pytest.raises(SystemExit, match="no manifest.json"):
        clh.cut(str(tmp_path))


def test_head_cut_check_runs_on_a_cut_toy_tree(tmp_path, capsys):
    """Smoke: the IR check feeds a last stage like the engine and finds identical outputs on CPU.
    (Its timing gate needs a real head; a toy head is too small to measure, so rc is not asserted.)"""
    import head_cut_check as hc
    _, tree = _tree(tmp_path, lora=True)
    clh.cut(str(tree))
    xml = str(tree / "stage0" / "stage.xml")
    hc.main(["--old", xml + ".pre-cut", "--new", xml, "--device", "CPU", "--tokens", "1,7,16",
             "--profile-tokens", "16", "--reps", "1", "--top", "2"])
    out = capsys.readouterr().out
    for t in (1, 7, 16):
        assert f"T={t}: logits max|d| 0.000000 (argmax same), final_hidden max|d| 0" in out
    assert "lm_head time at T=16" in out
    assert hc.main(["--profile-only", "--old", xml + ".pre-cut", "--device", "CPU", "--profile-tokens", "16",
                    "--reps", "1"]) in (0, 3)


def test_head_cut_check_finds_the_backup_weights():
    import head_cut_check as hc
    assert hc.bin_for("T/stage7/stage.xml") == "T/stage7/stage.bin"
    assert hc.bin_for("T/stage7/stage.xml.pre-cut") == "T/stage7/stage.bin.pre-cut"
    with pytest.raises(SystemExit):
        hc.bin_for("T/stage7/stage.bin")


@pytest.mark.parametrize("lora", [False, True])
def test_patched_stage_serializes_like_a_fresh_export(tmp_path, lora):
    """cut_lm_head.py on a saved stage writes the same stage.bin, and the same stage.xml up to the
    auto-numbered names of the toy's unnamed nodes (two builds in one process number them apart;
    the cut's own nodes are named, so a fresh export process and a patch agree on them)."""
    import re
    ov, fresh = _head_model(lora=lora, tmp_path=tmp_path)
    ws.cut_lm_head_to_last_position(fresh)
    (tmp_path / "fresh").mkdir()
    ov.save_model(fresh, str(tmp_path / "fresh" / "stage.xml"), compress_to_fp16=False)
    _, tree = _tree(tmp_path, lora=lora)
    clh.cut(str(tree))
    assert (tree / "stage0" / "stage.bin").read_bytes() == (tmp_path / "fresh" / "stage.bin").read_bytes()
    auto = re.compile(r'"(Constant|Multiply|MatMul|Add|Slice|Result)_\d+(\.\d+)?"')
    a, b = ((d / "stage.xml").read_text() for d in (tree / "stage0", tmp_path / "fresh"))
    assert auto.sub('"N"', a) == auto.sub('"N"', b)
    assert a.count(ws.LM_HEAD_CUT) == b.count(ws.LM_HEAD_CUT) == 5
