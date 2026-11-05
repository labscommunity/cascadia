"""`cascadia shard` reaches the sparse-MoE exporters, from the binary alone."""
import ast
import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

TOOLS = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, TOOLS)

import export_moe  # noqa: E402


def args(**kw):
    base = dict(
        model="/x", output_dir="/out", target="cpu", stage=None, layer_split=None, revision=None
    )
    return SimpleNamespace(**{**base, **kw})


@pytest.mark.parametrize(
    "model_type,exporter",
    [
        ("glm_moe_dsa", "export_glm5"),
        ("deepseek_v4", "export_deepseek_v4"),
        ("minimax_m2", "export_minimax_m2"),
        ("inkling_mm_model", "export_inkling"),
    ],
)
def test_each_family_routes_to_its_exporter(model_type, exporter):
    assert export_moe.exporter_for(model_type) == exporter
    assert export_moe.exporter_for("", model_type) == exporter


def test_other_models_stay_on_the_dense_path():
    assert export_moe.exporter_for("llama", "") is None


@pytest.mark.parametrize(
    "flags,named",
    [
        ({"target": "npu"}, "--target npu"),
        ({"stage": 1}, "--stage"),
        ({"layer_split": "1,2"}, "--layer-split"),
    ],
)
def test_flags_that_cannot_apply_are_refused(flags, named):
    with pytest.raises(SystemExit, match=named):
        export_moe.run_export(args(**flags), "export_glm5")


def test_runs_the_exporter_beside_it_on_a_local_checkpoint(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(subprocess, "call", lambda cmd: seen.append(cmd) or 0)
    export_moe.run_export(args(model=str(tmp_path)), "export_glm5")
    cmd = seen[0]
    assert cmd[0] == sys.executable
    assert cmd[2].endswith("export_glm5.py") and os.path.isfile(cmd[2])
    assert cmd[3:] == ["--model", str(tmp_path), "--out", "/out"]


def test_a_failed_exporter_fails_the_shard(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "call", lambda cmd: 3)
    with pytest.raises(SystemExit, match="exited with 3"):
        export_moe.run_export(args(model=str(tmp_path)), "export_glm5")


def test_the_sidecar_describes_a_sparse_moe_tree(tmp_path):
    cascadia_export = pytest.importorskip("cascadia_export")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"arch": "glm5", "num_layers": 3, "experts_format": "int4_bin"})
    )
    facts = cascadia_export.tree_facts(str(tmp_path))
    assert facts["stages"] is None, "one tree serves any stage count"
    assert facts["quant"] == "int4"
    assert facts["numLayers"] == 3

    # The format isn't known from the layout alone for an OpenVINO-IR expert tree.
    (tmp_path / "manifest.json").write_text(
        json.dumps({"arch": "minimax_m2", "num_layers": 3, "experts_format": "ov_ir"})
    )
    assert cascadia_export.tree_facts(str(tmp_path))["quant"] is None

    # Other manifests keep counting their stages.
    (tmp_path / "manifest.json").write_text(
        json.dumps({"arch": "qwen3_5_moe", "stages": [{}, {}], "num_layers": 8})
    )
    assert cascadia_export.tree_facts(str(tmp_path))["stages"] == 2


def _local_imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                yield os.path.relpath(
                    os.path.join(os.path.dirname(path), node.module or "__init__"), TOOLS
                ).replace(os.sep, ".")
            elif node.module:
                yield node.module


def _module_file(name):
    base = os.path.join(TOOLS, *name.split("."))
    for candidate in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(candidate):
            return candidate
    return None


def test_the_binary_embeds_everything_the_exporters_import():
    src = open(
        os.path.join(TOOLS, "..", "crates", "cascadia-cli", "src", "lib.rs"), encoding="utf-8"
    ).read()
    # ("name.py", SCRIPT_CONST) and ("dir/name.py", tool!("dir/name.py")) entries.
    embedded = set(re.findall(r'\(\s*"([\w/]+\.py)",\s*(?:\w+|tool!\("[\w/.]+"\)),?\s*\)', src))
    seen, queue = set(), [m + ".py" for m in set(export_moe.EXPORTERS.values())]
    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for name in _local_imports(os.path.join(TOOLS, rel)):
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                found = _module_file(".".join(parts[:i]))
                if found:
                    queue.append(os.path.relpath(found, TOOLS).replace(os.sep, "/"))
    assert seen <= embedded, f"not embedded: {sorted(seen - embedded)}"
