"""`cascadia shard` / `cascadia export` routing for vision-language models
(tools/cascadia_export.py): which sources get a vision tree, which are
refused, the tree's model_id, and the cascadia-export.json sidecar.

Run from the repo root with:

    python -m pytest tools/tests/test_vlm_export_routing.py -v

The cut and the optimum-intel export are stubbed; the tiny round trip at the
end runs both for real when optimum-intel is installed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types

import pytest

pytest.importorskip("openvino")

_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _TOOLS)
sys.path.insert(0, os.path.join(_TOOLS, "vlm_surgery"))
import cascadia_export as ce  # noqa: E402
import cut_vlm_lm as cut  # noqa: E402

# One config.json per cut_vlm_lm family, as the OpenVINO exports carry them.
FAMILY_CONFIGS = {
    "pixtral": {"model_type": "llava", "vision_config": {"model_type": "pixtral"}},
    "mistral3": {"model_type": "mistral3", "vision_config": {"model_type": "pixtral"}},
    "phi3_v": {"model_type": "phi3_v", "img_processor": {"name": "clip_vision_model"}},
    "internvl_chat": {"model_type": "internvl_chat",
                      "vision_config": {"model_type": "intern_vit_6b"}},
    "gemma3": {"model_type": "gemma3", "vision_config": {"model_type": "siglip_vision_model"}},
    "llava_next": {"model_type": "llava_next",
                   "vision_config": {"model_type": "clip_vision_model"}},
    "qwen2_5_vl": {"model_type": "qwen2_5_vl", "vision_config": {"model_type": "qwen2_5_vl"}},
    "qwen3_vl": {"model_type": "qwen3_vl", "vision_config": {"model_type": "qwen3_vl"}},
}

# Vision-language models no cutter handles.
UNCUT_VLMS = {
    "llava-1.5 (CLIP tower)": {"model_type": "llava",
                               "vision_config": {"model_type": "clip_vision_model"}},
    "qwen2_vl": {"model_type": "qwen2_vl", "vision_config": {"model_type": "qwen2_vl"}},
    "mllama": {"model_type": "mllama", "vision_config": {"model_type": "mllama_vision_model"}},
    "paligemma": {"model_type": "paligemma",
                  "vision_config": {"model_type": "siglip_vision_model"}},
    "idefics3": {"model_type": "idefics3", "vision_config": {"model_type": "idefics3_vision"}},
}

TEXT_CONFIG = {"model_type": "llama", "num_hidden_layers": 4}


def test_every_family_table_entry_has_a_fixture():
    assert sorted(FAMILY_CONFIGS) == sorted(cut.FAMILIES)


@pytest.mark.parametrize("family", sorted(FAMILY_CONFIGS))
def test_each_family_maps_from_its_config(family):
    cfg = FAMILY_CONFIGS[family]
    assert ce.is_vlm_config(cfg)
    assert cut.family_for_config(cfg) == family
    cut.check_family({**cfg, "text_config": {}}, family)


@pytest.mark.parametrize("name", sorted(UNCUT_VLMS))
def test_uncut_vlms_have_no_family(name):
    cfg = UNCUT_VLMS[name]
    assert ce.is_vlm_config(cfg)
    assert cut.family_for_config(cfg) is None


def test_text_models_are_not_vlms():
    assert not ce.is_vlm_config(TEXT_CONFIG)
    assert not ce.is_vlm_config({"model_type": "gemma3_text"})


def _args(model, out, **kw):
    a = dict(model=str(model), output_dir=str(out), num_stages=2, quantization="int4",
             target="cpu-gpu", layer_split=None, stage=None, text_only=False, revision=None)
    a.update(kw)
    return types.SimpleNamespace(**a)


def _ov_export(dirpath, cfg):
    """A dir that reads as an OpenVINO VLM export (empty IRs: the cut is stubbed)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "config.json").write_text(json.dumps(cfg))
    for f in ("openvino_language_model.xml", ce.VISION_IR, "openvino_text_embeddings_model.xml"):
        (dirpath / f).write_text("")
    return dirpath


@pytest.fixture
def cuts(monkeypatch):
    calls = []
    monkeypatch.setattr(ce, "_cut", lambda src, args, family, model_id:
                        calls.append((src, family, model_id)))
    return calls


def test_text_and_text_only_fall_through(tmp_path, cuts):
    assert ce.shard_vlm(_args("m", tmp_path), TEXT_CONFIG) is None
    cfg = FAMILY_CONFIGS["qwen3_vl"]
    assert ce.shard_vlm(_args("m", tmp_path, text_only=True), cfg) is None
    assert cuts == []


@pytest.mark.parametrize("name", sorted(UNCUT_VLMS))
def test_uncut_vlm_is_refused_naming_the_way_out(tmp_path, cuts, name):
    with pytest.raises(SystemExit) as e:
        ce.shard_vlm(_args("m", tmp_path), UNCUT_VLMS[name])
    msg = str(e.value)
    assert repr(UNCUT_VLMS[name]["model_type"]) in msg
    assert "--text-only" in msg and "cascadia export" in msg
    assert cuts == []


@pytest.mark.parametrize("extra,needle", [({"target": "npu"}, "--target npu"),
                                          ({"layer_split": "8"}, "--layer-split")])
def test_cut_path_refuses_npu_and_layer_split(tmp_path, cuts, extra, needle):
    src = _ov_export(tmp_path / "src", FAMILY_CONFIGS["gemma3"])
    with pytest.raises(SystemExit, match=needle):
        ce.shard_vlm(_args(src, tmp_path / "out", **extra), FAMILY_CONFIGS["gemma3"])


@pytest.mark.parametrize("family", sorted(FAMILY_CONFIGS))
def test_local_ov_export_is_cut_under_its_dir_name(tmp_path, cuts, family):
    src = _ov_export(tmp_path / f"{family}-int4-ov", FAMILY_CONFIGS[family])
    routed = ce.shard_vlm(_args(src, tmp_path / "out"), FAMILY_CONFIGS[family])
    assert cuts == [(str(src), family, f"{family}-int4-ov")]
    assert routed["identity"] == {"model_id": f"{family}-int4-ov", "repo": None,
                                  "revision": None}


def test_hf_ov_export_takes_the_repo_name_whatever_the_download_dir(tmp_path, monkeypatch, cuts):
    """R2: the tree's model_id comes from the repo, not from where the files
    landed, so two machines cutting one repo write one pipeline_config.json."""
    cfg = FAMILY_CONFIGS["qwen3_vl"]
    repo = "OpenVINO/Qwen3-VL-4B-Instruct-int4-ov"
    monkeypatch.setattr(ce, "source_files", lambda m, r=None: ["config.json", ce.VISION_IR,
                                                               "openvino_language_model.xml"])
    monkeypatch.setattr(ce, "source_identity", lambda m, r=None: {
        "model_id": m.rsplit("/", 1)[-1], "repo": m, "revision": "c0ffee"})
    for name in ("cache-a", "elsewhere-b"):
        d = _ov_export(tmp_path / name, cfg)
        monkeypatch.setattr(ce, "download", lambda r, rev, d=d: str(d))
        ce.shard_vlm(_args(repo, tmp_path / "out"), cfg)
    assert [c[2] for c in cuts] == ["Qwen3-VL-4B-Instruct-int4-ov"] * 2
    assert [c[1] for c in cuts] == ["qwen3_vl"] * 2


def test_cut_tree_model_id_follows_the_caller(tmp_path):
    """cut_vlm_lm.run writes the model_id it is given (and the dir name without one)."""
    import inspect

    assert "model_id" in inspect.signature(cut.run).parameters


def _raw(monkeypatch, cfg):
    monkeypatch.setattr(ce, "source_files", lambda m, r=None: ["config.json",
                                                               "model.safetensors"])
    monkeypatch.setattr(ce, "source_identity", lambda m, r=None: {
        "model_id": "tiny", "repo": None, "revision": None})


def test_raw_checkpoint_refused_without_optimum(tmp_path, monkeypatch, cuts):
    _raw(monkeypatch, FAMILY_CONFIGS["qwen2_5_vl"])
    monkeypatch.setattr(ce, "optimum_exports_vlm", lambda mt: None)
    with pytest.raises(SystemExit) as e:
        ce.shard_vlm(_args("raw", tmp_path), FAMILY_CONFIGS["qwen2_5_vl"])
    assert "optimum-intel" in str(e.value) and "pip install" in str(e.value)


def test_raw_checkpoint_refused_when_optimum_cannot_export_it(tmp_path, monkeypatch, cuts):
    _raw(monkeypatch, FAMILY_CONFIGS["qwen2_5_vl"])
    monkeypatch.setattr(ce, "optimum_exports_vlm", lambda mt: False)
    with pytest.raises(SystemExit, match="optimum-intel cannot export"):
        ce.shard_vlm(_args("raw", tmp_path), FAMILY_CONFIGS["qwen2_5_vl"])


def test_raw_checkpoint_refused_outside_the_exporters_transformers_bounds(
        tmp_path, monkeypatch, cuts):
    _raw(monkeypatch, FAMILY_CONFIGS["qwen2_5_vl"])
    monkeypatch.setattr(ce, "optimum_exports_vlm",
                        lambda mt: "its optimum-intel exporter needs transformers <= 5.0")
    with pytest.raises(SystemExit, match="transformers <= 5.0"):
        ce.shard_vlm(_args("raw", tmp_path), FAMILY_CONFIGS["qwen2_5_vl"])


def test_transformers_bounds_are_read_from_the_export_config():
    import functools

    import transformers

    class Old:
        MAX_TRANSFORMERS_VERSION = "4.0"

    class New:
        MIN_TRANSFORMERS_VERSION = "99.0"

    class Any_:
        pass

    assert "<= 4.0" in ce.transformers_bounds_refusal(functools.partial(Old))
    assert ">= 99.0" in ce.transformers_bounds_refusal(New)
    assert ce.transformers_bounds_refusal(Any_) is None
    assert transformers.__version__ in ce.transformers_bounds_refusal(New)


def test_transformers_pin_follows_the_export_configs_bounds():
    class Two:
        MAX_TRANSFORMERS_VERSION = "5.0"

    class Three:
        MAX_TRANSFORMERS_VERSION = "4.57.6"

    class Floor:
        MIN_TRANSFORMERS_VERSION = "5.5"

    class Both:
        MIN_TRANSFORMERS_VERSION = "4.51.0"
        MAX_TRANSFORMERS_VERSION = "4.53.3"

    # Exactly the bounds `transformers_bounds_refusal` compares: `==5.0.*` would
    # admit 5.0.1, which the `<= 5.0` bound refuses.
    assert ce.transformers_pin(Two) == "transformers<=5.0"
    assert ce.transformers_pin(Three) == "transformers<=4.57.6"
    assert ce.transformers_pin(Floor) == "transformers>=5.5"
    assert ce.transformers_pin(Both) == "transformers>=4.51.0,<=4.53.3"


def _bounds(pin):
    r = ce.BoundsRefusal("its optimum-intel exporter needs transformers <= 5.0 "
                         "(installed: 5.5.4)")
    r.pin = pin
    return r


def test_bounds_refusal_gives_a_side_environment_recipe(tmp_path, monkeypatch, cuts):
    """A Qwen VL raw checkpoint is refused with commands that do export it:
    a separate environment with the exporter's own transformers bound and
    this environment's optimum-intel / OpenVINO, then `cascadia shard` of the
    export directory."""
    pytest.importorskip("optimum.intel", exc_type=ImportError)
    _raw(monkeypatch, FAMILY_CONFIGS["qwen2_5_vl"])
    monkeypatch.setattr(ce, "optimum_exports_vlm", lambda mt: _bounds("transformers<=5.0"))
    with pytest.raises(SystemExit) as e:
        ce.shard_vlm(_args("Qwen/Qwen2.5-VL-3B-Instruct", tmp_path),
                     FAMILY_CONFIGS["qwen2_5_vl"])
    msg = str(e.value)
    import importlib.metadata as md

    assert "'transformers<=5.0'" in msg
    assert f"optimum-intel=={md.version('optimum-intel')}" in msg
    assert f"openvino=={md.version('openvino')}" in msg
    # The command checked end to end: asymmetric int4, no --sym.
    assert ("optimum-cli export openvino --model Qwen/Qwen2.5-VL-3B-Instruct "
            "--task image-text-to-text --weight-format int4 "
            "Qwen2.5-VL-3B-Instruct-int4-ov") in msg
    assert "cascadia shard --model Qwen2.5-VL-3B-Instruct-int4-ov" in msg
    assert "--num-stages 2" in msg
    assert "--revision" not in msg
    assert ".\n" not in msg.split("Export it in", 1)[1], "a stray period after the recipe"
    assert cuts == []


def test_recipe_keeps_the_weight_format_and_quotes_paths(tmp_path, monkeypatch):
    pytest.importorskip("optimum.intel", exc_type=ImportError)
    src = tmp_path / "my model"
    src.mkdir()
    recipe = ce.side_venv_recipe(str(src), "transformers<=5.0", None, "int8")
    assert f"--model '{src}'" in recipe
    assert "--weight-format int8 'my model-int8-ov'" in recipe
    assert "cascadia shard --model 'my model-int8-ov'" in recipe


def test_recipe_survives_an_sdk_openvino(monkeypatch):
    """An OpenVINO without pip metadata (the INSTALL.md SDK layout) still gives a
    recipe: the module's version, cut to its release, or no pin and a note."""
    import importlib.metadata as md

    real = md.version

    def no_meta(dist):
        if dist == "openvino":
            raise md.PackageNotFoundError(dist)
        return real(dist)

    monkeypatch.setattr(md, "version", no_meta)
    import openvino

    assert ce._installed_version("openvino", "openvino") == openvino.__version__.split("-")[0]
    monkeypatch.setattr(ce, "_installed_version",
                        lambda dist, module: None if dist == "openvino" else "2.2.0")
    recipe = ce.side_venv_recipe("org/m", "transformers<=5.0", 2)
    assert " openvino " in recipe + " " and "openvino's version could not be read" in recipe


def test_remote_code_refusal_wins_over_the_bounds_recipe():
    """Phi-3.5-vision and InternVL2 are outside the bounds too, but they ship
    remote code: no recipe may suggest running it."""
    cfg = {**FAMILY_CONFIGS["phi3_v"], "auto_map": {"AutoModel": "x.Y"}}
    assert "auto_map" in ce.raw_export_refusal(cfg, _bounds("transformers<=4.53.3"))
    assert ce.raw_export_recipe(cfg, _bounds("transformers<=4.53.3"), "m/phi", 2) is None


def test_raw_remote_code_checkpoint_is_refused(tmp_path, monkeypatch, cuts):
    cfg = {**FAMILY_CONFIGS["internvl_chat"], "auto_map": {"AutoModel": "x.Y"}}
    _raw(monkeypatch, cfg)
    monkeypatch.setattr(ce, "optimum_exports_vlm", lambda mt: True)
    with pytest.raises(SystemExit, match="auto_map"):
        ce.shard_vlm(_args("raw", tmp_path), cfg)


@pytest.mark.parametrize("quant,wf,sym", [("int4", "int4", True), ("int4_asym", "int4", False),
                                          ("int8", "int8", False), ("fp16", "fp16", False)])
def test_raw_checkpoint_is_exported_then_cut(tmp_path, monkeypatch, cuts, quant, wf, sym):
    _raw(monkeypatch, FAMILY_CONFIGS["qwen2_5_vl"])
    monkeypatch.setattr(ce, "optimum_exports_vlm", lambda mt: True)
    exports = []
    monkeypatch.setattr(ce, "optimum_export", lambda *a, **kw: exports.append((a, kw)))
    routed = ce.shard_vlm(_args("raw", tmp_path, quantization=quant),
                          FAMILY_CONFIGS["qwen2_5_vl"])
    (model, out, task, weight_format), kw = exports[0]
    assert (model, task, weight_format, kw["sym"]) == ("raw", ce.VLM_TASK, wf, sym)
    assert cuts == [(out, "qwen2_5_vl", "tiny")]
    assert routed["source_dir"] is None


def _optimum_argv(monkeypatch, model, revision):
    """Run optimum_export with subprocess.call and download stubbed; return
    (argv, download calls)."""
    calls, downloads = [], []
    monkeypatch.setattr(ce.subprocess, "call", lambda cmd: calls.append(cmd) or 0)

    def _download(repo, rev):
        downloads.append((repo, rev))
        return "/cache/" + repo.replace("/", "--")

    monkeypatch.setattr(ce, "download", _download)
    ce.optimum_export(model, "/out", ce.VLM_TASK, "int4", revision=revision, sym=True)
    return calls[0], downloads


def test_optimum_export_downloads_a_hub_id_at_its_revision(monkeypatch):
    argv, downloads = _optimum_argv(monkeypatch, "llava-hf/llava-1.5-7b-hf", "b234b804")
    assert downloads == [("llava-hf/llava-1.5-7b-hf", "b234b804")]
    assert "--revision" not in argv
    assert argv[argv.index("--model") + 1] == "/cache/llava-hf--llava-1.5-7b-hf"
    assert argv[-1] == "/out"


@pytest.mark.parametrize("revision", [None, "b234b804"])
def test_optimum_export_passes_a_local_dir_as_is(tmp_path, monkeypatch, revision):
    argv, downloads = _optimum_argv(monkeypatch, str(tmp_path), revision)
    assert downloads == []
    assert "--revision" not in argv
    assert argv[argv.index("--model") + 1] == str(tmp_path)


def test_optimum_export_without_a_revision_passes_the_hub_id(monkeypatch):
    argv, downloads = _optimum_argv(monkeypatch, "org/model", None)
    assert downloads == []
    assert "--revision" not in argv
    assert argv[argv.index("--model") + 1] == "org/model"


# --- export -------------------------------------------------------------------

def test_export_leaves_an_openvino_export_alone(tmp_path, capsys):
    src = _ov_export(tmp_path / "src", FAMILY_CONFIGS["qwen3_vl"])
    ce.export(str(src), str(tmp_path / "out"))
    assert "already an OpenVINO export" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def _raw_dir(tmp_path, cfg):
    d = tmp_path / "raw"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_export_refuses_the_qwen35_family(tmp_path):
    src = _raw_dir(tmp_path, {"model_type": "qwen3_5_moe", "vision_config": {}})
    with pytest.raises(SystemExit, match="Qwen3.5"):
        ce.export(str(src), str(tmp_path / "out"))


def test_export_refuses_a_vlm_optimum_cannot_export(tmp_path, monkeypatch):
    src = _raw_dir(tmp_path, UNCUT_VLMS["paligemma"])
    monkeypatch.setattr(ce, "optimum_exports_vlm", lambda mt: False)
    with pytest.raises(SystemExit, match="vision-language"):
        ce.export(str(src), str(tmp_path / "out"))


@pytest.mark.parametrize("cfg,task,mods", [
    (UNCUT_VLMS["qwen2_vl"], ce.VLM_TASK, ["text", "image", "video"]),
    ({**TEXT_CONFIG}, ce.TEXT_TASK, ["text"]),
])
def test_export_picks_the_task_and_writes_the_sidecar(tmp_path, monkeypatch, cfg, task, mods):
    src = _raw_dir(tmp_path, cfg)
    out = tmp_path / "out"
    monkeypatch.setattr(ce, "optimum_exports_vlm", lambda mt: True)

    def _export(model, out_dir, t, wf, revision=None, sym=False):
        assert (t, wf) == (task, "int4")
        os.makedirs(out_dir)
        with open(os.path.join(out_dir, "openvino_model.xml"), "w") as f:
            f.write("x" * 10)

    monkeypatch.setattr(ce, "optimum_export", _export)
    ce.export(str(src), str(out))
    side = json.loads((out / ce.SIDECAR).read_text())
    assert side["inputModalities"] == mods
    assert "stages" not in side
    assert side["source"] == {"repo": None, "revision": None}
    assert side["model_id"] == "raw" and side["sizeBytes"] == 10


# --- sidecar ------------------------------------------------------------------

def _tree(dirpath, cfg):
    dirpath.mkdir(parents=True)
    (dirpath / "pipeline_config.json").write_text(json.dumps(cfg))
    (dirpath / "stage_0").mkdir()
    (dirpath / "stage_0" / "openvino_model.bin").write_bytes(b"\0" * 100)
    return dirpath


def test_shard_sidecar_is_read_from_the_tree_and_stable(tmp_path):
    cfg = {"num_stages": 2, "num_layers": 36, "vision": {"family": "qwen3_vl"}}
    tree = _tree(tmp_path / "tree", cfg)
    ident = {"model_id": "Qwen3-VL-4B-Instruct-int4-ov",
             "repo": "OpenVINO/Qwen3-VL-4B-Instruct-int4-ov", "revision": "abc"}
    src = tmp_path / "src"
    src.mkdir()
    (src / "openvino_config.json").write_text(json.dumps({"dtype": "int4"}))
    ce.write_shard_sidecar(str(src), str(tree), ident)
    first = (tree / ce.SIDECAR).read_bytes()
    ce.write_shard_sidecar(str(src), str(tree), ident)
    assert (tree / ce.SIDECAR).read_bytes() == first, "the sidecar counts itself in sizeBytes"
    side = json.loads(first)
    size = (tree / "pipeline_config.json").stat().st_size + 100
    assert side == {
        "model_id": "Qwen3-VL-4B-Instruct-int4-ov",
        "source": {"repo": "OpenVINO/Qwen3-VL-4B-Instruct-int4-ov", "revision": "abc"},
        "stages": 2, "numLayers": 36, "quant": "int4", "sizeBytes": size,
        "inputModalities": ["text", "image", "video"],
    }
    assert b"\r" not in first


def test_text_tree_sidecar_uses_the_tree_quantization(tmp_path):
    tree = _tree(tmp_path / "tree", {"num_stages": 1, "num_layers": 4, "quantization": "int8"})
    ce.write_shard_sidecar("org/m", str(tree), {"model_id": "m", "repo": "org/m",
                                               "revision": "r"})
    side = json.loads((tree / ce.SIDECAR).read_text())
    assert (side["quant"], side["inputModalities"], side["stages"]) == ("int8", ["text"], 1)


def test_qwen35_manifest_tree(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "manifest.json").write_text(json.dumps(
        {"num_layers": 40, "stages": [{}, {}], "vision": {"embeddings": "x"}}))
    facts = ce.tree_facts(str(tree))
    assert (facts["stages"], facts["numLayers"], facts["vision"]) == (2, 40, True)


def test_no_tree_no_sidecar(tmp_path):
    ce.write_shard_sidecar("m", str(tmp_path), {"model_id": "m", "repo": None,
                                               "revision": None})
    assert not (tmp_path / ce.SIDECAR).exists()


# --- flat temp dir (R1) ---------------------------------------------------------

EMBEDDED = {
    "export_shards.py": "export_shards.py",
    "export_gemma4.py": "export_gemma4.py",
    "model_aliases.py": "model_aliases.py",
    "export_qwen36_moe.py": "qwen36_surgery/export_qwen36_moe.py",
    "export_gemma4_text.py": "gemma4_surgery/export_gemma4_text.py",
    "cut_vlm_lm.py": "vlm_surgery/cut_vlm_lm.py",
    "longrope.py": "vlm_surgery/longrope.py",
    "cascadia_export.py": "cascadia_export.py",
    "requirements.txt": "requirements.txt",
}


def test_scripts_run_from_one_flat_dir(tmp_path):
    """The CLI writes these files side by side into a temp dir
    (write_embedded_exporter); each entry point must import from there."""
    for name, rel in EMBEDDED.items():
        shutil.copy(os.path.join(_TOOLS, rel), tmp_path / name)
    for script in ("cut_vlm_lm.py", "cascadia_export.py"):
        r = subprocess.run([sys.executable, str(tmp_path / script), "--help"],
                           capture_output=True, text=True, cwd=tmp_path)
        assert r.returncode == 0, r.stderr
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); import cascadia_export as c; "
             "print(c.cut_family({'model_type': 'qwen3_vl', "
             "'vision_config': {'model_type': 'qwen3_vl'}})); print(c.install_line())")
    r = subprocess.run([sys.executable, "-I", "-c", probe, str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    family, line = r.stdout.splitlines()
    assert family == "qwen3_vl"
    assert '"optimum-intel>=' in line


# --- tiny round trip ------------------------------------------------------------

def test_tiny_vlm_export_then_cut_round_trip(tmp_path):
    """optimum-intel exports a tiny random LLaVA-NeXT, and the cut makes a
    2-stage vision tree from it. (LLaVA-NeXT because the exporter pins allow
    it: optimum-intel's Qwen VL exporters stop at transformers 5.0.)"""
    pytest.importorskip("optimum.intel", exc_type=ImportError)
    pytest.importorskip("transformers")
    from transformers import LlavaNextConfig, LlavaNextForConditionalGeneration

    if ce.optimum_exports_vlm("llava_next") is not True:
        pytest.skip(f"optimum-intel here: {ce.optimum_exports_vlm('llava_next')}")
    hub = "llava-hf/llava-v1.6-mistral-7b-hf"
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(hub)
    except (OSError, ImportError) as exc:  # offline, or no torchvision
        pytest.skip(f"needs {hub}'s processor files: {exc}")
    cfg = LlavaNextConfig(
        text_config=dict(model_type="mistral", hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, vocab_size=len(proc.tokenizer),
                         max_position_embeddings=4096),
        vision_config=dict(model_type="clip_vision_model", hidden_size=32,
                           intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                           image_size=336, patch_size=14, projection_dim=32),
        image_token_index=proc.tokenizer.convert_tokens_to_ids("<image>"),
        image_grid_pinpoints=proc.image_processor.image_grid_pinpoints,
    )
    raw = tmp_path / "tiny-llava-next"
    LlavaNextForConditionalGeneration(cfg).save_pretrained(raw)
    proc.save_pretrained(raw)
    ov_dir = tmp_path / "ov"
    ce.export(str(raw), str(ov_dir), weight_format="fp16")
    side = json.loads((ov_dir / ce.SIDECAR).read_text())
    assert side["inputModalities"] == ["text", "image", "video"] and "stages" not in side
    assert side["model_id"] == "tiny-llava-next" and side["numLayers"] == 4
    tree = tmp_path / "tree"
    routed = ce.shard_vlm(_args(ov_dir, tree), json.loads((ov_dir / "config.json").read_text()))
    pc = json.loads((tree / "pipeline_config.json").read_text())
    assert (pc["num_stages"], pc["vision"]["family"], pc["model_id"]) == (2, "llava_next", "ov")
    assert routed["identity"]["model_id"] == "ov"
