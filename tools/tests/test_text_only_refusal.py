"""`cascadia shard` fails clean on trees the generic text exporter cannot cut
(tools/export_shards.py): it refuses up front, before writing anything, and
a run that fails part-way leaves no pipeline_config.json.

Run from the repo root with:

    python -m pytest tools/tests/test_text_only_refusal.py -v

Every source is a stub dir (config.json plus empty files named like the real
ones); nothing is downloaded and the stage export is stubbed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pytest

_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

try:
    import export_shards
except ImportError as exc:
    if "torch" in str(exc) or "numpy" in str(exc):
        pytest.skip(
            f"export_shards.py needs torch/numpy at import time: {exc}",
            allow_module_level=True,
        )
    raise

QWEN3_VL = {"model_type": "qwen3_vl", "vision_config": {"model_type": "qwen3_vl"},
            "text_config": {"model_type": "qwen3_vl_text"}}
QWEN2_5_VL = {"model_type": "qwen2_5_vl", "vision_config": {"model_type": "qwen2_5_vl"}}
TINY_LLAMA = {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
              "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
              "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": 128}


@pytest.fixture(autouse=True)
def _restore_torch_default_dtype():
    """`export_shards.main()` sets torch's default dtype; undo it so other
    test modules are not poisoned by test ordering."""
    torch = pytest.importorskip("torch")
    before = torch.get_default_dtype()
    yield
    torch.set_default_dtype(before)


def _stub(dirpath, config, files=(), weight_map=None):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "config.json").write_text(json.dumps(config))
    for name in files:
        (dirpath / name).write_text("")
    if weight_map is not None:
        (dirpath / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors"
                                       for k in weight_map}}))
        (dirpath / "model-00001-of-00001.safetensors").write_text("")
    return dirpath


def _argv(model, out, *extra):
    return ["export_shards.py", "--model", str(model), "--output-dir", str(out),
            "--num-stages", "2", *extra]


def _refused(tmp_path, monkeypatch, model, *extra) -> str:
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", _argv(model, out, *extra))
    with pytest.raises(SystemExit) as e:
        export_shards.main()
    assert not out.exists(), "a refusal must not write anything to the output dir"
    msg = str(e.value)
    assert "\n" not in msg, f"one clear line, got: {msg!r}"
    return msg


def test_openvino_vlm_export_with_text_only_is_refused_up_front(tmp_path, monkeypatch):
    model = _stub(tmp_path / "ir", QWEN3_VL,
                  files=("openvino_language_model.xml", "openvino_language_model.bin",
                         "openvino_vision_embeddings_model.xml"))
    msg = _refused(tmp_path, monkeypatch, model, "--text-only")
    assert "`cascadia shard --text-only` cannot cut" in msg
    assert "OpenVINO export" in msg and "drop --text-only" in msg


def test_openvino_text_export_is_refused_up_front(tmp_path, monkeypatch):
    model = _stub(tmp_path / "ir", TINY_LLAMA,
                  files=("openvino_model.xml", "openvino_model.bin"))
    msg = _refused(tmp_path, monkeypatch, model)
    assert msg.startswith("`cascadia shard` cannot cut") and "OpenVINO export" in msg
    assert "--text-only" not in msg


def test_raw_qwen3_vl_with_text_only_is_refused_up_front(tmp_path, monkeypatch):
    # Qwen3-VL keeps its text tower under model.language_model.layers.*.
    model = _stub(tmp_path / "raw", QWEN3_VL,
                  weight_map=["model.language_model.layers.0.self_attn.q_proj.weight",
                              "model.visual.blocks.0.attn.qkv.weight"])
    msg = _refused(tmp_path, monkeypatch, model, "--text-only")
    assert "`cascadia shard --text-only` cannot cut" in msg
    assert "qwen3_vl" in msg and "model.layers.*" in msg


def test_vlm_with_model_layers_weights_still_passes_the_check(tmp_path):
    # Qwen2.5-VL ships its text tower as model.layers.*: the text exporter
    # reads it, so --text-only must keep working.
    model = _stub(tmp_path / "raw", QWEN2_5_VL,
                  weight_map=["model.layers.0.self_attn.q_proj.weight",
                              "visual.blocks.0.attn.qkv.weight"])
    args = argparse.Namespace(model=str(model), revision=None, text_only=True)
    assert export_shards.text_exporter_refusal(args, QWEN2_5_VL) is None


def _run_with_stub_stages(tmp_path, monkeypatch, fail_stage):
    pytest.importorskip("transformers")
    model = _stub(tmp_path / "llama", TINY_LLAMA)
    out = tmp_path / "out"
    monkeypatch.setattr(export_shards, "maybe_download", lambda m, r=None: str(model))
    monkeypatch.setattr(export_shards, "copy_tokenizer", lambda src, dst: None)

    def fake_stage(model_dir, output_dir, s, *a, **kw):
        if s["stage"] == fail_stage:
            raise RuntimeError(f"stage {fail_stage} failed")
        os.makedirs(os.path.join(output_dir, f"stage_{s['stage']}"), exist_ok=True)
        return 1.0

    monkeypatch.setattr(export_shards, "export_single_stage", fake_stage)
    monkeypatch.setattr(sys, "argv", _argv(model, out))
    return out


def test_failed_stage_leaves_no_pipeline_config(tmp_path, monkeypatch):
    out = _run_with_stub_stages(tmp_path, monkeypatch, fail_stage=1)
    with pytest.raises(RuntimeError, match="stage 1 failed"):
        export_shards.main()
    assert (out / "stage_0").is_dir()
    assert not (out / "pipeline_config.json").exists()
    assert not (out / "pipeline_config.json.tmp").exists()


def test_pipeline_config_is_written_after_every_stage(tmp_path, monkeypatch):
    out = _run_with_stub_stages(tmp_path, monkeypatch, fail_stage=None)
    export_shards.main()
    meta = json.loads((out / "pipeline_config.json").read_text())
    assert meta["num_stages"] == 2 and meta["num_layers"] == 2
    assert not (out / "pipeline_config.json.tmp").exists()
