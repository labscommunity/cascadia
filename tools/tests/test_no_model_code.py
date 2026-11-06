"""A downloaded model is data: nothing in it, and nothing in the folder someone
runs from, may execute during an export."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))

import cascadia_export as ce  # noqa: E402

# The fields MiniMaxAI/MiniMax-M2's config.json really has; that repo really ships
# auto_map too, so it has to export without its code rather than be refused.
MINIMAX = {
    "model_type": "minimax_m2",
    "hidden_size": 3072,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rotary_dim": 64,
    "num_hidden_layers": 2,
    "num_local_experts": 256,
    "num_experts_per_tok": 8,
    "intermediate_size": 1536,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5000000,
    "vocab_size": 200064,
}
GEMMA = {
    "model_type": "gemma4",
    "text_config": {"model_type": "gemma4_text", "num_hidden_layers": 2},
}


def repo_with_code(tmp_path, config):
    """A model folder whose config asks for its own code. Returns (repo, marker):
    the marker exists only if that code ran."""
    repo = tmp_path / "repo"
    repo.mkdir()
    marker = tmp_path / "code-ran"
    (repo / "own_code.py").write_text(f"open({str(marker)!r}, 'w').close()\nclass Own: pass\n")
    auto_map = {
        "AutoConfig": "own_code.Own",
        "AutoModelForCausalLM": "own_code.Own",
        "AutoTokenizer": ["own_code.Own", None],
    }
    (repo / "config.json").write_text(json.dumps({**config, "auto_map": auto_map}))
    (repo / "tokenizer_config.json").write_text(json.dumps({"auto_map": auto_map}))
    return repo, marker


def shard(repo, tmp_path):
    """`cascadia shard` on `repo`, through the same routing the binary runs."""
    env = {**os.environ, "HF_HOME": str(tmp_path / "hf"), "HF_HUB_OFFLINE": "1"}
    return subprocess.run(
        [sys.executable, "-u", str(TOOLS / "export_shards.py"), "--model", str(repo),
         "--output-dir", str(tmp_path / "out"), "--num-stages", "2", "--quantization", "fp16"],
        capture_output=True, text=True, timeout=600, env=env, stdin=subprocess.DEVNULL,
    )


@pytest.mark.parametrize("config", [MINIMAX, GEMMA], ids=["minimax_m2", "gemma4"])
def test_shard_never_runs_code_from_the_model(tmp_path, config):
    repo, marker = repo_with_code(tmp_path, config)
    r = shard(repo, tmp_path)
    assert not marker.exists(), f"the model's code ran\n{r.stdout[-800:]}\n{r.stderr[-800:]}"


def test_minimax_keeps_its_rotary_size_without_the_vendor_code(tmp_path):
    """The vendor's code turned `rotary_dim` into a factor and transformers' own class
    ignores it. Reading 128 instead of 64 would export a wrong model, with no error."""
    repo, _ = repo_with_code(tmp_path, MINIMAX)
    r = shard(repo, tmp_path)
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("[cfg] "))
    assert json.loads(line.removeprefix("[cfg] "))["partial_rotary_dim"] == 64


# --- Legacy .bin weights -------------------------------------------------------


class _WritesAFile:
    """Pickles to a call of open(path, "w"), which a full unpickle would make."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))


def _load(model_dir):
    import export_shards

    return export_shards.load_stage_weights(str(model_dir), 0, 1, True, False, False)


def test_a_bin_file_holding_more_than_weights_is_refused(tmp_path):
    torch = pytest.importorskip("torch")
    marker = tmp_path / "code-ran"
    torch.save({"model.layers.0.w": _WritesAFile(str(marker))}, tmp_path / "pytorch_model.bin")
    with pytest.raises(SystemExit, match="safetensors"):
        _load(tmp_path)
    assert not marker.exists(), "unpickling the file ran its code"


@pytest.mark.parametrize("zipped", [True, False], ids=["current-format", "legacy-format"])
def test_plain_bin_weights_still_load(tmp_path, zipped):
    """The legacy format can't be memory-mapped, so it takes the retry path."""
    torch = pytest.importorskip("torch")
    weights = {"model.layers.0.w": torch.ones(2, 2), "model.layers.5.w": torch.zeros(1)}
    torch.save(weights, tmp_path / "pytorch_model.bin", _use_new_zipfile_serialization=zipped)
    loaded = _load(tmp_path)
    assert list(loaded) == ["model.layers.0.w"]
    assert torch.equal(loaded["model.layers.0.w"], weights["model.layers.0.w"])


# --- The rest of the tree ------------------------------------------------------


def _shipped_sources():
    return [p for p in TOOLS.rglob("*.py") if "tests" not in p.relative_to(TOOLS).parts]


@pytest.mark.parametrize(
    "keyword,value", [("trust_remote_code", True), ("weights_only", False)]
)
def test_no_exporter_opts_in_to_running_model_code(keyword, value):
    found = [
        f"{path.relative_to(TOOLS)}:{node.lineno}"
        for path in _shipped_sources()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == keyword and isinstance(kw.value, ast.Constant) and kw.value.value is value
    ]
    assert not found, f"{keyword}={value}: {found}"


# --- The folder someone runs from ----------------------------------------------


def _argv_of_optimum_export(monkeypatch):
    monkeypatch.setattr(ce, "python_refusal", lambda: None)
    calls = []
    monkeypatch.setattr(ce.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    ce.optimum_export("/model", "/out", ce.VLM_TASK, "int4")
    return calls[0]


def test_the_optimum_step_ignores_the_working_folder(tmp_path, monkeypatch):
    pytest.importorskip("optimum.commands.optimum_cli")
    argv = _argv_of_optimum_export(monkeypatch)
    for name in ("json", "runpy", "argparse", "optimum", "transformers", "torch"):
        (tmp_path / f"{name}.py").write_text("open(__file__ + '.ran', 'w').close()\n")
    # Only the launcher: --help stops before any export work.
    launcher = argv[: argv.index("export")]
    r = subprocess.run(launcher + ["--help"], cwd=tmp_path, capture_output=True, text=True,
                       timeout=300, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr[-800:]
    assert "usage" in r.stdout.lower()
    assert not list(tmp_path.glob("*.ran")), r.stderr[-800:]
