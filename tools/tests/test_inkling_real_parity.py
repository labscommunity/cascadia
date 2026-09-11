"""End-to-end check of the real-weight parity harness on the tiny Inkling export.

Runs the Rust dumper (`cargo run -p cascadia-engine-sparse-moe --release --example
inkling_layer_dump`) on `crates/cascadia-engine-sparse-moe/tests/fixtures/inkling_export`, then
`tools/inkling_ref/real_layer_parity.py` against transformers on the same int4-dequantised weights:

- K = 4 (every layer, head loaded): every tensor within 0.1 % of the HF row RMS, decode and
  prefill paths alike; the logits argmax at every position matches HF, `reference.json`'s
  first-token argmaxes and (teacher-forced) its 8 greedy ids;
- K = 2: the partial-model path (no head, norm/unembed never loaded).

Skipped without cargo / torch / a transformers that ships Inkling. The first run pays for the
release build of the crate.

    python -m pytest tools/tests/test_inkling_real_parity.py -v
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")
try:
    from transformers import InklingForCausalLM  # noqa: F401
except ImportError:
    pytest.skip("transformers without native Inkling support (needs >= 5.16)", allow_module_level=True)
if shutil.which("cargo") is None:
    pytest.skip("cargo not on PATH", allow_module_level=True)

from inkling_ref import real_layer_parity  # noqa: E402

REPO = Path(_TOOLS_DIR).parent
EXPORT = REPO / "crates" / "cascadia-engine-sparse-moe" / "tests" / "fixtures" / "inkling_export"
# Tight bars for the tiny model vs HF float32 (measured 2026-09-10: rms(diff)/rms <= 0.063 %,
# worst element <= 0.26 % of its row's RMS / <= 0.1 % of the row's max, on every layer, both paths).
TIGHT_RMS = 1e-3     # 0.1 % of the row RMS, energy-wise
TIGHT_ELEM = 5e-3    # 0.5 % of the row RMS for the single worst element (bf16 write-back on each linear)
# HF in bfloat16 vs HF in float32 differs by 17 % of the row RMS at the tiny model's global layer 3
# (bf16 residual + bf16-rounded log-scaled position bias); the Rust shell vs HF-bf16 lands on the
# same number, so the bf16 mode is only ever an argmax-level check.
BF16_TOL = 0.25


@pytest.fixture(scope="module")
def reference():
    if not (EXPORT / "reference.json").exists():
        pytest.skip(f"{EXPORT}/reference.json missing (run export_inkling.py --tiny)")
    return json.loads((EXPORT / "reference.json").read_text())


def run_dump(layers: int, tokens: list[int], out: Path, experts: str | None = None) -> str:
    cmd = [shutil.which("cargo"), "run", "-p", "cascadia-engine-sparse-moe", "--release", "--example",
           "inkling_layer_dump", "--", "--export", str(EXPORT), "--layers", str(layers),
           "--tokens", ",".join(map(str, tokens)), "--out", str(out)]
    if experts:
        cmd += ["--experts", experts]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"inkling_layer_dump failed ({r.returncode}):\n{r.stdout}\n{r.stderr}"
    assert out.exists(), out
    return r.stdout


def _assert_tight(report: dict):
    assert report["pass"], json.dumps(report["tensors"], indent=1)
    for name, m in report["tensors"].items():
        assert m["frac_rms"] <= TIGHT_RMS, f"{name}: rms(diff)/rms {m['frac_rms']:.5f} > {TIGHT_RMS}"
        assert m["frac_max"] <= TIGHT_ELEM, f"{name}: max|diff|/rowRMS {m['frac_max']:.5f} > {TIGHT_ELEM}"
        assert m["cos_min"] > 1 - 1e-5, f"{name}: min cosine {m['cos_min']}"


def test_full_model_matches_hf_and_reference(tmp_path, reference):
    """K = num_layers: teacher-forced prompt + greedy[:-1], so the dump's logits cover every
    reference position."""
    prompt, greedy = reference["prompt_ids"], reference["greedy_ids"]
    tokens = prompt + greedy[:-1]
    dump = tmp_path / "dump_k4.safetensors"
    stdout = run_dump(4, tokens, dump)
    assert "dec-vs-pre" in stdout
    rep = real_layer_parity.compare_dump(EXPORT, 4, dump, dtype="float32", tol=0.02, log=print)
    assert rep["layers"] == rep["num_layers"] == 4
    assert set(rep["tensors"]) == {"embed_out", "logits_decode", "logits_prefill",
                                   *(f"layer{i}_out_{p}" for i in range(4) for p in ("decode", "prefill"))}
    _assert_tight(rep)
    T, P = len(tokens), len(prompt)
    for name in ("logits_decode", "logits_prefill"):
        m = rep["tensors"][name]
        assert m["argmax_agree"] == T
        assert m["argmax_rust"][:P] == reference["first_logits_argmax"]
        assert m["argmax_rust"][P - 1:] == greedy  # position P-1+i predicts greedy[i]
    assert rep["reference"] is not None
    for side in ("rust", "hf"):
        assert rep["reference"][side] == {"first_token_argmax": True, "greedy": True, "greedy_checked": len(greedy)}
    # the two Rust paths are bit-identical to each other
    from safetensors.torch import load_file
    import torch
    d = load_file(str(dump))
    for i in range(4):
        assert torch.equal(d[f"layer{i}_out_decode"], d[f"layer{i}_out_prefill"]), i
    assert torch.equal(d["logits_decode"], d["logits_prefill"])


def test_partial_model_path(tmp_path, reference):
    """K = 2 of 4: no head in the dump, norm/unembed not loaded on the HF side either."""
    tokens = reference["prompt_ids"]
    dump = tmp_path / "dump_k2.safetensors"
    stdout = run_dump(2, tokens, dump)
    assert "(head: no)" in stdout
    rep = real_layer_parity.compare_dump(EXPORT, 2, dump, dtype="float32", tol=0.02, log=print)
    assert rep["layers"] == 2 and rep["num_layers"] == 4
    assert set(rep["tensors"]) == {"embed_out", *(f"layer{i}_out_{p}" for i in range(2) for p in ("decode", "prefill"))}
    assert rep["reference"] is None
    _assert_tight(rep)
    # a K=2 dump must not be compared as K=3
    with pytest.raises(ValueError, match="written for --layers 2"):
        real_layer_parity.compare_dump(EXPORT, 3, dump, log=lambda *_: None)


def test_mmap_experts_within_tolerance(tmp_path, reference):
    """The real-model expert mode (int4 mmap kernels instead of eager f32 dequant) stays within
    the same band and the same argmaxes."""
    prompt, greedy = reference["prompt_ids"], reference["greedy_ids"]
    tokens = prompt + greedy[:-1]
    dump = tmp_path / "dump_k4_mmap.safetensors"
    stdout = run_dump(4, tokens, dump, experts="mmap")
    assert "experts=Mmap" in stdout
    rep = real_layer_parity.compare_dump(EXPORT, 4, dump, dtype="float32", tol=0.02, log=print)
    assert rep["experts"] == "mmap"
    _assert_tight(rep)
    assert rep["reference"]["rust"]["greedy"] and rep["reference"]["rust"]["first_token_argmax"]


def test_bfloat16_reference_path(tmp_path, reference):
    """`--dtype bfloat16` (the memory fallback on the real box) loads on the meta device, keeps the
    convs in f32, agrees on every argmax and stays inside HF's own bf16-vs-f32 band."""
    tokens = reference["prompt_ids"]
    dump = tmp_path / "dump_k4_bf16.safetensors"
    run_dump(4, tokens, dump)
    # bf16 reference = argmax-level check only: the reference itself rounds, so the ULP
    # criterion does not apply and rms is judged at the bf16 tolerance.
    rep = real_layer_parity.compare_dump(EXPORT, 4, dump, dtype="bfloat16", tol=BF16_TOL,
                                         rms_tol=BF16_TOL, scale_tol=BF16_TOL, log=print)
    assert rep["pass"], json.dumps(rep["tensors"], indent=1)
    assert rep["tensors"]["logits_prefill"]["argmax_agree"] == len(tokens)
    assert rep["reference"]["hf"]["greedy"] and rep["reference"]["rust"]["greedy"]
