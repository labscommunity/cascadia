"""Exporter decision options and decision.json/assets writers.

    python -m pytest tools/tests/test_decision_export.py -v
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "qwen36_surgery"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import export_qwen36_moe as ex  # noqa: E402


def test_manifest_records_final_hidden():
    spec = ex.ModelSpec(model_type="qwen3_5", hidden=8, num_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"])
    m = ex.build_manifest(spec, "src", final_hidden=True)
    assert m["final_hidden_output"] is True
    assert list(m)[-2:] == ["final_hidden_output", "stages"]
    assert "final_hidden_output" not in ex.build_manifest(spec, "src")


def test_chat_manifest_is_byte_identical_to_the_pre_decision_exporter():
    """Ranks compare manifest.json byte for byte at HELLO and kv_fingerprint hashes it, so a chat
    export must not grow a `final_hidden_output: false` key a tree from main does not have."""
    spec = ex.ModelSpec(model_type="qwen3_5", hidden=8, num_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"])
    main_manifest = {"arch": "qwen3_5", "family": spec.family, "hidden_size": 8, "num_layers": 4,
                     "layer_types": ["linear_attention"] * 3 + ["full_attention"], "source": "src",
                     "last_logits_only": True, "stages": []}
    assert json.dumps(ex.build_manifest(spec, "src"), indent=2) == json.dumps(main_manifest, indent=2)


# ---------------------------------------------------------------------------
# run_export wiring, with the IR work stubbed out (no model download, no real IR)
# ---------------------------------------------------------------------------

LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"]


class _FakeStage:
    def get_parameters(self):
        return []

    def get_sinks(self):
        return []


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """A model dir with config.json + placeholder IR files; extract_stage, ov.save_model and the
    weight_swap entry points are replaced with recorders."""
    ov = pytest.importorskip("openvino")
    import weight_swap

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": {"model_type": "qwen3_5_text", "hidden_size": 8, "num_hidden_layers": 8,
                        "layer_types": LAYER_TYPES, "vocab_size": 16},
    }))
    for aux in ("openvino_language_model.xml", "openvino_text_embeddings_model.xml",
                "openvino_text_embeddings_model.bin", "tokenizer.json"):
        (model / aux).write_text("stock " + aux)
    calls = []
    monkeypatch.setattr(ex, "extract_stage", lambda xml, spec, a, b, first, last: _FakeStage())
    monkeypatch.setattr(ov, "save_model", lambda *a, **kw: calls.append(("save",)))
    monkeypatch.setattr(weight_swap, "Weights", lambda d: ("W", d))
    reports = []

    def swap_stage(stage, w):
        calls.append(("swap", w))
        reports.append(object())  # per-call sentinel: check_swap must get this stage's report
        return reports[-1]

    def check_swap(report, layers, lt, last):
        assert report is reports[-1]
        assert lt == LAYER_TYPES
        calls.append(("check", list(layers), last))
    monkeypatch.setattr(weight_swap, "swap_stage", swap_stage)
    monkeypatch.setattr(weight_swap, "check_swap", check_swap)
    monkeypatch.setattr(weight_swap, "swap_embeddings",
                        lambda xml, w, out: calls.append(("emb", os.path.basename(xml), w)))
    monkeypatch.setattr(weight_swap, "add_final_hidden_output", lambda stage: calls.append(("fh",)))
    return model, tmp_path / "out", calls, weight_swap, monkeypatch


def _lora(monkeypatch, ws, calls, per_stage, targets, cfg_scale=2.0):
    monkeypatch.setattr(ws, "lora_scale_from_config", lambda f: cfg_scale)
    monkeypatch.setattr(ws, "lora_targets", lambda f: targets)
    monkeypatch.setattr(ws, "graft_lora", lambda stage, f, s: calls.append(("graft", s)) or per_stage)


def test_chat_export_unchanged_without_decision_flags(stubbed):
    model, out, calls, _, _ = stubbed
    ex.run_export(str(model), str(out), 2)
    assert [c[0] for c in calls] == ["save", "save"]
    m = json.loads((out / "manifest.json").read_text())
    assert "final_hidden_output" not in m
    assert (out / "openvino_text_embeddings_model.bin").read_text() == "stock openvino_text_embeddings_model.bin"


def test_weights_swaps_and_checks_every_stage_and_embeddings(stubbed, tmp_path):
    model, out, calls, _, _ = stubbed
    wdir = tmp_path / "ft"
    wdir.mkdir()
    (wdir / "tokenizer.json").write_text("fine-tune tokenizer")
    ft_cfg = (model / "config.json").read_text().replace('"vocab_size": 16', '"vocab_size": 16, "ft": 1')
    (wdir / "config.json").write_text(ft_cfg)
    ex.run_export(str(model), str(out), 2, weights=str(wdir))
    assert [c[0] for c in calls if c[0] in ("swap", "check")] == ["swap", "check", "swap", "check"]
    checks = [c for c in calls if c[0] == "check"]
    assert checks == [("check", [0, 1, 2, 3], False), ("check", [4, 5, 6, 7], True)]
    assert ("emb", "openvino_text_embeddings_model.xml", ("W", str(wdir))) in calls
    assert not (out / "openvino_text_embeddings_model.bin").exists()  # not copied from the stock IR
    assert (out / "tokenizer.json").read_text() == "fine-tune tokenizer"
    assert (out / "config.json").read_text() == ft_cfg
    assert (out / "manifest.json").exists()


def test_weights_with_other_architecture_refused_before_any_work(stubbed, tmp_path):
    model, out, calls, _, _ = stubbed
    wdir = tmp_path / "ft"
    wdir.mkdir()
    cfg = json.loads((model / "config.json").read_text())
    cfg["text_config"]["hidden_size"] = 16
    (wdir / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="not the IR's architecture"):
        ex.run_export(str(model), str(out), 2, weights=str(wdir))
    assert calls == []


@pytest.mark.parametrize("flag", ["weights", "lora"])
def test_validate_refused_with_weights_or_lora_before_any_work(stubbed, tmp_path, flag):
    model, out, calls, _, mp = stubbed
    mp.setattr(ex, "extract_stage", lambda *a: pytest.fail("extract_stage ran"))
    with pytest.raises(SystemExit, match="--validate"):
        ex.run_export(str(model), str(out), 2, validate=True, **{flag: str(tmp_path / "x")})
    assert calls == [] and not out.exists()


def test_lora_graft_total_must_match_adapter_targets(stubbed, tmp_path):
    model, out, calls, ws, mp = stubbed
    _lora(mp, ws, calls, per_stage=3, targets=7)
    out.mkdir()
    (out / "manifest.json").write_text("{}")  # stale manifest from an earlier run in the same dir
    with pytest.raises(SystemExit, match="grafted 6 LoRA branches but the adapter has 7 targets"):
        ex.run_export(str(model), str(out), 2, lora=str(tmp_path / "adapter_model.safetensors"))
    assert not (out / "manifest.json").exists()


def test_embeddings_swap_failure_leaves_no_manifest(stubbed, tmp_path):
    model, out, calls, ws, mp = stubbed

    def boom(*a):
        raise ValueError("no embedding Constant")
    mp.setattr(ws, "swap_embeddings", boom)
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        ex.run_export(str(model), str(out), 2, weights=str(tmp_path / "ft"))
    assert not (out / "manifest.json").exists()


def test_lora_scale_comes_from_adapter_config(stubbed, tmp_path):
    model, out, calls, ws, mp = stubbed
    _lora(mp, ws, calls, per_stage=3, targets=6, cfg_scale=2.0)
    ex.run_export(str(model), str(out), 2, lora=str(tmp_path / "adapter_model.safetensors"))
    assert [c for c in calls if c[0] == "graft"] == [("graft", 2.0), ("graft", 2.0)]


def test_lora_scale_override_warns(stubbed, tmp_path, capsys):
    model, out, calls, ws, mp = stubbed
    _lora(mp, ws, calls, per_stage=3, targets=6, cfg_scale=2.0)
    ex.run_export(str(model), str(out), 2, lora=str(tmp_path / "a.safetensors"), lora_scale=0.5)
    assert [c for c in calls if c[0] == "graft"] == [("graft", 0.5), ("graft", 0.5)]
    assert "WARNING: --lora-scale 0.5 overrides adapter_config.json's 2.0" in capsys.readouterr().out


def test_lora_config_errors_propagate(stubbed, tmp_path):
    model, out, calls, ws, mp = stubbed

    def refuse(f):
        raise ValueError("adapter_config.json has use_dora: true")
    mp.setattr(ws, "lora_scale_from_config", refuse)
    with pytest.raises(ValueError, match="use_dora"):
        ex.run_export(str(model), str(out), 2, lora=str(tmp_path / "a.safetensors"))


def test_missing_adapter_config_needs_explicit_scale(stubbed, tmp_path):
    model, out, calls, ws, mp = stubbed

    def missing(f):
        raise FileNotFoundError(f)
    mp.setattr(ws, "lora_scale_from_config", missing)
    with pytest.raises(SystemExit, match="pass --lora-scale"):
        ex.run_export(str(model), str(out), 2, lora=str(tmp_path / "a.safetensors"))


def test_final_hidden_only_on_last_stage_and_in_manifest(stubbed):
    model, out, calls, _, _ = stubbed
    ex.run_export(str(model), str(out), 2, final_hidden=True)
    assert [c[0] for c in calls] == ["save", "fh", "save"]
    assert json.loads((out / "manifest.json").read_text())["final_hidden_output"] is True


def test_cli_flags_reach_run_export(monkeypatch):
    seen = {}
    monkeypatch.setattr(ex, "run_export", lambda *a, **kw: seen.update(a=a, kw=kw))
    monkeypatch.setattr(sys, "argv", ["x", "--model", "m", "--out", "o", "--total", "8",
                                      "--weights", "w", "--lora", "l.safetensors",
                                      "--lora-scale", "0.25", "--final-hidden"])
    ex.main()
    assert seen["a"] == ("m", "o", 8, False, None)
    assert seen["kw"] == {"weights": "w", "lora": "l.safetensors", "lora_scale": 0.25, "final_hidden": True}


# ---------------------------------------------------------------------------
# decision_export.py: decision.json profile + decision/ head assets
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

import decision_export as de  # noqa: E402


def test_jev_profile_from_adapter_files(tmp_path):
    adapter = tmp_path / "adapter_vllm"
    adapter.mkdir()
    ids = list(range(100, 124))
    (adapter / "decision_head.json").write_text(json.dumps({
        "bias": [0.1] * 24, "verbalizer_ids": ids,
        "slots": {"num_slots": 24, "ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]},
                  "verbalizers": ["false", "true"] + [str(i) for i in range(6)] + [chr(65 + i) for i in range(16)]}}))
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"per_kind": {"noul": 1.0, "choice": 0.98, "score": 1.01}}))
    out = tmp_path / "tree"
    de.write_jev_profile(str(adapter), str(cal), str(out), max_tokens=4096)
    d = json.loads((out / "decision.json").read_text())
    assert d["family"] == "jev" and d["max_tokens"] == 4096
    assert d["jev"]["verbalizer_ids"] == ids and d["jev"]["slots"]["choice"] == [8, 24]
    assert d["jev"]["temperature"]["choice"] == 0.98
    assert json.loads((out / "decision" / "head.json").read_text()) == {"kind": "verbalizer"}


def test_kev_profile_writes_pointer_weights(tmp_path):
    torch = pytest.importorskip("torch")
    head = {"head": {"q.weight": torch.ones(256, 8), "q.bias": torch.zeros(256),
                     "k.weight": torch.full((256, 8), 2.0), "k.bias": torch.ones(256)},
            "temperature": 2.19}
    pt = tmp_path / "head.pt"
    torch.save(head, pt)
    out = tmp_path / "tree"
    de.write_kev_profile(str(pt), str(out), max_tokens=73728)
    d = json.loads((out / "decision.json").read_text())
    assert d == {"family": "kev", "max_tokens": 73728, "kev": {"temperature": 2.19}}
    kw = np.fromfile(out / "decision" / "pointer_k_w.f32", dtype="<f4")
    assert kw.shape == (256 * 8,) and kw[0] == 2.0
    assert json.loads((out / "decision" / "head.json").read_text()) == {"kind": "pointer", "dp": 256}


def test_kev_pointer_weights_are_row_major_out_by_in(tmp_path):
    # decision.rs `affine` reads w[o * hidden + i]: torch Linear weight (out, in), row-major.
    torch = pytest.importorskip("torch")
    qw = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    head = {"head": {"q.weight": qw, "q.bias": torch.tensor([7.0, 8.0, 9.0]),
                     "k.weight": qw.bfloat16(), "k.bias": torch.zeros(3)}, "temperature": 1.5}
    pt = tmp_path / "head.pt"
    torch.save(head, pt)
    out = tmp_path / "tree"
    de.write_kev_profile(str(pt), str(out))
    d = out / "decision"
    assert np.fromfile(d / "pointer_q_w.f32", dtype="<f4").tolist() == [0, 1, 2, 3, 4, 5]
    assert np.fromfile(d / "pointer_q_b.f32", dtype="<f4").tolist() == [7, 8, 9]
    assert np.fromfile(d / "pointer_k_w.f32", dtype="<f4").tolist() == [0, 1, 2, 3, 4, 5]
    assert json.loads((d / "head.json").read_text()) == {"kind": "pointer", "dp": 3}


class _Pickled:  # any non-allowlisted class is refused by torch.load(weights_only=True)
    pass


def test_kev_head_pt_is_loaded_weights_only(tmp_path):
    torch = pytest.importorskip("torch")
    pt = tmp_path / "head.pt"
    torch.save({"head": {}, "temperature": 1.0, "x": _Pickled()}, pt)
    with pytest.raises(Exception, match="(?i)weights_only|unsupported|global"):
        de.write_kev_profile(str(pt), str(tmp_path / "tree"))


def _clef_checkpoint(tmp_path, vocab=5, hidden=3, key="lm_head.weight", tied=False):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    ck = tmp_path / "clef"
    ck.mkdir()
    lm = torch.arange(vocab * hidden, dtype=torch.float32).reshape(vocab, hidden).bfloat16()
    save_file({key: lm}, str(ck / "model-00001.safetensors"))
    (ck / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "model-00001.safetensors"}}))
    (ck / "config.json").write_text(json.dumps({"tie_word_embeddings": tied, "text_config": {}}))
    return ck, lm


def test_clef_profile_writes_lm_head_f16_and_joint_head(tmp_path, monkeypatch):
    from decision import clef_head
    ck, lm = _clef_checkpoint(tmp_path)
    seen = []
    monkeypatch.setattr(clef_head, "export_joint_head",
                        lambda d, xml: seen.append((d, xml)) or open(xml, "w").close())
    out = tmp_path / "tree"
    de.write_clef_profile(str(ck), str(out), max_tokens=16384, chunk=2)  # chunk < vocab: streamed
    d = out / "decision"
    assert json.loads((out / "decision.json").read_text()) == {"family": "clef", "max_tokens": 16384, "clef": {}}
    assert json.loads((d / "head.json").read_text()) == {"kind": "joint", "vocab": 5}
    assert seen == [(str(ck), str(d / "joint_head.xml"))]
    got = np.fromfile(d / "lm_head.f16", dtype="<f2")
    assert got.shape == (5 * 3,) and got.tolist() == lm.float().flatten().tolist()


def test_clef_tied_embeddings_use_embed_tokens(tmp_path, monkeypatch):
    from decision import clef_head
    ck, lm = _clef_checkpoint(tmp_path, key="model.language_model.embed_tokens.weight", tied=True)
    monkeypatch.setattr(clef_head, "export_joint_head", lambda d, xml: open(xml, "w").close())
    out = tmp_path / "tree"
    de.write_clef_profile(str(ck), str(out))
    assert np.fromfile(out / "decision" / "lm_head.f16", dtype="<f2").tolist() == lm.float().flatten().tolist()


def test_clef_missing_lm_head_refused_without_partial_head_json(tmp_path, monkeypatch):
    from decision import clef_head
    ck, _ = _clef_checkpoint(tmp_path, key="model.language_model.embed_tokens.weight", tied=False)
    monkeypatch.setattr(clef_head, "export_joint_head", lambda d, xml: open(xml, "w").close())
    out = tmp_path / "tree"
    with pytest.raises(SystemExit, match="lm_head.weight"):
        de.write_clef_profile(str(ck), str(out))
    assert not (out / "decision" / "head.json").exists()


# CLI wiring: run_export stubbed (no IR), profile writers stubbed or synthetic.

@pytest.fixture
def cli(monkeypatch, tmp_path):
    calls = []

    def fake_run_export(*a, **kw):
        # decision assets must already be in place when the manifest (commit marker) is written
        calls.append(("export", a, kw, os.path.exists(os.path.join(a[1], "decision.json"))))
    monkeypatch.setattr(ex, "run_export", fake_run_export)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 262144, "hidden_size": 8}}))

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["decision_export.py", *argv])
        de.main()
    return run, calls, base, tmp_path


def test_cli_jev_lora_from_adapter_dir_and_no_validate(cli, monkeypatch):
    run, calls, base, tmp = cli
    seen = {}
    monkeypatch.setattr(de, "write_jev_profile",
                        lambda ad, cal, out, max_tokens: seen.update(ad=ad, cal=cal, mt=max_tokens)
                        or de._write(out, {"family": "jev"}, {"kind": "verbalizer"}))
    ad = tmp / "jev" / "adapter_vllm"
    run("jev", "--base-ir", str(base), "--adapter-dir", str(ad), "--out", str(tmp / "t"))
    (_, a, kw, assets_first), = calls
    assert a == (str(base), str(tmp / "t"), 8, False, None)
    assert kw == {"weights": None, "lora": os.path.join(str(ad), "adapter_model.safetensors"),
                  "lora_scale": None, "final_hidden": False}
    assert assets_first
    assert seen == {"ad": str(ad), "cal": str(tmp / "jev" / "calibration.json"), "mt": 32768}


def test_cli_kev_final_hidden_weights_lora(cli, monkeypatch):
    run, calls, base, tmp = cli
    seen = {}
    monkeypatch.setattr(de, "kev_head_meta", lambda pt: {"base": None, "weights": None})
    monkeypatch.setattr(de, "write_kev_profile",
                        lambda pt, out, hidden: seen.update(pt=pt, hidden=hidden) or de._write(out, {"family": "kev"}, {"kind": "pointer"}))
    pt = str(_kev_dir(tmp) / "H.pt")
    run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors", "--lora-scale", "2",
        "--head-pt", pt, "--stages", "2", "--out", str(tmp / "t"))
    (_, a, kw, assets_first), = calls
    assert a == (str(base), str(tmp / "t"), 2, False, None)
    assert kw == {"weights": "W", "lora": "L.safetensors", "lora_scale": 2.0, "final_hidden": True,
                  "tokenizer": str(tmp / "kev" / "tokenizer.json")}
    assert assets_first and seen == {"pt": pt, "hidden": 8}


def test_cli_clef_weights_default_to_clef_dir(cli, monkeypatch):
    run, calls, base, tmp = cli
    monkeypatch.setattr(de, "write_clef_profile",
                        lambda cd, out: de._write(out, {"family": "clef"}, {"kind": "joint"}))
    run("clef", "--base-ir", str(base), "--clef-dir", "C", "--out", str(tmp / "t"))
    (_, a, kw, assets_first), = calls
    assert a[3] is False and kw == {"weights": "C", "lora": None, "lora_scale": None, "final_hidden": True}
    assert assets_first


@pytest.mark.parametrize("family,missing", [("jev", "--adapter-dir"), ("kev", "--head-pt"), ("clef", "--clef-dir")])
def test_cli_family_inputs_required_before_export(cli, family, missing, capsys):
    run, calls, base, tmp = cli
    with pytest.raises(SystemExit):
        run(family, "--base-ir", str(base), "--out", str(tmp / "t"))
    assert missing in capsys.readouterr().err and calls == []


# Fix round 1: stale manifest, writer-side checks, config source for max_tokens.

def _jev_adapter(tmp_path, n_bias=24, choice=(8, 24), per_kind=None):
    adapter = tmp_path / "jev" / "adapter_vllm"
    adapter.mkdir(parents=True)
    (adapter / "decision_head.json").write_text(json.dumps({
        "bias": [0.0] * n_bias, "verbalizer_ids": list(range(24)),
        "slots": {"ranges": {"noul": [0, 2], "score": [2, 8], "choice": list(choice)},
                  "verbalizers": [str(i) for i in range(24)]}}))
    cal = tmp_path / "jev" / "calibration.json"
    cal.write_text(json.dumps({"per_kind": per_kind or {"noul": 1.0, "choice": 1.0, "score": 1.0}}))
    return adapter, cal


def test_cli_removes_stale_manifest_before_writing_assets(cli, monkeypatch):
    run, calls, base, tmp = cli
    out = tmp / "t"
    out.mkdir()
    (out / "manifest.json").write_text("{}")  # earlier export in the same tree

    def early_failure(*a, **kw):  # e.g. --weights architecture mismatch, before run_export's own removal
        raise SystemExit("--weights W is not the IR's architecture")
    monkeypatch.setattr(ex, "run_export", early_failure)
    adapter, _ = _jev_adapter(tmp)
    with pytest.raises(SystemExit, match="architecture"):
        run("jev", "--base-ir", str(base), "--adapter-dir", str(adapter), "--out", str(out))
    assert not (out / "manifest.json").exists()
    assert (out / "decision.json").exists()  # new assets, but no loadable tree


def test_cli_stale_manifest_removed_when_asset_writer_fails(cli, monkeypatch):
    run, calls, base, tmp = cli
    out = tmp / "t"
    out.mkdir()
    (out / "manifest.json").write_text("{}")

    def boom(*a, **kw):
        raise ValueError("bad head.pt")
    monkeypatch.setattr(de, "kev_head_meta", lambda pt: {"base": None, "weights": None})
    monkeypatch.setattr(de, "write_kev_profile", boom)
    with pytest.raises(ValueError):
        run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
            "--head-pt", str(_kev_dir(tmp) / "H.pt"), "--out", str(out))
    assert not (out / "manifest.json").exists() and calls == []


def test_jev_calibration_missing_kind_raises(tmp_path):
    adapter, cal = _jev_adapter(tmp_path, per_kind={"noul": 1.0, "choice": 0.98})
    with pytest.raises(ValueError, match="score"):
        de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"))


def test_jev_length_mismatch_raises(tmp_path):
    adapter, cal = _jev_adapter(tmp_path, n_bias=23)
    with pytest.raises(ValueError, match="lengths differ: 24/24/23"):
        de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"))


def test_jev_slot_range_out_of_bounds_raises(tmp_path):
    adapter, cal = _jev_adapter(tmp_path, choice=(8, 25))
    with pytest.raises(ValueError, match="choice"):
        de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"))


@pytest.mark.parametrize("kind,rng,match", [
    ("noul", (0, 3), "noul .* must be 2 wide"),
    ("score", (2, 7), "score .* must be 6 wide"),
    ("choice", (9, 8), "choice"),
])
def test_jev_slot_widths_mirror_the_server(tmp_path, kind, rng, match):
    # JevCodec::new refuses these at load; the exporter must not write a profile the server rejects
    adapter, cal = _jev_adapter(tmp_path)
    h = json.loads((adapter / "decision_head.json").read_text())
    h["slots"]["ranges"][kind] = list(rng)
    (adapter / "decision_head.json").write_text(json.dumps(h))
    with pytest.raises(ValueError, match=match):
        de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"))


def test_jev_missing_slot_range_raises(tmp_path):
    adapter, cal = _jev_adapter(tmp_path)
    h = json.loads((adapter / "decision_head.json").read_text())
    del h["slots"]["ranges"]["score"]
    (adapter / "decision_head.json").write_text(json.dumps(h))
    with pytest.raises(ValueError, match="score"):
        de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"))


def test_kev_hidden_mismatch_raises(tmp_path):
    torch = pytest.importorskip("torch")
    pt = tmp_path / "head.pt"
    torch.save({"head": {"q.weight": torch.ones(4, 8), "q.bias": torch.zeros(4),
                         "k.weight": torch.ones(4, 8), "k.bias": torch.zeros(4)}, "temperature": 1.0}, pt)
    with pytest.raises(ValueError, match="q.weight has hidden dim 8 but the base IR's hidden_size is 16"):
        de.write_kev_profile(str(pt), str(tmp_path / "tree"), hidden=16)
    de.write_kev_profile(str(pt), str(tmp_path / "ok"), hidden=8)


def test_cli_jev_max_tokens_from_weights_config(cli, monkeypatch):
    run, calls, base, tmp = cli
    seen = {}
    monkeypatch.setattr(de, "write_jev_profile",
                        lambda ad, cal, out, max_tokens: seen.update(mt=max_tokens))
    w = tmp / "ft"
    w.mkdir()
    (w / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 4096}}))
    run("jev", "--base-ir", str(base), "--weights", str(w), "--adapter-dir", str(tmp / "a"), "--out", str(tmp / "t"))
    assert seen == {"mt": 4096}  # the fine-tune's config (shipped in the tree), not the base IR's 262144


# Scoped-review fixes: Kev ships Kev's own tokenizer, and a kev export needs its backbone + adapter.

def test_tokenizer_override_replaces_the_weights_tokenizer_before_the_manifest(stubbed, tmp_path):
    model, out, calls, _, monkeypatch = stubbed
    wdir = tmp_path / "ft"
    wdir.mkdir()
    (wdir / "tokenizer.json").write_text("base tokenizer")
    (wdir / "config.json").write_text((model / "config.json").read_text())
    kev_tok = tmp_path / "kev" / "tokenizer.json"
    kev_tok.parent.mkdir()
    kev_tok.write_text("kev tokenizer")
    import shutil
    real_copy = shutil.copy2
    seen = []

    def copy2(src, dst):
        if os.path.basename(dst) == "tokenizer.json":
            seen.append((str(src), (out / "manifest.json").exists()))
        return real_copy(src, dst)
    monkeypatch.setattr(shutil, "copy2", copy2)
    ex.run_export(str(model), str(out), 2, weights=str(wdir), tokenizer=str(kev_tok))
    assert (out / "tokenizer.json").read_text() == "kev tokenizer"
    assert seen[-1] == (str(kev_tok), False)  # the override lands before the manifest (commit marker)
    assert (out / "manifest.json").exists()


def test_missing_tokenizer_override_refused_before_any_work(stubbed, tmp_path):
    model, out, calls, _, _ = stubbed
    with pytest.raises(SystemExit, match="tokenizer"):
        ex.run_export(str(model), str(out), 2, tokenizer=str(tmp_path / "nope" / "tokenizer.json"))
    assert calls == [] and not (out / "manifest.json").exists()


def _kev_dir(tmp, tokenizer=True):
    d = tmp / "kev"
    d.mkdir(exist_ok=True)
    if tokenizer:
        (d / "tokenizer.json").write_text("kev tokenizer")
    return d


def _stub_kev(monkeypatch, meta=None):
    monkeypatch.setattr(de, "kev_head_meta", lambda pt: meta or {"base": "Qwen/Qwen3.5-9B-Base", "weights": "lora"})
    monkeypatch.setattr(de, "write_kev_profile",
                        lambda pt, out, hidden: de._write(out, {"family": "kev"}, {"kind": "pointer"}))


def test_cli_kev_ships_the_tokenizer_beside_head_pt(cli, monkeypatch):
    run, calls, base, tmp = cli
    _stub_kev(monkeypatch)
    kd = _kev_dir(tmp)
    run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
        "--head-pt", str(kd / "head.pt"), "--out", str(tmp / "t"))
    (_, a, kw, _), = calls
    assert kw["tokenizer"] == str(kd / "tokenizer.json")


def test_cli_tokenizer_flag_overrides(cli, monkeypatch):
    run, calls, base, tmp = cli
    _stub_kev(monkeypatch)
    kd = _kev_dir(tmp, tokenizer=False)
    other = tmp / "other.json"
    other.write_text("t")
    run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
        "--head-pt", str(kd / "head.pt"), "--tokenizer", str(other), "--out", str(tmp / "t"))
    (_, a, kw, _), = calls
    assert kw["tokenizer"] == str(other)


def test_cli_kev_without_its_tokenizer_refused_before_any_write(cli, monkeypatch, capsys):
    run, calls, base, tmp = cli
    _stub_kev(monkeypatch)
    kd = _kev_dir(tmp, tokenizer=False)
    with pytest.raises(SystemExit):
        run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
            "--head-pt", str(kd / "head.pt"), "--out", str(tmp / "t"))
    assert "tokenizer.json" in capsys.readouterr().err
    assert calls == [] and not (tmp / "t").exists()


@pytest.mark.parametrize("drop", ["--weights", "--lora"])
def test_cli_kev_needs_weights_and_lora(cli, monkeypatch, capsys, drop):
    run, calls, base, tmp = cli
    _stub_kev(monkeypatch)
    kd = _kev_dir(tmp)
    argv = {"--weights": "W", "--lora": "L.safetensors"}
    del argv[drop]
    with pytest.raises(SystemExit):
        run("kev", "--base-ir", str(base), *[x for kv in argv.items() for x in kv],
            "--head-pt", str(kd / "head.pt"), "--out", str(tmp / "t"))
    err = capsys.readouterr().err
    assert drop in err and "Qwen/Qwen3.5-9B-Base" in err
    assert calls == [] and not (tmp / "t").exists()


def test_cli_kev_head_pt_recording_full_weights_refused(cli, monkeypatch, capsys):
    run, calls, base, tmp = cli
    _stub_kev(monkeypatch, meta={"base": "B", "weights": "full"})
    kd = _kev_dir(tmp)
    with pytest.raises(SystemExit):
        run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
            "--head-pt", str(kd / "head.pt"), "--out", str(tmp / "t"))
    assert "'full'" in capsys.readouterr().err and calls == []


def test_kev_head_meta_reads_base_and_weights_mode(tmp_path):
    torch = pytest.importorskip("torch")
    pt = tmp_path / "head.pt"
    torch.save({"head": {}, "temperature": 1.0, "base": "Qwen/Qwen3.5-9B-Base", "weights": "lora"}, pt)
    assert de.kev_head_meta(str(pt)) == {"base": "Qwen/Qwen3.5-9B-Base", "weights": "lora"}
    torch.save({"head": {}, "temperature": 1.0}, pt)
    assert de.kev_head_meta(str(pt)) == {"base": None, "weights": None}
    torch.save({"head": {}, "temperature": 1.0, "x": _Pickled()}, pt)
    with pytest.raises(Exception, match="(?i)weights_only|unsupported|global"):
        de.kev_head_meta(str(pt))


def test_cli_jev_and_clef_pass_no_tokenizer_by_default(cli, monkeypatch):
    run, calls, base, tmp = cli
    monkeypatch.setattr(de, "write_clef_profile",
                        lambda cd, out: de._write(out, {"family": "clef"}, {"kind": "joint"}))
    run("clef", "--base-ir", str(base), "--clef-dir", "C", "--out", str(tmp / "t"))
    (_, a, kw, _), = calls
    assert "tokenizer" not in kw


@pytest.mark.parametrize("content", [None, b"not a torch file", "list"])
def test_cli_bad_head_pt_is_a_clean_error(cli, monkeypatch, capsys, content):
    # A missing, unreadable or non-dict head.pt must exit with a message, not a traceback.
    run, calls, base, tmp = cli
    kd = _kev_dir(tmp)
    pt = kd / "head.pt"
    if content == "list":
        torch = pytest.importorskip("torch")
        torch.save([1, 2], pt)
    elif content is not None:
        pt.write_bytes(content)
    with pytest.raises(SystemExit) as e:
        run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
            "--head-pt", str(pt), "--out", str(tmp / "t"))
    assert e.value.code == 2
    assert str(pt) in capsys.readouterr().err
    assert calls == [] and not (tmp / "t").exists()
