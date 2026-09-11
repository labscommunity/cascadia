"""Tests for tools/export_inkling.py (+ tools/inkling_ref).

Run from the repo root:

    python -m pytest tools/tests/test_inkling_export.py -v

Covers: the config contract (real config.json semantics accepted, non-sigmoid gate rejected),
the de-interleave rule vs transformers' own `Interleave` op, the torch int4 packer vs
export_glm5's numpy reference (byte-identical), the --tiny round-trip (bins dequantize within
int4 tolerance, shells lossless, manifest fields, reference.json == a direct HF run on the
dequantized weights), idempotent re-runs, and the streaming --skip-missing-shards /
--delete-consumed-shards passes over a synthetic sharded checkpoint.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")
try:
    from transformers import InklingForCausalLM  # noqa: F401
except ImportError:
    pytest.skip("transformers without native Inkling support (needs >= 5.16)", allow_module_level=True)

import export_inkling  # noqa: E402
import inkling_ref  # noqa: E402
from inkling_ref import (MIN_ARGMAX_MARGIN, TINY_CONFIG, build_tiny_model, deinterleave,  # noqa: E402
                         dequant_expert_bin, greedy_reference, hf_state_to_checkpoint, interleave,
                         load_export_as_hf, select_prompt)


def _real_config():
    """thinkingmachines/Inkling config.json semantics per PORT_SPEC §1 (975B)."""
    return {
        "model_type": "inkling_mm_model",
        "architectures": ["InklingForConditionalGeneration"],
        "text_config": {
            "model_type": "inkling_text",
            "hidden_size": 6144, "num_hidden_layers": 66, "vocab_size": 201024, "unpadded_vocab_size": 200058,
            "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128,
            "swa_num_attention_heads": 64, "swa_num_key_value_heads": 16, "swa_head_dim": 128,
            "d_rel": 16, "rel_extent": 1024, "sliding_window_size": 512,
            "local_layer_ids": [i for i in range(66) if (i + 1) % 6],
            "dense_mlp_idx": 2, "dense_intermediate_size": 24576, "intermediate_size": 3072,
            "n_routed_experts": 256, "num_experts_per_tok": 6, "n_shared_experts": 2, "route_scale": 8.0,
            "rms_norm_eps": 1e-6, "log_scaling_n_floor": 128000, "log_scaling_alpha": 0.1,
            "logits_mup_width_multiplier": 24.0, "sconv_kernel_size": 4, "hidden_act": "silu",
            "eos_token_id": 200006,
            "gate_activation": "sigmoid", "norm_after_topk": True, "use_global_scale": True,
            "use_gate_bias": True, "use_sconv": True, "use_embed_norm": True, "shared_expert_sink": True,
            "q_bias": False, "o_bias": False, "final_logit_softcapping": None,
        },
        "mtp_config": {"num_nextn_predict_layers": 1},
    }


# --------------------------------------------------------------------------
# config contract
# --------------------------------------------------------------------------
def test_validate_accepts_real_config_semantics(tmp_path):
    cfg = _real_config()
    man = export_inkling.load_and_validate_config(cfg, strict=True)
    assert man["arch"] == "inkling"
    assert man["num_layers"] == 66 and man["hidden_size"] == 6144
    assert man["vocab_size"] == 201024 and man["unpadded_vocab_size"] == 200058
    assert (man["num_attention_heads"], man["num_kv_heads"], man["head_dim"]) == (64, 8, 128)
    assert (man["swa_num_attention_heads"], man["swa_num_kv_heads"], man["swa_head_dim"]) == (64, 16, 128)
    assert man["d_rel"] == 16 and man["rel_extent"] == 1024 and man["sliding_window"] == 512
    assert len(man["layer_types"]) == 66
    assert [i for i, t in enumerate(man["layer_types"]) if t == "global"] == [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
    assert man["dense_layers"] == [0, 1]
    assert man["dense_intermediate"] == 24576 and man["moe_intermediate"] == 3072
    assert (man["num_experts"], man["top_k"], man["n_shared_experts"], man["route_scale"]) == (256, 6, 2, 8.0)
    assert man["log_scaling_n_floor"] == 128000 and man["log_scaling_alpha"] == 0.1
    assert man["logits_mup_width_multiplier"] == 24.0 and man["conv_kernel_size"] == 4
    assert man["eos_token_ids"] == [200006] and man["hidden_act"] == "silu"
    assert man["experts_format"] == "int4_bin" and man["shell_backend"] == "rust_inkling"
    assert man["has_mtp"] is False
    # the CLI prints the same manifest
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, os.path.join(_TOOLS_DIR, "export_inkling.py"), "--validate", str(p)],
                       capture_output=True, text=True, check=True)
    printed = json.loads(r.stdout[: r.stdout.index("\n}") + 2])
    assert printed == man
    assert "[validate] OK" in r.stdout


def test_validate_layer_type_default_and_explicit_lists():
    cfg = _real_config()
    del cfg["text_config"]["local_layer_ids"]  # HF default: (i+1) % 6 != 0 -> sliding
    man = export_inkling.load_and_validate_config(cfg)
    assert man["layer_types"][5] == "global" and man["layer_types"][4] == "sliding"
    cfg["text_config"]["layer_types"] = ["hybrid"] * 66
    cfg["text_config"]["mlp_layer_types"] = ["sparse"] * 66
    man = export_inkling.load_and_validate_config(cfg)
    assert set(man["layer_types"]) == {"global"} and man["dense_layers"] == [] and man["dense_intermediate"] == 0


def test_validate_rejects_non_sigmoid_gate():
    cfg = _real_config()
    cfg["text_config"]["gate_activation"] = "softmax"
    with pytest.raises(SystemExit, match="gate_activation"):
        export_inkling.load_and_validate_config(cfg)


@pytest.mark.parametrize("key,bad", [("norm_after_topk", False), ("use_sconv", False), ("q_bias", True),
                                     ("final_logit_softcapping", 30.0), ("hidden_size", 6145),
                                     ("intermediate_size", 3000)])
def test_validate_rejects_other_contract_breaks(key, bad):
    cfg = _real_config()
    cfg["text_config"][key] = bad
    with pytest.raises(SystemExit):
        export_inkling.load_and_validate_config(cfg)


def test_validate_strict_requires_contract_keys():
    cfg = _real_config()
    del cfg["text_config"]["use_gate_bias"]
    export_inkling.load_and_validate_config(cfg)  # assumed with a warning
    with pytest.raises(SystemExit, match="strict"):
        export_inkling.load_and_validate_config(cfg, strict=True)


# --------------------------------------------------------------------------
# de-interleave rule vs transformers' Interleave op; int4 packer vs glm5 reference
# --------------------------------------------------------------------------
def test_deinterleave_matches_transformers_interleave():
    from transformers.core_model_loading import Interleave

    E, I, H = 3, 5, 4
    torch.manual_seed(1)
    w13 = torch.randn(E, 2 * I, H)
    gu = Interleave(dim=1).convert({"w13": w13}, ["w13"], ["gate_up"])["gate_up"]
    gate, up = deinterleave(w13, dim=1)
    assert torch.equal(gu[:, :I], gate) and torch.equal(gu[:, I:], up)
    assert torch.equal(gate, w13[:, 0::2]) and torch.equal(up, w13[:, 1::2])
    assert torch.equal(interleave(gate, up, dim=1), w13)
    # dense w13_dn uses Interleave(dim=0)
    w = torch.randn(2 * I, H)
    gu0 = Interleave(dim=0).convert({"w": w}, ["w"], ["gu"])["gu"]
    g0, u0 = deinterleave(w, dim=0)
    assert torch.equal(gu0[:I], g0) and torch.equal(gu0[I:], u0) and torch.equal(interleave(g0, u0, 0), w)
    # the reverse op round-trips
    back = Interleave(dim=1, inverse=True).convert({"gu": gu}, ["gu"], ["w13"])["w13"]
    assert torch.equal(back, w13)


def test_hf_checkpoint_name_mapping_round_trips():
    man = export_inkling.load_and_validate_config(TINY_CONFIG)
    model = build_tiny_model(man)
    sd = model.state_dict()
    ckpt = hf_state_to_checkpoint(sd)
    assert "model.llm.embed.weight" in ckpt and "model.llm.unembed.weight" in ckpt
    assert "model.llm.layers.1.mlp.experts.w13_weight" in ckpt
    assert "model.llm.layers.0.mlp.w13_dn.weight" in ckpt and "model.llm.layers.0.mlp.global_scale" in ckpt
    assert tuple(ckpt["model.llm.layers.1.attn.k_sconv.weight"].shape) == (32, 1, 4)
    assert tuple(ckpt["model.llm.layers.1.mlp.gate.weight"].shape) == (10, 64)
    back = inkling_ref.checkpoint_state_to_hf(ckpt)
    assert set(back) == set(sd)
    for k in sd:
        assert torch.equal(back[k], sd[k]), k


def test_pack_int4_matches_glm5_reference():
    import export_glm5

    torch.manual_seed(2)
    for shape in ((64, 64), (33, 96), (3072, 128)):
        w = (torch.randn(*shape) * 0.05).to(torch.bfloat16).float()
        w[0, :32] = 0.0  # an all-zero group -> scale 1.0 branch
        p, s = export_inkling.pack_int4(w)
        rp, rs = export_glm5._pack_int4_grouped(w.numpy())
        assert p == rp and s == rs, shape
    assert export_inkling.int4_bin_bytes(64, 32) == inkling_ref.int4_bin_bytes(64, 32)


# --------------------------------------------------------------------------
# --tiny round-trip
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_export(tmp_path_factory):
    out = tmp_path_factory.mktemp("inkling_tiny")
    summary = export_inkling.export_tiny(out, workers=2)
    return out, summary


def test_tiny_manifest_fields(tiny_export):
    out, summary = tiny_export
    assert summary["complete"] and summary["layers_done"] == 4
    man = json.loads((out / "manifest.json").read_text())
    expect = export_inkling.load_and_validate_config(TINY_CONFIG)
    assert man == expect
    assert man["layer_types"] == ["sliding", "sliding", "sliding", "global"]
    assert man["dense_layers"] == [0] and man["num_experts"] == 8 and man["top_k"] == 2
    assert man["unpadded_vocab_size"] == 120 and man["eos_token_ids"] == [127]
    ref = json.loads((out / "reference.json").read_text())
    assert set(ref) == {"prompt_ids", "greedy_ids", "first_logits_argmax"}
    assert len(ref["prompt_ids"]) == 12 and len(ref["greedy_ids"]) == 8 and len(ref["first_logits_argmax"]) == 12


def test_tiny_file_layout_and_shapes(tiny_export):
    from safetensors.torch import load_file

    out, _ = tiny_export
    emb = load_file(str(out / "embed.safetensors"))
    assert emb["embed.weight"].dtype == torch.bfloat16 and tuple(emb["embed.weight"].shape) == (128, 64)
    assert emb["embed_norm.weight"].dtype == torch.float32 and tuple(emb["embed_norm.weight"].shape) == (64,)
    head = load_file(str(out / "head.safetensors"))
    assert head["unembed.weight"].dtype == torch.bfloat16 and tuple(head["unembed.weight"].shape) == (128, 64)
    assert head["norm.weight"].dtype == torch.float32
    sh0 = load_file(str(out / "shells" / "layer_00.safetensors"))
    sh3 = load_file(str(out / "shells" / "layer_03.safetensors"))
    for sh in (sh0, sh3):
        assert sh["attn.wq_du.weight"].dtype == torch.bfloat16 and tuple(sh["attn.wq_du.weight"].shape) == (64, 64)
        assert tuple(sh["attn.wk_dv.weight"].shape) == (32, 64) and tuple(sh["attn.wr_du.weight"].shape) == (16, 64)
        assert sh["attn.k_sconv.weight"].dtype == torch.float32 and tuple(sh["attn.k_sconv.weight"].shape) == (32, 4)
        assert tuple(sh["attn_sconv.weight"].shape) == (64, 4) and sh["attn_norm.weight"].dtype == torch.float32
    assert tuple(sh0["attn.rel_logits_proj.proj"].shape) == (4, 4)   # sliding: extent == window
    assert tuple(sh3["attn.rel_logits_proj.proj"].shape) == (4, 8)   # global: rel_extent
    assert "mlp.global_scale" in sh0 and "mlp.gate.weight" not in sh0
    assert tuple(sh3["mlp.gate.weight"].shape) == (10, 64) and tuple(sh3["mlp.gate.bias"].shape) == (8,)
    assert tuple(sh3["mlp.gate.global_scale"].shape) == (1,) and sh3["mlp.gate.weight"].dtype == torch.float32
    assert (out / "experts" / "layer_00" / "dense.bin").stat().st_size == export_inkling.int4_bin_bytes(64, 64)
    for li in (1, 2, 3):
        d = out / "experts" / f"layer_{li:02d}"
        assert sorted(p.name for p in d.iterdir()) == \
            [f"expert_{e:03d}.bin" for e in range(8)] + ["expert_shared0.bin", "expert_shared1.bin"]
        assert all(p.stat().st_size == export_inkling.int4_bin_bytes(64, 32) for p in d.iterdir())
    assert not list(out.rglob("*.tmp"))


def test_tiny_bins_dequantize_within_int4_tolerance_and_shells_lossless(tiny_export):
    from safetensors.torch import load_file

    out, _ = tiny_export
    man = json.loads((out / "manifest.json").read_text())
    model = build_tiny_model(man)
    ckpt = hf_state_to_checkpoint(model.state_dict())

    def assert_int4(orig, deq):
        # symmetric int4 group-32: |w - deq| <= scale/2 + |q| * (bf16 rounding of the scale, 2^-9)
        wg = orig.reshape(orig.shape[0], -1, 32)
        scale = wg.abs().amax(-1) / 7.0
        tol = (scale / 2 + 7 * scale * 2 ** -9 + 1e-7)[:, :, None].expand_as(wg)
        assert torch.all((wg - deq.reshape_as(wg)).abs() <= tol)
        # ~9% mean abs error is what int4/g32 does to Gaussian weights; 20% would mean a layout bug
        rel = float((orig - deq).abs().mean()) / float(orig.abs().mean())
        assert 0.02 < rel < 0.2, rel

    w13 = ckpt["model.llm.layers.1.mlp.experts.w13_weight"]
    w2 = ckpt["model.llm.layers.1.mlp.experts.w2_weight"]
    for e in (0, 5):
        g, u, d = dequant_expert_bin(out / "experts" / "layer_01" / f"expert_{e:03d}.bin", 64, 32)
        assert_int4(w13[e, 0::2], g)
        assert_int4(w13[e, 1::2], u)
        assert_int4(w2[e], d)
    sw13 = ckpt["model.llm.layers.1.mlp.shared_experts.shared_w13_weight"]
    for s in (0, 1):
        g, u, d = dequant_expert_bin(out / "experts" / "layer_01" / f"expert_shared{s}.bin", 64, 32)
        assert_int4(sw13[s, 0::2], g)
        assert_int4(sw13[s, 1::2], u)
    g, u, d = dequant_expert_bin(out / "experts" / "layer_00" / "dense.bin", 64, 64)
    assert_int4(ckpt["model.llm.layers.0.mlp.w13_dn.weight"][0::2], g)
    assert_int4(ckpt["model.llm.layers.0.mlp.w13_dn.weight"][1::2], u)
    assert_int4(ckpt["model.llm.layers.0.mlp.w2_md.weight"], d)
    # shells: bf16 bits and f32 values are exactly the checkpoint's (weights are bf16-exact)
    for li in range(4):
        sh = load_file(str(out / "shells" / f"layer_{li:02d}.safetensors"))
        for suf, t in sh.items():
            orig = ckpt[f"model.llm.layers.{li}.{suf}"].float()
            if suf.endswith("sconv.weight"):
                orig = orig.reshape(t.shape)
            assert torch.equal(t.float(), orig), (li, suf)
    emb = load_file(str(out / "embed.safetensors"))
    assert torch.equal(emb["embed.weight"].float(), ckpt["model.llm.embed.weight"])


def test_tiny_reference_matches_direct_hf_run_on_dequantized_weights(tiny_export):
    out, _ = tiny_export
    ref = json.loads((out / "reference.json").read_text())
    model, man = load_export_as_hf(out)
    assert ref["prompt_ids"] == select_prompt(man)[0]
    direct, margins = greedy_reference(model, ref["prompt_ids"], 8)
    assert direct == ref
    assert min(margins["prompt"]) >= MIN_ARGMAX_MARGIN and min(margins["greedy"]) >= MIN_ARGMAX_MARGIN
    with torch.no_grad():
        logits = model(torch.tensor([ref["prompt_ids"]]), use_cache=False).logits[0]
    assert logits.shape == (12, 120)
    assert logits.argmax(-1).tolist() == ref["first_logits_argmax"]
    # the int4 round-trip model is NOT the original (quantization did something)
    orig = build_tiny_model(man)
    with torch.no_grad():
        lo = orig(torch.tensor([ref["prompt_ids"]]), use_cache=False).logits[0]
    assert float((lo - logits).abs().max()) > 0


def test_tiny_rerun_is_idempotent(tiny_export):
    out, _ = tiny_export
    before = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    man = export_inkling.load_and_validate_config(TINY_CONFIG)
    ckpt = hf_state_to_checkpoint(build_tiny_model(man).state_dict())
    summary = export_inkling.Exporter(export_inkling.DictSource(ckpt), man, out, workers=2).run()
    n_units = 2 + 4 + 1 + 3 * (8 + 2)
    assert summary["counts"] == {"skipped": n_units} and summary["complete"]
    after = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    for p, m in before.items():
        if p.name != "manifest.json":
            assert after[p] == m, p
    assert export_inkling.layers_done_check(out) is True


def test_layers_done_check_detects_missing_and_truncated(tiny_export):
    out, _ = tiny_export
    victim = out / "experts" / "layer_02" / "expert_003.bin"
    data = victim.read_bytes()
    try:
        victim.write_bytes(data[:-1])  # truncated file must not count as done, marker or not
        assert (out / ".layer_02.done").exists()
        assert export_inkling.layers_done_check(out) is False
        assert not (out / ".layer_02.done").exists()  # the audit drops the stale marker
        victim.unlink()
        assert export_inkling.layers_done_check(out) is False
        man = export_inkling.load_and_validate_config(TINY_CONFIG)
        ckpt = hf_state_to_checkpoint(build_tiny_model(man).state_dict())
        summary = export_inkling.Exporter(export_inkling.DictSource(ckpt), man, out, workers=2).run()
        assert summary["counts"]["done"] == 1 and summary["complete"]
        assert victim.read_bytes() == data
        assert export_inkling.layers_done_check(out) is True
    finally:
        victim.write_bytes(data)
        (out / ".layer_02.done").touch()


# --------------------------------------------------------------------------
# streaming passes over a synthetic sharded checkpoint (the real --model path)
# --------------------------------------------------------------------------
def _write_sharded_checkpoint(model_dir: Path, ckpt: dict):
    """3 shards split by layer (shard 0: embed/head/mtp + layer 0, 1: layers 1-2, 2: layer 3),
    bf16 tensors like the real checkpoint, plus model.safetensors.index.json + config.json."""
    from safetensors.torch import save_file

    def shard_of(name):
        if name.startswith("model.llm.layers."):
            li = int(name.split(".")[3])
            return 0 if li == 0 else (1 if li <= 2 else 2)
        return 0

    groups = {0: {}, 1: {}, 2: {}}
    for k, v in ckpt.items():
        groups[shard_of(k)][k] = v.to(torch.bfloat16).contiguous()
    groups[0]["model.mtp.layers.0.attn_norm.weight"] = torch.ones(64, dtype=torch.bfloat16)  # dropped
    names = {i: f"model-{i + 1:05d}-of-00003.safetensors" for i in groups}
    wm = {}
    model_dir.mkdir(parents=True, exist_ok=True)
    for i, g in groups.items():
        save_file(g, str(model_dir / names[i]))
        for k in g:
            wm[k] = names[i]
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (model_dir / "config.json").write_text(json.dumps(TINY_CONFIG))
    (model_dir / "tokenizer_config.json").write_text("{}")
    return names


def test_streaming_export_over_sharded_checkpoint(tmp_path, tiny_export):
    tiny_out, _ = tiny_export
    man = export_inkling.load_and_validate_config(TINY_CONFIG)
    ckpt = hf_state_to_checkpoint(build_tiny_model(man).state_dict())
    model_dir = tmp_path / "ckpt"
    names = _write_sharded_checkpoint(model_dir, ckpt)
    out = tmp_path / "export"
    # shard 2 (layer 3) has not "arrived" yet
    parked = tmp_path / names[2]
    (model_dir / names[2]).rename(parked)

    with pytest.raises(SystemExit, match="not on disk"):
        export_inkling.export_real(model_dir, out, workers=2)  # without --skip-missing-shards: loud
    s1 = export_inkling.export_real(model_dir, out, skip_missing=True, delete_consumed=True, workers=2)
    assert not s1["complete"] and s1["layers_done"] == 3 and s1["embed_done"] and s1["head_done"]
    assert s1["pending_shards"] == [names[2]]
    assert sorted(s1["shards_deleted"]) == [names[0], names[1]] and s1["shards_consumed"] == 2
    assert not (model_dir / names[0]).exists() and not (model_dir / names[1]).exists()
    assert not (out / "manifest.json").exists()
    assert (out / ".layer_00.done").exists() and not (out / ".layer_03.done").exists()
    assert export_inkling.layers_done_check(out, model_dir) is False

    parked.rename(model_dir / names[2])
    s2 = export_inkling.export_real(model_dir, out, skip_missing=True, delete_consumed=True, workers=2)
    assert s2["complete"] and s2["layers_done"] == 4 and s2["pending_shards"] == []
    assert s2["shards_deleted"] == [names[2]] and s2["shards_consumed"] == 3
    assert s2["counts"]["skipped"] == 2 + (1 + 1) + 2 * (1 + 10)  # embed, head, layer 0, layers 1-2
    assert (out / "manifest.json").exists() and (out / "tokenizer_config.json").exists()
    assert export_inkling.layers_done_check(out, model_dir) is True

    # byte-identical to the --tiny export (same weights, DictSource vs ShardSource)
    for p in tiny_out.rglob("*.bin"):
        assert (out / p.relative_to(tiny_out)).read_bytes() == p.read_bytes(), p
    from safetensors.torch import load_file
    for p in tiny_out.rglob("*.safetensors"):
        a, b = load_file(str(p)), load_file(str(out / p.relative_to(tiny_out)))
        assert set(a) == set(b) and all(torch.equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a), p
    assert json.loads((out / "manifest.json").read_text()) == json.loads((tiny_out / "manifest.json").read_text())


def test_layers_range_and_shards_only(tmp_path):
    man = export_inkling.load_and_validate_config(TINY_CONFIG)
    ckpt = hf_state_to_checkpoint(build_tiny_model(man).state_dict())
    model_dir = tmp_path / "ckpt"
    names = _write_sharded_checkpoint(model_dir, ckpt)
    out = tmp_path / "export"
    s = export_inkling.export_real(model_dir, out, layers=(1, 2), shards_only=True, delete_consumed=True, workers=2)
    assert not s["complete"] and s["layers_done"] == 0 and not s["embed_done"]
    assert sorted(p.name for p in (out / "experts").iterdir()) == ["layer_01", "layer_02"]
    assert not (out / "shells").exists() or not list((out / "shells").iterdir())
    assert s["shards_deleted"] == []  # shells of layers 1-2 still need shard 1
    assert all((model_dir / n).exists() for n in names.values())
    s = export_inkling.export_real(model_dir, out, delete_consumed=True, workers=2)
    assert s["complete"] and s["counts"]["skipped"] == 2 * 10
    assert sorted(s["shards_deleted"]) == sorted(names.values())


def test_gen_fixtures_writes_spec_tensor_set(tmp_path):
    from inkling_ref import gen_fixtures
    from safetensors.torch import load_file

    old = sys.argv
    sys.argv = ["gen_fixtures.py", "--out", str(tmp_path)]
    try:
        gen_fixtures.main()
    finally:
        sys.argv = old
    fx = load_file(str(tmp_path / "fixtures.safetensors"))
    for name, shape in {"prompt_ids": (12,), "greedy_ids": (8,), "final_logits": (12, 120),
                        "sconv_in": (8, 6), "sconv_w": (8, 4), "sconv_out": (8, 6),
                        "relpos_r": (4, 4), "relpos_proj": (4, 8), "relpos_bias": (4, 10),
                        "gate_logits": (10,), "gate_bias": (8,), "gate_idx": (2,), "gate_w": (2,), "gate_gamma": (2,),
                        "attn_x": (6, 64), "attn_out": (6, 64), "attn_out_sliding": (6, 64),
                        "moe_x": (1, 64), "moe_out": (1, 64)}.items():
        assert tuple(fx[name].shape) == shape, name
    for li in range(4):
        assert tuple(fx[f"layer{li}_out"].shape) == (12, 64)
    assert fx["prompt_ids"].dtype == torch.int64 and fx["gate_idx"].dtype == torch.int64
    assert fx["final_logits"].dtype == torch.float32
    assert "model.llm.layers.1.mlp.experts.w13_weight" in fx and "model.llm.embed.weight" in fx
    meta = json.loads((tmp_path / "fixtures.json").read_text())
    assert meta["greedy_ids"] == fx["greedy_ids"].tolist()
    assert max(meta["spec_vs_hf_max_rel_err"].values()) < 1e-4
    m = meta["argmax_top2_logit_margins"]
    assert min(m["prompt"] + m["greedy"]) >= MIN_ARGMAX_MARGIN
