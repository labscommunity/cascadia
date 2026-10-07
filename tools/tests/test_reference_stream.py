"""reference_stream: layer-streamed fp32 forward == the full HF model on a tiny random Qwen3.5.

    python -m pytest tools/tests/test_reference_stream.py -v
"""
import json
import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")
M = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "decision"))
import reference_stream as rs  # noqa: E402


def tiny(tmp_path):
    from safetensors.torch import save_file
    from transformers.models.qwen3_5 import Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                            layer_types=["linear_attention", "full_attention"],
                            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8)
    torch.manual_seed(0)
    model = M.Qwen3_5ForCausalLM(cfg).float().eval()
    # clone: tied embeddings share storage with lm_head and safetensors refuses shared tensors
    sd = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in sd}}))
    cfg.save_pretrained(tmp_path)
    return model


def test_streamed_hidden_matches_full_model(tmp_path):
    model = tiny(tmp_path)
    ids = [3, 9, 27, 1, 44, 5]
    with torch.no_grad():
        full = model(torch.tensor([ids]), output_hidden_states=True).hidden_states[-1][0]
    (h,) = rs.stream_hidden(str(tmp_path), [{"readout": {"prompt_ids": ids}}], threads=1)
    assert torch.allclose(h, full, atol=1e-4), float((h - full).abs().max())


def write_lora(path, model, modules, rank=2, seed=1):
    """A PEFT-shaped adapter over `modules` (paths under model.model, e.g. "layers.0.mlp.down_proj")."""
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(seed)
    t = {}
    for m in modules:
        w = model.get_submodule(m if m == "lm_head" else f"model.{m}").weight
        key = m if m == "lm_head" else f"model.{m}"
        t[f"base_model.model.{key}.lora_A.weight"] = torch.randn(rank, w.shape[1], generator=g) * 0.1
        t[f"base_model.model.{key}.lora_B.weight"] = torch.randn(w.shape[0], rank, generator=g) * 0.1
    save_file(t, str(path))
    return t


def test_lora_merged_per_layer_matches_merged_full_model(tmp_path):
    model = tiny(tmp_path)
    mods = ["layers.0.mlp.down_proj", "layers.0.linear_attn.in_proj_qkv", "layers.1.self_attn.q_proj"]
    t = write_lora(tmp_path / "adapter.safetensors", model, mods)
    scale = 2.0
    ids = [7, 2, 61, 30, 4]
    with torch.no_grad():
        (plain,) = rs.stream_hidden(str(tmp_path), [{"readout": {"prompt_ids": ids}}], threads=1)
        for m in mods:
            k = f"base_model.model.model.{m}"
            model.get_submodule(f"model.{m}").weight += scale * (t[f"{k}.lora_B.weight"] @ t[f"{k}.lora_A.weight"])
        full = model(torch.tensor([ids]), output_hidden_states=True).hidden_states[-1][0]
    (h,) = rs.stream_hidden(str(tmp_path), [{"readout": {"prompt_ids": ids}}],
                            lora=str(tmp_path / "adapter.safetensors"), lora_scale=scale, threads=1)
    assert torch.allclose(h, full, atol=1e-4), float((h - full).abs().max())
    assert not torch.allclose(plain, full, atol=1e-3), "the adapter changed nothing; the test proves nothing"


def test_adapter_module_matching_no_weight_fails(tmp_path):
    from safetensors.torch import save_file
    tiny(tmp_path)
    save_file({"base_model.model.model.layers.0.mlp.nope.lora_A.weight": torch.zeros(2, 32),
               "base_model.model.model.layers.0.mlp.nope.lora_B.weight": torch.zeros(32, 2)}, str(tmp_path / "bad.safetensors"))
    with pytest.raises(SystemExit, match="matched no decoder weight"):
        rs.stream_hidden(str(tmp_path), [{"readout": {"prompt_ids": [1, 2]}}],
                         lora=str(tmp_path / "bad.safetensors"), lora_scale=1.0, threads=1)


def test_streams_several_passes_of_different_lengths(tmp_path):
    model = tiny(tmp_path)
    seqs = [[3, 9, 27], [1, 44, 5, 8, 8, 2, 63]]
    hs = rs.stream_hidden(str(tmp_path), [{"readout": {"prompt_ids": s}} for s in seqs], threads=1)
    with torch.no_grad():
        for s, h in zip(seqs, hs):
            full = model(torch.tensor([s]), output_hidden_states=True).hidden_states[-1][0]
            assert h.shape == full.shape
            assert torch.allclose(h, full, atol=1e-4)


def test_lm_rows_reads_lm_head(tmp_path):
    model = tiny(tmp_path)
    assert torch.equal(rs.lm_rows(str(tmp_path), [5, 0, 5]), model.lm_head.weight.detach()[[5, 0, 5]])


# ---------------------------------------------------------------------------------------------
# family_probs / build_ref: option order and one entry per question

RECS = [
    {"id": "r0", "state": "s", "questions": {
        "a": {"type": "noul", "instructions": "?"},
        "b": {"type": "choice", "instructions": "?", "criteria": {"zeta": "z", "alpha": "a", "mid": "m"}},
        "c": {"type": "score", "instructions": "?", "criteria": ["lo", "mid", "hi"]}}},
    {"id": "r1", "state": "t", "questions": {"d": {"type": "noul", "instructions": "?"}}},
]


def kev_tree(tmp_path, hidden, dp=4, temperature=1.5):
    import numpy as np
    tree = tmp_path / "tree"
    (tree / "decision").mkdir(parents=True)
    (tree / "decision.json").write_text(json.dumps({"family": "kev", "kev": {"temperature": temperature}}))
    (tree / "decision" / "head.json").write_text(json.dumps({"kind": "pointer", "dp": dp}))
    g = torch.Generator().manual_seed(3)
    w = {"pointer_q_w": torch.randn(dp, hidden, generator=g), "pointer_q_b": torch.randn(dp, generator=g),
         "pointer_k_w": torch.randn(dp, hidden, generator=g), "pointer_k_b": torch.randn(dp, generator=g)}
    for n, v in w.items():
        v.numpy().astype("<f4").tofile(tree / "decision" / f"{n}.f32")
    return tree, w


def kev_dump(tmp_path):
    lines = []
    for rec in RECS:
        for qid, q in rec["questions"].items():
            n = {"noul": 2, "choice": len(q.get("criteria", [])), "score": len(q.get("criteria", []))}[q["type"]]
            ids = list(range(1, 4 + 2 * n))
            lines.append({"parts": [[qid, 0, n]], "readout": {"prompt_ids": ids, "head": {
                "kind": "pointer", "decide_pos": len(ids) - 1, "option_pos": [2 + 2 * i for i in range(n)]}}})
    p = tmp_path / "dump.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return p, lines


def test_kev_build_ref_matches_pointer_head_math(tmp_path):
    tiny(tmp_path)
    tree, w = kev_tree(tmp_path, hidden=32)
    dump, lines = kev_dump(tmp_path)
    ref = rs.build_ref("kev", str(tmp_path), str(dump), RECS, str(tree), threads=1)
    assert list(ref) == ["r0.a", "r0.b", "r0.c", "r1.d"]  # exactly one entry per question
    hs = rs.stream_hidden(str(tmp_path), lines, threads=1)
    for (key, got), line, h in zip(ref.items(), lines, hs):
        hd = line["readout"]["head"]
        q = w["pointer_q_w"] @ h[hd["decide_pos"]] + w["pointer_q_b"]
        k = h[hd["option_pos"]] @ w["pointer_k_w"].T + w["pointer_k_b"]
        want = torch.softmax(((k @ q) / math.sqrt(4) / 1.5).double(), 0).tolist()
        assert len(got) == len(want)
        assert got == pytest.approx(want, abs=1e-6), key
        assert sum(got) == pytest.approx(1.0)


def test_kev_build_ref_rejects_misaligned_dump(tmp_path):
    tiny(tmp_path)
    tree, _ = kev_tree(tmp_path, hidden=32)
    dump, lines = kev_dump(tmp_path)
    dump.write_text("".join(json.dumps(x) + "\n" for x in lines[:-1]))
    with pytest.raises(SystemExit, match="MISMATCH"):
        rs.build_ref("kev", str(tmp_path), str(dump), RECS, str(tree), threads=1)


def jev_tree(tmp_path):
    tree = tmp_path / "tree"
    (tree / "decision").mkdir(parents=True)
    # 2 noul + 3 choice + 6 score slots
    bias = [0.1 * i for i in range(11)]
    prof = {"family": "jev", "jev": {"verbalizers": [str(i) for i in range(11)], "verbalizer_ids": list(range(10, 21)),
                                     "bias": bias, "slots": {"noul": [0, 2], "choice": [2, 5], "score": [5, 11]},
                                     "temperature": {"noul": 1.0, "choice": 2.0, "score": 0.5}}}
    (tree / "decision.json").write_text(json.dumps(prof))
    (tree / "decision" / "head.json").write_text(json.dumps({"kind": "verbalizer"}))
    return tree, prof["jev"]


def test_jev_build_ref_bias_temperature_rescale_and_lm_head_lora(tmp_path):
    from codec_golden import rescale_score
    model = tiny(tmp_path)
    tree, j = jev_tree(tmp_path)
    t = write_lora(tmp_path / "adapter.safetensors", model, ["lm_head"])
    lines = []
    for rec in RECS:
        for qid, q in rec["questions"].items():
            lo, hi = j["slots"][q["type"]]
            if q["type"] == "choice":
                hi = lo + len(q["criteria"])
            lines.append({"parts": [[qid, 0, hi - lo]],
                          "readout": {"prompt_ids": [4, 8, 15, 16, 23 + len(lines)],
                                      "head": {"kind": "verbalizer", "token_ids": j["verbalizer_ids"][lo:hi]}}})
    dump = tmp_path / "dump.jsonl"
    dump.write_text("".join(json.dumps(x) + "\n" for x in lines))
    scale = 2.0
    ref = rs.build_ref("jev", str(tmp_path), str(dump), RECS, str(tree),
                       lora=str(tmp_path / "adapter.safetensors"), lora_scale=scale, threads=1)
    assert list(ref) == ["r0.a", "r0.b", "r0.c", "r1.d"]
    hs = rs.stream_hidden(str(tmp_path), lines, threads=1)
    W = model.lm_head.weight.detach() + scale * (t["base_model.model.lm_head.lora_B.weight"] @ t["base_model.model.lm_head.lora_A.weight"])
    kinds = ["noul", "choice", "score", "noul"]
    for (key, got), line, h, kind in zip(ref.items(), lines, hs, kinds):
        vids = line["readout"]["head"]["token_ids"]
        lo = j["slots"][kind][0]
        z = [(float(W[v] @ h[-1]) + j["bias"][lo + i]) / j["temperature"][kind] for i, v in enumerate(vids)]
        want = torch.softmax(torch.tensor(z, dtype=torch.float64), 0).tolist()
        if kind == "score":
            want = rescale_score(want, 3)
        assert got == pytest.approx(want, abs=1e-6), key


def test_clef_family_probs_reorders_noul_and_choice(tmp_path, monkeypatch):
    """Clef's head emits noul [true, false] and choices sorted; the reference reports option_keys order."""
    rec = RECS[0]
    logits = torch.tensor([2.0, -1.0,  0.5, 1.5, -0.5,  0.1, 0.2, 0.3])  # a: [true,false]; b: alpha,mid,zeta; c: 3
    head = {"kind": "joint", "questions": [
        {"qtype": 0, "span": [0, 1], "options": [{"span": [1, 2]}, {"span": [2, 3]}]},
        {"qtype": 1, "span": [0, 1], "options": [{"span": [1, 2]}, {"span": [2, 3]}, {"span": [3, 4]}]},
        {"qtype": 2, "span": [0, 1], "options": [{"span": [1, 2]}, {"span": [2, 3]}, {"span": [3, 4]}]}]}
    p = {"parts": [["a", 0, 2], ["b", 2, 5], ["c", 5, 8]], "readout": {"prompt_ids": [1, 2, 3, 4, 5], "head": head}}

    class FakeHead(torch.nn.Module):
        def forward(self, hidden, mask, lexical, option_q, types):
            assert mask.shape == (3 + 8, 5) and lexical.shape == (8, 32)
            assert option_q.tolist() == [0, 0, 1, 1, 1, 2, 2, 2] and types.tolist() == [0, 1, 2]
            return logits

    monkeypatch.setattr(rs, "clef_tensor_head", lambda clef_dir: FakeHead())
    tiny(tmp_path)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "decision.json").write_text(json.dumps({"family": "clef", "clef": {}}))
    out = rs.family_probs("clef", rec, [p], [torch.zeros(5, 32)], str(tree), str(tmp_path), None, None, "unused")
    sm = lambda v: torch.softmax(torch.tensor(v, dtype=torch.float64), 0).tolist()
    assert out["a"] == pytest.approx(sm([2.0, -1.0])[::-1])           # [false, true]
    s = sm([0.5, 1.5, -0.5])                                           # alpha, mid, zeta
    assert out["b"] == pytest.approx([s[2], s[0], s[1]])              # request order: zeta, alpha, mid
    assert out["c"] == pytest.approx(sm([0.1, 0.2, 0.3]))
