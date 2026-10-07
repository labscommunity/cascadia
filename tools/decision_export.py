"""Build a Cascadia decision-model tree (docs/decisions.md).

  jev:  python tools/decision_export.py jev  --base-ir STOCK_INT8_IR --adapter-dir JEV/adapter_vllm --out TREE [--stages N]
  kev:  python tools/decision_export.py kev  --base-ir STOCK_INT8_IR --weights QWEN35_9B_BASE --lora KEV/adapter_model.safetensors \
                                             --head-pt KEV/head.pt --out TREE
  clef: python tools/decision_export.py clef --base-ir STOCK_INT8_IR --weights CLEF_DIR --clef-dir CLEF_DIR --out TREE

The stage IRs come from export_qwen36_moe.run_export (one stage in memory at a time).
The decision profile and head assets are written BEFORE the stage export, so the
manifest run_export writes last stays the commit marker for the whole tree, and a
bad head.pt / adapter dir fails before the long IR export starts.

Tree layout (read by crates/cascadia-api/src/decisions/profile.rs and
HeadAssets::load in crates/cascadia-engine-openvino/src/decision.rs):
  decision.json                  {"family", "max_tokens", "jev"|"kev"|"clef": {...}}
  decision/head.json             {"kind": "verbalizer"} | {"kind": "pointer", "dp"} | {"kind": "joint", "vocab"}
  decision/pointer_{q,k}_{w,b}.f32   kev: little-endian f32, w row-major (dp, hidden)
  decision/joint_head.xml/.bin   clef: OpenVINO joint schema head
  decision/lm_head.f16           clef: little-endian f16, row-major (vocab, hidden)
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "qwen36_surgery"))
sys.path.insert(0, os.path.dirname(__file__))


def _write(out_dir, profile, head):
    out = Path(out_dir)
    (out / "decision").mkdir(parents=True, exist_ok=True)
    (out / "decision.json").write_text(json.dumps(profile, indent=1))
    (out / "decision" / "head.json").write_text(json.dumps(head))


def write_jev_profile(adapter_dir, calibration_json, out_dir, max_tokens=32768):
    h = json.loads((Path(adapter_dir) / "decision_head.json").read_text())
    cal = json.loads(Path(calibration_json).read_text())["per_kind"]
    slots = h["slots"]
    profile = {"family": "jev", "max_tokens": max_tokens, "jev": {
        "verbalizers": slots["verbalizers"], "verbalizer_ids": h["verbalizer_ids"], "bias": h["bias"],
        "slots": slots["ranges"], "temperature": {k: cal.get(k, 1.0) for k in ("noul", "choice", "score")}}}
    _write(out_dir, profile, {"kind": "verbalizer"})


def write_kev_profile(head_pt, out_dir, max_tokens=73728):
    import torch
    meta = torch.load(head_pt, map_location="cpu", weights_only=True)
    w = meta["head"]
    dp = w["q.weight"].shape[0]
    d = Path(out_dir) / "decision"
    d.mkdir(parents=True, exist_ok=True)
    for key, fname in (("q.weight", "pointer_q_w"), ("q.bias", "pointer_q_b"), ("k.weight", "pointer_k_w"), ("k.bias", "pointer_k_b")):
        w[key].float().contiguous().numpy().astype("<f4").tofile(d / f"{fname}.f32")
    _write(out_dir, {"family": "kev", "max_tokens": max_tokens, "kev": {"temperature": float(meta["temperature"])}},
           {"kind": "pointer", "dp": int(dp)})


def _lm_head_name(W, clef_dir):
    """lm_head.weight, or the input embeddings when the checkpoint ties them."""
    if "lm_head.weight" in W.idx:
        return "lm_head.weight"
    cfg_path = Path(clef_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    tied = cfg.get("tie_word_embeddings", (cfg.get("text_config") or {}).get("tie_word_embeddings", False))
    emb = W.prefix + "embed_tokens.weight"
    if tied and emb in W.idx:
        return emb
    raise SystemExit(f"{clef_dir}: no lm_head.weight in model.safetensors.index.json "
                     f"(and tie_word_embeddings is not set) - cannot write decision/lm_head.f16")


def write_clef_profile(clef_dir, out_dir, max_tokens=16384, chunk=8192):
    from safetensors import safe_open
    from weight_swap import Weights
    from decision import clef_head

    W = Weights(clef_dir)
    name = _lm_head_name(W, clef_dir)
    d = Path(out_dir) / "decision"
    d.mkdir(parents=True, exist_ok=True)
    clef_head.export_joint_head(clef_dir, str(d / "joint_head.xml"))
    # lm_head rows fp16 for the lexical prior, streamed in chunks (vocab x hidden)
    f = safe_open(Path(clef_dir) / W.idx[name], "pt")
    sl = f.get_slice(name)
    vocab, _hidden = sl.get_shape()
    with open(d / "lm_head.f16", "wb") as out:
        for a in range(0, vocab, chunk):
            sl[a:min(vocab, a + chunk)].float().numpy().astype("<f2").tofile(out)
    # head.json last: it names the vocab the lm_head.f16 just written holds
    _write(out_dir, {"family": "clef", "max_tokens": max_tokens, "clef": {}}, {"kind": "joint", "vocab": int(vocab)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["jev", "kev", "clef"])
    ap.add_argument("--base-ir", required=True, help="Intel stock int8 IR dir (OpenVINO/Qwen3.5-9B-int8-ov or Qwen3.8-27B-int8-ov)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", type=int, default=8, help="stage cut count: 8 for a single box, one per rank for a pipeline")
    ap.add_argument("--layer-split", default=None)
    ap.add_argument("--weights", default=None, help="HF checkpoint whose weights replace the IR's (clef: defaults to --clef-dir)")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=None, help="override adapter_config.json")
    ap.add_argument("--adapter-dir", default=None, help="jev: dir with decision_head.json + adapter_model.safetensors")
    ap.add_argument("--calibration", default=None, help="jev: calibration.json (default: ADAPTER_DIR/../calibration.json)")
    ap.add_argument("--head-pt", default=None, help="kev: head.pt")
    ap.add_argument("--clef-dir", default=None, help="clef: dir with joint_head.* and the checkpoint")
    a = ap.parse_args()
    need = {"jev": ("adapter_dir", "--adapter-dir"), "kev": ("head_pt", "--head-pt"), "clef": ("clef_dir", "--clef-dir")}
    attr, flag = need[a.family]
    if not getattr(a, attr):
        ap.error(f"{a.family} needs {flag}")
    from export_qwen36_moe import run_export

    # validate is always False: run_export refuses --validate with --weights/--lora
    # (decision trees are checked by the golden/parity gates instead).
    if a.family == "jev":
        lora = a.lora or os.path.join(a.adapter_dir, "adapter_model.safetensors")
        cal = a.calibration or os.path.join(os.path.dirname(os.path.abspath(a.adapter_dir)), "calibration.json")
        cfg = json.loads((Path(a.base_ir) / "config.json").read_text())
        mpe = (cfg.get("text_config") or cfg).get("max_position_embeddings", 32768)
        write_jev_profile(a.adapter_dir, cal, a.out, max_tokens=min(int(mpe), 32768))
        run_export(a.base_ir, a.out, a.stages, False, a.layer_split, weights=a.weights, lora=lora,
                   lora_scale=a.lora_scale, final_hidden=False)
    elif a.family == "kev":
        write_kev_profile(a.head_pt, a.out)
        run_export(a.base_ir, a.out, a.stages, False, a.layer_split, weights=a.weights, lora=a.lora,
                   lora_scale=a.lora_scale, final_hidden=True)
    else:
        write_clef_profile(a.clef_dir, a.out)
        run_export(a.base_ir, a.out, a.stages, False, a.layer_split, weights=a.weights or a.clef_dir,
                   lora=a.lora, lora_scale=a.lora_scale, final_hidden=True)
    print(f"decision tree written: {a.out}")


if __name__ == "__main__":
    main()
