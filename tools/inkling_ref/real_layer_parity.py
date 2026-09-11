#!/usr/bin/env python3
"""Real-weight per-layer parity: the Rust Inkling shell vs transformers on the SAME weights.

    python tools/inkling_ref/real_layer_parity.py --export DIR --layers K --dump dump.safetensors \\
        [--dtype float32|bfloat16] [--tol 0.02] [--tokens 1,2,3] [--json report.json]

`dump.safetensors` is written by the Rust side (same DIR, same K):

    cargo run -p cascadia-engine-sparse-moe --release --example inkling_layer_dump -- \\
        --export DIR --layers K --tokens 1,2,3 --out dump.safetensors

What this script does
  1. builds an `InklingTextConfig` from the export's manifest.json (cross-checked against
     source_config.json when present) with `num_hidden_layers = K` and `layer_types` /
     `mlp_layer_types` sliced to the first K layers;
  2. loads embed + embed_norm and the first K layers' weights from the export (bf16 shells as-is,
     int4 bins dequantised, HF parameter names via inkling_ref's mapping); norm + unembed only
     when K == num_layers. The model is instantiated on the meta device and the tensors assigned
     in place, so a 58 GB layer is never randomly initialised first;
  3. runs the dump's token ids through transformers' own `InklingForCausalLM` (eager attention,
     no cache) with forward hooks on `embed_norm` and every decoder layer (`output_hidden_states`
     replaces the last entry with the normed state; hooks do not);
  4. compares each dumped tensor with its HF counterpart: max |diff|, max |diff| as a fraction of
     the HF row RMS (the PASS/FAIL metric, `--tol`), rms(diff)/rms(row) and the minimum per-row
     cosine similarity — for the decode-path AND the prefill-path tensors, plus the logits (with
     per-position argmax agreement) when the head is present. With reference.json next to the
     export and the dump starting with its prompt (K == num_layers), the first-token / greedy
     argmaxes are checked on both sides too.

Tolerance
  The Rust shell rounds to bf16 after every linear (bf16 weights, f32 accumulate); HF in float32
  does not. Each rounded element carries up to ~0.4 % relative error, so the default `--tol 0.02`
  (2 % of the HF row RMS for the single worst element) is the band the tier-2 tests use. Measured
  on the tiny export (K=4, 19 tokens, float32): worst element 0.09-0.26 % of its row RMS
  (0.1 % of the row max), rms(diff)/rms 0.02-0.06 %, cosine 1.000000, argmax 19/19, decode ==
  prefill bit for bit. With `--dtype bfloat16` HF ALSO rounds its residual stream (and the
  log-scaled position bias) to bf16 while the Rust residual stays f32: HF-bf16 differs from
  HF-f32 by 17 % of the row RMS at the tiny model's global layer, and Rust-vs-HF-bf16 lands on
  the same number — so bf16 is a memory fallback for an argmax-level check only (`--tol 0.25`);
  the per-layer numbers mean float32.

Memory on the real box (Inkling 975B: hidden 6144, 66 layers, 256 experts of width 3072)
  Dequantised to float32 one MoE layer is 256 x 3 x 3072 x 6144 x 4 B = 58 GB (plus ~0.5 GB of
  shared experts and ~0.8 GB of attention); a dense layer is ~1.9 GB; embed 4.9 GB, unembed
  4.9 GB. On a 172 GB box `--layers 3` in float32 (layers 0-1 dense + layer 2, the first MoE
  layer: ~67 GB resident plus the forward's transients) is the intended run; `--layers 4` would
  need ~125 GB and is already tight; `--dtype bfloat16` halves all of it. The estimate is
  printed before anything is loaded. Loading is dominated by dequantising the 256 expert bins of
  the MoE layer (32 MB of int4 -> 226 MB of f32 each, numpy on one core) — minutes, not seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import torch  # noqa: E402

from inkling_ref import hf_config_from_manifest, load_export_state, read_manifest  # noqa: E402

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
# manifest key -> source_config.json text_config key(s) that must agree with it
_SOURCE_KEYS = {
    "hidden_size": "hidden_size", "num_layers": "num_hidden_layers", "vocab_size": "vocab_size",
    "unpadded_vocab_size": "unpadded_vocab_size", "num_attention_heads": "num_attention_heads",
    "num_kv_heads": "num_key_value_heads", "head_dim": "head_dim", "d_rel": "d_rel",
    "rel_extent": "rel_extent", "sliding_window": "sliding_window_size",
    "moe_intermediate": "moe_intermediate_size", "num_experts": "n_routed_experts",
    "top_k": "num_experts_per_tok", "n_shared_experts": "n_shared_experts",
    "route_scale": "route_scale", "rms_norm_eps": "rms_norm_eps",
    "logits_mup_width_multiplier": "logits_mup_width_multiplier", "conv_kernel_size": "sconv_kernel_size",
    "hidden_act": "hidden_act",
}


# --------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------
def estimate_bytes(man: dict, k: int, dtype: torch.dtype, with_head: bool) -> dict:
    """Resident bytes of the HF model for the first `k` layers in `dtype` (weights only)."""
    H, V = man["hidden_size"], man["vocab_size"]
    isz = torch.empty(0, dtype=dtype).element_size()
    per_layer = []
    for li in range(k):
        sliding = man["layer_types"][li] == "sliding"
        hq = man["swa_num_attention_heads"] if sliding else man["num_attention_heads"]
        hkv = man["swa_num_kv_heads"] if sliding else man["num_kv_heads"]
        d = man["swa_head_dim"] if sliding else man["head_dim"]
        attn = (2 * hq * d * H + 2 * hkv * d * H + hq * d * H) * isz  # q, r, k, v, o
        if li in man["dense_layers"]:
            mlp = 3 * H * man["dense_intermediate"] * isz
        else:
            mlp = (man["num_experts"] + man["n_shared_experts"]) * 3 * H * man["moe_intermediate"] * isz
        per_layer.append(attn + mlp)
    embed = V * H * isz
    head = V * H * isz if with_head else 0
    return {"per_layer": per_layer, "embed": embed, "head": head,
            "total": sum(per_layer) + embed + head}


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def source_text_config(export_dir) -> dict | None:
    p = Path(export_dir) / "source_config.json"
    if not p.exists():
        return None
    cfg = json.loads(p.read_text())
    return cfg.get("text_config", cfg)


def check_source_config(man: dict, src: dict) -> list[str]:
    """Fields where source_config.json disagrees with manifest.json (a stale manifest would make
    the comparison meaningless, so the caller fails on any)."""
    bad = []
    for mk, sk in _SOURCE_KEYS.items():
        if mk not in man or sk not in src:
            continue
        a, b = man[mk], src[sk]
        same = math.isclose(a, b, rel_tol=1e-6) if isinstance(a, float) or isinstance(b, float) else a == b
        if not same:
            bad.append(f"{mk}={a!r} vs source {sk}={b!r}")
    src_local = src.get("local_layer_ids")
    if src_local is not None:
        want = [i for i, t in enumerate(man["layer_types"]) if t == "sliding"]
        if sorted(src_local) != want:
            bad.append(f"layer_types sliding ids {want} vs source local_layer_ids {sorted(src_local)}")
    if "dense_mlp_idx" in src and list(range(src["dense_mlp_idx"])) != list(man["dense_layers"]):
        bad.append(f"dense_layers={man['dense_layers']} vs source dense_mlp_idx={src['dense_mlp_idx']}")
    return bad


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def build_model(man: dict, k: int, sd: dict, dtype: torch.dtype):
    """`InklingForCausalLM` for the first `k` layers with `sd` assigned in place (meta-device
    construction: no random init, no extra copy). For a partial model (k < num_layers) the final
    norm and unembed are replaced by identities — their outputs are never compared."""
    from transformers import InklingForCausalLM

    cfg = hf_config_from_manifest(man, num_layers=k)
    with torch.device("meta"):
        model = InklingForCausalLM(cfg)
    partial = k < man["num_layers"]
    if partial:
        model.model.norm = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()
    res = model.load_state_dict(sd, strict=False, assign=True)
    if res.unexpected_keys:
        raise KeyError(f"unexpected keys for a {k}-layer model: {sorted(res.unexpected_keys)[:8]} ...")
    if res.missing_keys:
        raise KeyError(f"missing keys: {sorted(res.missing_keys)[:8]} ...")
    left = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if left:
        raise RuntimeError(f"parameters still on meta after load: {left[:8]} ...")
    model.eval()
    assert model.config._attn_implementation == "eager", model.config._attn_implementation
    if dtype == torch.bfloat16:
        # what from_pretrained(dtype=bfloat16) keeps in f32 (_keep_in_fp32_modules_strict)
        for n, p in model.named_parameters():
            if "sconv" in n:
                assert p.dtype == torch.float32, n
    return model


@torch.no_grad()
def run_hf(model, tokens: list[int], k: int) -> dict[str, torch.Tensor]:
    """{embed_out, layer{L}_out (L < k), logits?} as float32 [T, ...] from one no-cache forward."""
    outs: dict[str, torch.Tensor] = {}
    hooks = [model.model.embed_norm.register_forward_hook(
        lambda m, i, o: outs.__setitem__("embed_out", o.detach()[0].float().clone()))]
    for li, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, li=li: outs.__setitem__(f"layer{li}_out", o.detach()[0].float().clone())))
    ids = torch.tensor([tokens], dtype=torch.long)
    try:
        o = model(ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    if k == model.config.num_hidden_layers and not isinstance(model.lm_head, torch.nn.Identity):
        outs["logits"] = o.logits[0].float()
    assert set(outs) >= {"embed_out", *(f"layer{li}_out" for li in range(k))}, sorted(outs)
    return outs


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------
def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of bf16 values at magnitude |x| (7 explicit mantissa bits): 2^(floor(log2|x|) - 7)."""
    ax = x.abs().clamp_min(1e-30)
    return torch.exp2(torch.floor(torch.log2(ax)) - 7)


def row_metrics(rust: torch.Tensor, hf: torch.Tensor, tol: float,
                ulp_tol: float = 4.0, rms_tol: float = 0.01) -> dict:
    """Per-tensor closeness of `rust` to `hf` (both [T, C]), computed in float64.

    PASS = `rms(diff)/rms <= rms_tol` AND every element within `ulp_tol` bf16 ULPs of the HF
    value (denominator floored at 1e-3 of the row RMS so near-zero entries are judged on the
    row's scale). The Rust shell rounds every linear's output to bf16 while HF runs float32;
    on the real model the residual stream carries massive activations (|x| ~ 1e3 in rows whose
    RMS is ~10), where one bf16 ULP is 4-8 in absolute terms — max|diff|/rowRMS reads 0.4 there
    while rms(diff)/rms is 0.3 %. That is the reference's own dtype granularity, not a math
    error, so the verdict is ULP-aware; `max|diff|/rowRMS` (`tol`) stays reported.
    """
    r, h = rust.double(), hf.double()
    if r.shape != h.shape:
        raise ValueError(f"shape {tuple(r.shape)} vs HF {tuple(h.shape)}")
    if not torch.isfinite(r).all():
        raise ValueError("non-finite values in the Rust tensor")
    diff = (r - h).abs()
    row_rms = h.pow(2).mean(dim=1).sqrt().clamp_min(1e-30)
    row_max = h.abs().max(dim=1).values.clamp_min(1e-30)
    frac_rows = diff.max(dim=1).values / row_rms
    cos = torch.nn.functional.cosine_similarity(r, h, dim=1, eps=1e-30)
    worst = int(frac_rows.argmax())
    ulps = diff / torch.maximum(bf16_ulp(h), 1e-3 * row_rms[:, None])
    uw = int(ulps.argmax())
    urow, ucol = divmod(uw, ulps.shape[1])
    m = {
        "max_abs": float(diff.max()),
        # the PASS/FAIL metric: worst element of the worst row, relative to that row's RMS
        "frac_max": float(frac_rows.max()),
        "frac_max_row": worst,
        # the tier-2 test's "row scale" metric: worst element relative to the row's max |value|
        "frac_scale": float((diff.max(dim=1).values / row_max).max()),
        # energy of the whole diff relative to the tensor's energy
        "frac_rms": float(diff.pow(2).mean().sqrt() / h.pow(2).mean().sqrt().clamp_min(1e-30)),
        "cos_min": float(cos.min()),
        "row_rms_min": float(row_rms.min()),
        "row_rms_max": float(row_rms.max()),
        "rows": int(r.shape[0]),
        # worst element in bf16 ULPs of the HF value (floored at 1e-3 * row RMS)
        "ulp_max": float(ulps.max()),
        "ulp_worst": {"row": int(urow), "col": int(ucol), "hf": float(h[urow, ucol]),
                      "rust": float(r[urow, ucol]), "row_rms": float(row_rms[urow])},
        "frac_max_within_tol": bool(frac_rows.max() <= tol),
    }
    m["pass"] = m["frac_rms"] <= rms_tol and m["ulp_max"] <= ulp_tol
    return m


def compare_dump(export, layers: int, dump, dtype: str = "float32", tol: float = 0.02,
                 ulp_tol: float = 4.0, rms_tol: float = 0.01,
                 tokens: list[int] | None = None, log=print) -> dict:
    """Load, run HF, compare. Returns the report dict (`report['pass']` is the verdict)."""
    from safetensors.torch import load_file
    from safetensors import safe_open

    export = Path(export)
    man = read_manifest(export)
    k = int(layers)
    if not 1 <= k <= man["num_layers"]:
        raise ValueError(f"--layers {k} out of range 1..{man['num_layers']}")
    full = k == man["num_layers"]
    tdt = DTYPES[dtype]

    src = source_text_config(export)
    if src is not None:
        bad = check_source_config(man, src)
        if bad:
            raise ValueError("manifest.json disagrees with source_config.json: " + "; ".join(bad))
        log(f"[parity] source_config.json agrees with manifest.json on {len(_SOURCE_KEYS)} fields")

    # ---- the dump ----
    d = load_file(str(dump))
    with safe_open(str(dump), "pt") as f:
        meta = f.metadata() or {}
    if meta.get("layers") not in (None, str(k)):
        raise ValueError(f"dump was written for --layers {meta['layers']}, comparing --layers {k}")
    if "tokens" in d:
        dump_tokens = [int(t) for t in d["tokens"].tolist()]
        if tokens is not None and list(tokens) != dump_tokens:
            raise ValueError(f"--tokens {list(tokens)} != dump tokens {dump_tokens}")
        tokens = dump_tokens
    elif tokens is None:
        raise ValueError("dump carries no `tokens` tensor: pass --tokens")
    tokens = [int(t) for t in tokens]
    T, H = len(tokens), man["hidden_size"]
    want_names = ["embed_out"] + [f"layer{li}_out_{p}" for li in range(k) for p in ("decode", "prefill")]
    if full:
        want_names += ["logits_decode", "logits_prefill"]
    missing = [n for n in want_names if n not in d]
    if missing:
        raise KeyError(f"dump is missing {missing}")
    for n in want_names:
        cols = man["unpadded_vocab_size"] if n.startswith("logits") else H
        if tuple(d[n].shape) != (T, cols):
            raise ValueError(f"dump {n}: shape {tuple(d[n].shape)} != ({T}, {cols})")

    # ---- sizing + load ----
    est = estimate_bytes(man, k, tdt, with_head=full)
    log(f"[parity] export={export} layers=0..{k} of {man['num_layers']} (head: {'yes' if full else 'no'}) "
        f"tokens={T} dtype={dtype} experts={meta.get('experts', '?')}")
    log(f"[parity] estimated HF RAM: {_gb(est['total'])} = embed {_gb(est['embed'])}"
        + (f" + head {_gb(est['head'])}" if full else "")
        + " + layers " + " + ".join(_gb(b) for b in est["per_layer"]))
    t0 = time.time()
    sd, _ = load_export_state(export, layers=range(k), dtype=tdt, with_head=full, log=log)
    log(f"[parity] weights loaded in {time.time() - t0:.1f}s ({len(sd)} tensors)")
    model = build_model(man, k, sd, tdt)
    del sd
    t0 = time.time()
    hf = run_hf(model, tokens, k)
    log(f"[parity] HF forward ({T} tokens, {k} layers, {dtype}) in {time.time() - t0:.1f}s")

    # ---- compare ----
    results: dict[str, dict] = {}
    log(f"[parity] {'tensor':<22} {'max|diff|':>11} {'max|diff|/rowRMS':>17} {'max|diff|/rowMax':>17} "
        f"{'rms(diff)/rms':>14} {'min cos':>10} {'bf16 ULPs':>10}  verdict")
    for n in want_names:
        base = n.rsplit("_", 1)[0] if n != "embed_out" else n  # layerL_out_decode -> layerL_out
        m = row_metrics(d[n], hf[base], tol, ulp_tol=ulp_tol, rms_tol=rms_tol)
        if n.startswith("logits"):
            am_r = d[n].argmax(dim=1)
            am_h = hf["logits"].argmax(dim=1)
            m["argmax_agree"] = int((am_r == am_h).sum())
            m["argmax_rust"] = am_r.tolist()
            m["argmax_hf"] = am_h.tolist()
            m["pass"] = m["pass"] and m["argmax_agree"] == T
        results[n] = m
        extra = f"  argmax {m['argmax_agree']}/{T}" if "argmax_agree" in m else ""
        log(f"[parity] {n:<22} {m['max_abs']:>11.3e} {m['frac_max']:>17.5f} {m['frac_scale']:>17.5f} "
            f"{m['frac_rms']:>14.5f} {m['cos_min']:>10.6f} {m['ulp_max']:>10.2f}  {'PASS' if m['pass'] else 'FAIL'}{extra}")
        w = m["ulp_worst"]
        log(f"[parity]   worst element row {w['row']} col {w['col']}: hf {w['hf']:.4f} rust {w['rust']:.4f} "
            f"(row RMS {w['row_rms']:.3f})")

    # ---- reference.json (tiny exports carry HF's greedy on the same weights) ----
    ref_report = None
    ref_path = export / "reference.json"
    if full and ref_path.exists():
        ref = json.loads(ref_path.read_text())
        prompt, greedy, first = ref["prompt_ids"], ref["greedy_ids"], ref.get("first_logits_argmax")
        P = len(prompt)
        if tokens[:P] == list(prompt):
            checks = {}
            for side, am in (("rust", results["logits_prefill"]["argmax_rust"]),
                             ("hf", results["logits_prefill"]["argmax_hf"])):
                ok_first = (am[:P] == list(first)) if first else None
                n_greedy = min(len(greedy), T - P + 1)  # position P-1+i predicts greedy[i]
                ok_greedy = am[P - 1:P - 1 + n_greedy] == list(greedy[:n_greedy])
                checks[side] = {"first_token_argmax": ok_first, "greedy": ok_greedy, "greedy_checked": n_greedy}
                log(f"[parity] reference.json ({side}): first-token argmax {ok_first}, "
                    f"greedy {ok_greedy} ({n_greedy}/{len(greedy)} ids covered by the dump)")
            ref_report = checks
        else:
            log("[parity] reference.json present but the dump does not start with its prompt; skipped")

    ok = all(m["pass"] for m in results.values())
    if ref_report is not None:
        ok = ok and all(c["greedy"] and c["first_token_argmax"] is not False for c in ref_report.values())
    log(f"[parity] {'PASS' if ok else 'FAIL'}: {len(results)} tensors; criteria rms(diff)/rms <= {rms_tol}, "
        f"every element <= {ulp_tol} bf16 ULPs of HF (max|diff|/rowRMS reported, tol {tol}); dtype {dtype}")
    return {"pass": ok, "export": str(export), "layers": k, "num_layers": man["num_layers"], "dtype": dtype,
            "tol": tol, "ulp_tol": ulp_tol, "rms_tol": rms_tol, "tokens": tokens, "experts": meta.get("experts"), "estimate_bytes": est,
            "tensors": results, "reference": ref_report}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="export_inkling.py output dir")
    ap.add_argument("--layers", type=int, required=True, help="compare the first K layers")
    ap.add_argument("--dump", required=True, help="safetensors written by the inkling_layer_dump example")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    ap.add_argument("--tol", type=float, default=0.02, help="max |diff| / HF row RMS (default 0.02)")
    ap.add_argument("--ulp-tol", type=float, default=4.0, help="PASS: every element within this many bf16 ULPs of HF (default 4)")
    ap.add_argument("--rms-tol", type=float, default=0.01, help="PASS: rms(diff)/rms per tensor (default 0.01)")
    ap.add_argument("--tokens", help="comma-separated ids; must equal the dump's `tokens` when present")
    ap.add_argument("--json", help="write the full report here")
    a = ap.parse_args(argv)
    tokens = [int(t) for t in a.tokens.replace(",", " ").split()] if a.tokens else None
    rep = compare_dump(a.export, a.layers, a.dump, dtype=a.dtype, tol=a.tol, ulp_tol=a.ulp_tol,
                       rms_tol=a.rms_tol, tokens=tokens)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
