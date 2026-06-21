#!/usr/bin/env python3
"""Quality probe: run MiniMax-M2 with FULL-PRECISION (fp32) experts.

Uses the exported OV embed/shells/head (so routing + attention are the
exact same graphs the engine runs) but dispatches the routed experts by
dequantizing the original block-FP8 weights to fp32 and running the SwiGLU
in numpy — i.e. the experts are effectively unquantized. This isolates
expert precision: if output becomes coherent here, the int4 experts are
the cause of the degradation (and int8 would help); if it still degrades,
expert precision is not the issue.

Needs no extra disk (reads the FP8 source already on disk). Slow — for a
short quality check, not throughput.

    python test_minimax_m2_fp_experts.py \
        --model-dir /media/tatef/extssd/m2/export \
        --fp8-dir   /media/tatef/extssd/m2/fp8 \
        --prompt "The capital of France is" --max-new 24 --rep-penalty 1.3 --rep-window 64
"""
import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov

import export_minimax_m2 as E  # StReader + dequant_fp8_block


def silu(x):
    return x / (1.0 + np.exp(-x))


def quant_dequant(w, mode):
    """Simulate weight quantization on an fp32 [out,in] matrix, matching the
    exporter: int8 = per-output-channel symmetric (max/127); int4 = per-
    32-col-group symmetric (max/7). 'none' returns w unchanged. Lets the
    probe compare FP8-precision experts vs int8/int4-*linear*-quant experts
    with everything else (routing, attention, the SwiGLU math) identical."""
    if mode == "none":
        return w
    if mode == "int8":
        s = np.abs(w).max(axis=1, keepdims=True) / 127.0
        s[s == 0] = 1.0
        return np.round(w / s).clip(-127, 127) * s
    if mode == "int4":
        out, inn = w.shape
        g = 32
        wg = w.reshape(out, inn // g, g)
        s = np.abs(wg).max(axis=2, keepdims=True) / 7.0
        s[s == 0] = 1.0
        return (np.round(wg / s).clip(-8, 7) * s).reshape(out, inn)
    if mode == "nf4":
        # NormalFloat-4: 16 quantile levels of a unit normal (QLoRA /
        # bitsandbytes), per-group absmax scale, nearest-level rounding.
        # Distribution-matched 4-bit — the candidate fix.
        levels = np.array([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
            0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
            0.7229568362236023, 1.0], dtype=np.float32)
        out, inn = w.shape
        g = 64
        wg = w.reshape(out, inn // g, g)
        absmax = np.abs(wg).max(axis=2, keepdims=True)
        absmax[absmax == 0] = 1.0
        wn = wg / absmax
        idx = np.abs(wn[..., None] - levels).argmin(axis=-1)
        return (levels[idx] * absmax).reshape(out, inn)
    raise ValueError(mode)


def sample(logits, history, rep_penalty, rep_window, temperature):
    work = logits.astype(np.float32).copy()
    if abs(rep_penalty - 1.0) > 1e-9 and history:
        hist = history if rep_window == 0 else history[-rep_window:]
        for tok in hist:
            if 0 <= tok < work.size:
                l = work[tok]
                work[tok] = l / rep_penalty if l > 0 else l * rep_penalty
    if temperature <= 0.0:
        return int(np.argmax(work))
    work = work / temperature
    work -= work.max()
    p = np.exp(work)
    p /= p.sum()
    return int(np.argmax(p))  # temperature path still argmax-of-softmax here (deterministic probe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--fp8-dir", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--rep-penalty", type=float, default=1.0)
    ap.add_argument("--rep-window", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--quant", choices=["none", "int8", "int4", "nf4"], default="none",
                    help="simulate expert weight quantization (none=fp32/FP8-precision)")
    ap.add_argument("--ov-experts", action="store_true",
                    help="dispatch the export's compiled OV expert IRs (like the Rust engine) "
                         "instead of numpy SwiGLU from the FP8 source — isolates OV-expert-IR "
                         "execution from the numpy reference")
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()

    md = Path(args.model_dir)
    man = json.loads((md / "manifest.json").read_text())
    H, KV, D, K = man["hidden_size"], man["num_kv_heads"], man["head_dim"], man["top_k"]
    layers = man.get("exported_layers") or list(range(man["num_layers"]))
    eos = set(man.get("eos_token_ids", []))

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(md / "tokenizer.json"))

    core = ov.Core()
    try:
        core.set_property("CPU", {"SNIPPETS_MODE": "DISABLE"})
    except Exception:
        pass
    embed = core.compile_model(str(md / "layer0" / "openvino_model.xml"), args.device)
    head = core.compile_model(str(md / "head" / "openvino_model.xml"), args.device)
    shells = {li: core.compile_model(str(md / "shells" / f"layer_{li:02d}" / "openvino_model.xml"),
                                     args.device) for li in layers}

    reader = E.StReader(Path(args.fp8_dir))
    expert_cache = {}

    def expert_f32(li, ei):
        key = (li, ei)
        w = expert_cache.get(key)
        if w is None:
            b = f"model.layers.{li}.block_sparse_moe.experts.{ei}"
            w = (quant_dequant(reader.weight(b + ".w1").numpy(), args.quant),   # gate
                 quant_dequant(reader.weight(b + ".w3").numpy(), args.quant),   # up
                 quant_dequant(reader.weight(b + ".w2").numpy(), args.quant))   # down
            if len(expert_cache) < 600:              # ~34G cap; speeds up short runs
                expert_cache[key] = w
        return w

    expert_ov_cache = {}

    def expert_ov(li, ei, apn):
        c = expert_ov_cache.get((li, ei))
        if c is None:
            c = core.compile_model(
                str(md / "experts" / f"layer_{li:02d}" / f"expert_{ei:03d}" / "openvino_model.xml"),
                args.device)
            if len(expert_ov_cache) < 4000:
                expert_ov_cache[(li, ei)] = c
        r = c.create_infer_request()
        r.infer({"x": apn.reshape(1, 1, H).astype(np.float32)})
        return np.array(r.get_tensor("y").data).reshape(-1)

    past_k = {li: np.zeros((1, KV, 0, D), np.float32) for li in layers}
    past_v = {li: np.zeros((1, KV, 0, D), np.float32) for li in layers}

    def step(token, pos):
        ids = np.array([[token]], dtype=np.int64)
        h = embed(ids)[embed.output("hidden")].astype(np.float32)
        for li in layers:
            req = shells[li].create_infer_request()
            req.infer({"x": h, "past_k": past_k[li], "past_v": past_v[li],
                       "past_seq_len": np.array(pos, dtype=np.int64)})
            apn = np.array(req.get_tensor("attn_out_post_norm").data).reshape(-1)
            residual = np.array(req.get_tensor("attn_residual").data)
            ids_r = np.array(req.get_tensor("routing_ids").data).reshape(-1)
            w_r = np.array(req.get_tensor("routing_weights").data).reshape(-1)
            pk = req.get_tensor("present_k").data
            pv = req.get_tensor("present_v").data
            past_k[li] = np.concatenate([past_k[li], pk], axis=2)
            past_v[li] = np.concatenate([past_v[li], pv], axis=2)
            moe = np.zeros_like(residual)
            for k in range(K):
                ei = int(ids_r[k])
                if args.ov_experts:
                    y = expert_ov(li, ei, apn).reshape(residual.shape)
                else:
                    gate, up, down = expert_f32(li, ei)
                    inter = silu(gate @ apn) * (up @ apn)   # [inter]
                    y = (down @ inter).reshape(residual.shape)
                moe = moe + float(w_r[k]) * y
            h = residual + moe
        logits = head(h.astype(np.float32))[head.output("logits")].reshape(-1)
        return logits

    prompt_ids = tok.encode(args.prompt, add_special_tokens=True).ids
    print(f"prompt={args.prompt!r} ids={prompt_ids}", flush=True)
    pos = 0
    logits = None
    for t in prompt_ids:
        logits = step(t, pos); pos += 1
    gen, history = [], []
    while len(gen) < args.max_new:
        nxt = sample(logits, history, args.rep_penalty, args.rep_window, args.temperature)
        gen.append(nxt); history.append(nxt)
        if nxt in eos:
            break
        logits = step(nxt, pos); pos += 1
    print("completion:", repr(tok.decode(gen)), flush=True)


if __name__ == "__main__":
    main()
