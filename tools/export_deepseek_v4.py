#!/usr/bin/env python3
"""Export DeepSeek-V4-Flash into cascadia's sparse-MoE layout
(see docs/architectures/deepseek-v4.md).

V4-Flash's shell math (Hyper-Connections Sinkhorn mixing, sliding-window +
learned-compression KV, DSA indexer, attention sinks, grouped low-rank O) is
NOT sanely traceable to OpenVINO, so — like Kimi K2.6 and unlike MiniMax-M2 —
the shell is exported as dequantized **safetensors** for a Rust shell backend
("rust_dsv4", Phase 2). Only the per-expert SwiGLU FFNs become int4 binaries
(cascadia-int4-gemm layout) or OV IRs.

Ground truth: the vendored CPU reference in tools/deepseek_v4_ref/ (upstream
inference/model.py + pure-torch kernels). --tiny random-inits that reference,
exports it, and records its greedy tokens + logits in reference.json for the
Phase-3 byte-match test.

Output layout:

    <out>/manifest.json
    <out>/embed.safetensors                    # embed.weight
    <out>/head.safetensors                     # head.weight, norm.weight, hc_head_*
    <out>/shells/layer_NN.safetensors          # attn + gate + hc params, dequantized
    <out>/experts/layer_NN/expert_EEE.bin      # routed experts (int4_bin default)
    <out>/experts/layer_NN/expert_shared.bin   # the always-on shared expert
    <out>/reference.json                       # --tiny only

Checkpoint conventions (verified against the repo's inference/convert.py):
    FP8 linears:  <base>.weight e4m3 [N,K] + <base>.scale [ceil(N/128),ceil(K/128)]
                  (e8m0 or f32) — dequant = weight * block-expanded scale.
    FP4 experts:  <base>.weight int8-packed [N,K/2] (low nibble = even element,
                  e2m1 LUT ±{0,.5,1,1.5,2,3,4,6}) + <base>.scale [N,K/32] e8m0.
    Tensor names are already reference-style (layers.N.attn.wq_a.weight, ...).

Usage:
    # tiny synthetic model for the correctness harness (no download, no OV):
    python export_deepseek_v4.py --tiny --out /tmp/dsv4_tiny

    # full model from a local checkout (run on a big-RAM box):
    python export_deepseek_v4.py --model /path/to/DeepSeek-V4-Flash \
        --out /path/to/export [--layers 0-3]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from deepseek_v4_ref.kernels_ref import e8m0_to_f32, dequant_fp4_weight

# --------------------------------------------------------------------------
# int4_bin packing — same layout as tools/export_minimax_m2.py (validated by
# the Rust test int4_bin_expert_matches_fp32_within_tolerance). Duplicated
# here (30 lines) so --tiny needs no openvino import via that module.
# --------------------------------------------------------------------------
_INT4_GROUP = 32


def _pack_int4_grouped(w: np.ndarray):
    """[out, in] fp32 -> (packed u8 [out, in/2], scales bf16-LE [out, in/32])."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    out, inn = w.shape
    g = _INT4_GROUP
    assert inn % g == 0, f"in_dim {inn} not divisible by group {g}"
    wg = w.reshape(out, inn // g, g)
    max_abs = np.abs(wg).max(axis=2)
    s = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float32)
    q = np.clip(np.round(wg / s[:, :, None]), -8, 7).astype(np.int32)
    nib = (q + 8).astype(np.uint8).reshape(out, inn)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    u = s.view(np.uint32)
    bf = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")
    return packed.tobytes(), bf.tobytes()


def export_expert_bin(w1: np.ndarray, w3: np.ndarray, w2: np.ndarray, path: Path):
    """gate(w1), up(w3), down(w2): packed nibbles then bf16 scales, concatenated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for w in (w1, w3, w2):
            packed, scale = _pack_int4_grouped(w)
            f.write(packed)
            f.write(scale)


# --------------------------------------------------------------------------
# Weight sources: in-memory reference (--tiny) or streamed safetensors (full)
# --------------------------------------------------------------------------
class RefSource:
    """Weight getter over the in-memory reference model (tiny mode)."""

    def __init__(self, model):
        self.sd = dict(model.state_dict())

    def has(self, name):
        return name in self.sd

    def get(self, name) -> torch.Tensor:
        """fp32 tensor (int tensors pass through)."""
        t = self.sd[name]
        if t.dtype in (torch.int32, torch.int64):
            return t
        return t.to(torch.float32)


class CkptSource:
    """Streamed weight getter over the published V4 checkpoint. Dequantizes
    FP8 (block 128x128 `.scale`) and FP4 (per-32 `.scale`) on read; never
    holds more than one tensor at a time."""

    def __init__(self, model_dir: Path):
        from safetensors import safe_open
        self._safe_open = safe_open
        self.dir = model_dir
        idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.wm = idx["weight_map"]
        self._open = {}

    def _raw(self, name):
        shard = self.wm[name]
        h = self._open.get(shard)
        if h is None:
            h = self._safe_open(str(self.dir / shard), framework="pt")
            self._open[shard] = h
        return h.get_tensor(name)

    def has(self, name):
        return name in self.wm

    def get(self, name) -> torch.Tensor:
        t = self._raw(name)
        if t.dtype in (torch.int32, torch.int64):
            return t
        sk = name.rsplit(".", 1)[0] + ".scale"
        if name.endswith(".weight") and self.has(sk):
            s = self._raw(sk)
            if t.dtype == torch.float8_e4m3fn:
                n, k = t.shape
                sf = e8m0_to_f32(s)
                sf = sf.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
                return t.to(torch.float32) * sf
            # FP4: int8-packed [N, K/2] (or float4_e2m1fn_x2), scale [N, K/32]
            return dequant_fp4_weight(t, s)
        return t.to(torch.float32)


# --------------------------------------------------------------------------
# Shell tensor inventory (per layer) — mirrors the reference module tree.
# --------------------------------------------------------------------------
# stored f32 (precision-sensitive / tiny): norms, ape, compressor mats,
# hc params, sinks, gate. Stored bf16: the big attention matrices.
_F32_SUFFIXES = (
    "attn_norm.weight", "ffn_norm.weight",
    "attn.q_norm.weight", "attn.kv_norm.weight", "attn.attn_sink",
    "ffn.gate.weight", "ffn.gate.bias",
    "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
    "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
)
_BF16_SUFFIXES = (
    "attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
    "attn.wo_a.weight", "attn.wo_b.weight",
)
_COMPRESSOR_KEYS = ("ape", "norm.weight", "wgate.weight", "wkv.weight")
_INDEXER_KEYS = ("weights_proj.weight", "wq_b.weight")


def shell_tensor_names(li: int, ratio: int, is_hash: bool):
    """(name, store_dtype) pairs for layer li. `ratio` from compress_ratios."""
    p = f"layers.{li}."
    out = []
    for s in _F32_SUFFIXES:
        if s == "ffn.gate.bias" and is_hash:
            continue  # hash layers have tid2eid instead of a selection bias
        out.append((p + s, torch.float32))
    if is_hash:
        out.append((p + "ffn.gate.tid2eid", torch.int32))
    for s in _BF16_SUFFIXES:
        out.append((p + s, torch.bfloat16))
    if ratio:
        for s in _COMPRESSOR_KEYS:
            out.append((p + "attn.compressor." + s, torch.float32))
        if ratio == 4:  # indexer only on overlap-compressed layers
            for s in _INDEXER_KEYS:
                out.append((p + "attn.indexer." + s, torch.bfloat16))
            for s in _COMPRESSOR_KEYS:
                out.append((p + "attn.indexer.compressor." + s, torch.float32))
    return out


def save_safetensors(path: Path, tensors: dict):
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))


# --------------------------------------------------------------------------
# Export core (shared by tiny + full)
# --------------------------------------------------------------------------
def export_all(src, cfg: dict, out: Path, layer_ids):
    ratios = cfg["compress_ratios"]

    # embed
    ep = out / "embed.safetensors"
    if not ep.exists():
        save_safetensors(ep, {"embed.weight": src.get("embed.weight").to(torch.bfloat16)})
        print("[embed] written", flush=True)

    # head (+ model-level hyper-connection head params)
    hp = out / "head.safetensors"
    if not hp.exists():
        save_safetensors(hp, {
            "head.weight": src.get("head.weight").to(torch.bfloat16),
            "norm.weight": src.get("norm.weight"),
            "hc_head_fn": src.get("hc_head_fn"),
            "hc_head_base": src.get("hc_head_base"),
            "hc_head_scale": src.get("hc_head_scale"),
        })
        print("[head] written", flush=True)

    for li in layer_ids:
        ratio = ratios[li]
        is_hash = li < cfg["n_hash_layers"]
        sp = out / "shells" / f"layer_{li:02d}.safetensors"
        if not sp.exists():
            tensors = {}
            for name, dt in shell_tensor_names(li, ratio, is_hash):
                tensors[name.removeprefix(f"layers.{li}.")] = (
                    src.get(name) if dt in (torch.int32,) else src.get(name).to(dt))
            save_safetensors(sp, tensors)

        edir = out / "experts" / f"layer_{li:02d}"
        base = f"layers.{li}.ffn"
        jobs = [(f"{base}.shared_experts", edir / "expert_shared")]
        jobs += [(f"{base}.experts.{ei}", edir / f"expert_{ei:03d}")
                 for ei in range(cfg["n_routed_experts"])]
        for wbase, dst in jobs:
            binp = dst.with_suffix(".bin")
            if binp.exists() and binp.stat().st_size > 0:
                continue
            w1 = src.get(wbase + ".w1.weight").numpy()
            w3 = src.get(wbase + ".w3.weight").numpy()
            w2 = src.get(wbase + ".w2.weight").numpy()
            export_expert_bin(w1, w3, w2, binp)
        print(f"[layer {li}] shell + {cfg['n_routed_experts']}+shared experts done",
              flush=True)


def write_manifest(cfg: dict, out: Path, layer_ids, experts_format: str, eos):
    manifest = {
        "arch": "deepseek_v4",
        "shell_backend": "rust_dsv4",          # Phase-2 engine backend name
        "experts_format": experts_format,
        "num_layers": cfg["n_layers"],
        "exported_layers": list(layer_ids),
        "hidden_size": cfg["hidden_size"],
        "vocab_size": cfg["vocab_size"],
        "eos_token_ids": eos,
        # attention
        "n_heads": cfg["n_heads"],
        "head_dim": cfg["head_dim"],
        "rope_head_dim": cfg["rope_head_dim"],
        "q_lora_rank": cfg["q_lora_rank"],
        "o_lora_rank": cfg["o_lora_rank"],
        "o_groups": cfg["o_groups"],
        "window_size": cfg["window_size"],
        "compress_ratios": list(cfg["compress_ratios"]),
        "rope_theta": cfg["rope_theta"],
        "compress_rope_theta": cfg["compress_rope_theta"],
        "rope_factor": cfg["rope_factor"],
        "beta_fast": cfg["beta_fast"],
        "beta_slow": cfg["beta_slow"],
        "original_seq_len": cfg["original_seq_len"],
        "index_n_heads": cfg["index_n_heads"],
        "index_head_dim": cfg["index_head_dim"],
        "index_topk": cfg["index_topk"],
        # hyper-connections
        "hc_mult": cfg["hc_mult"],
        "hc_sinkhorn_iters": cfg["hc_sinkhorn_iters"],
        "hc_eps": cfg["hc_eps"],
        "rms_norm_eps": cfg["norm_eps"],
        # moe
        "n_routed_experts": cfg["n_routed_experts"],
        "n_shared_experts": 1,
        "top_k": cfg["n_activated_experts"],
        "score_func": cfg["score_func"],
        "route_scale": cfg["route_scale"],
        "n_hash_layers": cfg["n_hash_layers"],
        "swiglu_limit": cfg["swiglu_limit"],
        "expert_intermediate": cfg["moe_inter_dim"],
        "has_shared_expert": True,
        # wire notes for the runner
        "activation_shape": "b,s,hc_mult,hidden",   # HC copies travel inter-stage
        "needs_input_ids": cfg["n_hash_layers"] > 0,  # hash gate reads token ids
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] {out/'manifest.json'}", flush=True)


# --------------------------------------------------------------------------
# Tiny mode
# --------------------------------------------------------------------------
TINY_KW = dict(
    max_batch_size=1, max_seq_len=64, dtype="bf16",
    vocab_size=128, dim=64, moe_inter_dim=32,
    n_layers=4, n_hash_layers=1, n_mtp_layers=1,
    n_heads=4, n_routed_experts=8, n_shared_experts=1, n_activated_experts=2,
    score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
    q_lora_rank=32, head_dim=80, rope_head_dim=16, o_groups=2, o_lora_rank=32,
    window_size=8, compress_ratios=(0, 4, 16, 4, 0),
    compress_rope_theta=160000.0, original_seq_len=0, rope_theta=10000.0,
    index_n_heads=2, index_head_dim=32, index_topk=4,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
)


# Medium config: keeps the real model's STRUCTURAL params that the tiny
# config never exercised (o_groups=8, rope_head_dim=64, YaRN on via
# original_seq_len>0, ratio-128 compressor, 3 hash layers) while staying
# small enough to run the torch reference fast. A dimension-dependent engine
# bug reproduces here with random weights — no checkpoint needed.
MED_KW = dict(
    max_batch_size=1, max_seq_len=64, dtype="bf16",
    vocab_size=256, dim=512, moe_inter_dim=128,
    n_layers=5, n_hash_layers=3, n_mtp_layers=1,
    n_heads=16, n_routed_experts=8, n_shared_experts=1, n_activated_experts=2,
    score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
    # real attention proportions: head_dim=512 (nope=448 != rope=64),
    # o_groups=8, index_head_dim=128 — the structural dims tiny never had
    q_lora_rank=256, head_dim=512, rope_head_dim=64, o_groups=16, o_lora_rank=128,
    window_size=8, compress_ratios=(0, 0, 0, 0, 0, 0),  # all-window + groups(8)!=heads(16)
    compress_rope_theta=160000.0, original_seq_len=256, rope_theta=10000.0,
    rope_factor=8.0, beta_fast=32.0, beta_slow=1.0,
    index_n_heads=8, index_head_dim=128, index_topk=16,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
)


# Big config: MED's real structural dims PLUS ratio-4/128 compressor+indexer
# layers, and enough layers (8, hash only layer 0) to split across 4 ranks so a
# ratio-128 compressor lands as the ENTRY layer of a mid rank — the exact real
# condition (real layer 11 = ratio-128 = rank1 entry) that MED punted on with
# all-zero ratios. Real attention scale: dim=4096 -> hc*dim=16384 wire width,
# n_heads=64, o_groups=8 (8 heads/group MQA grouping tiny/med never had).
BIG_KW = dict(
    max_batch_size=1, max_seq_len=64, dtype="bf16",
    vocab_size=256, dim=4096, moe_inter_dim=256,
    n_layers=8, n_hash_layers=1, n_mtp_layers=1,
    n_heads=64, n_routed_experts=8, n_shared_experts=1, n_activated_experts=2,
    score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
    q_lora_rank=1024, head_dim=512, rope_head_dim=64, o_groups=8, o_lora_rank=1024,
    window_size=128, compress_ratios=(0, 4, 128, 4, 128, 4, 128, 0, 0),
    compress_rope_theta=160000.0, original_seq_len=256, rope_theta=10000.0,
    rope_factor=8.0, beta_fast=32.0, beta_slow=1.0,
    index_n_heads=8, index_head_dim=128, index_topk=64,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
)


def _int4_roundtrip(w: torch.Tensor) -> torch.Tensor:
    """Pass a [out, in] weight through the exact int4_bin pack+dequant the
    Rust loader sees, so a --med reference and the Rust engine share the
    *same* lossy expert weights and any residual divergence is pure shell
    logic (not int4 quant noise). Mirrors `_pack_int4_grouped` + loader
    `dequant_int4`."""
    wn = np.ascontiguousarray(w.detach().float().cpu().numpy(), dtype=np.float32)
    out, inn = wn.shape
    gsz = _INT4_GROUP
    wg = wn.reshape(out, inn // gsz, gsz)
    max_abs = np.abs(wg).max(axis=2)
    s = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float32)
    q = np.clip(np.round(wg / s[:, :, None]), -8, 7)          # nibble-8 value
    u = np.ascontiguousarray(s).view(np.uint32)
    bf = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32)  # f32->bf16
    s_bf16 = (bf << 16).view(np.float32)                      # bf16 bits->f32
    w_rt = (q * s_bf16[:, :, None]).reshape(out, inn).astype(np.float32)
    return torch.from_numpy(w_rt).to(w.dtype)


def build_ref(kw, int4_experts=False):
    """Deterministically-initialized reference model for the given config.
    With int4_experts, round-trips the routed+shared expert weights through
    int4 so the reference matches the Rust engine's dequantized experts."""
    from deepseek_v4_ref.model import ModelArgs, Transformer
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    model = Transformer(ModelArgs(**kw))
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dtype in (torch.float32, torch.bfloat16, torch.float16):
                if "norm" in name and p.ndim == 1:
                    p.copy_(1.0 + 0.05 * torch.randn(p.shape, generator=g,
                                                     dtype=torch.float32).to(p.dtype))
                else:
                    p.copy_((0.05 * torch.randn(p.shape, generator=g,
                                                dtype=torch.float32)).to(p.dtype))
            elif p.dtype == torch.int32:  # tid2eid
                p.copy_(torch.randint(0, kw["n_routed_experts"], p.shape,
                                      generator=g, dtype=torch.int32))
        if int4_experts:
            for name, p in model.named_parameters():
                if (".experts." in name or ".shared_experts." in name) \
                        and name.endswith(".weight") and p.ndim == 2:
                    p.copy_(_int4_roundtrip(p))
    return model


def build_tiny():
    return build_ref(TINY_KW)


def tiny_reference_run(model, prompt, n_gen):
    """Greedy decode; returns tokens + logit snapshots for the harness."""
    snaps = {}
    layer_h, attn_h = {}, {}  # per-layer hidden + attn output (last prefill tok)

    def mk_hook(store, i):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            store[i] = o[0, -1].to(torch.float32).flatten().tolist()
        return hook

    ffn_in = {}

    def mk_pre_hook(i):
        def hook(_m, inp):
            x = inp[0]
            ffn_in[i] = x[0, -1].to(torch.float32).flatten().tolist()
        return hook

    handles = []
    for i, lyr in enumerate(model.layers):
        handles.append(lyr.register_forward_hook(mk_hook(layer_h, i)))
        handles.append(lyr.attn.register_forward_hook(mk_hook(attn_h, i)))
        handles.append(lyr.ffn.register_forward_pre_hook(mk_pre_hook(i)))
    with torch.no_grad():
        logits = model(torch.tensor([prompt]), 0)
        snaps["prefill"] = logits[0].to(torch.float32).tolist()
        snaps["prefill_hiddens"] = [layer_h[i] for i in range(len(model.layers))]
        snaps["prefill_attn"] = [attn_h[i] for i in range(len(model.layers))]
        snaps["prefill_ffn_in"] = [ffn_in[i] for i in range(len(model.layers))]
        for hnd in handles:
            hnd.remove()
        gen = [int(logits[0].argmax())]
        step_logits = [logits[0].to(torch.float32).tolist()]  # per-decode-step
        for step in range(n_gen - 1):
            pos = len(prompt) + step
            logits = model(torch.tensor([[gen[-1]]]), pos)
            if step == 0:
                snaps["decode0"] = logits[0].to(torch.float32).tolist()
            step_logits.append(logits[0].to(torch.float32).tolist())
            gen.append(int(logits[0].argmax()))
    snaps["step_logits"] = step_logits
    return gen, snaps


def cfg_from_args_obj(a) -> dict:
    keys = ("vocab_size", "moe_inter_dim", "n_layers", "n_hash_layers",
            "n_heads", "n_routed_experts", "n_activated_experts", "score_func",
            "route_scale", "swiglu_limit", "q_lora_rank", "head_dim",
            "rope_head_dim", "norm_eps", "o_groups", "o_lora_rank",
            "window_size", "compress_ratios", "compress_rope_theta",
            "original_seq_len", "rope_theta", "rope_factor", "beta_fast",
            "beta_slow", "index_n_heads", "index_head_dim", "index_topk",
            "hc_mult", "hc_sinkhorn_iters", "hc_eps")
    cfg = {k: getattr(a, k) for k in keys}
    cfg["hidden_size"] = a.dim
    return cfg


# --------------------------------------------------------------------------
def parse_layers(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return [x for x in out if 0 <= x < n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="local path of DeepSeek-V4-Flash checkout")
    ap.add_argument("--tiny", action="store_true", help="synthetic tiny model")
    ap.add_argument("--med", action="store_true",
                    help="synthetic medium model (real structural params: "
                         "o_groups=8, rope_head_dim=64, YaRN, ratio-128)")
    ap.add_argument("--big", action="store_true",
                    help="synthetic big model: real attention dims (4096/64h/"
                         "o_groups=8) + ratio-4/128 compressor/indexer layers "
                         "across 8 layers (4-rank splittable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="", help="subset e.g. 0-3; default all")
    ap.add_argument("--experts", choices=["int4_bin"], default="int4_bin",
                    help="dsv4 experts are always int4_bin; OV expert "
                         "acceleration is the optional experts_ov/ overlay built "
                         "separately (consumed by OvExperts::from_env)")
    ap.add_argument("--ref-prompt", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18",
                    help="comma token ids for the --tiny reference run")
    ap.add_argument("--ref-gen", type=int, default=8)
    args = ap.parse_args()
    if not args.tiny and not args.med and not args.big and not args.model:
        ap.error("need --model, --tiny, --med, or --big")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.tiny or args.med or args.big:
        kw = BIG_KW if args.big else (MED_KW if args.med else TINY_KW)
        tag = "big" if args.big else ("med" if args.med else "tiny")
        print(f"[{tag}] building deterministic reference ...", flush=True)
        model = build_ref(kw, int4_experts=(args.med or args.big))
        from deepseek_v4_ref.model import ModelArgs
        a = ModelArgs(**kw)
        cfg = cfg_from_args_obj(a)
        layer_ids = parse_layers(args.layers, cfg["n_layers"])
        src = RefSource(model)
        export_all(src, cfg, out, layer_ids)
        write_manifest(cfg, out, layer_ids, args.experts, eos=[cfg["vocab_size"] - 1])
        prompt = [int(x) for x in args.ref_prompt.split(",")]
        gen, snaps = tiny_reference_run(model, prompt, args.ref_gen)
        (out / "reference.json").write_text(json.dumps({
            "prompt_ids": prompt,
            "greedy_tokens": prompt + gen,
            "generated": gen,
            "prefill_logits": snaps["prefill"],
            "decode0_logits": snaps["decode0"],
            "step_logits": snaps["step_logits"],
            "prefill_hiddens": snaps["prefill_hiddens"],
            "prefill_attn": snaps["prefill_attn"],
            "prefill_ffn_in": snaps["prefill_ffn_in"],
        }, indent=2))
        print(f"[reference] generated={gen}", flush=True)
        print("[done tiny]", flush=True)
        return

    # ---- full checkpoint ----
    md = Path(args.model)
    hf = json.loads((md / "config.json").read_text())
    # config.json -> ModelArgs field names (see reference ModelArgs)
    cfg = dict(
        hidden_size=hf["hidden_size"],
        vocab_size=hf["vocab_size"],
        moe_inter_dim=hf["moe_intermediate_size"],
        n_layers=hf["num_hidden_layers"],
        n_hash_layers=hf.get("num_hash_layers", 0),
        n_heads=hf["num_attention_heads"],
        n_routed_experts=hf["n_routed_experts"],
        n_activated_experts=hf["num_experts_per_tok"],
        score_func=hf.get("scoring_func", "sqrtsoftplus"),
        route_scale=hf.get("routed_scaling_factor", 1.0),
        swiglu_limit=hf.get("swiglu_limit", 0.0),
        q_lora_rank=hf["q_lora_rank"],
        head_dim=hf["head_dim"],
        rope_head_dim=hf["qk_rope_head_dim"],
        norm_eps=hf.get("rms_norm_eps", 1e-6),
        o_groups=hf["o_groups"],
        o_lora_rank=hf["o_lora_rank"],
        window_size=hf["sliding_window"],
        compress_ratios=hf["compress_ratios"],
        compress_rope_theta=hf.get("compress_rope_theta", 40000.0),
        original_seq_len=(hf.get("rope_scaling") or {}).get(
            "original_max_position_embeddings", 0),
        rope_theta=hf.get("rope_theta", 10000.0),
        rope_factor=(hf.get("rope_scaling") or {}).get("factor", 1),
        beta_fast=(hf.get("rope_scaling") or {}).get("beta_fast", 32),
        beta_slow=(hf.get("rope_scaling") or {}).get("beta_slow", 1),
        index_n_heads=hf["index_n_heads"],
        index_head_dim=hf["index_head_dim"],
        index_topk=hf["index_topk"],
        hc_mult=hf.get("hc_mult", 4),
        hc_sinkhorn_iters=hf.get("hc_sinkhorn_iters", 20),
        hc_eps=hf.get("hc_eps", 1e-6),
    )
    print(f"[cfg] {json.dumps({k: v for k, v in cfg.items() if k != 'compress_ratios'})}",
          flush=True)
    layer_ids = parse_layers(args.layers, cfg["n_layers"])
    src = CkptSource(md)
    export_all(src, cfg, out, layer_ids)
    eos = []
    gc = md / "generation_config.json"
    if gc.exists():
        e = json.loads(gc.read_text()).get("eos_token_id")
        eos = e if isinstance(e, list) else ([e] if isinstance(e, int) else [])
    if not eos:
        e = hf.get("eos_token_id")
        eos = e if isinstance(e, list) else ([e] if isinstance(e, int) else [])
    write_manifest(cfg, out, layer_ids, args.experts, eos)
    # Carry the tokenizer + every serving sidecar. The chat template ships as a
    # standalone `chat_template.jinja` on modern DeepSeek checkpoints (instruct
    # variants) rather than inside tokenizer_config.json; without it the API
    # falls back to a generic formatter and instruct prompts degrade. Base
    # checkpoints simply won't have the file — the `.exists()` guard skips it.
    for fn in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
    ):
        if (md / fn).exists():
            shutil.copy(md / fn, out / fn)
    print("[done full]", flush=True)


if __name__ == "__main__":
    main()
