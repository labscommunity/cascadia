#!/usr/bin/env python3
"""Kimi-K3 exporter -> cascadia sparse-moe layout (arch == "kimi_k3").

Usage:
  --validate CONFIG.json     run the hard-fail config contract (no output)
  --tiny OUT                 write a deterministic tiny model
  --model DIR --out DIR      export a real checkpoint (resumable)
  --selftest                 fp4 pack/unpack round-trip

Layout written:
  <out>/manifest.json
  <out>/shells/layer_NN.safetensors     bf16 attention + norms + LatentMoE proj
  <out>/experts/layer_NN/expert_EEE.bin fp4 e2m1 + u8 E8M0 scales
  <out>/embed.safetensors, <out>/head.safetensors

K3's routed experts already ship as mxfp4 (e2m1 values + E8M0 group-32 scales),
so the real path REPACKS them byte-wise instead of dequantizing and
requantizing. Regrinding an already-4-bit grid onto a linear int4 grid would
lose the 0.5/1.5/3 levels for no benefit; everything else in the checkpoint
(attention, shared experts, dense MLP, lm_head) is bf16 already.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepseek_v4_ref.kernels_ref import (_pow2_round_up, _quantize_e2m1,
                                         dequant_fp4_weight, e8m0_to_f32)

FP4_GROUP = 32
FP4_MAX = 6.0

# e2m1 magnitude grid; index = nibble & 0x7, sign = nibble >> 3
_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


# --------------------------------------------------------------------------
# fp4 expert bin — packed nibbles then u8 E8M0 scales, per section.
# Section bytes: out*in/2 (nibbles) + out*(in/32) (scales).
# Low nibble = even column, matching the dsv4 convention and MmapExpert.
# --------------------------------------------------------------------------


def pack_fp4_from_f32(w: np.ndarray):
    """[out, in] fp32 -> (packed u8 [out, in/2], scales u8 [out, in/32]).

    Power-of-two (E8M0) group scales + RNE onto the e2m1 grid, matching the
    upstream mxfp4 convention. Used by --tiny; the real path repacks instead.
    """
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    out, inn = t.shape
    assert inn % FP4_GROUP == 0, f"in_dim {inn} not divisible by {FP4_GROUP}"
    g = t.reshape(out, inn // FP4_GROUP, FP4_GROUP)
    amax = g.abs().amax(-1).clamp_min(1e-30)
    s = _pow2_round_up(amax / FP4_MAX)                       # [out, ngroups]
    q = _quantize_e2m1((g / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))

    mag = q.abs().numpy().reshape(out, inn)
    sign = (q.numpy().reshape(out, inn) < 0).astype(np.uint8)
    idx = np.abs(mag[..., None] - _E2M1).argmin(-1).astype(np.uint8)
    nib = (idx | (sign << 3)).astype(np.uint8)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)

    # f32 power-of-two -> E8M0 biased exponent byte
    e = np.round(np.log2(s.numpy())).astype(np.int32) + 127
    return packed.tobytes(), np.clip(e, 0, 254).astype(np.uint8).tobytes()


def unpack_fp4(packed: bytes, scales: bytes, out: int, inn: int) -> np.ndarray:
    """Inverse of `pack_fp4_from_f32` — used by the round-trip selftest."""
    raw = np.frombuffer(packed, dtype=np.uint8).reshape(out, inn // 2)
    s = np.frombuffer(scales, dtype=np.uint8).reshape(out, inn // FP4_GROUP)
    b = torch.from_numpy(raw.copy())
    # must be viewed as e8m0 — `e8m0_to_f32` treats a plain uint8 as a literal
    # scale value rather than a biased exponent
    sc = torch.from_numpy(s.copy()).view(torch.float8_e8m0fnu)
    return dequant_fp4_weight(b, sc, group=FP4_GROUP).numpy()


def section_bytes(out: int, inn: int) -> int:
    return out * inn // 2 + out * (inn // FP4_GROUP)


def export_expert_bin(w1: np.ndarray, w3: np.ndarray, w2: np.ndarray, path: Path):
    """SiTU FFN: gate(w1), up(w3), down(w2) — nibbles then E8M0 scales each."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for w in (w1, w3, w2):
            packed, scale = pack_fp4_from_f32(w)
            f.write(packed)
            f.write(scale)


def repack_expert_bin(sections, path: Path):
    """Real path: write already-mxfp4 (packed, scale) pairs straight through."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for packed, scale in sections:
            f.write(np.ascontiguousarray(packed, dtype=np.uint8).tobytes())
            f.write(np.ascontiguousarray(scale, dtype=np.uint8).tobytes())


# --------------------------------------------------------------------------
# Config contract — hard-fail on anything the shell does not implement.
# --------------------------------------------------------------------------


class ConfigError(SystemExit):
    pass


def _require(cond, msg):
    if not cond:
        raise ConfigError(f"[export_kimi_k3] config contract violated: {msg}")


def load_and_validate_config(path: Path) -> dict:
    """Load HF config.json and enforce the k3 shell contract. Fails loudly on
    any surprise rather than silently mis-exporting."""
    raw = json.loads(Path(path).read_text())
    # multimodal wrapper: the text model lives under text_config; the ViT is dropped
    c = raw.get("text_config", raw)
    _require(c.get("model_type") in ("kimi_linear", "kimi_k3"),
             f"model_type must be kimi_linear/kimi_k3, got {c.get('model_type')!r}")

    def g(k):
        _require(k in c, f"missing key '{k}'")
        return c[k]

    la = c.get("linear_attn_config")
    _require(isinstance(la, dict), "linear_attn_config missing (hybrid KDA/MLA)")
    _require(bool(c.get("mla_use_nope")), "mla_use_nope must be true (no RoPE implemented)")
    _require(c.get("moe_router_activation_func") == "sigmoid",
             f"router must be sigmoid, got {c.get('moe_router_activation_func')!r}")
    _require(c.get("hidden_act") == "situ", f"hidden_act must be situ, got {c.get('hidden_act')!r}")
    _require(int(c.get("num_expert_group", 1)) == 1, "num_expert_group must be 1")
    _require(int(c.get("topk_group", 1)) == 1, "topk_group must be 1")
    _require(c.get("attn_res_block_size") is not None, "attn_res_block_size required")
    _require(int(c.get("num_nextn_predict_layers", 0)) == 0, "MTP layers not supported")

    n = int(g("num_hidden_layers"))
    kda = set(int(i) for i in la.get("kda_layers", []))
    _require(kda, "linear_attn_config.kda_layers is empty")
    # layer 0 appears in neither list and falls through to MLA; ids >= n are unused
    _require(max(kda) < n, f"kda_layers has an id >= num_hidden_layers ({n})")

    q = raw.get("quantization_config", c.get("quantization_config", {})) or {}
    fmt = q.get("format")
    _require(fmt in (None, "mxfp4-pack-quantized"),
             f"expected mxfp4-pack-quantized experts, got {fmt!r}")
    grp = (q.get("config_groups", {}).get("group_0", {}).get("weights", {}) or {})
    if grp:
        _require(int(grp.get("group_size", FP4_GROUP)) == FP4_GROUP,
                 f"expert group_size must be {FP4_GROUP}, got {grp.get('group_size')}")
        _require(int(grp.get("num_bits", 4)) == 4, "experts must be 4-bit")

    cfg = dict(
        hidden_size=int(g("hidden_size")),
        num_hidden_layers=n,
        first_k_dense_replace=int(c.get("first_k_dense_replace", 0)),
        attn_res_block_size=int(c["attn_res_block_size"]),
        kda_layers=sorted(kda),
        rms_norm_eps=float(c.get("rms_norm_eps", 1e-5)),
        situ_beta=float(c.get("activation_situ_beta", 1.0)),
        situ_linear_beta=c.get("activation_situ_linear_beta"),
        num_heads=int(la["num_heads"]),
        head_dim=int(la["head_dim"]),
        conv_size=int(la["short_conv_kernel_size"]),
        gate_lower_bound=la.get("gate_lower_bound"),
        use_full_rank_gate=bool(la.get("use_full_rank_gate", False)),
        num_attention_heads=int(g("num_attention_heads")),
        q_lora_rank=int(g("q_lora_rank")),
        kv_lora_rank=int(g("kv_lora_rank")),
        qk_nope_head_dim=int(g("qk_nope_head_dim")),
        qk_rope_head_dim=int(g("qk_rope_head_dim")),
        v_head_dim=int(g("v_head_dim")),
        mla_use_output_gate=bool(c.get("mla_use_output_gate", False)),
        num_experts=int(g("num_experts")),
        top_k=int(g("num_experts_per_token")),
        num_shared_experts=int(c.get("num_shared_experts") or 0),
        routed_expert_hidden_size=int(c.get("routed_expert_hidden_size") or g("hidden_size")),
        moe_intermediate_size=int(g("moe_intermediate_size")),
        latent_moe_use_norm=bool(c.get("latent_moe_use_norm", False)),
        routed_scaling_factor=float(c.get("routed_scaling_factor", 1.0)),
        moe_renormalize=bool(c.get("moe_renormalize", True)),
        intermediate_size=int(g("intermediate_size")),
        vocab_size=int(g("vocab_size")),
        eos_token_ids=[int(c.get("eos_token_id", 0))],
    )
    _require(cfg["routed_expert_hidden_size"] % FP4_GROUP == 0,
             "routed_expert_hidden_size must be a multiple of 32")
    _require(cfg["moe_intermediate_size"] % FP4_GROUP == 0,
             "moe_intermediate_size must be a multiple of 32")
    return cfg


def build_manifest(cfg: dict) -> dict:
    """The <out>/manifest.json the engine reads (arch == 'kimi_k3' sniff)."""
    m = dict(cfg)
    m["arch"] = "kimi_k3"
    m["experts_format"] = "fp4_bin"
    m["shell_backend"] = "rust_k3"
    m["expert_bin_bytes"] = (
        2 * section_bytes(cfg["moe_intermediate_size"], cfg["routed_expert_hidden_size"])
        + section_bytes(cfg["routed_expert_hidden_size"], cfg["moe_intermediate_size"])
    )
    return m


def write_manifest(cfg: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(build_manifest(cfg), indent=2) + "\n")
    print(f"[manifest] {out / 'manifest.json'}", flush=True)


# --------------------------------------------------------------------------
# Disk pre-flight
# --------------------------------------------------------------------------


def check_space(out: Path, cfg: dict):
    import shutil
    per_expert = build_manifest(cfg)["expert_bin_bytes"]
    n_moe = cfg["num_hidden_layers"] - cfg["first_k_dense_replace"]
    need = per_expert * cfg["num_experts"] * n_moe
    free = shutil.disk_usage(out).free
    print(f"[preflight] experts ~{need / 1e9:.1f} GB, free {free / 1e9:.1f} GB", flush=True)
    if need > free:
        raise SystemExit(
            f"[export_kimi_k3] not enough space: need ~{need / 1e9:.1f} GB for routed "
            f"experts alone, {free / 1e9:.1f} GB free")


# --------------------------------------------------------------------------
# Tiny deterministic export (for the correctness harness)
# --------------------------------------------------------------------------


def export_tiny(out: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kimi_k3_ref.tiny import tiny_cfg, tiny_weights

    cfg = tiny_cfg()
    w = tiny_weights(cfg)
    cfg = dict(cfg)
    cfg["kda_layers"] = sorted(cfg["kda_layers"])
    cfg.setdefault("eos_token_ids", [0])
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(cfg, out)

    save_file({"embed": w["embed"].contiguous()}, str(out / "embed.safetensors"))
    save_file({
        "norm": w["norm"].contiguous(),
        "lm_head": w["lm_head"].contiguous(),
        "output_attn_res_proj": w["output_attn_res_proj"].contiguous(),
        "output_attn_res_norm": w["output_attn_res_norm"].contiguous(),
    }, str(out / "head.safetensors"))

    for i, lw in enumerate(w["layers"]):
        flat = {}
        for k, v in lw.items():
            if k in ("attn", "moe"):
                continue
            flat[k] = v.contiguous()
        for k, v in lw["attn"].items():
            flat[f"attn.{k}"] = v.contiguous()
        if "moe" in lw:
            moe = lw["moe"]
            for k in ("gate_weight", "e_score_correction_bias",
                      "routed_expert_down_proj", "routed_expert_up_proj",
                      "routed_expert_norm", "shared_w1", "shared_w3", "shared_w2"):
                flat[f"moe.{k}"] = moe[k].contiguous()
            edir = out / "experts" / f"layer_{i:02d}"
            for e, ew in enumerate(moe["experts"]):
                export_expert_bin(ew["w1"].numpy(), ew["w3"].numpy(),
                                  ew["w2"].numpy(), edir / f"expert_{e:03d}.bin")
        (out / "shells").mkdir(parents=True, exist_ok=True)
        save_file(flat, str(out / "shells" / f"layer_{i:02d}.safetensors"))

    print(f"[tiny] wrote {cfg['num_hidden_layers']} layers -> {out}", flush=True)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def selftest():
    rng = np.random.default_rng(0)
    out_d, in_d = 8, 64
    w = rng.standard_normal((out_d, in_d)).astype(np.float32)
    packed, scale = pack_fp4_from_f32(w)
    assert len(packed) == out_d * in_d // 2, "packed size"
    assert len(scale) == out_d * (in_d // FP4_GROUP), "scale size"
    back = unpack_fp4(packed, scale, out_d, in_d)

    # every decoded value must sit exactly on scale * e2m1 grid
    s = np.frombuffer(scale, dtype=np.uint8).reshape(out_d, in_d // FP4_GROUP)
    s = np.exp2(s.astype(np.int32) - 127).repeat(FP4_GROUP, axis=1)
    on_grid = np.isin(np.abs(back / s).round(6), np.round(_E2M1, 6))
    assert on_grid.all(), "decoded values off the e2m1 grid"

    rel = np.abs(back - w).max() / np.abs(w).max()
    print(f"[selftest] fp4 round-trip ok: {out_d}x{in_d}, max rel err {rel:.3f}")
    assert rel < 0.35, f"fp4 round-trip error too large: {rel}"

    # a value already on the grid must survive exactly
    exact = (np.array([[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -6.0] * 8],
                      dtype=np.float32).reshape(1, 64))
    p2, s2 = pack_fp4_from_f32(exact)
    assert np.allclose(unpack_fp4(p2, s2, 1, 64), exact), "on-grid values must be exact"
    print("[selftest] on-grid values round-trip exactly")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=Path, help="validate a config.json against the k3 contract")
    ap.add_argument("--tiny", type=Path, help="write a deterministic tiny model to this dir")
    ap.add_argument("--model", type=Path, help="real checkpoint dir (with --out)")
    ap.add_argument("--out", type=Path, help="output dir for --model")
    ap.add_argument("--selftest", action="store_true", help="fp4 round-trip self-test")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if a.validate:
        cfg = load_and_validate_config(a.validate)
        print(json.dumps(build_manifest(cfg), indent=2))
        return
    if a.tiny:
        export_tiny(a.tiny)
        return
    if a.model:
        if not a.out:
            raise SystemExit("--model requires --out")
        cfg = load_and_validate_config(a.model / "config.json")
        a.out.mkdir(parents=True, exist_ok=True)
        check_space(a.out, cfg)
        write_manifest(cfg, a.out)
        raise SystemExit(
            "[export_kimi_k3] config + manifest OK. The streaming weight pass is "
            "not implemented yet: it needs the real checkpoint's tensor names to "
            "verify against, and this exporter hard-fails rather than guessing "
            "them. Run --validate against the real config.json first.")
    ap.print_help()


if __name__ == "__main__":
    main()
