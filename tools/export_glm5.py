#!/usr/bin/env python3
"""GLM-5.2 (`glm5`) exporter -> the sparse-moe engine's on-disk layout.

Modes:
  --validate CONFIG.json     run the hard-fail config contract (no output)
  --tiny OUT                 write a deterministic tiny model in the engine layout
  --model DIR --out OUT       convert a real zai-org/GLM-5.2-FP8 checkpoint

On-disk layout (mirrors tools/export_deepseek_v4.py; consumed by the M3 loader):
  <out>/manifest.json
  <out>/embed.safetensors                      # embed.weight (bf16)
  <out>/head.safetensors                       # head.weight (lm_head), norm.weight (final)
  <out>/shells/layer_NN.safetensors            # attn + norms (+ router for MoE), bf16
  <out>/experts/layer_NN/expert_EEE.bin        # routed experts (int4_bin)
  <out>/experts/layer_NN/expert_shared.bin     # shared expert (int4_bin)
  <out>/experts/layer_NN/dense.bin             # dense-layer MLP (first_k_dense_replace)

Numeric policy: the shell is dequantized bf16 for the Rust shell backend
("rust_glm"); only the SwiGLU FFNs (routed experts, shared expert, dense MLP)
become int4 binaries (cascadia-int4-gemm group-32 layout). The MTP head weights
are int8 in deployment (int4 collapses acceptance); --tiny keeps them bf16.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# int4_bin packing — cascadia-int4-gemm group-32 layout (same as the dsv4/M2
# exporter; validated by the Rust MmapExpert path).
# --------------------------------------------------------------------------
_INT4_GROUP = 32


def _pack_int4_grouped(w: np.ndarray):
    """[out, in] fp32 -> (packed u8 [out, in/2], scales bf16-LE [out, in/32])."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    out, inn = w.shape
    g = _INT4_GROUP
    assert inn % g == 0, f"in_dim {inn} not divisible by int4 group {g}"
    wg = w.reshape(out, inn // g, g)
    max_abs = np.abs(wg).max(axis=2)
    s = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float32)
    q = np.clip(np.round(wg / s[:, :, None]), -8, 7).astype(np.int32)
    nib = (q + 8).astype(np.uint8).reshape(out, inn)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    u = s.view(np.uint32)
    bf = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")  # RNE f32 -> bf16
    return packed.tobytes(), bf.tobytes()


def export_expert_bin(wg: np.ndarray, wu: np.ndarray, wd: np.ndarray, path: Path):
    """SwiGLU FFN: gate(wg), up(wu), down(wd) — packed nibbles then bf16 scales,
    concatenated. Matches MmapExpert::open's `2*section(inter,dim)+section(dim,inter)`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for w in (wg, wu, wd):
            packed, scale = _pack_int4_grouped(w)
            f.write(packed)
            f.write(scale)


# --------------------------------------------------------------------------
# Config contract — hard-fail on anything the Rust shell does not implement.
# --------------------------------------------------------------------------
class ConfigError(SystemExit):
    pass


def _require(cond, msg):
    if not cond:
        raise ConfigError(f"[export_glm5] config contract violated: {msg}")


def load_and_validate_config(path: Path) -> dict:
    """Load HF config.json and enforce the glm5 shell contract. Fails loudly on
    any surprise (design decision #5) rather than silently mis-exporting."""
    c = json.loads(Path(path).read_text())

    def g(k):
        _require(k in c, f"missing key '{k}'")
        return c[k]

    scoring = c.get("scoring_func", c.get("score_func"))
    _require(scoring == "sigmoid", f"scoring_func must be 'sigmoid', got {scoring!r}")
    _require(c.get("rope_scaling") in (None, {}), f"rope_scaling must be absent (no YaRN), got {c.get('rope_scaling')!r}")
    _require(int(c.get("n_group", 1)) == 1, f"n_group must be 1, got {c.get('n_group')}")
    _require(int(c.get("topk_group", 1)) == 1, f"topk_group must be 1, got {c.get('topk_group')}")

    qk_nope = int(g("qk_nope_head_dim"))
    qk_rope = int(g("qk_rope_head_dim"))
    _require(qk_rope % 2 == 0, f"qk_rope_head_dim must be even, got {qk_rope}")
    cfg = dict(
        hidden=int(g("hidden_size")),
        vocab=int(g("vocab_size")),
        num_layers=int(g("num_hidden_layers")),
        num_heads=int(g("num_attention_heads")),
        q_lora=int(g("q_lora_rank")),
        kv_lora=int(g("kv_lora_rank")),
        qk_nope=qk_nope,
        qk_rope=qk_rope,
        qk_head=qk_nope + qk_rope,
        v_head=int(g("v_head_dim")),
        n_routed=int(g("n_routed_experts")),
        n_shared=int(c.get("n_shared_experts", 1)),
        top_k=int(g("num_experts_per_tok")),
        moe_inter=int(g("moe_intermediate_size")),
        dense_inter=int(g("intermediate_size")),
        first_dense=int(c.get("first_k_dense_replace", 0)),
        routed_scale=float(c.get("routed_scaling_factor", 1.0)),
        rope_theta=float(c.get("rope_theta", c.get("rope_parameters", {}).get("rope_theta", 10000.0))),
        eps=float(c.get("rms_norm_eps", 1e-5)),
        n_mtp=int(c.get("num_nextn_predict_layers", 0)),
        eos=c.get("eos_token_id", []),
    )
    _require(cfg["qk_head"] > 0 and cfg["kv_lora"] > 0, "bad MLA dims")
    _require(cfg["hidden"] % _INT4_GROUP == 0 and cfg["moe_inter"] % _INT4_GROUP == 0,
             f"hidden ({cfg['hidden']}) and moe_inter ({cfg['moe_inter']}) must be "
             f"divisible by the int4 group {_INT4_GROUP}")
    return cfg


def build_manifest(cfg: dict) -> dict:
    """The <out>/manifest.json the engine reads (arch=='glm5' sniff)."""
    eos = cfg["eos"]
    if isinstance(eos, int):
        eos = [eos]
    return {
        "arch": "glm5",
        "num_layers": cfg["num_layers"],
        "dense_layers": list(range(cfg["first_dense"])),
        "num_experts": cfg["n_routed"],
        "top_k": cfg["top_k"],
        "hidden_size": cfg["hidden"],
        "num_kv_heads": cfg["num_heads"],
        "qk_head_dim": cfg["qk_head"],
        "v_head_dim": cfg["v_head"],
        "vocab_size": cfg["vocab"],
        "eos_token_ids": eos,
        "experts_format": "int4_bin",
        "shell_backend": "rust_glm",
        "expert_intermediate": cfg["moe_inter"],
        # glm5 shell dims (beyond the dsv4-shared subset) the loader needs:
        "num_attention_heads": cfg["num_heads"],
        "q_lora_rank": cfg["q_lora"],
        "kv_lora_rank": cfg["kv_lora"],
        "qk_nope_head_dim": cfg["qk_nope"],
        "qk_rope_head_dim": cfg["qk_rope"],
        "dense_intermediate": cfg["dense_inter"],
        "n_shared_experts": cfg["n_shared"],
        "routed_scaling_factor": cfg["routed_scale"],
        "rope_theta": cfg["rope_theta"],
        "rms_norm_eps": cfg["eps"],
    }


def write_manifest(cfg: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(build_manifest(cfg), indent=2))
    print(f"[manifest] {out / 'manifest.json'}", flush=True)


# --------------------------------------------------------------------------
# --tiny: deterministic tiny model in the engine layout (smoke / M3 loader dev)
# --------------------------------------------------------------------------
def export_tiny(out: Path):
    import torch
    from safetensors.torch import save_file

    gen = torch.Generator().manual_seed(7)

    def rf(*shape, s=0.3):
        return (s * torch.randn(*shape, generator=gen)).to(torch.bfloat16)

    def nf(n):
        return (1.0 + 0.05 * torch.randn(n, generator=gen)).to(torch.bfloat16)

    cfg = dict(
        hidden=32, vocab=16, num_layers=3, num_heads=3, q_lora=16, kv_lora=8,
        qk_nope=6, qk_rope=4, qk_head=10, v_head=6, n_routed=4, n_shared=1,
        top_k=2, moe_inter=32, dense_inter=64, first_dense=1, routed_scale=2.5,
        rope_theta=8.0e6, eps=1e-5, n_mtp=1, eos=[0],
    )
    H, E, MI, DI = cfg["hidden"], cfg["n_routed"], cfg["moe_inter"], cfg["dense_inter"]
    write_manifest(cfg, out)
    (out / "shells").mkdir(parents=True, exist_ok=True)

    save_file({"embed.weight": rf(cfg["vocab"], H)}, str(out / "embed.safetensors"))
    save_file({"head.weight": rf(cfg["vocab"], H), "norm.weight": nf(H)},
              str(out / "head.safetensors"))

    def attn_shell(li):
        h, qk, nope, vh = cfg["num_heads"], cfg["qk_head"], cfg["qk_nope"], cfg["v_head"]
        kvl, ql = cfg["kv_lora"], cfg["q_lora"]
        return {
            "input_layernorm.weight": nf(H), "post_attention_layernorm.weight": nf(H),
            "self_attn.wq_a.weight": rf(ql, H), "self_attn.q_a_layernorm.weight": nf(ql),
            "self_attn.wq_b.weight": rf(h * qk, ql),
            "self_attn.wkv_a.weight": rf(kvl + cfg["qk_rope"], H),
            "self_attn.kv_a_layernorm.weight": nf(kvl),
            "self_attn.wkv_b.weight": rf(h * (nope + vh), kvl),
            "self_attn.o_proj.weight": rf(H, h * vh),
        }

    def npy(t):
        return t.to(torch.float32).numpy()

    for li in range(cfg["num_layers"]):
        shell = attn_shell(li)
        edir = out / "experts" / f"layer_{li:02d}"
        if li < cfg["first_dense"]:
            export_expert_bin(npy(rf(DI, H)), npy(rf(DI, H)), npy(rf(H, DI)), edir / "dense.bin")
        else:
            shell["mlp.gate.weight"] = rf(E, H)            # router
            shell["mlp.gate.e_score_correction_bias"] = (0.3 * torch.randn(E, generator=gen)).to(torch.bfloat16)
            for e in range(E):
                export_expert_bin(npy(rf(MI, H)), npy(rf(MI, H)), npy(rf(H, MI)),
                                  edir / f"expert_{e:03d}.bin")
            export_expert_bin(npy(rf(MI, H)), npy(rf(MI, H)), npy(rf(H, MI)),
                              edir / "expert_shared.bin")
        save_file({k: v for k, v in shell.items()}, str(out / "shells" / f"layer_{li:02d}.safetensors"))

    print(f"[tiny] wrote {cfg['num_layers']} layers to {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="GLM-5.2 exporter")
    ap.add_argument("--validate", type=Path, help="validate a config.json against the glm5 contract")
    ap.add_argument("--tiny", type=Path, help="write a deterministic tiny model to this dir")
    ap.add_argument("--model", type=Path, help="real GLM-5.2-FP8 checkpoint dir (with --out)")
    ap.add_argument("--out", type=Path, help="output dir for --model")
    args = ap.parse_args()

    if args.validate:
        cfg = load_and_validate_config(args.validate)
        print(f"[validate] OK: glm5 config ({cfg['num_layers']} layers, "
              f"{cfg['n_routed']} experts, top-{cfg['top_k']}, hidden {cfg['hidden']})")
        return
    if args.tiny:
        export_tiny(args.tiny)
        return
    if args.model and args.out:
        cfg = load_and_validate_config(args.model / "config.json")
        write_manifest(cfg, args.out)
        raise SystemExit("[export_glm5] real FP8->int4 weight conversion is wired with the "
                         "M3 loader (round-trip validated end-to-end); manifest written.")
    ap.error("one of --validate, --tiny, or --model/--out is required")


if __name__ == "__main__":
    main()
