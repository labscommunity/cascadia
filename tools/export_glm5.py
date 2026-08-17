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
import os
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
# int3_bin packing — same group-32 symmetric layout as int4, but 3 bits/weight
# (values -4..3, LSB-first bitstream: 32 values -> 12 packed bytes) + bf16
# scales. ~22% fewer bytes than int4 (14 vs 18 bytes / 32 weights). For the cold
# expert tail (A2); dequant speed is irrelevant in the IO-bound regime.
# --------------------------------------------------------------------------
_INT3_GROUP = 32


def _pack_int3_grouped(w: np.ndarray):
    """[out, in] fp32 -> (packed u8 [out, in*3/8], scales bf16-LE [out, in/32]).
    Symmetric int3: s = max_abs/3, q = clip(round(w/s), -4, 3), stored q+4 in
    [0,7], LSB-first bitstream (32 vals -> 96 bits -> 12 bytes per group)."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    out, inn = w.shape
    g = _INT3_GROUP
    assert inn % g == 0, f"in_dim {inn} not divisible by int3 group {g}"
    wg = w.reshape(out, inn // g, g)
    max_abs = np.abs(wg).max(axis=2)
    s = np.where(max_abs > 0, max_abs / 3.0, 1.0).astype(np.float32)
    q = np.clip(np.round(wg / s[:, :, None]), -4, 3).astype(np.int32)
    u3 = (q + 4).astype(np.uint8)  # [out, ng, 32] in [0,7]
    # expand to LSB-first bits then packbits (little) -> 12 bytes/group.
    bits = ((u3[:, :, :, None] >> np.arange(3, dtype=np.uint8)) & 1).astype(np.uint8)
    bits = bits.reshape(out, inn // g, g * 3)  # [out, ng, 96]
    packed = np.packbits(bits, axis=-1, bitorder="little")  # [out, ng, 12]
    packed = packed.reshape(out, -1)  # [out, ng*12] = [out, in*3/8]
    u = s.view(np.uint32)
    bf = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")  # RNE f32 -> bf16
    return packed.tobytes(), bf.tobytes()


def _unpack_int3_grouped(packed: bytes, scale: bytes, out: int, inn: int) -> np.ndarray:
    """Reference inverse of `_pack_int3_grouped` (for the self-test + parity)."""
    g = _INT3_GROUP
    ng = inn // g
    p = np.frombuffer(packed, dtype=np.uint8).reshape(out, ng, 12)
    bits = np.unpackbits(p, axis=-1, bitorder="little").reshape(out, ng, g, 3)
    u3 = (bits * (1 << np.arange(3, dtype=np.uint8))).sum(axis=-1).astype(np.int32)
    q = u3 - 4
    s = np.frombuffer(scale, dtype="<u2").astype(np.uint32).reshape(out, ng)
    s = (s << 16).view(np.float32)
    return (q * s[:, :, None]).reshape(out, inn)


def export_expert_bin_int3(wg: np.ndarray, wu: np.ndarray, wd: np.ndarray, path: Path):
    """int3 counterpart of export_expert_bin (gate, up, down)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for w in (wg, wu, wd):
            packed, scale = _pack_int3_grouped(w)
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
        # DSA lightning indexer (GLM-5.2 / deepseek_v32). Optional here so a
        # non-DSA config still exports (dense); when present the weights are
        # emitted and a missing tensor fails loudly in export_real.
        index_topk=int(c.get("index_topk", 0)),
        index_n_heads=int(c.get("index_n_heads", 0)),
        index_head_dim=int(c.get("index_head_dim", 0)),
        # IndexShare topology: per-layer "full" (owns an indexer) | "shared"
        # (reuses the previous full layer's top-k). Absent -> every attention
        # layer is its own indexer (the tiny/dev default).
        indexer_types=list(c.get("indexer_types", [])),
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
        # DSA lightning indexer (0 = not exported -> dense causal attention).
        "index_topk": cfg.get("index_topk", 0),
        "index_n_heads": cfg.get("index_n_heads", 0),
        "index_head_dim": cfg.get("index_head_dim", 0),
        # Per-layer IndexShare topology. Default (absent): every layer is "full"
        # when an indexer is present. "shared" layers reuse the previous full
        # layer's top-k selection.
        "indexer_types": (cfg.get("indexer_types")
                          or (["full"] * cfg["num_layers"] if cfg.get("index_n_heads", 0) > 0 else [])),
        # MTP draft head (layer `num_layers`, DeepSeek-V3 nextn) for speculative
        # decode. True only when the head's weights were actually emitted to
        # <dir>/mtp.safetensors (block experts are bf16). export_real does not
        # emit it yet, so it stays false there.
        "has_mtp": bool(cfg.get("mtp_emitted", False)),
    }


def write_manifest(cfg: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(build_manifest(cfg), indent=2))
    print(f"[manifest] {out / 'manifest.json'}", flush=True)


# --------------------------------------------------------------------------
# --tiny: deterministic tiny model in the engine layout (smoke / M3 loader dev)
# --------------------------------------------------------------------------

# Default DSA indexer for the `--tiny` CLI path: small enough that ~10
# generated tokens cross the index_topk boundary cheaply, so a loader/parity
# test can exercise the sparse selection without a real-sized context. Kept as
# a module constant (rather than inlined at the call site) so the Rust test's
# fixture generator can build the byte-identical config instead of drifting.
TINY_INDEXER_NUM_LAYERS = 3
TINY_INDEXER_KW = dict(
    num_layers=TINY_INDEXER_NUM_LAYERS,
    index_n_heads=2,
    index_head_dim=16,
    index_topk=8,
    indexer_types=["full" if i % 2 == 0 else "shared" for i in range(TINY_INDEXER_NUM_LAYERS)],
)


def export_tiny(out: Path, num_layers: int = 3, first_dense: int = 1,
                index_n_heads: int = 0, index_head_dim: int = 8, index_topk: int = 0,
                indexer_types: "list | None" = None):
    import torch
    from safetensors.torch import save_file

    gen = torch.Generator().manual_seed(7)
    # Indexer weights are drawn from an INDEPENDENT stream so enabling DSA leaves
    # every other weight — and therefore the dense greedy tokens — untouched.
    geni = torch.Generator().manual_seed(11)

    def rf(*shape, s=0.3):
        return (s * torch.randn(*shape, generator=gen)).to(torch.bfloat16)

    def nf(n):
        return (1.0 + 0.05 * torch.randn(n, generator=gen)).to(torch.bfloat16)

    def rfi(*shape, s=0.3):
        return (s * torch.randn(*shape, generator=geni)).to(torch.bfloat16)

    # num_layers/first_dense are parametric so multi-rank pipeline fixtures (≥4
    # layers, for a middle-relay split) can be generated; the RNG stream is
    # sequential per layer, so the default (3, 1) reproduces the base fixture and
    # an N-layer export's first 3 layers match it byte-for-byte.
    cfg = dict(
        hidden=32, vocab=16, num_layers=num_layers, num_heads=3, q_lora=16, kv_lora=8,
        qk_nope=6, qk_rope=4, qk_head=10, v_head=6, n_routed=4, n_shared=1,
        top_k=2, moe_inter=32, dense_inter=64, first_dense=first_dense, routed_scale=2.5,
        rope_theta=8.0e6, eps=1e-5, n_mtp=1, eos=[0],
        index_n_heads=index_n_heads, index_head_dim=index_head_dim, index_topk=index_topk,
        indexer_types=(indexer_types
                       or (["full"] * num_layers if index_n_heads > 0 else [])),
    )
    H, E, MI, DI = cfg["hidden"], cfg["n_routed"], cfg["moe_inter"], cfg["dense_inter"]
    cfg["mtp_emitted"] = cfg["n_mtp"] > 0  # tiny always emits the MTP head below
    write_manifest(cfg, out)
    (out / "shells").mkdir(parents=True, exist_ok=True)

    save_file({"embed.weight": rf(cfg["vocab"], H)}, str(out / "embed.safetensors"))
    save_file({"head.weight": rf(cfg["vocab"], H), "norm.weight": nf(H)},
              str(out / "head.safetensors"))

    def attn_shell(li):
        h, qk, nope, vh = cfg["num_heads"], cfg["qk_head"], cfg["qk_nope"], cfg["v_head"]
        kvl, ql = cfg["kv_lora"], cfg["q_lora"]
        shell = {
            "input_layernorm.weight": nf(H), "post_attention_layernorm.weight": nf(H),
            "self_attn.wq_a.weight": rf(ql, H), "self_attn.q_a_layernorm.weight": nf(ql),
            "self_attn.wq_b.weight": rf(h * qk, ql),
            "self_attn.wkv_a.weight": rf(kvl + cfg["qk_rope"], H),
            "self_attn.kv_a_layernorm.weight": nf(kvl),
            "self_attn.wkv_b.weight": rf(h * (nope + vh), kvl),
            "self_attn.o_proj.weight": rf(H, h * vh),
        }
        # IndexShare: emit indexer weights only for "full" layers.
        if index_n_heads > 0 and cfg["indexer_types"][li] == "full":
            inh, ihd = index_n_heads, index_head_dim
            shell.update({
                "self_attn.indexer.wq_b.weight": rfi(inh * ihd, ql),
                "self_attn.indexer.wk.weight": rfi(ihd, H),
                "self_attn.indexer.weights_proj.weight": rfi(inh, H),
                "self_attn.indexer.k_norm.weight": (1.0 + 0.05 * torch.randn(ihd, generator=geni)).to(torch.bfloat16),
                "self_attn.indexer.k_norm.bias": (0.1 * torch.randn(ihd, generator=geni)).to(torch.bfloat16),
            })
        return shell

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

    # MTP draft head (DeepSeek-V3 nextn). Emitted LAST so the per-layer RNG
    # stream above is unchanged (pipeline fixtures stay byte-reproducible). The
    # block is a normal attn+MoE layer with NO indexer (draft window g ≪
    # index_topk ⇒ full causal anyway); its experts are stored bf16 (the shell's
    # native dtype), so tiny and the real bf16 head share one loader path.
    if cfg["n_mtp"] > 0:
        h, qk, nope, vh = cfg["num_heads"], cfg["qk_head"], cfg["qk_nope"], cfg["v_head"]
        kvl, ql = cfg["kv_lora"], cfg["q_lora"]
        mtp = {
            "mtp.enorm.weight": nf(H),
            "mtp.hnorm.weight": nf(H),
            "mtp.norm.weight": nf(H),          # shared_head.norm (== mtp_norm)
            "mtp.eh_proj.weight": rf(H, 2 * H),
            "mtp.block.input_layernorm.weight": nf(H),
            "mtp.block.post_attention_layernorm.weight": nf(H),
            "mtp.block.self_attn.wq_a.weight": rf(ql, H),
            "mtp.block.self_attn.q_a_layernorm.weight": nf(ql),
            "mtp.block.self_attn.wq_b.weight": rf(h * qk, ql),
            "mtp.block.self_attn.wkv_a.weight": rf(kvl + cfg["qk_rope"], H),
            "mtp.block.self_attn.kv_a_layernorm.weight": nf(kvl),
            "mtp.block.self_attn.wkv_b.weight": rf(h * (nope + vh), kvl),
            "mtp.block.self_attn.o_proj.weight": rf(H, h * vh),
            "mtp.block.mlp.gate.weight": rf(E, H),
            "mtp.block.mlp.gate.e_score_correction_bias": (0.3 * torch.randn(E, generator=gen)).to(torch.bfloat16),
        }
        for e in range(E):
            mtp[f"mtp.block.mlp.experts.{e}.gate.weight"] = rf(MI, H)
            mtp[f"mtp.block.mlp.experts.{e}.up.weight"] = rf(MI, H)
            mtp[f"mtp.block.mlp.experts.{e}.down.weight"] = rf(H, MI)
        mtp["mtp.block.mlp.shared.gate.weight"] = rf(MI, H)
        mtp["mtp.block.mlp.shared.up.weight"] = rf(MI, H)
        mtp["mtp.block.mlp.shared.down.weight"] = rf(H, MI)
        save_file(mtp, str(out / "mtp.safetensors"))

    print(f"[tiny] wrote {cfg['num_layers']} layers to {out}", flush=True)


# --------------------------------------------------------------------------
# Real checkpoint: stream + dequant a local zai-org/GLM-5.2-FP8 dir.
# --------------------------------------------------------------------------
class CkptSource:
    """Streamed weight getter over a GLM-5.2-FP8 checkpoint. Dequantizes FP8
    e4m3 (block 128x128 `.weight_scale_inv`) on read; norms / router / biases
    are stored full-precision (no scale) and pass through. Holds at most one
    shard handle per file."""

    def __init__(self, model_dir: Path):
        import torch  # noqa: F401
        from safetensors import safe_open

        self._safe_open = safe_open
        self.dir = Path(model_dir)
        idx = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.wm = idx["weight_map"]
        self._open = {}

    def has(self, name: str) -> bool:
        return name in self.wm

    def _raw(self, name: str):
        shard = self.wm[name]
        h = self._open.get(shard)
        if h is None:
            h = self._safe_open(str(self.dir / shard), framework="pt")
            self._open[shard] = h
        return h.get_tensor(name)

    def get(self, name: str):
        """Return the weight as an fp32 torch tensor (FP8-dequantized if a
        block scale exists)."""
        import torch

        t = self._raw(name)
        sk = name.rsplit(".", 1)[0] + ".weight_scale_inv"
        if name.endswith(".weight") and self.has(sk):
            w = t.to(torch.float32)
            s = self._raw(sk).to(torch.float32)              # [ceil(O/128), ceil(I/128)]
            o, i = w.shape
            s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:o, :i]
            # non-128-divisible dims (e.g. kv_a_proj_with_mqa [576,6144]) MUST
            # clip cleanly; a wrong-axis expand would corrupt silently.
            assert s.shape == w.shape, f"fp8 scale expand mismatch for {name}: {tuple(s.shape)} vs {tuple(w.shape)}"
            return w * s                                     # dequant = fp8 * scale_inv
        return t.to(torch.float32)


def _int4_bin_bytes(hidden: int, inter: int) -> int:
    """Byte size of one int4 expert/dense bin (gate+up+down, group-32)."""
    def sec(o, i):
        return o * i // 2 + o * (i // _INT4_GROUP) * 2
    return 2 * sec(inter, hidden) + sec(hidden, inter)


def check_space(out: Path, cfg: dict):
    """Pre-flight: estimate the export size and refuse to start if the disk
    can't hold it (a 439 GB export that runs out of space at layer 60 is an
    expensive failure). Override with GLM5_SKIP_SPACE_CHECK=1."""
    import shutil

    h = cfg["hidden"]
    n_dense = cfg["first_dense"]
    n_moe = cfg["num_layers"] - n_dense
    experts = n_moe * (cfg["n_routed"] + cfg["n_shared"]) * _int4_bin_bytes(h, cfg["moe_inter"])
    dense = n_dense * _int4_bin_bytes(h, cfg["dense_inter"])
    attn_params = (
        cfg["q_lora"] * h + cfg["num_heads"] * cfg["qk_head"] * cfg["q_lora"]
        + (cfg["kv_lora"] + cfg["qk_rope"]) * h
        + cfg["num_heads"] * (cfg["qk_nope"] + cfg["v_head"]) * cfg["kv_lora"]
        + h * cfg["num_heads"] * cfg["v_head"]
    )
    shell = cfg["num_layers"] * attn_params * 2 + 2 * cfg["vocab"] * h * 2  # bf16
    est = experts + dense + shell
    free = shutil.disk_usage(out).free
    print(f"[check_space] estimated output ~{est / 1e9:.1f} GB, free ~{free / 1e9:.1f} GB", flush=True)
    if free < est * 1.05 and os.environ.get("GLM5_SKIP_SPACE_CHECK") != "1":
        raise SystemExit(
            f"[export_glm5] insufficient disk: need ~{est / 1e9:.1f} GB (+5% margin), "
            f"have ~{free / 1e9:.1f} GB. Set GLM5_SKIP_SPACE_CHECK=1 to override."
        )


def export_real(model_dir: Path, out: Path):
    """Convert a local GLM-5.2-FP8 checkpoint to the engine layout: bf16 shell
    safetensors + int4 expert/dense bins + manifest. Maps HF names to the shell
    names the loader reads (q_a_proj->wq_a, kv_a_proj_with_mqa->wkv_a, ...).

    Resumable: writes a `.layer_NN.done` marker after each layer and skips
    completed layers on re-run, so a crash/OOM mid-export restarts cheaply."""
    import torch
    from safetensors.torch import save_file

    cfg = load_and_validate_config(model_dir / "config.json")
    src = CkptSource(model_dir)
    cfg["mtp_emitted"] = cfg["n_mtp"] > 0  # MTP draft head emitted below (bf16 block)
    (out / "shells").mkdir(parents=True, exist_ok=True)
    check_space(out, cfg)

    def bf16(name):
        return src.get(name).to(torch.bfloat16)

    def npy(name):
        return src.get(name).to(torch.float32).numpy()

    head_done = out / ".head.done"
    if head_done.exists():
        print("[head] skip (done)", flush=True)
    else:
        save_file({"embed.weight": bf16("model.embed_tokens.weight")}, str(out / "embed.safetensors"))
        save_file(
            {"head.weight": bf16("lm_head.weight"), "norm.weight": bf16("model.norm.weight")},
            str(out / "head.safetensors"),
        )
        head_done.touch()

    for li in range(cfg["num_layers"]):
        marker = out / f".layer_{li:02d}.done"
        if marker.exists():
            print(f"[layer {li:2d}/{cfg['num_layers']}] skip (done)", flush=True)
            continue
        p = f"model.layers.{li}."
        shell = {
            "input_layernorm.weight": bf16(p + "input_layernorm.weight"),
            "post_attention_layernorm.weight": bf16(p + "post_attention_layernorm.weight"),
            "self_attn.wq_a.weight": bf16(p + "self_attn.q_a_proj.weight"),
            "self_attn.q_a_layernorm.weight": bf16(p + "self_attn.q_a_layernorm.weight"),
            "self_attn.wq_b.weight": bf16(p + "self_attn.q_b_proj.weight"),
            "self_attn.wkv_a.weight": bf16(p + "self_attn.kv_a_proj_with_mqa.weight"),
            "self_attn.kv_a_layernorm.weight": bf16(p + "self_attn.kv_a_layernorm.weight"),
            "self_attn.wkv_b.weight": bf16(p + "self_attn.kv_b_proj.weight"),
            "self_attn.o_proj.weight": bf16(p + "self_attn.o_proj.weight"),
        }
        # DSA lightning indexer weights (bf16 shell). IndexShare: only "full"
        # layers own an indexer; "shared" layers reuse the previous full layer's
        # top-k (no tensors). Real GLM names: self_attn.indexer.{wq_b,wk,
        # weights_proj,k_norm}. Fail loudly if a "full" layer's tensor is absent.
        itypes = cfg.get("indexer_types") or []
        is_full = (cfg["index_n_heads"] > 0
                   and (li >= len(itypes) or itypes[li] == "full"))
        if is_full and cfg["index_n_heads"] > 0:
            ip = p + "self_attn.indexer."
            for name in ("wq_b.weight", "wk.weight", "weights_proj.weight",
                         "k_norm.weight", "k_norm.bias"):
                _require(src.has(ip + name),
                         f"'full' indexer layer {li} missing tensor '{ip + name}' "
                         f"— check the checkpoint's DSA indexer parameter names")
                shell["self_attn.indexer." + name] = bf16(ip + name)
        edir = out / "experts" / f"layer_{li:02d}"
        if li < cfg["first_dense"]:
            export_expert_bin(
                npy(p + "mlp.gate_proj.weight"), npy(p + "mlp.up_proj.weight"),
                npy(p + "mlp.down_proj.weight"), edir / "dense.bin",
            )
        else:
            shell["mlp.gate.weight"] = bf16(p + "mlp.gate.weight")
            shell["mlp.gate.e_score_correction_bias"] = \
                src.get(p + "mlp.gate.e_score_correction_bias").to(torch.bfloat16)
            for e in range(cfg["n_routed"]):
                ep = p + f"mlp.experts.{e}."
                export_expert_bin(
                    npy(ep + "gate_proj.weight"), npy(ep + "up_proj.weight"),
                    npy(ep + "down_proj.weight"), edir / f"expert_{e:03d}.bin",
                )
            sp = p + "mlp.shared_experts."
            export_expert_bin(
                npy(sp + "gate_proj.weight"), npy(sp + "up_proj.weight"),
                npy(sp + "down_proj.weight"), edir / "expert_shared.bin",
            )
        save_file(shell, str(out / "shells" / f"layer_{li:02d}.safetensors"))
        marker.touch()  # layer fully written -> resumable checkpoint
        print(f"[layer {li:2d}/{cfg['num_layers']}] shell + experts written", flush=True)

    # MTP draft head (DeepSeek-V3 nextn, layer `num_layers` == 78) for native
    # speculative decode. The block is a full attn + MoE layer; its experts are
    # dequantized to bf16 (the bf16-first MTP path — int4 collapses acceptance),
    # stored directly in mtp.safetensors so tiny and real share one loader path.
    # The block's DSA indexer is intentionally NOT exported: the draft window
    # (g ≪ index_topk) always attends the full causal range, so the loader runs
    # the block dense. Resumable via .mtp.done.
    mtp_done = out / ".mtp.done"
    if cfg["n_mtp"] > 0 and not mtp_done.exists():
        mp = f"model.layers.{cfg['num_layers']}."
        mtp = {
            "mtp.enorm.weight": bf16(mp + "enorm.weight"),
            "mtp.hnorm.weight": bf16(mp + "hnorm.weight"),
            "mtp.norm.weight": bf16(mp + "shared_head.norm.weight"),
            "mtp.eh_proj.weight": bf16(mp + "eh_proj.weight"),
            "mtp.block.input_layernorm.weight": bf16(mp + "input_layernorm.weight"),
            "mtp.block.post_attention_layernorm.weight": bf16(mp + "post_attention_layernorm.weight"),
            "mtp.block.self_attn.wq_a.weight": bf16(mp + "self_attn.q_a_proj.weight"),
            "mtp.block.self_attn.q_a_layernorm.weight": bf16(mp + "self_attn.q_a_layernorm.weight"),
            "mtp.block.self_attn.wq_b.weight": bf16(mp + "self_attn.q_b_proj.weight"),
            "mtp.block.self_attn.wkv_a.weight": bf16(mp + "self_attn.kv_a_proj_with_mqa.weight"),
            "mtp.block.self_attn.kv_a_layernorm.weight": bf16(mp + "self_attn.kv_a_layernorm.weight"),
            "mtp.block.self_attn.wkv_b.weight": bf16(mp + "self_attn.kv_b_proj.weight"),
            "mtp.block.self_attn.o_proj.weight": bf16(mp + "self_attn.o_proj.weight"),
            "mtp.block.mlp.gate.weight": bf16(mp + "mlp.gate.weight"),
            "mtp.block.mlp.gate.e_score_correction_bias":
                src.get(mp + "mlp.gate.e_score_correction_bias").to(torch.bfloat16),
        }
        for e in range(cfg["n_routed"]):
            ep = mp + f"mlp.experts.{e}."
            mtp[f"mtp.block.mlp.experts.{e}.gate.weight"] = bf16(ep + "gate_proj.weight")
            mtp[f"mtp.block.mlp.experts.{e}.up.weight"] = bf16(ep + "up_proj.weight")
            mtp[f"mtp.block.mlp.experts.{e}.down.weight"] = bf16(ep + "down_proj.weight")
        sp = mp + "mlp.shared_experts."
        mtp["mtp.block.mlp.shared.gate.weight"] = bf16(sp + "gate_proj.weight")
        mtp["mtp.block.mlp.shared.up.weight"] = bf16(sp + "up_proj.weight")
        mtp["mtp.block.mlp.shared.down.weight"] = bf16(sp + "down_proj.weight")
        save_file(mtp, str(out / "mtp.safetensors"))
        mtp_done.touch()
        print(f"[mtp] draft head written ({cfg['n_routed']}+1 bf16 experts)", flush=True)
    elif mtp_done.exists():
        print("[mtp] skip (done)", flush=True)

    # Written only now that every tensor it describes is on disk: a crash
    # earlier (dense loop, MTP block) must not leave a manifest claiming
    # weights (e.g. has_mtp) that were never exported.
    write_manifest(cfg, out)

    # Carry the serving sidecars from the source checkpoint. GLM-5.2 ships its
    # chat template as a standalone `chat_template.jinja`; without it the API
    # falls back to legacy `role: content` formatting and instruct/chat prompts
    # degrade (cascadia-api::load_chat_template_config_at reads the model root).
    # `.exists()` guards each — base checkpoints simply won't have some of them.
    import shutil

    for fn in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
    ):
        s = model_dir / fn
        if s.exists():
            shutil.copy(s, out / fn)
            print(f"[sidecar] {fn}", flush=True)

    print(f"[export] done -> {out}", flush=True)


def _selftest_fp8():
    """Validate the FP8 block-128 dequant math against a manual reference,
    with no checkpoint (torch float8_e4m3fn)."""
    import torch

    torch.manual_seed(0)
    o, i = 256, 384  # 2 x 3 blocks of 128
    w = (0.1 * torch.randn(o, i)).to(torch.float8_e4m3fn)
    s = (0.5 + torch.rand((o + 127) // 128, (i + 127) // 128))  # block scales
    # reference: expand + multiply
    sf = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:o, :i]
    ref = w.to(torch.float32) * sf
    # exercise the same expansion the CkptSource uses
    got = w.to(torch.float32) * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:o, :i]
    err = (got - ref).abs().max().item()
    assert err == 0.0, f"fp8 dequant selftest diverged: {err}"
    print("[selftest] fp8 block-128 dequant OK")


def main():
    ap = argparse.ArgumentParser(description="GLM-5.2 exporter")
    ap.add_argument("--validate", type=Path, help="validate a config.json against the glm5 contract")
    ap.add_argument("--tiny", type=Path, help="write a deterministic tiny model to this dir")
    ap.add_argument("--model", type=Path, help="real GLM-5.2-FP8 checkpoint dir (with --out)")
    ap.add_argument("--out", type=Path, help="output dir for --model")
    ap.add_argument("--selftest", action="store_true", help="run the FP8 dequant self-test")
    args = ap.parse_args()

    if args.selftest:
        _selftest_fp8()
        return
    if args.validate:
        cfg = load_and_validate_config(args.validate)
        print(f"[validate] OK: glm5 config ({cfg['num_layers']} layers, "
              f"{cfg['n_routed']} experts, top-{cfg['top_k']}, hidden {cfg['hidden']})")
        return
    if args.tiny:
        export_tiny(args.tiny, **TINY_INDEXER_KW)
        return
    if args.model and args.out:
        export_real(args.model, args.out)
        return
    ap.error("one of --validate, --tiny, --model/--out, or --selftest is required")


if __name__ == "__main__":
    main()
