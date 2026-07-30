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
  <out>/experts/layer_NN.bin            all experts of a layer, concatenated
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


# One packed file per LAYER, experts concatenated at a fixed stride.
#
# Per-expert files would mean 896 x 92 = 82,432 of them, and the runtime maps
# each expert set -- Linux's default vm.max_map_count is 65,530, so per-expert
# mappings fail outright at this scale. Per layer it is 92 mappings, and
# residency still works: mlock/madvise take sub-ranges of a mapping.


def append_expert_bin(w1: np.ndarray, w3: np.ndarray, w2: np.ndarray, f):
    """SiTU FFN: gate(w1), up(w3), down(w2) — nibbles then E8M0 scales each."""
    for w in (w1, w3, w2):
        packed, scale = pack_fp4_from_f32(w)
        f.write(packed)
        f.write(scale)


def append_repacked_expert(sections, f):
    """Real path: write already-mxfp4 (packed, scale) pairs straight through."""
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
    # `linear_attn_config` lists layers 1-INDEXED: kda_layers starts at 1 and
    # full_attn_layers runs to num_hidden_layers. Subtracting 1 yields a clean
    # partition of 0..n-1. Verified against the checkpoint's tensor index --
    # layer 0 carries KDA tensors (A_log) and layer 3 carries MLA ones.
    kda = set(int(i) - 1 for i in la.get("kda_layers", []))
    full = set(int(i) - 1 for i in la.get("full_attn_layers", []))
    _require(kda, "linear_attn_config.kda_layers is empty")
    _require(not (kda & full), "kda_layers and full_attn_layers overlap")
    _require(
        kda | full == set(range(n)),
        f"kda_layers + full_attn_layers must partition 0..{n - 1} after the "
        f"1-indexed shift (got {len(kda)} + {len(full)} covering "
        f"{len(kda | full)} of {n})",
    )

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


def export_tiny(out: Path, expert_roots: list | None = None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kimi_k3_ref.tiny import tiny_cfg, tiny_weights

    cfg = tiny_cfg()
    w = tiny_weights(cfg)
    cfg = dict(cfg)
    cfg["kda_layers"] = sorted(cfg["kda_layers"])
    cfg.setdefault("eos_token_ids", [0])
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(cfg, out)
    roots = plan_expert_roots(out, expert_roots, cfg) if expert_roots else None

    # Store the big matrices bf16, matching the real export, so the loader's
    # BF16 decode path is covered by the tiny fixtures. tiny_weights are already
    # bf16-rounded, so nothing is lost.
    def bf(t):
        return t.to(torch.bfloat16).contiguous()

    save_file({"embed": bf(w["embed"])}, str(out / "embed.safetensors"))
    save_file({
        "norm": w["norm"].contiguous(),
        "lm_head": bf(w["lm_head"]),
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
            # norms / A_log / dt_bias stay f32, as they do in the checkpoint
            flat[f"attn.{k}"] = v.contiguous() if v.ndim <= 1 else bf(v)
        if "moe" in lw:
            moe = lw["moe"]
            for k in ("gate_weight", "e_score_correction_bias",
                      "routed_expert_down_proj", "routed_expert_up_proj",
                      "routed_expert_norm", "shared_w1", "shared_w3", "shared_w2"):
                v = moe[k]
                flat[f"moe.{k}"] = v.contiguous() if v.ndim <= 1 else bf(v)
            edir = out / "experts"
            edir.mkdir(parents=True, exist_ok=True)
            wdir = roots[i] if roots else edir
            wdir.mkdir(parents=True, exist_ok=True)
            final = wdir / f"layer_{i:02d}.bin"
            with open(final, "wb") as ef:
                for ew in moe["experts"]:
                    append_expert_bin(ew["w1"].numpy(), ew["w3"].numpy(),
                                      ew["w2"].numpy(), ef)
            if roots:
                link = edir / f"layer_{i:02d}.bin"
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(final)
        (out / "shells").mkdir(parents=True, exist_ok=True)
        save_file(flat, str(out / "shells" / f"layer_{i:02d}.safetensors"))

    print(f"[tiny] wrote {cfg['num_hidden_layers']} layers -> {out}", flush=True)



# --------------------------------------------------------------------------
# Real checkpoint export (streaming, resumable)
# --------------------------------------------------------------------------

# Text model lives under `language_model.`; `vision_tower.` / `mm_projector.`
# are the ViT and are dropped. Verified against the checkpoint's
# model.safetensors.index.json (497,220 tensors, 1.56 TB).
PREFIX = "language_model.model."
LM_HEAD = "language_model.lm_head.weight"

# per-layer shell tensors: source suffix -> our shell key
COMMON = {
    "input_layernorm.weight": "input_layernorm",
    "post_attention_layernorm.weight": "post_attention_layernorm",
    "self_attention_res_proj.weight": "attn_res_proj",
    "self_attention_res_norm.weight": "attn_res_norm",
    "mlp_res_proj.weight": "mlp_res_proj",
    "mlp_res_norm.weight": "mlp_res_norm",
    "self_attn.g_proj.weight": "attn.g_proj",
    "self_attn.o_proj.weight": "attn.o_proj",
}
KDA_ONLY = {
    "self_attn.q_proj.weight": "attn.q_proj",
    "self_attn.k_proj.weight": "attn.k_proj",
    "self_attn.v_proj.weight": "attn.v_proj",
    "self_attn.q_conv1d.weight": "attn.q_conv1d",
    "self_attn.k_conv1d.weight": "attn.k_conv1d",
    "self_attn.v_conv1d.weight": "attn.v_conv1d",
    "self_attn.f_a_proj.weight": "attn.f_a_proj",
    "self_attn.f_b_proj.weight": "attn.f_b_proj",
    "self_attn.b_proj.weight": "attn.b_proj",
    "self_attn.o_norm.weight": "attn.o_norm",
    "self_attn.A_log": "attn.A_log",
    "self_attn.dt_bias": "attn.dt_bias",
}
MLA_ONLY = {
    "self_attn.q_a_proj.weight": "attn.q_a_proj",
    "self_attn.q_a_layernorm.weight": "attn.q_a_layernorm",
    "self_attn.q_b_proj.weight": "attn.q_b_proj",
    "self_attn.kv_a_proj_with_mqa.weight": "attn.kv_a_proj_with_mqa",
    "self_attn.kv_a_layernorm.weight": "attn.kv_a_layernorm",
    "self_attn.kv_b_proj.weight": "attn.kv_b_proj",
}
MOE = {
    "block_sparse_moe.gate.weight": "moe.gate_weight",
    "block_sparse_moe.gate.e_score_correction_bias": "moe.e_score_correction_bias",
    "block_sparse_moe.routed_expert_down_proj.weight": "moe.routed_expert_down_proj",
    "block_sparse_moe.routed_expert_up_proj.weight": "moe.routed_expert_up_proj",
    "block_sparse_moe.routed_expert_norm.weight": "moe.routed_expert_norm",
    "block_sparse_moe.shared_experts.gate_proj.weight": "moe.shared_w1",
    "block_sparse_moe.shared_experts.up_proj.weight": "moe.shared_w3",
    "block_sparse_moe.shared_experts.down_proj.weight": "moe.shared_w2",
}
DENSE = {
    "mlp.gate_proj.weight": "w1",
    "mlp.up_proj.weight": "w3",
    "mlp.down_proj.weight": "w2",
}


def layer_table(li: int, cfg: dict) -> dict:
    """Source-suffix -> shell-key map for layer `li`.

    One definition, shared by the export, the index pre-flight and the
    shard-usage map: if they drifted, `--free-source-shards` could delete a
    shard the export still needs.
    """
    t = dict(COMMON)
    t.update(KDA_ONLY if li in set(cfg["kda_layers"]) else MLA_ONLY)
    t.update(MOE if li >= cfg["first_k_dense_replace"] else DENSE)
    return t


def layer_tensor_names(li: int, cfg: dict) -> list[str]:
    """Every source tensor name the export reads for layer `li`."""
    base = f"{PREFIX}layers.{li}."
    names = [base + sfx for sfx in layer_table(li, cfg)]
    if li >= cfg["first_k_dense_replace"]:
        eb = f"{base}block_sparse_moe.experts."
        for e in range(cfg["num_experts"]):
            for w in ("w1", "w3", "w2"):
                names += [f"{eb}{e}.{w}.weight_packed", f"{eb}{e}.{w}.weight_scale"]
    return names


def top_tensor_names() -> list[str]:
    """Embed / head tensors, read before the layer loop."""
    return [
        PREFIX + "embed_tokens.weight",
        PREFIX + "norm.weight",
        LM_HEAD,
        PREFIX + "output_attn_res_proj.weight",
        PREFIX + "output_attn_res_norm.weight",
    ]


class CkptSource:
    """Lazily-opened safetensors shards, keyed by the index's weight_map."""

    def __init__(self, model_dir: Path):
        from safetensors import safe_open

        self._open = safe_open
        self.dir = model_dir
        idx = model_dir / "model.safetensors.index.json"
        if not idx.exists():
            raise SystemExit(f"[export_kimi_k3] missing {idx}")
        self.map = json.loads(idx.read_text())["weight_map"]
        self._handles = {}

    def _h(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = self._open(str(self.dir / shard), framework="pt")
            self._handles[shard] = h
        return h

    def has(self, name: str) -> bool:
        return name in self.map

    def get(self, name: str):
        if name not in self.map:
            raise ConfigError(f"tensor missing from the checkpoint index: {name}")
        return self._h(self.map[name]).get_tensor(name)

    def close_shard(self, shard: str):
        """Drop the handle so the file can be removed (required on Windows)."""
        self._handles.pop(shard, None)

    def close(self):
        self._handles.clear()



# Serving sidecars copied verbatim beside the weights, so an export is
# self-contained and a node never has to re-fetch from the hub.
SIDECARS = [
    "tiktoken.model",
    "tokenizer_config.json",
    "tokenization_kimi.py",
    "generation_config.json",
    "config.json",
]


def copy_sidecars(model_dir: Path, out: Path):
    r"""Carry the tokenizer + generation config into the export.

    K3 ships a tiktoken BPE (`tiktoken.model` + a `TikTokenTokenizer` class),
    NOT a HF `tokenizer.json`, and no chat template. The engine's API rank loads
    `tokenizer.json`, so a converted one is still required before this model can
    serve text — see docs/architectures/kimi-k3.md. We deliberately do not
    synthesise it here: the tiktoken `pat_str` uses Java/ICU character-class
    intersection (`&&[^\p{Han}]`), which the HF tokenizers and Rust regex
    engines do not accept, so a naive translation silently mis-splits text and
    looks like a model bug rather than a tokenizer bug.
    """
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in SIDECARS:
        src = model_dir / name
        if src.exists():
            (out / name).write_bytes(src.read_bytes())
            copied.append(name)
    print(f"[sidecars] copied {len(copied)}: {', '.join(copied) or 'none'}", flush=True)

    # Build the tokenizer.json the API rank needs. A validation failure is
    # fatal: a subtly wrong tokenizer presents as a model quality problem.
    tj = out / "tokenizer.json"
    if not tj.exists():
        if not (model_dir / "tiktoken.model").exists():
            print("[sidecars] NOTE no tiktoken.model — cannot build tokenizer.json; "
                  "only pre-tokenized input will work", flush=True)
        else:
            from kimi_k3_tokenizer import build_tokenizer_json, read_ranks, validate

            ranks = read_ranks(model_dir)
            tj.write_text(json.dumps(build_tokenizer_json(ranks), ensure_ascii=False))
            print(f"[sidecars] built tokenizer.json ({len(ranks):,} ranks)", flush=True)
            if validate(ranks, tj) != 0:
                raise SystemExit(
                    "[export_kimi_k3] tokenizer.json does not match reference "
                    "tiktoken — refusing to ship it")


def unused_shards(src: "CkptSource", cfg: dict) -> list:
    """Shards holding no text-model tensor — the ViT's own (95, 96 in the
    release). A text-only export never opens them, so they can be excluded from
    the download or freed up front."""
    used = set()
    for n in top_tensor_names():
        if src.has(n):
            used.add(src.map[n])
    for li in range(cfg["num_hidden_layers"]):
        for n in layer_tensor_names(li, cfg):
            if src.has(n):
                used.add(src.map[n])
    return sorted(set(src.map.values()) - used)


def parse_expert_roots(spec: str) -> list:
    """`dir[:max_layers],dir[:max_layers],...` -> [(Path, cap|None)]."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part and part.rsplit(":", 1)[1].isdigit():
            d, n = part.rsplit(":", 1)
            out.append((Path(d), int(n)))
        else:
            out.append((Path(part), None))
    return out


def plan_expert_roots(out: Path, roots: list, cfg: dict) -> dict:
    """Assign each MoE layer's expert bin to one of `roots`. Returns {layer: dir}.

    K3's experts are ~1.45 TB and a host may not have that on any single
    filesystem. Each layer's bin is written to its assigned root and symlinked
    back to `<out>/experts/layer_NN.bin`, so the export still looks like one
    directory and the loader (which just opens that path) is unaffected.

    The plan is persisted to `<out>/.expert_roots.json` and reused on resume:
    recomputing it would give different answers as free space changes, and a
    layer must not move between runs.
    """
    plan_p = out / ".expert_roots.json"
    if plan_p.exists():
        saved = json.loads(plan_p.read_text())
        print(f"[roots] reusing plan for {len(saved)} layers from {plan_p.name}",
              flush=True)
        return {int(k): Path(v) for k, v in saved.items()}

    import shutil
    # a LAYER's bin holds every expert, not one — expert_bin_bytes is per expert
    per = build_manifest(cfg)["expert_bin_bytes"] * cfg["num_experts"]
    headroom = 40 * 1024**3
    moe = [li for li in range(cfg["num_hidden_layers"])
           if li >= cfg["first_k_dense_replace"]]

    caps = []
    for r, want in roots:
        r.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(r).free
        cap = max(0, (free - headroom) // per)
        # an explicit cap can only lower what free space allows
        caps.append((r, min(cap, want) if want is not None else cap))
    total = sum(c for _, c in caps)
    if total < len(moe):
        raise SystemExit(
            f"[export_kimi_k3] expert roots hold {total} layers, need {len(moe)}.\n"
            + "\n".join(f"    {r}: room for {c} ({c * per / 1e9:.0f} GB)"
                         for r, c in caps)
            + f"\n  short by {(len(moe) - total) * per / 1e9:.0f} GB")

    plan, it = {}, iter(moe)
    for r, cap in caps:
        for _ in range(cap):
            li = next(it, None)
            if li is None:
                break
            plan[li] = r
    out.mkdir(parents=True, exist_ok=True)
    plan_p.write_text(json.dumps({str(k): str(v) for k, v in plan.items()}, indent=1))
    for r, cap in caps:
        n = sum(1 for v in plan.values() if v == r)
        if n:
            print(f"[roots] {n:>3} layers ({n * per / 1e9:6.1f} GB) -> {r}", flush=True)
    return plan


def build_shard_usage(src: "CkptSource", cfg: dict) -> dict:
    """shard -> the highest layer index that still reads it.

    `-1` marks the embed/head group, which is consumed before the layer loop.
    Derived from the same `layer_tensor_names` the export uses.
    """
    last: dict = {}
    for n in top_tensor_names():
        if src.has(n):
            last[src.map[n]] = max(last.get(src.map[n], -1), -1)
    for li in range(cfg["num_hidden_layers"]):
        for n in layer_tensor_names(li, cfg):
            if src.has(n):
                sh = src.map[n]
                last[sh] = max(last.get(sh, -1), li)
    return last


def free_consumed_shards(model_dir: Path, src: "CkptSource", last: dict, done_li: int):
    """Delete source shards no later layer will read. DESTRUCTIVE.

    Called only after the layer's `.done` marker is written, so a killed run
    never loses a shard whose output was not committed. A freed layer cannot be
    re-exported without re-downloading.
    """
    freed = 0
    for shard, last_li in list(last.items()):
        if last_li != done_li:
            continue
        p = model_dir / shard
        if p.exists():
            src.close_shard(shard)
            n = p.stat().st_size
            p.unlink()
            freed += n
        last.pop(shard, None)
    if freed:
        print(f"[free] released {freed / 1e9:.1f} GB of consumed source shards",
              flush=True)


def export_real(model_dir: Path, out: Path, cfg: dict, free_source: bool = False,
                expert_roots: list | None = None):
    """Stream the checkpoint into the sparse-moe layout, one layer at a time.

    Resumable: a `.layer_NN.done` marker is written after each layer and
    completed layers are skipped on a re-run.
    """
    src = CkptSource(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(cfg, out)
    copy_sidecars(model_dir, out)
    roots = plan_expert_roots(out, expert_roots, cfg) if expert_roots else None
    spare = unused_shards(src, cfg)
    if spare:
        print(f"[free] {len(spare)} shard(s) hold only vision tensors and are never "
              f"read: {', '.join(spare)}", flush=True)
        print("[free]   pass these to `hf download --exclude` to skip them entirely",
              flush=True)
    usage = build_shard_usage(src, cfg) if free_source else None
    if free_source:
        print(f"[free] --free-source-shards ON: {len(usage)} source shards will be "
              f"DELETED as they are consumed", flush=True)
        # never-read shards can go immediately
        for shard in spare:
            sp = model_dir / shard
            if sp.exists():
                src.close_shard(shard)
                sp.unlink()
                print(f"[free] removed vision-only {shard}", flush=True)
    n = cfg["num_hidden_layers"]
    kda = set(cfg["kda_layers"])
    (out / "shells").mkdir(parents=True, exist_ok=True)

    # embed + head. Dtypes are PRESERVED, not upcast: the source matrices are
    # already bf16 and the shell stores bf16, so widening to f32 would double
    # ~114 GB of shell for nothing — and push the export past the disk it has to
    # fit in. Norms / A_log / dt_bias stay f32 because that is how they ship.
    if not (out / ".head.done").exists():
        save_file({"embed": src.get(PREFIX + "embed_tokens.weight").contiguous()},
                  str(out / "embed.safetensors"))
        save_file({
            "norm": src.get(PREFIX + "norm.weight").contiguous(),
            "lm_head": src.get(LM_HEAD).contiguous(),
            "output_attn_res_proj": src.get(PREFIX + "output_attn_res_proj.weight").contiguous(),
            "output_attn_res_norm": src.get(PREFIX + "output_attn_res_norm.weight").contiguous(),
        }, str(out / "head.safetensors"))
        (out / ".head.done").write_text("ok\n")
        print("[head] embed + head written", flush=True)
        if usage is not None:
            free_consumed_shards(model_dir, src, usage, -1)

    for li in range(n):
        marker = out / f".layer_{li:02d}.done"
        if marker.exists():
            continue
        base = f"{PREFIX}layers.{li}."
        flat = {}
        table = layer_table(li, cfg)

        for suffix, key in table.items():
            t = src.get(base + suffix)
            # conv1d ships as [D, 1, K] (depthwise); the shell wants [D, K]
            if key.endswith("_conv1d") and t.dim() == 3:
                t = t.squeeze(1)
            flat[key] = t.contiguous()
        save_file(flat, str(out / "shells" / f"layer_{li:02d}.safetensors"))

        # routed experts: already mxfp4 -> repack the packed/scale pairs as-is
        if li >= cfg["first_k_dense_replace"]:
            edir = out / "experts"
            edir.mkdir(parents=True, exist_ok=True)
            # with --expert-roots the bin lives on another filesystem; the .part
            # is created THERE so the rename stays atomic and no transient data
            # lands on the export volume
            wdir = roots[li] if roots else edir
            wdir.mkdir(parents=True, exist_ok=True)
            eb = f"{base}block_sparse_moe.experts."
            tmp = wdir / f"layer_{li:02d}.bin.part"
            with open(tmp, "wb") as ef:
                for e in range(cfg["num_experts"]):
                    sections = []
                    for w in ("w1", "w3", "w2"):
                        packed = src.get(f"{eb}{e}.{w}.weight_packed")
                        scale = src.get(f"{eb}{e}.{w}.weight_scale")
                        sections.append((
                            packed.view(torch.uint8).numpy(),
                            scale.view(torch.uint8).numpy(),
                        ))
                    append_repacked_expert(sections, ef)
            # rename only once complete, so a killed run never leaves a short bin
            final = wdir / f"layer_{li:02d}.bin"
            tmp.rename(final)
            if roots:
                link = edir / f"layer_{li:02d}.bin"
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(final)
        marker.write_text("ok\n")
        print(f"[layer {li:02d}/{n - 1}] {'kda' if li in kda else 'mla'} done", flush=True)
        if usage is not None:
            free_consumed_shards(model_dir, src, usage, li)

    src.close()
    print(f"[done] {n} layers -> {out}", flush=True)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def check_index(index_path: Path, cfg: dict):
    """Verify every tensor the streaming pass will ask for exists in the
    checkpoint's index — a pre-flight that needs the index only, not the
    1.56 TB of weights. Catches a name-mapping mistake before a multi-day
    download rather than after it.
    """
    wm = json.loads(Path(index_path).read_text())["weight_map"]
    have = set(wm)
    want, missing = [], []

    want += top_tensor_names()
    for li in range(cfg["num_hidden_layers"]):
        want += layer_tensor_names(li, cfg)

    for nme in want:
        if nme not in have:
            missing.append(nme)

    text = {k for k in have if k.startswith("language_model.")}
    unused = text - set(want)
    print(f"[check-index] requested {len(want):,} tensors, {len(missing)} missing")
    print(f"[check-index] text-model tensors in index: {len(text):,}")
    print(f"[check-index] dropped (vision/projector): {len(have) - len(text):,}")
    if unused:
        print(f"[check-index] WARNING {len(unused)} text tensors unused, e.g.:")
        for k in sorted(unused)[:5]:
            print(f"    {k}")
    if missing:
        print("[check-index] MISSING, e.g.:")
        for k in missing[:10]:
            print(f"    {k}")
        raise SystemExit("[export_kimi_k3] index pre-flight FAILED")
    print("[check-index] OK — every required tensor is present")


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
    ap.add_argument("--expert-roots",
                    help="comma-separated dirs to spread the expert bins over, for "
                         "hosts with no single filesystem big enough. Each bin is "
                         "symlinked back into <out>/experts/. Append :N to cap a "
                         "dir at N layers, e.g. /mnt/a:33,/mnt/b")
    ap.add_argument("--free-source-shards", action="store_true",
                    help="DELETE each source shard once no remaining layer needs it. "
                         "Required to fit source+output on one disk; the freed "
                         "layers cannot be re-exported without re-downloading")
    ap.add_argument("--check-index", type=Path,
                    help="verify tensor names against model.safetensors.index.json "
                         "(needs --config); no weights required")
    ap.add_argument("--config", type=Path, help="config.json for --check-index")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if a.check_index:
        if not a.config:
            raise SystemExit("--check-index requires --config CONFIG.json")
        check_index(a.check_index, load_and_validate_config(a.config))
        return
    if a.validate:
        cfg = load_and_validate_config(a.validate)
        print(json.dumps(build_manifest(cfg), indent=2))
        return
    if a.tiny:
        er = parse_expert_roots(a.expert_roots) if a.expert_roots else None
        export_tiny(a.tiny, expert_roots=er)
        return
    if a.model:
        if not a.out:
            raise SystemExit("--model requires --out")
        cfg = load_and_validate_config(a.model / "config.json")
        a.out.mkdir(parents=True, exist_ok=True)
        check_space(a.out, cfg)
        er = parse_expert_roots(a.expert_roots) if a.expert_roots else None
        export_real(a.model, a.out, cfg, free_source=a.free_source_shards,
                    expert_roots=er)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
