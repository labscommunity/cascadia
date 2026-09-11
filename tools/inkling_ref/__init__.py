"""Inkling (Thinking Machines `thinkingmachines/Inkling`, 975B/41B-active MoE) reference helpers.

Shared by `tools/export_inkling.py --tiny`, `tools/inkling_ref/gen_fixtures.py` and
`tools/tests/test_inkling_export.py`:

- the tiny synthetic config (PORT_SPEC §4) as a *checkpoint-style* config.json dict;
- `build_tiny_model`: transformers' own `InklingForCausalLM` on that config, float32, every
  weight drawn from seed 7 and ROUNDED TO bf16 (so the engine's bf16 storage is lossless);
- the HF-module-name <-> checkpoint-name mapping (the reverse of transformers'
  `conversion_mapping.py` "inkling_mm_model" entry), incl. the gate/up (de)interleave rule;
- int4 group-32 dequant + `load_export_as_hf`: read an `export_inkling.py` output dir back
  into a fresh HF model, so `reference.json` compares like with like (int4-dequantized weights
  on both sides of the Rust/Python parity test).

transformers >= 5.16 ships `models/inkling` natively; HF *is* the oracle for this port.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

TINY_SEED = 7
PROMPT_LEN = 12
N_GEN = 8
INT4_GROUP = 32

# Tiny text config, spelled the way the real checkpoint's config.json spells it (SGLang-style
# keys incl. the contract flags export_inkling.py enforces). PORT_SPEC §4.
TINY_TEXT_CONFIG = {
    "model_type": "inkling_text",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "vocab_size": 128,
    "unpadded_vocab_size": 120,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 16,
    "d_rel": 4,
    "rel_extent": 8,
    "sliding_window_size": 4,
    "local_layer_ids": [0, 1, 2],
    "dense_mlp_idx": 1,
    "dense_intermediate_size": 64,
    "intermediate_size": 32,  # the checkpoint's `intermediate_size` is the MoE intermediate
    "moe_intermediate_size": 32,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 2,
    "route_scale": 8.0,
    "rms_norm_eps": 1e-6,
    "log_scaling_n_floor": 4,
    "log_scaling_alpha": 0.1,
    "logits_mup_width_multiplier": 2.0,
    "sconv_kernel_size": 4,
    "hidden_act": "silu",
    # >= unpadded_vocab_size, so the sliced logits can never produce it: greedy never stops early.
    "eos_token_id": 127,
    # contract flags (present in the real config.json; hard-checked by the exporter)
    "gate_activation": "sigmoid",
    "norm_after_topk": True,
    "use_global_scale": True,
    "use_gate_bias": True,
    "use_sconv": True,
    "use_embed_norm": True,
    "shared_expert_sink": True,
    "q_bias": False,
    "o_bias": False,
    "final_logit_softcapping": None,
}
TINY_CONFIG = {
    "model_type": "inkling_mm_model",
    "architectures": ["InklingForConditionalGeneration"],
    "text_config": TINY_TEXT_CONFIG,
}


# --------------------------------------------------------------------------
# gate/up interleave rule (transformers core_model_loading.Interleave)
# --------------------------------------------------------------------------
def interleave(gate: torch.Tensor, up: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Checkpoint layout along `dim`: row 2i = gate_i, row 2i+1 = up_i. This is the INVERSE of
    transformers' `Interleave(dim)` load-time op (`Interleave(dim, inverse=True)`)."""
    return torch.stack([gate, up], dim=dim + 1).flatten(dim, dim + 1).contiguous()


def deinterleave(w13: torch.Tensor, dim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """transformers `Interleave(dim)`: reshape `[2I] -> [I, 2]`, transpose -> `[2, I]`, so
    gate = rows 0::2 and up = rows 1::2 along `dim`."""
    n = w13.shape[dim]
    x = w13.unflatten(dim, (n // 2, 2))
    return x.select(dim + 1, 0).contiguous(), x.select(dim + 1, 1).contiguous()


# --------------------------------------------------------------------------
# HF module names <-> checkpoint names (reverse of conversion_mapping "inkling_mm_model")
# --------------------------------------------------------------------------
_GLOBAL_HF_TO_CKPT = {
    "model.embed_tokens.weight": "model.llm.embed.weight",
    "model.embed_norm.weight": "model.llm.embed_norm.weight",
    "model.norm.weight": "model.llm.norm.weight",
    "lm_head.weight": "model.llm.unembed.weight",
}
_LAYER_HF_TO_CKPT = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "mlp_norm.weight",
    "self_attn.q_proj.weight": "attn.wq_du.weight",
    "self_attn.k_proj.weight": "attn.wk_dv.weight",
    "self_attn.v_proj.weight": "attn.wv_dv.weight",
    "self_attn.r_proj.weight": "attn.wr_du.weight",
    "self_attn.o_proj.weight": "attn.wo_ud.weight",
    "self_attn.q_norm.weight": "attn.q_norm.weight",
    "self_attn.k_norm.weight": "attn.k_norm.weight",
    "self_attn.k_sconv.conv1d.weight": "attn.k_sconv.weight",
    "self_attn.v_sconv.conv1d.weight": "attn.v_sconv.weight",
    "self_attn.rel_logits_proj.proj": "attn.rel_logits_proj.proj",
    "attn_sconv.conv1d.weight": "attn_sconv.weight",
    "mlp_sconv.conv1d.weight": "mlp_sconv.weight",
    "mlp.gate.weight": "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias": "mlp.gate.bias",
    "mlp.gate.global_scale": "mlp.gate.global_scale",
    "mlp.global_scale": "mlp.global_scale",
    "mlp.experts.down_proj": "mlp.experts.w2_weight",
    "mlp.shared_experts.down_proj": "mlp.shared_experts.shared_w2_weight",
    "mlp.down_proj.weight": "mlp.w2_md.weight",
}
_LAYER_CKPT_TO_HF = {v: k for k, v in _LAYER_HF_TO_CKPT.items()}
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def hf_state_to_checkpoint(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """InklingForCausalLM state_dict -> checkpoint-named tensors (`model.llm.*`), gate/up
    RE-INTERLEAVED into the checkpoint's w13 layout, so the result looks exactly like a slice
    of thinkingmachines/Inkling."""
    out: dict[str, torch.Tensor] = {}
    parts: dict[int, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        v = v.detach()
        if k in _GLOBAL_HF_TO_CKPT:
            out[_GLOBAL_HF_TO_CKPT[k]] = v
            continue
        m = _LAYER_RE.match(k)
        if not m:
            raise KeyError(f"unmapped HF key {k!r}")
        li, suf = int(m.group(1)), m.group(2)
        p = f"model.llm.layers.{li}."
        if suf in _LAYER_HF_TO_CKPT:
            out[p + _LAYER_HF_TO_CKPT[suf]] = v
        elif suf == "mlp.experts.gate_up_proj":  # [E, 2I, H], HF chunk(2, dim=-1): first I = gate
            inter = v.shape[1] // 2
            out[p + "mlp.experts.w13_weight"] = interleave(v[:, :inter], v[:, inter:], dim=1)
        elif suf in ("mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj",
                     "mlp.gate_proj.weight", "mlp.up_proj.weight"):
            parts.setdefault(li, {})[suf] = v
        else:
            raise KeyError(f"unmapped HF layer key {k!r}")
    for li, d in parts.items():
        p = f"model.llm.layers.{li}."
        if "mlp.shared_experts.gate_proj" in d:
            out[p + "mlp.shared_experts.shared_w13_weight"] = interleave(
                d["mlp.shared_experts.gate_proj"], d["mlp.shared_experts.up_proj"], dim=1)
        if "mlp.gate_proj.weight" in d:
            out[p + "mlp.w13_dn.weight"] = interleave(d["mlp.gate_proj.weight"], d["mlp.up_proj.weight"], dim=0)
    return out


def checkpoint_state_to_hf(ckpt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Inverse of `hf_state_to_checkpoint` (what transformers' conversion mapping does on load)."""
    out: dict[str, torch.Tensor] = {}
    g2c = {v: k for k, v in _GLOBAL_HF_TO_CKPT.items()}
    lre = re.compile(r"^model\.llm\.layers\.(\d+)\.(.+)$")
    for k, v in ckpt.items():
        if k in g2c:
            out[g2c[k]] = v
            continue
        m = lre.match(k)
        if not m:
            raise KeyError(f"unmapped checkpoint key {k!r}")
        li, suf = int(m.group(1)), m.group(2)
        p = f"model.layers.{li}."
        if suf in _LAYER_CKPT_TO_HF:
            out[p + _LAYER_CKPT_TO_HF[suf]] = v
        elif suf == "mlp.experts.w13_weight":
            gate, up = deinterleave(v, dim=1)
            out[p + "mlp.experts.gate_up_proj"] = torch.cat([gate, up], dim=1)
        elif suf == "mlp.shared_experts.shared_w13_weight":
            gate, up = deinterleave(v, dim=1)
            out[p + "mlp.shared_experts.gate_proj"] = gate
            out[p + "mlp.shared_experts.up_proj"] = up
        elif suf == "mlp.w13_dn.weight":
            gate, up = deinterleave(v, dim=0)
            out[p + "mlp.gate_proj.weight"] = gate
            out[p + "mlp.up_proj.weight"] = up
        else:
            raise KeyError(f"unmapped checkpoint layer key {k!r}")
    return out


# --------------------------------------------------------------------------
# manifest -> HF config, tiny model, prompt
# --------------------------------------------------------------------------
def hf_config_from_manifest(man: dict):
    """`InklingTextConfig` for an export manifest (eager attention: the relative-position bias is
    a `position_bias` kwarg; eager is the path we validated against)."""
    from transformers import InklingTextConfig

    dense = set(man["dense_layers"])
    eos = man.get("eos_token_ids") or []
    return InklingTextConfig(
        vocab_size=man["vocab_size"],
        unpadded_vocab_size=man["unpadded_vocab_size"],
        hidden_size=man["hidden_size"],
        num_hidden_layers=man["num_layers"],
        num_attention_heads=man["num_attention_heads"],
        num_key_value_heads=man["num_kv_heads"],
        head_dim=man["head_dim"],
        swa_num_attention_heads=man["swa_num_attention_heads"],
        swa_num_key_value_heads=man["swa_num_kv_heads"],
        swa_head_dim=man["swa_head_dim"],
        sliding_window_size=man["sliding_window"],
        d_rel=man["d_rel"],
        rel_extent=man["rel_extent"],
        log_scaling_n_floor=man["log_scaling_n_floor"],
        log_scaling_alpha=man["log_scaling_alpha"],
        layer_types=["hybrid_sliding" if t == "sliding" else "hybrid" for t in man["layer_types"]],
        mlp_layer_types=["dense" if i in dense else "sparse" for i in range(man["num_layers"])],
        rms_norm_eps=man["rms_norm_eps"],
        conv_kernel_size=man["conv_kernel_size"],
        intermediate_size=man["dense_intermediate"] or man["moe_intermediate"],
        moe_intermediate_size=man["moe_intermediate"],
        hidden_act=man["hidden_act"],
        n_routed_experts=man["num_experts"],
        num_experts_per_tok=man["top_k"],
        n_shared_experts=man["n_shared_experts"],
        route_scale=man["route_scale"],
        logits_mup_width_multiplier=man["logits_mup_width_multiplier"],
        eos_token_id=eos[0] if eos else None,
        pad_token_id=None,
        bos_token_id=None,
        attn_implementation="eager",
    )


def _is_unit_param(name: str) -> bool:
    """RMSNorm weights and global scales are drawn around 1 (a ~0 norm weight would zero the
    residual stream); everything else around 0."""
    return name.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight",
                          "embed_norm.weight", "model.norm.weight", "global_scale"))


UNEMBED_STD = 0.5   # 10x the other weights: keeps top-1/top-2 logit gaps far above bf16 noise
# Sanity floor on the top-2 logit gap at every prompt position + greedy step (logits have std ~2,
# |max| ~6; bf16 write-back moves them by ~0.5%, so 0.15 is ~5x the expected noise). The chosen
# prompt is the best of N_PROMPT_CANDIDATES (typically ~0.22).
MIN_ARGMAX_MARGIN = 0.15
N_PROMPT_CANDIDATES = 512


def build_tiny_model(man: dict, seed: int = TINY_SEED):
    """transformers `InklingForCausalLM` on the tiny config, float32. Every state_dict entry is
    drawn from one seeded generator in state_dict order — N(0, 0.05) for weights, 1 + N(0, 0.05)
    for norms / global scales, N(0, UNEMBED_STD) for `unembed` (with 0.05 the logits have std 0.2
    over 120 candidates and the argmax ties at bf16 precision) — then ROUNDED TO bf16 and stored as
    f32 (bf16 shells lossless)."""
    from transformers import InklingForCausalLM

    cfg = hf_config_from_manifest(man)
    torch.manual_seed(0)  # HF's own init runs first; overwritten below
    model = InklingForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(seed)
    new = {}
    for k, v in model.state_dict().items():
        t = torch.randn(v.shape, generator=g) * (UNEMBED_STD if k == "lm_head.weight" else 0.05)
        if _is_unit_param(k):
            t = t + 1.0
        new[k] = t.to(torch.bfloat16).to(torch.float32)
    model.load_state_dict(new, strict=True)
    assert model.config._attn_implementation == "eager"
    return model


def prompt_ids(man: dict, seed: int = 0, n: int = PROMPT_LEN) -> list[int]:
    """Candidate prompt #seed (tokens < unpadded_vocab_size). The fixture prompt is chosen by
    `select_prompt`, which scans seeds for robust argmax margins."""
    g = torch.Generator().manual_seed(TINY_SEED + 1000 + seed)
    return torch.randint(0, man["unpadded_vocab_size"], (n,), generator=g).tolist()


def int4_roundtrip_state(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """HF state_dict with every FFN weight (routed / shared / dense) replaced by its int4 group-32
    pack -> dequant round-trip — exactly what `load_export_as_hf` sees after `export_inkling.py`."""
    import export_inkling  # tools/ is on sys.path for every entry point that uses this package

    def rt(w):
        out, inn = w.shape
        p, s = export_inkling.pack_int4(w)
        return dequant_int4_section(p + s, 0, out, inn)[0]

    new = dict(sd)
    for k, v in sd.items():
        if k.endswith("mlp.experts.gate_up_proj"):
            inter = v.shape[1] // 2
            new[k] = torch.stack([torch.cat([rt(v[e, :inter]), rt(v[e, inter:])], 0) for e in range(v.shape[0])])
        elif k.endswith(("mlp.experts.down_proj", "mlp.shared_experts.gate_proj",
                         "mlp.shared_experts.up_proj", "mlp.shared_experts.down_proj")):
            new[k] = torch.stack([rt(v[e]) for e in range(v.shape[0])])
        elif k.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")):
            new[k] = rt(v)
    return new


def int4_roundtrip_model(model, man: dict):
    from transformers import InklingForCausalLM

    m = InklingForCausalLM(hf_config_from_manifest(man)).float().eval()
    m.load_state_dict(int4_roundtrip_state(model.state_dict()), strict=True)
    return m


@torch.no_grad()
def argmax_margins(model, prompt: list[int], n_gen: int = N_GEN):
    """top-1 minus top-2 logit at every prompt position, then at every greedy step (no cache)."""
    ids = torch.tensor([prompt], dtype=torch.long)
    lg = model(ids, use_cache=False).logits[0]
    t2 = lg.topk(2, dim=-1).values
    prompt_m = (t2[:, 0] - t2[:, 1]).tolist()
    cur, gen_m = ids.clone(), []
    for _ in range(n_gen):
        l1 = model(cur, use_cache=False).logits[0, -1]
        top2 = l1.topk(2).values
        gen_m.append(float(top2[0] - top2[1]))
        cur = torch.cat([cur, torch.tensor([[int(l1.argmax())]])], dim=1)
    return {"prompt": prompt_m, "greedy": gen_m}


@torch.no_grad()
def argmax_margins_batched(model, prompts: torch.Tensor, n_gen: int = N_GEN):
    """prompts [N, T] (equal lengths, no padding) -> (prompt margins [N, T], greedy margins [N, n_gen])."""
    t2 = model(prompts, use_cache=False).logits.topk(2, dim=-1).values
    pm = t2[..., 0] - t2[..., 1]
    cur, gm = prompts.clone(), []
    for _ in range(n_gen):
        l1 = model(cur, use_cache=False).logits[:, -1]
        top2 = l1.topk(2, dim=-1).values
        gm.append(top2[:, 0] - top2[:, 1])
        cur = torch.cat([cur, l1.argmax(-1, keepdim=True)], dim=1)
    return pm, torch.stack(gm, dim=1)


def select_prompt(man: dict, model=None, min_margin: float = MIN_ARGMAX_MARGIN,
                  n_candidates: int = N_PROMPT_CANDIDATES):
    """The fixture prompt: of `n_candidates` seeded candidates, the one with the LARGEST minimum
    top-2 logit gap over its 12 prompt positions + 8 greedy steps, evaluated for BOTH the f32 tiny
    model and its int4 round-trip (the --tiny reference model), so the 'argmax token sequence
    EXACT' contract holds through bf16 write-back on either side. Returns (prompt, seed, margin)."""
    model = model or build_tiny_model(man)
    mq = int4_roundtrip_model(model, man)
    cands = torch.stack([torch.tensor(prompt_ids(man, s), dtype=torch.long) for s in range(n_candidates)])
    mins = None
    for m in (model, mq):
        pm, gm = argmax_margins_batched(m, cands)
        mm = torch.minimum(pm.min(dim=1).values, gm.min(dim=1).values)
        mins = mm if mins is None else torch.minimum(mins, mm)
    seed = int(mins.argmax())
    margin = float(mins[seed])
    if margin < min_margin:
        raise AssertionError(f"best of {n_candidates} prompt candidates has argmax margin {margin:.3f} < {min_margin}")
    return cands[seed].tolist(), seed, margin


@torch.no_grad()
def greedy_reference(model, prompt: list[int], n_gen: int = N_GEN):
    """{prompt_ids, greedy_ids, first_logits_argmax} from HF: `generate(do_sample=False)` (cached
    decode) cross-checked against a no-cache re-prefill loop. Also returns the argmax margins
    ({'prompt': [12], 'greedy': [8]}: how robust each argmax is to bf16 write-back)."""
    ids = torch.tensor([prompt], dtype=torch.long)
    first = model(ids, use_cache=False).logits[0].argmax(-1).tolist()
    gen = model.generate(ids, max_new_tokens=n_gen, do_sample=False)
    greedy = gen[0, ids.shape[1]:].tolist()
    cur, manual = ids.clone(), []
    for _ in range(n_gen):
        t = int(model(cur, use_cache=False).logits[0, -1].argmax())
        manual.append(t)
        cur = torch.cat([cur, torch.tensor([[t]])], dim=1)
    if manual != greedy:
        raise AssertionError(f"HF cached generate {greedy} != no-cache greedy {manual}")
    return ({"prompt_ids": list(prompt), "greedy_ids": greedy, "first_logits_argmax": first},
            argmax_margins(model, prompt, n_gen))


# --------------------------------------------------------------------------
# int4 group-32 bins (export_glm5._pack_int4_grouped layout) -> f32
# --------------------------------------------------------------------------
def int4_bin_bytes(hidden: int, inter: int) -> int:
    def sec(o, i):
        return o * i // 2 + o * (i // INT4_GROUP) * 2
    return 2 * sec(inter, hidden) + sec(hidden, inter)


def dequant_int4_section(buf: bytes, off: int, out_dim: int, in_dim: int):
    """One section -> (torch f32 [out_dim, in_dim], new offset). Mirrors the Rust
    `dequant_int4` / `tools/glm5_ref/load_export.py`."""
    ng = in_dim // INT4_GROUP
    packed_len = out_dim * in_dim // 2
    scale_len = out_dim * ng * 2
    packed = np.frombuffer(buf[off:off + packed_len], dtype=np.uint8)
    scales = np.frombuffer(buf[off + packed_len:off + packed_len + scale_len], dtype="<u2")
    lo = (packed & 0x0F).astype(np.int32) - 8
    hi = ((packed >> 4) & 0x0F).astype(np.int32) - 8
    nib = np.empty(out_dim * in_dim, dtype=np.int32)
    nib[0::2] = lo
    nib[1::2] = hi
    s = (scales.astype(np.uint32) << 16).view(np.float32).reshape(out_dim, ng)
    w = nib.reshape(out_dim, ng, INT4_GROUP).astype(np.float32) * s[:, :, None]
    return torch.from_numpy(w.reshape(out_dim, in_dim).copy()), off + packed_len + scale_len


def dequant_expert_bin(path, hidden: int, inter: int):
    """expert bin -> (gate [inter, hidden], up [inter, hidden], down [hidden, inter]) f32."""
    buf = Path(path).read_bytes()
    if len(buf) != int4_bin_bytes(hidden, inter):
        raise ValueError(f"{path}: {len(buf)} bytes, expected {int4_bin_bytes(hidden, inter)}")
    gate, off = dequant_int4_section(buf, 0, inter, hidden)
    up, off = dequant_int4_section(buf, off, inter, hidden)
    down, off = dequant_int4_section(buf, off, hidden, inter)
    return gate, up, down


def load_export_as_hf(export_dir):
    """Read an `export_inkling.py` output dir back into a fresh `InklingForCausalLM` (f32):
    bf16 shells as-is (lossless), int4 bins DEQUANTIZED. Returns (model, manifest)."""
    from safetensors.torch import load_file
    from transformers import InklingForCausalLM

    d = Path(export_dir)
    man = json.loads((d / "manifest.json").read_text())
    if man.get("arch") != "inkling":
        raise ValueError(f"{d}: manifest arch {man.get('arch')!r} != 'inkling'")
    H, I, Id = man["hidden_size"], man["moe_intermediate"], man["dense_intermediate"]
    dense = set(man["dense_layers"])
    sd: dict[str, torch.Tensor] = {}
    emb = load_file(str(d / "embed.safetensors"))
    sd["model.embed_tokens.weight"] = emb["embed.weight"].float()
    sd["model.embed_norm.weight"] = emb["embed_norm.weight"].float()
    head = load_file(str(d / "head.safetensors"))
    sd["lm_head.weight"] = head["unembed.weight"].float()
    sd["model.norm.weight"] = head["norm.weight"].float()
    for li in range(man["num_layers"]):
        p = f"model.layers.{li}."
        sh = load_file(str(d / "shells" / f"layer_{li:02d}.safetensors"))
        for suf, t in sh.items():
            hf = _LAYER_CKPT_TO_HF[suf]
            t = t.float()
            if suf.endswith("sconv.weight"):  # stored [C, K]; HF conv1d wants [C, 1, K]
                t = t.unsqueeze(1)
            sd[p + hf] = t.contiguous()
        edir = d / "experts" / f"layer_{li:02d}"
        if li in dense:
            g, u, dn = dequant_expert_bin(edir / "dense.bin", H, Id)
            sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"], sd[p + "mlp.down_proj.weight"] = g, u, dn
        else:
            gu, dw = [], []
            for e in range(man["num_experts"]):
                g, u, dn = dequant_expert_bin(edir / f"expert_{e:03d}.bin", H, I)
                gu.append(torch.cat([g, u], dim=0))
                dw.append(dn)
            sd[p + "mlp.experts.gate_up_proj"] = torch.stack(gu)
            sd[p + "mlp.experts.down_proj"] = torch.stack(dw)
            sg, su, sdn = [], [], []
            for s in range(man["n_shared_experts"]):
                g, u, dn = dequant_expert_bin(edir / f"expert_shared{s}.bin", H, I)
                sg.append(g)
                su.append(u)
                sdn.append(dn)
            sd[p + "mlp.shared_experts.gate_proj"] = torch.stack(sg)
            sd[p + "mlp.shared_experts.up_proj"] = torch.stack(su)
            sd[p + "mlp.shared_experts.down_proj"] = torch.stack(sdn)
    model = InklingForCausalLM(hf_config_from_manifest(man)).float().eval()
    model.load_state_dict(sd, strict=True)
    return model, man


__all__ = [
    "TINY_SEED", "PROMPT_LEN", "N_GEN", "INT4_GROUP", "TINY_TEXT_CONFIG", "TINY_CONFIG",
    "interleave", "deinterleave", "hf_state_to_checkpoint", "checkpoint_state_to_hf",
    "hf_config_from_manifest", "build_tiny_model", "prompt_ids", "select_prompt", "argmax_margins",
    "argmax_margins_batched", "int4_roundtrip_state", "int4_roundtrip_model", "greedy_reference",
    "UNEMBED_STD", "MIN_ARGMAX_MARGIN", "N_PROMPT_CANDIDATES",
    "int4_bin_bytes", "dequant_int4_section", "dequant_expert_bin", "load_export_as_hf",
]
