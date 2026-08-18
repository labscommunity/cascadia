"""Read a `tools/export_glm5.py` on-disk layout and run greedy generation via
the CPU reference. This is the parity target for the Rust `glm::loader`: both
read the SAME files (bf16 shell safetensors + int4 expert bins), so a
token-exact match validates the exporter write + the Rust loader read + the
int4 dequant round-trip all at once.
"""
import json
from pathlib import Path

import numpy as np
import torch

from glm5_ref.kernels_ref import model_ref

_G = 32


def _dequant_section(buf, off, out_dim, in_dim):
    """One int4 section -> (torch.f32 [out_dim, in_dim], new_off). Mirrors the
    Rust `glm::loader::dequant_int4` and `export_glm5._pack_int4_grouped`."""
    ng = in_dim // _G
    packed_len = out_dim * in_dim // 2
    scale_len = out_dim * ng * 2
    packed = np.frombuffer(buf[off:off + packed_len], dtype=np.uint8)
    scales = np.frombuffer(buf[off + packed_len:off + packed_len + scale_len], dtype="<u2")
    off2 = off + packed_len + scale_len
    lo = (packed & 0x0F).astype(np.int32) - 8
    hi = ((packed >> 4) & 0x0F).astype(np.int32) - 8
    nib = np.empty(out_dim * in_dim, dtype=np.int32)
    nib[0::2] = lo
    nib[1::2] = hi
    s_f32 = (scales.astype(np.uint32) << 16).view(np.float32).reshape(out_dim, ng)
    w = nib.reshape(out_dim, ng, _G).astype(np.float32) * s_f32[:, :, None]
    return torch.from_numpy(w.reshape(out_dim, in_dim).copy()), off2


def _dequant_expert_bin(path, hidden, inter):
    buf = Path(path).read_bytes()
    off = 0
    wg, off = _dequant_section(buf, off, inter, hidden)
    wu, off = _dequant_section(buf, off, inter, hidden)
    wd, off = _dequant_section(buf, off, hidden, inter)
    return (wg, wu, wd)


def load_export_and_generate(export_dir, prompt, n_gen):
    from safetensors.torch import load_file

    d = Path(export_dir)
    man = json.loads((d / "manifest.json").read_text())
    hidden, DI, MI = man["hidden_size"], man["dense_intermediate"], man["expert_intermediate"]
    E, dense_layers = man["num_experts"], set(man["dense_layers"])
    cfg = {
        "n_heads": man["num_attention_heads"], "qk_nope": man["qk_nope_head_dim"],
        "qk_rope": man["qk_rope_head_dim"], "v_head": man["v_head_dim"],
        "kv_lora": man["kv_lora_rank"], "q_lora": man["q_lora_rank"],
        "eps": man["rms_norm_eps"], "theta": man["rope_theta"],
        "top_k": man["top_k"], "scale": man["routed_scaling_factor"],
    }
    embed = load_file(d / "embed.safetensors")["embed.weight"].float()
    head = load_file(d / "head.safetensors")
    lm_head, final_norm = head["head.weight"].float(), head["norm.weight"].float()

    layers = []
    for li in range(man["num_layers"]):
        sh = load_file(d / "shells" / f"layer_{li:02d}.safetensors")
        attn = {
            "wq_a": sh["self_attn.wq_a.weight"].float(),
            "q_a_ln": sh["self_attn.q_a_layernorm.weight"].float(),
            "wq_b": sh["self_attn.wq_b.weight"].float(),
            "wkv_a": sh["self_attn.wkv_a.weight"].float(),
            "kv_a_ln": sh["self_attn.kv_a_layernorm.weight"].float(),
            "wkv_b": sh["self_attn.wkv_b.weight"].float(),
            "wo": sh["self_attn.o_proj.weight"].float(),
        }
        edir = d / "experts" / f"layer_{li:02d}"
        base = {"in_ln": sh["input_layernorm.weight"].float(),
                "post_ln": sh["post_attention_layernorm.weight"].float(), "attn": attn}
        if li in dense_layers:
            base["kind"] = "dense"
            base["mlp"] = _dequant_expert_bin(edir / "dense.bin", hidden, DI)
        else:
            rw = sh["mlp.gate.weight"].float()
            rb = sh["mlp.gate.e_score_correction_bias"].float()
            exps = [_dequant_expert_bin(edir / f"expert_{e:03d}.bin", hidden, MI) for e in range(E)]
            shd = _dequant_expert_bin(edir / "expert_shared.bin", hidden, MI)
            base["kind"] = "moe"
            base["mlp"] = (rw, rb, exps, shd)
        layers.append(base)

    return model_ref(cfg, {"embed": embed, "final_norm": final_norm,
                           "lm_head": lm_head, "layers": layers}, prompt, n_gen)
