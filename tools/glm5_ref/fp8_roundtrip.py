"""Build a tiny synthetic GLM-5.2-FP8 checkpoint (real HF tensor names, fp8
e4m3 weights + block-128 weight_scale_inv, bf16 norms/router/bias), run the
real exporter on it, and return the greedy tokens the exporter's output
produces. A Rust test then loads that same export and must reproduce them —
validating export_real's name mapping + FP8 dequant + int4 repack end-to-end
with no 744B checkpoint.
"""
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

# Tiny dims (int4 group 32-divisible; fp8 block 128 collapses to [1,1] here).
CFG = dict(
    hidden_size=32, intermediate_size=64, moe_intermediate_size=32, vocab_size=16,
    num_hidden_layers=3, num_attention_heads=3, q_lora_rank=16, kv_lora_rank=8,
    qk_nope_head_dim=6, qk_rope_head_dim=4, v_head_dim=6, n_routed_experts=4,
    n_shared_experts=1, num_experts_per_tok=2, first_k_dense_replace=1, n_group=1,
    topk_group=1, routed_scaling_factor=2.5, scoring_func="sigmoid", rope_theta=8.0e6,
    rms_norm_eps=1e-5, num_nextn_predict_layers=1, eos_token_id=[0],
    quantization_config=dict(quant_method="fp8", fmt="e4m3", activation_scheme="dynamic",
                             weight_block_size=[128, 128], modules_to_not_convert=[]),
)


def _to_fp8_blocked(w: torch.Tensor):
    """f32 [O,I] -> (fp8 e4m3 [O,I], weight_scale_inv f32 [ceil(O/128),ceil(I/128)]).
    dequant = fp8.f32 * scale_inv(expanded)."""
    o, i = w.shape
    nb_o, nb_i = (o + 127) // 128, (i + 127) // 128
    scale = torch.ones(nb_o, nb_i)
    q = torch.zeros(o, i)
    for bo in range(nb_o):
        for bi in range(nb_i):
            r0, r1 = bo * 128, min((bo + 1) * 128, o)
            c0, c1 = bi * 128, min((bi + 1) * 128, i)
            blk = w[r0:r1, c0:c1]
            s = (blk.abs().max().clamp_min(1e-6) / 448.0)
            scale[bo, bi] = s
            q[r0:r1, c0:c1] = (blk / s).clamp(-448, 448)
    return q.to(torch.float8_e4m3fn), scale


def build_synthetic_fp8_ckpt(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(CFG, indent=2))
    g = torch.Generator().manual_seed(11)

    def rnd(*shape, s=0.3):
        return s * torch.randn(*shape, generator=g)

    def nf(n):
        return 1.0 + 0.05 * torch.randn(n, generator=g)

    C = CFG
    H, QK = C["hidden_size"], C["qk_nope_head_dim"] + C["qk_rope_head_dim"]
    Hh, VH, KVL, QL = C["num_attention_heads"], C["v_head_dim"], C["kv_lora_rank"], C["q_lora_rank"]
    MI, DI, E, V = C["moe_intermediate_size"], C["intermediate_size"], C["n_routed_experts"], C["vocab_size"]

    tensors = {}

    def put_bf16(name, t):
        tensors[name] = t.to(torch.bfloat16)

    def put_fp8(name, t):
        q, s = _to_fp8_blocked(t.to(torch.float32))
        tensors[name] = q
        tensors[name.rsplit(".", 1)[0] + ".weight_scale_inv"] = s

    put_bf16("model.embed_tokens.weight", rnd(V, H))
    put_bf16("model.norm.weight", nf(H))
    put_bf16("lm_head.weight", rnd(V, H))
    for li in range(C["num_hidden_layers"]):
        p = f"model.layers.{li}."
        put_bf16(p + "input_layernorm.weight", nf(H))
        put_bf16(p + "post_attention_layernorm.weight", nf(H))
        put_fp8(p + "self_attn.q_a_proj.weight", rnd(QL, H))
        put_bf16(p + "self_attn.q_a_layernorm.weight", nf(QL))
        put_fp8(p + "self_attn.q_b_proj.weight", rnd(Hh * QK, QL))
        put_fp8(p + "self_attn.kv_a_proj_with_mqa.weight", rnd(KVL + C["qk_rope_head_dim"], H))
        put_bf16(p + "self_attn.kv_a_layernorm.weight", nf(KVL))
        put_fp8(p + "self_attn.kv_b_proj.weight", rnd(Hh * (C["qk_nope_head_dim"] + VH), KVL))
        put_fp8(p + "self_attn.o_proj.weight", rnd(H, Hh * VH))
        if li < C["first_k_dense_replace"]:
            put_fp8(p + "mlp.gate_proj.weight", rnd(DI, H))
            put_fp8(p + "mlp.up_proj.weight", rnd(DI, H))
            put_fp8(p + "mlp.down_proj.weight", rnd(H, DI))
        else:
            put_bf16(p + "mlp.gate.weight", rnd(E, H))
            put_bf16(p + "mlp.gate.e_score_correction_bias", 0.3 * torch.randn(E, generator=g))
            for e in range(E):
                ep = p + f"mlp.experts.{e}."
                put_fp8(ep + "gate_proj.weight", rnd(MI, H))
                put_fp8(ep + "up_proj.weight", rnd(MI, H))
                put_fp8(ep + "down_proj.weight", rnd(H, MI))
            sp = p + "mlp.shared_experts."
            put_fp8(sp + "gate_proj.weight", rnd(MI, H))
            put_fp8(sp + "up_proj.weight", rnd(MI, H))
            put_fp8(sp + "down_proj.weight", rnd(H, MI))

    save_file(tensors, str(out / "model.safetensors"))
    index = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in tensors}}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def roundtrip(work: Path, prompt, n_gen):
    """Build fp8 ckpt -> export_real -> greedy via the export reader."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import export_glm5
    from glm5_ref.load_export import load_export_and_generate

    import shutil

    ckpt = work / "glm5_fp8_ckpt"
    export = work / "glm5_fp8_export"
    build_synthetic_fp8_ckpt(ckpt)
    shutil.rmtree(export, ignore_errors=True)  # fixture: always a fresh export (no stale .done skips)
    export_glm5.export_real(ckpt, export)
    return export, load_export_and_generate(export, prompt, n_gen)
