"""Deterministic tiny synthetic K3 model — structure-faithful, small dims.

Keeps every structural feature that can go wrong (hybrid KDA/MLA layer mix,
AttnRes block boundaries, LatentMoE, shared experts, dense layer 0) while
staying small enough to run on CPU in a second.
"""
import torch


def tiny_cfg():
    return {
        "hidden_size": 64,
        "num_hidden_layers": 6,
        "first_k_dense_replace": 1,
        "attn_res_block_size": 2,          # boundaries at layers 0, 2, 4
        "kda_layers": {1, 2, 4, 5},        # layers 0, 3 fall through to MLA
        "rms_norm_eps": 1e-5,
        "situ_beta": 4.0,
        "situ_linear_beta": 25.0,
        # attention
        "num_heads": 4,
        "head_dim": 16,
        "conv_size": 4,
        "gate_lower_bound": -5.0,
        "q_lora_rank": 24,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        # moe
        "num_experts": 8,
        "top_k": 2,
        "num_shared_experts": 2,
        # both must be multiples of the fp4 group (32), as in the real config
        "routed_expert_hidden_size": 32,
        "moe_intermediate_size": 32,
        "routed_scaling_factor": 1.0,
        "moe_renormalize": True,
        "latent_moe_use_norm": True,
        "intermediate_size": 48,           # dense layer 0
        "vocab_size": 33,
    }


def _r(*shape, scale=0.05):
    return torch.randn(*shape) * scale


def tiny_weights(cfg, seed=0):
    torch.manual_seed(seed)
    h = cfg["hidden_size"]
    nh, hd = cfg["num_heads"], cfg["head_dim"]
    proj = nh * hd
    lat = cfg["routed_expert_hidden_size"]
    mi = cfg["moe_intermediate_size"]
    qh = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]

    layers = []
    for i in range(cfg["num_hidden_layers"]):
        is_kda = i in cfg["kda_layers"]
        if is_kda:
            attn = {
                "q_proj": _r(proj, h), "k_proj": _r(proj, h), "v_proj": _r(proj, h),
                "q_conv1d": _r(proj, cfg["conv_size"]),
                "k_conv1d": _r(proj, cfg["conv_size"]),
                "v_conv1d": _r(proj, cfg["conv_size"]),
                "f_a_proj": _r(hd, h), "f_b_proj": _r(proj, hd),
                "A_log": torch.rand(nh).uniform_(1, 16).log(),
                "dt_bias": _r(proj),
                "b_proj": _r(nh, h),
                "g_proj": _r(proj, h),
                "o_norm": torch.ones(hd) + _r(hd),
                "o_proj": _r(h, proj),
            }
        else:
            attn = {
                "q_a_proj": _r(cfg["q_lora_rank"], h),
                "q_a_layernorm": torch.ones(cfg["q_lora_rank"]) + _r(cfg["q_lora_rank"]),
                "q_b_proj": _r(nh * qh, cfg["q_lora_rank"]),
                "kv_a_proj_with_mqa": _r(cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"], h),
                "kv_a_layernorm": torch.ones(cfg["kv_lora_rank"]) + _r(cfg["kv_lora_rank"]),
                "kv_b_proj": _r(nh * (cfg["qk_nope_head_dim"] + cfg["v_head_dim"]),
                                cfg["kv_lora_rank"]),
                "g_proj": _r(nh * cfg["v_head_dim"], h),
                "o_proj": _r(h, nh * cfg["v_head_dim"]),
            }

        lw = {
            "attn": attn,
            "input_layernorm": torch.ones(h) + _r(h),
            "post_attention_layernorm": torch.ones(h) + _r(h),
            "attn_res_proj": _r(1, h), "attn_res_norm": torch.ones(h) + _r(h),
            "mlp_res_proj": _r(1, h), "mlp_res_norm": torch.ones(h) + _r(h),
        }
        if i >= cfg["first_k_dense_replace"]:
            lw["moe"] = {
                "gate_weight": _r(cfg["num_experts"], h),
                "e_score_correction_bias": _r(cfg["num_experts"]),
                "routed_expert_down_proj": _r(lat, h),
                "routed_expert_up_proj": _r(h, lat),
                "routed_expert_norm": torch.ones(lat) + _r(lat),
                "experts": [
                    {"w1": _r(mi, lat), "w3": _r(mi, lat), "w2": _r(lat, mi)}
                    for _ in range(cfg["num_experts"])
                ],
                "shared_w1": _r(mi * cfg["num_shared_experts"], h),
                "shared_w3": _r(mi * cfg["num_shared_experts"], h),
                "shared_w2": _r(h, mi * cfg["num_shared_experts"]),
            }
        else:
            ii = cfg["intermediate_size"]
            lw |= {"w1": _r(ii, h), "w3": _r(ii, h), "w2": _r(h, ii)}
        layers.append(lw)

    return {
        "embed": _r(cfg["vocab_size"], h, scale=0.5),
        "layers": layers,
        "output_attn_res_proj": _r(1, h),
        "output_attn_res_norm": torch.ones(h) + _r(h),
        "norm": torch.ones(h) + _r(h),
        "lm_head": _r(cfg["vocab_size"], h),
    }
