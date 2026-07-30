#!/usr/bin/env python3
"""Exercise `export_kimi_k3.export_real` against a synthetic checkpoint.

The real-checkpoint path is ~100 lines that only ever run against 1.5 TB on a
remote host for hours at a time — the worst possible place to discover a bug.
This builds a tiny checkpoint in the REAL HuggingFace layout (same tensor names,
same dtypes, same `model.safetensors.index.json`) and runs the actual export
over it, then checks the parts that are easy to get wrong:

  * full export produces the expected tree
  * a killed run resumes and lands byte-identical to a clean one
  * `--expert-roots` places bins off-volume and symlinks them back
  * `--free-source-shards` deletes only shards no later layer reads

`--with-loader` adds the check that closes the loop: it runs the exported tree
through the Rust `k3_run` example, so a drift between what the exporter writes
and what the loader expects fails here rather than on the remote host.

Run:  python tools/kimi_k3_ref/selftest_export_real.py [--with-loader]
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import export_kimi_k3 as X  # noqa: E402

# Tiny, but every structural feature the exporter branches on.
H, NL, NE = 64, 6, 8
LAT, MI, HD, NH = 32, 32, 16, 4
QL, KVL, NOPE, ROPE, VH = 24, 16, 16, 8, 16
DENSE_INTER = 48
KDA = [1, 2, 4, 5]  # 0-indexed; the config lists these 1-indexed


def tiny_config() -> dict:
    return {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "model_type": "kimi_k3",
        "text_config": {
            "model_type": "kimi_linear",
            "hidden_size": H,
            "num_hidden_layers": NL,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "intermediate_size": DENSE_INTER,
            "vocab_size": 33,
            "rms_norm_eps": 1e-5,
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "attn_res_block_size": 2,
            "num_experts": NE,
            "num_experts_per_token": 2,
            "num_shared_experts": 2,
            "moe_intermediate_size": MI,
            "routed_expert_hidden_size": LAT,
            "latent_moe_use_norm": True,
            "moe_router_activation_func": "sigmoid",
            "moe_renormalize": True,
            "routed_scaling_factor": 1.0,
            "num_expert_group": 1,
            "topk_group": 1,
            "num_nextn_predict_layers": 0,
            "num_attention_heads": NH,
            "q_lora_rank": QL,
            "kv_lora_rank": KVL,
            "qk_nope_head_dim": NOPE,
            "qk_rope_head_dim": ROPE,
            "v_head_dim": VH,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "eos_token_id": 7,
            # 1-INDEXED, as the real config is
            "linear_attn_config": {
                "num_heads": NH,
                "head_dim": HD,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": True,
                "kda_layers": [i + 1 for i in KDA],
                "full_attn_layers": [i + 1 for i in range(NL) if i not in KDA],
            },
        },
        "quantization_config": {
            "format": "mxfp4-pack-quantized",
            "config_groups": {
                "group_0": {"weights": {"group_size": 32, "num_bits": 4}}
            },
        },
    }


def build_checkpoint(root: Path, shards: int = 4) -> None:
    """Write a synthetic checkpoint with the REAL tensor names and dtypes."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(tiny_config()))
    X.load_and_validate_config(root / "config.json")  # config must parse

    rng = np.random.default_rng(0)
    bf = lambda *s: torch.tensor(rng.standard_normal(s) * 0.05, dtype=torch.bfloat16)
    f32 = lambda *s: torch.tensor(rng.standard_normal(s), dtype=torch.float32)
    u8 = lambda *s: torch.tensor(rng.integers(0, 256, s), dtype=torch.uint8)

    T: dict = {}
    P = X.PREFIX
    T[P + "embed_tokens.weight"] = bf(33, H)
    T[P + "norm.weight"] = f32(H)
    T[X.LM_HEAD] = bf(33, H)
    T[P + "output_attn_res_proj.weight"] = bf(1, H)
    T[P + "output_attn_res_norm.weight"] = f32(H)

    for li in range(NL):
        b = f"{P}layers.{li}."
        T[b + "input_layernorm.weight"] = f32(H)
        T[b + "post_attention_layernorm.weight"] = f32(H)
        T[b + "self_attention_res_proj.weight"] = bf(1, H)
        T[b + "self_attention_res_norm.weight"] = f32(H)
        T[b + "mlp_res_proj.weight"] = bf(1, H)
        T[b + "mlp_res_norm.weight"] = f32(H)
        T[b + "self_attn.g_proj.weight"] = bf(NH * (HD if li in KDA else VH), H)
        T[b + "self_attn.o_proj.weight"] = bf(H, NH * (HD if li in KDA else VH))
        if li in KDA:
            for n in ("q_proj", "k_proj", "v_proj"):
                T[b + f"self_attn.{n}.weight"] = bf(NH * HD, H)
            for n in ("q_conv1d", "k_conv1d", "v_conv1d"):
                T[b + f"self_attn.{n}.weight"] = bf(NH * HD, 1, 4)  # depthwise [D,1,K]
            T[b + "self_attn.f_a_proj.weight"] = bf(HD, H)
            T[b + "self_attn.f_b_proj.weight"] = bf(NH * HD, HD)
            T[b + "self_attn.b_proj.weight"] = bf(NH, H)
            T[b + "self_attn.o_norm.weight"] = f32(HD)
            # zero-padded to head_dim, exactly as the release ships it
            a = torch.zeros(HD, dtype=torch.float32)
            a[:NH] = torch.rand(NH) * 2.0
            T[b + "self_attn.A_log"] = a
            T[b + "self_attn.dt_bias"] = f32(NH * HD)
        else:
            T[b + "self_attn.q_a_proj.weight"] = bf(QL, H)
            T[b + "self_attn.q_a_layernorm.weight"] = f32(QL)
            T[b + "self_attn.q_b_proj.weight"] = bf(NH * (NOPE + ROPE), QL)
            T[b + "self_attn.kv_a_proj_with_mqa.weight"] = bf(KVL + ROPE, H)
            T[b + "self_attn.kv_a_layernorm.weight"] = f32(KVL)
            T[b + "self_attn.kv_b_proj.weight"] = bf(NH * (NOPE + VH), KVL)

        if li >= 1:
            m = b + "block_sparse_moe."
            T[m + "gate.weight"] = bf(NE, H)
            T[m + "gate.e_score_correction_bias"] = f32(NE)
            T[m + "routed_expert_down_proj.weight"] = bf(LAT, H)
            T[m + "routed_expert_up_proj.weight"] = bf(H, LAT)
            T[m + "routed_expert_norm.weight"] = f32(LAT)
            for n in ("gate_proj", "up_proj"):
                T[m + f"shared_experts.{n}.weight"] = bf(MI * 2, H)
            T[m + "shared_experts.down_proj.weight"] = bf(H, MI * 2)
            for e in range(NE):
                for w, (o, i) in (("w1", (MI, LAT)), ("w3", (MI, LAT)), ("w2", (LAT, MI))):
                    T[f"{m}experts.{e}.{w}.weight_packed"] = u8(o, i // 2)
                    T[f"{m}experts.{e}.{w}.weight_scale"] = u8(o, i // 32)
        else:
            for n in ("gate_proj", "up_proj"):
                T[b + f"mlp.{n}.weight"] = bf(DENSE_INTER, H)
            T[b + "mlp.down_proj.weight"] = bf(H, DENSE_INTER)

    # a couple of vision tensors, in their own trailing shard, as the release has
    vis = {
        "vision_tower.patch_embed.proj.weight": bf(8, 8),
        "mm_projector.proj.0.weight": bf(8, 8),
    }

    names = list(T)
    per = (len(names) + shards - 1) // shards
    wmap, total = {}, 0
    for si in range(shards):
        chunk = {n: T[n] for n in names[si * per:(si + 1) * per]}
        if not chunk:
            continue
        f = f"model-{si:05d}-of-{shards + 1:05d}.safetensors"
        save_file(chunk, str(root / f))
        for n in chunk:
            wmap[n] = f
        total += sum(v.numel() * v.element_size() for v in chunk.values())
    fv = f"model-{shards:05d}-of-{shards + 1:05d}.safetensors"
    save_file(vis, str(root / fv))
    for n in vis:
        wmap[n] = fv
    total += sum(v.numel() * v.element_size() for v in vis.values())
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": wmap})
    )


def tree_digest(out: Path) -> dict:
    """Content hash of every produced file, following symlinks."""
    import hashlib

    d = {}
    for p in sorted(out.rglob("*")):
        if p.is_dir() or p.name.startswith("."):
            continue
        real = p.resolve()
        d[str(p.relative_to(out))] = hashlib.md5(real.read_bytes()).hexdigest()
    return d


def check(name, ok, extra=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{extra}")
    return ok


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="k3_export_real_"))
    results = []
    try:
        ck = tmp / "ckpt"
        build_checkpoint(ck)
        cfg = X.load_and_validate_config(ck / "config.json")
        print(f"synthetic checkpoint: {len(list(ck.glob('*.safetensors')))} shards")

        # 1. clean export
        a = tmp / "out_clean"
        X.export_real(ck, a, cfg)
        da = tree_digest(a)
        results.append(check("clean export produced files", len(da) > 0, f": {len(da)}"))
        results.append(check(
            "expected tree",
            (a / "manifest.json").exists()
            and (a / "embed.safetensors").exists()
            and (a / "head.safetensors").exists()
            and len(list((a / "shells").glob("*.safetensors"))) == NL
            and len(list((a / "experts").glob("*.bin"))) == NL - 1,
        ))
        results.append(check(
            "markers written", len(list(a.glob(".layer_*.done"))) == NL))

        # 2. killed run -> resume, must equal the clean export
        b = tmp / "out_resume"
        X.export_real(ck, b, cfg)
        for li in (NL - 2, NL - 1):
            (b / f".layer_{li:02d}.done").unlink()
            for f in (b / "experts" / f"layer_{li:02d}.bin",
                      b / "shells" / f"layer_{li:02d}.safetensors"):
                if f.exists():
                    f.unlink()
        # a stray .part, as a killed run would leave
        (b / "experts" / f"layer_{NL - 1:02d}.bin.part").write_bytes(b"garbage")
        X.export_real(ck, b, cfg)
        results.append(check("resume == clean export", tree_digest(b) == da))
        results.append(check(
            "stray .part cleaned",
            not list((b / "experts").glob("*.part"))))

        # 3. --expert-roots: bins live elsewhere, symlinked back, same content
        c = tmp / "out_roots"
        r1, r2 = tmp / "r1", tmp / "r2"
        X.export_real(ck, c, cfg, expert_roots=[(r1, 2), (r2, None)])
        links = sorted((c / "experts").glob("*.bin"))
        results.append(check(
            "expert bins are symlinks",
            all(p.is_symlink() for p in links), f": {len(links)}"))
        results.append(check(
            "bins split across roots",
            len(list(r1.glob('*.bin'))) == 2 and len(list(r2.glob('*.bin'))) == NL - 3,
            f": r1={len(list(r1.glob('*.bin')))} r2={len(list(r2.glob('*.bin')))}"))
        results.append(check("split export == clean export", tree_digest(c) == da))

        # 4. --free-source-shards: source consumed, output still correct
        ck2 = tmp / "ckpt2"
        shutil.copytree(ck, ck2)
        d = tmp / "out_freed"
        X.export_real(ck2, d, cfg, free_source=True)
        left = list(ck2.glob("*.safetensors"))
        results.append(check(
            "source shards freed", len(left) == 0, f": {len(left)} left"))
        results.append(check("freed-run export == clean export", tree_digest(d) == da))

        # 5. the export actually loads and generates in the engine
        if "--with-loader" in sys.argv:
            import subprocess

            r = subprocess.run(
                ["cargo", "run", "--release", "-q", "--example", "k3_run",
                 "--", str(a), "--ids", "1 2 3", "4"],
                cwd=HERE.parents[1], capture_output=True, text=True)
            ok = r.returncode == 0 and "[k3_run] out ids:" in r.stdout
            results.append(check("k3_run loads and generates", ok))
            if not ok:
                print(r.stdout[-2000:] + r.stderr[-2000:])

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if all(results):
        print(f"PASS — {len(results)}/{len(results)} checks")
        return 0
    print(f"FAIL — {sum(results)}/{len(results)} passed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
