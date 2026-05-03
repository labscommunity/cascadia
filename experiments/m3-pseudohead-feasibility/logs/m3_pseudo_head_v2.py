"""M3 step 1 v2: REAL pseudo-head test with lm_head.weight + final model.norm.

The first attempt (m3_pseudo_head.py) was wrong on two counts:
  1. Used embed_tokens.weight (input embedding) — a different matrix from
     lm_head.weight. Llama 3.1 8B has untied weights.
  2. Skipped final RMSNorm (model.norm) which is applied before lm_head in the
     full forward.

Correct logit-lens projection at intermediate layer L:
    logits = lm_head( model.norm( hidden_at_L ) )

Per the logit-lens literature (Belrose+ Tuned Lens, LayerSkip):
  - Layer 16/32 ≈ 50% depth: typically 0-10% top-1 agreement on Llama-class
  - Layer 24/32 ≈ 75% depth: typically 30-50% on factual content

Tests both layer 16 (current v3 stage_0) and layer 24 (would need a different
stage cut — but we have v3 22/10 too which gives layer 22 hidden).
"""
import json
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from safetensors import safe_open

SOURCE_DIR = Path(r"C:\cascadia\models\llama-3.1-8b-src")
FULL_MODEL_XML = r"C:\cascadia\models\llama-3.1-8b-int4\openvino_model.xml"

# Stage_0 IRs at different depths
STAGE0_LAYER16 = r"C:\cascadia\shards_2stage_v3\stage_0\openvino_model.xml"
STAGE0_LAYER22 = r"C:\cascadia\shards_2stage_v3_22_10\stage_0\openvino_model.xml"

PROMPT_IDS = [128000, 70869, 279, 1401, 12062, 1990, 36821, 323, 67737, 37416]
N_TOKENS = 32

def load_lm_head_and_norm(source_dir: Path):
    """Load lm_head.weight + model.norm.weight from HF safetensors."""
    print("Loading lm_head.weight + model.norm.weight from safetensors...", flush=True)
    index_path = source_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    def fetch(name):
        path = source_dir / weight_map[name]
        with safe_open(path, framework="pt") as f:
            t = f.get_tensor(name)
        return t

    lm_head = fetch("lm_head.weight")
    norm_w = fetch("model.norm.weight")
    print(f"  lm_head.weight: shape={tuple(lm_head.shape)}, dtype={lm_head.dtype}", flush=True)
    print(f"  model.norm.weight: shape={tuple(norm_w.shape)}, dtype={norm_w.dtype}", flush=True)

    return (lm_head.to(torch.float32).numpy(),
            norm_w.to(torch.float32).numpy())

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Llama RMSNorm: x = weight * x / sqrt(mean(x^2) + eps)"""
    # x: [..., hidden]; weight: [hidden]
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    x_normed = x.astype(np.float32) / np.sqrt(var + eps)
    return (x_normed * weight).astype(np.float32)

def run_test(stage0_xml: str, layer_label: str,
             lm_head: np.ndarray, norm_w: np.ndarray,
             core: ov.Core, full_req):
    print(f"\n{'='*60}", flush=True)
    print(f"Testing pseudo-head at {layer_label} (IR: {stage0_xml})", flush=True)
    print(f"{'='*60}", flush=True)
    stage0 = core.compile_model(stage0_xml, "GPU")
    stage0_req = stage0.create_infer_request()
    print("Stage_0 inputs:", [sorted(p.get_names()) for p in stage0.inputs], flush=True)

    head_dim = 128
    rope_theta = 500000.0
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

    def cos_sin(positions: np.ndarray):
        angles = positions[:, None] * inv_freq[None, :]
        cos_h = np.cos(angles)
        sin_h = np.sin(angles)
        cos = np.concatenate([cos_h, cos_h], axis=-1)[None, :, :].astype(np.float16)
        sin = np.concatenate([sin_h, sin_h], axis=-1)[None, :, :].astype(np.float16)
        return cos, sin

    matches = 0
    total = 0
    matches_top5 = 0  # also track if real_token is in top-5 of pseudo
    print(f"\n{'pos':>4} {'real':>8} {'pseudo':>8} {'real_in_top5_of_pseudo':>22}", flush=True)
    sys.stdout.flush()

    # Prefill
    full_req.reset_state()
    cur_ids = np.array([PROMPT_IDS], dtype=np.int64)
    attn = np.ones_like(cur_ids, dtype=np.int64)
    pos_ids = np.arange(cur_ids.shape[1], dtype=np.int64).reshape(1, -1)
    beam = np.zeros(1, dtype=np.int32)
    full_req.set_input_tensor(0, ov.Tensor(cur_ids))
    full_req.set_input_tensor(1, ov.Tensor(attn))
    full_req.set_input_tensor(2, ov.Tensor(pos_ids))
    full_req.set_input_tensor(3, ov.Tensor(beam))
    full_req.infer()
    real_logits = full_req.get_output_tensor(0).data
    real_tok = int(np.argmax(real_logits[0, -1, :]))

    stage0_req.reset_state()
    cos, sin = cos_sin(np.arange(cur_ids.shape[1], dtype=np.float32))
    stage0_req.set_input_tensor(0, ov.Tensor(cur_ids))
    stage0_req.set_input_tensor(1, ov.Tensor(cos))
    stage0_req.set_input_tensor(2, ov.Tensor(sin))
    stage0_req.infer()
    hidden = stage0_req.get_output_tensor(0).data
    last_hidden = hidden[0, -1, :].astype(np.float32)
    # PROPER pseudo-projection: norm + lm_head
    normed = rms_norm(last_hidden[None, :], norm_w)[0]  # [hidden]
    pseudo_logits = normed @ lm_head.T  # [vocab]
    pseudo_tok = int(np.argmax(pseudo_logits))
    top5 = np.argpartition(pseudo_logits, -5)[-5:]
    in_top5 = real_tok in top5

    match = (real_tok == pseudo_tok)
    if match: matches += 1
    if in_top5: matches_top5 += 1
    total += 1
    print(f"{0:>4} {real_tok:>8} {pseudo_tok:>8} {('Y' if in_top5 else 'n'):>22}", flush=True)

    pos = cur_ids.shape[1]
    for step in range(1, N_TOKENS):
        ids = np.array([[real_tok]], dtype=np.int64)
        attn_step = np.ones((1, pos + 1), dtype=np.int64)
        pos_id = np.array([[pos]], dtype=np.int64)

        full_req.set_input_tensor(0, ov.Tensor(ids))
        full_req.set_input_tensor(1, ov.Tensor(attn_step))
        full_req.set_input_tensor(2, ov.Tensor(pos_id))
        full_req.set_input_tensor(3, ov.Tensor(beam))
        full_req.infer()
        real_logits = full_req.get_output_tensor(0).data
        real_tok_new = int(np.argmax(real_logits[0, -1, :]))

        cos1, sin1 = cos_sin(np.array([pos], dtype=np.float32))
        stage0_req.set_input_tensor(0, ov.Tensor(ids))
        stage0_req.set_input_tensor(1, ov.Tensor(cos1))
        stage0_req.set_input_tensor(2, ov.Tensor(sin1))
        stage0_req.infer()
        hidden = stage0_req.get_output_tensor(0).data
        last_hidden = hidden[0, -1, :].astype(np.float32)
        normed = rms_norm(last_hidden[None, :], norm_w)[0]
        pseudo_logits = normed @ lm_head.T
        pseudo_tok_new = int(np.argmax(pseudo_logits))
        top5 = np.argpartition(pseudo_logits, -5)[-5:]
        in_top5_new = real_tok_new in top5

        match = (real_tok_new == pseudo_tok_new)
        if match: matches += 1
        if in_top5_new: matches_top5 += 1
        total += 1
        print(f"{step:>4} {real_tok_new:>8} {pseudo_tok_new:>8} {('Y' if in_top5_new else 'n'):>22}", flush=True)

        real_tok = real_tok_new
        pos += 1

    pct1 = matches/total*100
    pct5 = matches_top5/total*100
    print(f"\n=== {layer_label} AGREEMENT ===", flush=True)
    print(f"  top-1 (pseudo == real): {matches}/{total} = {pct1:.1f}%", flush=True)
    print(f"  real-in-pseudo's-top-5: {matches_top5}/{total} = {pct5:.1f}%", flush=True)
    return pct1, pct5

def main():
    core = ov.Core()
    print(f"OV available: {core.available_devices}", flush=True)

    print(f"\nCompiling FULL Llama 8B INT4 reference on GPU...", flush=True)
    full = core.compile_model(FULL_MODEL_XML, "GPU")
    full_req = full.create_infer_request()

    lm_head, norm_w = load_lm_head_and_norm(SOURCE_DIR)

    results = {}
    results["layer_16"] = run_test(STAGE0_LAYER16, "layer 16/32 (50% depth)",
                                    lm_head, norm_w, core, full_req)
    results["layer_22"] = run_test(STAGE0_LAYER22, "layer 22/32 (69% depth)",
                                    lm_head, norm_w, core, full_req)

    print(f"\n{'='*60}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for label, (top1, top5) in results.items():
        print(f"  {label}: top-1={top1:.1f}%  top-5={top5:.1f}%", flush=True)

if __name__ == "__main__":
    main()
