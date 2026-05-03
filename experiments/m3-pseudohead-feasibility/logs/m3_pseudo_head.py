"""M3 step 1: feasibility test for the early-exit pseudo-head moonshot.

For Llama 3.1 8B INT4:
- Run stage_0 (16 layers + embed) standalone via OV. Capture intermediate hidden_state at layer 16 output.
- Load embed_tokens.weight from safetensors.
- Project hidden via embed.T to get pseudo_logits.
- Run the FULL Llama 8B (single-stage) and capture real argmax tokens.
- Compute per-position agreement rate: how often does pseudo_token (from layer-16 hidden + embed) match real_token (from full forward)?

If agreement rate is high (>40%), the pseudo-head is a viable speculation source.
If low (<25%), abandon the moonshot.
"""
import sys
import json
import numpy as np
from pathlib import Path

import openvino as ov
from safetensors import safe_open

STAGE0_XML = r"C:\cascadia\shards_2stage_v3\stage_0\openvino_model.xml"
SOURCE_DIR = Path(r"C:\cascadia\models\llama-3.1-8b-src")
FULL_MODEL_XML = r"C:\cascadia\models\llama-3.1-8b-int4\openvino_model.xml"

PROMPT = "Explain the key differences between RISC and CISC processors. Be specific."
N_TOKENS = 32  # generate this many; compare per-position agreement

def load_embed_matrix(source_dir: Path) -> np.ndarray:
    """Load via torch (which handles bf16) and convert to f16 numpy."""
    print("Loading embed_tokens.weight from safetensors via torch...", flush=True)
    import torch
    index_path = source_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    embed_file = index["weight_map"]["model.embed_tokens.weight"]
    full_path = source_dir / embed_file
    with safe_open(full_path, framework="pt") as f:
        weight = f.get_tensor("model.embed_tokens.weight")  # torch tensor, bf16 likely
    print(f"  embed_tokens.weight: shape={tuple(weight.shape)}, dtype={weight.dtype}", flush=True)
    return weight.to(torch.float16).numpy()

def main():
    core = ov.Core()
    print(f"OV available devices: {core.available_devices}", flush=True)

    # 1. Load full Llama 8B for real_token reference
    print(f"\nCompiling FULL model on GPU: {FULL_MODEL_XML}", flush=True)
    full = core.compile_model(FULL_MODEL_XML, "GPU")
    full_req = full.create_infer_request()

    # 2. Load stage_0 IR for hidden_state extraction
    print(f"\nCompiling STAGE_0 IR on GPU: {STAGE0_XML}", flush=True)
    stage0 = core.compile_model(STAGE0_XML, "GPU")
    stage0_req = stage0.create_infer_request()
    print("Stage_0 inputs:", [sorted(p.get_names()) for p in stage0.inputs], flush=True)
    print("Stage_0 outputs:", [sorted(p.get_names()) for p in stage0.outputs], flush=True)

    # 3. Load embed_tokens.weight
    embed = load_embed_matrix(SOURCE_DIR)
    print(f"  embed shape: {embed.shape}, dtype: {embed.dtype}", flush=True)
    embed_T = embed.T.astype(np.float16)  # [hidden, vocab]

    # 4. Tokenize prompt (just use the prompt as bytes for simplicity — proper tokenizer would be better)
    # We'll use a simple test by feeding integers — picking actual token ids requires a tokenizer
    # Use a fixed prompt token list for reproducibility
    prompt_ids = [128000, 70869, 279, 1401, 12062, 1990, 36821, 323, 67737, 37416]
    print(f"Prompt token ids: {prompt_ids}", flush=True)

    # 5. Run iterative generation: at each step, compare pseudo vs real
    matches = 0
    total = 0
    print(f"\n{'pos':>4} {'real_tok':>8} {'pseudo_tok':>10} {'match':>6}", flush=True)
    sys.stdout.flush()

    # Initial prefill of full model — run once with the whole prompt
    full_req.reset_state()
    cur_ids = np.array([prompt_ids], dtype=np.int64)
    attn = np.ones_like(cur_ids, dtype=np.int64)
    pos_ids = np.arange(cur_ids.shape[1], dtype=np.int64).reshape(1, -1)
    beam = np.zeros(1, dtype=np.int32)
    full_req.set_input_tensor(0, ov.Tensor(cur_ids))
    full_req.set_input_tensor(1, ov.Tensor(attn))
    full_req.set_input_tensor(2, ov.Tensor(pos_ids))
    full_req.set_input_tensor(3, ov.Tensor(beam))
    full_req.infer()
    real_logits = full_req.get_output_tensor(0).data  # [1, seq, vocab]
    real_tok = int(np.argmax(real_logits[0, -1, :]))

    # Same prefill on stage_0 — v3 shards expect (input_ids, cos, sin).
    # Compute Llama 3.1 rotary cos/sin locally.
    head_dim = 128
    rope_theta = 500000.0
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos_arr = np.arange(cur_ids.shape[1], dtype=np.float32)
    angles = pos_arr[:, None] * inv_freq[None, :]  # [seq, head_dim/2]
    cos_half = np.cos(angles)
    sin_half = np.sin(angles)
    cos = np.concatenate([cos_half, cos_half], axis=-1)[None, :, :].astype(np.float16)  # [1, seq, head_dim]
    sin = np.concatenate([sin_half, sin_half], axis=-1)[None, :, :].astype(np.float16)
    stage0_req.reset_state()
    stage0_req.set_input_tensor(0, ov.Tensor(cur_ids))
    stage0_req.set_input_tensor(1, ov.Tensor(cos))
    stage0_req.set_input_tensor(2, ov.Tensor(sin))
    stage0_req.infer()
    hidden = stage0_req.get_output_tensor(0).data  # [1, seq, hidden]
    last_hidden = hidden[0, -1, :].astype(np.float16)
    pseudo_logits = last_hidden @ embed_T  # [vocab]
    pseudo_tok = int(np.argmax(pseudo_logits))

    match = (real_tok == pseudo_tok)
    if match: matches += 1
    total += 1
    print(f"{0:>4} {real_tok:>8} {pseudo_tok:>10} {('Y' if match else 'n'):>6}", flush=True)

    # Decode loop: feed real_tok to both, compare next prediction
    pos = cur_ids.shape[1]
    for step in range(1, N_TOKENS):
        ids = np.array([[real_tok]], dtype=np.int64)
        attn_step = np.ones((1, pos + 1), dtype=np.int64)
        pos_id = np.array([[pos]], dtype=np.int64)

        # Full
        full_req.set_input_tensor(0, ov.Tensor(ids))
        full_req.set_input_tensor(1, ov.Tensor(attn_step))
        full_req.set_input_tensor(2, ov.Tensor(pos_id))
        full_req.set_input_tensor(3, ov.Tensor(beam))
        full_req.infer()
        real_logits = full_req.get_output_tensor(0).data
        real_tok_new = int(np.argmax(real_logits[0, -1, :]))

        # Stage_0 with v3 (cos, sin) — single position
        pos_arr = np.array([pos], dtype=np.float32)
        angles = pos_arr[:, None] * inv_freq[None, :]
        cos1 = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)[None, :, :].astype(np.float16)
        sin1 = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)[None, :, :].astype(np.float16)
        stage0_req.set_input_tensor(0, ov.Tensor(ids))
        stage0_req.set_input_tensor(1, ov.Tensor(cos1))
        stage0_req.set_input_tensor(2, ov.Tensor(sin1))
        stage0_req.infer()
        hidden = stage0_req.get_output_tensor(0).data
        last_hidden = hidden[0, -1, :].astype(np.float16)
        pseudo_logits = last_hidden @ embed_T
        pseudo_tok_new = int(np.argmax(pseudo_logits))

        match = (real_tok_new == pseudo_tok_new)
        if match: matches += 1
        total += 1
        print(f"{step:>4} {real_tok_new:>8} {pseudo_tok_new:>10} {('Y' if match else 'n'):>6}", flush=True)

        real_tok = real_tok_new
        pos += 1

    print(f"\n=== AGREEMENT RATE ===", flush=True)
    print(f"matches: {matches}/{total} = {matches/total*100:.1f}%", flush=True)

if __name__ == "__main__":
    main()
