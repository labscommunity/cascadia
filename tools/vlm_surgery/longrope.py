#!/usr/bin/env python3
"""Give a Phi-3 language model IR its long-RoPE switch back.

Phi-3's `su` / `longrope` RoPE (Phi-3.5-vision, and InternVL2-4B's Phi-3 tower)
picks its frequencies per forward pass in Hugging Face: the short factors while
the sequence is at most `original_max_position_embeddings` (4096) long, the
long factors past it (`seq_len = max(position_ids) + 1`). optimum-intel traces
the rotary block with the short factors folded into one constant
(`inv_freq [1, rot/2, 1]`), so an export (and every tree cut from it, and
ov-genai, which runs the same IR) keeps the short factors at any length and
differs from Hugging Face past 4096 positions. The attention factor
(`sqrt(1 + ln(max / original) / ln(original))`, 1.19 on these models) is
traced as a constant multiply on cos and sin, as Hugging Face applies it at
every length, so it needs no change.

`add_long_rope_switch` finds that constant by value (the short-factor
frequencies computed from config.json) and feeds its consumers
`Select(max(position_ids) >= original, long, <the constant>)`: one decision
per forward pass, as Hugging Face makes it. The short branch is the export's
own constant, so below the window the model computes exactly what it did.
A prompt longer than 4096 prefilled in one inference takes the long factors
throughout, and a decode that crosses 4096 keeps its cached keys and rotates the
new tokens with the long factors, as Hugging Face's cache does. A prompt
prefilled in chunks (ov-dist-spec, static trees) is not covered: the chunks
before 4096 keep the short factors, which a per-request decision fed by the host
would fix.

The decision is per pass over every row of `position_ids`, so a packed tree
(several requests side by side in one sequence) must not let any slot reach
`original`; `packed_variant.py` refuses such a tree.

As a tool, it writes a copy of an OpenVINO VLM export with the switch in its
`openvino_language_model.xml`, for ov-genai:

    python longrope.py --model <export dir> --output-dir <new export dir>

Other files are hardlinked when the filesystem allows it, else copied.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import openvino as ov
from openvino import opset13 as ops

LM_XML = "openvino_language_model.xml"
LONGROPE_TYPES = ("su", "longrope")


def text_config(cfg: dict) -> dict:
    """The language model's config: InternVL nests it as `llm_config`."""
    for key in ("llm_config", "text_config"):
        if isinstance(cfg.get(key), dict):
            return cfg[key]
    return cfg


def long_rope(text: dict) -> tuple[np.ndarray, np.ndarray, int] | None:
    """(short inv_freq, long inv_freq, original window) for a `su` / `longrope`
    config, as Hugging Face computes them; None for any other RoPE."""
    rs = text.get("rope_scaling") or {}
    if (rs.get("type") or rs.get("rope_type")) not in LONGROPE_TYPES:
        return None
    original = text.get("original_max_position_embeddings") or rs.get(
        "original_max_position_embeddings")
    if not original:
        raise ValueError("longrope config states no original_max_position_embeddings")
    head_dim = text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"]
    rot = int(head_dim * text.get("partial_rotary_factor", 1.0))
    # In float32 throughout, as Hugging Face computes inv_freq in torch.
    base = np.float32(text.get("rope_theta", 10000.0))
    expo = np.arange(0, rot, 2).astype(np.float32) / np.float32(rot)
    inv = []
    for key in ("short_factor", "long_factor"):
        factor = np.asarray(rs[key], dtype=np.float32)
        if factor.shape != expo.shape:
            raise ValueError(f"rope_scaling.{key} has {factor.size} entries, expected {expo.size}")
        inv.append(np.float32(1.0) / (factor * np.power(base, expo, dtype=np.float32)))
    return inv[0], inv[1], int(original)


def add_long_rope_switch(model: ov.Model, text: dict) -> int:
    """Replace each short-factor inv_freq constant in `model` with the
    position-dependent switch. Returns how many were replaced (0 when the
    config is not longrope). Raises when a longrope config's constant is not
    found, or the model takes no `position_ids`."""
    rope = long_rope(text)
    if rope is None:
        return 0
    short, long, original = rope
    pos = [p for p in model.get_parameters() if p.get_friendly_name() == "position_ids"]
    if not pos:
        raise ValueError("model takes no position_ids; cannot switch RoPE factors")
    pos = pos[0]
    if pos.get_partial_shape().rank.get_length() != 2:
        raise ValueError(f"position_ids {pos.get_partial_shape()} is not [batch, seq]")
    targets = []
    for op in model.get_ordered_ops():
        if op.get_type_name() != "Constant" or op.get_element_type() != ov.Type.f32:
            continue
        data = op.get_data()
        if data.size == short.size and np.allclose(data.ravel(), short, rtol=1e-5, atol=0):
            targets.append(op)
    if not targets:
        raise ValueError(
            "no short-factor inv_freq constant found; this export's rotary block is not "
            "the one optimum traces for Phi-3")
    as_i64 = ops.convert(pos, ov.Type.i64)
    last = ops.reduce_max(as_i64, ops.constant(np.array([0, 1], dtype=np.int64)), keep_dims=False)
    is_long = ops.greater_equal(last, ops.constant(np.int64(original)))
    for op in targets:
        shape = list(op.get_output_shape(0))
        consumers = list(op.output(0).get_target_inputs())
        switch = ops.select(
            is_long,
            ops.constant(long.reshape(shape)),
            op.output(0),
            name=f"{op.get_friendly_name()}/long_rope_switch",
        )
        for target in consumers:
            target.replace_source_output(switch.output(0))
    model.validate_nodes_and_infer_types()
    return len(targets)


def patch_export(model_dir: str, out_dir: str) -> int:
    with open(os.path.join(model_dir, "config.json")) as f:
        text = text_config(json.load(f))
    if long_rope(text) is None:
        raise SystemExit(f"{model_dir}: rope_scaling is not su/longrope; nothing to switch")
    if os.path.abspath(model_dir) == os.path.abspath(out_dir):
        raise SystemExit("--output-dir must differ from --model")
    os.makedirs(out_dir, exist_ok=True)
    model = ov.Core().read_model(os.path.join(model_dir, LM_XML))
    n = add_long_rope_switch(model, text)
    for name in os.listdir(model_dir):
        src = os.path.join(model_dir, name)
        dst = os.path.join(out_dir, name)
        if name in (LM_XML, LM_XML.replace(".xml", ".bin")) or not os.path.isfile(src):
            continue
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    lm_out = os.path.join(out_dir, LM_XML)
    for fn in (lm_out, lm_out.replace(".xml", ".bin")):
        if os.path.exists(fn):
            os.remove(fn)
    ov.save_model(model, lm_out, compress_to_fp16=False)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="OpenVINO VLM export directory")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    n = patch_export(args.model, args.output_dir)
    print(f"LONGROPE DONE: {n} switch(es) -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
