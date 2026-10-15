#!/usr/bin/env python3
"""Reference fixtures for the Rust Qwen3.5 vision preprocessing and M-RoPE.

Runs Hugging Face's own Qwen3.5 processor and `get_rope_index` (no model
weights) and writes what the Rust port must reproduce:

  <out>/<name>.png         deterministic synthetic inputs (lossless)
  <out>/reference.json     per image: processor output (fast and slow), and
                           per prompt: input_ids, grids, 3-row position ids,
                           rope delta

Pixel tensors are large, so each is summarised (shape, sum, abs-sum, sum of
squares, first/last 64 values, every 1009th value) rather than stored.

Usage (needs torch, torchvision, transformers==5.14.1, pillow, numpy):
  python gen_qwen35_fixtures.py --model <Qwen3.5/3.6 export dir> --out <dir>

The model dir only supplies config.json, the processor configs and the
tokenizer; the weights are not read.
"""
import argparse
import hashlib
import json
import os
import types

import numpy as np
import torch
import torchvision
import transformers
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qm


def synth(w, h, seed):
    """A deterministic image with gradients, edges and flat regions."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (x * 255.0 / max(w - 1, 1))
    g = (y * 255.0 / max(h - 1, 1))
    b = ((x + 2 * y + seed * 37) % 256)
    img = np.stack([r, g, b], axis=-1)
    # A few hard-edged blocks so resampling across edges is exercised.
    rng = np.random.default_rng(seed)
    for _ in range(6):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1, y1 = min(w, x0 + rng.integers(1, max(2, w // 3))), min(h, y0 + rng.integers(1, max(2, h // 3)))
        img[y0:y1, x0:x1] = rng.integers(0, 256, size=3)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


IMAGES = {
    "square_448": (448, 448, 1),
    "landscape_640x360": (640, 360, 2),
    "portrait_300x1000": (300, 1000, 3),
    "tiny_40x30": (40, 30, 4),
    "odd_1001x777": (1001, 777, 5),
}


def summary(t):
    a = t.detach().to(torch.float64).flatten().numpy()
    return {
        "shape": list(t.shape),
        "sum": float(a.sum()),
        "abs_sum": float(np.abs(a).sum()),
        "sq_sum": float((a * a).sum()),
        "head": [float(v) for v in a[:64]],
        "tail": [float(v) for v in a[-64:]],
        "stride_1009": [float(v) for v in a[::1009]],
    }


def fake_model(config):
    """Just enough of Qwen3_5MoeModel for get_rope_index: config and helper."""
    ns = types.SimpleNamespace(config=config)
    ns.get_vision_position_ids = types.MethodType(qm.Qwen3_5MoeModel.get_vision_position_ids, ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model)
    processor = AutoProcessor.from_pretrained(args.model)
    fast = AutoImageProcessor.from_pretrained(args.model, backend="torchvision")
    slow = AutoImageProcessor.from_pretrained(args.model, backend="pil")
    model = fake_model(config)
    image_token = config.image_token_id
    video_token = config.video_token_id

    ref = {
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
        "processor": {k: list(v) if isinstance(v, tuple) else v for k, v in fast.to_dict().items()
                      if isinstance(v, (int, float, str, list, tuple, dict, bool))},
        "token_ids": {"image": image_token, "video": video_token,
                      "vision_start": config.vision_start_token_id,
                      "vision_end": getattr(config, "vision_end_token_id", None)},
        "spatial_merge_size": config.vision_config.spatial_merge_size,
        "images": {}, "prompts": {},
    }

    imgs = {}
    for name, (w, h, seed) in IMAGES.items():
        img = synth(w, h, seed)
        path = os.path.join(args.out, f"{name}.png")
        img.save(path)
        img = Image.open(path).convert("RGB")  # exactly what the Rust test decodes
        imgs[name] = img
        entry = {"size": [w, h], "png_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
        outs = {}
        for kind, proc in (("fast", fast), ("slow", slow)):
            o = proc(images=[img], return_tensors="pt")
            outs[kind] = o["pixel_values"]
            entry[kind] = {"grid_thw": o["image_grid_thw"].tolist()[0], "pixel_values": summary(o["pixel_values"])}
        d = (outs["fast"].double() - outs["slow"].double()).abs()
        entry["fast_vs_slow"] = {"max_abs": float(d.max()), "mean_abs": float(d.mean())}
        ref["images"][name] = entry

    frames = np.stack([np.asarray(synth(128, 96, 10 + i)) for i in range(4)])
    prompts = {
        "single": [{"role": "user", "content": [
            {"type": "image", "image": imgs["square_448"]}, {"type": "text", "text": "Describe."}]}],
        "two_images": [{"role": "user", "content": [
            {"type": "text", "text": "A:"}, {"type": "image", "image": imgs["landscape_640x360"]},
            {"type": "text", "text": "B:"}, {"type": "image", "image": imgs["portrait_300x1000"]},
            {"type": "text", "text": "Compare."}]}],
        "multi_turn": [
            {"role": "user", "content": [{"type": "image", "image": imgs["tiny_40x30"]},
                                         {"type": "text", "text": "What?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "A dot."}]},
            {"role": "user", "content": [{"type": "image", "image": imgs["odd_1001x777"]},
                                         {"type": "text", "text": "And this?"}]}],
        "video": [{"role": "user", "content": [
            {"type": "video", "video": frames}, {"type": "text", "text": "What happens?"}]}],
        "text_only": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    }
    for name, messages in prompts.items():
        try:
            kw = {}
            if name == "video":
                kw = {"video_metadata": [{"fps": 2.0, "total_num_frames": len(frames), "frames_indices": list(range(len(frames)))}],
                      "do_sample_frames": False}
            o = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                              add_generation_prompt=True, return_tensors="pt", **kw)
        except Exception as e:  # recorded, not fatal: the other prompts still count
            ref["prompts"][name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{name}] processor failed: {e}")
            continue
        ids = o["input_ids"]
        mm = torch.zeros_like(ids, dtype=torch.int32)
        mm[ids == image_token] = 1
        mm[ids == video_token] = 2
        if "mm_token_type_ids" in o:
            assert torch.equal(o["mm_token_type_ids"].to(torch.int32), mm), "mm_token_type_ids disagree"
        igrid = o.get("image_grid_thw")
        vgrid = o.get("video_grid_thw")
        pos, delta = qm.Qwen3_5MoeModel.get_rope_index(
            model, ids, mm, image_grid_thw=igrid,
            video_grid_thw=vgrid.clone() if vgrid is not None else None,
            attention_mask=o.get("attention_mask"))
        entry = {
            "input_ids": ids[0].tolist(),
            "image_grid_thw": igrid.tolist() if igrid is not None else [],
            "video_grid_thw": vgrid.tolist() if vgrid is not None else [],
            "position_ids": pos[:, 0, :].tolist(),
            "rope_delta": int(delta.flatten()[0]),
        }
        if "pixel_values_videos" in o:
            entry["pixel_values_videos"] = summary(o["pixel_values_videos"])
        ref["prompts"][name] = entry
        print(f"[{name}] tokens={ids.shape[1]} images={entry['image_grid_thw']} videos={entry['video_grid_thw']} rope_delta={entry['rope_delta']}")

    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump(ref, f, separators=(",", ":"))
    for name, e in ref["images"].items():
        print(f"{name}: grid fast={e['fast']['grid_thw']} slow={e['slow']['grid_thw']} fast_vs_slow max={e['fast_vs_slow']['max_abs']:.4f} mean={e['fast_vs_slow']['mean_abs']:.5f}")


if __name__ == "__main__":
    main()
