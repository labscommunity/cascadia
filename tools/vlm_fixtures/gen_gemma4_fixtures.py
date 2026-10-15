#!/usr/bin/env python3
"""Reference fixtures for the Rust Gemma 4 image preprocessing and prompt layout.

Runs Hugging Face's own Gemma 4 processor (no language model weights) and
writes what the Rust port must reproduce:

  <out>/<name>.png         deterministic synthetic inputs (lossless; the same
                           images as the Qwen3.5 fixtures)
  <out>/reference.json     per image: the resized size, soft-token count,
                           position ids and a summary of pixel_values for the
                           torchvision (default) and PIL processors; per
                           prompt: the rendered template, input_ids and
                           mm_token_type_ids

Pixel tensors are large, so each is summarised (shape, sum, abs-sum, sum of
squares, first/last 64 values, every 1009th value) rather than stored.

With --run-ir the export's vision IR is also driven by the PIL processor's
output (`vision_ir`): the reference for the Rust preprocessing feeding the
same IR.

Usage (needs torch, torchvision, transformers==5.14.1, pillow, numpy; and
openvino for --run-ir):
  python gen_gemma4_fixtures.py --model <Gemma 4 export dir> --out <dir> [--run-ir]
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torchvision
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor


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
    "big_1600x1200": (1600, 1200, 6),
    # Extreme aspect ratios: one side floors to 0 and takes the edge-case
    # branch of get_aspect_ratio_preserving_size.
    "strip_1500x4": (1500, 4, 7),
    "pole_3x1200": (3, 1200, 8),
}

# Extra runs at a smaller soft-token budget: a strong downscale, which
# exercises the antialiased (widened) kernel.
LOW_BUDGET = {"big_1600x1200_soft70": ("big_1600x1200", 70)}


def summary(t):
    a = torch.as_tensor(t).detach().to(torch.float64).flatten().numpy()
    return {
        "shape": list(t.shape),
        "sum": float(a.sum()),
        "abs_sum": float(np.abs(a).sum()),
        "sq_sum": float((a * a).sum()),
        "head": [float(v) for v in a[:64]],
        "tail": [float(v) for v in a[-64:]],
        "stride_1009": [float(v) for v in a[::1009]],
    }


def run(proc, img, **kw):
    o = proc(images=[img], return_tensors="pt", **kw)
    pos = o["image_position_ids"][0]
    real = int((pos[:, 0] >= 0).sum())
    # Real positions are (x=col, y=row) in row-major order, so the last one
    # gives the patch grid.
    cols, rows = int(pos[real - 1, 0]) + 1, int(pos[real - 1, 1]) + 1
    return o, {
        "patch_grid_hw": [rows, cols],
        "resized_hw": [rows * proc.patch_size, cols * proc.patch_size],
        "num_patches": real,
        "num_soft_tokens": int(o["num_soft_tokens_per_image"][0]),
        "pixel_values": summary(o["pixel_values"]),
        "position_ids": summary(pos),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-ir", action="store_true", help="also run the vision IR (needs openvino)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model)
    fast = AutoImageProcessor.from_pretrained(args.model, backend="torchvision")
    slow = AutoImageProcessor.from_pretrained(args.model, backend="pil")
    tok = processor.tokenizer

    ref = {
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
        "processor": {k: list(v) if isinstance(v, tuple) else v for k, v in slow.to_dict().items()
                      if isinstance(v, (int, float, str, list, tuple, dict, bool))},
        "default_backend": type(processor.image_processor).__name__,
        "tokens": {"image": processor.image_token, "image_id": processor.image_token_id,
                   "boi": processor.boi_token, "boi_id": tok.convert_tokens_to_ids(processor.boi_token),
                   "eoi": processor.eoi_token, "eoi_id": tok.convert_tokens_to_ids(processor.eoi_token),
                   "pad_id": tok.pad_token_id},
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
            o, entry[kind] = run(proc, img)
            outs[kind] = o["pixel_values"]
        d = (outs["fast"].double() - outs["slow"].double()).abs()
        entry["fast_vs_slow"] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                                 "frac_differ": float((d > 0).double().mean())}
        ref["images"][name] = entry

    for name, (src, soft) in LOW_BUDGET.items():
        img = imgs[src]
        entry = {"size": list(img.size), "source": src, "max_soft_tokens": soft}
        outs = {}
        for kind, proc in (("fast", fast), ("slow", slow)):
            o, entry[kind] = run(proc, img, max_soft_tokens=soft)
            outs[kind] = o["pixel_values"]
        d = (outs["fast"].double() - outs["slow"].double()).abs()
        entry["fast_vs_slow"] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                                 "frac_differ": float((d > 0).double().mean())}
        ref["images"][name] = entry

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
        "text_only": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    }
    for name, messages in prompts.items():
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        o = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                          add_generation_prompt=True, return_tensors="pt")
        ids = o["input_ids"][0]
        # The processor drops num_soft_tokens_per_image from its output, so
        # name the images in prompt order; their counts are under `images`.
        names = [next(n for n, im in imgs.items() if im is c["image"])
                 for m in messages for c in m["content"] if c["type"] == "image"]
        entry = {"template": text, "input_ids": ids.tolist(), "images": names,
                 "num_soft_tokens": [ref["images"][n]["fast"]["num_soft_tokens"] for n in names]}
        if "mm_token_type_ids" in o:
            entry["mm_token_type_ids"] = o["mm_token_type_ids"][0].tolist()
        ref["prompts"][name] = entry
        print(f"[{name}] tokens={len(entry['input_ids'])} soft={entry['num_soft_tokens']}")

    if args.run_ir:
        ref["vision_ir"] = run_vision_ir(args.model, slow, imgs)

    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump(ref, f, separators=(",", ":"))
    for name, e in ref["images"].items():
        print(f"{name}: resized fast={e['fast']['resized_hw']} slow={e['slow']['resized_hw']} "
              f"soft={e['slow']['num_soft_tokens']} fast_vs_slow max={e['fast_vs_slow']['max_abs']:.5f} "
              f"differ={e['fast_vs_slow']['frac_differ']:.4f}")


def run_vision_ir(model_dir, proc, imgs):
    """The PIL processor's output driving the export's vision IR."""
    import openvino as ov
    core = ov.Core()
    ir = core.compile_model(os.path.join(model_dir, "openvino_vision_embeddings_model.xml"), "CPU")
    out = {}
    for name in ("tiny_40x30", "square_448", "landscape_640x360"):
        o = proc(images=[imgs[name]], return_tensors="np")
        emb = ir({"pixel_values": o["pixel_values"].astype(np.float32),
                  "image_position_ids": o["image_position_ids"].astype(np.int64)})[0]
        n = int(o["num_soft_tokens_per_image"][0])
        out[name] = {"num_soft_tokens": n, "out_shape": list(emb.shape),
                     "image_embeds": summary(torch.from_numpy(np.ascontiguousarray(emb[:n])))}
        print(f"[vision_ir] {name}: soft={n} out={list(emb.shape)}")
    return out


if __name__ == "__main__":
    main()
