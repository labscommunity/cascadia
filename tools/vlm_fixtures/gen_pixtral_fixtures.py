#!/usr/bin/env python3
"""Reference fixtures for the Rust Pixtral image preprocessing and prompt layout.

Also covers Mistral 3 (Ministral-3 and friends): the same `PixtralProcessor`
with `spatial_merge_size` 2, so images resize to whole `patch * merge` cells.

Runs Hugging Face's own Pixtral processor (no language model weights) and
writes what the Rust port must reproduce:

  <out>/<name>.png         deterministic synthetic inputs (the Gemma 4 set)
  <out>/reference.json     per image: the resized size, patch grid and a
                           summary of pixel_values for the torchvision
                           (default) and PIL processors; per prompt: the
                           rendered template, input_ids and image names;
                           with --run-ir, the vision IR's output

Usage (transformers==5.14.1, torch, torchvision, pillow, numpy; openvino
for --run-ir):
  python gen_pixtral_fixtures.py --model <Pixtral OV export> --out <dir> [--run-ir]
  python gen_pixtral_fixtures.py --model <Mistral 3 OV export> --out <dir> --run-ir \
      --hf-features <HF checkpoint>   # torch: HF get_image_features vs the OV chain
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-ir", action="store_true")
    ap.add_argument("--hf-features", default=None,
                    help="HF checkpoint: compare get_image_features with the OV vision chain")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model)
    fast = AutoImageProcessor.from_pretrained(args.model, backend="torchvision")
    slow = AutoImageProcessor.from_pretrained(args.model, backend="pil")
    tok = processor.tokenizer
    merge = int(getattr(processor, "spatial_merge_size", 1) or 1)
    patch = int(processor.patch_size)
    # PixtralProcessor hands the image processor patch * merge as the patch.
    cell = patch * merge
    ref = {
        "patch_size": patch, "spatial_merge_size": merge,
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
        "processor": {k: (list(v) if isinstance(v, tuple) else v) for k, v in slow.to_dict().items()
                      if isinstance(v, (int, float, str, list, tuple, dict, bool))},
        "tokens": {"image": processor.image_token_id, "image_break": processor.image_break_token_id,
                   "image_end": processor.image_end_token_id},
        "images": {}, "prompts": {},
    }
    imgs = {}
    for name, (w, h, seed) in IMAGES.items():
        img = synth(w, h, seed)
        path = os.path.join(args.out, f"{name}.png")
        img.save(path)
        img = Image.open(path).convert("RGB")
        imgs[name] = img
        entry = {"size": [w, h], "png_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
        outs = {}
        for kind, proc in (("fast", fast), ("slow", slow)):
            o = proc(images=[img], return_tensors="pt", patch_size={"height": cell, "width": cell})
            pv = o["pixel_values"][0]
            hh, ww = (int(x) for x in o["image_sizes"][0])
            outs[kind] = pv
            entry[kind] = {"resized_hw": [hh, ww], "grid": [hh // cell, ww // cell],
                           "pixel_values": summary(pv[:, :hh, :ww])}
        d = (outs["fast"].double() - outs["slow"].double()).abs()
        entry["fast_vs_slow"] = {"max_abs": float(d.max()), "mean_abs": float(d.mean())}
        ref["images"][name] = entry
        print(f"{name}: resized={entry['slow']['resized_hw']} grid={entry['slow']['grid']} "
              f"fast_vs_slow max={entry['fast_vs_slow']['max_abs']:.4f}", flush=True)

    prompts = {
        "single": [{"role": "user", "content": [
            {"type": "image", "image": imgs["square_448"]}, {"type": "text", "text": "Describe."}]}],
        "two_images": [{"role": "user", "content": [
            {"type": "text", "text": "A:"}, {"type": "image", "image": imgs["landscape_640x360"]},
            {"type": "text", "text": "B:"}, {"type": "image", "image": imgs["portrait_300x1000"]},
            {"type": "text", "text": "Compare."}]}],
        "text_only": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    }
    for name, messages in prompts.items():
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        o = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                          add_generation_prompt=True, return_tensors="pt")
        names = [next(n for n, im in imgs.items() if im is c["image"])
                 for m in messages for c in m["content"] if c["type"] == "image"]
        ref["prompts"][name] = {"template": text, "input_ids": o["input_ids"][0].tolist(),
                                "images": names,
                                "grids": [ref["images"][n]["fast"]["grid"] for n in names]}
        print(f"[{name}] tokens={o['input_ids'].shape[1]} images={names}", flush=True)

    def ov_chain(core, name):
        """The OV vision chain for one image: vision IR, then (Mistral 3) the
        host unfold and the projector IR. Returns (pixel input, rows)."""
        o = slow(images=[imgs[name]], return_tensors="np", patch_size={"height": cell, "width": cell})
        hh, ww = (int(x) for x in o["image_sizes"][0])
        pv = o["pixel_values"][:, :, :hh, :ww].astype(np.float32)
        emb = vis({"pixel_values": pv})[0]
        emb = emb[0] if emb.ndim == 3 else emb
        if merge > 1:
            ph, pw, d = hh // patch, ww // patch, emb.shape[-1]
            grid = torch.from_numpy(np.ascontiguousarray(emb)).view(ph, pw, d).permute(2, 0, 1)[None]
            un = torch.nn.functional.unfold(grid, kernel_size=merge, stride=merge)
            merged = un.view(d * merge * merge, -1).t().contiguous().numpy()
            emb = proj({proj.inputs[0].get_any_name(): merged})[0]
        return o, hh, ww, np.ascontiguousarray(emb)

    if args.run_ir or args.hf_features:
        import openvino as ov
        core = ov.Core()
        vis = core.compile_model(os.path.join(args.model, "openvino_vision_embeddings_model.xml"), "CPU")
        proj = (core.compile_model(os.path.join(args.model, "openvino_multi_modal_projector_model.xml"), "CPU")
                if merge > 1 else None)
        ref["vision_ir"] = {}
        for name in ("tiny_40x30", "square_448"):
            _, hh, ww, emb = ov_chain(core, name)
            ref["vision_ir"][name] = {"grid": [hh // cell, ww // cell], "out_shape": list(emb.shape),
                                      "image_embeds": summary(torch.from_numpy(emb))}
            print(f"[vision_ir] {name}: out={list(emb.shape)}", flush=True)

    if args.hf_features:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.hf_features, torch_dtype=torch.float32)
        model.eval()
        ref["hf_features"] = {}
        for name in ("tiny_40x30", "square_448"):
            o, hh, ww, emb = ov_chain(core, name)
            with torch.no_grad():
                feats = model.model.get_image_features(
                    pixel_values=torch.from_numpy(o["pixel_values"]).float(),
                    image_sizes=torch.tensor(o["image_sizes"]))
            hf = feats.pooler_output if hasattr(feats, "pooler_output") else feats
            hf = torch.cat(list(hf), 0).float().numpy() if isinstance(hf, (list, tuple)) else hf.float().numpy()
            d = np.abs(hf - emb)
            cos = float((hf * emb).sum() / (np.linalg.norm(hf) * np.linalg.norm(emb)))
            ref["hf_features"][name] = {"shape": list(hf.shape), "max_abs": float(d.max()),
                                        "mean_abs": float(d.mean()), "cosine": cos}
            print(f"[hf_features] {name}: hf={list(hf.shape)} ov={list(emb.shape)} "
                  f"max_abs={d.max():.4f} mean_abs={d.mean():.5f} cos={cos:.6f}", flush=True)

    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump(ref, f, separators=(",", ":"))
    print("FIXTURES DONE", flush=True)


if __name__ == "__main__":
    main()
