#!/usr/bin/env python3
"""Reference fixtures for the Rust Gemma 3 image preprocessing and prompt layout.

Runs Hugging Face's own Gemma 3 processor (no language model weights: only
the export's JSON configs, tokenizer and chat template) on the eight
synthetic images of the Gemma 4 / Pixtral fixtures and writes what the Rust
port must reproduce:

  <out>/reference.json   processor settings and framing token ids; per image
                         a summary of pixel_values for the PIL processor
                         (`slow`) and the torchvision one (`fast`, HF's
                         default) and how far they differ; per prompt the
                         rendered template (one `<start_of_image>` per
                         image), the processor's expanded text, input_ids
                         and token_type_ids; with --run-ir, the vision IR's
                         rows for two images

The images are read from --images (default: the Gemma 4 fixture PNGs), so
they are not stored twice.

Usage (transformers==5.14.1, torch, torchvision, pillow, numpy; openvino
for --run-ir):
  python gen_gemma3_fixtures.py --model <OpenVINO/gemma-3-12b-it-int4-ov dir> \
      --out <dir> [--images <png dir>] [--run-ir]
"""
import argparse
import json
import os

import numpy as np
import PIL
import torch
import torchvision
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGES = os.path.join(HERE, os.pardir, os.pardir, "crates", "cascadia-engine-openvino",
                              "tests", "fixtures", "gemma4_vision")

IMAGES = ("square_448", "landscape_640x360", "portrait_300x1000", "tiny_40x30",
          "odd_1001x777", "big_1600x1200", "strip_1500x4", "pole_3x1200")


def summary(t):
    a = torch.as_tensor(np.asarray(t)).detach().to(torch.float64).flatten().numpy()
    return {
        "shape": list(np.asarray(t).shape),
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
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--run-ir", action="store_true", help="also run the vision IR (needs openvino)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model)
    fast = AutoImageProcessor.from_pretrained(args.model, backend="torchvision")
    slow = AutoImageProcessor.from_pretrained(args.model, backend="pil")
    tok = processor.tokenizer
    ids = lambda s: tok.convert_tokens_to_ids(s)  # noqa: E731

    ref = {
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "pillow": PIL.__version__},
        "processor": {k: list(v) if isinstance(v, tuple) else v for k, v in slow.to_dict().items()
                      if isinstance(v, (int, float, str, list, tuple, dict, bool))},
        "image_seq_length": processor.image_seq_length,
        "full_image_sequence": processor.full_image_sequence,
        "default_backend": type(processor.image_processor).__name__,
        "tokens": {"image": tok.image_token, "image_id": ids(tok.image_token),
                   "boi": tok.boi_token, "boi_id": ids(tok.boi_token),
                   "eoi": tok.eoi_token, "eoi_id": ids(tok.eoi_token),
                   "pad_id": tok.pad_token_id},
        "images": {}, "prompts": {},
    }

    imgs = {}
    for name in IMAGES:
        img = Image.open(os.path.join(args.images, f"{name}.png")).convert("RGB")
        imgs[name] = img
        entry = {"size": list(img.size)}
        outs = {}
        for kind, proc in (("fast", fast), ("slow", slow)):
            o = proc(images=[img], return_tensors="pt")
            outs[kind] = o["pixel_values"][0]
            entry[kind] = {"pixel_values": summary(outs[kind]),
                           "num_crops": [int(x) for x in o["num_crops"]]}
        d = (outs["fast"].double() - outs["slow"].double()).abs()
        entry["fast_vs_slow"] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                                 "frac_differ": float((d > 0).double().mean())}
        ref["images"][name] = entry

    def user(*parts):
        content = []
        for p in parts:
            content.append({"type": "image", "image": imgs[p]} if p in imgs else
                           {"type": "text", "text": p})
        return {"role": "user", "content": content}

    prompts = {
        "single": [user("square_448", "Describe.")],
        "two_images": [user("A:", "landscape_640x360", "B:", "portrait_300x1000", "Compare.")],
        "back_to_back": [user("tiny_40x30", "odd_1001x777", "Same?")],
        "multi_turn": [
            user("tiny_40x30", "What?"),
            {"role": "assistant", "content": [{"type": "text", "text": "A dot."}]},
            user("And this?", "odd_1001x777")],
        "system": [{"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
                   user("square_448", "Colours?")],
        "text_only": [user("hello")],
    }
    for name, messages in prompts.items():
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [c["image"] for m in messages for c in m["content"] if c["type"] == "image"]
        o = processor(text=text, images=images or None, return_tensors="pt")
        # The processor's text after expansion, as its tokenizer saw it.
        expanded = text.replace(processor.boi_token, processor.full_image_sequence) \
            if images else text
        entry = {"template": text, "expanded": expanded,
                 "input_ids": o["input_ids"][0].tolist(),
                 "token_type_ids": o["token_type_ids"][0].tolist(),
                 "images": [n for n, im in imgs.items() for x in images if x is im]}
        ref["prompts"][name] = entry
        print(f"[{name}] tokens={len(entry['input_ids'])} images={len(images)}")

    if args.run_ir:
        ref["vision_ir"] = run_vision_ir(args.model, slow, imgs)

    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump(ref, f, separators=(",", ":"))
    for name, e in ref["images"].items():
        fs = e["fast_vs_slow"]
        print(f"{name}: fast_vs_slow max={fs['max_abs']:.5f} differ={fs['frac_differ']:.4f}")


def run_vision_ir(model_dir, proc, imgs):
    """The PIL processor's output driving the export's vision IR."""
    import openvino as ov
    ir = ov.Core().compile_model(os.path.join(model_dir, "openvino_vision_embeddings_model.xml"),
                                 "CPU")
    out = {}
    for name in ("tiny_40x30", "square_448"):
        o = proc(images=[imgs[name]], return_tensors="np")
        emb = ir({"pixel_values": o["pixel_values"].astype(np.float32)})[0]
        out[name] = {"out_shape": list(emb.shape), "image_embeds": summary(emb[0])}
        print(f"[vision_ir] {name}: out={list(emb.shape)}")
    return out


if __name__ == "__main__":
    main()
