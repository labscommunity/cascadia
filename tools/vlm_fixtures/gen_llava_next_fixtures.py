#!/usr/bin/env python3
"""Reference fixtures for the Rust LLaVA-NeXT image input (`llava_next_vision`).

Runs Hugging Face's own LLaVA-NeXT processor and the model's
`pack_image_features` (no language model or vision weights: packing is fed
synthetic features) and writes what the Rust port must reproduce:

  <out>/reference.json
    processor     the image processor's settings
    images.<name> per image (the Pixtral fixture PNGs): the selected
                  pinpoint, tile grid, number of 336-px views and a summary of
                  the PIL processor's pixel_values (bit-identity reference);
                  the torchvision processor's distance from it; HF's token
                  count; the packed feature layout as runs of view rows
    prompts.<name> the processor's rendered template and input_ids, and the
                  prompt Cascadia serves (the tokenizer's own chat template
                  over images-first content, tokenized without special tokens
                  and expanded), which must equal them
    counts        HF's token count for a sweep of image sizes
    vision_ir     with --run-ir only: the vision IR's packed rows for two
                  images (needs the export's .bin weights)

Usage (transformers==5.14.1, torch, torchvision, pillow, numpy, jinja2;
openvino for --run-ir):
  python gen_llava_next_fixtures.py --model <OpenVINO/llava-v1.6-mistral-7b-hf-int4-ov dir> \
      --images crates/cascadia-engine-openvino/tests/fixtures/pixtral_vision --out <dir>

The model dir needs only the JSON configs and tokenizer.json unless --run-ir.
"""
import argparse
import hashlib
import json
import os
from types import SimpleNamespace

import jinja2
import numpy as np
import torch
import torchvision
import transformers
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
from transformers.image_processing_utils import select_best_resolution
from transformers.models.llava_next.modeling_llava_next import LlavaNextModel

IMAGES = (
    "square_448", "landscape_640x360", "portrait_300x1000", "tiny_40x30",
    "odd_1001x777", "big_1600x1200", "strip_1500x4", "pole_3x1200",
)

SWEEP = (1, 2, 3, 4, 5, 7, 10, 14, 30, 40, 99, 100, 167, 200, 255, 299, 300, 301, 333,
         336, 337, 360, 447, 448, 449, 500, 599, 640, 671, 672, 673, 700, 777, 800, 1000,
         1001, 1007, 1008, 1009, 1200, 1500, 2000, 4000)


def summary(t):
    a = torch.as_tensor(t).detach().to(torch.float64).flatten().numpy()
    return {
        "shape": list(t.shape),
        "sum": float(a.sum()),
        "head": [float(v) for v in a[:64]],
        "tail": [float(v) for v in a[-64:]],
        "stride_1009": [float(v) for v in a[::1009]],
    }


def runs(idx):
    """Run-length form of a row-source list: [[start, len], ...], -1 = newline."""
    out = []
    for v in idx:
        if out and ((v == -1 and out[-1][0] == -1) or
                    (v != -1 and out[-1][0] != -1 and out[-1][0] + out[-1][1] == v)):
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


def packed_layout(cfg, n_views, h, w):
    """HF pack_image_features on features whose channel 0 names the source
    row (view * 576 + position); the newline vector is -1."""
    side = cfg.vision_config.image_size // cfg.vision_config.patch_size
    per = side * side
    src = torch.arange(n_views * per, dtype=torch.float32).view(n_views, per, 1)
    feat = torch.cat([src, -src - 1], dim=-1)
    newline = torch.tensor([-1.0, -2.0])
    fake = SimpleNamespace(config=cfg)
    packed, lens = LlavaNextModel.pack_image_features(
        fake, [feat], torch.tensor([[h, w]]), "default", newline)
    packed = packed if isinstance(packed, torch.Tensor) else torch.cat(list(packed), 0)
    ch0 = packed[:, 0]
    nl = ch0 == -1
    assert torch.equal(packed[nl, 1], torch.full((int(nl.sum()),), -2.0))
    assert torch.equal(packed[~nl, 1], -ch0[~nl] - 1)
    return [int(v) for v in ch0.tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True, help="dir with the Pixtral fixture PNGs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-ir", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model)
    slow = AutoImageProcessor.from_pretrained(args.model, backend="pil")
    fast = AutoImageProcessor.from_pretrained(args.model, backend="torchvision")
    cfg = AutoConfig.from_pretrained(args.model)
    tok = processor.tokenizer
    image_token = processor.image_token_id
    ref = {
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
        "processor": {k: (list(v) if isinstance(v, tuple) else v) for k, v in slow.to_dict().items()
                      if isinstance(v, (int, float, str, list, tuple, dict, bool))},
        "patch_size": processor.patch_size,
        "image_token": image_token,
        "images": {}, "prompts": {}, "counts": [],
    }
    pins = slow.image_grid_pinpoints
    tile = cfg.vision_config.image_size

    def hf_count(h, w):
        n = processor._get_number_of_features(h, w, tile, tile)
        return n - 1 if processor.vision_feature_select_strategy == "default" else n

    imgs = {}
    for name in IMAGES:
        path = os.path.join(args.images, f"{name}.png")
        img = Image.open(path).convert("RGB")
        imgs[name] = img
        w, h = img.size
        best = select_best_resolution((h, w), pins)
        o = slow(images=[img], return_tensors="pt")
        pv = o["pixel_values"][0]
        f = fast(images=[img], return_tensors="pt")["pixel_values"][0]
        assert f.shape == pv.shape
        n_views = pv.shape[0]
        grid = [best[0] // tile, best[1] // tile]
        assert n_views == 1 + grid[0] * grid[1]
        layout = packed_layout(cfg, n_views, h, w)
        count = hf_count(h, w)
        assert len(layout) == count, (name, len(layout), count)
        d = (f.double() - pv.double()).abs()
        ref["images"][name] = {
            "size": [w, h], "png_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "best_resolution": list(best), "grid": grid, "views": n_views,
            "pixel_values": summary(pv),
            "fast_vs_slow": {"max_abs": float(d.max()), "mean_abs": float(d.mean())},
            "tokens": count, "packed_runs": runs(layout),
        }
        print(f"{name}: best={best} grid={grid} views={n_views} tokens={count} "
              f"fast_vs_slow max={float(d.max()):.4f}", flush=True)

    for h in SWEEP:
        for w in SWEEP:
            ref["counts"].append([h, w, hf_count(h, w)])

    # What Cascadia serves: the tokenizer's own template (tokenizer_config.json,
    # with its bos) over one content string, images first as "<image>\n" and
    # the text parts back to back (MediaConfig::llava_next_native), tokenized
    # without special tokens, then each <image> expanded to its count.
    with open(os.path.join(args.model, "tokenizer_config.json")) as fh:
        tok_template = json.load(fh)["chat_template"]
    env = jinja2.Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    served_tpl = env.from_string(tok_template)

    prompts = {
        "single": [{"role": "user", "content": [
            {"type": "text", "text": "What is in this picture?"},
            {"type": "image", "image": imgs["square_448"]}]}],
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
        ids = o["input_ids"][0].tolist()
        names = [next(n for n, im in imgs.items() if im is c["image"])
                 for m in messages for c in m["content"] if c["type"] == "image"]
        served_msgs = []
        for m in messages:
            parts = m["content"]
            joined = ("<image>\n" * sum(c["type"] == "image" for c in parts)
                      + "".join(c["text"] for c in parts if c["type"] == "text"))
            served_msgs.append({"role": m["role"], "content": joined})
        served = served_tpl.render(messages=served_msgs, bos_token=tok.bos_token,
                                   eos_token=tok.eos_token, add_generation_prompt=True)
        served_ids = tok(served, add_special_tokens=False)["input_ids"]
        counts = iter(ref["images"][n]["tokens"] for n in names)
        expanded = []
        for t in served_ids:
            expanded.extend([t] * next(counts) if t == image_token else [t])
        assert expanded == ids, f"{name}: served prompt differs from the processor's"
        ref["prompts"][name] = {"template": text, "served_template": served,
                                "input_ids": ids, "images": names}
        print(f"[{name}] tokens={len(ids)} images={names} served==hf", flush=True)

    if args.run_ir:
        import openvino as ov
        vis = ov.Core().compile_model(
            os.path.join(args.model, "openvino_vision_embeddings_model.xml"), "CPU")
        newline = torch.tensor(cfg.image_newline if isinstance(cfg.image_newline, list)
                               else json.load(open(os.path.join(args.model, "config.json")))["image_newline"])
        ref["vision_ir"] = {}
        for name in ("tiny_40x30", "square_448"):
            img = imgs[name]
            w, h = img.size
            pv = slow(images=[img], return_tensors="np")["pixel_values"][0].astype(np.float32)
            emb = torch.from_numpy(np.ascontiguousarray(vis({"pixel_values": pv})[0]))
            packed, _ = LlavaNextModel.pack_image_features(
                SimpleNamespace(config=cfg), [emb], torch.tensor([[h, w]]), "default", newline)
            packed = packed if isinstance(packed, torch.Tensor) else torch.cat(list(packed), 0)
            ref["vision_ir"][name] = {"out_shape": list(emb.shape), "packed": summary(packed)}
            print(f"[vision_ir] {name}: views={list(emb.shape)} packed={list(packed.shape)}", flush=True)

    with open(os.path.join(args.out, "reference.json"), "w") as fh:
        json.dump(ref, fh, separators=(",", ":"))
    print("FIXTURES DONE", flush=True)


if __name__ == "__main__":
    main()
