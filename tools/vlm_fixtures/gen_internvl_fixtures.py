#!/usr/bin/env python3
"""Reference fixtures for the Rust InternVL2 image path (`internvl_chat`).

InternVL ships no processor class: its model card's `load_image`
(`dynamic_preprocess` + torchvision transforms) prepares the pixels and
`InternVLChatModel.chat` writes each `<image>` as `<img>` + `<IMG_CONTEXT>` *
256 * tiles + `</img>`. This script carries that code verbatim (from
OpenGVLab/InternVL2-4B, README.md and modeling_internvl_chat.py) and writes
what the Rust port (`crates/cascadia-engine-openvino/src/internvl_vision`)
must reproduce:

  <out>/reference.json
    config      image size, tiles, tokens per tile, mean/std
    tokens      <IMG_CONTEXT>, <img>, </img> ids
    ratios      Python's target-ratio order for max_num 12 and 6
    images      per image (the eight PNGs of tests/fixtures/pixtral_vision):
                the tile grid, the number of views and a summary of
                pixel_values
    prompts     the export's chat template rendered with each image as
                `<img><IMG_CONTEXT></img>\\n` (the API's tag), the expanded
                string `chat` would build, its ids (one BOS: the template
                writes it) and the images
    vision_ir   with --run-ir: the vision IR's rows for two images (needs the
                .bin weights)

Usage (torch, torchvision, transformers, pillow, numpy; openvino for
--run-ir):
  python gen_internvl_fixtures.py --model <OpenVINO/InternVL2-4B-int4-ov dir> \
      --images crates/cascadia-engine-openvino/tests/fixtures/pixtral_vision \
      --out crates/cascadia-engine-openvino/tests/fixtures/internvl_vision [--run-ir]
"""
import argparse
import hashlib
import json
import os

import numpy as np
import PIL
import torch
import torchvision
import torchvision.transforms as T
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

IMAGES = ["square_448", "landscape_640x360", "portrait_300x1000", "tiny_40x30",
          "odd_1001x777", "big_1600x1200", "strip_1500x4", "pole_3x1200"]

# --- verbatim from the OpenGVLab/InternVL2-4B model card -------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
# ---------------------------------------------------------------------------


def target_ratios(min_num, max_num):
    s = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
            for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    return sorted(s, key=lambda x: x[0] * x[1])


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
    ap.add_argument("--images", required=True, help="dir with the eight fixture PNGs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-ir", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    tok = AutoTokenizer.from_pretrained(args.model)
    size = int(cfg.get("force_image_size") or cfg["vision_config"]["image_size"])
    patch = int(cfg["vision_config"]["patch_size"])
    per_tile = int((size // patch) ** 2 * cfg["downsample_ratio"] ** 2)
    max_num = int(cfg["max_dynamic_patch"])
    min_num = int(cfg["min_dynamic_patch"])
    ctx, start, end = (tok.convert_tokens_to_ids(t) for t in ("<IMG_CONTEXT>", "<img>", "</img>"))
    ref = {
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "pillow": PIL.__version__},
        "config": {"image_size": size, "min_tiles": min_num, "max_tiles": max_num,
                   "use_thumbnail": cfg["use_thumbnail"], "tokens_per_tile": per_tile,
                   "image_mean": IMAGENET_MEAN, "image_std": IMAGENET_STD},
        "tokens": {"context": ctx, "start": start, "end": end},
        "ratios": {"12": target_ratios(1, 12), "6": target_ratios(1, 6)},
        "images": {}, "prompts": {},
    }
    transform = build_transform(size)
    imgs = {}
    for name in IMAGES:
        path = os.path.join(args.images, f"{name}.png")
        img = Image.open(path).convert("RGB")
        imgs[name] = img
        w, h = img.size
        grid = find_closest_aspect_ratio(w / h, target_ratios(min_num, max_num), w, h, size)
        tiles = dynamic_preprocess(img, min_num=min_num, max_num=max_num, image_size=size,
                                   use_thumbnail=cfg["use_thumbnail"])
        pv = torch.stack([transform(t) for t in tiles])
        ref["images"][name] = {
            "size": [w, h],
            "png_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "grid": list(grid), "views": len(tiles), "num_tokens": len(tiles) * per_tile,
            "pixel_values": summary(pv),
        }
        print(f"{name}: grid={grid} views={len(tiles)}", flush=True)

    tag = "<img><IMG_CONTEXT></img>\n"
    prompts = {
        "single": ([tag + "What is in this picture?"], ["square_448"]),
        "two_images": (["A:" + tag + "B:" + tag + "Compare."],
                       ["landscape_640x360", "portrait_300x1000"]),
        "tiled": (["Describe." + tag], ["big_1600x1200"]),
        "text_only": (["hello"], []),
    }
    for name, (contents, names) in prompts.items():
        messages = [{"role": "user", "content": c} for c in contents]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # `chat`: each image's `<image>` becomes <img> + context * 256 * tiles + </img>.
        expanded = text
        for n in names:
            run = "<img>" + "<IMG_CONTEXT>" * ref["images"][n]["num_tokens"] + "</img>"
            expanded = expanded.replace(tag, run + "\n", 1)
        ids = tok(expanded, add_special_tokens=False).input_ids
        collapsed = tok(text, add_special_tokens=False).input_ids
        assert ids.count(ctx) == sum(ref["images"][n]["num_tokens"] for n in names)
        ref["prompts"][name] = {"template": text, "input_ids": ids, "template_ids": collapsed,
                                "images": names,
                                "num_tokens": [ref["images"][n]["num_tokens"] for n in names]}
        print(f"[{name}] tokens={len(ids)} images={names}", flush=True)

    if args.run_ir:
        import openvino as ov
        core = ov.Core()
        vis = core.compile_model(os.path.join(args.model, "openvino_vision_embeddings_model.xml"), "CPU")
        ref["vision_ir"] = {}
        for name in ("tiny_40x30", "square_448"):
            tiles = dynamic_preprocess(imgs[name], min_num=min_num, max_num=max_num,
                                       image_size=size, use_thumbnail=cfg["use_thumbnail"])
            pv = torch.stack([transform(t) for t in tiles]).numpy()
            out = vis({"pixel_values": pv})[0]
            ref["vision_ir"][name] = {"views": len(tiles), "out_shape": list(out.shape),
                                      "image_embeds": summary(torch.from_numpy(out))}
            print(f"[vision_ir] {name}: {list(out.shape)}", flush=True)

    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump(ref, f, separators=(",", ":"))
    print("FIXTURES DONE", flush=True)


if __name__ == "__main__":
    main()
