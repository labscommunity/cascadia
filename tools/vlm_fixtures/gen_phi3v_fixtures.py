#!/usr/bin/env python3
"""Reference fixtures for the Rust Phi-3.5-vision image path (`phi3_v`).

Runs the export's own remote-code processor (`processing_phi3_v.py`:
`Phi3VImageProcessor`, `Phi3VProcessor`) and the feature assembly of
`modeling_phi3_v.py` (`reshape_hd_patches_2x2merge`, `add_image_newline`,
`hd_feature_transform`), with no language-model weights, and writes what the
Rust port (`crates/cascadia-engine-openvino/src/phi3v_vision`) must reproduce:

  <out>/reference.json
    processor   num_crops, num_img_tokens, mean/std
    tokens      <|image|> id, BOS id
    images      per image (the eight PNGs of tests/fixtures/pixtral_vision):
                image_sizes, num_img_tokens (also from size alone), and
                summaries of pixel_values: the global view and the crops
    assembly    hd_feature_transform on synthetic vision features (an exact
                formula, so no IR is needed) with the config's glb_GN/sub_GN
                and an identity projection: shape and summary
    separators  the config's glb_GN and sub_GN, flattened (the Rust test
                assembles with them)
    prompts     the rendered template (tags as the API writes them), the
                processor's input_ids (images as -1, -2, ...), each chunk's
                own ids, and the images
    vision_ir   with --run-ir: the vision IR + host assembly + projection IR
                rows for two images (needs the .bin weights; run it on a
                machine that has the export)

Usage (torch, torchvision, transformers, pillow, numpy; openvino for
--run-ir):
  python gen_phi3v_fixtures.py --model <OpenVINO/Phi-3.5-vision-instruct-int4-ov dir> \
      --images crates/cascadia-engine-openvino/tests/fixtures/pixtral_vision \
      --out crates/cascadia-engine-openvino/tests/fixtures/phi3v_vision [--run-ir]
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import types

import numpy as np
import PIL
import torch
import torchvision
import transformers
from PIL import Image
from transformers import AutoTokenizer

IMAGES = ["square_448", "landscape_640x360", "portrait_300x1000", "tiny_40x30",
          "odd_1001x777", "big_1600x1200", "strip_1500x4", "pole_3x1200"]


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


def load_processing(model_dir):
    spec = importlib.util.spec_from_file_location(
        "processing_phi3_v", os.path.join(model_dir, "processing_phi3_v.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def embedding_methods(model_dir):
    """`Phi3ImageEmbedding`'s three assembly methods, lifted out of
    modeling_phi3_v.py (importing the module needs the CLIP/flash-attention
    stack of the transformers it was written for; these use only torch)."""
    src = open(os.path.join(model_dir, "modeling_phi3_v.py")).read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Phi3ImageEmbedding")
    want = {"hd_feature_transform", "reshape_hd_patches_2x2merge", "add_image_newline"}
    ns = {"torch": torch, "nn": torch.nn}
    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in want:
            exec(compile(ast.Module(body=[fn], type_ignores=[]), "modeling_phi3_v.py", "exec"), ns)
    return {k: ns[k] for k in want}


class _Identity(torch.nn.Module):
    """A projection that returns its input, with the `bias` the method reads
    for device and dtype."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x


def synthetic_features(views, side=24, d=1024):
    """Exact in f32; the Rust test builds the same values."""
    v = torch.arange(views, dtype=torch.int64)[:, None, None]
    p = torch.arange(side * side, dtype=torch.int64)[None, :, None]
    c = torch.arange(d, dtype=torch.int64)[None, None, :]
    return (((v * 7919 + p * 31 + c * 7) % 1000).to(torch.float32) / 8.0) - 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True, help="dir with the eight fixture PNGs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-ir", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pm = load_processing(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    pre = json.load(open(os.path.join(args.model, "preprocessor_config.json")))
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    ip = pm.Phi3VImageProcessor(num_crops=pre["num_crops"], image_mean=pre["image_mean"],
                                image_std=pre["image_std"], num_img_tokens=pre["num_img_tokens"])
    proc = pm.Phi3VProcessor(ip, tok)
    image_id = tok.convert_tokens_to_ids("<|image|>")
    ref = {
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "pillow": PIL.__version__},
        "processor": {"num_crops": pre["num_crops"], "num_img_tokens": pre["num_img_tokens"],
                      "image_mean": pre["image_mean"], "image_std": pre["image_std"]},
        "tokens": {"image": image_id, "bos": tok.bos_token_id},
        "images": {}, "prompts": {},
    }

    imgs = {}
    for name in IMAGES:
        path = os.path.join(args.images, f"{name}.png")
        img = Image.open(path).convert("RGB")
        imgs[name] = img
        o = ip.preprocess([img], return_tensors="pt")
        h, w = (int(x) for x in o["image_sizes"][0])
        n = (h // 336) * (w // 336)
        pv = o["pixel_values"][0]
        ntok = int(o["num_img_tokens"][0])
        assert ntok == ip.calc_num_image_tokens_from_image_size(*img.size), name
        ref["images"][name] = {
            "size": list(img.size),
            "png_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "image_size_hw": [h, w], "crops": [h // 336, w // 336], "num_img_tokens": ntok,
            "global": summary(pv[0]), "crops_pixels": summary(pv[1:1 + n]),
        }
        print(f"{name}: hd={h}x{w} crops={n} tokens={ntok}", flush=True)

    # Feature assembly on synthetic features with the real separators.
    meth = embedding_methods(args.model)
    emb = types.SimpleNamespace(
        hd_transform_order="sub_glb",
        img_projection=torch.nn.Sequential(_Identity()),
        glb_GN=torch.tensor(cfg["glb_GN"], dtype=torch.float32),
        sub_GN=torch.tensor(cfg["sub_GN"], dtype=torch.float32),
    )
    for k, f in meth.items():
        setattr(emb, k, types.MethodType(f, emb))
    ref["separators"] = {"glb_GN": emb.glb_GN.flatten().tolist(),
                         "sub_GN": emb.sub_GN.flatten().tolist()}
    ref["assembly"] = {}
    for name in ("landscape_640x360", "portrait_300x1000"):
        h, w = ref["images"][name]["image_size_hw"]
        views = 1 + (h // 336) * (w // 336)
        feat = synthetic_features(views)
        # hd_feature_transform takes (num_images, num_crops + 1, 576, 1024).
        out = emb.hd_feature_transform(feat[None], [[h, w]])
        ref["assembly"][name] = {"image_size_hw": [h, w], "views": views, "rows": summary(out)}
        assert out.shape[0] == ref["images"][name]["num_img_tokens"], name
        print(f"[assembly] {name}: {list(out.shape)}", flush=True)

    # Prompts as the API renders them: parts back to back, each image as
    # `<|image_N|>\n` (MediaConfig::phi3v_native).
    tag = lambda i: f"<|image_{i}|>\n"
    prompts = {
        "single": ([tag(1) + "What is in this picture?"], ["square_448"]),
        "two_images": (["A:" + tag(1) + "B:" + tag(2) + "Compare."],
                       ["landscape_640x360", "portrait_300x1000"]),
        "text_first": (["Describe this image." + tag(1)], ["odd_1001x777"]),
        "text_only": (["hello"], []),
    }
    for name, (contents, names) in prompts.items():
        messages = [{"role": "user", "content": c} for c in contents]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if names:
            o = proc(text, [imgs[n] for n in names], return_tensors="pt")
        else:
            o = proc(text, None, return_tensors="pt")
        import re
        chunks = re.split(r"<\|image_\d+\|>", text)
        ref["prompts"][name] = {
            "template": text, "input_ids": o["input_ids"][0].tolist(),
            "chunks": chunks, "chunk_ids": [tok(c).input_ids for c in chunks],
            "images": names,
            "num_img_tokens": [ref["images"][n]["num_img_tokens"] for n in names],
        }
        print(f"[{name}] tokens={o['input_ids'].shape[1]} images={names}", flush=True)

    if args.run_ir:
        import openvino as ov
        core = ov.Core()
        vis = core.compile_model(os.path.join(args.model, "openvino_vision_embeddings_model.xml"), "CPU")
        proj = core.compile_model(os.path.join(args.model, "openvino_vision_projection_model.xml"), "CPU")
        ref["vision_ir"] = {}
        for name in ("tiny_40x30", "square_448"):
            o = ip.preprocess([imgs[name]], return_tensors="pt")
            h, w = (int(x) for x in o["image_sizes"][0])
            views = 1 + (h // 336) * (w // 336)
            feat = torch.from_numpy(vis({"pixel_values": o["pixel_values"][0, :views].numpy()})[0])
            merged = emb.hd_feature_transform(feat[None], [[h, w]])
            rows = proj({proj.inputs[0].get_any_name(): merged[None].numpy()})[0][0]
            ref["vision_ir"][name] = {"views": views, "out_shape": list(rows.shape),
                                      "image_embeds": summary(torch.from_numpy(rows))}
            print(f"[vision_ir] {name}: {list(rows.shape)}", flush=True)

    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump(ref, f, separators=(",", ":"))
    print("FIXTURES DONE", flush=True)


if __name__ == "__main__":
    main()
