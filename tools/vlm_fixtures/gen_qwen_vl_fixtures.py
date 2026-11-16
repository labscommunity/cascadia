#!/usr/bin/env python3
"""Reference fixtures for the Rust Qwen2.5-VL / Qwen3-VL front end on ov-runtime.

Runs Hugging Face's own processor, `get_rope_index` and the vision host-input
helpers (no model weights) and writes what the Rust port must reproduce:

  <out>/reference.json   per family (`qwen2_5_vl`, `qwen3_vl`):
    processor   the image processor constants
    images      per input image, the PIL (slow) and torchvision (fast)
                processors' `grid_thw` and a summary of `pixel_values`
    prompts     per chat: the expanded `input_ids`, image grids, HF's 3-row
                `position_ids` and `rope_delta`
    encoder     per grid, the vision IRs' host inputs: Qwen2.5-VL's
                `window_index` / `cu_window_seqlens` and rotary table,
                Qwen3-VL's bilinear position indices and weights and rotary
                table, and `cu_seqlens`

The input images are the Qwen3.5 fixtures' PNGs (`--images`, default
`crates/cascadia-engine-openvino/tests/fixtures/qwen35_vision`), so no new
binaries are checked in. Pixel tensors are summarised (shape, sums, first and
last 64 values, every 1009th value) rather than stored.

Usage (needs torch, torchvision, transformers==5.14.1, pillow, numpy):
  python gen_qwen_vl_fixtures.py --qwen2-5-vl <OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov dir> \
      --qwen3-vl <OpenVINO/Qwen3-VL-4B-Instruct-int4-ov dir> --out <dir>

The model dirs only supply config.json, the processor configs, the chat
template and the tokenizer; no IR or weight file is read.

  --run-ir FAMILY   instead drive that family's vision IRs (needs openvino and
                    the export with its weights, so run it where the model is)
                    with HF's preprocessing and host inputs, and write
                    <out>/vision_ir_<FAMILY>.json: per image the grid and a
                    summary of the image rows (and Qwen3-VL's DeepStack rows).
                    `tests/qwen_vl_vision_ir.rs` compares the Rust chain to it.
"""
import argparse
import glob
import json
import os
import types

import numpy as np
import torch
import torchvision
import transformers
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
from transformers import vision_utils as vu
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as q25
from transformers.models.qwen3_vl import modeling_qwen3_vl as q3

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGES = os.path.join(HERE, os.pardir, os.pardir, "crates", "cascadia-engine-openvino",
                              "tests", "fixtures", "qwen35_vision")

# A strong downscale exercises the antialiased (widened) kernel path.
LOW_BUDGET = {"big_1600x1200_maxpx262144": ("big_1600x1200", 262144)}


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


def fake_model(cls, config):
    """Just enough of the VLM model class for get_rope_index."""
    ns = types.SimpleNamespace(config=config)
    ns.get_vision_position_ids = types.MethodType(cls.get_vision_position_ids, ns)
    return ns


def prompts_for(imgs):
    q = "What is in this picture? Answer in one sentence."
    return {
        "single": [{"role": "user", "content": [
            {"type": "image", "image": imgs["square_448"]}, {"type": "text", "text": q}]}],
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


def family(model_dir, name, imgs):
    config = AutoConfig.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    fast = AutoImageProcessor.from_pretrained(model_dir, backend="torchvision")
    slow = AutoImageProcessor.from_pretrained(model_dir, backend="pil")
    cls = q25.Qwen2_5_VLModel if name == "qwen2_5_vl" else q3.Qwen3VLModel
    model = fake_model(cls, config)
    image_token, video_token = config.image_token_id, config.video_token_id
    vc = config.vision_config
    ref = {
        "model_type": config.model_type,
        "processor": {k: list(v) if isinstance(v, tuple) else v for k, v in fast.to_dict().items()
                      if isinstance(v, (int, float, str, list, tuple, dict, bool))},
        "token_ids": {"image": image_token, "video": video_token,
                      "vision_start": config.vision_start_token_id,
                      "vision_end": config.vision_end_token_id},
        "spatial_merge_size": vc.spatial_merge_size,
        "images": {}, "prompts": {},
    }

    for img_name, img in imgs.items():
        entry = {"size": list(img.size)}
        outs = {}
        for kind, proc in (("fast", fast), ("slow", slow)):
            o = proc(images=[img], return_tensors="pt")
            outs[kind] = o["pixel_values"]
            entry[kind] = {"grid_thw": o["image_grid_thw"].tolist()[0],
                           "pixel_values": summary(o["pixel_values"])}
        d = (outs["fast"].double() - outs["slow"].double()).abs()
        entry["fast_vs_slow"] = {"max_abs": float(d.max()), "mean_abs": float(d.mean())}
        ref["images"][img_name] = entry
    for img_name, (src, max_px) in LOW_BUDGET.items():
        img = imgs[src]
        entry = {"size": list(img.size), "source": src, "max_pixels": max_px}
        for kind, proc in (("fast", fast), ("slow", slow)):
            o = proc(images=[img], max_pixels=max_px, min_pixels=proc.size["shortest_edge"],
                     return_tensors="pt")
            entry[kind] = {"grid_thw": o["image_grid_thw"].tolist()[0],
                           "pixel_values": summary(o["pixel_values"])}
        ref["images"][img_name] = entry

    for p_name, messages in prompts_for(imgs).items():
        o = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                          add_generation_prompt=True, return_tensors="pt")
        ids = o["input_ids"]
        mm = torch.zeros_like(ids, dtype=torch.int32)
        mm[ids == image_token] = 1
        mm[ids == video_token] = 2
        igrid = o.get("image_grid_thw")
        pos, delta = cls.get_rope_index(model, ids, mm, image_grid_thw=igrid,
                                        attention_mask=o.get("attention_mask"))
        # The prompt as the API hands it to rank 0: one pad per image.
        flat, prev = [], None
        for t in ids[0].tolist():
            if not (t == image_token and prev == image_token):
                flat.append(t)
            prev = t
        ref["prompts"][p_name] = {
            "prompt_ids": flat,
            "input_ids": ids[0].tolist(),
            "image_sizes": [list(c["image"].size) for m in messages
                            for c in (m["content"] if isinstance(m["content"], list) else [])
                            if c.get("type") == "image"],
            "image_grid_thw": igrid.tolist() if igrid is not None else [],
            "position_ids": pos[:, 0, :].tolist(),
            "rope_delta": int(delta.flatten()[0]),
        }
        print(f"[{name}/{p_name}] tokens={ids.shape[1]} grids={ref['prompts'][p_name]['image_grid_thw']} "
              f"rope_delta={ref['prompts'][p_name]['rope_delta']}")

    head_dim = vc.hidden_size // vc.num_heads
    grids = {n: e["slow"]["grid_thw"] for n, e in ref["images"].items()}
    enc = {"hidden_size": vc.hidden_size, "out_hidden_size": vc.out_hidden_size,
           "rotary_dim": head_dim // 2, "grids": {}}
    if name == "qwen2_5_vl":
        rotary = q25.Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        enc.update(window_size=vc.window_size, patch_size=vc.patch_size)
    else:
        rotary = q3.Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        side = int(vc.num_position_embeddings ** 0.5)
        enc.update(num_grid_per_side=side, deepstack_visual_indexes=list(vc.deepstack_visual_indexes))
    for g_name, g in grids.items():
        gt = torch.tensor([g])
        rot = rotary(vu.get_vision_position_ids(gt, vc.spatial_merge_size))
        e = {"grid_thw": g, "rotary_pos_emb": summary(rot),
             "cu_seqlens": vu.get_vision_cu_seqlens(gt).tolist()}
        if name == "qwen2_5_vl":
            wi, cu = vu.get_vision_window_index(gt, vc.spatial_merge_size, vc.window_size, vc.patch_size)
            e["window_index"] = wi.tolist()
            e["cu_window_seqlens"] = cu.tolist()
        else:
            idx, wts = vu.get_vision_bilinear_indices_and_weights(gt, side, vc.spatial_merge_size)
            e["indices"] = idx.flatten().tolist() if idx.shape[1] <= 3000 else summary(idx)
            e["weights"] = summary(wts)
        enc["grids"][g_name] = e
    ref["encoder"] = enc
    return ref


def run_vision_irs(model_dir, name, imgs):
    """HF's preprocessing and host inputs driving the export's vision IRs."""
    import openvino as ov
    config = AutoConfig.from_pretrained(model_dir)
    proc = AutoImageProcessor.from_pretrained(model_dir, backend="pil")
    vc = config.vision_config
    head_dim = vc.hidden_size // vc.num_heads
    core = ov.Core()
    ir = lambda s: core.compile_model(  # noqa: E731
        os.path.join(model_dir, f"openvino_vision_embeddings{s}_model.xml"), "CPU")
    patch, merger = ir(""), ir("_merger")
    out = {}
    for img_name in ("tiny_40x30", "square_448", "landscape_640x360"):
        o = proc(images=[imgs[img_name]], return_tensors="pt")
        pv, g = o["pixel_values"], o["image_grid_thw"]
        pe = torch.from_numpy(patch({"hidden_states": pv.numpy()})[0])
        pos_ids = vu.get_vision_position_ids(g, vc.spatial_merge_size)
        cu = vu.get_vision_cu_seqlens(g).tolist()
        n = pe.shape[0]
        mask = torch.full((1, n, n), float("-inf"))
        for a, b in zip(cu[:-1], cu[1:]):
            mask[0, a:b, a:b] = 0
        if name == "qwen2_5_vl":
            rot = q25.Qwen2_5_VisionRotaryEmbedding(head_dim // 2)(pos_ids)
            wi, cw = vu.get_vision_window_index(g, vc.spatial_merge_size, vc.window_size,
                                                vc.patch_size)
            wmask = torch.full((1, n, n), float("-inf"))
            cw = cw.tolist()
            for a, b in zip(cw[:-1], cw[1:]):
                wmask[0, a:b, a:b] = 0
            res = merger({"hidden_states": pe.numpy(), "attention_mask": mask.numpy(),
                          "window_attention_mask": wmask.numpy(), "window_index": wi.numpy(),
                          "rotary_pos_emb": rot.numpy()})
            entry = {"image_embeds": summary(torch.from_numpy(res[0]))}
        else:
            side = int(vc.num_position_embeddings ** 0.5)
            pos_ir = ir("_pos")
            idx, wts = vu.get_vision_bilinear_indices_and_weights(g, side, vc.spatial_merge_size)
            corners = torch.from_numpy(pos_ir({"input": idx.numpy()})[0])
            hidden = pe + (corners * wts[:, :, None]).sum(0)
            rot = q3.Qwen3VLVisionRotaryEmbedding(head_dim // 2)(pos_ids)
            res = merger({"hidden_states": hidden.numpy(), "attention_mask": mask.numpy(),
                          "rotary_pos_emb": rot.numpy()})
            by = {next(iter(k.get_names())): v for k, v in res.items()}
            entry = {"image_embeds": summary(torch.from_numpy(by["last_hidden_state"])),
                     "deepstack": summary(torch.from_numpy(by["deepstack_feature_lists"]))}
        entry["grid_thw"] = g.tolist()[0]
        out[img_name] = entry
        print(f"[vision_ir {name}] {img_name}: grid={entry['grid_thw']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen2-5-vl")
    ap.add_argument("--qwen3-vl")
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-ir", choices=("qwen2_5_vl", "qwen3_vl"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    imgs = {os.path.splitext(os.path.basename(p))[0]: Image.open(p).convert("RGB")
            for p in sorted(glob.glob(os.path.join(args.images, "*.png")))}
    if args.run_ir:
        model = args.qwen2_5_vl if args.run_ir == "qwen2_5_vl" else args.qwen3_vl
        if not model:
            raise SystemExit(f"--run-ir {args.run_ir} needs that family's model dir")
        ref = run_vision_irs(model, args.run_ir, imgs)
        with open(os.path.join(args.out, f"vision_ir_{args.run_ir}.json"), "w") as f:
            json.dump(ref, f, separators=(",", ":"))
        return
    if not (args.qwen2_5_vl and args.qwen3_vl):
        raise SystemExit("--qwen2-5-vl and --qwen3-vl are both required")
    ref = {
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
        "families": {
            "qwen2_5_vl": family(args.qwen2_5_vl, "qwen2_5_vl", imgs),
            "qwen3_vl": family(args.qwen3_vl, "qwen3_vl", imgs),
        },
    }
    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump(ref, f, separators=(",", ":"))
    for fam, r in ref["families"].items():
        for n, e in r["images"].items():
            extra = f" fast_vs_slow max={e['fast_vs_slow']['max_abs']:.4f}" if "fast_vs_slow" in e else ""
            print(f"{fam} {n}: grid slow={e['slow']['grid_thw']} fast={e['fast']['grid_thw']}{extra}")


if __name__ == "__main__":
    main()
