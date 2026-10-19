#!/usr/bin/env python3
"""Video reference fixtures for the Rust Qwen VL front ends (native video).

Runs Hugging Face's own processor, video processor and `get_rope_index` on
synthetic videos (no model weights) and writes what the Rust port must
reproduce for Qwen2.5-VL, Qwen3-VL and Qwen3.5/3.6:

  <out>/video_reference.json   per family (`qwen2_5_vl`, `qwen3_vl`,
                               `qwen3_5_moe`):
    video_processor  the video processor constants
    videos           per synthetic video: its frames (count, size, fps,
                     frame indices, the frame times in ms the API would
                     send), the video processor's `video_grid_thw` and a
                     summary of `pixel_values_videos`
    prompts          per chat: the rendered template (one
                     `<|vision_start|><|video_pad|><|vision_end|>` per video,
                     as the API writes it), HF's expanded `input_ids`, those
                     ids with each pad run collapsed to one pad and as text,
                     the media in prompt order, the grids,
                     `second_per_grid_ts` (Qwen2.5-VL), HF's 3-row
                     `position_ids` and `rope_delta`

Frames are a closed-form pattern (`frame_pixels`) the Rust test rebuilds, so
no binaries are checked in. Images are the Qwen3.5 fixtures' PNGs.

Usage (needs torch, torchvision, transformers==5.14.1, pillow, numpy):
  python gen_qwen_vl_video_fixtures.py --qwen2-5-vl <Qwen2.5-VL export> \
      --qwen3-vl <Qwen3-VL export> --qwen3-5 <Qwen3.5/3.6 export or HF dir> --out <dir>

Each dir only supplies config.json, the processor configs, the chat template
and the tokenizer.
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
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as q25
from transformers.models.qwen3_vl import modeling_qwen3_vl as q3
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q35

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGES = os.path.join(HERE, os.pardir, os.pardir, "crates", "cascadia-engine-openvino",
                              "tests", "fixtures", "qwen35_vision")


def frame_pixels(frames, height, width, seed):
    """RGB8 `[frames, height, width, 3]`: a smooth gradient that moves per
    frame plus a small texture, so resizes and frame pairs both matter."""
    f = np.arange(frames).reshape(-1, 1, 1, 1)
    y = np.arange(height).reshape(1, -1, 1, 1)
    x = np.arange(width).reshape(1, 1, -1, 1)
    c = np.arange(3).reshape(1, 1, 1, -1)
    v = x * 3 + y * 5 + f * (17 + seed) + c * 71 + ((x * y + seed) % 23) * 2
    return (v % 256).astype(np.uint8)


# name: (frames, width, height, fps, seed)
VIDEOS = {
    "odd5": (5, 200, 120, 2.0, 1),
    "even4": (4, 96, 64, 2.0, 2),
    "one1": (1, 160, 96, 2.0, 3),
    "api8": (8, 448, 252, 1.0, 4),
    "slow4": (4, 96, 64, 0.5, 5),
    "frac4": (4, 96, 64, 1.5, 6),
}


def summary(t):
    a = t.detach().to(torch.float64).flatten().numpy()
    return {
        "shape": list(t.shape),
        "sum": float(a.sum()),
        "abs_sum": float(np.abs(a).sum()),
        "head": [float(v) for v in a[:64]],
        "tail": [float(v) for v in a[-64:]],
        "stride_1009": [float(v) for v in a[::1009]],
    }


def metadata(name):
    n, w, h, fps, _ = VIDEOS[name]
    return {"fps": fps, "frames_indices": list(range(n)), "total_num_frames": n,
            "duration": n / fps, "width": w, "height": h}


def video(name):
    n, w, h, _, seed = VIDEOS[name]
    return frame_pixels(n, h, w, seed)


def fake_model(cls, config):
    """Just enough of the VLM model class for get_rope_index."""
    ns = types.SimpleNamespace(config=config)
    ns.get_vision_position_ids = types.MethodType(cls.get_vision_position_ids, ns)
    return ns


def chats(imgs):
    vid = lambda n: {"type": "video", "name": n}  # noqa: E731
    img = lambda n: {"type": "image", "image": imgs[n], "name": n}  # noqa: E731
    txt = lambda t: {"type": "text", "text": t}  # noqa: E731
    return {
        "video_odd": [{"role": "user", "content": [vid("odd5"), txt("What happens?")]}],
        "video_even": [{"role": "user", "content": [vid("even4"), txt("Describe it.")]}],
        "one_frame": [{"role": "user", "content": [vid("one1"), txt("And this?")]}],
        "api_like": [{"role": "user", "content": [vid("api8"), txt("What is shown?")]}],
        "slow": [{"role": "user", "content": [vid("slow4"), txt("Slow.")]}],
        "fractional_fps": [{"role": "user", "content": [vid("frac4"), txt("Frac.")]}],
        "two_videos": [{"role": "user", "content": [
            txt("A:"), vid("odd5"), txt("B:"), vid("slow4"), txt("Compare.")]}],
        "video_and_images": [
            {"role": "user", "content": [img("tiny_40x30"), txt("What?")]},
            {"role": "assistant", "content": [txt("A dot.")]},
            {"role": "user", "content": [vid("even4"), txt("then"), img("landscape_640x360"),
                                         vid("one1"), txt("Now?")]}],
    }


def collapse(ids, pads):
    out, prev = [], None
    for t in ids:
        if not (t in pads and t == prev):
            out.append(t)
        prev = t
    return out


def family(model_dir, imgs):
    config = AutoConfig.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    vp = processor.video_processor
    name = config.model_type
    cls = {"qwen2_5_vl": q25.Qwen2_5_VLModel, "qwen3_vl": q3.Qwen3VLModel,
           "qwen3_5_moe": q35.Qwen3_5MoeModel}[name]
    model = fake_model(cls, config)
    image_token, video_token = config.image_token_id, config.video_token_id
    vc = config.vision_config
    ref = {
        "model_type": name,
        "video_processor": {k: list(v) if isinstance(v, tuple) else v
                            for k, v in vp.to_dict().items()
                            if isinstance(v, (int, float, str, list, tuple, dict, bool))},
        "token_ids": {"image": image_token, "video": video_token,
                      "vision_start": config.vision_start_token_id,
                      "vision_end": config.vision_end_token_id},
        "spatial_merge_size": vc.spatial_merge_size,
        "tokens_per_second": getattr(vc, "tokens_per_second", None),
        "videos": {}, "prompts": {},
    }
    for v_name, (n, w, h, fps, _) in VIDEOS.items():
        o = vp(videos=[video(v_name)], video_metadata=[metadata(v_name)], do_sample_frames=False,
               return_tensors="pt")
        ref["videos"][v_name] = {
            "frames": n, "width": w, "height": h, "fps": fps,
            "timestamps_ms": [round(i * 1000 / fps) for i in range(n)],
            "video_grid_thw": o["video_grid_thw"].tolist()[0],
            "pixel_values_videos": summary(o["pixel_values_videos"]),
        }

    for p_name, messages in chats(imgs).items():
        # The template only needs the part types; the processor gets the media.
        plain = [{"role": m["role"], "content": [{k: v for k, v in c.items() if k == "type" or
                                                  k == "text"} for c in m["content"]]}
                 for m in messages]
        text = processor.apply_chat_template(plain, add_generation_prompt=True, tokenize=False)
        media = [c for m in messages for c in m["content"] if c["type"] in ("image", "video")]
        images = [c["image"] for c in media if c["type"] == "image"]
        videos = [video(c["name"]) for c in media if c["type"] == "video"]
        metas = [metadata(c["name"]) for c in media if c["type"] == "video"]
        kw = {"images": images} if images else {}
        if videos:
            kw.update(videos=videos, video_metadata=metas, do_sample_frames=False)
        o = processor(text=[text], return_tensors="pt", **kw)
        ids = o["input_ids"]
        mm = torch.zeros_like(ids, dtype=torch.int32)
        mm[ids == image_token] = 1
        mm[ids == video_token] = 2
        igrid, vgrid = o.get("image_grid_thw"), o.get("video_grid_thw")
        extra = {}
        if name == "qwen2_5_vl" and vgrid is not None:
            extra["second_per_grid_ts"] = o["second_per_grid_ts"]
        pos, delta = cls.get_rope_index(model, ids, mm, image_grid_thw=igrid, video_grid_thw=vgrid,
                                        attention_mask=o.get("attention_mask"), **extra)
        flat = collapse(ids[0].tolist(), (image_token, video_token))
        entry = {
            "template_text": text,
            "media": [{"kind": c["type"], "name": c["name"]} for c in media],
            "prompt_ids": flat,
            "prompt_text": processor.tokenizer.decode(flat),
            "input_ids": ids[0].tolist(),
            "image_grid_thw": igrid.tolist() if igrid is not None else [],
            "video_grid_thw": vgrid.tolist() if vgrid is not None else [],
            "position_ids": pos[:, 0, :].tolist(),
            "rope_delta": int(delta.flatten()[0]),
        }
        if "second_per_grid_ts" in extra:
            spg = extra["second_per_grid_ts"]
            entry["second_per_grid_ts"] = [float(x) for x in spg]
        ref["prompts"][p_name] = entry
        print(f"[{name}/{p_name}] tokens={ids.shape[1]} video_grids={entry['video_grid_thw']} "
              f"rope_delta={entry['rope_delta']}")
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen2-5-vl", required=True)
    ap.add_argument("--qwen3-vl", required=True)
    ap.add_argument("--qwen3-5", required=True)
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    imgs = {os.path.splitext(os.path.basename(p))[0]: Image.open(p).convert("RGB")
            for p in sorted(glob.glob(os.path.join(args.images, "*.png")))}
    ref = {
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
        "families": {},
    }
    for d in (args.qwen2_5_vl, args.qwen3_vl, args.qwen3_5):
        f = family(d, imgs)
        ref["families"][f["model_type"]] = f
    with open(os.path.join(args.out, "video_reference.json"), "w") as f:
        json.dump(ref, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
