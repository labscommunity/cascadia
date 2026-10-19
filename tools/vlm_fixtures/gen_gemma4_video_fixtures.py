#!/usr/bin/env python3
"""Video reference fixtures for the Rust Gemma 4 native video front end.

Runs Hugging Face's own `Gemma4Processor` / `Gemma4VideoProcessor` (no model
weights) on synthetic videos and writes what the Rust port
(`crates/cascadia-engine-openvino/src/gemma4_vision/video.rs`) must reproduce:

  <out>/video_reference.json
    video_processor  the video processor constants
    tokens           `<|image|>`, `<|video|>`, begin/end-of-image and pad ids,
                     and the text config's `use_bidirectional_attention` and
                     `sliding_window`
    videos           per synthetic video: its frames (count, size, the frame
                     times in ms the API would send), whether HF samples it,
                     the frames HF keeps, soft tokens per frame, the resized
                     frame size, and summaries of `pixel_values_videos` and
                     `video_position_ids`
    prompts          per chat: the rendered template (one `<|video|>` per
                     video, one `<|image|>` per image, as the API writes
                     them), the text HF tokenizes after expanding the videos
                     (`expanded_text`), the media in prompt order,
                     `input_ids`, `mm_token_type_ids` and the attention block
                     ids of `get_block_sequence_ids_for_mask`; on a model with
                     `use_bidirectional_attention == "vision"` (26B-A4B / 31B)
                     also `masks`: HF's sliding and full attention masks
                     (`create_masks_for_vision_model`), each row as the
                     `[first, last]` key it may attend (always one range)

Frames are a closed-form pattern (`frame_pixels`, the Qwen video fixtures'
one) the Rust test rebuilds, so no binaries are checked in. Images are the
Gemma 4 image fixtures' PNGs.

A video's frame times reach HF as `fps` and `frames_indices` (HF's timestamps
are `index / fps`); every video but `long40` is pre-sampled
(`do_sample_frames=False`, as the API samples), `long40` lets HF sample 32 of
its 40 frames.

Usage (needs torch, torchvision, transformers>=5.14.1, pillow, numpy):
  python gen_gemma4_video_fixtures.py --model <Gemma 4 dir> [--model <dir> ...] \
      --out <dir> [--name video_reference.json]

A dir only supplies config.json, processor_config.json, the chat template
and the tokenizer (e.g. google/gemma-4-E4B-it without its weights). Several
`--model`s must give the same reference (26B-A4B and 31B share their
processor, tokenizer and attention layout), which is written once with every
source named in `sources`.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torchvision
import transformers
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers.models.gemma4.modeling_gemma4 import (
    create_masks_for_vision_model,
    get_block_sequence_ids_for_mask,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGES = os.path.join(HERE, os.pardir, os.pardir, "crates", "cascadia-engine-openvino",
                              "tests", "fixtures", "gemma4_vision")


def frame_pixels(frames, height, width, seed):
    """RGB8 `[frames, height, width, 3]`: a smooth gradient that moves per
    frame plus a small texture (the Qwen video fixtures' pattern)."""
    f = np.arange(frames).reshape(-1, 1, 1, 1)
    y = np.arange(height).reshape(1, -1, 1, 1)
    x = np.arange(width).reshape(1, 1, -1, 1)
    c = np.arange(3).reshape(1, 1, 1, -1)
    v = x * 3 + y * 5 + f * (17 + seed) + c * 71 + ((x * y + seed) % 23) * 2
    return (v % 256).astype(np.uint8)


# name: (width, height, frame times in ms, seed, let HF sample)
VIDEOS = {
    # The API's 4 s clip at 1 fps, scaled into its 448x448 frame budget.
    "api4": (448, 336, [0, 1000, 2000, 3000], 1, False),
    "odd3": (200, 120, [0, 500, 1000], 2, False),
    "one1": (160, 96, [0], 3, False),
    # Times across minute boundaries, past an hour (mm keeps counting).
    "minutes3": (96, 64, [59_999, 60_000, 3_725_500], 4, False),
    # Tall: one side floors to a single 48 px step.
    "tall2": (64, 400, [0, 2000], 5, False),
    # 40 frames at 2 fps: HF samples 32 of them.
    "long40": (64, 48, [i * 500 for i in range(40)], 6, True),
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
    w, h, ts, _, _ = VIDEOS[name]
    # Times in ms as frame indices at 1000 fps: HF's `index / fps` is then
    # exactly the time in seconds.
    return {"fps": 1000.0, "frames_indices": list(ts), "total_num_frames": len(ts),
            "duration": (ts[-1] + 1) / 1000.0, "width": w, "height": h}


def sampled_metadata(name):
    # `long40`: HF samples by index, so give it its true rate (2 fps).
    w, h, ts, _, _ = VIDEOS[name]
    return {"fps": 2.0, "total_num_frames": len(ts), "duration": len(ts) / 2.0,
            "width": w, "height": h}


def video(name):
    w, h, ts, seed, _ = VIDEOS[name]
    return frame_pixels(len(ts), h, w, seed)


def video_kwargs(names):
    """Processor kwargs for videos in prompt order (sampling is per call, so
    a call holds only pre-sampled videos or only `long40`)."""
    sample = {VIDEOS[n][4] for n in names}
    assert len(sample) == 1, names
    if sample.pop():
        return {"videos": [video(n) for n in names],
                "video_metadata": [sampled_metadata(n) for n in names], "do_sample_frames": True}
    return {"videos": [video(n) for n in names], "video_metadata": [metadata(n) for n in names],
            "do_sample_frames": False}


def chats(imgs):
    vid = lambda n: {"type": "video", "name": n}  # noqa: E731
    img = lambda n: {"type": "image", "image": imgs[n], "name": n}  # noqa: E731
    txt = lambda t: {"type": "text", "text": t}  # noqa: E731
    return {
        "api_like": [{"role": "user", "content": [vid("api4"), txt("What happens?")]}],
        "odd": [{"role": "user", "content": [txt("Watch:"), vid("odd3"), txt("Describe it.")]}],
        "one_frame": [{"role": "user", "content": [vid("one1"), txt("And this?")]}],
        "minutes": [{"role": "user", "content": [vid("minutes3"), txt("When?")]}],
        "tall": [{"role": "user", "content": [vid("tall2"), txt("Tall.")]}],
        "sampled": [{"role": "user", "content": [vid("long40"), txt("Long.")]}],
        "two_videos": [{"role": "user", "content": [
            txt("A:"), vid("odd3"), txt("B:"), vid("one1"), txt("Compare.")]}],
        "video_and_images": [
            {"role": "user", "content": [img("tiny_40x30"), txt("What?")]},
            {"role": "assistant", "content": [txt("A dot.")]},
            {"role": "user", "content": [vid("odd3"), txt("then"), img("landscape_640x360"),
                                         vid("one1"), txt("Now?")]}],
    }


def mask_ranges(config, mm):
    """HF's sliding / full attention masks for a prompt of `mm_token_type_ids`
    `mm` (`[1, L]`), each row as the one `[first, last]` key range it sees."""
    text = config.get_text_config()
    text._attn_implementation = "sdpa"
    n = mm.shape[1]
    out = create_masks_for_vision_model(
        config=text, inputs_embeds=torch.zeros(1, n, 8), attention_mask=torch.ones(1, n, dtype=torch.long),
        past_key_values=None, position_ids=torch.arange(n).unsqueeze(0),
        block_sequence_ids=get_block_sequence_ids_for_mask(mm, device="cpu"))
    ranges = {}
    for kind, m in out.items():
        if m is None:  # sdpa leaves a plain causal mask implicit
            m = torch.ones(n, n, dtype=torch.bool).tril()
        m = m.reshape(m.shape[-2], m.shape[-1]).to(torch.bool)
        rows = []
        for q in range(n):
            keys = torch.nonzero(m[q]).flatten().tolist()
            assert keys == list(range(keys[0], keys[-1] + 1)), (kind, q)
            rows.append([keys[0], keys[-1]])
        ranges[kind] = rows
    return ranges


def reference(model, images_dir):
    config = AutoConfig.from_pretrained(model)
    processor = AutoProcessor.from_pretrained(model)
    vp = processor.video_processor
    tok = processor.tokenizer
    imgs = {os.path.splitext(os.path.basename(p))[0]: Image.open(p).convert("RGB")
            for p in sorted(glob.glob(os.path.join(images_dir, "*.png")))}

    ref = {
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
        "model_type": config.model_type,
        "video_processor": {k: list(v) if isinstance(v, tuple) else v
                            for k, v in vp.to_dict().items()
                            if isinstance(v, (int, float, str, list, tuple, dict, bool))},
        "tokens": {"image": processor.image_token_id, "video": processor.video_token_id,
                   "boi": tok.convert_tokens_to_ids(processor.boi_token),
                   "eoi": tok.convert_tokens_to_ids(processor.eoi_token),
                   "pad": config.text_config.pad_token_id,
                   "use_bidirectional_attention":
                       getattr(config.text_config, "use_bidirectional_attention", None),
                   "sliding_window": config.text_config.sliding_window},
        "videos": {}, "prompts": {},
    }
    assert config.video_token_id == processor.video_token_id

    for name, (w, h, ts, _, sample) in VIDEOS.items():
        o = vp(return_tensors="pt", return_metadata=True, **video_kwargs([name]))
        meta = o["video_metadata"][0]
        kept = [int(i) for i in meta.frames_indices]
        pos = o["video_position_ids"]
        real = int((pos[0, :, 0] >= 0).sum())
        cols, rows = int(pos[0, real - 1, 0]) + 1, int(pos[0, real - 1, 1]) + 1
        if sample:
            # Sampled indices are frame numbers at 2 fps; the times follow.
            kept_ms = [ts[i] for i in kept]
        else:
            kept_ms = kept  # indices are the times at 1000 fps
            kept = [ts.index(i) for i in kept]
        ref["videos"][name] = {
            "frames": len(ts), "width": w, "height": h, "timestamps_ms": ts, "sampled": sample,
            "kept_frames": kept, "kept_ms": kept_ms,
            "timestamps": [f"{int(s // 60):02d}:{int(s % 60):02d}" for s in meta.timestamps],
            "soft_tokens_per_frame": int(o["num_soft_tokens_per_video"][0]),
            "resized_hw": [rows * vp.patch_size, cols * vp.patch_size],
            "num_patches": real,
            "pixel_values_videos": summary(o["pixel_values_videos"]),
            "video_position_ids": summary(pos),
        }
        print(f"[video {name}] kept={len(kept)} soft/frame={ref['videos'][name]['soft_tokens_per_frame']} "
              f"resized={ref['videos'][name]['resized_hw']}")

    for p_name, messages in chats(imgs).items():
        plain = [{"role": m["role"], "content": [{k: v for k, v in c.items() if k in ("type", "text")}
                                                 for c in m["content"]]} for m in messages]
        text = processor.apply_chat_template(plain, add_generation_prompt=True, tokenize=False)
        media = [c for m in messages for c in m["content"] if c["type"] in ("image", "video")]
        images = [c["image"] for c in media if c["type"] == "image"]
        names = [c["name"] for c in media if c["type"] == "video"]
        kw = {"images": images} if images else {}
        if names:
            kw.update(video_kwargs(names))
        o = processor(text=[text], return_tensors="pt", **kw)
        ids = o["input_ids"][0].tolist()
        # The text HF tokenized, recovered from its ids and checked to
        # tokenize back to them.
        expanded = tok.decode(ids, skip_special_tokens=False)
        assert tok(expanded, add_special_tokens=False)["input_ids"] == ids, p_name
        mm = o["mm_token_type_ids"]
        blocks = get_block_sequence_ids_for_mask(mm, device="cpu")[0].tolist()
        ref["prompts"][p_name] = {
            "template_text": text,
            "expanded_text": expanded,
            "media": [{"kind": c["type"], "name": c["name"]} for c in media],
            "input_ids": ids,
            "mm_token_type_ids": mm[0].tolist(),
            "block_ids": blocks,
        }
        if ref["tokens"]["use_bidirectional_attention"] == "vision":
            ref["prompts"][p_name]["masks"] = mask_ranges(config, mm)
        print(f"[{p_name}] tokens={len(ids)} blocks={max(blocks) + 1}")
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, action="append")
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="video_reference.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    refs = [reference(m, args.images) for m in args.model]
    for m, r in zip(args.model[1:], refs[1:]):
        assert r == refs[0], f"{m} differs from {args.model[0]}"
    ref = refs[0]
    ref["sources"] = [os.path.basename(os.path.normpath(m)) for m in args.model]
    with open(os.path.join(args.out, args.name), "w") as f:
        json.dump(ref, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
