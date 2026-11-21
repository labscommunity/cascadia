#!/usr/bin/env python3
"""XR1 fixtures from Xiaomi's own processor (`processing_mibot.py`), no model weights.

Builds one RoboCasa365 request the way Xiaomi's `eval_robocasa365/entry.py` does: three cameras of
four frames, each centre-cropped to 0.95 and resized back (bilinear), a 4-row state history of 14
values zero-padded to 60, the task in Xiaomi's chat messages. Saves what the processor makes, and the
cropped frames a client sends, as an uncompressed npz.

    uv venv --python 3.12 /tmp/xr1-venv
    uv pip install --python /tmp/xr1-venv/bin/python torch==2.11.0 torchvision==0.26.0 \
        transformers==4.57.1 numpy pillow
    # PROC: the non-weight files of XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 @ 3a6d0293
    /tmp/xr1-venv/bin/python -I tools/xr1_processor_fixture.py make PROC 32 OUT.npz
    /tmp/xr1-venv/bin/python -I tools/xr1_processor_fixture.py crop IN.npz OUT.npz

`make PROC SIDE OUT` uses seeded random frames of SIDE x SIDE. `crop` adds the cropped frames
(`crop_<camera>`) to a reference npz that holds only the raw `obs_<camera>` histories.
"""
import sys

import numpy as np
from PIL import Image

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
LABELS = ("Left camera", "Right camera", "Wrist camera")
TASK = "pick the apple from the counter and place it in the sink"


def center_crop(image, ratio=0.95):
    """entry.py `center_crop`."""
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    w, h = pil.size
    cw, ch = max(1, int(w * ratio)), max(1, int(h * ratio))
    left, top = (w - cw) // 2, (h - ch) // 2
    return pil.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def make(proc_dir, side, out):
    import torch
    from transformers import AutoProcessor

    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(3, 4, side, side, 3), dtype=np.uint8)
    cropped = np.stack([[np.asarray(center_crop(f)) for f in cam] for cam in raw])
    state14 = rng.uniform(-1.0, 1.0, size=(4, 14)).astype(np.float32)
    state = np.zeros((1, 4, 60), np.float32)
    state[0, :, :14] = state14

    proc = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True, use_fast=False)
    content = []
    for i, label in enumerate(LABELS):
        content.append({"type": "text", "text": ("\n" if i else "") + label + ": "})
        content.append({"type": "video", "video": [Image.fromarray(f) for f in cropped[i]]})
    content.append({"type": "text", "text": f"\n\nGenerate robot actions for the task:\n{TASK} /no_cot"})
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]
    inputs = proc.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",
                                      do_resize=False, state=state, robot_type="robocasa365")
    ids = inputs["input_ids"][0]
    text = proc.tokenizer.decode(ids)
    assert proc.tokenizer(text)["input_ids"] == ids.tolist(), "decode does not round-trip"
    np.savez(
        out,
        task=np.array(TASK),
        prompt=np.array(text),
        frames=cropped,
        state14=state14,
        input_ids=ids.numpy().astype(np.int64),
        attention_mask=inputs["attention_mask"][0].numpy().astype(np.int64),
        pixel_values=inputs["pixel_values_videos"].float().numpy(),
        video_grid_thw=inputs["video_grid_thw"].numpy().astype(np.int64),
        state=inputs["state"].to(torch.float32).numpy(),
    )
    print(f"{out}: {len(ids)} tokens, grid {inputs['video_grid_thw'].tolist()}, "
          f"pixel_values {list(inputs['pixel_values_videos'].shape)}")


def crop(src, out):
    z = dict(np.load(src))
    for cam in CAMERAS:
        z[f"crop_{cam}"] = np.stack([np.asarray(center_crop(f)) for f in z[f"obs_{cam}"]])
    np.savez(out, **z)
    print(f"{out}: added crop_<camera> for {', '.join(CAMERAS)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["make"] and len(sys.argv) == 5:
        make(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif sys.argv[1:2] == ["crop"] and len(sys.argv) == 4:
        crop(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
