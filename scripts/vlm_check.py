#!/usr/bin/env python3
"""Live image + video checks against a running cascadia VLM server, through the
unmodified `openai` SDK.

It builds its own media, so it needs no files from the server's box: a red
circle on white (PNG, written with the standard library) and a 4 s clip, red
for 2 s then blue (MP4, made with a local ffmpeg). Each check prints
`PASS <name>` or `FAIL <name>: <why>`; the last line is `VLM CHECK ALL PASS`
or `VLM CHECK FAILED <n>`, and the exit code follows it.

  1. /v1/models lists `image` (and `video`, unless --no-video) in input_modalities
  2. a chat image is answered (the circle is red) and counted in prompt_tokens
  3. the same, streamed, ends with a finish_reason
  4. /v1/responses takes it as an `input_image`
  5. a `video_url` clip is described red, then blue      (skipped with --no-video)
  6. an `https://` image is refused with 400 (remote URLs are off by default)
  7. bytes that are not an image are refused with 400

Run:   python3 scripts/vlm_check.py --base-url http://127.0.0.1:8099/v1
Needs: pip install -r tests/conformance/requirements-openai-sdk.txt; ffmpeg on PATH
       (or --video <clip.mp4>, or --no-video). --dry-run builds the media only.
"""
import argparse
import base64
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib


def red_circle_png(size=448) -> bytes:
    rows = []
    c, r2 = size / 2, (size * 0.35) ** 2
    for y in range(size):
        row = bytearray(b"\x00")  # filter: none
        for x in range(size):
            inside = (x - c) ** 2 + (y - c) ** 2 <= r2
            row += b"\xdc\x14\x14" if inside else b"\xff\xff\xff"
        rows.append(bytes(row))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


def red_then_blue_mp4(ffmpeg: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        out = f"{d}/clip.mp4"
        subprocess.run(
            [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=red:s=448x448:d=2:r=10",
             "-f", "lavfi", "-i", "color=blue:s=448x448:d=2:r=10", "-filter_complex",
             "[0][1]concat=n=2:v=1[v]", "-map", "[v]", "-pix_fmt", "yuv420p", out],
            check=True)
        with open(out, "rb") as f:
            return f.read()


def data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def says(text: str, word: str) -> bool:
    return re.search(rf"\b{word}\b", text or "", re.I) is not None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--model", help="served model id (default: the first in /v1/models)")
    ap.add_argument("--video", help="an MP4 to send instead of the generated clip")
    ap.add_argument("--no-video", action="store_true", help="skip the video checks")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per-request seconds")
    ap.add_argument("--dry-run", action="store_true", help="build the media and stop")
    args = ap.parse_args()

    png = red_circle_png()
    clip = None
    if not args.no_video:
        if args.video:
            clip = open(args.video, "rb").read()
        else:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                print("ffmpeg not on PATH: pass --video <clip.mp4> or --no-video")
                return 2
            clip = red_then_blue_mp4(ffmpeg)
    print(f"media: png {len(png)} B, clip {len(clip) if clip else 0} B")
    if args.dry_run:
        print("VLM CHECK DRY RUN")
        return 0

    import openai

    client = openai.OpenAI(base_url=args.base_url, api_key="unused", timeout=args.timeout)
    failed = 0

    def check(name, fn):
        nonlocal failed
        try:
            why = fn()
        except Exception as e:  # a check that raises is a failure, not a crash
            why = f"{type(e).__name__}: {e}"
        if why:
            failed += 1
            print(f"FAIL {name}: {why}", flush=True)
        else:
            print(f"PASS {name}", flush=True)

    models = client.models.list().data
    model = args.model or models[0].id
    print(f"model: {model}")
    img = {"type": "image_url", "image_url": {"url": data_url("image/png", png)}}
    question = "What color is the circle? Answer in one word."

    def chat(content, **kw):
        return client.chat.completions.create(
            model=model, temperature=0, max_tokens=64,
            messages=[{"role": "user", "content": content}], **kw)

    def models_check():
        m = next(m for m in models if m.id == model).model_dump()
        mods = set(m.get("input_modalities") or [])
        want = {"text", "image"} | (set() if args.no_video else {"video"})
        return None if want <= mods else f"input_modalities {sorted(mods)}, want {sorted(want)}"

    def image_check():
        text_only = chat(question).usage.prompt_tokens
        r = chat([{"type": "text", "text": question}, img])
        answer = r.choices[0].message.content
        if not says(answer, "red"):
            return f"answer {answer!r} does not say red"
        if r.usage.prompt_tokens <= text_only + 16:
            return f"prompt_tokens {r.usage.prompt_tokens} vs {text_only} text-only"
        print(f"  image answer {answer!r}, prompt_tokens {r.usage.prompt_tokens}")

    def stream_check():
        text, finish = "", None
        for ev in chat([img, {"type": "text", "text": question}], stream=True):
            if ev.choices:
                text += ev.choices[0].delta.content or ""
                finish = ev.choices[0].finish_reason or finish
        if not says(text, "red"):
            return f"streamed {text!r} does not say red"
        return None if finish else "no finish_reason"

    def responses_check():
        r = client.responses.create(model=model, temperature=0, max_output_tokens=64, input=[
            {"role": "user", "content": [
                {"type": "input_text", "text": question},
                {"type": "input_image", "image_url": data_url("image/png", png)}]}])
        return None if says(r.output_text, "red") else f"output {r.output_text!r}"

    def video_check():
        r = chat([{"type": "video_url", "video_url": {"url": data_url("video/mp4", clip)}},
                  {"type": "text", "text": "The video shows one solid color, then another. "
                                           "Which two colors, in order?"}])
        answer = r.choices[0].message.content or ""
        red, blue = re.search(r"\bred\b", answer, re.I), re.search(r"\bblue\b", answer, re.I)
        if not (red and blue and red.start() < blue.start()):
            return f"answer {answer!r} is not red then blue"
        print(f"  video answer {answer!r}, prompt_tokens {r.usage.prompt_tokens}")

    def refused(content):
        try:
            chat(content)
        except openai.BadRequestError as e:
            msg = str(e.body.get("message") if isinstance(e.body, dict) else e)
            # A server that takes no images refuses every image; that proves nothing here.
            if "does not accept image input" in msg:
                return f"refused for the wrong reason: {msg}"
            print(f"  400: {msg}")
            return None
        return "served, want 400"

    check("models-modalities", models_check)
    check("chat-image", image_check)
    check("chat-image-stream", stream_check)
    check("responses-image", responses_check)
    if clip:
        check("chat-video", video_check)
    check("remote-url-refused", lambda: refused(
        [{"type": "text", "text": question},
         {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]))
    check("bad-image-refused", lambda: refused(
        [{"type": "text", "text": question},
         {"type": "image_url", "image_url": {"url": data_url("image/png", b"not a png")}}]))

    print(f"VLM CHECK ALL PASS" if not failed else f"VLM CHECK FAILED {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
