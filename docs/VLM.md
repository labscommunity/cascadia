# Image input (VLM)

Cascadia takes images the way the OpenAI API does: `image_url` content parts in
`/v1/chat/completions` messages. This page says which models and engines take them, how
to export each one, and what the limits are. The design and its verification history are
in [rfcs/vlm-support.md](rfcs/vlm-support.md). Video is served as sampled frames, or
whole where the engine takes it natively (see [Video](#video)); audio is not supported.

## The request

```json
{"model": "m", "messages": [{"role": "user", "content": [
  {"type": "text", "text": "What is in this picture?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "auto"}}
]}]}
```

- `data:` URLs (PNG, JPEG, WEBP, GIF) work by default. `https://` URLs only with
  `--api-allow-image-urls` (public addresses only, each redirect re-checked).
- Up to 8 images per request (`--api-max-images`). An image over about 1 MP is downscaled (`detail: "low"`:
  512x512; `"original"` is served as `"high"`). The chat and Responses bodies may carry
  20 MiB of images on top of the text limit; the text itself (everything but image URLs)
  stays held to that limit (`413`), and `/v1/completions` gets no image allowance.
- A request's images may declare 100 MP of source pixels together (two images at the
  50 MP per-image cap), checked from their headers before any decode. Two images decode
  at once across the server; a request that waits more than 10 s for a decode slot gets
  the capacity `503` with `Retry-After`.
- Images can sit anywhere in a user message and in earlier turns.
- `/v1/models` lists `input_modalities` (`["text", "image"]` when images are accepted,
  `["text", "image", "video"]` when ffmpeg was found too).
- Errors are OpenAI-shaped: a model or tree without image input answers `400`, an
  undecodable image `400`, a prompt that does not fit the context `413`
  (`context_length_exceeded`). Text that spells a model's own image tag (for example
  `<|image|>` or `[IMG]`) is refused with `400`. On `ov-genai` that covers GenAI's
  universal `<ov_genai_image_N>` / `<ov_genai_video_N>` and the export's native image and
  video tags (by `config.json` `model_type`; every known one for an unlisted type), which
  GenAI would otherwise read as media slots.
- A text-only content array joins its text parts back to back, as an image message's do
  (and as Hugging Face's templates write one). The deprecated `function` role is served as
  `tool`.

### The Responses API

`POST /v1/responses` takes the same images as `input_image` parts and serves them through
the chat path, so every limit and error above applies unchanged:

```json
{"model": "m", "instructions": "Answer in one line.", "input": [{"role": "user", "content": [
  {"type": "input_text", "text": "What is in this picture?"},
  {"type": "input_image", "image_url": "data:image/png;base64,...", "detail": "auto"}
]}]}
```

`input` may also be a plain string; `instructions` becomes a system message;
`max_output_tokens`, `temperature`, `top_p` and `reasoning.effort` map onto their chat
counterparts, and `stream: true` sends OpenAI's event sequence (`response.created` through
`response.completed`, or `response.incomplete` at the token cap and for a cancelled
turn, whose status is `cancelled`). Cascadia keeps no state
and serves no tools here: `store: true`, `previous_response_id`, `conversation`,
`background`, `tools`, `include`, `prompt`, `stop`, structured `text.format`,
`reasoning.summary`, `truncation: "auto"`, `file_id` images, `input_file` and
`input_audio` parts and non-message input items are refused with a `400` whose `param`
names the field.

## Video

OpenAI's APIs take no video. Their documented way is to sample frames and send them as
`image_url` parts, which works here unchanged: raise `--api-max-images` for more frames.

Cascadia also takes the `video_url` extension vLLM, SGLang and OpenRouter use, and samples
the frames itself:

```json
{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,...", "detail": "auto"}}
```

and in `/v1/responses` OpenRouter's `{"type": "input_video", "video_url": "data:video/mp4;base64,..."}`
(`{"url": ...}` is accepted too; `file_id` is a `400`, code `video_file_id_not_supported`).

- `data:video/mp4`, `video/webm`, `video/quicktime` and `video/x-matroska` URLs; `https://`
  only with `--api-allow-image-urls`, under the same checks as images (a fetched video may
  be typed `video/*`, `application/octet-stream` or not at all: its bytes are checked). The
  container must be MP4/MOV or Matroska/WebM; anything else is a `400`. A WebM that states
  no duration (as a browser's MediaRecorder writes it) has it read from its packets (first
  to last); timestamps in labels count from the clip's start, whatever its first timestamp.
- The video becomes `ceil(duration x --api-video-fps)` frames (default 1 per second), at
  most `--api-max-video-frames` (default 16), spread uniformly from 0 s. Each frame enters
  the prompt where the video was, as `Frame k (t s): ` followed by one image (`Video j frame
  k (t s): ` when a request has several videos), exactly as if the caller had sent those
  parts. Frames do not count against `--api-max-images`.
- Frame size: `detail: "auto"` (the default) fits each frame to
  `--api-max-video-frame-pixels` (default 448x448), `"low"` to a quarter of that (224x224,
  never more than the image low-detail budget), the smallest; `"high"` to the larger of that
  and the image budget (about 1 MP). Aspect ratio is kept.
- As frames, each costs one image's tokens on that family: Gemma 3, 256 per frame plus its
  markers (16 frames are over 4 K tokens); Qwen3-VL and Qwen3.5 (16x16 patches), about 196
  for a 448x448 frame; Qwen2.5-VL (14x14 patches), about 256. A video that does not fit the
  context gets the usual `413` `context_length_exceeded`.
- Limits, all checked from the container before any frame is decoded:
  `--api-max-videos` (default 1 per request; 0 turns video off), `--api-max-video-seconds`
  (600), `--api-max-video-source-pixels` (3840x2160 per source frame), and a fixed work
  bound of 600 s of 1080p30 per video (frames x pixels), so 4K30 is held to 150 s. The
  frames are counted from the container's packets by a demux-only scan, so the bound is
  only as good as the container's packets: it is advisory against a crafted file, which
  the source-frame cap, the decode deadline and the two decode slots still bound.
  `--api-max-video-mb` (32 MiB per request, at most 1024; on top of the image allowance,
  and the chat and Responses body limits grow by it only when video is on).
  `--api-video-decode-timeout` (30 s for all of a request's videos; past it ffmpeg is
  killed and the request gets a `400` asking for a shorter or lower-resolution clip).
  Videos decode under the request's one decode slot.
- A clip with more than 32 source frames per sampled frame (past about 16 s at 30 fps) and
  at least one keyframe per sampled frame is read with one seek per frame instead of one
  pass: 60 s of 4K30 with a 2 s GOP takes about 3.6 s instead of 9 s on an M-series Mac. A
  clip with rare keyframes (a browser recording, `keyint=infinite`) is read in one pass,
  since each seek would decode most of it. ffmpeg's memory follows the source frame size,
  about 170 MB at 4K; it is not capped otherwise.
- Decoding needs ffmpeg and ffprobe on the server: `--api-ffmpeg PATH`, else `ffmpeg` on
  `PATH` (ffprobe beside it or on `PATH`). A decode-only LGPL build is enough. ffmpeg runs
  as a subprocess on a private temp file (removed afterwards), restricted to local files,
  with the container forced. Without ffmpeg a video part is a `400` naming `--api-ffmpeg`,
  and `/v1/models` does not list `video`.
- Everything above holds on every engine; what differs is how the frames reach the model.

### Native video (ov-genai)

On the `ov-genai` engine, an export whose `config.json` `model_type` GenAI preprocesses
video for gets each video whole: its sampled frames go to GenAI as one `[T, H, W, 3]`
video, behind `<ov_genai_video_j>` where the caller put it (videos numbered from 0, apart
from the images), with no per-frame labels. The vision encoder then sees the frames
together, with their times, and spends far fewer tokens on them. Measured end to end
through `video_url` on GenAI 2026.4.1 (Windows, x86): a 4 s clip (a red circle for 2 s,
then a blue square), sampled at 1 fps to 4 frames of 448x448, every frame passed as
sampled; each model described it correctly.

| `model_type` (export) | native video | prompt tokens |
|---|---|---|
| `gemma4` (Gemma 4 26B-A4B) | yes | 336 (as frames: 1117) |
| `qwen3_vl` (Qwen3-VL-4B) | yes | 447 (as frames: 871) |
| `qwen3_5_moe` (Qwen3.6-35B-A3B) | yes | 451 |
| `qwen2_5_vl` (Qwen2.5-VL-7B) | yes | 565 |
| `qwen2_vl`, `qwen3_5` | yes (GenAI's Qwen video path; not measured) | |
| `llava_next_video` | yes (in GenAI's video list; untested) | |
| `internvl_chat` (InternVL2-4B), `llava_next`, every other type | no: GenAI has no video preprocessing for them | as frames (InternVL2-4B: 1117) |

Cascadia passes each video's frame indices and sampling rate (`VideoMetadata`), so GenAI
uses the frames as sampled. Without them, GenAI 2026.4.1 resamples Qwen3-VL's video as if
it were a 24 fps source, keeping 4 of 16 frames with the wrong times. With them, a
16-frame clip of eight colours (two frames each) costs Qwen3-VL-4B 1655 prompt tokens and
is answered with all eight in order, on both GenAI pipelines. Because the indices are
given, GenAI logs "Frames indices already provided in video metadata, skipping ...
sampling" once per native video request; that warning is expected.

Qwen's encoder takes frames in pairs and pads an odd count with a copy of the last
frame, so on the Qwen types an odd frame count is rounded up to the next even one when
`--api-max-video-frames` allows (a 3 s clip at 1 fps gives 4 frames, at 0, 0.75, 1.5 and
2.25 s): the extra frame costs nothing. The other limits and flags are the ones above;
there is no separate native frame cap (16 frames are 8 Qwen pairs, about 1600 tokens at
448x448). On the staged engines, Qwen VL trees and the `gemma4` engine's Gemma 4 E2B / E4B
trees take video natively too (next sections); every other engine (`ov-dist-spec`,
`ov-runtime` on other families, Gemma 4 26B / 31B included) and every other export takes
video as labelled frames. Requests are the same either way.

### Native video (ov-runtime Qwen VL, qwen35)

`ov-runtime` Qwen2.5-VL and Qwen3-VL trees and `qwen35` Qwen3.5 / 3.6 trees take each video
whole too: the API writes Qwen's own `<|vision_start|><|video_pad|><|vision_end|>` where the
caller put the video, and rank 0 does what Hugging Face's processor does with it. Frame
pairs fill the vision encoder's temporal patch (an odd count rounds up as above), so a
video costs about half the tokens of its frames as images, with no `Frame k` labels:

- Qwen3-VL and Qwen3.5 write each frame pair as `<t seconds>` and its tokens (`t` the
  pair's mean time), size frames with their video processor's budget over all frames
  (`video_preprocessor_config.json`, which the cut copies) and position each pair as its
  own grid. Their video processor refuses a frame under 32 px a side, so a video whose
  frames scale below that (a tiny clip, or `detail: "low"` on a very wide one) is a
  `400` naming the bound; send its frames as `image_url` parts instead.
- Qwen2.5-VL writes one run of tokens and puts pair `k` at time position
  `floor(k * 2 / sampled fps * tokens_per_second)`, the next text one past the video's
  last position: Qwen's original `get_rope_index`, which the model was trained with
  (transformers 5.20.1 spaces the pairs the same way). transformers 5.14 / 5.15
  differ: they truncate `2 / fps` to whole seconds and start the next text inside the
  video's time range. At the default 1 fps a fractional `2 / fps` is routine (a 5 s clip
  gives 6 frames, 1.67 s per pair), so this server follows the original, not 5.14.

The token counts equal transformers 5.14.1's for the same frames, and the positions
equal its `get_rope_index` on Qwen3-VL and Qwen3.5 and the original code's on Qwen2.5-VL
(`tests/qwen_vl_video_parity.rs`). Video works in every mode images do on these trees
(stateful, static, packed; one box or several): rank 0 encodes it and the stages see the
same rows and position frames as for an image.

Measured on pawan-04 (CPU) with the SDK video suite's 4 s clip (4 frames): a Qwen3-VL-4B
tree takes 450 prompt tokens natively (one encode, cached for the follow-up turns) where
the same clip as labelled frames took 871, and ov-genai's native path 447 (GenAI 2026.4.1
replaces the whole `<|vision_start|><|video_pad|><|vision_end|>`, dropping the outer pair
HF keeps around the per-pair markup; this server follows HF); stateful,
static (`--prefill-seq 256 --shared-prefill`) and packed (`--packed-slots 2
--packed-seq 64`) trees answer the same and count the same 450. A Qwen2.5-VL-7B tree
takes 565, as ov-genai does. A two-stage Qwen3-VL tree
across pawan-04 (rank 0) and pawan-02 also counts 450 and answers the same. The `qwen35`
engine's video path is pinned by the parity tests but not yet run live (no Qwen3.6
vision tree on the nodes).

### Native video (gemma4 engine, Gemma 4 E2B / E4B)

Gemma 4 has no temporal patch: Hugging Face's `Gemma4Processor` (and GenAI 2026.4.1's
`InputsEmbedderGemma4`) encode each frame through the image encoder on its own, at the video
processor's smaller budget (`processor_config.json` `video_processor`: 70 soft tokens a
frame on E2B / E4B, against 280 for an image), and write the chat template's one
`<|video|>` as, per frame, its time as `mm:ss`, a space, then `<|image>`, `<|video|>` once
per soft token and `<image|>`, frames joined by a space. The `gemma4` engine does the same:
the API writes `<|video|>` where the caller put the video (with every sampled frame; no
pairing, no `Frame k` labels), rank 0 expands it in the text before tokenizing (minutes keep
counting past the hour, as HF's), encodes the frames, and hands stage 0 the video soft
tokens as `<|image|>`, the id its IR splices rows at (HF embeds both kinds the same way).
A video of more than the processor's `num_frames` (32) would keep 32 frames, spaced as HF
samples them; the API's 16-frame cap keeps every frame. E2B / E4B attend causally
(`use_bidirectional_attention` is unset), so frames need no attention block of their own;
the encode cache and the KV warm-resume key take the video's hash, which covers its frame
times. Only the `gemma4` engine opts in: Gemma 4 26B / 31B on `ov-runtime` keep frames.

Token ids, soft-token counts, frame sampling, `mm:ss` labels, `mm_token_type_ids` and the
attention blocks (one per frame) equal transformers 5.20.1's, and frame pixels are within
two 8-bit levels of its torchvision-only video processor
(`tests/gemma4_video_parity.rs`, fixtures from
`tools/vlm_fixtures/gen_gemma4_video_fixtures.py`).

Measured on pawan-04 (CPU) with the SDK video suite's 4 s clip on a one-stage E4B tree
(`trees\e4bv-1stage`): 332 prompt tokens natively where the same clip as labelled frames
took 1109, every check passing ("The first shape is a red circle, and the last shape is
a blue square."), the frames encoded once (about 4 s) and served from the encode cache
on the follow-up requests.

## What takes images

| Model | Engine | Placement | How to get the model dir |
|---|---|---|---|
| Qwen3.5 / Qwen3.6 VLM | `ov-genai` | one box | the OpenVINO VLM export as published (it has `openvino_vision_embeddings_model.xml`) |
| Qwen3.5 / Qwen3.6 VLM | `qwen35` | 2+ machines | `tools/qwen36_surgery/export_qwen36_moe.py`; the vision IRs in the source dir are carried into the tree |
| Gemma 4 26B-A4B / 31B | `ov-genai` | one box | the OpenVINO VLM export as published |
| Qwen2.5-VL, Qwen3-VL, InternVL2, Phi-3.5-vision, LLaVA-NeXT (llava-v1.6), Gemma 3 | `ov-genai` | one box | the OpenVINO exports as published (`OpenVINO/<model>-int4-ov`) |
| Gemma 4 26B-A4B / 31B | `ov-runtime` | 2+ machines, stateful, static or packed | `tools/gemma4_surgery/export_gemma4_text.py --vision` |
| Gemma 4 E2B / E4B | `gemma4` | 1+ machines | `tools/export_gemma4.py --vision DIR` |
| Gemma 3 (4B / 12B / 27B) | `ov-runtime` | 1+ machines, stateful, static or packed | `tools/vlm_surgery/cut_vlm_lm.py --family gemma3` |
| Pixtral 12B | `ov-runtime` | 1+ machines, stateful, static or packed | `tools/vlm_surgery/cut_vlm_lm.py --family pixtral` |
| Mistral 3 (Ministral-3, Mistral Small 3.1/3.2) | `ov-runtime` | 1+ machines, stateful, static or packed | export with optimum-intel, then `tools/vlm_surgery/cut_vlm_lm.py --family mistral3` |
| Phi-3.5-vision | `ov-runtime` | 1+ machines, stateful, static or packed | `tools/vlm_surgery/cut_vlm_lm.py --family phi3_v` on `OpenVINO/Phi-3.5-vision-instruct-int4-ov` |
| InternVL2 (Phi-3 text tower, e.g. InternVL2-4B) | `ov-runtime` | 1+ machines, stateful, static or packed | `tools/vlm_surgery/cut_vlm_lm.py --family internvl_chat` on `OpenVINO/InternVL2-4B-int4-ov` |
| LLaVA-NeXT (llava-v1.6-mistral-7b) | `ov-runtime` | 1+ machines, stateful, static or packed | `tools/vlm_surgery/cut_vlm_lm.py --family llava_next` on `OpenVINO/llava-v1.6-mistral-7b-hf-int4-ov` |
| Pixtral 12B target, Ministral-3-3B draft | `ov-dist-spec` | 2+ machines, stateful | `cut_vlm_lm.py --family pixtral --all-logits` for the target, `tools/vlm_surgery/make_text_draft.py` for the draft |
| Qwen2.5-VL (7B) | `ov-runtime` | 1+ machines, stateful, static or packed | `tools/vlm_surgery/cut_vlm_lm.py --family qwen2_5_vl` |
| Qwen3-VL (4B) | `ov-runtime` | 1+ machines, stateful, static or packed | `tools/vlm_surgery/cut_vlm_lm.py --family qwen3_vl` |

Every other engine and tree refuses images with a `400`: `ov-dist-spec` on a v5 (text)
target, `sparse-moe`, `ov-genai` on a text-only export, any tree without a `vision`
block. `ov-dist-spec` refuses a Gemma 4 target tree at load. Not supported:

- **Pixtral on `ov-genai`.** GenAI 2026.4.1 preprocesses Pixtral as LLaVA-1.5 and
  describes a different picture; `ov-genai` serves such an export as text only. Use
  `ov-runtime` (a one-stage tree runs on one box).
- **Llama 3.2 Vision** (`mllama`), refused at export: its text tower has cross-attention.
- **MiniCPM-V 4.7**: GenAI 2026.4.1 does not know its model type (`minicpmv4_7`) and the
  engine fails to load.
- Images on `ov-dist-spec` for any target but Pixtral / Mistral 3 (Gemma 3, Phi-3.5-vision,
  InternVL2, LLaVA-NeXT, Qwen2.5-VL, Qwen3-VL): such a tree is refused at load. Serve
  these on `ov-runtime`.
- Trees made with `export_shards.py` carry no vision tower (it notes when it drops one);
  cut vision trees with `tools/vlm_surgery/cut_vlm_lm.py`.

## Exporting

### Gemma 4 26B / 31B (ov-runtime)

```bash
python tools/gemma4_surgery/export_gemma4_text.py \
    --model <OpenVINO gemma-4 VLM export> --output-dir <tree> --num-stages 4 --vision
```

Stage 0 takes `inputs_embeds`; rank 0 embeds the prompt, runs the vision encoder on CPU
and splices the image rows in. Every stage also takes `token_type_ids` (Gemma attends
bidirectionally inside an image), which travels between ranks with the hidden states.

### Gemma 4 E2B / E4B (gemma4 engine)

```bash
python tools/export_gemma4.py --model <google/gemma-4-E4B-it snapshot> \
    --output-dir <tree> --num-stages 2 --vision <OpenVINO/gemma-4-E4B-it-fp16-ov dir>
```

Pass the snapshot path to `--model` (a pinned download is not resolvable by repo id
offline). `--vision` takes the vision encoder IR and processor configs from the
OpenVINO export.

### Pixtral (ov-runtime)

```bash
python tools/vlm_surgery/cut_vlm_lm.py --model <OpenVINO/pixtral-12b-int4-ov dir> \
    --output-dir <tree> --num-stages 2 --family pixtral
```

`--num-stages 1` cuts a one-stage tree for a single box. `--stage I` cuts one stage on a
worker that holds only that one.

### Gemma 3 (ov-runtime)

```bash
python tools/vlm_surgery/cut_vlm_lm.py --model <OpenVINO/gemma-3-12b-it-int4-ov dir> \
    --output-dir <tree> --num-stages 2 --family gemma3
```

The published export already fits: its language model takes `inputs_embeds` and
`token_type_ids`, and its vision IR (SigLIP, the 4x4 pool and the projector) returns
256 rows per image at the text width. Rank 0 resizes each image to 896x896 (bilinear,
as HF's processor), expands each `<start_of_image>` of the rendered prompt into
HF's `\n\n<start_of_image>` + 256 x `<image_soft_token>` + `<end_of_image>\n\n` before
tokenizing, and splices the image rows in. Every stage takes `token_type_ids`: Gemma 3
opens each image both ways in its full and sliding masks, so the types ride the wire as
a third frame, as for Gemma 4. For the static path pass `--prefill-seq` of at least 256
and keep `--context` + `--prefill-seq` within 1025 (sliding window 1024), for example
`--context 768 --prefill-seq 256` (two images). Pan-and-scan (off in the published
processor config) is refused at load.

### Mistral 3 (ov-runtime)

No OpenVINO export is published, so make one with optimum-intel (`main`; release 2.2.0
lacks the `ministral3` text tower), then cut it like Pixtral:

```bash
pip install "optimum-intel[openvino] @ https://github.com/huggingface/optimum-intel/archive/<commit>.zip" nncf
optimum-cli export openvino --model <mistralai/Ministral-3-3B-Instruct-2512-BF16 dir> \
    --task image-text-to-text --weight-format int4 <export dir>
python tools/vlm_surgery/cut_vlm_lm.py --model <export dir> --output-dir <tree> \
    --num-stages 2 --family mistral3
```

The export holds four IRs; rank 0 runs the vision tower, regroups each 2x2 block of
patches and runs `openvino_multi_modal_projector_model.xml` on them. Exporting on Windows
is fine: the chat template it writes has CRLF line endings, which the server normalises.

### Phi-3.5-vision (ov-runtime)

```bash
python tools/vlm_surgery/cut_vlm_lm.py --model <OpenVINO/Phi-3.5-vision-instruct-int4-ov dir> \
    --output-dir <tree> --num-stages 2 --family phi3_v
```

The export holds four IRs: the CLIP tower (`openvino_vision_embeddings_model.xml`, one
336-px view to 576 patch rows), the projector (`openvino_vision_projection_model.xml`),
the text embeddings and the language model. Rank 0 does the processor's HD transform
(up to `num_crops` 4 crops of 336 px plus a global view), merges each view's patches 2x2,
lays the crops out with the `sub_GN` / `glb_GN` separator rows from `config.json`, and
runs the projector. Each image is `<|image_N|>` followed by a newline in the prompt, as
the model card writes it; rank 0 tokenizes the text between tags chunk by chunk, each
with its BOS, as `Phi3VProcessor` does. The tree also stops on `<|end|>`.

### InternVL2 (ov-runtime)

```bash
python tools/vlm_surgery/cut_vlm_lm.py --model <OpenVINO/InternVL2-4B-int4-ov dir> \
    --output-dir <tree> --num-stages 2 --family internvl_chat
```

The vision IR already ends in the projector (448-px tile to 256 rows). Rank 0 cuts the
image into the closest grid of at most `max_dynamic_patch` (12) tiles plus a thumbnail,
as the model card's `load_image` does, and writes each image as
`<img>` + `<IMG_CONTEXT>` x 256 per tile + `</img>`, a newline after it.

### LLaVA-NeXT (ov-runtime)

```bash
python tools/vlm_surgery/cut_vlm_lm.py --model <OpenVINO/llava-v1.6-mistral-7b-hf-int4-ov dir> \
    --output-dir <tree> --num-stages 2 --family llava_next
```

The export's vision IR is the CLIP tower plus projector (`[n, 3, 336, 336]` to
`[n, 576, 4096]`); `image_newline` is a 4096-float array in its `config.json`, which the
cut keeps as `config.vlm.json`. Rank 0 picks the image's pinpoint, encodes the 336x336
base view and the tiles in one inference and packs the rows as HF does (the base view,
then the tile grid without its padding, `image_newline` after each row). A message's
images go before its text (`<image>\n` each), as the processor's template writes them.

### Qwen2.5-VL / Qwen3-VL (ov-runtime)

```bash
python tools/vlm_surgery/cut_vlm_lm.py --model <OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov dir> \
    --output-dir <tree> --num-stages 2 --family qwen2_5_vl
python tools/vlm_surgery/cut_vlm_lm.py --model <OpenVINO/Qwen3-VL-4B-Instruct-int4-ov dir> \
    --output-dir <tree> --num-stages 2 --family qwen3_vl
```

Rank 0 preprocesses images as HF's Qwen2-VL processor does, runs the vision IRs
(Qwen2.5-VL: patch embedding and the window-attention merger; Qwen3-VL: Qwen3.5's three)
and expands each `<|image_pad|>` to the image's tokens; a video goes through the same IRs
a frame pair at a time ([native video](#native-video-ov-runtime-qwen-vl-qwen35)). Every stage takes M-RoPE
`position_ids [3, B, T]`: rank 0 computes them per span (images take a 2-D grid; decode
continues from `rope_delta`) and sends them after the hidden states, so the stages record
`position_rows: 3` and the tree `vision.mrope_rows: 3`; ranks on a build without the frame
refuse the tree at load. Qwen3-VL adds the vision tower's DeepStack features to the image
tokens after decoder layers 0-2, so stage 0 (on rank 0) must hold those layers: the cut
refuses a split that puts them elsewhere, and the features never cross the wire.

### Pixtral with speculative decoding (ov-dist-spec)

Only the target sees images; the draft is a text model fed the prompt without its
image ids, so it changes the acceptance rate, never the answer.
The target's last stage must return logits for every position (`--all-logits`), and the
draft must be one IR that takes `input_ids`:

```bash
python tools/vlm_surgery/cut_vlm_lm.py --model <OpenVINO/pixtral-12b-int4-ov dir> \
    --output-dir <tree> --num-stages 2 --family pixtral --all-logits
python tools/vlm_surgery/make_text_draft.py --model <ministral3-3b-int4-ov export> \
    --output-dir <draft dir>
```

`make_text_draft.py` wires `openvino_text_embeddings_model.xml` in front of
`openvino_language_model.xml` and saves `openvino_model.xml` (inputs `input_ids`,
`attention_mask`, `position_ids`, `beam_idx`, stateful KV). Serving and logs are in
[engines/ov-dist-spec.md](engines/ov-dist-spec.md#image-input-pixtral-target). With
`kv_coord`, image turns there prefill cold and are not captured for warm resume.

### Static and packed trees (NPU, or CPU/GPU without OpenVINO state)

Convert a stateful vision tree from the tools above:

```bash
python tools/vlm_surgery/static_variant.py --tree <stateful tree> --output-dir <static tree> \
    --context 2048 --prefill-seq 256 [--shared-prefill] [--stage I]
```

- `--prefill-seq C` adds a chunked prefill (C rows per inference). On CPU/GPU,
  `--shared-prefill` writes one IR that serves both decode and prefill, so the weights
  are compiled once; the NPU needs the two separate static IRs (omit the flag).
- **LLaVA-NeXT** spends up to 2928 tokens on an image, so give it `--context 4096` or
  more; it attends causally, so any `--prefill-seq` works. For packed slots a 448x448
  image (2956 prompt tokens with the SDK test's text) must fit one slot's region, so a
  two-slot variant needs `--context` near 6016 (region 3007); a narrower context refuses
  such an image with a 413. Verified static and packed on CPU (prompt_tokens 2956).
- **Gemma 4** needs `--prefill-seq` at least one image wide (280), and `--context` plus
  `--prefill-seq` must stay within the sliding window + 1 (1025): for example
  `--context 736 --prefill-seq 288`, which holds about two images.
- **Qwen2.5-VL / Qwen3-VL** carry M-RoPE `position_ids [3, T]` on every stage, which both
  the static IRs (`[3, 1, seq]`) and the packed IR (`[3, 1, S]`) now take; `static_variant.py`
  and `packed_variant.py` pin them from the stateful tree, and rank 0 computes each span's
  rows (decode continues from `rope_delta`) and ships them to the other stages — on the
  static wire after the hidden, on the packed wire as a `[1, 3, S]` frame. Qwen attends
  causally, so any `--prefill-seq` / `--packed-seq` works and a chunk boundary may fall
  inside an image. Qwen3-VL's DeepStack rows feed stage 0 (rank 0) per inference,
  gathered by image-token order, so they never cross the wire; the cut keeps the DeepStack
  layers in stage 0, and the static/packed IRs pin `deepstack_visual_embeds [layers, seq,
  hidden]` (the gather reads at most one row per image token, so padded rows are never
  touched).
- Packed slots (`--packed-slots N`, continuous batching on static shards): build the
  packed IR per stage of the static tree with `python tools/packed_variant.py <stage dir>
  --slots N [--packed-seq S]`, then serve every rank with `--packed-slots N`. On CPU/GPU
  run it on the same `--shared-prefill` tree: the packed IR sits beside the shared decode
  IR, so one tree serves both the static and the packed paths. A prefill chunk is
  `--packed-seq` rows; a decode step runs narrower when the stage has a narrow width:
  pass `--dynamic-seq` on CPU (the packed IR keeps its query width dynamic, so a
  decode step runs at its live rows) or `--decode-seq D` (a second, static `D`-row
  decode IR, `D` at least the slot count; meant for the NPU, measured on CPU only).
  Packed output on an Intel GPU is wrong at every width (the load warns). Without either, a decode step is
  `--packed-seq` rows too (CPU, Qwen3-VL-4B, 4 slots: 1.92 s a step at 64 rows,
  0.75 s with `--decode-seq 4`, 0.94 s with `--dynamic-seq`; see
  [perf/NPU_PACKED_SLOTS.md](perf/NPU_PACKED_SLOTS.md)). A bidirectional
  family (Gemma 3 / Gemma 4) keeps `--packed-seq` at least one image wide (below). A
  request whose prompt does not fit one slot's region `(context - 1 - packed_prefix) / N`
  is refused with a 413.
- **Gemma 4 on packed slots** (26B-A4B: about 18 GB per packed stage at 288 rows, so one
  stage per 32 GB node): same two steps. `packed_variant.py` finds the stage's two
  mask producers (sliding window and full attention), tells them apart from the tree's
  `config.json` `layer_types`, and gives each its own input (`attn_mask_sliding_4d`,
  `attn_mask_4d`); the runtime builds both masks, with each image open both ways. An
  image must fit one packed inference, so pass `--packed-seq` at least one image wide:

  ```bash
  python tools/vlm_surgery/static_variant.py --tree <stateful tree> --output-dir <static tree> \
      --context 736 --prefill-seq 288
  for i in 0 1 2 3; do
    python tools/packed_variant.py <static tree>/stage_$i --slots 2 --packed-seq 288
  done
  ```

  The load log line `packed multi-slot decode active` names the mask ports, the window
  and the KV geometry (`8x256 x6, 2x512 x1`, ...) each stage loaded.
- **Gemma 3 on packed slots** takes the same two-producer recipe (sliding window + full
  attention, `token_type_ids`), so build its static tree with `--context 768
  --prefill-seq 256` (sliding window 1024) and `--packed-seq 256` (one image wide). Each
  slot's region is `(context - 1) / slots`, so two images do not fit one slot of a
  two-slot 768-context tree (they are refused per request with a 413); one 448x448 image
  (286 tokens) fits.
- On CPU/GPU the packed variant is built from the same `--shared-prefill` static tree as
  the static path, so one tree serves both.

## Serving

```bash
# one box
cascadia run <model dir> --api 127.0.0.1:8099                       # ov-genai
cascadia worker --total 1 --rank 0 --engine ov-runtime --model <tree> --api 127.0.0.1:8099

# two machines (rank 1 first)
cascadia worker --total 2 --rank 1 --engine ov-runtime --model <tree> --listen <host>:9201
cascadia worker --total 2 --rank 0 --engine ov-runtime --model <tree> --next <host>:9201 --api 127.0.0.1:8099

# speculative decoding on a Pixtral --all-logits tree (rank 1 first)
cascadia worker --total 2 --rank 1 --engine ov-dist-spec --model <tree> --listen <host>:9201
cascadia worker --total 2 --rank 0 --engine ov-dist-spec --model <tree> --draft-model <draft dir> \
    --spec-k 4 --next <host>:9201 --api 127.0.0.1:8099
```

Image input switches on by itself when the model dir or tree carries a vision encoder
(`openvino_vision_embeddings_model.xml`, and for `ov-runtime`/`gemma4`/`ov-dist-spec` a `vision` block
in `pipeline_config.json`). The startup log says so (`accepting image_url content parts`)
or says why not. Static trees: add `--device NPU` (and `--ov-cache-dir` so a restart
loads compiled blobs instead of recompiling, which takes about ten minutes per stage IR).
Verified on the Lunar Lake NPU with Pixtral (one 12B stage per 32 GB node) and Ministral-3-3B
(one box), and on the Arc 140V iGPU (`--device GPU`) with Gemma 3, Qwen3-VL and Ministral-3;
a node
needs RAM for both the decode and prefill IRs of every stage it holds. Gemma 4 26B-A4B
is not practical on the NPU: one stage's decode IR had not compiled after 90 minutes
(its mixture-of-experts layers); serve it from CPU static or stateful trees. Qwen3-VL 4B
static and packed (`--packed-slots 2`) trees passed the SDK image suite on CPU — one box and
two machines (the M-RoPE positions ride the stage wire) — with the prompt-token count
matching the stateful tree (223), and the packed concurrent test too. Gemma 3 12B packed
(`--packed-slots 2`, sliding and full masks) passed the concurrent test across two machines
(286 tokens per image, no slot leak); it is slow on CPU, about 200-300 s a request.

## Limits worth knowing

- Prompt tokens count the image's tokens (Gemma 4: at most 280 per image; Gemma 3: 256 per
  image plus its markers and blank lines, 286 for a 448x448 image with the SDK test's text;
  Pixtral: one per 16x16 patch plus row breaks, 832 for a 448x448 image with its text;
  LLaVA-NeXT: 576 plus its unpadded tile rows, at most 2928, so 2945 for a 448x448 image
  with its text; Phi-3.5-vision: `(crops + 1) * 144 + 1 + (crop rows + 1) * 12`, 757 for a
  448x448 image; InternVL2: 256 per tile, one tile for 448x448, up to 13 tiles (3328
  tokens) for an image whose aspect ratio matches a 12-tile grid, small images included;
  Qwen: about one per 1000 pixels, a 448x448 image is 196 tokens on Qwen3.5 / Qwen3-VL
  and 256 on Qwen2.5-VL). The counts match Hugging Face's processor for the same
  request (`ov-genai` counts 2956 for that LLaVA-NeXT request).
- Phi-3.5-vision and InternVL2: these exports carry Phi-3's RoPE frequencies as a traced
  constant, with no switch to the long factors past 4096 positions, so longer sequences
  differ from Hugging Face; ov-genai runs the same IRs. The load log says so.
- Packed slots: a prompt must fit one slot's region, `(context - 1 - packed_prefix) /
  slots`; two Pixtral images do not fit two slots of a 2048 context, and one 448x448
  LLaVA-NeXT image (2945 tokens with its text) needs a region of at least that.
- Gemma 4 packed slots: a request whose image is wider than `--packed-seq` rows is
  refused with a 400 (lower detail sends a smaller image). A 288-row variant decodes
  at a 288-row cost unless it was built with `--dynamic-seq` (CPU) or
  `--decode-seq` (static shapes, meant for the NPU); `static_variant.py` still caps `--context` + `--prefill-seq` at the sliding
  window + 1 (1025), which leaves two slots of 367 tokens at `--context 736`. Every
  stage must run the same build: the packed plan frame carries image ids (four rows
  instead of three), and a stage refuses the other kind.
- Rank 0 keeps each encoded image (keyed on a hash of its pixels) so a chat's earlier
  images are not re-encoded every turn: `CASCADIA_IMAGE_CACHE_MB` per vision front end,
  default 256, 0 to disable. With `kv_coord`, `ov-runtime`, `gemma4` and `qwen35` also
  resume a chat's KV past its earlier images (keyed on each image's hash), so a follow-up
  turn prefills only its new text: about 1-1.5 s instead of 11-14 s on CPU (Ministral-3-3B,
  Gemma 4 E4B, Qwen3.6-35B-A3B on two machines). Static and packed shards still re-prefill
  images; the single-box `qwen35` prefix cache keys on the image too.
- The image encode cache and image warm-resume are shared across all callers of a server,
  as the prefix cache is: the API sets no tenant, so a caller who has an image can tell
  from the latency whether someone sent the same image recently. Serve callers who must
  not learn that from separate servers.
