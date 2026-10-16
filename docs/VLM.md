# Image input (VLM)

Cascadia takes images the way the OpenAI API does: `image_url` content parts in
`/v1/chat/completions` messages. This page says which models and engines take them, how
to export each one, and what the limits are. The design and its verification history are
in [rfcs/vlm-support.md](rfcs/vlm-support.md). Video and audio are not supported.

## The request

```json
{"model": "m", "messages": [{"role": "user", "content": [
  {"type": "text", "text": "What is in this picture?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "auto"}}
]}]}
```

- `data:` URLs (PNG, JPEG, WEBP, GIF) work by default. `https://` URLs only with
  `--api-allow-image-urls` (public addresses only, each redirect re-checked).
- Up to 8 images per request. An image over about 1 MP is downscaled (`detail: "low"`:
  512x512). The request body may carry 20 MiB of images on top of the text limit.
- Images can sit anywhere in a user message and in earlier turns.
- `/v1/models` lists `input_modalities` (`["text", "image"]` when images are accepted).
- Errors are OpenAI-shaped: a model or tree without image input answers `400`, an
  undecodable image `400`, a prompt that does not fit the context `413`
  (`context_length_exceeded`). Text that spells a model's own image tag (for example
  `<|image|>` or `[IMG]`) is refused with `400`.

## What takes images

| Model | Engine | Placement | How to get the model dir |
|---|---|---|---|
| Qwen3.5 / Qwen3.6 VLM | `ov-genai` | one box | the OpenVINO VLM export as published (it has `openvino_vision_embeddings_model.xml`) |
| Qwen3.5 / Qwen3.6 VLM | `qwen35` | 2+ machines | `tools/qwen36_surgery/export_qwen36_moe.py`; the vision IRs in the source dir are carried into the tree |
| Gemma 4 26B-A4B / 31B | `ov-genai` | one box | the OpenVINO VLM export as published |
| Qwen2.5-VL, Qwen3-VL, InternVL2, Phi-3.5-vision, LLaVA-NeXT (llava-v1.6), Gemma 3 | `ov-genai` | one box | the OpenVINO exports as published (`OpenVINO/<model>-int4-ov`) |
| Gemma 4 26B-A4B / 31B | `ov-runtime` | 2+ machines, stateful or static | `tools/gemma4_surgery/export_gemma4_text.py --vision` |
| Gemma 4 E2B / E4B | `gemma4` | 1+ machines | `tools/export_gemma4.py --vision DIR` |
| Pixtral 12B | `ov-runtime` | 1+ machines, stateful, static or packed | `tools/vlm_surgery/cut_vlm_lm.py --family pixtral` |
| Mistral 3 (Ministral-3, Mistral Small 3.1/3.2) | `ov-runtime` | 1+ machines | export with optimum-intel, then `tools/vlm_surgery/cut_vlm_lm.py --family mistral3` |

Every other engine and tree refuses images with a `400`: `ov-dist-spec`, `sparse-moe`,
`ov-genai` on a text-only export, any tree without a `vision` block. Not supported:

- **Pixtral on `ov-genai`.** GenAI 2026.4.1 preprocesses Pixtral as LLaVA-1.5 and
  describes a different picture; `ov-genai` serves such an export as text only. Use
  `ov-runtime` (a one-stage tree runs on one box).
- **Llama 3.2 Vision** (`mllama`), refused at export: its text tower has cross-attention.
- **MiniCPM-V 4.7**: GenAI 2026.4.1 does not know its model type (`minicpmv4_7`) and the
  engine fails to load.
- Staged (multi-machine) serving of the GenAI-only families above: their trees are not
  exported with the vision tower yet (`export_shards.py` notes when it drops one).

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

### Static and packed trees (NPU, or CPU/GPU without OpenVINO state)

Convert a stateful vision tree from the tools above:

```bash
python tools/vlm_surgery/static_variant.py --tree <stateful tree> --output-dir <static tree> \
    --context 2048 --prefill-seq 256 [--shared-prefill] [--stage I]
```

- `--prefill-seq C` adds a chunked prefill (C rows per inference). On CPU/GPU,
  `--shared-prefill` writes one IR that serves both decode and prefill, so the weights
  are compiled once; the NPU needs the two separate static IRs (omit the flag).
- **Gemma 4** needs `--prefill-seq` at least one image wide (280), and `--context` plus
  `--prefill-seq` must stay within the sliding window + 1 (1025): for example
  `--context 736 --prefill-seq 288`, which holds about two images.
- Packed slots (`--packed-slots N`, continuous batching on static shards): build the
  packed IR per stage with `python tools/packed_variant.py <stage dir> --slots N
  [--packed-seq S]`. Pixtral only; Gemma 4 trees are refused on packed slots.

## Serving

```bash
# one box
cascadia run <model dir> --api 127.0.0.1:8099                       # ov-genai
cascadia worker --total 1 --rank 0 --engine ov-runtime --model <tree> --api 127.0.0.1:8099

# two machines (rank 1 first)
cascadia worker --total 2 --rank 1 --engine ov-runtime --model <tree> --listen <host>:9201
cascadia worker --total 2 --rank 0 --engine ov-runtime --model <tree> --next <host>:9201 --api 127.0.0.1:8099
```

Image input switches on by itself when the model dir or tree carries a vision encoder
(`openvino_vision_embeddings_model.xml`, and for `ov-runtime`/`gemma4` a `vision` block
in `pipeline_config.json`). The startup log says so (`accepting image_url content parts`)
or says why not. Static trees: add `--device NPU` (and `--ov-cache-dir` so a restart
loads compiled blobs instead of recompiling, which takes about ten minutes per stage IR).
Verified on the Lunar Lake NPU with Pixtral (one 12B stage per 32 GB node); a node
needs RAM for both the decode and prefill IRs of every stage it holds. Gemma 4 26B-A4B
is not practical on the NPU: one stage's decode IR had not compiled after 90 minutes
(its mixture-of-experts layers); serve it from CPU static or stateful trees.

## Limits worth knowing

- Prompt tokens count the image's tokens (Gemma 4: at most 280 per image; Pixtral: one per
  16x16 patch plus row breaks, 832 for a 448x448 image with its text; Qwen: about one per
  1000 pixels). The counts match Hugging Face's processor for the same request.
- Packed slots: a prompt must fit one slot's region, `(context - 1 - packed_prefix) /
  slots`; two Pixtral images do not fit two slots of a 2048 context.
- Rank 0 keeps each encoded image (keyed on a hash of its pixels) so a chat's earlier
  images are not re-encoded every turn: `CASCADIA_IMAGE_CACHE_MB` per vision front end,
  default 256, 0 to disable. With `kv_coord`, `ov-runtime`, `gemma4` and `qwen35` also
  resume a chat's KV past its earlier images (keyed on each image's hash), so a follow-up
  turn prefills only its new text: about 1-1.5 s instead of 11-14 s on CPU (Ministral-3-3B,
  Gemma 4 E4B, Qwen3.6-35B-A3B on two machines). Static and packed shards still re-prefill
  images; the single-box `qwen35` prefix cache keys on the image too.
