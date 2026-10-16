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

## What takes images

| Model | Engine | Placement | How to get the model dir |
|---|---|---|---|
| Qwen3.5 / Qwen3.6 VLM | `ov-genai` | one box | the OpenVINO VLM export as published (it has `openvino_vision_embeddings_model.xml`) |
| Qwen3.5 / Qwen3.6 VLM | `qwen35` | 2+ machines | `tools/qwen36_surgery/export_qwen36_moe.py`; the vision IRs in the source dir are carried into the tree |
| Gemma 4 26B-A4B / 31B | `ov-genai` | one box | the OpenVINO VLM export as published |
| Qwen2.5-VL, Qwen3-VL, InternVL2, Phi-3.5-vision, LLaVA-NeXT (llava-v1.6), Gemma 3 | `ov-genai` | one box | the OpenVINO exports as published (`OpenVINO/<model>-int4-ov`) |
| Gemma 4 26B-A4B / 31B | `ov-runtime` | 2+ machines, stateful, static or packed | `tools/gemma4_surgery/export_gemma4_text.py --vision` |
| Gemma 4 E2B / E4B | `gemma4` | 1+ machines | `tools/export_gemma4.py --vision DIR` |
| Pixtral 12B | `ov-runtime` | 1+ machines, stateful, static or packed | `tools/vlm_surgery/cut_vlm_lm.py --family pixtral` |
| Mistral 3 (Ministral-3, Mistral Small 3.1/3.2) | `ov-runtime` | 1+ machines | export with optimum-intel, then `tools/vlm_surgery/cut_vlm_lm.py --family mistral3` |
| Pixtral 12B target, Ministral-3-3B draft | `ov-dist-spec` | 2+ machines, stateful | `cut_vlm_lm.py --family pixtral --all-logits` for the target, `tools/vlm_surgery/make_text_draft.py` for the draft |
| Qwen2.5-VL (7B) | `ov-runtime` | 1+ machines, stateful | `tools/vlm_surgery/cut_vlm_lm.py --family qwen2_5_vl` |
| Qwen3-VL (4B) | `ov-runtime` | 1+ machines, stateful | `tools/vlm_surgery/cut_vlm_lm.py --family qwen3_vl` |

Every other engine and tree refuses images with a `400`: `ov-dist-spec` on a v5 (text)
target, `sparse-moe`, `ov-genai` on a text-only export, any tree without a `vision`
block. `ov-dist-spec` refuses a Gemma 4 target tree at load. Not supported:

- **Pixtral on `ov-genai`.** GenAI 2026.4.1 preprocesses Pixtral as LLaVA-1.5 and
  describes a different picture; `ov-genai` serves such an export as text only. Use
  `ov-runtime` (a one-stage tree runs on one box).
- **Llama 3.2 Vision** (`mllama`), refused at export: its text tower has cross-attention.
- **MiniCPM-V 4.7**: GenAI 2026.4.1 does not know its model type (`minicpmv4_7`) and the
  engine fails to load.
- Staged (multi-machine) serving of the other GenAI-only families above (InternVL2,
  Phi-3.5-vision, LLaVA-NeXT, Gemma 3): their trees are not exported with the vision
  tower yet (`export_shards.py` notes when it drops one).
- Qwen2.5-VL / Qwen3-VL on static, packed or `ov-dist-spec` trees: those paths feed 1-D
  positions and these models need M-RoPE positions on every stage, so the tree is
  refused at load (`ov-dist-spec` offers no image input for it).

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

### Qwen2.5-VL / Qwen3-VL (ov-runtime)

```bash
python tools/vlm_surgery/cut_vlm_lm.py --model <OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov dir> \
    --output-dir <tree> --num-stages 2 --family qwen2_5_vl
python tools/vlm_surgery/cut_vlm_lm.py --model <OpenVINO/Qwen3-VL-4B-Instruct-int4-ov dir> \
    --output-dir <tree> --num-stages 2 --family qwen3_vl
```

Rank 0 preprocesses images as HF's Qwen2-VL processor does, runs the vision IRs
(Qwen2.5-VL: patch embedding and the window-attention merger; Qwen3-VL: Qwen3.5's three)
and expands each `<|image_pad|>` to the image's tokens. Every stage takes M-RoPE
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
- **Gemma 4** needs `--prefill-seq` at least one image wide (280), and `--context` plus
  `--prefill-seq` must stay within the sliding window + 1 (1025): for example
  `--context 736 --prefill-seq 288`, which holds about two images.
- Packed slots (`--packed-slots N`, continuous batching on static shards): build the
  packed IR per stage of the static tree with `python tools/packed_variant.py <stage dir>
  --slots N [--packed-seq S]`, then serve every rank with `--packed-slots N`.
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
(its mixture-of-experts layers); serve it from CPU static or stateful trees.

## Limits worth knowing

- Prompt tokens count the image's tokens (Gemma 4: at most 280 per image; Pixtral: one per
  16x16 patch plus row breaks, 832 for a 448x448 image with its text; Qwen: about one per
  1000 pixels, a 448x448 image is 196 tokens on Qwen3.5 / Qwen3-VL and 256 on Qwen2.5-VL).
  The counts match Hugging Face's processor for the same request.
- Packed slots: a prompt must fit one slot's region, `(context - 1 - packed_prefix) /
  slots`; two Pixtral images do not fit two slots of a 2048 context.
- Gemma 4 packed slots: a request whose image is wider than `--packed-seq` rows is
  refused with a 400 (lower detail sends a smaller image). Every inference is
  `--packed-seq` rows, decode included, so a 288-row variant decodes at a 288-row
  cost; `static_variant.py` still caps `--context` + `--prefill-seq` at the sliding
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
