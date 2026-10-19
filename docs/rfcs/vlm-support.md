# RFC draft: image input (VLM) support across Cascadia

Operator guide (models, export commands, flags, limits): [../VLM.md](../VLM.md).

**Status:** research draft, 2026-10-14. Based on `feat/qwen35-moe-gptq-ir-builder`
(PR #174, HEAD `9ecc2083`). Nothing here is implemented. Claims marked
**[verified]** were read in this tree or in the cited upstream source; claims
marked **[inferred]** still need checking (see [Open questions](#open-questions)).

## Summary

Every Qwen3.5-family IR we build already contains a vision tower, and the ov-genai
shim already opens those IRs with OpenVINO GenAI's `VLMPipeline`. What stops image
input is the plumbing around them. The API only accepts string content, the task
type has no media field, the shim passes an empty image list, and the staged engine
fills M-RoPE positions with plain 1-D positions.

Qwen3.5 injects vision features **only** through `inputs_embeds`; there is no
DeepStack. So a multi-node deployment needs the vision encoder on rank 0 only.
Downstream ranks need correct positions, but no image data.

Recommended order:

1. Accept OpenAI image content parts in the API, using OpenAI's wire format and error
   objects exactly, and carry the images on `GenerationTask`. This first fixes two
   existing gaps: content-part arrays fail today even for text-only messages, and
   non-streaming errors are not in OpenAI's shape.
2. Pass the images to `VLMPipeline` in the shim. This gives single-box image input for any
   VLM-layout export, with no exporter change.
3. Add a rank-0 vision step to the staged `qwen35` engine. It splices vision rows into
   `inputs_embeds` and sends real M-RoPE positions downstream. This is what serves
   397B-A17B with images.
4. Other families (Gemma 4, Mistral 3, Inkling) follow later, each needing an export
   that keeps its vision tower and a stage 0 that takes `inputs_embeds`.

## Where we are today

| Layer | State | Reference |
|---|---|---|
| HTTP API | `ChatMessage.content: String`. An OpenAI content-part array fails to deserialize, so the request is rejected before the handler runs. Routes: chat/completions, completions only. | `crates/cascadia-api/src/lib.rs:648-659` [verified] |
| Size limits | 64 KiB body by default in the library. The CLI's `--api-max-body-mb` defaults to 1 MiB and also sets `max_prompt_bytes`. One base64 photo uses most of that. | `lib.rs:35,47,527,2279`; `cascadia-cli/src/lib.rs:2341` [verified] |
| Chat template | minijinja, rendered in the API. Messages reach the template as `{"role", "content": <string>}`, so HF VLM templates never see image parts. | `lib.rs:1334,1430-1497,1447` [verified] |
| Tokenization | Happens in the engine (the `tokenizers` crate, or GenAI's OV tokenizer). The API sends rendered text. | `qwen36.rs:729-741`, `genai.rs:440` [verified] |
| Task type | `GenerationTask { prompt: String, … resume_token_ids }`: no media field. It never crosses the wire except in tests. | `cascadia-types/src/task.rs:57-107` [verified] |
| ov-genai engine | Detects a VLM-layout dir and builds `VLMPipeline` "served text-only". Rejects `--cb` and `--draft-model` on VLM layouts. | `genai.rs:241-286` [verified] |
| Shim | `vlm->generate(prompt, std::vector<ov::Tensor>{}, …)`: always an empty image list. CB `add_request` takes a string only. | `cascadia-ov-genai-shim/cpp/shim.cpp:399-410,451-525` [verified] |
| Staged `qwen35` | Rank 0 runs `openvino_text_embeddings_model.xml` on the host (`embed_seq`) and feeds `inputs_embeds` to stage 0. Positions are `[4,1,n]` with every row `t0..t1`. Downstream ranks rebuild positions from the frame's `pos`. | `qwen36.rs:555-562,949-980,994-1030,1689-1692` [verified] |
| Prefix / KV caches | Keyed on token ids only (qwen35 `PrefixCache`, sparse-moe `kv_prefix_cache`, `KvCoordination::lookup`). | `prefix_cache.rs:179-193,335`; `cascadia-engine/src/lib.rs:196-215` [verified] |
| Exporters | The #174 builder emits `openvino_vision_embeddings{,_pos,_merger}_model.xml`. `export_qwen36_moe.py` cuts only the language model and copies only the text-embeddings IR. The Gemma 4 surgery and `export_shards.py` drop vision. Inkling drops `model.visual.*`. | `gptq_to_ov.py:477-487`; `export_qwen36_moe.py:406,445-452`; `export_gemma4_text.py`; `export_inkling.py:66` [verified] |
| Roadmap | No RFC or issue. Docs say "vision out of scope" (qwen3.6, qwen3.8, mistral, inkling, SHARDING.md:106). | [verified] |

### Which families have vision upstream

| Family | Vision upstream | Path to images |
|---|---|---|
| Qwen3.5 / 3.6 MoE (35B-A3B, 397B-A17B), Qwen3.8-27B dense | Yes: 27-layer ViT + merger, interleaved M-RoPE `[11,11,10]` | **First target.** VLM IRs already built; ov-genai works once the shim passes images; staged path per this RFC |
| Gemma 4 (E2B/E4B/31B; 26B-A4B rejected today) | Yes | The exporter must stop dropping vision; stage 0 currently takes `input_ids` plus per-layer embeddings derived from ids |
| Mistral Small 3.1 (`mistral3`, Pixtral) | Yes | Generic exporter keeps `text_config` only |
| Inkling | Yes (`model.visual.*` hMLP, plus audio) | sparse-moe; vision weights dropped at export |
| Kimi K2.6 | Yes (MoonViT, about 400M, projector to 7168) | sparse-moe, native K2.6 path; about 1T parameters, built for one large node, not these 32 GB nodes; vision not exported |
| GLM-5.x, DeepSeek-V4, MiniMax-M2, R1-distill, Phi-3/4 | No, as exported (GLM-5V and DeepSeek-V4-Flash-Vision-Exp are separate checkpoints) | Out of scope |

## How a Qwen3.5 image request works

Checked against transformers 5.14.1, `models/qwen3_5_moe/modeling_qwen3_5_moe.py` [verified]:

1. **Preprocessing** (`Qwen2VLImageProcessorFast`): smart-resize to a multiple of
   `patch_size · merge = 16 · 2 = 32` px within min/max pixel bounds, then bicubic resize,
   normalize, and patchify to `pixel_values [num_patches, C·T·16·16]` with
   `temporal_patch_size = 2` (a still image is duplicated in time). It also produces
   `image_grid_thw = [1, H/16, W/16]`.
   - The real `preprocessor_config.json` for both Qwen3.5-35B-A3B and 397B-A17B, fetched
     from the HF hub 2026-10-14 [verified]:
     - `shortest_edge` = 65,536 px (256²);
     - `longest_edge` = 16,777,216 px (4096²);
     - `image_mean` = `image_std` = `[0.5, 0.5, 0.5]`.
   - So the model will take a 16.7 MP image of about **16k tokens**. Cascadia's default
     pixel cap must sit far below the model's own bound.
2. **Token expansion:** the chat template emits `<|vision_start|><|image_pad|><|vision_end|>`.
   The processor expands `<|image_pad|>` to `grid_h·grid_w / 4` copies, so a
   448×448 image becomes 196 tokens.
3. **Vision tower:** the ViT runs, then the 2×2 merger gives `[n_img_tokens, hidden]`.
   The #174 builder splits this into three IRs: patch embed (`vision_embeddings`),
   learned position embed (`vision_embeddings_pos`), and blocks plus merger
   (`vision_embeddings_merger`).
4. **Splice:** `inputs_embeds.masked_scatter(input_ids == image_token_id, image_embeds)`
   (`modeling_qwen3_5_moe.py:1666`). **This is the only way vision enters the language
   model.** `Qwen3_5MoeTextModel.forward` takes `inputs_embeds` and `position_ids` and
   nothing vision-specific (`:1264-1331`). Grepping the file finds no `deepstack` and
   no `visual_pos_masks`. (Qwen3-VL *does* have DeepStack, injecting into the first 3
   decoder layers. That model is not in our catalogue. If it is ever added, the vision
   output must reach the stage that owns layers 0-2.)
5. **Positions** (`get_rope_index`, `:1402-1493`):
   - Text runs advance all three rows (t, h, w) together.
   - An image at `current_pos` gets `t = current_pos`, `h = current_pos + row`,
     `w = current_pos + col`.
   - `current_pos` then advances by `max(grid_h, grid_w) / 2`, **not** by the token
     count. This is a difference from Qwen2.5-VL.
   - `rope_delta = max_pos + 1 − seq_len` is negative after an image.
   - Decode positions are `kv_len + rope_delta` in all rows (`:1613-1623`).
   - The language model takes 4 rows; row 0 is the text position used for mask
     construction (`:1283-1295`).
   - OpenVINO GenAI 2026.4.1 builds exactly this for the exported IR [verified,
     `src/cpp/src/visual_language/qwen3_5/classes.cpp:41-108`]:
     - `position_ids [4,B,T]`;
     - row 0 is the plain token index `0..T`;
     - rows 1-3 are the 3D vision positions;
     - in decode, row 0 is `history_size` and rows 1-3 come from `rope_delta`.
   - The same file states "Qwen3.5 LM has no additional vision-related inputs"
     (`:110-113`), which confirms there is no DeepStack. Qwen3-VL, by contrast, feeds
     `deepstack_visual_embeds` and `visual_pos_masks`.

## Design

### A. API and task (all engines)

**Constraint: Cascadia is OpenAI API compatible.** Image input must use OpenAI's wire
format exactly, with no Cascadia-specific fields. An unmodified `openai` Python or JS
client, or any tool built on one, has to work unchanged. Where we can't support
something, we fail the way OpenAI fails: same status code, same error object.

**Existing gaps to fix first.** Both are independent of VLM and both are [verified]:
1. A text-only message sent as a content-part array,
   `"content": [{"type":"text","text":"hi"}]`, fails to deserialize on *every* engine
   today. The cause is `de_null_content` taking `Option<String>` (`lib.rs:685`). OpenAI
   SDKs and many agent frameworks send this form.
2. Non-streaming errors are `{"error": "<string>"}` (`lib.rs:2246,2274,2284,…`). OpenAI's
   format is `{"error": {"message", "type", "param", "code"}}`. Streaming errors already
   use an object (`lib.rs:2724`) but leave out `param` and `code`. SDKs parse
   `error.message` and `error.type`, so a string-valued `error` breaks their exception
   mapping.

**Request shape (Chat Completions)**, matching OpenAI:
```json
{"role": "user", "content": [
  {"type": "text", "text": "What is in this image?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,…", "detail": "auto"}}
]}
```
- **Content type.** `ChatMessage.content` becomes untagged `null | String | Vec<ContentPart>`.
  `ContentPart` is tagged on `type`: `text { text }` and `image_url { image_url: { url, detail? } }`.
- **`detail`.** Accept `low | high | auto`; it is optional and defaults to `auto`. It maps
  to our pixel budget:
  - `low` means a fixed small resize, about 512×512 or fewer tokens;
  - `high` and `auto` mean the model's max-pixels bound.
  Unknown values return a 400, as OpenAI does.
- **Which roles may carry which parts.** Image parts are valid only on `user` messages.
  `system`, `developer` and `assistant` messages may use text-only part arrays; an image
  part there is a 400 naming the `param` (e.g. `messages[0].content[1]`).
- **Other OpenAI part types.** `input_audio`, `file` and `refusal` deserialize but return
  OpenAI's "unsupported content type" style error. They must not fall through to a serde
  failure.
- **Image sources.** OpenAI accepts `https` URLs and `data:` URIs.
  - `data:` URIs are always on.
  - `https` URLs are a real compatibility expectation; many clients pass public links.
    They are supported but **off by default**, enabled with `--api-allow-image-urls`.
    They then need an allow-list, a byte cap, a timeout, and no private or link-local
    addresses after DNS resolution, to block SSRF.
  - With fetching off, a URL image returns a 400 with an explicit message, never a
    silent drop.
  - Supported formats are OpenAI's set (PNG, JPEG, WebP, non-animated GIF), decoded with
    the `image` crate. Anything else is a 400.
- **Responses API.** Cascadia has no `/v1/responses` route today. If it is added, its
  `{"type":"input_image","image_url":…}` part maps onto the same internal `ImageInput`.
  The decisions-API branch already handles the `input_image` shape.
- **`/v1/completions`.** It stays text-only. OpenAI's legacy Completions API has no
  image input either, so this is compatible.
- **`usage`.** `prompt_tokens` counts the expanded image tokens, as OpenAI's does, so
  clients that budget by usage see real cost.
- **Errors.** Every new rejection uses the OpenAI error object with the right `type`
  (`invalid_request_error`), `param`, and status:
  - 400: bad part, unsupported image, images sent to a text-only model;
  - 413: over the byte cap.
- **`/v1/models`.** OpenAI's list has no capability field. Advertising vision support
  there needs an additive field that clients ignore (e.g. `"capabilities": {"vision": true}`),
  so the standard schema stays untouched.
- **Limits.** Add a per-image byte cap, an image-count cap and a total pixel cap,
  separate from `max_prompt_bytes`. That limit should count rendered *text* only. The
  body limit needs a separate, larger default when vision is enabled, for example
  `--api-max-image-mb`. Defaults should accept what typical OpenAI clients send, which
  is a few base64 phone photos. The exact byte cap is a policy choice, not an OpenAI
  number we verified. The **pixel** cap matters more for cost: Qwen3.5 allows 16.7 MP
  (about 16k tokens), so cap per image (e.g. about 1-2 MP for `auto`/`high`) and
  downscale rather than reject.
- **Conformance tests.** Drive the server with the official `openai` Python SDK.
  Exercise text parts, `image_url` with a data URI, `image_url` with an https URL,
  multiple images, streaming with `include_usage`, and each error path, checking the
  SDK raises the expected exception class. There is no SDK-based suite today
  [verified]. `tests-e2e` drives the binary over raw HTTP with `reqwest` against the
  mock engine, and `tests/conformance` covers only `--elastic`. Nothing imports
  `openai`. The SDK suite is new. It can run against the mock engine for wire-format
  checks, with one test against a real VLM IR.
- **Template.** Pass structured parts into the template (`{"type":"image"}` /
  `{"type":"text","text":…}`) so the model's own `chat_template.jinja` emits the vision
  tokens. The existing `sanitize_numeric_dot_index` hook already copes with templates
  that branch on list content.
  - The ov-genai "thinking ON" legacy path (`lib.rs:1744-1751`) sends only the last user
    message. With images it must keep the image order consistent with the prompt.
- **Task.** Add `media: Vec<MediaInput>` to `GenerationTask`, where
  `MediaInput { kind: MediaKind, frames: Arc<[u8]>, num_frames, width, height,
  timestamps_ms: Option<Vec<u32>>, sha256 }` and `MediaKind` is `Image` (one frame,
  no timestamps) or, later, `Video`. Decode once, in the API. Engines get pixels, never
  URLs. `GenerationTask` is in-process only, so this costs no wire versioning. See
  "Video: out of scope, kept cheap" for why the shape is general from day one.
- **Capability gate.** Each engine reports `supports_images()`. Image parts sent to a
  text-only engine get an OpenAI-shaped 400 (`invalid_request_error`, "model X does not
  accept image input"), not a serde error.

### B. Single box: ov-genai `VLMPipeline`

- **ABI.** Extend the shim with an images argument:
  `cascadia_pipeline_generate(..., const uint8_t* const* rgbs, const size_t* hw, size_t n)`.
  Build `ov::Tensor{u8, [1,H,W,3]}` per image (zero-copy over the Rust buffer for the
  duration of the call) and call `vlm->generate(prompt, images, cfg, streamer)`.
  - GenAI does the resizing, patchifying, normalizing and token expansion itself, so
    Cascadia sends raw RGB only.
  - **Placeholders** [verified, GenAI 2026.4.1 `inputs_embedder.cpp:620-660`,
    `qwen2vl/classes.hpp:122`]:
    - GenAI looks for the native tag `<|vision_start|><|image_pad|><|vision_end|>`,
      which is exactly what the HF chat template emits for an image part. It maps tags
      to images in order and expands each one to the right token count itself.
    - The tag count **must equal** the image count, or GenAI asserts.
    - Universal tags (`<ov_genai_image_i>`) are also accepted, but the two kinds cannot
      be mixed in one prompt.
    - With no tags at all, GenAI *prepends* the tags to the prompt. For a pre-templated
      prompt that would put the image before `<|im_start|>`, so the API must always
      render image parts through the template.
    - Re-templating is not a problem [verified on pawan-04, 2026-10-15]. The engine
      already sets `apply_chat_template=false` when the API renders the template
      (`cascadia-cli/src/lib.rs:2293`). A pre-templated prompt with the native tag and
      templating off answered correctly on the 35B IR (see Phase 0 results).
  - Upstream also ships a C API (`ov_genai_vlm_pipeline_generate(..., const ov_tensor_t** rgbs, size_t num_images, ...)`).
    We keep our own shim for consistency.
- **Batching.** `VLMPipeline` handles one request at a time. GenAI 2026.4.1's
  `ContinuousBatchingPipeline` **does** take images [verified]:
  `add_request(id, prompt, const std::vector<ov::Tensor>& images, config)` and a
  videos overload (`continuous_batching_pipeline.hpp:284-285`).
  - That makes `--cb` with images a shim change: add an images-taking `add_request`
    next to the existing string one at `shim.cpp:451-525`.
  - Commit `0d82544a` (Tate, 2026-08-04, only on `origin/perf/spec-scheduler-cache`)
    only lifts the `--cb` rejection for VLM layouts in `genai.rs`, for text-only
    requests. Port it as the first step.
- **Effort.** Small: API work from A, about 100 lines of shim and FFI, and a test with
  the 35B IR.

### C. Staged multi-node: `qwen35` engine

Rank 0 already owns the tokenizer, the text-embeddings IR and the decode driver.
Everything image-related stays there.

1. **Vision runner on rank 0** (new module, e.g. `qwen35_vision.rs`). It compiles the
   three vision IRs through the shim's generic `Runtime`.
   - **Preprocess** in Rust (`image` + `fast_image_resize` with a bicubic filter that
     matches the PIL resample), using constants from `preprocessor_config.json`.
     Parity-test it against the HF processor on a fixed image set.
   - **Host glue [verified, GenAI 2026.4.1 `qwen3_vl/classes.cpp:455-622`; Qwen3.5
     reuses `VisionEncoderQwen3VL`].** The vision IRs are not self-contained. The host
     must:
     - compute bilinear-interpolation corner indices from `grid_thw` and feed them to
       `vision_embeddings_pos` (`input [4,N]` i64, output `[4,N,1152]`). The weights
       are not an IR input: the host applies them and adds the weighted 4-corner sum
       to the patch embeddings (`:455-513`);
     - compute `rotary_pos_emb` (`[N,36]` f32) for the merger (`get_rotary_pos_emb`);
     - build a block-diagonal `attention_mask` (`[1,N,N]` f32). The 35B export takes
       the mask, not `cu_seqlens` [verified from the IR on pawan-04] (`:609-621`);
     - read `last_hidden_state` back out (`[N/4, 2048]`, one row per image token).

     None of this is in our tree. Rank-0 vision is therefore a port of GenAI's
     Apache-2.0 Qwen2-VL/Qwen3-VL embedder logic: preprocessing, the pieces above, and
     `create_position_ids`. That is several hundred lines of C++ to Rust, not a thin
     wrapper. The alternative is a small C++ entry point in our shim that links GenAI's
     *internal* `InputsEmbedder`. That is less code but depends on non-public GenAI
     headers that change between releases.
   - **Device placement.** The tower is about 0.9 GB uncompressed. Rank 0 can run it on
     the iGPU while decode runs elsewhere. It is not measured yet.
2. **Splice.** Tokenize the rendered prompt and expand each `<|image_pad|>` to the
   image's token count. Then `embed_seq` the ids and overwrite the image rows with the
   vision output before `chain_pass` (`qwen36.rs:1689`, `:2424`).
   - Prefill chunking (`PREFILL_CHUNK = 256`) can split an image span. That is fine for
     embeddings because rows are independent, as long as positions are sliced from the
     same precomputed tensor.
3. **Positions.** Compute the full `[4, 1, T]` position tensor and `rope_delta` once per
   task, as in `get_rope_index`. `chain_pass_inner` then slices it instead of building
   `t0..t1` (`qwen36.rs:997`). Decode uses `step + rope_delta` in rows 1-3, while row 0
   stays the plain token index (prefill `0..T`, decode `history_size`). This matches
   GenAI's Qwen3.5 embedder [verified, see above]. The 35B IR's language model takes
   `position_ids [4,?,?]` i64 and `inputs_embeds [?,?,2048]` [verified from the IR].
4. **Wire.** Downstream ranks currently derive positions from the FORWARD header's
   `pos`. Two options:
   - **(a)** Add an optional i64 position tensor to FORWARD under a new append-only
     frame kind.
   - **(b)** Send `rope_delta` once per task and the image spans (offset, `grid_thw`) so
     each rank recomputes positions.

   Option (a) is simpler and costs `4·n·8` bytes per chunk, which is negligible next to
   `n·hidden·4`. Hidden states are unchanged and image bytes never leave rank 0.
5. **Prefix cache.**
   - Make the key `(token_ids with each image span replaced by a per-image content hash)`.
     This follows vLLM's `mm_hash`.
   - DeltaNet state can't be trimmed, so a hit is only useful up to the first differing
     image.
   - Simplest first cut: skip snapshot *restore* for tasks with images, but still
     snapshot text-only prefixes.
   - `KvCoordination::lookup` needs the same treatment before cross-node reuse is
     allowed for image tasks.
6. **Encoder cache.** An LRU of `sha256 → (embeds, grid_thw)` on rank 0 lets multi-turn
   chats that resend the same image skip the ViT. Optional; add it after the basics work.
7. **Export.** `export_qwen36_moe.py` copies the three vision IRs,
   `preprocessor_config.json` and `video_preprocessor_config.json`, and records them in
   `manifest.json` (e.g. `"vision": {"embeddings": …, "pos": …, "merger": …}`) so the
   engine can detect vision support from the manifest.

### D. Other families (later)

Each family needs two things:
- an exporter that keeps the vision tower;
- a stage 0 that accepts `inputs_embeds`:
  - `ov-runtime`, `dist-spec` and `gemma4` gather embeddings inside the stage-0 IR from
    `input_ids`;
  - sparse-moe already has hidden-in entry points below `embed_token`
    (`staged.rs:80-87`).

Gemma 4 has an extra problem. Its per-layer embeddings are derived from `input_ids`
[inferred], so image positions need a defined id or a zero row. Treat each family as
its own project once A-C have landed.

## Hard problems and risks

- **M-RoPE correctness.** Off-by-one mistakes in positions don't crash; they degrade
  answers quietly. We need a parity harness: logits from HF transformers (35B, one
  image) against the Cascadia staged chain, the same way `--validate` works for text
  today.
- **Vision host glue.** Confirmed: the merger needs host-side rotary embeddings and
  attention masks, and the pos IR needs host-computed interpolation. Rank-0 vision is
  therefore a port of GenAI's Qwen-VL embedder (several hundred lines), with its own
  parity tests. This is the largest single item in Phase 2.
- **Toolchain freshness.** Qwen3.5 VLM exports need recent optimum-intel, transformers
  5.x and OpenVINO ≥ 2026.4. One external report describes an image-shape bug in GenAI
  2026.3.1 that is fixed on master. We pin 2026.4.1, and PR #160 notes 2026.5 fixed a
  VLM IR load SIGSEGV. Phase 0 must establish which version actually works.
- **Memory on 64 GB ranks.** The CPU plugin holds about 2× the IR. Adding about 1 GB of
  vision plus activations for large images to rank 0 needs a budget. Pixel caps also
  bound the ViT's quadratic attention cost.
- **Cache poisoning.** Any cache keyed on token ids alone would serve KV from a
  different image. This must be fixed before any image task touches a cache.
- **Security.** Remote image fetch (SSRF), decompression bombs (pixel caps before full
  decode) and per-tenant byte budgets.

## Phased plan

| Phase | Scope | Exit criterion |
|---|---|---|
| 0. Verify | Introspect the 35B and 397B VLM IR inputs (`position_ids` rows, merger inputs). Run Python `VLMPipeline` with an image on 35B under our pinned OpenVINO. Check for a CB images overload. | Written answers to every open question below |
| 1a. OpenAI compat fixes | Text content-part arrays on all engines; OpenAI error object everywhere | The unmodified `openai` SDK works against every engine with array content, and its error paths raise the right exception classes |
| 1b. API + single box | Rest of section A plus section B; capability gate; limits | The unmodified `openai` SDK sends an `image_url` request to `cascadia run --engine ov-genai` on the 35B VLM IR and gets a coherent answer with correct `usage`; text-only engines return OpenAI-shaped 400s |
| 2. Staged rank-0 vision | Section C items 1-5 plus the exporter (C7) | 35B six-stage chain matches HF logits on image prompts (top-1, top-5 within tolerance); prefix cache never hits across different images |
| 3. 397B with images | Serve the #174 artifact across 64 GB ranks; measure ViT latency and placement | The open #174 test-plan item ("Serve stage shards…") passes with an image prompt |
| 4. Other families | Gemma 4, Mistral 3, Inkling, one at a time | Per-family RFC |

## Open questions

Answered from source on 2026-10-14 (GenAI tag `2026.4.1.0`, transformers 5.14.1, HF
hub configs); details are in the sections above:

1. ~~Position input layout.~~ **Answered:** `[4,B,T]`; row 0 is the token index, rows
   1-3 are M-RoPE (GenAI `qwen3_5/classes.cpp:41-108`). The 35B IR's `position_ids` is
   `[4,?,?]` i64.
2. ~~Merger inputs.~~ **Answered** from the 35B IR:
   - embeddings: `hidden_states [?,?]` f32 -> `[?,1152]`;
   - pos: `input [4,?]` i64 -> `[4,?,1152]` (weights applied on the host);
   - merger: `hidden_states [?,1152]`, `attention_mask [1,?,?]` f32,
     `rotary_pos_emb [?,36]` f32 -> `[?,2048]`.
   The #174 builder's own output should be checked the same way when one exists.
3. ~~Placeholder.~~ **Answered:** the native tag
   `<|vision_start|><|image_pad|><|vision_end|>`, one per image, with the count checked.
   Re-templating is not an issue (see B).
4. ~~Continuous batching with images.~~ **Answered: yes** in 2026.4.1
   (`continuous_batching_pipeline.hpp:284-285`).
5. ~~Rust vs GenAI internals.~~ **Decided:** write the rank-0 vision glue in Rust,
   after Phase 1b, and test it against GenAI's output on the same images.
7. ~~End to end on our pinned GenAI.~~ **Answered: yes.** See Phase 0 results.

6. ~~Video in scope?~~ **Decided: no**, but every image-path interface is built so
   video slots in without rework. See "Video: out of scope, kept cheap".

No questions remain open.

## Phase 1b: what was built and verified

Verified on pawan-04 against GenAI 2026.4.1 and the 35B export before coding:

- **Universal tags work.** `<ov_genai_image_N>` placed in a pre-rendered prompt
  (`apply_chat_template=false`) answers correctly for one image and keeps order for two;
  a tag/image count mismatch raises "Missing image/video with index 1". The API inserts
  universal tags, so the same code serves every GenAI VLM family.
- **CB images overload honours `apply_chat_template`.** Off: the pre-rendered prompt is
  used as is (218 input tokens, correct answer). On: GenAI wraps it again (227 tokens,
  thinking switches on). Unlike the string overload, no shim-side templating is needed.
- **Exact prompt length is available.** `VLMDecodedResults.perf_metrics` and the CB
  handle's `get_perf_metrics()` both report input tokens including image tokens (218 for
  one 448x448 image), so `usage.prompt_tokens` is exact.

Built (local commits on `pawan/vlm-research`):

- Task: `GenerationTask.media: Vec<MediaInput>` (frame stack, kind-tagged) and
  `prompt_rendered`.
- Shim: `cascadia_media_t` list on `cascadia_pipeline_generate` and
  `cascadia_cb_add_request`; `cascadia_cb_handle_input_tokens`. Media on a plain LLM
  export is an error.
- Engine: media passed on both paths; `skip_chat_template` also when `prompt_rendered`;
  prompt tokens from perf metrics with the tokenizer count as fallback.
- API: `image_url` parts (object or bare-string form) on `user` messages; `data:` URLs
  only (PNG, JPEG, WEBP, GIF); header check before decode (50 MP, 16384 a side, 200:1);
  downscale to 1 MP (512x512 for `detail: low`); SHA-256 over the pixels; at most 8 per
  request. Image requests always render the template API-side, and every image must
  reach the rendered prompt exactly once or the request is a 400. Image requests take
  the in-flight permit before decoding, and at most 2 decodes run at once.
- CLI: enabled for `--engine ov-genai` when `openvino_vision_embeddings_model.xml` is
  present; body limit grows by 20 MiB of image bytes.

Follow-ups, also built:

- `/v1/models` reports `input_modalities` (`["text", "image"]` on a VLM, else `["text"]`).
- Remote images behind `--api-allow-image-urls` (off by default): `https` on port 443
  only, no credentials, no proxy; every resolved address must be public and the
  connection is pinned to it (no rebinding); at most 3 redirects, each re-checked;
  `image/*` only; the request's 20 MiB byte budget; one 10 s deadline per image across
  DNS, hops and body, and 20 s for all of a request's fetches. Errors never reveal what a
  refused host resolved to or why a connection failed.
- Malformed-body 400s carry serde's reason and JSON path only.
- `tests/conformance/test_openai_sdk.py`: the pinned `openai` SDK against the mock
  engine, run in CI.

## Phase 2: what was built and verified (2026-10-15)

Rank-0 vision for the staged `qwen35` engine, in Rust (`crates/cascadia-engine-openvino/src/qwen35_vision/`):

- **Preprocessing** (`mod.rs`, `resample.rs`): `smart_resize`, a bicubic resize that
  reproduces Pillow's fixed-point `Resample.c`, rescale/normalise in the PIL backend's
  dtypes, temporal padding, patch layout. Bit-identical in f32 to transformers 5.14.1's PIL
  processor on seven fixtures (including a 2.8x antialiased downscale), within one 8-bit
  level of the torchvision processor.
- **M-RoPE** (`rope.rs`): `get_rope_index` for images and videos (per-frame split), the
  4-row layout, decode positions from `rope_delta`. Exact against HF on one image, two
  images, multi-turn, a 4-frame video and text only.
- **Vision IR host inputs** (`encoder.rs`): bilinear indices/weights (torch.linspace's f32
  rounding reproduced), weighted corner sum, `rotary_pos_emb`, per-frame `cu_seqlens` and
  the merger's block-diagonal mask; `VisionEncoder` drives the three IRs. On pawan-04 the
  Rust chain matches HF's processor and `vision_utils` driving the same IRs at every stage
  (gaps ~1e-16 on sampled values).
- **Engine** (`qwen36.rs`, `prompt.rs`): admission encodes images and expands each
  `<|image_pad|>`; prefill splices image rows over pad embeddings (also across chunk
  boundaries); `chain_pass` takes explicit positions. Pipelines send `FRAME_FORWARD_MROPE`
  (kind 13) for image turns only, gated by the `MROPE_CAP` bit in the HELLO_ACK capability
  field; image turns are refused (never answered text-only) without vision IRs, on a chain
  lacking the bit, when resumed, or past the window. Image turns skip every prefix/KV cache.
- **API/CLI/exporter**: `MediaConfig::qwen35_native` (native tag, reserved tags refused in
  caller text); enabled for `--engine qwen35` when the tree has the vision IRs;
  `export_qwen36_moe.py` copies them and records `manifest.vision`.

End to end on pawan-04 (Qwen3.6-35B-A3B 4-stage tree, single box, CPU), unmodified
`openai` SDK: one image "A red square and a blue circle." (`prompt_tokens` 228), two images
described in order (433), streaming and refusals as in Phase 1b. Answers and token counts
are identical to the GenAI `VLMPipeline` path. Image encode about 1.8 s per image on CPU.
The pipeline wire is covered by an in-process two-rank test over a real socket.

Still open:

- ~~A multi-rank run on hardware with images.~~ Done 2026-10-15: two machines over
  Tailscale, rank 0 (stage 0 + vision IRs) on pawan-04 and rank 1 (stage 1) on pawan-02,
  the Qwen3.6-35B 2-stage tree. HELLO negotiated `mrope=true`; the full SDK check passes
  (one image "A red square and a blue circle.", `prompt_tokens` 228, 13 s; two images in
  order; streaming; refusals).
- ~~Prefix reuse for image turns.~~ Done: single-box image turns key the local prefix
  cache on `VisionPrompt::cache_key` (pad tokens replaced by pseudo-ids from the image's
  pixel hash, row and tenant). Measured on pawan-04: a follow-up turn on the same image
  restored past it (72.5 s -> 11.5 s); a different image reused only the system prompt and
  answered about its own image. The local text prefix cache is not tenant-namespaced (a
  pre-existing property); image spans are, once tenants are asserted.
- ~~Vision IR placement.~~ `--vision-device GPU` (default: the stage device). Measured on
  pawan-04 (Arc 140V, OV 2026.4.1), stages on CPU: warm encode of a 640x360 image 1.9 s on
  CPU vs 188 ms on the iGPU; two images 4.4 s vs 0.46 s; SDK answers identical. The iGPU
  runs f16 and its final embeddings drift up to 14% (sampled max-abs, relative) from the f32
  reference, so CPU stays the default; the f32 precision hint on the iGPU gives wrong merger
  output in this OV release (patch/position stages exact) and is not offered. The iGPU
  shares system memory: on a 32 GB box holding the 35B stages it left 0.2 GB free.
- HF-logit parity on image prompts (Phase 2 exit criterion) beyond the GenAI answer match
  (needs the original weights); 397B (Phase 3).

## Other families on the single-box path (2026-10-15)

`--engine ov-genai` uses GenAI's universal `<ov_genai_image_N>` tag, so any VLM GenAI
supports works without family code. Verified with Gemma 4 26B-A4B (int4, CPU, pawan-04)
and no code change: the full SDK check passes (one image "There is a red square and a
blue circle.", `prompt_tokens` 289; two images in order; streaming; OpenAI-shaped
refusals). Staged engines need family host code and a stage 0 that takes embeddings
(section D); Gemma 4 on `ov-runtime` is next.

## Gemma 4 on the staged `ov-runtime` engine (2026-10-15)

The 26B-A4B and 31B have no real per-layer embeddings (the input is `[?,?,?,0]`), so the
section D problem does not arise for them; E2B/E4B are still refused by the surgery.

- **Export.** `tools/gemma4_surgery/export_gemma4_text.py --vision` keeps stage 0's
  `inputs_embeds` (tagged `grafted_inputs_embeds`) and every stage's `token_type_ids`,
  copies the text embedding and vision IRs and the processor configs, keeps the VLM
  config as `config.vlm.json`, and writes a `vision` block into `pipeline_config.json`.
  Checked in Python on the 2-stage 26B-A4B tree before any Rust: stage 0 then stage 1
  agree with the uncut language model on the next token for a text and an image prompt
  (max logit gap about 4% of the largest logit, the f16 hop), and greedy decoding
  through the stages answers "There is a red square and a blue circle in this picture."
- **Preprocessing** (`gemma4_vision`): HF's aspect-preserving size (edge cases
  included), Pillow-exact bicubic shared with Qwen, rescale without normalisation,
  `(row, col)` patches with `(x, y)` positions padded to 2520 rows with `-1`, and the
  `<|image>` + N x `<|image|>` + `<image|>` expansion. Bit-identical to HF's PIL
  processor on nine fixtures (`tests/fixtures/gemma4_vision`); HF's default torchvision
  processor is not Pillow-exact (about 45% of values differ, by at most 2/255), so that
  is the tolerance against it.
- **Engine.** Rank 0 embeds every step through the text embedding IR, encodes images
  on CPU and splices their rows into the prefill. `token_type_ids` (1 on image soft
  tokens) cross the wire as a third frame `[lead] [hidden] [types]`, because every
  stage's sliding-window layers attend bidirectionally inside an image and only rank 0
  knows where images are. The stateful path prefills the whole prompt in one inference,
  so an image is never split across chunks; static, chunked-prefill and packed shards
  serve Gemma 4 too since 2026-10-16 (static only; see "Gemma 4 on the static
  path"). Turns with
  images first skipped KV warm-resume and capture, because every same-size image has
  the same ids; they resume since 2026-10-16 (see "KV warm-resume for image turns").
- **API.** `MediaConfig::gemma4_native` uses the template's own `<|image|>` and reserves
  the image/video/audio markers in user text; the CLI enables it for `ov-runtime` when
  the pipeline has a `vision` block.
- **Fixed on the way:** `ov-runtime`'s single-task stream reported `usage.prompt_tokens`
  0 and always finished `"stop"`; it now reports both, like the packed path.

Verified with both ranks of the 2-stage 26B-A4B tree on pawan-02 over the TCP wire
(CPU): the SDK check passes 11/11. That covers one image (`prompt_tokens` 289), two
images in order (554), streaming with low detail, text only, a same-size image right
after another (described correctly, so no stale KV), an image from an earlier turn,
the reserved marker refused, and OpenAI-shaped refusals. Image encode is about 6 s per
image on CPU; a one-image prefill is about 12 s; decode is 6-10 tok/s with both ranks on one
box.

**Across two machines** (4-stage tree, layers 0-6 / 7-13 / 14-20 / 21-29): ranks 0-1 on
pawan-02 and ranks 2-3 on pawan-04, with rank 1 sending to rank 2 over Tailscale. The SDK
check passes 11/11 with the same answers and `prompt_tokens` (289 and 554). One-image
prefill is about 20-26 s and decode 2-5 tok/s on CPU across the link. pawan-04 had 2.8 GB
free, so it needed no new weights: the `--vision` surgery leaves every stage after
stage 0 byte-identical to the text-only tree's (`openvino_model.bin` SHA-256 equal for
stages 1-3), so pawan-04 runs the new 2-3 MB `.xml` graphs over hardlinks to the text
tree's existing weights. Only stage 0 (and rank 0's embedding and vision IRs) are new
data. Any node already holding a text tree of the same split can serve the later
stages of a vision tree this way.

**Gemma 4 31B** (dense, 60 layers) passes the same way: the `--vision` surgery accepts it
(its per-layer input is empty, like 26B-A4B's), stages 1-3 are again byte-identical to
the text tree's, and the 4-stage tree across pawan-02 and pawan-04 passes the SDK check
11/11 (`prompt_tokens` 289 and 554; one-image prefill about 80 s on CPU).

## Gemma 4 E2B/E4B on the `gemma4` engine (2026-10-15)

E2B/E4B have real per-layer embeddings and KV sharing across layers, so the IR surgery
refuses them; their staged path is the `gemma4` engine (`tools/export_gemma4.py`, traced
from PyTorch). Its stage 0 embeds ids inside the IR, so the image splice goes in the
graph: `--vision DIR` adds a trailing `image_features [M, hidden]` input, image tokens
embed and look up their per-layer embedding as the pad id, their rows are replaced in
order by the image features, and the per-layer projection reads the spliced embedding,
which is what HF's `Gemma4Model.forward` does. E2B/E4B use plain causal attention
(`use_bidirectional_attention: null`), so no `token_type_ids` cross the wire. The encoder
is OpenVINO's published vision IR (`OpenVINO/gemma-4-E2B-it-fp16-ov`, Apache-2.0).

Verified with E2B, 2 stages, across pawan-02 (rank 0) and pawan-04 (rank 1): the SDK
check passes 11/11, and the answers are word-for-word HF PyTorch's (fp32, same images):
"The image contains a red square and a blue circle." and "the first picture contains
red and blue, and the second picture contains green and yellow." Prompt token counts
then differed from HF's by 1 and 4 (285 vs 284, 550 vs 546); fixed since, see "Prompt
tokens match HF" below. Encode is about 3 s per image. The stage-1 IR exported on each node is byte-identical in weights.

**Gemma 4 E4B** (`google/gemma-4-E4B-it`, Apache-2.0 and not gated; vision IR from
`OpenVINO/gemma-4-E4B-it-fp16-ov`) passes the same way: 11/11 across two machines,
"The two shapes are a red square and a blue circle.", stage 1 byte-identical across nodes.

## Pixtral on `ov-runtime` (2026-10-15)

The first non-Gemma family on the staged path, from `OpenVINO/pixtral-12b-int4-ov`
(LLaVA layout: Mistral text tower, Pixtral encoder; Apache-2.0).

- **Cut.** `tools/vlm_surgery/cut_vlm_lm.py` slices an OpenVINO VLM export's language
  model (which already takes `inputs_embeds`) into stages with the Gemma surgery's stage
  extraction, and writes a `vision` block (`family: "pixtral"`, `token_type_ids: false`):
  Pixtral attends causally, so the wire stays `[lead] [hidden]`.
- **Rank 0** (`pixtral_vision`): HF's Pixtral resize (fit the longest edge in 1024 with
  floor, then round each side up to 16-px patches), CLIP normalisation, the vision IR,
  and each `[IMG]` expanded to its rows of `[IMG]` closed by `[IMG_BREAK]` and, last,
  `[IMG_END]`. Bit-identical to HF's PIL processor on the eight fixture images; that
  surfaced a Pillow 12 rule (an image more than 100:1 tall shrinks height first, in two
  calls) now mirrored in the shared resampler.
- **Verified** across pawan-02 and pawan-04 (2 stages, CPU): 11/11, e.g. "There is a red
  square and a blue circle." and "first picture has red and blue, second picture has
  green and yellow."; one 448x448 image is 832 prompt tokens, encoded in about 1 s.
- **GenAI on the same export is wrong.** Single-box `ov-genai` needs two fixes to load it
  (the repo ships no tokenizer IRs; GenAI 2026.4.1 rejects `patch_size` as an object) and
  then answers incorrectly ("a square and a triangle. The square is blue", "black and
  white"), with 903 instead of 832 prompt tokens: its Pixtral preprocessing differs from
  HF's. The staged path, with HF-exact preprocessing on the same IRs, answers correctly,
  so this is GenAI's, not the export's. Pixtral is served by `ov-runtime`, staged or
  as a one-stage tree on a single box (next section).
- **Root cause (2026-10-16).** GenAI 2026.4.1 has no Pixtral path: `model_type:
  "llava"` maps it onto LLaVA-1.5 (`src/cpp/src/visual_language/llava/classes.cpp`,
  tag `2026.4.1.0`), which resizes the shortest edge, center-crops a fixed square, and
  expands each image as a flat run plus "\n", without `[IMG_BREAK]` after each patch
  row or `[IMG_END]`. Pixtral keeps the aspect ratio (longest edge fit to whole 16-px
  patches) and needs those tokens. `ov-genai` now serves a Pixtral export as text only
  (`genai::unsupported_vision`; the CLI logs why and the API answers images with a
  400), so it no longer gives wrong image answers. A fix belongs upstream: a Pixtral
  `InputsEmbedder` doing what `pixtral_vision` here does.

## Pixtral on `ov-genai`: no fix on our side (2026-10-19)

Traced in GenAI `2026.4.1.0` (paths under `src/cpp/src/`):

- **Model type.** `VLMConfig` reads `config.json` `model_type` (`visual_language/
  vlm_config.cpp:61`); the export says `"llava"`, which maps to `LLAVA` (line 18).
  `vision_config.model_type` is never read, and there is no `pixtral` or `mistral3`
  entry, so a Mistral 3 export fails to load (`Unsupported 'mistral3' VLM model type`,
  line 45). `inputs_embedder.cpp:366` and `vision_encoder.cpp:117` build the LLaVA-1.5
  classes.
- **Preprocessing** (`visual_language/llava/classes.cpp:11-33`): resize so the shortest
  edge is `size.shortest_edge`, center-crop to `crop_size`, CLIP-normalise. Pixtral's
  `preprocessor_config.json` has neither key (it has `size.longest_edge`), so the
  defaults apply: 336 and 336x336 (`processor_config.hpp:45-47`). Every image becomes a
  cropped square of 21x21 16-px patches; HF keeps the aspect ratio, fits the longest edge
  in 1024 and rounds each side up to whole patches.
- **Expansion** (`classes.cpp:92-113`): each image becomes `shape[1]` copies of
  `im_start` plus `"\n"`. `im_start` is the hard-coded `"<image>"` (`vlm_config.hpp:55`;
  no config key sets it), which the Pixtral (tekken) tokenizer splits into ordinary text:
  `<image>` alone is `<` `image` `>` (1060 5497 1062), but 441 of them and the newline
  are `<`, then `image` `><` (5497 4812) repeated, then `image` `>\n` (1561): 883 tokens.
  The image id is the last piece of `<image>` alone, `>` (1062; `classes.cpp:133-137`),
  which never occurs in the expanded run, so the merge (`utils.cpp:1038-1073`, which
  overwrites runs of that id) finds nothing and **no image row reaches the model**: it
  answers from the text `<image><image>...`. That is the 903: 20 text tokens + 883,
  against HF's 832 (20 + 28 x 28 `[IMG]` + 27 `[IMG_BREAK]` + `[IMG_END]`; checked on
  pawan-04 with the export's tokenizer). GenAI has no `[IMG_BREAK]` or `[IMG_END]` at all.
- **Newer GenAI.** `master` at `3869223` (2026-10-19) only adds `minicpmv4_7` to the
  model-type map and leaves `llava/` unchanged: no Pixtral path upstream yet.

Nothing inside our control yields HF's prompt. The pipelines take no `inputs_embeds`
(the public `VLMPipeline` and `ContinuousBatchingPipeline` take text and images only), so
the shim cannot hand GenAI `pixtral_vision`'s rows. Config edits cannot change the square
crop (fixed at load; the per-image options set only `patch_size`), the `<image>` tag or
the extra `"\n"`. A wrapped vision IR that appends the break and end rows, plus a
tokenizer IR that makes `<image>` one token, would still crop every image square and add
a newline: HF-exact for square images only, and fragile. So `ov-genai` keeps Pixtral
exports text only, and the API's `400` for an image now carries the reason and the
`ov-runtime` commands (`Config::no_media_reason`).

## Static (NPU) and packed paths for vision trees (2026-10-16)

`ov-runtime`'s static path (stateless shards, host-side KV ring) and its packed
multi-slot path (several requests in one inference, built on the static shards) now
serve causal vision trees. Pixtral is the one today.

- **Export.** `tools/vlm_surgery/static_variant.py` turns a stateful tree from
  `cut_vlm_lm.py` into the static contract `export_shards --target npu` writes. Per
  stage: `beam_idx` becomes the constant `[0]`; each KV `ReadValue` becomes a
  `past_key_values.{i}` input and each `Assign` a `present.{i}` output (f16, numbered
  by layer within the stage); the head's last-position slice is dropped so a prefill
  chunk returns every row; every shape is pinned. `--prefill-seq C` adds the chunked
  prefill IR. OpenVINO's own `stateful_to_stateless_transformation` rejects cut stages
  (their `beam_idx` Gathers do not match its pattern). `tools/packed_variant.py`
  then builds the packed IR from the static one unchanged: the optimum IR also feeds
  every SDPA from one additive f32 mask node.
- **Rank 0** embeds the prompt once (text rows from the embedding IR, image rows
  spliced in) and feeds `inputs_embeds` rows: one per seq=1 step, `C` per prefill
  chunk, or the slot's rows per packed inference. A decode step embeds the sampled
  token. Downstream stages are unchanged.
- **Packed slots** hold their own spliced prompt rows. The shared prefix (`--packed-prefix`)
  is matched and populated only up to a prompt's first image token: every image of one
  size has the same ids, so matching past it would reuse another image's K/V. Images
  are encoded at `submit`, so a prompt that outgrows the slot's region is refused there
  with the same 413 `context_length_exceeded` a long text prompt gets.
- **Gemma 4** (`token_type_ids` trees) was refused here at first; it is served on the
  static path since 2026-10-16 (next section) and on packed slots since the same day
  (see "Gemma 4 on packed slots").
- **Verified** across pawan-02 and pawan-04 (2 stages, CPU, context 2048):
  - static, prefill chunks of 256: 11/11, the same answers and token counts as the
    stateful tree ("There is a red square and a blue circle.", 832 prompt tokens; two
    images 1650);
  - packed, 2 slots of 32 rows: a red/blue and a green/yellow image of the same size
    plus a text prompt sent at once each get their own answer.
- **Single box.** `cut_vlm_lm.py --num-stages 1` now cuts a one-stage tree, so Pixtral
  is served on one box by `ov-runtime` (GenAI answers it wrongly, see above). On
  pawan-02: stateful 11/11 (one image in 27 s); packed with 2 slots and
  `--packed-prefix 64` (which is single-stage only) 10/11 with the two-image check
  refused 413 as designed (1650 tokens, 991-token slots), the same-size image swap
  answering about the new image with the shared prefix live, and the concurrent check
  passing.
- **Unit test:** `tools/tests/test_vlm_static_variant.py` converts a toy two-layer
  stateful stage and checks the port contract and that one stateless step with the
  stateful past fed explicitly matches the stateful step.

## Gemma 4 on the static path (2026-10-16)

Two things kept Gemma 4 off the static path; both are handled now.

- **Mixed KV geometry.** A stage mixes sliding layers (8 KV heads x 256) and global
  layers (2 x 512). `static_variant.py` records each layer's geometry in
  `stage_config.kv_layers`, and the host ring (`StaticKv`) sizes, feeds and absorbs
  every layer with its own heads and width.
- **Bidirectional attention inside an image.** The exported IR builds its image
  groups from `token_type_ids`, front-padding the past's types as text: an image sees
  itself bidirectionally only if the whole image is in the current inference. So a
  Gemma static tree needs a chunked prefill at least one image wide (280 tokens at
  most), and rank 0 never ends a chunk inside an image (`image_safe_take`; relays split
  their frames the same way). Every inference feeds its rows' token types, and the
  types frame rides the static wire as on the stateful one. Packed slots served
  `token_type_ids` trees later (next section).
- **Sliding window.** The IR masks the sliding window by ring index, and the ring's
  padding between the valid past and the query window makes the index distance run
  ahead of the real one. `static_variant.py` therefore refuses a context plus prefill
  width beyond the window (1024 + 1): 736 + 288 here, room for two images.
- **One compiled model on CPU/GPU.** Compiling a separate prefill IR doubles a
  stage's weights in RAM, and two 26B stages per 32 GB node ran out of memory.
  `--shared-prefill` writes one IR with a dynamic query width over the fixed past; the
  runtime runs both decode and prefill chunks on that one compilation
  (`stage_config.prefill_shares_decode`). The NPU still needs the two static IRs.
- **Verified:** Gemma 4 26B-A4B, 4 static stages across pawan-02 and pawan-04 (CPU,
  context 736, prefill 288, shared IR): 11/11, "There is a red square and a blue
  circle.", 288 / 550 prompt tokens, which is HF's processor count on the 26B template
  (its generation prompt adds an empty thought channel, hence 4 more than E2B).

## Gemma 4 on packed slots (2026-10-16)

**Memory.** On the 26B-A4B tree a packed stage (`--packed-seq 288`) compiles to about
18 GB resident (its mixture-of-experts layers at 288 rows), so two stages do not fit on
one 32 GB node: the second rank failed to allocate. Gemma 4 26B on packed slots needs
one stage per node (four nodes for the four-stage tree); it was not run end to end on
the two-node cluster. The mask, window, image-block, per-slot failure and wire logic are
covered by unit tests.

Packed slots (`--packed-slots N`) refused Gemma 4 trees because their host mask had no
image blocks and the packed IR builder expected one mask. They serve them now, single-
and multi-stage, text and image requests in the same inference:

- **Build.** `static_variant.py` as for the static path, then `tools/packed_variant.py
  <stage> --slots N --packed-seq S` per stage. A Gemma 4 stage feeds its SDPAs from two
  mask producers (`layers.0.self_attn/.../Slice_11` sliding, `layers.5...` full: the
  shared masks keep the name of the first layer that built them). The tool classifies
  each from `config.json` `layer_types` by that index, cross-checked against the layers
  the producer feeds when the stage has one SDPA per layer, replaces each with its own
  Parameter (`attn_mask_sliding_4d`, `attn_mask_4d`, both `[1,1,S,T]`), drops inputs
  left without a consumer (`attention_mask`, `token_type_ids`) and records
  `packed_masks`, `packed_sliding_window` and `packed_dropped_inputs` in
  `stage_config.json`. One full-attention producer (every other family) takes the
  original edit; its output is byte-identical to before.
- **Per-layer KV.** `PackedKv` sizes each layer's ring from `stage_config.kv_layers`
  (8 x 256 sliding, 2 x 512 global), as `StaticKv` does.
- **Sliding window.** The sliding port gets the same mask with every column whose
  absolute position is `window` (1024) or more behind the query row blocked (HF: visible
  iff `0 <= q - k < window`). Positions are known from the slot's layout: the shared
  prefix holds `0..`, the region the slot's own tokens from `shared_use` on, and once it
  has slid, its pinned sink columns keep the first own tokens while the rest hold the
  latest `region - sink`. A query row is at its slot's next position plus its order. So
  unlike the static IR, which masks by ring index, the packed window is exact for any
  region size.
- **Image blocks.** Rank 0 numbers each contiguous run of image soft tokens in a
  prompt (0, 1, ...); same-slot rows of one image are open both ways in both masks, on
  top of causal order. A packed prefill chunk never ends inside an image
  (`packed::image_safe_take`); submit refuses a prompt whose image is wider than
  `packed_seq` with a 400 (`UnsupportedInput`), and the load log warns when `packed_seq`
  is narrower than the widest image.
- **Wire.** Downstream stages rebuild the plan from the plan frame. On a
  `token_type_ids` tree it carries a fourth row, each row's image id (`-1` for text and
  idle rows): `[1,4,S]` instead of `[1,3,S]`. The row count is the version: a stage that
  masks image blocks refuses a 3-row frame, every other stage (and any build from
  before) refuses a 4-row one, so mixed builds fail loudly instead of masking
  differently. Other trees send the unchanged 3-row frame.
- **Still refused**, by name: a packed IR that still reads the 2D `attention_mask`
  (some other node consumes it), and a mixed-geometry Gemma stage whose packed IR has
  no sliding port (built without `layer_types`). `token_type_ids` kept by the tool
  (something besides the masks reads it) is fed 1 on image rows.
- **Constraints.** Every packed prefill chunk is `packed_seq` rows, and so is a decode
  step unless the stage has a narrower width (`packed_variant.py --dynamic-seq` on
  CPU/GPU, `--decode-seq` on the NPU), so `--packed-seq 288` (one image) alone makes
  each decode step a 288-row inference. `static_variant.py` still
  caps context plus prefill at the window + 1, because the static IR it also writes
  masks by ring index; at `--context 736` two slots hold 367 tokens each, one image
  and its text.
- **Verified** locally only (unit tests: masks for causal, window across the slide and
  the sinks, the shared prefix, image blocks and cross-slot isolation, mixed-geometry
  absorb, the wire round trip and version checks; a toy two-producer stage in
  `tools/tests/test_packed_variant.py` whose packed IR, fed the producers' own masks,
  matches the original). Not yet run on the 26B tree.

## On the NPU (2026-10-16)

Pixtral's static tree (2 stages, context 2048, prefill 256, separate decode and prefill
IRs) runs on the Lunar Lake NPU (Intel AI Boost, NPU 4000) of both nodes, `--device NPU
--ov-cache-dir`: 11/11, the same answers and HF's token counts (832 / 1648), one image
in 18 s against 53 s on CPU. The vision encoder and text embeddings stay on CPU. Each
stage IR takes about ten minutes to compile; the compile transient left at least 4.7 GB
free on a 32 GB node holding one 12B stage (decode + prefill).

Gemma 4 26B-A4B was sized on the NPU one stage at a time (stage 1 of the 4-stage vision
tree, `static_variant.py --context 736 --prefill-seq 288`, separate decode and prefill
IRs as the NPU needs, 11 GB on disk). Its decode IR alone did not finish compiling in 90
minutes (Pixtral: ten), using 9-11 GB of process memory throughout and still growing
when stopped; node free memory went from 28 to 17 GB. So Gemma 26B on these NPUs is not
practical: eight such compiles for four stages, two stages per 32 GB node each holding a
decode and a prefill compilation. The likely cost is the mixture-of-experts layers
(128 experts per layer, all compiled for static shapes). Gemma 4 26B runs on CPU static
and stateful trees (see above); on the NPU, Pixtral and other dense models are the fit.

## Mistral 3 on `ov-runtime` (2026-10-16)

`Mistral3ForConditionalGeneration` (Ministral-3 3B/8B/14B, Mistral Small 3.1/3.2) is
Pixtral's processor and vision tower plus a learned 2x2 patch merger. No OpenVINO
export is published; optimum-intel `main` (commit `d304fb56`; release 2.2.0 lacks the
`ministral3` text tower) exports it as four IRs: language model, text embeddings, vision
embeddings (tower + projector norm) and multi-modal projector (merger linear + MLP).

- **Rank 0** (`pixtral_vision` with `spatial_merge` 2): images resize to whole 28-px
  cells (longest edge 1540), the vision IR returns one row per 14-px patch,
  `merge_patches` regroups each 2x2 block channel-major exactly like `torch.unfold`, and
  the projector IR maps them to the text width. Each `[IMG]` expands to rows of tokens
  closed by `[IMG_BREAK]` and `[IMG_END]`.
- **Cut**: `cut_vlm_lm.py --family mistral3` (copies the projector IR into the root).
- **Parity**: the Rust preprocessing is bit-identical to HF's PIL processor on the
  eight fixture images and the prompt expansion reproduces its ids
  (`tests/fixtures/mistral3_vision`); the OV image chain against HF
  `get_image_features` (fp32): cosine 0.9994 on a 2x2-token image, 0.988 on 448x448,
  the vision IR's weights being int8 (as in OpenVINO's own Pixtral export).
- **Verified** with Ministral-3-3B-Instruct-2512 (Apache-2.0), int4: 11/11 on one box
  and across pawan-02 and pawan-04 (2 stages), "Square (red) and circle (blue).", 815 /
  1091 prompt tokens, which is HF's `apply_chat_template` count. The two machines'
  exports are byte-identical (stage 1 weights `1B0F50ED...`).
- **Found on the way:** optimum-cli on Windows writes `chat_template.jinja` with CRLF.
  Jinja2 normalises line endings; minijinja did not, so the template's default system
  prompt came out 51 tokens longer than HF's. `build_chat_env` now normalises `\r\n`
  and `\r` to `\n` for every template.

## More families on `ov-genai`, one box (2026-10-16)

OpenVINO's published int4 exports, served with `cascadia run <dir>` on pawan-02 (CPU)
through the same API path (universal `<ov_genai_image_N>` tags, parts back to back):

| Model | GenAI family | SDK check | One-image prompt tokens |
|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | `qwen2_5_vl` | 11/11 | 294 |
| Qwen3-VL-4B-Instruct | `qwen3_vl` | 11/11 | 223 |
| InternVL2-4B | `internvl_chat` | 11/11 | 282 |
| Phi-3.5-vision-instruct | `phi3_v` | 11/11 | 785 |
| llava-v1.6-mistral-7b | `llava_next` | 11/11 | 2956 (tiled, one image in 56 s) |
| Gemma-3-4B-it | `gemma3` | 11/11 | 286 |
| MiniCPM-V-4.7-1B | `minicpmv4_7` | does not load: GenAI 2026.4.1 rejects the model type | - |

## Prompt tokens match HF (2026-10-16)

A message with images reached the chat template as one string whose parts the API
joined with "\n". HF's templates write a content array's parts back to back (Gemma 4's
also trims each text part), and GenAI inserts its own tags the same way, so every join
cost one extra token. `MediaConfig` now joins parts with nothing between them and, for
Gemma 4, trims text parts (`trim_text_parts`). Text-only arrays keep "\n".

| Model (engine) | One image | Two images | HF |
|---|---|---|---|
| Gemma 4 E2B (`gemma4`, 2 machines) | 284 (was 285) | 546 (was 550) | 284 / 546 |
| Pixtral (`ov-runtime`, one box) | 832 | 1648 (was 1650) | 832 / 1648 |
| Qwen3.6-35B-A3B (`ov-genai`, one box, thinking off) | 227 | 429 | 227 / 429 |

All three pass 11/11 after the change. The staged `qwen35` path (Qwen3.6, 2 stages
across both machines) re-served on 2026-10-16: 11/11, 227 / 429 tokens, HF's counts.

## Branch review (2026-10-16)

Four read-only reviews (API, `ov-runtime`, vision preprocessing and the other engines,
the Python tools) found no high-severity defect; the fixes for what they found are in
the three `fix(...)` commits after this RFC's prompt-token entry. Re-verified after the
fixes: Gemma 4 E2B on `gemma4` across two machines 11/11 (284/546 tokens), Qwen3.6 on
`ov-genai` 11/11 (227/429), Pixtral packed with `--packed-prefix 64` 10/11 plus the
designed 413 and the concurrent check (image prompts now log `shared_prefix=18`, the
text before the first image).

Left open, not caused by this branch: downstream packed stages never `admit`, so with
`--packed-prefix` on a multi-stage tree their prefix bookkeeping would drift from rank
0's; the CLI refuses `--packed-prefix` on more than one stage, which keeps this
unreachable.

## KV warm-resume for image turns (2026-10-16)

The `kv_coord` cache keys a captured turn (prompt plus reply) on its token ids, and a
follow-up turn resumes from the longest cached strict prefix. Image soft tokens are one
id repeated, the same for every image of one size, so `ov-runtime`, `gemma4` and the
`qwen35` pipeline kept image turns out of the cache: every chat turn re-prefilled every
earlier image.

`vision_kv` builds the key instead: the prompt ids with each image's soft tokens
replaced by `pad_cache_id(tenant, sha256, row)`, the ids the `qwen35` prefix cache
already uses (high bit set, so never a real token id; per row, so two images agree only
if their pixels do). Capture, lookup and the downstream CAPTURE/RESTORE epochs all use
the key. A cached prefix whose KV depth falls strictly inside an image is refused (cold),
and on a resume only the images past the prefix are spliced into the suffix prefill.
Tenants namespace the stand-ins as they do the text cache. `qwen35` keys on
`VisionPrompt::cache_key`, the same ids; its image rows and M-RoPE positions are looked
up by absolute index, so the suffix prefill needed no change (decode positions, index
plus `rope_delta`, are what the next turn's prompt computes for those tokens).

Verified with a three-turn chat per image, two same-size images in different colours
(`sdk_vlm_kv.py`: A red square / blue circle, B green / yellow), greedy:

| Tree | Turn 1 (cold) | Turns 2-3 (warm) | Resumed at |
|---|---|---|---|
| Ministral-3-3B, `ov-runtime`, one stage | 11 s | 1.0 s, 0.9 s | 824 of 838, 840 of 850 |
| Ministral-3-3B, `ov-runtime`, two machines | 13-16 s | 1.0 s, 0.9 s | same |
| Gemma 4 E4B, `gemma4`, one stage | 13-14 s | 1.1-1.4 s, 0.9-1.0 s | 296 of 317, 318 of 335 |
| Qwen3.6-35B-A3B, `qwen35`, two machines | 11-13 s | 1.4-1.5 s, 1.3 s | 235 of 260, 261 of 282 |

All 24 answers are right. Conversation B's first turn misses (`n_entries` 1, A's entry
present) and never resumes from A's KV. On two machines rank 1 captures each turn under
the epoch rank 0 later restores. Gemma 4 E4B needs `enable_thinking: false` for this
test: a reply cut off inside the thought channel re-renders differently, so the next
turn misses (a cold reprefill, not a wrong answer).

The single-box `qwen35` run (`tree-oss4-vision`, one process) also answers 6/6, with
follow-ups at 3-6 s against 20-56 s cold, but from the local prefix cache: `kv_coord`
misses every follow-up there, text-only turns included, because that tree's chat template
re-renders an earlier reply differently from the tokens generated, so no whole-sequence
key is a prefix of the next prompt. The prefix cache snapshots at the last
`<|im_start|>` and is unaffected.

Still re-prefilled: static and packed shards (the cache is stateful-only). A plane peer
on another chain looks entries up by real token ids, so image entries are only resumed
on the chain that captured them.

## Final regression pass (2026-10-16)

Two read-only reviews of everything after the earlier branch review (engines; API, CLI
and tools) found nothing high or medium; the five low findings they verified are fixed:
Mistral 3's `spatial_merge` no longer resets to 1 when `processor_config.json` lacks it,
`cut_vlm_lm.py --family mistral3` requires the projector IR and `processor_config.json`,
GenAI's text-only hint names `--family mistral3`, the image encode cache keys on the
tenant as well as the pixel hash, and a cached prefix that ends inside an image is skipped
without being dropped (`take_warm_where`). The workspace passes 1195 tests (1301 with
`kv_coord`); clippy reports the same 228 warnings as `main`; formatting is clean.

One binary built from the result (and its `kv_coord` twin) then passed the 11-check SDK
test on every serving path, on CPU:

| Engine, placement | Model | Prompt tokens (1 / 2 images) |
|---|---|---|
| `ov-genai`, one box | Qwen3.6-35B-A3B | 227 |
| `ov-genai`, one box | Gemma 3 4B | 286 |
| `ov-runtime` stateful, one box | Ministral-3-3B | 815 |
| `ov-runtime` stateful, one box | Pixtral 12B (re-cut) | 832 |
| `ov-runtime` static, 4 stages on two machines | Gemma 4 26B-A4B (re-converted, shared prefill IR) | 288 / 550 |
| `gemma4`, one box | Gemma 4 E4B | 284 |
| `qwen35`, two machines | Qwen3.6-35B-A3B | 227 |

With `kv_coord`, a text-only chat still warm-resumes (535 of 545 tokens on turn 2), and
the image chat check (`sdk_vlm_kv.py`) passes 6/6 with its follow-ups resumed.

## GPU and NPU (2026-10-16)

The same 11-check SDK test, with the same binary as the final regression pass, on
pawan-02's Arc 140V iGPU (`--device GPU`) and Lunar Lake NPU (`--device NPU`); the vision
encoder and text embeddings stay on CPU. Answers and prompt token counts equal the CPU
runs.

| Device | Engine | Model | Result | Prompt tokens |
|---|---|---|---|---|
| GPU | `ov-genai` | Gemma 3 4B | 11/11 | 286 |
| GPU | `ov-genai` | Qwen3-VL-4B | 11/11 | 223 |
| GPU | `ov-runtime` stateful | Ministral-3-3B | 11/11 | 815 |
| NPU | `ov-runtime` static (context 2048, prefill 256) | Ministral-3-3B | 11/11 | 815 |
| NPU | `ov-runtime` static, 2 stages on two machines | Pixtral 12B | 11/11 (earlier) | 832 |

The Ministral NPU run, compile included, took about ten minutes (`--ov-cache-dir` makes
a restart load the blobs). Not run on GPU: the staged multi-machine trees and the 26B /
31B models, whose two stages per node already fill the shared memory on CPU.

## Phi-3.5-vision and InternVL2 on `ov-runtime` (2026-10-16)

Both were single-box only (`ov-genai`, 785 and 282 prompt tokens on the SDK check). Both
have a Phi-3 text tower that takes `inputs_embeds` and attends causally, so they cut and
serve like Pixtral: `cut_vlm_lm.py --family phi3_v | internvl_chat`, `token_type_ids`
off, the wire `[lead] [hidden]`.

**IR facts** (read from the published `.xml` files; no weights fetched):

| Export | Vision IRs | Host work on rank 0 |
|---|---|---|
| `OpenVINO/Phi-3.5-vision-instruct-int4-ov` | `openvino_vision_embeddings_model.xml`: `pixel_values [N,3,336,336]` to `[N,576,1024]` (CLIP layer -2, class token dropped); `openvino_vision_projection_model.xml`: `input [1,T,4096]` to `[1,T,3072]` | HD transform, global view, 2x2 merge, crop tiling, `sub_GN` / `glb_GN` rows (in `config.json`) |
| `OpenVINO/InternVL2-4B-int4-ov` | `openvino_vision_embeddings_model.xml`: `pixel_values [N,3,448,448]` to `[N,256,3072]` (InternViT, pixel shuffle and `mlp1` inside) | tiling and thumbnail only |

Both language models take `inputs_embeds`, `attention_mask`, `position_ids`, `beam_idx`
(64 KV states, 32 layers, `layers.N.input_layernorm` boundaries). Their rotary
`inv_freq` is a constant `[1,48,1]` traced once; there is no switch to the su/longrope
long factors past `original_max_position_embeddings` (4096) in the graph, so a cut
serves them exactly as the export and GenAI do, HF-exact up to 4096 positions. The load
log warns.

**Phi-3.5-vision** (`phi3v_vision`; source: the export's `processing_phi3_v.py` and
`modeling_phi3_v.py`):

- `HD_transform`: transpose a portrait image, scale the long side to the largest
  `k * 336` with `k * ceil(k / ratio) <= num_crops` (4), resize with Pillow bilinear
  (torchvision's default on a PIL image; the shared resampler gained the filter), pad
  the short side to whole 336-px crops with white, centred. The global view is
  `F.interpolate(..., (336, 336), mode="bicubic")` on the normalised tensor (torch's CPU
  kernel, ported in f32). The vision IR gets the global view, then the crops row-major;
  the processor's zero padding to `num_crops + 1` views is skipped (unused rows).
- `hd_feature_transform`: each 24x24 grid merged 2x2 into 12x12 rows of 4096, crops
  tiled into one grid, every grid row closed by `sub_GN`, then `glb_GN`, then the global
  view's 12 rows each closed by `sub_GN`; then the projection IR. An image is
  `(crops + 1) * 144 + 1 + (crop rows + 1) * 12` tokens (757 for 448x448).
- Prompt: `Phi3VProcessor` cuts the prompt at each `<|image_N|>` and tokenizes every
  chunk on its own, so each chunk after an image starts with its own `<s>` (and the
  tokenizer's `▁`); the soft tokens are negative ids embedded as id 0. Rank 0 does the
  same through `Frontend::encode_prompt` (text-only prompts too, which gives them the
  BOS the template lacks) and marks soft tokens with `<|image|>` (32044). The API writes
  `<|image_N|>\n` (numbered from 1: `MediaConfig` placeholders take `{n}`), the model
  card's and GenAI's form, and reserves `<|image_` and `<|image|>`.
- The export's `generation_config.json` stops on `<|endoftext|>` only; the cut adds
  `<|end|>`, the template's turn end (InternVL2's Phi-3 tower lists both).

**InternVL2** (`internvl_vision`; source: the OpenGVLab/InternVL2-4B model card and
`modeling_internvl_chat.py`): `dynamic_preprocess` picks the tile grid of at most
`max_dynamic_patch` (12) tiles whose ratio is closest to the image's, visiting grids in
Python's `sorted(set(...), key=area)` order (embedded for 1..12: ties in closeness
depend on it), resizes with Pillow bicubic, cuts 448-px tiles row-major and adds a
448x448 thumbnail when there is more than one; ImageNet normalisation. The prompt
carries `<img><IMG_CONTEXT></img>\n` per image (three special tokens, so tokenizing it
and widening `<IMG_CONTEXT>` to 256 per tile gives the ids of the string `chat`
builds). The reference prompt is the export's chat template (which writes `<s>`), as
Cascadia renders it, not `chat`'s Chinese system prompt; optimum's `preprocess_inputs`
tokenizes that template again with a BOS, which doubles it.

**Parity** (`tests/phi3v_vision_parity.rs`, `tests/internvl_vision_parity.rs`, fixtures
from `tools/vlm_fixtures/gen_{phi3v,internvl}_fixtures.py` on the eight Pixtral fixture
images, transformers 5.14.1, torch 2.14, Pillow 12.3): sizes, grids and token counts
equal from size alone; Phi crops and InternVL tiles bit-identical; the Phi global view
within one f32 ulp (torch's kernel fuses multiply-adds); the Phi feature assembly
bit-identical to `hd_feature_transform` on synthetic features with the real separators;
the prompt layouts reproduce the processors' ids. HF prompt tokens on the fixture
prompts:

| Model | One image (448x448) | Two images (640x360, 300x1000) |
|---|---|---|
| Phi-3.5-vision | 774 (757 image) | 1403 |
| InternVL2-4B | 269 (256 + 2 framing) | 1809 (3 + 4 tiles) |

The SDK check's 785 / 282 on `ov-genai` is the same 757 / 258 image tokens plus a
question 11 tokens longer in both families, so the staged counts are expected to equal
GenAI's there; not yet run on the cluster.

**Paths.** Causal, so the static (`static_variant.py`) and packed paths take both under
the Pixtral rules: the shared prefix stops at the first image token, the expanded length
is computed from image sizes before any encode (a too-long prompt is refused at submit
with no encoder work), the encode cache keys on (tenant, sha256), and `vision_kv` keys
image turns on the soft-token runs. `ov-dist-spec` still refuses non-Pixtral targets.
An InternVL2 image can take 13 tiles (3328 tokens) whatever its size, so packed slots
need a region that holds one.

**Not yet verified on the cluster:** the cut on the real language models (the toy tests
cover the config layouts), the vision IR rows against `--run-ir` fixtures, and the SDK
check across two machines.

## Qwen2.5-VL and Qwen3-VL on `ov-runtime` (2026-10-16)

Both served one box only through `ov-genai` (294 / 223 prompt tokens on the SDK check's
448x448 image). They now take images on staged `ov-runtime` trees, reusing the `qwen35`
machinery: the Qwen2-VL processor port, `get_rope_index`, pad expansion and the
`pad_cache_id` image keys.

**IR facts** (`OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov`, `OpenVINO/Qwen3-VL-4B-Instruct-int4-ov`,
read from the published `.xml` files) [verified]:

| IR | Qwen2.5-VL-7B | Qwen3-VL-4B |
|---|---|---|
| language model | `inputs_embeds [?,?,3584]`, `attention_mask [?,?]`, `position_ids [3,?,?]`, `beam_idx` | the same at 2560, plus `visual_pos_masks [?,?]` boolean and `deepstack_visual_embeds [?,?,2560]` |
| vision patch | `hidden_states [?,?]` -> `[N,1280]` | `hidden_states [?,?]` -> `[N,1024]` |
| vision positions | none | `input [4,?]` -> `[4,N,1024]` (Qwen3.5's bilinear table) |
| vision merger | `hidden_states [N,1280]`, `attention_mask [1,N,N]`, `window_attention_mask [1,N,N]`, `window_index [N/4]`, `rotary_pos_emb [N,40]` -> `[N/4,3584]` | `hidden_states [N,1024]`, `attention_mask`, `rotary_pos_emb [N,32]` -> `last_hidden_state [N/4,2560]` and `deepstack_feature_lists [3,N/4,2560]` |

**M-RoPE.** transformers 5.14.1 `Qwen2_5_VLModel.get_rope_index` and
`Qwen3VLModel.get_rope_index` give a still image `t = start`, `h/w = start + row/col` and
advance by `max(h, w) / merge`; Qwen2.5-VL's `tokens_per_second` spacing multiplies the
temporal index only for videos (`time_interval = 1` for images; videos: "Native video on
the staged engines" below). That is exactly
`qwen35_vision::rope::mrope_positions`, so both reuse it; GenAI 2026.4.1's
`create_position_ids` (`qwen2vl/classes.cpp`) builds the same. The language model takes
HF's three rows (not Qwen3.5's four) and decode positions are `index + rope_delta` on
all three (`get_generation_phase_position_ids`). The `mrope_section` split
(Qwen2.5-VL `[16,24,24]` in contiguous blocks, `apply_multimodal_rotary_pos_emb`;
Qwen3-VL `[24,20,20]` interleaved, `apply_interleaved_mrope`) is inside the language
model IR, so the host feeds both the same rows. Parity: the size-only expansion equals
HF's `input_ids` and the positions equal `get_rope_index` on one image, two images,
multi-turn and text only, for both families (`tests/qwen_vl_vision_parity.rs`).

**Wire.** A tree's `vision.mrope_rows` (3) and every stage's `position_rows` (3) switch
on a fourth frame `[lead] [hidden] [positions]`, I64 `[1, 3, T]`, on every step, decode
included (24 bytes a token). Rank 0 computes it from the turn's positions; other stages
bind `position_ids [3, 1, T]` from it and forward it. Versioning: a build without the
family refuses the tree (`vision.family`), a tree cut before the fields refuses to load
(`re-cut`), and a peer that does not expect the frame reads it as a lead and fails on the
shape. The static and packed paths pin 1-D positions, so these trees are refused there.

**DeepStack (Qwen3-VL).** `deepstack_visual_indexes [5, 11, 17]` are the *vision*
layers the features come from; the language model adds feature `k` to the image tokens'
hidden states after decoder layer `k`, `k < 3` (`Qwen3VLTextModel.forward`,
`_deepstack_process`), which the IR does with `cumsum(visual_pos_masks)` gathers from
`deepstack_visual_embeds[k]` feeding layers 1-3's input norms [verified in the IR]. So
only stage 0 needs the rows, and stage 0 runs on rank 0 beside the encoder: they never
cross the wire. `cut_vlm_lm.py` refuses a stage 0 shorter than three layers or a later
stage that keeps a DeepStack input, and turns `visual_pos_masks` into an i64 input
converted to boolean in the graph (the runtime feeds no boolean tensors). Text and
decode steps feed an all-zero mask and one zero row per layer, as GenAI does.

**Qwen2.5-VL vision host inputs.** `window_index` / `cu_window_seqlens` port
`vision_utils.get_vision_window_index` (windows of 112 / 2 / 14 = 4 merge units, empty
padded windows dropped as `unique_consecutive` does) and the window mask follows GenAI's
`get_window_attention_mask`; the rotary table and per-frame mask are Qwen3.5's. Exact
against HF on seven images (`window_index`, `cu_window_seqlens`), and the preprocessing is
bit-identical to the PIL processor for both families (CLIP mean/std and 14-px patches
for Qwen2.5-VL).

**Hardening carried over.** The expansion is sized from image dimensions before any
encode and checked against the encoder's; encodes are cached on (tenant, sha256);
`<|vision_start|>`, `<|image_pad|>`, `<|vision_end|>` and `<|video_pad|>` are reserved in
caller text (`MediaConfig::qwen35_native`); `kv_coord` keys image turns on `vision_kv`
and, since positions are looked up by absolute index, a warm suffix needs no change (the
DeepStack rows are trimmed with the image rows).

**Not yet verified on hardware**: cutting the published IRs (the tests cut toy models
with the export's interface, `tools/tests/test_vlm_qwen_cut.py`), the IR parity test
(`tests/qwen_vl_vision_ir.rs` with `gen_qwen_vl_fixtures.py --run-ir`) and the SDK check
on two machines.

## Engines that do not take images (2026-10-15)

Every engine either serves images or refuses them with an OpenAI 400; none answers
from the text alone.

| Engine | Images | How a request with images is answered |
|---|---|---|
| `ov-genai` | VLM exports with the vision IR | served; on a text-only export the API offers no image input (400), and `submit` refuses too |
| `qwen35` | trees with the vision IRs | served; otherwise 400 at the API and at `submit` |
| `ov-runtime` | Gemma 4 surgery trees (`vision.family = "gemma4"`, stateful, static and packed paths), Pixtral trees (`"pixtral"`, stateful, static and packed paths), Mistral 3 trees (`"mistral3"`), LLaVA-NeXT trees (`"llava_next"`), Phi-3.5-vision and InternVL2 trees (`"phi3_v"`, `"internvl_chat"`) and Qwen2.5-VL / Qwen3-VL trees (`"qwen2_5_vl"` / `"qwen3_vl"`, stateful only) | served; other trees 400; an unknown `vision.family` fails to load |
| `gemma4` (`gemma4_cached` trees) | E2B/E4B trees from `export_gemma4.py --vision` | served; other trees 400 at the API and at `submit` |
| `ov-dist-spec` | Pixtral / Mistral 3 target trees cut with `--all-logits` (`vision.family` `"pixtral"` / `"mistral3"`); the draft stays text-only | served; v5 (text) targets 400 at the API and at `submit`; a Gemma 4 target fails to load |
| `sparse-moe` | no | 400 |
| `mock` | accepts and ignores | test stand-in for the API's image tests |

A body over the size limit stays a 413, now with a message pointing at
`input_modalities`. `tools/export_shards.py` rejects Llama 3.2 Vision (`mllama`, whose
text tower has cross-attention) and notes at export time when it drops a checkpoint's
vision tower (Mistral 3, Qwen2-VL, LLaVA).

## Pixtral on `ov-dist-spec` (2026-10-16)

Design: only the target sees images. Rank 0 expands and encodes them with the Pixtral
front end and feeds stage 0 `inputs_embeds` (the text-embedding rows of the expanded
prompt with the image rows spliced in); every later span (the drafted tokens and the
correction) is embedded through the same table with no image rows, so the mask-based
rewind and the `valid_mask` only ever cover text rows. The draft (Ministral-3-3B,
`tools/vlm_surgery/make_text_draft.py`) is fed the expanded ids, image tokens as plain
tokens; its ids match Pixtral's apart from seven renamed reserved specials, and the
target's tokenizer tokenizes once. Pixtral's text tower uses 1-D RoPE, so dist-spec's
1-D positions hold.

What changed to make a `cut_vlm_lm.py` tree a dist-spec target: stage 0 takes
`inputs_embeds` (the driver now resolves it when the tree has a `vision` block), and the
last stage must emit logits for every fed position. The cut sliced the head to the last
row; `--all-logits` keeps them all and records `"logits": "all"` in the last stage's
config, which rank 0 and the last worker check at load (an older cut without the field
is refused). The API's image tag for these trees is `pixtral_native`.

Not covered: KV warm resume of image turns under `kv_coord` (they prefill cold and are
not captured), Gemma 4 targets (their `token_type_ids` frame does not ride the wire),
static or packed dist-spec.

Fixed after the first cluster run: the draft accepted 0 of 40 drafted tokens, because
it read the 784 `[IMG]` placeholders as text and degenerated ("UserUserUser...";
`tools/vlm_surgery/check_text_draft.py` shows the draft itself matches Ministral's
embeddings + LM pipeline token for token, rewinds included). The draft is now prefilled
with the image ids removed (its KV is its own, so its positions need not match the
target's). The prefill's all-rows logits overflowed the wire (432 MB for two images,
`MAX_TENSOR_BYTES` 256 MB) and scaled with the prompt (about 17 GB for eight large images):
the prefill now goes in 512-row chunks (frame kind 10) answered with the last row's logits
only, and verify spans keep every row. `usage.prompt_tokens` reports the expanded count.

Verified on pawan-02 (CPU, both ranks on one box; target `cut_vlm_lm.py --num-stages 2
--family pixtral --all-logits`, draft `make_text_draft.py` from `ministral3-3b-int4-ov`,
`--spec-k 4`): the 11-check SDK test passes, answers equal `ov-runtime`'s ("There is a red
square and a blue circle.", 832 prompt tokens, two images 1648). Draft acceptance 25-35%
on image and text turns, about two tokens per target step (10 tokens in 5 steps for one
image, 14 in 6 for two).

## Gemma 3 on `ov-runtime` (2026-10-16)

Gemma 3 was served only on one box by `ov-genai`. The staged path now takes it, from
the published `OpenVINO/gemma-3-12b-it-int4-ov` (Gemma license, not gated).

- **The export.** `openvino_language_model.xml` takes `attention_mask`, `position_ids`,
  `token_type_ids`, `inputs_embeds [?,?,3840]` and `beam_idx` (stateful KV) and returns
  logits `[?,?,262208]`; `openvino_text_embeddings_model.xml` maps `input` ids to
  `[?,?,3840]` (the sqrt(hidden) scale is inside it); `openvino_vision_embeddings_model.xml`
  maps `pixel_values [?,3,?,?]` to `[?,256,3840]`: SigLIP, the 4x4 average pool, the
  soft-embedding norm and the projection are all in the IR, so rank 0 needs no projector
  step. In the language model graph `token_type_ids` reaches all 48 attention layers.
- **Masks.** transformers 5.14.1 `modeling_gemma3.py`, `create_masks_for_vision_model`:
  full layers use OR(causal, same image block), sliding layers AND(window, OR(causal,
  same block)); blocks are contiguous runs of `token_type_ids == 1`
  (`get_block_sequence_ids_for_mask`). So, unlike Gemma 4's sliding-only case, both
  masks open each image, which is what the packed host masks already do. The Gemma 4
  machinery is reused unchanged: `token_type_ids` on every stage, the third wire frame,
  `image_safe_take` on the static path (an image is 256 rows, so `--prefill-seq 256` or
  more) and the 1025 window cap in `static_variant.py`.
- **Prompt.** `Gemma3Processor` replaces each `<start_of_image>` of the rendered
  template on the text with `"\n\n<start_of_image>" + 256 x "<image_soft_token>" +
  "<end_of_image>\n\n"` (`full_image_sequence`) and then tokenizes. The blank lines
  merge with their neighbours (`"user\n" + "\n\n"` is the single token 109; two images
  back to back give one `"\n\n\n\n"`, 110), so the expansion cannot be done on ids as
  for Pixtral and Gemma 4: rank 0 expands the prompt text of a turn with images before
  tokenizing it (`Frontend::prompt_text`), and `gemma3_vision::check_expanded` then
  checks one 256-token run per image. The soft token id 262144 is inside the vocabulary
  (262208), so it embeds as itself, as in HF; its rows are overwritten either way.
- **Preprocessing** (`gemma3_vision`): resize to 896x896 ignoring the aspect ratio with
  `resample: 2` (bilinear; the shared Pillow-exact resampler gained the bilinear
  kernel), rescale 1/255, normalise with mean/std 0.5. Pan-and-scan defaults off in the
  processor (`do_pan_and_scan: null` in the config) and is refused if switched on.
- **API.** `MediaConfig::gemma3_native`: the template's own `<start_of_image>`; reserved
  `<start_of_image>`, `<image_soft_token>`, `<end_of_image>`; text parts trimmed, as the
  template trims each one. The CLI enables it for `ov-runtime` on a `gemma3` tree.
  `ov-dist-spec` refuses the tree (token types), as for Gemma 4.
- **Cut.** `cut_vlm_lm.py --family gemma3` checks `gemma3` / `siglip_vision_model`,
  refuses a language model without `token_type_ids`, and refuses a stage that lost them;
  `config.json` keeps `sliding_window` and `layer_types`.
- **Parity** (`tests/fixtures/gemma3_vision`, made with transformers 5.14.1 and
  Pillow 12.3 from the export's configs and tokenizer, no weights): pixel values
  bit-identical to HF's PIL processor on the eight fixture images (the torchvision
  processor differs by one 8-bit level); the text expansion equals the processor's
  expanded text; the processor's `token_type_ids` equal what rank 0 sends; and,
  with `CASCADIA_GEMMA3_TOKENIZER` set to the export's `tokenizer.json`, the Rust
  tokenizer gives HF's ids on every prompt. Prompt tokens: one image 270, two images 535.
- **Verified** with Gemma 3 12B (`OpenVINO/gemma-3-12b-it-int4-ov`, public, revision
  60f9d951), cut into 2 stages (layers 0-23 / 24-47) on each node and served across
  pawan-02 (rank 0) and pawan-04 (rank 1), CPU: the 11-check SDK test passes, "There is a
  red square and a blue circle.", 286 / 550 prompt tokens, which equals `ov-genai` on the
  same export on one box (also 11/11). Not run: packed slots and the static path.

## LLaVA-NeXT on `ov-runtime` (2026-10-16)

`LlavaNextForConditionalGeneration` (llava-v1.6; here `llava-hf/llava-v1.6-mistral-7b-hf`,
Mistral-7B text tower, 1-D RoPE, causal) served across machines from the public
`OpenVINO/llava-v1.6-mistral-7b-hf-int4-ov` export (Apache-2.0), which `ov-genai`
already serves on one box.

- **The export.** Read from its `.xml` and JSON files (no weights): four IRs, as for
  Pixtral. `openvino_vision_embeddings_model.xml` takes `pixel_values [?, 3, ?, ?]` and
  returns `last_hidden_state [?, 576, 4096]`: the CLIP tower's penultimate layer with
  the CLS row dropped (`vision_feature_select_strategy: "default"`) and the projector
  (`multi_modal_projector.linear_1/2`) inside. There is no `image_newline` IR:
  optimum-intel writes the 4096 floats into `config.json`. The language model is the
  usual stateful `inputs_embeds` IR (`__module.model.layers.N.input_layernorm`), and the
  image token is `<image>` (32000).
- **Rank 0** (`llava_next_vision`), from transformers 5.14.1
  (`image_processing_pil_llava_next.py`, `processing_llava_next.py`,
  `modeling_llava_next.py`):
  - `select_best_resolution` picks the pinpoint (most kept pixels, then least waste);
    the image resizes into it with its aspect ratio (`get_patch_output_size`, `ceil`),
    is centred on black (`divmod` padding) and cut into 336-px tiles row-major, after a
    base view squashed to 336x336. Every resize is Pillow's bicubic (a same-size resize
    is a copy), then rescale and CLIP normalisation.
  - All views go through the vision IR in one inference (at most five on llava-v1.6's
    pinpoints).
  - Packing is `pack_image_features`: the base view's 576 rows, then the tiles stitched
    into one `(rows*24) x (cols*24)` grid, the padding rows or columns dropped
    (`unpad_image`, Python's `int(round(x, 7))` reproduced), `image_newline` after each
    kept row. An extreme strip keeps no tile rows at all (1500x4: 576 tokens); a pole
    keeps no columns but every row's newline (3x1200: 648).
  - The token count is `LlavaNextProcessor._get_number_of_features` minus the CLS,
    from the size alone: used before encoding (packed submit, `expanded_len`), and
    checked against the encoder's at submit on packed slots.
  - Every soft token, newline rows included, is `<image>`, so the expansion is a run of
    `<image>` per image; `vision_kv` keys and spans and the `(tenant, sha256)` encode
    cache apply unchanged.
- **Prompt.** HF's processor template writes `<image>\n` for each image of a message,
  then its text parts back to back (`[INST] <image>\nWhat is in this picture? [/INST]`),
  whatever the parts' order. `MediaConfig::llava_next_native` does the same
  (`images_first`, placeholder `<image>\n`, `<image>` reserved), and the tree's
  tokenizer template adds `<s>`. The fixture generator renders this and checks the ids
  equal the processor's: 2945 tokens for one 448x448 image (`ov-genai`: 2956), 4199
  for two (640x360, 300x1000).
- **Parity** (`tests/llava_next_vision_parity.rs`, fixtures from
  `tools/vlm_fixtures/gen_llava_next_fixtures.py`): pixel values bit-identical to the
  PIL processor on the eight Pixtral fixture images (the torchvision one differs by up
  to 0.24 on 3x1200); pinpoint, grid and token count equal HF's on those and on a
  43x43 sweep of sizes; packing reproduces HF's row order on synthetic features; prompt
  expansion reproduces the processor's ids. The vision IR's own output was not compared
  here (no weights fetched); `--run-ir` on the cluster adds it.
- **Cut.** `cut_vlm_lm.py --family llava_next` requires `preprocessor_config.json` and a
  hidden-wide `image_newline`, and refuses the `full` feature strategy.
- **Static and packed.** The family is causal, so Pixtral's rules apply: no image
  blocks, a prefill chunk may split an image, a packed prompt is sized from its images
  and refused with a 413 before encoding when it outgrows a slot's region. A static
  tree needs a context of 4096 or more for one image. Neither path has been run on
  hardware with this model.
- **Not covered:** `ov-dist-spec` (refused at load: Pixtral / Mistral 3 targets only).
  Not yet run on the cluster.

## Every GenAI-only family across machines (2026-10-17)

The families that only `ov-genai` served (one box) now also run staged on `ov-runtime`,
each cut into 2 stages by `cut_vlm_lm.py --family <f>` from its public OpenVINO int4 export
(no Hugging Face token needed), stage 0 on pawan-02 and stage 1 on pawan-04 (each node
cut its own stage from its own download), CPU, with one binary from this branch. The
11-check SDK test passes on every one, and the prompt token counts equal `ov-genai`'s
on the same export except where `ov-genai` is the one that is off:

| Family (`--family`) | Export | Result | Prompt tokens (1 image) | `ov-genai` |
|---|---|---|---|---|
| Gemma 3 (`gemma3`) | `OpenVINO/gemma-3-12b-it-int4-ov` | 11/11 | 286 (two images 550) | 286 |
| LLaVA-NeXT (`llava_next`) | `OpenVINO/llava-v1.6-mistral-7b-hf-int4-ov` | 11/11 | 2956 | 2956 |
| Phi-3.5-vision (`phi3_v`) | `OpenVINO/Phi-3.5-vision-instruct-int4-ov` | 11/11 | 785 | 785 |
| InternVL2 (`internvl_chat`) | `OpenVINO/InternVL2-4B-int4-ov` | 11/11 | 280 | 282 |
| Qwen2.5-VL (`qwen2_5_vl`) | `OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov` | 11/11 | 294 | 294 |
| Qwen3-VL (`qwen3_vl`) | `OpenVINO/Qwen3-VL-4B-Instruct-int4-ov` | 11/11 | 223 | 223 |

InternVL2: 280 is HF's count for this request (the export's chat template, then the
model card's 256 `<IMG_CONTEXT>` per tile: `<s>` + `<|user|>` 2, question 17, image block
259, `<|end|>` + `<|assistant|>` 2); GenAI's 282 adds a second BOS and a newline.
Qwen2.5-VL needed one fix found here: its `rope_scaling` carries both `type` and
`rope_type`, which the config reader's serde alias rejected as a duplicate field
(`080698fb`). Qwen VL trees carry M-RoPE positions as their own wire frame (checked at
load) and refuse the static and packed paths; Qwen3-VL's DeepStack features enter after
LM layers 0-2, inside stage 0 on rank 0, so they never cross the wire.

## Video: out of scope, kept cheap

Video ships in no phase of this RFC. The image work must still leave nothing that
would have to be redone to add it. Each item below is a requirement on the image work,
checked in review and, where marked, by a test:

1. **Task shape.** `GenerationTask.media` is a list of `MediaInput` with a `kind` and
   a frame count, not `images: Vec<ImageInput>` (section A). An image is one frame.
2. **Vision runner input.** The rank-0 runner takes a frame stack `[T,H,W,3]` plus
   optional timestamps. An image is `T=1`, padded to the temporal patch size (2) the
   same way the HF processor does. No image-only entry point.
3. **Positions.** The position code takes a full `(t, h, w)` grid per media item and
   the optional timestamps, following `get_rope_index`'s video branch. **Test:** the
   parity suite against HF `get_rope_index` includes a synthetic video grid
   (`t>1`, with timestamps), not only images. This is the piece that is costly to
   retrofit, so it is tested now.
4. **Caches.** Prefix-cache and encoder-cache keys use the per-item `sha256` of the
   decoded frames, regardless of kind.
5. **Shim and FFI.** The shim's media entry point takes a list of tensors plus a kind
   per tensor, so GenAI's `add_request(..., videos, ...)` overload is a new branch,
   not a new ABI.
6. **API.** The content-part parser has one dispatch point for media parts. `video_url`
   gets an OpenAI-shaped 400 naming the type (as today) until video lands.
7. **Exporter.** `video_preprocessor_config.json` is copied and listed in the manifest
   alongside the image config (C7).

Adding video later is then: a decoder and frame sampler in the API with its own
limits (duration, frame count, container size), the `video_url` part behind a flag,
the video branch in the shim, and `MediaKind::Video` reaching the runner. The
positions, caches, task shape and wire stay as they are.

### Native video on the staged engines (V2)

`ov-runtime` Qwen2.5-VL / Qwen3-VL trees and the `qwen35` engine's Qwen3.5 / 3.6 trees
take a `MediaKind::Video` item whole (`qwen35_vision/video.rs`, `qwen_vl_vision.rs`,
`qwen36.rs`); the API routes it there per family (`STAGED_NATIVE_VIDEO` in
`cascadia-api/src/media.rs`) behind Qwen's own `<|vision_start|><|video_pad|><|vision_end|>`,
the tag HF's chat template renders for a video. Every other family keeps frames, and the
engines' `refuse_video` stays for them.

- **The IRs need no re-export.** Each family's patch-embedding IR takes
  `hidden_states [?, 3 * 2 * P * P]`: one row per patch over the temporal patch of two
  frames (a Conv3d), which an image fills by duplicating its frame. A video's consecutive
  frames fill it; an odd count pads with the last frame, as HF does and as GenAI's
  `encode_frames_with_config` does by encoding the trailing frame as an image. The merger
  attends within one temporal group (`cu_seqlens` has one sequence per group; Qwen2.5-VL's
  windows never cross one) and every other op is per patch, so rank 0 encodes the groups
  one by one, which keeps the merger's `[1, N, N]` masks one group's size. On the
  Qwen3-VL-4B and Qwen2.5-VL-7B exports, HF's video preprocessing driving the IRs (CPU,
  OpenVINO 2026.1) gives the same rows, and on Qwen3-VL the same DeepStack rows (max
  |diff| 0.0), for a 5-frame video encoded whole or per group.
- **Size.** Qwen3-VL and Qwen3.5 (`Qwen3VLVideoProcessor`) resize every frame with a
  `smart_resize` whose budget counts all frames (4096..25165824 pixels from the export's
  video config, found where `AutoVideoProcessor` looks); Qwen2.5-VL
  (`Qwen2VLVideoProcessor`) resizes each frame with its image budget. HF's video
  processors resize with torchvision only; rank 0's Pillow-exact resize is within two
  8-bit levels of it.
- **Markup.** Qwen3-VL and Qwen3.5 replace the template's `<|video_pad|>` (its own
  `<|vision_start|>` / `<|vision_end|>` stay around the result) with, per temporal group,
  `<{t:.1f} seconds><|vision_start|>` + `h * w / 4` pads + `<|vision_end|>`, `t` the mean of
  the group's first and last frame time; rank 0 writes the timestamps into the prompt text
  before tokenizing and widens each pad on ids. Qwen2.5-VL writes one run of
  `t * h * w / 4` pads, no timestamps.
- **Positions.** Qwen3-VL / Qwen3.5: one `t = 1` grid per group (`get_rope_index` splits
  `video_grid_thw` the same way). Qwen2.5-VL: one grid whose group `k` sits at
  `start + floor(k * second_per_grid_t * tokens_per_second)` (f32, as HF holds
  `second_per_grid_ts`), `second_per_grid_t = 2 / sampled_fps` from the frame times, and
  the next token starts one past the largest position so far. That is Qwen's original
  `get_rope_index` (transformers 4.x), which the model was trained with; transformers
  5.20.1 computes the same temporal steps (checked; 5.2 reportedly too), and GenAI
  2026.4.1 and vLLM start the next text the same way. transformers 5.14 / 5.15 depart from it twice, and we do not follow them:
  they truncate `second_per_grid_t` to whole seconds before multiplying (a transient
  regression), and they start the next text `max(h, w) / 2` after the run's start, inside
  the video's time range. The second also breaks warm resume: a later turn's prompt,
  positioned cold, would put the text after the video below the `rope_delta`-based
  positions the earlier turn decoded with (16 frames at 1 fps of 448x252: the video's
  last group at `start + 28`, the next text at `start + 16`). With the API's default
  1 fps sampling, a fractional `second_per_grid_t` is routine (a 5 s clip gives 6 frames,
  1.67 s per group), so the two would differ on most requests. The fixtures take
  Qwen2.5-VL's positions from a port of the original code in the generator
  (`qwen25_rope_index`), checked equal to 5.14.1 where the two agree (no video, one
  group) and to 5.20.1's temporal steps inside every video run.
- **Feeding.** Once positions are fixed, rank 0 feeds every `<|video_pad|>` as
  `<|image_pad|>`. The model never reads a pad's own embedding (its row is replaced, as
  HF's `masked_scatter` does), and Qwen3-VL's `visual_pos_masks` cover image and video
  tokens alike in HF, so the splice, the DeepStack mask and window, the KV / prefix and
  encode cache keys (the video's pixel hash) and every mode's paths take video rows with
  no second id. A video is therefore served in stateful, static and packed modes and
  across machines (the `[3, T]` positions frame) wherever images are.
- **Parity** (`tests/qwen_vl_video_parity.rs`, fixtures from
  `tools/vlm_fixtures/gen_qwen_vl_video_fixtures.py`, transformers 5.14.1 for the
  processors, Qwen2.5-VL positions as above): grids and
  pixels for odd / even / one-frame videos and an 8-frame 448x252 clip on all three
  families; the text rank 0 tokenizes, the expanded ids, `get_rope_index` positions and
  `rope_delta` for one video, two videos, videos with images across turns, and
  Qwen2.5-VL at 0.5, 1, 1.5 and 2 frames a second; and (`rope.rs`) a later turn's
  positions, computed cold, equal the ones the earlier turn fed over the shared prefix,
  for both video layouts.

## Phase 0 results (pawan-04, 2026-10-15)

Setup: Core Ultra 7 258V, 32 GB, CPU device; Python `openvino-genai==2026.4.1.0`
(our pin) in a throwaway venv; `C:\cascadia\models\Qwen3.6-35B-A3B-int4-ov` (optimum
export, `qwen3_5_moe`, language model 17.4 GB int4, vision IRs about 0.4 GB). Test image:
448x448 synthetic, a red square left and a blue circle right.

| Run | Input tokens | TTFT | TPOT | Answer |
|---|---|---|---|---|
| plain prompt + image (GenAI templates, thinking on) | 227 | 43.3 s (first call) | 163 ms | starts reasoning about "two distinct geometric shapes" |
| plain prompt, no image | 29 | 2.7 s | 139 ms | text-only reasoning |
| pre-templated + native tag, `apply_chat_template=false` | 229 | 21.5 s | 140 ms | "A red square and a blue circle." |

- The image path works end to end on our pinned GenAI with the 35B export.
- A 448x448 image costs 198 tokens: (448/16)^2 / 4 = 196 image tokens plus the two
  vision tags. This matches the token math in "How a Qwen3.5 image request works".
- The pre-templated run is exactly what our API plus engine will send, so Phase 1b
  needs no prompt-format work beyond rendering the tag through the template.
- On CPU one small image adds about 19 s to TTFT after warm-up (21.5 s against
  2.7 s for text). That covers the vision encoder and about 200 extra prefill tokens;
  this run did not split the two. Pixel caps and iGPU placement for the
  vision IRs matter for latency, not only memory.
- Model load took 88 s.

## Sources

- In tree: files cited above, on branch `pawan/vlm-research` = `9ecc2083`.
- transformers 5.14.1, `models/qwen3_5_moe/modeling_qwen3_5_moe.py` (`get_rope_index`,
  `get_placeholder_mask`, `Qwen3_5MoeTextModel.forward`).
- OpenVINO GenAI: `src/c/include/openvino/genai/c/vlm_pipeline.h` (master); VLMPipeline
  and ContinuousBatchingPipeline API reference, docs.openvino.ai 2025/2026; OVMS
  continuous-batching VLM demo.
- Qwen3-VL technical report (arXiv 2511.21631), for DeepStack in Qwen3-VL only.
- vLLM encoder cache and EPD disaggregation (vllm.ai blog, 2025-12-15); llama.cpp
  `tools/mtmd` README.
- OpenAI images-vision guide (content-part formats).
