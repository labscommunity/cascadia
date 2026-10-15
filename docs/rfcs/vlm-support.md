# RFC draft: image input (VLM) support across Cascadia

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
| Kimi K2.6, GLM-5.x, DeepSeek-V4, MiniMax-M2, R1-distill, Phi-3/4 | Not stated or no | Out of scope |

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
  are refused for vision trees. Turns with images skip KV warm-resume and capture,
  because every same-size image has the same ids.
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
differ from HF's by 1 and 4 (285 vs 284, 550 vs 546), the chat template's whitespace
between text parts rather than the image path; noted as a follow-up. Encode is about
3 s per image. The stage-1 IR exported on each node is byte-identical in weights.

## Engines that do not take images (2026-10-15)

Every engine either serves images or refuses them with an OpenAI 400; none answers
from the text alone.

| Engine | Images | How a request with images is answered |
|---|---|---|
| `ov-genai` | VLM exports with the vision IR | served; on a text-only export the API offers no image input (400), and `submit` refuses too |
| `qwen35` | trees with the vision IRs | served; otherwise 400 at the API and at `submit` |
| `ov-runtime` | Gemma 4 vision trees (`vision.family = "gemma4"`) | served; other trees 400; an unknown `vision.family` fails to load |
| `gemma4` (`gemma4_cached` trees) | E2B/E4B trees from `export_gemma4.py --vision` | served; other trees 400 at the API and at `submit` |
| `ov-dist-spec` | no | 400; draft and target would both need the expanded image prompt |
| `sparse-moe` | no | 400 |
| `mock` | accepts and ignores | test stand-in for the API's image tests |

A body over the size limit stays a 413, now with a message pointing at
`input_modalities`. `tools/export_shards.py` rejects Llama 3.2 Vision (`mllama`, whose
text tower has cross-attention) and notes at export time when it drops a checkpoint's
vision tower (Mistral 3, Qwen2-VL, LLaVA).

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
