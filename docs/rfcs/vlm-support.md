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
- **Task.** Add `images: Vec<ImageInput>` to `GenerationTask`, where
  `ImageInput { rgb: Arc<[u8]>, width, height, sha256 }`. Decode once, in the API.
  Engines get pixels, never URLs. `GenerationTask` is in-process only, so this costs no
  wire versioning.
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
    - Still to check: whether the shim's `generate` call re-applies the chat template to
      an already-rendered prompt.
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
     - compute bilinear-interpolation `input` indices and `weights` for
       `vision_embeddings_pos` from `grid_thw`, then add the weighted 4-corner sum to
       the patch embeddings (`:455-513`);
     - compute `rotary_pos_emb` for the merger (`get_rotary_pos_emb`);
     - pass either `cu_seqlens` or a block-diagonal `attention_mask`, depending on how
       the IR was exported (`:609-621`);
     - read `last_hidden_state` back out.

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
   GenAI's Qwen3.5 embedder [verified, see above]. Still check it once against the #174
   IR's `position_ids` input shape.
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
   1-3 are M-RoPE (GenAI `qwen3_5/classes.cpp:41-108`). One confirmation against the
   #174 IR's `model.inputs` remains.
2. ~~Merger inputs.~~ **Answered:**
   - merger: `hidden_states`, `rotary_pos_emb`, and `cu_seqlens` or `attention_mask`;
   - pos IR: `input` indices and `weights`.
   All of these are host-computed. Which of `cu_seqlens` and `attention_mask` the #174
   IR takes still needs reading from `model.inputs`.
3. ~~Placeholder.~~ **Answered:** the native tag
   `<|vision_start|><|image_pad|><|vision_end|>`, one per image, with the count checked.
   **Still open:** whether our shim's `generate` re-applies the chat template.
4. ~~Continuous batching with images.~~ **Answered: yes** in 2026.4.1
   (`continuous_batching_pipeline.hpp:284-285`).

Still open:

5. Port GenAI's embedder logic to Rust, or link GenAI internals from the shim? This is
   now a real trade-off; see C1.
6. Is video in scope? Qwen3.5 supports it (timestamped frames, `get_rope_index` splits
   `video_grid_thw`, and CB has a videos overload). This RFC treats it as a later
   extension.
7. Does the image path work end to end on our pinned GenAI (2026.4.1) with a
   Qwen3.5-35B VLM IR? This needs model files: run once through Python `VLMPipeline`
   before Phase 1b.

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
