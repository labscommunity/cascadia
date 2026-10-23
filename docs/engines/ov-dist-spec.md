# `ov-dist-spec` — distributed speculative decoding

Combines pipeline parallelism with speculative decoding: the **driver** holds a small fast draft model plus stage 0 of the target; **workers** hold downstream stages and respond to FORWARD frames over TCP.

## When to use

- The target model fits across `N` machines but not on one.
- You have a draft model that shares the target's tokenizer (e.g. Llama-3.2-1B for a Llama-3.1 target).
- You want the spec-decode token-rate boost without losing the ability to run a model bigger than one node.
- Your generations are long-form (roughly ≥ 1,000 tokens): below that, the per-round network coordination doesn't amortize and a good single-node config tends to win.

## Shard format

Requires **v5 shards** (or a causal VLM tree: Pixtral / Mistral 3, Phi-3.5-vision, InternVL2, LLaVA-NeXT; see [Image input](#image-input-pixtral-target)) (canonical optimum-style inputs: `input_ids|hidden_states, attention_mask, position_ids, beam_idx`). v3 shards are rejected at load time. Generate via `cascadia shard` — its exporter (`tools/export_shards.py`) emits `v5_canonical_inputs` shards; see [../SHARDING.md](../SHARDING.md).

Driver expects the full pipeline directory:
```text
<pipeline_dir>/
  pipeline_config.json
  tokenizer/                 (optional; falls back to model_id's HF cache)
  stage_0/
    openvino_model.xml
    openvino_model.bin
    stage_config.json
  stage_1/...
```

Workers can hold just their own stage as a flat dir:
```text
<worker_model_dir>/
  openvino_model.xml
  openvino_model.bin
  stage_config.json
```

## Wire protocol

`crates/cascadia-engine-openvino/src/dist_spec.rs` defines three frames:

- **FORWARD** (driver → worker → next worker): `[kind=1][logical_pos_start]` + `attention_mask` (int64 `[1, total_seq_len]`) + `hidden_states` (float16 `[1, new_tokens, hidden_size]`)
- **RESET** (driver → workers, propagated): `[kind=3]`
- **LOGITS_RESPONSE** (last worker → upstream): `[kind=4]` + `logits` (float16 `[1, new_tokens, vocab_size]`)

Mask-based rewind is **driver-side only** — the worker just sees a new `attention_mask` on the next FORWARD. There is no REWIND frame.

## Topology

```text
driver (rank 0, has draft + stage_0 + tokenizer)
       ⇣ FORWARD
worker (rank 1, has stage_1)
       ⇡ LOGITS_RESPONSE
```

## Example

```bash
# Worker (last stage, listens on TB/LAN port):
cascadia worker --rank 1 --total 2 --engine ov-dist-spec --device GPU \
              --model /shards/shards_2stage_v5_beam_stage_1 \
              --listen 10.10.10.2:9100

# Driver (rank 0, holds draft, serves the API or stdin):
cascadia worker --rank 0 --total 2 --engine ov-dist-spec --device GPU \
              --model /shards/shards_2stage_v5_beam \
              --next 10.10.10.2:9100 --api :8000 \
              --draft-model /models/llama-3.2-1b-int4-ov \
              --spec-k 4
```

## Image input (Pixtral target)

`ov-dist-spec` takes `image_url` parts when the target is a causal tree with 1-D positions
(Pixtral or Mistral 3, Phi-3.5-vision, InternVL2, LLaVA-NeXT) cut by
`tools/vlm_surgery/cut_vlm_lm.py --all-logits`; Gemma 3 / 4 (token types) and Qwen VL
(M-RoPE) targets are refused at load. Phi-3.5-vision and InternVL2 targets carry Phi-3's
long-RoPE switch, which decides once per inference; this engine prefills in chunks, so a
prompt past 4096 tokens keeps the short factors in its earlier chunks and differs from
Hugging Face (the load log warns; see [../VLM.md](../VLM.md#limits-worth-knowing)).
Only the target sees images:

- **Rank 0** expands each `[IMG]` placeholder into the image's rows of `[IMG]` closed by
  `[IMG_BREAK]` / `[IMG_END]`, runs the vision encoder (CPU, cached per tenant by image
  hash: `CASCADIA_IMAGE_CACHE_MB`), embeds the expanded prompt through
  `openvino_text_embeddings_model.xml` and overwrites the image positions with the
  encoder's rows. Stage 0 takes `inputs_embeds`. Drafted and accepted tokens are embedded
  through the same table with no image rows, so image rows only ever enter the prompt
  prefill, never a speculative span, and the mask-based rewind only touches text rows.
- **The draft** is text-only and is fed the prompt with the image ids (`[IMG]`,
  `[IMG_BREAK]`, `[IMG_END]`) dropped: hundreds of placeholders read as text are likely noise
  to it (0 of 40 drafts accepted on pawan-02). Its cache is its own, so its positions
  need not match the target's. Verification keeps the output the target's greedy output;
  the draft only moves the acceptance rate. It must share the target's token ids
  (Ministral-3-3B for Pixtral 12B: the same tekken vocabulary).
- **Prefill** runs in 512-row chunks (target and draft) sent as `FORWARD_LAST_ROW`
  (kind 10), which the last stage answers with the final row's logits only; verify spans
  keep every row. Wire and memory per inference stay bounded whatever the prompt (8 1 MP
  images expand to about 33k tokens: one 512-row chunk is 512 x 131072 f32 logits, 256
  MiB inside the last stage, and one 256 KiB row on the wire). A worker built before
  kind 10 refuses it loudly (`bad kind 10`): upgrade every rank together.
- **Window**: `submit` answers a prompt at or past the target's `max_position_embeddings`
  with a 400 `context_length_exceeded`, sizing image turns from their dimensions before any encode.
- **Workers** are unchanged: hidden states and the mask travel as for any v5 tree.

What a target tree must provide, and how a `cut_vlm_lm.py` tree meets it:

| Requirement | v5 shard (`export_shards.py`) | VLM tree (`cut_vlm_lm.py`) |
|---|---|---|
| `pipeline_config.json`, `stage_N/{openvino_model.xml, stage_config.json}`, `tokenizer/tokenizer.json` | yes | yes |
| `export_version` not `v3*` | `v5_canonical_inputs` | `vlm_lm_surgery_v1` |
| stage 0: `has_embed: true`; inputs `attention_mask`, `position_ids`, `beam_idx` plus | `input_ids` | `inputs_embeds` (rank 0 embeds) |
| middle and last stages: `hidden_states` (f16), `attention_mask`, `position_ids`, `beam_idx` | yes | yes |
| last stage: `has_head: true`, logits for **every** fed position | yes | only with `--all-logits` (`"logits": "all"` in its `stage_config.json`) |
| rank 0 vision files: `vision` block, `config.vlm.json`, text-embedding and vision IRs | - | yes |

A last stage that slices its head to the last position (a cut without `--all-logits`,
`"logits": "last"`, or an older `vlm_lm_surgery_v1` cut without the field) is refused at
load by rank 0 and by the last worker. A Gemma 4 vision tree is refused at load (its
`token_type_ids` frame does not ride the dist-spec wire). A v5 target keeps refusing
images with a 400.

With `kv_coord`, an image turn always prefills cold and is not captured for warm resume
(its key would be placeholder ids that every image of one size shares); text turns on
the same engine capture and resume as before. A resumed turn (`resume_token_ids`) cannot
carry images.

`usage.prompt_tokens` counts the expanded prompt (image soft tokens included).

Video arrives as labelled frames (`Frame k (t s): ` and one image per frame; the engine
takes no native video), so a video is a multi-image prompt here: the SDK video suite's
4 s clip is 4 frames, 3322 prompt tokens on Pixtral 12B, accepted at 0.22 with
Ministral-3-3B at K=4 (pawan-04, CPU). The answer can repeat the frame labels.

Check a draft against its export (greedy draft vs embeddings + language model, and a
masked-rewind run that must match):

```bash
python tools/vlm_surgery/check_text_draft.py --export <ministral3-3b-int4-ov> \
    --draft <ministral3-3b-text-draft> --image-tokens 784
```

Build the draft and the target tree (Windows paths; models in `C:\vlm\models`, trees in `C:\vlm\trees`):

```bash
# Draft: compose Ministral-3-3B's text embeddings into its language model
python tools/vlm_surgery/make_text_draft.py \
    --model C:\vlm\models\ministral3-3b-int4-ov \
    --output-dir C:\vlm\models\ministral3-3b-text-draft

# Target: two stages with logits for every position
python tools/vlm_surgery/cut_vlm_lm.py \
    --model C:\vlm\models\pixtral-12b-int4-ov \
    --output-dir C:\vlm\trees\pixtral-12b-distspec-2 \
    --num-stages 2 --family pixtral --all-logits
```

The draft directory gets one `openvino_model.xml` (inputs `input_ids`, `attention_mask`,
`position_ids`, `beam_idx`; the language model's KV state and logits), the tokenizer
files and a flattened `config.json` whose `vocab_size` rank 0 checks prompt ids against.

```bash
# rank 1 (last stage) first
cascadia worker --total 2 --rank 1 --engine ov-dist-spec --device GPU \
    --model C:\vlm\trees\pixtral-12b-distspec-2 --listen <rank1-host>:9201
# rank 0: draft + stage 0 + vision front end, serves the API
cascadia worker --total 2 --rank 0 --engine ov-dist-spec --device GPU \
    --model C:\vlm\trees\pixtral-12b-distspec-2 \
    --draft-model C:\vlm\models\ministral3-3b-text-draft \
    --spec-k 4 --next <rank1-host>:9201 --api 127.0.0.1:8099
```

What rank 0 logs: `vision target: rank 0 embeds every span and encodes images (...)` at
load, `accepting image_url content parts` from the API, then per image turn
`ov-dist-spec: images encoded` (images, prompt tokens, encode_ms), `image rows spliced
into the target's stage 0 prefill` (rows), and `ov-dist-spec done` with `drafted`,
`accepted`, `accept` and `images`. The same `--all-logits` tree also serves `ov-runtime`
(it reads the last row), so a greedy `ov-runtime` answer on it is the reference output.

## Picking K

K is the draft length per spec round. Higher K = more parallel target verifications but more wasted work when the draft is wrong.

Measured on Llama-3.1-8B INT4 target + Llama-3.2-1B INT4 draft, two hosts linked via Thunderbolt:

| K | 64-tok factual | accept | 256-tok creative | accept |
|---|----------------|--------|------------------|--------|
| 3 | 15.84 tok/s    | 0.76   | 14.68 tok/s      | 0.55   |
| **4** | **16.11**  | 0.62   | **15.52**        | 0.50   |
| 5 | **16.87**      | 0.65   | 13.91            | 0.42   |
| 6 | 15.59          | 0.60   | 12.70            | 0.38   |

**K=4 is the recommended default.** K=5 wins narrowly on short factual prompts but loses ~11% on long creative prompts. Push K higher only if your accept rate is consistently ≥ 0.7.

## Performance notes

- Mask-based rewind (v5) is ~free vs the ~40 ms-per-call physical rewind that v3 shards forced. The win shows up most on long generations and low-accept prompts.
- Per-FORWARD network cost dominates on slow links. Thunderbolt 4/5 between hosts on AC power gives a 25% throughput bump over LAN; on battery the link drops under load.
- `--device GPU` resolves to whatever Intel iGPU/dGPU OpenVINO sees; `cascadia engines` does not currently list devices.
