# Phi-3 / Phi-4 support

Both Phi-3 and Phi-4 report `model_type: "phi3"` (Microsoft chose to
keep the family identifier across major versions; Phi-4 is structurally
a refresh of the Phi-3 architecture). Cascadia's `detect_architecture`
matches `"phi"` and dispatches to `Phi3DecoderLayer`, so the variants
listed below load and trace.

## Variants and config quirks

| Model | `model_type` | Quirks not yet honoured |
|-------|--------------|-------------------------|
| `microsoft/Phi-3-mini-4k-instruct` | `phi3` | `partial_rotary_factor = 1.0` (none) |
| `microsoft/Phi-3-mini-128k-instruct` | `phi3` | `partial_rotary_factor = 1.0`; LongRoPE scaling |
| `microsoft/Phi-3-medium-4k-instruct` | `phi3` | `partial_rotary_factor = 1.0` |
| `microsoft/Phi-3-medium-128k-instruct` | `phi3` | LongRoPE scaling |
| `microsoft/Phi-3-small-8k-instruct` | `phi3` | tied embeddings; sliding window |
| `microsoft/phi-4` | `phi3` | `partial_rotary_factor = 1.0`; works on default 16k context |
| `microsoft/Phi-4-mini-instruct` | `phi3` | **`partial_rotary_factor = 0.75`** (the trailing 25% of each head is NOT rotated); LongRoPE scaling; `sliding_window = 262144`; `vocab_size = 200064` |
| `microsoft/Phi-4-reasoning` | `phi3` | Like Phi-4 14B + chain-of-thought training |
| `microsoft/Phi-4-mini-reasoning` | `phi3` | Like Phi-4-mini + reasoning |

## What works today

Phi-3 / Phi-4 (including Phi-4-mini) export and run; **partial rotary is
handled** (#69). LongRoPE long-context extension is not modeled (a SOFT
quirk — exact within the original context window, degrades beyond it):

- ✅ `microsoft/phi-4` (14B).
- ✅ `microsoft/Phi-3-mini-4k-instruct`.
- ✅ `microsoft/Phi-3-medium-4k-instruct`.
- ✅ `microsoft/Phi-4-mini-instruct` — `partial_rotary_factor=0.75` is
  honoured; only LongRoPE long-context (>4k) degrades.

## Partial rotary (`partial_rotary_factor`) — handled (#69)

Phi-4 Mini sets `partial_rotary_factor = 0.75`. With `head_dim = 96`,
RoPE rotates the leading `int(0.75 * 96) = 72` dims and leaves `[72, 96)`
untouched. `TracedRotaryEmbedding` now emits cos/sin at width
`rotary_dim = int(partial_rotary_factor * head_dim)`, and `apply_rotary`
rotates only that leading slice (`rotate_half` splitting at
`rotary_dim/2`), concatenating the untouched tail — matching
transformers' Phi/Phi3 `apply_rotary_pos_emb` byte-for-byte on the
APPLIED q/k (not just cos/sin), verified on transformers 4.57 + 5.9.

## What's still dropped

### 1. LongRoPE scaling

Phi-3 Mini 128k and Phi-4 Mini set:

```json
"rope_scaling": {
  "type": "longrope",
  "short_factor": [1.0, 1.1, 1.2, ...],
  "long_factor": [1.0, 1.4, 2.1, ...],
  "original_max_position_embeddings": 4096
}
```

LongRoPE applies different per-dim scaling factors depending on whether
the current sequence is shorter or longer than
`original_max_position_embeddings`. The exporter bakes plain RoPE and
does not apply the short/long factors; `check_export_quirks` emits a
SOFT warning and proceeds (it is correct within the original context
window).

**Symptom:** output quality is fine up to ~4k tokens; degrades rapidly
beyond.

**Fix path:** bake the LongRoPE short/long `inv_freq` into
`TracedRotaryEmbedding` (select by sequence length), or model it in the
runtime for a v3-style external-rotary path.

### 2. Sliding window

Phi-3 Small uses sliding-window attention with a fixed window. Cascadia
treats all layers as full causal. For Phi-3 Small specifically this is
usually fine because the window is large.

## Phi-4-multimodal

`microsoft/Phi-4-multimodal-instruct` has `model_type:
"phi4mm_for_causal_lm"` and a different decoder layout (vision + audio
encoders + LoRA adapters per modality). Not supported. Use the text
backbone separately if extractable.
