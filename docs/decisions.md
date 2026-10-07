# Decisions API

Cascadia can serve **decision models**: fine-tuned Qwen3.5-family checkpoints that answer typed
questions (yes/no, pick one, rate on a scale) about a piece of text by reading probabilities off a
single prefill pass. They never generate text. A model directory turns into a decision server when
it carries a `decision.json` next to the shard tree:

```bash
cascadia run /models/kev-9b-int8 --engine qwen35 --api 127.0.0.1:8000
```

The server logs `decision.json found; serving /v1/decisions (OpenAI format) and /v1/systemone` and
exposes two routes over the same core:

| Route | Wire format |
|---|---|
| `POST /v1/decisions` | [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions). OpenAI SDK clients work unchanged. |
| `POST /v1/systemone` | Jev/SystemOne format (TypeSafe's system-one-adapter). |

Both routes run identical prompts through the same model; only the request and response shapes
differ. Only `--engine qwen35` serves decisions. With any other engine a `decision.json` is ignored
with a startup warning, and both routes answer 501. A `decision.json` that fails to load (bad JSON,
missing `tokenizer.json`) is logged and decisions stay off; chat keeps working.

> **Security.** Like the rest of the HTTP API, these routes are plaintext and unauthenticated. See
> [SECURITY.md](../SECURITY.md).

## `POST /v1/decisions` (OpenAI format)

### Request

```json
{
  "model": "kev-9b",
  "input": "I was charged twice for order 7 and support has not replied in three days.",
  "questions": [
    {"type": "predicate", "name": "urgent", "instructions": "Does this need a reply today?"},
    {"type": "choice", "name": "department", "instructions": "Which team should handle this?",
     "choices": [
       {"value": "billing", "description": "Payments, refunds, invoices."},
       {"value": "technical", "description": "Bugs and outages."},
       {"value": "other"}
     ]},
    {"type": "score", "name": "severity", "instructions": "How badly is the customer affected?",
     "levels": [
       {"label": "Cosmetic", "description": "Appearance only."},
       {"label": "Workaround available"},
       {"label": "Fully blocked"}
     ]}
  ]
}
```

- `model` is accepted and ignored: one server serves one model.
- `input` is a string, or a list of messages `{"role": "user", "content": ...}`. `content` is a
  string or a list of `{"type": "input_text", "text": ...}` parts. A message's parts are
  concatenated, and messages are joined with `"\n"`. Only `role: "user"` is accepted (an omitted
  `role` counts as `"user"`).
  `input_image` parts are a 400: this server is text-only.
- `questions[].name` is required, non-empty and unique. Answers come back in request order.
- `choices[].value` is a string or a number; any other JSON type is a 400. Duplicate values are a
  400.
- `choices[].description` and `levels[].description` are optional.

### Response

```json
{
  "answers": [
    {"type": "predicate", "name": "urgent", "probability": 0.8},
    {"type": "choice", "name": "department", "choice": "billing",
     "probabilities": [
       {"value": "billing", "probability": 0.95},
       {"value": "technical", "probability": 0.03},
       {"value": "other", "probability": 0.02}
     ],
     "confidence": 0.925},
    {"type": "score", "name": "severity", "score": 1.1,
     "probabilities": [
       {"value": 0, "label": "Cosmetic", "probability": 0.1},
       {"value": 1, "label": "Workaround available", "probability": 0.7},
       {"value": 2, "label": "Fully blocked", "probability": 0.2}
     ],
     "confidence": 0.55}
  ]
}
```

- `predicate`: `probability` is p(true).
- `choice`: `choice` is the most likely value (the first one on a tie), echoed with its original
  JSON type. `probabilities` lists every choice in request order.
- `score`: `score` is the probability-weighted level index, Σ i·pᵢ. `probabilities` lists every
  level with its index and label.
- The response is exactly `{"answers": [...]}`. The `refusal` answer type is never emitted,
  because these models do not refuse.

### Errors

Errors use OpenAI's envelope, which the OpenAI SDKs parse into typed exceptions:

```json
{"error": {"message": "between 1 and 32 questions are allowed, got 0", "type": "invalid_request_error", "code": null}}
```

`type` is `invalid_request_error` for 400 and 413, and `server_error` for 501 and 503. A body that
is not valid JSON, or does not match the request schema, is rejected by the HTTP framework before
the handler runs, with its default 4xx body rather than this envelope (the same as the other
routes). SDK clients only send well-formed bodies.

### Mapping notes

The OpenAI request is translated into the internal (SystemOne) form before the model sees it:

- `predicate` becomes a yes/no question with no criteria.
- `choice` becomes a choice with criteria `{value: description}` in request order. A number value
  is keyed by Python's `str()` of the number (`2` → `"2"`, `1e5` → `"100000.0"`); a missing
  description is `null`.
- **Score levels are rendered as one text per level**, `"label: description"`, or just `"label"`
  when the description is missing or empty, before the codec sees them. This approximates the single-text
  levels the models were trained on; nothing in the reference implementations says which form
  they score best on. JEV never sees level text (see [JEV score answers](#jev-score-answers-are-an-approximation)),
  so only Kev and Clef are affected.
- **OpenAI's docs describe questions as independent.** JEV and Kev do decide each question in its
  own pass. Clef decides all of a request's questions in one joint pass, by design: that is how
  it was trained, and its answers to one question can depend on the others asked alongside it.

### OpenAI SDK example

```python
from openai import OpenAI

client = OpenAI(base_url="http://HOST:PORT/v1", api_key="unused")

result = client.decisions.create(
    model="kev-9b",  # ignored; any string works
    input="I was charged twice for order 7.",
    questions=[
        {"type": "predicate", "name": "urgent", "instructions": "Does this need a reply today?"},
        {
            "type": "choice",
            "name": "department",
            "instructions": "Which team should handle this?",
            "choices": [
                {"value": "billing", "description": "Payments, refunds, invoices."},
                {"value": "technical", "description": "Bugs and outages."},
            ],
        },
    ],
)
for answer in result.answers:
    print(answer)
```

The server does not check `api_key`; the SDK only requires that one is set.

## `POST /v1/systemone` (Jev/SystemOne format)

### Request

```json
{
  "model": "kev-9b",
  "state": {"ticket": "I was charged twice for order 7.", "customer_tier": "gold"},
  "questions": {
    "urgent": {"type": "noul", "instructions": "Does this need a reply today?"},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, refunds, invoices.", "technical": "Bugs and outages."}},
    "severity": {"type": "score", "instructions": "How badly is the customer affected?",
                 "criteria": ["Cosmetic", "Workaround available", "Fully blocked"]}
  }
}
```

- `state` is any JSON value. The codecs render it exactly as the reference servers do, so key
  order is kept as sent.
- `questions` is an object keyed by question id. Answers come back in the same order.
  - `noul`: yes/no. `criteria` is optional.
  - `choice`: `criteria` is an object, option name → description.
  - `score`: `criteria` is an array of level descriptions, lowest first.
- `images` and `videos` must be absent or empty (400 otherwise). Image parts inside `state` are
  also a 400.

### Response

```json
{
  "model": "kev-9b",
  "answers": {
    "urgent": {"type": "noul", "noul": 0.8},
    "department": {"type": "choice", "choice": "billing", "confidence": 0.9,
                   "probabilities": {"billing": 0.95, "technical": 0.05}},
    "severity": {"type": "score", "score": 1.1, "confidence": 0.55,
                 "legend": {"0": "Cosmetic", "1": "Workaround available", "2": "Fully blocked"},
                 "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}}
  },
  "usage": {"input_tokens": 412, "output_tokens": 0}
}
```

`usage.input_tokens` is the number of prompt tokens prefilled across all passes. `output_tokens`
is always 0: decision models do not decode.

### Errors

Errors are `{"error": "<message>"}` with the status codes below.

## Status codes

Both routes share these statuses.

| Status | When |
|---|---|
| 200 | Every question answered. |
| 400 | Validation: no questions or more than `max_questions`; an empty question name, or (OpenAI format) a duplicate question name or choice value; a choice with no options or more than the model allows; a score with fewer than 2 levels; images; a request the tokenizer cannot encode; Clef's option cap (see [Limits](#limits)). |
| 413 | Too long: the state is more than `max_total_tokens × 8` bytes (checked before tokenizing); one pass is longer than `max_tokens` (JEV and Kev; Clef truncates the state instead); the request's prefill work is more than `max_total_tokens`; or the engine reports the prompt too long. |
| 501 | The model serves no decisions: no `decision.json`, an engine other than `qwen35`, or the engine reports the model has no decision head. |
| 503 | The server is at capacity (`engine at capacity; retry after current requests complete`), or any other engine error. Decisions share the API's concurrency permits with chat requests. |

Error messages are deliberately terse:

- **Tokenizer failures** return the fixed message `request could not be tokenized`. The
  tokenizer's own text can carry vocabulary or file-path detail, so it is logged, not returned.
- **Validation messages** (for example `image parts in state are not supported (text-only server)`,
  or Clef's `N options across all questions; the limit is 4096`) are returned as-is: they name a
  problem the caller can fix.
- **Engine errors** return a fixed message per category (`decision engine error`, `engine at
  capacity or unavailable`, `decision prompt too long`). A request that finds every concurrency
  permit taken gets `engine at capacity; retry after current requests complete`. The engine's text (paths, OpenVINO
  messages) is logged at `warn` and never returned.

## Limits

| Limit | Source | Default |
|---|---|---|
| Questions per request | `max_questions` in `decision.json` | 32 |
| Tokens in one pass | `max_tokens` in `decision.json` | required field |
| Prefill tokens per request | `max_total_tokens` in `decision.json` | 4 × `max_tokens` |
| Options per choice | per model family | JEV: 16 for the shipped JEV profiles (the profile's choice slot count); Kev and Clef: 256 |
| Options per request (Clef) | engine readout cap | 4096 across all questions, because Clef answers them in one pass |
| Levels per score | per model family | 2 to 256 |
| Request body | `--api-max-body-mb` | 1 MiB |

- **Clef truncates the state to fit `max_tokens`** (as Cloudflare's `encode_record` does), so a
  long state never makes a Clef pass too long; JEV and Kev return 413 when a pass exceeds
  `max_tokens`. The request-level `max_total_tokens` byte pre-check still applies to every family.
- **JEV and Kev run one prefill pass per question**, so `max_questions` bounds the engine work of
  one request. Clef runs one pass for the whole request.
- **Prefill work** is the sum of all passes' prompt lengths. Kev repeats the state at the start of
  every question's pass. On a single box with the prefix cache on (`--total 1` and
  `--prefix-cache-gb` > 0, the default), the engine prefills that shared prefix once and resumes it
  for each question, so it counts once: a Kev request with a 64K-token state and 5 questions costs
  about 64K plus five short branches. In pipeline mode (`--total` > 1) or with
  `--prefix-cache-gb 0`, every pass counts in full, because the engine re-prefills the state per
  question.
- **Raise `--api-max-body-mb` for long states.** The default 1 MiB body cap (it applies to every API route) holds roughly 250K
  tokens of English, but JSON escaping and a long Kev state with many questions can exceed it.
  The flag applies to every route, chat included.
- **Shared concurrency permits.** Decision requests take the same concurrency permits as chat
  requests. A request that finds none free gets 503 (`engine at capacity; retry after current
  requests complete`) at once rather than queueing.

## Confidence

**Confidence follows TypeSafe's system-one-adapter 0.2.1**, the formulas Kev's own server also
uses, for every model family and on both routes:

- **choice:** `(p_max − 1/K) / (1 − 1/K)` over the K options. 0 at a uniform distribution, 1 at
  certainty, and 1 for a single option.
- **score:** `max(0, 1 − E|level − mode| / D)` over the L levels, where `mode` is the most likely
  level (the first on a tie) and `D` is the mean absolute deviation of a uniform distribution over
  the levels around their midpoint.

OpenAI's documented examples match these formulas exactly: choice probabilities
[0.95, 0.02, 0.01, 0.02] give 0.93 (the server reports 0.9333), and score probabilities
[0.1, 0.7, 0.2] give 0.55.

Cloudflare's own Clef server reports the maximum probability as confidence instead. On the spike's
records the two differ by 0.05 on average and by up to 0.36 for Clef. Probabilities and the chosen
answer are identical either way; only the `confidence` field differs.

Probabilities are a softmax computed in f64 and rounded to 4 decimals (half-to-even on the exact
binary value, like Python's `round(x, 4)`), so they match the reference servers digit for digit.

## JEV score answers are an approximation

JEV only knows a fixed 0–5 scale and never sees the level descriptions. The server maps its 6-way
distribution linearly onto the requested L levels (`rescale_score`). On the 25 four-level score
questions in the parity set, this mapping agreed with models that do read the descriptions (Kev
and Clef) more often than truncation did: 20 vs 19 answers consistent with their choices, and a
mean expected-score gap of 0.08 vs 0.13. Those two models agree with each other on only 15 of the
25 score questions, so treat JEV score answers as indicative. For exact score semantics, use Kev
or Clef.

## Prompt fidelity

Each model family has a codec that reproduces its reference implementation's prompt text exactly:
key order, separators, and Python `str()` / `json.dumps` formatting.

- **Numbers.** Floats in request JSON are parsed exactly and rendered like Python's `repr`
  (`1e-05`, `1.0`, `0.1`), as the reference servers would print them. Integers outside the
  i64/u64 range are the one known gap: they are rendered as floats, where Python would keep every
  digit.
- **Caller text is sanitized.** Hugging Face tokenizers match special tokens inside plain text, so
  a state or option text containing `<|im_end|>` could otherwise forge the prompt's structure and
  steer the decision. The server rewrites every `<|name|>` sequence, and every other angle-bracket
  added token of the model's tokenizer (for example `<think>` or `<tool_call>`), to `<¦…¦>`
  (`<|im_end|>` → `<¦im_end¦>`). This is stricter than the reference servers on purpose, for
  prompt-injection safety, so prompts differ from the references only for caller text that
  contains such strings. The prompt template's own special tokens are unaffected.

## `--decision-dump FILE`

```bash
cascadia run /models/kev-9b-int8 --engine qwen35 --decision-dump /tmp/kev-passes.jsonl
```

`cascadia run` and `cascadia worker` accept `--decision-dump FILE`. Every decision pass is
appended to FILE as one JSON line, `{"parts": [[question_id, start, end], ...], "readout": {...}}`:
the prompt token ids, the readout head layout, and which logit range answers which question. The
parity and golden tooling reads this file to compare the server's prompts with the reference
encoders.

- The pass is written **before the engine runs**, so a pass the engine later rejects is still
  recorded.
- **Privacy.** The file holds request-derived data: the token ids decode back to the caller's
  state and questions. Use it for parity checks only, never on a production endpoint. The server
  logs a warning at startup when the flag is set.
- The file is opened once at startup in append mode with permissions 0600. An existing file is
  tightened to 0600, and a symlink at the path is refused. A failure to open it stops startup.
- It is a startup flag only; there is no environment variable for it.
- It takes effect only when a decision model is loaded. Otherwise the server logs that
  `--decision-dump` is ignored and opens no file.

## Limitations

- **Text only.** Images in either format are a 400.
- **Batch 1.** The engine runs one decision pass at a time; concurrent requests beyond the engine's
  capacity get 503.
- **int8 weights only.** Decision trees are built from Intel's int8 OpenVINO exports. int4 measured
  87.8–96.3% top-1 agreement with full precision, against 98.8–100% for int8, and is not supported.
- **JEV choice questions take at most 16 options** for the shipped JEV profiles (the profile's
  choice slot count).
- **`--engine qwen35` only.** Other engines cannot run decision passes.

## Models

Each supported decision model has a section below with its `decision.json`, export steps and
measured parity.
