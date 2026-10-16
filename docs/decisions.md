# Decisions API

Cascadia can serve **decision models**: fine-tuned Qwen3.5-family checkpoints that answer typed
questions (yes/no, pick one, rate on a scale) about a piece of text by reading probabilities off a
single prefill pass. They never generate text. A model directory turns into a decision server when
it carries a `decision.json` next to the shard tree:

```bash
cascadia run /models/kev-9b-int8 --engine qwen35 --device CPU --api 127.0.0.1:8000
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
missing `tokenizer.json`) is logged and decisions stay off; chat keeps working. So is one that is
there but cannot be read, such as a dangling symlink or a directory: it logs `decision.json present
but unusable`, never passes for a plain chat model.

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
- `choices[].value` is a string or a number; any other JSON type is a 400. Numbers are read
  exactly, by the same reader as `/v1/systemone`: an integer keeps every digit at any size, and a
  float is the correctly rounded double of its literal. Duplicate values are a
  400. Values are compared by their internal key, Python's `str()` for a number (see
  [Mapping notes](#mapping-notes)), so `2` and `"2"` count as the same value and are a 400 too.
- `choices[].description` and `levels[].description` are optional.

### Response

```json
{
  "id": "dec-3f2a9c0e8b7d4e61a5c2f0b9d8e7a6c5",
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
  ],
  "usage": {"input_tokens": 412, "output_tokens": 0, "total_tokens": 412}
}
```

- `id` is unique per response: `dec-` and 32 hex digits.
- `usage.input_tokens` is the number of prompt tokens prefilled across all passes, the same count
  `/v1/systemone` reports. `output_tokens` is always 0 (decision models do not decode), so
  `total_tokens` equals `input_tokens`.
- `predicate`: `probability` is p(true).
- `choice`: `choice` is the most likely value (the first one on a tie), echoed with its original
  JSON type. `probabilities` lists every choice in request order.
- `score`: `score` is the probability-weighted level index, Σ i·pᵢ. `probabilities` lists every
  level with its index and label.
- The response is exactly `{"id", "answers", "usage"}`. The `refusal` answer type is never
  emitted, because these models do not refuse.

### Errors

Errors use OpenAI's envelope, which the OpenAI SDKs parse into typed exceptions:

```json
{"error": {"message": "between 1 and 32 questions are allowed, got 0", "type": "invalid_request_error", "code": null}}
```

`type` is `invalid_request_error` for 400 and 413, and `server_error` for 501 and 503. A body that
does not match the request schema is a 400 in this envelope too, never the framework's 422 text.
An unknown or malformed content part is named, for example `unsupported content part type
'input_file'` (an unknown type is echoed up to 64 characters, then `…`) or `input_text parts need
a string "text"`. A key repeated in the same object is a 400 (`invalid request body: duplicate
field ...`), not last-one-wins; any other schema mismatch (a missing
`questions`, an unknown question `type`) is `invalid request body: <parser message>`. A body that
is not valid JSON (400), a missing `content-type: application/json` (415) or a body over
`--api-max-body-mb` (413) keeps the framework's status and message, inside this envelope with
`type` `invalid_request_error`.

### Mapping notes

The OpenAI request is translated into the internal (SystemOne) form before the model sees it:

- `predicate` becomes a yes/no question with no criteria.
- `choice` becomes a choice with criteria `{value: description}` in request order. A number value
  is keyed by Python's `str()` of the parsed number, exactly as `/v1/systemone` renders the same
  literal (`2` → `"2"`, `1e5` → `"100000.0"`, `1e20` → `"1e+20"`, a 30-digit integer → all 30
  digits); a missing description is `null`. `-0` is the integer `0`, keyed `"0"` as Python's
  `json.loads("-0")` is, so `0` and `-0` are duplicates; `-0.0` is a float, keyed `"-0.0"`. The
  answer echoes the value as sent, every digit included.
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

- `state` is any JSON value, rendered exactly as each reference server renders it: JEV and Kev
  keep object keys in the order sent; Clef sorts them (`json.dumps(sort_keys=True)`, as
  Cloudflare's `encode_record` does).
- `questions` is an object keyed by question id. Answers come back in the same order.
  - `noul`: yes/no. `criteria` is optional.
  - `choice`: `criteria` is an object, option name → description.
  - `score`: `criteria` is an array of level descriptions, lowest first.
- `images` and `videos` must be absent, `null` or empty (400 otherwise).
- A list `state` is read differently per family, as by each reference. JEV reads its elements as
  message parts (serve_decide `_parts`), so an image part there (`{"image": ...}` or
  `{"type": "image_url", ...}`) is a 400, and so is a `{"type": "text"}` part without a string
  `text` (`text parts in state need a string "text"`; serve_decide would fail on it). Kev and
  Clef render list elements as plain data, so a catalogue record with an `image` field is just
  text to them.

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

- `legend` (score answers) maps each level index to its level. Kev returns each level as its
  reference renders it: text, `key: value` lines for an object and Python `str()` for a scalar
  (`true` → `"True"`, `2` → `"2"`). JEV and Clef echo the JSON value sent. So a score with
  `"criteria": [{"name": "x"}, true]` gets `{"0": "name: x", "1": "True"}` from Kev and
  `{"0": {"name": "x"}, "1": true}` from JEV and Clef. The OpenAI route is the same for every
  family: its `label` comes from the request's `levels`.

`usage.input_tokens` is the number of prompt tokens prefilled across all passes. `output_tokens`
is always 0: decision models do not decode.

### Errors

Errors are `{"error": "<message>"}` with the status codes below. Unlike the OpenAI route, a body
that is not valid JSON or does not match the request schema (a missing `questions`, an unknown
question `type`, a value nested deeper than 128 levels) is rejected by the HTTP framework before
the handler runs, with its default 4xx text body rather than this shape.

## Status codes

Both routes share these statuses.

| Status | When |
|---|---|
| 200 | Every question answered. |
| 400 | Validation: no questions or more than `max_questions`; an empty question name, or (OpenAI format) a duplicate question name or choice value; a choice with no options or more than the model allows; a score with fewer levels than the model allows (2 for JEV, 1 for Kev and Clef) or more; images; an image part, or a text part without a string `text`, in a list `state` (JEV); a request the tokenizer cannot encode; Clef's option cap (see [Limits](#limits)). |
| 413 | Too long: the state is more than `max_total_tokens × 8` bytes (checked before tokenizing); Kev's whole rendered text (state, instructions, options and levels) is more than `max_total_tokens × 8` bytes; Kev's rendered state is more than 524,288 bytes (65,536 × 8, checked before tokenizing); one Kev question's instructions and option texts render to more than 8 bytes per token the pass has left after the state (`max_tokens` minus the state's tokens, checked before tokenizing); Kev's state is more than 65,536 tokens (`SERVE_MAX_STATE`, `<state>` included) or reaches `max_tokens`; one pass is longer than `max_tokens` (JEV and Kev; Clef truncates the state instead); Clef's schema alone fills the joint pass, leaving no room for a non-empty state; the request's prefill work is more than `max_total_tokens` (JEV stops tokenizing at the first question that takes it over: `needs at least N`); a Kev request that needs the shared-state discount when the prefix cache can never hold that state (`this request needs N prefill tokens without prefix reuse (the shared prefix cannot fit the prefix cache); the per-request limit is M`, refused before any prefill once the engine knows the snapshot's size, which its warmup measures at startup; see [Limits](#limits) for when the first such request can still spend one pass); or the engine reports the prompt too long. |
| 501 | The model serves no decisions: no `decision.json` (or one that fails its load checks; see the startup log), an engine other than `qwen35`, the engine has no decision head (`decision/` missing on the single box or the pipeline's last rank), the tree was exported without `--final-hidden` (Kev and Clef), or a pipeline rank cannot serve decision passes (an older build, rank 0 or the last rank without `config.json` `vocab_size`, or the last rank without `decision/`; a middle rank only relays and needs neither). Rank 0 refuses the pipeline cases before any prefill. |
| 503 | The server is at capacity (`engine at capacity; retry after current requests complete`): every concurrency permit is taken, or a Kev pass that must resume its shared state from the prefix cache found the snapshot evicted by other requests (see [Limits](#limits)); retrying can succeed. Any other engine error is `decision engine error`, including non-finite logits or probabilities (never a 200 with `null`s). Decisions share the API's concurrency permits with chat requests. |

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
| Questions per request | `max_questions` in `decision.json` | 32 (Clef: at most 256, the joint head's cap) |
| Tokens in one pass | `max_tokens` in `decision.json` | required field |
| Prefill tokens per request | `max_total_tokens` in `decision.json` | 4 × `max_tokens` |
| Options per choice | per model family | JEV: 16 for the shipped JEV profiles (the profile's choice slot count); Kev: 255 (Kev's `MAX_OPTIONS`); Clef: 256 |
| Options per request (Clef) | engine readout cap | 4096 across all questions, because Clef answers them in one pass |
| Tokens in the joint pass (Clef) | `MAX_JOINT_PROMPT` and the head's input limit | the smaller of `max_tokens`, 16,384 and 320 MiB ÷ (4 × width): 16,384 for both Clef-flash (width 4096) and Clef-27B (width 5120) |
| Levels per score | per model family | JEV: 2 to 256; Clef: 1 to 256; Kev: 1 to 255 |
| State tokens (Kev) | Kev's `SERVE_MAX_STATE` | 65,536, `<state>` included; only the whole pass may reach `max_tokens`. The rendered state is also capped at 8 bytes per token (524,288 bytes) before tokenizing |
| Concurrent tokenizations | fixed | min(4, CPU cores), separate from the request permits |
| Request body | `--api-max-body-mb` | 1 MiB |

- **`decision.json` is checked at load.** `max_tokens`, `max_total_tokens` and `max_questions`
  must be at least 1, a JEV choice slot range at least 1 wide, and Clef's `max_total_tokens` must
  cover one full joint pass. `max_tokens` above the engine's 131,072-token decision prompt cap,
  and Clef's `max_questions` above 256, are clamped with a warning, so a request over the
  engine's own limits is a 400 or 413 rather than a 503. Calibration values must keep
  `(logit + bias) / temperature` finite: every JEV and Kev `temperature` in [0.001, 1000] and
  every JEV `bias` in [-10,000, 10,000] (shipped profiles use temperatures near 1 and biases of a
  few units). A profile that fails these checks
  disables decisions with a startup warning. The startup log names the family and the
  effective limits.
- **Clef truncates the state to fit `max_tokens`** (as Cloudflare's `encode_record` does), so a
  long state never makes a Clef pass too long; JEV and Kev return 413 when a pass exceeds
  `max_tokens`. Clef's effective bound is also capped by the joint head's inputs, each of which
  must fit the native shim's 320 MiB per-input limit (`MAX_TENSOR_BYTES`, mirrored as
  `MAX_HEAD_INPUT_BYTES`). The limit is sized to Clef-27B's `[16384, 5120]` f32 hidden rows,
  exactly 320 MiB, so both Clef models reach the full 16,384 tokens; only a wider tree is
  tightened (width 8192: 10,240). The `[questions + options, T]` span mask gives 2^26 × 1.25 ÷
  (questions + options), at least 19,275 within the 256-question and 4096-option caps, so it
  does not bind today. The state is truncated to the tightest of these; a schema that alone
  exceeds it is a 413, and the engine refuses an over-limit joint readout (413) before any
  prefill. The joint head runs where the final hidden rows are (the single box, or the last
  pipeline rank), so the hidden input never crosses the wire: pipeline prefill sends 256-token
  chunks (5 MiB at width 5120), far under the transport's 256 MiB frame cap. In a mixed
  pipeline whose last rank predates this limit, that rank still refuses a Clef-27B readout over
  13,107 tokens with the same 413 (`decision_too_long`), but only after the chain prefill: rank
  0 cannot tell from HELLO which limit the last rank holds. The wire format is unchanged. The request-level `max_total_tokens` byte pre-check still applies to every family.
- **JEV and Kev run one prefill pass per question**, so `max_questions` bounds the engine work of
  one request. Clef runs one pass for the whole request.
- **Prefill work** is the sum of all passes' prompt lengths. Kev repeats the state at the start of
  every question's pass. On a single box with the prefix cache on (`--total 1` and
  `--prefix-cache-gb` > 0, the default), the engine prefills that shared prefix once and resumes it
  for each question, so it counts once: a Kev request with a 64K-token state and 5 questions costs
  about 64K plus five short branches. In pipeline mode (`--total` > 1) or with
  `--prefix-cache-gb 0`, every pass counts in full, because the engine re-prefills the state per
  question.
- **The discount is never exceeded.** A request whose budget covers every pass in full runs as
  usual. A request that needs the discount (its passes in full exceed `max_total_tokens`) relies
  on the prefix cache:
  - Before its first pass prefills anything, the engine checks that the cache can hold a snapshot
    of the state at all: the cache is on, the state is at least 16 tokens (`MIN_PREFIX_TOKENS`),
    and its snapshot is not larger than the cache. The snapshot's size is known before the first
    request: the warmup copies its one-token state once and reads from it the exact size at every
    length (each attention layer's bytes per token plus the fixed recurrent state; the startup log
    prints `fixed` and `per_token`). If the snapshot cannot fit, the request is a 413 that names
    its full need, with no prefill spent, the first request after startup included. A state
    under 16 tokens never earns the discount, so it is budgeted in full.
  - When the warmup could not read that size (the warmup failed, the state copy failed, or the
    state has no readable attention layout; the startup log says so), the engine knows only a
    lower bound: the tree's `config.json` attention dims at half a byte per element, or the
    warmup state's total. A state whose real snapshot lies between that bound and the cache size
    then spends one full pass (its prefill and the snapshot copy) before the 413. That happens
    at most once per size per server start: the refused snapshot is measured, and later requests
    of that size or longer are refused before any prefill.
  - The first pass then always snapshots the state, even when it resumed a shorter cached state
    that this one extends (a log or transcript that grew since the last request).
  - Every later pass resumes that snapshot. If another request's passes evicted it, the request
    stops at that pass with the capacity 503 instead of prefilling the state again past its
    budget. Retry, or raise `max_total_tokens` or `--prefix-cache-gb`.
- **Raise `--api-max-body-mb` for long states.** The default 1 MiB body cap (it applies to every API route) holds roughly 250K
  tokens of English, but JSON escaping and a long Kev state with many questions can exceed it.
  The flag applies to every route, chat included.
- **Shared concurrency permits.** Decision requests take the same concurrency permits as chat
  requests. A request that finds none free gets 503 (`engine at capacity; retry after current
  requests complete`) at once rather than queueing. A request holds its permit until its
  tokenization has finished, even if the client disconnects first. Decision requests count in the
  request total, the in-flight gauge and the `capacity` rejection metric like chat requests.
- **Bounded tokenizer memory.** Tokenizing a long state takes a few hundred MB of transient memory,
  so at most min(4, CPU cores) decision requests tokenize at once; the others wait their turn
  (holding their permit) rather than failing. Kev also refuses a state that renders to more than
  8 bytes per allowed token (524,288 bytes) before tokenizing it, and a question whose
  instructions and option texts render to more than 8 bytes per token the pass has left after the
  state; honest text tokenizes at about 4 bytes per token.

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

- **Numbers.** Numbers in request JSON, on both routes, are parsed exactly and rendered as the
  reference servers would print them: integers of any size with every digit, floats like Python's
  `repr` (`1e-05`, `1.0`, `0.1`, `1e+20`).
- **Caller text is sanitized.** Hugging Face tokenizers match special tokens inside plain text, so
  a state or option text containing `<|im_end|>` could otherwise forge the prompt's structure and
  steer the decision. The server rewrites every `<|name|>` sequence, and every other angle-bracket
  added token of the model's tokenizer (for example `<think>` or `<tool_call>`), to `<¦…¦>`
  (`<|im_end|>` → `<¦im_end¦>`). This is stricter than the reference servers on purpose, for
  prompt-injection safety, so prompts differ from the references only for caller text that
  contains such strings. The prompt template's own special tokens are unaffected.

## `--decision-dump FILE`

```bash
mkdir -m 700 -p ~/parity
cascadia run /models/kev-9b-int8 --engine qwen35 --device CPU --api 127.0.0.1:8000 --decision-dump ~/parity/kev-passes.jsonl
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
  tightened to 0600. The path must be absent or a regular file this user owns with a single
  link: a symlink, FIFO, device or hard-linked file is refused. A failure to open it stops
  startup. Keep it in a directory only you can write (not a shared `/tmp`), and bind the API to
  loopback while it is on: the file grows by every pass of every request.
- `--decision-dump-max-bytes BYTES` closes the dump for good once a request's lines would take
  the file past BYTES: that request and every later one are not written, even one that would
  still fit, so the file is always a prefix of the run. A failed write (a full disk) closes it
  the same way and cuts the file back to its last whole line. Each case logs a warning once. The
  default is unlimited. Lines are written off the API's async workers, so a large dump does not
  stall chat; the write holds the request's concurrency permit until it ends, even if the client
  disconnects first.
- It is a startup flag only; there is no environment variable for it.
- It takes effect only when a decision model is loaded. Otherwise the server logs that
  `--decision-dump` is ignored and opens no file.

## Build a decision tree

A decision tree is an int8 shard tree plus `decision.json` and the head assets under `decision/`.
`tools/decision_export.py` builds one from Intel's stock int8 OpenVINO IR and the model family's
fine-tuned files, one family per run (`python tools/decision_export.py --help` lists every flag).
`STOCK_INT8_IR` is a local copy of the stock IR (for example `OpenVINO/Qwen3.5-9B-int8-ov`), and
the other upper-case names are the family's downloaded files:

```bash
# JEV: the stock IR plus JEV's verbalizer LoRA adapter.
python tools/decision_export.py jev --base-ir STOCK_INT8_IR \
  --adapter-dir JEV/adapter_vllm --out /models/jev-9b-int8

# Kev: the Base checkpoint, Kev's LoRA and its pointer head (--weights and --lora are required).
python tools/decision_export.py kev --base-ir STOCK_INT8_IR \
  --weights QWEN35_9B_BASE --lora KEV/adapter_model.safetensors \
  --head-pt KEV/head.pt --out /models/kev-9b-int8

# Clef: the Clef checkpoint and its joint schema head.
python tools/decision_export.py clef --base-ir STOCK_INT8_IR \
  --clef-dir CLEF_DIR --out /models/clef-flash-int8
```

- `--stages N` sets the stage cut count: 8 for a single box, one per rank for a pipeline.
- JEV takes `--weights` only for a fine-tuned backbone that does not already hold the adapter
  merged, stated with `--weights-not-merged` (check with `tools/decision/backbone_diff.py`).
- Kev ships the `tokenizer.json` beside `head.pt`, not the Base checkpoint's (`--tokenizer FILE`
  overrides it for any family).
- Kev and Clef trees are exported with the final hidden-state output their heads read; JEV's
  verbalizer reads only the logits.
- A LoRA adapter is grafted as plain linear branches only. The exporter and
  `tools/decision/reference_stream.py` refuse an adapter with embedding LoRA keys
  (`lora_embedding_A/B`), any non-LoRA tensor, a non-empty `modules_to_save` in
  `adapter_config.json` (as well as DoRA, rank/alpha patterns and `lora_bias`), and the export
  fails unless every adapter module is grafted exactly once.
- Kev refuses a `head.pt` whose head the server's pointer readout cannot run: `option_isolation`
  or `special_embeddings` set, `weights` other than `lora`, a `head_dim` that is not the q/k row
  count, or head tensors other than `q`/`k` weight and bias.
- The manifest is written last, so a tree whose export stopped part way is not loadable.

## Pipeline reply timeouts

In pipeline mode (`--total` > 1) rank 0 waits for each reply from the next rank for the activation
timeout (`CASCADIA_ACTIVATION_TIMEOUT_SECS`, default 60 s): the TOKEN after each FORWARD, and for a
decision pass the RESET_ACK, DECIDE_ACK and DECISION. When a reply is later than that, rank 0 fails
that task, and the late reply still arrives afterwards. What happens next depends on where the
wait stopped:

- **Before the reply's first byte** (the usual case: the next rank is still computing). The link
  is still frame-aligned. Every reply header carries the task epoch, so the next task skips the
  late frames of earlier tasks until it reads its own reply, then carries on normally. The skip is
  bounded: at most 16 stale frames and 32 MiB, within that task's own timeout. The log shows
  `pipeline reply timed out; its late reply will be skipped`, then one
  `skipped a late reply of an earlier task` per frame skipped.
- **Mid-frame, mid-request, or on a reply with no task tag** (CAPTURE_ACK, used only with KV
  coordination), or after any other wire error on the link. Rank 0 can no longer find frame
  boundaries, so it marks the link desynced and logs one error,
  `qwen36 pipeline link desynced after …; restart the pipeline`. Every later task, chat and
  decisions alike, fails at once with that message, and nothing more is written to the link.
  Restart both ranks to recover.
- **The startup handshake** (HELLO). A rank 1 that is still loading can answer after the timeout.
  The first request then fails, and the next one waits for that HELLO's reply instead of sending a
  second HELLO. A long load therefore costs one failed request rather than a restart. The JEV-27B
  runbook below still raises the timeout, so that no request fails.

A middle rank applies the same rules to its own downstream link. The wire format is unchanged, so
ranks of older builds work with this one as before. No metric reports the desynced state yet;
watch for the error line.

## Limitations

- **Text only.** Images in either format are a 400 (`images`, `videos`, OpenAI `input_image`
  parts, and JEV image parts in a list `state`).
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

### JEV-9B

**Model:** [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) (verbalizer LoRA on Qwen3.5-9B),
served on Intel's stock [OpenVINO/Qwen3.5-9B-int8-ov](https://huggingface.co/OpenVINO/Qwen3.5-9B-int8-ov)
with the adapter grafted at export. No `--weights`: the backbone is stock.

```bash
hf download OpenVINO/Qwen3.5-9B-int8-ov --include "openvino_language_model.*" --include "openvino_text_embeddings_model.*" \
    --include config.json --include "tokenizer*.json" --include generation_config.json --include "chat_template*" \
    --local-dir ~/models/qwen35-9b-int8
hf download autotrust/JEV-9B --include "adapter_vllm/*" --include calibration.json --include tokenizer.json \
    --include tokenizer_config.json --local-dir ~/models/jev9b
python tools/decision_export.py jev --base-ir ~/models/qwen35-9b-int8 --adapter-dir ~/models/jev9b/adapter_vllm \
    --out ~/trees/jev9b-int8           # 8 stages; 201 LoRA branches grafted (25 per stage, 26 on the last)
cascadia run ~/trees/jev9b-int8 --engine qwen35 --device CPU --api 127.0.0.1:8080
```

**Measured parity** (single box, CPU, Intel Core Ultra 7 258V, 32 GB, OpenVINO GenAI 2026.4.0; the 50
records / 82 questions of the spike set):

| Check | Result |
|---|---|
| Prompt layout vs serve_decide + `decision_head.json` (82 + 4 synthetic) | 86/86 |
| Prompts byte-equal to the spike's `jev.py` (choices sorted) | 82/82 |
| Answers vs the spike's saved full-precision reference | 82/82 |
| int8 vs streamed fp32 reference on the server's own prompts | **82/82** (choice 33/33, predicate 24/24, score 25/25), max \|Δp\| 0.0175, no flips |
| `/v1/decisions` vs `/v1/systemone` | 82/82 identical |

The warmup measured the prefix-cache snapshot model at 53.5 MB fixed + 64 KiB per token.

**Apple Silicon:** OpenVINO's ARM CPU plugin expands quantized weights when it compiles, so one process
serving all 8 stages needs far more than the 8.3 GB of int8 weights (it went to ~37 GB of swap on a 36 GB
M4 Max). Serve the 9B models on x86 (Intel) hosts; on a Mac, only one stage per process fits.

### Kev-9B

**Model:** [jaredpalmer/kev-9b](https://huggingface.co/jaredpalmer/kev-9b) (LoRA + pointer head on
Qwen3.5-9B-Base). The base is not the instruct model that Intel's int8 IR was built from, so the export
swaps the base weights into that IR (`--weights`, checked per stage by `check_swap`), grafts the adapter,
ships Kev's own tokenizer and swaps the text embeddings. The repo's `adapter_config.json` is used as is.

```bash
hf download Qwen/Qwen3.5-9B-Base --revision 68c46c4b3498877f3ef123c856ecfde50c39f404 --local-dir ~/models/qwen35-9b-base
hf download jaredpalmer/kev-9b adapter_model.safetensors adapter_config.json head.pt tokenizer.json --local-dir ~/models/kev9b
python tools/decision_export.py kev --base-ir ~/models/qwen35-9b-int8 --weights ~/models/qwen35-9b-base \
    --lora ~/models/kev9b/adapter_model.safetensors --head-pt ~/models/kev9b/head.pt \
    --out ~/trees/kev9b-int8           # 8 stages; 248 LoRA branches grafted (31 per stage)
cascadia run ~/trees/kev9b-int8 --engine qwen35 --device CPU --api 127.0.0.1:8080
```

**Measured parity** (same host and set as JEV-9B):

| Check | Single box (8 stages, 1 process) | Pipeline (2 stages, 2 processes) |
|---|---|---|
| Passes vs `kev.questions` (82 + 4 synthetic) | 86/86 | 86/86 |
| Answers vs the spike's saved full-precision reference | 82/82 | 82/82 |
| int8 vs streamed fp32 reference | **81/82** (choice 33/33, predicate 24/24, score 24/25), max \|Δp\| 0.034; the one flip, `r43.severity`, has a reference margin of 0.007 | **82/82**, max \|Δp\| 0.029, no flips |
| `/v1/decisions` vs `/v1/systemone` | 62/62 identical | — |

The fp32 and route rows were measured on trees exported after the `q_asym` fix (see Clef 27B); before it,
the same checks gave max \|Δp\| 0.069 (single box) and 0.083 (pipeline). The prompt-layout row does not
depend on the weights; the spike-answers row predates the fix.

The route check covers the 62 questions whose state is a string. Kev and Clef render an object state
structurally, while an OpenAI `input` is always text, so an object state sent as `json.dumps` text renders
differently by design. `openai_equivalence.py --all-states` is for JEV only.

**Prefix reuse.** A 9,001-token state with 5 predicate questions (45,095 tokens without reuse) prefills
the state once (about 395 s on this CPU), then resumes the snapshot for every later pass
(`warm_prefix=9002`). With `max_total_tokens` lowered to 20,000:

| Server | Result |
|---|---|
| Default prefix cache | 200, answers identical to the unlimited run (the later passes must resume the snapshot) |
| `--prefix-cache-gb 0.001` | 413 in 0.02 s: `needs 45095 prefill tokens without prefix reuse (the shared prefix cannot fit the prefix cache); the per-request limit is 20000` |
| `--prefix-cache-gb 0` | 413 in 0.02 s: `needs 45095 prefill tokens; the per-request limit is 20000` |
| Pipeline (no prefix discount) | 413 in 0.02 s, the plain message; nothing reaches the dump |

### Clef-flash

**Model:** [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (a full fine-tune of
Qwen3.5-9B, all 32 layers changed, plus a joint head over every position's final hidden state). The
export swaps all of Clef's weights into Intel's int8 IR (`--weights`) and installs the joint head from
the same directory (`--clef-dir`).

```bash
hf download Cloudflare/clef-flash --local-dir ~/models/clef-flash      # 17.8 GB (BF16)
python tools/decision_export.py clef --base-ir ~/models/qwen35-9b-int8 --weights ~/models/clef-flash \
    --clef-dir ~/models/clef-flash --out ~/trees/clef-flash-int8
cascadia run ~/trees/clef-flash-int8 --engine qwen35 --device CPU --api 127.0.0.1:8080
```

**Measured parity** (same host and set as JEV-9B; one joint pass per record, 50 passes for 82 questions):

| Check | Single box (8 stages, 1 process) | Pipeline (2 stages, 2 processes) |
|---|---|---|
| Records vs the spike's `encode_record` (50 + 4 synthetic) | 54/54 | 54/54 |
| Answers vs the spike's saved full-precision reference | 82/82 | 82/82 |
| int8 vs streamed fp32 reference | **81/82** (choice 33/33, predicate 24/24, score 24/25), max \|Δp\| 0.039; the one flip, `r07.severity`, has a reference margin of 0.012 | **81/82**, max \|Δp\| 0.036; the same `r07.severity` near-tie |
| `/v1/decisions` vs `/v1/systemone` (string states) | 62/62 identical | — |

As for Kev-9B, the fp32 and route rows come from trees exported after the `q_asym` fix (before it: max
\|Δp\| 0.072 single box, 0.069 pipeline, no flips); the spike-answers row predates the fix. The same
request sent twice returns 200 both times with identical answers (joint passes always prefill cold). For
comparison, the spike's int4 export on Apple Silicon agreed on 79/82. `r07.severity` is a genuine
near-tie: the reference gives its two leading options 0.419 and 0.407.

### JEV-27B

**Model:** [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) (verbalizer LoRA on Qwen3.8-27B),
served across two machines on Intel's stock
[OpenVINO/Qwen3.8-27B-int8-ov](https://huggingface.co/OpenVINO/Qwen3.8-27B-int8-ov). The 27B models were
never measured in the spike, so these numbers are the first ones.

**Backbone check.** `backbone_diff.py` against `Qwen/Qwen3.8-27B` reports `VERDICT: STOCK`: all 851
backbone tensors are identical, `lm_head` included. The export therefore grafts the adapter onto the
stock IR with no `--weights`, and the reference applies the adapter (`--lora`) on JEV-27B's own BF16
shards.

```bash
hf download OpenVINO/Qwen3.8-27B-int8-ov --local-dir ~/models/qwen38-27b-int8
hf download autotrust/JEV-27B --include "adapter_vllm/*" --include calibration.json --include tokenizer.json \
    --include tokenizer_config.json --include config.json --local-dir ~/models/jev27b
python tools/decision_export.py jev --base-ir ~/models/qwen38-27b-int8 --adapter-dir ~/models/jev27b/adapter_vllm \
    --stages 2 --out ~/trees/jev27b-int8   # 401 LoRA branches grafted (200 + 201)
# host B (rank 1), then host A (rank 0, API on loopback)
cascadia worker --model ~/trees/jev27b-int8 --engine qwen35 --device CPU --rank 1 --total 2 --listen 0.0.0.0:7801
CASCADIA_ACTIVATION_TIMEOUT_SECS=1800 cascadia worker --model ~/trees/jev27b-int8 --engine qwen35 --device CPU \
    --rank 0 --total 2 --next HOST_B:7801 --api 127.0.0.1:8080
```

Each host needs the whole tree at the same path. The export is deterministic: two hosts that each
downloaded from Hugging Face and exported with the same commit produced byte-identical trees (13/13
files by SHA-256), so a slow link between the hosts never has to carry the 28 GB tree.

**Measured parity** (rank 0 and rank 1 on two Intel Core Ultra 7 258V / 32 GB hosts, CPU, OpenVINO GenAI
2026.4.0, linked over a relayed Tailscale path of about 0.4 MB/s; the 50 records / 82 questions of the
spike set):

| Check | Result |
|---|---|
| Prompt layout vs serve_decide + `decision_head.json` (82 + 4 synthetic) | 86/86 |
| int8 (2 machines) vs streamed fp32 reference | **82/82** (choice 33/33, predicate 24/24, score 25/25), max \|Δp\| 0.0187, no flips |
| `/v1/decisions` vs `/v1/systemone` (`--all-states`) | 82/82 identical |

**Reference memory.** `reference_stream.py` keeps one fp32 decoder layer resident (about 8 GB private
for 27B). On a 32 GB host it still cannot share memory with a running rank 0 (13–25 GB), so run the
reference with the ranks stopped, then restart them for the parity verdict.

### Clef 27B

**Model:** [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) (Qwen3.8-27B with a joint head over
every position's final hidden state), served across two machines like JEV-27B. Against stock Qwen3.8-27B,
`backbone_diff.py` finds layers 0–39, the embeddings, the norms and `lm_head` identical; only the
projection weights of layers 40–63 are fine-tuned. The export swaps the whole checkpoint into Intel's
int8 IR (`--weights`), as for Clef-flash.

```bash
hf download Cloudflare/clef --local-dir ~/models/clef      # 55 GB (BF16)
python tools/decision_export.py clef --base-ir ~/models/qwen38-27b-int8 --weights ~/models/clef \
    --clef-dir ~/models/clef --stages 2 --out ~/trees/clef-int8
```

The export holds a whole stage's new int8 constants (12–15 GB) and needs about 16 GB free; it is the one
export above the 7 GB cap. Serve it with the two `cascadia worker` commands shown for JEV-27B.

**Measured parity** (same two hosts and set as JEV-27B; one joint pass per record, 50 passes for 82
questions):

| Check | Result |
|---|---|
| Records vs the spike's `encode_record` (50 + 4 synthetic) | 54/54 |
| int8 (2 machines) vs streamed fp32 reference | **81/82** (choice 32/33, predicate 24/24, score 25/25), max \|Δp\| 0.067; the one flip, `r41.team`, has a reference margin of 0.048 |
| `/v1/decisions` vs `/v1/systemone` (string states) | 62/62 identical |

**The swap must keep zero in each row's int8 range.** The first Clef 27B tree measured 78/82 with two
confident flips and max |Δp| 0.30. The cause was the swap's quantizer: it scaled each row over
[min, max], while NNCF (which built Intel's IR) uses [min(w, 0), max(w, 0)]. The two agree on every
projection row, but a 4-tap linear-attention `conv1d` channel is single-signed in about a quarter of
rows, and those rows saturated. That garbled the convolution in all 48 linear-attention layers of every
`--weights` tree. With the fix, the swapped `conv1d` matches Intel's constants in at least 99.9% of
entries.
