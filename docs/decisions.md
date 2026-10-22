# Decisions API

Cascadia can serve **decision models**: fine-tuned Qwen3.5-family checkpoints that answer typed
questions (yes/no, pick one, rate on a scale) about a piece of text by reading probabilities off a
single prefill pass. They never generate text. A model directory turns into a decision server when
it carries a `decision.json` next to the shard tree:

```bash
cascadia run /models/kev-9b-int8 --engine qwen35 --device CPU --api 127.0.0.1:8000
```

The server logs `decision.json loaded; /v1/decisions (OpenAI format) and /v1/systemone will be
served` before the model load, and once the API is up it exposes two routes over the same core:

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
- `strategy` (optional, top level): how a JEV choice with more options than its trained slots is
  answered, `"auto"` (the default), `"single"`, `"tournament"` or `"permute"`. See
  [JEV choices past the trained slots](#jev-choices-past-the-trained-slots).

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

`type` is `invalid_request_error` for 400, 413 and 415, and `server_error` for 501 and 503. A body that
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
  answer echoes the same JSON number: an integer with every digit, a float in serde_json's shortest
  form (`1E2` comes back as `100.0`, `2.50` as `2.5`), which parses to the same value.
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
- `strategy` (optional): as on the OpenAI route, `"auto"` (the default; `null` too), `"single"`,
  `"tournament"` or `"permute"`. See
  [JEV choices past the trained slots](#jev-choices-past-the-trained-slots).
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

Both routes share these statuses. Request-body errors differ by route (see each route's Errors
section): the OpenAI route answers all of them in its envelope, `/v1/systemone` with the
framework's plain-text body.

| Status | When |
|---|---|
| 200 | Every question answered. |
| 400 | A body that is not valid JSON; on the OpenAI route, also a body that does not match the request schema (a missing `questions`, an unknown question `type`, a repeated key). Validation: no questions or more than `max_questions`; an empty question name, or (OpenAI format) a duplicate question name or choice value; a choice with no options or more than the model allows; a score with fewer levels than the model allows (2 for JEV, 1 for Kev and Clef) or more; images; an image part, or a text part without a string `text`, in a list `state` (JEV); a request the tokenizer cannot encode; Clef's option cap (see [Limits](#limits)); a `strategy` other than `"auto"` on Kev or Clef (`strategy applies to JEV only`); on the OpenAI route, a `strategy` that is none of the four values. |
| 413 | The body is over `--api-max-body-mb`. Too long: the state is more than `max_total_tokens × 8` bytes (checked before tokenizing); Kev's whole rendered text (state, instructions, options and levels) is more than `max_total_tokens × 8` bytes; Kev's rendered state is more than 524,288 bytes (65,536 × 8, checked before tokenizing); one Kev question's instructions and option texts render to more than 8 bytes per token the pass has left after the state (`max_tokens` minus the state's tokens, checked before tokenizing); Kev's state is more than 65,536 tokens (`SERVE_MAX_STATE`, `<state>` included) or reaches `max_tokens`; one pass is longer than `max_tokens` (JEV and Kev; Clef truncates the state instead); Clef's schema alone fills the joint pass, leaving no room for a non-empty state; the request's prefill work is more than `max_total_tokens` (JEV stops tokenizing at the first pass that takes it over: `needs at least N`; a JEV tournament's final counts at its worst case before any pass runs, see [JEV choices past the trained slots](#jev-choices-past-the-trained-slots)); a Kev request that needs the shared-state discount when the prefix cache can never hold that state (`this request needs N prefill tokens without prefix reuse (the shared prefix cannot fit the prefix cache); the per-request limit is M`, refused before any prefill once the engine knows the snapshot's size, which its warmup measures at startup; see [Limits](#limits) for when the first such request can still spend one pass); or the engine reports the prompt too long. |
| 415 | The request lacks `content-type: application/json`. |
| 422 | `/v1/systemone` only: valid JSON that does not match the request schema, a `strategy` that is none of the four values included (the OpenAI route answers 400 instead). |
| 501 | The model serves no decisions: no `decision.json` (or one that fails its load checks; see the startup log), an engine other than `qwen35`, the engine has no decision head (`decision/` missing on the single box or the pipeline's last rank), the tree was exported without `--final-hidden` (Kev and Clef), or a pipeline rank cannot serve decision passes (an older build, rank 0 or the last rank without `config.json` `vocab_size`, or the last rank without `decision/`; a middle rank only relays and needs neither). Rank 0 refuses the pipeline cases before any prefill. |
| 503 | The server is at capacity (`engine at capacity; retry after current requests complete`): every decision permit is taken (`--max-decision-requests`, default 16), or a Kev pass that must resume its shared state from the prefix cache found the snapshot evicted by other requests (see [Limits](#limits)); retrying can succeed. Any other engine error is `decision engine error`, including non-finite logits or probabilities (never a 200 with `null`s) and a task the engine failed (see [Engine failures](#engine-failures)). Decisions have their own admission pool: chat requests never take its permits, and decisions never take chat's. Chat can still cause this 503 by evicting a Kev snapshot. |

Error messages are deliberately terse:

- **Tokenizer failures** return the fixed message `request could not be tokenized`. The
  tokenizer's own text can carry vocabulary or file-path detail, so it is logged, not returned.
- **Validation messages** (for example `image parts in state are not supported (text-only server)`,
  or Clef's `N options across all questions; the limit is 4096`) are returned as-is: they name a
  problem the caller can fix.
- **Engine errors** return a fixed message per category (`decision engine error`, `engine at
  capacity or unavailable`, `decision prompt too long`). A request that finds every decision
  permit taken gets `engine at capacity; retry after current requests complete`. The engine's text (paths, OpenVINO
  messages) is logged at `warn` and never returned.

## Limits

| Limit | Source | Default |
|---|---|---|
| Questions per request | `max_questions` in `decision.json` | 32 (Clef: at most 256, the joint head's cap) |
| Tokens in one pass | `max_tokens` in `decision.json` | required field |
| Prefill tokens per request | `max_total_tokens` in `decision.json` | 4 × `max_tokens` |
| Options per choice | per model family | JEV: as many as the profile's `choice_labels`, at most 256 (serve_decide's `MAX_OPTIONS`); a profile without them, the choice slot count (16 for the shipped JEV profiles); Kev: 255 (Kev's `MAX_OPTIONS`); Clef: 256 |
| Options per request (Clef) | engine readout cap | 4096 across all questions, because Clef answers them in one pass |
| Tokens in the joint pass (Clef) | `MAX_JOINT_PROMPT` and the head's input limit | the smaller of `max_tokens`, 16,384 and 320 MiB ÷ (4 × width): 16,384 for both Clef-flash (width 4096) and Clef-27B (width 5120) |
| Levels per score | per model family | JEV: 2 to 256; Clef: 1 to 256; Kev: 1 to 255 |
| State tokens (Kev) | Kev's `SERVE_MAX_STATE` | 65,536, `<state>` included; only the whole pass may reach `max_tokens`. The rendered state is also capped at 8 bytes per token (524,288 bytes) before tokenizing |
| Concurrent decision requests | `--max-decision-requests` | 16 (1 to 240), a pool separate from chat's (16 unless `CASCADIA_API_MAX_CONCURRENT` sets it); the two together at most 256; one more is a 503 |
| Concurrent tokenizations | fixed | min(4, CPU cores), separate from the decision permits |
| Request body | `--api-max-body-mb` | 1 MiB |

- **`decision.json` is checked at load.** `max_tokens`, `max_total_tokens` and `max_questions`
  must be at least 1, a JEV choice slot range at least 1 wide, and Clef's `max_total_tokens` must
  cover one full joint pass. `max_tokens` above the engine's 131,072-token decision prompt cap,
  and Clef's `max_questions` above 256, are clamped with a warning, so a request over the
  engine's own limits is a 400 or 413 rather than a 503. Calibration values must keep
  `(logit + bias) / temperature` finite: every JEV and Kev `temperature` in [0.001, 1000] and
  every JEV `bias` in [-10,000, 10,000] (shipped profiles use temperatures near 1 and biases of a
  few tenths). A profile that fails these checks
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
  pipeline whose last rank predates this limit, that rank refuses a Clef-27B readout over 13,107
  tokens only after the chain prefill, because rank 0 cannot tell from HELLO which limit the last
  rank holds. A last rank from commit fd87d505 onward refuses it with the same 413
  (`decision_too_long`); an older one fails it in the native shim, which is a 503. The wire
  format is unchanged. The request-level `max_total_tokens` byte pre-check still applies to every
  family.
- **JEV and Kev run one prefill pass per question**, so `max_questions` bounds the engine work of
  one request. A JEV choice past the trained slots runs its plan's passes instead (up to 17, see
  [JEV choices past the trained slots](#jev-choices-past-the-trained-slots)); every one of them
  counts toward `max_total_tokens`. Clef runs one pass for the whole request.
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
- **A separate admission pool.** Decision requests take permits from a pool of their own, sized
  by `--max-decision-requests N` on `cascadia run` and `cascadia worker` (default 16). Chat
  requests keep their own pool of C: 16, unless the `CASCADIA_API_MAX_CONCURRENT` environment
  variable sets it. Decisions and chat never take each other's permits, so a burst of decisions
  never uses up chat's permits, and a burst of chat never uses up the decisions'. The server as a
  whole admits up to C chat + N decision requests at once: 32 by default, where both used to
  share 16. Each one holds its body (up to `--api-max-body-mb`) and, for a decision, its tokenized
  passes, so size memory for C + N. N runs from 1 to 240, and
  anything else is a startup error that names the limit. The reason: every admitted request can
  have a pass waiting in the engine's 256-task pending queue, and C + N must fit there.
  Otherwise decisions would fill the queue and chat would get `QueueFull` 503s. So with
  decisions enabled, a server whose C + N exceeds 256 (a raised C with a large N) refuses to
  start, naming both knobs. A decision
  request that finds no permit free gets 503 (`engine at capacity; retry after current requests
  complete`) at once rather than queueing. A request holds its permit until its last pass and
  tokenization have finished, even if the client disconnects first. Decision requests count in
  the request total, the in-flight gauge and the `capacity` rejection metric like chat requests.
- **The pool sets admission, not parallelism.** The engine still runs one pass at a time, for
  decisions and chat alike. A larger pool lets more requests wait their turn instead of getting a
  503, at the cost of latency for each; a smaller one refuses sooner.
- **Kev and the prefix cache.** Every admitted Kev request snapshots its state into the one prefix
  cache. Chat snapshots its turn boundaries into the same cache. So more admitted Kev requests
  evict each other's snapshots more often, and so does a burst of chat. A request whose snapshot
  was evicted stops with the capacity 503 (see above), so chat traffic can still cause decision
  503s. Raise `--max-decision-requests` together with `--prefix-cache-gb`, or keep it small.
- **Dashboard.** `/api/stats` `max_concurrent` is chat's pool only (C), while
  `requests_in_flight` counts chat and decision requests together. It can therefore read up to
  C + N against a ceiling of C.
- **Bounded tokenizer memory.** Tokenizing a long state takes a few hundred MB of transient memory,
  so at most min(4, CPU cores) decision requests tokenize at once; the others wait their turn
  (holding their decision permit) rather than failing. Kev also refuses a state that renders to more than
  8 bytes per allowed token (524,288 bytes) before tokenizing it, and a question whose
  instructions and option texts render to more than 8 bytes per token the pass has left after the
  state; honest text tokenizes at about 4 bytes per token.

## JEV choices past the trained slots

JEV's head has a fixed number of trained choice slots (16, `A`..`P`, for the shipped JEV models).
A choice question within them is one pass, exactly as before, whatever `strategy` says. A choice
with more options, up to the profile's `choice_labels` (at most 256, written by the exporter), is
answered the way the reference server answers it (serve_decide `s1_dist`), by one of three plans:

| `strategy` | Passes | What runs |
|---|---|---|
| `single` | 1 | One pass over all n options, labelled `A`..`P`, then `Q`..`Z`, `AA`, ... |
| `tournament` | ⌈n/16⌉ + 1 | Contiguous groups of at most 16 options (⌈n/16⌉ groups, the first n mod ⌈n/16⌉ one larger), then a final over 16: each group's winner and the best of the rest by in-group probability. A finalist gets its final probability times the finalists' joint in-group mass; any other option keeps its in-group probability scaled by its group's share of the final. The whole is normalised. |
| `permute` | 4 | The options in request order, then in three orders drawn by CPython's `random.Random(zlib.crc32(question))` (`rng.sample`); the answer is the mean of the four passes. |

- **Choosing a plan.** The request's `strategy` (top level, both routes) picks one. `"auto"`, `null`
  or no field means the profile's default: `strategy` in the `jev` object of `decision.json`
  (`"single"`, `"tournament"` or `"permute"`; default `"single"`, the reference's default for the
  Qwen-based JEV models). The exporter does not write it; set it by hand. Kev and Clef answer any
  option count in one pass, so a `strategy` other than `"auto"` is a 400 there
  (`strategy applies to JEV only`).
- **Every pass is an ordinary JEV choice pass** over its own options, labelled by position from
  `A` (a tournament group or final of at most 16 options reads the trained slots), with the
  trained bias for the positions the trained slots cover and none beyond, and the choice
  temperature.
- **Budget.** Every pass counts toward `max_total_tokens` before any runs, and each must fit
  `max_tokens`. A tournament's final depends on the group passes, so it counts at its worst case:
  the 16 costliest option lines, labelled `A`..`P`. A line's cost is its text tokenized as the
  prompt holds it, with the space after `)` and the newline (`" {text}\n"`), not the text alone:
  Qwen's tokenizer joins that space to a following word (`Merkel` is 3 tokens alone, ` Merkel`
  1) and splits it off before digits or CJK text. A request that does not fit is a 413 with no
  pass run. With Qwen's tokenizer these line costs add up, so the real final is never longer; for
  a tokenizer where they did not, the request would stop with a 413 before the final rather than
  overspend.
- **Order.** The engine runs one pass at a time: every question's round-1 passes in request
  order, then every tournament's final. serve_decide runs a plan's passes concurrently; the
  result is the same.
- **The answer** covers all n options in request order, like any choice: `choice` is the most
  likely option (the first on a tie) and `confidence` the usual choice confidence. `usage`
  counts every pass.

Where this differs from serve_decide:

- **Options.** serve_decide takes an `options` list of strings; here they are the choice's
  criteria in request order (the OpenAI route's `choices` in order), and each option's text is
  the codec's line, `name: description` (`name` alone without a description), sanitized like all
  caller text. Groups, permutations and the worst-case final are over those lines.
- **The permute seed** is serve_decide's `req.question`: the question's `instructions` exactly
  as received, before sanitizing (a non-string value as its `json.dumps`), or the question id when
  there are none; the prompt shows the same text sanitized.
- **Arithmetic.** The combinations are serve_decide's, in f64 and in its order. Its sums (each
  pass's softmax normaliser, the tournament's `among` and final normaliser) use Python's `sum()`,
  which compensates rounding from CPython 3.12 on, where the server adds left to right; so
  probabilities can differ in the last bit or so, as for every JEV pass today, far below the 4
  decimals returned.
- **A second round** needs the tokenizer again: a tournament request holds its decision permit
  and waits for a tokenization slot once more between its rounds.

**Checking wide choices.** The parity tools take `--wide` to drive `WIDE_RECORDS`
(`tools/decision/records.py`) instead of the 50 parity records: four requests with choices of 17,
24, 40 + 100 and 256 options, each sent once per strategy (12 records, 17 questions; the `single`
requests also carry one native choice and one predicate). Start the server with `--decision-dump` and no other client, then:

```bash
python tools/decision/codec_golden.py jev --url http://127.0.0.1:8080 --dump ~/parity/jev.jsonl \
    --tokenizer ~/models/jev9b --adapter-dir ~/models/jev9b/adapter_vllm --wide     # prompts and plans
python tools/decision/parity_check.py --url http://127.0.0.1:8080 --dump ~/parity/jev.jsonl --wide   # writes jev.wide.parity.jsonl
python tools/decision/reference_stream.py jev --weights JEV_BF16 --lora ~/models/jev9b/adapter_vllm/adapter_model.safetensors \
    --tree ~/trees/jev9b-int8 --dump ~/parity/jev.wide.parity.jsonl --out ~/parity/jev.wide.ref.json --wide
python tools/decision/parity_check.py --url http://127.0.0.1:8080 --ref ~/parity/jev.wide.ref.json --wide  # the gate
```

- `codec_golden.py --wide` (JEV only, no `--answers`) checks each question ran the strategy its
  request asked for, re-derives every round-1 pass's prompt and label ids with serve_decide's
  layout and its single-token labels (`A`..`Z`, `AA`, ...; the first 16 must be the trained
  head's), with the permute orders drawn by CPython's own `random`, and reads a tournament's
  final back from its prompt: a sorted set of 16 with a member of every group, re-derived in the
  same layout. It writes `<dump>.wide.golden.jsonl`. The records always name their strategy, so
  the profile's default (`auto`) is not exercised here.
- The dump holds no logits, so the combinations are checked where probabilities exist:
  `reference_stream.py --wide` computes each pass in fp32 (the trained bias on the first 16
  positions, none beyond) and combines them with Python ports of serve_decide's formulas. A
  tournament's finalists are the reference's own, picked from its fp32 group passes: when they are
  the ones the server ran (read back from its final through the tree's `tokenizer.json`), the
  server's final is replayed; otherwise a second, small fp32 sweep runs the final over the
  reference's finalists, and a `FINALISTS` line names the options swapped. A server that picks the
  wrong finalists therefore differs from the reference. `parity_check.py --wide` then gates the
  server's answers against it.
- The gate is the 50 records' top-1 gate (`--min-agree 0.96`, `--confident 0.05`) plus a
  probability bound: `--max-dp` (default 0.05 with `--wide`) fails the run when any multi-pass
  question's max |p_server − p_ref| exceeds it, since a combination bug can keep every top-1.
  int8 JEV-9B measured max |Δp| 0.018 on CPU and 0.024 on GPU on the 50 records; the output
  prints the multi-pass and native max |dp| side by side. Over 17 questions 0.96 allows no flip;
  pass `--min-agree 0.94` to allow one near-tie.
- The wide files never replace the 50-record ones: `jev.wide.parity.jsonl` and
  `jev.wide.golden.jsonl` sit beside `jev.parity.jsonl` and `jev.golden.jsonl`.

**Measured** (JEV-9B int8, single box, 2026-10-17): `codec_golden.py --wide` matched all 62 passes
(every strategy, 17–256 options) to the reference layout and trained head. The wide gate against
the streamed fp32 reference passed 17/17 on both devices, with every tournament's finalists equal
to the reference's: max |Δp| 0.019 on CPU (native questions 0.006) and 0.048 on GPU (native
0.005), so the strategies add nothing measurable on CPU and the GPU's margin is its own precision.
The 50-record parity on the same build was unchanged (80/82 on GPU, max |Δp| 0.024). JEV-27B
across two hosts answered a 40-option choice with `single` in 69 s and with `tournament` (three
groups and a final) in 67 s, both 200 with all 40 probabilities summing to 1; a 16-option choice
took 17 s as before.

## System 2 (evaluated, not shipped)

JEV's reference server can also answer with System 2: the base model reasons over the question in
its chat template, its answer distribution `p2` is read back after the thought and blended with
System 1's `p1`. Cascadia built it (Tasks 41 to 45, with a switchable-LoRA export that ran the
base model from the same tree) and removed it in Task 53, after an A/B on hardware:

- **Setup.** typed-decisions on pawan-03's GPU (Intel Arc 140V), 8 cases, the 16 questions that
  thought.
- **Accuracy.** The blend never changed an argmax: 7/16 correct with and without it. The read-back
  `p2` put 0.975 mean mass on System 1's own argmax.
- **Calibration.** KL and Brier got worse with every `p2` weight `w`: 0.236 / 0.137 at `w = 0`
  (System 1), 0.482 / 0.290 at `w = 0.5` (the reference's mix), 3.331 / 0.575 at `w = 1`. The
  paired bootstrap of blend − System 1 gave accuracy [0, 0], KL [+0.158, +0.334] and Brier
  [+0.098, +0.204].
- **Cost.** One thinking question took 90 to 1,300 s on the batch-1 engine, and long generations
  hit the intermittent native infer exception (BUG-1, see [Engine failures](#engine-failures)).

What remains of it:

- **Requests.** `thinking`, `threshold`, `think_budget` and `return_reasoning` are ignored on both
  routes, like any field the request schemas do not know (neither refuses unknown fields), whatever
  their value. Answers carry no `thinking` object, and `usage.output_tokens` is 0.
- **Trees.** `decision_export.py` no longer takes `--switchable-lora`. A tree exported with it has
  a `lora_scale` stage input that nothing feeds, and fails the load with `switchable trees are no
  longer supported; re-export without --switchable-lora` (the engine checks `manifest.json` and
  each compiled stage, the API checks `jev.lora_scale` in `decision.json`). Re-export it without
  the flag; `trees/jev9b-int8-sw` on the test boxes is such a tree. The `jev.answer_labels`,
  `jev.answer_label_ids` and `jev.end_think_id` that trees exported in between carry are ignored,
  and the exporter no longer writes them.
- **Pipeline wire.** As before Task 41: HELLO_ACK capability bits 0 (capture_v2) and 1 (decision
  frames) only, no `lora_scale` key in the HELLO payload, and `pos = 0` in RESET and RESTORE. A
  rank built between Tasks 42 and 53 still interoperates on a baked tree: its extra bit 2 is
  dropped by the AND along the chain, and its extra HELLO key is ignored.

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
- **No BOS token.** The codecs encode prompts without a beginning-of-sequence token, as the
  Qwen3.5 references do (Qwen's tokenizer adds none). A decision model on a backbone that
  expects one (Gemma's `<bos>`, for example) would get a different prompt from its reference;
  only Qwen3.5-family trees are supported.
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

- A JEV choice pass's line also has `"strategy"` (`"native"` for a choice within the trained
  slots, else `"single"`, `"permute"` or `"tournament"`) and `"round"` (1, or 2 for a
  tournament's final), so the tools can rebuild the combination. A question's passes are written
  in the order they run: its round-1 passes with the other questions', then the finals once round
  1 has finished. Other lines are unchanged.

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
  stall chat; the write holds the request's decision permit until it ends, even if the client
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

- `--stages N` sets the stage cut count. The default, 1, is the tree for a single box: one stage
  holds the whole model, which serves faster (see [One stage or eight](#one-stage-or-eight)).
  A pipeline passes one per rank (`--stages 2` for two machines), as before. `--stages 8` cuts a
  single-box tree for a machine with too little RAM for the 1-stage export: the export then holds
  one stage at a time. Exporter peak RAM (process tree working set) measured on pawan-01, 1 stage
  against 8: JEV 8.25 against 3.1 GiB, Kev 20.3 against 5.1 GiB, Clef-flash 20.6 against 5.3 GiB.
- A 1-stage export checks its memory before it writes anything. It estimates its peak RAM from
  the base IR's int8 weight files (`openvino_language_model.bin` plus
  `openvino_text_embeddings_model.bin`, 7.40 + 0.95 = 8.35 GiB for Qwen3.5-9B): 2.5 times their
  size with `--weights` (Kev, Clef; the measured peaks are 2.43 and 2.47 times), 1.0 times without
  (JEV's graft; measured 0.99 times). That is 20.9 GiB for a 9B swap and 8.35 GiB for JEV, above
  every measured 1-stage peak. When the estimate exceeds the machine's physical RAM, the export is
  refused: `the export needs about 20.9 GiB of RAM at --stages 1 (the base IR's 8.3 GiB of int8
  weights, x2.5 for the --weights swap), but this machine has 16.0 GiB: export with --stages 8
  instead ...`. `--skip-memory-check` exports anyway. When the RAM cannot be read, nothing is
  refused. Multi-stage exports (`--stages 2` and up: pipelines, and the low-memory single box) are
  not checked, as before: only the 1-stage peaks were measured, and multi-stage peaks do not scale
  from them (Clef-27B at `--stages 2` needs about 16 GB). The check uses only the Python standard
  library (`sysconf` on Linux and macOS, `GlobalMemoryStatusEx` on
  Windows).
- A 1-stage tree also loads slower (warm load to `API serving` 21-24 s against 7-13 s for 8
  stages), and a fault recovery leaks the whole model, so its engine recompiles at most once (see
  [Engine failures](#engine-failures)).
- `--max-tokens N` sets `decision.json` `max_tokens` for any family. The default is 32,768 for
  JEV, 73,728 for Kev and 16,384 for Clef, each lowered to the tree's `max_position_embeddings`
  (from the `config.json` the tree ships: the one in `--weights`, which for Clef defaults to
  `--clef-dir`, when it has one, else the base IR's; with neither, the default stands). An `N`
  above that window, or above the server's own cap, is refused rather than lowered. The cap is
  131,072 for JEV and Kev; for Clef it is the joint pass bound at the base IR's width (16,384
  for both Clef models, 10,240 at width 8192; see [Limits](#limits)).
- The exporter refuses calibration the server would refuse at load (see [Limits](#limits)): a JEV
  or Kev `temperature` outside [0.001, 1000] or a JEV `bias` outside [-10,000, 10,000]. The error
  names the field, such as `jev.temperature.choice` or `jev.bias[3]`.
- JEV takes `--weights` only for a fine-tuned backbone that does not already hold the adapter
  merged, stated with `--weights-not-merged` (check with `tools/decision/backbone_diff.py`).
- Kev ships the `tokenizer.json` beside `head.pt`, not the Base checkpoint's (`--tokenizer FILE`
  overrides it for any family).
- JEV writes its choice labels to the `jev` object of `decision.json`: `choice_labels` (strings)
  and `choice_label_ids` (their token ids), at most 256 of each. The labels are `A`..`Z`, then
  `AA`, `AB`, ..., `ZZ`, keeping each one that is a single token of the `tokenizer.json` the tree
  ships and that stays that token in an option line (`x\n{label}) y`). The first labels must be
  the trained choice verbalizers (`A`..`P` for JEV-9B), so the export is refused when their ids
  differ from the `verbalizer_ids` of the choice slots; the error names both lists. A JEV export
  with no `tokenizer.json` (from `--tokenizer`, `--weights` or the base IR) is refused. Both
  fields are optional: a tree exported without them still loads.
- Kev and Clef trees are exported with the final hidden-state output their heads read; JEV's
  verbalizer reads only the logits.
- Clef takes no `--lora` or `--lora-scale`: `--clef-dir` is the full checkpoint its joint head
  was trained on, and an adapter grafted on top would move the hidden states the head reads.
- A LoRA adapter is grafted as plain linear branches only. The exporter and
  `tools/decision/reference_stream.py` refuse an adapter with embedding LoRA keys
  (`lora_embedding_A/B`), any non-LoRA tensor, a non-empty `modules_to_save` in
  `adapter_config.json` (as well as DoRA, rank/alpha patterns and `lora_bias`), and the export
  fails unless every adapter module is grafted exactly once.
- Kev refuses a `head.pt` whose head the server's pointer readout cannot run: `option_isolation`
  or `special_embeddings` set, `weights` other than `lora`, a `head_dim` that is not an integer
  equal to the q/k row count, no saved head (`head: None`), or head tensors other than `q`/`k`
  weight and bias.
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
  `qwen36 pipeline link desynced after …; restart the pipeline`. Nothing more is written to the
  link. A lost link ends the same way: when rank 1 dies, rank 0's next reply fails with a wire
  error. The desync also fails the engine with that reason, so rank 0 exits for its supervisor
  (see [Engine failures](#engine-failures)); until it exits, every task, chat and decisions
  alike, fails at once with `qwen36 engine failed: <the desync message>; restart the server`.
  With the relays under restart-on-exit supervisors too, the pipeline reconnects on its own.
  Measured on pawan-03 (two ranks on one host, kev9b-int8-2st, warm model cache, a test
  supervisor that restarts a rank 2 s after it exits; the systemd unit's `RestartSec=5` adds
  about 3 s to each restart):
  - Rank 0 killed: the pipeline served again about 4 s after the kill.
  - Rank 1 killed: about 12 s. Rank 1 restarted within 3 s. Rank 0 failed its next exchange,
    answered 503 while it gave in-flight responses the 3 s grace, then exited, restarted and
    reconnected. Before this change, rank 0 stayed up, desynced, and answered 503 for 15 minutes
    until a manual restart.

  Each restart counts against the supervisor's start limit (the systemd unit allows 3 starts per
  minute). If one rank cannot come back, the ranks around it keep failing their link and exiting
  until they hit that limit too, and then they stay stopped. Fix the dead rank, then reset and
  start the units (`systemctl reset-failed` and `systemctl start`).
- **The startup handshake** (HELLO). A rank 1 that is still loading can answer after the timeout.
  The first request then fails, and the next one waits for that HELLO's reply instead of sending a
  second HELLO. In the served 2-rank topology a long load therefore costs one failed request
  rather than a restart. The JEV-27B runbook below still raises the timeout, so that no request
  fails.

These guarantees are for the served 2-rank topology. In a longer chain:

- A slow-loading rank below a middle rank makes the middle's own HELLO forward time out. The
  middle then answers rank 0 with a handshake refusal, which poisons rank 0 until restart, so
  there a long load costs a restart.
- A middle rank drains late replies on its own downstream link by the same rules. A desynced
  middle cannot pass the restart message upstream, but it fails its engine and exits for its
  supervisor like any relay whose engine failed (below), so rank 0 sees its link close, desyncs,
  and exits too. The desync reason is in the middle rank's log.

The wire format is unchanged, so ranks of older builds work with this one as before. Every rank
of a chain must be built with the same features: a chain that mixes builds with and without KV
coordination (the `kv_coord` feature) is unsupported. No metric reports the desynced state yet;
watch for the error line.

**Measured on hardware** (JEV-27B across two hosts, rank 0 with a 40 s timeout): rank 1 was frozen
for 70 s during a 1,967-token request. That request returned 503, rank 0 logged the late TOKEN
reply as skipped, and the next task logged `skipped a late reply of an earlier task` (epoch 3,
then its own epoch 4). The two requests after it returned 200 with answers identical to a run
without the timeout, in normal time.

## Engine failures

The qwen35 engine (`cascadia_engine_openvino::qwen36`) can fail a task in a native call. Chat and
decision tasks share this path, so what follows holds for both.

**The error comes first.** The engine returns a failed task's error chunk at once and scrubs the
stage requests (`reset_all`, a native `recreate_request` per stage after a warm turn) when the
next task is admitted. A scrub that blocked used to hold the error, the request and the engine.
A decision pass that fails is the 503 `decision engine error` (or the capacity 503 or 413 an
engine refusal maps to); chat gets its 5xx. On a single box, a native infer failure is first
retried once (below), so its 5xx now appears only when the retry fails too or the recovery fails
the engine.

**The engine's line.** Each failed task logs one `warn` line, `qwen36 task failed`: the task id,
`phase` (`prefill`, `decode` or `head`), `prompt`, `warm_prefix` (prompt tokens resumed from the
prefix cache, 0 for a cold prefill), `generated` and the reason. It covers every failure site:
single-box prefill, decode and decision head, and the pipeline head's prefill, decode and
decision. The older per-site lines (`decode failed`, `prefill failed`, ...) remain. The error
chunk carries the same `warm_prefix` (`Chunk::warm_prefix`, additive, omitted when unknown). The
engine's text is logged and never returned to the client.

### Retry after a failed infer

On a single box (no pipeline), a task whose native infer fails during its prefill or its decode is
retried once instead of failed. These are the failures that start the recovery below (OpenVINO's
`unknown C++ exception in infer`, BUG-1, is one). The client sees only the extra latency and no
chunk of the failed attempt. A decision pass re-runs from the start, so
its answer is the one an uninterrupted pass gives. A chat keeps every byte already sent, but its
continuation is not always byte-identical to an uninterrupted run: re-prefilling the generated
tokens takes a different numeric path than decoding them one at a time, so a later near-tie can
go the other way. On pawan-01 and pawan-03 (Arc 140V, OpenVINO 2026.4.0 as a fault source, 3
real faults in 240 rounds of 1600 tokens) every retried chat returned 200 with all 1600 tokens
and differed from the clean runs of the same prompt. A plain-OpenVINO replay of the same resume
on the 8-stage IRs continued token-identically across the seam and first diverged 93 and 152
tokens later, both times into a coherent alternative wording.

- **What is retried.** Only a native infer failure of the task being served. Tokenize errors,
  refusals, budget and window limits, cancels, and a failure that fails the engine itself (a
  recompiled stage's first infer) fail the task as before. Each task is retried at most once.
- **Chat and completions** resume over the tokens already generated: the retry prefills the
  prompt plus those tokens (the Option B resume path), so `max_tokens` still bounds the whole
  answer, and decode continues from there. The bytes already sent to the client are carried
  over, so no text is repeated or lost, even when an incomplete UTF-8 character was being held
  back. The final chunk's `prompt_tokens` leaves the resumed tokens out, so usage matches an
  uninterrupted run.
- **Decision passes** re-run unchanged: a pass sends nothing before its final chunk. Only a
  failed prefill is retried. A failure of the decision head fails the pass as before: no head
  runs an infer the recovery covers (the verbalizer and pointer heads are host math, and the
  joint head's own infer is not recovered), so a re-run could not be trusted to answer.
- **When.** The retry is admitted at once, ahead of every queued task, in the same engine step as
  the failure. That admission first runs the failed task's scrub and the recovery below, exactly
  as the next task's admission would, and then prefills the prompt again. In the worst case the
  failing step therefore waits for the 30 s idle wait (`STAGE_IDLE_TIMEOUT`), then the recompile
  (up to 180 s, `RECOMPILE_TIMEOUT`), then the recompiled stage's first infer (up to 30 s), so up
  to 240 s for each stuck stage (at most the engine's recompile cap: 2, or 1 when its largest stage
  is over 4 GiB, as on a 1-stage tree). After that comes the re-prefill of the prompt plus the
  tokens already generated. No timer bounds the re-prefill; it takes as long as a prefill of that
  length. (The recompiled stage's first infer is the first infer of that re-prefill.) That step
  returns nothing, and the engine reports it to the runner as progress
  (`Engine::progressed_without_chunks`). The runner closes a stream after 3 engine steps in a row
  that produce nothing (`MAX_CONSECUTIVE_EMPTY_STEPS`); neither the retried stream nor a stream
  queued behind it counts the retry step toward that. A step that waits on the recovery is still
  one step, however long it waits.
- **When it still fails.** If the recovery fails the engine, the retried task gets the usual
  `qwen36 engine failed: ...; restart the server` refusal and the server exits for its
  supervisor (below). If the retry of a chat or prefill pass itself fails, the task fails with
  the usual error plus `(on the retry after an infer failure)`; a decision head that fails on the
  retry fails with the head's own error, without the suffix. Either way the next task's admission
  runs the scrub, as before.
- **Logs.** A retry logs one `warn` line, `qwen36: retrying task after infer failure`, with the
  task id, `phase`, `prompt`, `warm_prefix`, `generated` (the tokens the failed attempt
  generated), `resume_seed` (the tokens of a peer resume it came with, not counted in
  `generated`), `attempt` (2, the retry) and the reason. `qwen36 task failed` is logged only when
  the task really fails. The per-site `prefill failed` and `decode failed` lines still mark each
  failed infer.

Pipeline mode does not retry: a failed task on the pipeline head fails as before.

### After a failed infer

On the GPU (pawan-03, Arc 140V), a decode infer threw `unknown C++ exception in infer`. 25 ms
later the next admission's scrub logged `stage reset failed ... Infer Request is busy`, and the
server then sat idle for 18 minutes, every request waiting. The failed infer had left the
OpenVINO request marked busy. OpenVINO refuses `reset_state` on a busy request. Its destructor
waits, with no timeout, for the run to finish. So the scrub's next `recreate_request`, which
released the old request, never returned.

What happens now:

1. The engine remembers that an infer failed. The next scrub, normally at the next task's
   admission, first checks every stage request (and the embeddings), without blocking.
2. A request that is still busy gets OpenVINO's `cancel()`, then `wait_for` in 100 ms slices,
   for up to 30 s. Once it is idle, the scrub resets or rebuilds it as before, and the task is
   served.
3. If it is still not idle after 30 s, the engine abandons the request. The request is never
   destroyed, because its destructor would block: it is leaked, with its device memory. The
   engine then recompiles that stage (or the embeddings): it compiles a fresh copy from the same
   IR, with the same xml path, device and plugin config that the load used. It installs the
   copy in place of the old one, and the task continues on it. The task that ran the recovery is
   served after the wait and the compile, and later tasks are served as usual.

**Why recompile, not a fresh request.** On pawan-01 (Arc 140V, build 435c874e) stage6 read
`Cancelled` for the whole 30 s after the fault. A fresh request from the same compiled model was
created in 1 ms, but its first infer failed with a shape-inference error, so the engine failed.
On the same GPU, a fresh request from a healthy compiled model served in every case tested, so
the fault corrupts state the GPU plugin keeps per compiled model. Only a new compile escapes it.
With the recompile (build 0c61635a) the same soak on pawan-01 recovered from the fault twice,
on stage3 and on stage6, and served the next request each time.

**The old stage is never dropped.** Its destructor could wait forever on the stuck request, so
the whole old copy (compiled model and weights) is leaked, whatever the recompile does. The same
holds when the engine fails instead (cap reached, no recipe, a request that cannot be abandoned)
and for a recompiled copy whose first infer fails or does not return: that copy is leaked too,
never left to drop with the engine. **Each recovery costs about one stage of memory**: the
stage's weight file, 1.49 GB for stage6 and 2.47 GB for stage7 of the 8-stage 9B int8 tree, on a
GPU that shares the host's RAM (31.6 GB on p01). On a 1-stage tree (the single-box default) the
one stage is the whole model, so a recovery leaks about 9 GB.

A recompiled stage needs nothing re-bound. Every input is set on each infer, and its variable
state starts empty. The scrub skips its reset and rebuild (it is fresh and has never run). The
next task then reads its input names and sets its inputs (`input_names`, `set_input`: host-side
tensor copies, not bounded) and runs its first infer, which is bounded.

**The two calls that can wait on the device are bounded.** Each runs on a helper thread, as
OpenVINO's usual synchronous call:

- The recompile is bounded by 180 s (`RECOMPILE_TIMEOUT`). A cold compile without the model
  cache can take tens of seconds; a cache-warm one measured 1.2 to 1.7 s on p01.
- The recompiled stage's first infer is bounded by 30 s, the same as the idle wait.

If either does not return in time, its thread and whatever it holds are leaked, never destroyed,
and the engine fails. The scrub's `reset_state` and `recreate_request` on the other stages are
not bounded: they run on compiled models that did not fault.

**When a restart is still needed.** The engine fails in these cases, and the server then exits
so its supervisor restarts it (below):

- The recompile fails, does not return within 180 s, or its helper thread dies.
- A recompiled stage's first infer fails, or does not return within 30 s. The device is then
  suspect, not the stage. This is a deliberate choice: a single recurrence of the intermittent
  fault on that first infer looks the same as a broken device, so it fails the engine even when
  a second recompile might have served.
- This engine has already recompiled as many stages as its cap, and another request is
  abandoned. The cap is per engine (one engine per process in production). Each recompile leaks
  a stage's weights, so the cap stops a fault that recurs from exhausting memory. The cap follows
  the size of what one recompile would leak, the largest weight file among the engine's local
  stages and embeddings: 1 (`LARGE_LEAK_MAX_RECOMPILES`) above 4 GiB (`LARGE_LEAK_BYTES`), else 2
  (`MAX_RECOMPILES`). The 8-stage 9B tree's stages (1.49 GB for stage6, 2.47 GB for stage7) keep
  2. A 1-stage 9B tree (the single-box default; its one stage is the whole model, about 9 GB) and
  a 27B 2-stage pipeline rank (12-15 GB per stage) get 1: two such leaks are not safe on a GPU
  that shares the host's RAM. There the first fault recovers and the request is retried (above);
  the second fails the engine, and the server exits for a supervisor restart (below), about 22 s
  warm for a 1-stage 9B tree. If a stage's weight file size cannot be read at load, the cap
  stays 2, the pre-Task 58 behaviour. The engine logs its cap at load (`qwen36: stage recompile
  cap`, with `total`, `local_stages`, `largest_leak_bytes` and `cap`), and every `recompile`
  recovery line carries `recompiled` and `cap`.
- The request could not be abandoned (the shim is out of memory), or its state cannot be read.

A failed engine fails the task that ran the recovery, and every later one at once, with
`backend error: qwen36 engine failed: <reason>; restart the server`. For example: `stage6's
infer request was still Cancelled ... abandoned; this engine has already recompiled 2 stages,
the cap of 2, so stage6 is not recompiled` (`... recompiled 1 stages, the cap of 1, so stage0 is
not recompiled` on a 1-stage tree), or `...; recompiling stage6 timed out after 180000
ms`. Chat and decisions return their 5xx. Only a restart recovers it.

**The server exits so the supervisor restarts it.** On rank 0 or a single box (`cascadia worker
--api` and `cascadia run`), the server notices the failure within 200 ms and logs one `error`
line: `engine failed; exiting so the supervisor restarts the server`, with the reason. It stops
accepting connections and new requests: an idle keep-alive connection is closed at once, and a
connection with a response in flight is closed after that response. It gives the responses
already in flight 3 s to reach their clients (the failing request's 5xx and any requests already
refused), runs the same bounded close as a SIGTERM (up to 10 s), and exits non-zero. Both
production units restart on a non-zero exit: systemd's `Restart=on-failure` and NSSM's `AppExit
Default Restart` (see [docs/deploy/README.md](deploy/README.md)). A warm restart (model cache
warm) measured about 5 s to `API serving` on pawan-03 for an 8-stage tree, and 21-24 s (about
22 s) on pawan-01 for a 1-stage tree, after the supervisor's own delay (systemd `RestartSec=5`).
If the close overruns its 10 s, the process exits non-zero at once instead of waiting on it.
Requests that arrive while the server is down get a connection refused, so clients should
retry. A SIGTERM or SIGINT still exits 0; if its close overruns the same 10 s, the process logs
`runner.close() overran after a shutdown signal; forcing a non-zero exit` and exits non-zero at
once too. Only qwen35 reports a failed engine; the other engines never do.

To debug a failed engine in place, pass `--keep-serving-on-engine-failure` (worker and `run`).
The server then stays up as before Task 56: it refuses every request with a 5xx, and `/health`
keeps answering and reports `degraded` after a failed completion.

**After a second recovery the engine is slower.** On pawan-01 (Arc 140V), an engine that holds two
leaked stage copies (about 3.2 GB) decodes 12-20% slower per stage, the embeddings about 2x, and a
chat took +16% (126 to 146 ms/token); this matches the +15% seen in the soak after the second
recovery. One leaked copy, or any number of recompiles whose old copies are freed, showed no
slowdown, so the cost is holding two leaked copies, not the recompile or the fault. Freeing a
faulted copy is unsafe (its destructor can hang), so the engine keeps serving, slower. At the
engine's second successful recompile it logs one `warn` line, once per engine: `qwen36: engine
degraded: 2 stage copies leaked by fault recovery (N bytes); decode is slower until the server
restarts (measured +16% with two on an Arc 140V); restart recommended`, with `leaked_copies`,
`leaked_bytes` and `cap` fields. The rule keys on successful recompiles: a recovery that fails
also leaks a copy, but it fails the engine, so no later recompile can warn. With the cap of 1 (a
1-stage tree, or a big pipeline stage) only one recompile can succeed, so this warning never fires
there: the one leaked copy (about 9 GB for a 1-stage 9B tree) stays until the restart, and the
next fault fails the engine instead of leaking a second. Its decode cost was not measured. The
code would name a helper thread leaked by a timed-out recompile in the text, but that cannot
happen today: a timed-out recompile fails the engine. The API process's `/metrics` exposes
`cascadia_engine_stage_recompiles_total` (counter: successful recompiles) and
`cascadia_engine_leaked_stage_bytes` (gauge: the weight bytes of every stage copy fault recovery
leaked, whether or not its recompile succeeded). Only the API process (rank 0) serves `/metrics`,
so a pipeline relay's recompiles are counted in the relay's own process and never exposed; they
show only in that relay's log. Restart to recover full speed.

Each recovery step is one `error` line, `qwen36: stage recovery`, with `stage`, `action`
(`probe`, `cancel`, `wait`, `abandon`, `recompile`, `first_infer`, `leak`, `reset` or `recreate`),
`outcome` and `elapsed_ms`. `leak` is logged when the engine fails with the faulted copy still in
its slot (cap reached, no recipe, or a request that could not be abandoned), with `leaked_bytes`
and `process_leaked`. The `recompile` line also carries:

- `recompiled`: this engine's count of recompiles, this one included;
- `cap`: the engine's recompile cap (2, or 1 when its largest stage is over 4 GiB);
- `leaked_bytes`: the size of the leaked stage's weight file;
- `process_leaked`: the stage copies this process has leaked on purpose (a copy left with a
  helper thread that timed out or died is not included);
- `leaked_requests`: every request the native shim has leaked in this process.

Its outcome is `ok`, `failed`, `timeout` or `refused: cap`. The recompiled stage's first infer
logs `first_infer` with `ok`, `failed`, `timeout` or `helper thread died`. The scrub logs
`skipped: recompiled` for the fresh stage. A worst-case recovery therefore takes up to 30 s for
the idle wait plus 180 s for the compile plus 30 s for the first infer, and each wait is
bounded. A failed engine also logs `qwen36: engine failed` once, `refusing task (engine
failed)` per task, and, unless the server keeps serving, the server's `engine failed; exiting`
line.

The shim also guards itself:

- `recreate_request` refuses a busy request.
- Destroying a runtime leaks a busy request rather than waiting on it, and logs it
  (`action=abandon outcome=leaked`).

**Pipeline mode.** A pipeline task whose infer fails is not retried (see [Retry after a failed
infer](#retry-after-a-failed-infer)). A downstream rank runs the same recovery inside the head's
RESET. It abandons the stuck request and recompiles the stage the same way, and then acks, so the
head's task is served. The head sees only a slower ack. The recovery takes up to 30 s of idle wait
plus the compile (1 to 2 s cache-warm, up to 180 s) for each stuck stage, and the time adds up
across ranks: a middle rank recovers before it forwards the RESET downstream. One stuck stage with a
cache-warm compile fits under the head's default 60 s reply bound. A cold compile, two stuck stages
(on one rank, or on two), or a lower configured bound can exceed it. The head then fails that task
with `RESET not acked`. The protocol does not desync: acks carry the task's epoch, and the head
skips a late one. The next task is served on the recompiled stage.

A downstream rank whose engine failed (one of the cases above, or a desynced downstream link of
its own; see [Pipeline reply timeouts](#pipeline-reply-timeouts)) does not ack the next RESET. Its
step returns a connection-fatal error, so the stage process exits for its supervisor to rebuild it.
The head sees the socket close and fails the task at once with `qwen36 pipeline RESET not acked:
socket closed`, instead of waiting out its reply bound on every task. The reason is in the
downstream rank's log. No frame changed, so a peer without this change handles it as any closed
link.

During the recovery the engine is held, so other requests wait for it. Every wait is bounded;
before Task 46 none was.

### BUG-1 (open)

The `unknown C++ exception in infer` itself is not fixed; the engine only recovers afterwards.
On hardware it showed up in long generations on the GPU (System 2's thinking, above, and once
after a prefix-cache restore); chat generations of that length can hit it too. Its cause is
unknown. One lead: OpenVINO's `InferRequest` wrapper turns any C++ exception into an
`ov::Exception`, so an error the shim reports as "unknown" did not pass through it as a C++
exception. On Windows that suggests a structured (SEH) exception from the driver or plugin. That
would also explain why the request stayed busy: OpenVINO's own cleanup never ran. On pawan-01
such a request never came back to idle: it stayed `Cancelled` for the full 30 s, and a fresh
request on the same compiled model then failed its first infer. The engine now recompiles the
stage (above). On pawan-01 (build 0c61635a, chat soak) the fault hit stage3 and later stage6 in
one run; both times the recompile took under 0.4 s, the recompiled stage's first infer
succeeded, and the next request was served. If a recompiled stage's first infer ever fails, the
engine fails. On OpenVINO 2026.4.0 a 1-stage tree ran 0 faults in 100 rounds on pawan-01, where
the 8-stage tree faults in about 3% of rounds.

## Devices

Decision trees run on `--device CPU` or `--device GPU` with the same files; no export or flag
changes. Measured on an Intel Core Ultra 7 258V (Arc 140V iGPU), single box, default GPU precision
(no `--ov-config`), against the same streamed fp32 references and gate as the CPU rows below:

| Model | GPU parity | max \|Δp\| | `/v1/decisions` vs `/v1/systemone` | CPU parity (for comparison) |
|---|---|---|---|---|
| JEV-9B | **80/82** PASS | 0.024 | 62/62 identical | 82/82 |
| Kev-9B | **82/82** PASS | 0.031 | 62/62 identical | 81/82 |
| Clef-flash | **82/82** PASS | 0.053 | 62/62 identical | 81/82 |

Every flip in these runs had a reference margin under 0.05, so no f32 override was needed. On the
typed-decisions benchmark (400 cases, five questions each) the GPU served all three models with
no errors. With the box otherwise idle, JEV-9B took 3.3 s per case against 34 s on the CPU, and
Clef-flash 1.3 s; Kev-9B's median was 13 s, measured while another job shared the CPU. GPU probabilities differ from the CPU's by up to about 0.16 on near-ties, so a caller that
compares raw probabilities across hosts should keep one device. The 27B models were validated on
CPU only.

**NPU: not supported.** `--device NPU` on the same host (Intel AI Boost, OpenVINO 2026.4) fails
at load, before any request: the NPU compiler rejects the first stage with `Missing upper bound
for one or more nodes`, because the qwen35 trees have unbounded dimensions (the embeddings
input is `[?, ?]`, and every stage takes a `[1, n, width]` hidden of any `n`). The NPU needs
static or bounded shapes. Cascadia's existing NPU path, for other engines, compiles pre-exported
static decode and prefill-chunk variants. Bringing decision trees to it would need the same for
every qwen35 stage: a static prefill chunk with padding, a static readout, and a bounded
linear-attention and KV state. That is a separate project, not a flag.

### One stage or eight

A single box serves a 1-stage tree (the exporter's default) faster than the same weights cut into
8 stages. Measured on pawan-01 (Arc 140V, `--device GPU`, OpenVINO 2026.4.1, build bd02a0f5), same
weights exported both ways:

| Model | td40 decision p50, 8 → 1 stage | 1600-token chat, 8 → 1 stage | GPU parity (both) |
|---|---|---|---|
| JEV-9B | 3.46 → 2.51 s (-27.7%) | 211.7 → 184.9 s (-12.7%) | 80/82 |
| Kev-9B | 2.52 → 1.92 s (-23.8%) | 211.6 → 185.6 s (-12.3%) | 82/82 |
| Clef-flash | 1.38 → 1.17 s (-15.0%) | 188.7 → 174.2 s (-7.7%) | 82/82 |

`bench_cases.py compare` found no confident flips between the two layouts. The costs: warm load
to `API serving` 21-24 s against 7-13 s; exporter peak RAM 8.25 / 20.3 / 20.6 GiB against 3.1 / 5.1
/ 5.3 GiB (JEV / Kev / Clef-flash), which is why a low-memory machine exports `--stages 8`; and a
fault recovery leaks the whole model (about 9 GB) instead of one stage (1.5-2.5 GB), so the engine
recompiles at most once (see [Engine failures](#engine-failures)). On OpenVINO 2026.4.0 the
1-stage tree also ran 0 BUG-1 faults in 100 rounds, where 8 stages fault in about 3% of rounds.

### Measuring

Per-pass timings are logged at debug level on target `cascadia::perf`; start the server with
`RUST_LOG=info,cascadia::perf=debug` to see them. With plain `RUST_LOG=info` they cost nothing
beyond a few clock reads per stage infer. Two events come from the API and one from the engine:

- `decision request done` (API, per request): `family`, `passes`, `input_tokens`, `encode_ms`,
  `passes_ms`, `total_ms`.
- `decision pass done` (API, per pass): `task`, `pass_ms`, `prompt`, `prefilled`.
- `qwen36 decision pass done` (engine, per pass, single box only; a pipeline tree logs none):
  `task`, `head`, `prompt`, `warm_prefix`, `prefilled`, `spans`, `span_max`, `admit_ms`,
  `admit_reset_ms`, `restore_ms`, `prime_ms`, `set_state_ms`, `snapshot_ms`, `embed_ms`, `infer_ms`,
  `stage_ms` (per stage, comma-separated), `host_copy_ms`, `head_ms`, `reset_ms`, `recreate`,
  `total_ms`. `reset_ms` and `recreate` are the pass's own reset, which runs at admission
  (`reset_ms` equals `admit_reset_ms`): a cold pass's reset, with `recreate=true` when the
  previous pass had restored a prefix-cache state. In the default restore mode a warm hit runs no reset (`reset_ms` 0,
  `recreate=false`), and after another decision pass its `prime_ms` is 0 too. With
  `CASCADIA_QWEN36_RESTORE_CLEAR=recreate` or `reset` a hit scrubs first as before, so it
  recreates and primes.

**Primed requests between passes (single box).** A decision pass no longer scrubs the stage
requests when it finishes. Before, the finish recreated every stage request (because the pass had
restored a state), and the next pass's restore then had to run one priming fold on the new
requests before `set_state`, since the GPU plugin drops a state set onto a request that has never
run. Now the finished pass's requests stay live and primed, and the next decision pass that hits
the prefix cache writes the restored state straight over them. This is safe because the shim's
`set_state_blob` writes every VariableState or fails (the blob's state count must equal the
request's, and every state must apply); a failed restore recreates the requests and prefills
cold. Every other admission (a cold decision pass, chat, a kv_coord restore, the scrub after a
failed task, a stage recovery) runs the deferred scrub first (a recreate after a warm pass, else
the cheap reset), so nothing prefills on the finished pass's state. Pipelines are unchanged: they
reset after every pass. The cost of that scrub moves rather than disappearing: a chat that follows
a decision pass pays that scrub (a recreate after a warm pass, else the cheap reset) at its own
admission (before its first token), where before the decision request paid it at its finish.
Measured on pawan-03 (Arc 140V, 40 td40 cases, build 8f6d8f1a vs 0c61635a): a Kev restore fell
from about 300 ms to about 91-100 ms with `prime_ms` 0 and no recreate, and Kev p50 fell from
3.56 s to 2.41 s (-32%, p90 4.13 s to 2.85 s). Answers were identical (max |dp| 0, no flips) for
Kev, JEV and Clef; JEV (3.50 to 3.41 s) and Clef (1.41 to 1.38 s) stay within noise, since their
passes rarely restore. A temperature-0 chat returned the same text before and after the decision
runs.

`tools/decision/bench_cases.py` drives a running server and compares runs:

```bash
# Once: the typed-decisions test split -> the 40-case set (first 10 per workflow, round-robin).
python tools/decision/bench_cases.py cases td-test.json td40.jsonl
# Post every case to /v1/systemone, one at a time; prints per-case latency and p50/mean/p90.
python tools/decision/bench_cases.py run --url http://127.0.0.1:8080 --model kev-9b \
    --cases td40.jsonl --out kev-new.jsonl
# Run-to-run parity and latency deltas; exits 1 on max |dp| > --max-dp (0.05), a flip where the
# first run's margin is >= --confident (0.05), or different case sets.
python tools/decision/bench_cases.py compare kev-base.jsonl kev-new.jsonl
# Median and p90 of every numeric field of the engine's per-pass lines in a server log.
python tools/decision/bench_cases.py logsum server.log
```

The case file and the run files hold request content (states, questions and answers), like
`--decision-dump`: `run` and `cases` write them with mode 0600 and never overwrite one. Keep them
private.

## Limitations

- **Text only.** Images in either format are a 400 (`images`, `videos`, OpenAI `input_image`
  parts, and JEV image parts in a list `state`).
- **Batch 1.** The engine runs one decision pass at a time. Concurrent requests beyond
  `--max-decision-requests` get 503; the admitted ones wait their turn for the engine.
- **int8 weights only.** Decision trees are built from Intel's int8 OpenVINO exports. int4 measured
  87.8–96.3% top-1 agreement with full precision, against 98.8–100% for int8, and is not supported.
- **JEV choices past the trained slots are multi-pass.** They take up to 256 options, but the
  model was trained on 16: the extra passes cost prefill, and the answers are serve_decide's
  approximations (see [JEV choices past the trained slots](#jev-choices-past-the-trained-slots)).
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
    --out ~/trees/jev9b-int8           # 1 stage; 201 LoRA branches grafted (--stages 8: 25 per stage, 26 on the last)
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
M4 Max). Serve the 9B models on x86 (Intel) hosts; on a Mac, only one stage per process fits (export
with `--stages 8`).

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
    --out ~/trees/kev9b-int8           # 1 stage; 248 LoRA branches grafted (--stages 8: 31 per stage)
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

The export holds a whole stage's new int8 constants (12–15 GB) and needs about 16 GB free. The exporter's
memory check applies to 1-stage exports only, so this 2-stage export is not checked. Serve it with the two `cascadia worker` commands shown for JEV-27B.

**Measured parity** (same two hosts and set as JEV-27B; one joint pass per record, 50 passes for 82
questions):

| Check | Result |
|---|---|
| Records vs the spike's `encode_record` (50 + 4 synthetic) | 54/54 |
| int8 (2 machines) vs streamed fp32 reference | **81/82** (choice 32/33, predicate 24/24, score 25/25), max \|Δp\| 0.067; the one flip, `r41.team`, has a reference margin of 0.048 |
| `/v1/decisions` vs `/v1/systemone` (string states) | 62/62 identical |
| Long joint pass (2 machines, CPU) | 9,609 tokens: 200 in 18 min; 16,384 tokens (a longer state, truncated to the cap): 200 in 37 min |

At 16,384 tokens the last rank (which runs the joint head) peaked at 20.7 GB private memory and a
28.0 GB working set on a 32 GB host, with under 1 GB of physical memory free during the pass;
rank 0 peaked at 13.3 GB private. Leave the last rank's host otherwise idle for requests this long.

**The swap must keep zero in each row's int8 range.** The first Clef 27B tree measured 78/82 with two
confident flips and max |Δp| 0.30. The cause was the swap's quantizer: it scaled each row over
[min, max], while NNCF (which built Intel's IR) uses [min(w, 0), max(w, 0)]. The two agree on every
projection row, but a 4-tap linear-attention `conv1d` channel is single-signed in about a quarter of
rows, and those rows saturated. That garbled the convolution in all 48 linear-attention layers of every
`--weights` tree. With the fix, the swapped `conv1d` matches Intel's constants in at least 99.9% of
entries.
