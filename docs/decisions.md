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
- `strategy` (optional, top level): how a JEV choice with more options than its trained slots is
  answered, `"auto"` (the default), `"single"`, `"tournament"` or `"permute"`. See
  [JEV choices past the trained slots](#jev-choices-past-the-trained-slots).
- `thinking`, `threshold`, `think_budget`, `return_reasoning` (optional, top level, JEV only):
  adaptive thinking. See [JEV adaptive thinking (System 2)](#jev-adaptive-thinking-system-2).

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
  `/v1/systemone` reports. `output_tokens` is the tokens System 2 generated, 0 unless a question
  thought (decision passes do not decode). `total_tokens` is their sum.
- With thinking `auto` or `on`, every non-score answer also carries `thinking` (see
  [JEV adaptive thinking (System 2)](#jev-adaptive-thinking-system-2)); otherwise answers are
  exactly as shown.
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
- `thinking`, `threshold`, `think_budget`, `return_reasoning` (optional, JEV only): as on the
  OpenAI route. See [JEV adaptive thinking (System 2)](#jev-adaptive-thinking-system-2).
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
is the tokens System 2 generated, 0 unless a question thought (decision passes do not decode).
With thinking `auto` or `on`, every non-score answer also carries `thinking`.

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
| 400 | A body that is not valid JSON; on the OpenAI route, also a body that does not match the request schema (a missing `questions`, an unknown question `type`, a repeated key). Validation: no questions or more than `max_questions`; an empty question name, or (OpenAI format) a duplicate question name or choice value; a choice with no options or more than the model allows; a score with fewer levels than the model allows (2 for JEV, 1 for Kev and Clef) or more; images; an image part, or a text part without a string `text`, in a list `state` (JEV); a request the tokenizer cannot encode; Clef's option cap (see [Limits](#limits)); a `strategy` other than `"auto"` on Kev or Clef (`strategy applies to JEV only`); on the OpenAI route, a `strategy` that is none of the four values; thinking that cannot run (see [JEV adaptive thinking (System 2)](#jev-adaptive-thinking-system-2): any System 2 field on Kev or Clef, a score question with an explicit `thinking`, a tree that is not switchable or lacks System 2's export, System 2 disabled at startup for a missing chat template, more options than answer labels, `threshold` outside [0, 1], `think_budget` outside 1..=`max_tokens`). |
| 413 | The body is over `--api-max-body-mb`. Too long: the state is more than `max_total_tokens × 8` bytes (checked before tokenizing); Kev's whole rendered text (state, instructions, options and levels) is more than `max_total_tokens × 8` bytes; Kev's rendered state is more than 524,288 bytes (65,536 × 8, checked before tokenizing); one Kev question's instructions and option texts render to more than 8 bytes per token the pass has left after the state (`max_tokens` minus the state's tokens, checked before tokenizing); Kev's state is more than 65,536 tokens (`SERVE_MAX_STATE`, `<state>` included) or reaches `max_tokens`; one pass is longer than `max_tokens` (JEV and Kev; Clef truncates the state instead); Clef's schema alone fills the joint pass, leaving no room for a non-empty state; the request's prefill work is more than `max_total_tokens` (JEV stops tokenizing at the first pass that takes it over: `needs at least N`; a JEV tournament's final counts at its worst case before any pass runs, see [JEV choices past the trained slots](#jev-choices-past-the-trained-slots); so does System 2 at its `think_budget` under `thinking: "on"` (`auto` skips a question instead), see [JEV adaptive thinking (System 2)](#jev-adaptive-thinking-system-2)); a Kev request that needs the shared-state discount when the prefix cache can never hold that state (`this request needs N prefill tokens without prefix reuse (the shared prefix cannot fit the prefix cache); the per-request limit is M`, refused before any prefill once the engine knows the snapshot's size, which its warmup measures at startup; see [Limits](#limits) for when the first such request can still spend one pass); or the engine reports the prompt too long. |
| 415 | The request lacks `content-type: application/json`. |
| 422 | `/v1/systemone` only: valid JSON that does not match the request schema, a `strategy` that is none of the four values included (the OpenAI route answers 400 instead). |
| 501 | The model serves no decisions: no `decision.json` (or one that fails its load checks; see the startup log), an engine other than `qwen35`, the engine has no decision head (`decision/` missing on the single box or the pipeline's last rank), the tree was exported without `--final-hidden` (Kev and Clef), or a pipeline rank cannot serve decision passes (an older build, rank 0 or the last rank without `config.json` `vocab_size`, or the last rank without `decision/`; a middle rank only relays and needs neither). Rank 0 refuses the pipeline cases before any prefill. |
| 503 | The server is at capacity (`engine at capacity; retry after current requests complete`): every decision permit is taken (`--max-decision-requests`, default 16), or a Kev pass that must resume its shared state from the prefix cache found the snapshot evicted by other requests (see [Limits](#limits)); retrying can succeed. Any other engine error is `decision engine error`, including non-finite logits or probabilities (never a 200 with `null`s), and a failed System 2 task under `thinking: "on"` (under `auto` the question answers from System 1 instead, see [Failures](#failures)). Decisions have their own admission pool: chat requests never take its permits, and decisions never take chat's. Chat can still cause this 503 by evicting a Kev snapshot. |

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
| Thinking tokens per question (JEV System 2) | request `think_budget` | 4096 (or `max_tokens` when smaller); at most `max_tokens` |
| Concurrent decision requests | `--max-decision-requests` | 16 (1 to 240), a pool separate from chat's 16; one more is a 503 |
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
  requests keep their own pool of 16. Decisions and chat never take each other's permits, so a
  burst of decisions never uses up chat's permits, and a burst of chat never uses up the
  decisions'. The server as a whole admits up to 16 chat + N decision requests at once: 32 by
  default, where both used to share 16. Each one holds its body (up to `--api-max-body-mb`) and,
  for a decision, its tokenized passes, so size memory for 16 + N. N runs from 1 to 240, and
  anything else is a startup error that names the limit. The reason: every admitted request can
  have a pass waiting in the engine's 256-task pending queue, and 16 + N must fit there.
  Otherwise decisions would fill the queue and chat would get `QueueFull` 503s. A decision
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
- **Dashboard.** `/api/stats` `max_concurrent` is chat's pool only (16), while
  `requests_in_flight` counts chat and decision requests together. It can therefore read up to
  16 + N against a ceiling of 16.
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

## JEV adaptive thinking (System 2)

A JEV request can ask the tree's base model to reason before it answers, as the reference server
does (serve_decide `decide` and `s2_dist`). System 1 is JEV's usual answer (`p1`, every plan
above). System 2 runs the same tree with its LoRA switched off, so the tree must be switchable
(`--switchable-lora`, [Runtime LoRA scale](#runtime-lora-scale)). The base model thinks over the
question in its chat template, the server reads its answer distribution `p2` after the thought,
and the answer becomes the blend `p = (1 − mix) · p1 + mix · p2`.

### Request fields

Top level, on both routes, JEV only (on Kev and Clef any of them, other than `thinking: "off"`
and `return_reasoning: false`, is a 400 `thinking, threshold, think_budget and return_reasoning
apply to JEV only`):

| Field | Values | Default |
|---|---|---|
| `thinking` | `"off"`, `"auto"`, `"on"`; `null` is the default | the profile's, `"off"` |
| `threshold` | `auto` thinks when System 1's leading probability is below it, in [0, 1] | the profile's, 0.8 |
| `think_budget` | most tokens the base model may think, 1..=`max_tokens` | 4096, or `max_tokens` (decision.json) when smaller |
| `return_reasoning` | `true` returns the thought in `thinking.reasoning` | `false` |

With `thinking` off (or absent with the default profile) the other three fields are ignored, as
the reference ignores them, and every response is byte for byte what it was before thinking
existed.

**Why the default budget is 4096, not the reference's.** serve_decide's default is no limit
beyond the context window (262,144 tokens for Qwen3.5). The engine decodes one token at a time
on a batch-1 slot, so on our hardware a window-long thought is hours of decoding per question, and
its worst case (the generation plus
the read-back, both counted up front) is more than a request's `max_total_tokens` can hold. 4096
keeps a thinking question to minutes and lets `on` over a few questions fit the default
`max_total_tokens` (4 × `max_tokens`).

### Triggers

- `on` thinks for every noul and choice question; `auto` only for one whose System 1 leading
  probability `max(p1)` is below `threshold` (equal does not think).
- Score questions never think (the reference). A request that sets `thinking` to `auto` or `on`
  and holds a score question is a 400 (`thinking is supported for noul and choice questions; s is
  a score question`); under a profile whose default thinks, a score question just answers.
- Each question thinks on its own, in request order, after every System 1 pass. A wide choice
  ([above](#jev-choices-past-the-trained-slots)) thinks over all its options in one System 2
  pass, whatever its System 1 plan.

### What runs

1. **Prompt.** The tree's chat template (the one chat uses) is rendered for one user turn whose
   content is `XQXCONTENTXQX`, with the generation prompt and thinking on, and split around that
   placeholder. The user content is the state (as System 1 renders it), then
   `"\n\nQuestion: {question}\n\nOptions:\n{options}\n\nThink it through carefully, then give your
   final answer on the last line in the form: Answer: (X)"`, where the options are lines
   `"({label}) {option}"` labelled with the profile's answer labels (`A`, `B`, ...). A noul
   shows `(A) Yes (true)` and `(B) No (false)`. Caller text is sanitized as in System 1.
2. **Thinking.** The engine decodes from that prompt on the base model (`lora_scale = 0`),
   greedily, for at most `think_budget` tokens. The server stops at the end-of-think token
   (`</think>`): the engine does not stop at a token id, so the server drops the stream there,
   which cancels the task.
3. **Read-back.** One Verbalizer readout, also on the base model, over the prompt, the thought
   (decoded with special tokens kept), `</think>`, `"\n\n"` and `"Answer: ("`, scores the first
   n answer label ids. `p2` is their plain softmax: no bias and no temperature. A noul's
   (A) Yes / (B) No become `[P(false), P(true)]`.
4. **Blend.** `p = (1 − mix) · p1 + mix · p2`, with the profile's `mix` (0.5). `p2` alone is close
   to one-hot and over-confident; the blend keeps System 1's calibration. Confidence, `choice`
   and `score` are computed from `p` as usual.

### Response fields

Every noul and choice answer of a request whose mode is `auto` or `on` carries `thinking`:

```json
"urgent": {"type": "noul", "noul": 0.5845,
           "thinking": {"mode": "on", "used": true, "threshold": 0.8, "budget": 512,
                        "think_tokens": 187, "finished_within_budget": true,
                        "reasoning": "The customer was charged twice ..."}}
```

- `mode`, `threshold`, `budget`: the request's resolved settings.
- `used`: the base model thought and was blended in. An `auto` answer that did not think has only
  `mode`, `used: false`, `threshold` and `budget`, and its probabilities are System 1's.
- `reason`: `"budget"` on an `auto` answer that would have thought (System 1 was below the
  threshold) but whose System 2 worst case did not fit what the request had left (see below).
  `"error"` on an `auto` answer whose System 2 failed in the engine (see
  [Failures](#failures)). Either way its probabilities are System 1's.
- `think_tokens`: tokens generated, the closing `</think>` included.
- `finished_within_budget`: the thought closed, or the model stopped on its own, before the
  budget ran out. A thought cut at the budget is still read back, with `</think>` appended.
- `reasoning`: the thought, trimmed, only with `return_reasoning: true`.

On the OpenAI route the same object sits on each answer. `usage.output_tokens` counts every
thinking token, and `usage.input_tokens` adds each generation's and read-back's prompt.

### Limits and budgets

- **Switchable trees only.** System 2 runs the base model through `lora_scale = 0`, which a
  baked tree refuses at the engine. The API checks first: a tree without `jev.lora_scale` is a
  400 (`thinking needs a switchable tree ...`). So is a tree without System 2's export
  (`jev.answer_labels`, `jev.end_think_id`; re-export it), and a choice with more options than
  answer labels.
- **Chat template.** It is rendered once at startup. If a JEV that could think has no template,
  or it does not render a user turn, the server logs a warning and disables System 2 for that
  tree: System 1 keeps serving, a profile default that thinks is ignored, and a request that sets
  `thinking` to `auto` or `on` is a 400 (`thinking is disabled on this server: the model's chat
  template did not load or render`).
- **Budget, up front.** Before any pass runs, each question that may think (every noul and choice
  under `auto` too) is reserved at its worst case against what `max_total_tokens` leaves after
  System 1: its prompt plus `think_budget` generated, then the read-back over both plus its tail.
  - `on` is strict: a request that cannot afford every question is a 413 naming the budget
    (`... for thinking at think_budget N ...; the per-request limit is M (lower think_budget)`),
    before any engine call.
  - `auto` reserves question by question, in request order. A question that does not fit (nor
    the pass limits below) answers from System 1 alone with `reason: "budget"` if it would have
    thought. `auto` never makes a request fail; a profile default of `auto` cannot turn a working
    request into a 413.
  - The real read-back is checked again before it runs (a 413 under `on`, a budget skip under
    `auto`).
- **Pass lengths.** A System 2 prompt plus the read-back's tail must fit `max_tokens`, like any
  pass. The read-back adds up to `think_budget` tokens of thought on top, so it may exceed
  `max_tokens`, up to the engine's decision prompt cap: the model's window (config.json
  `max_position_embeddings`, read at load as the engine reads it), at most 131,072. Both are
  checked up front at the budget: a 413 under `on`, a budget skip under `auto`.
- **Admission.** System 2 runs inside the request: the same decision permit, the same engine
  queue, no extra admission. A thinking request holds its permit for as long as it thinks.
- `--decision-dump` records System 1 passes only.

### Failures

A System 2 task can fail in the engine:
- its generation is refused at submit;
- an engine error arrives before the first token or mid-stream. On hardware this was a decode
  infer that threw: `decode failed: backend error: unknown C++ exception in infer`;
- its stream ends with no final chunk;
- its read-back fails.

The request learns of each one as soon as the engine reports it:

- `on`: the request fails like any engine failure, on both routes. That is the 503 `decision
  engine error`, or the capacity 503 or 413 that an engine refusal maps to, as for System 1
  passes. No further engine task is submitted.
- `auto`: the question answers from System 1 alone with `thinking: {"used": false, "reason":
  "error"}`, and the request is a 200. **The request submits no further engine work.** Every later
  question that would have thought also answers from System 1 with `reason: "error"`. Questions
  that would not have thought report as usual. The reason: what failed may still hold the engine
  (see below), and the next task would wait behind it. System 1 has already run by then. Tokens
  the generation streamed before it failed count in `usage`.

**The API's line.** Each failure is one `warn` line, `System 2 task failed`, with these fields:
- `task`: the task id, `think-…` for the generation or `dec-…` for the read-back;
- `phase`: `generate` or `readback`;
- `prompt_tokens`;
- `cache`: `hit` or `miss`, or `unknown` when the engine did not say. That happens for a refusal
  at submit, a stream the runner or API ended, or an engine that does not report it;
- `warm_prefix`: the prompt tokens resumed from the prefix cache, 0 on a miss;
- `generated`: the tokens streamed before the failure;
- the mode and the reason.

The engine reports the cache state on its error chunk (`Chunk::warm_prefix`, an additive field
that is omitted when unknown).

**The engine's line.** The engine (`cascadia_engine_openvino::qwen36`) also logs one `warn` line
per failed task, `qwen36 task failed`. It has the task id, `phase` (`prefill`, `decode` or
`head`), `prompt`, `warm_prefix`, `generated` and the reason. This covers every failure site:
single-box prefill, decode and decision head, and the pipeline head's prefill, decode and
decision. The older per-site lines (`decode failed`, `prefill failed`, ...) remain. The engine's
text is logged and never returned to the client.

**What the fix covers.** The engine returns a failed task's error before it scrubs the stage
requests. The scrub (`reset_all`, which runs a native `recreate_request` per stage after a warm
turn) now runs when the next task is admitted. On the GPU, a request once waited more than 15
minutes after a failed decode infer. The engine logged `decode failed`, the API never logged
receiving the error, and an exact replay later completed cleanly. Two explanations fit:

1. *The API was still listening.* The scrub, which used to run before the error chunk was
   returned, blocked, so the chunk never left the engine. This fix covers that case: the error
   is returned first, and the request gets its 503 (`on`) or its System 1 answers (`auto`, which
   submits no further task) at once.
2. *The API had already stopped listening.* The failing decode belonged to a think task the API
   had dropped at `</think>`, while its cancel was deferred behind another request's step. The
   error went to a dropped stream, by design, and the API was waiting on the read-back. That
   read-back's admission now runs the scrub. If the scrub blocks, the read-back waits as before.
   **This fix does not cover that case.**

**Telling them apart.** Match the engine's `qwen36 task failed` (or `decode failed`) task id
against the API's lines:
- the same `think-…` id in an API `System 2 task failed` line means case 1, which is covered;
- no API line for that id, with the request still waiting on a later `dec-…` task, means case 2.

**Known limit.** A native scrub that truly blocks still holds the engine lock. It does so at the
next task's admission: that task, and every request behind it, chat included, waits. There is no
engine-level timeout on native calls. The API adds none either, since a timeout cannot free the
engine. Chat requests share the engine path and get the same reordering.

### decision.json

The exporter (`tools/decision_export.py jev`, Task 43) writes, from the tokenizer the tree ships:

- `answer_labels` / `answer_label_ids`: serve_decide's rule, `A`..`Z` then `AA`..`ZZ`, each kept
  when `"Answer: ({label})"` encodes as `"Answer: ("` plus exactly two more tokens (the label's,
  kept, and the `)`), at most 256.
- `end_think_id`: the id of `</think>` when it is one token (omitted otherwise: no thinking).

A profile with answer labels must have at least 2 (a noul shows two options), or it fails the
load.

The adaptive-thinking settings default to serve_decide's Qwen profile and can be overridden under
`jev.system2`: `thinking` (`"off"`), `threshold` (0.8), `mix` (0.5), `end_think` (`"</think>"`,
which must encode to `end_think_id`) and `after_think` (`"\n\n"`). A profile whose default
`thinking` is not `off` must be able to think (switchable, with the export), or it fails the load.

### Differences from the reference

- **Greedy decoding.** The engine decodes greedily; serve_decide samples the thought with a seed
  (`crc32(question + options)`). The reasoning text, and so `p2`, differ from the reference's for
  the same input.
- **Option text.** Choices show System 1's option line text (`name: description`, or `name` when
  the description is empty), not the bare option string the reference shows, so System 1 and
  System 2 see the same options.
- **Read-back tokens.** The thought is decoded from the generated ids and the read-back prompt is
  tokenized from text, as the reference renders its read-back. The thought is model output and is
  not sanitized, also as in the reference.
- **No `reasoning_effort`, `chat_template_kwargs`, `debug` or `system2_only`.** The template's own
  `reasoning_effort` default applies (the reference's without kwargs).
- **Several questions.** The reference answers one question per request; here every eligible
  question thinks on its own, and the budget counts all of them.

### Measured

JEV-9B switchable tree (`--switchable-lora`, int8), single box, GPU (Intel Arc 140V, Core Ultra 7
258V, 32 GB, OpenVINO GenAI 2026.4.0), `think_budget` 4096, threshold 0.8. Three requests: a
refund case (a noul plus a choice), an ambiguous department choice and an arithmetic choice.

| Request | `off` (System 1) | `on` | `auto` |
|---|---|---|---|
| Refund noul | 0.9964 | 0.9979, 491 thought tokens | 0.9964, no thought (confident) |
| Refund department | billing 1.0 | billing 0.9999, 602 thought tokens | billing 1.0, no thought |
| Ambiguous department | billing 0.6142 / technical 0.3845 | billing 0.8071 / technical 0.1923, 1429 thought tokens | same as `on` (below threshold, thought) |
| Arithmetic cost | $8 0.9926 | $8 0.9961, 708 thought tokens | $8 0.9926, no thought |
| Latency per request | 0.2–0.5 s | 92–188 s | 0.3–0.5 s when confident, 185 s when it thinks |

Every thought finished within budget. The same sequence gave the same answers across three runs.
One earlier run hit an `unknown C++ exception in infer` on the ambiguous request's think task under
`auto`, after a prefix-cache restore; two later runs of the same sequence (one on the build with
the [Failures](#failures) handling) did not reproduce it. The cause is open; the failure lines
above are what a recurrence must be diagnosed from.

System 1 on the same switchable tree matches the baked tree: CPU 82/82 (max |Δp| 0.0175,
identical to the baked tree), GPU 81/82 (max |Δp| 0.0269; the flip is fp16 rounding).

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
encoders. System 2's generation and read-back (see
[JEV adaptive thinking (System 2)](#jev-adaptive-thinking-system-2)) are not written.

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

- `--stages N` sets the stage cut count: 8 for a single box, one per rank for a pipeline.
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
- JEV takes `--switchable-lora` to make the adapter switchable at runtime, the first step toward
  adaptive thinking (System 2), which runs without the adapter. The scale is no longer baked
  into the grafted branches: every stage that carries one gains a scalar f32 input `lora_scale`,
  and each branch adds `lora_scale · (x·Aᵀ)·Bᵀ`. Feeding the trained scale gives the JEV model and
  feeding 0 removes the adapter, from the same tree. Scale 0 is the stock base model only when
  the tree was exported from the stock IR with no `--weights` (the shipped JEV recipe above); a
  tree exported with `--weights` runs that backbone at 0. The trained scale (`lora_alpha / r` from
  `adapter_config.json`, or `--lora-scale`; 2.0 for the shipped adapters) is written to the `jev`
  object of `decision.json` as `lora_scale`. A tree without that field has its scale baked in and
  no such input; that is the default, and a default export is unchanged byte for byte. The qwen35
  engine feeds the input on every pass (see [Runtime LoRA scale](#runtime-lora-scale)). Kev and
  Clef refuse the flag.
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

## Runtime LoRA scale

The qwen35 engine serves a switchable JEV tree (`--switchable-lora`, above) as either model:

- At load the engine reads which stages take `lora_scale` from `manifest.json` (each stage's
  `inputs`) and checks every compiled stage IR against it; a stage whose IR and manifest entry
  disagree fails the load. A tree is switchable when any stage takes the input. Rank 0 (or the
  single box) also needs `jev.lora_scale` in `decision.json`: a switchable tree without it fails
  the load.
- A task's `lora_scale` (`GenerationTask::lora_scale`) picks the scale. `None` is the trained
  scale, used by chat and every System 1 pass; `Some(0.0)` removes the adapter (System 2's
  generation and read-back, see
  [JEV adaptive thinking (System 2)](#jev-adaptive-thinking-system-2)). The
  scale must be a finite number ≥ 0 (`-0.0` is 0). The engine feeds it to every stage that has the
  input on every infer: prefill, decode, the decision readout's prefill, the priming fold before a
  prefix-cache restore, and warmup (which runs the trained scale). Stages without a LoRA branch
  have no input and are not fed.
- A tree whose scale is baked in has no input and runs as before. A task that sets `lora_scale`
  on it (any `Some`, 0 included) is refused at submit with `backend error: tree has no switchable
  LoRA`, never served with the adapter on.
- The single box's prefix cache scopes every snapshot by the scale it was taken at, so a
  base-model state and a LoRA state of the same tokens never restore for each other. The scopes
  share one byte budget and one LRU. A `require_warm` pass whose prefix was cached at the other
  scale gets `decision_cold`, as if it had been evicted.
- With `kv_coord`, the KV plane keys its blobs by tokens alone, so on a switchable tree only
  trained-scale turns are captured into it or warm-resumed from it; a scale-0 turn runs cold.
- In a pipeline every rank feeds its own stage, and rank 0 carries the task's scale to the other
  ranks. HELLO_ACK capability bit 2 (`CAP_LORA_SCALE`, value 4, ANDed along the chain like the
  decision bit) says a rank takes the scale, and rank 0's HELLO payload carries `"lora_scale": 1`
  for the same thing upstream. On a switchable tree, rank 0 or a middle rank refuses a chain
  whose ACK lacks the bit, and a downstream rank refuses (NAKs) an upstream whose HELLO lacks the
  key, so an old build on either side fails the handshake with a reason naming `lora_scale`. A
  tree with its scale baked in ignores both, and old and new ranks keep working together.
- The scale rides in the `pos` field of the task-start frames, RESET and (with `kv_coord`)
  RESTORE, as its f32 bits. Those frames carried `pos = 0`, which an older rank ignores, and the
  field is set only on a switchable tree whose chain agreed on the bit, so the frame layout is
  unchanged. Each rank relays the scale it received down the chain.

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
  second HELLO. In the served 2-rank topology a long load therefore costs one failed request
  rather than a restart. The JEV-27B runbook below still raises the timeout, so that no request
  fails.

These guarantees are for the served 2-rank topology. In a longer chain:

- A slow-loading rank below a middle rank makes the middle's own HELLO forward time out. The
  middle then answers rank 0 with a handshake refusal, which poisons rank 0 until restart, so
  there a long load costs a restart.
- A middle rank drains late replies on its own downstream link by the same rules. But a desynced
  middle cannot pass the restart message upstream, so rank 0 sees only timeouts.

The wire format is unchanged, so ranks of older builds work with this one as before. Every rank
of a chain must be built with the same features: a chain that mixes builds with and without KV
coordination (the `kv_coord` feature) is unsupported. No metric reports the desynced state yet;
watch for the error line.

**Measured on hardware** (JEV-27B across two hosts, rank 0 with a 40 s timeout): rank 1 was frozen
for 70 s during a 1,967-token request. That request returned 503, rank 0 logged the late TOKEN
reply as skipped, and the next task logged `skipped a late reply of an earlier task` (epoch 3,
then its own epoch 4). The two requests after it returned 200 with answers identical to a run
without the timeout, in normal time.

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
