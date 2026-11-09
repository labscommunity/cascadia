# The Responses API (`POST /v1/responses`)

Cascadia serves OpenAI's Responses API by mapping each request onto the chat path: the
request becomes a `/v1/chat/completions` request and is admitted by the same code, so the
chat template, sampling, image input, limits and every chat refusal apply unchanged
(images: see [VLM.md](VLM.md#the-responses-api)). Only the request and response shapes
differ. It serves function and custom tools, returns reasoning as reasoning items, and keeps
recent output items in memory so `item_reference` works. It is enough for the Codex CLI,
the OpenAI Agents SDK, `@ai-sdk/openai` and n8n's OpenAI Chat Model (Responses on).

Cascadia keeps no conversation state (`previous_response_id` and `conversation` are refused):
send the whole conversation in `input` every turn. The output items of responses whose
`store` is not `false` are kept in a bounded in-memory cache (16384 items / 32 MiB, on by
default) so `item_reference` resolves (see [The item cache](#the-item-cache)).

## Request fields

| Field | Behaviour |
|---|---|
| `model`, `input`, `instructions` | `input` is a string (one user message) or an array of items (see [Input items](#input-items)). `instructions` becomes a leading `system` message. |
| `max_output_tokens` | The chat path's `max_tokens`. Unset, it follows the server's unset default (256 tokens today, the chat endpoint's default). The response echoes the cap in force. |
| `temperature`, `top_p` | As on chat (`temperature` unset = `0.0`). |
| `reasoning.effort` | The chat endpoint's `reasoning_effort`. |
| `reasoning.summary` | Accepted; no summary is generated. Echoed as `null`. |
| `stream` | `true` sends OpenAI's event sequence (see [Streaming](#streaming)). `false`, `null` or absent: one JSON response. |
| `tools` | `function`, `custom` and `namespace` tools are served; hosted tools are dropped (see [Tools](#tools)). `[]` or absent = no tools. |
| `tool_choice` | `"auto"` and `"none"` (`"none"`: calls are not parsed). `"required"`, a named `function` / `custom` tool and `allowed_tools` are refused with `400` ("not enforced yet"); a hosted tool type is refused with `400`. |
| `parallel_tool_calls` | `false` keeps only the first call the model makes. Echoed (absent = `true`). |
| `max_tool_calls` | A non-negative integer or `null` (else `400`), echoed; no effect (it caps hosted tool calls, and none run here). |
| `store` | `true` or absent (OpenAI's default is `true`): the response's output items are kept in the [item cache](#the-item-cache). `false`: nothing is kept. Echoed (`true` when absent). |
| `include` | Each value must be one of OpenAI's `include` values, else `400`. `reasoning.encrypted_content` adds `encrypted_content` to reasoning items; the others add nothing here. |
| `metadata` | Echoed, held to OpenAI's limits (an object of at most 16 string pairs, keys up to 64 characters, values up to 512); anything else is a `400`. |
| `text` | Only `{"format": {"type": "text"}}`; other `text` fields such as `verbosity` are ignored. |
| `truncation` | Only `"disabled"`: a prompt over the window is a `400` `context_length_exceeded`, never trimmed. |
| `background` | Only `false`. |
| `top_logprobs` | Only `0`. |

**Ignored**: `client_metadata`, `prompt_cache_key`, `prompt_cache_options`,
`prompt_cache_retention`, `safety_identifier`, `user`, `service_tier`, `text.verbosity`,
`stream_options`, `access_programs`, `context_management`, `moderation`, and any other
field the server does not know (as on the chat endpoint).

**Refused** with a `400` whose `param` names the field: `previous_response_id`,
`conversation`, `background: true`, `prompt`, `top_logprobs` above 0, `stop`, a
`text.format` other than `text`, `truncation` other than `disabled`, `file_id` images and
videos, `input_file` and `input_audio` message parts.

## Input items

`input` is read in order. A model's turn in the Responses API is several items (reasoning,
a message, tool calls); they are gathered back into one assistant chat turn.

| Item | Becomes |
|---|---|
| `message`, role `user` / `system` | That chat message. `developer` is served as `system`. A `system` right after a `system`, or a `user` right after a `user`, is merged into it, the texts joined by a blank line (so `developer, developer, user, user` reaches the template as `system, user`). |
| `message`, role `assistant` | The assistant turn's content. |
| `reasoning` | Replayed as `<think>{text}</think>` at the front of the assistant turn's content. The text is the item's `content[].text`, else Cascadia's own `encrypted_content` decoded. A reasoning item with neither (another provider's blob, a summary-only item) is skipped. |
| `function_call`, `custom_tool_call` | A tool call on the assistant turn (`call_id` is the call's id; a `namespace` is flattened as the tool was served; a custom call's `input` becomes `{"input": ...}` arguments). |
| `function_call_output`, `custom_tool_call_output` | A `tool` message. `output` is a string or text parts, joined. An image or file part is replaced by `[image omitted: tool output images are not supported]` / `[file omitted: tool output files are not supported]` and logged, so a Codex `view_image` result does not fail the request (and every later one that resends it). |
| `item_reference` | Replaced by the cached item with that `id` (see [The item cache](#the-item-cache)), then read as above. An item with an `id` and no `type`, `role` or `content` is an `item_reference`. |
| `compaction`, `compaction_summary`, `compaction_trigger` | `400`: remote compaction is not supported; compact locally. |

Any other item type is a `400` naming `input[i]`, as is a call or call output without a
`call_id`.

## Tools

- **`function`**: served as a chat tool, unchanged.
- **`namespace`**: its nested `function` and `custom` tools are served flattened as
  `<namespace>__<name>` (the shape Codex names its MCP tools in). A call to one comes back
  with its own `name` and the `namespace` field. A namespace may hold only `function` and
  `custom` tools.
- **`custom`** (freeform): served as a function taking one string parameter `input`. A
  `grammar` format is written into the tool's description for the model; it is not enforced
  (there is no constrained decoding on this path). The call comes back as a
  `custom_tool_call` with that `input`.
- **Tools this server cannot serve**: OpenAI-hosted ones such as `web_search*`,
  `file_search`, `code_interpreter`, `computer_use*`, `image_generation` and `mcp`, and the
  client-side `shell`, `local_shell`, `apply_patch` and `tool_search` types. They are dropped
  with one warning in the server log and left out of the echoed `tools`; the rest of the
  request is served.

An unknown tool type, a tool without a `name`, or two tools served under the same name are
refused with a `400`. Tool calls are parsed out of the model's output as the chat endpoint
parses them. The model's chat template must render the tools: with no template, a tools
request is a `400`, or a `500` on `ov-genai` with thinking on (`sycl-llama` templates them
itself). A malformed call stays message text. A call to a name not in `tools` comes back as
a plain `function_call` under that name.

## Output

`output` holds, in order: a reasoning item (if the model reasoned), a message (if there is
answer text), then one item per tool call. Ids are `rs_`, `msg_`, `fc_` and `ctc_` plus a
random UUID; a call's `call_id` is the parser's `call_...`. `status` is `completed`;
`incomplete` (with `incomplete_details.reason: "max_output_tokens"`) at the token cap; or
`cancelled` after `/v1/cancel/<response id>` (the response id is the task id). An empty
generation returns `output: []`.

### Reasoning items

With thinking on, text the model writes inside `<think>...</think>` becomes a `reasoning`
item (`content: [{"type": "reasoning_text", ...}]`, `summary: []`) instead of message text,
and its share of the output tokens is reported in
`usage.output_tokens_details.reasoning_tokens`. A reply that never opens `<think>` is
returned unchanged as message text.

Cascadia tells whether the model starts inside `<think>` (Qwen3-2507-style templates open it
in the generation prompt) from the rendered prompt. Where the engine renders the prompt
itself, a probe of the chat template at load decides. Limits:

- `sycl-llama`: llama-server moves reasoning out of the text into `reasoning_content`; for
  `/v1/responses` the engine forwards it as text inside `<think>...</think>`, ahead of the
  answer, so it becomes a reasoning item. `/v1/chat/completions` still drops it. A replayed
  reasoning item goes back to llama-server as the assistant turn's `reasoning_content`,
  except on a turn with both text and tool calls when the template refuses that (gpt-oss) or
  cannot be read (no template in the GGUF, or one set with `--chat-template*` in the llama
  args); there it is left out.
- On engine-rendered paths whose template cannot be probed (no chat template loaded, or a
  probe render error), a prompt-opened model's reasoning stays in the message text.

With `include: ["reasoning.encrypted_content"]`, each finished reasoning item carries
`encrypted_content`. It is **not encrypted**: it is the reasoning text in base64 behind a
`cascadia.v1.` prefix, an opaque blob for clients (like Codex) that replay reasoning from
it. Cascadia decodes its own blobs when they come back in `input`; anyone holding the blob
can read it.

### Streaming

```
response.created, response.in_progress
reasoning:  output_item.added, content_part.added, reasoning_text.delta*, reasoning_text.done,
            content_part.done, output_item.done
message:    output_item.added, content_part.added, output_text.delta*, output_text.done,
            content_part.done, output_item.done
each call:  output_item.added, function_call_arguments.delta, function_call_arguments.done,
            output_item.done (custom: custom_tool_call_input.delta / .done)
response.completed | response.incomplete | response.failed
```

Each event has an `event:` line and a `data:` payload with its `type` and a
`sequence_number` counting from 0; there is no `[DONE]`. A call's arguments arrive in one
delta, once the call is parsed. The last event carries the full response with every item. A
cancelled response ends on `response.incomplete` with `status: "cancelled"`. An engine
failure after the stream started ends on `response.failed`, with the items sent so far. The
`error` stream event is never sent: a failure before the stream starts is an HTTP error.

## The item cache

`@ai-sdk/openai` (which sends `store: true` by default) replays the previous step's items
as `item_reference`s, by id. So the output items of every response whose `store` is not
`false` are kept in process memory, written before the response body or its last event, so
the client's next request finds them. The cache holds at most 16384 items and 32 MiB,
evicting the least recently used first (a lookup by `item_reference` counts as a use, so
the earliest items of a conversation that keeps referencing them stay); failed responses
and streams the client dropped are not kept.
Items are found only by their random id; there is no per-caller scoping.

The cache is lost on restart and items are evicted under load, so an `item_reference` can
miss. A miss is a `400` naming `input[i]`: `Item with id '<id>' not found (this server keeps
stored items in memory only; it may have restarted or evicted it; send the item itself, or
use store: false)`. For long AI SDK chats, turn it off so the SDK sends the items
themselves:

```js
await generateText({ model: openai.responses("qwen3-8b"), prompt,
  providerOptions: { openai: { store: false } } });
```

References expanded in one request count against the prompt byte cap (`--api-max-body-mb`);
past it the request is a `413` `prompt_too_large`.

## ov-genai with thinking on

`ov-genai` normally applies its own chat template when thinking is on, and that template
sees only the last user message. On `/v1/responses`, when the model has a chat template, the
API renders the full prompt itself instead (instructions, developer messages, history, tools),
and ov-genai uses it as is. On `/v1/chat/completions` the same happens when the request has
tools or a tool-call history; plain multi-turn chat without tools keeps the engine's own
template. If the API's render fails, ov-genai falls back to its own template (which sees
only the last user message), and a request with tools is then a `500`.

On an NPU, ov-genai's pipeline takes at most `MAX_PROMPT_LEN` prompt tokens (1024 unless
`--npu-max-prompt-len` sets it). A longer prompt is a `400` `context_length_exceeded` naming
that limit. A Codex request (its instructions and tools) is usually longer than 1024 tokens,
so serve Codex from the GPU or CPU, or raise `--npu-max-prompt-len`.

## Codex CLI

Point Codex at Cascadia with a custom provider in `~/.codex/config.toml`:

```toml
model = "qwen3-8b"
model_provider = "cascadia"
model_context_window = 32768
model_auto_compact_token_limit = 24000
[model_providers.cascadia]
name = "cascadia"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
capabilities = { remote_compaction = "unsupported" }
```

Set `model` to the served model's id, `model_context_window` to its context window and
`model_auto_compact_token_limit` below it. `remote_compaction = "unsupported"` makes Codex
compact the conversation locally (the server refuses `compaction_trigger` items).

- Codex never sends `max_output_tokens`, so each turn is capped at the server's unset
  default (256 tokens today). A long Codex turn (a big reasoning block, a long patch) can end
  `incomplete` at that cap.
- The model sees only Codex's `function`, `namespace` and `custom` tools. Its `web_search`
  tool (and any other type in the list above) is dropped, logged once per request. Its MCP
  tools arrive as namespaces and are served as `<namespace>__<name>`.
- A tool output with a screenshot (`view_image`) is replaced by a note in the prompt. On a
  server without image input the request body still carries the image, so raise
  `--api-max-body-mb` above its size, or every request that resends it is a `413`.

## Using the mapping from another crate

`cascadia_api::responses_api` exposes this mapping without the engine, for a gateway that
serves the Responses API over its own OpenAI chat-completion backends. `validate_and_map`
runs every check `/v1/responses` runs, in the same order (the refused fields, `tools`,
`tool_choice`, `input`, then an empty conversation), and returns the chat messages, the
served tools (`tools.chat_tools`) and the checked `tool_choice`; `item_reference` items are
resolved through a closure, for example over the gateway's own `ItemCache`. A `Refusal`
gives the status and the error body the server sends (`status()`, `to_error_json()`).
Replayed reasoning arrives as a `<think>…</think>` prefix of the assistant turn's content;
`strip_replayed_reasoning` removes it for a backend that cannot take it, and
`ChatMessage::to_openai_json` writes a message as a chat request's `messages[]` entry.
`fold_system_turns` and `folds_to_try` retry a conversation whose template refuses a later
`system` turn.

On the way back, `OutputBuilder` splits the streamed text, with reasoning folded in as
`<think>…</think>`, into reasoning and message items, and `OutputBuilder::push_calls` takes
the backend's already parsed tool calls (build the builder with `tools_active: false`, so
text is not scanned for calls; text fed after the calls is ignored). `ResponseEvents`
renders the same SSE events, sequence numbers and terminal event as this server's stream,
and returns the final items and response object. `End::from_finish_reason` reads the chat
`finish_reason` (`stop`, `tool_calls`, `function_call`, `length`, `content_filter`) and
returns `None` for anything else; a backend error goes to `ResponseEvents::finish` as
`Some(error)`, which ends the stream with `response.failed`.

The gateway decides the chat request's sampling settings itself. The server maps them so:
`max_output_tokens` is `max_tokens` (unset: 256, the chat endpoint's default, echoed on the
response); `temperature` unset is `0.0`; `top_p` is passed through when set (echoed as `1.0`
when unset); `reasoning.effort` is the chat request's `reasoning_effort`. `ResponseShell`
echoes them by these rules, so a gateway that picks other defaults should send them to its
backends and keep the echo in mind.

```rust
use cascadia_api::responses_api::*;
use std::sync::Arc;

let req: ResponsesRequest = serde_json::from_slice(&body)?;
let cache = ItemCache::default();
let resolve = |id: &str, budget: usize| {
    cache.get_within(id, budget).map_err(|_| Refusal::too_large("input", MAX_BYTES))
};
let mapped = match validate_and_map(&req, &resolve, MAX_BYTES) {
    Ok(m) => m,
    Err(refusal) => return reply(refusal.status(), refusal.to_error_json()),
};
let messages: Vec<_> = mapped.messages.iter().map(ChatMessage::to_openai_json).collect();
// Send `messages`, `mapped.tools.chat_tools` and `mapped.tool_choice` to a backend, then:
let shell = ResponseShell::new(&req, new_response_id(), &mapped.tools, mapped.tool_choice);
let parallel = req.parallel_tool_calls.unwrap_or(true);
let mut builder =
    OutputBuilder::new(true, false, false, Arc::new(mapped.tools), parallel, mapped.wants_encrypted);
let mut events = ResponseEvents::new(shell);
send(events.start());
for delta in content_deltas {
    send(events.push(builder.feed(&delta, 1)));
}
send(events.push(builder.push_calls(tool_calls)));
let (end, error) = match End::from_finish_reason(&finish_reason) {
    Some(end) => (end, None),
    None => (End::Cancelled, Some(format!("backend finished with `{finish_reason}`"))),
};
let failed = error.is_some();
let (closing, items, _response) =
    events.finish(&mut builder, end, prompt_tokens, completion_tokens, error);
if !failed && req.store != Some(false) {
    cache.insert_all(&items); // before the terminal event is sent; a failed one is not kept
}
send(closing);
```

For a non-streaming reply, the returned response object is the body when the backend
finished. A backend error is different there: the server answers a non-streaming request
whose generation failed with an HTTP 5xx error body, not a `failed` response object, so a
gateway that matches it should do the same rather than send `finish`'s object.
`crates/cascadia-api/tests/responses_api.rs` drives the streaming flow end to end.

## Known limitations

- Codex sends `developer` messages in the middle of a history; they become `system` turns
  after an assistant or tool turn. Templates that take a leading system turn only (Gemma 3,
  Mistral) or none at all (Gemma 2) refuse them, so when the loaded template refuses a
  request's system turns, they are folded into user turns where they stand: into the next
  user turn, their text first, or as a user turn of their own before the next assistant turn,
  or at the end when nothing follows. The later ones are folded first, every one if that is
  not enough. The model then reads them as user text, not as system instructions. Templates
  that take them (Gemma 4, Qwen) see them unchanged. `/v1/chat/completions` does not fold.
  On `sycl-llama`, which renders the GGUF's own template, the decision is made with the
  template the GGUF carries (`tokenizer.chat_template`, read at startup; the log says whether
  one was found); a GGUF without one, or a server whose `--llama-args` set the template
  (`--chat-template*`), is not folded. A request that template refuses however it is folded
  (Gemma 3 has no tool role, so a history with tool outputs) is a `400` naming the template's
  message, as where the API renders, rather than a failure once the stream has started.
- `tool_choice` cannot force a call yet, and structured output (`text.format`) is not
  served.
- `reasoning.summary` produces no summary; reasoning is returned whole.

## Testing

To try the API without a model, run the mock engine with a fixture whose chat template
renders tools:

```bash
cascadia run tests/fixtures/mock-tools-model --engine mock
```

The model id is `mock-tools-model`. It echoes by default; as a test aid, a user turn that
contains `__mock_tool_call__` on a request with tools gets one scripted call of the first
tool; the turn after its `function_call_output` gets the fixed answer
`mock: got <first 40 characters of the output>`; and a later user turn without the sentinel,
after one that had it, gets `mock: ok`. So a client's tool loop runs end to end. It is not a
feature of real models.

`scripts/responses-clients-check.sh --bin target/debug/cascadia` runs the OpenAI Agents SDK,
`@ai-sdk/openai`, n8n and Codex CLI tool loops against this mock and ends `RESPONSES_CLIENTS: ALL PASS`.

`scripts/responses-live-check.py <base-url>` checks a running server, any engine: a text
answer, a tool call (plain and streamed), the turn after its output, Codex's two request
bodies, and a later `developer` turn. Every body and stream event is validated against the
pinned OpenAI spec, and no `<think>` marker may reach the answer text. It ends
`RESPONSES_LIVE: ALL PASS`. Add `--mock` against the mock engine. The tool checks need a
model that calls tools.
