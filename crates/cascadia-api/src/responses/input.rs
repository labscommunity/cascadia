//! Responses `input` items, mapped onto chat messages (D9).
//!
//! The walk keeps one pending assistant turn: held reasoning text, the
//! assistant message's content and the tool calls after it. A Responses
//! output (reasoning, message, calls) is one chat assistant turn, so the
//! items that make one up are gathered and flushed as one message; a tool
//! output, or any non-assistant message, ends the turn.
//!
//! Reasoning is replayed as `<think>{text}</think>` at the front of the
//! turn's content (D8): its `content[].text`, else our own
//! `encrypted_content` decoded, else nothing (another provider's blob or a
//! summary-only item cannot be rendered, so it is dropped).

use super::*;
use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;

/// What starts our `encrypted_content`: the reasoning text in base64, not
/// encrypted (the client already holds it in `content`); the version lets
/// the format change.
pub(super) const ENCRYPTED_PREFIX: &str = "cascadia.v1.";

/// A tool output's image part, which the chat path cannot take as a tool
/// result. Codex's `view_image` sends one; a 400 would poison the history
/// Codex resends on every later turn.
const IMAGE_OMITTED: &str = "[image omitted: tool output images are not supported]";
const FILE_OMITTED: &str = "[file omitted: tool output files are not supported]";

/// `text` as our `encrypted_content`.
#[cfg_attr(not(test), allow(dead_code))] // written by the output builder
pub(super) fn encode_reasoning(text: &str) -> String {
    format!("{ENCRYPTED_PREFIX}{}", STANDARD.encode(text))
}

/// The reasoning text in one of our own `encrypted_content` blobs; `None`
/// for another provider's blob, bad base64 or bad UTF-8.
pub(super) fn decode_reasoning(blob: &str) -> Option<String> {
    let bytes = STANDARD.decode(blob.strip_prefix(ENCRYPTED_PREFIX)?).ok()?;
    String::from_utf8(bytes).ok()
}

/// The assistant turn being gathered.
#[derive(Default)]
struct Turn {
    /// Reasoning texts held for the turn, in order.
    reasoning: Vec<String>,
    /// The assistant message's `content` and `parts`.
    content: Option<(String, Vec<ContentPart>)>,
    calls: Vec<ToolCall>,
}

impl Turn {
    /// Push the turn, if any, as one assistant message: the held reasoning
    /// as a `<think>` prefix (no newline: templates split at `</think>`),
    /// then the content, then the calls. A non-empty `parts` gets the prefix
    /// as its first part too, as the image and part-trimming paths rebuild
    /// `content` from `parts`.
    fn flush(&mut self, out: &mut Vec<ChatMessage>) {
        let turn = std::mem::take(self);
        if turn.reasoning.is_empty() && turn.content.is_none() && turn.calls.is_empty() {
            return;
        }
        let (mut content, mut parts) = turn.content.unwrap_or_default();
        if !turn.reasoning.is_empty() {
            let think = format!("<think>{}</think>", turn.reasoning.join("\n"));
            content.insert_str(0, &think);
            if !parts.is_empty() {
                parts.insert(0, ContentPart::Text(think));
            }
        }
        out.push(ChatMessage {
            tool_calls: (!turn.calls.is_empty()).then_some(turn.calls),
            parts,
            ..text_message("assistant", content)
        });
    }
}

/// Looks a stored item up by id against the remaining byte budget: the item
/// and its stored size, `Ok(None)` if unknown, `Err` (a 413) if over budget.
pub(super) type Resolve<'a> = dyn Fn(&str, usize) -> Result<Option<(Value, usize)>, Refusal> + 'a;

/// `input` as chat messages, `instructions` first as a `system` message.
/// Each item's content parts are rewritten into the chat endpoint's part
/// types and parsed by its own [`parse_content`], so text joining and image
/// placement match `/v1/chat/completions` exactly.
///
/// `resolve` and `max_expand_bytes` serve `item_reference` expansion: not
/// taken yet, so such an item is refused like any unknown type.
pub(super) fn input_messages(
    req: &ResponsesRequest,
    resolve: &Resolve,
    max_expand_bytes: usize,
) -> Result<Vec<ChatMessage>, Refusal> {
    let _ = (resolve, max_expand_bytes);
    let mut out = Vec::new();
    if let Some(instructions) = &req.instructions {
        out.push(text_message("system", instructions.clone()));
    }
    let items = match &req.input {
        Value::String(s) => {
            out.push(text_message("user", s.clone()));
            return Ok(out);
        }
        Value::Array(items) => items,
        _ => {
            return Err(Refusal::new(
                "input",
                "`input` must be a string or an array of input items",
            ))
        }
    };
    let mut turn = Turn::default();
    for (i, item) in items.iter().enumerate() {
        let at = format!("input[{i}]");
        // `type` is optional on the shorthand message form `{role, content}`.
        let kind = item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        match kind {
            "message" => {
                let message = input_message(item, &at)?;
                if message.role == "assistant" {
                    if turn.content.is_some() || !turn.calls.is_empty() {
                        turn.flush(&mut out);
                    }
                    turn.content = Some((message.content, message.parts));
                } else {
                    turn.flush(&mut out);
                    out.push(message);
                }
            }
            // A reasoning item with no text we can render changes nothing.
            "reasoning" => {
                if let Some(text) = reasoning_text(item) {
                    // After content it opens the next turn; after calls
                    // whose outputs have not come yet, or after reasoning,
                    // it joins this one (OpenAI interleaves `reasoning,
                    // call, reasoning, call` in one output).
                    if turn.content.is_some() {
                        turn.flush(&mut out);
                    }
                    turn.reasoning.push(text);
                }
            }
            "function_call" | "custom_tool_call" => turn.calls.push(tool_call(item, kind, &at)?),
            "function_call_output" | "custom_tool_call_output" => {
                let output = tool_output(item, &at)?;
                turn.flush(&mut out);
                out.push(output);
            }
            _ => {
                return Err(Refusal::new(
                    &at,
                    format!(
                        "input item type `{kind}` is not supported; `message`, `reasoning`, \
                         `function_call`, `custom_tool_call` and their outputs are accepted \
                         (no stored items)"
                    ),
                ))
            }
        }
    }
    turn.flush(&mut out);
    Ok(out)
}

/// A `message` item as the chat message it is.
fn input_message(item: &Value, at: &str) -> Result<ChatMessage, Refusal> {
    let role = match item.get("role").and_then(Value::as_str) {
        Some(r @ ("user" | "assistant" | "system")) => r,
        // The chat templates know no `developer` turn; OpenAI ranks the
        // two alike.
        Some("developer") => "system",
        Some(other) => {
            return Err(Refusal::new(
                format!("{at}.role"),
                format!("role `{other}` is not supported"),
            ))
        }
        None => {
            return Err(Refusal::new(
                format!("{at}.role"),
                "input message is missing its `role`",
            ))
        }
    };
    let content = match item.get("content") {
        Some(Value::String(s)) => Value::String(s.clone()),
        Some(Value::Array(parts)) => Value::Array(
            parts
                .iter()
                .enumerate()
                .map(|(j, p)| chat_part(p, &format!("{at}.content[{j}]")))
                .collect::<Result<_, _>>()?,
        ),
        _ => {
            return Err(Refusal::new(
                format!("{at}.content"),
                "`content` must be a string or an array of content parts",
            ))
        }
    };
    let (content, parts) =
        parse_content(Some(content)).map_err(|e| Refusal::new(format!("{at}.content"), e))?;
    Ok(ChatMessage {
        parts,
        ..text_message(role, content)
    })
}

/// A reasoning item's text: its `content[].text` back to back, else our
/// own `encrypted_content` decoded; `None` when neither gives any text.
fn reasoning_text(item: &Value) -> Option<String> {
    let text: String = item
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect();
    if !text.is_empty() {
        return Some(text);
    }
    item.get("encrypted_content")
        .and_then(Value::as_str)
        .and_then(decode_reasoning)
        .filter(|t| !t.is_empty())
}

/// `item[key]` as a string, else a refusal naming `<at>.<key>`; an empty
/// one too when `non_empty` (an id or a name).
fn str_field<'a>(
    item: &'a Value,
    key: &str,
    at: &str,
    non_empty: bool,
) -> Result<&'a str, Refusal> {
    match item.get(key).and_then(Value::as_str) {
        Some(s) if !(non_empty && s.is_empty()) => Ok(s),
        _ => Err(Refusal::malformed(
            format!("{at}.{key}"),
            format!("this input item needs a string `{key}`"),
        )),
    }
}

/// A `function_call` or `custom_tool_call` item as a chat tool call: `id`
/// = `call_id`, the name flattened as the tool was served (D2), a custom
/// call's `input` as `{"input": …}` arguments (D3).
fn tool_call(item: &Value, kind: &str, at: &str) -> Result<ToolCall, Refusal> {
    let call_id = str_field(item, "call_id", at, true)?;
    let name = str_field(item, "name", at, true)?;
    let namespace = item
        .get("namespace")
        .and_then(Value::as_str)
        .filter(|ns| !ns.is_empty());
    let arguments = if kind == "custom_tool_call" {
        json!({"input": str_field(item, "input", at, false)?}).to_string()
    } else {
        str_field(item, "arguments", at, false)?.to_owned()
    };
    Ok(ToolCall {
        id: call_id.to_owned(),
        r#type: "function".into(),
        function: FunctionCall {
            name: ToolMap::flat(namespace, name),
            arguments,
        },
    })
}

/// A `function_call_output` or `custom_tool_call_output` item as a `tool`
/// message: `output` as sent, or its text parts back to back, an image or
/// file part replaced by a note.
fn tool_output(item: &Value, at: &str) -> Result<ChatMessage, Refusal> {
    let call_id = str_field(item, "call_id", at, true)?;
    let content = match item.get("output") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .enumerate()
            .map(|(j, p)| output_part(p, call_id, &format!("{at}.output[{j}]")))
            .collect::<Result<String, _>>()?,
        _ => {
            return Err(Refusal::malformed(
                format!("{at}.output"),
                "`output` must be a string or an array of content parts",
            ))
        }
    };
    Ok(ChatMessage {
        tool_call_id: Some(call_id.to_owned()),
        ..text_message("tool", content)
    })
}

/// One part of a tool output as text.
fn output_part<'a>(part: &'a Value, call_id: &str, at: &str) -> Result<&'a str, Refusal> {
    match part.get("type").and_then(Value::as_str).unwrap_or("") {
        "input_text" | "output_text" => part.get("text").and_then(Value::as_str).ok_or_else(|| {
            Refusal::malformed(format!("{at}.text"), "a text part needs a string `text`")
        }),
        "input_image" => {
            warn!(
                call_id,
                "responses: tool output image omitted; not supported"
            );
            Ok(IMAGE_OMITTED)
        }
        "input_file" => {
            warn!(
                call_id,
                "responses: tool output file omitted; not supported"
            );
            Ok(FILE_OMITTED)
        }
        "" => Err(Refusal::new(
            format!("{at}.type"),
            "content part is missing its `type`",
        )),
        other => Err(Refusal::new(
            format!("{at}.type"),
            format!(
                "tool output part type `{other}` is not supported; only `input_text`, \
                 `input_image` and `input_file` parts are accepted"
            ),
        )),
    }
}

pub(super) fn text_message(role: &str, content: String) -> ChatMessage {
    ChatMessage {
        role: role.to_owned(),
        content,
        name: None,
        tool_calls: None,
        tool_call_id: None,
        parts: Vec::new(),
    }
}

/// One Responses content part as the chat endpoint's `text` / `image_url` /
/// `video_url` part. `output_text` and `refusal` are a prior assistant turn's
/// text. `input_video` is OpenRouter's extension (OpenAI has no video input):
/// `video_url` is the URL string, or leniently `{url}` as in chat.
pub(super) fn chat_part(part: &Value, at: &str) -> Result<Value, Refusal> {
    let kind = part.get("type").and_then(Value::as_str).unwrap_or("");
    let field = |name: &str| part.get(name).and_then(Value::as_str);
    match kind {
        "input_text" | "output_text" | "refusal" => {
            let key = if kind == "refusal" { "refusal" } else { "text" };
            match field(key) {
                Some(text) => Ok(json!({"type": "text", "text": text})),
                None => Err(Refusal::new(
                    format!("{at}.{key}"),
                    format!("content part of type `{kind}` needs a string `{key}`"),
                )),
            }
        }
        "input_image" => match (field("image_url"), part.get("file_id")) {
            (Some(url), _) => Ok(json!({"type": "image_url", "image_url": {
                "url": url, "detail": field("detail")}})),
            (None, Some(f)) if !f.is_null() => Err(Refusal::new(
                format!("{at}.file_id"),
                "`file_id` images are not supported: this server keeps no files; \
                 send the image as a `data:` URL in `image_url`",
            )),
            _ => Err(Refusal::new(
                format!("{at}.image_url"),
                "content part of type `input_image` needs `image_url`",
            )),
        },
        "input_video" => {
            // A string is a video file; an array is a frame list (OVMS's
            // form), with `fps` and `detail` beside it or in `{url, ...}`. A
            // file's `fps` is not forwarded: the server samples it at its own
            // rate.
            let video_url = part.get("video_url");
            let inner = |key: &str| {
                video_url
                    .and_then(|v| v.get(key))
                    .filter(|f| !f.is_null())
                    .or(part.get(key))
            };
            let url = match video_url {
                Some(u @ (Value::String(_) | Value::Array(_))) => Some(u),
                Some(obj) => obj.get("url").filter(|u| u.is_string() || u.is_array()),
                None => None,
            };
            let fps = url.filter(|u| u.is_array()).and(inner("fps"));
            match (url, part.get("file_id")) {
                (Some(url), _) => Ok(json!({"type": "video_url", "video_url": {
                    "url": url, "detail": inner("detail"), "fps": fps}})),
                (None, Some(f)) if !f.is_null() => Err(Refusal::new(
                    format!("{at}.file_id"),
                    "`file_id` videos are not supported: this server keeps no files; \
                     send the video as a `data:` URL in `video_url`",
                )
                .with_code("video_file_id_not_supported")),
                _ => Err(Refusal::new(
                    format!("{at}.video_url"),
                    "content part of type `input_video` needs `video_url`",
                )),
            }
        }
        "input_file" => Err(Refusal::new(
            at,
            "file inputs (`input_file`) are not supported",
        )),
        "input_audio" => Err(Refusal::new(
            at,
            "audio inputs (`input_audio`) are not supported",
        )),
        "" => Err(Refusal::new(
            format!("{at}.type"),
            "content part is missing its `type`",
        )),
        other => Err(Refusal::new(
            format!("{at}.type"),
            format!(
                "content part type `{other}` is not supported; only `input_text`, \
                 `input_image` and `input_video` parts are accepted"
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk(input: Value) -> Result<Vec<ChatMessage>, Refusal> {
        let req: ResponsesRequest =
            serde_json::from_value(json!({"model": "m", "input": input})).unwrap();
        input_messages(&req, &|_, _| Ok(None), 1 << 20)
    }

    fn call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            r#type: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

    fn rs(text: &str) -> Value {
        json!({"type": "reasoning", "summary": [],
            "content": [{"type": "reasoning_text", "text": text}]})
    }

    fn fc(id: &str) -> Value {
        json!({"type": "function_call", "call_id": id, "name": "f", "arguments": "{}"})
    }

    fn out(id: &str) -> Value {
        json!({"type": "function_call_output", "call_id": id, "output": id})
    }

    /// (role, content, tool_calls, tool_call_id) of a message.
    type Shape<'a> = (&'a str, &'a str, Option<&'a Vec<ToolCall>>, Option<&'a str>);

    fn shape(m: &[ChatMessage]) -> Vec<Shape<'_>> {
        m.iter()
            .map(|m| {
                (
                    m.role.as_str(),
                    m.content.as_str(),
                    m.tool_calls.as_ref(),
                    m.tool_call_id.as_deref(),
                )
            })
            .collect()
    }

    /// (a) Codex's turn-2 tail: reasoning and its call are one assistant
    /// turn, the output a tool message.
    #[test]
    fn codex_tail_is_one_turn_then_a_tool_message() {
        let m = walk(json!([
            {"type": "reasoning", "id": "rs_1", "summary": [],
                "content": [{"type": "reasoning_text", "text": "I should list."}],
                "encrypted_content": "opaque-blob-1"},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "a.txt\n"},
        ]))
        .unwrap();
        let calls = vec![call("call_1", "exec_command", "{\"cmd\":\"ls\"}")];
        assert_eq!(
            shape(&m),
            [
                (
                    "assistant",
                    "<think>I should list.</think>",
                    Some(&calls),
                    None
                ),
                ("tool", "a.txt\n", None, Some("call_1")),
            ]
        );
    }

    /// (b) An assistant message and the calls after it are one turn.
    #[test]
    fn assistant_message_and_calls_are_one_turn() {
        let m = walk(json!([
            {"role": "assistant", "content": "checking"},
            fc("c1"),
            fc("c2"),
        ]))
        .unwrap();
        let calls = vec![call("c1", "f", "{}"), call("c2", "f", "{}")];
        assert_eq!(shape(&m), [("assistant", "checking", Some(&calls), None)]);
    }

    /// (c) A namespaced call is named as the tool was served.
    #[test]
    fn namespaced_call_is_flattened() {
        let m = walk(
            json!([{"type": "function_call", "call_id": "c", "namespace": "mcp__x",
            "name": "click", "arguments": "{}"}]),
        )
        .unwrap();
        assert_eq!(
            m[0].tool_calls.as_ref().unwrap()[0].function.name,
            "mcp__x__click"
        );
    }

    /// (d) A custom call's `input` becomes `{"input": …}` arguments; its
    /// output a tool message.
    #[test]
    fn custom_call_and_output() {
        let m = walk(json!([
            {"type": "custom_tool_call", "call_id": "c", "name": "apply_patch",
                "input": "*** Begin"},
            {"type": "custom_tool_call_output", "call_id": "c", "output": "Done"},
        ]))
        .unwrap();
        let calls = vec![call("c", "apply_patch", r#"{"input":"*** Begin"}"#)];
        assert_eq!(
            shape(&m),
            [
                ("assistant", "", Some(&calls), None),
                ("tool", "Done", None, Some("c")),
            ]
        );
    }

    /// (e) Our own blob is read back; a foreign or broken blob, or a
    /// summary-only item, is dropped.
    #[test]
    fn reasoning_blob_is_decoded_or_dropped() {
        let m = walk(json!([{"type": "reasoning", "summary": [],
            "encrypted_content": encode_reasoning("r")}]))
        .unwrap();
        assert_eq!(shape(&m), [("assistant", "<think>r</think>", None, None)]);
        let bad_utf8 = format!(
            "{ENCRYPTED_PREFIX}{}",
            base64::engine::general_purpose::STANDARD.encode([0xff, 0xfe])
        );
        for blob in ["opaque-blob-1", "cascadia.v1.!!!", bad_utf8.as_str()] {
            let m = walk(json!([{"type": "reasoning", "summary": [],
                "encrypted_content": blob}]))
            .unwrap();
            assert!(m.is_empty(), "{blob}: {m:?}");
        }
        let m = walk(json!([{"type": "reasoning", "id": "rs_1", "summary": []}])).unwrap();
        assert!(m.is_empty(), "{m:?}");
    }

    /// (f) A turn of only reasoning is flushed before the next user message.
    #[test]
    fn reasoning_only_turn_is_flushed_first() {
        let m = walk(json!([rs("r"), {"role": "user", "content": "next"}])).unwrap();
        assert_eq!(
            shape(&m),
            [
                ("assistant", "<think>r</think>", None, None),
                ("user", "next", None, None),
            ]
        );
    }

    /// (g) Output parts: text joined back to back; images and files are
    /// replaced by a note.
    #[test]
    fn output_parts_are_joined_and_media_omitted() {
        let output = |parts: Value| {
            walk(json!([{"type": "function_call_output", "call_id": "c", "output": parts}]))
                .unwrap()
                .remove(0)
                .content
        };
        assert_eq!(
            output(json!([{"type": "input_text", "text": "a"},
                {"type": "input_text", "text": "b"}])),
            "ab"
        );
        assert_eq!(
            output(json!([{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}])),
            "[image omitted: tool output images are not supported]"
        );
        assert_eq!(
            output(json!([{"type": "input_file", "file_data": "x", "filename": "a.pdf"}])),
            "[file omitted: tool output files are not supported]"
        );
    }

    /// (h) A call or output without `call_id` is refused naming it.
    #[test]
    fn missing_call_id_is_refused() {
        for item in [
            json!({"type": "function_call", "name": "f", "arguments": "{}"}),
            json!({"type": "custom_tool_call", "name": "f", "input": ""}),
            json!({"type": "function_call_output", "output": "x"}),
        ] {
            let err = walk(json!([item])).unwrap_err();
            assert_eq!(err.param, "input[0].call_id", "{item}");
            assert_eq!(err.status, StatusCode::BAD_REQUEST);
        }
    }

    /// (m) OpenAI's interleaved reasoning and calls stay one turn.
    #[test]
    fn interleaved_reasoning_and_calls_are_one_turn() {
        let m = walk(json!([
            rs("r1"),
            fc("A"),
            rs("r2"),
            fc("B"),
            out("A"),
            out("B")
        ]))
        .unwrap();
        let calls = vec![call("A", "f", "{}"), call("B", "f", "{}")];
        assert_eq!(
            shape(&m),
            [
                ("assistant", "<think>r1\nr2</think>", Some(&calls), None),
                ("tool", "A", None, Some("A")),
                ("tool", "B", None, Some("B")),
            ]
        );
    }

    /// (p) Reasoning items in a row join before the assistant's text.
    #[test]
    fn consecutive_reasoning_joins() {
        let m = walk(json!([rs("r1"), rs("r2"), {"role": "assistant", "content": "a"}])).unwrap();
        assert_eq!(
            shape(&m),
            [("assistant", "<think>r1\nr2</think>a", None, None)]
        );
    }

    /// (s) An array-form assistant message gets the prefix in `parts` too.
    #[test]
    fn think_prefix_edits_parts_too() {
        let m = walk(json!([rs("r"), {"role": "assistant", "content": [
            {"type": "output_text", "text": "x", "annotations": []}]}]))
        .unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].content, "<think>r</think>x");
        assert_eq!(m[0].parts[0], ContentPart::Text("<think>r</think>".into()));
        assert_eq!(m[0].parts[1], ContentPart::Text("x".into()));
    }

    /// A reasoning item with no renderable text is skipped: it neither
    /// flushes the assistant's content nor holds, so the call after it
    /// stays in the same turn.
    #[test]
    fn unrenderable_reasoning_keeps_the_turn() {
        let m = walk(json!([
            {"role": "assistant", "content": "a"},
            {"type": "reasoning", "id": "rs_x", "encrypted_content": "foreign"},
            fc("A"),
        ]))
        .unwrap();
        let calls = vec![call("A", "f", "{}")];
        assert_eq!(shape(&m), [("assistant", "a", Some(&calls), None)]);
    }

    #[test]
    fn reasoning_blob_round_trips() {
        let text = "pensée — 考える\n🙂";
        let blob = encode_reasoning(text);
        assert!(blob.starts_with(ENCRYPTED_PREFIX), "{blob}");
        assert_eq!(decode_reasoning(&blob).as_deref(), Some(text));
    }
}
