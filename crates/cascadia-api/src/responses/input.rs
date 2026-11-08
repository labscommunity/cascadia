//! Responses `input` items, mapped onto chat messages.

use super::*;

/// `input` as chat messages, `instructions` first as a `system` message.
/// Each item's content parts are rewritten into the chat endpoint's part
/// types and parsed by its own [`parse_content`], so text joining and image
/// placement match `/v1/chat/completions` exactly.
pub(super) fn input_messages(req: &ResponsesRequest) -> Result<Vec<ChatMessage>, Refusal> {
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
    for (i, item) in items.iter().enumerate() {
        let at = format!("input[{i}]");
        // `type` is optional on the shorthand message form `{role, content}`.
        let kind = item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        if kind != "message" {
            return Err(Refusal::new(
                &at,
                format!(
                    "input item type `{kind}` is not supported; only `message` items are \
                     accepted (no tools or stored items)"
                ),
            ));
        }
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
        out.push(ChatMessage {
            role: role.to_owned(),
            content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            parts,
        });
    }
    Ok(out)
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
