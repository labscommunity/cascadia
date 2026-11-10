//! The OpenAI error envelope: `{"error":{"message","type","param","code"}}`.
//!
//! Every refusal this crate sends is built here, so the shape stays the same
//! on every route; an engine error that reaches a stream after its 200 is
//! sent as an SSE event whose `error` is this same object. Another server that embeds these routes can call
//! [`error_body`] to answer its own refusals in the same shape.

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;

/// The envelope's `type`: OpenAI's split between the caller's fault
/// (`invalid_request_error`, any 4xx) and the server's (`server_error`).
pub fn error_type(status: StatusCode) -> &'static str {
    if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    }
}

/// `{"error":{"message","type","param","code"}}`. `param` names the request
/// field at fault and `code` a machine-readable reason; both are present as
/// `null` when unset, as OpenAI sends them.
pub fn error_body(
    status: StatusCode,
    message: &str,
    param: Option<&str>,
    code: Option<&str>,
) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": message,
            "type": error_type(status),
            "param": param,
            "code": code,
        }
    })
}

/// Seconds a "busy" 503 asks the client to wait before retrying. Slots free
/// as soon as a running request ends, so a short wait is enough.
pub const RETRY_AFTER_SECS: u32 = 1;

/// Sets `Retry-After: <secs>` on a 503 and leaves any other status alone.
/// The header is added in place, so the response's body and extensions
/// survive; another server can wrap its own capacity refusals with it.
pub fn with_retry_after(mut resp: Response, secs: u32) -> Response {
    if resp.status() == StatusCode::SERVICE_UNAVAILABLE {
        resp.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from(secs));
    }
    resp
}

/// The leading decimal digits of `s` as a count; `None` when `s` does not
/// start with a digit or the number does not fit.
fn leading_count(s: &str) -> Option<u32> {
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    s[..end].parse().ok()
}

/// `(prompt tokens, window)` read from an engine's over-window text (the
/// inner text of `EngineError::PromptTooLong`, without the `prompt too long: `
/// prefix).
///
/// The tokens are the digits right after the first `prompt is ` (an
/// `at least ` there is skipped). The window is the digits right after
/// whichever of `context is `, `window is ` or `holds ` comes first in the
/// text: the packed runtime's text names its region with `holds ` and later
/// explains it with `(per-slot context is (static_context - …`, so the
/// earliest marker wins, not the first in the list. Either is `None` when its
/// marker is missing or not followed by digits.
pub fn prompt_counts(detail: &str) -> (Option<u32>, Option<u32>) {
    let tokens = detail.find("prompt is ").and_then(|i| {
        let rest = &detail[i + "prompt is ".len()..];
        leading_count(rest.strip_prefix("at least ").unwrap_or(rest))
    });
    let window = ["context is ", "window is ", "holds "]
        .iter()
        .filter_map(|m| detail.find(m).map(|i| i + m.len()))
        .min()
        .and_then(|at| leading_count(&detail[at..]));
    (tokens, window)
}

/// OpenAI's wording for a prompt over the context window, which client
/// libraries match to recognise context overflow: every form starts with
/// "This model's maximum context length is". `noun` is the request field
/// that carries the prompt (`messages`, `prompt` or `input`). With a known
/// `window` the message states it, and the prompt's `tokens` when known (a
/// prompt at or under the window, which engines refuse when it leaves no room
/// to generate, says so); without one it says the context is exceeded, with
/// the tokens when known, and no number right after "is" for a client to
/// read as the limit. A non-empty `detail` (the engine's own text) follows in
/// parentheses.
pub fn context_length_message(
    noun: &str,
    window: Option<u32>,
    tokens: Option<u32>,
    detail: &str,
) -> String {
    let mut msg = match (window, tokens) {
        (Some(n), Some(m)) if m > n => format!(
            "This model's maximum context length is {n} tokens. However, your {noun} resulted \
             in {m} tokens. Please reduce the length of the {noun}."
        ),
        (Some(n), Some(m)) => format!(
            "This model's maximum context length is {n} tokens. However, your {noun} resulted \
             in {m} tokens, which leaves no room for the completion. Please reduce the length \
             of the {noun}."
        ),
        (Some(n), None) => format!(
            "This model's maximum context length is {n} tokens. However, your {noun} is longer \
             than that. Please reduce the length of the {noun}."
        ),
        (None, Some(m)) => format!(
            "This model's maximum context length is exceeded: your {noun} resulted in {m} \
             tokens, which exceeds the available context size. Please reduce the length of the \
             {noun}."
        ),
        (None, None) => format!(
            "This model's maximum context length is exceeded: your {noun} exceeds the \
             available context size. Please reduce the length of the {noun}."
        ),
    };
    if !detail.is_empty() {
        msg.push_str(&format!(" ({detail})"));
    }
    msg
}

/// Most of a non-JSON error body that is read to reuse as the message. Such
/// bodies are short text (a rejection's reason); a longer one is not read.
const MAX_TEXT_BODY_BYTES: usize = 64 * 1024;

/// Response middleware: under `/v1/`, a 4xx/5xx whose `Content-Type` is not
/// JSON (axum's empty 404 and 405, a path rejection's text, a fallback's
/// plain "not found") becomes the OpenAI envelope with the same status.
///
/// A 404 reads `Invalid URL ({METHOD} {path})` and a 405
/// `Method {METHOD} is not allowed for {path}`; any other status keeps its
/// body text, or the status's reason when the body is empty. The 405's
/// `Allow` header is added by axum after this runs, so it is not in the text,
/// but it still reaches the client. Headers other than `Content-Type` and
/// `Content-Length` are kept.
///
/// A JSON body passes through untouched, so applying the layer twice is
/// harmless, and a 1xx/2xx/3xx body (a stream included) is never read.
/// Apply it as the outermost layer, after any `merge`: the merged router's
/// fallback is only covered by a layer added afterwards.
pub async fn openai_error_bodies(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let method = req.method().clone();
    let path = req
        .extensions()
        .get::<axum::extract::OriginalUri>()
        .map(|u| u.path().to_owned())
        .unwrap_or_else(|| req.uri().path().to_owned());
    let resp = next.run(req).await;

    let status = resp.status();
    let under_v1 = path == "/v1" || path.starts_with("/v1/");
    if !under_v1
        || !(status.is_client_error() || status.is_server_error())
        || is_json(resp.headers())
    {
        return resp;
    }

    let (mut parts, body) = resp.into_parts();
    let message = match status {
        StatusCode::NOT_FOUND => format!("Invalid URL ({method} {path})"),
        StatusCode::METHOD_NOT_ALLOWED => format!("Method {method} is not allowed for {path}"),
        _ => {
            let text = axum::body::to_bytes(body, MAX_TEXT_BODY_BYTES)
                .await
                .map(|b| String::from_utf8_lossy(&b).trim().to_owned())
                .unwrap_or_default();
            if text.is_empty() {
                status
                    .canonical_reason()
                    .unwrap_or("request failed")
                    .to_owned()
            } else {
                text
            }
        }
    };
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    let body = error_body(status, &message, None, None).to_string();
    Response::from_parts(parts, axum::body::Body::from(body))
}

/// `application/json` or a `+json` type, parameters ignored.
fn is_json(headers: &axum::http::HeaderMap) -> bool {
    let Some(ct) = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    let essence = ct
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    essence == "application/json" || essence.ends_with("+json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;

    #[test]
    fn server_errors_carry_null_param_and_code() {
        let v = error_body(
            StatusCode::SERVICE_UNAVAILABLE,
            "engine is busy",
            None,
            None,
        );
        assert_eq!(
            v,
            serde_json::json!({
                "error": {
                    "message": "engine is busy",
                    "type": "server_error",
                    "param": null,
                    "code": null,
                }
            })
        );
    }

    #[test]
    fn client_errors_carry_param_and_code() {
        let v = error_body(
            StatusCode::BAD_REQUEST,
            "bad tool_choice",
            Some("tool_choice"),
            Some("invalid_value"),
        );
        assert_eq!(
            v,
            serde_json::json!({
                "error": {
                    "message": "bad tool_choice",
                    "type": "invalid_request_error",
                    "param": "tool_choice",
                    "code": "invalid_value",
                }
            })
        );
    }

    #[test]
    fn type_follows_the_status_class() {
        assert_eq!(
            error_type(StatusCode::PAYLOAD_TOO_LARGE),
            "invalid_request_error"
        );
        assert_eq!(
            error_type(StatusCode::INTERNAL_SERVER_ERROR),
            "server_error"
        );
    }

    #[derive(Clone, Copy)]
    struct Marker;

    #[test]
    fn retry_after_is_set_on_a_503_in_place() {
        let mut resp = StatusCode::SERVICE_UNAVAILABLE.into_response();
        resp.extensions_mut().insert(Marker);
        let resp = with_retry_after(resp, RETRY_AFTER_SECS);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers()[header::RETRY_AFTER], "1");
        assert!(resp.extensions().get::<Marker>().is_some());
    }

    #[test]
    fn retry_after_is_not_set_on_other_statuses() {
        for status in [
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_REQUEST,
            StatusCode::OK,
        ] {
            let resp = with_retry_after(status.into_response(), RETRY_AFTER_SECS);
            assert!(
                resp.headers().get(header::RETRY_AFTER).is_none(),
                "{status}"
            );
        }
    }

    /// One row per engine's over-window wording, numbers filled in.
    #[test]
    fn prompt_counts_reads_every_engine_wording() {
        type Row = (&'static str, (Option<u32>, Option<u32>));
        let rows: &[Row] = &[
            // ov-genai, text
            (
                "prompt is 900 tokens but the model's context is 512; send a shorter prompt",
                (Some(900), Some(512)),
            ),
            // ov-genai, media
            (
                "prompt is at least 900 tokens with its images and videos expanded but the \
                 model's context is 512; send a shorter prompt (or fewer or smaller images)",
                (Some(900), Some(512)),
            ),
            // gemma4
            (
                "prompt is 900 tokens with its images and videos expanded; the model's window is \
                 512 (max_position_embeddings)",
                (Some(900), Some(512)),
            ),
            // packed runtime, in full: its tail says "context is" after "holds"
            (
                "prompt is 207 tokens but this worker's per-slot KV region holds \
                 127; raise --static-context or lower --packed-slots (per-slot context is \
                 (static_context - 1 - packed_prefix) / packed_slots)",
                (Some(207), Some(127)),
            ),
            // static runtime tree
            (
                "prompt is 2000 tokens with its images expanded but this static tree's KV window \
                 holds 1023 (static_context - 1); send a shorter prompt (or fewer or smaller \
                 images), or convert the tree with a larger context",
                (Some(2000), Some(1023)),
            ),
            // stateful runtime tree
            (
                "prompt is 5000 tokens but the model's context is 4096; send a \
                 shorter prompt (or fewer or smaller images)",
                (Some(5000), Some(4096)),
            ),
            // distributed speculative decoding
            (
                "prompt is 900 tokens; the target's window is 512 (max_position_embeddings)",
                (Some(900), Some(512)),
            ),
            // qwen36, text and media
            (
                "prompt is 900 tokens; the model's window is 512 (max_position_embeddings)",
                (Some(900), Some(512)),
            ),
            (
                "prompt is 900 tokens with its images; the model's window is 512",
                (Some(900), Some(512)),
            ),
            // mock engine
            (
                "prompt is 9000 tokens; the model's window is 8192",
                (Some(9000), Some(8192)),
            ),
            // a count without a window
            ("prompt is at least 5875 tokens", (Some(5875), None)),
            // a count too large to be one
            (
                "prompt is 99999999999 tokens; the model's window is 512",
                (None, Some(512)),
            ),
            // neither
            ("3 images do not fit the context", (None, None)),
            ("stage 1 hung up", (None, None)),
        ];
        for (detail, want) in rows {
            assert_eq!(prompt_counts(detail), *want, "{detail}");
        }
    }

    #[test]
    fn a_marker_without_digits_is_no_count() {
        assert_eq!(
            prompt_counts("prompt is long; the model's window is unknown"),
            (None, None)
        );
    }

    /// Every form, exact. All of them start with "This model's maximum
    /// context length is", the text released client libraries match.
    #[test]
    fn context_length_message_forms() {
        let rows = [
            (
                Some(8192),
                Some(9000),
                "This model's maximum context length is 8192 tokens. However, your messages \
                 resulted in 9000 tokens. Please reduce the length of the messages.",
            ),
            (
                Some(8192),
                Some(8192),
                "This model's maximum context length is 8192 tokens. However, your messages \
                 resulted in 8192 tokens, which leaves no room for the completion. Please \
                 reduce the length of the messages.",
            ),
            (
                Some(8192),
                Some(100),
                "This model's maximum context length is 8192 tokens. However, your messages \
                 resulted in 100 tokens, which leaves no room for the completion. Please \
                 reduce the length of the messages.",
            ),
            (
                Some(8192),
                None,
                "This model's maximum context length is 8192 tokens. However, your messages is \
                 longer than that. Please reduce the length of the messages.",
            ),
            (
                None,
                Some(5875),
                "This model's maximum context length is exceeded: your messages resulted in \
                 5875 tokens, which exceeds the available context size. Please reduce the \
                 length of the messages.",
            ),
            (
                None,
                None,
                "This model's maximum context length is exceeded: your messages exceeds the \
                 available context size. Please reduce the length of the messages.",
            ),
        ];
        for (window, tokens, want) in rows {
            assert_eq!(
                context_length_message("messages", window, tokens, ""),
                want,
                "{window:?} {tokens:?}"
            );
        }
        assert_eq!(
            context_length_message("input", None, None, ""),
            "This model's maximum context length is exceeded: your input exceeds the \
             available context size. Please reduce the length of the input."
        );
    }

    #[test]
    fn context_length_message_appends_the_detail() {
        assert_eq!(
            context_length_message(
                "messages",
                Some(127),
                Some(207),
                "prompt is 207 tokens but this worker's per-slot KV region holds 127"
            ),
            "This model's maximum context length is 127 tokens. However, your messages \
             resulted in 207 tokens. Please reduce the length of the messages. (prompt is 207 \
             tokens but this worker's per-slot KV region holds 127)"
        );
    }
}
