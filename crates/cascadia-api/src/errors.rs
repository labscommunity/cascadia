//! The OpenAI error envelope: `{"error":{"message","type","param","code"}}`.
//!
//! Every refusal this crate sends is built here, so the shape stays the same
//! on every route. Another server that embeds these routes can call
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
}
