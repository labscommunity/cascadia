//! The OpenAI error envelope: `{"error":{"message","type","param","code"}}`.
//!
//! Every refusal this crate sends is built here, so the shape stays the same
//! on every route. Another server that embeds these routes can call
//! [`error_body`] to answer its own refusals in the same shape.

use axum::http::StatusCode;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
