//! Typed decisions over decision models (docs/decisions.md): POST /v1/decisions (OpenAI format,
//! openai.rs) and POST /v1/systemone (Jev/SystemOne format, handler.rs), sharing one core.

pub mod schema;

use axum::http::StatusCode;

#[derive(Debug, thiserror::Error)]
pub enum DecisionError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    TooLong(String),
    #[error("this model does not serve decisions")]
    NotServed,
    #[error("{0}")]
    Engine(String),
}

impl DecisionError {
    pub fn status(&self) -> StatusCode {
        match self {
            DecisionError::BadRequest(_) => StatusCode::BAD_REQUEST,
            DecisionError::TooLong(_) => StatusCode::PAYLOAD_TOO_LARGE,
            DecisionError::NotServed => StatusCode::NOT_IMPLEMENTED,
            DecisionError::Engine(_) => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}
