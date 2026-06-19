use thiserror::Error;

/// Top-level error type for cascadia-types.
///
/// Engines and runners typically wrap these in their own error enums so
/// callers can pattern-match without inspecting nested chains.
#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid shard layout: {0}")]
    InvalidShard(String),

    #[error("invalid task: {0}")]
    InvalidTask(String),
}
