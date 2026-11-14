//! Diffusion contracts shared by standalone runtimes and authenticated fleet
//! integrations. Network authentication and placement belong to the caller;
//! artifact identity, tensor bounds, and execution ownership do not.

pub mod lease;
pub mod manifest;
pub mod wire;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid diffusion contract: {0}")]
    Invalid(String),
    #[error("diffusion worker is at capacity")]
    Capacity,
    #[error("diffusion lease is unknown, expired, or owned by another caller")]
    Lease,
    #[error("diffusion call is replayed or out of order")]
    Replay,
    #[error("diffusion worker must be restarted after an uncertain execution")]
    Quarantined,
    #[error("diffusion I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("diffusion JSON failed: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

pub mod rpc;
pub mod worker;
