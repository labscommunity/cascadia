//! Control envelopes for the version-one stage protocol. An Execute request or
//! Executed response is immediately followed by exactly `bytes` tensor-frame
//! bytes. Callers must bound the envelope before deserializing it.
use crate::{lease::LeaseId, worker::Status};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use uuid::Uuid;
pub const CONTROL_LIMIT: usize = 16 * 1024;
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    Catalog,
    Reserve {
        boot: Uuid,
        manifest: String,
        components: BTreeSet<String>,
        ttl_ms: u32,
    },
    Renew {
        lease: LeaseId,
        ttl_ms: u32,
    },
    Release {
        lease: LeaseId,
    },
    Execute {
        bytes: u32,
    },
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case", deny_unknown_fields)]
pub enum Response {
    Catalog { worker: Option<Status> },
    Reserved { lease: LeaseId },
    Done,
    Executed { bytes: u32 },
    Error { code: Code },
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Code {
    Invalid,
    Capacity,
    Lease,
    Replay,
    Unavailable,
}
impl From<&crate::Error> for Code {
    fn from(error: &crate::Error) -> Self {
        match error {
            crate::Error::Capacity => Self::Capacity,
            crate::Error::Lease => Self::Lease,
            crate::Error::Replay => Self::Replay,
            crate::Error::Quarantined | crate::Error::Io(_) => Self::Unavailable,
            _ => Self::Invalid,
        }
    }
}
