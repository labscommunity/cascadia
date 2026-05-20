use serde::{Deserialize, Serialize};

/// A progress event emitted while a shard is loading.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LoadProgress {
    pub bytes_loaded: u64,
    pub bytes_total: Option<u64>,
    pub message: String,
}

impl LoadProgress {
    pub fn message(msg: impl Into<String>) -> Self {
        Self {
            bytes_loaded: 0,
            bytes_total: None,
            message: msg.into(),
        }
    }

    pub fn ready() -> Self {
        Self {
            bytes_loaded: 1,
            bytes_total: Some(1),
            message: "ready".into(),
        }
    }
}
