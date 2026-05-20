use serde::{Deserialize, Serialize};

/// Network address of a single peer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PeerEndpoint {
    pub host: String,
    pub port: u16,
}

impl PeerEndpoint {
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
        }
    }
}

impl std::fmt::Display for PeerEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.host, self.port)
    }
}

/// A pipeline stage's view of its neighbours.
///
/// `upstream` sends activations to us (None on the first stage).
/// `downstream` receives activations from us (None on the last stage).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct PeerLayout {
    pub upstream: Option<PeerEndpoint>,
    pub downstream: Option<PeerEndpoint>,
}

impl PeerLayout {
    pub fn single_stage() -> Self {
        Self::default()
    }

    pub fn first_of(downstream: PeerEndpoint) -> Self {
        Self {
            upstream: None,
            downstream: Some(downstream),
        }
    }

    pub fn last_of(upstream: PeerEndpoint) -> Self {
        Self {
            upstream: Some(upstream),
            downstream: None,
        }
    }

    pub fn middle(upstream: PeerEndpoint, downstream: PeerEndpoint) -> Self {
        Self {
            upstream: Some(upstream),
            downstream: Some(downstream),
        }
    }
}
