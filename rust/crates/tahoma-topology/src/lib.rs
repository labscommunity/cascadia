//! Cluster topology graph.
//!
//! Mirrors the richer parts of `tahoma/shared/topology.py`. Nodes are
//! advertised by discovery; edges store *measured* latency + bandwidth
//! (per rainier's 1,200+ experiments showing latency drives placement on
//! Intel fleets). A missing edge means we never probed it — placement
//! should treat unmeasured edges pessimistically rather than as missing.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// One node's advertised capacity. Populated by discovery + heartbeats.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NodeInfo {
    pub node_id: String,
    pub host: String,
    pub port: u16,
    #[serde(default = "default_namespace")]
    pub namespace: String,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default)]
    pub memory_mb: u64,
    #[serde(default)]
    pub engines: Vec<String>,
    #[serde(default = "now_ts")]
    pub last_seen: f64,
}

fn default_namespace() -> String {
    "default".into()
}
fn default_device() -> String {
    "CPU".into()
}

impl NodeInfo {
    pub fn new(node_id: impl Into<String>, host: impl Into<String>, port: u16) -> Self {
        Self {
            node_id: node_id.into(),
            host: host.into(),
            port,
            namespace: default_namespace(),
            device: default_device(),
            memory_mb: 0,
            engines: Vec::new(),
            last_seen: now_ts(),
        }
    }
}

/// Per-link measurements; populated by occasional probes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EdgeMetrics {
    #[serde(default)]
    pub latency_ms: f64,
    #[serde(default)]
    pub bandwidth_mbps: f64,
    #[serde(default)]
    pub last_measured: f64,
}

/// In-memory directed graph of nodes + measured edges.
///
/// Thread-safe via interior `RwLock`. Cheap clone (`Arc` inside).
#[derive(Default, Clone)]
pub struct Topology {
    inner: Arc<RwLock<TopologyInner>>,
}

#[derive(Default, Debug)]
struct TopologyInner {
    nodes: HashMap<String, NodeInfo>,
    edges: HashMap<(String, String), EdgeMetrics>,
}

impl Topology {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&self, mut info: NodeInfo) {
        info.last_seen = now_ts();
        let mut inner = self.inner.write();
        inner.nodes.insert(info.node_id.clone(), info);
    }

    pub fn remove_node(&self, node_id: &str) {
        let mut inner = self.inner.write();
        inner.nodes.remove(node_id);
        inner
            .edges
            .retain(|(s, d), _| s != node_id && d != node_id);
    }

    pub fn measure(
        &self,
        src: impl Into<String>,
        dst: impl Into<String>,
        latency_ms: f64,
        bandwidth_mbps: f64,
    ) {
        let mut inner = self.inner.write();
        inner.edges.insert(
            (src.into(), dst.into()),
            EdgeMetrics {
                latency_ms,
                bandwidth_mbps,
                last_measured: now_ts(),
            },
        );
    }

    pub fn nodes(&self) -> Vec<NodeInfo> {
        self.inner.read().nodes.values().cloned().collect()
    }

    pub fn node(&self, node_id: &str) -> Option<NodeInfo> {
        self.inner.read().nodes.get(node_id).cloned()
    }

    pub fn edges(&self) -> Vec<((String, String), EdgeMetrics)> {
        self.inner
            .read()
            .edges
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    pub fn edge(&self, src: &str, dst: &str) -> Option<EdgeMetrics> {
        self.inner
            .read()
            .edges
            .get(&(src.to_string(), dst.to_string()))
            .cloned()
    }

    /// Drop nodes we haven't heard from in `max_age_s` seconds. Returns
    /// the dropped node IDs.
    pub fn expire_stale(&self, max_age_s: f64) -> Vec<String> {
        let cutoff = now_ts() - max_age_s;
        let mut inner = self.inner.write();
        let stale: Vec<String> = inner
            .nodes
            .iter()
            .filter(|(_, info)| info.last_seen < cutoff)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &stale {
            inner.nodes.remove(id);
            inner.edges.retain(|(s, d), _| s != id && d != id);
        }
        stale
    }
}
