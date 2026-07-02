//! Cluster topology graph.
//!
//! Mirrors the richer parts of `cascadia/shared/topology.py`. Nodes are
//! advertised by discovery; edges store *measured* latency + bandwidth
//! (1,200+ placement experiments showed latency drives placement on
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

/// One node's advertised capacity **+ election advertisement**. Populated by
/// discovery + heartbeats. The `model_hash`/`cluster_size`/`ring_digest` fields
/// carry mDNS auto-ring (#89) negotiation state, not capacity — see
/// docs/superpowers/specs/2026-07-01-mdns-auto-ring-design.md.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NodeInfo {
    pub node_id: String,
    pub host: String,
    /// Activation-relay / latency-probe port (the port peers connect to for
    /// the pipeline transport and RTT probes).
    pub port: u16,
    /// OpenAI-compatible API / dashboard port, set on coordinator nodes that
    /// serve `--api`. `None` for relay-only stages. The dashboard shows this
    /// as the node's reachable address rather than the relay `port`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_port: Option<u16>,
    #[serde(default = "default_namespace")]
    pub namespace: String,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default)]
    pub memory_mb: u64,
    /// CPU brand string (e.g. "Intel(R) Core(TM) Ultra 7 258V"). Empty if
    /// unknown. Advertised so the dashboard can show per-node system specs.
    #[serde(default)]
    pub cpu_model: String,
    /// Logical CPU count. 0 if unknown.
    #[serde(default)]
    pub cpu_cores: u32,
    /// Operating system label (e.g. "Windows 11", "macOS 15.5"). Empty if
    /// unknown.
    #[serde(default)]
    pub os: String,
    #[serde(default)]
    pub engines: Vec<String>,
    /// FNV-1a hash of the worker's `--model` string (auto-ring model match).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_hash: Option<String>,
    /// Operator's `--cluster-size` (auto-ring participant marker). `None` =
    /// not an auto-ring participant (manual/dashboard-only worker).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_size: Option<u32>,
    /// Phase-2 agreement digest of this node's resolved member ordering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ring_digest: Option<String>,
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
            api_port: None,
            namespace: default_namespace(),
            device: default_device(),
            memory_mb: 0,
            cpu_model: String::new(),
            cpu_cores: 0,
            os: String::new(),
            engines: Vec::new(),
            model_hash: None,
            cluster_size: None,
            ring_digest: None,
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

    /// Refresh an existing node's `last_seen` without re-cloning the whole
    /// `NodeInfo`. Used by the self-heartbeat loop. No-op if the node isn't
    /// present (a peer we haven't discovered yet).
    pub fn touch(&self, node_id: &str) {
        let mut inner = self.inner.write();
        if let Some(n) = inner.nodes.get_mut(node_id) {
            n.last_seen = now_ts();
        }
    }

    pub fn remove_node(&self, node_id: &str) {
        let mut inner = self.inner.write();
        inner.nodes.remove(node_id);
        inner.edges.retain(|(s, d), _| s != node_id && d != node_id);
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

    /// Record a latency measurement WITHOUT clobbering a previously-measured
    /// `bandwidth_mbps` on the same edge. The latency-probe loop measures
    /// only RTT; using `measure(.., 0.0)` would reset bandwidth to 0 on
    /// every tick. Preserves the existing edge's bandwidth (0.0 for a new
    /// edge until a bandwidth probe sets it).
    pub fn record_latency(&self, src: impl Into<String>, dst: impl Into<String>, latency_ms: f64) {
        let key = (src.into(), dst.into());
        let mut inner = self.inner.write();
        let bandwidth_mbps = inner.edges.get(&key).map_or(0.0, |e| e.bandwidth_mbps);
        inner.edges.insert(
            key,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_defaults_election_fields_to_none() {
        let n = NodeInfo::new("n1", "127.0.0.1", 9100);
        assert_eq!(n.model_hash, None);
        assert_eq!(n.cluster_size, None);
        assert_eq!(n.ring_digest, None);
    }

    #[test]
    fn election_fields_omitted_from_json_when_none() {
        let n = NodeInfo::new("n1", "127.0.0.1", 9100);
        let j = serde_json::to_string(&n).unwrap();
        assert!(
            !j.contains("model_hash"),
            "None fields must be skipped: {j}"
        );
        assert!(!j.contains("ring_digest"), "{j}");
    }
}
