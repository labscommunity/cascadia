//! Zero-config mDNS peer discovery.
//!
//! Mirrors `tahoma/discovery/__init__.py`. Each node publishes a service
//! of type `_tahoma._tcp.local.` whose TXT record carries node id,
//! namespace, advertised device, available memory, and supported engines.
//! Discovered peers are added to the shared [`Topology`] graph.

use std::collections::HashMap;
use std::net::{IpAddr, UdpSocket};
use std::time::Duration;

use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use parking_lot::Mutex;
use tahoma_topology::{NodeInfo, Topology};
use thiserror::Error;
use tracing::{debug, info, warn};

pub const SERVICE_TYPE: &str = "_tahoma._tcp.local.";

#[derive(Debug, Error)]
pub enum Error {
    #[error("mdns error: {0}")]
    Mdns(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<mdns_sd::Error> for Error {
    fn from(e: mdns_sd::Error) -> Self {
        Self::Mdns(e.to_string())
    }
}

/// Best-effort local LAN IP: the address this host uses to reach the
/// gateway. Falls back to `127.0.0.1` if no route can be probed.
pub fn local_ip() -> IpAddr {
    let socket = match UdpSocket::bind("0.0.0.0:0") {
        Ok(s) => s,
        Err(_) => return IpAddr::from([127, 0, 0, 1]),
    };
    socket
        .set_read_timeout(Some(Duration::from_millis(500)))
        .ok();
    if socket.connect("8.8.8.8:80").is_err() {
        return IpAddr::from([127, 0, 0, 1]);
    }
    socket
        .local_addr()
        .map(|sa| sa.ip())
        .unwrap_or_else(|_| IpAddr::from([127, 0, 0, 1]))
}

fn props_from_node(info: &NodeInfo) -> HashMap<String, String> {
    let mut p = HashMap::new();
    p.insert("node_id".into(), info.node_id.clone());
    p.insert("namespace".into(), info.namespace.clone());
    p.insert("device".into(), info.device.clone());
    p.insert("memory_mb".into(), info.memory_mb.to_string());
    p.insert("engines".into(), info.engines.join(","));
    p
}

fn node_from_service(srv: &ServiceInfo, expected_namespace: &str) -> Option<NodeInfo> {
    let props = srv.get_properties();
    let get = |k: &str| -> Option<String> {
        props.iter().find(|p| p.key() == k).map(|p| p.val_str().to_string())
    };

    let node_id = get("node_id")?;
    let namespace = get("namespace").unwrap_or_else(|| "default".into());
    if namespace != expected_namespace {
        return None;
    }
    let device = get("device").unwrap_or_else(|| "CPU".into());
    let memory_mb = get("memory_mb")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0u64);
    let engines = get("engines")
        .map(|s| {
            s.split(',')
                .filter(|p| !p.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    let host = srv
        .get_addresses()
        .iter()
        .next()
        .map(|a| a.to_string())
        .unwrap_or_else(|| "127.0.0.1".into());
    let port = srv.get_port();

    Some(NodeInfo {
        node_id,
        host,
        port,
        namespace,
        device,
        memory_mb,
        engines,
        last_seen: 0.0,
    })
}

/// Manages an mDNS daemon, advertises this node, and merges discovered
/// peers into a shared `Topology`.
pub struct DiscoveryService {
    daemon: Option<ServiceDaemon>,
    advertised: Mutex<Option<String>>,
    topology: Topology,
    namespace: String,
}

impl DiscoveryService {
    pub fn new(topology: Topology, namespace: impl Into<String>) -> Self {
        Self {
            daemon: None,
            advertised: Mutex::new(None),
            topology,
            namespace: namespace.into(),
        }
    }

    /// Start the mDNS daemon, register `node`, and begin browsing.
    pub fn start(&mut self, node: NodeInfo) -> Result<()> {
        let daemon = ServiceDaemon::new()?;
        let host_name = format!("{}.local.", node.node_id);
        let info = ServiceInfo::new(
            SERVICE_TYPE,
            &node.node_id,
            &host_name,
            node.host.as_str(),
            node.port,
            Some(props_from_node(&node)),
        )?;
        let registered_name = info.get_fullname().to_string();
        daemon.register(info)?;
        info!(node_id = %node.node_id, port = node.port, "tahoma service registered");
        *self.advertised.lock() = Some(registered_name);

        let receiver = daemon.browse(SERVICE_TYPE)?;
        let topology = self.topology.clone();
        let namespace = self.namespace.clone();
        std::thread::spawn(move || {
            for event in receiver.iter() {
                match event {
                    ServiceEvent::ServiceResolved(srv) => {
                        if let Some(node) = node_from_service(&srv, &namespace) {
                            debug!(node_id = %node.node_id, "discovered");
                            topology.add_node(node);
                        }
                    }
                    ServiceEvent::ServiceRemoved(_, fullname) => {
                        if let Some(node_id) = fullname.split('.').next() {
                            debug!(%node_id, "peer removed");
                            topology.remove_node(node_id);
                        }
                    }
                    other => debug!(?other, "discovery event"),
                }
            }
        });

        self.daemon = Some(daemon);
        Ok(())
    }

    pub fn close(&mut self) {
        if let Some(daemon) = self.daemon.take() {
            if let Some(name) = self.advertised.lock().take() {
                let _ = daemon.unregister(&name);
            }
            if let Err(e) = daemon.shutdown() {
                warn!(error = ?e, "mdns shutdown failed");
            }
        }
    }
}

impl Drop for DiscoveryService {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_ip_is_callable() {
        // Just verify it returns *something* without panicking; the
        // value is environment-dependent.
        let _ip = local_ip();
    }

    #[test]
    fn props_from_node_serialises_engines() {
        let mut info = NodeInfo::new("n1", "127.0.0.1", 9100);
        info.engines = vec!["ov-genai".into(), "mock".into()];
        let p = props_from_node(&info);
        assert_eq!(p.get("node_id"), Some(&"n1".to_string()));
        assert_eq!(p.get("engines"), Some(&"ov-genai,mock".to_string()));
    }
}
