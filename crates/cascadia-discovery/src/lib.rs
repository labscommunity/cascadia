//! Zero-config mDNS peer discovery.
//!
//! Mirrors `cascadia/discovery/__init__.py`. Each node publishes a service
//! of type `_cascadia._tcp.local.` whose TXT record carries node id,
//! namespace, advertised device, available memory, and supported engines.
//! Discovered peers are added to the shared [`Topology`] graph.

use std::collections::HashMap;
use std::net::{IpAddr, UdpSocket};
use std::time::Duration;

use cascadia_topology::{NodeInfo, Topology};
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use parking_lot::Mutex;
use thiserror::Error;
use tracing::{debug, info, warn};

pub const SERVICE_TYPE: &str = "_cascadia._tcp.local.";

/// Debounce window for browse Removed/Resolved pairs. Must be shorter than
/// the worker's --settle (>=1s) so a truly-dead peer clears before commit,
/// and long enough to absorb a re-register's Removed->Resolved gap.
const DEBOUNCE_WINDOW: Duration = Duration::from_millis(750);

/// Debounce for browse Removed/Resolved: a Removed followed by a Resolved for
/// the same id within the window is a TXT refresh (keep membership), not a leave.
struct RemoveDebounce {
    window: Duration,
    pending: Mutex<HashMap<String, std::time::Instant>>,
}

impl RemoveDebounce {
    fn new(window: Duration) -> Self {
        Self {
            window,
            pending: Mutex::new(HashMap::new()),
        }
    }
    fn on_removed(&self, id: &str, now: std::time::Instant) {
        self.pending.lock().insert(id.to_string(), now);
    }
    /// Returns true if this Resolved cancelled a pending removal (refresh).
    fn on_resolved(&self, id: &str, now: std::time::Instant) -> bool {
        if let Some(t) = self.pending.lock().remove(id) {
            return now.duration_since(t) <= self.window;
        }
        false
    }
    /// Ids whose Removed timer expired at `now` (=> truly remove).
    fn expired(&self, now: std::time::Instant) -> Vec<String> {
        let mut g = self.pending.lock();
        let dead: Vec<String> = g
            .iter()
            .filter(|(_, t)| now.duration_since(**t) > self.window)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &dead {
            g.remove(id);
        }
        dead
    }
}

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

/// Best-effort local LAN IP — the address peers should use to reach this
/// host for the activation relay + latency probes.
///
/// 1. Default-route egress probe (no packets sent; just selects the
///    outbound interface). Picks the real default interface over docker/VPN
///    ones and works on any routable network.
/// 2. If that yields nothing usable (an **air-gapped / isolated LAN with no
///    default route** — the project's primary deployment), fall back to the
///    first private, non-loopback IPv4 from the interface list. Without this
///    the node would advertise `127.0.0.1` and every peer would probe its
///    own loopback, filling the latency matrix with bogus self-loops.
/// 3. Loopback as a last resort.
pub fn local_ip() -> IpAddr {
    if let Some(ip) = egress_route_ip() {
        if !ip.is_loopback() {
            return ip;
        }
    }
    if let Ok(ifaces) = if_addrs::get_if_addrs() {
        if let Some(ip) = ifaces
            .iter()
            .map(|i| i.ip())
            .find(|ip| matches!(ip, IpAddr::V4(v4) if v4.is_private()))
        {
            return ip;
        }
    }
    IpAddr::from([127, 0, 0, 1])
}

/// Select the source IP of the default route via a connected (but never
/// sent-on) UDP socket. `None` when there is no route (air-gapped).
fn egress_route_ip() -> Option<IpAddr> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket
        .set_read_timeout(Some(Duration::from_millis(500)))
        .ok();
    socket.connect("8.8.8.8:80").ok()?;
    socket.local_addr().ok().map(|sa| sa.ip())
}

/// Cap a free-form spec string before it goes into the mDNS TXT record.
/// TXT records have a tight size budget (each entry ≤255 bytes, and some
/// responders enforce a small single-packet total), so an unexpectedly
/// long CPU brand / OS string can't push the record past the limit and
/// get the whole advertisement dropped.
fn txt_trunc(s: &str) -> String {
    const MAX: usize = 64;
    if s.len() <= MAX {
        s.to_owned()
    } else {
        s.char_indices()
            .take_while(|(i, _)| *i < MAX)
            .map(|(_, c)| c)
            .collect()
    }
}

fn props_from_node(info: &NodeInfo) -> HashMap<String, String> {
    let mut p = HashMap::new();
    p.insert("node_id".into(), info.node_id.clone());
    p.insert("namespace".into(), info.namespace.clone());
    p.insert("device".into(), info.device.clone());
    p.insert("memory_mb".into(), info.memory_mb.to_string());
    p.insert("cpu_model".into(), txt_trunc(&info.cpu_model));
    p.insert("cpu_cores".into(), info.cpu_cores.to_string());
    p.insert("os".into(), txt_trunc(&info.os));
    p.insert("engines".into(), info.engines.join(","));
    if let Some(api_port) = info.api_port {
        p.insert("api_port".into(), api_port.to_string());
    }
    if let Some(h) = &info.model_hash {
        p.insert("model_hash".into(), h.clone());
    }
    if let Some(cs) = info.cluster_size {
        p.insert("cluster_size".into(), cs.to_string());
    }
    if let Some(d) = &info.ring_digest {
        p.insert("ring_digest".into(), d.clone());
    }
    p
}

fn node_from_service(srv: &ServiceInfo, expected_namespace: &str) -> Option<NodeInfo> {
    let props = srv.get_properties();
    let get = |k: &str| -> Option<String> {
        props
            .iter()
            .find(|p| p.key() == k)
            .map(|p| p.val_str().to_string())
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
    let cpu_model = get("cpu_model").unwrap_or_default();
    let cpu_cores = get("cpu_cores")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0u32);
    let os = get("os").unwrap_or_default();
    let api_port = get("api_port").and_then(|s| s.parse().ok());
    let engines = get("engines")
        .map(|s| {
            s.split(',')
                .filter(|p| !p.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    // Election fields. Guard hash shape (16 lowercase hex) so a corrupt or
    // hostile TXT value degrades to "absent" instead of poisoning matching.
    let hex16 = |s: String| -> Option<String> {
        (s.len() == 16
            && s.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()))
        .then_some(s)
    };
    let model_hash = get("model_hash").and_then(hex16);
    let cluster_size = get("cluster_size").and_then(|s| s.parse().ok());
    let ring_digest = get("ring_digest").and_then(hex16);

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
        api_port,
        namespace,
        device,
        memory_mb,
        cpu_model,
        cpu_cores,
        os,
        engines,
        model_hash,
        cluster_size,
        ring_digest,
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
    /// JoinHandle for the background browse-loop thread. Held so we
    /// don't lose the handle (preventing it from being silently
    /// orphaned on drop) and so close() can wait for the thread to
    /// exit cleanly. The loop polls with `recv_timeout` and breaks as
    /// soon as `receiver.is_disconnected()` is true, which happens
    /// once the daemon shuts down and drops its sender.
    browse_thread: Option<std::thread::JoinHandle<()>>,
}

impl DiscoveryService {
    pub fn new(topology: Topology, namespace: impl Into<String>) -> Self {
        Self {
            daemon: None,
            advertised: Mutex::new(None),
            topology,
            namespace: namespace.into(),
            browse_thread: None,
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
        info!(node_id = %node.node_id, port = node.port, "cascadia service registered");
        *self.advertised.lock() = Some(registered_name);

        let receiver = daemon.browse(SERVICE_TYPE)?;
        let topology = self.topology.clone();
        let namespace = self.namespace.clone();
        let handle = std::thread::spawn(move || {
            let debounce = RemoveDebounce::new(DEBOUNCE_WINDOW);
            loop {
                match receiver.recv_timeout(Duration::from_millis(200)) {
                    Ok(ServiceEvent::ServiceResolved(srv)) => {
                        if let Some(node) = node_from_service(&srv, &namespace) {
                            debounce.on_resolved(&node.node_id, std::time::Instant::now());
                            debug!(node_id = %node.node_id, "discovered");
                            topology.add_node(node);
                        }
                    }
                    Ok(ServiceEvent::ServiceRemoved(_, fullname)) => {
                        // The mDNS fullname is `<node_id>.<SERVICE_TYPE>`.
                        // Strip the service-type suffix rather than splitting
                        // on the first '.', so a node_id containing a dot
                        // (e.g. an FQDN-ish hostname) isn't truncated to the
                        // wrong id (which would leave the dead peer in the
                        // topology forever).
                        let node_id = fullname
                            .strip_suffix(SERVICE_TYPE)
                            .and_then(|s| s.strip_suffix('.'))
                            .unwrap_or(&fullname);
                        // Do NOT remove yet — debounce absorbs re-register flaps.
                        debounce.on_removed(node_id, std::time::Instant::now());
                    }
                    Ok(other) => debug!(?other, "discovery event"),
                    // Daemon shut down -> channel closed -> exit so close()'s
                    // join() returns (mdns-sd Receiver is flume; the concrete
                    // error type isn't re-exported, so use is_disconnected()).
                    Err(_) if receiver.is_disconnected() => break,
                    Err(_) => {} // timeout tick
                }
                // Sweep on EVERY iteration, not only on timeout.
                for id in debounce.expired(std::time::Instant::now()) {
                    debug!(%id, "peer removed (debounce expired)");
                    topology.remove_node(&id);
                }
            }
        });
        self.browse_thread = Some(handle);

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
        // Best-effort join of the browse loop. Once the daemon is
        // shut down its sender drops, so the loop's next
        // `recv_timeout` (at most 200ms out) observes
        // `receiver.is_disconnected()` and breaks. We don't fail
        // close() on a panicked thread.
        if let Some(handle) = self.browse_thread.take() {
            let _ = handle.join();
        }
    }

    /// Re-advertise this node's TXT (ring_digest / api_port changed) without a
    /// membership gap: mdns-sd re-registers the same fullname in place
    /// (HashMap replace + unsolicited announce). Peers that do see a
    /// Removed/Resolved pair absorb it via the browse-loop debounce.
    pub fn update_txt(&mut self, node: NodeInfo) -> Result<()> {
        let daemon = self
            .daemon
            .as_ref()
            .ok_or_else(|| Error::Mdns("discovery not started".into()))?;
        let host_name = format!("{}.local.", node.node_id);
        let info = ServiceInfo::new(
            SERVICE_TYPE,
            &node.node_id,
            &host_name,
            node.host.as_str(),
            node.port,
            Some(props_from_node(&node)),
        )?;
        daemon.register(info)?;
        Ok(())
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

    // Exercises the DECODE half (node_from_service) — previously untested —
    // so the encode/decode pair can't silently drift when a NodeInfo field
    // is added to one side only.
    #[test]
    fn props_round_trip_through_service_info() {
        let mut node = NodeInfo::new("host.lan-r0", "192.168.0.5", 9100);
        node.api_port = Some(8000);
        node.device = "GPU".into();
        node.memory_mb = 32000;
        node.cpu_model = "Intel(R) Core(TM) Ultra 7 258V".into();
        node.cpu_cores = 16;
        node.os = "Windows 11".into();
        node.engines = vec!["ov-runtime".into(), "mock".into()];

        let info = ServiceInfo::new(
            SERVICE_TYPE,
            &node.node_id,
            "host.local.",
            node.host.as_str(),
            node.port,
            Some(props_from_node(&node)),
        )
        .expect("build ServiceInfo");

        let decoded = node_from_service(&info, "default").expect("decode");
        assert_eq!(decoded.node_id, "host.lan-r0");
        assert_eq!(decoded.api_port, Some(8000));
        assert_eq!(decoded.device, "GPU");
        assert_eq!(decoded.memory_mb, 32000);
        assert_eq!(decoded.cpu_model, "Intel(R) Core(TM) Ultra 7 258V");
        assert_eq!(decoded.cpu_cores, 16);
        assert_eq!(decoded.os, "Windows 11");
        assert_eq!(decoded.engines, vec!["ov-runtime", "mock"]);
    }

    // A node_id containing a '.' must round-trip through the mDNS fullname
    // suffix-strip used by ServiceRemoved (regression: split('.').next()
    // truncated it).
    #[test]
    fn removed_fullname_strips_service_type_not_first_dot() {
        let node_id = "host.lan-r0";
        let fullname = format!("{node_id}.{SERVICE_TYPE}");
        let stripped = fullname
            .strip_suffix(SERVICE_TYPE)
            .and_then(|s| s.strip_suffix('.'))
            .unwrap_or(&fullname);
        assert_eq!(stripped, "host.lan-r0");
    }

    #[test]
    fn election_fields_round_trip_when_present() {
        let mut node = NodeInfo::new("host-192-168-0-5", "192.168.0.5", 9100);
        node.model_hash = Some("af63dc4c8601ec8c".into());
        node.cluster_size = Some(3);
        node.ring_digest = Some("cbf29ce484222325".into());
        let info = ServiceInfo::new(
            SERVICE_TYPE,
            &node.node_id,
            "host.local.",
            node.host.as_str(),
            node.port,
            Some(props_from_node(&node)),
        )
        .expect("build ServiceInfo");
        let decoded = node_from_service(&info, "default").expect("decode");
        assert_eq!(decoded.model_hash, Some("af63dc4c8601ec8c".into()));
        assert_eq!(decoded.cluster_size, Some(3));
        assert_eq!(decoded.ring_digest, Some("cbf29ce484222325".into()));
    }

    #[test]
    fn absent_election_fields_decode_to_none_not_empty() {
        // A worker (old binary / manual path) advertises none of the 3 fields.
        let node = NodeInfo::new("host-192-168-0-9", "192.168.0.9", 9100);
        let info = ServiceInfo::new(
            SERVICE_TYPE,
            &node.node_id,
            "host.local.",
            node.host.as_str(),
            node.port,
            Some(props_from_node(&node)),
        )
        .unwrap();
        let decoded = node_from_service(&info, "default").unwrap();
        assert_eq!(decoded.model_hash, None, "must be None, not Some(\"\")");
        assert_eq!(decoded.cluster_size, None, "must be None, not Some(0)");
        assert_eq!(decoded.ring_digest, None);
    }

    #[test]
    fn malformed_election_values_decode_to_none() {
        let node = NodeInfo::new("h", "127.0.0.1", 9100);
        let mut props = props_from_node(&node);
        props.insert("cluster_size".into(), "not-a-number".into());
        // Non-16-hex model_hash (e.g. hostile/corrupt TXT) is treated as absent.
        props.insert("model_hash".into(), "zz-not-hex".into());
        let info = ServiceInfo::new(
            SERVICE_TYPE,
            &node.node_id,
            "h.local.",
            node.host.as_str(),
            node.port,
            Some(props),
        )
        .unwrap();
        let decoded = node_from_service(&info, "default").unwrap();
        assert_eq!(decoded.cluster_size, None);
        assert_eq!(decoded.model_hash, None);
    }
}

#[cfg(test)]
mod debounce_tests {
    use super::*;
    use std::time::{Duration, Instant};
    #[test]
    fn resolved_within_window_is_a_refresh() {
        let d = RemoveDebounce::new(Duration::from_millis(500));
        let t0 = Instant::now();
        d.on_removed("n1", t0);
        assert!(d.on_resolved("n1", t0 + Duration::from_millis(100)));
        assert!(d.expired(t0 + Duration::from_secs(5)).is_empty());
    }
    #[test]
    fn removed_without_resolve_expires_to_true_removal() {
        let d = RemoveDebounce::new(Duration::from_millis(500));
        let t0 = Instant::now();
        d.on_removed("n2", t0);
        assert_eq!(
            d.expired(t0 + Duration::from_secs(1)),
            vec!["n2".to_string()]
        );
    }
}
