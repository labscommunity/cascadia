//! `cascadia discover` — list Cascadia peers on the local network.
//!
//! This is the read-only half of zero-config clustering: it uses the
//! existing mDNS [`DiscoveryService`] to browse `_cascadia._tcp.local.`
//! for a few seconds and prints the peers it finds, including the
//! `host:port` you'd pass to another worker's `--next`.
//!
//! It deliberately does NOT auto-assign ranks, agree on a stage count,
//! or order peers into a pipeline — that full auto-ring formation is
//! tracked separately (see issue #52, follow-ups). Today `worker` still
//! takes explicit `--rank`/`--total`/`--next`; `discover` just removes
//! the "what's my peer's address?" guesswork.

use std::time::Duration;

use anyhow::Result;
use cascadia_discovery::{local_ip, DiscoveryService};
use cascadia_topology::{NodeInfo, Topology};
use clap::Parser;

/// Browse the LAN for Cascadia peers and print what's advertising.
#[derive(Parser, Debug, Clone)]
pub struct DiscoverArgs {
    /// Discovery namespace to browse. Peers in a different namespace are
    /// ignored (matches the worker's namespace partitioning).
    #[arg(long, default_value = "default")]
    pub namespace: String,

    /// How long to listen for peer announcements, in seconds. mDNS
    /// announces are fast (~2.5 s in practice); the default gives a
    /// comfortable margin without making the command feel slow.
    #[arg(long, default_value_t = 5)]
    pub timeout: u64,
}

pub async fn cmd_discover(args: DiscoverArgs) -> Result<()> {
    let topology = Topology::new();

    // We have to advertise to browse (the daemon registers + browses in
    // one shot). Advertise a clearly-labelled, short-lived probe node so
    // a human reading another box's discovery view can tell this wasn't a
    // real worker. It unregisters on close().
    let self_id = format!("discover-probe-{}", std::process::id());
    let mut self_node = NodeInfo::new(self_id.clone(), local_ip().to_string(), 0);
    self_node.namespace = args.namespace.clone();
    self_node.engines = vec!["(discover-probe)".into()];

    let mut svc = DiscoveryService::new(topology.clone(), args.namespace.clone());
    svc.start(self_node)
        .map_err(|e| anyhow::anyhow!("failed to start mDNS discovery: {e}"))?;

    println!(
        "Browsing {:?} for cascadia peers in namespace {:?} ({}s)...\n",
        cascadia_discovery::SERVICE_TYPE,
        args.namespace,
        args.timeout
    );
    tokio::time::sleep(Duration::from_secs(args.timeout)).await;

    // Exclude our own probe from the listing.
    let mut peers: Vec<NodeInfo> = topology
        .nodes()
        .into_iter()
        .filter(|n| n.node_id != self_id)
        .collect();
    peers.sort_by(|a, b| a.node_id.cmp(&b.node_id));

    svc.close();

    if peers.is_empty() {
        println!("No peers found.");
        println!();
        println!("If you expected peers, check that:");
        println!("  • the other workers are running and on the same LAN / subnet");
        println!("  • multicast/mDNS isn't blocked by a firewall or the network");
        println!(
            "  • they use the same --namespace (this run used {:?})",
            args.namespace
        );
        println!();
        println!("Note: workers do not yet advertise on the network automatically —");
        println!("auto-ring formation is tracked in issue #52. For now, wire peers");
        println!("manually with --listen / --next host:port.");
        return Ok(());
    }

    println!(
        "{:24}  {:21}  {:8}  {:>8}  engines",
        "node_id", "host:port", "device", "mem_mb"
    );
    println!("{}", "-".repeat(86));
    for p in &peers {
        let addr = relay_addr(&p.host, p.port);
        let engines = if p.engines.is_empty() {
            "-".to_string()
        } else {
            p.engines.join(",")
        };
        println!(
            "{:24}  {:21}  {:8}  {:>8}  {}",
            p.node_id, addr, p.device, p.memory_mb, engines
        );
    }
    println!();
    if let Some(p) = peers.iter().find(|p| has_tcp_relay(p.port)) {
        println!(
            "Pass a peer's host:port to a worker's --next, e.g. --next {}:{}",
            p.host, p.port
        );
    }
    if peers.iter().any(|p| !has_tcp_relay(p.port)) {
        println!(
            "(in-host unix) peers listen on a unix socket, not a TCP port: wire \
             them from the same host with --next unix:/path.sock"
        );
    }
    Ok(())
}

/// A node advertising relay port 0 has no TCP relay endpoint: it is an
/// in-host unix-socket stage (#17), so there is nothing to dial or probe
/// at `host:0`.
pub(crate) fn has_tcp_relay(port: u16) -> bool {
    port != 0
}

/// The relay address column: `host:port`, or an in-host unix marker for a
/// port-0 node (never the undialable `host:0`).
fn relay_addr(host: &str, port: u16) -> String {
    if has_tcp_relay(port) {
        format!("{host}:{port}")
    } else {
        format!("{host} (in-host unix)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_zero_renders_as_in_host_unix_not_host_zero() {
        assert!(has_tcp_relay(9100));
        assert!(!has_tcp_relay(0));
        assert_eq!(relay_addr("10.0.0.2", 9100), "10.0.0.2:9100");
        assert_eq!(relay_addr("10.0.0.2", 0), "10.0.0.2 (in-host unix)");
    }
}
