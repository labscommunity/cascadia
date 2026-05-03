use tahoma_topology::{NodeInfo, Topology};

#[test]
fn add_node_records_last_seen() {
    let t = Topology::new();
    let node = NodeInfo::new("n1", "127.0.0.1", 9100);
    t.add_node(node.clone());
    let got = t.node("n1").expect("node should exist");
    assert!(got.last_seen > 0.0);
}

#[test]
fn remove_node_drops_associated_edges() {
    let t = Topology::new();
    t.add_node(NodeInfo::new("n1", "h", 1));
    t.add_node(NodeInfo::new("n2", "h", 2));
    t.measure("n1", "n2", 1.0, 100.0);
    t.measure("n2", "n1", 1.0, 100.0);
    assert_eq!(t.edges().len(), 2);
    t.remove_node("n1");
    assert_eq!(t.edges().len(), 0);
    assert!(t.node("n1").is_none());
    assert!(t.node("n2").is_some());
}

#[test]
fn measure_records_edge_metrics() {
    let t = Topology::new();
    t.add_node(NodeInfo::new("a", "h", 1));
    t.add_node(NodeInfo::new("b", "h", 2));
    t.measure("a", "b", 5.0, 1000.0);
    let edge = t.edge("a", "b").expect("edge should exist");
    assert_eq!(edge.latency_ms, 5.0);
    assert_eq!(edge.bandwidth_mbps, 1000.0);
    assert!(edge.last_measured > 0.0);
}

#[test]
fn expire_stale_drops_old_nodes() {
    let t = Topology::new();
    t.add_node(NodeInfo::new("n1", "h", 1));
    t.add_node(NodeInfo::new("n2", "h", 2));
    // Both were just added; with a generous max_age, none should be stale.
    let stale = t.expire_stale(60.0);
    assert!(stale.is_empty());
    assert_eq!(t.nodes().len(), 2);

    // A negative max_age forces every node into the stale set.
    let stale = t.expire_stale(-1.0);
    assert_eq!(stale.len(), 2);
    assert!(t.nodes().is_empty());
}
