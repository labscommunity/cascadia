//! Auto-ring election (#89): pure ring resolution + membership digest, and
//! the two-phase mDNS barrier. See docs/superpowers/specs/2026-07-01-mdns-auto-ring-design.md.

use cascadia_topology::NodeInfo;
use cascadia_types::hash::fnv1a_hex;

#[derive(Debug, thiserror::Error)]
pub enum RingError {
    #[error("self ({0}) not in member set")]
    SelfMissing(String),
    #[error("empty member set")]
    Empty,
    #[error("auto-ring: {0}")]
    Election(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct RingAssignment {
    pub rank: u32,
    pub total: u32,
    /// Next stage's advertised relay endpoint; None on the last rank.
    pub next: Option<(String, u16)>,
    /// The coordinator's (rank 0's) advertised host — for operator logs.
    pub coordinator_host: String,
}
impl RingAssignment {
    pub fn is_first(&self) -> bool {
        self.rank == 0
    }
    pub fn is_last(&self) -> bool {
        self.next.is_none()
    }
}

/// Deterministic total order: (memory_mb DESC, node_id ASC). node_id is unique
/// (topology key), so the order is total — no undetermined tie-break.
fn sorted(members: &[NodeInfo]) -> Vec<&NodeInfo> {
    let mut v: Vec<&NodeInfo> = members.iter().collect();
    v.sort_by(|a, b| {
        b.memory_mb
            .cmp(&a.memory_mb)
            .then_with(|| a.node_id.cmp(&b.node_id))
    });
    v
}

/// PURE. Digest of the RESOLVED ORDER (spec F1): FNV over the '\n'-joined
/// "memory_mb:node_id" pairs in sort order. Digesting the order (not just the
/// id set) certifies rank assignment: a restart-ghost advertising a different
/// memory_mb yields a different digest, so agreement can never mis-wire ranks.
pub fn member_digest(members: &[NodeInfo]) -> String {
    let joined = sorted(members)
        .iter()
        .map(|n| format!("{}:{}", n.memory_mb, n.node_id))
        .collect::<Vec<_>>()
        .join("\n");
    fnv1a_hex(joined.as_bytes())
}

/// PURE. Locate self in the sorted order -> rank/next/total/coordinator.
pub fn resolve_ring(members: &[NodeInfo], self_id: &str) -> Result<RingAssignment, RingError> {
    if members.is_empty() {
        return Err(RingError::Empty);
    }
    let order = sorted(members);
    let idx = order
        .iter()
        .position(|n| n.node_id == self_id)
        .ok_or_else(|| RingError::SelfMissing(self_id.to_string()))?;
    Ok(RingAssignment {
        rank: idx as u32,
        total: order.len() as u32,
        next: order.get(idx + 1).map(|n| (n.host.clone(), n.port)),
        coordinator_host: order[0].host.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn node(id: &str, host: &str, mem: u64) -> NodeInfo {
        let mut n = NodeInfo::new(id, host, 9100);
        n.memory_mb = mem;
        n
    }

    #[test]
    fn rank0_is_max_memory_and_chain_is_correct() {
        let m = vec![
            node("c", "10.0.0.3", 16000),
            node("a", "10.0.0.1", 32000),
            node("b", "10.0.0.2", 16000),
        ];
        let a = resolve_ring(&m, "a").unwrap();
        assert_eq!(
            (a.rank, a.total, a.is_first(), a.is_last()),
            (0, 3, true, false)
        );
        assert_eq!(a.next, Some(("10.0.0.2".into(), 9100))); // b: 16000, tie-break id
        assert_eq!(a.coordinator_host, "10.0.0.1");
        let last = resolve_ring(&m, "c").unwrap();
        assert_eq!((last.rank, last.is_last()), (2, true));
        assert_eq!(last.coordinator_host, "10.0.0.1");
    }

    #[test]
    fn permutation_invariant() {
        let base = vec![
            node("a", "h1", 32000),
            node("b", "h2", 16000),
            node("c", "h3", 8000),
        ];
        let want = resolve_ring(&base, "b").unwrap();
        let shuffled = vec![base[2].clone(), base[0].clone(), base[1].clone()];
        assert_eq!(resolve_ring(&shuffled, "b").unwrap(), want);
    }

    #[test]
    fn same_set_different_memory_yields_different_digest() {
        let s1 = vec![node("a", "h1", 32000), node("b", "h2", 16000)];
        let mut s2 = s1.clone();
        s2[0].memory_mb = 8000; // ghost with changed memory
        assert_ne!(
            member_digest(&s1),
            member_digest(&s2),
            "digest must cover order (F1)"
        );
    }

    #[test]
    fn digest_is_independent_of_input_vec_order() {
        let a = vec![node("a", "h1", 32000), node("b", "h2", 16000)];
        let b = vec![a[1].clone(), a[0].clone()];
        assert_eq!(member_digest(&a), member_digest(&b));
    }

    #[test]
    fn self_absent_errors() {
        assert!(matches!(
            resolve_ring(&[node("a", "h", 1)], "zzz"),
            Err(RingError::SelfMissing(_))
        ));
    }

    #[test]
    fn memory_zero_sorts_last() {
        let m = vec![node("a", "h1", 0), node("b", "h2", 16000)];
        assert_eq!(resolve_ring(&m, "b").unwrap().rank, 0);
        assert_eq!(resolve_ring(&m, "a").unwrap().rank, 1);
    }

    #[test]
    fn n1_degenerate_self_only_ring() {
        let m = vec![node("solo", "10.0.0.7", 8000)];
        let r = resolve_ring(&m, "solo").unwrap();
        assert_eq!(
            (r.rank, r.total, r.next.clone(), r.is_first(), r.is_last()),
            (0, 1, None, true, true)
        );
    }
}
