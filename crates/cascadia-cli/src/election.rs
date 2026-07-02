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
/// Precondition: node_id must not contain ':' or '\n' (auto-ring ids are
/// sanitized to [A-Za-z0-9-]; see auto_node_id), else pair joining is ambiguous.
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

pub struct ElectParams {
    pub self_id: String,
    pub namespace: String, // discovery already filters; kept for messages
    pub model_hash: String,
    pub engine: String,
    pub cluster_size: u32,
}

/// Outcome of one evaluation of the current member snapshot.
#[derive(Debug, PartialEq)]
pub enum Step {
    Wait,                 // < N participants; keep polling
    Conflict(String),     // model/engine mismatch on a participant -> abort
    Ready(Vec<String>),   // exactly N participants (node_ids)
    TooMany(Vec<String>), // > N participants (transient ghost or real surplus)
}

/// PURE. Classify the current snapshot. Self-contained: filters to
/// participants (cluster_size == mine) internally, so passing raw
/// topo.nodes() is safe — manual/dashboard workers never affect the count.
/// Conflict checks run BEFORE the count (spec: mismatch preemption).
pub fn phase1_step(members: &[NodeInfo], p: &ElectParams) -> Step {
    let participants: Vec<&NodeInfo> = members
        .iter()
        .filter(|m| m.cluster_size == Some(p.cluster_size))
        .collect();
    for m in &participants {
        match &m.model_hash {
            Some(h) if *h == p.model_hash => {}
            _ => {
                return Step::Conflict(format!(
                    "peer {} model_hash {:?} != {} (version skew or wrong --model)",
                    m.node_id, m.model_hash, p.model_hash
                ))
            }
        }
        if !m.engines.iter().any(|e| e == &p.engine) {
            return Step::Conflict(format!("peer {} engine != {}", m.node_id, p.engine));
        }
    }
    let n = p.cluster_size as usize;
    let ids: Vec<String> = participants.iter().map(|m| m.node_id.clone()).collect();
    match participants.len() {
        len if len < n => Step::Wait,
        len if len == n => Step::Ready(ids),
        _ => Step::TooMany(ids),
    }
}

/// PURE. Phase-2 commit predicate: exactly N members AND every member
/// (including self) advertises my digest (spec F2 cardinality guard).
pub fn agreed(members: &[NodeInfo], my_digest: &str, n: u32) -> bool {
    members.len() == n as usize
        && members
            .iter()
            .all(|m| m.ring_digest.as_deref() == Some(my_digest))
}

use cascadia_discovery::DiscoveryService;
use cascadia_topology::Topology;
use std::time::Duration;

pub struct ElectTimeouts {
    pub discover: Duration,
    pub agree: Duration,
    pub settle: Duration,
}

/// Member view = participants from the topology, with self's entry overlaid
/// (spec F5: self is injected explicitly, never via mDNS self-reflection).
fn current_members(topo: &Topology, me: &NodeInfo, p: &ElectParams) -> Vec<NodeInfo> {
    let mut v: Vec<NodeInfo> = topo
        .nodes()
        .into_iter()
        .filter(|n| n.cluster_size == Some(p.cluster_size))
        .filter(|n| n.node_id != me.node_id)
        .collect();
    v.push(me.clone());
    v
}

pub async fn elect(
    topo: &Topology,
    disco: &mut DiscoveryService,
    self_node: &NodeInfo,
    p: &ElectParams,
    t: &ElectTimeouts,
) -> Result<(RingAssignment, String), RingError> {
    // ---- Phase 1: collect exactly N matched participants ----
    let mut me = self_node.clone();
    let discover_deadline = tokio::time::Instant::now() + t.discover;
    let mut stable_since: Option<(String, tokio::time::Instant)> = None; // (digest, since)
    let members = loop {
        let members = current_members(topo, &me, p);
        match phase1_step(&members, p) {
            Step::Conflict(msg) => return Err(RingError::Election(msg)),
            Step::Ready(_) => {
                // Settle: the RESOLVED ORDER (member_digest — excludes
                // last_seen churn) must hold for t.settle before Phase 2.
                let d = member_digest(&members);
                match &stable_since {
                    Some((prev, since)) if *prev == d => {
                        if since.elapsed() >= t.settle {
                            break members;
                        }
                    }
                    _ => stable_since = Some((d, tokio::time::Instant::now())),
                }
            }
            Step::TooMany(ids) => {
                // Transient ghosts get debounce-pruned; only a surplus that
                // persists to the deadline is fatal.
                tracing::warn!(
                    ?ids,
                    n = p.cluster_size,
                    "more than N participants; waiting for prune"
                );
                stable_since = None;
            }
            Step::Wait => {
                tracing::info!(
                    have = members.len(),
                    want = p.cluster_size,
                    "waiting for peers"
                );
                stable_since = None;
            }
        }
        if tokio::time::Instant::now() >= discover_deadline {
            let ids: Vec<String> = members.iter().map(|m| m.node_id.clone()).collect();
            return Err(RingError::Election(format!(
                "discovered {} of {} participants in namespace '{}' after {:?}: {:?}",
                members.len(),
                p.cluster_size,
                p.namespace,
                t.discover,
                ids
            )));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    };

    // ---- Phase 2: agree via digest ----
    let mut digest = member_digest(&members);
    me.ring_digest = Some(digest.clone());
    // F5/livelock fix: our digest must be visible in the SAME view `agreed`
    // reads. update_txt only informs peers; write the local topology too.
    topo.add_node(me.clone());
    disco
        .update_txt(me.clone())
        .map_err(|e| RingError::Election(format!("cannot advertise digest (mDNS daemon): {e}")))?;

    let mut agree_deadline = tokio::time::Instant::now() + t.agree;
    loop {
        let members = current_members(topo, &me, p);
        if agreed(&members, &digest, p.cluster_size) {
            let ring = resolve_ring(&members, &p.self_id)?;
            return Ok((ring, digest));
        }
        let now_digest = member_digest(&members);
        if now_digest != digest && members.len() == p.cluster_size as usize {
            // Set/order shifted while stable-sized: adopt + re-advertise.
            // The commit predicate MUST track what we advertise.
            digest = now_digest;
            me.ring_digest = Some(digest.clone());
            topo.add_node(me.clone());
            if let Err(e) = disco.update_txt(me.clone()) {
                tracing::warn!(error = %e, "digest re-advertise failed; peers may not converge");
            }
            agree_deadline = tokio::time::Instant::now() + t.agree; // restart window
        }
        if tokio::time::Instant::now() >= agree_deadline {
            let views: Vec<(String, Option<String>)> = members
                .iter()
                .map(|m| (m.node_id.clone(), m.ring_digest.clone()))
                .collect();
            return Err(RingError::Election(format!(
                "ring membership did not converge in namespace '{}'; my digest {} vs peer views {:?}",
                p.namespace, digest, views
            )));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
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

#[cfg(test)]
mod elect_tests {
    use super::*;
    fn part(id: &str, mem: u64, hash: &str, cs: u32) -> NodeInfo {
        let mut n = NodeInfo::new(id, "h", 9100);
        n.memory_mb = mem;
        n.model_hash = Some(hash.into());
        n.cluster_size = Some(cs);
        n.engines = vec!["ov-runtime".into()];
        n
    }
    fn params() -> ElectParams {
        ElectParams {
            self_id: "a".into(),
            namespace: "default".into(),
            model_hash: "h".into(),
            engine: "ov-runtime".into(),
            cluster_size: 3,
        }
    }
    #[test]
    fn under_n_waits() {
        assert_eq!(
            phase1_step(&[part("a", 1, "h", 3), part("b", 1, "h", 3)], &params()),
            Step::Wait
        );
    }
    #[test]
    fn exactly_n_ready() {
        let m = vec![
            part("a", 1, "h", 3),
            part("b", 1, "h", 3),
            part("c", 1, "h", 3),
        ];
        assert!(matches!(phase1_step(&m, &params()), Step::Ready(_)));
    }
    #[test]
    fn over_n_toomany() {
        let m = vec![
            part("a", 1, "h", 3),
            part("b", 1, "h", 3),
            part("c", 1, "h", 3),
            part("d", 1, "h", 3),
        ];
        assert!(matches!(phase1_step(&m, &params()), Step::TooMany(_)));
    }
    #[test]
    fn model_mismatch_conflicts_and_names_offender() {
        let m = vec![
            part("a", 1, "h", 3),
            part("bad-peer", 1, "WRONG", 3),
            part("c", 1, "h", 3),
        ];
        match phase1_step(&m, &params()) {
            Step::Conflict(msg) => assert!(msg.contains("bad-peer"), "offender not named: {msg}"),
            other => panic!("expected Conflict, got {other:?}"),
        }
    }
    #[test]
    fn engine_mismatch_conflicts() {
        let mut m = vec![
            part("a", 1, "h", 3),
            part("b", 1, "h", 3),
            part("c", 1, "h", 3),
        ];
        m[1].engines = vec!["mock".into()];
        assert!(matches!(phase1_step(&m, &params()), Step::Conflict(_)));
    }
    #[test]
    fn conflict_preempts_count() {
        // Both a mismatch AND >N present: mismatch must fire first (spec order).
        let m = vec![
            part("a", 1, "h", 3),
            part("b", 1, "WRONG", 3),
            part("c", 1, "h", 3),
            part("d", 1, "h", 3),
        ];
        assert!(matches!(phase1_step(&m, &params()), Step::Conflict(_)));
    }
    #[test]
    fn non_participants_are_invisible_unfiltered() {
        // A manual worker (no cluster_size, different model) passed RAW:
        // neither counted nor a conflict.
        let mut manual = NodeInfo::new("m", "h", 9100);
        manual.model_hash = Some("other".into()); // no cluster_size
        let m = vec![part("a", 1, "h", 3), part("b", 1, "h", 3), manual];
        assert_eq!(phase1_step(&m, &params()), Step::Wait);
    }
    #[test]
    fn n1_self_only_is_ready() {
        let p = ElectParams {
            cluster_size: 1,
            ..params()
        };
        assert!(matches!(
            phase1_step(&[part("a", 1, "h", 1)], &p),
            Step::Ready(_)
        ));
    }
    #[test]
    fn agreement_requires_len_n_and_unanimous() {
        let mut m = vec![
            part("a", 1, "h", 3),
            part("b", 1, "h", 3),
            part("c", 1, "h", 3),
        ];
        let d = member_digest(&m);
        for x in &mut m {
            x.ring_digest = Some(d.clone());
        }
        assert!(agreed(&m, &d, 3));
        m.pop(); // now 2 -> cardinality guard fails
        assert!(!agreed(&m, &d, 3));
    }
    #[test]
    fn agreement_fails_on_one_disagreeing_digest() {
        let mut m = vec![
            part("a", 1, "h", 3),
            part("b", 1, "h", 3),
            part("c", 1, "h", 3),
        ];
        let d = member_digest(&m);
        m[0].ring_digest = Some(d.clone());
        m[1].ring_digest = Some(d.clone());
        m[2].ring_digest = Some("0000000000000000".into());
        assert!(!agreed(&m, &d, 3));
    }
    #[test]
    fn agreement_fails_when_self_digest_missing() {
        // Regression for the self-write-back livelock: if self's entry lacks
        // the digest, unanimity must be false (which is why elect MUST write
        // its digest into the local topology, not only the mDNS TXT).
        let mut m = vec![
            part("a", 1, "h", 3),
            part("b", 1, "h", 3),
            part("c", 1, "h", 3),
        ];
        let d = member_digest(&m);
        m[1].ring_digest = Some(d.clone());
        m[2].ring_digest = Some(d.clone());
        // m[0] (self) has ring_digest None
        assert!(!agreed(&m, &d, 3));
    }
}
