//! Exclusive execution ownership. Caller cancellation never means engine idle.
use crate::{
    manifest::{digest, identifier, invalid},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use uuid::Uuid;
const MAX_TTL: Duration = Duration::from_secs(120);
const MAX_LIFETIME: Duration = Duration::from_secs(3600);
const MAX_CALLS: u32 = 2048;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeaseId {
    pub boot: Uuid,
    pub id: Uuid,
}
#[derive(Clone)]
pub struct Gate(Arc<Mutex<State>>);
struct State {
    boot: Uuid,
    draining: bool,
    quarantined: bool,
    lease: Option<Lease>,
}
struct Lease {
    id: LeaseId,
    owner: String,
    manifest: String,
    components: BTreeSet<String>,
    expires: Instant,
    deadline: Instant,
    next: u32,
    active: Option<u32>,
    cancelled: bool,
}
/// Dropping an unfinished execution quarantines the worker. Only a positively
/// observed engine completion permits reuse; restarting creates a new boot ID.
pub struct Execution {
    gate: Gate,
    lease: LeaseId,
    sequence: u32,
    finished: bool,
}
impl Default for Gate {
    fn default() -> Self {
        Self::new()
    }
}
impl Gate {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(State {
            boot: Uuid::new_v4(),
            draining: false,
            quarantined: false,
            lease: None,
        })))
    }
    pub fn boot(&self) -> Uuid {
        self.0.lock().unwrap().boot
    }
    pub fn drain(&self) {
        self.0.lock().unwrap().draining = true;
    }
    pub fn draining(&self) -> bool {
        self.0.lock().unwrap().draining
    }
    pub fn quarantined(&self) -> bool {
        self.0.lock().unwrap().quarantined
    }
    pub fn occupied(&self, now: Instant) -> bool {
        let mut state = self.0.lock().unwrap();
        reap(&mut state, now);
        state.lease.is_some() || state.quarantined
    }
    pub fn reserve(
        &self,
        boot: Uuid,
        owner: &str,
        manifest: &str,
        components: BTreeSet<String>,
        ttl: Duration,
        now: Instant,
    ) -> Result<LeaseId> {
        digest(manifest)?;
        ttl_valid(ttl)?;
        if owner.is_empty() || owner.len() > 256 || components.is_empty() || components.len() > 16 {
            return invalid("invalid reservation");
        }
        for component in &components {
            identifier(component)?;
        }
        let mut state = self.0.lock().unwrap();
        reap(&mut state, now);
        if state.quarantined {
            return Err(Error::Quarantined);
        }
        if boot != state.boot {
            return Err(Error::Lease);
        }
        if state.draining || state.lease.is_some() {
            return Err(Error::Capacity);
        }
        let id = LeaseId {
            boot,
            id: Uuid::new_v4(),
        };
        state.lease = Some(Lease {
            id,
            owner: owner.into(),
            manifest: manifest.into(),
            components,
            expires: now + ttl,
            deadline: now + MAX_LIFETIME,
            next: 1,
            active: None,
            cancelled: false,
        });
        Ok(id)
    }
    pub fn renew(&self, owner: &str, id: LeaseId, ttl: Duration, now: Instant) -> Result<()> {
        ttl_valid(ttl)?;
        let mut state = self.0.lock().unwrap();
        if state.quarantined {
            return Err(Error::Quarantined);
        }
        let lease = authorized(&mut state, owner, id, now)?;
        lease.expires = (now + ttl).min(lease.deadline);
        Ok(())
    }
    pub fn release(&self, owner: &str, id: LeaseId) -> Result<()> {
        let mut state = self.0.lock().unwrap();
        let lease = state
            .lease
            .as_mut()
            .filter(|l| l.id == id && l.owner == owner)
            .ok_or(Error::Lease)?;
        lease.cancelled = true;
        if lease.active.is_none() {
            state.lease = None;
        }
        Ok(())
    }
    pub fn begin(
        &self,
        owner: &str,
        id: LeaseId,
        manifest: &str,
        component: &str,
        sequence: u32,
        now: Instant,
    ) -> Result<Execution> {
        let mut state = self.0.lock().unwrap();
        if state.quarantined {
            return Err(Error::Quarantined);
        }
        let lease = authorized(&mut state, owner, id, now)?;
        if lease.manifest != manifest || !lease.components.contains(component) {
            return Err(Error::Lease);
        }
        if sequence != lease.next || sequence > MAX_CALLS {
            return Err(Error::Replay);
        }
        if lease.active.is_some() {
            return Err(Error::Capacity);
        }
        lease.next += 1;
        lease.active = Some(sequence);
        Ok(Execution {
            gate: self.clone(),
            lease: id,
            sequence,
            finished: false,
        })
    }
}
fn ttl_valid(ttl: Duration) -> Result<()> {
    if ttl.is_zero() || ttl > MAX_TTL {
        return invalid("lease TTL must be in (0,120] seconds");
    }
    Ok(())
}
fn reap(state: &mut State, now: Instant) {
    if state.lease.as_ref().is_some_and(|l| {
        l.active.is_none() && (l.cancelled || now >= l.expires || now >= l.deadline)
    }) {
        state.lease = None;
    }
}
fn authorized<'a>(
    state: &'a mut State,
    owner: &str,
    id: LeaseId,
    now: Instant,
) -> Result<&'a mut Lease> {
    state
        .lease
        .as_mut()
        .filter(|l| {
            l.id == id && l.owner == owner && !l.cancelled && now < l.expires && now < l.deadline
        })
        .ok_or(Error::Lease)
}
impl Execution {
    /// Call only after the engine is known idle, including a confirmed failure.
    pub fn complete(mut self, now: Instant) {
        let mut state = self.gate.0.lock().unwrap();
        match state.lease.as_mut() {
            Some(lease) if lease.id == self.lease && lease.active == Some(self.sequence) => {
                lease.active = None;
                reap(&mut state, now);
            }
            _ => state.quarantined = true,
        }
        self.finished = true;
    }
}
impl Drop for Execution {
    fn drop(&mut self) {
        if !self.finished {
            self.gate.0.lock().unwrap().quarantined = true;
        }
    }
}
