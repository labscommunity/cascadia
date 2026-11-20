//! Runtime-independent worker admission and detached execution. Integrations
//! supply an executor and an authenticated caller identity; they never release
//! engine ownership on a network timeout or a disconnected client.
use crate::{
    lease::{Gate, LeaseId},
    manifest::{invalid, Manifest},
    wire, Error, Result,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant},
};
use uuid::Uuid;

pub type ExecuteFuture = Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send>>;
pub trait Executor: Send + Sync + 'static {
    /// Return only after the engine has finished. `Err(Error::Rejected)` means
    /// this call did not run, or ran to completion and was rejected. Any other
    /// error is uncertain and permanently quarantines this worker instance.
    fn execute(&self, frame: Vec<u8>) -> ExecuteFuture;
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Status {
    pub version: u32,
    pub boot: Uuid,
    pub manifest: String,
    pub components: BTreeSet<String>,
    pub available: bool,
}
impl Status {
    pub fn validate(&self) -> Result<()> {
        crate::manifest::digest(&self.manifest)?;
        if self.version != 1 || self.components.len() > 16 {
            return invalid("invalid worker status");
        }
        for name in &self.components {
            crate::manifest::identifier(name)?;
        }
        Ok(())
    }
}
pub struct Worker {
    gate: Gate,
    ready: AtomicBool,
    observer: OnceLock<Arc<dyn Fn() + Send + Sync>>,
    manifest: Arc<Manifest>,
    id: String,
    components: BTreeSet<String>,
    executor: Arc<dyn Executor>,
}
impl Worker {
    pub fn new(
        manifest: Arc<Manifest>,
        expected: String,
        components: BTreeSet<String>,
        executor: Arc<dyn Executor>,
    ) -> Result<Arc<Self>> {
        if manifest.id()? != expected
            || components.len() > 16
            || components
                .iter()
                .any(|name| !manifest.components.contains_key(name))
        {
            return invalid("worker artifacts differ from pinned contract");
        }
        Ok(Arc::new(Self {
            gate: Gate::new(),
            ready: AtomicBool::new(true),
            observer: OnceLock::new(),
            manifest,
            id: expected,
            components,
            executor,
        }))
    }
    pub fn status(&self) -> Status {
        Status {
            version: 1,
            boot: self.gate.boot(),
            manifest: self.id.clone(),
            components: self.components.clone(),
            available: self.ready.load(Ordering::Acquire)
                && !self.gate.occupied(Instant::now())
                && !self.gate.draining(),
        }
    }
    /// Registers a callback run once each execution ends, from a guard's `Drop`
    /// that may run during a panic unwind, so it must not panic or the process aborts.
    pub fn observe_state(&self, observer: Arc<dyn Fn() + Send + Sync>) {
        let _ = self.observer.set(observer);
    }
    fn changed(&self) {
        if let Some(observer) = self.observer.get() {
            observer();
        }
    }
    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Release);
    }
    pub fn occupied(&self) -> bool {
        self.gate.occupied(Instant::now())
    }
    pub fn quarantined(&self) -> bool {
        self.gate.quarantined()
    }
    pub fn drain(&self) {
        self.gate.drain();
    }
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    pub fn reserve(
        &self,
        owner: &str,
        boot: Uuid,
        manifest: &str,
        components: BTreeSet<String>,
        ttl: Duration,
    ) -> Result<LeaseId> {
        if !self.ready.load(Ordering::Acquire) {
            return Err(Error::Capacity);
        }
        if manifest != self.id || !components.is_subset(&self.components) {
            return Err(Error::Lease);
        }
        let result = self
            .gate
            .reserve(boot, owner, manifest, components, ttl, Instant::now());
        self.changed();
        result
    }
    pub fn renew(&self, owner: &str, id: LeaseId, ttl: Duration) -> Result<()> {
        self.gate.renew(owner, id, ttl, Instant::now())
    }
    pub fn release(&self, owner: &str, id: LeaseId) -> Result<()> {
        let result = self.gate.release(owner, id);
        self.changed();
        result
    }
    pub async fn execute(self: &Arc<Self>, owner: &str, frame: Vec<u8>) -> Result<Vec<u8>> {
        let (header, _) = wire::decode(&frame)?;
        let component = self
            .manifest
            .components
            .get(&header.component)
            .ok_or_else(|| Error::Invalid("unknown component".into()))?;
        wire::expect(&header, &component.inputs)?;
        let execution = self.gate.begin(
            owner,
            header.lease,
            &header.manifest,
            &header.component,
            header.sequence,
            Instant::now(),
        )?;
        let expected = component.outputs.clone();
        let worker = self.clone();
        // This task and guard own the inference. Dropping the caller's future
        // merely detaches the join handle; it cannot make the engine reusable.
        let result = tokio::spawn(async move {
            // Outlives the block below that owns `execution`, so the observer
            // sees a quarantine even when the executor panics.
            let _changed = Changed(worker.clone());
            let executor = worker.executor.clone();
            let result = async move {
                let response = match executor.execute(frame).await {
                    Err(Error::Rejected) => {
                        execution.complete(Instant::now());
                        return Err(Error::Rejected);
                    }
                    other => other?,
                };
                // A malformed reply means the engine is broken; quarantine it.
                let (reply, _) = wire::decode(&response)?;
                if reply.manifest != header.manifest
                    || reply.component != header.component
                    || reply.lease != header.lease
                    || reply.sequence != header.sequence
                {
                    return invalid("stage response belongs to another execution");
                }
                wire::expect(&reply, &expected)?;
                execution.complete(Instant::now());
                Ok(response)
            }
            .await;
            result
        })
        .await;
        // Every failure after `begin` except a rejection left the worker
        // quarantined; report that, not the cause, so callers cannot retry it.
        match result {
            Ok(Err(Error::Rejected)) => Err(Error::Rejected),
            Ok(Ok(response)) => Ok(response),
            Ok(Err(error)) => {
                tracing::warn!(%error, "diffusion worker quarantined");
                Err(Error::Quarantined)
            }
            Err(error) => {
                tracing::warn!(%error, "diffusion worker quarantined");
                Err(Error::Quarantined)
            }
        }
    }
}
/// Notifies the state observer once per execution when the task ends.
struct Changed(Arc<Worker>);
impl Drop for Changed {
    fn drop(&mut self) {
        self.0.changed();
    }
}
