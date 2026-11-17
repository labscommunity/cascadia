//! physicalai's remote-inference server over Zenoh, so Intel's
//! `RemoteInferenceModel` can use a Cascadia policy with no code change.
//!
//! Mirrors `physicalai/inference/remote/server.py`: a peer session that
//! listens on one TCP endpoint only (no connect, no scouting, no shared
//! memory), three queryables under `physicalai/inference/<name>`, and one
//! worker that answers in order. A queued predict is superseded by a newer
//! one, and a predict whose `budget_ms` has passed when the model is free is
//! answered `expired`. There is no authentication, as in physicalai, so the
//! listener is loopback unless the caller opts in.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use cascadia_engine::EngineError;
use cascadia_types::{ActionChunk, Observation, PolicySpec};
use sha2::{Digest, Sha256};
use tokio::sync::OwnedSemaphorePermit;
use zenoh::query::{Query, Queryable};
use zenoh::{Session, Wait};

use super::physicalai::{
    check_name, decode_control, decode_predict, encode_error, encode_metadata,
    encode_predict_reply, encode_reset_ok, key_prefix, read_seq, ControlError, ErrorCode,
    ServerIdentity, Timings, MAX_REQUEST_BYTES, NO_SEQ,
};
use super::server::{is_engine_fault, SharedPolicy};
use super::{check_reply, max_payload_bytes, PackageIdentity};

/// A setup failure: a bad address or name, or Zenoh refusing to open.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ZenohError(pub String);

fn err<T>(msg: impl Into<String>) -> Result<T, ZenohError> {
    Err(ZenohError(msg.into()))
}

/// physicalai's `derive_port("inference", name)`: the port its client
/// connects to when none is given.
pub fn derive_port(name: &str) -> u16 {
    let digest = Sha256::digest(key_prefix(name).as_bytes());
    let n = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
    (20_000 + n % 40_000) as u16
}

/// Where and under which name to listen.
#[derive(Clone, Debug)]
pub struct ZenohListen {
    /// The slot name the client is given; keys live under `physicalai/inference/<name>`.
    pub name: String,
    pub host: String,
    /// `None` derives the port from the name, as physicalai does.
    pub port: Option<u16>,
    /// Allow a host other than loopback. physicalai's transport has no
    /// authentication, so anything that can reach the port can drive the robot.
    pub allow_all_interfaces: bool,
}

use super::is_loopback_host as is_loopback;

impl ZenohListen {
    pub fn port(&self) -> u16 {
        self.port.unwrap_or_else(|| derive_port(&self.name))
    }

    /// The `tcp/<host>:<port>` endpoint, after the name and host checks.
    pub fn endpoint(&self) -> Result<String, ZenohError> {
        check_name(&self.name).map_err(|e| ZenohError(e.0))?;
        let host = self.host.as_str();
        if host.is_empty() || host.contains('/') || host.chars().any(char::is_whitespace) {
            return err("host must be a non-empty hostname or IP address");
        }
        if !is_loopback(host) && !self.allow_all_interfaces {
            return err(format!(
                "the Zenoh listener has no authentication, so it binds loopback only; \
                 {host} needs --policy-allow-all-interfaces"
            ));
        }
        let port = self.port();
        if port == 0 {
            return err("port must be between 1 and 65535");
        }
        let host = if host.contains(':') && !host.starts_with('[') {
            format!("[{host}]")
        } else {
            host.to_string()
        };
        Ok(format!("tcp/{host}:{port}"))
    }
}

/// physicalai's secure defaults: a peer listening on `endpoint` only, with
/// no outgoing connects, no multicast or gossip scouting, and no shared memory.
pub fn secure_config(endpoint: &str, max_message_bytes: u64) -> Result<zenoh::Config, ZenohError> {
    let mut c = zenoh::Config::default();
    let ep = serde_json::to_string(&[endpoint]).expect("a string list serializes");
    for (k, v) in [
        ("mode", "\"peer\"".to_string()),
        ("connect/endpoints", "[]".to_string()),
        ("listen/endpoints", ep),
        ("scouting/multicast/enabled", "false".to_string()),
        ("scouting/gossip/enabled", "false".to_string()),
        (
            "transport/link/rx/max_message_size",
            max_message_bytes.to_string(),
        ),
        ("transport/shared_memory/enabled", "false".to_string()),
    ] {
        c.insert_json5(k, &v)
            .map_err(|e| ZenohError(format!("zenoh config {k}: {e}")))?;
    }
    Ok(c)
}

/// One computed call and when its compute started (after any wait for the model).
pub struct Computed {
    pub chunk: ActionChunk,
    pub started: Instant,
}

/// Why a call produced no actions.
#[derive(Debug)]
pub enum BackendError {
    /// The observation was refused: `invalid_request`.
    Invalid(String),
    /// The model or upstream failed: `model_error`.
    Model(String),
    /// The budget ran out before or while computing: `expired`.
    Expired,
}

/// What answers the queued calls: a local engine, or an upstream server.
pub trait Backend: Send + Sync + 'static {
    /// Run one call; `deadline` is when the client stops waiting.
    fn infer(&self, obs: &Observation, deadline: Instant) -> Result<Computed, BackendError>;
    fn reset(&self) {}
}

/// The engine this process loaded, shared with the WebSocket front end.
struct LocalBackend {
    policy: SharedPolicy,
    /// The Zenoh front end counts as one connection for as long as it serves.
    _slot: OwnedSemaphorePermit,
}

impl Backend for LocalBackend {
    fn infer(&self, obs: &Observation, deadline: Instant) -> Result<Computed, BackendError> {
        let mut engine = self.policy.lock();
        let started = Instant::now();
        // physicalai checks the budget when the worker picks the call up; the
        // model may have been busy with a WebSocket call until now.
        if started > deadline {
            return Err(BackendError::Expired);
        }
        let res = self.policy.infer_guarded(&mut **engine, obs);
        drop(engine);
        match res {
            Ok(chunk) => match check_reply(&chunk) {
                Ok(()) => Ok(Computed { chunk, started }),
                Err(e) => Err(BackendError::Model(e.0)),
            },
            Err(EngineError::InvalidConfig(m)) => Err(BackendError::Invalid(m)),
            Err(e) if !is_engine_fault(&e) => Err(BackendError::Invalid(e.to_string())),
            Err(e) => Err(BackendError::Model(e.to_string())),
        }
    }
}

/// What the handshake advertises besides the fresh `server_id`.
#[derive(Clone, Debug)]
pub struct Advert {
    pub spec: PolicySpec,
    pub identity: PackageIdentity,
    pub backend: String,
    pub device: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cmd {
    Metadata,
    Predict,
    Reset,
}

struct Job {
    query: Query,
    cmd: Cmd,
    seq: i128,
    received: Instant,
    decoded: Instant,
    deadline: Instant,
    obs: Option<Observation>,
}

/// Calls answered, superseded and expired since start.
#[derive(Debug, Default)]
pub struct Counts {
    pub done: AtomicU64,
    pub superseded: AtomicU64,
    pub expired: AtomicU64,
}

struct Core {
    spec: PolicySpec,
    max_bytes: u64,
    metadata: Vec<u8>,
    server_id: String,
    predict_key: String,
    reset_key: String,
    metadata_key: String,
    queue: Mutex<VecDeque<Job>>,
    cond: Condvar,
    stopping: AtomicBool,
    backend: Box<dyn Backend>,
    counts: Counts,
}

fn reply_err(query: &Query, code: ErrorCode, seq: i128) {
    if let Err(e) = query.reply_err(encode_error(code, seq)).wait() {
        tracing::debug!(error = %e, "could not send a physicalai error reply");
    }
}

impl Core {
    /// [`super::server::count_infer`] for a call refused before the engine.
    fn count(&self, result: &str) {
        super::server::count_infer(self.spec.model.as_str(), result);
    }

    fn drop_metric(&self, reason: &str) {
        cascadia_metrics::POLICY_DROPPED_TOTAL
            .with_label_values(&[self.spec.model.as_str(), reason])
            .inc();
    }

    fn on_query(&self, query: Query, cmd: Cmd) {
        let received = Instant::now();
        let payload = query
            .payload()
            .map(|p| p.to_bytes().into_owned())
            .unwrap_or_default();
        if self.stopping.load(Ordering::SeqCst) {
            return reply_err(&query, ErrorCode::ShuttingDown, read_seq(&payload));
        }
        if payload.len() as u64 > self.max_bytes {
            self.count("invalid");
            return reply_err(&query, ErrorCode::InvalidRequest, NO_SEQ);
        }
        let refused = |why: &str| {
            self.count("invalid");
            tracing::info!(command = ?cmd, reason = why, "physicalai request refused");
            reply_err(&query, ErrorCode::InvalidRequest, read_seq(&payload));
        };
        match cmd {
            Cmd::Metadata | Cmd::Reset => match decode_control(&payload, self.max_bytes) {
                Err(ControlError::UnsupportedVersion) => {
                    reply_err(&query, ErrorCode::UnsupportedVersion, NO_SEQ)
                }
                Err(ControlError::Invalid(e)) => refused(e.message()),
                Ok(()) if cmd == Cmd::Metadata => {
                    if let Err(e) = query
                        .reply(&self.metadata_key, self.metadata.clone())
                        .wait()
                    {
                        tracing::debug!(error = %e, "could not send physicalai metadata");
                    }
                }
                Ok(()) => self.enqueue(Job {
                    query,
                    cmd,
                    seq: NO_SEQ,
                    received,
                    decoded: Instant::now(),
                    deadline: received + Duration::from_millis(i32::MAX as u64),
                    obs: None,
                }),
            },
            Cmd::Predict => match decode_predict(&payload, &self.spec, self.max_bytes) {
                Err(e) => refused(e.message()),
                Ok(req) => self.enqueue(Job {
                    query,
                    cmd,
                    seq: i128::from(req.seq),
                    received,
                    decoded: Instant::now(),
                    deadline: received + Duration::from_millis(req.budget_ms),
                    obs: Some(req.observation),
                }),
            },
        }
    }

    /// Queue `job`, superseding a queued call of the same kind.
    fn enqueue(&self, job: Job) {
        let mut q = self.queue.lock().expect("queue lock");
        if self.stopping.load(Ordering::SeqCst) {
            return reply_err(&job.query, ErrorCode::ShuttingDown, job.seq);
        }
        if let Some(i) = q.iter().position(|p| p.cmd == job.cmd) {
            let old = q.remove(i).expect("position is in range");
            reply_err(&old.query, ErrorCode::Superseded, old.seq);
            self.counts.superseded.fetch_add(1, Ordering::Relaxed);
            self.drop_metric("superseded");
        }
        q.push_back(job);
        self.cond.notify_one();
    }

    fn worker(&self) {
        loop {
            let job = {
                let mut q = self.queue.lock().expect("queue lock");
                loop {
                    if let Some(job) = q.pop_front() {
                        break job;
                    }
                    if self.stopping.load(Ordering::SeqCst) {
                        return;
                    }
                    q = self.cond.wait(q).expect("queue lock");
                }
            };
            self.run(job);
        }
    }

    fn run(&self, job: Job) {
        let expired = || {
            self.counts.expired.fetch_add(1, Ordering::Relaxed);
            self.drop_metric("expired");
            tracing::info!(seq = %job.seq, "physicalai request expired");
            reply_err(&job.query, ErrorCode::Expired, job.seq);
        };
        if Instant::now() > job.deadline {
            return expired();
        }
        if job.cmd == Cmd::Reset {
            self.backend.reset();
            if let Err(e) = job.query.reply(&self.reset_key, encode_reset_ok()).wait() {
                tracing::debug!(error = %e, "could not send the reset reply");
            }
            self.counts.done.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let obs = job
            .obs
            .as_ref()
            .expect("a predict job carries its observation");
        let seq = u64::try_from(job.seq).expect("a predict seq is non-negative");
        match self.backend.infer(obs, job.deadline) {
            Ok(c) => {
                let t = Timings {
                    decode_s: (job.decoded - job.received).as_secs_f64(),
                    queue_s: c
                        .started
                        .saturating_duration_since(job.decoded)
                        .as_secs_f64(),
                    compute_s: c.started.elapsed().as_secs_f64(),
                };
                let reply = encode_predict_reply(&c.chunk, seq, &self.server_id, t);
                if let Err(e) = job.query.reply(&self.predict_key, reply).wait() {
                    tracing::debug!(error = %e, "could not send a predict reply");
                }
                self.counts.done.fetch_add(1, Ordering::Relaxed);
            }
            Err(BackendError::Expired) => expired(),
            Err(BackendError::Invalid(m)) => {
                tracing::info!(seq, reason = %m, "physicalai request refused");
                reply_err(&job.query, ErrorCode::InvalidRequest, job.seq);
            }
            Err(BackendError::Model(m)) => {
                tracing::warn!(seq, error = %m, "physicalai request failed");
                reply_err(&job.query, ErrorCode::ModelError, job.seq);
            }
        }
    }
}

/// A running physicalai front end. [`ZenohFrontEnd::stop`] answers queued
/// calls `shutting_down` and closes the session.
pub struct ZenohFrontEnd {
    session: Session,
    queryables: Vec<Queryable<()>>,
    core: Arc<Core>,
    worker: Option<std::thread::JoinHandle<()>>,
    endpoint: String,
}

impl ZenohFrontEnd {
    /// Serve the shared engine. It takes one of the engine's connection slots.
    pub async fn serve_local(
        policy: SharedPolicy,
        listen: &ZenohListen,
        identity: PackageIdentity,
    ) -> Result<Self, ZenohError> {
        let Some(slot) = policy.try_slot() else {
            return err(format!(
                "every one of the {} policy connection slots is taken",
                policy.max_connections()
            ));
        };
        let advert = Advert {
            spec: policy.spec().clone(),
            device: policy.spec().device.clone(),
            identity,
            backend: "openvino".into(),
        };
        let backend = LocalBackend {
            policy,
            _slot: slot,
        };
        Self::start(listen, advert, Box::new(backend)).await
    }

    /// Serve `backend` under `listen`, advertising `advert` and a fresh `server_id`.
    pub async fn start(
        listen: &ZenohListen,
        advert: Advert,
        backend: Box<dyn Backend>,
    ) -> Result<Self, ZenohError> {
        let endpoint = listen.endpoint()?;
        // Frames may come at the camera's native size and are resized here.
        let max_bytes = MAX_REQUEST_BYTES.max(u64::from(max_payload_bytes(&advert.spec)));
        let config = secure_config(&endpoint, max_bytes)?;
        let id = ServerIdentity {
            server_id: uuid::Uuid::new_v4().simple().to_string(),
            name: listen.name.clone(),
            policy_name: advert.identity.policy_name.clone(),
            manifest_sha256: advert.spec.manifest_sha256.clone(),
            backend: advert.backend.clone(),
            device: advert.device.clone(),
        };
        let prefix = key_prefix(&listen.name);
        let core = Arc::new(Core {
            metadata: encode_metadata(&id, &advert.spec, max_bytes),
            spec: advert.spec,
            max_bytes,
            server_id: id.server_id.clone(),
            metadata_key: format!("{prefix}/metadata"),
            predict_key: format!("{prefix}/predict"),
            reset_key: format!("{prefix}/reset"),
            queue: Mutex::new(VecDeque::new()),
            cond: Condvar::new(),
            stopping: AtomicBool::new(false),
            backend,
            counts: Counts::default(),
        });
        if !is_loopback(&listen.host) {
            tracing::warn!(
                endpoint = %endpoint,
                "physicalai server reachable without authentication; use only on a trusted network"
            );
        }
        let session = zenoh::open(config)
            .await
            .map_err(|e| ZenohError(format!("open the Zenoh session on {endpoint}: {e}")))?;
        let worker_core = core.clone();
        let worker = std::thread::Builder::new()
            .name("physicalai-worker".into())
            .spawn(move || worker_core.worker())
            .map_err(|e| ZenohError(format!("spawn the worker: {e}")))?;
        let mut me = Self {
            session,
            queryables: Vec::new(),
            core,
            worker: Some(worker),
            endpoint,
        };
        for (key, cmd) in [
            (me.core.metadata_key.clone(), Cmd::Metadata),
            (me.core.predict_key.clone(), Cmd::Predict),
            (me.core.reset_key.clone(), Cmd::Reset),
        ] {
            let core = me.core.clone();
            let declared = me
                .session
                .declare_queryable(key.as_str())
                .callback(move |q| core.on_query(q, cmd))
                .complete(true)
                .await;
            match declared {
                Ok(q) => me.queryables.push(q),
                Err(e) => {
                    me.stop().await;
                    return err(format!("declare {key}: {e}"));
                }
            }
        }
        tracing::info!(
            endpoint = %me.endpoint,
            name = %listen.name,
            server_id = %me.core.server_id,
            "serving physicalai remote inference"
        );
        Ok(me)
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn server_id(&self) -> &str {
        &self.core.server_id
    }

    pub fn counts(&self) -> &Counts {
        &self.core.counts
    }

    pub async fn stop(mut self) {
        self.core.stopping.store(true, Ordering::SeqCst);
        {
            let mut q = self.core.queue.lock().expect("queue lock");
            for job in q.drain(..) {
                reply_err(&job.query, ErrorCode::ShuttingDown, job.seq);
            }
            self.core.cond.notify_all();
        }
        for q in self.queryables.drain(..) {
            if let Err(e) = q.undeclare().await {
                tracing::debug!(error = %e, "undeclare a physicalai queryable");
            }
        }
        if let Some(w) = self.worker.take() {
            let _ = tokio::task::spawn_blocking(move || w.join()).await;
        }
        if let Err(e) = self.session.close().await {
            tracing::debug!(error = %e, "close the Zenoh session");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listen(host: &str, allow: bool) -> ZenohListen {
        ZenohListen {
            name: "cell1".into(),
            host: host.into(),
            port: Some(7447),
            allow_all_interfaces: allow,
        }
    }

    #[test]
    fn endpoints_are_loopback_unless_allowed() {
        assert_eq!(
            listen("127.0.0.1", false).endpoint().unwrap(),
            "tcp/127.0.0.1:7447"
        );
        assert_eq!(listen("::1", false).endpoint().unwrap(), "tcp/[::1]:7447");
        assert_eq!(
            listen("localhost", false).endpoint().unwrap(),
            "tcp/localhost:7447"
        );
        for host in ["0.0.0.0", "::", "192.168.1.20", "robot.lan"] {
            let e = listen(host, false).endpoint().unwrap_err().0;
            assert!(e.contains("--policy-allow-all-interfaces"), "{host}: {e}");
            assert!(listen(host, true).endpoint().is_ok(), "{host}");
        }
        assert!(listen("a b", true).endpoint().is_err());
        assert!(listen("", true).endpoint().is_err());
        let mut bad = listen("127.0.0.1", false);
        bad.name = "a/b".into();
        assert!(bad.endpoint().is_err());
        bad.name = ".x".into();
        assert!(bad.endpoint().is_err());
    }

    #[test]
    fn ports_match_physicalai() {
        let want: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/physicalai/ports.json"))
                .unwrap();
        for (name, port) in want.as_object().unwrap() {
            assert_eq!(
                u64::from(derive_port(name)),
                port.as_u64().unwrap(),
                "{name}"
            );
        }
    }

    fn spec() -> PolicySpec {
        PolicySpec {
            model: "mock".into(),
            device: "CPU".into(),
            cameras: vec!["image".into(), "wrist".into()],
            height: 16,
            width: 16,
            state_len: 3,
            chunk: [2, 3],
            noise: cascadia_types::NoiseMode::Input,
            noise_shape: Some(vec![2, 3]),
            resize: cascadia_types::ResizeConvention::TopLeft,
            prompt_max_bytes: 64,
            attribution: None,
            manifest_sha256: "ab".repeat(32),
            precision: "f32".into(),
            package_sha256: String::new(),
            server_resize: Some(cascadia_types::ServerResize::Letterbox),
            rtc_chunk_size: None,
        }
    }

    /// Fails every call with the error its function builds.
    struct Failing(PolicySpec, fn() -> EngineError);

    impl cascadia_engine::PolicyEngine for Failing {
        fn spec(&self) -> &PolicySpec {
            &self.0
        }
        fn warmup(&mut self) -> cascadia_engine::EngineResult<Vec<u32>> {
            Ok(Vec::new())
        }
        fn infer(&mut self, _: &Observation) -> cascadia_engine::EngineResult<ActionChunk> {
            Err((self.1)())
        }
    }

    /// A refusal is `invalid_request` and leaves readiness up; an engine
    /// failure is `model_error` and clears it.
    #[test]
    fn refusals_leave_readiness_and_faults_clear_it() {
        let cases: [(fn() -> EngineError, bool); 3] = [
            (|| EngineError::PromptTooLong("over".into()), false),
            (|| EngineError::InvalidConfig("bad".into()), false),
            (|| EngineError::Backend("device lost".into()), true),
        ];
        for (make, fault) in cases {
            let policy = SharedPolicy::new(Box::new(Failing(spec(), make)), 1).unwrap();
            // Chat's counters too; the registry is process-global, so deltas only grow.
            let rejected = || {
                cascadia_metrics::API_REJECTED_TOTAL
                    .with_label_values(&["invalid_request"])
                    .get()
            };
            let failed = || {
                cascadia_metrics::TASKS_FAILED_TOTAL
                    .with_label_values(&[policy.spec().model.as_str()])
                    .get()
            };
            let (rejected_before, failed_before) = (rejected(), failed());
            let backend = LocalBackend {
                _slot: policy.try_slot().unwrap(),
                policy: policy.clone(),
            };
            let obs = Observation {
                images: Vec::new(),
                state: Vec::new(),
                prompt: String::new(),
                seed: None,
                noise: None,
                rtc: None,
            };
            let deadline = Instant::now() + Duration::from_secs(10);
            let e = backend.infer(&obs, deadline).err().unwrap();
            let e_name = format!("{e:?}");
            match (fault, e) {
                (false, BackendError::Invalid(_)) | (true, BackendError::Model(_)) => {}
                _ => panic!("{} answered {e_name}", make()),
            }
            assert_eq!(policy.is_ready(), !fault, "{}", make());
            if fault {
                assert!(failed() > failed_before, "{}", make());
            } else {
                assert!(rejected() > rejected_before, "{}", make());
            }
        }
    }

    /// A Zenoh client in this process against the front end on a mock engine:
    /// the handshake, a predict from physicalai's fixture, a refusal, and the
    /// shared connection slot.
    #[tokio::test(flavor = "multi_thread")]
    async fn serves_the_shared_engine_over_zenoh() {
        let policy =
            SharedPolicy::new(Box::new(cascadia_engine::MockPolicyEngine::new(spec())), 2).unwrap();
        let port = 20_000 + (std::process::id() % 20_000) as u16;
        let listen = ZenohListen {
            name: "cell1".into(),
            host: "127.0.0.1".into(),
            port: Some(port),
            allow_all_interfaces: false,
        };
        let id = PackageIdentity {
            policy_name: "smolvla".into(),
            warmup_ms: Some(5),
        };
        let front = ZenohFrontEnd::serve_local(policy.clone(), &listen, id)
            .await
            .unwrap();
        let held = policy.try_slot().unwrap();
        assert!(
            policy.try_slot().is_none(),
            "the Zenoh front end holds one of two slots"
        );
        drop(held);

        let mut c = zenoh::Config::default();
        for (k, v) in [
            ("mode", "\"client\"".to_string()),
            ("connect/endpoints", format!("[\"tcp/127.0.0.1:{port}\"]")),
            ("listen/endpoints", "[]".into()),
            ("scouting/multicast/enabled", "false".into()),
        ] {
            c.insert_json5(k, &v).unwrap();
        }
        let client = zenoh::open(c).await.unwrap();
        let get = |key: &'static str, payload: Vec<u8>| {
            let client = &client;
            async move {
                let r = client
                    .get(format!("physicalai/inference/cell1/{key}"))
                    .payload(payload)
                    .timeout(Duration::from_secs(10))
                    .await
                    .unwrap()
                    .recv_async()
                    .await
                    .unwrap();
                match r.result() {
                    Ok(s) => Ok(s.payload().to_bytes().into_owned()),
                    Err(e) => Err(e.payload().to_bytes().into_owned()),
                }
            }
        };
        let fx = |n: &str| {
            std::fs::read(format!(
                "{}/tests/fixtures/physicalai/{n}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap()
        };
        let meta = get("metadata", fx("metadata-request.msgpack"))
            .await
            .unwrap();
        assert!(meta.windows(7).any(|w| w == b"smolvla"));
        assert!(meta.windows(32).any(|w| w == front.server_id().as_bytes()));
        let reply = get("predict", fx("predict-raw.msgpack")).await.unwrap();
        assert!(reply.windows(6).any(|w| w == b"action"));
        let mut bad = fx("predict-raw.msgpack");
        let at = bad.windows(5).position(|w| w == b"wrist").unwrap();
        bad[at..at + 5].copy_from_slice(b"other");
        let e = get("predict", bad).await.unwrap_err();
        assert_eq!(e, encode_error(ErrorCode::InvalidRequest, 3));
        assert_eq!(
            get("reset", fx("metadata-request.msgpack")).await.unwrap(),
            encode_reset_ok()
        );
        assert_eq!(front.counts().done.load(Ordering::Relaxed), 2);
        client.close().await.unwrap();
        front.stop().await;
        assert!(policy.try_slot().is_some(), "stopping frees the slot");
    }

    #[test]
    fn the_secure_config_takes_every_key() {
        secure_config("tcp/127.0.0.1:7447", 1 << 20).unwrap();
    }
}
