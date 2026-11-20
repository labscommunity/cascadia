use cascadia_diffusion::{
    manifest::Manifest,
    wire,
    worker::{ExecuteFuture, Executor, Worker},
    Error,
};
use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::sync::Notify;
struct HeldEngine {
    entered: Arc<Notify>,
    finish: Arc<Notify>,
    corrupt: bool,
}
impl Executor for HeldEngine {
    fn execute(&self, frame: Vec<u8>) -> ExecuteFuture {
        let entered = self.entered.clone();
        let finish = self.finish.clone();
        let corrupt = self.corrupt;
        Box::pin(async move {
            entered.notify_one();
            finish.notified().await;
            if corrupt {
                Err(Error::Invalid("engine reset".into()))
            } else {
                Ok(frame)
            }
        })
    }
}
#[tokio::test]
async fn network_cancellation_does_not_release_engine_and_unknown_failure_quarantines() {
    for corrupt in [false, true] {
        let entered = Arc::new(Notify::new());
        let finish = Arc::new(Notify::new());
        let manifest: Manifest =
            serde_json::from_str(include_str!("fixtures/manifest.json")).unwrap();
        let id = manifest.id().unwrap();
        let worker = Worker::new(
            Arc::new(manifest),
            id.clone(),
            BTreeSet::from(["unet".into()]),
            Arc::new(HeldEngine {
                entered: entered.clone(),
                finish: finish.clone(),
                corrupt,
            }),
        )
        .unwrap();
        let lease = worker
            .reserve(
                "peer",
                worker.status().boot,
                &id,
                BTreeSet::from(["unet".into()]),
                Duration::from_secs(60),
            )
            .unwrap();
        let (mut header, values) = wire::decode(include_bytes!("fixtures/frame.bin")).unwrap();
        header.lease = lease;
        let raw = wire::encode(&header, &values).unwrap();
        let executor = worker.clone();
        let request = tokio::spawn(async move { executor.execute("peer", raw).await });
        entered.notified().await;
        request.abort();
        worker.release("peer", lease).unwrap();
        assert!(worker.occupied());
        assert!(!worker.status().available);
        finish.notify_one();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if worker.quarantined() || !worker.occupied() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(worker.quarantined(), corrupt);
        assert_eq!(worker.status().available, !corrupt);
    }
}
type Reply = Box<dyn Fn(Vec<u8>) -> cascadia_diffusion::Result<Vec<u8>> + Send + Sync>;
struct Scripted(Reply);
impl Executor for Scripted {
    fn execute(&self, frame: Vec<u8>) -> ExecuteFuture {
        let reply = (self.0)(frame);
        Box::pin(async move { reply })
    }
}
/// Runs one call through a fresh worker, then releases its lease, and returns
/// the call's result and how often the state observer fired during the call.
async fn run_once(reply: Reply) -> (Arc<Worker>, cascadia_diffusion::Result<Vec<u8>>, usize) {
    let manifest: Manifest = serde_json::from_str(include_str!("fixtures/manifest.json")).unwrap();
    let id = manifest.id().unwrap();
    let worker = Worker::new(
        Arc::new(manifest),
        id.clone(),
        BTreeSet::from(["unet".into()]),
        Arc::new(Scripted(reply)),
    )
    .unwrap();
    let lease = worker
        .reserve(
            "peer",
            worker.status().boot,
            &id,
            BTreeSet::from(["unet".into()]),
            Duration::from_secs(60),
        )
        .unwrap();
    let changes = Arc::new(AtomicUsize::new(0));
    let counter = changes.clone();
    worker.observe_state(Arc::new(move || {
        counter.fetch_add(1, Ordering::SeqCst);
    }));
    let (mut header, values) = wire::decode(include_bytes!("fixtures/frame.bin")).unwrap();
    header.lease = lease;
    let raw = wire::encode(&header, &values).unwrap();
    let result = worker.execute("peer", raw).await;
    let changed = changes.load(Ordering::SeqCst);
    worker.release("peer", lease).unwrap();
    (worker, result, changed)
}
#[tokio::test]
async fn rejected_call_leaves_worker_idle_and_notifies() {
    let (worker, result, changed) = run_once(Box::new(|_| Err(Error::Rejected))).await;
    assert!(matches!(result, Err(Error::Rejected)));
    assert!(!worker.quarantined());
    assert!(!worker.occupied());
    assert!(worker.status().available);
    assert_eq!(changed, 1);
}
struct PanicEngine {
    entered: Arc<Notify>,
    finish: Arc<Notify>,
}
impl Executor for PanicEngine {
    fn execute(&self, _: Vec<u8>) -> ExecuteFuture {
        let entered = self.entered.clone();
        let finish = self.finish.clone();
        Box::pin(async move {
            entered.notify_one();
            finish.notified().await;
            panic!("engine panicked");
        })
    }
}
#[tokio::test]
async fn panic_with_the_caller_gone_quarantines_and_notifies() {
    let entered = Arc::new(Notify::new());
    let finish = Arc::new(Notify::new());
    let manifest: Manifest = serde_json::from_str(include_str!("fixtures/manifest.json")).unwrap();
    let id = manifest.id().unwrap();
    let worker = Worker::new(
        Arc::new(manifest),
        id.clone(),
        BTreeSet::from(["unet".into()]),
        Arc::new(PanicEngine {
            entered: entered.clone(),
            finish: finish.clone(),
        }),
    )
    .unwrap();
    let lease = worker
        .reserve(
            "peer",
            worker.status().boot,
            &id,
            BTreeSet::from(["unet".into()]),
            Duration::from_secs(60),
        )
        .unwrap();
    // Each observer call records whether the quarantine was already visible.
    let (sender, mut seen) = tokio::sync::mpsc::unbounded_channel();
    let observed = Arc::downgrade(&worker);
    worker.observe_state(Arc::new(move || {
        let _ = sender.send(observed.upgrade().is_some_and(|w| w.quarantined()));
    }));
    let (mut header, values) = wire::decode(include_bytes!("fixtures/frame.bin")).unwrap();
    header.lease = lease;
    let raw = wire::encode(&header, &values).unwrap();
    let executor = worker.clone();
    let request = tokio::spawn(async move { executor.execute("peer", raw).await });
    entered.notified().await;
    request.abort();
    assert!(request.await.unwrap_err().is_cancelled());
    finish.notify_one();
    let quarantined = tokio::time::timeout(Duration::from_secs(5), seen.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(quarantined);
    assert!(worker.quarantined());
    assert!(seen.try_recv().is_err());
}
#[tokio::test]
async fn uncertain_failure_quarantines_and_notifies() {
    let (worker, result, changed) =
        run_once(Box::new(|_| Err(Error::Invalid("engine reset".into())))).await;
    assert!(matches!(result, Err(Error::Quarantined)));
    assert!(worker.quarantined());
    assert!(!worker.status().available);
    assert_eq!(changed, 1);
}
#[tokio::test]
async fn reply_for_another_execution_or_schema_quarantines() {
    let other_execution: Reply = Box::new(|frame| {
        let (mut header, values) = wire::decode(&frame)?;
        header.sequence += 1;
        wire::encode(&header, &values)
    });
    let other_schema: Reply = Box::new(|frame| {
        let (mut header, values) = wire::decode(&frame)?;
        header.tensors[0].name = "other".into();
        wire::encode(&header, &values)
    });
    for reply in [other_execution, other_schema] {
        let (worker, result, changed) = run_once(reply).await;
        assert!(matches!(result, Err(Error::Quarantined)), "{result:?}");
        assert!(worker.quarantined());
        assert_eq!(changed, 1);
    }
}
