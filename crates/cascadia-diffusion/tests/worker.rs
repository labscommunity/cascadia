use cascadia_diffusion::{
    manifest::Manifest,
    wire,
    worker::{ExecuteFuture, Executor, Worker},
    Error,
};
use std::{collections::BTreeSet, sync::Arc, time::Duration};
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
