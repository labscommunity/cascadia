//! Async lookahead prefetch for streamed MoE experts.
//!
//! Decode is bottlenecked on reading routed experts from disk. While the main
//! thread computes layer L, a single background worker warms layer L+1's likely
//! experts into the OS page cache, so the main thread's mmap access soft-faults
//! instead of blocking on disk.
//!
//! Prediction: layer L+1's own router run on an attention-free proxy of L's
//! output (see `MoeLayer::predict_experts`). Only experts that are NOT already
//! resident are enqueued (checked on the main thread before enqueue). A shared
//! `cur_layer` marker lets the worker skip any layer the main thread has already
//! reached — warming it would be wasted.
//!
//! Warming is best-effort and never changes output: a wrong or late prediction
//! only spends read bandwidth. The main thread never blocks on the worker.

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Bin file + on-disk length for one expert — what the worker needs to warm its
/// pages. `None` for non-mmap experts (bf16/eager; the test path).
pub type Bin = Option<(PathBuf, u64)>;

/// One sparse layer's routed-expert bins, indexed by expert id.
pub struct LayerBins {
    pub routed: Vec<Bin>,
}

/// Every owned layer's bins, indexed by LOCAL layer index; `None` for dense
/// layers (no routed experts to prefetch).
pub type LookaheadTable = Vec<Option<LayerBins>>;

/// A warm request: warm routed expert `eid` of local layer `layer`.
#[derive(Clone, Copy)]
struct Job {
    layer: u16,
    eid: u16,
}

/// Request ring depth. Bounded; a full ring drops the request — a stale
/// prediction is worthless, so never stall the compute thread for it.
const RING: usize = 4096;

/// Sequential read chunk for warming a bin's pages without a full-bin heap
/// allocation (read + discard through a small reused buffer).
const WARM_CHUNK: usize = 1 << 20; // 1 MiB

/// Background prefetch worker + the shared marker the main thread uses to keep
/// the worker from warming a layer it has already reached.
pub struct Lookahead {
    tx: Option<SyncSender<Job>>,
    /// The layer the main thread is currently on. The worker drops any job for a
    /// layer `<=` this — the main thread already owns it.
    cur_layer: Arc<AtomicI32>,
    stop: Arc<AtomicBool>,
    loads: Arc<AtomicU64>,
    drops: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
}

impl Lookahead {
    /// Spawn the prefetch worker over `table`.
    pub fn new(table: LookaheadTable) -> Self {
        let table = Arc::new(table);
        let cur_layer = Arc::new(AtomicI32::new(-1));
        let stop = Arc::new(AtomicBool::new(false));
        let loads = Arc::new(AtomicU64::new(0));
        let drops = Arc::new(AtomicU64::new(0));
        let (tx, rx) = sync_channel::<Job>(RING);

        let worker = {
            let table = Arc::clone(&table);
            let cur_layer = Arc::clone(&cur_layer);
            let stop = Arc::clone(&stop);
            let loads = Arc::clone(&loads);
            let drops = Arc::clone(&drops);
            std::thread::Builder::new()
                .name("glm5-lookahead".into())
                .spawn(move || {
                    let mut buf = vec![0u8; WARM_CHUNK];
                    // Single consumer: recv blocks until a job or all senders drop.
                    while let Ok(job) = rx.recv() {
                        if stop.load(Ordering::Acquire) {
                            break;
                        }
                        // Drop only layers the main thread is already PAST. A job
                        // for the current layer is kept: the worker warms experts
                        // the main thread hasn't reached yet within that layer.
                        if (job.layer as i32) < cur_layer.load(Ordering::Acquire) {
                            drops.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        let bin = table
                            .get(job.layer as usize)
                            .and_then(|o| o.as_ref())
                            .and_then(|lb| lb.routed.get(job.eid as usize))
                            .and_then(|b| b.as_ref());
                        match bin {
                            Some((path, _len)) if warm_file(path, &mut buf) => {
                                loads.fetch_add(1, Ordering::Relaxed);
                            }
                            _ => {
                                drops.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                })
                .expect("spawn glm5-lookahead worker")
        };

        Self {
            tx: Some(tx),
            cur_layer,
            stop,
            loads,
            drops,
            worker: Some(worker),
        }
    }

    /// Publish the layer the main thread is now computing. Call before computing
    /// each layer so the worker can skip layers already reached.
    #[inline]
    pub fn set_cur_layer(&self, local_layer: usize) {
        self.cur_layer.store(local_layer as i32, Ordering::Release);
    }

    /// Enqueue a warm for routed expert `eid` of local layer `layer`.
    /// Non-blocking: a full ring drops the request. Never blocks decode.
    #[inline]
    pub fn enqueue(&self, layer: usize, eid: u32) {
        if let Some(tx) = &self.tx {
            if tx
                .try_send(Job {
                    layer: layer as u16,
                    eid: eid as u16,
                })
                .is_err()
            {
                self.drops.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// `(warmed, dropped)` counters.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.loads.load(Ordering::Relaxed),
            self.drops.load(Ordering::Relaxed),
        )
    }
}

impl Drop for Lookahead {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        drop(self.tx.take()); // close the channel -> worker recv() returns Err -> exits
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
        let (w, d) = self.stats();
        eprintln!("[glm5] lookahead prefetch: warmed={w} dropped={d}");
    }
}

/// Sequentially read `path` through `buf` (discarding) to pull its pages into
/// the OS page cache. Returns whether the file opened + read without error.
/// Best-effort: any I/O error is swallowed (a failed warm is harmless).
fn warm_file(path: &std::path::Path, buf: &mut [u8]) -> bool {
    let Ok(mut f) = File::open(path) else {
        return false;
    };
    loop {
        match f.read(buf) {
            Ok(0) => return true,
            Ok(_) => {}
            Err(_) => return false,
        }
    }
}
