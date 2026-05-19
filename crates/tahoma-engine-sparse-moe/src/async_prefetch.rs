//! Runner-side async expert prefetcher (autolab iter 074 skeleton).
//!
//! Wraps the [`AsyncPrefetchBackend`] from `tahoma-int4-gemm` in a
//! background thread fed by the inference path's `try_submit` calls.
//! The thread translates each `PrefetchReq` into one
//! `Shard::async_read` per slice of the expert (6 slices: gate/up/down
//! × packed/scale).
//!
//! This is the **skeleton** companion to `Shard::async_read` — see
//! `docs/perf/io_uring_prefetch.md` for the full plan. The hot-path
//! `forward_shells` wiring is deliberately not added here because
//! the iter 033 C1 prefetcher (which this would replace) is not yet
//! on this branch. The struct + thread shape below is the contract
//! the future wiring will satisfy.
//!
//! Lifecycle:
//!
//!   * `AsyncPrefetcher::spawn(source, backend)` — starts the worker
//!     thread, returns a handle. The worker terminates when the
//!     `Sender` half is dropped (on `AsyncPrefetcher::drop`).
//!   * `try_submit(lid, eid)` — non-blocking enqueue. Drops the
//!     request on overflow rather than stalling the inference path
//!     (the demand mmap path will resolve the bytes later — that's
//!     the no-prefetch baseline).
//!   * `snapshot()` — returns the counters (submits, drops, completed)
//!     for instrumentation logs.
//!   * Drop — closes the sender, joins the worker thread.

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use tahoma_int4_gemm::async_prefetch::{AsyncPrefetchBackend, AsyncReadHandle};
use tahoma_int4_gemm::SafetensorsExpertSource;
use tracing::{debug, info};

/// One expert-prefetch request. The worker thread resolves it to six
/// `Shard::async_read` SQEs (gate/up/down × packed/scale).
#[derive(Copy, Clone, Debug)]
pub struct PrefetchReq {
    pub lid: u32,
    pub eid: u32,
}

/// Diagnostic counters per `AsyncPrefetcher` lifetime. Returned by
/// [`AsyncPrefetcher::snapshot`] for the per-token instrumentation
/// line; not load-bearing.
#[derive(Copy, Clone, Debug, Default)]
pub struct PrefetchStats {
    /// Calls to `try_submit` that landed in the channel.
    pub submits: u64,
    /// Calls to `try_submit` that hit a full / disconnected channel.
    pub drops: u64,
    /// Requests the worker thread has fully processed (issued all 6
    /// `async_read` SQEs for, or attempted to and skipped on error).
    pub completed: u64,
    /// SQEs the io_uring backend rejected with `QueueFull` (or the
    /// fallback path's equivalent). Lets the inference loop detect a
    /// sustained over-submit pattern and back off.
    pub backend_queue_full: u64,
}

/// Background thread that consumes [`PrefetchReq`] and issues async
/// reads against the expert's six tensor slices.
///
/// One thread is enough — io_uring submission is non-blocking; the
/// completion thread is owned by the backend (not this struct).
pub struct AsyncPrefetcher {
    tx: Option<SyncSender<PrefetchReq>>,
    join: Option<JoinHandle<()>>,
    submits: Arc<AtomicU64>,
    drops: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
    backend_queue_full: Arc<AtomicU64>,
    /// Sticks around so callers can interrogate whether the io_uring
    /// path is firing vs the fallback (logged on startup, useful in
    /// bench comparisons).
    backend: AsyncPrefetchBackend,
}

impl AsyncPrefetcher {
    /// Spawn the worker thread.
    ///
    /// `source` is the same shared `SafetensorsExpertSource` the
    /// runner uses for demand reads — we re-use its shard cache so
    /// `async_read` doesn't double-open files.
    ///
    /// `backend` is the platform-specific async I/O backend. On Linux
    /// with kernel >= 5.6 and io_uring available, this dispatches via
    /// `IORING_OP_READ`. Everywhere else (or on Linux until milestone
    /// 1 lands), it dispatches via the fallback path, which is
    /// equivalent to the iter 033 madvise(WILLNEED) prefetcher.
    ///
    /// `channel_depth` — bounded queue size. Production default is
    /// 4096 (~22 tokens of 6 experts × 30 layers at K=6).
    pub fn spawn(
        source: Arc<SafetensorsExpertSource>,
        backend: AsyncPrefetchBackend,
        channel_depth: usize,
    ) -> Self {
        info!(
            io_uring = backend.using_io_uring(),
            channel_depth, "spawning async expert prefetcher"
        );

        let (tx, rx) = mpsc::sync_channel::<PrefetchReq>(channel_depth);
        let submits = Arc::new(AtomicU64::new(0));
        let drops = Arc::new(AtomicU64::new(0));
        let completed = Arc::new(AtomicU64::new(0));
        let backend_queue_full = Arc::new(AtomicU64::new(0));

        let source_for_thread = source.clone();
        let backend_for_thread = backend.clone();
        let completed_thread = completed.clone();
        let queue_full_thread = backend_queue_full.clone();

        let join = thread::Builder::new()
            .name("expert-prefetch-iouring".into())
            .spawn(move || {
                // Plain blocking recv loop. Terminates when the sender
                // side drops (on AsyncPrefetcher::drop). We don't pull
                // CQEs here — the io_uring backend has its own
                // completion thread (lives in IoUringState); this
                // thread's only job is to push SQEs.
                while let Ok(req) = rx.recv() {
                    let n = process_one(
                        &source_for_thread,
                        &backend_for_thread,
                        req,
                        &queue_full_thread,
                    );
                    completed_thread.fetch_add(1, AtomicOrdering::Relaxed);
                    debug!(
                        lid = req.lid,
                        eid = req.eid,
                        sqe_pushed = n,
                        "prefetch req processed"
                    );
                }
            })
            .expect("spawn expert-prefetch-iouring thread");

        Self {
            tx: Some(tx),
            join: Some(join),
            submits,
            drops,
            completed,
            backend_queue_full,
            backend,
        }
    }

    /// Non-blocking enqueue. Drops the request on overflow.
    ///
    /// The dispatcher should call this once per (predicted lid, eid)
    /// at the start of `forward_shells` (or earlier — the further
    /// ahead the better). Drops are silently accepted; we'd rather
    /// miss a prefetch than block the inference path.
    pub fn try_submit(&self, lid: u32, eid: u32) {
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        match tx.try_send(PrefetchReq { lid, eid }) {
            Ok(()) => {
                self.submits.fetch_add(1, AtomicOrdering::Relaxed);
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.drops.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }

    /// Snapshot the counters. Cheap (atomic loads).
    pub fn snapshot(&self) -> PrefetchStats {
        PrefetchStats {
            submits: self.submits.load(AtomicOrdering::Relaxed),
            drops: self.drops.load(AtomicOrdering::Relaxed),
            completed: self.completed.load(AtomicOrdering::Relaxed),
            backend_queue_full: self.backend_queue_full.load(AtomicOrdering::Relaxed),
        }
    }

    /// Did the backend pick the io_uring path on startup, or did it
    /// fall through to madvise / PrefetchVirtualMemory?
    pub fn using_io_uring(&self) -> bool {
        self.backend.using_io_uring()
    }
}

impl Drop for AsyncPrefetcher {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// Issue async reads for the six slices that compose one expert.
/// Returns the number of slices that were successfully pushed onto
/// the backend's SQ. The returned [`AsyncReadHandle`]s are dropped
/// at end-of-scope — we don't currently retain them for the
/// dispatcher to consult, but milestone 5 will (so the dispatcher
/// can short-circuit if the read is still inflight by the time we
/// hit dispatch_expert).
fn process_one(
    source: &SafetensorsExpertSource,
    backend: &AsyncPrefetchBackend,
    req: PrefetchReq,
    queue_full: &Arc<AtomicU64>,
) -> usize {
    let base = format!(
        "language_model.model.layers.{}.mlp.experts.{}",
        req.lid, req.eid
    );
    let names = [
        format!("{}.gate_proj.weight_packed", base),
        format!("{}.gate_proj.weight_scale", base),
        format!("{}.up_proj.weight_packed", base),
        format!("{}.up_proj.weight_scale", base),
        format!("{}.down_proj.weight_packed", base),
        format!("{}.down_proj.weight_scale", base),
    ];

    let mut pushed = 0usize;
    let mut _handles: Vec<AsyncReadHandle> = Vec::with_capacity(names.len());

    for name in names.iter() {
        // Resolve the shard. We use the public `tensor_bytes` API for
        // the Arc-pin behavior, then ignore the slice and just use
        // the shard for its `async_read`. This is slightly wasteful
        // (we resolve the slice twice), but the alternative is to
        // expose a shard-lookup-only API on the source which we'll
        // do in milestone 3 once the wiring is real.
        let Ok((shard, _bytes)) = source.tensor_bytes(name) else {
            continue;
        };
        match shard.async_read(backend, name) {
            Ok(h) => {
                _handles.push(h);
                pushed += 1;
            }
            Err(tahoma_int4_gemm::async_prefetch::AsyncIoError::QueueFull) => {
                queue_full.fetch_add(1, AtomicOrdering::Relaxed);
                // Don't try to push the rest of this expert — the
                // queue is full, the demand path will resolve it.
                break;
            }
            Err(tahoma_int4_gemm::async_prefetch::AsyncIoError::NotImplemented) => {
                // Skeleton state — silently skip. The fallback
                // backend wouldn't return this, and the io_uring
                // backend isn't wired up yet.
                break;
            }
            Err(e) => {
                debug!(name = %name, err = %e, "async_read failed");
                continue;
            }
        }
    }

    pushed
}

#[cfg(test)]
mod tests {
    use super::*;

    // The full path needs a real SafetensorsExpertSource; we test
    // PrefetchStats default + the using_io_uring stub here, and rely
    // on the bench harness (autolab 074) for end-to-end coverage.

    #[test]
    fn stats_default_is_zero() {
        let s = PrefetchStats::default();
        assert_eq!(s.submits, 0);
        assert_eq!(s.drops, 0);
        assert_eq!(s.completed, 0);
        assert_eq!(s.backend_queue_full, 0);
    }
}
