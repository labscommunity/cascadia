//! Runner-side async expert prefetcher.
//!
//! Spawns a worker thread fed by the inference path's `try_submit` calls.
//! Each `PrefetchReq` resolves to six prefetch hits per expert
//! (gate / up / down × packed / scale). The actual hint mechanism is
//! one of two backends, selected at construction:
//!
//!   * [`PrefetchBackendKind::Madvise`] — `madvise(MADV_WILLNEED)` on
//!     each tensor's byte range via [`Shard::advise_willneed`]. iter 033
//!     baseline; works on every Unix.
//!   * [`PrefetchBackendKind::IoUring`] — `IORING_OP_READ` SQEs via
//!     [`Shard::async_read`] + the iter 097 [`AsyncPrefetchBackend`].
//!     Linux-only; falls through to a no-op on every other platform
//!     (use `Madvise` there).
//!
//! Both backends are pure side-effects on the OS page cache. The
//! inference path's `dispatch_expert` resolves bytes through the same
//! mmap regardless of which prefetcher fired — so the runner's output
//! is bit-identical across backends. The
//! `bit_exact_output_across_backends` test below pins that contract
//! against a synthetic safetensors-shaped tempfile.
//!
//! Lifecycle:
//!
//!   * [`AsyncPrefetcher::spawn`] — starts the worker thread, returns a
//!     handle. The worker terminates when the `Sender` half is dropped
//!     (on `AsyncPrefetcher::drop`).
//!   * [`AsyncPrefetcher::try_submit`] — non-blocking enqueue. Drops the
//!     request on overflow rather than stalling the inference path (the
//!     demand mmap path will resolve the bytes later — that's the
//!     no-prefetch baseline).
//!   * [`AsyncPrefetcher::snapshot`] — returns the counters for
//!     instrumentation logs.
//!   * Drop — closes the sender, joins the worker thread.

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use tahoma_int4_gemm::async_prefetch::{
    AsyncPrefetchBackend, AsyncReadHandle, PrefetchBackendKind,
};
use tahoma_int4_gemm::SafetensorsExpertSource;
use tracing::{debug, info};

/// One expert-prefetch request. The worker thread resolves it to six
/// hits against the expert's tensor slices (gate / up / down × packed /
/// scale).
#[derive(Copy, Clone, Debug)]
pub struct PrefetchReq {
    pub lid: u32,
    pub eid: u32,
}

/// Diagnostic counters per [`AsyncPrefetcher`] lifetime. Returned by
/// [`AsyncPrefetcher::snapshot`] for the per-token instrumentation line;
/// not load-bearing.
#[derive(Copy, Clone, Debug, Default)]
pub struct PrefetchStats {
    /// Calls to `try_submit` that landed in the channel.
    pub submits: u64,
    /// Calls to `try_submit` that hit a full / disconnected channel.
    pub drops: u64,
    /// Requests the worker thread has fully processed (issued all 6
    /// hints / SQEs for, or attempted to and skipped on error).
    pub completed: u64,
    /// SQEs the io_uring backend rejected with `QueueFull` (madvise path
    /// never raises this). Lets the inference loop detect a sustained
    /// over-submit pattern and back off.
    pub backend_queue_full: u64,
    /// Tensor slices the worker thread successfully prefetched (sum
    /// across all completed requests). For madvise this is six per
    /// expert on a healthy filesystem; for io_uring it's six minus any
    /// SQ rejections.
    pub slices_pushed: u64,
}

/// Background thread that consumes [`PrefetchReq`] and issues prefetch
/// hints against the expert's six tensor slices.
///
/// One thread is enough — both madvise (cheap syscall) and io_uring
/// submission (non-blocking) are fast; the actual I/O happens
/// asynchronously in the kernel (madvise) or the io_uring reaper
/// thread.
pub struct AsyncPrefetcher {
    tx: Option<SyncSender<PrefetchReq>>,
    join: Option<JoinHandle<()>>,
    submits: Arc<AtomicU64>,
    drops: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
    backend_queue_full: Arc<AtomicU64>,
    slices_pushed: Arc<AtomicU64>,
    /// Which path is firing (frozen at construction).
    kind: PrefetchBackendKind,
    /// Sticks around so callers can interrogate whether the io_uring
    /// path is actually live vs the fallback (logged on startup,
    /// useful in bench comparisons). Only meaningful when
    /// `kind == PrefetchBackendKind::IoUring`.
    backend: AsyncPrefetchBackend,
}

impl AsyncPrefetcher {
    /// Spawn the worker thread.
    ///
    /// `source` is the same shared `SafetensorsExpertSource` the runner
    /// uses for demand reads — we re-use its shard cache so prefetch
    /// doesn't double-open files.
    ///
    /// `kind` selects the prefetch path (see [`PrefetchBackendKind`]).
    /// For `IoUring`, the backend is constructed via
    /// [`AsyncPrefetchBackend::new`] which probes for io_uring at init
    /// time and silently falls back to a no-op on non-Linux / WSL2 /
    /// seccomp-blocked containers. For `Madvise`, the backend is still
    /// constructed (cheap) but never used; the worker calls
    /// `SafetensorsExpertSource::prefetch_expert_madvise` directly.
    ///
    /// `channel_depth` — bounded queue size. Production default is 4096
    /// (~22 tokens of 6 experts × 30 layers at K=6).
    pub fn spawn(
        source: Arc<SafetensorsExpertSource>,
        kind: PrefetchBackendKind,
        channel_depth: usize,
    ) -> Self {
        // Construct the io_uring backend either way — it's cheap (~one
        // syscall + a NOP probe on Linux), and exposing `using_io_uring`
        // on the AsyncPrefetcher lets the runner's startup log
        // distinguish "asked for io_uring, got it" from "asked for
        // io_uring, fell back". On the Madvise path the backend is
        // never touched.
        let backend = AsyncPrefetchBackend::new();
        info!(
            backend = kind.as_str(),
            using_io_uring = backend.using_io_uring(),
            channel_depth,
            "spawning async expert prefetcher"
        );

        let (tx, rx) = mpsc::sync_channel::<PrefetchReq>(channel_depth);
        let submits = Arc::new(AtomicU64::new(0));
        let drops = Arc::new(AtomicU64::new(0));
        let completed = Arc::new(AtomicU64::new(0));
        let backend_queue_full = Arc::new(AtomicU64::new(0));
        let slices_pushed = Arc::new(AtomicU64::new(0));

        let source_for_thread = source.clone();
        let backend_for_thread = backend.clone();
        let completed_thread = completed.clone();
        let queue_full_thread = backend_queue_full.clone();
        let slices_thread = slices_pushed.clone();
        let kind_for_thread = kind;

        let thread_name = match kind {
            PrefetchBackendKind::Madvise => "expert-prefetch-madvise".into(),
            PrefetchBackendKind::IoUring => "expert-prefetch-iouring".into(),
        };

        let join = thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                // Plain blocking recv loop. Terminates when the sender
                // side drops (on AsyncPrefetcher::drop). We don't pull
                // CQEs here — the io_uring backend has its own
                // completion thread (lives in IoUringState); this
                // thread's only job is to push SQEs / madvise hints.
                while let Ok(req) = rx.recv() {
                    let n = process_one(
                        &source_for_thread,
                        &backend_for_thread,
                        kind_for_thread,
                        req,
                        &queue_full_thread,
                    );
                    slices_thread.fetch_add(n as u64, AtomicOrdering::Relaxed);
                    completed_thread.fetch_add(1, AtomicOrdering::Relaxed);
                    debug!(
                        lid = req.lid,
                        eid = req.eid,
                        slices_pushed = n,
                        backend = kind_for_thread.as_str(),
                        "prefetch req processed"
                    );
                }
            })
            .expect("spawn expert-prefetch thread");

        Self {
            tx: Some(tx),
            join: Some(join),
            submits,
            drops,
            completed,
            backend_queue_full,
            slices_pushed,
            kind,
            backend,
        }
    }

    /// Non-blocking enqueue. Drops the request on overflow.
    ///
    /// The dispatcher should call this once per (predicted lid, eid) at
    /// the start of `forward_shells` (or earlier — the further ahead
    /// the better). Drops are silently accepted; we'd rather miss a
    /// prefetch than block the inference path.
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
            slices_pushed: self.slices_pushed.load(AtomicOrdering::Relaxed),
        }
    }

    /// Which backend was selected at construction.
    pub fn kind(&self) -> PrefetchBackendKind {
        self.kind
    }

    /// Did the underlying io_uring backend pick the real io_uring path
    /// at init, or did it fall through to the no-op Fallback? Only
    /// meaningful when `kind() == IoUring`; on the Madvise path this
    /// just reflects whether io_uring *would have worked* (the worker
    /// thread doesn't use the backend either way).
    pub fn using_io_uring(&self) -> bool {
        self.backend.using_io_uring()
    }

    /// Synchronously drain the request channel and process any backlog
    /// before returning. Test-only: production paths rely on the
    /// background thread's natural drain in Drop. The deterministic test
    /// in this module uses this to assert the worker has finished
    /// processing every request before checking the counters.
    ///
    /// Takes `&mut self` so the test can call it on a borrow without
    /// fighting Drop. Subsequent `try_submit` calls become no-ops (the
    /// sender is gone); subsequent `snapshot` calls return the final
    /// counter values.
    #[cfg(test)]
    pub fn flush_for_test(&mut self) -> PrefetchStats {
        // Drop the sender → worker thread exits its recv loop after
        // draining. Then join the worker so the counters are stable.
        drop(self.tx.take());
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
        self.snapshot()
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

/// Issue prefetch hints for the six slices that compose one expert.
///
/// Returns the number of slices that were successfully hinted. For the
/// io_uring path that's the number of SQEs pushed; for the madvise
/// path it's the number of `MADV_WILLNEED` calls that succeeded.
/// Either way, the bytes flow through the same demand path
/// (`Shard::slice` + mmap fault) later — prefetch is purely a page-cache
/// warming side-effect.
///
/// The returned [`AsyncReadHandle`]s (io_uring only) are dropped at
/// end-of-scope. We don't currently retain them for the dispatcher to
/// consult; a future milestone may, so the dispatcher can short-circuit
/// when the read is still inflight.
fn process_one(
    source: &SafetensorsExpertSource,
    backend: &AsyncPrefetchBackend,
    kind: PrefetchBackendKind,
    req: PrefetchReq,
    queue_full: &Arc<AtomicU64>,
) -> usize {
    match kind {
        PrefetchBackendKind::Madvise => {
            // One madvise per slice through the shared shard cache.
            // `prefetch_expert_madvise` skips missing tensors silently
            // and returns the count of hits.
            source.prefetch_expert_madvise(req.lid, req.eid)
        }
        PrefetchBackendKind::IoUring => process_one_io_uring(source, backend, req, queue_full),
    }
}

fn process_one_io_uring(
    source: &SafetensorsExpertSource,
    backend: &AsyncPrefetchBackend,
    req: PrefetchReq,
    queue_full: &Arc<AtomicU64>,
) -> usize {
    let names = SafetensorsExpertSource::expert_tensor_names(req.lid, req.eid);

    let mut pushed = 0usize;
    let mut _handles: Vec<AsyncReadHandle> = Vec::with_capacity(names.len());

    for name in names.iter() {
        // Resolve the shard. We use the public `tensor_bytes` API for
        // the Arc-pin behavior, then ignore the slice and just use the
        // shard for its `async_read`. Slightly wasteful (we resolve
        // the slice twice), but the alternative is to expose a
        // shard-lookup-only API which a future milestone may do.
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
                // Don't try to push the rest of this expert — the queue
                // is full, the demand path will resolve it.
                break;
            }
            Err(tahoma_int4_gemm::async_prefetch::AsyncIoError::NotImplemented) => {
                // Reserved for future use; current backend never emits.
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
    use std::collections::HashMap;
    use std::io::Write;
    use std::path::PathBuf;

    /// Synthesize a tiny safetensors shard + an index file pointing at
    /// it, containing the six tensor slices for `(layer=L, expert=E)`
    /// at a fixed `(L, E)` and arbitrary `payload` bytes. Returns the
    /// model dir path (caller must keep the TempDir alive).
    ///
    /// We use this to exercise the prefetcher end-to-end without a real
    /// K2.6 checkpoint — the model dir layout (one shard +
    /// `model.safetensors.index.json`) is what `SafetensorsExpertSource`
    /// reads, and that's enough to drive `prefetch_expert_madvise` and
    /// `tensor_bytes` for both backends.
    fn make_fake_model_dir(payload: &[u8]) -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let model_dir = tmp.path().to_path_buf();

        let shard_name = "model-00001-of-00001.safetensors";
        let shard_path = model_dir.join(shard_name);

        let layer = 1u32;
        let expert = 0u32;
        let base = format!("language_model.model.layers.{layer}.mlp.experts.{expert}");
        let names = [
            format!("{base}.gate_proj.weight_packed"),
            format!("{base}.gate_proj.weight_scale"),
            format!("{base}.up_proj.weight_packed"),
            format!("{base}.up_proj.weight_scale"),
            format!("{base}.down_proj.weight_packed"),
            format!("{base}.down_proj.weight_scale"),
        ];

        // Build the safetensors metadata JSON: six tensors back-to-back
        // in the data section, each `payload.len()` bytes long.
        let mut json = serde_json::Map::new();
        for (i, n) in names.iter().enumerate() {
            let start = (i * payload.len()) as u64;
            let end = ((i + 1) * payload.len()) as u64;
            let entry = serde_json::json!({
                "dtype": "U8",
                "shape": [payload.len()],
                "data_offsets": [start, end],
            });
            json.insert(n.clone(), entry);
        }
        let json_bytes = serde_json::to_vec(&json).expect("json");

        let mut shard_file = std::fs::File::create(&shard_path).expect("create shard");
        shard_file
            .write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write header len");
        shard_file.write_all(&json_bytes).expect("write json");
        for _ in 0..names.len() {
            shard_file.write_all(payload).expect("write payload");
        }
        shard_file.flush().expect("flush shard");
        drop(shard_file);

        // Build the index JSON pointing every tensor at the one shard.
        let mut weight_map: HashMap<&str, &str> = HashMap::new();
        for n in names.iter() {
            weight_map.insert(n.as_str(), shard_name);
        }
        let idx = serde_json::json!({ "weight_map": weight_map });
        std::fs::write(
            model_dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&idx).expect("idx json"),
        )
        .expect("write idx");

        (tmp, model_dir)
    }

    #[test]
    fn stats_default_is_zero() {
        let s = PrefetchStats::default();
        assert_eq!(s.submits, 0);
        assert_eq!(s.drops, 0);
        assert_eq!(s.completed, 0);
        assert_eq!(s.backend_queue_full, 0);
        assert_eq!(s.slices_pushed, 0);
    }

    /// Parsing the env-var-style backend name. Spelling variants accepted
    /// for ergonomics (`io-uring` vs `io_uring`).
    #[test]
    fn backend_kind_parse() {
        assert_eq!(
            PrefetchBackendKind::from_str_ci("madvise").unwrap(),
            PrefetchBackendKind::Madvise
        );
        assert_eq!(
            PrefetchBackendKind::from_str_ci("io-uring").unwrap(),
            PrefetchBackendKind::IoUring
        );
        assert_eq!(
            PrefetchBackendKind::from_str_ci("IO_URING").unwrap(),
            PrefetchBackendKind::IoUring
        );
        assert!(PrefetchBackendKind::from_str_ci("nonesuch").is_err());
    }

    /// End-to-end: spawn each backend variant against the same synthetic
    /// model dir, push the same prefetch requests through both, drain
    /// the worker threads, and assert:
    ///
    ///   1. Both backends process every request (completed counter).
    ///   2. The on-disk shard bytes are *unchanged* afterward —
    ///      proves that neither backend mutates the model. Since
    ///      `dispatch_expert` reads via the same mmap regardless of
    ///      which prefetch ran, byte-identical inputs to the kernel
    ///      mean byte-identical outputs.
    ///   3. The shard's `slice()` returns the same bytes after each
    ///      prefetch backend runs — pinning the "page-cache-only
    ///      side-effect" invariant directly.
    ///
    /// This is the structural proof that `--prefetch-backend madvise`
    /// and `--prefetch-backend io-uring` produce bit-identical model
    /// output. We can't load a real K2.6 checkpoint in CI, but the
    /// prefetcher's only knob into the inference path is the page
    /// cache, and the test fixes the bytes against any unintended
    /// mutation.
    #[test]
    fn bit_exact_output_across_backends() {
        // Use a payload that's distinctive enough that an accidental
        // mutation would show up immediately.
        let payload: Vec<u8> = (0u8..255).cycle().take(8192).collect();
        let (_tmp, model_dir) = make_fake_model_dir(&payload);

        let source =
            Arc::new(SafetensorsExpertSource::open(model_dir.clone()).expect("open source"));

        // Snapshot the on-disk bytes for the gate_packed slice before
        // any prefetch runs.
        let baseline_bytes = source
            .tensor_bytes("language_model.model.layers.1.mlp.experts.0.gate_proj.weight_packed")
            .expect("baseline lookup")
            .1
            .to_vec();

        // --- Madvise path ---
        let mut pf = AsyncPrefetcher::spawn(source.clone(), PrefetchBackendKind::Madvise, 64);
        for _ in 0..4 {
            pf.try_submit(1, 0);
        }
        let stats_madvise = pf.flush_for_test();
        assert_eq!(
            stats_madvise.submits, 4,
            "madvise: expected 4 submits, got {stats_madvise:?}"
        );
        assert_eq!(
            stats_madvise.completed, 4,
            "madvise: expected 4 completed, got {stats_madvise:?}"
        );
        // Madvise should hit 6 slices per request when the filesystem
        // supports it (Linux/macOS); on platforms where memmap2's
        // `advise_range` is a no-op the count may be 0, but the rest of
        // the contract still holds. We accept either as long as it's
        // consistent (== 0 or == 24).
        assert!(
            stats_madvise.slices_pushed == 0 || stats_madvise.slices_pushed == 24,
            "madvise: expected 0 or 24 slice pushes, got {stats_madvise:?}"
        );

        let after_madvise = source
            .tensor_bytes("language_model.model.layers.1.mlp.experts.0.gate_proj.weight_packed")
            .expect("after-madvise lookup")
            .1
            .to_vec();
        assert_eq!(
            after_madvise, baseline_bytes,
            "madvise prefetch must not mutate shard bytes"
        );

        // --- IoUring path ---
        let mut pf = AsyncPrefetcher::spawn(source.clone(), PrefetchBackendKind::IoUring, 64);
        for _ in 0..4 {
            pf.try_submit(1, 0);
        }
        let stats_iouring = pf.flush_for_test();
        assert_eq!(
            stats_iouring.submits, 4,
            "io-uring: expected 4 submits, got {stats_iouring:?}"
        );
        assert_eq!(
            stats_iouring.completed, 4,
            "io-uring: expected 4 completed, got {stats_iouring:?}"
        );

        let after_iouring = source
            .tensor_bytes("language_model.model.layers.1.mlp.experts.0.gate_proj.weight_packed")
            .expect("after-iouring lookup")
            .1
            .to_vec();
        assert_eq!(
            after_iouring, baseline_bytes,
            "io-uring prefetch must not mutate shard bytes"
        );

        // The clinching assertion: the bytes the kernel would read for
        // `dispatch_expert` are identical across both backends. Since
        // the kernel is pure functional over its inputs, that means
        // the routing/MoE output is identical too.
        assert_eq!(
            after_madvise, after_iouring,
            "bytes seen by dispatch_expert must be identical across backends"
        );
    }
}
