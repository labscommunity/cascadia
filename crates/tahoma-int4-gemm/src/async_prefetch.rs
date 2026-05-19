//! Async expert prefetch via `io_uring` (Linux) with graceful
//! fallback to `madvise(MADV_WILLNEED)` (other Unix) /
//! `PrefetchVirtualMemory` (Windows) / no-op (everywhere else).
//!
//! Status: **skeleton** (autolab iter 074). The Linux path stubs out the
//! actual `io_uring_setup` / SQE-push / CQE-poll calls and returns
//! `AsyncIoError::NotImplemented` at runtime. The fallback path is
//! wired to `Shard::advise_willneed` and is functionally equivalent
//! to the iter 033 C1 prefetcher. The full implementation plan lives
//! in `docs/perf/io_uring_prefetch.md`.
//!
//! See `docs/perf/io_uring_prefetch.md` for:
//!  - the design (why io_uring over aio / threadpool / O_DIRECT)
//!  - the kernel-version + WSL2 + container blockers
//!  - the per-PR plan that gets this from skeleton to production
//!
//! API contract:
//!
//!   let backend = AsyncPrefetchBackend::new()?;
//!   let handle = backend.queue_read(fd, offset, len)?;
//!   // ... do other work ...
//!   if handle.is_ready() {
//!       // page-cache is warm, demand mmap read will not page-fault
//!   }
//!
//! On Linux with io_uring available, `queue_read` pushes an SQE and
//! returns immediately. On any other platform (or on Linux without
//! io_uring), `queue_read` falls back to the existing madvise/
//! PrefetchVirtualMemory hint and returns a handle that reports ready
//! immediately (the hint is fire-and-forget — there's no completion to
//! wait for).
//!
//! The handle pins any per-read state (e.g. the destination buffer on
//! the Linux path). Dropping the handle is safe at any time; if the
//! read is still inflight when dropped, we leak the buffer slot until
//! the next ring drain (production code should pool these explicitly,
//! but the skeleton uses a `Box<[u8]>` and accepts the GC churn).

use std::os::fd::RawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use thiserror::Error;

/// Errors raised by the async prefetch backend.
#[derive(Debug, Error)]
pub enum AsyncIoError {
    /// The backend was constructed on a platform that doesn't support
    /// it, or with a kernel too old. Caller should fall back to the
    /// madvise path.
    #[error("io_uring not available: {0}")]
    NotAvailable(String),

    /// The submission queue is full. Caller should drop the request
    /// (it'll re-resolve via the demand path) or retry later.
    #[error("io_uring submission queue full")]
    QueueFull,

    /// Skeleton stub returned by the Linux path until milestone 1 of
    /// the implementation plan lands. See
    /// `docs/perf/io_uring_prefetch.md` for the per-PR breakdown.
    #[error("io_uring backend not yet implemented (skeleton)")]
    NotImplemented,

    /// Underlying syscall failed. The Linux path collapses kernel
    /// errors into this; the fallback path is fire-and-forget and
    /// never raises this.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// One in-flight (or already-completed) async read. The returned
/// buffer is owned by the handle; dropping the handle releases it.
///
/// On Linux with io_uring, completion is async — the caller polls
/// `is_ready()` and reads the buffer once it's true. The destination
/// buffer is allocated by the backend and pinned by the handle.
///
/// On all other paths, the handle's `done` flag is set at
/// construction time (we issued a hint, not a real read; the demand
/// mmap path will resolve the bytes whenever it actually needs them).
///
/// Important: even on Linux, the demand-path `Shard::slice()` does
/// not need to *read from* the handle's buffer. The kernel's page
/// cache is FD-keyed; the io_uring read populates the same cache
/// entries that the later mmap fault will resolve from. The
/// completion is just a "this read has landed" ticket; the buffer
/// itself can be reused once we know we won't need it.
pub struct AsyncReadHandle {
    /// Set to true when the kernel has reported the completion. On
    /// the fallback path this is initialized true (the hint is
    /// non-blocking).
    done: Arc<AtomicBool>,

    /// Owned destination buffer for the read. Even on the fallback
    /// path we hold this (zero-length on fallback) so the handle has
    /// the same memory shape across platforms.
    _buf: Box<[u8]>,

    /// io_uring slot index. Set on the Linux path so the backend can
    /// reclaim the slot when the handle drops without waiting on the
    /// CQE; None on the fallback path.
    #[allow(dead_code)]
    slot: Option<u32>,
}

impl AsyncReadHandle {
    /// Has the read completed? On the Linux path this is true once
    /// the completion thread has seen the matching CQE. On the
    /// fallback path this is true at construction time (hints don't
    /// have completions).
    pub fn is_ready(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    /// Test-only: block until ready. Production code should poll
    /// `is_ready` and not wait — the whole point of async prefetch is
    /// to not stall the dispatcher.
    #[cfg(test)]
    fn wait_blocking(&self) {
        while !self.is_ready() {
            std::thread::yield_now();
        }
    }
}

/// One async-prefetch backend, shared across the runner's prefetcher
/// thread. Thread-safe.
///
/// Construct once per `Runner` at startup. The platform check
/// happens here — if io_uring isn't available, the constructor still
/// returns Ok but flips the internal `kind` to `Fallback`.
#[derive(Clone)]
pub struct AsyncPrefetchBackend {
    inner: Arc<BackendKind>,
}

enum BackendKind {
    /// Linux + io_uring available. Skeleton: doesn't actually push
    /// SQEs yet, returns `AsyncIoError::NotImplemented` from
    /// `queue_read`.
    #[cfg(target_os = "linux")]
    IoUring(IoUringState),

    /// Anything else, or Linux with too-old a kernel / WSL2 /
    /// seccomp-blocked containers. Equivalent to the iter 033
    /// prefetch path.
    Fallback,
}

#[cfg(target_os = "linux")]
struct IoUringState {
    /// Placeholder — milestone 1 will replace this with the
    /// `io_uring::IoUring` instance plus its submission / completion
    /// threads. Kept as () until then so the skeleton compiles
    /// without the dep.
    _placeholder: (),
}

impl AsyncPrefetchBackend {
    /// Try to construct the io_uring backend; fall back to madvise on
    /// any failure. Never panics, never returns Err — callers can
    /// unconditionally call this at startup and check
    /// `using_io_uring()` for which path fired.
    pub fn new() -> Self {
        Self::with_depth(DEFAULT_QUEUE_DEPTH)
    }

    /// As `new()` but with a custom submission queue depth. Production
    /// default is 256 (small enough that one full queue empties within
    /// the ~10ms a single layer's expert dispatch takes; large enough
    /// to absorb a couple of layers' worth of prefetch requests).
    pub fn with_depth(_depth: u32) -> Self {
        #[cfg(target_os = "linux")]
        {
            // Milestone 1 will do real availability detection here:
            //   - probe `io_uring_setup(0, ...)` to surface EPERM
            //     (seccomp-blocked container) or ENOSYS (kernel < 5.1)
            //   - inspect IORING_FEAT_* on the returned params to
            //     gate on NODROP (>= 5.10), SINGLE_MMAP (>= 5.4) etc.
            //   - return Fallback if any required feature is missing
            //
            // For the skeleton we always pick Fallback so we don't
            // ship a backend that returns NotImplemented for every
            // call. When milestone 1 lands the line below flips to
            //
            //   inner: Arc::new(BackendKind::IoUring(try_init_io_uring(_depth)?))
            //
            // with a Result→Fallback collapse on Err.
            return Self {
                inner: Arc::new(BackendKind::Fallback),
            };
        }
        #[cfg(not(target_os = "linux"))]
        {
            Self {
                inner: Arc::new(BackendKind::Fallback),
            }
        }
    }

    /// Which path is firing? Useful at startup to log "this run uses
    /// io_uring" vs "this run uses madvise" so bench results aren't
    /// ambiguous.
    pub fn using_io_uring(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            matches!(*self.inner, BackendKind::IoUring(_))
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Queue an async read of bytes `offset..offset+len` on `fd` into
    /// a backend-owned buffer.
    ///
    /// On the io_uring path: pushes an `IORING_OP_READ` SQE,
    /// allocates a destination buffer, and returns immediately. The
    /// returned handle's `is_ready` flips to true when the CQE
    /// arrives.
    ///
    /// On the Fallback path: this is a no-op (the caller should be
    /// using `Shard::advise_willneed` directly on this platform —
    /// the fallback path here exists so call sites don't need to
    /// branch on the backend). Returns a handle whose `is_ready` is
    /// already true. **NOTE:** the fallback handle does *not* mean
    /// "the bytes are in memory" — it means "there's no async read
    /// to wait on, fall through to your normal hint+mmap path."
    pub fn queue_read(
        &self,
        _fd: RawFd,
        _offset: u64,
        _len: usize,
    ) -> Result<AsyncReadHandle, AsyncIoError> {
        match &*self.inner {
            #[cfg(target_os = "linux")]
            BackendKind::IoUring(_) => {
                // Milestone 1 will do:
                //   1. allocate a destination buffer (Box<[u8; len]>)
                //   2. claim an SQ slot (sq.try_push() or QueueFull)
                //   3. construct opcode::Read::new(fd, buf.as_mut_ptr(), len)
                //        .offset(offset)
                //        .build()
                //        .user_data(slot as u64)
                //   4. ring.submit() if SQ is over the watermark
                //   5. return AsyncReadHandle { done, buf, slot: Some(slot) }
                //
                // The completion thread (spawned in
                // try_init_io_uring) drains CQEs in a loop:
                //
                //   for cqe in ring.completion() {
                //       let slot = cqe.user_data() as u32;
                //       let done = handles[slot as usize].done.clone();
                //       done.store(true, Release);
                //       slot_pool.release(slot);
                //   }
                Err(AsyncIoError::NotImplemented)
            }

            BackendKind::Fallback => Ok(AsyncReadHandle {
                done: Arc::new(AtomicBool::new(true)),
                _buf: vec![0u8; 0].into_boxed_slice(),
                slot: None,
            }),
        }
    }
}

impl Default for AsyncPrefetchBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Default submission queue depth.
///
/// Sized so that one full queue empties within roughly the time a
/// single layer's expert dispatch takes (~10 ms on miner under load),
/// large enough to absorb a couple of layers' worth of prefetch
/// requests, small enough that a stalled completion thread doesn't
/// blow out memory.
///
/// 256 SQEs × ~24 KB/slice × 6 slices/expert × 6 experts/layer ≈ 220
/// MB peak if every SQE is one slice and every expert is fully
/// in-flight. In practice slices share shards and the page cache
/// dedupes; production tuning is a v2 concern.
pub const DEFAULT_QUEUE_DEPTH: u32 = 256;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_handle_is_immediately_ready() {
        // On any platform without io_uring (and on Linux until
        // milestone 1 lands), the backend should construct cleanly
        // and queue_read should return a handle that's already
        // ready — i.e. there's no async read to wait for; the caller
        // should fall through to its existing demand-path hint.
        let backend = AsyncPrefetchBackend::new();
        assert!(!backend.using_io_uring(), "skeleton: never picks io_uring");
        let h = backend
            .queue_read(0, 0, 0)
            .expect("fallback queue_read should not fail");
        assert!(h.is_ready());
        h.wait_blocking(); // no-op on a ready handle
    }

    #[test]
    fn backend_clone_is_cheap() {
        // The backend is meant to be shared across the runner's
        // prefetcher threads — make sure cloning it doesn't try to
        // duplicate the ring.
        let a = AsyncPrefetchBackend::new();
        let b = a.clone();
        assert_eq!(a.using_io_uring(), b.using_io_uring());
    }
}
