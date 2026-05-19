//! Async expert prefetch via `io_uring` (Linux) with graceful
//! fallback to `madvise(MADV_WILLNEED)` (other Unix) /
//! `PrefetchVirtualMemory` (Windows) / no-op (everywhere else).
//!
//! Status: **milestone 1** (autolab iter 097). The Linux path is now
//! real: `AsyncPrefetchBackend::with_depth` constructs an
//! `io_uring::IoUring`, probes it with a NOP SQE / CQE round-trip
//! (catches WSL2 / Docker seccomp / kernel < 5.1), spawns a
//! completion-reaper thread, and serves `queue_read` via
//! `IORING_OP_READ`. On any probe / construction failure the backend
//! emits a `warn!` and falls through to the same Fallback path used
//! on every non-Linux platform.
//!
//! See `docs/perf/io_uring_prefetch.md` for:
//!  - the design (why io_uring over aio / threadpool / O_DIRECT)
//!  - the kernel-version + WSL2 + container blockers
//!  - the per-PR plan that gets this from skeleton to production
//!
//! API contract:
//!
//!   let backend = AsyncPrefetchBackend::new();
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
//! read is still inflight when dropped, the backing buffer stays
//! allocated until the reaper thread sees the CQE and releases the
//! slot.

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

    /// Reserved for future milestones (kept for ABI stability —
    /// removing it would break downstream `match` arms in
    /// `tahoma-engine-sparse-moe`). Milestone 1 never returns it.
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
    ///
    /// On the io_uring path this buffer is the destination the kernel
    /// writes into. The reaper thread holds a parallel `Arc` to the
    /// same buffer via the slot table, so dropping the handle while
    /// the read is inflight doesn't unmap the kernel's destination.
    #[allow(dead_code)]
    buf: Arc<UnsafeBuf>,

    /// io_uring slot index. Set on the Linux path so the reaper can
    /// release the slot when the CQE lands; None on the fallback
    /// path.
    #[allow(dead_code)]
    slot: Option<u32>,
}

/// Wrapper around a `Vec<u8>` whose pointer is handed to the kernel
/// across thread boundaries. We treat it as opaque from Rust's POV
/// (no concurrent reads) — the reaper thread owns the right to flip
/// `done` once the kernel has finished writing. The Arc<>+Drop
/// machinery handles the lifetime; the `unsafe impl Send + Sync` is
/// the explicit acknowledgement that we're shipping a raw buffer
/// across threads. The `Vec<u8>` field looks dead to the compiler
/// (we never read through `&self.0`), but its allocation is
/// load-bearing — the kernel writes into the pointer we captured
/// from `as_mut_ptr()` before wrapping; dropping the Vec would
/// free the destination out from under an inflight SQE.
struct UnsafeBuf(#[allow(dead_code)] Vec<u8>);

// Safety: we never hand out `&UnsafeBuf` across threads — the buffer
// is written by the kernel via the raw pointer we passed in the SQE,
// and read by nobody (the handle's only API is `is_ready`). The
// reaper thread only reads `done`, not the buffer's contents. So
// shipping the Arc across thread boundaries is sound; the &mut [u8]
// access pattern is single-threaded (one SQE per slot at a time).
unsafe impl Send for UnsafeBuf {}
unsafe impl Sync for UnsafeBuf {}

impl AsyncReadHandle {
    /// Has the read completed? On the Linux path this is true once
    /// the reaper thread has seen the matching CQE. On the
    /// fallback path this is true at construction time (hints don't
    /// have completions).
    pub fn is_ready(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    /// Test-only: block until ready. Production code should poll
    /// `is_ready` and not wait — the whole point of async prefetch is
    /// to not stall the dispatcher.
    ///
    /// Gated to non-Linux because only the fallback-path test on
    /// non-Linux dev machines calls it; the Linux test in this module
    /// has its own bounded-spin wait with a 5s panic cap.
    #[cfg(all(test, not(target_os = "linux")))]
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
    /// Linux + io_uring available + functional NOP probe.
    #[cfg(target_os = "linux")]
    IoUring(linux::IoUringState),

    /// Anything else, or Linux with too-old a kernel / WSL2 /
    /// seccomp-blocked containers. Equivalent to the iter 033
    /// prefetch path.
    Fallback,
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
    pub fn with_depth(depth: u32) -> Self {
        #[cfg(target_os = "linux")]
        {
            // io_uring depth must be a power of two — round up
            // silently rather than reject the call.
            let depth = depth.next_power_of_two().max(8);
            match linux::IoUringState::try_init(depth) {
                Ok(state) => Self {
                    inner: Arc::new(BackendKind::IoUring(state)),
                },
                Err(e) => {
                    tracing::warn!(
                        err = %e,
                        "io_uring init failed, falling back to madvise/PrefetchVirtualMemory \
                         (kernel<5.1, WSL2, or seccomp-blocked container?)"
                    );
                    Self {
                        inner: Arc::new(BackendKind::Fallback),
                    }
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = depth;
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
        fd: RawFd,
        offset: u64,
        len: usize,
    ) -> Result<AsyncReadHandle, AsyncIoError> {
        match &*self.inner {
            #[cfg(target_os = "linux")]
            BackendKind::IoUring(state) => state.queue_read(fd, offset, len),

            BackendKind::Fallback => {
                let _ = (fd, offset, len);
                Ok(AsyncReadHandle {
                    done: Arc::new(AtomicBool::new(true)),
                    buf: Arc::new(UnsafeBuf(Vec::new())),
                    slot: None,
                })
            }
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

// ----------------------------------------------------------------------------
// Linux io_uring backend (milestone 1).
// ----------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod linux {
    //! Real io_uring backend: `IoUring` + reaper thread + slot pool.
    //!
    //! Lifecycle:
    //!
    //!   * `try_init(depth)` — constructs `IoUring::new(depth)`,
    //!     probes with a NOP, spawns the reaper. Returns `Err` on any
    //!     failure (caller collapses to Fallback).
    //!   * `queue_read(fd, off, len)` — allocates a destination
    //!     buffer, claims a slot, pushes an `OP_READ` SQE, returns
    //!     the handle.
    //!   * `Drop` — pushes a NOP with the SHUTDOWN sentinel user_data,
    //!     joins the reaper thread.
    //!
    //! Thread safety: the `IoUring` instance is `Send + Sync`. The
    //! `SubmissionQueue` half is `!Send + !Sync` per upstream's
    //! design, so we serialize SQ pushes through a `Mutex<()>` and
    //! re-borrow `submission_shared()` for each push. The reaper
    //! thread owns the CQ via `completion_shared()` and blocks on
    //! `submit_and_wait(1)`.
    //!
    //! Slot pool: a fixed-size `Vec<SlotEntry>` indexed by user_data.
    //! Each entry has an `in_use` flag (claimed by `queue_read`,
    //! cleared by the reaper) plus an `Arc<AtomicBool>` for the
    //! handle's `done` flag and an `Arc<UnsafeBuf>` keeping the
    //! buffer alive across the kernel-write window.

    use std::io;
    use std::os::fd::RawFd;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    use io_uring::{opcode, types, IoUring};

    use super::{AsyncIoError, AsyncReadHandle, UnsafeBuf};

    /// Sentinel user_data for the shutdown NOP pushed by Drop. Picked
    /// at the top of the u64 range so it can't collide with a real
    /// slot index (slot pools never approach 2^63).
    const SHUTDOWN_TOKEN: u64 = u64::MAX;

    /// Reaper poll interval when no CQEs are pending. Short enough
    /// that Drop joins promptly; long enough that an idle backend
    /// doesn't burn a core.
    const REAPER_IDLE: Duration = Duration::from_millis(10);

    pub struct IoUringState {
        ring: Arc<IoUring>,
        /// Serializes pushes on the SQ side. `SubmissionQueue` is
        /// `!Send + !Sync`, so we re-borrow it inside the lock on
        /// each push.
        sq_lock: Arc<Mutex<()>>,
        /// Slot table — indexed by the u64 user_data we attach to each
        /// SQE. Fixed size = queue depth.
        slots: Arc<Vec<SlotEntry>>,
        /// Round-robin allocator hint for `claim_slot` (the actual
        /// claim is a CAS on `in_use`; the hint just avoids always
        /// retrying from index 0).
        next_slot_hint: Arc<AtomicU32>,
        /// Signaled by Drop to break the reaper out of its
        /// submit_and_wait loop after the shutdown SQE lands.
        shutting_down: Arc<AtomicBool>,
        /// Reaper thread handle. Joined in Drop.
        reaper: Option<JoinHandle<()>>,
    }

    struct SlotEntry {
        in_use: AtomicBool,
        /// Per-slot state, replaced on each claim. `Mutex` so the
        /// reaper can swap in `None` after the CQE lands without
        /// racing the next `claim_slot`.
        active: Mutex<Option<ActiveSlot>>,
    }

    struct ActiveSlot {
        done: Arc<AtomicBool>,
        /// Kept alive while the kernel writes. Released when the
        /// reaper sees the CQE.
        _buf: Arc<UnsafeBuf>,
    }

    impl IoUringState {
        pub fn try_init(depth: u32) -> Result<Self, io::Error> {
            // 1. Construct the ring. Fails with ENOSYS on kernels
            //    that don't have io_uring_setup at all (pre-5.1).
            let ring = IoUring::new(depth).map_err(|e| {
                io::Error::new(e.kind(), format!("IoUring::new({depth}) failed: {e}"))
            })?;

            // 2. Functional probe — submit a NOP, wait for the CQE.
            //    Catches WSL2 (where io_uring_setup succeeds but
            //    submission is blocked) and seccomp-restricted
            //    containers (where setup succeeds but io_uring_enter
            //    is filtered). Cheaper than waiting until first read.
            probe_nop(&ring)?;

            // 3. Build slot pool, share the ring, spawn reaper.
            let slots: Vec<SlotEntry> = (0..depth)
                .map(|_| SlotEntry {
                    in_use: AtomicBool::new(false),
                    active: Mutex::new(None),
                })
                .collect();
            let slots = Arc::new(slots);
            let ring = Arc::new(ring);
            let sq_lock = Arc::new(Mutex::new(()));
            let shutting_down = Arc::new(AtomicBool::new(false));

            let reaper = spawn_reaper(ring.clone(), slots.clone(), shutting_down.clone())?;

            tracing::info!(
                depth,
                "io_uring backend initialized (NOP probe ok, reaper thread spawned)"
            );

            Ok(Self {
                ring,
                sq_lock,
                slots,
                next_slot_hint: Arc::new(AtomicU32::new(0)),
                shutting_down,
                reaper: Some(reaper),
            })
        }

        pub fn queue_read(
            &self,
            fd: RawFd,
            offset: u64,
            len: usize,
        ) -> Result<AsyncReadHandle, AsyncIoError> {
            // Zero-length reads are a no-op — there's nothing for
            // io_uring to do, so return a ready handle without
            // touching the ring.
            if len == 0 {
                return Ok(AsyncReadHandle {
                    done: Arc::new(AtomicBool::new(true)),
                    buf: Arc::new(UnsafeBuf(Vec::new())),
                    slot: None,
                });
            }

            // Allocate the destination buffer. The kernel will write
            // bytes here via the raw pointer in the SQE; we keep an
            // Arc to it both in the handle and the slot table so the
            // earliest-dropped party doesn't yank it out from under
            // the kernel.
            let mut raw = vec![0u8; len];
            let ptr = raw.as_mut_ptr();
            let buf = Arc::new(UnsafeBuf(raw));

            let slot_idx = self.claim_slot().ok_or(AsyncIoError::QueueFull)?;
            let done = Arc::new(AtomicBool::new(false));

            // Stash the active state so the reaper can flip `done`
            // and drop the buffer when the CQE lands.
            {
                let mut active = self.slots[slot_idx as usize]
                    .active
                    .lock()
                    .expect("slot mutex poisoned");
                *active = Some(ActiveSlot {
                    done: done.clone(),
                    _buf: buf.clone(),
                });
            }

            // Build the SQE. `opcode::Read` with a single contiguous
            // destination is the right primitive for our use case —
            // each tensor slice is one mmap range, so iovec /
            // Readv would just add an indirection.
            let sqe = opcode::Read::new(types::Fd(fd), ptr, len as u32)
                .offset(offset)
                .build()
                .user_data(slot_idx as u64);

            let push_res = {
                let _guard = self.sq_lock.lock().expect("sq lock poisoned");
                // Safety: we hold sq_lock for the duration of this
                // borrow, so no other thread is touching
                // submission_shared. The SQE references `ptr`, which
                // is kept alive by `buf` in the slot table until the
                // reaper clears the slot.
                unsafe {
                    let mut sq = self.ring.submission_shared();
                    sq.sync();
                    sq.push(&sqe)
                }
            };

            if push_res.is_err() {
                // SQ full — release the slot and tell the caller to
                // back off. Caller treats this as a non-fatal drop
                // (demand path resolves the bytes).
                self.release_slot(slot_idx);
                return Err(AsyncIoError::QueueFull);
            }

            // Kick the kernel. `submit()` is non-blocking — it
            // returns as soon as the ring is enqueued. The reaper
            // does the actual CQE wait.
            self.ring.submit().map_err(|e| {
                // Submit failed after push succeeded — slot is
                // already in the kernel's queue, but it's safer to
                // surface the error than pretend the read worked.
                AsyncIoError::Io(io::Error::new(e.kind(), format!("io_uring submit: {e}")))
            })?;

            Ok(AsyncReadHandle {
                done,
                buf,
                slot: Some(slot_idx),
            })
        }

        /// Find a free slot via CAS scan. Returns None if every slot
        /// is in flight.
        fn claim_slot(&self) -> Option<u32> {
            let n = self.slots.len() as u32;
            let start = self.next_slot_hint.load(Ordering::Relaxed) % n;
            for offset in 0..n {
                let i = (start + offset) % n;
                let slot = &self.slots[i as usize];
                if slot
                    .in_use
                    .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    self.next_slot_hint.store((i + 1) % n, Ordering::Relaxed);
                    return Some(i);
                }
            }
            None
        }

        fn release_slot(&self, idx: u32) {
            let slot = &self.slots[idx as usize];
            *slot.active.lock().expect("slot mutex poisoned") = None;
            slot.in_use.store(false, Ordering::Release);
        }
    }

    impl Drop for IoUringState {
        fn drop(&mut self) {
            // Signal shutdown, push a NOP so the reaper wakes from
            // its submit_and_wait, then join.
            self.shutting_down.store(true, Ordering::Release);

            // Best-effort shutdown SQE. If push fails (ring genuinely
            // wedged), the reaper's idle poll will still eventually
            // see shutting_down=true and exit.
            let sqe = opcode::Nop::new().build().user_data(SHUTDOWN_TOKEN);
            {
                if let Ok(_guard) = self.sq_lock.lock() {
                    unsafe {
                        let mut sq = self.ring.submission_shared();
                        sq.sync();
                        let _ = sq.push(&sqe);
                    }
                }
                let _ = self.ring.submit();
            }

            if let Some(j) = self.reaper.take() {
                let _ = j.join();
            }
        }
    }

    fn probe_nop(ring: &IoUring) -> Result<(), io::Error> {
        const PROBE_TOKEN: u64 = 0xDEAD_BEEF_CAFE_BABE;

        let sqe = opcode::Nop::new().build().user_data(PROBE_TOKEN);
        // Safety: we have exclusive access — this is called from
        // try_init before the ring is shared with any other thread.
        unsafe {
            let mut sq = ring.submission_shared();
            sq.push(&sqe)
                .map_err(|e| io::Error::other(format!("NOP probe SQE push failed: {e:?}")))?;
        }
        ring.submit_and_wait(1).map_err(|e| {
            io::Error::new(e.kind(), format!("NOP probe submit_and_wait failed: {e}"))
        })?;

        // Drain the CQE and verify the token + a non-negative result.
        // Safety: again exclusive — only this thread touches the ring
        // during init.
        let mut cq = unsafe { ring.completion_shared() };
        match cq.next() {
            Some(cqe) => {
                if cqe.user_data() != PROBE_TOKEN {
                    return Err(io::Error::other(format!(
                        "NOP probe CQE user_data mismatch: got {:#x} want {:#x}",
                        cqe.user_data(),
                        PROBE_TOKEN
                    )));
                }
                if cqe.result() < 0 {
                    return Err(io::Error::from_raw_os_error(-cqe.result()));
                }
                Ok(())
            }
            None => Err(io::Error::other(
                "NOP probe: no CQE returned after submit_and_wait(1)",
            )),
        }
    }

    fn spawn_reaper(
        ring: Arc<IoUring>,
        slots: Arc<Vec<SlotEntry>>,
        shutting_down: Arc<AtomicBool>,
    ) -> Result<JoinHandle<()>, io::Error> {
        thread::Builder::new()
            .name("io-uring-reaper".into())
            .spawn(move || reaper_loop(ring, slots, shutting_down))
    }

    fn reaper_loop(ring: Arc<IoUring>, slots: Arc<Vec<SlotEntry>>, shutting_down: Arc<AtomicBool>) {
        loop {
            // submit_and_wait(0) returns immediately if the CQ is
            // non-empty, otherwise blocks until at least one CQE
            // lands or the call is interrupted. We loop with a short
            // timeout-via-poll pattern instead of submit_and_wait(1)
            // so shutdown can break us out even if no real CQEs
            // arrive.
            //
            // The actual submission of pending SQEs is done by
            // queue_read's `ring.submit()` call — we don't need to
            // re-submit here.
            //
            // Drain whatever CQEs are pending.
            let mut drained = 0usize;
            // Safety: we are the sole CQ consumer thread.
            let mut cq = unsafe { ring.completion_shared() };
            cq.sync();
            for cqe in &mut cq {
                drained += 1;
                let user_data = cqe.user_data();
                if user_data == SHUTDOWN_TOKEN {
                    // Sentinel — drained alongside any final reads.
                    continue;
                }
                let idx = user_data as usize;
                if idx >= slots.len() {
                    // Bogus token (shouldn't happen) — log and
                    // continue.
                    tracing::warn!(
                        user_data,
                        "io-uring-reaper: CQE with out-of-range user_data, ignoring"
                    );
                    continue;
                }
                let slot = &slots[idx];
                let active = slot.active.lock().expect("slot mutex poisoned").take();
                if let Some(active) = active {
                    // result() < 0 is errno — we still flip done
                    // (the caller's contract is "ticket that the
                    // read attempt has finished"; demand path
                    // resolves the bytes regardless of whether the
                    // prefetch landed).
                    if cqe.result() < 0 {
                        tracing::debug!(
                            errno = -cqe.result(),
                            slot = idx,
                            "io-uring-reaper: prefetch SQE returned errno; falling back to demand path"
                        );
                    }
                    active.done.store(true, Ordering::Release);
                }
                slot.in_use.store(false, Ordering::Release);
            }

            if shutting_down.load(Ordering::Acquire) && drained == 0 {
                // Shutdown signaled and CQ is empty — exit.
                break;
            }

            if drained == 0 {
                // Nothing to do — short sleep. We deliberately don't
                // use submit_and_wait(1) here because the SQ side is
                // owned by queue_read; calling submit_and_wait from
                // two threads is allowed but the io-uring 0.6 API
                // makes it awkward to do safely with the shared SQ
                // borrow pattern.
                thread::sleep(REAPER_IDLE);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_handle_is_immediately_ready_on_non_linux() {
        // On non-Linux platforms the backend always picks Fallback;
        // queue_read should return an already-ready handle without
        // touching any FD.
        #[cfg(not(target_os = "linux"))]
        {
            let backend = AsyncPrefetchBackend::new();
            assert!(!backend.using_io_uring());
            let h = backend
                .queue_read(0, 0, 0)
                .expect("fallback queue_read should not fail");
            assert!(h.is_ready());
            h.wait_blocking();
        }
    }

    #[test]
    fn backend_clone_is_cheap_and_consistent() {
        // The backend is meant to be shared across the runner's
        // prefetcher threads — make sure cloning it doesn't try to
        // duplicate the ring, and both clones agree about which
        // path is firing.
        let a = AsyncPrefetchBackend::new();
        let b = a.clone();
        assert_eq!(a.using_io_uring(), b.using_io_uring());
    }

    #[test]
    fn backend_constructs_without_panic_on_any_platform() {
        // The whole point of the graceful-fallback contract is that
        // callers can unconditionally construct this at startup
        // without worrying about platform / kernel / container
        // capability. Verify it on whatever platform is running CI.
        let _ = AsyncPrefetchBackend::new();
        let _ = AsyncPrefetchBackend::with_depth(8);
        let _ = AsyncPrefetchBackend::with_depth(16);
    }

    /// Linux-specific end-to-end test: construct the backend and
    /// verify it either lit up io_uring OR fell through cleanly. On
    /// the io_uring path, actually read a few bytes from a tempfile
    /// and confirm the handle goes ready.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_io_uring_or_clean_fallback() {
        use std::io::Write;
        use std::os::fd::AsRawFd;

        let backend = AsyncPrefetchBackend::with_depth(8);

        if !backend.using_io_uring() {
            // Acceptable outcome on dev machines (WSL2, old kernels,
            // seccomp-blocked container CI). The constructor should
            // have logged a warn; we verified above that it didn't
            // panic.
            return;
        }

        // io_uring is up — do a real read.
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let payload = b"the quick brown fox jumps over the lazy dog";
        tmp.write_all(payload).expect("write tempfile");
        tmp.flush().expect("flush tempfile");
        let fd = tmp.as_file().as_raw_fd();

        let h = backend
            .queue_read(fd, 0, payload.len())
            .expect("io_uring queue_read should succeed on a healthy ring");

        // Block (with a sanity cap) until the reaper flips done.
        let start = std::time::Instant::now();
        while !h.is_ready() {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(5),
                "io_uring read did not complete within 5s"
            );
            std::thread::yield_now();
        }
    }
}
