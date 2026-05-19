# io_uring async expert prefetch (iter 074)

Design doc + skeleton for replacing the `madvise(MADV_WILLNEED)` hint
chain used by the iter 033 C1 expert prefetcher with **true async I/O**
on Linux via `io_uring`.

This is the scoping deliverable for autolab iter 074. Status: **plan +
skeleton, no production code yet.** The implementation lift is
estimated at 1-2 engineer-weeks; this doc captures the path so it can
be picked up cold.

---

## TL;DR

- `madvise(MADV_WILLNEED)` is a **hint**. The kernel may or may not act
  on it; the actual read happens at page-fault time, fed by the same
  per-process / per-file readahead window as the demand path.
- Iter 070 measured this: a 7-feature cache-attack stack with ~1700
  `madvise` calls/token regressed -32% vs the K=6 baseline because
  prefetch and demand reads contend for the same readahead queue.
- `io_uring` lets us issue **explicit** `IORING_OP_READ` SQEs into a
  separate ring. The prefetcher owns the ring; demand reads stay on
  the mmap page-fault path. The two queues don't compete for the same
  Linux readahead window — only for the underlying NVMe bandwidth.
- Predicted speedup: **~2x on highly-contended NVMe** (separating the
  prefetch queue from the demand queue eliminates the iter 070
  pathological case). On uncontended NVMe + warm cache, expected to
  be roughly even with iter 033 (we don't beat already-warm pages, we
  just don't fight them).

## What we're replacing

Current C1 chain (iter 033 + iter 038 Windows port):

```text
Prefetcher thread ──recv()─► PrefetchReq{lid, eid}
                              │
                              ├─► Shard::advise_willneed(name)
                              │     #[cfg(unix)] mmap.advise_range(WillNeed, off, len)  ──► madvise(2)
                              │     #[cfg(windows)] PrefetchVirtualMemory(...)
                              │
                              └─► (no completion, no signal — fire-and-forget hint)
```

Properties:

| Trait                      | madvise(WILLNEED)                                              |
| -------------------------- | -------------------------------------------------------------- |
| Syscall                    | One per slice (6 per expert × 60 layers × N_predicted)        |
| Completion signal          | None — pure hint                                               |
| Bypasses readahead window  | No — feeds the same per-FD readahead queue as demand reads     |
| Bypasses page cache        | No — populates page cache same as a `read()`                   |
| Backpressure               | None — kernel queues whatever the inflight prefetcher submits |

The "no backpressure / shared readahead window" combination is what
iter 070 hit: 7 features × multi-layer × top-N + the speculative
chain submits more readahead requests than the NVMe can absorb, and
the kernel serves them strictly-FIFO with the demand-path reads
in line behind.

## What we're building

```text
Prefetcher thread ──recv()─► PrefetchReq{lid, eid}
                              │
                              ├─► Shard::async_read(off, len) ──► io_uring_enter (SQE: OP_READ → ring buffer)
                              │                                            │
                              │                                            ▼
                              │                                   completion thread polls CQE
                              │                                   marks PrefetchSlot as ready
                              │
Demand path ──► Shard::demand_slice(name) ──► first checks PrefetchSlot::is_ready()
                              │                       ├─ ready: read from io_uring buffer (already warm)
                              │                       └─ not ready: fall through to mmap[..] (page-faults like today)
                              │
                              ▼
                       (kernel page cache resolves overlap — the io_uring read
                        populated the page cache, so the mmap fault hits a warm page)
```

Key properties:

| Trait                      | io_uring async read                                              |
| -------------------------- | ---------------------------------------------------------------- |
| Syscall                    | Batched — many SQEs per `io_uring_enter`                         |
| Completion signal          | Yes — CQE per SQE, polled by background thread                   |
| Bypasses readahead window  | Effectively yes — explicit read, not advisory                    |
| Bypasses page cache        | Optional (`O_DIRECT` mode) — default is page-cache-backed        |
| Backpressure               | Built-in — bounded SQ depth, `try_push` fails when ring is full  |

The composition pattern (the critical bit): **prefetched bytes go into
a separate buffer (allocated per slot, owned by the prefetcher); demand
reads stay on the mmap path.** This works because:

1. Linux's page cache is FD-keyed, not vma-keyed. An `io_uring` read on
   the safetensors fd populates the same page-cache entries that a
   later page-fault through the mmap will resolve from. So the
   bytes-on-disk → bytes-in-memory traffic happens once.
2. We don't need to redirect the demand-path GEMM to read from the
   prefetch buffer — when the prefetch lands, the mmap path is already
   served from cache. The prefetch buffer is essentially a "ticket"
   that the read has happened; it can be reused (ring slot) or freed
   immediately after the corresponding CQE.

(Alt design: bypass the mmap on the demand path too, read from the
prefetch buffer directly. Considered and rejected — it would invert
the existing layout and force a copy on the demand path. The
page-cache overlap pattern is cheaper and matches how `tahoma-int4-gemm`
already reads.)

## Why io_uring, not aio / threadpool / O_DIRECT alone

- **aio (libaio):** historically broken for buffered I/O on Linux,
  falls back to sync on the common case. Largely deprecated.
- **Thread pool of pread():** works, but every read costs a thread +
  context switch. The point of the prefetcher is to keep CPU on
  inference; spinning up dozens of pread threads burns the same
  CPU budget we're trying to free up for GEMV.
- **O_DIRECT + sync read:** valid, but mutually exclusive with mmap
  on the same FD (`O_DIRECT` requires aligned reads and bypasses the
  page cache — the demand-mmap path would then page-fault to disk
  again, double-reading). Tracked as a follow-up.
- **io_uring + buffered read:** what this iter ships. Single syscall
  batches dozens of SQEs, completions come back on a separate ring,
  page cache is shared with mmap.

## API skeleton

Target API in `tahoma-int4-gemm`:

```rust
// New: a handle representing one in-flight async read. The caller may
// poll it for completion (`is_ready`), wait on it (`wait`), or drop
// it (cancels the read on a best-effort basis).
pub struct AsyncReadHandle {
    /// Slot index inside the ring. None on non-Linux (read completed
    /// synchronously via the fallback path).
    #[cfg(target_os = "linux")]
    slot: Option<u32>,
    /// Shared completion flag flipped by the completion thread when the
    /// CQE for this SQE arrives.
    done: Arc<AtomicBool>,
    /// Owned buffer the read drained into. Released back to the ring
    /// pool when this handle is dropped.
    _buf: Box<[u8]>,
}

impl AsyncReadHandle {
    /// Has the kernel returned a CQE for this read?
    pub fn is_ready(&self) -> bool { /* ... */ }

    /// Block (parking lot CondVar) until ready. Test-only — the
    /// production prefetcher polls is_ready and otherwise does nothing.
    pub fn wait(&self) { /* ... */ }
}

impl Shard {
    /// Queue an async read of bytes `off..off+len` into a ring-owned
    /// buffer. Returns a handle the caller can poll.
    ///
    /// Linux: pushes an `IORING_OP_READ` SQE; returns immediately.
    /// Non-Linux: degrades to the existing `madvise/PrefetchVirtualMemory`
    /// hint and returns a handle that is immediately ready.
    pub fn async_read(&self, off: usize, len: usize) -> Result<AsyncReadHandle, AsyncIoError>;
}
```

Target API in `tahoma-engine-sparse-moe` (replacing the iter 033
prefetcher, which doesn't exist on `main` yet — we land both the new
prefetcher *and* its async backend in the same series of PRs):

```rust
pub struct AsyncPrefetcher {
    tx: SyncSender<PrefetchReq>,
    join: JoinHandle<()>,
    /// Inflight handles, keyed by (lid, eid). Polled by the
    /// dispatcher to short-circuit if the read has already landed.
    inflight: Arc<DashMap<(u32, u32), Vec<AsyncReadHandle>>>,
    drops: Arc<AtomicU64>,
    submits: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
}

impl AsyncPrefetcher {
    /// Linux: spawns the io_uring submission + completion threads.
    /// Non-Linux: spawns the iter 033-style madvise/PrefetchVirtualMemory
    /// thread; `inflight` is empty (the demand path falls through to
    /// mmap unconditionally).
    pub fn spawn(source: Arc<SafetensorsExpertSource>, depth: u32) -> Self;
    pub fn try_submit(&self, lid: u32, eid: u32);
    pub fn snapshot(&self) -> PrefetchStats;
}
```

## Per-PR plan (6 milestones, 1-2 weeks)

| # | Title                                                | Touched crates                                  |
|---|------------------------------------------------------|-------------------------------------------------|
| 1 | `async_prefetch` module skeleton + Linux backend     | `tahoma-int4-gemm`                              |
| 2 | `Shard::async_read` API + fallback path              | `tahoma-int4-gemm`                              |
| 3 | `AsyncPrefetcher` wrapping iter 033's `Prefetcher`   | `tahoma-engine-sparse-moe`                      |
| 4 | Wire into `forward_shells` (cfg `TAHOMA_PREFETCH_BACKEND=iouring`)| `tahoma-engine-sparse-moe`           |
| 5 | Instrumentation + counters (sqe_submitted, cqe_seen, queue_depth, blocked_on_inflight) | both    |
| 6 | Bench harness + measured speedup vs iter 033         | autolab `074_iouring_bench/`                    |

PR #74 shipped the **skeleton** for milestones 1+2+3 (everything
constructs cleanly but the Linux path returns `NotImplemented` / the
backend always picks `Fallback`). **Iter 097 (`perf/io-uring-milestone1-097`)
flips milestone 1 from skeleton to real:** `AsyncPrefetchBackend::with_depth`
now constructs `io_uring::IoUring`, probes it with a NOP SQE/CQE
round-trip, spawns a reaper thread, and serves `queue_read` via
`IORING_OP_READ`. Milestones 2–6 still need wiring (the demand-path
overlap pattern, the dispatcher hook, instrumentation, and the bench
harness).

## Blockers / portability notes

- **Linux kernel >= 5.1** for io_uring at all; >= 5.6 for the
  `IORING_OP_READ` with arbitrary FDs we need; >= 5.10 strongly
  preferred for `IORING_FEAT_NODROP` (without it, CQE overflow
  silently drops completions and we'd have to handle that). The
  `io-uring` crate gates older features at runtime, so we just
  check at startup and fall back to madvise if the kernel is too
  old.

- **WSL2:** io_uring is **not available** in WSL2 under the standard
  Microsoft kernel as of mid-2026. Our matias Windows fleet runs
  via WSL2 for some research workloads. The fallback path handles
  this transparently — the runtime feature probe returns false and
  we go down the existing `madvise` arm. Logged at info level on
  startup so anyone bench-comparing knows which path fired.

- **Containers:** Docker / podman / k8s frequently strip
  `io_uring_setup(2)` via seccomp profile. The default Docker
  seccomp profile **blocked** io_uring entirely until docker 23
  (2023-02), and many k8s clusters still ship older profiles.
  Detect with a probe SQE at startup; on EPERM, fall back.

- **NVMe queue depth:** the speedup is gated on the NVMe being
  capable of high queue depth (modern consumer NVMe: 32-64;
  enterprise: 128+). On a slow SATA SSD or a contended HDD, the
  bottleneck shifts to the device queue and io_uring won't help.

- **Page cache pressure:** the prefetcher still populates the page
  cache, so the same RSS-pressure story applies (iter 054 expert
  pinning composes the same way). io_uring with `IORING_OP_NOP`
  + `O_DIRECT` could bypass the page cache for the dispatch path,
  but then we lose the mmap-friendly overlap pattern. Stays
  page-cache-backed for v1.

- **Buffer alignment:** for buffered reads, no alignment required.
  If we ever pivot to `O_DIRECT`, buffers must be 512-byte aligned;
  the safetensors `data_offsets` are word-aligned but not always
  512-aligned. Tracked as a v2 concern.

- **File descriptor lifetime:** the existing `Shard` keeps the
  mmap alive via `Mmap`. For io_uring we need the raw `RawFd`
  too; we'll re-open the file alongside the mmap. The fd is
  owned by `Shard` and dropped with it — no `unsafe` lifetime
  juggling.

## Why this is worth doing (and when it isn't)

iter 070's main finding: **at 7 features × max settings, the
composed chain regressed -32% vs the K=6 baseline.** The hypothesis
was that ~1700 madvise/token saturate the kernel readahead queue
and starve demand reads. If that's right, separating the prefetch
queue from the demand queue should let the chain compose without
the readahead-greedy regression.

If the hypothesis is wrong (e.g. the regression is actually about
RSS pressure from pinning + speculative prefetch reservations,
not readahead-queue contention), io_uring won't help and we'll
see the same number on the bench. That's a useful negative
result either way — it tells us the limit is NVMe bandwidth,
not the kernel's scheduling.

## Non-goals (deliberately deferred)

- Replacing the *mmap* layout itself (would need to redesign the
  GEMM kernel to take owned buffers, not slices into mmap pages).
- O_DIRECT mode — needs alignment + page-cache-bypass story.
- Multi-FD scatter-gather reads (one SQE per slice today; could
  batch with `IORING_OP_READV`).
- io_uring-driven *fixed* buffers (registered with the kernel
  for zero-copy). Useful follow-up but not needed for v1.
- Polled vs interrupt-driven completion mode (SQPOLL). Worth
  trying for the dispatch path, but requires CAP_SYS_NICE.
