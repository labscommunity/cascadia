//! Mmap-based safetensors reader that exposes K2.6 experts directly
//! from the model's on-disk format.
//!
//! Avoids duplicating the ~553 GB of expert weights — the safetensors
//! shards are already on disk and have the same packed-int4 layout the
//! kernel expects. We just need to find each (layer, expert) tensor's
//! byte range and pass slices into the existing kernel.
//!
//! Format reminder: each safetensors shard is
//!
//! ```text
//!   bytes 0..8         : header_len u64 little-endian
//!   bytes 8..8+header_len  : UTF-8 JSON metadata
//!   bytes 8+header_len.. : raw tensor data
//! ```
//!
//! Metadata format::
//!
//!   {
//!     "<tensor_name>": {
//!       "dtype": "BF16" | "I32" | ...,
//!       "shape": [...],
//!       "data_offsets": [start, end]   // relative to start of data section
//!     },
//!     "__metadata__": {...}
//!   }

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

#[cfg(unix)]
use memmap2::Advice;
use memmap2::Mmap;
use parking_lot::RwLock;
#[cfg(windows)]
use windows_sys::Win32::System::Memory::{
    PrefetchVirtualMemory, VirtualLock, VirtualUnlock, WIN32_MEMORY_RANGE_ENTRY,
};
#[cfg(windows)]
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use crate::format::GemmError;

/// Per-shard mmap + lookup table: tensor name → (data start, length) in
/// the mmap. Public because it appears in the return type of
/// `tensor_bytes` / `layer0` / `embed_tokens` (callers pin the Arc to
/// keep the returned slice valid), but the fields are all private.
pub struct Shard {
    mmap: Mmap,
    data_start: usize,
    tensors: HashMap<String, (usize, usize)>, // (data_offset_in_data_section, length)
}

impl Shard {
    fn open(path: &Path) -> Result<Self, GemmError> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        if mmap.len() < 8 {
            return Err(GemmError::Truncated {
                expected: 8,
                actual: mmap.len(),
            });
        }
        let mut hdr = [0u8; 8];
        hdr.copy_from_slice(&mmap[..8]);
        let header_len = u64::from_le_bytes(hdr) as usize;
        if 8 + header_len > mmap.len() {
            return Err(GemmError::Truncated {
                expected: 8 + header_len,
                actual: mmap.len(),
            });
        }
        let json_bytes = &mmap[8..8 + header_len];
        let json: serde_json::Value = serde_json::from_slice(json_bytes).map_err(|e| {
            GemmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("safetensors json parse: {e}"),
            ))
        })?;
        let map = json.as_object().ok_or_else(|| {
            GemmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "safetensors json not object",
            ))
        })?;
        let mut tensors = HashMap::with_capacity(map.len());
        for (k, v) in map {
            if k == "__metadata__" {
                continue;
            }
            let offsets = v
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .ok_or_else(|| {
                    GemmError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("missing data_offsets for {k}"),
                    ))
                })?;
            if offsets.len() != 2 {
                return Err(GemmError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("data_offsets for {k} not length 2"),
                )));
            }
            let start = offsets[0].as_u64().unwrap_or(0) as usize;
            let end = offsets[1].as_u64().unwrap_or(0) as usize;
            if end < start {
                continue;
            }
            tensors.insert(k.clone(), (start, end - start));
        }
        Ok(Self {
            mmap,
            data_start: 8 + header_len,
            tensors,
        })
    }

    fn slice(&self, tensor_name: &str) -> Option<&[u8]> {
        self.tensors.get(tensor_name).map(|&(off, len)| {
            let start = self.data_start + off;
            &self.mmap[start..start + len]
        })
    }

    /// Hint the OS that we'll need a tensor's byte range soon. On Unix
    /// this wraps `madvise(MADV_WILLNEED)`; on Windows it wraps
    /// `PrefetchVirtualMemory` (kernel32, Win8+). Best-effort: errors
    /// swallowed because both APIs are advisory and the worst case is
    /// the read happens on demand later (which is exactly the
    /// no-prefetch baseline).
    fn advise_willneed(&self, tensor_name: &str) {
        let Some(&(_off, _len)) = self.tensors.get(tensor_name) else {
            return;
        };
        // memmap2's `advise_range` (Unix) and `PrefetchVirtualMemory`
        // (Windows) both round the start down and the length up to the
        // next page boundary internally, so we don't align here.
        #[cfg(unix)]
        {
            let start = self.data_start + _off;
            let _ = self.mmap.advise_range(Advice::WillNeed, start, _len);
        }
        #[cfg(windows)]
        {
            self.win_prefetch_range(_off, _len);
        }
    }

    /// autolab iter 054 (expert pinning): lock one tensor's byte range
    /// into RAM so it can never be paged out. On Unix wraps
    /// `mlock(addr, len)`; on Windows wraps `VirtualLock(addr, len)`.
    /// Returns the number of bytes actually attempted to pin (= the
    /// tensor's on-disk length, *not* page-rounded — the OS rounds
    /// internally and that doesn't change accounting at the granularity
    /// we care about). Returns 0 on failure so the caller can decide
    /// whether to bail (e.g. RLIMIT_MEMLOCK exhausted) without us
    /// holding any partial state.
    ///
    /// **Critical contract:** the returned byte count is what we'd need
    /// to `munlock` later. Callers track the total via the
    /// `SafetensorsExpertSource::pinned_bytes` counter for diagnostics
    /// and budget enforcement.
    fn pin_range(&self, tensor_name: &str) -> usize {
        let Some(&(_off, _len)) = self.tensors.get(tensor_name) else {
            return 0;
        };
        if _len == 0 {
            return 0;
        }
        #[cfg(unix)]
        {
            let start = self.data_start + _off;
            // SAFETY: `start..start+len` is inside the live mmap (validated
            // at Shard::open via the data_offsets check) and we never deref
            // the pointer — only hand it to the kernel. `mlock` does not
            // require alignment; the kernel page-rounds internally.
            let addr = unsafe { self.mmap.as_ptr().add(start) } as *const libc::c_void;
            let rc = unsafe { libc::mlock(addr, _len) };
            if rc == 0 {
                _len
            } else {
                0
            }
        }
        #[cfg(windows)]
        {
            self.win_lock_range(_off, _len)
        }
        #[cfg(not(any(unix, windows)))]
        {
            // No-op fallback: pinning isn't available. Returning 0 makes
            // the caller treat the expert as unpinned, which degrades
            // gracefully to the C1-prefetch-only baseline.
            let _ = _off;
            0
        }
    }

    /// Inverse of `pin_range`. Best-effort: errors swallowed so a
    /// double-unpin (or unpin after a Shard recreation that lost the
    /// pin) doesn't poison the source. Returns the byte count the
    /// caller should subtract from `pinned_bytes` if non-zero.
    fn unpin_range(&self, tensor_name: &str) -> usize {
        let Some(&(_off, _len)) = self.tensors.get(tensor_name) else {
            return 0;
        };
        if _len == 0 {
            return 0;
        }
        #[cfg(unix)]
        {
            let start = self.data_start + _off;
            // SAFETY: same as pin_range. munlock is idempotent on Linux
            // (returns 0 even if the range wasn't locked); other Unixen
            // may return EINVAL — we ignore either way.
            let addr = unsafe { self.mmap.as_ptr().add(start) } as *const libc::c_void;
            let rc = unsafe { libc::munlock(addr, _len) };
            if rc == 0 {
                _len
            } else {
                0
            }
        }
        #[cfg(windows)]
        {
            self.win_unlock_range(_off, _len)
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = _off;
            0
        }
    }

    /// Windows arm of `pin_range`. `VirtualLock` requires SE_LOCK_MEMORY
    /// privilege OR a working-set big enough to hold the locked region
    /// (auto-grown when locking under `SetProcessWorkingSetSizeEx` is
    /// configured). Returns 0 on failure — caller falls back to
    /// C1-prefetch-only.
    #[cfg(windows)]
    fn win_lock_range(&self, off: usize, len: usize) -> usize {
        if len == 0 {
            return 0;
        }
        let start = self.data_start + off;
        let base = self.mmap.as_ptr();
        // SAFETY: pointer is inside the live mmap; only handed to kernel.
        let addr = unsafe { base.add(start) } as *const core::ffi::c_void;
        // SAFETY: VirtualLock signature accepts a const pointer; non-zero
        // return is success. Failure is non-fatal (caller treats as unpinned).
        let ok = unsafe { VirtualLock(addr, len) };
        if ok != 0 {
            len
        } else {
            0
        }
    }

    /// Windows arm of `unpin_range`. `VirtualUnlock` is best-effort:
    /// returns failure if the range was already unlocked, which we
    /// ignore (idempotent semantics).
    #[cfg(windows)]
    fn win_unlock_range(&self, off: usize, len: usize) -> usize {
        if len == 0 {
            return 0;
        }
        let start = self.data_start + off;
        let base = self.mmap.as_ptr();
        // SAFETY: see win_lock_range.
        let addr = unsafe { base.add(start) } as *const core::ffi::c_void;
        unsafe {
            let _ = VirtualUnlock(addr, len);
        }
        len
    }

    /// Windows arm of `advise_willneed`. Resolves a single tensor's
    /// (offset, length) inside this mmap to a virtual-address range and
    /// hands it to `PrefetchVirtualMemory` for async page-in.
    ///
    /// Semantics MS documents for `PrefetchVirtualMemory`:
    ///   * "purely a performance optimization … treated as a strong
    ///     hint by the system and is subject to usual physical memory
    ///     constraints where it can completely or partially fail under
    ///     low-memory conditions."
    ///   * Returns nonzero on success, zero on failure (we ignore both —
    ///     the read will just happen synchronously on first access).
    ///   * Available on Windows 8 / Server 2012 and up.
    ///
    /// We call it once per tensor (one entry in the
    /// `WIN32_MEMORY_RANGE_ENTRY` array). The expert-prefetch caller
    /// invokes us six times per expert (gate/up/down × packed/scale)
    /// which is consistent with how the Unix path iterates.
    /// Calling with batches of one keeps the code simple and avoids
    /// holding a `Vec` of entries that would have to outlive the
    /// `unsafe` call; the per-call cost is just one cross-DLL hop
    /// (~µs), same order of magnitude as `madvise` on Linux.
    #[cfg(windows)]
    fn win_prefetch_range(&self, off: usize, len: usize) {
        if len == 0 {
            return;
        }
        // Tensor offset is within the mmap's data section; compute the
        // raw virtual address by adding it to the mmap's base ptr.
        // `Mmap::as_ptr()` returns a `*const u8` aimed at byte 0 of the
        // mapped view. Adding `data_start + off` lands us at the first
        // tensor byte. SAFETY: the resulting pointer is inside the live
        // mmap; we never deref it, only hand it to the kernel.
        let start = self.data_start + off;
        let base = self.mmap.as_ptr();
        let addr = unsafe { base.add(start) } as *mut core::ffi::c_void;

        let entry = WIN32_MEMORY_RANGE_ENTRY {
            VirtualAddress: addr,
            NumberOfBytes: len,
        };
        // SAFETY: hProcess from GetCurrentProcess is a pseudo-handle (no
        // close required), the entry array lives on the stack for the
        // duration of the call, and Flags must be 0 per MSDN. Return
        // value is intentionally ignored (advisory API).
        unsafe {
            let _ = PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
        }
    }
}

/// Source of safetensors-backed expert weights. Caches mmaps lazily.
///
/// Construct once per process with the model directory; clone is cheap
/// (Arc'd).
#[derive(Clone)]
pub struct SafetensorsExpertSource {
    model_dir: PathBuf,
    /// Map from tensor name → shard filename, from
    /// `model.safetensors.index.json`.
    weight_map: Arc<HashMap<String, String>>,
    /// Lazy mmap cache: shard filename → Shard.
    shards: Arc<RwLock<HashMap<String, Arc<Shard>>>>,
    /// autolab iter 054 (expert pinning): cumulative bytes successfully
    /// `mlock`/`VirtualLock`'d via `pin_expert` minus the bytes returned
    /// to the OS via `unpin_expert`. Used by the runner for budget
    /// enforcement and diagnostics. Sum of *attempted* sizes (not page-
    /// rounded) — close enough for the warn-threshold check.
    pinned_bytes: Arc<AtomicU64>,
    /// autolab iter 054 (expert pinning): set of (layer, expert) pairs
    /// currently pinned. Used to (a) make pin_expert idempotent (skip
    /// double-pin which would double-count bytes) and (b) drive unpin
    /// of the full set when the runner is being torn down or a new
    /// pin set is being installed.
    pinned_experts: Arc<RwLock<HashMap<(u32, u32), usize>>>,
}

impl SafetensorsExpertSource {
    /// Open the model dir and parse `model.safetensors.index.json`.
    pub fn open(model_dir: impl Into<PathBuf>) -> Result<Self, GemmError> {
        let model_dir = model_dir.into();
        let idx_path = model_dir.join("model.safetensors.index.json");
        let idx_bytes = std::fs::read(&idx_path)?;
        let idx: serde_json::Value = serde_json::from_slice(&idx_bytes).map_err(|e| {
            GemmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("safetensors index json parse: {e}"),
            ))
        })?;
        let weight_map = idx
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| {
                GemmError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "weight_map missing",
                ))
            })?;
        let mut map: HashMap<String, String> = HashMap::with_capacity(weight_map.len());
        for (k, v) in weight_map {
            if let Some(s) = v.as_str() {
                map.insert(k.clone(), s.to_string());
            }
        }
        Ok(Self {
            model_dir,
            weight_map: Arc::new(map),
            shards: Arc::new(RwLock::new(HashMap::new())),
            pinned_bytes: Arc::new(AtomicU64::new(0)),
            pinned_experts: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    fn shard_for(&self, tensor_name: &str) -> Result<Arc<Shard>, GemmError> {
        let shard_name = self.weight_map.get(tensor_name).ok_or_else(|| {
            GemmError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("tensor {tensor_name} not in weight_map"),
            ))
        })?;
        if let Some(s) = self.shards.read().get(shard_name) {
            return Ok(Arc::clone(s));
        }
        // Race: another thread may insert it; tolerated.
        let path = self.model_dir.join(shard_name);
        let s = Arc::new(Shard::open(&path)?);
        self.shards
            .write()
            .insert(shard_name.clone(), Arc::clone(&s));
        Ok(s)
    }

    fn slice(&self, tensor_name: &str) -> Result<(Arc<Shard>, &'static [u8]), GemmError> {
        let s = self.shard_for(tensor_name)?;
        let bytes = s.slice(tensor_name).ok_or_else(|| {
            GemmError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("tensor {tensor_name} not in shard"),
            ))
        })?;
        // Cast lifetime: bytes are tied to Arc<Shard>; we return both
        // so caller pins it.
        let static_bytes: &'static [u8] = unsafe { std::mem::transmute(bytes) };
        Ok((s, static_bytes))
    }

    /// Names of the six tensors that compose one expert. Same enumeration
    /// `expert()` uses; pulled out so prefetch can iterate them without
    /// materializing the full `SafetensorsExpert`.
    fn expert_tensor_names(layer: u32, expert: u32) -> [String; 6] {
        let base = format!(
            "language_model.model.layers.{}.mlp.experts.{}",
            layer, expert
        );
        [
            format!("{}.gate_proj.weight_packed", base),
            format!("{}.gate_proj.weight_scale", base),
            format!("{}.up_proj.weight_packed", base),
            format!("{}.up_proj.weight_scale", base),
            format!("{}.down_proj.weight_packed", base),
            format!("{}.down_proj.weight_scale", base),
        ]
    }

    /// autolab iter 029 (C1): issue `madvise(MADV_WILLNEED)` on every
    /// byte slice for one expert. Non-blocking — the kernel queues
    /// async readahead. Returns the number of slices for which madvise
    /// succeeded (caller can ignore; useful for instrumentation).
    ///
    /// Designed to be called from a dedicated prefetch thread that's
    /// fed by the inference path. Holds no locks across madvise calls
    /// (the shard lookup briefly takes the inner RwLock).
    pub fn prefetch_expert(&self, layer: u32, expert: u32) -> usize {
        let mut hits = 0usize;
        for name in Self::expert_tensor_names(layer, expert).iter() {
            // Resolve the shard. If the tensor isn't in the weight map
            // we silently skip — prefetch is best-effort.
            if !self.weight_map.contains_key(name) {
                continue;
            }
            // Open-or-clone the shard. We tolerate the open cost because
            // (a) prefetch runs off the hot path and (b) once cached
            // every subsequent prefetch on that shard is one HashMap
            // hit + one madvise syscall.
            let shard = match self.shard_for(name) {
                Ok(s) => s,
                Err(_) => continue,
            };
            shard.advise_willneed(name);
            hits += 1;
        }
        hits
    }

    /// autolab iter 054 (expert pinning): `mlock` (Linux/macOS/BSD) or
    /// `VirtualLock` (Windows) every byte slice that makes up one
    /// expert (six tensors: gate / up / down × packed / scale). Returns
    /// the number of bytes successfully pinned for this call — `0` on
    /// any failure path (tensor missing, shard open failure, syscall
    /// error). Idempotent: pinning an already-pinned expert is a no-op
    /// and returns `0` (does not double-count toward `pinned_bytes`).
    ///
    /// Strategy: composes with C1 prefetch — pinning short-circuits the
    /// expert's pages from ever being evicted under memory pressure,
    /// while C1 prefetch hides the page-in latency for the unpinned
    /// tail. On K2.6 with a heavy-tailed expert hit distribution, pinning
    /// the top ~10% per layer (38 of 384) covers ~80% of dispatches at
    /// ~47 GB total cost — easily fits on a 133 GB miner with the page
    /// cache for the unpinned 90% behind it.
    ///
    /// Critical: callers MUST verify `rlimit_memlock_soft()` is large
    /// enough to hold the target set before kicking off bulk pin calls;
    /// once `mlock` starts returning ENOMEM, partial pins remain and
    /// the page cache will start thrashing as the OS tries to free
    /// pages that are locked.
    pub fn pin_expert(&self, layer: u32, expert: u32) -> usize {
        // Idempotency check — already pinned ⇒ no-op. Drop the read
        // lock before doing the syscall work to keep contention low
        // when many layers fire pin_expert in parallel during the
        // warmup-completion handoff.
        if self.pinned_experts.read().contains_key(&(layer, expert)) {
            return 0;
        }
        let mut total = 0usize;
        for name in Self::expert_tensor_names(layer, expert).iter() {
            if !self.weight_map.contains_key(name) {
                continue;
            }
            let shard = match self.shard_for(name) {
                Ok(s) => s,
                Err(_) => continue,
            };
            total += shard.pin_range(name);
        }
        if total > 0 {
            // Record after successful pin so partial failures don't
            // leave us with a counter mismatch on unpin. A `(layer,
            // expert)` row may exist with `0` bytes (full failure) —
            // store it only if any pages took.
            self.pinned_experts.write().insert((layer, expert), total);
            self.pinned_bytes
                .fetch_add(total as u64, AtomicOrdering::Relaxed);
        }
        total
    }

    /// Inverse of `pin_expert`. Returns the byte count actually released
    /// (0 if the expert wasn't pinned). Safe to call from any thread.
    pub fn unpin_expert(&self, layer: u32, expert: u32) -> usize {
        let prev_bytes = match self.pinned_experts.write().remove(&(layer, expert)) {
            Some(b) => b,
            None => return 0,
        };
        let mut released = 0usize;
        for name in Self::expert_tensor_names(layer, expert).iter() {
            if !self.weight_map.contains_key(name) {
                continue;
            }
            let shard = match self.shard_for(name) {
                Ok(s) => s,
                Err(_) => continue,
            };
            released += shard.unpin_range(name);
        }
        // Subtract the *recorded* pin size, not the freshly-returned
        // count — the two should match in steady state, but if mmap
        // hot-reload ever changed offsets between pin and unpin we'd
        // want the accounting to follow what we promised on the way in.
        self.pinned_bytes
            .fetch_sub(prev_bytes as u64, AtomicOrdering::Relaxed);
        released
    }

    /// autolab iter 054: unpin every expert this source has pinned.
    /// Useful at runner teardown or before installing a new pin set.
    /// Returns `(experts_unpinned, bytes_released_recorded)`.
    pub fn unpin_all_experts(&self) -> (usize, u64) {
        let to_unpin: Vec<(u32, u32)> = self.pinned_experts.read().keys().copied().collect();
        let n = to_unpin.len();
        let mut released = 0u64;
        for (layer, expert) in to_unpin {
            let bytes = self.unpin_expert(layer, expert);
            released += bytes as u64;
        }
        (n, released)
    }

    /// Snapshot the cumulative pinned-bytes counter (sum of
    /// pin_expert returns minus unpin_expert returns).
    pub fn pinned_bytes(&self) -> u64 {
        self.pinned_bytes.load(AtomicOrdering::Relaxed)
    }

    /// Snapshot the count of (layer, expert) pairs currently pinned.
    pub fn pinned_expert_count(&self) -> usize {
        self.pinned_experts.read().len()
    }

    /// autolab iter 054: read the process's soft RLIMIT_MEMLOCK on Unix.
    /// Returns `Some(soft_bytes)` or `None` if the syscall fails or
    /// the limit is `RLIM_INFINITY` (which we report as `u64::MAX` so
    /// callers can compare directly). On non-Unix returns `None` —
    /// Windows has no equivalent rlimit (instead the working-set size
    /// caps `VirtualLock`, which auto-grows).
    pub fn rlimit_memlock_soft() -> Option<u64> {
        #[cfg(unix)]
        {
            let mut rlim = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            // SAFETY: RLIMIT_MEMLOCK is a valid resource id and rlim is
            // owned + initialized to zero.
            let rc = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rlim) };
            if rc != 0 {
                return None;
            }
            // RLIM_INFINITY is the sentinel — report as MAX so callers
            // can do a single `>= required` check without special-casing.
            if rlim.rlim_cur == libc::RLIM_INFINITY {
                Some(u64::MAX)
            } else {
                Some(rlim.rlim_cur as u64)
            }
        }
        #[cfg(not(unix))]
        {
            None
        }
    }

    /// Helper: bytes required to pin the named expert (sum of the six
    /// tensor slice lengths). Returns `0` if the expert isn't in the
    /// weight map — useful as a probe before deciding to pin.
    pub fn expert_size_bytes(&self, layer: u32, expert: u32) -> u64 {
        let mut total = 0u64;
        for name in Self::expert_tensor_names(layer, expert).iter() {
            let shard = match self.shard_for(name) {
                Ok(s) => s,
                Err(_) => return 0,
            };
            if let Some(&(_off, len)) = shard.tensors.get(name) {
                total += len as u64;
            }
        }
        total
    }

    /// Fetch one expert's six tensor slices (gate/up/down × packed/scale).
    /// Returns a struct holding Arc references to the mmaps so the
    /// slices stay valid as long as the result is alive.
    pub fn expert(&self, layer: u32, expert: u32) -> Result<SafetensorsExpert, GemmError> {
        let base = format!(
            "language_model.model.layers.{}.mlp.experts.{}",
            layer, expert
        );
        let mut shard_pins = Vec::with_capacity(6);
        let mut slices = [&[][..]; 6];
        for (i, (proj, suf)) in [
            ("gate_proj", "weight_packed"),
            ("gate_proj", "weight_scale"),
            ("up_proj", "weight_packed"),
            ("up_proj", "weight_scale"),
            ("down_proj", "weight_packed"),
            ("down_proj", "weight_scale"),
        ]
        .iter()
        .enumerate()
        {
            let name = format!("{}.{}.{}", base, proj, suf);
            let (shard, bytes) = self.slice(&name)?;
            shard_pins.push(shard);
            slices[i] = bytes;
        }
        Ok(SafetensorsExpert {
            _pins: shard_pins,
            gate_packed: slices[0],
            gate_scale: slices[1],
            up_packed: slices[2],
            up_scale: slices[3],
            down_packed: slices[4],
            down_scale: slices[5],
        })
    }
}

/// One expert's weights backed by safetensors mmaps. The Arc refs in
/// `_pins` keep the shard mmaps alive for the byte slices.
pub struct SafetensorsExpert {
    _pins: Vec<Arc<Shard>>,
    pub gate_packed: &'static [u8],
    pub gate_scale: &'static [u8],
    pub up_packed: &'static [u8],
    pub up_scale: &'static [u8],
    pub down_packed: &'static [u8],
    pub down_scale: &'static [u8],
}

unsafe impl Send for SafetensorsExpert {}
unsafe impl Sync for SafetensorsExpert {}

impl SafetensorsExpertSource {
    /// Generic tensor lookup: returns raw bytes for any named tensor in
    /// the model. Useful for fetching shell weights (attention,
    /// layernorm, router, shared expert) which aren't expert-indexed.
    pub fn tensor_bytes(
        &self,
        tensor_name: &str,
    ) -> Result<(Arc<Shard>, &'static [u8]), GemmError> {
        self.slice(tensor_name)
    }
}

/// One shell's weights for layer L. All bf16 except for one f32 bias
/// (`gate_correction_bias`). The Arc shards in `_pins` keep the
/// safetensors mmaps alive while these slices are in use.
pub struct SafetensorsShell {
    _pins: Vec<Arc<Shard>>,
    pub layer: u32,

    /// `input_layernorm.weight` — bf16 [hidden].
    pub input_norm: &'static [u8],
    /// `self_attn.q_a_proj.weight` — bf16 [q_lora_rank=1536, hidden=7168].
    pub q_a_proj: &'static [u8],
    /// `self_attn.q_a_layernorm.weight` — bf16 [q_lora_rank].
    pub q_a_norm: &'static [u8],
    /// `self_attn.q_b_proj.weight` — bf16 [heads*qk_head_dim, q_lora_rank].
    pub q_b_proj: &'static [u8],
    /// `self_attn.kv_a_proj_with_mqa.weight` — bf16 [kv_lora_rank+qk_rope, hidden].
    pub kv_a_proj: &'static [u8],
    /// `self_attn.kv_a_layernorm.weight` — bf16 [kv_lora_rank=512].
    pub kv_a_norm: &'static [u8],
    /// `self_attn.kv_b_proj.weight` — bf16 [heads*(qk_nope+v_head), kv_lora_rank].
    pub kv_b_proj: &'static [u8],
    /// `self_attn.o_proj.weight` — bf16 [hidden, heads*v_head].
    pub o_proj: &'static [u8],

    /// `post_attention_layernorm.weight` — bf16 [hidden].
    pub post_norm: &'static [u8],

    /// `mlp.gate.weight` — bf16 [n_routed_experts=384, hidden].
    pub router_weight: &'static [u8],
    /// `mlp.gate.e_score_correction_bias` — f32 [n_routed_experts].
    pub router_bias: &'static [u8],

    /// `mlp.shared_experts.gate_proj.weight` — bf16 [intermediate=2048, hidden].
    pub shared_gate: &'static [u8],
    /// `mlp.shared_experts.up_proj.weight` — bf16 [intermediate, hidden].
    pub shared_up: &'static [u8],
    /// `mlp.shared_experts.down_proj.weight` — bf16 [hidden, intermediate].
    pub shared_down: &'static [u8],
}

unsafe impl Send for SafetensorsShell {}
unsafe impl Sync for SafetensorsShell {}

/// Layer 0 of K2.6 is dense (not MoE) — same MLA attention as a shell,
/// but the MLP is a single SwiGLU instead of a router + 384 experts +
/// shared expert.
pub struct SafetensorsLayer0 {
    _pins: Vec<Arc<Shard>>,
    pub layer: u32,

    /// `input_layernorm.weight` — bf16 [hidden].
    pub input_norm: &'static [u8],
    /// `self_attn.q_a_proj.weight` — bf16 [q_lora_rank, hidden].
    pub q_a_proj: &'static [u8],
    pub q_a_norm: &'static [u8],
    pub q_b_proj: &'static [u8],
    pub kv_a_proj: &'static [u8],
    pub kv_a_norm: &'static [u8],
    pub kv_b_proj: &'static [u8],
    pub o_proj: &'static [u8],

    pub post_norm: &'static [u8],

    /// `mlp.gate_proj.weight` — bf16 [intermediate_dense=18432, hidden].
    pub gate_proj: &'static [u8],
    /// `mlp.up_proj.weight` — bf16 [intermediate_dense, hidden].
    pub up_proj: &'static [u8],
    /// `mlp.down_proj.weight` — bf16 [hidden, intermediate_dense].
    pub down_proj: &'static [u8],
}

unsafe impl Send for SafetensorsLayer0 {}
unsafe impl Sync for SafetensorsLayer0 {}

impl SafetensorsExpertSource {
    /// Fetch one layer's shell tensors. Mmaps the relevant safetensors
    /// shards lazily (with internal caching).
    pub fn shell(&self, layer: u32) -> Result<SafetensorsShell, GemmError> {
        let base = format!("language_model.model.layers.{layer}");
        let names: [&str; 14] = [
            "input_layernorm.weight",
            "self_attn.q_a_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate.weight",
            "mlp.gate.e_score_correction_bias",
            "mlp.shared_experts.gate_proj.weight",
            "mlp.shared_experts.up_proj.weight",
            "mlp.shared_experts.down_proj.weight",
        ];
        let mut pins = Vec::with_capacity(names.len());
        let mut slices: [&'static [u8]; 14] = [&[]; 14];
        for (i, suf) in names.iter().enumerate() {
            let full = format!("{}.{}", base, suf);
            let (shard, bytes) = self.slice(&full)?;
            pins.push(shard);
            slices[i] = bytes;
        }
        Ok(SafetensorsShell {
            _pins: pins,
            layer,
            input_norm: slices[0],
            q_a_proj: slices[1],
            q_a_norm: slices[2],
            q_b_proj: slices[3],
            kv_a_proj: slices[4],
            kv_a_norm: slices[5],
            kv_b_proj: slices[6],
            o_proj: slices[7],
            post_norm: slices[8],
            router_weight: slices[9],
            router_bias: slices[10],
            shared_gate: slices[11],
            shared_up: slices[12],
            shared_down: slices[13],
        })
    }

    /// Fetch the dense layer-0 tensors (attention + SwiGLU MLP).
    /// Mmaps the relevant safetensors shards lazily.
    pub fn layer0(&self) -> Result<SafetensorsLayer0, GemmError> {
        let layer: u32 = 0;
        let base = format!("language_model.model.layers.{layer}");
        let names: [&str; 12] = [
            "input_layernorm.weight",
            "self_attn.q_a_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];
        let mut pins = Vec::with_capacity(names.len());
        let mut slices: [&'static [u8]; 12] = [&[]; 12];
        for (i, suf) in names.iter().enumerate() {
            let full = format!("{base}.{suf}");
            let (shard, bytes) = self.slice(&full)?;
            pins.push(shard);
            slices[i] = bytes;
        }
        Ok(SafetensorsLayer0 {
            _pins: pins,
            layer,
            input_norm: slices[0],
            q_a_proj: slices[1],
            q_a_norm: slices[2],
            q_b_proj: slices[3],
            kv_a_proj: slices[4],
            kv_a_norm: slices[5],
            kv_b_proj: slices[6],
            o_proj: slices[7],
            post_norm: slices[8],
            gate_proj: slices[9],
            up_proj: slices[10],
            down_proj: slices[11],
        })
    }

    /// Fetch the model's input embedding table — bf16
    /// `[vocab_size, hidden_size]`, flat row-major. Returns the
    /// pinned shard reference plus a slice into the mmap.
    pub fn embed_tokens(&self) -> Result<(Arc<Shard>, &'static [u8]), GemmError> {
        self.slice("language_model.model.embed_tokens.weight")
    }
}

/// autolab iter 054 (expert pinning): tests cover the on-disk pinning
/// pipeline end-to-end against a hand-rolled synthetic safetensors
/// shard. We can't easily run against real K2.6 weights from a unit
/// test (the model is ~480 GB on disk, hundreds of shards), so we
/// build the smallest legal shard that has the six per-expert tensors
/// for one (layer, expert) pair and verify pin/unpin accounting +
/// idempotency + size probe.
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a tiny safetensors shard at `path` containing exactly the
    /// six per-expert tensors for `(layer, expert)`. Each tensor is
    /// filled with `byte_size` random-looking bytes — the actual
    /// content doesn't matter, only the offsets and lengths. Returns
    /// the per-tensor byte size so tests can compute expected totals.
    fn write_test_shard(
        path: &std::path::Path,
        layer: u32,
        expert: u32,
        byte_size: usize,
    ) -> usize {
        let names = SafetensorsExpertSource::expert_tensor_names(layer, expert);
        // Build a JSON header with `data_offsets` for each tensor.
        let mut tensors_json = serde_json::Map::new();
        for (i, name) in names.iter().enumerate() {
            let start = i * byte_size;
            let end = start + byte_size;
            tensors_json.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "U8",
                    "shape": [byte_size],
                    "data_offsets": [start, end],
                }),
            );
        }
        let header = serde_json::Value::Object(tensors_json);
        let header_bytes = serde_json::to_vec(&header).expect("json serialize");
        let header_len = header_bytes.len() as u64;
        let mut f = std::fs::File::create(path).expect("create test shard");
        f.write_all(&header_len.to_le_bytes()).expect("write len");
        f.write_all(&header_bytes).expect("write header");
        // Now write `byte_size * 6` bytes of data.
        let data = vec![0xABu8; byte_size * 6];
        f.write_all(&data).expect("write data");
        f.sync_all().expect("fsync");
        byte_size
    }

    /// Build a minimal `model.safetensors.index.json` pointing every
    /// per-expert tensor for (layer, expert) at the named shard.
    fn write_test_index(dir: &std::path::Path, layer: u32, expert: u32, shard_name: &str) {
        let names = SafetensorsExpertSource::expert_tensor_names(layer, expert);
        let mut weight_map = serde_json::Map::new();
        for name in names.iter() {
            weight_map.insert(name.clone(), serde_json::Value::String(shard_name.into()));
        }
        let idx = serde_json::json!({
            "metadata": {"total_size": 0},
            "weight_map": serde_json::Value::Object(weight_map),
        });
        let p = dir.join("model.safetensors.index.json");
        std::fs::write(&p, serde_json::to_vec_pretty(&idx).unwrap()).expect("write index");
    }

    #[test]
    fn expert_tensor_names_six_entries_with_correct_template() {
        let names = SafetensorsExpertSource::expert_tensor_names(7, 42);
        assert_eq!(names.len(), 6);
        // Verify the template (layer, expert) substitution matches the
        // `expert()` path's enumeration.
        assert_eq!(
            names[0],
            "language_model.model.layers.7.mlp.experts.42.gate_proj.weight_packed"
        );
        assert_eq!(
            names[5],
            "language_model.model.layers.7.mlp.experts.42.down_proj.weight_scale"
        );
    }

    #[test]
    fn expert_size_bytes_sums_six_tensor_lengths() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let shard_name = "model-00001-of-00001.safetensors";
        let shard_path = tmp.path().join(shard_name);
        let per_tensor_bytes = 256usize;
        write_test_shard(&shard_path, 0, 0, per_tensor_bytes);
        write_test_index(tmp.path(), 0, 0, shard_name);
        let src = SafetensorsExpertSource::open(tmp.path()).expect("open source");
        // 6 tensors × 256 B each = 1536 B.
        assert_eq!(
            src.expert_size_bytes(0, 0),
            (per_tensor_bytes * 6) as u64,
            "expert_size_bytes should sum the six tensor slice lengths"
        );
        // Missing expert → 0 (probe is safe to call on unknown ids).
        assert_eq!(src.expert_size_bytes(0, 999), 0);
    }

    #[test]
    fn pin_expert_then_unpin_round_trips_counters() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let shard_name = "model-00001-of-00001.safetensors";
        let shard_path = tmp.path().join(shard_name);
        // 4 KB per tensor — one page on most arches. Total per expert
        // is 24 KB, well below any reasonable RLIMIT_MEMLOCK so the
        // test is robust to environments where the limit isn't bumped.
        let per_tensor_bytes = 4096usize;
        write_test_shard(&shard_path, 1, 2, per_tensor_bytes);
        write_test_index(tmp.path(), 1, 2, shard_name);
        let src = SafetensorsExpertSource::open(tmp.path()).expect("open source");
        assert_eq!(src.pinned_bytes(), 0);
        assert_eq!(src.pinned_expert_count(), 0);

        let bytes_pinned = src.pin_expert(1, 2);
        // On environments where mlock is denied (low RLIMIT_MEMLOCK,
        // restricted CI, etc.) the syscall may fail and we record 0.
        // The test still asserts the accounting is consistent.
        if bytes_pinned == 0 {
            assert_eq!(src.pinned_bytes(), 0);
            assert_eq!(src.pinned_expert_count(), 0);
            return; // Can't test the success path here.
        }
        assert_eq!(
            bytes_pinned,
            per_tensor_bytes * 6,
            "expected to pin all six tensors"
        );
        assert_eq!(src.pinned_bytes(), bytes_pinned as u64);
        assert_eq!(src.pinned_expert_count(), 1);

        // Idempotent: pinning the same expert again is a no-op and
        // doesn't double-count.
        let bytes_pinned_again = src.pin_expert(1, 2);
        assert_eq!(bytes_pinned_again, 0);
        assert_eq!(src.pinned_bytes(), bytes_pinned as u64);
        assert_eq!(src.pinned_expert_count(), 1);

        // Unpin: clears the counters back to zero.
        let released = src.unpin_expert(1, 2);
        assert!(released > 0, "unpin should return non-zero on success");
        assert_eq!(src.pinned_bytes(), 0);
        assert_eq!(src.pinned_expert_count(), 0);

        // Unpinning an unpinned expert is a no-op.
        let released_again = src.unpin_expert(1, 2);
        assert_eq!(released_again, 0);
        assert_eq!(src.pinned_bytes(), 0);
    }

    #[test]
    fn unpin_all_releases_every_pinned_expert() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let shard_name = "model-00001-of-00001.safetensors";
        let shard_path = tmp.path().join(shard_name);
        let per_tensor_bytes = 4096usize;
        // Three experts in the same layer; the index uses the same
        // shard file but the test_shard writer covers only one expert
        // — for unpin_all coverage we re-use the same tensor names by
        // pinning different (layer, expert) pairs and tolerate that
        // only one pair actually has weights. unpin_all should still
        // be a no-op for the absent ones (skip), and clear the present one.
        write_test_shard(&shard_path, 3, 5, per_tensor_bytes);
        write_test_index(tmp.path(), 3, 5, shard_name);
        let src = SafetensorsExpertSource::open(tmp.path()).expect("open source");
        let pinned = src.pin_expert(3, 5);
        if pinned == 0 {
            // mlock denied in this env; can't exercise unpin_all here.
            return;
        }
        // Try to pin a phantom expert — should be a no-op (no tensors
        // in weight_map) and add nothing to the counter.
        let phantom = src.pin_expert(3, 99);
        assert_eq!(phantom, 0);
        assert_eq!(src.pinned_expert_count(), 1);
        let (n, released) = src.unpin_all_experts();
        assert_eq!(n, 1);
        assert_eq!(released, pinned as u64);
        assert_eq!(src.pinned_bytes(), 0);
    }

    #[test]
    fn pin_expert_with_unknown_id_returns_zero() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let shard_name = "model-00001-of-00001.safetensors";
        let shard_path = tmp.path().join(shard_name);
        write_test_shard(&shard_path, 0, 0, 64);
        write_test_index(tmp.path(), 0, 0, shard_name);
        let src = SafetensorsExpertSource::open(tmp.path()).expect("open source");
        // Layer / expert pair that isn't in the index → all six lookups
        // miss the weight_map and pin_expert reports zero without
        // updating the counter.
        let bytes = src.pin_expert(99, 99);
        assert_eq!(bytes, 0);
        assert_eq!(src.pinned_bytes(), 0);
        assert_eq!(src.pinned_expert_count(), 0);
    }

    #[test]
    fn rlimit_memlock_soft_returns_a_value_on_unix() {
        #[cfg(unix)]
        {
            let lim = SafetensorsExpertSource::rlimit_memlock_soft();
            assert!(lim.is_some(), "getrlimit(RLIMIT_MEMLOCK) should succeed");
            // We don't assert a specific minimum — CI may run with the
            // 64 KiB default on Linux; this just ensures the syscall
            // wiring works.
        }
        #[cfg(not(unix))]
        {
            assert!(SafetensorsExpertSource::rlimit_memlock_soft().is_none());
        }
    }
}
