//! Memory-mapped int4_bin expert — the production path for the real 43-layer
//! model, where eagerly dequantizing every expert to f32 would need ~285 GB
//! of RAM per rank. Weights stay packed on disk; each forward decodes the int4
//! nibbles of the rows it touches straight into a fused SIMD dot against the
//! activation (no f32 scratch row — see `dequant_row_dot`).
//!
//! Numerics match the eager [`Expert`](super::model::Expert) path within bf16
//! tolerance, **not bitwise**: the per-row nibble decode matches
//! `loader::dequant_int4`, but the fused dequant+dot reorders the f32 summation
//! and fuses the multiply-add, so results differ by a few bf16 ULP.
//! `dsv4_expert_mmap.rs` validates that tolerance plus the exact-greedy tokens.
//! (Assumes `in_dim % 32 == 0`, guaranteed by the int4 group=32 packing.)

use std::fs::File;
use std::path::Path;

use half::bf16;
use memmap2::Mmap;

use super::loader::LoadError;
use super::math::to_bf16;

/// Minimal FFI for the Win32 page-cache primitives used by [`MmapExpert::pin`]
/// and [`MmapExpert::prefetch`] — the counterparts of `mlock`/`madvise(WILLNEED)`.
/// Declared directly (kernel32 exports these) to keep the shim dependency-free.
#[cfg(windows)]
mod winmem {
    use core::ffi::c_void;

    /// `WIN32_MEMORY_RANGE_ENTRY` — a virtual-address range for `PrefetchVirtualMemory`.
    #[repr(C)]
    pub struct Win32MemoryRangeEntry {
        pub virtual_address: *mut c_void,
        pub number_of_bytes: usize,
    }

    /// `PSAPI_WORKING_SET_EX_INFORMATION` — one queried virtual address plus the
    /// returned attributes block. `Valid` is bit 0 of `virtual_attributes`.
    #[repr(C)]
    pub struct WorkingSetExInfo {
        pub virtual_address: *mut c_void,
        pub virtual_attributes: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        /// `BOOL VirtualLock(LPVOID lpAddress, SIZE_T dwSize)` — nonzero on success.
        pub fn VirtualLock(address: *const c_void, size: usize) -> i32;
        /// `BOOL K32EmptyWorkingSet(HANDLE)` — moves every removable page of the
        /// process working set to the standby list. `VirtualLock`'d pages and the
        /// hard working-set minimum are respected; standby pages soft-fault back
        /// (no disk I/O) on next touch.
        pub fn K32EmptyWorkingSet(process: isize) -> i32;
        /// `HANDLE GetCurrentProcess(void)` — the current-process pseudo handle.
        pub fn GetCurrentProcess() -> isize;
        /// `BOOL PrefetchVirtualMemory(HANDLE, ULONG_PTR NumberOfEntries,
        /// PWIN32_MEMORY_RANGE_ENTRY, ULONG Flags)` — async read-ahead hint.
        pub fn PrefetchVirtualMemory(
            process: isize,
            number_of_entries: usize,
            addresses: *const Win32MemoryRangeEntry,
            flags: u32,
        ) -> i32;
        /// `BOOL QueryWorkingSetEx(HANDLE, PVOID buffer, DWORD size)` — fills the
        /// `virtual_attributes` of each entry; bit 0 (`Valid`) = page resident.
        pub fn QueryWorkingSetEx(process: isize, buffer: *mut c_void, size: u32) -> i32;
        /// `BOOL SetProcessWorkingSetSizeEx(HANDLE, SIZE_T min, SIZE_T max, DWORD flags)`
        /// — sets the working-set quota. The lockable-page count is bounded by the
        /// minimum, so this must raise it before large `VirtualLock`s.
        pub fn SetProcessWorkingSetSizeEx(
            process: isize,
            min: usize,
            max: usize,
            flags: u32,
        ) -> i32;
    }

    /// `QUOTA_LIMITS_HARDWS_MIN_ENABLE` — enforce the minimum working set (so the
    /// locked pages actually get the quota).
    pub const QUOTA_LIMITS_HARDWS_MIN_ENABLE: u32 = 0x0000_0001;
}

/// Raise this process's minimum working set so `VirtualLock` can pin up to
/// `bytes` of pages. Windows caps lockable pages at (minimum working set −
/// overhead), which defaults to a few MB, so without this every large expert pin
/// fails with `ERROR_WORKING_SET_QUOTA` and nothing stays resident. No-op off
/// Windows. Best-effort: logs on failure, callers still fall back to page cache.
pub fn reserve_lockable(bytes: usize) {
    #[cfg(windows)]
    {
        let min = bytes.saturating_add(32 * 1024 * 1024); // overhead margin
        let max = min.saturating_add(min / 4);
        // SAFETY: pseudo-handle from GetCurrentProcess; the call only adjusts this
        // process's own working-set quota.
        let ok = unsafe {
            winmem::SetProcessWorkingSetSizeEx(
                winmem::GetCurrentProcess(),
                min,
                max,
                winmem::QUOTA_LIMITS_HARDWS_MIN_ENABLE,
            )
        };
        if ok == 0 {
            eprintln!(
                "[glm5] SetProcessWorkingSetSizeEx({min}) failed: {} — hot-expert pins may not stick",
                std::io::Error::last_os_error()
            );
        } else {
            RESERVED_WS_MIN.store(min, std::sync::atomic::Ordering::Relaxed);
        }
    }
    #[cfg(not(windows))]
    {
        let _ = bytes;
    }
}

/// The hard working-set minimum applied by [`reserve_lockable`] (bytes; 0 if
/// none). [`trim_working_set`] re-applies it after an `EmptyWorkingSet`, which
/// resets the quota to defaults on some Windows builds.
#[cfg(windows)]
static RESERVED_WS_MIN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Trim every removable page of this process's working set to the standby
/// list, then re-apply the pinned floor. `VirtualLock`'d pages stay resident;
/// trimmed pages stay in RAM (standby counts toward `MemAvailable`) and
/// soft-fault back without disk I/O on next touch.
///
/// Why: the mmap expert path grows the working set far past the pinned floor,
/// and Windows trims lazily — under-pressure nodes sit at a few hundred MB of
/// `MemAvailable` while holding gigabytes of clean, re-readable pages. The
/// scheduler's memory rule (correctly) refuses to route to such a node. Moving
/// the excess to standby makes the metric — and the node — schedulable again.
/// No-op (returns `false`) off Windows.
pub fn trim_working_set() -> bool {
    #[cfg(windows)]
    {
        // SAFETY: pseudo-handle; both calls only affect this process.
        let ok = unsafe { winmem::K32EmptyWorkingSet(winmem::GetCurrentProcess()) };
        let min = RESERVED_WS_MIN.load(std::sync::atomic::Ordering::Relaxed);
        if min > 0 {
            let max = min.saturating_add(min / 4);
            // SAFETY: as above.
            unsafe {
                winmem::SetProcessWorkingSetSizeEx(
                    winmem::GetCurrentProcess(),
                    min,
                    max,
                    winmem::QUOTA_LIMITS_HARDWS_MIN_ENABLE,
                );
            }
        }
        ok != 0
    }
    #[cfg(not(windows))]
    {
        false
    }
}

const G: usize = 32; // int4 quant group (columns per bf16 scale)

/// Byte size of one packed `[out, in]` section: nibbles then scales.
fn section_bytes(out_dim: usize, in_dim: usize) -> usize {
    out_dim * in_dim / 2 + out_dim * (in_dim / G) * 2
}

/// One expert's int4_bin file, mmap'd. Layout (exporter contract):
/// w1 (gate) `[inter, dim]`, w3 (up) `[inter, dim]`, w2 (down) `[dim, inter]`,
/// each as packed nibbles followed by bf16-LE per-32 scales.
pub struct MmapExpert {
    mmap: Mmap,
    path: std::path::PathBuf,
    dim: usize,
    pub inter: usize,
}

impl MmapExpert {
    pub fn open(path: &Path, dim: usize, inter: usize) -> Result<Self, LoadError> {
        let f = File::open(path)?;
        let len = f.metadata()?.len() as usize;
        let want = 2 * section_bytes(inter, dim) + section_bytes(dim, inter);
        if len < want {
            return Err(LoadError::ExpertBin(path.display().to_string(), len));
        }
        let mmap = unsafe { Mmap::map(&f)? };
        Ok(Self {
            mmap,
            path: path.to_path_buf(),
            dim,
            inter,
        })
    }

    /// Read this expert's whole bin into an owned buffer in ONE bulk sequential
    /// read (light-R1). Called concurrently across a layer's routed experts so
    /// the reads happen up-front, off the compute threads, at full sequential
    /// bandwidth — instead of faulting mmap pages in mid-GEMV. The returned bytes
    /// are byte-identical to `self.mmap`, so [`Self::swiglu_from`] is bit-exact
    /// vs the mmap path.
    ///
    /// If the file has shrunk since [`Self::open`] validated its length (e.g. the
    /// bin was replaced/truncated under a running node), `std::fs::read` still
    /// returns `Ok` with a short buffer — which would slice out of bounds inside
    /// [`Self::swiglu_from`]. Re-check the length here and return `Err` on a short
    /// read so callers route through their mmap fallback instead of panicking.
    pub fn read_bytes(&self) -> std::io::Result<Vec<u8>> {
        let buf = std::fs::read(&self.path)?;
        let want = 2 * section_bytes(self.inter, self.dim) + section_bytes(self.dim, self.inter);
        if buf.len() < want {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "{}: expert bin shrank to {} bytes (need {want})",
                    self.path.display(),
                    buf.len()
                ),
            ));
        }
        Ok(buf)
    }

    /// On-disk size of this expert's int4 bin, i.e. the bytes streamed for it at
    /// 0% cache hit. Used by the decode profiler's residency accounting.
    #[inline]
    pub fn bin_len(&self) -> usize {
        self.mmap.len()
    }

    /// This expert's on-disk int4 bin path — for the lookahead prefetch worker to
    /// warm its pages into the OS cache off the compute thread.
    pub fn bin_path(&self) -> &Path {
        &self.path
    }

    /// Whether this expert's bin is (almost) entirely resident in RAM right
    /// now: at least 90% of 64 pages sampled across it are in core. A resident
    /// expert is cheaper to compute straight off the mapping than to copy
    /// first; a paged-out one streams faster as one bulk sequential read
    /// ([`Self::read_bytes`]). `false` when the OS query is unavailable, so
    /// the caller keeps the streaming path.
    pub fn mostly_resident(&self) -> bool {
        let (resident, probed) = self.resident_pages_sampled(64);
        probed > 0 && resident * 10 >= probed * 9
    }

    /// Estimate how many of this expert's mapped pages are resident in RAM right
    /// now, by probing `samples` pages spread evenly across the bin. Returns
    /// `(resident, probed)`; `(0, 0)` if the OS query is unavailable or fails.
    ///
    /// `QueryWorkingSetEx` on Windows probes this mapping's process working set,
    /// not the complete system file cache. A buffered `read_bytes` can leave the
    /// mapping invalid even while its backing pages are cached in RAM; a later
    /// mmap access can then soft-fault without disk I/O. Treat this as a lower
    /// bound on cached bytes, not a true file-cache miss count. Unix uses mincore.
    /// Best-effort and read-only; it never faults pages in to bias the sample.
    pub fn resident_pages_sampled(&self, samples: usize) -> (usize, usize) {
        const PAGE: usize = 4096;
        let len = self.mmap.len();
        if len == 0 || samples == 0 {
            return (0, 0);
        }
        let base = self.mmap.as_ptr() as usize;
        let npages = len.div_ceil(PAGE);
        let want = samples.min(npages);
        let step = (npages / want).max(1);
        let addrs: Vec<usize> = (0..npages)
            .step_by(step)
            .take(want)
            .map(|p| base + p * PAGE)
            .collect();

        #[cfg(windows)]
        {
            use core::ffi::c_void;
            let mut buf: Vec<winmem::WorkingSetExInfo> = addrs
                .iter()
                .map(|&a| winmem::WorkingSetExInfo {
                    virtual_address: a as *mut c_void,
                    virtual_attributes: 0,
                })
                .collect();
            let bytes = (buf.len() * core::mem::size_of::<winmem::WorkingSetExInfo>()) as u32;
            // SAFETY: `buf` is a valid array of `buf.len()` entries; QueryWorkingSetEx
            // writes only the `virtual_attributes` field of each. Read-only otherwise.
            let ok = unsafe {
                winmem::QueryWorkingSetEx(
                    winmem::GetCurrentProcess(),
                    buf.as_mut_ptr() as *mut c_void,
                    bytes,
                )
            };
            if ok == 0 {
                return (0, 0);
            }
            let res = buf.iter().filter(|e| e.virtual_attributes & 1 == 1).count();
            (res, buf.len())
        }
        #[cfg(unix)]
        {
            use core::ffi::c_void;
            extern "C" {
                fn mincore(addr: *mut c_void, length: usize, vec: *mut u8) -> i32;
            }
            let (mut resident, mut probed) = (0usize, 0usize);
            for &a in &addrs {
                let mut v = [0u8; 1];
                // SAFETY: [a, a+PAGE) lies within the mapped region (npages pages
                // from base); mincore writes one residency byte into `v`.
                let r = unsafe { mincore(a as *mut c_void, PAGE, v.as_mut_ptr()) };
                if r == 0 {
                    probed += 1;
                    if v[0] & 1 == 1 {
                        resident += 1;
                    }
                }
            }
            (resident, probed)
        }
        #[cfg(not(any(unix, windows)))]
        {
            (0, 0)
        }
    }

    /// SwiGLU FFN over an explicitly-read byte buffer (the R1 path). Mirrors
    /// `crate::glm::ffn::swiglu_mmap` exactly, but reads weights from `data`
    /// (whole-expert buffer) rather than the mmap.
    pub fn swiglu_from(&self, data: &[u8], x: &[f32]) -> Vec<f32> {
        let (inter, dim) = (self.inter, self.dim);
        let mut h = vec![0.0f32; inter];
        self.gemv_on(data, 0, inter, dim, x, &mut h);
        let mut u = vec![0.0f32; inter];
        self.gemv_on(data, section_bytes(inter, dim), inter, dim, x, &mut u);
        for (hi, &ui) in h.iter_mut().zip(&u) {
            *hi = (*hi / (1.0 + (-*hi).exp())) * ui; // silu(gate)*up — matches ffn::swiglu_mmap
        }
        let mut out = vec![0.0f32; dim];
        self.gemv_on(
            data,
            2 * section_bytes(inter, dim),
            dim,
            inter,
            &h,
            &mut out,
        );
        out
    }

    /// y = W x with W dequantized row-by-row; y[o] rounded to bf16 exactly
    /// like `linear_bf16` over an eagerly-dequantized W.
    ///
    /// Output rows are independent: dequant + dot each on its own core (rayon),
    /// with a per-row scratch buffer. Bit-identical to the sequential version
    /// (same per-row accumulation order), just spread across the CPU — this is
    /// the real-model MoE hot path (256 experts, mmap int4).
    fn gemv(&self, sec_off: usize, out_dim: usize, in_dim: usize, x: &[f32], y: &mut [f32]) {
        self.gemv_on(&self.mmap, sec_off, out_dim, in_dim, x, y);
    }

    /// `gemv` over an arbitrary byte source (`self.mmap` or an R1 read buffer).
    /// The two are byte-identical, so the result is bitwise the same either way.
    fn gemv_on(
        &self,
        data: &[u8],
        sec_off: usize,
        out_dim: usize,
        in_dim: usize,
        x: &[f32],
        y: &mut [f32],
    ) {
        use rayon::prelude::*;
        debug_assert_eq!(x.len(), in_dim);
        debug_assert_eq!(y.len(), out_dim);
        let ng = in_dim / G;
        let packed = &data[sec_off..sec_off + out_dim * in_dim / 2];
        let scales = &data
            [sec_off + out_dim * in_dim / 2..sec_off + out_dim * in_dim / 2 + out_dim * ng * 2];
        let row_bytes = in_dim / 2;
        // Each output row is an independent fused dequant+dot: the int4 nibbles
        // are unpacked straight into the FMA against x (no f32 scratch row, no
        // scalar unpack), rayon across rows. Same value as dequant-then-dot,
        // modulo f32 summation order (see `dequant_row_dot`).
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && !(is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl"))
        {
            use std::sync::OnceLock;
            static ROWS: OnceLock<usize> = OnceLock::new();
            let rows = *ROWS.get_or_init(|| {
                std::env::var("CASCADIA_INT4_GEMV_ROWS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|r| matches!(r, 1 | 2 | 4))
                    .unwrap_or(1)
            });
            match rows {
                2 => {
                    gemv_tiled_avx2::<2>(packed, scales, x, y);
                    return;
                }
                4 => {
                    gemv_tiled_avx2::<4>(packed, scales, x, y);
                    return;
                }
                _ => {}
            }
        }
        y.par_iter_mut().enumerate().for_each(|(o, yy)| {
            let prow = &packed[o * row_bytes..(o + 1) * row_bytes];
            let srow = &scales[o * ng * 2..(o + 1) * ng * 2];
            *yy = to_bf16(dequant_row_dot(prow, srow, x, in_dim));
        });
    }

    /// Mirror of `Expert::forward`: silu(clamp(w1 x)) * clamp(w3 x)
    /// [* route_w] -> w2, with the same bf16 rounding points.
    pub fn forward(&self, x: &[f32], dim: usize, limit: f32, route_w: Option<f32>) -> Vec<f32> {
        debug_assert_eq!(dim, self.dim);
        let inter = self.inter;
        let w1_off = 0;
        let w3_off = section_bytes(inter, dim);
        let w2_off = 2 * section_bytes(inter, dim);
        let mut gate = vec![0.0f32; inter];
        let mut up = vec![0.0f32; inter];
        self.gemv(w1_off, inter, dim, x, &mut gate);
        self.gemv(w3_off, inter, dim, x, &mut up);
        let mut h = vec![0.0f32; inter];
        for i in 0..inter {
            let mut g = gate[i];
            let mut u = up[i];
            if limit > 0.0 {
                u = u.clamp(-limit, limit);
                g = g.min(limit);
            }
            let s = g / (1.0 + (-g).exp()); // silu
            let mut v = s * u;
            if let Some(w) = route_w {
                v *= w;
            }
            h[i] = to_bf16(v);
        }
        let mut out = vec![0.0f32; dim];
        self.gemv(w2_off, dim, inter, &h, &mut out);
        out
    }

    /// `mlock` the mapped range so the OS never evicts it (hot-expert pinning).
    /// Best-effort: returns the error (e.g. `RLIMIT_MEMLOCK` exceeded) so the
    /// caller can fall back to the OS page cache for this expert.
    #[cfg(unix)]
    pub fn pin(&self) -> std::io::Result<()> {
        self.mmap.lock()
    }

    /// Windows equivalent of `mlock`: `VirtualLock` pins the mapped range in the
    /// process working set so it is not paged out. Bounded by the working set
    /// (the analogue of `RLIMIT_MEMLOCK`); on failure returns the OS error so the
    /// caller falls back to the page cache — same best-effort contract as Unix.
    #[cfg(windows)]
    pub fn pin(&self) -> std::io::Result<()> {
        // SAFETY: `as_ptr()`/`len()` describe the live, valid mmap range for the
        // lifetime of `self`; VirtualLock only reads those bounds.
        let ok = unsafe { winmem::VirtualLock(self.mmap.as_ptr().cast(), self.mmap.len()) };
        if ok == 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    /// Any other target (no page-locking primitive): no-op — callers fall back to
    /// the OS page cache.
    #[cfg(not(any(unix, windows)))]
    pub fn pin(&self) -> std::io::Result<()> {
        Ok(())
    }

    /// `madvise(WILLNEED)` — ask the OS to start reading this expert's pages into
    /// the cache without blocking. Issued for the *next* layer's likely experts
    /// while the current layer computes, so the NVMe read overlaps compute
    /// instead of stalling the GEMV. Best-effort hint; a failure is ignored.
    #[cfg(unix)]
    #[inline]
    pub fn prefetch(&self) {
        let _ = self.mmap.advise(memmap2::Advice::WillNeed);
    }

    /// Windows equivalent of `madvise(WILLNEED)`: `PrefetchVirtualMemory` asks the
    /// memory manager to read the range into RAM asynchronously. Best-effort; the
    /// result is ignored (same contract as the Unix hint).
    #[cfg(windows)]
    #[inline]
    pub fn prefetch(&self) {
        let entry = winmem::Win32MemoryRangeEntry {
            virtual_address: self.mmap.as_ptr() as *mut core::ffi::c_void,
            number_of_bytes: self.mmap.len(),
        };
        // SAFETY: `entry` describes the live mmap range; PrefetchVirtualMemory
        // only reads it and issues an async read-ahead hint.
        unsafe {
            let _ = winmem::PrefetchVirtualMemory(winmem::GetCurrentProcess(), 1, &entry, 0);
        }
    }

    /// Any other target: no prefetch hint available — no-op.
    #[cfg(not(any(unix, windows)))]
    #[inline]
    pub fn prefetch(&self) {}

    /// SwiGLU section GEMVs exposed for shells with a different activation
    /// contract (the glm5 shell applies f32 `silu·up`, no clamp, route outside).
    /// Each returns the bf16-rounded fused int4 dequant-dot (same kernel as
    /// [`Self::forward`]). `dim`/`inter` are the ones passed to `open`.
    pub fn gemv_gate(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; self.inter];
        self.gemv(0, self.inter, self.dim, x, &mut y);
        y
    }
    pub fn gemv_up(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; self.inter];
        self.gemv(
            section_bytes(self.inter, self.dim),
            self.inter,
            self.dim,
            x,
            &mut y,
        );
        y
    }
    pub fn gemv_down(&self, h: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; self.dim];
        self.gemv(
            2 * section_bytes(self.inter, self.dim),
            self.dim,
            self.inter,
            h,
            &mut y,
        );
        y
    }
}

/// Fused int4 dequant + dot for one output row: `Σ_k (nibble_k - 8) * scale(g(k)) * x[k]`,
/// where `packed_row` is `in_dim/2` nibble bytes (low nibble = even col, high = odd,
/// interleaved per byte) and `scales_row` is one bf16-LE scale per 32-column group.
/// AVX2+FMA on x86_64 (runtime-detected); scalar fallback otherwise. Returns the raw
/// f32 dot; the caller rounds to bf16 (matching `Expert::forward`).
#[inline]
fn dequant_row_dot(packed_row: &[u8], scales_row: &[u8], x: &[f32], in_dim: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 (16-wide) where available — ~2x the AVX2 lane count on the
        // Xeon export host + AVX-512 nodes; the Lunar Lake AI-PCs have no
        // AVX-512 and fall through to the AVX2 path.
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: avx512{f,bw,vl} detected; loads stay within packed_row
            // (in_dim/2 B), scales_row (in_dim/G*2 B) and x (in_dim).
            return unsafe { dequant_row_dot_avx512(packed_row, scales_row, x, in_dim) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2+fma detected at runtime; every load stays within
            // packed_row (in_dim/2 bytes), scales_row (in_dim/G*2 bytes) and x (in_dim).
            return unsafe { dequant_row_dot_avx2(packed_row, scales_row, x, in_dim) };
        }
    }
    dequant_row_dot_scalar(packed_row, scales_row, x, in_dim)
}

/// AVX-512 fused dequant+dot — the AVX2 strategy at 512-bit: per 32-col group,
/// decode 16 packed bytes to 32 int8, interleave to column order, sign-extend to
/// 2×16-wide f32, scale, and FMA against x into one 16-wide accumulator. Same
/// value as the scalar/AVX2 paths within f32 lane-order ULP (the caller rounds
/// to bf16).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn dequant_row_dot_avx512(
    packed_row: &[u8],
    scales_row: &[u8],
    x: &[f32],
    in_dim: usize,
) -> f32 {
    use core::arch::x86_64::*;
    let ng = in_dim / G;
    let lo_mask = _mm_set1_epi8(0x0F);
    let bias = _mm_set1_epi8(8);
    let mut acc = _mm512_setzero_ps();
    let xp = x.as_ptr();
    for g in 0..ng {
        let s = half::bf16::from_le_bytes([scales_row[g * 2], scales_row[g * 2 + 1]]).to_f32();
        let sv = _mm512_set1_ps(s);
        let pk = _mm_loadu_si128(packed_row.as_ptr().add(g * (G / 2)) as *const __m128i);
        let low = _mm_and_si128(pk, lo_mask);
        let high = _mm_and_si128(_mm_srli_epi16::<4>(pk), lo_mask);
        let low_s = _mm_sub_epi8(low, bias);
        let high_s = _mm_sub_epi8(high, bias);
        // low/high nibble of byte i = cols 2i, 2i+1 -> interleave to column order.
        let il = _mm_unpacklo_epi8(low_s, high_s); // cols 0..15  (16 int8)
        let ih = _mm_unpackhi_epi8(low_s, high_s); // cols 16..31
        let c0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(il)); // 16 f32
        let c1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(ih));
        let base = g * G;
        let x0 = _mm512_loadu_ps(xp.add(base));
        let x1 = _mm512_loadu_ps(xp.add(base + 16));
        acc = _mm512_fmadd_ps(_mm512_mul_ps(c0, sv), x0, acc);
        acc = _mm512_fmadd_ps(_mm512_mul_ps(c1, sv), x1, acc);
    }
    _mm512_reduce_add_ps(acc)
}

/// Scalar reference for `dequant_row_dot` (non-x86 / no-AVX2). Same nibble decode
/// as `loader::dequant_int4`.
fn dequant_row_dot_scalar(packed_row: &[u8], scales_row: &[u8], x: &[f32], in_dim: usize) -> f32 {
    let ng = in_dim / G;
    let mut acc = 0.0f32;
    for g in 0..ng {
        let s = bf16::from_le_bytes([scales_row[g * 2], scales_row[g * 2 + 1]]).to_f32();
        for i in 0..G / 2 {
            let byte = packed_row[g * (G / 2) + i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            acc += (lo as f32 * s) * x[g * G + 2 * i];
            acc += (hi as f32 * s) * x[g * G + 2 * i + 1];
        }
    }
    acc
}

/// AVX2+FMA fused dequant+dot. Ports `cascadia_int4_gemm::kernel_avx512`'s strategy
/// to 256-bit lanes: load 16 packed bytes (one 32-col group), split lo/hi nibbles,
/// subtract 8, interleave to column order, sign-extend i8→i32→f32, scale, and FMA
/// against x into an 8-wide accumulator; horizontal-sum at the end.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_row_dot_avx2(
    packed_row: &[u8],
    scales_row: &[u8],
    x: &[f32],
    in_dim: usize,
) -> f32 {
    use core::arch::x86_64::*;
    let ng = in_dim / G;
    let lo_mask = _mm_set1_epi8(0x0F);
    let bias = _mm_set1_epi8(8);
    let mut acc = _mm256_setzero_ps();
    let xp = x.as_ptr();
    for g in 0..ng {
        // NB: `use core::arch::x86_64::*` brings an intrinsic `bf16` into scope,
        // so qualify the half crate's type explicitly.
        let s = half::bf16::from_le_bytes([scales_row[g * 2], scales_row[g * 2 + 1]]).to_f32();
        let sv = _mm256_set1_ps(s);
        let pk = _mm_loadu_si128(packed_row.as_ptr().add(g * (G / 2)) as *const __m128i);
        let low = _mm_and_si128(pk, lo_mask);
        let high = _mm_and_si128(_mm_srli_epi16::<4>(pk), lo_mask);
        let low_s = _mm_sub_epi8(low, bias);
        let high_s = _mm_sub_epi8(high, bias);
        // interleave to [col0, col1, ...]: low/high nibble of byte i = cols 2i, 2i+1.
        let il = _mm_unpacklo_epi8(low_s, high_s); // cols 0..15
        let ih = _mm_unpackhi_epi8(low_s, high_s); // cols 16..31
        let c0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(il));
        let c1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(il)));
        let c2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ih));
        let c3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(ih)));
        let base = g * G;
        let x0 = _mm256_loadu_ps(xp.add(base));
        let x1 = _mm256_loadu_ps(xp.add(base + 8));
        let x2 = _mm256_loadu_ps(xp.add(base + 16));
        let x3 = _mm256_loadu_ps(xp.add(base + 24));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(c0, sv), x0, acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(c1, sv), x1, acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(c2, sv), x2, acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(c3, sv), x3, acc);
    }
    // horizontal sum of the 8 lanes
    let lo128 = _mm256_castps256_ps128(acc);
    let hi128 = _mm256_extractf128_ps::<1>(acc);
    let s128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

// ───────────────────── multi-input (row-batched) int4 GEMM ─────────────────────
//
// `gemv_on` streams a whole section (9.4 MB of nibbles + 1.2 MB of scales at
// the real dims) through the memory bus for ONE activation. When a block of
// rows shares an expert (multi-stream decode, prefill) that is `n` full passes
// over the same bytes. The kernels below decode each weight group ONCE and
// feed it to all `n` activations, so the expert's bytes cross the bus once per
// call.
//
// Bit-identity contract: for every (weight row, input) pair the arithmetic is
// the single-input kernel's, operation for operation — accumulator starts at
// +0, groups ascend, the same `c·scale` product feeds the same FMA chain
// (AVX2: 4 × 8-lane, AVX-512: 2 × 16-lane, scalar: lo then hi nibble), the
// same horizontal reduction, the same `to_bf16`. Only the loop nest around the
// pair changes (inputs inside weight groups inside a tile of weight rows), and
// no operation mixes two pairs, so the f32 bits are the single-input bits.
// `tests::row_gemm` pins that on the RAW dot (before the bf16 rounding, which
// would mask most reorderings) for each ISA.

/// Which fused dequant-dot kernel runs. The three differ in lane structure and
/// so in f32 bits; a multi-input kernel has to mirror the active one.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Int4Isa {
    Scalar,
    Avx2,
    Avx512,
}

impl Int4Isa {
    /// The kernel [`dequant_row_dot`] — and so `gemv_on` — dispatches to on
    /// this CPU. (`CASCADIA_INT4_GEMV_ROWS` tiling is the AVX2 kernel's bits:
    /// `tiled_rows_preserve_avx2_bits_across_scales_and_odd_row_counts`.)
    pub fn active() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
            {
                return Self::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Self::Avx2;
            }
        }
        Self::Scalar
    }

    /// Whether this CPU can run the kernel (forcing an unsupported one would
    /// be an illegal instruction).
    pub fn supported(self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => {
                is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512vl")
            }
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }
}

/// Weight rows per GEMM tile (`CASCADIA_INT4_GEMM_ROWS` = 1|2|4|8, default 8).
/// A tile's rows share one sweep over the inputs, so the inputs come out of L2
/// once per tile instead of once per weight row, and `rows × n` independent FMA
/// chains keep the FMA ports busy where one chain is latency-bound. A schedule
/// knob only: every setting produces the same bits.
fn gemm_tile_rows() -> usize {
    use std::sync::OnceLock;
    static ROWS: OnceLock<usize> = OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("CASCADIA_INT4_GEMM_ROWS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|r| matches!(r, 1 | 2 | 4 | 8))
            .unwrap_or(GEMM_TILE_ROWS)
    })
}

const GEMM_TILE_ROWS: usize = 8;

/// Ceiling on one tile's accumulator scratch (see `gemm_section`).
const GEMM_ACC_BYTES: usize = 16 * 1024;

/// How far ahead of the group being decoded the packed-row prefetch reaches.
#[cfg(target_arch = "x86_64")]
const GEMM_PREFETCH: usize = 512;

/// Pull every tile row's nibble stream (one cache line = 4 groups) and scale
/// stream (one line = 32 groups) ahead of group `g` by hand: a tile interleaves
/// `2 × rows` slow streams, which the hardware prefetcher loses track of —
/// measured, a tile of 8 only beats a tile of 2 with this. Hints only (no
/// fault, no effect on values); `wrapping_add` because the last ones point past
/// the tile.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn gemm_prefetch(packed: &[u8], scales: &[u8], rows: usize, (rb, sb): (usize, usize), g: usize) {
    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
    if !g.is_multiple_of(4) {
        return;
    }
    for r in 0..rows {
        let nibbles = packed
            .as_ptr()
            .wrapping_add(r * rb + g * (G / 2) + GEMM_PREFETCH);
        // SAFETY: prefetch never dereferences; any address is allowed.
        unsafe { _mm_prefetch::<_MM_HINT_T0>(nibbles as *const i8) };
        if g.is_multiple_of(32) {
            let scale = scales.as_ptr().wrapping_add(r * sb + g * 2 + 64);
            // SAFETY: as above.
            unsafe { _mm_prefetch::<_MM_HINT_T0>(scale as *const i8) };
        }
    }
}

/// Most inputs one GEMM pass carries; a larger block (prefill's 128-row blocks)
/// is split into balanced passes. Bounds the scratch (`n × in_dim` f32 of
/// group-major inputs — 1.5 MB at 64 × 6144 — plus the `[out][n]` outputs) and
/// keeps the input block near the per-core L2. Measured on a 1 MB-L2 Xeon: 64
/// per pass 0.22 ms/row, 32 → 0.24, 16 → 0.26 — by then the weight stream is
/// amortized 64× anyway. A multi-stream decode frame (≤ 64 rows) is one pass.
pub const GEMM_MAX_INPUTS: usize = 64;

/// One AVX2 accumulator (8 lanes), 32-byte aligned so a store never straddles
/// a cache line.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Acc8([f32; 8]);

/// One AVX-512 accumulator (16 lanes), cache-line aligned.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Acc16([f32; 16]);

/// `n` inputs of `in_dim` re-laid GROUP-major: `out[(g·n + j)·G + c] =
/// xs[j][g·G + c]`. For one weight group the kernel then walks the `n` inputs'
/// 32 columns back to back (one sequential stream instead of `n` strided
/// ones). Pure copies, so the values the FMAs see are the callers' bits.
fn group_major(xs: &[&[f32]], in_dim: usize) -> Vec<f32> {
    let n = xs.len();
    let mut out = vec![0.0f32; n * in_dim];
    for (j, x) in xs.iter().enumerate() {
        assert_eq!(x.len(), in_dim, "int4 gemm: input {j} length");
        for (g, grp) in x.chunks_exact(G).enumerate() {
            out[(g * n + j) * G..(g * n + j + 1) * G].copy_from_slice(grp);
        }
    }
    out
}

/// [`group_major`] from a `[in_dim][n]` matrix (`cols[c·n + j]`, the layout a
/// GEMM writes) — the gate/up product feeding the down projection.
fn group_major_from_cols(cols: &[f32], n: usize, in_dim: usize) -> Vec<f32> {
    debug_assert_eq!(cols.len(), n * in_dim);
    let mut out = vec![0.0f32; n * in_dim];
    for g in 0..in_dim / G {
        for c in 0..G {
            let src = &cols[(g * G + c) * n..(g * G + c + 1) * n];
            for (j, &v) in src.iter().enumerate() {
                out[(g * n + j) * G + c] = v;
            }
        }
    }
    out
}

/// `Y = W·X` for `n` inputs over one packed section. `xg` is the group-major
/// input block ([`group_major`]); `yt` is `[out_dim][n]` (`yt[o·n + j]` = row
/// `o` · input `j`). Rayon over tiles of weight rows — each task carries ALL
/// `n` inputs through its rows, so the split never touches a (row, input)
/// pair's arithmetic. `round` applies the GEMV's `to_bf16` (off only in the
/// raw-dot tests).
#[allow(clippy::too_many_arguments)]
fn gemm_section(
    packed: &[u8],
    scales: &[u8],
    in_dim: usize,
    xg: &[f32],
    n: usize,
    yt: &mut [f32],
    isa: Int4Isa,
    tile: usize,
    round: bool,
) {
    assert!(
        n > 0 && tile > 0 && in_dim.is_multiple_of(G),
        "int4 gemm: shape"
    );
    assert!(isa.supported(), "int4 gemm: {isa:?} not supported here");
    let out_dim = yt.len() / n;
    let (rb, sb) = (in_dim / 2, in_dim / G * 2);
    assert_eq!(yt.len(), out_dim * n, "int4 gemm: output length");
    assert_eq!(xg.len(), n * in_dim, "int4 gemm: input length");
    assert!(
        packed.len() >= out_dim * rb && scales.len() >= out_dim * sb,
        "int4 gemm: section length"
    );
    // Keep a tile's accumulators (`tile · n` of them) well inside L1 next to
    // the group's input window: at 64 inputs a tile of 8 AVX-512 accumulators
    // is 32 KB — a whole L1D — and measured 1.5× slower per row than at 32.
    let lane_bytes = match isa {
        Int4Isa::Avx512 => 64,
        Int4Isa::Avx2 => 32,
        Int4Isa::Scalar => 4,
    };
    let mut tile = tile;
    while tile > 2 && tile * n * lane_bytes > GEMM_ACC_BYTES {
        tile /= 2;
    }
    #[cfg(target_arch = "x86_64")]
    if isa == Int4Isa::Avx512 {
        return gemm_drive(
            packed,
            scales,
            (rb, sb),
            n,
            yt,
            tile,
            round,
            Acc16([0.0; 16]),
            // SAFETY: `isa.supported()` checked avx512{f,bw,vl}; slice bounds
            // are asserted inside the kernel.
            |p, s, rows, acc, y| unsafe { gemm_tile_avx512(p, s, rows, xg, n, in_dim, acc, y) },
        );
    }
    #[cfg(target_arch = "x86_64")]
    if isa == Int4Isa::Avx2 {
        return gemm_drive(
            packed,
            scales,
            (rb, sb),
            n,
            yt,
            tile,
            round,
            Acc8([0.0; 8]),
            // SAFETY: `isa.supported()` checked avx2+fma; bounds asserted inside.
            |p, s, rows, acc, y| unsafe { gemm_tile_avx2(p, s, rows, xg, n, in_dim, acc, y) },
        );
    }
    gemm_drive(
        packed,
        scales,
        (rb, sb),
        n,
        yt,
        tile,
        round,
        0.0f32,
        |p, s, rows, acc, y| gemm_tile_scalar(p, s, rows, xg, n, in_dim, acc, y),
    );
}

/// The rayon split shared by the three kernels: tiles of `tile` weight rows
/// (the last may be short), a per-task accumulator scratch (`tile · n` of `A`).
#[allow(clippy::too_many_arguments)]
fn gemm_drive<A, K>(
    packed: &[u8],
    scales: &[u8],
    (rb, sb): (usize, usize),
    n: usize,
    yt: &mut [f32],
    tile: usize,
    round: bool,
    zero: A,
    kernel: K,
) where
    A: Copy + Send + Sync,
    K: Fn(&[u8], &[u8], usize, &mut [A], &mut [f32]) + Sync,
{
    use rayon::prelude::*;
    yt.par_chunks_mut(n * tile).enumerate().for_each_init(
        || vec![zero; n * tile],
        |acc, (t, out)| {
            let first = t * tile;
            let rows = out.len() / n;
            kernel(
                &packed[first * rb..(first + rows) * rb],
                &scales[first * sb..(first + rows) * sb],
                rows,
                acc,
                out,
            );
            if round {
                for v in out.iter_mut() {
                    *v = to_bf16(*v);
                }
            }
        },
    );
}

/// Scalar tile: `dequant_row_dot_scalar`'s chain per (row, input) — `acc +=
/// (nibble·s)·x[c]`, columns (lo nibble then hi) and groups ascending. The
/// group's 32 `nibble·s` weights are built once for all inputs, and the inputs
/// run four at a time so four independent add chains overlap (one chain is
/// latency-bound); neither touches a pair's operation order.
#[allow(clippy::too_many_arguments)]
fn gemm_tile_scalar(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    xg: &[f32],
    n: usize,
    in_dim: usize,
    acc: &mut [f32],
    y: &mut [f32],
) {
    let ng = in_dim / G;
    let (rb, sb) = (in_dim / 2, ng * 2);
    let acc = &mut acc[..rows * n];
    acc.fill(0.0);
    for g in 0..ng {
        let xgrp = &xg[g * n * G..(g + 1) * n * G];
        for r in 0..rows {
            let si = r * sb + g * 2;
            let s = bf16::from_le_bytes([scales[si], scales[si + 1]]).to_f32();
            let bytes = &packed[r * rb + g * (G / 2)..r * rb + (g + 1) * (G / 2)];
            let mut w = [0.0f32; G];
            for (i, &byte) in bytes.iter().enumerate() {
                w[2 * i] = ((byte & 0x0F) as i32 - 8) as f32 * s;
                w[2 * i + 1] = (((byte >> 4) & 0x0F) as i32 - 8) as f32 * s;
            }
            let mut quads = acc[r * n..(r + 1) * n].chunks_exact_mut(4);
            let mut xquads = xgrp.chunks_exact(4 * G);
            for (a, x) in (&mut quads).zip(&mut xquads) {
                let (mut a0, mut a1, mut a2, mut a3) = (a[0], a[1], a[2], a[3]);
                for (c, &wc) in w.iter().enumerate() {
                    a0 += wc * x[c];
                    a1 += wc * x[G + c];
                    a2 += wc * x[2 * G + c];
                    a3 += wc * x[3 * G + c];
                }
                (a[0], a[1], a[2], a[3]) = (a0, a1, a2, a3);
            }
            let rest = quads.into_remainder();
            for (a, x) in rest.iter_mut().zip(xquads.remainder().chunks_exact(G)) {
                let mut v = *a;
                for (&wc, &xc) in w.iter().zip(x) {
                    v += wc * xc;
                }
                *a = v;
            }
        }
    }
    y[..rows * n].copy_from_slice(acc);
}

/// One weight group's four 8-lane `c·scale` vectors (columns 0..7, 8..15,
/// 16..23, 24..31) — exactly the products `dequant_row_dot_avx2` feeds its FMAs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn group_weights_avx2(packed16: *const u8, scale: f32) -> [core::arch::x86_64::__m256; 4] {
    use core::arch::x86_64::*;
    let lo_mask = _mm_set1_epi8(0x0F);
    let bias = _mm_set1_epi8(8);
    let sv = _mm256_set1_ps(scale);
    let pk = _mm_loadu_si128(packed16 as *const __m128i);
    let low_s = _mm_sub_epi8(_mm_and_si128(pk, lo_mask), bias);
    let high_s = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16::<4>(pk), lo_mask), bias);
    // low/high nibble of byte i = cols 2i, 2i+1 -> interleave to column order.
    let il = _mm_unpacklo_epi8(low_s, high_s); // cols 0..15
    let ih = _mm_unpackhi_epi8(low_s, high_s); // cols 16..31
    [
        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(il)), sv),
        _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(il))),
            sv,
        ),
        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ih)), sv),
        _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(ih))),
            sv,
        ),
    ]
}

/// AVX2 tile: `dequant_row_dot_avx2`'s chain per (row, input) — the group's
/// four `c·scale` vectors are built once and FMA'd against each input's 32
/// columns into that pair's own 8-lane accumulator, then the same horizontal
/// sum. Weight rows go through the input loop two at a time so both share each
/// input's four loads (the loop is load-port-bound otherwise); the two rows'
/// chains never mix. Writes the RAW dots (`y[r·n + j]`); the driver rounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_tile_avx2(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    xg: &[f32],
    n: usize,
    in_dim: usize,
    acc: &mut [Acc8],
    y: &mut [f32],
) {
    use core::arch::x86_64::*;
    let ng = in_dim / G;
    let (rb, sb) = (in_dim / 2, ng * 2);
    assert!(packed.len() >= rows * rb && scales.len() >= rows * sb);
    assert!(xg.len() >= n * in_dim && acc.len() >= rows * n && y.len() >= rows * n);
    let scale = |r: usize, g: usize| {
        let si = r * sb + g * 2;
        half::bf16::from_le_bytes([scales[si], scales[si + 1]]).to_f32()
    };
    let pp = packed.as_ptr();
    let ap = acc.as_mut_ptr() as *mut f32;
    for a in 0..rows * n {
        _mm256_store_ps(ap.add(a * 8), _mm256_setzero_ps());
    }
    for g in 0..ng {
        let xgp = xg.as_ptr().add(g * n * G);
        let mut r = 0;
        gemm_prefetch(packed, scales, rows, (rb, sb), g);
        while r + 2 <= rows {
            let [u0, u1, u2, u3] = group_weights_avx2(pp.add(r * rb + g * (G / 2)), scale(r, g));
            let [v0, v1, v2, v3] =
                group_weights_avx2(pp.add((r + 1) * rb + g * (G / 2)), scale(r + 1, g));
            let (ar, br) = (ap.add(r * n * 8), ap.add((r + 1) * n * 8));
            for j in 0..n {
                let xp = xgp.add(j * G);
                let (x0, x1) = (_mm256_loadu_ps(xp), _mm256_loadu_ps(xp.add(8)));
                let (x2, x3) = (_mm256_loadu_ps(xp.add(16)), _mm256_loadu_ps(xp.add(24)));
                let mut a = _mm256_load_ps(ar.add(j * 8));
                a = _mm256_fmadd_ps(u0, x0, a);
                a = _mm256_fmadd_ps(u1, x1, a);
                a = _mm256_fmadd_ps(u2, x2, a);
                a = _mm256_fmadd_ps(u3, x3, a);
                _mm256_store_ps(ar.add(j * 8), a);
                let mut b = _mm256_load_ps(br.add(j * 8));
                b = _mm256_fmadd_ps(v0, x0, b);
                b = _mm256_fmadd_ps(v1, x1, b);
                b = _mm256_fmadd_ps(v2, x2, b);
                b = _mm256_fmadd_ps(v3, x3, b);
                _mm256_store_ps(br.add(j * 8), b);
            }
            r += 2;
        }
        if r < rows {
            let [w0, w1, w2, w3] = group_weights_avx2(pp.add(r * rb + g * (G / 2)), scale(r, g));
            let ar = ap.add(r * n * 8);
            for j in 0..n {
                let xp = xgp.add(j * G);
                let mut a = _mm256_load_ps(ar.add(j * 8));
                a = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp), a);
                a = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp.add(8)), a);
                a = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp.add(16)), a);
                a = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp.add(24)), a);
                _mm256_store_ps(ar.add(j * 8), a);
            }
        }
    }
    for (i, value) in y[..rows * n].iter_mut().enumerate() {
        // horizontal sum of the 8 lanes — `dequant_row_dot_avx2`'s order
        let a = _mm256_load_ps(ap.add(i * 8));
        let s128 = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
        let shuf = _mm_movehdup_ps(s128);
        let sums = _mm_add_ps(s128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        *value = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
    }
}

/// One weight group's two 16-lane `c·scale` vectors (columns 0..15, 16..31) —
/// the products `dequant_row_dot_avx512` feeds its FMAs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn group_weights_avx512(packed16: *const u8, scale: f32) -> [core::arch::x86_64::__m512; 2] {
    use core::arch::x86_64::*;
    let lo_mask = _mm_set1_epi8(0x0F);
    let bias = _mm_set1_epi8(8);
    let sv = _mm512_set1_ps(scale);
    let pk = _mm_loadu_si128(packed16 as *const __m128i);
    let low_s = _mm_sub_epi8(_mm_and_si128(pk, lo_mask), bias);
    let high_s = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16::<4>(pk), lo_mask), bias);
    let il = _mm_unpacklo_epi8(low_s, high_s); // cols 0..15
    let ih = _mm_unpackhi_epi8(low_s, high_s); // cols 16..31
    [
        _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(il)), sv),
        _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(ih)), sv),
    ]
}

/// AVX-512 tile: `dequant_row_dot_avx512`'s chain per (row, input) — two
/// 16-lane `c·scale` vectors per group, FMA'd in column order into that pair's
/// own accumulator, then the same `_mm512_reduce_add_ps`. Weight rows pair up
/// through the input loop like the AVX2 tile.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_tile_avx512(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    xg: &[f32],
    n: usize,
    in_dim: usize,
    acc: &mut [Acc16],
    y: &mut [f32],
) {
    use core::arch::x86_64::*;
    let ng = in_dim / G;
    let (rb, sb) = (in_dim / 2, ng * 2);
    assert!(packed.len() >= rows * rb && scales.len() >= rows * sb);
    assert!(xg.len() >= n * in_dim && acc.len() >= rows * n && y.len() >= rows * n);
    let scale = |r: usize, g: usize| {
        let si = r * sb + g * 2;
        half::bf16::from_le_bytes([scales[si], scales[si + 1]]).to_f32()
    };
    let pp = packed.as_ptr();
    let ap = acc.as_mut_ptr() as *mut f32;
    for a in 0..rows * n {
        _mm512_store_ps(ap.add(a * 16), _mm512_setzero_ps());
    }
    for g in 0..ng {
        let xgp = xg.as_ptr().add(g * n * G);
        let mut r = 0;
        gemm_prefetch(packed, scales, rows, (rb, sb), g);
        while r + 2 <= rows {
            let [u0, u1] = group_weights_avx512(pp.add(r * rb + g * (G / 2)), scale(r, g));
            let [v0, v1] =
                group_weights_avx512(pp.add((r + 1) * rb + g * (G / 2)), scale(r + 1, g));
            let (ar, br) = (ap.add(r * n * 16), ap.add((r + 1) * n * 16));
            for j in 0..n {
                let xp = xgp.add(j * G);
                let (x0, x1) = (_mm512_loadu_ps(xp), _mm512_loadu_ps(xp.add(16)));
                let mut a = _mm512_load_ps(ar.add(j * 16));
                a = _mm512_fmadd_ps(u0, x0, a);
                a = _mm512_fmadd_ps(u1, x1, a);
                _mm512_store_ps(ar.add(j * 16), a);
                let mut b = _mm512_load_ps(br.add(j * 16));
                b = _mm512_fmadd_ps(v0, x0, b);
                b = _mm512_fmadd_ps(v1, x1, b);
                _mm512_store_ps(br.add(j * 16), b);
            }
            r += 2;
        }
        if r < rows {
            let [w0, w1] = group_weights_avx512(pp.add(r * rb + g * (G / 2)), scale(r, g));
            let ar = ap.add(r * n * 16);
            for j in 0..n {
                let xp = xgp.add(j * G);
                let mut a = _mm512_load_ps(ar.add(j * 16));
                a = _mm512_fmadd_ps(w0, _mm512_loadu_ps(xp), a);
                a = _mm512_fmadd_ps(w1, _mm512_loadu_ps(xp.add(16)), a);
                _mm512_store_ps(ar.add(j * 16), a);
            }
        }
    }
    for (i, value) in y[..rows * n].iter_mut().enumerate() {
        *value = _mm512_reduce_add_ps(_mm512_load_ps(ap.add(i * 16)));
    }
}

/// One output row's raw dot on an explicit kernel ([`dequant_row_dot`] with the
/// dispatch lifted out) — the per-row reference the forced paths use.
#[inline]
fn dequant_row_dot_with(
    isa: Int4Isa,
    packed_row: &[u8],
    scales_row: &[u8],
    x: &[f32],
    in_dim: usize,
) -> f32 {
    // SAFETY (both): callers assert `isa.supported()`; loads stay within
    // packed_row (in_dim/2 B), scales_row (in_dim/G·2 B) and x (in_dim).
    #[cfg(target_arch = "x86_64")]
    if isa == Int4Isa::Avx512 {
        return unsafe { dequant_row_dot_avx512(packed_row, scales_row, x, in_dim) };
    }
    #[cfg(target_arch = "x86_64")]
    if isa == Int4Isa::Avx2 {
        return unsafe { dequant_row_dot_avx2(packed_row, scales_row, x, in_dim) };
    }
    let _ = isa;
    dequant_row_dot_scalar(packed_row, scales_row, x, in_dim)
}

impl MmapExpert {
    /// [`Self::swiglu_from`] for `xs.len()` inputs at once, returned flat
    /// `[n, dim]` (input-major). Each weight group is unpacked once and
    /// applied to every input, so the expert's bytes cross the memory bus once
    /// per call instead of once per input. Row `j` of the result is
    /// BIT-IDENTICAL to `self.swiglu_from(data, xs[j])` on the same machine
    /// (see the contract above); one input IS that call.
    pub fn swiglu_rows_from(&self, data: &[u8], xs: &[&[f32]]) -> Vec<f32> {
        match xs {
            [] => Vec::new(),
            [x] => self.swiglu_from(data, x),
            _ => self.swiglu_rows_with(data, xs, Int4Isa::active(), gemm_tile_rows()),
        }
    }

    /// [`Self::swiglu_rows_from`] over the mapping itself — per row the bits of
    /// `glm::ffn::swiglu_mmap` (the `AnyExpert::Mmap` forward).
    pub fn swiglu_rows(&self, xs: &[&[f32]]) -> Vec<f32> {
        self.swiglu_rows_from(&self.mmap, xs)
    }

    /// The multi-input SwiGLU on an explicit kernel and tile (tests / benches:
    /// the AVX2 path on an AVX-512 box). Panics if `isa` is unsupported here.
    #[doc(hidden)]
    pub fn swiglu_rows_from_forced(
        &self,
        data: &[u8],
        xs: &[&[f32]],
        isa: Int4Isa,
        tile_rows: usize,
    ) -> Vec<f32> {
        assert!(isa.supported(), "int4 gemm: {isa:?} not supported here");
        self.swiglu_rows_with(data, xs, isa, tile_rows.max(1))
    }

    /// [`Self::swiglu_from`] on an explicit kernel: `gemv_rows` = 2|4 selects
    /// the tiled AVX2 GEMV (`CASCADIA_INT4_GEMV_ROWS`), anything else the
    /// per-row rayon path. The single-input baseline for tests / benches; with
    /// `Int4Isa::active()` it is `swiglu_from`'s own arithmetic.
    #[doc(hidden)]
    pub fn swiglu_from_forced(
        &self,
        data: &[u8],
        x: &[f32],
        isa: Int4Isa,
        gemv_rows: usize,
    ) -> Vec<f32> {
        assert!(isa.supported(), "int4 gemv: {isa:?} not supported here");
        let (inter, dim) = (self.inter, self.dim);
        let sec = section_bytes(inter, dim);
        let mut h = vec![0.0f32; inter];
        gemv_forced(data, 0, inter, dim, x, &mut h, isa, gemv_rows);
        let mut u = vec![0.0f32; inter];
        gemv_forced(data, sec, inter, dim, x, &mut u, isa, gemv_rows);
        for (hi, &ui) in h.iter_mut().zip(&u) {
            *hi = (*hi / (1.0 + (-*hi).exp())) * ui;
        }
        let mut out = vec![0.0f32; dim];
        gemv_forced(data, 2 * sec, dim, inter, &h, &mut out, isa, gemv_rows);
        out
    }

    fn swiglu_rows_with(&self, data: &[u8], xs: &[&[f32]], isa: Int4Isa, tile: usize) -> Vec<f32> {
        use rayon::prelude::*;
        let (inter, dim) = (self.inter, self.dim);
        let sec = section_bytes(inter, dim);
        let section = |off: usize, out_dim: usize, in_dim: usize| {
            let ng = in_dim / G;
            (
                &data[off..off + out_dim * in_dim / 2],
                &data[off + out_dim * in_dim / 2..off + out_dim * in_dim / 2 + out_dim * ng * 2],
            )
        };
        let (gate_p, gate_s) = section(0, inter, dim);
        let (up_p, up_s) = section(sec, inter, dim);
        let (down_p, down_s) = section(2 * sec, dim, inter);
        let n = xs.len();
        let mut out = vec![0.0f32; n * dim];
        if n == 0 {
            return out;
        }
        // Balanced passes of at most GEMM_MAX_INPUTS inputs.
        let pass = n.div_ceil(n.div_ceil(GEMM_MAX_INPUTS));
        for (chunk, dst) in xs.chunks(pass).zip(out.chunks_mut(pass * dim)) {
            let m = chunk.len();
            let xg = group_major(chunk, dim);
            let mut h = vec![0.0f32; inter * m]; // [inter][m]
            gemm_section(gate_p, gate_s, dim, &xg, m, &mut h, isa, tile, true);
            let mut u = vec![0.0f32; inter * m];
            gemm_section(up_p, up_s, dim, &xg, m, &mut u, isa, tile, true);
            // silu(gate)·up — `swiglu_from`'s expression, elementwise (so any
            // split is value-neutral); serial it would cost n × 3072 `exp`s.
            h.par_chunks_mut(4096)
                .zip(u.par_chunks(4096))
                .for_each(|(hc, uc)| {
                    for (hi, &ui) in hc.iter_mut().zip(uc) {
                        *hi = (*hi / (1.0 + (-*hi).exp())) * ui;
                    }
                });
            let hg = group_major_from_cols(&h, m, inter);
            let mut yt = vec![0.0f32; dim * m]; // [dim][m]
            gemm_section(down_p, down_s, inter, &hg, m, &mut yt, isa, tile, true);
            for (j, row) in dst.chunks_mut(dim).enumerate() {
                for (o, v) in row.iter_mut().enumerate() {
                    *v = yt[o * m + j];
                }
            }
        }
        out
    }
}

/// `gemv_on` with the kernel and tiling passed in instead of detected / read
/// from the environment. `gemv_on` itself is left untouched (it is the
/// production single-input path); `tests::row_gemm` pins the two together.
#[allow(clippy::too_many_arguments)]
fn gemv_forced(
    data: &[u8],
    sec_off: usize,
    out_dim: usize,
    in_dim: usize,
    x: &[f32],
    y: &mut [f32],
    isa: Int4Isa,
    gemv_rows: usize,
) {
    use rayon::prelude::*;
    let ng = in_dim / G;
    let packed = &data[sec_off..sec_off + out_dim * in_dim / 2];
    let scales =
        &data[sec_off + out_dim * in_dim / 2..sec_off + out_dim * in_dim / 2 + out_dim * ng * 2];
    let row_bytes = in_dim / 2;
    #[cfg(target_arch = "x86_64")]
    if isa == Int4Isa::Avx2 {
        match gemv_rows {
            2 => return gemv_tiled_avx2::<2>(packed, scales, x, y),
            4 => return gemv_tiled_avx2::<4>(packed, scales, x, y),
            _ => {}
        }
    }
    let _ = gemv_rows;
    y.par_iter_mut().enumerate().for_each(|(o, yy)| {
        let prow = &packed[o * row_bytes..(o + 1) * row_bytes];
        let srow = &scales[o * ng * 2..(o + 1) * ng * 2];
        *yy = to_bf16(dequant_row_dot_with(isa, prow, srow, x, in_dim));
    });
}

#[cfg(test)]
mod tests {
    use super::{dequant_row_dot, G};

    /// The active `dequant_row_dot` (AVX2 on x86, scalar elsewhere) must match a
    /// straightforward dequant-then-dot reference within f32 rounding — a wrong
    /// nibble decode or SIMD lane order shows up as a large relative error.
    /// Fixture-free so it runs on any node.
    #[test]
    fn fused_dequant_dot_matches_reference() {
        let in_dim = 256usize; // 8 groups of 32
        let ng = in_dim / G;
        let mut packed = vec![0u8; in_dim / 2];
        let mut scales = vec![0u8; ng * 2];
        let mut x = vec![0f32; in_dim];
        for (i, b) in packed.iter_mut().enumerate() {
            *b = ((i * 37 + 11) & 0xFF) as u8; // spans every nibble value
        }
        for g in 0..ng {
            let s = half::bf16::from_f32(0.05 + 0.013 * g as f32).to_le_bytes();
            scales[g * 2] = s[0];
            scales[g * 2 + 1] = s[1];
        }
        for (i, xi) in x.iter_mut().enumerate() {
            *xi = ((i as f32) * 0.13).sin() * 0.5;
        }
        // reference: dequant each element (col order lo,hi per byte) then sum
        let mut refv = 0.0f64;
        for g in 0..ng {
            let s = half::bf16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]).to_f32();
            for i in 0..G / 2 {
                let byte = packed[g * (G / 2) + i];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                refv += (lo as f32 * s * x[g * G + 2 * i]) as f64;
                refv += (hi as f32 * s * x[g * G + 2 * i + 1]) as f64;
            }
        }
        let got = dequant_row_dot(&packed, &scales, &x, in_dim) as f64;
        let rel = (got - refv).abs() / refv.abs().max(1e-6);
        assert!(rel < 1e-4, "fused={got} ref={refv} rel={rel}");
    }

    use super::MmapExpert;
    use std::path::PathBuf;

    /// A routed expert from the committed GLM-5.2 fixture (hidden = inter = 32).
    fn fixture_expert() -> MmapExpert {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/glm5_export/experts/layer_01/expert_000.bin");
        MmapExpert::open(&path, 32, 32).expect("open fixture expert")
    }

    /// The hot/cold correctness guarantee reduces to this: computing a cold
    /// expert from its whole-bin READ buffer (`swiglu_from` over `read_bytes`)
    /// must be BIT-for-bit identical to the mmap kernel (`swiglu_mmap`) the hot
    /// and fallback paths use. The end-to-end parity tests only check argmax
    /// tokens; this pins the actual bitwise equality the claim rests on, and by
    /// extension the read-failure fallback (which IS `swiglu_mmap`).
    #[test]
    fn swiglu_from_read_bytes_is_bit_identical_to_mmap_kernel() {
        use crate::glm::ffn::swiglu_mmap;
        let m = fixture_expert();
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.13).sin() * 0.5).collect();
        let via_mmap = swiglu_mmap(&m, &x);
        let via_read = m.swiglu_from(&m.read_bytes().expect("full bin reads"), &x);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&via_mmap),
            bits(&via_read),
            "cold-read swiglu_from diverged from the mmap kernel bit-for-bit"
        );
    }

    /// `open` validated the bin length, but a file that shrank AFTER open (a bin
    /// re-synced/truncated under a running node) would otherwise return a short
    /// buffer that slices out of bounds downstream. `read_bytes` must reject it
    /// with `Err` so the hot/cold + R1 paths fall back to mmap instead of
    /// panicking.
    ///
    /// Unix-only: Windows refuses to shrink a file that still has a mapped
    /// section open (OS error 1224), so the `set_len` setup below can't even
    /// run there — the OS prevents the very scenario this guards against, so
    /// there is nothing for `read_bytes` to reject.
    #[cfg(not(windows))]
    #[test]
    fn read_bytes_rejects_a_bin_that_shrank_after_open() {
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/glm5_export/experts/layer_01/expert_000.bin");
        let tmp = std::env::temp_dir().join(format!("glm5_short_{}.bin", std::process::id()));
        std::fs::copy(&src, &tmp).unwrap();
        let m = MmapExpert::open(&tmp, 32, 32).expect("open full bin");
        assert!(
            m.read_bytes().is_ok(),
            "full bin reads OK before truncation"
        );
        // Shrink the backing file under the already-mmap'd expert. Only
        // read_bytes (fs::read) touches it afterwards, so no mmap SIGBUS.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&tmp)
            .unwrap()
            .set_len(16)
            .unwrap();
        let err = m.read_bytes().expect_err("short bin must error");
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
        std::fs::remove_file(&tmp).ok();
    }
}

/// Tile output rows while keeping each row's original AVX2 FMA chain. The
/// caller has checked AVX2/FMA and excluded the different AVX-512 reduction.
#[cfg(target_arch = "x86_64")]
fn gemv_tiled_avx2<const ROWS: usize>(packed: &[u8], scales: &[u8], x: &[f32], y: &mut [f32]) {
    use rayon::prelude::*;
    let n = x.len();
    let rb = n / 2;
    let sb = n / G * 2;
    y.par_chunks_mut(ROWS).enumerate().for_each(|(tile, out)| {
        let first = tile * ROWS;
        if out.len() == ROWS {
            // SAFETY: dispatch checked CPU features. All rows have n/2 packed
            // bytes and n/32 bf16 scales; n is group-aligned in the bin format.
            unsafe {
                dequant_rows_avx2::<ROWS>(
                    &packed[first * rb..(first + ROWS) * rb],
                    &scales[first * sb..(first + ROWS) * sb],
                    x,
                    out,
                )
            };
        } else {
            for (r, value) in out.iter_mut().enumerate() {
                let row = first + r;
                // SAFETY: same feature and row bounds as the complete tiles.
                *value = to_bf16(unsafe {
                    dequant_row_dot_avx2(
                        &packed[row * rb..(row + 1) * rb],
                        &scales[row * sb..(row + 1) * sb],
                        x,
                        n,
                    )
                });
            }
        }
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_rows_avx2<const ROWS: usize>(
    packed: &[u8],
    scales: &[u8],
    x: &[f32],
    y: &mut [f32],
) {
    use core::arch::x86_64::*;
    let n = x.len();
    let ng = n / G;
    let mask = _mm_set1_epi8(15);
    let bias = _mm_set1_epi8(8);
    let mut acc = [_mm256_setzero_ps(); ROWS];
    for g in 0..ng {
        let xp = x.as_ptr().add(g * G);
        let x0 = _mm256_loadu_ps(xp);
        let x1 = _mm256_loadu_ps(xp.add(8));
        let x2 = _mm256_loadu_ps(xp.add(16));
        let x3 = _mm256_loadu_ps(xp.add(24));
        for (r, a) in acc.iter_mut().enumerate() {
            let si = (r * ng + g) * 2;
            let scale = half::bf16::from_le_bytes([scales[si], scales[si + 1]]).to_f32();
            let sv = _mm256_set1_ps(scale);
            let pk = _mm_loadu_si128(packed.as_ptr().add(r * n / 2 + g * G / 2).cast());
            let low = _mm_sub_epi8(_mm_and_si128(pk, mask), bias);
            let high = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16::<4>(pk), mask), bias);
            let il = _mm_unpacklo_epi8(low, high);
            let ih = _mm_unpackhi_epi8(low, high);
            let c0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(il));
            let c1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(il)));
            let c2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ih));
            let c3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(ih)));
            *a = _mm256_fmadd_ps(_mm256_mul_ps(c0, sv), x0, *a);
            *a = _mm256_fmadd_ps(_mm256_mul_ps(c1, sv), x1, *a);
            *a = _mm256_fmadd_ps(_mm256_mul_ps(c2, sv), x2, *a);
            *a = _mm256_fmadd_ps(_mm256_mul_ps(c3, sv), x3, *a);
        }
    }
    for (a, value) in acc.into_iter().zip(y) {
        let s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
        let shuf = _mm_movehdup_ps(s);
        let sums = _mm_add_ps(s, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        *value = to_bf16(_mm_cvtss_f32(_mm_add_ss(sums, shuf2)));
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tiled_int4_tests {
    use super::*;

    #[test]
    fn tiled_rows_preserve_avx2_bits_across_scales_and_odd_row_counts() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let mut seed = 4431u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as u32
        };
        for n in [32, 64, 96, 256, 3072, 6144] {
            for rows in [1, 2, 3, 4, 5, 17] {
                let data: Vec<u8> = (0..n * rows / 2 + 1).map(|_| next() as u8).collect();
                let packed = &data[1..];
                let mut scales = vec![0u8];
                for _ in 0..rows * n / G {
                    let scale = half::bf16::from_f32((next() % 127) as f32 / 4096.0);
                    scales.extend(scale.to_le_bytes());
                }
                let scales = &scales[1..];
                let x: Vec<f32> = (0..n)
                    .map(|_| (next() >> 8) as f32 / 16777216.0 - 0.5)
                    .collect();
                let want: Vec<f32> = (0..rows)
                    .map(|r| {
                        to_bf16(unsafe {
                            dequant_row_dot_avx2(
                                &packed[r * n / 2..(r + 1) * n / 2],
                                &scales[r * n / G * 2..(r + 1) * n / G * 2],
                                &x,
                                n,
                            )
                        })
                    })
                    .collect();
                let mut a = vec![f32::NAN; rows];
                let mut b = vec![f32::NAN; rows];
                gemv_tiled_avx2::<2>(packed, scales, &x, &mut a);
                gemv_tiled_avx2::<4>(packed, scales, &x, &mut b);
                for r in 0..rows {
                    assert_eq!(
                        a[r].to_bits(),
                        want[r].to_bits(),
                        "2-row n={n} rows={rows} r={r}"
                    );
                    assert_eq!(
                        b[r].to_bits(),
                        want[r].to_bits(),
                        "4-row n={n} rows={rows} r={r}"
                    );
                }
            }
        }
    }
}

/// The multi-input kernels against the single-input ones, on the RAW f32 dot.
/// (The GEMV's bf16 rounding keeps 8 significant bits, so it hides almost every
/// summation-order slip — `bf16_rounding_would_hide_an_order_slip` counts it —
/// and an output-level comparison alone would have no teeth.) Each kernel is
/// called directly, so the AVX2 path is exercised on an AVX-512 box too.
#[cfg(test)]
mod row_gemm_tests {
    use super::*;

    fn isas() -> Vec<Int4Isa> {
        [Int4Isa::Scalar, Int4Isa::Avx2, Int4Isa::Avx512]
            .into_iter()
            .filter(|i| i.supported())
            .collect()
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 32) as u32
        }
    }

    /// A random `[out_dim, in_dim]` section (every nibble value, scales of the
    /// real model's order incl. zero and negative) + `n` inputs.
    fn section(
        rng: &mut Rng,
        out_dim: usize,
        in_dim: usize,
        n: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<Vec<f32>>) {
        let packed: Vec<u8> = (0..out_dim * in_dim / 2)
            .map(|_| rng.next() as u8)
            .collect();
        let mut scales = Vec::with_capacity(out_dim * in_dim / G * 2);
        for _ in 0..out_dim * in_dim / G {
            let s = (rng.next() % 255) as f32 / 8192.0 - 127.0 / 8192.0;
            scales.extend(half::bf16::from_f32(s).to_le_bytes());
        }
        let xs = (0..n)
            .map(|_| {
                (0..in_dim)
                    .map(|_| (rng.next() >> 8) as f32 / 16777216.0 - 0.5)
                    .collect()
            })
            .collect();
        (packed, scales, xs)
    }

    /// Per-(row, input) raw dots from the SINGLE-input kernel, `[out_dim][n]`.
    fn per_row(
        isa: Int4Isa,
        packed: &[u8],
        scales: &[u8],
        xs: &[Vec<f32>],
        out_dim: usize,
        in_dim: usize,
    ) -> Vec<f32> {
        use rayon::prelude::*;
        let (rb, sb, n) = (in_dim / 2, in_dim / G * 2, xs.len());
        let mut want = vec![0.0f32; out_dim * n];
        want.par_chunks_mut(n).enumerate().for_each(|(o, w)| {
            for (j, x) in xs.iter().enumerate() {
                w[j] = dequant_row_dot_with(
                    isa,
                    &packed[o * rb..(o + 1) * rb],
                    &scales[o * sb..(o + 1) * sb],
                    x,
                    in_dim,
                );
            }
        });
        want
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        isa: Int4Isa,
        tile: usize,
        round: bool,
        packed: &[u8],
        scales: &[u8],
        xs: &[Vec<f32>],
        out_dim: usize,
        in_dim: usize,
    ) -> Vec<f32> {
        let refs: Vec<&[f32]> = xs.iter().map(Vec::as_slice).collect();
        let xg = group_major(&refs, in_dim);
        let mut yt = vec![f32::NAN; out_dim * xs.len()];
        gemm_section(
            packed,
            scales,
            in_dim,
            &xg,
            xs.len(),
            &mut yt,
            isa,
            tile,
            round,
        );
        yt
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    #[test]
    fn gemm_raw_dots_are_the_single_input_kernels_bits() {
        let mut rng = Rng(0x1A2B_3C4D);
        for isa in isas() {
            for (out_dim, in_dim) in [(1, 32), (3, 64), (5, 96), (17, 256), (9, 3072), (6, 6144)] {
                for n in [1, 2, 3, 5, 8, 17] {
                    let (packed, scales, xs) = section(&mut rng, out_dim, in_dim, n);
                    let want = per_row(isa, &packed, &scales, &xs, out_dim, in_dim);
                    for tile in [1, 2, 4, 8] {
                        let got = gemm(isa, tile, false, &packed, &scales, &xs, out_dim, in_dim);
                        assert_eq!(
                            bits(&got),
                            bits(&want),
                            "{isa:?} raw {out_dim}x{in_dim} n={n} tile={tile}"
                        );
                        let got = gemm(isa, tile, true, &packed, &scales, &xs, out_dim, in_dim);
                        let rounded: Vec<f32> = want.iter().map(|&v| to_bf16(v)).collect();
                        assert_eq!(
                            bits(&got),
                            bits(&rounded),
                            "{isa:?} bf16 {out_dim}x{in_dim} n={n} tile={tile}"
                        );
                    }
                }
            }
        }
    }

    /// The real sections (gate/up `[3072, 6144]`, down `[6144, 3072]`), every
    /// weight row, the default tile.
    #[test]
    fn gemm_raw_dots_match_at_the_real_dims() {
        let mut rng = Rng(0x5EED_0042);
        for isa in isas() {
            for (out_dim, in_dim) in [(3072, 6144), (6144, 3072)] {
                let (packed, scales, xs) = section(&mut rng, out_dim, in_dim, 17);
                let want = per_row(isa, &packed, &scales, &xs, out_dim, in_dim);
                for n in [2, 3, 5, 8, 17] {
                    let got = gemm(
                        isa,
                        GEMM_TILE_ROWS,
                        false,
                        &packed,
                        &scales,
                        &xs[..n],
                        out_dim,
                        in_dim,
                    );
                    for o in 0..out_dim {
                        assert_eq!(
                            bits(&got[o * n..(o + 1) * n]),
                            bits(&want[o * 17..o * 17 + n]),
                            "{isa:?} {out_dim}x{in_dim} n={n} row {o}"
                        );
                    }
                }
            }
        }
    }

    /// `dequant_row_dot_scalar` with the groups walked LAST-to-first: the same
    /// products, one summation-order slip.
    fn reversed_groups_dot(packed_row: &[u8], scales_row: &[u8], x: &[f32], in_dim: usize) -> f32 {
        let mut acc = 0.0f32;
        for g in (0..in_dim / G).rev() {
            let s = bf16::from_le_bytes([scales_row[g * 2], scales_row[g * 2 + 1]]).to_f32();
            for i in 0..G / 2 {
                let byte = packed_row[g * (G / 2) + i];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                acc += (lo as f32 * s) * x[g * G + 2 * i];
                acc += (hi as f32 * s) * x[g * G + 2 * i + 1];
            }
        }
        acc
    }

    /// Teeth: the raw comparison above must FAIL for a kernel that sums in a
    /// different order — reversed groups against the scalar kernel, and the
    /// three lane structures against each other.
    #[test]
    fn a_different_summation_order_changes_the_raw_bits() {
        let mut rng = Rng(0x7EE7);
        let (out_dim, in_dim, n) = (64, 6144, 4);
        let (packed, scales, xs) = section(&mut rng, out_dim, in_dim, n);
        let (rb, sb) = (in_dim / 2, in_dim / G * 2);
        let scalar = gemm(
            Int4Isa::Scalar,
            4,
            false,
            &packed,
            &scales,
            &xs,
            out_dim,
            in_dim,
        );
        let mut reversed = vec![0.0f32; out_dim * n];
        for o in 0..out_dim {
            for (j, x) in xs.iter().enumerate() {
                reversed[o * n + j] = reversed_groups_dot(
                    &packed[o * rb..(o + 1) * rb],
                    &scales[o * sb..(o + 1) * sb],
                    x,
                    in_dim,
                );
            }
        }
        let differing = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .filter(|(p, q)| p.to_bits() != q.to_bits())
                .count()
        };
        let total = out_dim * n;
        let slipped = differing(&scalar, &reversed);
        assert!(
            slipped * 2 > total,
            "reversed group order changed only {slipped}/{total} raw dots"
        );
        let all = isas();
        for (a, &ia) in all.iter().enumerate() {
            for &ib in &all[a + 1..] {
                let ya = gemm(ia, 4, false, &packed, &scales, &xs, out_dim, in_dim);
                let yb = gemm(ib, 4, false, &packed, &scales, &xs, out_dim, in_dim);
                let d = differing(&ya, &yb);
                assert!(
                    d * 2 > total,
                    "{ia:?} vs {ib:?}: only {d}/{total} raw dots differ"
                );
            }
        }
    }

    /// Why the comparisons above are on the raw dot: after the GEMV's bf16
    /// rounding the same order slip survives in (almost) no output.
    #[test]
    fn bf16_rounding_would_hide_an_order_slip() {
        let mut rng = Rng(0xB16);
        let (out_dim, in_dim, n) = (64, 6144, 4);
        let (packed, scales, xs) = section(&mut rng, out_dim, in_dim, n);
        let (rb, sb) = (in_dim / 2, in_dim / G * 2);
        let rounded = gemm(
            Int4Isa::Scalar,
            4,
            true,
            &packed,
            &scales,
            &xs,
            out_dim,
            in_dim,
        );
        let mut hidden = 0;
        for o in 0..out_dim {
            for (j, x) in xs.iter().enumerate() {
                let slip = to_bf16(reversed_groups_dot(
                    &packed[o * rb..(o + 1) * rb],
                    &scales[o * sb..(o + 1) * sb],
                    x,
                    in_dim,
                ));
                hidden += (slip.to_bits() == rounded[o * n + j].to_bits()) as usize;
            }
        }
        assert!(
            hidden * 10 >= out_dim * n * 9,
            "expected bf16 rounding to mask most order slips, masked {hidden}/{}",
            out_dim * n
        );
    }

    /// `gemv_forced` on the active kernel is `gemv_on` (the production
    /// single-input path the forced baseline stands in for).
    #[test]
    fn forced_gemv_on_the_active_kernel_is_gemv_on() {
        let mut rng = Rng(0xF0CE);
        let (out_dim, in_dim) = (37, 256);
        let (packed, scales, xs) = section(&mut rng, out_dim, in_dim, 1);
        let mut data = packed.clone();
        data.extend_from_slice(&scales);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("e.bin");
        // Any file of a valid length: gemv_on reads `data`, not the mapping.
        std::fs::write(&path, vec![0u8; 3 * section_bytes(32, 32)]).unwrap();
        let m = MmapExpert::open(&path, 32, 32).unwrap();
        let mut want = vec![0.0f32; out_dim];
        m.gemv_on(&data, 0, out_dim, in_dim, &xs[0], &mut want);
        for rows in [1, 2, 4] {
            let mut got = vec![0.0f32; out_dim];
            gemv_forced(
                &data,
                0,
                out_dim,
                in_dim,
                &xs[0],
                &mut got,
                Int4Isa::active(),
                rows,
            );
            assert_eq!(bits(&got), bits(&want), "gemv_rows={rows}");
        }
    }
}
