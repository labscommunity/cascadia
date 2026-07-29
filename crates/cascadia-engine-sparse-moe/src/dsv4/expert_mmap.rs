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
    pub fn read_bytes(&self) -> std::io::Result<Vec<u8>> {
        std::fs::read(&self.path)
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

    /// Estimate how many of this expert's mapped pages are resident in RAM right
    /// now, by probing `samples` pages spread evenly across the bin. Returns
    /// `(resident, probed)`; `(0, 0)` if the OS query is unavailable or fails.
    ///
    /// This is the decode profiler's TRUE expert-cache hit signal: mmap page
    /// faults are not counted by the process read counter on Windows (paging I/O
    /// is charged elsewhere), so residency is probed directly — `QueryWorkingSetEx`
    /// on Windows, `mincore` on unix. Best-effort and read-only.
    pub fn resident_pages_sampled(&self, samples: usize) -> (usize, usize) {
        resident_pages_sampled_ptr(self.mmap.as_ptr() as usize, self.mmap.len(), samples)
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
}

/// Sampled page-residency of an arbitrary mapping — `(resident, probed)`.
///
/// Shared leaf: the k3 fp4 expert maps use this too. Extracted from
/// [`MmapExpert::resident_pages_sampled`], which now delegates here.
pub fn resident_pages_sampled_ptr(base: usize, len: usize, samples: usize) -> (usize, usize) {
    const PAGE: usize = 4096;
    if len == 0 || samples == 0 {
        return (0, 0);
    }
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
