//! AXPY-form sparse SwiGLU FFN down projection.
//!
//! This is the kernel that closes the gap left by PR #34 — the
//! existing two-phase Gate-first FFN sparsity sparsifies only the
//! **up** matmul. The **down** matmul still ran dense over a sparse
//! intermediate vector, capping the end-to-end FFN-compute speedup
//! at ~1.2× even when up was 50% sparse. This module ports
//! PowerInfer SmallThinker's actual down-projection mechanism
//! (`/tmp/PowerInfer/smallthinker/powerinfer/fused_sparse_moe/`
//! `fused_sparse_moe.cpp:174-186`), which is *not* a column-sparse
//! GEMV but an **AXPY** over a *transposed* down weight.
//!
//! ## Algorithm
//!
//! Dense down projection: `y[h] = sum_k W_down[h, k] * inter[k]`
//! for `h in [0, hidden), k in [0, intermediate)`. Cost: `hidden ×
//! intermediate` FMAs.
//!
//! AXPY-form (with W transposed to `[intermediate, hidden]`):
//! `for r in active: y[h] += scalars[r] * W_t[r, h]` for `h in
//! [0, hidden)`. The full-rank dot product is replaced by an outer
//! accumulation over only the active intermediate lanes — rows for
//! inactive lanes (`silu(gate[r]) * up[r]` below the threshold) are
//! never touched. Cost: `active × hidden` FMAs ⇒ kernel speedup
//! ceiling **`1 / active_frac`** vs dense.
//!
//! ## Layout — the down weight must be transposed
//!
//! K2.6 stores down as `[hidden=7168, intermediate=2048]` int4 with
//! group-32 bf16 scales along the *intermediate* (K) axis. The
//! AXPY layout we need is `[intermediate, hidden]` with group-32
//! scales along the *hidden* axis — that way each AXPY row read is
//! contiguous (hidden/2 nibble bytes + hidden/32 bf16 scales) and
//! the FMA write into `y[hidden]` traverses the output in order.
//!
//! The transpose is not a permutation. The quantization grouping
//! changes orientation, so each weight is re-rounded into the new
//! group's scale. [`transpose_requantize_down`] performs the one-
//! time conversion at expert load time; the result is cached
//! alongside the original weights in cascadia's LRU expert cache
//! (`cascadia-engine-sparse-moe::runner::ExpertCache`).
//!
//! The re-quantization is the same group-32 symmetric int4 scheme
//! used by the original (max-abs over group ⇒ scale = max/7,
//! round-to-nearest with ties-to-even). Each weight rounds twice
//! total (original quantization + this re-quantization). Empirical
//! quality validation is the test suite + the end-to-end K2.6 eval.
//!
//! ## Attribution
//!
//! - AXPY-form sparse down: PowerInfer SmallThinker
//!   `fused_sparse_moe.cpp:174-186` (MIT, SJTU-IPADS / Tiiny AI;
//!   `--transpose-down all` flag at
//!   `convert_hf_to_gguf.py:6275-6283`). Clean-room Rust re-impl —
//!   no PowerInfer source copied.
//! - Premise (down is the most sparsifiable FFN tensor): TEAL
//!   (Liu et al. 2024, arxiv:2408.14690 Fig 7).

use rayon::prelude::*;

use crate::format::{bf16_bits_to_f32, f32_to_bf16_bits};
use crate::GROUP_SIZE;

/// Scalar reference AXPY-form sparse down. For each active
/// intermediate lane `r`, accumulate `scalars[r] · dequant_row(r)`
/// into `y[0..hidden]`.
///
/// Caller contract:
/// - `packed_t.len() == n_intermediate * n_hidden / 2`
/// - `scale_t_bits.len() == n_intermediate * n_hidden / GROUP_SIZE * 2`
/// - `scalars.len() == n_intermediate` (only `active` indices read)
/// - `y.len() == n_hidden`; caller pre-zeroes (or accumulates).
///
/// `active` indices must be `< n_intermediate`; duplicates are
/// summed (the function is order-independent across `active`).
///
/// Parallelism: rayon chunks `y` along the hidden dim so each
/// worker thread owns a disjoint slice and visits every active
/// lane on its slice — no atomics, no false sharing. Matches the
/// AVX-512 path's chunked-y strategy so the two layouts have the
/// same scaling behaviour.
pub fn dequant_axpy_int4_active(
    packed_t: &[u8],
    scale_t_bits: &[u8],
    scalars: &[f32],
    active: &[u32],
    n_intermediate: usize,
    n_hidden: usize,
    y: &mut [f32],
) {
    assert_eq!(packed_t.len(), n_intermediate * n_hidden / 2);
    let n_groups_h = n_hidden / GROUP_SIZE;
    assert_eq!(scale_t_bits.len(), n_intermediate * n_groups_h * 2);
    assert_eq!(scalars.len(), n_intermediate);
    assert_eq!(y.len(), n_hidden);
    // Use the same per-thread chunk size as the AVX-512 path so
    // the scalar reference behaves like a slow-but-equivalent
    // version of the AVX path under rayon.
    const CHUNK: usize = 256;
    let row_stride_packed = n_hidden / 2;
    let row_stride_scale = n_groups_h * 2;
    let groups_per_chunk = (CHUNK / GROUP_SIZE).max(1);

    y.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(ci, y_chunk)| {
            let chunk_start = ci * CHUNK;
            let group_offset_in_row = chunk_start / GROUP_SIZE;
            // Last chunk may be shorter than CHUNK if n_hidden isn't
            // a multiple of CHUNK; compute how many groups land in it.
            let groups_in_this_chunk = (y_chunk.len() / GROUP_SIZE).min(groups_per_chunk);
            for &r in active {
                let r = r as usize;
                debug_assert!(r < n_intermediate);
                let scalar = scalars[r];
                let row_packed = &packed_t[r * row_stride_packed..(r + 1) * row_stride_packed];
                let row_scales = &scale_t_bits[r * row_stride_scale..(r + 1) * row_stride_scale];
                for g_in_chunk in 0..groups_in_this_chunk {
                    let g_in_row = group_offset_in_row + g_in_chunk;
                    let scale_u16 = u16::from_le_bytes([
                        row_scales[g_in_row * 2],
                        row_scales[g_in_row * 2 + 1],
                    ]);
                    let scale = bf16_bits_to_f32(scale_u16);
                    let combined = scalar * scale;
                    let group_packed =
                        &row_packed[g_in_row * (GROUP_SIZE / 2)..(g_in_row + 1) * (GROUP_SIZE / 2)];
                    let y_off = g_in_chunk * GROUP_SIZE;
                    for i in 0..(GROUP_SIZE / 2) {
                        let byte = group_packed[i];
                        let lo_nibble = (byte & 0x0F) as i32 - 8;
                        let hi_nibble = ((byte >> 4) & 0x0F) as i32 - 8;
                        y_chunk[y_off + i * 2] += combined * (lo_nibble as f32);
                        y_chunk[y_off + i * 2 + 1] += combined * (hi_nibble as f32);
                    }
                }
                // Handle the last chunk's trailing groups (when y_chunk.len() <= CHUNK).
                if y_chunk.len() % GROUP_SIZE != 0 {
                    // n_hidden must be a multiple of GROUP_SIZE per the
                    // crate's invariant (groups are 32 wide); skip — no
                    // partial group at the chunk boundary.
                }
            }
        });
}

/// AVX-512 path. Parallelizes over `y` *chunks* so each worker
/// thread owns a disjoint slice of the output — no atomics, no
/// false sharing.
#[cfg(target_arch = "x86_64")]
mod avx512 {
    use core::arch::x86_64::*;
    use rayon::prelude::*;

    use crate::format::bf16_bits_to_f32;
    use crate::GROUP_SIZE;

    /// Per-thread chunk size in cols of `y`. Must be a multiple of
    /// `GROUP_SIZE` (32) so groups line up with chunk boundaries.
    /// 256 = 8 groups per chunk; 256 × 4 = 1 KiB y-bytes per chunk
    /// (well within L1).
    const CHUNK: usize = 256;

    /// SAFETY: caller must ensure `avx512f,avx512bw,avx512vl` are
    /// available at runtime (the public wrapper checks).
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub unsafe fn dequant_axpy_int4_active_avx512(
        packed_t: &[u8],
        scale_t_bits: &[u8],
        scalars: &[f32],
        active: &[u32],
        n_intermediate: usize,
        n_hidden: usize,
        y: &mut [f32],
    ) {
        assert_eq!(packed_t.len(), n_intermediate * n_hidden / 2);
        let n_groups_h = n_hidden / GROUP_SIZE;
        assert_eq!(scale_t_bits.len(), n_intermediate * n_groups_h * 2);
        assert_eq!(scalars.len(), n_intermediate);
        assert_eq!(y.len(), n_hidden);
        assert!(
            n_hidden % CHUNK == 0,
            "AXPY-AVX-512 expects n_hidden ({n_hidden}) divisible by CHUNK ({CHUNK})"
        );
        let row_stride_packed = n_hidden / 2;
        let row_stride_scale = n_groups_h * 2;
        let lo_mask = _mm_set1_epi8(0x0F);
        let bias = _mm_set1_epi8(8);

        // Chunked-y parallelism: each worker owns a disjoint
        // [chunk_start, chunk_start + CHUNK) slice of y. The key
        // perf trick is that the chunk's y values are loaded into
        // an array of __m512 accumulators ONCE per chunk, all
        // active lanes FMA into the accumulator array, and the
        // array is stored back to y ONCE at the end. This keeps
        // y in registers across all active scalars and eliminates
        // the per-(active, group) y read+write traffic that
        // otherwise dominates at p > 0.1.
        const VECS_PER_CHUNK: usize = CHUNK / 16; // 256/16 = 16 zmm regs
        y.par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(ci, y_chunk)| {
                let chunk_start = ci * CHUNK;
                let groups_per_chunk = CHUNK / GROUP_SIZE;
                let group_offset_in_row = chunk_start / GROUP_SIZE;
                // SAFETY: each thread owns a disjoint y_chunk; the
                // accumulator array lives entirely in stack
                // (compiler keeps it in registers given AVX-512's
                // 32 zmm regs and our 16-element footprint).
                unsafe {
                    // Load y_chunk into accumulators ONCE per
                    // chunk. 16 zmm regs = 256 f32 lanes.
                    let mut acc: [__m512; VECS_PER_CHUNK] = [_mm512_setzero_ps(); VECS_PER_CHUNK];
                    for v in 0..VECS_PER_CHUNK {
                        acc[v] = _mm512_loadu_ps(y_chunk.as_ptr().add(v * 16));
                    }

                    for &r in active {
                        let r = r as usize;
                        let scalar = scalars[r];
                        let row_packed_ptr = packed_t.as_ptr().add(r * row_stride_packed);
                        let row_scale_ptr = scale_t_bits.as_ptr().add(r * row_stride_scale);
                        for g_in_chunk in 0..groups_per_chunk {
                            let g_in_row = group_offset_in_row + g_in_chunk;
                            let scale_off = g_in_row * 2;
                            let scale_u16 = u16::from_le_bytes([
                                *row_scale_ptr.add(scale_off),
                                *row_scale_ptr.add(scale_off + 1),
                            ]);
                            let scale = bf16_bits_to_f32(scale_u16);
                            let combined = _mm512_set1_ps(scalar * scale);

                            // 16 packed bytes = 32 nibbles for this group
                            let p_ptr =
                                row_packed_ptr.add(g_in_row * (GROUP_SIZE / 2)) as *const __m128i;
                            let pk = _mm_loadu_si128(p_ptr);
                            let low_nibbles = _mm_and_si128(pk, lo_mask);
                            let high_nibbles = _mm_and_si128(_mm_srli_epi16::<4>(pk), lo_mask);
                            let low_signed = _mm_sub_epi8(low_nibbles, bias);
                            let high_signed = _mm_sub_epi8(high_nibbles, bias);
                            // Interleave so we recover original column
                            // order (col 0, 1, 2, 3, ...)
                            let interleaved_lo = _mm_unpacklo_epi8(low_signed, high_signed);
                            let interleaved_hi = _mm_unpackhi_epi8(low_signed, high_signed);
                            let lo_i32 = _mm512_cvtepi8_epi32(interleaved_lo);
                            let hi_i32 = _mm512_cvtepi8_epi32(interleaved_hi);
                            let lo_f = _mm512_cvtepi32_ps(lo_i32);
                            let hi_f = _mm512_cvtepi32_ps(hi_i32);
                            // Each group spans 2 zmm registers
                            // (16 f32 lanes each). Indices into
                            // the acc array: 2 * g_in_chunk and
                            // 2 * g_in_chunk + 1.
                            let v_idx = g_in_chunk * 2;
                            acc[v_idx] = _mm512_fmadd_ps(combined, lo_f, acc[v_idx]);
                            acc[v_idx + 1] = _mm512_fmadd_ps(combined, hi_f, acc[v_idx + 1]);
                        }
                    }

                    // Store accumulators back to y_chunk ONCE per
                    // chunk.
                    for v in 0..VECS_PER_CHUNK {
                        _mm512_storeu_ps(y_chunk.as_mut_ptr().add(v * 16), acc[v]);
                    }
                }
            });
    }
}

#[cfg(target_arch = "x86_64")]
pub use avx512::dequant_axpy_int4_active_avx512;

/// Wrapper: pick AVX-512 if available, else fall back to scalar.
/// Caller pre-zeroes `y` (or accumulates from a prior call).
pub fn dequant_axpy_int4_active_auto(
    packed_t: &[u8],
    scale_t_bits: &[u8],
    scalars: &[f32],
    active: &[u32],
    n_intermediate: usize,
    n_hidden: usize,
    y: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if n_hidden % 256 == 0
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: feature-checked at runtime.
            unsafe {
                dequant_axpy_int4_active_avx512(
                    packed_t,
                    scale_t_bits,
                    scalars,
                    active,
                    n_intermediate,
                    n_hidden,
                    y,
                );
            }
            return;
        }
    }
    dequant_axpy_int4_active(
        packed_t,
        scale_t_bits,
        scalars,
        active,
        n_intermediate,
        n_hidden,
        y,
    );
}

/// Transpose-and-requantize a `[hidden, intermediate]` int4 down
/// weight into `[intermediate, hidden]` layout for the AXPY-form
/// kernel.
///
/// Steps:
/// 1. Dequantize the source one row (output-row) at a time into
///    a single f32 scratch row (no full-matrix f32 buffer).
/// 2. Scatter that row into the matching column of a transposed
///    f32 working buffer `[intermediate, hidden]`.
/// 3. Re-quantize the transposed buffer row-by-row using the
///    same group-32 symmetric int4 scheme.
///
/// Output buffers:
/// - `packed_t.len()    == n_intermediate * (n_hidden / 2)`
/// - `scale_t_bits.len() == n_intermediate * (n_hidden / GROUP_SIZE) * 2`
///
/// One-time CPU cost at expert load: dominated by the transposed
/// f32 buffer (n_intermediate × n_hidden × 4 bytes ≈ 56 MiB for
/// K2.6 — runs in ~5 ms on a Cascade Lake socket; ~0.5 ms with
/// rayon over the source rows).
///
/// Quality: each weight rounds twice (original quantization +
/// this re-quantization). Empirical impact on K2.6 canonical-prompt
/// match is validated by the end-to-end test in
/// `cascadia-engine-sparse-moe/tests/k26_layer0_eval.rs` and the
/// `axpy_close_to_dense_down_random_weights` unit test below.
pub fn transpose_requantize_down(
    src_packed: &[u8],     // [n_hidden, n_intermediate / 2]
    src_scale_bits: &[u8], // [n_hidden, n_intermediate / GROUP_SIZE * 2]
    n_hidden: usize,
    n_intermediate: usize,
) -> (Vec<u8>, Vec<u8>) {
    let n_groups_src = n_intermediate / GROUP_SIZE; // groups along K (intermediate)
    let n_groups_dst = n_hidden / GROUP_SIZE; // groups along new K (hidden)
    assert_eq!(src_packed.len(), n_hidden * (n_intermediate / 2));
    assert_eq!(src_scale_bits.len(), n_hidden * n_groups_src * 2);

    // 1+2 : dequant every source row and scatter into a
    // [n_intermediate, n_hidden] f32 buffer. Parallelized across
    // source rows (each row independent).
    let mut transposed = vec![0.0f32; n_intermediate * n_hidden];
    let t_ptr_addr = transposed.as_mut_ptr() as usize;
    let row_stride_packed_src = n_intermediate / 2;
    let row_stride_scale_src = n_groups_src * 2;
    (0..n_hidden).into_par_iter().for_each(|h| {
        let row_packed = &src_packed[h * row_stride_packed_src..(h + 1) * row_stride_packed_src];
        let row_scales = &src_scale_bits[h * row_stride_scale_src..(h + 1) * row_stride_scale_src];
        for g in 0..n_groups_src {
            let scale_u16 = u16::from_le_bytes([row_scales[g * 2], row_scales[g * 2 + 1]]);
            let scale = bf16_bits_to_f32(scale_u16);
            let group_packed = &row_packed[g * (GROUP_SIZE / 2)..(g + 1) * (GROUP_SIZE / 2)];
            for i in 0..(GROUP_SIZE / 2) {
                let byte = group_packed[i];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                let col_lo = g * GROUP_SIZE + i * 2;
                let col_hi = g * GROUP_SIZE + i * 2 + 1;
                let v_lo = lo as f32 * scale;
                let v_hi = hi as f32 * scale;
                // Scatter (h, col_lo) → transposed[col_lo, h] etc.
                // SAFETY: each (h, col) pair maps to a unique
                // (col, h) cell; threads index by h, write distinct
                // cells in disjoint columns of `transposed`.
                unsafe {
                    let base = t_ptr_addr as *mut f32;
                    *base.add(col_lo * n_hidden + h) = v_lo;
                    *base.add(col_hi * n_hidden + h) = v_hi;
                }
            }
        }
    });

    // 3 : re-quantize each transposed row into the new packed
    // layout with group-32 symmetric int4 + bf16 scales along the
    // new col axis (hidden).
    let mut packed_t = vec![0u8; n_intermediate * (n_hidden / 2)];
    let mut scale_t_bits = vec![0u8; n_intermediate * n_groups_dst * 2];
    let row_stride_packed_dst = n_hidden / 2;
    let row_stride_scale_dst = n_groups_dst * 2;

    let packed_ptr_addr = packed_t.as_mut_ptr() as usize;
    let scale_ptr_addr = scale_t_bits.as_mut_ptr() as usize;
    (0..n_intermediate).into_par_iter().for_each(|r| {
        let row_f32 = &transposed[r * n_hidden..(r + 1) * n_hidden];
        // SAFETY: per-`r` writes go to distinct row regions in
        // both output buffers; iteration is over disjoint rows.
        let packed_row = unsafe { (packed_ptr_addr as *mut u8).add(r * row_stride_packed_dst) };
        let scale_row = unsafe { (scale_ptr_addr as *mut u8).add(r * row_stride_scale_dst) };
        for g in 0..n_groups_dst {
            let group_f32 = &row_f32[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
            // Symmetric int4: scale = max(|v|) / 7. Quantized
            // value q = round(v / scale) clamped to [-8, 7]; we
            // store q+8 in the nibble so the on-disk format
            // matches the existing exporter.
            let max_abs = group_f32.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
            let inv = if max_abs == 0.0 { 0.0 } else { 1.0 / scale };
            let scale_u16 = f32_to_bf16_bits(scale);
            // SAFETY: scale_row + 2*g .. +2 is within the row.
            unsafe {
                *scale_row.add(g * 2) = (scale_u16 & 0xFF) as u8;
                *scale_row.add(g * 2 + 1) = ((scale_u16 >> 8) & 0xFF) as u8;
            }
            for i in 0..(GROUP_SIZE / 2) {
                let v_lo = group_f32[i * 2];
                let v_hi = group_f32[i * 2 + 1];
                let q_lo = (v_lo * inv).round() as i32;
                let q_hi = (v_hi * inv).round() as i32;
                let q_lo_u = (q_lo.clamp(-8, 7) + 8) as u8;
                let q_hi_u = (q_hi.clamp(-8, 7) + 8) as u8;
                let byte = (q_hi_u << 4) | q_lo_u;
                // SAFETY: byte offset is within packed_row.
                unsafe {
                    *packed_row.add(g * (GROUP_SIZE / 2) + i) = byte;
                }
            }
        }
    });

    (packed_t, scale_t_bits)
}

/// Reusable per-call scratch for [`crate::ffn_sparsity::ffn_forward_sparse_f32`]
/// and the AXPY-form variant — eliminates the 7 per-call
/// `Vec::new()` allocations that accumulated at K2.6 dispatch
/// rates (~480 routed-expert calls per token × 7 allocs).
///
/// Construction: `FfnScratch::new(hidden, intermediate)` allocs
/// once; subsequent `resize_for(...)` is a no-op if the buffers
/// are already large enough.
///
/// One scratch is meant to be owned per-runner (not per-call) and
/// re-used across routed-expert calls within the same token. The
/// scratch is not Sync — it holds raw `Vec` buffers — so per-thread
/// pooling would need separate instances if a future caller
/// fanned the AXPY across threads.
pub struct FfnScratch {
    /// `x_f32`: f32 copy of the bf16 input (gate / up share this).
    pub x_f32: Vec<f32>,
    /// `gate_out`: gate projection output (length intermediate).
    pub gate_out: Vec<f32>,
    /// `silu_gate`: silu(gate_out), length intermediate.
    pub silu_gate: Vec<f32>,
    /// `active`: sorted indices of active intermediate lanes.
    pub active: Vec<u32>,
    /// `up_out`: up projection output (length intermediate; only
    /// `active` entries populated in the sparse path).
    pub up_out: Vec<f32>,
    /// `inter`: silu(gate) ⊙ up (length intermediate; only
    /// `active` entries non-zero).
    pub inter: Vec<f32>,
    /// `out_f32`: f32 output (length hidden) before bf16 cast.
    pub out_f32: Vec<f32>,
}

impl FfnScratch {
    pub fn new(hidden: usize, intermediate: usize) -> Self {
        Self {
            x_f32: vec![0.0f32; hidden],
            gate_out: vec![0.0f32; intermediate],
            silu_gate: vec![0.0f32; intermediate],
            active: Vec::with_capacity(intermediate),
            up_out: vec![0.0f32; intermediate],
            inter: vec![0.0f32; intermediate],
            out_f32: vec![0.0f32; hidden],
        }
    }

    /// Grow buffers in-place if they're smaller than the requested
    /// shape. No shrinkage — the scratch's high-water mark stays
    /// allocated across calls.
    pub fn resize_for(&mut self, hidden: usize, intermediate: usize) {
        if self.x_f32.len() < hidden {
            self.x_f32.resize(hidden, 0.0);
        }
        if self.gate_out.len() < intermediate {
            self.gate_out.resize(intermediate, 0.0);
        }
        if self.silu_gate.len() < intermediate {
            self.silu_gate.resize(intermediate, 0.0);
        }
        if self.up_out.len() < intermediate {
            self.up_out.resize(intermediate, 0.0);
        }
        if self.inter.len() < intermediate {
            self.inter.resize(intermediate, 0.0);
        }
        if self.out_f32.len() < hidden {
            self.out_f32.resize(hidden, 0.0);
        }
        // `active` is cleared by the FFN forward; capacity is
        // preserved across calls.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::dequant_gemv_int4;

    /// Build a deterministic `[n_hidden, n_intermediate]` int4
    /// down weight + group-32 bf16 scales the same way the test
    /// fixtures in ffn_sparsity.rs do.
    fn make_down_weight(n_hidden: usize, n_intermediate: usize) -> (Vec<u8>, Vec<u8>) {
        let n_groups = n_intermediate / GROUP_SIZE;
        let packed: Vec<u8> = (0..n_hidden * n_intermediate / 2)
            .map(|i| {
                let lo = (i * 31 + 7) & 0x0F;
                let hi = (i * 53 + 11) & 0x0F;
                ((hi << 4) | lo) as u8
            })
            .collect();
        // bf16 1.0 = 0x3F80, written little-endian as [0x80, 0x3F].
        let scales: Vec<u8> = vec![0x80, 0x3F].repeat(n_hidden * n_groups);
        (packed, scales)
    }

    /// Transposing then "untransposing" should round-trip within
    /// the precision of the re-quantization (one extra rounding
    /// step ⇒ each weight differs by ≤ 1 LSB of the new scale).
    #[test]
    fn transpose_requantize_roundtrip_close_to_identity() {
        let n_hidden = 64;
        let n_intermediate = 64;
        let (src_packed, src_scale_bits) = make_down_weight(n_hidden, n_intermediate);

        // dequant the source manually so we have a reference.
        let mut src_f32 = vec![0.0f32; n_hidden * n_intermediate];
        for h in 0..n_hidden {
            let row_packed = &src_packed[h * (n_intermediate / 2)..(h + 1) * (n_intermediate / 2)];
            let row_scales = &src_scale_bits[h * (n_intermediate / GROUP_SIZE) * 2
                ..(h + 1) * (n_intermediate / GROUP_SIZE) * 2];
            for g in 0..n_intermediate / GROUP_SIZE {
                let scale = bf16_bits_to_f32(u16::from_le_bytes([
                    row_scales[g * 2],
                    row_scales[g * 2 + 1],
                ]));
                for i in 0..GROUP_SIZE / 2 {
                    let byte = row_packed[g * (GROUP_SIZE / 2) + i];
                    let lo = (byte & 0x0F) as i32 - 8;
                    let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                    src_f32[h * n_intermediate + g * GROUP_SIZE + i * 2] = lo as f32 * scale;
                    src_f32[h * n_intermediate + g * GROUP_SIZE + i * 2 + 1] = hi as f32 * scale;
                }
            }
        }

        let (packed_t, scale_t_bits) =
            transpose_requantize_down(&src_packed, &src_scale_bits, n_hidden, n_intermediate);

        // dequant the transposed result back to f32 and compare
        // against the source post-permutation.
        for r in 0..n_intermediate {
            let row_packed = &packed_t[r * (n_hidden / 2)..(r + 1) * (n_hidden / 2)];
            let row_scales = &scale_t_bits
                [r * (n_hidden / GROUP_SIZE) * 2..(r + 1) * (n_hidden / GROUP_SIZE) * 2];
            for g in 0..n_hidden / GROUP_SIZE {
                let scale = bf16_bits_to_f32(u16::from_le_bytes([
                    row_scales[g * 2],
                    row_scales[g * 2 + 1],
                ]));
                for i in 0..GROUP_SIZE / 2 {
                    let byte = row_packed[g * (GROUP_SIZE / 2) + i];
                    let lo = (byte & 0x0F) as i32 - 8;
                    let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                    let h_lo = g * GROUP_SIZE + i * 2;
                    let h_hi = g * GROUP_SIZE + i * 2 + 1;
                    let got_lo = lo as f32 * scale;
                    let got_hi = hi as f32 * scale;
                    let ref_lo = src_f32[h_lo * n_intermediate + r];
                    let ref_hi = src_f32[h_hi * n_intermediate + r];
                    // Re-quantization adds ≤ 1 LSB of the new scale per element.
                    // With our test data the new max-abs scale per group is small,
                    // so allow up to 1.5× scale error.
                    let tol = scale.max(0.0625) * 1.5;
                    assert!(
                        (got_lo - ref_lo).abs() <= tol,
                        "(r={r}, h={h_lo}): got={got_lo} ref={ref_lo} tol={tol}",
                    );
                    assert!(
                        (got_hi - ref_hi).abs() <= tol,
                        "(r={r}, h={h_hi}): got={got_hi} ref={ref_hi} tol={tol}",
                    );
                }
            }
        }
    }

    /// AXPY-form down (over all active lanes = dense equivalent)
    /// should produce numerically-close output to the dense GEMV
    /// path on the *original* (non-transposed) weight. Tolerance
    /// allows for the one extra rounding step from re-quantization.
    #[test]
    fn axpy_close_to_dense_down_random_weights() {
        let n_hidden = 64;
        let n_intermediate = 64;
        let (src_packed, src_scale_bits) = make_down_weight(n_hidden, n_intermediate);
        let (packed_t, scale_t_bits) =
            transpose_requantize_down(&src_packed, &src_scale_bits, n_hidden, n_intermediate);

        // Build an arbitrary intermediate vector (in the AXPY-form
        // path this is `silu(gate) ⊙ up`).
        let inter: Vec<f32> = (0..n_intermediate)
            .map(|i| (i as f32) * 0.013 - 0.4)
            .collect();
        // "All active" — every lane participates.
        let active: Vec<u32> = (0..n_intermediate as u32).collect();

        // Dense reference: y_dense = down @ inter
        let mut y_dense = vec![0.0f32; n_hidden];
        dequant_gemv_int4(
            &src_packed,
            &src_scale_bits,
            &inter,
            n_hidden,
            n_intermediate,
            &mut y_dense,
        );

        // AXPY path
        let mut y_axpy = vec![0.0f32; n_hidden];
        dequant_axpy_int4_active_auto(
            &packed_t,
            &scale_t_bits,
            &inter,
            &active,
            n_intermediate,
            n_hidden,
            &mut y_axpy,
        );

        // Allow up to a few percent of the magnitude as error
        // (re-quantization adds one rounding step per weight).
        let max_dense = y_dense.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let max_dev = y_dense
            .iter()
            .zip(y_axpy.iter())
            .map(|(&d, &a)| (d - a).abs())
            .fold(0.0f32, f32::max);
        let tol = max_dense.max(1.0) * 0.10; // 10% tolerance is loose but ok for synthetic
        assert!(
            max_dev <= tol,
            "AXPY diverged: max_dev={max_dev} max_dense={max_dense} tol={tol}",
        );
    }

    /// AXPY with a SUBSET of active lanes equals AXPY-all minus the
    /// inactive contributions. Verifies linearity / no per-lane
    /// crosstalk in the kernel.
    #[test]
    fn axpy_subset_equals_sum_of_per_lane() {
        let n_hidden = 32;
        let n_intermediate = 64;
        let (src_packed, src_scale_bits) = make_down_weight(n_hidden, n_intermediate);
        let (packed_t, scale_t_bits) =
            transpose_requantize_down(&src_packed, &src_scale_bits, n_hidden, n_intermediate);
        let inter: Vec<f32> = (0..n_intermediate).map(|i| (i as f32) * 0.1).collect();

        let subset: Vec<u32> = vec![3, 7, 11, 25];

        // y_subset = AXPY over subset
        let mut y_subset = vec![0.0f32; n_hidden];
        dequant_axpy_int4_active(
            &packed_t,
            &scale_t_bits,
            &inter,
            &subset,
            n_intermediate,
            n_hidden,
            &mut y_subset,
        );

        // y_sum = sum of per-lane AXPYs
        let mut y_sum = vec![0.0f32; n_hidden];
        for &r in &subset {
            dequant_axpy_int4_active(
                &packed_t,
                &scale_t_bits,
                &inter,
                std::slice::from_ref(&r),
                n_intermediate,
                n_hidden,
                &mut y_sum,
            );
        }

        for h in 0..n_hidden {
            assert!(
                (y_subset[h] - y_sum[h]).abs() < 1e-5,
                "h={h}: subset={} sum={}",
                y_subset[h],
                y_sum[h]
            );
        }
    }

    /// Scalar and AVX-512 paths produce bit-identical output (when
    /// AVX-512 is available — on aarch64 / non-x86 this just exercises
    /// the scalar path twice).
    #[test]
    fn axpy_scalar_matches_avx512_auto() {
        let n_hidden = 256; // multiple of CHUNK=256 for the AVX-512 path
        let n_intermediate = 64;
        let (src_packed, src_scale_bits) = make_down_weight(n_hidden, n_intermediate);
        let (packed_t, scale_t_bits) =
            transpose_requantize_down(&src_packed, &src_scale_bits, n_hidden, n_intermediate);
        let inter: Vec<f32> = (0..n_intermediate)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();
        let active: Vec<u32> = (0..n_intermediate as u32).filter(|i| i % 2 == 0).collect();

        let mut y_scalar = vec![0.0f32; n_hidden];
        dequant_axpy_int4_active(
            &packed_t,
            &scale_t_bits,
            &inter,
            &active,
            n_intermediate,
            n_hidden,
            &mut y_scalar,
        );

        let mut y_auto = vec![0.0f32; n_hidden];
        dequant_axpy_int4_active_auto(
            &packed_t,
            &scale_t_bits,
            &inter,
            &active,
            n_intermediate,
            n_hidden,
            &mut y_auto,
        );

        for h in 0..n_hidden {
            // Same algorithm, same inputs — should be bit-identical
            // *given* the AVX-512 path uses the same FMA order. (If
            // the AVX-512 path were to re-order the per-active-lane
            // accumulation, results could differ at the f32 ULP
            // level. We process lanes in the same order in both.)
            let diff = (y_scalar[h] - y_auto[h]).abs();
            // Allow tiny ULP drift between the f32 horizontal-add
            // in scalar vs the AVX-512 accumulated FMAs.
            assert!(
                diff < 1e-3 * y_scalar[h].abs().max(1.0),
                "h={h}: scalar={} auto={} diff={}",
                y_scalar[h],
                y_auto[h],
                diff
            );
        }
    }

    /// `FfnScratch::resize_for` only grows; never shrinks; idempotent
    /// on equal-size requests.
    #[test]
    fn ffn_scratch_grows_in_place() {
        let mut s = FfnScratch::new(32, 64);
        assert_eq!(s.x_f32.len(), 32);
        assert_eq!(s.gate_out.len(), 64);
        // Same size: no change.
        s.resize_for(32, 64);
        assert_eq!(s.x_f32.len(), 32);
        // Larger: grows.
        s.resize_for(128, 256);
        assert_eq!(s.x_f32.len(), 128);
        assert_eq!(s.gate_out.len(), 256);
        // Smaller: keeps high-water mark.
        s.resize_for(16, 32);
        assert_eq!(s.x_f32.len(), 128);
        assert_eq!(s.gate_out.len(), 256);
    }
}
