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
        Ok(Self { mmap, dim, inter })
    }

    /// y = W x with W dequantized row-by-row; y[o] rounded to bf16 exactly
    /// like `linear_bf16` over an eagerly-dequantized W.
    ///
    /// Output rows are independent: dequant + dot each on its own core (rayon),
    /// with a per-row scratch buffer. Bit-identical to the sequential version
    /// (same per-row accumulation order), just spread across the CPU — this is
    /// the real-model MoE hot path (256 experts, mmap int4).
    fn gemv(&self, sec_off: usize, out_dim: usize, in_dim: usize, x: &[f32], y: &mut [f32]) {
        use rayon::prelude::*;
        debug_assert_eq!(x.len(), in_dim);
        debug_assert_eq!(y.len(), out_dim);
        let ng = in_dim / G;
        let packed = &self.mmap[sec_off..sec_off + out_dim * in_dim / 2];
        let scales = &self.mmap
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

    /// Batched mirror of [`Self::forward`] for `rows` activation rows
    /// (`xs = [rows, dim]`) that all routed to THIS expert. Each int4 weight
    /// row is dequantized ONCE and reused across every activation row, so the
    /// nibble unpack — the decode hot path's dominant cost — is paid per weight
    /// instead of per (weight, token). `route_ws[r]` scales row r (as `route_w`
    /// scales the single-token path). Returns `[rows, dim]`, bit-identical to
    /// calling [`Self::forward`] on each row (scalar path; see `gemv_batch`).
    pub fn forward_batch(
        &self,
        xs: &[f32],
        rows: usize,
        dim: usize,
        limit: f32,
        route_ws: &[f32],
    ) -> Vec<f32> {
        debug_assert_eq!(dim, self.dim);
        debug_assert_eq!(xs.len(), rows * dim);
        debug_assert_eq!(route_ws.len(), rows);
        let inter = self.inter;
        let w1_off = 0;
        let w3_off = section_bytes(inter, dim);
        let w2_off = 2 * section_bytes(inter, dim);
        let mut gate = vec![0.0f32; rows * inter];
        let mut up = vec![0.0f32; rows * inter];
        self.gemv_batch(w1_off, inter, dim, xs, rows, &mut gate);
        self.gemv_batch(w3_off, inter, dim, xs, rows, &mut up);
        let mut h = vec![0.0f32; rows * inter];
        for r in 0..rows {
            let rw = route_ws[r];
            for i in 0..inter {
                let mut g = gate[r * inter + i];
                let mut u = up[r * inter + i];
                if limit > 0.0 {
                    u = u.clamp(-limit, limit);
                    g = g.min(limit);
                }
                let s = g / (1.0 + (-g).exp()); // silu
                                                // rw == 1.0 for the shared/no-route case is an exact f32 no-op,
                                                // so this matches `forward`'s `route_w: None` path bit-for-bit.
                let v = s * u * rw;
                h[r * inter + i] = to_bf16(v);
            }
        }
        let mut out = vec![0.0f32; rows * dim];
        self.gemv_batch(w2_off, dim, inter, &h, rows, &mut out);
        out
    }

    /// Batched `gemv`: `ys[r] = W xs[r]` for every row, with W dequantized ONCE
    /// per output row and reused across all `rows` inputs. `xs = [rows, in_dim]`,
    /// `ys = [rows, out_dim]`. Parallelized over output rows (each owns a column
    /// of a transposed scratch, then scattered into row-major `ys`).
    ///
    /// Scalar dequant+dot in strict column order — bit-identical to the scalar
    /// [`dequant_row_dot`] the single-token [`Self::gemv`] uses, so on the scalar
    /// path `forward_batch` equals per-row `forward` exactly. (The AVX2 fused
    /// kernel accumulates in a different lane order; a matching AVX2 batch kernel
    /// is a follow-up for on-node throughput + on-node bit-exactness.)
    fn gemv_batch(
        &self,
        sec_off: usize,
        out_dim: usize,
        in_dim: usize,
        xs: &[f32],
        rows: usize,
        ys: &mut [f32],
    ) {
        use rayon::prelude::*;
        debug_assert_eq!(xs.len(), rows * in_dim);
        debug_assert_eq!(ys.len(), rows * out_dim);
        let ng = in_dim / G;
        let packed = &self.mmap[sec_off..sec_off + out_dim * in_dim / 2];
        let scales = &self.mmap
            [sec_off + out_dim * in_dim / 2..sec_off + out_dim * in_dim / 2 + out_dim * ng * 2];
        let row_bytes = in_dim / 2;
        // Transposed output [out_dim, rows] so each output row `o` owns a
        // contiguous `[rows]` slice (no aliasing under rayon); transpose after.
        let mut yt = vec![0.0f32; out_dim * rows];
        yt.par_chunks_mut(rows).enumerate().for_each(|(o, ycol)| {
            let prow = &packed[o * row_bytes..(o + 1) * row_bytes];
            let srow = &scales[o * ng * 2..(o + 1) * ng * 2];
            let wrow = dequant_row_f32(prow, srow, in_dim); // unpack ONCE
            for r in 0..rows {
                let x = &xs[r * in_dim..(r + 1) * in_dim];
                // strict column-order dot == dequant_row_dot_scalar's order.
                let mut acc = 0.0f32;
                for k in 0..in_dim {
                    acc += wrow[k] * x[k];
                }
                ycol[r] = to_bf16(acc);
            }
        });
        for o in 0..out_dim {
            for r in 0..rows {
                ys[r * out_dim + o] = yt[o * rows + r];
            }
        }
    }
}

/// Dequantize one packed int4 weight row into f32 column order, folding the
/// per-32-group bf16 scale: `out[k] = (nibble_k - 8) * scale(g(k))`. Matches the
/// element decode in [`dequant_row_dot_scalar`], so a strict column-order dot of
/// the result reproduces that kernel's summation exactly.
fn dequant_row_f32(packed_row: &[u8], scales_row: &[u8], in_dim: usize) -> Vec<f32> {
    let ng = in_dim / G;
    let mut out = vec![0.0f32; in_dim];
    for g in 0..ng {
        let s = bf16::from_le_bytes([scales_row[g * 2], scales_row[g * 2 + 1]]).to_f32();
        for i in 0..G / 2 {
            let byte = packed_row[g * (G / 2) + i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out[g * G + 2 * i] = lo as f32 * s;
            out[g * G + 2 * i + 1] = hi as f32 * s;
        }
    }
    out
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
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2+fma detected at runtime; every load stays within
            // packed_row (in_dim/2 bytes), scales_row (in_dim/G*2 bytes) and x (in_dim).
            return unsafe { dequant_row_dot_avx2(packed_row, scales_row, x, in_dim) };
        }
    }
    dequant_row_dot_scalar(packed_row, scales_row, x, in_dim)
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
    use super::{dequant_row_dot, MmapExpert, G};

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

    /// `forward_batch` over N rows must equal calling `forward` on each row —
    /// exactly, bit-for-bit, on the scalar path (this machine). Guards the
    /// batch-union prefill kernel: dequant-once-reuse-across-rows preserves the
    /// per-token summation order. Builds a deterministic int4_bin on disk and
    /// mmaps it through the real `MmapExpert::open`.
    #[test]
    fn forward_batch_matches_per_token() {
        use std::io::Write;
        let (dim, inter) = (64usize, 128usize);
        // int4_bin layout (exporter contract): w1[inter,dim], w3[inter,dim],
        // w2[dim,inter], each = packed nibbles then bf16-LE per-32 scales.
        let mut buf = Vec::new();
        let mut push_section = |out_dim: usize, in_dim: usize, seed: u64| {
            let ng = in_dim / G;
            for k in 0..(out_dim * in_dim / 2) {
                buf.push(((k as u64 * 131 + seed * 7 + 17) & 0xFF) as u8);
            }
            for j in 0..(out_dim * ng) {
                let s = half::bf16::from_f32(0.03 + 0.001 * ((j as u64 + seed) % 40) as f32)
                    .to_le_bytes();
                buf.push(s[0]);
                buf.push(s[1]);
            }
        };
        push_section(inter, dim, 1); // w1
        push_section(inter, dim, 2); // w3
        push_section(dim, inter, 3); // w2

        let path = std::env::temp_dir().join(format!("dsv4_fb_{}.int4bin", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&buf)
            .unwrap();
        let e = MmapExpert::open(&path, dim, inter).unwrap();

        let rows = 5usize;
        let mut xs = vec![0f32; rows * dim];
        for (i, v) in xs.iter_mut().enumerate() {
            *v = ((i as f32) * 0.017).sin() * 0.7;
        }
        let route_ws: Vec<f32> = (0..rows).map(|r| 0.5 + 0.1 * r as f32).collect();
        let limit = 7.0f32;

        let batch = e.forward_batch(&xs, rows, dim, limit, &route_ws);
        for r in 0..rows {
            let single = e.forward(&xs[r * dim..(r + 1) * dim], dim, limit, Some(route_ws[r]));
            for c in 0..dim {
                assert_eq!(
                    batch[r * dim + c].to_bits(),
                    single[c].to_bits(),
                    "row {r} col {c}: batch {} != single {}",
                    batch[r * dim + c],
                    single[c]
                );
            }
        }
        let _ = std::fs::remove_file(&path);
    }
}
